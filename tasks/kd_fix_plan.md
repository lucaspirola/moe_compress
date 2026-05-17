# Stage 2.5 Router-KD overfit — fix plan

## Verified diagnosis (against the code)

Stage 2.5 = `stage5_router_kd.run()` with `stage_key="stage2p5"`; config block
`stage5_router_kd` in `configs/qwen36_35b_a3b_30pct.yaml`.

Confirmed against the actual source:

1. **Fixed corpus, replayed N× with no reshuffle.**
   `batches = iter_batches(calib, batch_size=...)` (`stage5_router_kd.py:617`)
   materialises the batch list **once** (`utils/calibration.py:249-262` —
   eager list, despite the `iter_` name). The training loop
   `for epoch in range(s5["epochs"])` (`:931`) then iterates the *same* list
   object every epoch — `for i, batch in enumerate(batches)` (`:942`). No
   shuffle anywhere. So epochs 2 & 3 are byte-identical replays.

2. **`total_steps` = (batches // grad_accum) × epochs.**
   `stage5_router_kd.py:785`: `total_steps = (len(batches) // grad_accum) * s5["epochs"]`.
   With `max_calibration_samples=6000`, `batch_size=8`, `grad_accum=1`,
   `epochs=3`: `(6000/8 // 1) × 3 = 750 × 3 = 2250`. Matches the analysis.

3. **T-ramp spans the *full* 2250 steps.**
   `_current_T()` (`:815-819`): linear `T_start→T_end` over `total_optim_steps`.
   `kd_temperature_start=4.0 → kd_temperature_end=1.0`. As T→1 the soft-target
   regularisation vanishes and the loss sharpens onto the teacher's exact modes
   on the *training* sequences — the back-half drift the analysis describes.

4. **`save_best` selects on EMA of the training-batch `raw_kl`.**
   `:1104` accumulates `kl_loss.detach() / T²` (temperature-invariant raw KL)
   over the log window; `:1133-1145` averages it and EMAs it
   (`best_metric_ema_alpha=0.2`); `:1152-1155` writes `best.pt` whenever
   `ema < best_raw_kl_ema`. This is a windowed **training-loss** proxy, NOT a
   held-out metric. It catches the drift only because the T-ramp also makes the
   training `raw_kl` rise in the back half. `best_step` pins early
   (~step 250 on S0) and `:1235-1256` reloads that snapshot at export — so
   ~88% of the schedule is discarded compute.

5. **`teacher_logits_cache` is hard-rejected for `epochs>1`.**
   `:894-903` raises `RuntimeError` when `epochs>1 and teacher_logits_cache is
   not None`. So the multi-epoch config can't use the cache and pays
   ~1500 redundant 35B-teacher forwards. The config comment at
   `qwen36_35b_a3b_30pct.yaml:326-329` documents this. With `epochs=1` the
   guard is inert and the cache is usable.

### Corrections to the prior analysis

- The analysis cited "the existing runtime guard at `stage5_router_kd.py:744`"
  for the cache/epochs conflict. The real guard is at **`:894-903`**
  (line 744 is unrelated — optimizer-state device-migration logging). The
  config comment also points at `:744`; both references are stale. The guard
  is correct in behaviour; only the line number is wrong.
- The cache code already *accepts* `epochs=1` (`:325-346` validates
  `cache_n >= epochs_cfg * cfg_n`; with `epochs=1` that is just `cfg_n`).
  So re-enabling the cache for `epochs=1` needs **no code change** — only the
  config epochs bump + a comment refresh.
- Otherwise the analysis is accurate.

## Prioritized fixes

### P0 — Safe default changes (zero quality loss; we export step ~250 anyway)

| # | Change | File | Type | Effect | Risk |
|---|--------|------|------|--------|------|
| 1 | `epochs: 3 → 1` | config | default | 2250→750 steps; drops the redundant epochs 2&3 that only fed the overfit. We already export the step-~250 best, so quality is unchanged. ~−40 min wall/row. | None — best-checkpoint export means the discarded steps had no effect on the exported model. |
| 2 | `max_calibration_samples: 6000 → 3000` | config | default | 750→375 steps. The 6000 bump (2026-05-13) existed only to feed `epochs=3`; with `epochs=1` and an early-stop at ~step 250, 3000 samples (375 steps) still comfortably covers the useful horizon. ~halves Stage-2.5 calib build + wall. | Low. If early-stop does not fire before step 375 the run simply ends at 375 — still past the observed step-250 optimum. |
| 3 | Re-enable `teacher_logits_cache` for `epochs=1` | (no code) | doc-only | With `epochs=1` the `:894` guard is inert and `:325-346` accepts a `cfg_n`-sized cache. Cache becomes usable → skips the live 35B teacher entirely. | None — guard already allows it; only the stale config comment is refreshed. |

### P1 — Early stopping (new, config-gated, default-OFF → no behaviour change)

New config key `early_stop_patience` (default `0` = disabled). When `>0`:
patience-based on the **existing** `raw_kl` EMA — the same metric `save_best`
already tracks. Each log window: if the EMA did **not** improve on
`best_raw_kl_ema`, increment a no-improve counter; on improvement reset it.
When the counter reaches `early_stop_patience` consecutive non-improving log
windows, break out of the training loop cleanly. The save-best/`best.pt`
machinery is untouched — the run still exports the best snapshot, it just
stops walking the discarded back half.

- File: `stage5_router_kd.py`.
- Type: config-gated, default `0`. With the default the loop body is
  byte-identical to current `main` (the patience counter is computed but
  never triggers a break; guarded by `if _early_stop_patience > 0`).
- Effect: cuts the ~88% wasted back-half compute *automatically*, without
  hand-tuning `epochs`. Complements P0#1/#2 as the principled stopping rule.
- Risk: Low. Stops strictly later than the best step. Counter persists across
  the checkpoint/resume payload so a resumed run does not lose patience state.

### P1 — Per-epoch reshuffle (new, config-gated, default-OFF)

New config key `shuffle_batches_each_epoch` (default `false`). When `true`,
the batch *iteration order* is permuted per epoch with a deterministic,
epoch-seeded RNG. Only relevant when `epochs>1`. Implemented but **default
off** so the `epochs=1` production path is unaffected and the flag-off path
is byte-identical to `main`.

- Rationale: with `epochs=1` (the new default) this is moot, but it is the
  correct mitigation if a future ablation re-raises `epochs`. Cheap and
  clearly correct (a permutation of an existing list).
- Refuses to combine with `teacher_logits_cache` (the cache is indexed by
  positional `(epoch*len+i)`; a shuffled order would mispair) — fail-loud,
  consistent with the existing `:894` guard.
- Risk: Low; gated and default-off.

### P2 — Result-changing knobs (config-gated, default = current behaviour)

These change the exported model, so they are **opt-in** for the next ablation
round and must default to today's values:

| Knob | New config key | Default (= current) | When set |
|------|----------------|---------------------|----------|
| T-ramp horizon | `kd_temperature_ramp_fraction` | `1.0` (ramp over full schedule) | `<1.0` finishes the 4.0→1.0 ramp within the first fraction of steps, then holds at `T_end`. Flattens / shortens the late sharpening. |
| Held-out save-best | `save_best_holdout_fraction` | `0.0` (select on training raw_kl, as today) | `>0.0` reserves that fraction of calibration batches as a held-out slice; `save_best` selects on the held-out KL instead of the training-window EMA. |
| Cosine LR floor | `lr_min_ratio` | `0.10` (already a config key) | set `0.0` to let the cosine decay to true zero. |

`lr_min_ratio` is already a plumbed config key (`:795`, `:809`) — no code
change needed; the plan only documents that the next round can set it to 0.
T-ramp-fraction and held-out-save-best require new, gated code paths.

**Scope decision:** P0 + P1 are implemented in this pass (they are the
overfit fix). For P2, `kd_temperature_ramp_fraction` is implemented (small,
self-contained, clearly default-safe). `save_best_holdout_fraction` is the
largest change (carve a held-out slice, extra eval forward pass, resume-state
implications) and is **deferred** — documented here as the recommended next
step but not implemented, to keep this pass surgical and the flag-off path
provably identical. `lr_min_ratio=0` needs no code.

## Test plan

- Run existing `test_stage5_merge_repair.py` + the stage5 smoke tests; record
  pre-existing failures vs regressions.
- Add `test_stage5_early_stop.py` (CPU-only): early-stop triggers after
  `patience` non-improving windows; does NOT trigger while improving;
  flag-off (`patience=0`) path leaves the loop count unchanged.
