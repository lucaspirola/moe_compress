# PLAN — Distillation-plugin Tier-1 CPU optimizations

**Branch:** `plan/distill-opt` (base `origin/main` @ `0ae66896942cd8ecbca1cf62c81aaec60f889590`)
**Scope:** PLAN ONLY. No production edits in this branch. All file:line citations verified against `origin/main` blobs on 2026-05-31.
**Host for testing:** RTX 5080, 16 GB VRAM (`nvidia-smi`: `16303 MiB`). This is the binding VRAM constraint for Lever 1.

## Tier-1 contract (applies to all four levers)

Every lever here is **mechanism-only / byte-identical**: it changes *when* / *how* a tensor is moved between host and device, or *how much* host RAM is held, but it does NOT change:
- the CPU bytes that are uploaded (same `.to(dtype=bf16)` source tensors),
- the order of GPU ops in the AdamW / forward loops,
- any RNG draw (no `randperm`/`Generator` touched),
- any numeric value entering a loss or an optimizer step.

The two affected plugins are **default-OFF** (`block_refine.enabled=False`; `expert_distill_steps=0`), but the user has confirmed the recovery run WILL enable both. So these are real hot paths, not dead code.

Threading / `lsa_pool` does NOT apply: neither path calls `linear_sum_assignment`; both are GPU-bound sequential loops. Do not propose it.

---

## Lever 1 — block_refine: hoist per-step re-upload (BIGGEST)

**File / function:** `max_quality/src/moe_compress/stage3/plugins/block_refine.py` → `_phase_c5_block_refine`.

### Current behavior (verified)
The per-block training loop (lines **509–512**) re-uploads the SAME CPU tensors on every one of the `epochs * len(batches)` steps:

```
509  for epoch in range(epochs):
510      for bi, _ in enumerate(batches):
511          x_s = X_student[bi].to(device=device, dtype=student_dtype)
512          target = teacher_targets[bi].to(device=device, dtype=student_dtype)
```

- `X_student[bi]` is a constant bf16 CPU tensor for the whole block (built once at line **314** via `_capture_block_input`, mutated only at block advance, line **598**).
- `teacher_targets[bi]` is built ONCE per block (lines **495–503** live path, or sliced from the cache at **473–476**) and is constant across all `epochs` (default 25) of that block.

So across 25 epochs × `n_batches` steps, the identical H2D copy is issued ~25× per batch per block. The measurement pass reported ~103 GB redundant H2D per block and ~20% of each step's wall time.

### After (restructure)
Hoist the H2D **once per block**, before the epoch loop, into device-resident lists; index those inside the loop with no `.to()`:

```
# --- before the epoch loop, after teacher_targets is finalized (~line 504) ---
x_s_dev      = [X_student[bi].to(device=device, dtype=student_dtype) for bi in range(len(batches))]
target_dev   = [teacher_targets[bi].to(device=device, dtype=student_dtype) for bi in range(len(batches))]

# --- inside the loop (replaces 511-512) ---
for epoch in range(epochs):
    for bi, _ in enumerate(batches):
        x_s    = x_s_dev[bi]
        target = target_dev[bi]
        out = s_layer(x_s, **student_kwargs_all.get(layer_idx, {}))
        ...

# --- after the AdamW loop, BEFORE _advance_streams (~line 597) ---
del x_s_dev, target_dev
# (optional) torch.cuda.empty_cache() only if a VRAM-tight fallback is selected — see Risks.
```

**Byte-identity argument:** `Tensor.to(device, dtype)` is a pure copy/cast. Doing it once and reusing the result is bit-identical to doing it every step — the source CPU bytes never change between steps (no in-place mutation of `X_student[bi]` / `teacher_targets[bi]` inside the loop; the only mutation is at block advance, line 598, which runs AFTER `del`). The forward `s_layer(x_s, ...)`, the loss, the optimizer step, the LR schedule, and `step` increment are untouched. Identical op order, identical RNG (none in this loop), identical values.

### GPU-memory cost (the binding risk on the 16 GB host)
Resident working set per block = `x_s_dev` + `target_dev`. Each is `n_batches` tensors of shape `(batch_size, seq_len, hidden)` bf16.
- The measurement pass cited **~2.15 GB per list, ~4.3 GB total resident per block** (2× ~2.15 GB). VERIFY at implementation time by computing `n_batches * batch_size * seq_len * hidden * 2 bytes` from the live config — the recovery config's `block_refine.batch_size`, `calibration.num_sequences`, `calibration.sequence_length`, and the model `hidden_size` set the true number. (Note: `X_student`/`teacher_targets` lists already span ALL batches in CPU RAM today; Lever 1 only changes *where* the per-block copy lives, GPU vs CPU.)
- On the 16 GB 5080 the resident 30B-A3B student layer + teacher target forward already lives on-device; adding ~4.3 GB of resident activations must be checked against headroom. If the resident set does not fit, use the **pinned-once fallback** below.

### Pinned-once fallback (VRAM-tight)
If the full per-block resident set blows the budget: keep the lists on CPU but make them **pinned** (see Lever 2) and upload per-step with `non_blocking=True`. This recovers most of the H2D-throughput win (pinned + async) without the resident-VRAM cost. The H2D is still issued per step, but at 3.25× throughput and overlapped with compute. This is strictly a fallback — Lever 1's hoist is preferred when VRAM allows.

### Free point
`del x_s_dev, target_dev` immediately before `_advance_streams` (line 598). `_advance_streams` does its own per-batch `.to(device)` of the CPU `X_student`/`X_teacher` lists, so freeing the hoisted copies first maximizes headroom for the advance forward. Do NOT add an unconditional `torch.cuda.empty_cache()` (it serializes the stream and is a perf regression); only consider it inside the VRAM-tight fallback path.

---

## Lever 2 — block_refine: pinned + `non_blocking` H2D

**File / function:** same file; producers of the CPU bf16 lists.

### Current behavior (verified)
The CPU bf16 tensors are pageable (`.to(device="cpu")` with no `.pin_memory()`):
- `_capture_block_input._hook` (line **294**): `captured[...] = t.detach().to(dtype=torch.bfloat16, device="cpu")`.
- live teacher target (line **503**): `teacher_targets.append(out.detach().to(dtype=torch.bfloat16, device="cpu"))`.
- `_advance_streams` (lines **624**, **629**): `new_s/new_t.append(out_*.detach().to(dtype=torch.bfloat16, device="cpu"))`.

Pageable→device copies are synchronous and ~3.25× slower than pinned async copies (measurement pass).

### After
1. Allocate the CPU-side tensors as **pinned** at the producer sites. Append `.pin_memory()` to the four producer `.to(..., device="cpu")` calls (294, 503, 624, 629). The cache-slice path (lines 473–476) produces views into a single large cached tensor; pin the per-batch slice copy at the consumption site instead (see below) rather than the whole cache.
2. At the H2D site, add `non_blocking=True`. With Lever 1 applied, the H2D is the **once-per-block** hoist (`x_s_dev`/`target_dev` build) — pin those source tensors so the single bulk upload is async. Without Lever 1 (fallback path), pin + `non_blocking` on the per-step `.to()` at 511–512.

**Bound on pinned RAM:** pin only the *resident working set*, i.e. the `X_student` + `teacher_targets` + `X_teacher` lists that are live for the current block, NOT every historical block. These lists are already replaced wholesale at each block advance (line 598 reassigns `X_student, X_teacher`), so pinned RAM is bounded by `2–3 × n_batches × batch_size × seq_len × hidden × 2 bytes` — same order as the Lever 1 resident set. Pinned host RAM is non-pageable; on a many-GB working set verify the host has the headroom (the recovery host has ample system RAM; this is not the 16 GB GPU budget). If pinned-RAM pressure is observed, fall back to pinning only `x_s_dev`/`target_dev` sources for the hoisted upload and leave `X_teacher` pageable.

**Byte-identity argument:** `.pin_memory()` returns a copy in pinned host memory with identical bytes; `non_blocking=True` only relaxes the host-side synchronization of the copy — PyTorch still orders the copy before the consuming kernel on the same stream, so the consumed values are identical. No numeric change.

**Stacking:** Lever 2 stacks under Lever 1 — pin the once-uploaded copy's *source* so the single bulk H2D is both async and pinned-throughput.

---

## Lever 3 — block_hidden_cache: lazy / prefetch load

**File / function:** `max_quality/src/moe_compress/stage3/plugins/block_hidden_cache.py` → `Stage3BlockHiddenCacheProvider._load_layers` (lines **144–177**) and `on_load` (lines **179–291**).

### Current behavior (verified)
`_load_layers` (called from `on_load`, line **215**) eagerly `load_block_hidden(...)`s EVERY per-layer sidecar up-front (loop at **161–176**), and `on_load` reshapes ALL of them into `teacher_targets_cache` (loop at **240–283**), a `dict[layer_idx -> Tensor]` held for the whole Stage 3 pass. Measurement: ~21 s up-front stall + ~52 GB host RAM held for the duration.

`block_refine` consumes one layer's entry at a time (lines 463–476: `teacher_targets_cache.get(int(layer_idx))`), in `all_indices` order, with a full 25-epoch train between consecutive layer reads.

### Miss-detection contract (MUST preserve — this is the hard part)
Today the cache is **all-or-nothing at start**: `on_load` validates EVERY layer's prompt-count (I2 guard, lines 247–258) and token-count (lines 263–274) up-front and returns `None` (cache miss → live teacher forward) if ANY layer fails. The `block_refine` consumer therefore sees either a fully-hydrated dict or no dict at all. The orchestrator's `dispatch_first("on_load", ...)` either wins (all layers good) or the slot is absent (`ctx.has("teacher_targets_cache")` is False).

A lazy / per-layer load moves the miss point from "before Stage 3 starts" to "during layer `i`'s consumption" — and a malformed layer `i` discovered mid-pass would be discovered AFTER layers `0..i-1` already trained against cached targets, which is FINE numerically (each layer's target is independent), but the all-or-nothing miss-detection semantic must be preserved so a partial cache never silently mixes cached + live targets in a way the operator can't see.

### After — recommended design (eager-validate, lazy-materialize)
Split the two costs that are conflated today:
1. **Validation stays eager and all-or-nothing** (cheap, metadata-only): in `on_load`, walk the sidecars dir and validate `n_prompts_in_subset` + `n_tokens == n_prompts*seq_len` for EVERY layer using ONLY the sidecar header / shape metadata — WITHOUT materializing the reshaped `[n_prompts, seq_len, hidden]` tensor or holding the raw `hidden_states` in RAM. This requires `load_block_hidden` (or a new metadata-only helper in `cached_calibration_signals`) to expose `n_prompts_in_subset` and `hs.shape[0]` without retaining the full payload. CHECK the `BlockHiddenPayload` / `load_block_hidden` API: if it always loads the full tensor (likely, since it's a `torch.load`), add a lightweight `peek_block_hidden(jsonl_path, layer_idx) -> (n_prompts_in_subset, n_tokens, hidden)` that reads only metadata (e.g. via `torch.load(..., mmap=True)` shape inspection, or a stored header field). If a cheap metadata read is not feasible, fall back to design (b) below.
2. **Materialization becomes lazy**: instead of populating `teacher_targets_cache` with eager tensors, populate it with a **lazy mapping** — a small object that, on `cache.get(layer_idx)`, `torch.load`s + reshapes that one layer's sidecar on demand and returns the `[n_prompts, seq_len, hidden]` tensor. Because `block_refine` reads each layer exactly once and never re-reads, a plain lazy dict (load-on-get, no caching of prior layers) holds only ~1 layer's tensor in RAM at a time (~52 GB / n_layers). The consumer's `cached.get(int(layer_idx))` call site (block_refine line 465) is unchanged.

This preserves the contract: validation is still complete and up-front (operator sees a clean hit/miss decision before any training), while the 52 GB / 21 s materialization is amortized lazily over the pass.

### Optional prefetch (only if the lazy stall per layer is material)
Layer `i`'s `torch.load` (~52 GB / n_layers ≈ ~1–2 GB) can be prefetched on a background thread during layer `i-1`'s 25-epoch train (the train is GPU-bound; a CPU-side `torch.load` overlaps cleanly). Implement as a one-deep prefetch in the lazy mapping: when `get(i)` is called, kick a daemon thread to `torch.load(i+1)`. KEEP this OFF the byte-identity-critical path: prefetch only changes load timing, never values. Recommend deferring prefetch to a follow-up unless the per-layer lazy stall is measured to matter — the lazy-materialize step alone removes the 52 GB held-RAM and the 21 s up-front stall.

### Alternative design (b) — if metadata-only peek is infeasible
If `load_block_hidden` cannot be made to peek cheaply, keep `_load_layers` eager for VALIDATION (it must read each sidecar to validate), but DON'T retain the reshaped tensors: validate each, then `del` the payload, and store in `teacher_targets_cache` a lazy loader keyed by `(jsonl_path, layer_idx)` that re-loads on `get`. This still removes the 52 GB held-RAM (only one layer materialized at a time) but does NOT remove the ~21 s up-front validation stall (each sidecar is read once for validation). The held-RAM win is the larger one; accept the stall if the peek is infeasible.

**Byte-identity argument:** the cached tensor returned by `get(layer_idx)` is the SAME reshaped `[n_prompts, seq_len, hidden]` bf16 tensor (same `hs.reshape(n_prompts, seq_len, -1).contiguous()`, line 278) whether materialized eagerly or lazily. `block_refine` slices it identically (lines 473–476). No value change; only allocation timing.

---

## Lever 4 — expert_distill: snapshot → members only

**File / function:** `max_quality/src/moe_compress/stage2/plugins/expert_distill.py` → `_snapshot_pre_merge_layer_experts` (lines **489–515**) and its caller `ExpertDistillPlugin.pre_merge_snapshot` (lines **987–1009**).

### Current behavior (verified)
`_snapshot_pre_merge_layer_experts` snapshots ALL `num_routed_experts` (loop `for eid in range(n)`, lines 510–514), each `.detach().cpu().clone()` of gate/up/down. But `_distill_merged_group` reads ONLY group members: `pre_merge_weights[m]["..."] for m in members` (v1 lines 677–679; v2 reads the same `pre_merge_weights[m]` per member). Non-member entries are NEVER read.

The v1-waste note in the docstring (lines 499–505) already flags this and prescribes the fix: narrow to `set().union(*grouped.values())`.

### `grouped` availability (verified)
- `grouped` is committed to ctx at `stage2/orchestrator.py` line **608** (inside `_run_assignment`, called at line **1567**).
- `pre_merge_snapshot` runs in `_STAGE2_POST_ASSIGN_PHASES` (`orchestrator.py` line **205**, first phase), walked at line **1568** — AFTER `_run_assignment`. So `grouped` IS in ctx when the snapshot hook fires. ✓
- `grouped` shape (verified `stage2/grouping.py:63` `_build_grouped_from_assignment`): `{centroid_id: [centroid_id, *absorbed_member_ids]}`. Orphan non-centroids are promoted to singleton groups (`_promote_orphans`, grouping.py). So `set().union(*grouped.values())` = every centroid + every assigned/promoted child = EXACTLY the experts that any `_distill_merged_group` call can read via `members`.

### `reads` declaration (correction to the task brief)
The brief says "the plan must add `grouped` to the snapshot hook's `reads`". **`grouped` is ALREADY in the plugin-level `reads` tuple** (`expert_distill.py` line **896**: `"layer_ref", "pre_merge_weights", "grouped", "freq", "layer_input_acc"`). So no `reads` change is needed — the registry contract already declares it. What IS needed is for the `pre_merge_snapshot` hook BODY (currently reads only `layer_ref`, line 999) to also `ctx.get("grouped")` and pass it down. Confirm `grouped` is set-once-before-read in the registry's ordering check (it is: set at orch 1567/608, read at orch 1568) so the read does not trip a "read-before-write" registry assertion.

### After
1. Add an optional `members` param to the helper (keeps the existing 1-arg test call `_snapshot_pre_merge_layer_experts(layer_ref)` working — see test gate):
```
def _snapshot_pre_merge_layer_experts(
    layer_ref: MoELayerRef,
    members: "set[int] | None" = None,
) -> dict[int, dict[str, torch.Tensor]]:
    banks = build_banks(layer_ref)
    out: dict[int, dict[str, torch.Tensor]] = {}
    eids = range(layer_ref.num_routed_experts) if members is None else sorted(members)
    for eid in eids:
        out[eid] = {name: banks[name].get(eid).detach().cpu().clone() for name in MATRIX_NAMES}
    return out
```
2. In `pre_merge_snapshot` (line 1004–1008), compute the member set from `grouped` and pass it:
```
grouped = ctx.get("grouped")
needed = set().union(*grouped.values()) if grouped else None
pre_merge_weights = (
    _snapshot_pre_merge_layer_experts(layer_ref, members=needed)
    if self.expert_distill_steps > 0 else None
)
```
(Guard `if grouped` so an empty/None grouped degrades to full snapshot — defensive, never expected on the live path.)

**Byte-identity argument:** `_distill_merged_group` reads `pre_merge_weights[m]` only for `m in members`, and every such `m` is in `set().union(*grouped.values())`. The narrowed dict contains exactly those keys (same `.detach().cpu().clone()` bytes). Non-member keys were dead — removing them cannot change any distilled output. `skip_singletons` and the `len(members) <= 1` early-return mean some present keys go unread too; that's still a superset of what's read, so safe.

**Savings:** ~1.2 s + ~0.6 GB host RAM per layer (measurement pass) — the snapshot drops from `num_routed_experts` (e.g. 128/256) clones to the union of group members (≈ num_groups + absorbed, typically far fewer).

---

## Ordering

1. **Lever 4** first — fully self-contained, covered by behavioral tests, lowest risk, smallest diff. Lands and verifies independently.
2. **Lever 3** second — independent file, contract-sensitive but no interaction with Levers 1/2.
3. **Lever 1** third — the big win; needs the new bit-identity gate test (below) in place first.
4. **Lever 2** last — stacks under Lever 1 (pin the hoisted source). Implementing after Lever 1 means the pin sites are the once-per-block upload, not the per-step path.

Rationale: ascending risk / ascending need-for-new-gate. Lever 4 and Lever 3 are gated by existing/extendable behavioral tests; Levers 1+2 need the new block_refine bit-identity harness, so build that once and land 1+2 against it.

---

## Verification gates

### Lever 4 (byte-identical; existing behavioral tests + one new equality test)
- Existing: `max_quality/tests/test_stage2_expert_distill.py`, `test_stage2_plugin_expert_distill.py`.
- **CAUTION:** `test_snapshot_pre_merge_layer_experts_makes_independent_cpu_clones` (test file lines 106–129) asserts `set(snap.keys()) == set(range(n))` — i.e. it asserts the FULL snapshot. The optional-`members`-defaults-to-None signature keeps this test green (it calls `_snapshot_pre_merge_layer_experts(layer_ref)` with no `members`). Do NOT change that test's call; the narrowing only happens on the plugin path where `grouped` is passed.
- **NEW test** (`test_stage2_plugin_expert_distill.py`): drive `ExpertDistillPlugin.pre_merge_snapshot` + `merge` on the `tiny_model` fixture with a known `grouped` in ctx, once with the narrowed snapshot (production path) and once forcing a FULL snapshot (`members=None`), and assert the resulting distilled bank weights (`distill_state` + the written-back `gate/up/down` for each centroid) are **bit-identical** (`torch.equal`). This proves the narrowing is a pure dead-key removal. Seed RNG identically for both runs.

### Lever 3 (byte-identical; extend existing cache tests)
- Existing: `max_quality/tests/test_stage3_block_hidden_cache.py` (covers `test_load_miss_no_dir`, `test_load_hit_populates_teacher_targets_cache`, `test_token_count_mismatch_falls_through_to_miss`, `test_prompt_count_divergence_falls_through_to_miss`, `test_c1_batch_size_decoupled`).
- **NEW tests:**
  - `test_lazy_get_returns_same_tensor_as_eager`: build sidecars, call `on_load`, then `cache.get(layer_idx)` for each layer; assert each returned tensor `torch.equal` to the eager `hs.reshape(n_prompts, seq_len, -1).contiguous()`.
  - `test_partial_cache_still_all_or_nothing_miss`: corrupt ONE layer's token-count; assert `on_load` returns `None` (full miss) BEFORE any training — i.e. the validation stays eager/all-or-nothing. This is the contract-preservation gate.
  - (if prefetch is implemented) `test_prefetch_does_not_change_values`: same as the first test with prefetch enabled.
- These run on CPU; no GPU needed.

### Levers 1 + 2 (NO existing byte-identical golden — build the gate)
The Stage 3 golden snapshot (`test_stage3_golden_snapshot.py`) runs with `block_refine` **OFF** (docstring lines 18–22: "captured with `block_refine` OFF"). So it CANNOT gate Levers 1–3 of block_refine. The mechanism-only argument (above) is necessary but NOT sufficient — build the real gate:

- **NEW test** `max_quality/tests/test_stage3_block_refine_optimized_equiv.py`:
  1. Tiny fixture: a small synthetic decoder stack with a `FactoredExperts` MoE layer (reuse the `tiny_model` Stage 3 fixture / the same construction `test_stage3_plugin_block_refine.py` uses for plumbing), a small `calib_tensor`, `epochs` small (e.g. 2) but `> 1` so the hoist's "reuse across epochs" path is exercised, `batch_size` such that `n_batches >= 2`.
  2. Run `_phase_c5_block_refine` to completion under a deterministic seed and snapshot the trained `FactoredExperts` U/V slots + the four RMSNorm scales (`input_layernorm`, `post_attention_layernorm`, `self_attn.q_norm`, `self_attn.k_norm`).
  3. **Compare old vs new code path.** Two viable mechanizations:
     - (preferred) Parametrize a single test over a module-level flag / monkeypatch-free toggle that selects the hoisted-upload path vs the legacy per-step path. **No monkeypatching production code** (project rule). Implement the toggle as an explicit kwarg or a tiny internal branch added WITH the optimization, defaulting to the optimized path, with the legacy path retained behind the kwarg ONLY for the test — OR
     - (cleaner, recommended) Capture a GOLDEN of the trained factors from `origin/main`'s pre-optimization code FIRST (run the new test against `origin/main` once, save the state_dict tensors as a golden artifact under `max_quality/tests/golden/stage3_block_refine/`), then assert the post-optimization run reproduces it bit-identically via `torch.equal`. Same determinism caveat as the existing golden (same wheel / same host for seed + verify; document it in the test docstring).
  4. Assert `torch.equal` on every trained tensor (U/V for each MATRIX_NAME, all 4 norms). bf16 trained factors are restored to original dtype at lines 538–542; compare in that final dtype.
  5. Run BOTH the live-teacher-target path AND the cache-hit path (`teacher_targets_cache` populated) so the hoist is verified for both target sources.
- **Determinism note:** PyTorch CPU/GPU op order is unchanged by Levers 1+2, so `torch.equal` (not `allclose`) is the correct assertion. If a GPU non-determinism surfaces (it should not, since op order is identical), that is a RED FLAG that the change is not actually mechanism-only — investigate, do not relax to `allclose`.
- The golden-capture variant is preferred because it pins against the ACTUAL pre-change bytes, not a re-derivation. Capture it from `origin/main` (this base) before touching block_refine.

### Full-suite regression
After all four levers: run the Stage 2 + Stage 3 test suites (`pytest max_quality/tests/test_stage2_expert_distill.py max_quality/tests/test_stage2_plugin_expert_distill.py max_quality/tests/test_stage3_block_hidden_cache.py max_quality/tests/test_stage3_plugin_block_refine.py max_quality/tests/test_stage3_golden_snapshot.py max_quality/tests/test_stage3_block_refine_optimized_equiv.py -v`) plus the existing Stage 3 golden (must remain byte-identical since block_refine is OFF there — proves no accidental change to the OFF path).

---

## Risks & rollback

| Lever | Risk | Mitigation / rollback |
|---|---|---|
| 1 | **GPU OOM on 16 GB 5080** from ~4.3 GB resident per-block working set on top of the resident student+teacher layers. | Compute the exact resident bytes from the live config before enabling on-host. If it doesn't fit, use the pinned-once fallback (Lever 2 path, per-step async H2D — keeps the throughput win, drops the resident cost). Rollback: revert the hoist, keep per-step `.to()`. |
| 1 | `del` placement wrong → tensors freed before last use, or held into `_advance_streams` reducing advance headroom. | `del x_s_dev, target_dev` strictly AFTER the epoch loop and BEFORE `_advance_streams` (line 598). Covered by the equivalence test (a premature free would crash or change values). |
| 1 | Resume-from-checkpoint path (lines 339–371) does NOT hit the hoist (it `continue`s before the train loop). | No change needed there; the hoist is only in the train branch. Verify the resume test still passes. |
| 2 | **Pinned host RAM pressure** — pinned memory is non-pageable. | Bound to the current block's working set (lists are replaced at block advance). If pressure observed, pin only the Lever-1 hoist source (`x_s_dev`/`target_dev`), leave `X_teacher` pageable. Rollback: drop `.pin_memory()`, keep `non_blocking=False`. |
| 3 | **Miss-detection contract regression** — a partial/lazy cache silently mixing cached + live targets. | Validation stays eager & all-or-nothing (design step 1). NEW test `test_partial_cache_still_all_or_nothing_miss` is the explicit gate. Rollback: revert to eager materialize (current code). |
| 3 | `load_block_hidden` may not support cheap metadata peek → can't avoid the 21 s stall. | Fall to alternative design (b): eager validate (accept stall), lazy materialize (kill the 52 GB held-RAM, the bigger win). |
| 3 | Background prefetch thread races / holds an extra layer in RAM. | Prefetch is OPTIONAL and deferred; one-deep only (≤2 layers resident). Recommend shipping lazy-materialize without prefetch first. |
| 4 | `grouped` empty/None at snapshot time → narrowing drops everything. | `if grouped` guard falls back to full snapshot. Verified `grouped` is set (orch 608) before the phase (orch 1568); the guard is defensive only. |
| 4 | Existing `set(snap.keys()) == set(range(n))` test breaks. | Optional `members=None` default preserves the full-snapshot signature; the existing test passes unchanged. |

**Global rollback:** each lever is an independent commit; revert any one without affecting the others. The default-OFF gates mean a bad lever cannot affect a run that hasn't enabled the plugin.

---

## Testing plan on the host RTX 5080 (16 GB)

1. **CPU-only tests** (Levers 3 + 4 + the equivalence harness's CPU mode): `pytest` the listed files. No VRAM concern.
2. **Lever 1 VRAM budget check (BEFORE enabling on a real run):** compute resident bytes from the live recovery config (`n_batches * batch_size * seq_len * hidden_size * 2 * 2` for both lists) and compare to `nvidia-smi` free VRAM with the student+teacher layer resident. If the equivalence test fixture is tiny it won't reveal OOM — so this is a manual arithmetic gate, documented in the implementation PR, not a unit test.
3. **Equivalence test on GPU** (if the host can run the tiny fixture on CUDA): run `test_stage3_block_refine_optimized_equiv.py` with `device=cuda` to exercise the real H2D path; `torch.equal` must hold. The tiny fixture's working set is KB-scale, so no OOM risk there.
4. **Golden non-regression:** `test_stage3_golden_snapshot.py` (block_refine OFF) must stay byte-identical — proves the OFF path is untouched.
5. Capture the Lever-1/2 golden artifact from `origin/main` (this base) on THIS host before any block_refine edit, so seed and verify share wheel + platform (existing golden determinism caveat).

---

## Section-by-section outline (for the report)

1. Tier-1 contract
2. Lever 1 — block_refine per-step re-upload hoist (current 509–512, after-sketch, ~4.3 GB resident budget on 16 GB host, pinned-once fallback, free at 598)
3. Lever 2 — block_refine pinned + non_blocking H2D (producers 294/503/624/629, bounds pinned RAM to the block working set, stacks under Lever 1)
4. Lever 3 — block_hidden_cache lazy/prefetch (eager-validate / lazy-materialize, preserves all-or-nothing miss contract, optional one-deep prefetch deferred)
5. Lever 4 — expert_distill snapshot→members (helper 489–515 + hook 999–1008; `grouped` ALREADY in reads line 896; narrow via `set().union(*grouped.values())`)
6. Ordering (4 → 3 → 1 → 2)
7. Verification gates (Lever 4 equality test; Lever 3 lazy+all-or-nothing tests; Levers 1+2 NEW bit-identity golden harness — the real gate; no-monkeypatch toggle/golden capture from origin/main)
8. Risks & rollback (Lever-1 GPU budget; Lever-3 miss contract; per-lever independent revert)
9. Host RTX 5080 testing plan
