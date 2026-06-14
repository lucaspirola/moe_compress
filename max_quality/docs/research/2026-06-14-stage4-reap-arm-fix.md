# Stage-4 (EoRA) audit for the live reap-s234 arm — crash / OOM / degrade

**Date:** 2026-06-14
**Branch:** `research/stage4-reap-arm-fix` (does NOT touch main; the live run is untouched)
**Trigger:** time-sensitive — the live reap-s234 arm is in Stage 3 now, ~1-2 h from Stage 4.

## Live run config (as launched)

The four key knobs are NOT in any committed YAML. They are synthesized at launch by
`run_reap_ream_35pct.py::build_arm_config` from the box launcher's CLI flags
(`max_quality/scripts/box_run_ablation.sh`):

```
--whitening-cov shift --num-gpus 2 --num-sequences 4000 --only reap-s234,ream-s234
```

- `stage4_eora.whitening_cov = "shift"` + `stage3_svd.persist_shift_covariance = true`
  — injected at `run_reap_ream_35pct.py:543-555` (gated on `--whitening-cov shift`).
- `multi_gpu.factor_workers = 2`, `multi_gpu.alpha_workers = 2`, `multi_gpu.eora_workers = 2`
  — injected at `run_reap_ream_35pct.py:579-582` (`= num_gpus`, unconditionally overwrites
  the template's `eora_workers: 4`).
- `model.device_map = balanced` — set only in the **uncommitted** on-box
  `/root/work/box_reap.yaml` model block (the runner passes it through verbatim at
  `run_reap_ream_35pct.py:897`; the committed template default is `auto`).
- Arm `reap-s234`: `method="faithful_prune"`, seed repo `pirola/reap-s234-stage2p5-final`,
  `stage_windows=((3,6),)` → resumes the seeded `stage2p5_final` from HF and runs
  **Stage 3→6 only** (no in-run Stage 2 / 2.5).
- cov collection stays **1-GPU / in-process** even at `--num-gpus 2` (cov-DP and profile-DP
  were dropped in commit `57171f5` as fp-divergent), so `_dp_replicas <= 1`.

The `30pct` lineage config also has `aa_svd.cross_covariance: true` and
`block_refine.enabled: true` (config `qwen36_35b_a3b_30pct_multigpu.yaml:326,333`).

Where each knob is READ in code:
- `eora_workers` → `stage4/orchestrator.py:69-85` (`_resolve_eora_workers`), set on ctx at `:120`.
- `factor_workers` → `stage3/orchestrator.py:107-124` (`_resolve_factor_workers`), set at `:878`,
  consumed in `stage3/plugins/aa_svd_factor.py:630`.
- `whitening_cov` → `stage4/plugins/eora_inputs.py:363`; lookup selection
  `stage4/plugins/eora_compensation.py:610-632`.
- `persist_shift_covariance` → `stage3/orchestrator.py:969`.

---

## RISK 1 — shift-cov dependency (highest crash risk) → **VERDICT: SAFE (will not crash)**

`whitening_cov=shift` makes Stage 4 raise `FileNotFoundError` if
`artifacts_dir/_stage3_shift_covariance.pt` is absent
(`stage4/plugins/eora_inputs.py:363-372`). The question is whether Stage 3 (running now,
`persist_shift_covariance=true`) actually writes that file with content.

**Sequencing — confirmed correct (`stage3/orchestrator.py` `run()`):**
`collect_covariances` (`:534`) → factor loop (`:883`, prefetcher set `:880-881`) →
`refine_blocks` Phase C.5 (`:930`) → finalize → `_consolidate_shift_covariance` (`:969-978`)
→ **then** `rmtree(_stage3_bcov_partial)` (`:979-981`). The comment at `:968` is explicit:
`# MUST run BEFORE the _stage3_bcov_partial rmtree.`

**Do the per-layer B-cov spills still exist (with content) at the consolidate call?** YES.
- In-process collection spills **per-layer, at each window's end** and blocks on every spill
  future before returning: `covariance_collection.py:1030-1041` (`spill_executor.submit(
  B_acc.spill_layer_to_disk, ...)`) + the `finally` join at `:1069-1076`
  ("All cov layer spills durable on disk.").
- `spill_layer_to_disk` writes atomically (`torch.save(tmp)` → `os.replace`,
  `activation_hooks.py:1168-1169`) and only evicts the in-RAM dict (`:1187-1188`) — it
  never deletes the `.pt`.
- Factoring does NOT consume-and-delete: `aa_svd_factor.py` has **zero**
  `unlink`/`os.remove`/`rmtree`; after each layer it calls only `B_acc.unload_layer(...)`
  (`aa_svd_factor.py:838`), which is RAM-only (`activation_hooks.py:1296-1300`).
- `BcovLayerPrefetcher` (`activation_hooks.py:1318-1397`) is read-only:
  `_read_layer_payload` (`:1196-1236`) does `torch.load` and never unlinks. The only
  `unlink` in the file (`:1172`) is `.tmp` cleanup on a *failed* save.
- `_consolidate_shift_covariance` (`covariance_collection.py:211-244`) merges the per-layer
  `covariance` dicts verbatim and returns `len(merged) > 0` for a real run.

**Key format matches** (a silent mis-key would have been a second crash vector — it is not one):
- Spill key tuple = `(layer_idx, expert_idx, matrix_name)` (`activation_hooks.py:1023`,
  dataclass field `:971`). Values are `[d,d]` tensors.
- matrix_name strings present: **`"gate_proj"` and `"down_proj"` only**; `up_proj` is aliased
  to `gate_proj` upstream — `InputCovarianceAccumulator.update` early-returns on `up_proj`
  when `_alias_gate_up` (default True): `activation_hooks.py:1001-1002,974`.
- Stage-4 lookup builds `key=(layer,e,name)` and redirects up_proj → `(layer,e,"gate_proj")`
  (`eora_compensation.py:816-827`). For down_proj it looks up `(layer,e,"down_proj")`.
- `down_proj` cov IS captured (unconditional in Stage 3: `intermediate_cb` always wired,
  `covariance_collection.py:893`; `capture_experts` defaults `capture_intermediate=True`,
  `activation_hooks.py:1581`; only the teacher sets it False, `:908`). So every Stage-4
  lookup resolves to a key the shift cov actually contains.
- The anchor `A_cov` uses the identical key convention (same accumulator class; loaded the
  same way at `eora_inputs.py:221`), so anchor↔shift cannot mis-key.

**Manifest guard** also present: shift load validates `.MANIFEST.json` (torn-write guard,
`eora_inputs.py:373-392`); `_consolidate_shift_covariance` writes that manifest
(`covariance_collection.py:235-243`).

**Conclusion:** For this fresh Stage-3 run (in-process cov, cross_cov + block_refine on,
persist on), the spill dir is fully populated when consolidate runs; a non-empty
`_stage3_shift_covariance.pt` (+ manifest) is written before the rmtree. Stage 4 finds it,
loads a populated dict, and does NOT hit the empty-cov `RuntimeError`
(`eora_compensation.py:622-626`). **No crash. No fix needed.**

**The ONLY way this crashes** (does not apply here): the spill dir is empty at `:972`
(manually wiped on a partial resume, or a future code change that frees spills during
factoring) → `_consolidate_shift_covariance` writes a 0-key cov → Stage 4 loads it (file
exists, so no `FileNotFoundError`) → `_resolve_whitening_lookup` raises `RuntimeError`
("`whitening_cov='shift'` but no shift_cov was loaded", `eora_compensation.py:622-626`).
Worth a one-line operator check (below) but not a code change.

---

## RISK 2 — `device_map=balanced` + `eora_workers=2` (never live-validated) → **VERDICT: SAFE, low OOM risk; one robustness gap (needs-fix, future-only)**

Stage 4 does NOT reload the model; it operates on the in-memory student already sharded by
`balanced` across cuda:0/cuda:1. EoRA widens `FactoredExperts.U/V` in place.

**Device placement (traced):**
- Per layer, `dev = fe.gate_proj_U.device` (`eora_compensation.py:753`,
  `stage4/orchestrator.py:179`) — the device this layer's experts live on under the balanced
  shard (cuda:0 OR cuda:1, layer-dependent).
- `eora_workers=2` → `effective_workers = min(2, len(eligible))` (`:835`). With
  `worker_devices=None` (the orchestrator sets only `eora_workers`, NOT
  `eora_worker_devices` — `stage4/orchestrator.py:120`), `_resolve_worker_devices` returns
  `[cuda:0, cuda:1]` unconditionally (`eora_compensation.py:585-586`).
- So band 0 solves on cuda:0, band 1 on cuda:1 — **independent of where `dev` (the layer's
  home) is**. `_solve_expert_tile` relocates every input to `target_device`: `W_orig`
  (CPU-resident, `eora_inputs.py:300`) and `U_e/V_e` (on `dev`) via `.to(target_device)`
  (`eora_compensation.py:444-447`); the cov `A` is moved to `target_device` inside
  `_eigh_spectrum`/`_compute_eora_factors` (`:213,:293`); a memoized gate spectrum is
  relocated to `delta.device` (`:327-331`). Results gather back to `dev` on the worker thread
  (`_solve_one`, `:880-881`). Final ascending-e assembly into `U_corr/V_corr` is on `dev`
  (`:893-897`). **All cross-device hops are explicit `.to()` — functionally correct.**

**Memory / OOM:** the per-expert transients are tiny. `A` is `[d_in,d_in]` fp32
(d_in = hidden ≈ 2048 for gate/up → 16 MB; ≈ moe_intermediate for down → smaller); `delta`
is `[d_out,d_in]` fp32; eigh/Gram (`eora_compensation.py:355-373`) operate on `[n_keep,n_keep]`
or `[d_out,d_out]`. With 2 worker threads each holding one expert's transients
(O(tens of MB) ×2 concurrent), the added VRAM is sub-GB per device on top of the resident
balanced shard (~25 GB/GPU for a ~49 GB student on 2× H200 141 GB). **Will fit comfortably;
no OOM.** No full-model-per-device duplication occurs — workers only relocate single-expert
tensors, never the model.

**Balanced-vs-worker interaction — the one real gap (robustness, not a live crash):**
`_resolve_worker_devices` assumes CUDA devices `cuda:0..cuda:(n-1)` exist and are usable
(`eora_compensation.py:585-586`). On a 2× H200 box with both GPUs visible this holds. But if
the box ever exposes a *non-contiguous* or partial CUDA_VISIBLE_DEVICES, or `device_count()`
< the device a balanced layer actually lives on, the worker could target a device the shard
never used. On the current 2-GPU box this is benign (both devices exist; gathers are
explicit). **Not a live-run risk for reap-s234.** Future hardening (post-run): have the
orchestrator pass `eora_worker_devices` derived from the model's actual occupied devices
(`{p.device for p in model.parameters() if p.is_cuda}`) instead of relying on the
`cuda:0..n-1` assumption in `_resolve_worker_devices`. Code change → future runs only.

**Caveat from memory (`project_multigpu_stage3_landed`):** the N-GPU task-parallel EoRA
(`e395ad0`) was never live-validated on real ≥2 GPU. The code path is byte-identical to
serial by construction (disjoint-row gather, ascending-e assembly — `eora_compensation.py:507-512`),
and the trace above shows every cross-device hop is explicit, so the risk is **low**. The
live run is itself the first validation; if it fails it will be a loud device-placement error
at the first widened layer, not silent corruption (the `f.result()` re-raise at
`eora_compensation.py:550-551` surfaces any worker exception).

---

## RISK 3 — `whitening_cov=shift` correctness on the reap arm → **VERDICT: SAFE (correct cov, no key mismatch)**

- The shift cov is the **post-2.5 student B = S = X'ᵀX'** (the acov paper-fix), consolidated
  from the same B-cov spills the AA-SVD factoring used — NOT the stale anchor. Source:
  `_consolidate_shift_covariance` reads `_stage3_bcov_partial/layer_*.pt`
  (`covariance_collection.py:211-244`, docstring `:214` "the post-2.5 shift cov S=X'ᵀX'").
- Loaded into the `shift_cov` ctx slot only when `whitening_cov ∈ {shift, anchored_adaptive}`
  (`eora_inputs.py:364-396`); for `shift`, `_resolve_whitening_lookup` returns the dict
  directly (`eora_compensation.py:621-626`) and raises loudly if it is empty.
- Applied in the EoRA √Λ basis: `whitening_lookup.get(cov_key)` → `A` →
  `_eigh_spectrum(A)` → `Q' = Q·√Λ` (`eora_compensation.py:827,457,338`). This matches
  upstream EoRA whitening with the sequentially-compressed shift input (eora.py:478, per the
  module docstring `:77-91`).
- **No per-expert key mismatch:** shift-cov keys are exactly
  `(layer,e,"gate_proj")` and `(layer,e,"down_proj")`; Stage-4 iteration looks up
  gate→`gate_proj`, up→`gate_proj` (redirected), down→`down_proj` — all present (Risk 1
  key analysis). down_proj cov is captured, so down_proj whitening is real (not the
  `A=None` plain-SVD fallback).
- **Graceful degrade if a key were ever missing** (does not occur here): `dict.get` returns
  `None` → `_compute_eora_factors` short-circuits to `_plain_svd_padded` (unweighted SVD,
  zero-padded) at `eora_compensation.py:308-309` — degrade, not crash.

---

## RISK 4 — seeded reap arm (resume at Stage 3, no in-run 2.5) deltas Stage 4 might assume → **VERDICT: SAFE**

- Stage 4 reads NO Stage-1/2.5 artifacts. Its inputs are: `_stage3_original_weights.pt`
  (`originals`, `eora_inputs.py:300`), `_stage2_input_covariance.pt` (anchor `A_cov`, only
  consulted under `whitening_cov=anchor` / `anchored_adaptive`), and — for this arm —
  `_stage3_shift_covariance.pt`. The seeded `stage2p5_final` provides the student; Stage 3
  produces all three Stage-4 inputs in-run. No assumption about an in-run Stage 2.5 exists.
- `originals` is the Stage-3 pre-factor weights, written by Stage 3 regardless of how the
  student was seeded. The faithful-prune (`reap-s234`) method changes which experts exist,
  but the per-`(layer,e,name)` originals/cov keying is expert-index-based and self-consistent
  within the run.
- Resume safety: Stage 4 has its own `_stage4_partial` crash-resume
  (`stage4/orchestrator.py:182-215`) independent of the Stage-3 seed.

---

## Concrete fixes & restart implications

| Risk | Verdict | Fix | Restart? |
|------|---------|-----|----------|
| 1 shift-cov dependency | SAFE | None. (Optional operator check below.) | n/a |
| 2 balanced + eora_workers=2 | SAFE; low OOM | None for the live run. Future hardening: pass `eora_worker_devices` from the model's occupied devices instead of `cuda:0..n-1`. | future-only |
| 3 shift whitening correctness | SAFE | None. | n/a |
| 4 seeded-arm deltas | SAFE | None. | n/a |

**No fix forces a restart-from-Stage-3.** No live crash/OOM was found. Because the config and
modules are already read/imported by the running subprocess, ANY config or code change would
require a restart anyway — and none is warranted.

### Optional zero-risk operator check (no restart, run on the box when Stage 3 finishes)

Before Stage 4 starts, confirm the shift cov was written non-empty:

```bash
python3 - <<'PY'
import torch, glob, os
p = glob.glob(os.path.join(os.environ.get("ARTIFACTS_DIR","."), "**/_stage3_shift_covariance.pt"), recursive=True)
assert p, "shift cov NOT written — Stage 4 will FileNotFoundError"
d = torch.load(p[0], map_location="cpu", weights_only=False)
cov = d.get("covariance", {})
assert len(cov) > 0, "shift cov EMPTY — Stage 4 will RuntimeError"
mats = {k[2] for k in cov}
print(f"OK: {len(cov)} keys, matrices={sorted(mats)}")  # expect gate_proj + down_proj
PY
```

If this passes, Stage 4 on the reap-s234 arm is clear on all four risks.

---

## Evidence index (file:line)

- shift-cov FileNotFoundError gate: `stage4/plugins/eora_inputs.py:363-372`
- shift-cov manifest validation: `eora_inputs.py:373-394`
- consolidate-before-rmtree: `stage3/orchestrator.py:968-981`
- `_consolidate_shift_covariance`: `stage3/plugins/covariance_collection.py:211-244`
- per-layer spill + join: `covariance_collection.py:1030-1041,1069-1076`
- `spill_layer_to_disk` (atomic, RAM-only evict): `utils/activation_hooks.py:1135-1194,1187-1188`
- `unload_layer` (RAM-only): `activation_hooks.py:1296-1300`; factor calls it: `aa_svd_factor.py:838`
- `BcovLayerPrefetcher` (read-only): `activation_hooks.py:1318-1397`, read `:1196-1236`
- spill key tuple: `activation_hooks.py:1023,971`; up_proj alias: `:1001-1002,974`
- down_proj capture default: `activation_hooks.py:1581`; wiring `covariance_collection.py:893`
- Stage-4 cov lookup keys: `eora_compensation.py:816-827`
- whitening lookup select + empty-cov RuntimeError: `eora_compensation.py:610-632,621-626`
- `_eigh_spectrum` (A→device, √Λ): `eora_compensation.py:213-238`
- `_compute_eora_factors` (A=None → plain SVD): `eora_compensation.py:293,308-309`
- `_solve_expert_tile` (inputs → target_device): `eora_compensation.py:444-447,457,464-467`
- worker device resolution: `eora_compensation.py:571-587`; bands `:835-869`
- `_run_expert_bands` (threaded, re-raise): `eora_compensation.py:482-568`
- `_resolve_eora_workers`: `stage4/orchestrator.py:69-85,120`
- orchestrator does NOT set `eora_worker_devices`: `stage4/orchestrator.py:120`
- run override builder: `run_reap_ream_35pct.py:543-555,579-582`; arm spec `:214-227`
- launcher CLI: `max_quality/scripts/box_run_ablation.sh`
- `factor_workers` resolve/consume: `stage3/orchestrator.py:107-124,878`; `aa_svd_factor.py:630`
