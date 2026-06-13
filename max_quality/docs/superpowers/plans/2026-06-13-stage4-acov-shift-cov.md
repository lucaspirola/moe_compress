# Stage-4 EoRA shift-cov whitening — implementation plan

**Branch:** `feat/stage4-acov-shift-cov` (worktree `/home/lucas/ai/wt-acov`)
**Code root:** `max_quality/`
**Spec source (read fully):** `max_quality/docs/research/2026-06-13-acov-capture-point.md`

## 0. Goal & converged verdict (from the spec)

Upstream EoRA (`NVlabs/EoRA::llama_sequential_eigen` @ `6a42e2e`,
`eora.py:447-466,512,518-526`) whitens its low-rank correction `ΔW` with the
**sequentially-compressed SHIFT** input `X'`, NOT the original anchor. Our
Stage-4 EoRA whitens with the frozen **original-calibration anchor `A`** (the
Stage-2 sidecar `A_cov`), which is a measured **deviation** from upstream EoRA.

The post-2.5 shift covariance `X'ᵀX'` is **already collected** by Stage-3's
live dual-forward — it is the student-side **B-cov** accumulator (`B_acc`,
code symbol for paper-`S = X_postᵀX_post`), computed per `(layer, expert,
matrix)` on the live post-2.5 compressed student
(`stage3/plugins/covariance_collection.py:486-499,639-661`). It is what
upstream EoRA actually whitens with, and Stage-4 currently never receives it.

This plan makes EoRA's whitening covariance **opt-in selectable** via
`stage4_eora.whitening_cov`:
- `"anchor"` (DEFAULT) → current behaviour, **byte-identical** (no golden regen).
- `"shift"` → whiten with the post-2.5 shift cov (upstream-EoRA-faithful).
- `"anchored_adaptive"` → both anchor + shift, AA-SVD-consistent (see §7 scope).

Keep the original `A` capture untouched — it is AA-SVD's anchor, consumed by
Stage-3 factorization (spec §5.1). We are **adding** a shift cov for Stage-4,
not removing the anchor.

## 1. Verified plumbing (cite file:line)

### 1.1 Where the shift cov is produced (Stage 3)
- `B_acc : InputCovarianceAccumulator` accumulates `S = X_postᵀX_post` per
  `(layer_idx, expert_idx, matrix_name)` on the **student** forward:
  - student-side update: `input_cb` →
    `B_acc.update_grouped(li, e, "gate_proj", …)` (`covariance_collection.py:657`)
    and `intermediate_cb` → `B_acc.update_grouped(li, e, "down_proj", …)`
    (`covariance_collection.py:783`). `up_proj` is **aliased to gate_proj**
    inside the accumulator (`activation_hooks.py:1001-1002` early-return on
    `up_proj`; `get()` gate→up fallback `:1310-1314`).
  - Per-layer it is `finalize_layer`d and `spill_layer_to_disk`'d to
    `artifacts_dir / "_stage3_bcov_partial" / layer_{idx}.pt`
    (`covariance_collection.py:996,1003`).
- Spill payload format (`activation_hooks.py:1157-1161`):
  ```python
  payload = {
      "format_version": 1,
      "covariance": snapshot,        # {(layer,expert,matrix) -> Tensor[d,d]}
      "tokens": {k: token_count…},
  }
  ```
  Stored dtype = `B_acc.storage_dtype` (= `stage3_svd.bcov_storage_dtype`,
  default `bfloat16`; `orchestrator.py:214`).
- **CRITICAL — the spill dir is DELETED on Stage-3 success**
  (`stage3/orchestrator.py:887-889`):
  ```python
  if (artifacts_dir / "_stage3_bcov_partial").exists():
      shutil.rmtree(artifacts_dir / "_stage3_bcov_partial", …)
  ```
  So today the shift cov does NOT survive past Stage 3. We must **persist a
  named copy** before that cleanup runs.

### 1.2 Where the whitening cov is consumed (Stage 4)
- `EoraInputsPlugin.load_eora_inputs` loads `A_cov` (anchor) onto the ctx
  (`eora_inputs.py:347`); `writes` includes `"A_cov"` (`:93-96`).
- `EoraCompensationPlugin.compensate_layer` reads `"A_cov"` off ctx
  (`eora_compensation.py:620,678`) and fetches the per-expert whitening matrix
  at the single swap point `_inputs_for(e)` (`eora_compensation.py:754-766`):
  ```python
  cov_key = (ref.layer_idx, e, "gate_proj") if name == "up_proj" else key
  A = A_cov.get(cov_key)
  return W_orig, U_e, V_e, A
  ```
- `A` then flows through `_solve_expert_tile(…, A, …)`
  (`eora_compensation.py:811-816`) → `_eigh_spectrum(A, …)`
  (`:451`) / `_compute_eora_factors(delta, A, …)` (`:458`). The whitening cov
  is **fully decoupled** from the target `delta = W_orig − U·V`
  (`_solve_expert_tile:438-442`). So swapping `A` swaps ONLY the whitening
  basis — the target stays `W_orig`-anchored (upstream-faithful; spec §3).

### 1.3 Key/shape compatibility (verified)
- `A_cov`, `B_acc.covariance`, and the proposed `shift_cov` all use the
  IDENTICAL key tuple `(layer_idx, expert_idx, matrix_name)`
  (`activation_hooks.py:1023`) and `[d,d]` shape. All store only `gate_proj`
  + `down_proj`; `up_proj` is served by the gate→up fallback
  (`aa_svd_factor.py:408-415 _cov_lookup`; `InputCovarianceAccumulator.get`
  `:1310-1314`). So a `shift_cov` dict plugs into `_inputs_for`'s existing
  `cov_key`-rewrite logic unchanged.

### 1.4 Stage3→Stage4 artifact handoff
- Both stages receive the SAME `artifacts_dir` (`run_pipeline.py:305,317`).
  Precedent: `_stage3_original_weights.pt` is written by Stage 3
  (`stage3/orchestrator.py:654,666`) and read by Stage 4
  (`eora_inputs.py:222,296`). The shift cov follows the exact same pattern.

## 2. Design decisions

### D1 — Stage-3 ride-along: persist the shift cov as a named artifact
Add a Stage-3 step (config-gated, default OFF for byte-identity of *artifacts
on disk* — but see D5) that, **before** the `_stage3_bcov_partial` cleanup,
copies the finalized per-layer B-cov spills into a durable named artifact
`artifacts_dir / "_stage3_shift_covariance.pt"` (single consolidated `.pt`,
mirroring `_stage2_input_covariance.pt`'s `{"covariance": {...}}` layout +
an atomic manifest sibling, mirroring `_stage3_original_weights.pt`).
**Near-zero cost:** the Grams already exist on disk in `_stage3_bcov_partial/`;
this is a read-spills + consolidate + atomic-save, no extra forward pass.

Rationale for a fresh consolidated `.pt` (vs. just *not deleting* the spill
dir): the spill dir is bf16, per-layer, and named/structured for crash-resume;
a single named artifact with a manifest is the durable cross-stage contract
(matches every other Stage→Stage handoff and gets the torn-write guard for
free).

### D2 — Config knob `stage4_eora.whitening_cov`
Read in `compensate_layer` via the existing `.get(key, default)` pattern
(`eora_compensation.py:687` shows the precedent `s4.get("log_residuals",
False)`):
```python
whitening_cov = str(s4.get("whitening_cov", "anchor"))
```
Values: `"anchor"` (default) | `"shift"` | `"anchored_adaptive"`. Unknown
value → raise `ValueError` (loud, like `cov_capture_mode`
`covariance_collection.py:546-550`).

### D3 — Stage-3 emission gate `stage3_svd.persist_shift_covariance`
The Stage-3 ride-along (D1) is gated by `stage3_svd.get(
"persist_shift_covariance", False)`. Default **False** → Stage 3 behaves
exactly as today (writes nothing new, deletes the spill dir as before) →
Stage-3 golden untouched. An A/B run sets it `True` to emit the artifact AND
sets `stage4_eora.whitening_cov: "shift"` to consume it.

### D4 — Stage-4 loads the shift cov only when needed
`EoraInputsPlugin.load_eora_inputs` loads `_stage3_shift_covariance.pt` into a
new ctx slot `"shift_cov"` ONLY when `stage4_eora.whitening_cov` ∈ {`"shift"`,
`"anchored_adaptive"`}. When `"anchor"` (default), the load is skipped entirely
→ no new I/O, no behaviour change. If the knob requests shift but the artifact
is absent → raise `FileNotFoundError` (loud; an operator asked for shift
whitening and the Stage-3 ride-along didn't run).

### D5 — How DEFAULT stays byte-identical (the golden gate)
1. `stage4_eora.whitening_cov` absent → `.get(..., "anchor")` → `"anchor"`.
2. In `"anchor"` mode, `compensate_layer` builds `whitening_lookup = A_cov`
   (the SAME object) and `_inputs_for` does `whitening_lookup.get(cov_key)` —
   identical bytes to today's `A_cov.get(cov_key)`.
3. `EoraInputsPlugin` skips the shift load (D4) → `"shift_cov"` slot never set.
4. Stage-3 `persist_shift_covariance` absent → False → no artifact written,
   spill dir deleted as before.
⇒ Stage-3 golden (`tests/test_stage3_golden_snapshot.py`) and Stage-4 golden
(`tests/test_stage4_golden_snapshot.py`, both fp32+bf16 variants) are
**unchanged — NO regen**. The golden configs embed `config["stage4_eora"]`
verbatim into `eora_ranks.json`; since the default config files do NOT add a
`whitening_cov` key, the serialized block is byte-identical. (We do NOT add
`whitening_cov` to the default golden test config — see §5 Task 7.)

### D6 — `whitening_lookup` is a thin selector, not a kernel change
The EoRA kernel (`_compute_eora_factors`, `_eigh_spectrum`,
`_solve_expert_tile`) is **untouched** — it already takes `A` as an opaque
`[d,d]` covariance. The ONLY change is *which dict* `_inputs_for` reads `A`
from. This keeps the result-changing surface minimal and the diff reviewable.

## 3. The shift-cov plumbing (exact)

```
Stage 3 (persist_shift_covariance: true):
  … existing dual-forward fills B_acc per layer, spills to _stage3_bcov_partial/ …
  [NEW] before rmtree(_stage3_bcov_partial):
     consolidate spills -> artifacts_dir/_stage3_shift_covariance.pt
       payload = {"format_version": 1, "covariance": {(l,e,m)->Tensor[d,d]}}
       + atomic_torch_save + write_manifest_last (schema_version=1)
  … existing rmtree(_stage3_bcov_partial) …

Stage 4 (whitening_cov: "shift"):
  EoraInputsPlugin.load_eora_inputs:
     [NEW] if whitening_cov in {"shift","anchored_adaptive"}:
        validate manifest -> load _stage3_shift_covariance.pt
        ctx.set("shift_cov", payload["covariance"])
  EoraCompensationPlugin.compensate_layer:
     [NEW] resolve whitening_lookup from whitening_cov:
        "anchor"            -> A_cov
        "shift"             -> shift_cov
        "anchored_adaptive" -> AnchoredAdaptiveLookup(A_cov, shift_cov)  (§7)
     _inputs_for(e): A = whitening_lookup.get(cov_key)   # was A_cov.get(cov_key)
```

## 4. Out of scope (note explicitly in the plan)
- **The A/B experiment itself** (arm A=anchor vs arm A*=shift, scored by
  Stage-6-alt thermometer + `stage4_eora.log_residuals` drop) needs a GPU and
  a full Stage-3+4 run. It is **run separately**; this branch only lands the
  opt-in machinery + tests.
- **The REAM merge-anchor mis-attribution bugfix** (spec §3 merge case;
  `stage2/shared_io.py:308-326` copies one constituent's Gram verbatim into a
  merged slot) is a **separate Stage-2 branch** — explicitly NOT in this
  Stage-3+4 branch.
- **`anchored_adaptive` full AA-SVD closed form** (anchor `A=X` + shift `B=X'`
  + cross `C=XX'ᵀ` via `M=W·C·S⁻¹·R`) is a generalization beyond vanilla
  EoRA. This plan includes the *config value + a defensible reduction*
  (§7) but the core deliverable + A/B is **anchor vs shift**. If the full
  AA-form is wanted, it is a follow-up that reuses Stage-3's `C_acc`
  (`covariance_collection.py:496-503`) — flagged as an open question (§9).

---

## 5. Bite-sized TDD tasks

> Run all test commands from `max_quality/`. Each task: write/adjust the test
> FIRST (red), then implement (green). Keep the default path byte-identical at
> every step.

### Task 0 — baseline: capture current golden green
**Command:**
```bash
cd max_quality
pytest tests/test_stage3_golden_snapshot.py tests/test_stage4_golden_snapshot.py -v
```
Confirm both pass BEFORE any change. This is the byte-identity baseline the
whole plan must preserve. (No code change.)

### Task 1 — Stage-3: consolidate-and-persist helper (unit, pure)
**TDD:** new test `tests/test_stage3_shift_cov_persist.py`:
- Build two fake per-layer spill files in a tmp dir using
  `InputCovarianceAccumulator.spill_layer_to_disk` (format_version 1, a couple
  of `(layer, expert, "gate_proj"|"down_proj")` Grams).
- Call the new helper `_consolidate_shift_covariance(spill_dir, out_path,
  moe_layer_indices, storage_dtype)`.
- Assert: `out_path` exists, its manifest sibling exists, `torch.load(out_path)
  ["covariance"]` has the EXACT union of keys across the spill files, each
  tensor `[d,d]`, dtype == requested storage dtype.

**Implement** in `stage3/plugins/covariance_collection.py` (module-level, near
`_reduce_spilled_cov_dirs`):
```python
def _consolidate_shift_covariance(
    spill_dir: Path, out_path: Path, layer_indices, *, storage_dtype,
) -> int:
    """Consolidate per-layer B-cov spills (the post-2.5 shift cov S=X'ᵀX')
    into one durable named artifact for Stage-4 EoRA shift whitening.

    Reads each ``spill_dir/layer_{idx}.pt`` (format_version 1, keyed
    (layer,expert,matrix) -> Tensor[d,d]; up_proj aliased to gate_proj
    upstream), merges the per-layer ``covariance`` dicts, and atomically
    saves ``{"format_version": 1, "covariance": {...}}`` + a manifest sibling.
    Near-zero cost: pure disk read+merge, no forward pass. Returns key count.
    """
    merged: dict = {}
    for li in layer_indices:
        p = Path(spill_dir) / f"layer_{li}.pt"
        if not p.exists():
            continue
        payload = torch.load(p, map_location="cpu", weights_only=True)
        cov = payload.get("covariance", {}) if isinstance(payload, dict) else {}
        for k, t in cov.items():
            merged[k] = t.to(storage_dtype)
    from moe_compress.utils.atomic_io import atomic_torch_save, write_manifest_last
    atomic_torch_save(out_path, {"format_version": 1, "covariance": merged})
    manifest = out_path.with_suffix(out_path.suffix + ".MANIFEST.json")
    try:
        manifest.unlink(missing_ok=True)
    except OSError:
        pass
    write_manifest_last(out_path, manifest, schema_version=1,
                        extra_meta={"n_keys": len(merged),
                                    "artifact": "stage3_shift_covariance"},
                        compute_sha256=False)
    return len(merged)
```
**Command:** `pytest tests/test_stage3_shift_cov_persist.py -v`

### Task 2 — Stage-3 orchestrator: gated ride-along call
**TDD:** add a test in the same file (or
`tests/test_stage3_orchestrator.py` if it exists) that runs the orchestrator
finalize path with `stage3_svd.persist_shift_covariance: true` on the tiny
fixture and asserts `_stage3_shift_covariance.pt` + manifest exist after run;
and with the key absent/false, asserts the file does NOT exist (default
byte-identity).

**Implement** in `stage3/orchestrator.py`, INSIDE the success-cleanup block,
**before** the `_stage3_bcov_partial` rmtree (currently line 887):
```python
# Shift-cov ride-along (opt-in): persist the post-2.5 per-expert input Gram
# (B_acc = S = X'ᵀX', the post-2.5 SHIFT cov) as a durable artifact for
# Stage-4 EoRA shift whitening (upstream-EoRA-faithful). Default OFF →
# byte-identical (no artifact written, spill dir deleted below as before).
if bool(s3.get("persist_shift_covariance", False)):
    from .plugins.covariance_collection import _consolidate_shift_covariance
    _shift_path = artifacts_dir / "_stage3_shift_covariance.pt"
    _n = _consolidate_shift_covariance(
        bcov_spill_dir, _shift_path,
        [ref.layer_idx for ref in moe_layers],
        storage_dtype=B_cov_dtype,
    )
    log.info("Stage 3: persisted post-2.5 shift covariance (%d keys) -> %s",
             _n, _shift_path)
# … existing rmtree(_stage3_bcov_partial) follows unchanged …
```
Note: place it where `bcov_spill_dir` and `moe_layers` are still in scope
(both are — see `orchestrator.py:255,126`) and BEFORE the rmtree at `:887`.
**Command:** `pytest tests/test_stage3_orchestrator.py tests/test_stage3_shift_cov_persist.py -v` (use whichever orchestrator test file exists; the agent confirmed `tests/test_stage3_golden_snapshot.py` runs the full S0→S4 pipeline so it also exercises this path with the default-off config → must stay green).

### Task 3 — Stage-4 inputs: load shift_cov when requested
**TDD:** new test `tests/test_stage4_shift_cov_inputs.py`:
- Write a fake `_stage3_shift_covariance.pt` (+ manifest via the same atomic
  writer) into a tmp `artifacts_dir`.
- Drive `EoraInputsPlugin.load_eora_inputs` (mirror the fixture style in
  `tests/test_stage4_plugin_inputs.py`) with config
  `stage4_eora.whitening_cov: "shift"`; assert `ctx.get("shift_cov")` is the
  loaded `{(l,e,m)->Tensor}` dict with the right keys/shapes.
- With `whitening_cov: "anchor"` (or absent), assert `ctx.has("shift_cov")` is
  **False** (load skipped).
- With `whitening_cov: "shift"` but the artifact ABSENT, assert it raises
  `FileNotFoundError`.

**Implement** in `eora_inputs.py`:
- Add `"shift_cov"` to `writes` (`:93-96`) — note it is conditionally set.
- At the end of `load_eora_inputs` (after the `A_cov`/originals block, near
  `:347`), add:
```python
s4 = config.get("stage4_eora", {})
whitening_cov = str(s4.get("whitening_cov", "anchor"))
if whitening_cov in ("shift", "anchored_adaptive"):
    shift_path = artifacts_dir / "_stage3_shift_covariance.pt"
    if not shift_path.exists():
        raise FileNotFoundError(
            f"stage4_eora.whitening_cov={whitening_cov!r} requires the "
            f"post-2.5 shift covariance at {shift_path}, but it is absent. "
            "Re-run Stage 3 with stage3_svd.persist_shift_covariance=true."
        )
    # Validate manifest (torn-write guard), mirror originals block :262-295.
    shift_manifest = shift_path.with_suffix(shift_path.suffix + ".MANIFEST.json")
    if shift_manifest.exists():
        from moe_compress.utils.atomic_io import (
            ManifestMismatchError, read_and_validate_manifest,
        )
        try:
            read_and_validate_manifest(shift_path, shift_manifest,
                                       expected_schema_version=1)
        except ManifestMismatchError as exc:
            raise RuntimeError(
                f"Stage 4: shift covariance manifest validation FAILED — {exc}. "
                f"Delete {shift_path.name} + {shift_manifest.name} and re-run "
                "Stage 3 with persist_shift_covariance=true."
            ) from exc
    shift_payload = torch.load(shift_path, map_location="cpu")
    ctx.set("shift_cov", shift_payload.get("covariance", {}))
    log.info("Stage 4: loaded post-2.5 shift covariance (%d keys) for "
             "whitening_cov=%r", len(ctx.get("shift_cov")), whitening_cov)
```
**Command:** `pytest tests/test_stage4_shift_cov_inputs.py -v`

### Task 4 — Stage-4 compensation: select the whitening lookup
**TDD:** new test `tests/test_stage4_whitening_cov_select.py`:
- Unit-level: assert that `compensate_layer` reads `A` from `shift_cov` when
  `whitening_cov="shift"` and from `A_cov` when `"anchor"`. The cleanest
  observable seam is to factor the selector into a tiny pure helper and test
  it directly (see implement below): `_resolve_whitening_lookup(whitening_cov,
  A_cov, shift_cov)` returns the right object and raises on unknown value.

**Implement** in `eora_compensation.py`:
- Add `"shift_cov"` to `compensate_layer.reads` (`:620`) as an optional slot.
- Add module-level helper:
```python
def _resolve_whitening_lookup(whitening_cov: str, A_cov, shift_cov):
    """Select the EoRA whitening covariance dict per stage4_eora.whitening_cov.

    'anchor' (default) -> the original-calibration anchor A_cov (byte-identical
    to historical behaviour). 'shift' -> the post-2.5 SHIFT cov (upstream-EoRA-
    faithful; spec 2026-06-13-acov-capture-point.md). 'anchored_adaptive' ->
    AnchoredAdaptiveLookup(A_cov, shift_cov) (see §7). Unknown value raises.
    """
    if whitening_cov == "anchor":
        return A_cov
    if whitening_cov == "shift":
        if not shift_cov:
            raise ValueError(
                "stage4_eora.whitening_cov='shift' but no shift_cov was loaded "
                "(EoraInputsPlugin did not populate it — wiring bug)."
            )
        return shift_cov
    if whitening_cov == "anchored_adaptive":
        return _AnchoredAdaptiveLookup(A_cov, shift_cov)   # §7
    raise ValueError(
        f"stage4_eora.whitening_cov must be 'anchor', 'shift', or "
        f"'anchored_adaptive', got {whitening_cov!r}"
    )
```
- In `compensate_layer`, after `s4 = config["stage4_eora"]` (`:686`):
```python
whitening_cov = str(s4.get("whitening_cov", "anchor"))
shift_cov = ctx.get("shift_cov") if ctx.has("shift_cov") else None
whitening_lookup = _resolve_whitening_lookup(whitening_cov, A_cov, shift_cov)
```
- In `_inputs_for` (`:765`), change ONE line:
```python
A = whitening_lookup.get(cov_key)   # was: A = A_cov.get(cov_key)
```
  `A_cov` (default `whitening_lookup` IS `A_cov`) → byte-identical.
**Command:** `pytest tests/test_stage4_whitening_cov_select.py -v`

### Task 5 — result-changing integration test (shift ≠ anchor, both valid)
**TDD:** new test `tests/test_stage4_shift_cov_result.py` (mirror the tiny-model
fixture in `tests/test_stage4_golden_snapshot.py` / `tests/conftest.py`):
- Run the tiny Stage-0→4 pipeline TWICE on the same fixture:
  arm-anchor (default) and arm-shift (`persist_shift_covariance: true` in S3 +
  `whitening_cov: "shift"` in S4). Use a SHIFT cov that genuinely differs from
  the anchor (the tiny pipeline already produces distinct post-2.5 B-cov; if
  the fixture's anchor==shift numerically, inject a fake differing
  `_stage3_shift_covariance.pt`).
- Assert: the two `eora_ranks.json` / widened U·V are **NOT byte-identical**
  (shift is result-changing by design) AND both are valid (finite, correct
  shapes, ranks within budget). This pins that the swap actually takes effect
  and produces a different-but-valid result.
- Also assert the shift cov dict has the right per-expert keys/shapes
  (`(layer,expert,"gate_proj"|"down_proj")`, `[d,d]`).
**Command:** `pytest tests/test_stage4_shift_cov_result.py -v`

### Task 6 — docstring fix (false "post-merge A re-collected")
**Implement** — replace `eora_compensation.py:77-85` ("Activation-cov reuse"
block). The current text falsely claims EoRA whitens with a "post-merge A
re-collected for Stage 4". Per spec §1.3/§5.4, nothing in Stage 4 re-collects;
it whitens with the original-calibration anchor (or its index-remap). New text:
```
Activation-cov whitening (whitening_cov knob)
---------------------------------------------
By DEFAULT (``stage4_eora.whitening_cov="anchor"``) this plugin whitens the
EoRA residual ``ΔW`` with the **original-calibration anchor** ``A_cov`` (the
Stage-2 sidecar, or its Stage-2 index-remap). NOTHING in Stage 4 re-collects
``A`` — the prior "post-merge A re-collected for Stage 4" claim was false
(see docs/research/2026-06-13-acov-capture-point.md §1.3). Upstream EoRA
(NVlabs/EoRA @ 6a42e2e, eora.py:512,518-526) instead whitens with the
sequentially-compressed **SHIFT** ``X'``; whitening with the anchor is a
measured deviation. Set ``whitening_cov="shift"`` to whiten with the post-2.5
shift cov (Stage-3 ride-along ``_stage3_shift_covariance.pt``) for
upstream-EoRA fidelity, or ``"anchored_adaptive"`` for the AA-SVD-consistent
anchor+shift form. The ``ΔW = W_orig − Ŵ`` *target* is always original-anchored
(``_solve_expert_tile`` :438-442) — original enters only the target, never the
whitening basis, matching upstream EoRA (eora.py:478).
```
(No test — doc only. Verify it no longer says "re-collected".)

### Task 7 — confirm goldens still green (DEFAULT byte-identity gate)
Do **not** add `whitening_cov` to any default config or golden test config.
**Command:**
```bash
cd max_quality
pytest tests/test_stage3_golden_snapshot.py tests/test_stage4_golden_snapshot.py \
       tests/test_stage4_plugin_inputs.py tests/test_stage4_plugin_compensation.py \
       tests/test_stage4_orchestrator.py -v
```
ALL must pass with NO golden regen. If any byte-diff appears, a default-path
change leaked — STOP and fix before proceeding (do not regen).

### Task 8 — full Stage-4 + Stage-3 suite + whole-impl review
**Command:**
```bash
cd max_quality
pytest tests/ -k "stage3 or stage4 or eora or covariance or shift" -v
```
Then a whole-implementation review (per MEMORY: "run the FULL suite + final
review" — per-task green misses seam bugs). Confirm: default path byte-identical
(goldens green), shift path result-changing+valid, no cross-file regression.

---

## 6. Config knob summary (for an A/B run config)
```yaml
stage3_svd:
  persist_shift_covariance: true     # NEW, default false — emit the ride-along artifact
stage4_eora:
  whitening_cov: shift               # NEW, default "anchor" — anchor | shift | anchored_adaptive
  log_residuals: true                # existing — log-only EoRA residual drop for the A/B metric
```
Arm A (current): omit both keys. Arm A* (upstream-faithful): both as above.

## 7. `anchored_adaptive` — scoped reduction (defensible, not full AA-SVD)
Vanilla EoRA's kernel takes a single `[d,d]` whitening cov `A`. The AA-SVD
closed form (`M=W·C·S⁻¹·R`) needs anchor `A`, shift `S=X'ᵀX'`, AND cross
`C=XX'ᵀ` — a different kernel than `_compute_eora_factors`. To keep this branch
EoRA-shaped and reviewable, `anchored_adaptive` ships as a **whitening-cov
blend** via a tiny lookup wrapper:
```python
class _AnchoredAdaptiveLookup:
    """anchored_adaptive whitening: blend anchor + shift into a single [d,d]
    cov for the EoRA √Λ basis. Reduction (NOT the full AA-SVD M=W·C·S⁻¹·R —
    that needs cross-cov C and a different kernel; see §9 open question). The
    blend is A_eff = A_anchor (when shift missing) else 0.5*(A_anchor+A_shift),
    keeping EoRA's single-cov contract."""
    def __init__(self, A_cov, shift_cov):
        self._a, self._s = A_cov, shift_cov or {}
    def get(self, key):
        a = self._a.get(key); s = self._s.get(key)
        if a is None: return s
        if s is None: return a
        return 0.5 * (a.to(torch.float32) + s.to(torch.float32))
```
This is documented as a **reduction**, not paper-exact AA-SVD. The core A/B
deliverable is anchor vs shift; `anchored_adaptive` is the optional third arm.
The full AA closed form is the open question §9.

## 8. Risk / byte-identity checklist
- [ ] `stage4_eora.whitening_cov` defaults to `"anchor"` via `.get(...,
      "anchor")`; default config files do NOT set it.
- [ ] `"anchor"` mode → `whitening_lookup IS A_cov` → `_inputs_for` byte-identical.
- [ ] `EoraInputsPlugin` skips shift load unless knob ∈ {shift, anchored_adaptive}.
- [ ] Stage-3 `persist_shift_covariance` defaults False → no artifact, spill
      rmtree unchanged → Stage-3 golden untouched.
- [ ] No change to `_compute_eora_factors` / `_eigh_spectrum` /
      `_solve_expert_tile` kernels (only the dict `A` is read from).
- [ ] Goldens (S3, S4 fp32+bf16) green with NO regen (Task 7).
- [ ] Shift path produces a DIFFERENT-but-valid result (Task 5).

## 9. Open questions (raise, don't substitute)
1. **Full AA-SVD `anchored_adaptive`?** The shipped form is a cov-blend
   reduction (§7), not `M=W·C·S⁻¹·R`. Do we want the full AA closed form
   (reusing Stage-3's `C_acc` cross-cov, `covariance_collection.py:496-503`)?
   That is a new kernel + a 4th input (C) into Stage 4 — a larger follow-up.
   Flagging per spec §5.3 "AA" arm; default plan ships the reduction.
2. **Shift-cov storage dtype.** B_acc spills in `bcov_storage_dtype` (default
   bf16). EoRA's anchor `A_cov` is typically fp16. `_eigh_spectrum` casts to
   fp32 before `eigh` (`:206`) so both are fine, but the noise floor is
   dtype-keyed (`_NOISE_FLOOR_BY_DTYPE[storage_dtype]`). Stage 4 passes
   `a_storage_dtype` (the anchor's dtype) to `_eigh_spectrum` even for the
   shift cov. Confirm we want the anchor's noise floor for the shift path, or a
   shift-specific `s_storage_dtype`. Default plan: reuse `a_storage_dtype`
   (simplest; the floor is relative to λ_max so dtype-class is close). Note as
   a knob if the A/B shows truncation sensitivity.
3. **down_proj shift cov.** B_acc DOES collect `down_proj` (factored,
   allclose-not-bitwise; `covariance_collection.py:783`), so the shift cov has
   `down_proj` keys — good, EoRA whitens all three matrices. (The anchor path's
   down_proj comes from the Stage-2 sidecar; parity holds.)
4. **REAM merge case.** For merge arms (REAM), the *shift* cov is per-survivor-
   slot (correct, post-2.5) whereas the *anchor* mis-attributes a single
   constituent's Gram (spec §3 merge wart). So `whitening_cov: "shift"` ALSO
   incidentally fixes the merge mis-attribution for the whitening basis — but
   the REAM anchor bugfix proper is the separate Stage-2 branch (§4).
