# PLAN_PLUGIN_09 — `merge_step="mergemoe"` (MergeMoE T₁=Q·P† closed-form)

**Status**: Ready for implementation
**Branch**: `feat/plugin_09_s2_mm` (base `0e7fd35`)
**Spec source**: `tasks/SC_STAGE12_COMPREHENSIVE_PLAN.md` §7 row `S2_MM` + §5.1 R1 (paper math).
**Paper**: *MergeMoE: Efficient Compression of MoE Models via Expert Output Merging*, arXiv:2510.14436. Equations 3–6.

---

## 1. Goal

Add a single new config knob `stage2_reap_ream.merge_step` that switches the
**merge math** (NOT the assignment, NOT the cost matrix, NOT the permutation
alignment) from the current frequency-weighted convex combination to the
MergeMoE closed-form `T₁ = Q · P†` solution.

| `merge_step` | Where it acts | Math used |
| --- | --- | --- |
| `"freq_weighted"` (default) | `_merge_experts_inplace` and `_tentative_merged_weights` | unchanged — `W_merged = Σ_j b_j · perm_j(W_j)`, byte-identical to current main |
| `"mergemoe"` | same two call sites | gate/up = freq-weighted average (same as default); down = `W'_D · T₁` with `T₁ = Q·P†` (closed-form least-squares) |

**Critical invariant**: with `merge_step="freq_weighted"` (default) the entire
Stage-2 pipeline is **byte-identical** to current main. All existing tests pass
unchanged. The new path is opt-in only.

---

## 2. MergeMoE math anchor (paper §3.2–§4, Eqs. 3–6)

For a merge cluster `C_i` with N members (centroid `c` + non-centroids `n_1…n_{N-1}`):

### 2.1 Frequency weights (Theorem 1)
```
b_j = f_j / Σ_{k ∈ C_i} f_k     for j ∈ C_i
```
Same as the current freq-weighted code. (If `Σ f = 0`, fall back to equal
weights — F2-FREQ-WEIGHT-FLOOR, same guard as today.)

### 2.2 Stacked weight matrices (paper Eq. 3 block notation)

For permutation-aligned cluster members (each non-centroid first permuted to
the centroid's intermediate-neuron ordering, same as today):

```
W'_G = [ W_G^1 ;  W_G^2 ; … ; W_G^N ]      (vertical stack, (N·d_int, d_hidden))
W'_U = [ W_U^1 ;  W_U^2 ; … ; W_U^N ]      (vertical stack, (N·d_int, d_hidden))
W'_D = [ b_1·W_D^1, b_2·W_D^2, …, b_N·W_D^N ]  (horizontal stack, (d_hidden, N·d_int))
```

### 2.3 Compression matrices (paper Eq. 4)

```
T₂ = T₃ = [b_1·I, b_2·I, …, b_N·I]     shape (d_int, N·d_int)
```

So:
- `W_G^merged = T₂ · W'_G = Σ_j b_j · W_G^j`   ← same as freq-weighted avg today
- `W_U^merged = T₃ · W'_U = Σ_j b_j · W_U^j`   ← same as freq-weighted avg today

For gate/up the MergeMoE merged weights are **identical** to today's
freq-weighted result. No code change needed for gate/up arithmetic.

### 2.4 T₁ (paper Eq. 5–6, the new math)

T₁ has shape `(N·d_int, d_int)` and solves the calibration least-squares
problem:

```
P = σ(T₂·W'_G·X̂) ⊙ (T₃·W'_U·X̂)            shape (d_int,      T)
Q = σ(W'_G·X̂)    ⊙ (W'_U·X̂)               shape (N·d_int,    T)

T₁ = Q · P†                                  shape (N·d_int, d_int)    (Eq. 6)
```

with `X̂` = calibration tokens that **arrive at this MoE layer's input**
(same `_LayerInputAccumulator` reservoir that `cost_alignment="output"` already
captures). `σ` = SiLU. `†` = Moore-Penrose pseudoinverse.

**Interpretation**: `P` is the SwiGLU intermediate activation that the
*merged* expert produces on `X̂`. `Q` is the **vertically-stacked** SwiGLU
intermediate activation that each *original* expert produces on `X̂`. T₁
chooses how to mix the original experts' activation rows into the merged
expert's activation rows so that — after applying the (already-fixed)
`W'_D` — the merged expert reproduces the cluster's freq-weighted output on
the calibration tokens.

### 2.5 Merged `W_D` (the only code change vs freq-weighted)

```
W_D^merged = W'_D · T₁                       shape (d_hidden, d_int)
```

Splitting `T₁` into `N` row-blocks each of shape `(d_int, d_int)`:

```
T₁ = [ T₁^(1) ; T₁^(2) ; … ; T₁^(N) ]

W_D^merged = Σ_j b_j · W_D^j · T₁^(j)
```

This is the **only** weight whose math diverges from today's freq-weighted
path; gate and up stay byte-identical.

---

## 3. Code-change locations

Two files touched; one tiny new helper file.

### 3.1 `stage2/mergemoe.py` (NEW)

Single self-contained helper module. Contains:

- `_swiglu_intermediate(W_gate, W_up, x) -> (T, d_int)` — local mirror of the
  SwiGLU intermediate activation `σ(gate·x) ⊙ (up·x)`. (We do **not** reuse
  `output_space_cost._swiglu_forward` because that returns the full
  `W_down`-projected output; MergeMoE needs the pre-down intermediate. Trivial
  helper, ~5 lines.)

- `_mergemoe_compute_merged_down(*, member_gates, member_ups, member_downs,
  member_perms, weights, X_hat, dtype) -> torch.Tensor`

  Inputs:
  - `member_gates`, `member_ups`, `member_downs`: list of N permutation-aligned
    weight tensors in the model's native dtype. The centroid's perm is
    identity; non-centroids' perms come from the existing `_PermAlignCache` or
    a fresh `_permutation_align_to_centroid` call.
  - `weights`: 1-D numpy/python array of length N, freq-weighted (already
    normalized to Σ = 1 by caller; same array shape `_merge_experts_inplace`
    builds today).
  - `X_hat`: `(T, d_hidden)` calibration tokens, float32. Caller is responsible
    for the token cap + device placement.
  - `dtype`: model's native dtype to cast the returned merged down to.

  Algorithm (all in fp32, returning native dtype at the end):

  1. Build P = `σ(W_G^merged · X̂) ⊙ (W_U^merged · X̂)`, shape `(T, d_int)`.
     Here `W_G^merged = Σ_j b_j W_G^j`, `W_U^merged = Σ_j b_j W_U^j` (same as
     gate/up freq-weighted avg).
  2. For each member j, compute `Q_j = σ(W_G^j · X̂) ⊙ (W_U^j · X̂)`, shape
     `(T, d_int)`.
  3. Stack: `Q_stack = concat([Q_1, …, Q_N], dim=1)` shape `(T, N·d_int)`.
     Convention: paper writes Q with shape `(N·d_int, T)`, we operate with
     the transpose throughout so we can use PyTorch's `torch.linalg.lstsq`
     which is row-major-friendly. The math is unchanged: solving
     `T₁ P = Q` (paper) is equivalent to solving `Pᵀ · T₁ᵀ = Qᵀ` and we
     read T₁ off the result.
  4. Solve `min ‖P_t · T₁ᵀ − Q_stackᵀ_t ‖_F²` where the subscript `_t`
     transposes the data-batch axis to rows. Concretely:
     ```python
     # P shape (T, d_int)  ; Q_stack shape (T, N·d_int)
     # solve P · X = Q_stack  for X of shape (d_int, N·d_int)
     X = torch.linalg.lstsq(P, Q_stack, driver="gels" or "gelsd").solution
     # Then T₁ = Xᵀ has shape (N·d_int, d_int) as in the paper.
     T1 = X.transpose(0, 1).contiguous()
     ```
     (Implementation note: `torch.linalg.lstsq` solves `A·X = B` for `X`,
     so passing `A=P`, `B=Q_stack` yields `X = P† · Q_stack`, i.e.
     `Xᵀ = Q_stackᵀ · (P†)ᵀ = Q · P†` per paper convention. ✓)
  5. Split T₁ row-wise into N blocks of shape `(d_int, d_int)`:
     `T1_blocks = T1.view(N, d_int, d_int)`.
  6. Compute `W_D^merged = Σ_j b_j · W_D^j · T1_blocks[j]`, shape `(d_hidden, d_int)`.
  7. Cast back to `dtype` and return.

  **Rank-deficiency / conditioning guard (paper-fidelity deviation; see Risk
  Mitigation §581 of the comprehensive plan)**: before calling lstsq, compute
  `cond(P)` via `torch.linalg.cond`. If `cond(P) > 1e8` (rank-deficient or
  near-singular), log a warning and fall back to the freq-weighted down
  (i.e. return `Σ_j b_j · W_D^j`). This is **not** in the paper — it is a
  documented project deviation `D-mergemoe-cond-fallback`. Threshold matches
  the comprehensive-plan §581 risk-mitigation specification.

  **Per-token cap**: caller passes a sub-sampled `X_hat`; the same
  `cost_output_token_cap` knob and per-layer deterministic seed used by
  `_output_space_cost` are reused. Default `1024` tokens.

### 3.2 `stage2/plugins/output_space_cost.py:_tentative_merged_weights`

Add `merge_step` + `layer_inputs` + `token_cap` parameters. Branch on
`merge_step`:

```python
def _tentative_merged_weights(
    layer_ref, centroid_id, child_id, freq, ream_acc, perm_cache, banks,
    *,
    merge_step: str = "freq_weighted",
    layer_inputs: torch.Tensor | None = None,
    token_cap: int = 1024,
) -> dict[str, torch.Tensor]:
    # ... compute w_c, w_m, perm, perm_t (UNCHANGED) ...

    merged: dict[str, torch.Tensor] = {}
    # gate_proj and up_proj — always freq-weighted (same under both modes).
    # ...

    if merge_step == "mergemoe":
        # Replace down_proj only.
        from ..mergemoe import _mergemoe_compute_merged_down
        ...
        merged["down_proj"] = _mergemoe_compute_merged_down(...)
    else:
        # Default: freq-weighted down (existing code, unchanged).
        ...

    return merged
```

The default keyword argument `merge_step="freq_weighted"` keeps the existing
call sites in `_output_space_cost` byte-identical.

**Performance note**: `_output_space_cost` calls `_tentative_merged_weights`
inside an O(n_nc × topk) loop. Each MergeMoE call adds a small lstsq solve.
We do NOT extend MergeMoE to the cost matrix in this PR: the cost matrix
keeps using freq-weighted tentative merges (it is a cost *proxy*, not the
actual merge). `_output_space_cost` callers do NOT pass `merge_step` and so
get the default. The only call site that opts into `merge_step="mergemoe"`
is the actual merge step at `merging.py:_merge_experts_inplace`.

### 3.3 `stage2/merging.py:_merge_experts_inplace`

Add `merge_step` + `layer_inputs` parameters (default `"freq_weighted"` /
`None`). Branch the inner loop's `down_proj` arithmetic:

```python
def _merge_experts_inplace(
    layer_ref, grouped, freq,
    *,
    freq_weighted: bool,
    scores=None,
    ream_acc=None,
    perm_cache=None,
    merge_step: str = "freq_weighted",
    layer_inputs: torch.Tensor | None = None,
    token_cap: int = 1024,
) -> None:
    ...
    for centroid, members in grouped.items():
        if len(members) <= 1:
            continue
        # ... freq/saliency weights (UNCHANGED) ...
        # ... perm alignment per member (UNCHANGED) ...

        if merge_step == "mergemoe" and layer_inputs is not None and len(members) >= 2:
            # Build aligned member weight lists (gate, up, down) and call the
            # mergemoe helper for down only. gate/up follow the existing
            # freq-weighted path (byte-identical to default).
            ...
        else:
            # Existing freq-weighted code, unchanged.
            ...
```

When `merge_step="freq_weighted"` (default): the entire function is
**byte-identical** to today — no detours, the new branch is dead code.

When `merge_step="mergemoe"` and `layer_inputs is None`: fall back to
freq-weighted with a one-line `log.warning(...)` (the merge would otherwise
fail; this gives the caller a clear actionable message). Code path documented
as a defensive fallback, NOT a normal mode.

### 3.4 `stage2/plugins/layer_merge.py` (call-site update)

Pass the new knobs through to `_merge_experts_inplace`:

```python
def merge(self, ctx):
    ...
    layer_input_acc = ctx.get("layer_input_acc")
    layer_inputs = layer_input_acc.buffer if layer_input_acc is not None else None

    _merge_experts_inplace(
        layer_ref, grouped, freq,
        freq_weighted=self.s2["ream"]["frequency_weighted_merge"],
        scores=scores,
        ream_acc=ream_acc,
        perm_cache=perm_cache,
        merge_step=self.merge_step,
        layer_inputs=layer_inputs,
        token_cap=self.cost_output_token_cap,
    )
```

Also: when `merge_step="mergemoe"` and the layer-input accumulator is
**disabled** (i.e. no distillation and `cost_alignment != "output"`), we need
to **force-enable** it in `on_layer_setup`. Update the
`_need_layer_inputs` clause:

```python
_need_layer_inputs = (
    self.expert_distill_steps > 0
    or self.cost_alignment_cfg == "output"
    or self.merge_step == "mergemoe"           # NEW
)
_layer_input_cap = (
    max(
        self.expert_distill_token_cap if self.expert_distill_steps > 0 else 0,
        self.cost_output_token_cap if self.cost_alignment_cfg == "output" else 0,
        self.cost_output_token_cap if self.merge_step == "mergemoe" else 0,   # NEW
    )
    if _need_layer_inputs else 0
)
```

This keeps the default `merge_step="freq_weighted"` path byte-identical
(`_need_layer_inputs` boolean for runs with no distill and `cost_alignment !=
"output"` stays `False`).

Add `merge_step` as a `__init__` kwarg on `LayerMergePlugin` (mirroring
`cost_alignment_cfg`).

### 3.5 `stage2/orchestrator.py` (config validation + plumbing)

Add to `run()`'s config-parse block (around L867 next to
`cost_alignment_cfg`):

```python
merge_step: str = str(s2.get("merge_step", "freq_weighted")).lower()
if merge_step not in ("freq_weighted", "mergemoe"):
    raise ValueError(
        f"stage2_reap_ream.merge_step={merge_step!r}; "
        "expected 'freq_weighted' or 'mergemoe'."
    )
```

Pass `merge_step` to `LayerMergePlugin(...)` and emit it under the
`stage2/config/...` Trackio namespace:
```python
"stage2/config/merge_step": merge_step,
```

Also update the resume path call at L787: pass `merge_step=merge_step,
layer_inputs=None`. On resume the calibration buffer is gone; the freq-weighted
fallback message will fire for any MergeMoE-mode resume that hits a multi-member
merge. **Document this in the resume docstring** as
`D-mergemoe-resume-fallback` — same posture as the existing `scores=None on
the resume path` deviation for saliency mode (orchestrator.py:782–786). No new
risk: the partial JSON already captures `final_kept_ids`; on resume the
already-merged weights are loaded from disk, not recomputed. The fallback path
only fires if a USER manually deletes a heal-weights file mid-resume, which is
out of scope.

---

## 4. Tests

New file: `max_quality/tests/test_stage2_plugin_mergemoe_step.py`. Tests:

### 4.1 Hand-checked closed-form on a small case
- 2 experts, `d_hidden=4`, `d_int=3`, freq = (3, 1) → b = (0.75, 0.25).
- Build random X̂ shape `(8, 4)`, manually compute `P`, `Q`, `T₁ = Q · P†`,
  `W_D^merged = Σ_j b_j · W_D^j · T₁_block_j`.
- Call `_mergemoe_compute_merged_down(...)`. Assert
  `torch.allclose(result, hand_computed, atol=1e-5, rtol=1e-4)`.

### 4.2 `merge_step="freq_weighted"` is the default and byte-identical
- Build two random expert weight sets (no permutation needed for a 2-member
  group with identity perm — `_PermAlignCache` returns identity for paired
  identical activation seeds).
- Call `_merge_experts_inplace` with `merge_step` unset (i.e. default).
- Snapshot the merged centroid's gate/up/down.
- Reset the model.
- Call `_merge_experts_inplace` with `merge_step="freq_weighted"` explicitly.
- Assert byte-identical tensors (no `atol`).

### 4.3 `merge_step="mergemoe"` routes through the new path
- Same setup as 4.2 but with `merge_step="mergemoe"` and a non-None
  `layer_inputs` calibration tensor.
- Assert gate/up unchanged vs freq-weighted (since T₂ = T₃ = freq weights, gate
  and up are mathematically identical).
- Assert down_proj differs from freq-weighted result (sanity that the new
  branch fired and computed something non-trivial).
- Assert finite (no NaN/Inf).

### 4.4 `merge_step="mergemoe"` with `layer_inputs=None` falls back + warns
- Call `_merge_experts_inplace(..., merge_step="mergemoe", layer_inputs=None)`.
- Capture log records. Assert exactly one WARNING fired naming
  `freq-weighted fallback`.
- Assert the merged weights match the freq-weighted result byte-identical.

### 4.5 Conditioning guard fires on a degenerate calibration
- Build `X̂` that makes `P` rank-deficient (e.g. `X̂` is rank 1 — all tokens
  the same vector).
- Call `_mergemoe_compute_merged_down(...)`.
- Assert one WARNING was logged mentioning `cond` and the freq-weighted
  fallback was used.

### 4.6 Orchestrator config validation
- Build a stage-2 cfg with `merge_step="banana"`. Assert ValueError mentioning
  the legal values.

### 4.7 `OutputSpaceCostPlugin._tentative_merged_weights` is unchanged on
default
- Pin: call `_tentative_merged_weights` with the existing argument set (no
  `merge_step` kwarg). Verify it still returns the exact same dict shapes /
  dtypes / values as before. (Guards against accidentally changing the cost
  path.)

### 4.8 Full Stage-2 run smoke (existing tests)
The existing `test_stage2_*` suite should pass byte-identical on default
`merge_step="freq_weighted"`. Supervisor runs the full suite as Gate 2.

---

## 5. Gates

1. New test file green: `pytest max_quality/tests/test_stage2_plugin_mergemoe_step.py`.
2. All existing Stage-2 tests green:
   `pytest max_quality/tests/test_stage2_*` (byte-identical on default
   `merge_step="freq_weighted"`).
3. Full suite green: `pytest max_quality/tests/`.

---

## 6. Paper-fidelity deviations to document

In the new `stage2/mergemoe.py` module docstring:

1. **D-mergemoe-cond-fallback**: `cond(P) > 1e8` falls back to freq-weighted
   down. Not in the paper; project-pragmatic per comprehensive plan §581 risk
   mitigation.
2. **D-mergemoe-perm-alignment**: paper does not specify a per-cluster
   intermediate-neuron alignment because it uses cosine-similarity clustering
   on `[W_U; W_G]`. This project uses SC's output-cost assignment and the
   existing `_permutation_align_to_centroid` Hungarian neuron alignment —
   each non-centroid is permuted into the centroid's neuron basis before the
   MergeMoE stack is built. This is **strictly compatible** with the paper's
   math (the paper's `W_G^j` becomes our `perm_j(W_G^j)`); listed here for
   audit-trail honesty.
3. **D-mergemoe-token-cap**: `cost_output_token_cap` (default 1024) bounds
   the calibration sample size per layer. Paper §3.3 uses 128 sequences ×
   2048 tokens ≈ 262k tokens; we use 1024 to keep the per-layer wall-clock
   tractable. Acceptable: the closed-form lstsq is well-conditioned at
   `T ≥ N·d_int`; for the project's `N ≤ 9` and `d_int = 512`, `T ≥ 4608`
   would be ideal but the SC cost loop reuses the same buffer at `T = 1024`
   without degradation, so we match it.
4. **D-mergemoe-fp32-solve**: lstsq runs in fp32 (the data flows in bf16 in
   production); result cast back to native dtype before bank.set. Matches the
   project's existing fp32 numerical-precision posture (see `merging.py`
   L124-125 `.to(torch.float32)` upcast in freq-weighted code).

---

## 7. Out of scope (deferred)

- T₂ and T₃ **per-cluster optimization** (paper sec. 4.2 mentions a joint
  optimization — `gate_proj` and `up_proj` could ALSO be fitted to the
  calibration X̂ via a second lstsq, in addition to T₁). The paper's
  primary recipe and our `S2_MM` row keep T₂/T₃ as the simple freq-weighted
  closed form (their Eq. 4), so this is paper-aligned. If `S2_MM` shows
  signal but undershoots, a follow-up plan can add the T₂/T₃ optimization.
- MergeMoE as a **cost matrix** rather than merge step (cost would estimate
  pseudoinverse-merge damage via the same closed form). Out of scope; only
  the merge step is changed here.
- `S2_MM_SEQ` (MergeMoE + REAM sequential): requires Plugin #10 / REAM
  sequential first. Not in this PR.

---

## 8. Done = ?

- Branch `feat/plugin_09_s2_mm` exists with three commits (or one squashed
  feat commit) on top of `0e7fd35`.
- `_merge_experts_inplace(..., merge_step="freq_weighted")` is byte-identical
  to today.
- `_merge_experts_inplace(..., merge_step="mergemoe")` runs the closed-form
  `T₁ = Q · P†`; produces a finite, hand-checked correct merged down_proj.
- Orchestrator validates `merge_step` ∈ {`freq_weighted`, `mergemoe`}.
- All three gates green.
- Paper-fidelity review-loop closes with no unaddressed deviations.
- Code-quality review-loop closes with no unaddressed findings.
