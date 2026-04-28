# Strategy A Pipeline — Status & Second-Opinion Report

**Date:** 2026-04-27 (Europe/Lisbon)
**Author:** Lucas Pirola (with Claude)
**Pipeline:** `max_quality` Strategy A — 30% parameter reduction on Qwen3.6-35B-A3B
**Working tree:** `/home/lucas/ai/moe_compress/max_quality` @ commit `49aa721`

This document is a complete narrative of what has been done so far, what
decisions were made and why, and where every artifact currently lives on the
Hugging Face Hub. It is intended for an outside reviewer ("second opinion")
who has not been following the day-to-day work.

---

## 1. The starting point

We started from the public model
[`Qwen/Qwen3.6-35B-A3B`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B), a 35 B
parameter sparse Mixture-of-Experts (MoE) model. Each of its 40 transformer
blocks contains 256 routed experts plus 1 shared expert; each token is routed
to top-8 experts. The model is referenced from
[`hf_jobs/submit.sh`](../hf_jobs/submit.sh#L22) and
[`configs/qwen36_35b_a3b_30pct.yaml`](../configs/qwen36_35b_a3b_30pct.yaml).

**Why this model.** Compared to Qwen3-32B-A3B, Qwen3.6-35B-A3B's higher
expert count (256 vs ~32) gives more redundancy headroom for expert
pruning while keeping `topk=8` active per token. It also has native
`transformers≥4.57` support via the fused `Qwen3_5MoeExperts` module, which
the entire pipeline relies on.

**Goal.** Compress the model by 30% in parameter count without unacceptable
quality loss (≤ +3% relative WikiText-2 PPL, ≤ ±1.5 pp on zero-shot ARC-C /
HellaSwag, ≤ ±3 pp on generative HumanEval / MATH-500 — see
[`stage6_validate.py`](../src/moe_compress/stage6_validate.py)).

---

## 2. The pipeline at a glance

The pipeline has 7 stages, registered in
[`src/moe_compress/run_pipeline.py`](../src/moe_compress/run_pipeline.py):

| Stage | Name                            | Output checkpoint dir | Done? |
|-------|---------------------------------|-----------------------|-------|
| 0     | Super-Expert Blacklist          | (none — JSON only)    | ✅    |
| 1     | GRAPE Budget Allocation         | (none — JSON only)    | ✅    |
| 2     | REAP Score + REAM Merge         | `stage2_pruned/`      | ✅    |
| 3     | AA-SVD Factorization            | `stage3_svd/`         | ❌ **stuck — see §6** |
| 4     | EoRA Residual Compensation      | `stage4_eora/`        | ⏸    |
| 5     | Router-only Knowledge Distill.  | `stage5_final/`       | ⏸    |
| 6     | Validation (PPL + zero-shot)    | (eval JSON only)      | ⏸    |

Per-stage Hub upload is the durability boundary (see §4).

---

## 3. What each stage does and why

The implementation lives under
[`src/moe_compress/`](../src/moe_compress/). What follows is the *purpose*
of each stage, not its code structure.

### Stage 0 — Super-Expert Blacklist
[`stage0_super_experts.py`](../src/moe_compress/stage0_super_experts.py)

Profiles maximum activation magnitudes per expert across a calibration
sample, z-scores them within each layer, and blacklists outliers (>2.5σ,
max 4 per layer, ≤5% global cap). **Purpose:** remove a small number of
"super-expert" outliers whose activation tails would otherwise dominate
the reconstruction errors of every later stage.

### Stage 1 — GRAPE Budget Allocation
[`stage1_grape.py`](../src/moe_compress/stage1_grape.py)

Computes per-layer pairwise expert redundancy (cosine / MSE / CKA distance
matrices) and solves a convex LP that targets ~22% expert reduction +
~11% SVD reduction = 30% total. **Purpose:** spend the parameter budget
non-uniformly — high-redundancy layers prune deeper, low-redundancy layers
keep more capacity.

### Stage 2 — REAP Score + REAM Merge
[`stage2_reap_ream.py`](../src/moe_compress/stage2_reap_ream.py)

Scores each pair of routed experts for merge cost (gate error
δ_gate, expert error δ̃_expert) and runs a frequency-weighted Hungarian
assignment to merge similar experts down to the per-layer budget. While
running, taps both `gate_up_in` (input to gate/up_proj) and `intermediate`
(input to down_proj) and accumulates input covariance matrices A and B
that all later stages consume. **Purpose:** reduce expert count from
128 → ~90–100 per layer and capture the post-prune input distribution.

### Stage 3 — AA-SVD Factorization
[`stage3_svd.py`](../src/moe_compress/stage3_svd.py)

Replaces each expert's 3 dense matrices (`gate_proj`, `up_proj`,
`down_proj`) with rank-k factors `W ≈ U·V` using one-sided
**activation-aware SVD** (Yuan et al. / SVD-LLM). The formulation
(`stage3_svd.py:350`) is:

```
M = W · L_B       (L_B = chol(B + ε·I), B = post-prune input covariance)
U Σ Vᵀ = svd(M)
U_k = U[:,:k] · S[:k]
V_k = Vᵀ[:k,:] · L_B⁻¹
⇒ U_k · V_k ≈ W   (rank-k optimal in the B-weighted Frobenius norm)
```

**Purpose:** another large parameter cut on top of expert merging,
but compressed *along the directions the model actually uses* rather than
plain Frobenius optimal.

### Stage 4 — EoRA Residual Compensation
[`stage4_eora.py`](../src/moe_compress/stage4_eora.py)

For each factored matrix, computes the residual ΔW = W_original − U·V,
projects it into the top-r eigenspace of the activation covariance, takes
a rank-r SVD of that projection, and *widens* (U, V) along the rank
dimension. Capped at 3% of Stage 3 savings. **Purpose:** recover quality
lost to rank truncation with zero extra router work — purely a stronger
factorization.

### Stage 5 — Router-only Knowledge Distillation
[`stage5_router_kd.py`](../src/moe_compress/stage5_router_kd.py)

Freezes all expert weights and topology; trains *only* the router gate
weights via KL divergence to match the teacher router's logits. The
teacher is loaded in NF4 quantization to fit alongside the student on a
single 80 GB A100. **Purpose:** adapt routing decisions to the new (pruned
+ factorized) expert set without touching expert internals.

### Stage 6 — Validation
[`stage6_validate.py`](../src/moe_compress/stage6_validate.py)

Runs WikiText-2 PPL, zero-shot (ARC-C, HellaSwag) and generative
(HumanEval, MATH-500) evaluations on both the compressed student and the
original teacher; compares deltas to the SLA thresholds. **Purpose:** hard
quality gate before declaring a successful run.

---

## 4. Key design decisions

### 4.1 Calibration data — Nemotron-Cascade-2-SFT-Data
*Commit [`8f65188`](https://github.com/lucaspirola/moe_compress/commit/8f65188), 2026-04-24.*

The earlier C4:Math:Code (0:0.5:0.5) split was math-skewed and not
representative of deployment. We switched to the Nemotron-Cascade-2-SFT
mix: chat 56% / math 21% / science 11% / instruction 3.3% / agent
3.3% / SWE 2% / terminal 3.3%. Same calibration set is used in stages 0,
2, 3 and 5. **Why it matters:** MoE expert routing is task-aware; the
calibration distribution defines which experts get treated as redundant.

### 4.2 Activation-aware SVD (one-sided ASVD, B only)
*Implemented at [`stage3_svd.py:350`](../src/moe_compress/stage3_svd.py#L350).*

We chose one-sided ASVD (the SVD-LLM / Yuan et al. formulation) over
two-sided ASVD: only the *post-prune* input covariance B is used inside
the SVD. The pre-prune covariance A is reserved for the optional L-BFGS
refine in `_per_matrix_refine`. **Why:** B reflects the input distribution
the compressed model will actually see; A would distort the factorization
toward the original model's input space (this was the bug fixed in
[`49aa721`](https://github.com/lucaspirola/moe_compress/commit/49aa721) —
see §5).

### 4.3 Per-projection rank bias
*Commit [`0b6a5ac`](https://github.com/lucaspirola/moe_compress/commit/0b6a5ac), 2026-04-27.*

Empirical reconstruction errors at uniform rank are wildly different
across the 3 projections: gate_proj ≈ 998 k, up_proj ≈ 468 k, down_proj
≈ 5 k. Reasons: SiLU amplifies gate errors; down_proj is inherently
near low-rank. The config now has
[`per_projection_weight: {gate: 1.75, up: 1.35, down: 0.35}`](../configs/qwen36_35b_a3b_30pct.yaml#L81)
which budget-neutrally shifts rank from down → gate/up.

### 4.4 Per-stage Hub durability
*Commit [`b1697c7`](https://github.com/lucaspirola/moe_compress/commit/b1697c7), 2026-04-26;
detailed rationale in [`docs/huggingface_jobs_and_buckets.md`](huggingface_jobs_and_buckets.md).*

HF Jobs bucket FUSE mounts are **not durable** under SIGKILL or timeout.
Each stage 2–5 commits its checkpoint to a *per-stage* HF model repo
immediately after the stage completes (with sidecars hoisted under
`artifacts/`). This is the only durability boundary; the bucket is treated
as a scratch cache. The pattern is:

```
<base_repo>-stage2   ← Stage 2 output + covariance sidecar
<base_repo>-stage3   ← Stage 3 output + originals sidecar
<base_repo>-stage4   ← Stage 4 output
<base_repo>-stage5   ← Final compressed model
```

### 4.5 Target ratio = 30%
[`configs/qwen36_35b_a3b_30pct.yaml:18`](../configs/qwen36_35b_a3b_30pct.yaml#L18).
Internally split into 22% expert reduction (Stage 1) + 11% SVD reduction
(Stage 3); EoRA (Stage 4) is capped at 3% spend-back of Stage 3 savings,
so the net is ~30%.

### 4.6 Other recent fixes
- [`b693aa9`](https://github.com/lucaspirola/moe_compress/commit/b693aa9): VRAM OOM at layer 27/40 — dense expert module and new FactoredExperts briefly coexisted on GPU; dense is now offloaded to CPU before allocation.
- [`b8ac530`](https://github.com/lucaspirola/moe_compress/commit/b8ac530): the metric `rel_recon_err` was measuring `W − U·V` (Frobenius), which doesn't match the AA-SVD objective; replaced with `‖(W−U·V)·L_B‖ / ‖W·L_B‖`.
- [`49aa721`](https://github.com/lucaspirola/moe_compress/commit/49aa721): the *factorization itself* was producing `U·V ≈ W·A`, not `W` (the formula included A); replaced with the correct `M = W·L_B`. **This is the fix that motivates the current Stage 3 attempt.**

---

## 5. What ran and where the artifacts are

### 5.1 Stage 2 — DONE
- **HF repo:** [`pirola/qwen3-6-35b-a3b-strategy-a-30pct-20260425-1634-stage2`](https://huggingface.co/pirola/qwen3-6-35b-a3b-strategy-a-30pct-20260425-1634-stage2)
- **Job ID:** `69ececbbd70108f37acdea1f` (completed 2026-04-25 21:46 UTC)
- **Contents:**
  - Pruned model weights at the repo root (sharded safetensors).
  - `artifacts/_stage2_input_covariance.pt` — 69.8 GB sidecar; required input for Stage 3.
  - `artifacts/stage1_budgets.json`, `artifacts/stage2_layer_mse.json`, etc.
  - `artifacts/merge_map.json` — which experts were merged into which.
  - `job_status.txt` — `SUCCESS`.

This is the input the Stage 3 attempts resume from
(`PRIOR_STAGE_REPO=…stage2`).

### 5.2 Stage 3 — IN PROGRESS, BLOCKED
- **First attempt:** job `69ef4bb6d70108f37ace07c0` (2026-04-27 11:42 UTC) — cancelled after sitting in queue.
- **Second attempt:** job `69ef5b3fd2c8bd8662bd0dc1` (2026-04-27 13:48 UTC).
  - Picked up A100-large at 18:36 UTC after a long queue wait.
  - B-cov collection: completed all 40 layers cleanly (~20 s/layer, GPU 99%, VRAM 56.5 / 85.9 GB).
  - **Factorization phase: catastrophic numerical failure** (see §6).
  - **Cancelled** at 21:42 CEST after observing Trackio metrics. Nothing of value was uploaded — the would-be repo `pirola/qwen3-6-35b-a3b-strategy-a-30pct-20260427-1836-stage3` does not exist.

### 5.3 Stages 4–6 — NOT YET RUN
Cannot proceed without a valid Stage 3 output.

---

## 6. The current blocker (where we want a second opinion)

> **Superseded by §6.5 below.** The dominant root cause turned out to be
> bfloat16 covariance storage, not the regularizer. The hypothesis recorded
> in this section ("rank-deficient B + 1e-6·I") is correct as far as it goes
> but missed the upstream cause. Read §6.5 for the resolution.

### 6.1 What we observed

Trackio run
[`qwen3-6-35b-a3b-strategy-a-30pct-20260427-1836`](https://huggingface.co/spaces/pirola/trackio)
shows `stage3/recon_rel_err/{gate,up,down}_proj` over the first 15 layers:

| Layer | gate_proj | up_proj  | down_proj |
|------:|----------:|---------:|----------:|
| 0     | 0.46      | 0.46     | **696**   |
| 1     | 0.54      | 0.54     | **514**   |
| 2     | 0.55      | 0.55     | 354       |
| 3     | 0.55      | 0.55     | 489       |
| 4     | **451 564** | **20 086** | 903     |
| 5     | 66 136    | 21 741   | 1 084     |
| 6     | 89 031    | 42 843   | 840       |
| 7     | 591 718   | 141 949  | 1 508     |
| 8     | 645 269   | 179 105  | 2 243     |
| 9     | 441 962   | 159 728  | 2 815     |
| 10    | 51 465    | 20 480   | 1 806     |
| ...   | ...       | ...      | ...       |

These are **relative** errors. The activation-weighted SVD truncation
identity guarantees `rel_err ∈ [0, 1]` in exact arithmetic, so anything
above 1 is a numerical artifact. We're seeing 2 to 6 orders of magnitude
above the bound.

Two distinct failure modes:
1. **Layers 0–3:** gate / up at ~0.5 (already 5–10× a healthy value),
   `down_proj` at 350–700 (impossible).
2. **Layer 4 onwards:** gate / up explode to 10⁴–10⁶.

### 6.2 Our diagnosis

We believe both modes are the same root cause: **`B` is severely
rank-deficient** and the regularization is **absolute, not relative**, so
`L_B` is so ill-conditioned that the Cholesky back-solve
(`solve_triangular`) returns a `V_k` whose entries are 10⁵–10⁷, and the
metric `(W − U·V) · L_B` then loses ~all its significant digits to
catastrophic cancellation in float32.

**Why `B` is rank-deficient.** `B = Σ xᵀx` is summed over calibration
tokens; rank(B) ≤ #tokens. For `down_proj`, `dim(B) = intermediate_size`
(≈ 6144). Calibration uses ~512–1024 tokens per (layer, expert), so
rank(B) ≪ dim(B). For `gate_proj` / `up_proj`, `dim(B) = hidden_size`
(≈ 2048) — also rank-deficient but less catastrophically. This explains
why `down_proj` breaks first (layer 0) and `gate/up` break later (layer 4+
when activation diversity has dropped further with depth).

**Why `1e-6 · I` doesn't save us.** This is an *absolute* floor. Typical
diagonal entries of `B` for these matrices are O(10⁻³ — 10²) depending
on the layer, so 1e-6 is several orders of magnitude below the natural
scale and effectively no regularization at all in the deficient
directions.

**Why the regression tests passed.** The 7 numerical-correctness tests
added in
[`49aa721`](https://github.com/lucaspirola/moe_compress/commit/49aa721)
use random or well-conditioned `B`. None covers
`X ∈ ℝ^{500 × 6144}, B = Xᵀ X`. That's the gap that let the bug ship.

### 6.3 Proposed fix (not yet applied)

In [`_aa_svd`](../src/moe_compress/stage3_svd.py#L350):

1. **Relative regularization.** Replace
   ```python
   B_reg = B + 1e-6 * I
   ```
   with
   ```python
   eps = 1e-3
   B_reg = B + eps * B.diag().mean() * I
   ```
   This bounds `cond(L_B)` to ~10³ regardless of `B`'s scale or rank.

2. **Numerically stable metric.** Compute
   `rel_err = sqrt(sum(S[k:]²) / sum(S²))` from the singular values of
   `M = W · L_B` directly. Identical math to the current expression but
   no cancellation.

3. **Optional: clip rank to effective rank.**
   `k_eff = min(k, num_singular_values_above(σ_max · 1e-6))`. Asking SVD
   to keep `k = 2000` directions when `B` only spans ~500 of them is
   fitting noise; cheap to add.

4. **New regression test.** `B = Xᵀ X` with `X ∈ ℝ^{500 × 6144}`. Assert
   `rel_err ≤ 1` and `‖(W − U·V)·L_B‖ / ‖W·L_B‖` matches the rank-k
   truncation bound.

Estimated change: ~30 lines in `stage3_svd.py`, ~50 lines of tests, local
test suite < 30 s.

### 6.4 Questions for the second-opinion reviewer

1. Does the diagnosis hold up — is rank-deficient `B` + absolute
   regularization the right root cause, or is there something subtler we
   missed (e.g. the Cholesky path is correct and the bug is actually in
   how `B` was accumulated in Stage 2)?
2. Is the proposed relative regularization the right fix, or is there a
   more principled approach (e.g. pseudo-inverse via truncated SVD of `B`
   itself, which avoids Cholesky entirely)?
3. Should we go further and **clip k to rank(B)** rather than relying on
   regularization — i.e. accept that we cannot meaningfully factorize
   into more directions than calibration spans?
4. Is there value in increasing the calibration token count for Stage 2
   to push `rank(B)` up, even though it's still much less than 6144?

---

## 6.5 Resolution & re-run

Diagnosis converged on **bfloat16 covariance storage** as the dominant root
cause: fp32 accumulation on GPU is correct, but
`InputCovarianceAccumulator.finalize_layer` casts to `storage_dtype` before
spilling to disk. Both Stage 2 and Stage 3 defaulted that dtype to bf16
(7 mantissa bits), which round-tripped the small eigenvalues of `B = XᵀX`
to zero — manufacturing rank deficiency that no `1e-6·I` regularization
rescues. CI did not catch it because `tests/conftest.py` overrides
`covariance_storage_dtype` to `float32`.

Fixes applied (full plan kept in private operator notes; key decisions
captured here):

- `stage3_svd._aa_svd` rewritten around a **truncated symmetric
  eigendecomposition of B** with a relative noise floor (`λ > 1e-6·λ_max`).
  Effective-rank-clipped `k_eff` replaces Cholesky-of-(B+εI); rel-err is
  computed from singular value tails of `M = W·L_B`, removing the
  cancellation that produced `rel_err ≫ 1`.
- `stage4_eora._compute_eora_factors` hardened the same way: drop the
  `1e-6·I`, use eigh with the relative threshold, zero-pad U/V when the
  effective rank is below `r` so the caller's fixed-shape tensors stay
  consistent.
- Storage dtype switched **bf16 → float16** at both call sites
  (`covariance_storage_dtype: float16`; new `stage3_svd.bcov_storage_dtype:
  float16`). Same byte count as bf16, but fp16's 10 mantissa bits remove
  the silent precision loss at the source.
- `per_projection_weight` reset to uniform (1/1/1) — the prior
  1.75/1.35/0.35 was tuned against bug-artifact recon errors. Re-tune from
  the (now trustworthy) `stage3/recon_rel_err/{name}` after one Stage 3
  re-run.
- New regression tests (the bf16 covariance bug's primary guard is the first):
  - [`tests/test_aa_svd_bf16_quantized.py`](../tests/test_aa_svd_bf16_quantized.py)
  - [`tests/test_aa_svd_fp16_quantized.py`](../tests/test_aa_svd_fp16_quantized.py)
  - [`tests/test_eora_bf16_A.py`](../tests/test_eora_bf16_A.py)
  - [`tests/conftest.py`](../tests/conftest.py) adds `tiny_config_bf16`;
    [`tests/test_smoke_stage3.py`](../tests/test_smoke_stage3.py) is now
    parametrized over both fp32 and bf16 covariance storage.
- `rel_err` semantics shifted: it is now the relative singular-value-tail
  ratio of `M = W·L_B`, equal to the prior `‖(W−UV)L_B‖/‖WL_B‖` whenever
  `k_eff < rank(M)`. Trackio numbers from runs after the fix are not
  directly comparable to the bug-era numbers in the 10²–10⁶ range.
- `stage3/k_eff_clip_count/{name}` now logged per layer to surface budget
  vs effective-rank under-spend.

Re-run sequence (HF Jobs). Each job is gated by the prior stage's Hub
upload (the bucket FUSE mount is non-durable on cancel/timeout, so per-stage
Hub commits are the only safe handoff). Per-job env vars:
`RESUME_FROM_STAGE=N`, `STOP_AFTER_STAGE=N`, and `PRIOR_STAGE_REPO=` the
previous job's `<base>-stageN` repo. See
[`docs/huggingface_jobs_and_buckets.md`](huggingface_jobs_and_buckets.md)
for the full durability rationale.

| Job | Stages | `--timeout` | Reason |
|---|---|---|---|
| A | 2 only | 5h | Regenerate fp16 A sidecar (~3 h on a100-large) |
| B | 3 only | 3h | Math fixes + new B with float16 storage (~30 min) |
| C | 4 + 5 chained | 6h | Stage 4 consumes A; Stage 5 trains router (~5 h) |
| D | 6 only | 5h | Final quality gate (~1–2 h) |

Total ~12–18 GPU hours. Per-stage verification gates are in §5 of the plan.

---

## 7. References

- Code repo on Hub (snapshots of the pipeline, downloaded into HF Jobs at run time): [`pirola/moe-compress-code`](https://huggingface.co/datasets/pirola/moe-compress-code)
- Trackio dashboard: [`pirola/trackio`](https://huggingface.co/spaces/pirola/trackio)
- Bucket (scratch, non-durable): `hf://buckets/pirola/moe-cache`
- Detailed durability rationale: [`docs/huggingface_jobs_and_buckets.md`](huggingface_jobs_and_buckets.md)
- HF Jobs operations runbook: [`docs/hf_jobs_operations.md`](hf_jobs_operations.md)
- Stage memory profiles: [`docs/stage_memory_profiles.md`](stage_memory_profiles.md)
