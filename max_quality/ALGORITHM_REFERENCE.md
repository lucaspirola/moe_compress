# Strategy A — Maximum Quality MoE Compression: Algorithm Reference

**Pipeline:** `max_quality/` in [`pirola/moe-compress`](https://huggingface.co/datasets/pirola/moe-compress/tree/main/max_quality)
**Target model:** [`Qwen/Qwen3.6-35B-A3B`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) — 35B parameter sparse MoE, 256 routed experts per layer, top-8 routing, 40 MoE decoder layers, `moe_intermediate_size=512`, `hidden_size=2048`.
**Goal:** 30% total parameter reduction with ≤3% relative WikiText-2 PPL increase and ≤1.5pp zero-shot accuracy drop.
**Config:** [`configs/qwen36_35b_a3b_30pct.yaml`](configs/qwen36_35b_a3b_30pct.yaml)
**Code review date:** 2026-04-29

This document is the **single authoritative reference** for the algorithms implemented in this pipeline. Every formula, every paper citation, every hyperparameter, and every known deviation from the cited papers is documented here. Future code reviews should verify the implementation against this document, not against the original papers directly — the deviations are deliberate and documented.

---

## Table of Contents

1. [Introduction and Pipeline Overview](#1-introduction-and-pipeline-overview)
2. [Calibration Data](#2-calibration-data)
3. [Budget Solver](#3-budget-solver)
4. [Stage 0 — Super Expert Detection](#4-stage-0--super-expert-detection)
5. [Stage 1 — GRAPE Budget Allocation](#5-stage-1--grape-budget-allocation)
6. [Stage 2 — REAP Scoring + REAM Pseudo-Pruning](#6-stage-2--reap-scoring--ream-pseudo-pruning)
6.5. [Stage 2.5 — Post-Merge Router Calibration](#65-stage-25--post-merge-router-calibration)
7. [Stage 3 — Non-Uniform SVD Factorization](#7-stage-3--non-uniform-svd-factorization)
8. [Stage 4 — EoRA Residual Compensation](#8-stage-4--eora-residual-compensation)
9. [Stage 5 — Router Knowledge Distillation (Final)](#9-stage-5--router-knowledge-distillation-final)
10. [Stage 6 — Validation](#10-stage-6--validation)
11. [Protected Components](#11-protected-components)
12. [Durability and Crash-Resume Model](#12-durability-and-crash-resume-model)
13. [Known Deviations from Papers](#13-known-deviations-from-papers)
14. [References](#14-references)

---

## 1. Introduction and Pipeline Overview

The pipeline compresses MoE models through a sequence of complementary techniques applied in a fixed order. The ordering is not arbitrary — each stage's output depends on prior stages, and later stages must see the final expert behaviour to calibrate correctly.

```
Stage 0: Super Expert Detection     → blacklist JSON (experts too important to touch)
   ↓
Budget Solver                        → ep:sp decomposition (how much to prune vs. factorise)
   ↓
Stage 1: GRAPE Budget Allocation     → per-layer expert budgets (non-uniform)
   ↓
Stage 2: REAP Score + REAM Merge     → pruned model checkpoint + input covariances (A)
   ↓
Stage 2.5: Post-Merge Router KD      → router weights calibrated to merged expert landscape
   ↓
Stage 3: Non-Uniform SVD             → factored model (FactoredExperts) + cross-covariance (C)
   ↓
Stage 4: EoRA Compensation           → widened factors (rank recovery)
   ↓
Stage 5: Router KD (Final)           → router weights recalibrated to factored+compensated model
   ↓
Stage 6: Validation                  → quality metrics + pass/fail gate
```

The total parameter reduction compounds multiplicatively:

```
(1 − expert_prune_ratio) × (1 − svd_rank_ratio) ≈ (1 − target_ratio)
```

applied to the **compressible pool** (routed expert weights only). Non-compressible parameters (attention, shared experts, embeddings, lm_head, layer norms, router weights) count toward the total denominator but are never modified (except router weights in Stages 2.5 and 5).

**Hardware:** Full pipeline runs on a single H200 (141 GB VRAM). Stages 2.5, 3, 4, and 5 keep the student model resident between stages — no inter-stage reload. Stage 5 runs twice: once after Stage 2 (as Stage 2.5) and once after Stage 4 (final). Stages 0–1 are weight-space only and run on any machine.

---

## 2. Calibration Data

**Source:** [`nvidia/Nemotron-Cascade-2-SFT-Data`](https://huggingface.co/datasets/nvidia/Nemotron-Cascade-2-SFT-Data)

| Subset | Weight | Rationale |
|--------|--------|-----------|
| chat | 0.56 | Dominant deployment traffic |
| math | 0.21 | Heavy reasoning tail |
| science | 0.11 | Reasoning diversity |
| instruction_following | 0.034 | Instruction adherence |
| conversational_agent | 0.033 | Tool-use / agent traces |
| terminal_agent | 0.033 | CLI agent coverage |
| swe | 0.02 | Software engineering |

**Why this matters:** MoE expert routing is task-aware. The calibration distribution defines which experts are treated as redundant. A math-skewed calibration would protect math experts at the expense of chat experts, producing a model biased toward the calibration distribution.

**Tokenization:** Rows are rendered via the tokenizer's chat template (`apply_chat_template`), then concatenated with EOS separators and chunked to `sequence_length=2048`.

**Disjoint draws per stage:** Each stage uses a different `seed_offset` from the base seed (`1337`), ensuring Stages 0, 2, 3, and 5 draw from independent shuffles of the same distribution.

| Stage | seed_offset | num_sequences | sequence_length |
|-------|-------------|---------------|-----------------|
| 0 | +1 | 100 | 2048 |
| 2 | +0 | 1024 | 2048 |
| 3 (B-cov) | +2 | 512 | 2048 |
| 5 (KD) | +5 | 3000 | 512 |

---

## 3. Budget Solver

**File:** [`budget/solver.py`](src/moe_compress/budget/solver.py)

Given a `target_total_reduction` (e.g., 0.30) and an `expert_svd_ratio` (e.g., 2.0 meaning pruning removes 2× the params that SVD removes), the solver iteratively finds `(expert_prune_ratio, svd_rank_ratio)` such that the projected total savings hits the target within a 0.5% tolerance.

**Algorithm:**
1. Analytical starting point: `sp ≈ target × total_params / (expert_params × (ratio + 1))`
2. Project savings: `ep × expert_params` → surviving experts → `after_prune × (1 − sp)` → total savings
3. If outside tolerance, scale both knobs by `target / projected`, maintaining the ep:sp ratio exactly
4. Converges in ≤3 iterations for typical targets

**Floor constraints:** No layer can go below `min_experts_per_layer` (default 128 for 256-expert layers) or below the number of blacklisted experts. The solver enforces these when projecting the expert budget.

---

## 4. Stage 0 — Super Expert Detection

**File:** [`stage0_super_experts.py`](src/moe_compress/stage0_super_experts.py)
**Paper:** Super Experts in MoE Models (2507.23279)
**Hardware:** H200. Original BF16 model (~70 GB) leaves 71 GB VRAM headroom — no quantization or memory tricks needed.

### What

Identifies a small set of "super experts" — experts with disproportionately large activation magnitudes — that must never be pruned, merged, or factored because their removal causes catastrophic quality collapse (e.g., −21.7% on Qwen3-30B).

### Why

Super experts carry outsized influence on model output despite being activated at normal frequency. Pruning them produces errors that no downstream compensation (SVD, EoRA, Router KD) can recover.

### How

Stage 0 is already single-pass: all 40 MoE layers are instrumented simultaneously and `run_calibration` runs once over all 100 samples.

1. Run 100 calibration samples through the model with `instrument_experts` callbacks recording `max(|down_proj_output|)` per (layer, expert)
2. Per layer: z-score the max activation values
3. Flag experts exceeding `mean + 2.5σ`
4. Cap at `max_blacklisted_per_layer=4` per layer, `global_blacklist_cap_pct=5%` globally
5. Emit `stage0_blacklist.json`

### Formula

```
blacklisted(l, e) = max_activation(l, e) > μ_l + z · σ_l
```

where `z = 2.5` (configurable), `μ_l` and `σ_l` are the mean and standard deviation of max activations across all experts in layer `l`.

### Resume

Stage 0 is stateless (JSON-only output, no model mutation). Re-running is cheap and always safe.

---

## 5. Stage 1 — GRAPE Budget Allocation

**File:** [`stage1_grape.py`](src/moe_compress/stage1_grape.py)
**Paper:** GRAPE: Greedy Redundancy-Aware Pruning for MoE (2604.06542), §3.2–3.3, Algorithm 1
**Hardware:** H200. Pure weight-space computation — no forward passes, no calibration data. Model stays loaded from Stage 0. Distance matrix computation (`cdist`, cosine similarity on 256-expert banks) runs on GPU.

### What

Computes a **non-uniform** per-layer expert budget: layers with more redundant experts donate more of their capacity, while layers with diverse experts keep more.

### Why

Uniform pruning wastes budget — some layers have highly redundant experts (high pairwise similarity) while others are diverse. GRAPE's +2.45% peak on Mixtral-8x22B at the 4-expert setting (paper Table 2) demonstrates the value of non-uniform allocation.

### How — GRAPE Algorithm 1 (entropy-aware greedy merge with restart)

1. **Compute pairwise distance matrices** `D^l` for all MoE layers using a pluggable metric (cosine similarity on flattened expert weight vectors by default — paper §3.2 explicitly allows "CKA, MSE, or other similarity measures")

2. **Initialize** each expert as its own cluster. Compute per-layer redundancy `R^l = Σ_{i≠j} D^l_{ij}` (Eq. 11, sum form). Set entropy threshold `Ê = E_0 × (1 − γ)` where `E_0` is the initial cross-layer entropy and `γ=0.1` (Eq. 10 — γ is project-chosen; paper gives no default).

3. **Greedy loop** until total surviving experts ≤ `global_expert_budget`:
   - If all layers frozen → **restart** (unfreeze all)
   - Pick `l* = argmax R^l` among unfrozen layers above their floor
   - Pick `(i*, j*) = argmax D^{l*}_{ij}` (most similar pair)
   - **Merge:** zero out `j*`'s row/column in `D^{l*}`, update `R^{l*}`
   - Decrement `cluster_counts[l*]`
   - If entropy drops below `Ê` → **freeze** layer `l*`

4. **Floor constraints:** `min_experts_per_layer=64`, plus `early_layer_bonus=+8` for first 4 MoE layers and `late_layer_bonus=+8` for last 5 MoE layers (protects input embedding propagation and output generation).

### Key Formulas

```
R^l = Σ_{i≠j} D^l_{ij}                          (Eq. 11 — sum, not mean)
R̃^l = (R^l − min R) / (max R − min R)           (Eq. 3 — for logging only)
Ê = E_0 × (1 − γ)                                (Eq. 10 — entropy threshold)
E = −Σ_l (n_l / N_total) × log(n_l / N_total)   (cross-layer entropy)
```

### Resume

Stage 1 is stateless (JSON-only output). Re-running is cheap and always safe.

### Correctness Notes

- The `R^l` update zeroes out the merged expert's *entire* row and column (not just the pair), preventing the absorbed expert from being re-selected in future iterations. This is more correct than the paper's pseudocode line 12 (`R^l ← R^l − 2·D[i*,j*]`), which appears to assume upper-triangular unique-pair counting.
- If the budget cannot be reached (all layers hit their floors), a warning is logged but the pipeline continues with the achieved budget.

---

## 6. Stage 2 — REAP Scoring + REAM Pseudo-Pruning

**File:** [`stage2_reap_ream.py`](src/moe_compress/stage2_reap_ream.py)
**Papers:**
- REAP: Routing-Expert Activation Pruning (2510.13999), Eq. 9
- REAM: Routing Expert Activation Merging (2604.04356), §3–4, Eq. 5–8
**Hardware:** H200. Model (70 GB BF16) stays loaded from Stage 1. 71 GB VRAM headroom enables `batch_size=24+` (A100 was capped at 6 due to 10 GB headroom) and single-pass profiling across all layers.

### What

Reduces the number of routed experts per layer from 256 to ~180–200 by merging similar experts (not deleting — merged experts' knowledge is preserved via frequency-weighted averaging). Simultaneously collects input covariance matrices (A) consumed by Stages 3 and 4.

### Why

Expert merging preserves more knowledge than deletion. REAM's pseudo-pruning (scoring + assignment + merge) was shown to retain 98.4% of the original model's quality on Qwen3-30B at ~22% expert reduction, outperforming pure pruning methods.

### How

**Single-pass profiling on H200:** All 40 MoE layers are instrumented simultaneously and `run_calibration` runs once over the full calibration set (not once per layer as on A100). After the pass completes, each layer's merge decisions and covariance accumulation are finalised sequentially (weight-space only — fast). Merges are applied layer by layer in order so each layer sees the updated activations from prior merges.

**Per-layer merge execution (sequential — must see prior merges):**

#### Step 1: REAP Scoring (Paper 2510.13999, Eq. 9)

For each expert `j`, compute importance as the conditional average of gate-weighted output norm over active tokens:

```
S_j = (1/|X_j|) × Σ_{x ∈ X_j} g_j(x) · ‖f_j(x)‖₂
```

where `X_j = {x | j ∈ TopK(σ(x))}`, `g_j(x)` is the post-softmax routing weight, and `f_j(x)` is the expert output vector.

#### Step 2: REAM Cost Matrix (Paper 2604.04356, Eq. 5 & 8)

**Activation-space similarities** (NOT weight-space):

- **δ_gate(i,j)** (Eq. 5): Cosine distance between **pre-softmax** router logit profile vectors. Each expert's profile is a vector indexed by global token position, containing the pre-softmax logit for that token (captured via `capture_router_outputs` pre-forward hook on the router module, which recomputes `F.linear(hidden, router.weight)`). Pre-softmax logits can be negative and unbounded, giving the full `[-1, 1]` cosine similarity range.

- **δ̃_expert(i,j)** (Eq. 8): Per-token cosine similarity of gated expert outputs `σ(x)_e · E_e(x)`, averaged over the full calibration set X. Gate weights naturally suppress inactive tokens (σ(x)_e → 0 for non-active tokens), so the denominator is |X| not the jointly-active count — matching paper Eq. 8. Accumulated incrementally per batch via `finalize_batch` in `activation_hooks.py`.

- **δ_REAM(i,j) = δ_gate(i,j) + δ̃_expert(i,j)** (Eq. 7): Unweighted sum (paper uses equal weight 1.0).

#### Step 3: Greedy Pseudo-Pruning Assignment (Paper §4)

Top-N'_l experts by REAP score become **centroids** (protected from removal). Non-centroids are assigned to centroids via **greedy pseudo-pruning**:

1. Iterate centroids in descending saliency order
2. For each centroid, absorb the most similar (highest δ_REAM) unassigned non-centroid
3. Repeat until all non-centroids are assigned

#### Step 4: Frequency-Weighted Merge (Paper Eq. 6)

```
W_merged = Σ_i (freq_i / Σ_j freq_j) × P_i(W_i)
```

where `P_i` is a neuron permutation alignment (Hungarian algorithm on combined cost matrix `C = C_wt + C_act`) that aligns each child expert's intermediate neurons to the centroid before averaging. `C_wt` is the gate+up Frobenius weight distance; `C_act` is the per-neuron mean activation magnitude L2 distance from `ReamCostAccumulator.get_neuron_mean`.

#### Step 5: Router Resize

Remove merged experts' rows from `gate.weight`. Update `num_experts` on the MoE block.

### Covariance Side-Collection

During the profiling forward pass, two covariance matrices are accumulated per (layer, expert):
- **A_gate_up** (`gate_proj`): Input covariance for gate_proj and up_proj (shared tensor)
- **A_down** (`down_proj`): Input covariance for down_proj (intermediate activations)

Stored in `_stage2_input_covariance.pt` (fp16 storage to avoid bf16 precision loss). On H200 with `batch_size=24+`, the covariance accumulates more signal per pass, improving the A-weighted SVD quality in Stage 3.

### Budget Bump Loop

Two safety gates can raise the effective target if merge quality is poor:
- **`max_merge_group_size=3`**: If any group exceeds this, bump target
- **`ream_cost_sigma_threshold=1.5`**: If mean cost exceeds `running_mean × (1 + 1.5)`, bump target (inactive for first 4 layers)

### Resume

Per-layer atomic checkpointing to `_stage2_partial/`:
- `merge_{layer_idx}.json`: centroid IDs, groupings, frequencies, merge map
- `layer_{layer_idx}.pt`: covariance snapshot for this layer

On resume, completed layers are replayed from partial files (fast, no forward pass). The model must be passed in pre-merge state (Stage 1 output) — a guard checks `num_routed_experts` matches the pre-merge count.

**Critical invariant:** Covariance remapping (`_remap_covariance_for_layer`) must happen BEFORE the snapshot. Snapshotting before remapping persists pre-merge expert keys, corrupting Stage 3 inputs on resume.

---

## 6.5. Stage 2.5 — Post-Merge Router Calibration

**File:** [`stage5_router_kd.py`](src/moe_compress/stage5_router_kd.py) (same code as Stage 5)
**Paper:** Router Knowledge Distillation for MoE Compression (2603.02217)
**Hardware:** H200 required (teacher BF16 ~70 GB + student BF16 ~50 GB = ~120 GB VRAM)

### What

Runs the Router KD algorithm (identical to Stage 5) on the Stage 2 output — before SVD factorization. Trains only `mlp.gate.weight`; all expert weights remain frozen.

### Why

After Stage 2, the router has been **resized** (rows for deleted experts removed) but never retrained. The surviving router weights were calibrated for the original 256-expert landscape. They now route among ~180–200 merged experts whose weight distributions have shifted. Stage 3's covariance collection runs on this degraded routing — better routing at this point means the cross-covariance and B-covariance collected by Stage 3 are more representative of actual inference-time token distribution per expert.

Stage 2.5 is distinct from Stage 5: Stage 5 recalibrates routers after SVD factorization and EoRA. Stage 2.5 recalibrates after merging only. Both are needed: the model changes again in Stages 3 and 4, making Stage 2.5's routers stale again — Stage 5 corrects this. The full chain is: merge → heal routers (2.5) → factorize → compensate → heal routers again (5).

### How

Identical to Stage 5 (§9), with two differences:

| Parameter | Stage 2.5 | Stage 5 |
|---|---|---|
| Input model | Stage 2 output (dense merged experts) | Stage 4 output (FactoredExperts + EoRA) |
| Teacher precision | BF16 — both models fit on H200 | BF16 on H200; NF4 fallback for A100 |
| Checkpoint prefix | `_stage2p5_partial/` | `_stage5_partial/` |
| Hub artifact | `<base>-stage2p5` | `<base>-stage5` |

### Resume

Same step-boundary checkpointing as Stage 5, under `_stage2p5_partial/`.

---

## 7. Stage 3 — Non-Uniform SVD Factorization

**File:** [`stage3_svd.py`](src/moe_compress/stage3_svd.py)
**Papers:**
- D-Rank: Spectral entropy for rank allocation (2509.25622), Eq. 2, 6
- AA-SVD: Anchored Adaptive SVD (2604.02119), Theorem 3.2, Corollary 3.3
- SVD-LLM V2: Heterogeneous rank allocation (2503.12340)
- Swift-SVD+: Dynamic rank allocation (2604.01609), Algorithm 2
**Hardware:** H200. Pruned student model stays resident from Stage 2.5. Original BF16 model also loaded for cross-covariance dual-forward (~120 GB VRAM total).

### What

Replaces each surviving expert's 3 dense matrices (`gate_proj`, `up_proj`, `down_proj`) with rank-k factors `W ≈ U · V`, where `U ∈ ℝ^{d_out × k}` and `V ∈ ℝ^{k × d_in}`. Rank `k` varies across (layer, matrix_type) groups — high-entropy matrices get more rank, low-entropy matrices get less.

### Why

SVD factorization reduces parameters from `d_out × d_in` to `k × (d_out + d_in)` per expert. With activation-aware rank allocation and weighting, the factorization concentrates error into directions the model rarely uses, preserving quality far better than plain truncated SVD.

### How

#### Phase A: Covariance Collection (B and cross-covariance C)

**B-covariance** `B = X_post^T X_post`: Run the pruned (post-Stage-2) model on fresh calibration data to collect per-(layer, expert, matrix) input covariances reflecting the distribution the compressed model will see after merging.

**Cross-covariance** `C = X_pre^T X_post` (H200 only): Run both the original model and the pruned model on the same calibration batch simultaneously. For each batch: forward original → collect `X_pre`; forward pruned → collect `X_post`; accumulate `C += X_pre^T @ X_post`. This gives the exact cross-covariance required by AA-SVD Theorem 3.2 and eliminates the auto-covariance approximation.

Requires H200 (141 GB VRAM): original BF16 (~70 GB) + pruned BF16 (~50 GB) = ~120 GB. On A100 the cross-covariance path is skipped and Phase C falls back to the auto-covariance approximation.

**Single-pass collection:** All 40 MoE layers are hooked simultaneously in a single calibration pass (not one pass per layer). Peak CPU RAM is ~88 GB for all accumulated covariances; this is feasible on H200 instances. Per-layer spill to disk still applies after each layer's accumulation is finalised, keeping the resident footprint bounded.

All covariances are written to `_stage3_bcov_partial/layer_{idx}.pt`. Background I/O thread overlaps spill with the next batch's forward pass.

#### Phase B: D-Rank Allocation (Paper 2509.25622, Eq. 2 & 6)

For each (layer, matrix_type) group:

```
p_i = σ_i² / Σ_j σ_j²                        (normalized squared singular values)
R_eff = exp(−Σ_i p_i · log(p_i))              (Eq. 2 — effective rank)
k_g = √(R_eff(g) / ω) × T_budget / Σ_{g'} √(R_eff(g') / ω)   (Eq. 7 — rank allocation; Eq. 6 gives proportionality only)
```

where `ω = n_experts × (d_out + d_in)` is the per-rank parameter cost and `T_budget` is the global rank budget derived from `svd_rank_ratio`.

**Per-projection bias** (budget-neutral): `gate_proj=1.33`, `up_proj=0.67`, `down_proj=1.0`. Derived from SwiGLU error sensitivity: gate errors are amplified by SiLU; up errors are bounded; down errors propagate to all downstream layers.

#### Phase B.2: Swift-SVD+ Per-Expert Rank Redistribution (Paper 2604.01609, Algorithm 2)

Within each (layer, matrix_type) group, D-Rank gives a uniform rank `k_g` to every expert. Swift-SVD+ refines this by redistributing the group's total rank budget `k_g × N_experts` across individual experts using a blending score:

```
s_i = β_i^α · (log(e + ε*_i))^{1-α}
```

where:
- `β_i = σ_i² / Σ_j σ_j²` — spectral energy proportion (how much of the group's total spectral energy this expert contributes)
- `ε*_i = √(Σ_{j>k̄} σ_j² / Σ_j σ_j²)` — reconstruction error at the group's mean rank `k̄` (higher = this expert needs more rank)
- `α ∈ [0, 1]` — balances the two signals

**α grid search:** For each projection type (if `per_group_type: true`), evaluate all `α ∈ {0.0, 0.1, ..., 1.0}` and pick the α that minimises total tail spectral energy across all experts in that type. This takes seconds (purely weight-space SVD, no forward passes).

Per-expert ranks are stored in the `FactoredExperts` slot at the max rank across experts in the group (zero-padded for experts with lower rank). `effective_ranks` tracks the true per-expert rank for honest parameter counting.

#### Phase C: Hybrid Activation-Aware SVD

For each (layer, expert, matrix):

**When cross-covariance C and B are available (H200 path — paper-exact):**

```
M = W · C · B⁻¹ · L_B        where C = X_pre^T X_post
```

This is the exact AA-SVD Theorem 3.2 formula. `L_B` is the eigendecomposition-based square root of B.

**When only A (pre-prune auto-cov) and B are available (A100 fallback):**

```
M = W · A_cov · B⁻¹ · L_B    where A_cov = X_pre^T X_pre  (auto-covariance approximation)
```

The two coincide when pre/post distributions are similar (light pruning). See §13 for deviation status.

**When only B is available (A = None):** Falls back to Corollary 3.3:

```
M = W · L_B
```

Then: `SVD(M) = U Σ V^T`, `U_k = U[:,:k] · S[:k]`, `V_k = V^T[:k,:] · L_B⁻¹`.

**Numerical safeguards:**
- Eigendecomposition replaces Cholesky (handles rank-deficient B natively)
- Dtype-aware noise floor: `bf16→1e-2`, `fp16→1e-3`, `fp32→1e-6`
- `k_eff = min(k, r_eff)` — never allocates rank beyond B's effective rank
- Zero-padding when `k_eff < k` so FactoredExperts tensors stay shape-stable

#### Phase D: L-BFGS Block Refinement (currently enabled)

Per-matrix activation-weighted reconstruction refinement:

```
min_{U,V} ‖(W − U·V) · A^{1/2}‖²_F
```

using L-BFGS with strong Wolfe line search (`lbfgs_steps=100`, `lbfgs_history=10`). B-weighted regression is monitored — if refine worsens the B-weighted norm, a warning is logged.

### Resume

- B-cov spill files at `_stage3_bcov_partial/layer_{idx}.pt` — layers whose spill files already exist are skipped on re-entry
- Spill directory is cleaned up on successful Stage 3 completion
- Original weights snapshot (`_stage3_original_weights.pt`) is saved for Stage 4 residual computation

---

## 8. Stage 4 — EoRA Residual Compensation

**File:** [`stage4_eora.py`](src/moe_compress/stage4_eora.py)
**Paper:** EoRA: Training-Free Compensation for Compressed LLMs (2410.21271), Algorithm 1
**Hardware:** H200. Activation covariance reused from Stage 2 — no additional forward passes needed. FactoredExperts model stays resident from Stage 3; `_stage3_original_weights.pt` remains in CPU RAM. No inter-stage reload.

### What

For each factored expert matrix, computes the residual `ΔW = W_original − U·V` and adds a rank-r correction that concentrates on the **most important input directions** (as measured by the pre-prune activation covariance eigenvalues). The correction is appended to the existing factored representation by widening U and V along the rank dimension.

### Why

EoRA recovers quality lost to rank truncation in Stage 3. The paper reports +10.84pp ARC-C on LLaMA3-8B (in the paper's 3-bit quantization experiment — not applicable to our BF16 pipeline, cited for magnitude context only). The key innovation over naive SVD of the residual is the √Λ-weighted eigenspace projection, which concentrates the correction rank budget on directions the model actually uses.

### How — Paper Algorithm 1

For each (layer, expert, matrix):

1. **Residual:** `ΔW = W_orig − U_old · V_old` — shape `[d_out × d_in]`

2. **Eigendecompose** activation covariance `A = X̃X̃^T = QΛQ^T` — shape `[d_in × d_in]`. Keep only `n_keep` eigenvectors above the dtype-aware noise floor.

3. **√Λ-scaled projection:** `Q' = Q_keep · √Λ_keep` — shape `[d_in × n_keep]`. This is the **full** signal eigenspace, NOT truncated to `r`. The √Λ scaling importance-weights each direction by its activation variance.

4. **Full projection:** `ΔW' = ΔW · Q'` — shape `[d_out × n_keep]`

5. **Rank-r SVD:** `SVD(ΔW') → U', Σ', V'^T`. Take top `take_eff = min(r, min(d_out, n_keep))`.

6. **Correction factors:**
   - `U_corr = U'[:, :take_eff] · Σ'[:take_eff]` — shape `[d_out × take_eff]`
   - `V_corr = V'^T[:take_eff] · (√Λ)⁻¹ · Q^T` — shape `[take_eff × d_in]` (back-projected to original weight space)

7. **Widen:** `new_U = [U_old | U_corr]`, `new_V = [V_old; V_corr]` — algebraically equivalent to `Ŵ·x + B'·A·x`

### Budget

`compensation_budget_pct=3%` of Stage 3 savings per matrix, capped at `eigenspace_rank_cap=64` rank per expert.

### Correctness Notes

- The √Λ scaling is the **core** innovation of EoRA. Without it, the algorithm degenerates to the Act-S baseline (naive truncated-eigenvector projection).
- Pre-truncating to `r` eigenvectors before SVD (the previous bug) eliminates the joint optimization that makes EoRA better than Act-S. The SVD must operate on the full `[d_out × n_keep]` projected error to optimally select the best `r` directions.
- The back-projection `V_corr = (V'^T · (√Λ)⁻¹) · Q^T` is critical — without the `(√Λ)⁻¹` term, the correction adapter operates in eigenspace rather than weight space, and the errors compound.

### Resume

Per-layer atomic checkpointing to `_stage4_partial/layer_{layer_idx}.pt` (format_version=1). Each checkpoint contains the full FactoredExperts U/V state, ranks, effective ranks, and parameter counts. On resume, completed layers are loaded directly; failed layers re-run from the Stage 3 output.

Stage 4 deletes `_stage3_original_weights.pt` and `_stage2_input_covariance.pt` on success (both are already durable on per-stage Hub repos).

---

## 9. Stage 5 — Router Knowledge Distillation (Final)

**File:** [`stage5_router_kd.py`](src/moe_compress/stage5_router_kd.py)
**Paper:** Router Knowledge Distillation for MoE Compression (2603.02217), Eq. 3, Table 1, §F.3
**Hardware:** H200. EoRA-compensated student model stays resident from Stage 4. Teacher loads in BF16 (~70 GB); combined VRAM ~120 GB.

### What

Trains **only** the router gate weights to match the original (uncompressed) teacher's output distribution via vocabulary-level KL divergence. All expert weights are frozen.

### Why

After Stages 2–4, the expert weights have changed but the router weights still reflect the original expert set. The router sends tokens to suboptimal experts, degrading quality. Router KD recalibrates routing decisions to the new expert landscape by matching the teacher's next-token prediction distribution — the gradient signal flows backward through the routing decisions, naturally adapting them.

The paper explicitly uses vocabulary-level output distillation (not router-gate-level): "By distilling output logits rather than matching router gate values explicitly, Router KD avoids requiring the teacher and student to share identical expert sets or gate dimensionalities."

### How — Vocabulary-Level KD (Paper Eq. 3)

```
L_RKD = (τ² / N_x) × Σ_t m_{t+1} · KL(softmax(z_T^t / τ) ‖ softmax(z_S^t / τ))
```

where `z_T, z_S ∈ ℝ^{|V|}` are teacher/student vocabulary logits, `m_{t+1} ∈ {0,1}` is the padding mask, `N_x = Σ_t m_{t+1}` is the count of unmasked positions, and `τ=1.0` is the temperature. Calibration sequences are fully packed so `m_{t+1}=1` everywhere in practice.

**Implementation:**
1. Teacher forward pass (no_grad) → vocabulary logits `[B, L, |V|]`
2. Student forward pass (with gradients) → vocabulary logits `[B, L, |V|]`
3. Shift logits: position `t` predicts token `t+1` (standard causal LM)
4. Chunked KL: process `chunk_size=128` sequence positions at a time to bound peak memory at `B × chunk × |V| × 4` bytes (~300 MB at B=4, |V|=150K vs ~1.2 GB unchunked)
5. `F.kl_div(log_softmax(student/τ), softmax(teacher/τ))` = KL(teacher ‖ student) — correct forward KL direction

### Hyperparameters (Paper Table 1, §F.3)

| Parameter | Value | Source |
|-----------|-------|--------|
| Optimizer | AdamW | Paper |
| Learning rate | **5×10⁻⁵** | Paper Table 1 (implementation previously used 1e-5; corrected to match paper) |
| Epochs | 1 | Paper |
| Batch size | 4 | Adapted (paper: 2) |
| Gradient accumulation | 2 | Adapted (paper: 4) |
| **Effective batch size** | **8** | Same as paper (4×2 = 2×4) |
| Max sequence length | 512 | Paper |
| KD temperature (τ) | 1.0 | Paper |
| Max calibration samples | 3000 | Paper |
| Teacher precision | BF16 on H200; NF4 (bitsandbytes, ~17 GB) on A100 | H200 (141 GB VRAM) fits teacher+student in BF16. NF4 is a fallback for the 80 GB A100 only. Student is always BF16. |

### Teacher Loading

On H200 (141 GB VRAM), both teacher and student load in BF16 (~70 GB + ~50 GB = ~120 GB). On A100 (80 GB VRAM), the teacher is quantized to NF4 via bitsandbytes (~17 GB) so the two can co-reside. The student — the pipeline model being compressed — is **always BF16 throughout all stages**. No stage operates on a quantized model. Alternatively, precomputed teacher vocabulary logits can be loaded from a cache file (`teacher_logits_cache` config key) to skip the live teacher entirely.

### Resume

Step-boundary checkpointing to `_stage5_partial/step_{N}.pt` (every 100 optimizer steps). Each checkpoint contains router parameter state + optimizer state. On resume, up to `grad_accum−1` batches of gradient signal from the last incomplete accumulation window are silently dropped. Only the two most recent checkpoints are retained.

---

## 10. Stage 6 — Validation

**File:** [`stage6_validate.py`](src/moe_compress/stage6_validate.py)
**Hardware:** Runs on the same H200 instance as Stage 5 (student model stays resident).

### What

Evaluates the compressed model against the uncompressed teacher on 5 metrics, enforces hard quality gates, and produces an imatrix file for downstream GGUF quantization. On H200 the larger VRAM headroom allows larger batch sizes for WikiText-2 and zero-shot evals, reducing wall-clock time. Generative evals (HumanEval, MATH-500) can use vLLM for significantly faster throughput than the default HuggingFace generate loop.

### Metrics

| Metric | Method | Threshold |
|--------|--------|-----------|
| WikiText-2 PPL | Standard next-token NLL → exp(mean_NLL), seq_len=2048 | ≤ +3% relative |
| ARC-C accuracy | lm-eval harness, 0-shot | ≤ 1.5pp absolute drop |
| HellaSwag accuracy | lm-eval harness, 0-shot | ≤ 1.5pp absolute drop |
| HumanEval pass@1 | exec-based evaluation (in-process) | ≤ 3pp absolute drop |
| MATH-500 accuracy | SymPy symbolic equivalence + \\boxed{} extraction + numeric fallback | ≤ 3pp absolute drop |

### Measured Reduction

The actual parameter reduction is computed from live parameter counts (accounting for effective ranks in FactoredExperts). Must be ≥ 30.0%.

### imatrix Generation

As a zero-overhead side-channel of the student evaluation pass, Stage 6 collects all text fed to the model across every benchmark — WikiText-2 documents, ARC-C and HellaSwag contexts (via `lm-eval log_samples`), HumanEval prompts, and MATH-500 problems — into a single multi-domain calibration file. After the teacher is freed, the final frozen model is converted to F16 GGUF and `llama-imatrix` runs on the combined text, producing `imatrix.gguf`.

This imatrix is significantly richer than the standard WikiText-only calibration: it covers general language, commonsense reasoning, code, and math — the same domains the model will face in real use. The better imatrix directly improves the quality of all I-variant GGUF quants (IQ4_XS, IQ3_M, IQ2_S, etc.) without any additional inference cost.

**Artifacts:**

| File | Description |
|------|-------------|
| `stage6_eval.json` | Quality gate results (metrics + pass/fail) |
| `calibration_imatrix.txt` | Combined eval text; always written (usable even without llama.cpp) |
| `model_f16.gguf` | Intermediate F16 GGUF of the compressed student |
| `imatrix.gguf` | Final importance matrix for GGUF quantization |

llama.cpp is built in the background by the job entrypoint (daemon thread, starts when Stage 1 begins) so the ~5-minute build does not add to wall-clock time. If llama.cpp is unavailable, `calibration_imatrix.txt` is still written and Stage 6 passes normally.

### Resume

Stage 6 is stateless. Re-running is always safe.

---

## 11. Protected Components

These are **never** modified by any compression stage:

- **Shared expert** (`mlp.shared_expert`) at every MoE layer
- **Attention weights** (DeltaNet linear attention + full attention projections)
- **Embeddings** and **lm_head**
- **Layer norms** (RMSNorm)
- **Router weights** — except Stage 5, which updates *only* these
- **Super experts** on the Stage 0 blacklist

---

## 12. Durability and Crash-Resume Model

HF Jobs bucket FUSE mounts are **not durable** under SIGKILL or timeout. The durability boundary is per-stage Hub uploads:

```
<base_repo>-stage2   ← Stage 2 output + covariance sidecar
<base_repo>-stage3   ← Stage 3 output + originals sidecar
<base_repo>-stage4   ← Stage 4 output
<base_repo>-stage5   ← Final compressed model
```

Each heavy stage (2–5) uploads its checkpoint to a per-stage Hub repo immediately on completion. The bucket is treated as scratch cache only.

**Within-stage crash-resume** uses per-layer/per-step atomic checkpoint files written to `_stage{N}_partial/` directories. Format version is checked on load; corrupt or incompatible files cause immediate failure with actionable error messages.

**Critical safety property:** Every partial checkpoint file is written via `.tmp` → `os.replace` (atomic on POSIX). A SIGKILL mid-write leaves at most a `.tmp` file, never a truncated final `.pt`.

---

## 13. Known Deviations from Papers

| Stage | Deviation | Paper Says | Implementation Does | Justification |
|-------|-----------|-----------|-------------------|---------------|
| 0 | Detection threshold | 2507.23279 Eq. 6 + Algorithm 1: two-stage process — first detect MA-formation layers L, then global P_{99.5} AND 0.1·a_max AND l∈L — all three required | Per-layer z-score (mean + 2.5σ); no global statistics; no MA-formation layer pre-filtering; all 40 MoE layers profiled | Avoids two-pass global stat collection and MA-pattern detection; per-layer z-score is conservative in practice (SEs produce z > 10 typically) |
| 0 | Blacklist caps | Paper: purely threshold-based, no caps | max_blacklisted_per_layer=4 and global_blacklist_cap_pct=5% | Safety guardrails against over-blacklisting |
| 1 | Weight-space D^l metric | Paper experiments likely use activation-based CKA | Cosine similarity on flattened weight vectors | Paper §3.2 explicitly allows "CKA, MSE, or other similarity measures" |
| 1 | γ entropy tolerance | 2604.06542 Eq. 10: γ∈[0,1], no default given | γ=0.1 (project-chosen, not from paper) | Paper leaves γ unspecified; 0.1 chosen empirically |
| 1 | Floor and bonus constraints | GRAPE has no floor constraints or layer bonuses | min_experts_per_layer=64; early_layer_bonus=+8 (first 4); late_layer_bonus=+8 (last 5) | Protects embedding propagation and output generation quality |
| 1 | D^l update after merge | 2604.06542 Algorithm 1 lines 11–12: zero only pair entry D_{i*,j*} and D_{j*,i*}; update R^l ← R^l − 2·D_{i*,j*} | Zeros the absorbed expert's entire row and column in D^l; recomputes R^l from updated matrix | Prevents the absorbed expert from influencing future pair-selection; the paper's update assumes the merged expert cannot be re-selected, but only zeroing the pair entry leaves stale similarity values that can distort R^l and layer selection in subsequent iterations |
| 3 | AA-SVD uses auto-covariance on A100 | 2604.02119 Theorem 3.2 requires cross-covariance `X_pre^T X_post` | H200: exact cross-cov via dual-forward; A100: substitutes `X_pre^T X_pre` | Cross-covariance requires both models in VRAM simultaneously (~120 GB); A100 fallback only |
| 3 | D-Rank ω adapted for MoE | 2509.25622 Eq. 7: ω = d₁ + n·d₂ (layers per group × dimensions) | ω = n_experts × (d_out + d_in) | D-Rank targets shared-basis layer groups; adapted for MoE expert groups |
| 3 | Swift-SVD+ β and ε* | 2604.01609 Alg. 2: β = end-to-end layer importance [1,2]; ε* = raw Frobenius loss ‖XW−XW*‖_F | β = per-expert spectral energy share; ε* = normalized tail energy ratio | Layer importance requires extra forward passes; spectral proxy adapted for within-group expert redistribution |
| 3 | α selection criterion | 2604.01609 §3.2.2: select α by validation-set end-to-end performance | Minimises total tail spectral energy — no forward passes | Validation evaluation per α requires 11× model-scale forward passes |
| 4 | Eigenspace noise-floor truncation | 2410.21271 Alg. 1: full Q ∈ ℝ^{k×k} used; QQ^T = I guarantees Theorem 1 exactness | Eigenvectors below noise floor discarded; n_keep < k retained before SVD | Suppresses near-zero noise directions; weakens Theorem 1 exactness but improves numerical stability |
| 5 | Calibration data source | 2603.02217 §F.3 Table 1: calibration dataset = c4 (used identically across all experiments) | Multi-domain Nemotron-Cascade-2-SFT-Data with weighted subsets (chat 0.56, math 0.21, science 0.11, etc.) | Task-aware calibration better matches target deployment distribution; c4 is general pre-training data with limited reasoning/code coverage |
| 5 | Effective batch size 4×2 instead of 2×4 | 2603.02217 Table 1: batch_size=2, grad_accum=4 | batch_size=4, grad_accum=2 | Same effective batch (8); adapted for A100 VRAM headroom with NF4-quantized teacher |

---

## 14. References

| ID | Paper | Year | Used In |
|----|-------|------|---------|
| 2507.23279 | Super Experts in MoE Models | 2025 | Stage 0 |
| 2604.06542 | GRAPE: Greedy Redundancy-Aware Pruning for MoE | 2026 | Stage 1 |
| 2510.13999 | REAP: Routing-Expert Activation Pruning | 2025 | Stage 2 (scoring) |
| 2604.04356 | REAM: Routing Expert Activation Merging | 2026 | Stage 2 (merging) |
| 2509.25622 | D-Rank: Spectral Entropy Rank Allocation | 2025 | Stage 3 (rank budget) |
| 2604.02119 | AA-SVD: Anchored Adaptive SVD for LLMs | 2026 | Stage 3 (factorization) |
| 2604.01609 | Swift-SVD+: Dynamic Non-Uniform Rank Allocation | 2026 | Stage 3 (α grid search) |
| 2503.12340 | SVD-LLM V2: Per-Type Rank Allocation | 2025 | Stage 3 (motivation) |
| 2410.21271 | EoRA: Training-Free Compensation for Compressed LLMs | 2024 | Stage 4 |
| 2603.02217 | Router Knowledge Distillation for MoE Compression | 2026 | Stage 5 |

---

*This document was generated from a full algorithmic review of the max_quality codebase on 2026-04-28; §13 updated 2026-04-29 after a per-stage paper compliance audit including full methodology-section cross-reference of all 10 cited papers. All formulas were verified against the cited papers' methodology sections. All deviations are deliberate and documented. For the original validation audit, see the archived [VALIDATED_STRATEGIES.md](https://huggingface.co/pirola/moe-compression-workflow/blob/main/VALIDATED_STRATEGIES.md).*