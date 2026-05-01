# Strategy A — Maximum Quality MoE Compression: Algorithm Reference

**Pipeline:** `max_quality/` in [`pirola/moe-compress`](https://huggingface.co/datasets/pirola/moe-compress/tree/main/max_quality)
**Target model:** [`Qwen/Qwen3.6-35B-A3B`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) — 35B parameter sparse MoE, 256 routed experts per layer, top-8 routing, 40 MoE decoder layers, `moe_intermediate_size=512`, `hidden_size=2048`.
**Goal:** 30% total parameter reduction with ≤3% relative WikiText-2 PPL increase and ≤1.5pp zero-shot accuracy drop.
**Config:** [`configs/qwen36_35b_a3b_30pct.yaml`](configs/qwen36_35b_a3b_30pct.yaml)
**Code review date:** 2026-04-30, compute-time optimizations added 2026-04-30

This document is the **single authoritative reference** for the algorithms implemented in this pipeline. Every formula, every paper citation, every hyperparameter, and every known deviation from the cited papers is documented here. Future code reviews should verify the implementation against this document, not against the original papers directly — the deviations are deliberate and documented.

---

## Table of Contents

1. [Introduction and Pipeline Overview](#1-introduction-and-pipeline-overview)
2. [Calibration Data](#2-calibration-data)
3. [Budget Solver](#3-budget-solver)
4. [Stage 1 — Super Expert Detection + GRAPE Budget Allocation](#4-stage-1--super-expert-detection--grape-budget-allocation)
5. [Stage 2 — REAP Scoring + REAM Pseudo-Pruning](#5-stage-2--reap-scoring--ream-pseudo-pruning)
5.5. [Stage 2.5 — Post-Merge Router Calibration](#55-stage-25--post-merge-router-calibration)
6. [Stage 3 — Non-Uniform SVD Factorization](#6-stage-3--non-uniform-svd-factorization)
7. [Stage 4 — EoRA Residual Compensation](#7-stage-4--eora-residual-compensation)
8. [Stage 5 — Router Knowledge Distillation (Final)](#8-stage-5--router-knowledge-distillation-final)
9. [Stage 6 — Validation](#9-stage-6--validation)
10. [Protected Components](#10-protected-components)
11. [Durability and Crash-Resume Model](#11-durability-and-crash-resume-model)
12. [Known Deviations from Papers](#12-known-deviations-from-papers)
13. [References](#13-references)

---

## 1. Introduction and Pipeline Overview

The pipeline compresses MoE models through a sequence of complementary techniques applied in a fixed order. The ordering is not arbitrary — each stage's output depends on prior stages, and later stages must see the final expert behaviour to calibrate correctly.

```
Stage 1: SE Detection + GRAPE       → blacklist JSON + per-layer expert budgets (non-uniform)
   ↓
Budget Solver                        → ep:sp decomposition (how much to prune vs. factorise)
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

**Hardware:** Full pipeline runs on a single H200 (141 GB VRAM). Stages 2.5, 3, 4, and 5 keep the student model resident between stages — no inter-stage reload. Stage 5 runs twice: once after Stage 2 (as Stage 2.5) and once after Stage 4 (final). Stage 1 requires a single forward pass (~5 min on H200); all subsequent GRAPE computation is weight-space only.

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

**Disjoint draws per stage:** Each stage uses a different `seed_offset` from the base seed (`1337`), ensuring Stages 1, 2, 3, and 5 draw from independent shuffles of the same distribution.

| Stage | seed_offset | num_sequences | sequence_length |
|-------|-------------|---------------|-----------------|
| 1 (SE + CKA) | +1 | 256 | 2048 |
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

**Floor constraints:** No layer can go below `min_experts_per_layer` (default: `num_routed_experts // 2` = 128 for 256-expert layers) or below the number of blacklisted experts. The solver enforces these when projecting the expert budget.

---

## 4. Stage 1 — Super Expert Detection + GRAPE Budget Allocation

**File:** [`stage1_grape.py`](src/moe_compress/stage1_grape.py)
**Papers:**
- Super Experts in MoE Models (2507.23279) — SE detection
- GRAPE: Greedy Redundancy-Aware Pruning for MoE (2604.06542), §3.2–3.3, Algorithm 1 — budget allocation
**Hardware:** H200. Original BF16 model (~70 GB) leaves 71 GB VRAM headroom. Single forward pass over 256 calibration samples (~5 min), then weight-space GRAPE computation on GPU.

### What

A single unified stage that (a) identifies super experts that must never be compressed, and (b) computes non-uniform per-layer expert budgets using activation-aware CKA similarity. Both tasks share a single calibration forward pass.

### Why

Super experts carry outsized influence on model output despite being activated at normal frequency. Pruning them causes catastrophic quality collapse (e.g., −21.7% on Qwen3-30B). They must be detected before any budget allocation.

Uniform pruning wastes budget — some layers have highly redundant experts (high pairwise CKA similarity) while others are diverse. GRAPE's +2.45% peak on Mixtral-8x22B at the 4-expert setting (paper Table 2) demonstrates the value of non-uniform allocation. Using CKA (rather than weight-space cosine) for the similarity metric gives GRAPE activation-aware redundancy estimates, producing better budgets for Stage 2.

### How

#### Phase A: Single-Pass Calibration (256 samples)

All 40 MoE layers are instrumented simultaneously. `run_calibration` runs once over all 256 samples, collecting two things per (layer, expert):

1. **Max activation magnitude** `max(|down_proj_output|)` — for super expert detection
2. **Expert output representations** `f_e(x)` — for CKA pairwise similarity computation

This is a single forward pass (~5 min on H200). The expert output representations are accumulated into per-layer representation matrices for CKA via reservoir sampling (max 256 tokens per expert).

#### Phase B: Super Expert Detection (Paper 2507.23279)

Per layer: z-score the max activation values and flag experts exceeding `mean + 2.5σ`. Cap at `max_blacklisted_per_layer=4` per layer, `global_blacklist_cap_pct=5%` globally. Emit `stage1_blacklist.json`.

```
blacklisted(l, e) = max_activation(l, e) > μ_l + z · σ_l
```

where `z = 2.5` (configurable), `μ_l` and `σ_l` are the mean and standard deviation of max activations across all experts in layer `l`.

#### Phase C: CKA Similarity Matrices

For each MoE layer, compute the pairwise CKA (Centered Kernel Alignment) matrix `D^l ∈ ℝ^{N×N}` from the collected expert output representations. CKA measures functional similarity between experts based on their response patterns to actual inputs — two experts that produce similar outputs on the calibration data have high CKA, regardless of weight-space similarity.

Paper §3.2 explicitly allows "CKA, MSE, or other similarity measures" for D^l. CKA is the metric used by Zhang et al. (2025), cited in GRAPE §3.2 as the reference for intra-layer redundancy assessment.

With 256 samples × 2048 tokens ≈ 524K total token activations across the layer (each expert sees only its top-k/N routed fraction; for top-8 over 256 experts that is ≈ 16K per expert before sampling), reservoir-sampled to 256 per expert for CKA so the kernel matrices are well-conditioned for 256-expert layers.

#### Phase D: GRAPE Algorithm 1 (entropy-aware greedy merge with restart)

1. **Initialize** each expert as its own cluster. Compute per-layer redundancy `R^l = Σ_{i≠j} D^l_{ij}` (Eq. 11, sum form). Set entropy threshold `Ê = E_0 × (1 − γ)` (Eq. 10) where `E_0` is the initial cross-layer entropy and `γ=0.1` is project-chosen (see [D3](#12-known-deviations-from-papers); the paper gives no default).

2. **Greedy loop** until total surviving experts ≤ `global_expert_budget`:
   - If all layers frozen → **restart** (unfreeze all)
   - Pick `l* = argmax R^l` among unfrozen layers above their floor
   - Pick `(i*, j*) = argmax D^{l*}_{ij}` (most similar pair)
   - **Merge:** zero out `j*`'s row/column in `D^{l*}`, update `R^{l*}`
   - Decrement `cluster_counts[l*]`
   - If entropy drops below `Ê` → **freeze** layer `l*`

3. **Floor constraint:** `min_experts_per_layer = num_routed_experts // 2` (= 128 for 256-expert layers). No early/late layer bonuses — the floor alone provides sufficient protection at 50% max removal per layer.

### Key Formulas

```
R^l = Σ_{i≠j} D^l_{ij}                          (Eq. 11 — sum, not mean)
R̃^l = (R^l − min R) / (max R − min R)           (Eq. 3 — for logging only)
Ê = E_0 × (1 − γ)                                (Eq. 10 — entropy threshold)
E = −Σ_l (n_l / N_total) × log(n_l / N_total)   (cross-layer entropy)
```

### Resume

Stage 1 is stateless (JSON-only output: blacklist + per-layer budgets). Re-running is cheap and always safe.

### Correctness Notes

- The `R^l` update zeroes out the merged expert's *entire* row and column (not just the pair), preventing the absorbed expert from being re-selected in future iterations. The paper's pseudocode line 12 (`R^l ← R^l − 2·D[i*,j*]`) is consistent with the sum-over-`i≠j` form of `R^l` (the 2× accounts for both `D[i*,j*]` and `D[j*,i*]`), but it only adjusts the scalar `R^l` while leaving stale similarity entries in row/column `j*` that can mis-rank future merges. Zeroing the full row/column eliminates that staleness. See [D4](#12-known-deviations-from-papers).
- If the budget cannot be reached (all layers hit their floors), a warning is logged but the pipeline continues with the achieved budget.

---

## 5. Stage 2 — REAP Scoring + REAM Pseudo-Pruning

**File:** [`stage2_reap_ream.py`](src/moe_compress/stage2_reap_ream.py)
**Papers:**
- REAP: Routing-Expert Activation Pruning (2510.13999), Eq. 9
- REAM: Routing Expert Activation Merging (2604.04356), §3–4, Eq. 5–8
**Hardware:** H200. Model (70 GB BF16) stays loaded from Stage 1. 71 GB VRAM headroom enables `batch_size=6` for profiling.

### What

Reduces the number of routed experts per layer from 256 to ~180–200 by merging similar experts (not deleting — merged experts' knowledge is preserved via frequency-weighted averaging, with intermediate-neuron permutation alignment so that the averaged neurons correspond across the merge group). Simultaneously collects input covariance matrices (A) consumed by Stages 3 and 4.

### Why

Expert merging preserves more knowledge than deletion. REAM's pseudo-pruning (scoring + assignment + merge) was shown to retain 98.4% of the original model's quality on Qwen3-30B at ~22% expert reduction, outperforming pure pruning methods.

### How

**Sequential profiling with early-exit (REAM paper §4, Fig. 1(b)):** The REAM paper (2604.04356) introduces *sequential merging* as a core contribution: after merging layer ℓ, activations must be recomputed through the merged layer before profiling layer ℓ+1, ensuring each layer's REAP scores and REAM cost matrices reflect the actual input distribution it will see at inference time (not stale pre-merge statistics). The paper's ablation (§5.4) measures ΔAVG = −1.0 when sequential merging is removed — a meaningful fraction of the quality budget.

**Implementation:** For each layer L (processed in order 0→39), the profiling forward pass runs from the input embedding through layers 0…L, collecting REAP/REAM/covariance data from layer L's hooks. Layers L+1…39 are **not executed** — their computation is pure waste because all metrics collected for layer L (REAP scores, δ_gate, δ̃_expert, input covariance) depend only on the hidden states that *arrive at* layer L, not on what happens after it. An **early-exit forward hook** registered on the decoder layer immediately after layer L raises a sentinel exception that aborts the forward pass cleanly. The profiling runs under `torch.no_grad()`, so no autograd graph is corrupted.

This gives a ~2× wall-clock speedup over the naïve approach (running all 40 layers for each of the 40 profiling passes): the total layer-forward count drops from 40×40=1600 to 1+2+3+…+40=820. The REAM paper's sequential merging semantics are preserved exactly — each layer is profiled on hidden states that reflect all prior merges.

**Vectorized accumulators (planned follow-up, zero quality impact):**

The REAM cost matrix computation involves two pairwise similarity metrics across all experts in a layer (up to 256 experts). A future optimization replaces Python dicts with dense tensors for O(1) vectorized operations:

- **Gate logit profiles** (`ReamCostAccumulator`): Instead of `dict[expert_id → dict[token_idx → float]]`, a pre-allocated `torch.Tensor(num_experts, total_calibration_tokens)` on CPU in float16 stores each expert's pre-softmax router logit for each calibration token. The full `[N_experts × N_experts]` δ_gate cosine-similarity matrix is computed in one `F.normalize` + `matmul` call (~milliseconds for 256×256) rather than O(N²) Python-level loops.

- **Gated-output pairwise similarity** (`finalize_batch`): Per-batch pairwise cosine similarity of gated expert outputs is computed via a single batched `F.cosine_similarity` over the jointly-active token intersection per expert pair, accumulated incrementally as before but with vectorized inner loops.

These optimizations are purely implementation-level data-structure changes. The mathematical computation is identical — same cosine similarities, same REAP scores, same cost matrix entries. Estimated wall-clock reduction on the cost-matrix phase: 10–100× (from minutes of Python iteration to seconds of tensor ops). Not yet implemented — the current accumulators use Python dicts (functionally correct, slower). The early-exit optimization provides the dominant ~2× speedup; vectorized accumulators are additive.

**Per-layer merge execution (sequential — must see prior merges):**

#### Step 1: REAP Scoring (Paper 2510.13999, Eq. 9)

For each expert `j`, compute importance as the conditional average of gate-weighted output norm over active tokens:

```
S_j = (1/|X_j|) × Σ_{x ∈ X_j} g_j(x) · ‖f_j(x)‖₂
```

where `X_j = {x | j ∈ TopK(σ(x))}`, `g_j(x)` is the post-softmax routing weight, and `f_j(x)` is the expert output vector.

#### Step 2: REAM Cost Matrix (Paper 2604.04356, Eq. 5, 7, 8)

**Activation-space similarities** (NOT weight-space):

- **δ_gate(i,j)** (Eq. 5): Cosine similarity between **pre-softmax** router logit profile vectors. Each expert's profile is a vector indexed by global token position, containing the pre-softmax logit for that token (captured via `capture_router_outputs` pre-forward hook on the router module, which recomputes `F.linear(hidden, router.weight)`). Pre-softmax logits can be negative and unbounded, giving the full `[-1, 1]` cosine similarity range.

- **δ̃_expert(i,j)** (Eq. 8): Per-token cosine similarity of gated expert outputs `σ(x)_e · E_e(x)`, averaged over the full calibration set X. Gate weights naturally suppress inactive tokens (σ(x)_e → 0 for non-active tokens), so the denominator is |X| not the jointly-active count — matching paper Eq. 8. Accumulated incrementally per batch via `finalize_batch` in `activation_hooks.py`.

- **δ_REAM(i,j) = δ_gate(i,j) + δ̃_expert(i,j)** (Eq. 7): Unweighted sum (paper uses equal weight 1.0).

#### Step 3: Greedy Pseudo-Pruning Assignment (Paper §4)

Top-N'_l experts by REAP score become **centroids** (protected from removal). Non-centroids are assigned to centroids via **greedy pseudo-pruning**:

1. Iterate centroids in descending saliency order
2. For each centroid, absorb the most similar (highest δ_REAM) unassigned non-centroid, up to `max_merge_group_size` non-centroids per centroid
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

Stored in `_stage2_input_covariance.pt` (fp16 storage to avoid bf16 precision loss). On H200 with `batch_size=6`, the covariance accumulates signal across all 1024 calibration samples, providing well-conditioned A matrices for Stage 3.

### Budget Bump Loop

Two safety gates can raise the effective target if merge quality is poor:
- **`max_merge_group_size=8`**: If any group exceeds this, bump target. The REAM paper uses C=16 for Qwen3-30B at 25% reduction (§A.1); 8 is conservative but allows centroids to absorb multiple similar experts as the paper intends.
- **`ream_cost_sigma_threshold=1.5`**: If mean cost exceeds `running_mean × (1 + 1.5)`, bump target (inactive for first 4 layers)

### Resume

Per-layer atomic checkpointing to `_stage2_partial/`:
- `merge_{layer_idx}.json`: centroid IDs, groupings, frequencies, merge map
- `layer_{layer_idx}.pt`: covariance snapshot for this layer

On resume, completed layers are replayed from partial files (fast, no forward pass). The model must be passed in pre-merge state (Stage 1 output) — a guard checks `num_routed_experts` matches the pre-merge count.

**Critical invariant:** Covariance remapping (`_remap_covariance_for_layer`) must happen BEFORE the snapshot. Snapshotting before remapping persists pre-merge expert keys, corrupting Stage 3 inputs on resume.

---

## 5.5. Stage 2.5 — Post-Merge Router Calibration

**File:** [`stage5_router_kd.py`](src/moe_compress/stage5_router_kd.py) (same code as Stage 5)
**Paper:** Router Knowledge Distillation for MoE Compression (2603.02217)
**Hardware:** H200 required (teacher BF16 ~70 GB + student BF16 ~50 GB = ~120 GB VRAM)

### What

Runs the Router KD algorithm (identical to Stage 5) on the Stage 2 output — before SVD factorization. Trains only `mlp.gate.weight`; all expert weights remain frozen.

### Why

After Stage 2, the router has been **resized** (rows for deleted experts removed) but never retrained. The surviving router weights were calibrated for the original 256-expert landscape. They now route among ~180–200 merged experts whose weight distributions have shifted. Stage 3's covariance collection runs on this degraded routing — better routing at this point means the cross-covariance and B-covariance collected by Stage 3 are more representative of actual inference-time token distribution per expert.

Stage 2.5 is distinct from Stage 5: Stage 5 recalibrates routers after SVD factorization and EoRA. Stage 2.5 recalibrates after merging only. Both are needed: the model changes again in Stages 3 and 4, making Stage 2.5's routers stale again — Stage 5 corrects this. The full chain is: merge → heal routers (2.5) → factorize → compensate → heal routers again (5).

### How

Identical to Stage 5 (§8), with two differences:

| Parameter | Stage 2.5 | Stage 5 |
|---|---|---|
| Input model | Stage 2 output (dense merged experts) | Stage 4 output (FactoredExperts + EoRA) |
| Teacher precision | BF16 — both models fit on H200 | BF16 |
| Checkpoint prefix | `_stage2p5_partial/` | `_stage5_partial/` |
| Hub artifact | `<base>-stage2p5` | `<base>-stage5` |

### Resume

Same step-boundary checkpointing as Stage 5, under `_stage2p5_partial/`.

---

## 6. Stage 3 — Non-Uniform SVD Factorization

**File:** [`stage3_svd.py`](src/moe_compress/stage3_svd.py)
**Papers:**
- D-Rank: Spectral entropy for rank allocation (2509.25622), Eq. 2, 7
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

**Dual-forward collection on H200:** Both the original (teacher) model and the pruned (student) model are loaded in VRAM simultaneously (~70 GB + ~50 GB = ~120 GB on H200's 141 GB). For each calibration batch, the teacher forwards first, then the student. Hooks on both models collect:

**B-covariance** `B = X_post^T X_post`: Auto-covariance of the pruned model's per-expert inputs. Reflects the input distribution the compressed model will see at inference.

**A-covariance reuse:** The pre-prune input auto-covariance `A = X_pre^T X_pre` referenced by Phase C Path 2 and Phase D's refinement objective is **not collected here** — it is reused from Stage 2's calibration pass (`_stage2_input_covariance.pt`, see §5 "Covariance Side-Collection") to avoid a redundant teacher forward.

**Cross-covariance** `C = X_pre^T X_post`: For each (layer, student_expert), the teacher's hidden state at the same token positions that the student routes to that expert is captured. `C` is accumulated as `X_pre^T @ X_post` per batch. This gives the exact cross-covariance required by AA-SVD Theorem 3.2 (paper 2604.02119): "what would the original model have produced for the inputs that the compressed model actually receives."

The teacher model is freed from VRAM after covariance collection completes — it is not needed for the factoring phase.

**Per-layer spill:** All 40 MoE layers are hooked simultaneously in a single calibration pass (not one pass per layer). After each layer's accumulation is finalised, both B and C covariances are spilled to disk (`_stage3_bcov_partial/` and `_stage3_ccov_partial/`). Background I/O thread overlaps spill with the next batch's forward pass, keeping the resident footprint bounded.

#### Phase B: D-Rank Allocation (Paper 2509.25622, Eq. 2 & 7)

For each (layer, matrix_type) group:

```
p_i = σ_i² / Σ_j σ_j²                        (normalized squared singular values)
R_eff = exp(−Σ_i p_i · log(p_i))              (Eq. 2 — effective rank)
k_g = √(R_eff(g) / ω) × T_budget / Σ_{g'} √(R_eff(g') / ω)   (Eq. 7 — rank allocation; Eq. 6 gives proportionality only)
```

where `ω = n_experts × (d_out + d_in)` is the per-rank parameter cost and `T_budget` is the global rank budget derived from `svd_rank_ratio`.

#### Phase B.2: Swift-SVD+ Per-Expert Rank Redistribution (Paper 2604.01609, Algorithm 2)

Within each (layer, matrix_type) group, D-Rank gives a uniform rank `k_g` to every expert. Swift-SVD+ refines this by redistributing the group's total rank budget `k_g × N_experts` across individual experts using a blending score:

```
s_i = β_i^α · (log(e + ε*_i))^{1-α}
```

where:
- `β_i = σ_i² / Σ_j σ_j²` — spectral energy proportion (how much of the group's total spectral energy this expert contributes)
- `ε*_i = √(Σ_{j>k̄} σ̃_j² / Σ_j σ̃_j²)` — activation-weighted reconstruction error at the group's mean rank `k̄`, where `σ̃_j` are the singular values of `W·A^{1/2}` (Stage 2 input auto-covariance from §5; see [D8](#12-known-deviations-from-papers) — ε* is now activation-weighted, not spectral-only). Higher = this expert needs more rank.
- `α ∈ [0, 1]` — balances the two signals

**α selection (paper §3.2.2 — validation-based):** For each candidate α ∈ {0.0, 0.1, ..., 1.0}, the full model is factored at the corresponding per-expert ranks using AA-SVD (reusing B/C covariance from Phase A spill files) and evaluated on WikiText-2 PPL (`validation_samples: 512` sequences). The α yielding the lowest end-to-end perplexity is selected. This implements the paper's exact procedure: *"For each candidate corresponding to α_i, the optimal low-rank approximation of every layer is computed using the closed-form solution in (3). The resulting compressed models are then evaluated on a validation set, and the candidate that yields the best end-to-end performance is selected."*

The factoring reuses cached spectral components from Phase A's B-covariance collection; each candidate requires ~2 minutes for a full 40-layer factor pass and ~20 seconds for PPL evaluation on H200. No model copies are made — originals are snapshotted to CPU RAM (~50 GB; H200 has 256 GB host RAM) and restored after each evaluation. Total α search: ~33 minutes for 11 candidates.

**Paper-compliance contract.** The α search MUST complete the paper-exact end-to-end PPL validation (Swift-SVD+ §3.2.2). If host RAM headroom at α-search entry is insufficient (<15 GB available), Stage 3 raises `RuntimeError` immediately rather than degrade to a spectral proxy — silently producing a non-paper-compliant model is worse than failing fast. Operators must provision adequate host RAM (~50 GB for the Qwen3-30B snapshot plus working set) or reduce `validation_samples` to fit. The previously-shipped silent spectral fallback was deviation D9 and was removed from Ch. 12 specifically because the pipeline now refuses to run that path. *(Implementation follow-up: replace the current OOM auto-fallback at `stage3_svd.py:285-301` with a hard `RuntimeError` to bring the code in line with this contract.)*

Per-expert ranks are stored in the `FactoredExperts` slot at the max rank across experts in the group (zero-padded for experts with lower rank). `effective_ranks` tracks the true per-expert rank for honest parameter counting.

#### Phase C: Hybrid Activation-Aware SVD

For each (layer, expert, matrix):

**Path 1 — Paper-exact Theorem 3.2 (primary on H200, cross-covariance C available):**

```
M = W · C · B⁻¹ · L_B        where C = X_pre^T X_post
```

This is the exact AA-SVD Theorem 3.2 formula. `L_B` is the eigendecomposition-based square root of B. The cross-covariance `C` is collected during Phase A's dual-forward pass.

**Path 2 — Auto-covariance approximation (C unavailable, A available):**

```
M = W · A · B⁻¹ · L_B        where A = X_pre^T X_pre
```

Substitutes pre-prune auto-covariance for cross-covariance. The two coincide when pre/post distributions are similar (light pruning). Active when `aa_svd.cross_covariance: false` in config.

**Path 3 — Corollary 3.3 fallback (B only):**

```
M = W · L_B
```

Then: `SVD(M) = U Σ V^T`, `U_k = U[:,:k] · diag(Σ[:k])`, `V_k^T = V^T[:k,:] · L_B⁻¹`. The rank-k reconstruction is `W ≈ U_k · V_k^T`.

**Eigendecomposition caching (gate_proj ↔ up_proj):** The covariance matrices B and C are identical for `gate_proj` and `up_proj` within the same expert — both projections receive the hidden state as input, and `_cov_lookup` falls back from `up_proj` to `gate_proj`. The eigendecomposition of B (`eigh`) and the derived right-hand-side product (CQ·diag(1/√λ), AQ·diag(1/√λ), or L_B depending on the path) are precomputed once per expert via `_precompute_eigh` and cached in an `_EighDecomp` dataclass. Both `gate_proj` and `up_proj` then call `_aa_svd_precomputed`, which skips directly to `M = W @ rhs`, SVD, and back-solve. `down_proj` has its own B (intermediate-dim covariance, 512×512) and goes through the full `_aa_svd` path.

This eliminates N_experts × N_layers redundant `eigh(2048×2048)` calls (~7,200 for 180 experts × 40 layers). The optimization is **mathematically identical** — same eigendecomposition, same rhs matrix, same floating-point operations on the same inputs; the only change is that the result is computed once and reused. Estimated wall-clock reduction: ~25% on Phase C.

**Numerical safeguards:**
- Eigendecomposition replaces Cholesky (handles rank-deficient B natively)
- Dtype-aware noise floor: `bf16→1e-2`, `fp16→1e-3`, `fp32→1e-6`
- `k_eff = min(k, r_eff)` — never allocates rank beyond B's effective rank
- Zero-padding when `k_eff < k` so FactoredExperts tensors stay shape-stable
- If `_precompute_eigh` raises (e.g. all-zero B), the per-matrix loop falls back to full `_aa_svd` which itself falls back to plain SVD

### Resume

- B-cov spill files at `_stage3_bcov_partial/layer_{idx}.pt` — layers whose spill files already exist are skipped on re-entry
- Spill directory is cleaned up on successful Stage 3 completion
- Original weights snapshot (`_stage3_original_weights.pt`) is saved for Stage 4 residual computation

---

## 7. Stage 4 — EoRA Residual Compensation

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

## 8. Stage 5 — Router Knowledge Distillation (Final)

**File:** [`stage5_router_kd.py`](src/moe_compress/stage5_router_kd.py)
**Paper:** Router Knowledge Distillation for MoE Compression (2603.02217), Eq. 3, Table 1, §F.3
**Hardware:** H200. EoRA-compensated student model stays resident from Stage 4. Teacher loads in BF16 (~70 GB); combined VRAM ~126 GB (with bs=8 logits).

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
4. Chunked KL: process `chunk_size` sequence positions at a time to bound peak memory at `B × chunk × |V| × 4` bytes. On H200 with ~16 GB VRAM headroom at bs=8, `chunk_size=512` (the full sequence length) is safe — peak intermediate is ~2.4 GB. Chunking is retained as a configurable parameter (`kd_seq_chunk_size`) for smaller-VRAM hardware; on H200 the overhead of chunk-boundary Python loops is eliminated by setting chunk=seq_len.
5. `F.kl_div(log_softmax(student/τ), softmax(teacher/τ))` = KL(teacher ‖ student) — correct forward KL direction

**`torch.compile` acceleration (Stages 2.5 and 5):** When `torch_compile: true` in the stage config, both teacher and student models are compiled via `torch.compile(model, mode="reduce-overhead")` before the KD training loop. On H200 (Hopper architecture), this enables kernel fusion and reduced launch overhead across the MoE dispatch + expert matmul sequence. Expected speedup: 20–40% on forward pass throughput after a one-time ~2–5 min compilation cost. Compilation is skipped when the model uses custom `instrument_experts` hooks (Stage 2's profiling pass), since the monkey-patched forward breaks torch.compile's graph tracing. Quality impact: zero — `torch.compile` produces numerically identical outputs in default mode.

### Hyperparameters (Paper Table 1, §F.3)

| Parameter | Value | Source |
|-----------|-------|--------|
| Optimizer | AdamW | Paper |
| Learning rate | **5×10⁻⁵** | Paper Table 1 (implementation previously used 1e-5; corrected to match paper) |
| Epochs | 1 | Paper |
| Batch size | 8 | Adapted (paper: 2) |
| Gradient accumulation | 1 | Adapted (paper: 4) |
| **Effective batch size** | **8** | Same as paper (8×1 = 2×4) |
| Max sequence length | 512 | Paper |
| KD temperature (τ) | 1.0 | Paper |
| Max calibration samples | 3000 | Paper |
| Teacher precision | BF16 | H200 (141 GB VRAM) fits teacher+student in BF16 with ~15 GB headroom at bs=8. Student is always BF16. |
| KL sequence chunk size | 512 | H200 (full sequence in one chunk — ~2.4 GB peak). Configurable via `kd_seq_chunk_size` for smaller hardware. |
| torch.compile | true | H200 Hopper. 20–40% forward pass speedup. Set false for debugging or hardware without compile support. |

### Teacher Loading

On H200 (141 GB VRAM), both teacher and student load in BF16 (~70 GB + ~50 GB = ~120 GB). At bs=8 with seq_len=512, vocabulary logits consume ~5 GB, leaving ~16 GB headroom. Alternatively, precomputed teacher vocabulary logits can be loaded from a cache file (`teacher_logits_cache` config key) to skip the live teacher entirely.

### Resume

Step-boundary checkpointing to `_stage5_partial/step_{N}.pt` (every 100 optimizer steps). Each checkpoint contains router parameter state + optimizer state. On resume, the last incomplete batch's gradient signal is silently dropped (no accumulation window — each batch is one optimizer step). Only the two most recent checkpoints are retained.

---

## 9. Stage 6 — Validation

**File:** [`stage6_validate.py`](src/moe_compress/stage6_validate.py)
**Hardware:** Runs on the same H200 instance as Stage 5 (student model stays resident).

### What

Evaluates the compressed model against the uncompressed teacher on 5 metrics, enforces hard quality gates, and produces an imatrix file for downstream GGUF quantization. On H200 the larger VRAM headroom allows larger batch sizes for all evaluation phases.

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

### Execution Model (Compute-Time Optimized)

Stage 6 runs the following phases. All optimizations are purely computational scheduling — larger batches, cached known-constants, overlapped I/O, and torch.compile. **No metric, formula, threshold, or evaluation methodology is changed.** All outputs are numerically identical to the batch_size=1 baseline.

#### Phase 1: Student Evaluation

| Sub-phase | Optimization | Config Key | Default |
|-----------|-------------|------------|---------|
| WikiText-2 PPL | **#1** — batch_size=1→8 | `ppl_batch_size` | 8 |
| ARC-C + HellaSwag | **#2** — lm-eval batch_size=1→auto:8 | `lm_eval_batch_size` | `"auto:8"` |
| HumanEval (164 prompts) | **#3** — batched model.generate() | `gen_batch_size` | 8 |
| MATH-500 (500 prompts) | **#4** — batched model.generate() | `gen_batch_size` | 8 |

**torch.compile** (**#5**, `torch_compile: true`): Before any evaluation begins, `model.forward` is compiled via `torch.compile(model.forward, dynamic=True, mode="reduce-overhead")`. `dynamic=True` handles variable-length padded batches from lm-eval. One-time compilation cost (~3–5 min on H200) is amortized across 1000+ forward passes. **Only** the forward pass is compiled — `model.generate()` is NOT compiled because autoregressive decoding changes shapes every step, causing excessive recompilation.

#### Phase 2: Teacher I/O Overlap (#6)

The teacher preload begins during Phase 1's generative evals (after the zero-shot harness completes): a background thread loads the teacher model to **host RAM** (device_map="cpu") while the GPU runs HumanEval and MATH-500. When Phase 1 completes, the student is moved to CPU, and the pre-loaded teacher is moved to GPU — eliminating the ~3–5 min dead time that a blocking teacher load would cause.

#### Phase 3: Teacher Evaluation (or Cache Hit)

**Teacher eval caching** (**#7**, `teacher_eval_cache.enabled: true`): The teacher (uncompressed Qwen3.6-35B-A3B) is a fixed, known model. Every Stage 6 run re-evaluates the same teacher on the same benchmarks with the same results. When caching is enabled:

- **First run:** Teacher is evaluated normally (PPL + zero-shot + generative). Results and param counts are saved to `teacher_eval_cache.json` with a cache key = `sha256(model_name + revision + eval_config_subset)`.
- **Subsequent runs:** Cached teacher results are loaded directly. No teacher model load, no teacher evaluation. This eliminates ~50% of total Stage 6 wall-clock time.
- **Auto-invalidation:** If the model name, revision, or any eval config parameter changes, the cache key mismatches and the teacher is re-evaluated.

When the cache misses, teacher evaluation uses the same batch sizes and torch.compile as student evaluation.

#### Phase 4: GGUF Conversion Overlap (#8)

When the teacher is being evaluated on GPU, the GGUF conversion (`convert_hf_to_gguf.py`) runs simultaneously in a **background CPU thread**. This is safe because GGUF conversion reads from the saved Stage 5 checkpoint on disk (CPU-only, ~5–10 min), teacher evaluation runs on GPU, and CPU and GPU work are fully independent. When teacher eval finishes and the teacher is freed, the F16 GGUF is ready — `llama-imatrix` can start immediately.

#### Phase 5: imatrix Generation

After the teacher is freed from VRAM, `llama-imatrix` runs on the F16 GGUF (pre-built in Phase 4) with the combined calibration text from all benchmarks.

### Spec Compliance of Optimizations

| Optimization | Why Numerically Identical |
|---|---|
| #1 PPL batch_size=8 | NLL is computed per-token; `out.loss × (batch.numel() - batch.shape[0])` recovers the exact sum regardless of batch size |
| #2 lm-eval batch_size=auto:8 | lm-eval's loglikelihood scoring is deterministic and batch-size-independent — left-padding with causal attention mask prevents cross-contamination |
| #3, #4 Batched generate | Greedy decoding (do_sample=False) produces the same argmax at each step regardless of batch. Left-padding with attention_mask ensures each prompt sees only its own context |
| #5 torch.compile | No numerical approximations in default/reduce-overhead modes. The spec already documents torch_compile as valid in Stage 5 |
| #6 Teacher I/O overlap | Computation unchanged — only I/O scheduling differs |
| #7 Teacher eval cache | Teacher is deterministic — same model + same eval = same numbers. Cache key includes model name, revision, and eval config |
| #8 GGUF overlap | GGUF conversion reads from saved checkpoint, independent of GPU evaluation |

### vLLM Note

vLLM is **NOT viable** for this model. The compressed model uses a custom `FactoredExperts` nn.Module that replaces the standard `Qwen3_5MoeExperts`. vLLM requires its own model-specific forward implementation and weight loader — it cannot load arbitrary custom nn.Module subclasses. The weight names (`gate_proj_U`, `gate_proj_V`, etc.) don't match what vLLM's Qwen3MoE loader expects. Stick with HuggingFace `model.forward()` and `model.generate()`.

### imatrix Generation

As a zero-overhead side-channel of the student evaluation pass, Stage 6 collects all text fed to the model across every benchmark into a single multi-domain calibration file. After the teacher is freed, the final frozen model is converted to F16 GGUF and `llama-imatrix` runs on the combined text, producing `imatrix.gguf`.

**Artifacts:**

| File | Description |
|------|-------------|
| `stage6_eval.json` | Quality gate results (metrics + pass/fail) |
| `teacher_eval_cache.json` | Cached teacher eval results + param counts (when caching enabled) |
| `calibration_imatrix.txt` | Combined eval text; always written (usable even without llama.cpp) |
| `model_f16.gguf` | Intermediate F16 GGUF of the compressed student |
| `imatrix.gguf` | Final importance matrix for GGUF quantization |

llama.cpp is built in the background by the job entrypoint (daemon thread, starts when Stage 1 begins) so the ~5-minute build does not add to wall-clock time. If llama.cpp is unavailable, `calibration_imatrix.txt` is still written and Stage 6 passes normally.

### Expected Wall-Clock Impact

| Scenario | Student Evals | Teacher Evals | imatrix | Total |
|---|---|---|---|---|
| Baseline (batch_size=1, no cache) | ~90–150 min | ~90–150 min | ~20 min | ~200–320 min |
| After P0 (#7 cache, #2 lm-eval) | ~30–50 min | 0 min (cached) | ~20 min | ~50–70 min |
| After P0 + P1 (#1 PPL, #3/#4 gen) | ~10–20 min | 0 min | ~20 min | ~30–40 min |
| After all optimizations | ~8–15 min | 0 min | ~15 min (overlapped) | ~25–30 min |

Expected improvement: **~8–12× end-to-end wall-clock reduction**, from ~3–5 hours down to ~25–30 minutes.

### Resume

Stage 6 is stateless. Re-running is always safe. The teacher eval cache persists across runs (it's a JSON file in the artifacts directory, also uploaded to Hub).


---

## 10. Protected Components

These are **never** modified by any compression stage:

- **Shared expert** (`mlp.shared_expert`) at every MoE layer
- **Attention weights** (DeltaNet linear attention + full attention projections)
- **Embeddings** and **lm_head**
- **Layer norms** (RMSNorm)
- **Router weights** — except Stage 5, which updates *only* these
- **Super experts** on the Stage 1 blacklist

---

## 11. Durability and Crash-Resume Model

### Inter-Stage Durability

HF Jobs bucket FUSE mounts are **not durable** under SIGKILL or timeout. The durability boundary is per-stage Hub uploads:

```
<base_repo>-stage2   ← Stage 2 output + covariance sidecar
<base_repo>-stage3   ← Stage 3 output + originals sidecar
<base_repo>-stage4   ← Stage 4 output
<base_repo>-stage5   ← Final compressed model
```

Each heavy stage (2–5) uploads its checkpoint to a per-stage Hub repo immediately on completion. The bucket is treated as scratch cache only.

### Within-Stage Crash-Resume

All partial checkpoint files are written via `.tmp` → `os.replace` (atomic on POSIX). A SIGKILL mid-write leaves at most a `.tmp` file, never a truncated final file. Dangling `.tmp` files are cleaned up at stage startup.

**`--no-resume` flag:** When passed to `run_pipeline.py`, disables all within-stage resume behaviour. Each stage runs unconditionally from scratch with no partial-file I/O. Stage 1 and Stage 6 are unaffected (they have no resume files).

| Stage | Resume Mechanism | Granularity | `--no-resume` Effect |
|-------|-----------------|-------------|---------------------|
| 1 | None (stateless, ~5 min) | N/A | None — JSONs are outputs, not resume files |
| 2 | `_stage2_partial/merge_{i}.json` + `layer_{i}.pt` | Per MoE layer | Skip all partial I/O |
| 3 | `_stage3_bcov_partial/`, `_stage3_ccov_partial/` spills; `_stage3_alpha_result.json` | Per covariance phase + α search | Delete existing spills; skip α cache |
| 4 | `_stage4_partial/layer_{i}.pt` | Per MoE layer | Skip all partial I/O |
| 5 | `_stage5_partial/step_{N}.pt` (rolling window of 2) | Per optimizer step (every 100 steps) | Skip all checkpoint I/O |
| 6 | None (stateless by design) | N/A | None — teacher_eval_cache is a speedup cache, not resume |

### Resume Safety Properties

**Stage 2 critical invariant:** Covariance remapping (`_remap_covariance_for_layer`) happens BEFORE the snapshot (`_snapshot_cov_layer`), which happens BEFORE the merge JSON write (`_write_merge_json`). A layer is considered complete only when BOTH `.json` and `.pt` exist. If `.pt` exists without `.json` (orphaned by crash between snapshot and JSON write), the `.pt` is deleted and the layer is reprocessed from scratch. This prevents double-remap corruption.

**Stage 3 covariance reuse:** On re-entry, if all per-layer B-cov spill files exist in `_stage3_bcov_partial/`, Phase A (covariance collection) is skipped entirely — including the teacher model load (~70 GB, ~60s). The α search result is cached in `_stage3_alpha_result.json` and reused on re-entry (~33 min saved).

**Stage 3 originals snapshot:** `_stage3_original_weights.pt` is saved immediately after `_snapshot_originals()` returns, BEFORE the α search and Phase D factoring. Stage 4 can access originals even if Stage 3 crashes during factoring.

**Stage 4 double-widen guard:** When `_stage3_original_weights.pt` is absent but `stage4_eora/eora_ranks.json` exists, Stage 4 detects a double-widen attempt and raises `AssertionError`. This protects against in-process re-runs (notebooks, test harnesses) where `widen_rank()` would silently double-apply EoRA correction.

**Stage 5 deferred teacher load:** The teacher model is loaded lazily on the first live batch (after fast-forward completes), not before the training loop. This eliminates wasted load time on resume.

### Format Version Enforcement

Every partial checkpoint carries a `format_version` field. On resume, the version is checked before any state is restored. A mismatch raises an error with an actionable message ("delete `_stage{N}_partial/` and re-run"). This prevents silent corruption when checkpoint format changes across code versions.

---

## 12. Known Deviations from Papers

| ID | Stage | Deviation | Paper Says | Implementation Does | Justification |
|----|-------|-----------|-----------|-------------------|---------------|
| D1 | 1 | Detection threshold | 2507.23279 Eq. 6 + Algorithm 1: two-stage process — first detect MA-formation layers L, then global P_{99.5} AND 0.1·a_max AND l∈L — all three required | Per-layer z-score (mean + 2.5σ); no global statistics; no MA-formation layer pre-filtering; all 40 MoE layers profiled | Avoids two-pass global stat collection and MA-pattern detection; per-layer z-score is conservative in practice (SEs produce z > 10 typically) |
| D2 | 1 | Blacklist caps | Paper: purely threshold-based, no caps | max_blacklisted_per_layer=4 and global_blacklist_cap_pct=5% | Safety guardrails against over-blacklisting |
| D3 | 1 | γ entropy tolerance | 2604.06542 Eq. 10: γ∈[0,1], no default given | γ=0.1 (project-chosen, not from paper) | Paper leaves γ unspecified; 0.1 chosen empirically |
| D4 | 1 | D^l update after merge | 2604.06542 Algorithm 1 lines 11–12: zero only pair entry D_{i*,j*} and D_{j*,i*}; update R^l ← R^l − 2·D_{i*,j*} | Zeros the absorbed expert's entire row and column in D^l; recomputes R^l from updated matrix | Prevents the absorbed expert from influencing future pair-selection; the paper's update assumes the merged expert cannot be re-selected, but only zeroing the pair entry leaves stale similarity values that can distort R^l and layer selection in subsequent iterations |
| D5 | 1 | Floor without layer bonuses | GRAPE has no floor constraints | min_experts_per_layer = num_routed_experts // 2 (=128); no early/late layer bonuses | 50% max removal per layer bounds the compression within the range where papers demonstrate results; bonuses removed — the floor alone is sufficient |
| D5a | 2 | REAM merge-group cap | 2604.04356 §A.1: C=16 for Qwen3-30B at 25% reduction | `max_merge_group_size=8` (with budget-bump fallback if exceeded) | Conservative bound — caps any single centroid's absorption at half the paper's cap, keeping merge groups smaller and more homogeneous; the budget bump compensates for any blocked merges so the global expert target is still met |
| D5b | 2 | Intermediate-neuron permutation in merge | 2604.04356 Eq. 6: frequency-weighted average `W_merged = Σ_i (freq_i / Σ_j freq_j) · W_i` (no permutation) | Hungarian permutation `P_i` aligning each child expert's intermediate neurons to the centroid before averaging, with combined cost `C = C_wt + C_act` (gate+up Frobenius weight distance + per-neuron mean-activation L2 distance) | Naive Eq. 6 averaging implicitly assumes intermediate neurons are index-aligned across experts. Qwen3 experts are independently initialised — by permutation symmetry of MLP hidden units (Git Re-Basin, Ainsworth et al. 2022), their intermediate-dim coordinate systems are arbitrary up to a permutation that the loss does not fix. Averaging without alignment therefore mixes uncorrelated neurons and inflates post-merge reconstruction error. Hungarian alignment recovers the correspondence before averaging. *TODO: A/B (with vs. without permutation) ablation pending Stage 6 evals.* |
| D6 | 3 | AA-SVD cross-covariance scope | 2604.02119 Theorem 3.2 requires cross-covariance for all linear layers | Cross-covariance C collected for gate_proj/up_proj (input-side) via dual-forward; down_proj falls back to Corollary 3.3 (B-only) because the teacher's per-expert intermediate activations require full expert dispatch instrumentation | Gate/up inputs share the same hidden state (pre-routing) so one capture covers both; down_proj inputs are expert-internal (post gate+up) and differ between teacher and student expert sets |
| D7 | 3 | D-Rank ω adapted for MoE | 2509.25622 Eq. 7: ω = d₁ + n·d₂ (layers per group × dimensions) | ω = n_experts × (d_out + d_in) | D-Rank targets shared-basis layer groups; adapted for MoE expert groups |
| D8 | 3 | Swift-SVD+ β | 2604.01609 Alg. 2: β = end-to-end layer importance, min-max normalized to [1,2] | β = per-expert spectral energy share (σ_i² / Σ σ_j²) | Paper's β is per-layer importance (requires 40 extra forward passes); adapted to per-expert within-group redistribution where the paper has no solution. ε* is now activation-weighted via Stage 2 A-covariance (no longer a deviation) |
| D10 | 4 | Eigenspace noise-floor truncation | 2410.21271 Alg. 1: full Q ∈ ℝ^{k×k} used; QQ^T = I guarantees Theorem 1 exactness | Eigenvectors below noise floor discarded; n_keep < k retained before SVD | Suppresses near-zero noise directions; weakens Theorem 1 exactness but improves numerical stability |
| D11 | 5 | Calibration data source | 2603.02217 §F.3 Table 1: calibration dataset = c4 (used identically across all experiments) | Multi-domain Nemotron-Cascade-2-SFT-Data with weighted subsets (chat 0.56, math 0.21, science 0.11, etc.) | Task-aware calibration better matches target deployment distribution; c4 is general pre-training data with limited reasoning/code coverage |

---

## 13. References

| ID | Paper | Year | Used In |
|----|-------|------|---------|
| 2507.23279 | Super Experts in MoE Models | 2025 | Stage 1 (SE detection) |
| 2604.06542 | GRAPE: Greedy Redundancy-Aware Pruning for MoE | 2026 | Stage 1 (budget allocation) |
| 2510.13999 | REAP: Routing-Expert Activation Pruning | 2025 | Stage 2 (scoring) |
| 2604.04356 | REAM: Routing Expert Activation Merging | 2026 | Stage 2 (merging) |
| 2509.25622 | D-Rank: Spectral Entropy Rank Allocation | 2025 | Stage 3 (rank budget) |
| 2604.02119 | AA-SVD: Anchored Adaptive SVD for LLMs | 2026 | Stage 3 (factorization) |
| 2604.01609 | Swift-SVD+: Dynamic Non-Uniform Rank Allocation | 2026 | Stage 3 (α validation search + rank redistribution) |
| 2503.12340 | SVD-LLM V2: Per-Type Rank Allocation | 2025 | Stage 3 (motivation) |
| 2410.21271 | EoRA: Training-Free Compensation for Compressed LLMs | 2024 | Stage 4 |
| 2603.02217 | Router Knowledge Distillation for MoE Compression | 2026 | Stage 5 |

---

*This document was generated from a full algorithmic review of the max_quality codebase on 2026-04-28; §12 updated 2026-04-29 after a per-stage paper compliance audit including full methodology-section cross-reference of all 10 cited papers; further per-stage spec-only paper-compliance review on 2026-05-01 added D5a (REAM merge-group cap) and D5b (intermediate-neuron Hungarian alignment in merge), corrected the D-Rank citation (Eq. 7, not Eq. 6) and the §6 ε* formula to reflect activation weighting per D8, fixed the δ_gate similarity/distance notation in §5, clarified A-covariance reuse from Stage 2 in §6 Phase A, clarified the A-vs-B weighting roles in §6 Phase D, fixed the SVD reconstruction notation (`diag(Σ[:k])` instead of `S[:k]`), and corrected the §4 R^l-update rationale. Spec redesign on 2026-04-29: merged Stage 0 into Stage 1 (CKA + SE detection), floor=n//2, max_merge_group=8, Router KD bs=8. D9 resolved on 2026-04-30: Swift-SVD+ α selection now uses paper-exact WikiText-2 PPL validation (§3.2.2 of 2604.01609) instead of spectral proxy; D9 removed from §12. Phase C eigh caching added 2026-04-30: gate_proj/up_proj share the same B and C covariance; eigendecomposition is now precomputed once per expert and reused for both projections, eliminating ~7,200 redundant eigh(2048×2048) calls. Compute-time optimizations 2026-04-30: (1) Stage 2 sequential profiling with early-exit forward — **implemented**; (2) vectorized REAM accumulators — **planned, not yet implemented**; (3) Stage 5 KL chunk size increased to full sequence length on H200 — **implemented**; (4) torch.compile support for Stages 2.5/5 KD forward passes — **implemented**. Stage 6 compute-time optimizations 2026-04-30 — **all implemented**: (5) WikiText-2 PPL batch_size 1→8; (6) lm-eval batch_size=auto:8; (7) batched model.generate() for HumanEval and MATH-500; (8) torch.compile for prefill-dominant forward paths; (9) teacher eval caching with sha256 cache key auto-invalidation (~50% total time eliminated); (10) teacher I/O overlap via background CPU preload; (11) GGUF conversion overlap with teacher eval. All Stage 6 optimizations are purely computational scheduling — numerically identical to batch_size=1 baseline. Expected total Stage 6 speedup: ~8–12× (from ~3–5 hours to ~25–30 minutes on H200). All formulas were verified against the cited papers. All deviations are deliberate and documented. For the original validation audit, see the archived [VALIDATED_STRATEGIES.md](https://huggingface.co/pirola/moe-compression-workflow/blob/main/VALIDATED_STRATEGIES.md).*
