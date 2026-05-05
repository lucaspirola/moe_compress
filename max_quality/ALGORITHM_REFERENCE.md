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

**Hardware:** Full pipeline runs on a single H200 (141 GB VRAM). Stages 2.5, 3, 4, and 5 keep the student model resident between stages — no inter-stage reload. Stage 5 runs twice: once after Stage 2 (as Stage 2.5) and once after Stage 4 (final). Stage 1 requires two sequential forward passes over the calibration set (~5 min combined on H200); all subsequent GRAPE computation is weight-space only.

> **Performance estimates:** All timing and speedup figures in this document (e.g. "~5 min Stage 1", "8–12× speedup for Stage 6", "~50% saving via teacher cache") are projected estimates based on algorithmic analysis and H200 hardware specifications. They have not been measured on the actual pipeline run. Actual H200 benchmarks are pending.

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
| 2 | +0 | 4000 | 2048 |
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
**Hardware:** H200. Original BF16 model (~70 GB) leaves 71 GB VRAM headroom. Two sequential forward passes over 256 calibration samples (Algorithm 1 Stage 1 then Stage 2, ~5 min combined), then weight-space GRAPE computation on GPU.

### What

A single unified stage that (a) identifies super experts that must never be compressed, and (b) computes non-uniform per-layer expert budgets using activation-aware CKA similarity. SE detection uses two sequential forward passes over the calibration data (Algorithm 1 Stage 1 builds L, Stage 2 builds A); the CKA representations for GRAPE are collected during the Stage 2 pass (Phase B).

### Why

Super experts carry outsized influence on model output despite being activated at normal frequency. Pruning them causes catastrophic quality collapse (e.g., −21.7% relative average accuracy drop across 9 benchmarks (Table 3, non-thinking mode) on Qwen3-30B-A3B). They must be detected before any budget allocation.

Uniform pruning wastes budget — some layers have highly redundant experts (high pairwise CKA similarity) while others are diverse. GRAPE's +2.45% peak on Mixtral-8x22B at the 4-expert setting (paper Table 2) demonstrates the value of non-uniform allocation. Using CKA (rather than weight-space cosine) for the similarity metric gives GRAPE activation-aware redundancy estimates, producing better budgets for Stage 2.

### How

#### Phase A: MA-Formation Layer Detection (Paper 2507.23279, Algorithm 1 (Appendix L) Stage 1)

Before scoring individual experts, Stage 1 constructs the set `L` of MA-formation layers. This is a dedicated pre-pass over the calibration data that scans each decoder layer's hidden state for the presence of a massive activation (MA) pattern.

**Algorithm (Algorithm 1, lines 3–12 — condensed rendering):**

```
L ← ∅
for each batch x ∈ D:
    for each decoder layer l:
        Compute hidden activations H_l(x)
        if MA pattern detected in H_l(x):
            L ← L ∪ {l}
        end if
    end for
end for
```

**MA pattern detection:** A layer `l` is added to `L` if its post-MoE hidden state `H_l(x)` contains at least one element whose magnitude is orders of magnitude larger than the bulk — i.e., a massive activation (Sun et al., 2024). The paper specifies no formula for the MA pattern detector; any detector that reliably identifies the presence of massive activations in the hidden state is compliant. One implementation-defined approach is to check whether `max |H_l(x)|` exceeds a large multiple of a robust high-quantile of `|H_l(x)|` for that batch, indicating a clear outlier channel; the specific multiplier is an implementation choice not from the paper.

**Why L matters:** The paper documents that some experts also produce extreme down_proj output magnitudes outside the MA-formation layers — these are called "outlier experts" (Table 7: L1E8, L47E48, L47E100 for Qwen3-30B-A3B; see Appendix C). Note: Table 6 of the paper lists the first outlier expert as "Layer 47 Expert 8" for this model, while Table 7 lists it as "Layer 1 Expert 8" — these two tables are internally inconsistent; this spec follows Table 7 (L1E8). These outlier experts do not contribute to MA formation and are not SEs. Not all outlier experts are excluded by the L-filter: L1E8 sits in Layer 1, which is an MA-formation layer (l ∈ L); Table 7 lists it as an outlier expert that is not classified as an SE, implying it fails the magnitude thresholds rather than being excluded by the L-filter. L47E48 and L47E100 sit outside L and are excluded by the L-filter. The l ∈ L constraint ensures that late-layer outlier experts outside L could not be blacklisted even if their magnitudes were large enough to satisfy the P99.5 and 0.1·a_max thresholds. Appendix C establishes that outlier experts lack the mechanistic significance of SEs but does not assert they would or would not pass the numerical thresholds.

**Properties of L:** MA formation in MoE models typically begins in the first 1–3 decoder layers and then stabilises — Mixtral exhibits this in a single layer, Qwen3-30B-A3B in three consecutive early layers. The MA pattern, once established, propagates stably across all subsequent layers via residual connections, so `L` is a small set of early layers (not the full layer stack). Note: this three-layer observation applies to Qwen3-30B-A3B (the paper's subject model); the pipeline's target model (Qwen3.6-35B-A3B) has a different architecture and its `L` will be determined empirically at runtime.

#### Phase B: Calibration Pass 2 — Expert Magnitude + CKA (256 samples)

All MoE layers are instrumented simultaneously. `run_calibration` runs once over all 256 samples (this is the second of the two passes; it is driven by Algorithm 1 (Appendix L) Stage 2, which covers expert magnitude collection for l ∈ L — the CKA collection for GRAPE is performed in the same pass as a pipeline efficiency choice but is not specified by Algorithm 1), collecting two things per (layer, expert):

1. **Max activation magnitude** `max_{x∈D} |h_{l,e}(x) · W^{l,e}_{down_proj}|` — for super expert detection. Here `h_{l,e}(x)` is the intermediate activation entering the down_proj of expert `e` in layer `l`, and the magnitude is measured at the down_proj **output** (post-weight-multiplication), exactly as stated in Algorithm 1 line 19.
2. **Expert output representations** `f_e(x)` — for CKA pairwise similarity computation

The expert output representations are accumulated into per-layer representation matrices for CKA via reservoir sampling (max 256 tokens per expert).

#### Phase C: Super Expert Detection (Paper 2507.23279, Eq. 6 + Algorithm 1 Stage 2)

Using the MA-formation layer set `L` constructed in Phase A, Stage 1 computes the global set `A = {a_{l,e}}` of max down_proj output magnitudes restricted to layers in `L`, then applies the three-way AND criterion from Eq. 6. Emit `stage1_blacklist.json`.

**Algorithm (Algorithm 1, lines 13–32):**

```
A ← ∅
for each batch x ∈ D:
    for each layer l ∈ L:
        for each expert e in layer l:
            a_{l,e} ← running max of |h_{l,e}(x) · W^{l,e}_{down_proj}| across batches
            A ← A ∪ {a_{l,e}}

# Note: a_{l,e} is maintained as a running maximum accumulated across batches,
# consistent with how Algorithm 1 is actually implemented.

P99.5 ← Percentile_99.5(A)     # global, across all (l,e) with l ∈ L
a_max ← max(A)                  # global maximum

S ← ∅
for each (l, e) with a_{l,e} ∈ A:
    if a_{l,e} > P99.5  AND  a_{l,e} > 0.1 · a_max:
        S ← S ∪ {(l, e)}
return S  # emitted as stage1_blacklist.json
```

**Three-way AND criterion (Eq. 6):**

```
blacklisted(l, e) = a_{l,e} > P99.5(A)  AND  a_{l,e} > 0.1 · a_max  AND  l ∈ L
```

where:
- `A = {a_{l,e}}` is the set of all max down_proj output magnitudes across all experts in all MA-formation layers
- `P99.5(A) = Percentile_99.5(A)` is the 99.5th percentile of this global set
- `a_max = max(A)` is the global maximum across all (l, e) in L
- `L` is the MA-formation layer set from Phase A

> **Note on §3.2.1 vs Algorithm 1:** The paper's §3.2.1 prose defines A as covering all such values across the entire model, while Algorithm 1 (Appendix L, lines 15–23, Stage 2) contains the inner loop restricted to l ∈ L (line 16). The spec follows Algorithm 1 as the more precise procedural definition — A is computed only over MA-formation layers.

All three conditions are required simultaneously. The l ∈ L constraint is enforced implicitly by restricting A to MA-formation layers — only experts in those layers are candidates.

**Empirical scale:** SEs account for fewer than 0.5% of all experts across the MoE models studied in the paper (Table 1: 0.05% for Qwen3-30B-A3B, 0.06% for DeepSeek-R1, 0.11% for DeepSeek-V2-Lite-Chat, 0.39% for Mixtral-8x7B-Instruct-v0.1). The paper sets no hard cap on SE count — the three-way AND criterion is purely threshold-based and the extremely tight thresholds (P99.5 AND 0.1·a_max) are self-limiting by design.

**Calibration check:** For models where the paper has published the canonical SE set, the Phase C output should be verified against it as a model-specific regression check. For example, for Qwen3-30B-A3B the paper reports (Table 2): Layer 1 Expert 68, Layer 2 Expert 92, Layer 3 Expert 82. This is provided as a verification reference for that specific model checkpoint — it is not a hardcoded list and other models will produce different SE sets. A compliant implementation should reproduce the paper's canonical set on the paper's target model before being applied to new models.

#### Phase D: CKA Similarity Matrices

For each MoE layer, compute the pairwise CKA (Centered Kernel Alignment) matrix `D^l ∈ ℝ^{N×N}` from the collected expert output representations. CKA measures functional similarity between experts based on their response patterns to actual inputs — two experts that produce similar outputs on the calibration data have high CKA, regardless of weight-space similarity.

Paper §3.2 explicitly allows "CKA, MSE, or other similarity measures" for D^l. CKA is the metric used by Zhang et al. (2025), cited in GRAPE §3.2 as the reference for intra-layer redundancy assessment.

With 256 samples × 2048 tokens ≈ 524K total token activations across the layer (each expert sees only its top-k/N routed fraction; for top-8 over 256 experts that is ≈ 16K per expert before sampling), reservoir-sampled to 256 per expert for CKA so the kernel matrices are well-conditioned for 256-expert layers.

#### Phase E: GRAPE Algorithm 1 (entropy-aware greedy merge with restart)

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

Reduces the number of routed experts per layer from 256 to ~180–200 by merging similar experts (not deleting — merged experts' knowledge is preserved via frequency-weighted averaging, with intermediate-neuron permutation alignment so that the averaged neurons correspond w.r.t. the centroid expert (REAM §4)). Simultaneously collects input covariance matrices (A) consumed by Stages 3 and 4.

### Why

Expert merging preserves more knowledge than deletion. REAM's pseudo-pruning (scoring + assignment + merge) was shown to retain ~98.5% of the original model's quality (derived: 69.8/70.9 GEN average, Table 1) on Qwen3-30B-A3B at ~25% expert reduction (128→96 experts, Table 1), outperforming pure pruning methods.

### How

**Sequential profiling with early-exit (REAM paper §4, Fig. 1(b)):** The REAM paper (2604.04356) introduces *sequential merging* as a core contribution: after merging layer ℓ, activations must be recomputed through the merged layer before profiling layer ℓ+1, ensuring each layer's REAP scores and REAM cost matrices reflect the actual input distribution it will see at inference time (not stale pre-merge statistics). The paper's ablation (§5.4) measures ΔAVG = −1.0 when sequential merging is removed — a meaningful fraction of the quality budget.

**Implementation:** For each layer L (processed in order 0→39), the profiling forward pass runs from the input embedding through layers 0…L, collecting REAP/REAM/covariance data from layer L's hooks. Layers L+1…39 are **not executed** — their computation is pure waste because all metrics collected for layer L (REAP scores, δ_gate, δ̃_expert, input covariance) depend only on the hidden states that *arrive at* layer L, not on what happens after it. An **early-exit forward hook** registered on the decoder layer immediately after layer L raises a sentinel exception that aborts the forward pass cleanly. The profiling runs under `torch.no_grad()`, so no autograd graph is corrupted.

This gives a ~2× wall-clock speedup over the naïve approach (running all 40 layers for each of the 40 profiling passes): the total layer-forward count drops from 40×40=1600 to 1+2+3+…+40=820. The REAM paper's sequential merging semantics are preserved exactly — each layer is profiled on hidden states that reflect all prior merges.

**Vectorized accumulators (planned follow-up, zero quality impact):**

The REAM cost matrix computation involves two pairwise similarity metrics across all experts in a layer (up to 256 experts). A future optimization replaces Python dicts with dense tensors for O(1) vectorized operations:

- **Gate logit profiles** (`ReamCostAccumulator`): Instead of `dict[expert_id → dict[token_idx → float]]`, a pre-allocated `torch.Tensor(num_experts, total_calibration_tokens)` on CPU in float16 stores each expert's pre-softmax router logit for each calibration token. The full `[N_experts × N_experts]` δ_gate cosine-similarity matrix is computed in one `F.normalize` + `matmul` call (~milliseconds for 256×256) rather than O(N²) Python-level loops. **Memory note:** at the updated calibration size of 4000 × 2048 = 8.19M tokens with 256 experts, the logit tensor is 256 × 8.19M × 2 bytes ≈ 4.2 GB per layer in FP16 on host RAM. This is materially larger than the prior 1024-sequence budget (~1.1 GB). The H200's host RAM (512 GB) comfortably accommodates this; the tensor is allocated and freed per layer, not held across layers simultaneously.

- **Gated-output pairwise similarity** (`finalize_batch`): Per-batch pairwise cosine similarity of gated expert outputs is computed via a single batched `F.cosine_similarity` over the jointly-active token intersection per expert pair, accumulated incrementally as before but with vectorized inner loops.

These optimizations are purely implementation-level data-structure changes. The mathematical computation is identical — same cosine similarities, same REAP scores, same cost matrix entries. Estimated wall-clock reduction on the cost-matrix phase: 10–100× (from minutes of Python iteration to seconds of tensor ops). Not yet implemented — the current accumulators use Python dicts (functionally correct, slower). The early-exit optimization provides the dominant ~2× speedup; vectorized accumulators are additive.

**Per-layer merge execution (sequential — must see prior merges):**

#### Step 1: REAP Scoring (Paper 2510.13999, Eq. 9)

> **Note on routing weight notation:** REAP (2510.13999) uses `g_j(x)` for the post-softmax routing weight, masked to zero for non-top-k experts. REAM (2604.04356) uses `σ(x)_j` for the full unmasked softmax (always positive). This spec uses `π(x)_j` as shorthand for the hard top-k masked version (not REAM's notation). In §5 below, `g_j(x)` follows REAP notation (masked, zero for non-active); the REAM Eq. 8 formula written with σ is interpreted using π(x)_j (hard top-k masked) rather than σ(x)_j; see the implementation note in Step 2 (δ̃_expert).

For each expert `j`, compute importance as the conditional average of gate-weighted output norm over active tokens:

```
S_j = (1/|X_j|) × Σ_{x ∈ X_j} g_j(x) · ‖f_j(x)‖₂
```

where `X_j = {x | j ∈ TopK(σ(x))}`, `g_j(x)` is the post-softmax routing weight, and `f_j(x)` is the expert output vector.

#### Step 2: REAM Cost Matrix (Paper 2604.04356, Eq. 5, 7, 8)

**Activation-space similarities** (NOT weight-space):

- **δ_gate(i,j)** (Eq. 5): Cosine similarity between **pre-softmax** router logit profile vectors. Each expert's profile is a vector indexed by global token position, containing the pre-softmax logit for that token (captured via `capture_router_outputs` pre-forward hook on the router module, which recomputes `F.linear(hidden, router.weight)`). Pre-softmax logits are used rather than post-softmax probabilities.

- **δ̃_expert(i,j)** (Eq. 8): Per-token cosine similarity of gated expert outputs `σ(x)_e · E_e(x)`, averaged over the full calibration set X. REAM Eq. 8 is written with the full unmasked softmax σ (strictly positive for all experts). The implementation uses the hard top-k masked version π(x), treating non-active experts as exact zeros. This is a deliberate implementation choice; the implementation must guard against undefined cosine similarity for zero-vector pairs (e.g., treating `sim(0, v) = 0`). The denominator is |X| not the jointly-active count — matching paper Eq. 8. Accumulated incrementally per batch via `finalize_batch` in `activation_hooks.py`.

- **δ_REAM(i,j) = δ_gate(i,j) + δ̃_expert(i,j)** (Eq. 7): Unweighted sum (paper uses equal weight 1.0).

#### Step 3: Greedy Pseudo-Pruning Assignment (Paper §4)

Top-N'_l experts by REAP score become **centroids** (protected from removal). Non-centroids are assigned to centroids via **greedy pseudo-pruning**:

1. Iterate centroids in descending saliency order
2. For each centroid, absorb the most similar (highest δ_REAM) unassigned non-centroid, up to `max_merge_group_size` non-centroids per centroid
3. Iterate all centroids once; centroids that receive no non-centroid assignments form singleton groups and pass through unchanged; non-centroids that are not assigned to any centroid are removed from the gate weight matrix (REAM §4)

#### Step 4: Frequency-Weighted Merge (Paper Eq. 6)

```
W_merged = Σ_i (freq_i / Σ_j freq_j) × P_i(W_i)
```

where the denominator `Σ_j freq_j` sums over merge group members only (not all N experts). `P_i` denotes the neuron permutation alignment as described in the paper's surrounding text (Hungarian algorithm on combined cost matrix `C = C_wt + C_act`) that aligns each child expert's intermediate neurons to the centroid before averaging; it is not an explicit formula component in the paper. `C_wt` is the gate+up Frobenius weight distance (implementation choice: gate_proj and up_proj; paper does not specify); `C_act` is the per-neuron mean activation L2 distance, where activation vectors H̄ are normalized before computing the distance (normalization method unspecified in the paper). `freq_i` is the count of calibration tokens for which expert i is in the top-k active set, equivalent to `S_i^freq × |X|` in the paper's notation (REAM Eq. 2).

#### Step 5: Router Resize

Remove merged experts' rows from `gate.weight`. Update `num_experts` on the MoE block.

### Covariance Side-Collection

During the profiling forward pass, two covariance matrices are accumulated per (layer, expert):
- **A_gate_up** (`gate_proj`): Input covariance for gate_proj and up_proj (shared tensor)
- **A_down** (`down_proj`): Input covariance for down_proj (intermediate activations)

Stored in `_stage2_input_covariance.pt` (fp32 storage; Swift-SVD paper 2604.01609 certifies FP32 for covariance accumulation; FP32 also avoids numerical degradation in eigendecomposition). On H200 with `batch_size=6`, the covariance accumulates signal across all 4000 calibration samples, providing well-conditioned A matrices for Stage 3.

### Budget Bump Loop

Two safety gates can raise the effective target if merge quality is poor:
- **`max_merge_group_size=8`**: If any group exceeds this, bump target. The REAM paper uses C=16 for Qwen3-30B at 25% reduction (128→96 experts) and C=32 for 50% reduction (128→64 experts) — suggesting C scales with compression depth rather than pool size; C=8 is deliberately conservative (see D5a). The budget-bump fallback ensures the global expert target is still met.
- **`ream_cost_sigma_threshold=1.5`**: If mean cost exceeds `running_mean × (1 + 1.5)`, bump target (inactive for first 4 layers)

### Resume

Per-layer atomic checkpointing to `_stage2_partial/` (see §11 for the `.tmp + os.replace` idiom and `.pt`-before-`.json` ordering invariant):
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
- D-Rank: Spectral entropy for rank allocation (2509.25622), Eq. 1, 2, 7
- AA-SVD: Anchored Adaptive SVD (2604.02119), Theorem 3.2, Corollary 3.3
- SVD-LLM V2: Heterogeneous rank allocation (2503.12340)
- Swift-SVD: Dynamic rank allocation (2604.01609), Algorithm 2
**Hardware:** H200. Pruned student model stays resident from Stage 2.5. Original BF16 model also loaded for cross-covariance dual-forward (~120 GB VRAM total).

### What

Replaces each surviving expert's 3 dense matrices (`gate_proj`, `up_proj`, `down_proj`) with rank-k factors `W ≈ U · V`, where `U ∈ ℝ^{d_out × k}` and `V ∈ ℝ^{k × d_in}`. Rank `k` varies across (layer, matrix_type) groups — high-entropy matrices get more rank, low-entropy matrices get less.

### Why

SVD factorization reduces parameters from `d_out × d_in` to `k × (d_out + d_in)` per expert. With activation-aware rank allocation and weighting, the factorization concentrates error into directions the model rarely uses, preserving quality far better than plain truncated SVD.

### How

#### Phase A: Covariance Collection (B and cross-covariance C)

**Dual-forward collection on H200:** Both the original (teacher) model and the pruned (student) model are loaded in VRAM simultaneously (~70 GB + ~50 GB = ~120 GB on H200's 141 GB). For each calibration batch, the teacher forwards first, then the student. Hooks on both models collect:

**B-covariance** `B = X_post^T X_post`: Auto-covariance of the pruned model's per-expert inputs. Reflects the input distribution the compressed model will see at inference.

**A-covariance reuse:** The pre-prune input auto-covariance `A = X_pre^T X_pre` referenced by Phase C Path 2 and by Phase B.2's activation-weighted `ε*` (per D8) is **not collected here** — it is reused from Stage 2's calibration pass (`_stage2_input_covariance.pt`, see §5 "Covariance Side-Collection") to avoid a redundant teacher forward.

**Cross-covariance** `C = X_pre^T X_post`: For each (layer, student_expert), the teacher's hidden state at the same token positions that the student routes to that expert is captured. `C` is accumulated as `X_pre^T @ X_post` per batch. This gives the exact cross-covariance required by AA-SVD Theorem 3.2 (paper 2604.02119): "what would the original model have produced for the inputs that the compressed model actually receives."

The teacher model is freed from VRAM after covariance collection completes — it is not needed for the factoring phase.

**Per-layer spill:** All 40 MoE layers are hooked simultaneously in a single calibration pass (not one pass per layer). After each layer's accumulation is finalised, both B and C covariances are spilled to disk (`_stage3_bcov_partial/` and `_stage3_ccov_partial/`). Background I/O thread overlaps spill with the next batch's forward pass, keeping the resident footprint bounded.

#### Phase B: D-Rank Allocation (Paper 2509.25622, Eq. 1, 2 & 7)

For each (layer, matrix_type) group, effective rank is computed from the **whitened** weight matrix, not raw W. This makes rank allocation input-distribution-aware.

**Step B.1 — FP64 Cholesky whitening (Paper 2509.25622, §3.1, inputs to Eq. 1–2):**

For each group `g` (identified by (layer, matrix_type)), retrieve the pre-prune input auto-covariance A_g = X_pre^T X_pre from Stage 2 (A_gate_up / A_down from _stage2_input_covariance.pt — this is the A-covariance, not the B-covariance defined in Phase A):
- For `gate_proj` and `up_proj` groups: use `A_gate_up` from `_stage2_input_covariance.pt` (hidden-state input covariance, dimension `hidden_size × hidden_size`)
- For `down_proj` groups: use `A_down` from `_stage2_input_covariance.pt` (intermediate-activation input covariance, dimension `moe_intermediate_size × moe_intermediate_size`)

Compute the Cholesky factor in **FP64**:

```
X_g^T X_g = S_g S_g^T,   S_g = chol(X_g^T X_g)    [computed in FP64 per paper; S_g is lower-triangular]
```

Then compute the whitened weight matrix `S_g · W_g^T` (where `W_g^T` transposes from PyTorch's stored `[d_out × d_in]` to `[d_in × d_out]`, giving a `[d_in × d_out]` result; applied per-expert). The singular values used for effective rank are those of `S_g · W_g^T`, not of raw `W_g`. The per-group covariance `X_g^T X_g` is the average over all experts in the group (shared input distribution for the same matrix type within a layer).

> **Note:** Using the group-average covariance for whitening (rather than a per-expert covariance) is a deliberate efficiency choice — collecting per-expert input covariances would require per-expert dispatch instrumentation equivalent to the cross-covariance infrastructure. The group average is a valid approximation when experts within a group see similar input distributions (which the REAM merging of similar experts enforces).

**Step B.2 — Effective rank from whitened SVD:**

```
σ_i = singular values of (S_g · W_g^T)           (whitened singular values; W_g^T transposes from stored [d_out×d_in] to [d_in×d_out])
p_i = σ_i² / Σ_j σ_j²                            (Eq. 1 — normalized squared singular values)
R_eff(g) = exp(−Σ_i p_i · log(p_i))              (Eq. 2 — effective rank from whitened spectrum)
```

**Step B.3 — Rank allocation:**

```
k_g = √(R_eff(g) / ω) × T_budget / Σ_{g'} √(R_eff(g') · ω)   (Eq. 7 — rank allocation; Eq. 6 gives proportionality only)
```

where `ω = n_experts × (d_out + d_in)` is the per-rank parameter cost and `T_budget` is the global rank budget derived from `svd_rank_ratio`.

**Per-projection bias** (budget-neutral): `gate_proj=1.33`, `up_proj=0.67`, `down_proj=1.0`. The ratio `gate:up:down = 4:2:3` is adopted from jangq's MLP-asymmetry analysis for SwiGLU quantization (`397B-MLP-ASYMMETRY.md` §3.1), translated from bit space to rank space. Rationale: gate errors are amplified multiplicatively through `SiLU(gate)·up`; down errors propagate to the residual stream of every downstream layer; up errors are bounded and linear. The multipliers sum to 3.0 across the three projection types, preserving the global rank budget on the type-average. See [D7a](#12-known-deviations-from-papers). The mean rank `k̄` used in `ε*` is the bias-adjusted group rank (i.e., after applying the gate/up/down multipliers from Step B.3 — deviation from paper: the paper defines k̄ as the plain uniform rank k̄ = (m×n)/(m+n) × ρ; see D7a).

#### Phase B.2: Swift-SVD Per-Expert Rank Redistribution (Paper 2604.01609, Algorithm 2)

Within each (layer, matrix_type) group, D-Rank gives a uniform rank `k_g` to every expert. Swift-SVD refines this by redistributing the group's total rank budget `k_g × N_experts` across individual experts using a blending score:

```
s_i = β_i^α · (log(e + ε*_i))^{1-α}
```

where e ≈ 2.718 (Euler's number, per paper notation 2604.01609 Eq. 12)

- `β_i = σ_i² / Σ_j σ_j²` — spectral energy proportion (how much of the group's total spectral energy this expert contributes; see [D8](#12-known-deviations-from-papers))
- `ε*_i = √(Σ_{j>k̄} σ̃_j² / Σ_j σ̃_j²)` — activation-weighted reconstruction error at the group's mean rank `k̄`, where `σ̃_j` are the singular values of `A^{1/2}·W` (Stage 2 input auto-covariance from §5; see [D8](#12-known-deviations-from-papers) — ε* is now activation-weighted, not spectral-only). Higher = this expert needs more rank. This equals `‖A^{1/2}·W − A^{1/2}·Ŵ(k̄)‖_F / ‖A^{1/2}·W‖_F` = `‖A^{1/2}·(W − Ŵ(k̄))‖_F / ‖A^{1/2}·W‖_F` (relative reconstruction error in the activation-weighted norm, where A^{1/2} left-multiplies W as in ‖XW − XW_k‖_F = ‖A^{1/2}(W − W_k)‖_F), i.e., the activation-weighted analogue of the paper's spectral ε*. (Deviation from paper: the paper's ε* is absolute truncation error; the spec normalizes to a relative ratio for cross-expert comparability — see §12 D-eps-star.)
- `α ∈ [0, 1]` — balances the two signals

**α selection (paper §3.2.2 — validation-based):** For each candidate α ∈ {0.0, 0.1, ..., 1.0}, the full model is factored at the corresponding per-expert ranks using the closed-form solution from Swift-SVD Eq. 3: W*_k = W V_k V_k^T, where V_k are the top-k right singular vectors of W^T A W (activation-weighted weight covariance, computed as W^T @ A_g @ W for each expert; A_g is the Stage 2 pre-prune input covariance from _stage2_input_covariance.pt — paper-exact per Theorem 3.1 / Eq. 3) and evaluated on WikiText-2 PPL (`validation_samples: 512` sequences). The α yielding the lowest end-to-end perplexity is selected. This implements the paper's exact procedure: *"For each candidate corresponding to α_i, the optimal low-rank approximation of every layer is computed using the closed-form solution in (3). The resulting compressed models are then evaluated on a validation set, and the candidate that yields the best end-to-end performance is selected."*

The factoring reuses cached spectral components from Phase A's B-covariance collection; each candidate requires ~2 minutes for a full 40-layer factor pass and ~20 seconds for PPL evaluation on H200. No model copies are made — originals are snapshotted to CPU RAM (~50 GB; H200 has 256 GB host RAM) and restored after each evaluation. Total α search: ~33 minutes for 11 candidates.

**Paper-compliance contract.** The α search MUST complete the paper-exact end-to-end PPL validation (Swift-SVD §3.2.2). If host RAM headroom at α-search entry is insufficient (<15 GB available), Stage 3 raises `RuntimeError` immediately rather than degrade to a spectral proxy — silently producing a non-paper-compliant model is worse than failing fast. Operators must provision adequate host RAM (~50 GB for the Qwen3-30B snapshot plus working set) or reduce `validation_samples` to fit. The previously-shipped silent spectral fallback was deviation D9 and was removed from Ch. 12 specifically because the pipeline now refuses to run that path.

> **Implementation follow-up:** Replace the current OOM auto-fallback at `stage3_svd.py:285-301` with a hard `RuntimeError` to bring the code in line with this contract.

**Minimal-rank floor (paper 2604.01609 Algorithm 2):** After rank redistribution, a floor is applied per expert:

```
k_i ← max(k_i, floor(k̄ · δ))    where δ = 0.5
```

The paper explicitly warns that δ = 0 is numerically unstable. This floor ensures no expert can receive rank 0 after redistribution.

Per-expert ranks are stored in the `FactoredExperts` slot at the max rank across experts in the group (zero-padded for experts with lower rank). `effective_ranks` tracks the true per-expert rank for honest parameter counting.

#### Phase C: Hybrid Activation-Aware SVD

For each (layer, expert, matrix):

**Path 1 — Anchored-adaptive objective, Theorem 3.2 (primary on H200, cross-covariance C available; see D-AASVD-objective):**

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

#### Phase C.5: Block-Level Iterative Refinement (Paper 2604.02119, Algorithm 2, §3.3)

After all linear sub-layers (attention projections and MLP gate/up/down projections for all experts) within a single decoder block have been individually factorized via Phase C (Paths 1/2/3), AA-SVD performs a **block-level joint refinement pass** that jointly optimizes the factorized weight factors and the block's normalization parameters to minimize the block's output error against the original model. This is the central contribution of AA-SVD over standard per-layer SVD.

**Block definition:** One transformer block `ℒ_i` = one decoder layer, comprising all its linear layers (attention projections + MLP gate/up/down projections for all routed experts in that layer), all non-linear operations, and all normalization layers (RMSNorm pre-attention and pre-MLP, post-attention residual). The refinement is applied sequentially, block by block, after each block's Phase C factorization — upstream blocks remain frozen while the current block is refined. For Qwen3's architecture, each block includes the sliding-window SDPA attention and the MoE MLP, both of whose RMSNorm pre-norms are updated during refinement.

**Objective (Section 3.3):**

```
ℓ_i = E_{X∼𝒟_i}[‖ℒ_i(X) − ℒ'_i(X')‖²]
```

where:
- `ℒ_i(X)` — the original (unfactorized) block's output on calibration activations `X` (the hidden states arriving at block `i` from the original model)
- `ℒ'_i(X')` — the compressed block's output on shifted calibration activations `X'` (the hidden states produced by the already-refined upstream compressed blocks)

In practice this is the mean over the calibration batch at each gradient step.

This anchors the compressed block to the original block's output while conditioning on the **actual shifted input** produced by compression of prior layers — precisely the anchored-adaptive formulation of Theorem 3.2 extended to the block level.

**Optimization procedure (Algorithm 2, line 9; Appendix B.2):**

Minimize `ℓ_i` jointly over:
1. All factorized weight factors `{U_j, V_j}` for every linear layer `j` in block `i`
2. Block-local parameters `θ_i` — the **RMSNorm scale parameters** (and biases, if any) within block `i`

Optimizer: **AdamW**, learning rate `1×10⁻⁴`, cosine learning rate schedule with linear warmup, **25 epochs** over the calibration data, batch size 32. All parameters in items 1 and 2 are updated simultaneously in each gradient step — this is a joint optimization, not an alternating coordinate-descent loop.

**Convergence:** Fixed epoch count of **25 epochs**. No delta-objective threshold is specified by the paper; training always runs for the full 25 epochs.

**Interaction with Paths 1/2/3:** Phase C factorization (Paths 1/2/3) provides the initialization for `{U_j, V_j}`. Phase C.5 refines these initializations via gradient descent; it does not re-invoke the Theorem 3.2 closed form. The B/C covariances computed in Phase A are used only for the Phase C initialization — Phase C.5 uses the calibration activations directly via forward passes through the (partially compressed) model.

**RMSNorm scope:** Only the RMSNorm layers **within** block `i` (pre-attention and pre-MLP norms of that specific decoder layer) have their scale parameters updated. Norms in all other blocks remain frozen. This covers exactly the `input_layernorm` (pre-attention) and `post_attention_layernorm` (pre-MLP) of each decoder layer — not the model-level `norm` or any embedding norms. See §10 for the updated protected-component policy.

**Sequential execution (Algorithm 2 lines 2–11):** After block `i`'s Phase C.5 completes, the refined compressed block `ℒ'_i` is used to produce `X'_{i+1}` (the input to block `i+1`) via a forward pass through `ℒ'_i`. This updated `X'` feeds into block `i+1`'s Phase C factorization and Phase C.5 refinement. Blocks are processed strictly in order 0 → (N_layers − 1).

### Resume

- B-cov spill files at `_stage3_bcov_partial/layer_{idx}.pt` — layers whose spill files already exist are skipped on re-entry
- Spill directory is cleaned up on successful Stage 3 completion
- Original weights snapshot (`_stage3_original_weights.pt`) is saved for Stage 4 residual computation

---

## 7. Stage 4 — EoRA Residual Compensation

**File:** [`stage4_eora.py`](src/moe_compress/stage4_eora.py)
**Paper:** EoRA: Training-Free Compensation for Compressed LLMs (2410.21271), Algorithm 1
**Hardware:** H200. One calibration forward pass to collect mean per-expert input activations X̃ = mean(X_expert). FactoredExperts model stays resident from Stage 3; `_stage3_original_weights.pt` remains in CPU RAM.

### What

For each factored expert matrix, computes the residual `ΔW = W_original − U·V` and adds a rank-r correction that concentrates on the **most important input directions** (as measured by the rank-1 outer product of the mean per-expert input activation X̃). The correction is appended to the existing factored representation by widening U and V along the rank dimension.

### Why

EoRA recovers quality lost to rank truncation in Stage 3. The paper reports +10.84pp ARC-C on LLaMA3-8B (in the paper's 3-bit quantization experiment — not applicable to our BF16 pipeline, cited for magnitude context only). The key innovation over naive SVD of the residual is the √Λ-weighted eigenspace projection, which concentrates the correction rank budget on directions the model actually uses.

### How — Paper Algorithm 1

For each (layer, expert, matrix):

1. **Residual:** `ΔW = W_orig − U_old · V_old` — shape `[d_out × d_in]`

2. **Compute mean activation and eigendecompose:** For each (layer, expert), collect `X̃_expert = mean_{tokens routed to expert}(X)` — the mean input activation vector, shape `[d_in]`. Form the rank-1 outer product `A = X̃ X̃^T`, shape `[d_in × d_in]`. Eigendecompose: `A = Q Λ Q^T`. Since A is rank-1, this gives exactly one non-zero eigenvalue λ₁ = ‖X̃‖² with eigenvector q₁ = X̃/‖X̃‖, so `n_keep = 1` in the non-degenerate case (the noise floor keeps only eigenvalues above a dtype-aware threshold). Note: the rank-1 structure means n_keep is typically 1, and the correction adapter is effectively a single-direction rank-1 update per matrix.

3. **√Λ-scaled projection:** `Q' = Q_keep · √Λ_keep` — shape `[d_in × n_keep]`. This is the **full** signal eigenspace, NOT truncated to `r`. The √Λ scaling importance-weights each direction by its activation variance.

4. **Full projection:** `ΔW' = ΔW · Q'` — shape `[d_out × n_keep]`

5. **Rank-r SVD:** `SVD(ΔW') → U', Σ', V'^T`. Take top `take_eff = min(r, min(d_out, n_keep))`.

6. **Correction factors:**
   - `U_corr = U'[:, :take_eff] · Σ'[:take_eff]` — shape `[d_out × take_eff]`
   - `V_corr = V'^T[:take_eff] · (√Λ_keep)⁻¹ · Q_keep^T` — shape `[take_eff × d_in]` (back-projected to original weight space)

7. **Widen:** `new_U = [U_old | U_corr]`, `new_V = [V_old; V_corr]` — algebraically equivalent to `Ŵ·x + B'·A·x`

### Budget

`compensation_budget_pct=3%` of Stage 3 savings per matrix, capped at `eigenspace_rank_cap=128` rank per expert (paper default).

### Correctness Notes

- The √Λ scaling is the **core** innovation of EoRA. Without it, the algorithm degenerates toward ZeroQuant-V2 (plain SVD on ΔW with no activation weighting) — Act-S is a separate method that uses per-channel L1-magnitude diagonal scaling, unrelated to eigenvector projection.
- Pre-truncating to `r` eigenvectors before SVD (the previous bug) eliminates the joint optimization that makes EoRA better than Act-S. The SVD must operate on the full `[d_out × n_keep]` projected error to optimally select the best `r` directions.
- The back-projection `V_corr = (V'^T · (√Λ)⁻¹) · Q^T` is critical — without the `(√Λ)⁻¹` term, the correction adapter operates in eigenspace rather than weight space, and the errors compound.

### Resume

Per-layer atomic checkpointing to `_stage4_partial/layer_{layer_idx}.pt` (format_version=1). Each checkpoint contains the full FactoredExperts U/V state, ranks, effective ranks, and parameter counts. On resume, completed layers are loaded directly; failed layers re-run from the Stage 3 output.

Stage 4 deletes `_stage3_original_weights.pt` on success (already durable on the per-stage Hub repo). `_stage2_input_covariance.pt` is not read by Stage 4 and is NOT deleted here — it is used by Stage 3 and cleaned up by that stage's own success handler.

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

These are **never** modified by any compression stage, with the exception noted below for Stage 3 block refinement:

- **Shared expert** (`mlp.shared_expert`) at every MoE layer
- **Attention weights** (DeltaNet linear attention + full attention projections)
- **Embeddings** and **lm_head**
- **Layer norms** (RMSNorm) — **except** during Stage 3 Phase C.5 block refinement, where the RMSNorm scale parameters (`θ_i`) within the block currently being refined are updated as required by AA-SVD Algorithm 2. Only the norms of the block under active refinement are updated; all other blocks' norms remain frozen throughout. After Stage 3 completes, no further norm modifications occur.
- **Router weights** — except Stage 5 (and Stage 2.5), which update *only* these
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

All partial checkpoint files are written via a durable atomic write sequence:
1. Write data to `<path>.tmp`
2. `fsync(<path>.tmp)` — flushes file data and metadata to storage
3. `fsync(parent_dir)` — ensures the `.tmp` file's directory entry is durable
4. `os.replace(<path>.tmp, <path>)` — atomic rename on POSIX

This sequence survives SIGKILL, training-framework timeout, kernel panic, and power loss. A crash at any point before step 4 completes leaves at most a `.tmp` file, never a truncated or partially-visible final file. Dangling `.tmp` files are cleaned up at stage startup.

> **Implementation follow-up:** The current code performs `.tmp` → `os.replace` without the two `fsync` calls. Replace the write helper at `checkpoint_utils.py` (or equivalent) with the above 4-step sequence to bring the implementation in line with this spec.

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

**Stage 3 originals snapshot:** `_stage3_original_weights.pt` is saved immediately after `_snapshot_originals()` returns, BEFORE the α search and Phase C factoring. Stage 4 can access originals even if Stage 3 crashes during factoring.

**Stage 4 double-widen guard:** When `_stage3_original_weights.pt` is absent but `stage4_eora/eora_ranks.json` exists, Stage 4 detects a double-widen attempt and raises `AssertionError`. This protects against in-process re-runs (notebooks, test harnesses) where `widen_rank()` would silently double-apply EoRA correction.

**Stage 5 deferred teacher load:** The teacher model is loaded lazily on the first live batch (after fast-forward completes), not before the training loop. This eliminates wasted load time on resume.

### Format Version Enforcement

Every partial checkpoint carries a `format_version` field. On resume, the version is checked before any state is restored. A mismatch raises an error with an actionable message ("delete `_stage{N}_partial/` and re-run"). This prevents silent corruption when checkpoint format changes across code versions.

---

## 12. Known Deviations from Papers

| ID | Stage | Deviation | Paper Says | Implementation Does | Justification |
|----|-------|-----------|-----------|-------------------|---------------|
| D3 | 1 | γ entropy tolerance | 2604.06542 Eq. 10: γ∈[0,1], no default given | γ=0.1 (project-chosen, not from paper) | Paper leaves γ unspecified; 0.1 chosen empirically |
| D4 | 1 | D^l update after merge | 2604.06542 Algorithm 1 lines 11–12: zero only pair entry D_{i*,j*} and D_{j*,i*}; update R^l ← R^l − 2·D_{i*,j*} | Zeros the absorbed expert's entire row and column in D^l; recomputes R^l from updated matrix | Prevents the absorbed expert from influencing future pair-selection; the paper's update assumes the merged expert cannot be re-selected, but only zeroing the pair entry leaves stale similarity values that can distort R^l and layer selection in subsequent iterations |
| D5 | 1 | Floor without layer bonuses | GRAPE has no floor constraints | min_experts_per_layer = num_routed_experts // 2 (=128); no early/late layer bonuses | 50% max removal per layer bounds the compression within the range where papers demonstrate results; bonuses removed — the floor alone is sufficient |
| D5a | 2 | REAM merge-group cap | 2604.04356 §A.1: C=16 for Qwen3-30B at 25% reduction | `max_merge_group_size=8` (with budget-bump fallback if exceeded) | Conservative bound — caps any single centroid's absorption at half the paper's cap, keeping merge groups smaller and more homogeneous; the budget bump compensates for any blocked merges so the global expert target is still met |
| D5b | 2 | Cost matrix choice for neuron permutation alignment in merge | 2604.04356 Eq. 6: frequency-weighted average with neuron permutation alignment (Ainsworth et al., 2023) w.r.t. the centroid expert; cost matrix C unspecified | Hungarian permutation `P_i` with cost matrix `C = C_wt + C_act` (gate+up Frobenius weight distance + per-neuron mean-activation L2 distance) | Paper prescribes permutation but leaves the cost form open. Spec uses `C_wt + C_act`: weight-space Frobenius distance captures structural similarity; activation-weighted neuron L2 distance captures functional importance. *TODO: Ablation of cost matrix choice (C_wt only vs. C_wt + C_act vs. activation-only) pending Stage 6 evals.* |
| D-protocol-blend | 2.5 | Protocol combination: REAM + Router KD in sequence | 2604.04356 (REAM): explicitly evaluates "without any fine-tuning after compression"; 2603.02217 (Router KD): designed as a standalone step, not as a post-REAM patch | Spec applies Router KD (Stage 2.5) immediately after the REAM merge | Router KD restores routing accuracy degraded by weight averaging; REAM's static evaluation does not cover post-merge routing drift. Combined protocol not ablated against REAM-static-only baseline: empirical_pending |
| D6 | 3 | AA-SVD cross-covariance scope | 2604.02119 Theorem 3.2 requires cross-covariance for all linear layers | Cross-covariance C collected for gate_proj/up_proj (input-side) via dual-forward; down_proj falls back to Corollary 3.3 (B-only) because the teacher's per-expert intermediate activations require full expert dispatch instrumentation | Gate/up inputs share the same hidden state (pre-routing) so one capture covers both; down_proj inputs are expert-internal (post gate+up) and differ between teacher and student expert sets |
| D-AASVD-objective | 3 | AA-SVD primary objective variant | 2604.02119 §4.3 Table 5 recommends input-aware (A=B=X, Corollary 3.3 with pre-prune covariance) + block refinement as primary recipe (PPL 6.89 at ρ=0.8 LLaMA-7B) | Spec uses anchored-adaptive (A=X_pre, B=X_post, Theorem 3.2) + block refinement (Path 1). Quality gap ~0.2 PPL at ρ=0.8 on LLaMA-7B; Qwen3-30B-A3B comparison empirical_pending. | Anchored-adaptive is the paper's central theoretical contribution and expected to outperform in high-compression regimes where upstream drift is larger; empirical validation on Qwen3-30B-A3B pending |
| D7 | 3 | D-Rank ω adapted for MoE | 2509.25622 Eq. 7: ω = d₁ + n·d₂ (layers per group × dimensions) | ω = n_experts × (d_out + d_in) | D-Rank targets shared-basis layer groups; adapted for MoE expert groups |
| D7a | 3 | Per-projection rank bias | 2509.25622: D-Rank Eq. 7 produces a single `k_g` per (layer, matrix_type) group; no per-projection-type multiplier | Group ranks from Eq. 7 are scaled by `gate=1.33, up=0.67, down=1.0` (sum=3.0, budget-neutral on type-average) before per-expert redistribution | Adapted from jangq's GGUF bit-allocation insight (`gate:up:down ≈ 4:2:3`, see `397B-MLP-ASYMMETRY.md` §3.1): SwiGLU forward couples gate errors multiplicatively via SiLU, while down errors propagate to the residual stream. The ratio translates the same physical asymmetry from bit space to rank space. *TODO: empirical re-tune from clean per-projection `recon_rel_err` once Stage 6 evals are available; current values inherited unchanged from a prior bf16-bug-tainted run and are theoretically- (not empirically-) grounded.* |
| D8 | 3 | Swift-SVD β | 2604.01609 Alg. 2: β = end-to-end layer importance, min-max normalized to [1,2] | β = per-expert spectral energy share (σ_i² / Σ σ_j²) in range (0,1] | Paper's β is per-layer end-to-end importance (requires 40 extra forward passes), min-max normalized to [1,2]; adapted to per-expert spectral energy share (σ_i²/Σσ_j²) in range (0,1]. The range difference changes blending behavior: paper's β∈[1,2] means β^α always amplifies; spec's β∈(0,1] can suppress low-energy experts. This is intentional — per-expert spectral energy within a group is the natural adaptation of per-layer importance for MoE expert redistribution. ε* is now activation-weighted via Stage 2 A-covariance (no longer a deviation) |
| D-eps-star | 3 | Swift-SVD ε* normalization | 2604.01609 Eq. 4: ε*_k = (Σ_{j>k} σ_j²)^{1/2} — absolute truncation error | ε*_i = √(Σ_{j>k̄} σ̃_j² / Σ_j σ̃_j²) — relative ratio (normalized by total spectral energy) | Normalization makes ε* scale-invariant across experts with different total spectral energy, enabling meaningful cross-expert comparison within the redistribution step; the log() in the blending score damps large outliers regardless. Additionally, σ̃_j = sv(A^{1/2}·W), not sv(W·A^{1/2}) — A^{1/2} left-multiplies W to match the activation-weighted output error ‖XW − XW_k‖_F. |
| D10 | 4 | Eigenspace noise-floor truncation | 2410.21271 Alg. 1: full Q ∈ ℝ^{k×k} used; QQ^T = I guarantees Theorem 1 exactness | Not applicable after rank-1 fix: A = X̃X̃^T is rank-1 by construction, so there is exactly one non-zero eigenvalue (λ₁ = ‖X̃‖²) and one eigenvector (q₁ = X̃/‖X̃‖). The noise-floor threshold keeps only eigenvalues above a dtype-aware floor; for a rank-1 matrix this retains n_keep=1 in the non-degenerate case and n_keep=0 only if X̃≈0 (degenerate expert, skipped). No Theorem 1 exactness is weakened — the rank-1 structure makes k=1 the full eigenspace. |
| D11 | 2, 5 | Calibration data source | 2603.02217 §F.3 Table 1: c4; 2510.13999 §4: c4 + evol-codealpaca (used identically across all experiments) | Multi-domain Nemotron-Cascade-2-SFT-Data with weighted subsets (chat 0.56, math 0.21, science 0.11, etc.) | Task-aware calibration better matches target deployment distribution; c4 and evol-codealpaca are general pre-training / instruction-tuning data with limited reasoning/code coverage relative to the target deployment mix |
| D-cal-size | 2 | Calibration sequence count | 2604.04356 §4: 3072 sequences × 512 tokens (1.57M tokens total); 2510.13999: 1024 sequences × 2048 tokens (2.1M tokens total) | 4000 sequences × 2048 tokens (8.19M tokens total) (Nemotron weighted subsets) | Exceeds both papers' calibration volumes (5.2× REAM in tokens, 3.9× REAP in tokens); longer 2048-token sequences match the deployment context length and capture more inter-token routing patterns per sequence. Task-aware Nemotron dataset documented in D11 |

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
| 2604.01609 | Swift-SVD: Theoretical Optimality Meets Practical Efficiency in Low-Rank LLM Compression | 2026 | Stage 3 (α validation search + rank redistribution) |
| 2503.12340 | SVD-LLM V2: Per-Type Rank Allocation | 2025 | Stage 3 (motivation) |
| 2410.21271 | EoRA: Training-Free Compensation for Compressed LLMs | 2024 | Stage 4 |
| 2603.02217 | Router Knowledge Distillation for MoE Compression | 2026 | Stage 5 |

---

*This document was generated from a full algorithmic review of the max_quality codebase on 2026-04-28; §12 updated 2026-04-29 after a per-stage paper compliance audit including full methodology-section cross-reference of all 10 cited papers; further per-stage spec-only paper-compliance review on 2026-05-01 added D5a (REAM merge-group cap) and D5b (intermediate-neuron Hungarian alignment in merge), corrected the D-Rank citation (Eq. 7, not Eq. 6) and the §6 ε* formula to reflect activation weighting per D8, fixed the δ_gate similarity/distance notation in §5, clarified A-covariance reuse from Stage 2 in §6 Phase A, clarified the A-vs-B weighting roles in §6 Phase D, fixed the SVD reconstruction notation (`diag(Σ[:k])` instead of `S[:k]`), and corrected the §4 R^l-update rationale. Spec redesign on 2026-04-29: merged Stage 0 into Stage 1 (CKA + SE detection), floor=n//2, max_merge_group=8, Router KD bs=8. D9 resolved on 2026-04-30: Swift-SVD α selection now uses paper-exact WikiText-2 PPL validation (§3.2.2 of 2604.01609) instead of spectral proxy; D9 removed from §12. Phase C eigh caching added 2026-04-30: gate_proj/up_proj share the same B and C covariance; eigendecomposition is now precomputed once per expert and reused for both projections, eliminating ~7,200 redundant eigh(2048×2048) calls. Compute-time optimizations 2026-04-30: (1) Stage 2 sequential profiling with early-exit forward — **implemented**; (2) vectorized REAM accumulators — **planned, not yet implemented**; (3) Stage 5 KL chunk size increased to full sequence length on H200 — **implemented**; (4) torch.compile support for Stages 2.5/5 KD forward passes — **implemented**. Stage 6 compute-time optimizations 2026-04-30 — **all implemented**: (5) WikiText-2 PPL batch_size 1→8; (6) lm-eval batch_size=auto:8; (7) batched model.generate() for HumanEval and MATH-500; (8) torch.compile for prefill-dominant forward paths; (9) teacher eval caching with sha256 cache key auto-invalidation (~50% total time eliminated); (10) teacher I/O overlap via background CPU preload; (11) GGUF conversion overlap with teacher eval. All Stage 6 optimizations are purely computational scheduling — numerically identical to batch_size=1 baseline. Expected total Stage 6 speedup: ~8–12× (from ~3–5 hours to ~25–30 minutes on H200). SE detection rewritten 2026-05-05 (audit triage findings F-0006, F-0037, F-0016, F-0015, F-0012): §4 Phase B replaced with paper-exact Algorithm 1 criterion (three-way AND: a_{l,e} > P99.5(A) AND > 0.1·a_max AND l ∈ L); Phase A.5 added for MA-formation layer detection pre-pass; empirical SE scale stated as < 0.5% (no hard cap); canonical SE verification reference added. D1 (per-layer z-score deviation) and D2 (blacklist caps deviation) removed from §12 — spec now complies with the paper on these points. Stage 3 spec updated 2026-05-05 (audit triage findings F-ch12-missing-0001 CRITICAL and F-ch12-missing-0004 HIGH): (1) §6 Phase B rewritten to include FP64 Cholesky whitening per D-Rank paper 2509.25622 Eq. 1 — effective rank is now computed from SVD of `S_g · W_g` (whitened) not raw `W_g`; covariance `X_g^T X_g` sourced from Stage 2 `_stage2_input_covariance.pt` (A_gate_up for gate/up, A_down for down_proj); (2) §6 Phase C.5 added — AA-SVD block-level joint refinement per paper 2604.02119 Algorithm 2 §3.3: after each block's Phase C factorization, all factorized weight factors `{U_j, V_j}` and block-local RMSNorm scale parameters `θ_i` are jointly optimized via AdamW (lr=1e-4, 25 epochs, cosine schedule, batch 32) to minimize block output MSE against the original model; blocks processed sequentially 0→N-1; (3) §10 Protected Components updated to carve out RMSNorm scale parameters `θ_i` during Stage 3 Phase C.5 block refinement only (paper-required exception to the general RMSNorm protection rule). No §12 D-rows added or removed for these two changes — the spec now describes the paper-exact algorithms. All formulas were verified against the cited papers. All deviations are deliberate and documented. For the original validation audit, see the archived [VALIDATED_STRATEGIES.md](https://huggingface.co/pirola/moe-compression-workflow/blob/main/VALIDATED_STRATEGIES.md).*
