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
| 1 (SE + CKA) | +1 | 1024 | 2048 |
| 2 | +0 | 4000 | 2048 |
| 3 (B-cov) | +2 | 512 | 2048 |
| 5 (KD) | +5 | 3000 | 512 |

---

## 3. Budget Solver

**File:** [`budget/solver.py`](src/moe_compress/budget/solver.py)

> **Project-original heuristic.** The analytical starting point and iterative scaling rule below are not derived from any cited paper; they are an engineering layer that converts a target compression ratio (params or VRAM) into a per-stage parameter budget consumed by Stages 1–3. Because this is glue, not a divergence *from* a paper, no §12 D-row is warranted.

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
**Hardware:** H200. Original BF16 model (~70 GB) leaves 71 GB VRAM headroom. Two sequential forward passes over 256 calibration samples (Algorithm 1 Stage 1 then Stage 2, ~5 min combined; project-measured walltime, not paper-derived), then weight-space GRAPE computation on GPU.

### What

A single unified stage that (a) identifies super experts that must never be compressed, and (b) computes non-uniform per-layer expert budgets using activation-aware CKA similarity. SE detection uses two sequential forward passes over the calibration data (Algorithm 1 Stage 1 builds L, Stage 2 builds A); the CKA representations for GRAPE are collected during the Stage 2 pass (Phase B).

### Why

Super experts carry outsized influence on model output despite being activated at normal frequency. Pruning them causes catastrophic quality collapse (e.g., −21.7% relative average accuracy drop across 9 benchmarks (Table 3, non-thinking mode) on Qwen3-30B-A3B). They must be detected before any budget allocation.

Uniform pruning wastes budget — some layers have highly redundant experts (high pairwise CKA similarity) while others are diverse. GRAPE's +2.45% peak on Mixtral-8x22B at the prune-4-of-8 setting (paper Table 1; "4e" means 4 experts pruned per layer, 50% prune ratio) demonstrates the value of non-uniform allocation. Using CKA (rather than weight-space cosine) for the similarity metric gives GRAPE activation-aware redundancy estimates, producing better budgets for Stage 2.

### How

<!-- MA-formation detection (formerly §4 Phase A) — CONSUMED by
     stage1/plugins/ma_detection.py (commit pending). The plugin docstring
     is now the single source of truth; cross-verified against
     audit/spec_compliance/01_papers/2507.23279/source.md and against
     official-code commit ZunhaiSu/Super-Experts-Profilling @
     573aead3127ae593ba267758b832944f8fed1485. -->

<!-- SHARED — re-check at end. Sampling parameters paragraph below mixes
     Phase A (ma_detection — bs=32), Phase B (aimer + sink_token +
     three_way_and shared CalibrationEngine — bs=8), and Phase D
     (ablation_filter — bs=8). Resolve in Phase 9 by moving the
     non-MA-detection sentences into the consuming plugins. -->
**Sampling parameters (project-specified — items 4-5 of the deviation row in §12):** Phase A uses `phase_a_batch_size = 32` (Phase A only tracks max magnitudes, batch-size invariant; v4 ran at bs=4 with 76.2 GB VRAM headroom). Phase B uses `phase_b_batch_size = 8` — every Phase B accumulator (`DownProjMaxAccumulator`, `ExpertOutputAccumulator` reservoir sampling, `SinkTokenRoutingAccumulator` vectorized reduction) handles arbitrary `B`; the prior `bs=1` was inherited from a per-token routing-instrumentation path that the vectorization eliminated. The live H200 run at bs=1 used 94.6/150.8 GB VRAM (37% free, ~25 GB non-model state); at bs=8 the activation portion scales ~8× while the CKA reservoir + max accumulator are batch-invariant, yielding a projected ~120-130 GB total with ~20-30 GB headroom. Cuts Phase B forward-pass count from 1024 to 128. Phase D (ablation filter) uses `ablation_filter_batch_size = 8` for its held-out forward passes — Phase B accumulators stay resident through Phase D since Phase E (CKA) still consumes them; held-out cache adds little overhead. The 2026-05-10 H200 first-run telemetry (job 6a00caf0) showed model+Phase B+Phase E reservoir at ~99 GB resident before Phase D, leaving ~40 GB free; bs=32 OOM'd on the `ForCausalLMLoss` bf16→fp32 logits upcast (tried 60.6 GB), so bs=8 was selected (logits upcast ~9.5 GB; matches the v4-proven Phase F batch size). `num_calibration_samples = 1024` (down from 4000; saturates per-layer max and the 256-token reservoir while staying within the < 5% Frobenius drift threshold reported in 2603.18492's calibration sensitivity figure). See §2 calibration table for the cross-stage view.

<!-- SHARED — re-check at end. "Why L matters" describes the L-filter
     mechanic, which is enforced by three_way_and (Eq. 6's l ∈ L
     restriction) and inspected by aimer/sink_token/magnitude_topk via
     their candidate-gate logic. Resolve in Phase 9 by relocating into
     three_way_and (primary consumer of L). -->
**Why L matters:** The paper documents that some experts also produce extreme down_proj output magnitudes outside the MA-formation layers — these are called "outlier experts" (Table 7: L1E8, L47E48, L47E100 for Qwen3-30B-A3B; see Appendix C). Tables 6 and 7 are internally inconsistent for the first outlier expert in this model (Table 6: "Layer 47 Expert 8"; Table 7: "Layer 1 Expert 8") (the Table-6 entry 'Layer 47 Expert 8' is almost certainly a typo for 'Layer 1 Expert 8'; spec follows Table 7's L1E8 reading); this spec follows Table 7 (L1E8). These outlier experts do not contribute to MA formation and are not SEs. Not all outlier experts are excluded by the L-filter: L1E8 sits in Layer 1, which is an MA-formation layer (l ∈ L); Table 7 lists it as an outlier expert that is not classified as an SE, implying it fails the magnitude thresholds rather than being excluded by the L-filter (spec inference; paper does not explicitly classify why L1E8 fails the SE criterion). L47E48 and L47E100 sit outside L and are excluded by the L-filter. The l ∈ L constraint ensures that late-layer outlier experts outside L could not be blacklisted even if their magnitudes were large enough to satisfy the P99.5 and 0.1·a_max thresholds. Appendix C establishes that outlier experts lack the mechanistic significance of SEs but does not assert they would or would not pass the numerical thresholds.

<!-- SHARED — re-check at end. "Properties of L" is referenced by
     ma_detection (which produces L) and by all downstream consumers of L
     (three_way_and, aimer, magnitude_topk, sink_token). Resolve in
     Phase 9. -->
**Properties of L:** MA formation in MoE models typically begins in the first 1–3 decoder layers and then stabilises — Mixtral exhibits this in a single layer (paper §3.2.2 / Table 2: Mixtral-8x7B-Instruct SE at "Layer 1 Expert 3"), Qwen3-30B-A3B in three consecutive early layers. The MA pattern, once established, propagates stably across all subsequent layers via residual connections, so `L` is a small set of early layers (not the full layer stack). Note: this three-layer observation applies to Qwen3-30B-A3B (the paper's subject model); the pipeline's target model (Qwen3.6-35B-A3B) has a different architecture and its `L` will be determined empirically at runtime.

#### Phase B: Calibration Pass 2 — Expert Magnitude + CKA (256 samples)

All MoE layers are instrumented simultaneously. `run_calibration` runs once over all 256 samples (this is the second of the two passes; it is driven by Algorithm 1 (Appendix L) Stage 2, which covers expert magnitude collection for l ∈ L (Phase B = magnitude + CKA + sink-routing collection); candidate generation is described in Phase C (Phase C = candidate-set construction over four detectors); ablation filtering and final blacklist construction happen in Phase D — the CKA collection for GRAPE is performed in the same pass as a pipeline efficiency choice but is not specified by Algorithm 1), collecting two things per (layer, expert):

1. **Max activation magnitude** `max_{x∈D} |h_{l,e}(x) · W^{l,e}_{down_proj}|` — for super expert detection. Here `h_{l,e}(x)` is the intermediate activation entering the down_proj of expert `e` in layer `l`, and the magnitude is measured at the down_proj **output** (post-weight-multiplication), exactly as stated in Algorithm 1 line 19.
2. **Expert output representations** `f_e(x)` — for CKA pairwise similarity computation

The expert output representations are accumulated into per-layer representation matrices for CKA via reservoir sampling (**cap = 256 tokens per expert**, project-specified; see [D-ma-detector](#12-known-deviations-from-papers) which bundles all Stage 1 sampling/threshold project choices).

#### Phase C: Candidate Generation

Phase C produces a **candidate set** by union of four detectors. The candidate set is broad on purpose — false candidates cost ablation time in Phase D but cannot reach the final blacklist without ablation evidence. Each candidate carries a `provenance` list naming which detector(s) flagged it.

**Candidate sources:**

<!-- 1. Three-way AND criterion (paper Eq. 6) — CONSUMED by
     stage1/plugins/three_way_and.py (commit pending). Full paper Eq. 6
     transcription + verified source.md line refs + official-code citation
     + D-SE-A / D-a-max-fraction deviation analysis now live in the plugin
     docstring. -->

<!-- 2. AIMER bottom-pct (arXiv:2603.18492 Eq. 4) — CONSUMED by
     stage1/plugins/aimer.py (commit pending). Full paper Eq. 4 +
     official-code citation + two deviations (down_proj-only scoring;
     repurposed as SE-candidate signal) now live in the plugin
     docstring. -->


<!-- 3. Sink-token routing (arXiv:2507.23279 Figures 6/20/21 — descriptive
     observation; detection criterion project-original, D-sink-token-routing) —
     CONSUMED by stage1/plugins/sink_token.py (commit pending). Full
     deviation rationale + v6-vs-v4 threshold archaeology + official-code
     citation now live in the plugin docstring. -->

4. **Magnitude top-K in `l ∈ L`** (see [D-magnitude-topk-candidates](#12-known-deviations-from-papers)): for each `l ∈ L`, top-`magnitude_topk_per_l_layer` (=16) experts by `per_expert_max(l, ·)` not already flagged by sources 1-3. Catches SEs whose magnitude doesn't quite cross the three-way AND but is still large; K=16 = 2× the model's active-experts-per-token (top-8 routing).

**De-duplication** is by `(layer, expert)` pair; provenance is the union of all sources that flagged the candidate. Final filter is Phase D's ablation pass.

**Empirical scale (paper 2507.23279):** SEs account for fewer than 0.5% of all experts across the MoE models studied (Table 1: 0.05% for Qwen3-30B-A3B, 0.06% for DeepSeek-R1, 0.11% for DeepSeek-V2-Lite-Chat, 0.39% for Mixtral-8x7B-Instruct-v0.1). Source 1 alone reproduces the paper's canonical SE set on Qwen3-30B-A3B (Table 2: L1E68, L2E92, L3E82); v6 broadens the candidate pool with sources 2-4 to catch architecture-shifted SEs that the static three-way AND threshold misses on Qwen3.6.

#### Phase D: Ablation Filter (replaces static-threshold blacklist construction; see [D-causal-ablation-validation](#12-known-deviations-from-papers))

Phase D ablates each candidate produced by Phase C and keeps only those with measurable causal impact.

**Procedure:**

1. **Held-out slice**: 100 calibration samples drawn with a deterministic seed offset distinct from Phase A/B, cached at `_calibration_cache_phase_d/`.

2. **Baseline**: forward over the held-out slice with no ablation; record mean per-token NLL `baseline_nll`.

3. **For each candidate `(l, e)`**: install a forward hook that zeros expert `e`'s `down_proj` output during the forward; measure `ablated_nll`; ΔNLL = `ablated_nll − baseline_nll`. Remove hook; restore.

4. **Filter**: blacklist = `{(l, e) | ΔNLL > ablation_filter_threshold}` (default 0.001 ≈ 0.1% PPL impact). Per-candidate ΔNLL retained in artifact for audit.

**Cost**: ~`|candidates|` × forward-pass time. At `ablation_filter_batch_size = 8` and 100 holdout samples (~13 batches per candidate), each candidate takes ~15 sec → ~15–30 min for a 60–100 candidate set. Phase D runs while Phase B's accumulators are still resident (Phase E (CKA) consumes them), so Phase D's resident memory is Phase B's footprint **plus** the held-out cache (~99 GB resident before Phase D on H200 per job 6a00caf0; bs=8 puts the per-batch logits upcast at ~9.5 GB with ample headroom).

**Why this is load-bearing**: the v4 run produced a 158-expert blacklist of which only 5 had measurable ablation impact (144 dead-weight, 9 false positives that *hurt* PPL when protected). Static thresholds are fragile across architectures; ablation is ground truth.

#### Phase E: CKA Distance Matrices

For each MoE layer, compute the pairwise CKA **distance** matrix `D^l ∈ ℝ^{N×N}` where `D^l_{ij} = 1 − CKA(f_i, f_j)` (distance, not raw similarity: 0 = identical, 1 = maximally different). CKA measures functional similarity between experts based on their response patterns to actual inputs — two experts that produce similar outputs on the calibration data have high CKA and thus low distance D^l_{ij}. The paper's `D^l` is a *similarity* (GRAPE 2604.06542 line 245) and selects pairs with `argmax`; this spec uses the distance form `1 − CKA` and `argmin`, an equivalent sign-flip documented at [D-cka-distance](#12-known-deviations-from-papers).

Paper §3.3 explicitly allows "CKA, mean squared error, or other similarity measures" for D^l. CKA is the metric used by Zhang et al. (2025), cited in GRAPE §3.2 as the reference for intra-layer redundancy assessment.

With 256 samples × 2048 tokens ≈ 524K total token activations across the layer (each expert sees only its top-k/N routed fraction; for top-8 over 256 experts that is ≈ 16K per expert before sampling), reservoir-sampled to 256 per expert for CKA so the kernel matrices are well-conditioned for 256-expert layers.

#### Phase F: GRAPE Budget Allocation

**SE blacklist interaction (spec-original integration of Phases A and B; see [D-se-blacklist-merge](#12-known-deviations-from-papers)):** Before the greedy loop, each SE's row and column in `D^l` are zeroed so SEs never participate in pair selection and their distances do not contribute to `R^l`. SE cluster slots are subtracted from both `cluster_counts` and the global budget (`effective_budget = global_budget − total_SEs`), so the loop terminates when the non-SE surviving count meets the non-SE budget. The floor is also applied to the non-SE pool only: `floor_l = max(min_experts − |SE_l|, 0)`.

1. **Initialize** each expert as its own cluster. Compute per-layer redundancy `R^l = Σ_{i≠j} D^l_{ij}` (Eq. 11, applied to the distance matrix per [D-cka-distance](#12-known-deviations-from-papers)). Note: the distance↔similarity identity `R^l_dist = N(N−1) − R^l_sim` assumes the i=j diagonal is excluded (Σ_{i≠j}); including it would break the offset because 1−CKA(f_i,f_i)=0 vs CKA(f_i,f_i)=1. Set entropy threshold `Ê = E × (1 − γ)` (Eq. 10) where `E` is the initial cross-layer entropy (paper notation: unsubscripted `E` defined at initialization; some implementations write `E_0`) and `γ=0.1` is project-chosen (see [D3](#12-known-deviations-from-papers); the paper gives no default).

2. **Greedy loop** until total surviving experts ≤ `effective_budget = global_expert_budget − total_SEs` (paper notation: K; see [D-se-blacklist-merge](#12-known-deviations-from-papers) for the SE subtraction):
   - If all layers frozen and budget not yet met → **restart** (unfreeze all; if budget already met, the outer loop exits normally). The same iteration that clears `frozen` then immediately runs argmin layer/pair selection and performs one merge; the entropy gate is then re-evaluated at the end of the same iteration (paper Algorithm 1 lines 11–12) — this "one extra merge per restart cycle to escape local optima" is documented at [D-grape-restart-merge](#12-known-deviations-from-papers).
   - Pick `l* = argmin R^l` among unfrozen layers strictly above their floor (smallest total pairwise distance = most redundant layer; floor enforced during greedy by skipping any layer at-or-below floor when picking argmin/argmax — see [D5](#12-known-deviations-from-papers); distance-vs-similarity sign-flip vs paper's `argmax` per [D-cka-distance](#12-known-deviations-from-papers)) (distance form: smaller R^l = more redundant; equivalent to paper's argmax R^l_sim under R^l_dist = N(N−1) − R^l_sim per [D-cka-distance](#12-known-deviations-from-papers)).
   - Pick `(i*, j*) = argmin D^{l*}_{ij}` (most similar pair = smallest distance; same distance-vs-similarity inversion per [D-cka-distance](#12-known-deviations-from-papers))
   - **Merge:** zero out `j*`'s row/column in `D^{l*}` (spec deviation D4 vs paper line 9, which zeros only the pair entries); update `R^{l*}` from the modified matrix (paper line 10).
   - Decrement `cluster_counts[l*]` (Stage 1 implicit; tracked by spec).
   - Recompute cross-layer entropy `E` from updated `cluster_counts` (paper line 11) (only `cluster_counts[l*]` changes per iteration; entropy is recomputed from the full `cluster_counts` vector but the result reflects only the single-layer change).
   - If `E < Ê` → **freeze** layer `l*` (paper line 12); else continue. (See [D-grape-restart-merge](#12-known-deviations-from-papers) for the second restart path: lag-corrected post-selection restart, which is project-original.)

3. **Floor constraint:** `min_experts_per_layer = num_routed_experts // 2` (= 128 for 256-expert layers). Applied to the non-SE pool per layer. No early/late layer bonuses — the floor alone provides sufficient protection at 50% max removal per layer (see [D5](#12-known-deviations-from-papers)).

### Key Formulas

```
R^l = Σ_{i≠j} D^l_{ij}                          (Eq. 11, applied to the distance matrix per [D-cka-distance]; smaller distance-R^l = more redundant layer. Smaller distance-R^l ↔ larger similarity-R^l under the constant offset N(N−1), so argmin on the distance matrix selects the same layer as argmax on the similarity matrix.)
R̃^l = (R^l − min R) / (max R − min R)           (Eq. 3 — for logging only; under the distance convention from [D-cka-distance], the explicit identity is R̃^l_dist = 1 − R̃^l_sim. Derivation: under R^l_dist = N(N−1) − R^l_sim, the min/max swap roles: min R_dist = N(N−1) − max R_sim and max R_dist = N(N−1) − min R_sim. Then R̃^l_dist = (R^l_dist − min R_dist) / (max R_dist − min R_dist) = (max R_sim − R^l_sim) / (max R_sim − min R_sim) = 1 − R̃^l_sim. Polarity is inverted vs paper Eq. 3 but the cross-layer ranking is preserved.)
Ê = E × (1 − γ)                                  (Eq. 10 — entropy threshold; paper notation: unsubscripted `E` defined at initialization)
E = −Σ_l (c_l / C_total) × log(c_l / C_total)   (cross-layer entropy; c_l = |C_l| is the cluster count for layer l, C_total = Σ_l |C_l| — matches paper Eq. 9 notation p_l = |C_l|/Σ_{l'} |C_{l'}|) (canonical paper numbering; the OCR'd source.md reflowed Eq. labels — cross-reference may show Eqs. 7/8 in the markdown rendering)
```

### Resume

Stage 1 is stateless (JSON-only output: blacklist + per-layer budgets). Re-running is cheap and always safe.

### Correctness Notes

- The `R^l` update zeroes out the merged expert's *entire* row and column (not just the pair), preventing the absorbed expert from being re-selected in future iterations. The paper's pseudocode line 10 (`R^l ← R^l − 2·D[i*,j*]`) is consistent with the sum-over-`i≠j` form of `R^l` (the 2× accounts for both `D[i*,j*]` and `D[j*,i*]`), but it only adjusts the scalar `R^l` while leaving stale similarity entries in row/column `j*` that can mis-rank future merges. Zeroing the full row/column eliminates that staleness. See [D4](#12-known-deviations-from-papers).
- If the budget cannot be reached (all layers hit their floors), a warning is logged but the pipeline continues with the achieved budget.

### Blacklist Output (`stage1_blacklist.json`)

Stage 1 emits `stage1_blacklist.json` containing the ablation-validated Super Expert blacklist — `(layer, expert)` index pairs whose Phase D ΔNLL exceeded `ablation_filter_threshold`. The candidate pool from Phase C (which can include three-way AND, AIMER, sink-token, and magnitude-top-K provenance) is recorded separately under `aimer.candidates`, `sink_token.candidates`, and `magnitude_topk.candidates` for audit; the per-candidate ΔNLL is in the companion `stage1_ablation_filter.json`. Final blacklist size is bounded by both the candidate pool and the threshold cut — typical Qwen3.6 runs produce 10–30 ablation-validated SEs out of 50–100 candidates, well below the paper's < 0.5% empirical scale.

Shared experts (`mlp.shared_expert`) are **not in the blacklist** and are not processed by Stage 1 at all. They live in a separate model attribute, distinct from the routed `mlp.experts` list, and are architecturally invisible to `iter_moe_layers`, GRAPE, and REAM. No explicit exclusion is needed — they are simply never candidates.

**What GRAPE contributes to Stage 2 is per-layer budgets (N'_l), not individual expert blacklists.** N'_l ≥ `min_experts_per_layer` (128) for every layer due to the floor constraint. Stage 2 uses N'_l as the target centroid count for REAM per layer; REAP scores then determine which N'_l routed non-blacklisted experts become centroids.

> Stage 1 spec touch-up 2026-05-07 (iter+3): clarified Phase F entropy-recompute scope (only c_{l*} changes per iter); §4 Phase E citation §3.2 → §3.3; GRAPE K vs effective_budget cross-link in Phase F step 2; minor wording polish on Mixtral percentage and first-MoE-layer scope. (Phase letters updated 2026-05-10 v6 rename: pre-v6 D/E referred to CKA/GRAPE, now E/F.)

---

## 5. Stage 2 — REAP Scoring + REAM Pseudo-Pruning

**File:** [`stage2_reap_ream.py`](src/moe_compress/stage2_reap_ream.py)
**Papers:**
- REAP: Routing-Expert Activation Pruning (2510.13999), Eq. 9
- REAM: Routing Expert Activation Merging (2604.04356), §3–4, Eq. 4–8 (Eq. 7 the aggregator)
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

- **Gate logit profiles** (`ReamCostAccumulator`): Instead of `dict[expert_id → dict[token_idx → float]]`, a pre-allocated `torch.Tensor(num_experts, total_calibration_tokens)` on CPU in float16 stores each expert's pre-softmax router logit for each calibration token. The full `[N_experts × N_experts]` δ_gate similarity matrix is computed in one `F.normalize` + `matmul` call (~milliseconds for 256×256): normalize each row to unit length, compute the gram matrix (inner products = cosine similarities of unit vectors), convert to Euclidean distances via `d = sqrt(2 − 2·cos)`, then apply `dist2sim`. This replaces O(N²) Python-level loops. **Memory note:** at the updated calibration size of 4000 × 2048 = 8.19M tokens with 256 experts, the logit tensor is 256 × 8.19M × 2 bytes ≈ 4.2 GB per layer in FP16 on host RAM. This is materially larger than the prior 1024-sequence budget (~1.1 GB). The H200's host RAM (512 GB) comfortably accommodates this; the tensor is allocated and freed per layer, not held across layers simultaneously.

- **Gated-output pairwise similarity** (`finalize_batch`): Per-batch pairwise cosine similarity of gated expert outputs is computed via a single batched `F.cosine_similarity` over the jointly-active token intersection per expert pair, accumulated incrementally as before but with vectorized inner loops.

These optimizations are purely implementation-level data-structure changes. The mathematical computation is identical — same cosine similarities, same REAP scores, same cost matrix entries. Estimated wall-clock reduction on the cost-matrix phase: 10–100× (from minutes of Python iteration to seconds of tensor ops). Not yet implemented — the current accumulators use Python dicts (functionally correct, slower). The early-exit optimization provides the dominant ~2× speedup; vectorized accumulators are additive.

**Per-layer merge execution (sequential — must see prior merges):**

#### Blacklisted Expert Exclusion

Before any REAP/REAM computation, **super experts (SEs)** are excluded from the routed expert pool — they are not candidates for the centroid set and not candidates for the non-centroid set; their weights pass through Stage 2 unchanged. SEs are identified by the (layer, expert) pairs in `stage1_blacklist.json`. Placing an SE in the centroid set would allow non-centroid weights to be merged into it, modifying the SE's weights — defeating the purpose of blacklisting.

**Shared experts** (`mlp.shared_expert`) are never in scope: they live in a separate model attribute, are not indexed as routed experts, and are never iterated by `iter_moe_layers`. No explicit exclusion logic is needed for them in Stage 2.

GRAPE outputs per-layer **budgets** (N'_l — how many routed experts to keep per layer), not individual expert blacklists. The floor constraint (min 128 per layer) is enforced on N'_l; REAM then selects which N'_l routed non-SE experts become centroids via REAP score.

All counts below (N'_l, feasibility checks, group sizes) refer to non-SE routed experts only.

#### Step 1: REAP Scoring (Paper 2510.13999, Eq. 9)

> **Note on routing weight notation:** REAP (2510.13999) uses `g_j(x)` for the post-softmax routing weight, masked to zero for non-top-k experts. REAM (2604.04356) uses `σ(x)_j` for the **full unmasked softmax** (always strictly positive for every expert on every token). In §5 below, `g_j(x)` follows REAP notation (masked, zero for non-active) and is taken **as dispatched in the model's forward pass** — for Qwen3-MoE this is the renormalized top-k weight (top-k softmax outputs renormalized to sum=1 over the top-k set); REAP's paper Eq. 9 is silent on renormalization, and the spec uses the dispatched value to match the experts' actual contribution to the forward output. The REAM Eq. 8 formula uses `σ(x)_j` as the **full unmasked softmax** (no top-k mask, no renormalization) — confirmed by the reference implementation: `F.softmax(router_logits, dim=-1)` over all experts with no top-k masking (`ream/moe_utils.py` lines 157–158, 173–174 — upstream REAM repository, not this codebase).

For each expert `j`, compute importance as the conditional average of gate-weighted output norm over active tokens (Eq. 9):

```
S_j = (1/|X_j|) × Σ_{x ∈ X_j} g_j(x) · ‖f_j(x)‖₂
```

where `X_j = {x | j ∈ TopK(σ(x))}`, `g_j(x)` is the post-softmax routing weight, and `f_j(x)` is the expert output vector.

#### Step 2: REAM Cost Matrix (Paper 2604.04356, Eq. 5, 7, 8)

**Activation-space similarities** (NOT weight-space; higher = more similar; both components scaled to [0, 1]):

- **δ_gate(i,j)** (Eq. 5): Similarity between **pre-softmax** router logit profile vectors. Each expert's profile is a vector of length |X| (one pre-softmax logit per calibration token). Profiles are **L2-row-normalized** (each expert's profile vector is unit-normalized), then pairwise Euclidean distances are computed; `dist2sim` converts to similarity by dividing by the matrix-wide maximum distance and subtracting from 1. δ_gate ∈ [0, 1]; higher = more similar. (Reference: `ream/ream.py` lines 37–41; reference paths point at the upstream REAM repository, not this codebase.) The L2-norm + Euclidean + `dist2sim` chain is a monotone transform of the paper's raw cosine into [0, 1] (since `dist = √(2 − 2·cos)` on unit vectors, and `1 − dist/max(dist)` is monotone-decreasing in `dist`); greedy ranking is preserved [D-ream-similarity-rescale].

- **δ̃_expert(i,j)** (Eq. 8): `(1/|X|) Σ_{x∈X} sim(σ(x)_i · E_i(x), σ(x)_j · E_j(x))` — mean per-token cosine similarity of the two experts' **full-softmax-gated** outputs (`σ(x)_i` is the full unmasked softmax weight), averaged over all |X| calibration tokens (not just jointly-active tokens). The raw cosine similarity ∈ [−1, 1] is **rescaled to [0, 1]** as `(cosine_sim + 1) / 2` [D-ream-similarity-rescale]; this rescaling is monotone, so greedy ranking is preserved. δ̃_expert ∈ [0, 1]; higher = more similar. (Reference: `ream/ream.py` lines 99–113, `moe_utils.py` lines 157–158, 173–174.) **Sparse-routing note:** Eq. 8 calls for `σ(x)_e · E_e(x)`; we use the full-softmax weight `σ(x)_e` directly (paper-faithful — note: this is the un-renormalized softmax, not the dispatched top-k routing weight) but `E_e(x)` is only computed on top-k tokens, so non-jointly-active tokens contribute zero to the numerator and still appear in the denominator |X| (see [D-ream-sparse-routing]). **NaN-handling:** if a jointly-active token produces a zero gated-output vector for one of the two experts (extremely rare; cosine is undefined), the per-token cosine is treated as `0` (after rescale: `0.5`, neutral) rather than excluded from the |X| denominator — same `[D-ream-sparse-routing]` rationale: skipped/degenerate token positions count toward |X| but contribute zero similarity signal.

- **δ_REAM(i,j) = (δ_gate(i,j) + δ̃_expert(i,j)) / 2**: Equal-weight average of gate and expert similarities; both components already in [0, 1]. δ_REAM ∈ [0, 1]; **higher = more similar**. The working distance is `cost(i,j) = 1 − δ_REAM(i,j) ∈ [0, 1]`; lower cost = more similar — the greedy assignment selects non-centroids with the **lowest cost**. (Reference: `ream/ream.py` lines 46–53.) Note: REAM Eq. 7 sums the two components (`δ_REAM = δ_g + δ̃_E`); the spec's mean is monotone in each component cosine; the joint greedy ranking matches the paper's exactly only when components agree on the pair, and is a project-original re-weighting otherwise. The /2 normalization keeps cost values in [0,1] for cross-stage diagnostic comparability [D-ream-aggregation].

#### Step 3: Greedy Pseudo-Pruning Assignment (Paper §4)

**Feasibility check:** Before the greedy pass, validate that `N'_l × max_merge_group_size ≥ N_l − N'_l` (where N_l is the total number of non-blacklisted routed experts in the layer, N'_l is the centroid count, and `max_merge_group_size` caps the **non-centroids** absorbed into each centroid — see Step 4). If this is violated, bump `effective_target` by 1 (or by `ceil(effective_target × cost_bump_ratio)` whichever is larger) and retry. If `effective_target` reaches `n_experts` without achieving feasibility, fall back to zero-merge: keep all non-protected experts as centroids (no merges performed for this layer). This guarantees no expert weights are lost at the cost of not meeting the compression target. (Reference: `ream/ream.py` lines 60–62 — upstream repository.)

**Quality-gate exhaustion (last-resort apply-anyway):** distinct from the feasibility-fallback above, if the **quality** gate (`ream_cost_sigma_threshold`, see [D-ream-budget-bump]) is still active when `effective_target` reaches `n_experts`, the spec applies the most recent above-threshold assignment instead of zero-merging — the rationale is that quality-gate failures often coincide with naturally high cost layers where any single-cap-respecting assignment is the best available, and zero-merging would unnecessarily cost compression. This is **project-original behavior**, not in the REAM paper, and is documented under [D-ream-budget-bump].

**Orphan-singleton promotion:** if the capped greedy assignment leaves any non-centroid unassigned (rare edge case where every centroid's cap is saturated by lower-cost candidates), the orphan is promoted to a singleton centroid for that layer (no merge). This is also project-original (the spec's Step-3 feasibility check is designed to prevent this, but the promotion path exists as a defensive safety net). Documented under [D-ream-budget-bump].

Top-N'_l experts by REAP score become **centroids**. Non-centroids are assigned to centroids via a **single-pass greedy algorithm**: iterate centroids in **descending saliency order** (most salient centroid first — order is important); for each centroid, absorb up to `max_merge_group_size` unassigned non-centroids with the **lowest cost** (most similar), in order. The loop exits early once all non-centroids are assigned. **Every non-centroid is guaranteed to be assigned** — the feasibility check ensures full coverage. (Reference: `ream/ream.py` lines 63–87.)

#### Step 4: Frequency-Weighted Merge (Paper Eq. 6)

```
W_merged = Σ_i (freq_i / Σ_j freq_j) × P_i(W_i)
        (Note: paper Eq. 6 uses raw `S_i^freq = freq_i/|X|` weights without group-renormalization, so paper-Eq.-6 weights generally do not sum to 1 over a merge group. The spec's group-renormalized form `freq_i/Σ_j freq_j` produces a convex combination — necessary for a well-formed weight average — and is mathematically equivalent to renormalizing paper-Eq.-6's `S_i^freq` weights post-hoc within the group. The spec form differs from a literal reading of Eq. 6 but is the only consistent way to interpret "weighted average" since the paper does not explicitly state how the un-normalized weights should be combined into a single tensor.)
```

where the denominator `Σ_j freq_j` sums over merge group members only (not all N experts). `P_i` denotes the neuron permutation alignment as described in the paper's surrounding text (Hungarian algorithm on combined cost matrix `C = C_act + C_wt`) that aligns each child expert's intermediate neurons to the centroid before averaging; it is not an explicit formula component in the paper. `C_wt` is the gate+up Frobenius weight distance (implementation choice: gate_proj and up_proj; paper does not specify). Implementation: `C_wt[p,q] = ‖W^p_gate − W^q_gate‖_F + ‖W^p_up − W^q_up‖_F` (sum of independent gate and up Frobenius distances — note: this is **not** the block-Frobenius `‖[W^p_gate, W^p_up] − [W^q_gate, W^q_up]‖_F = √(‖ΔW_gate‖_F² + ‖ΔW_up‖_F²)`; the spec uses the L1-of-Frobenius-distances aggregation instead, then min-max normalizes the resulting matrix as a single C_wt component). (See D5b for the C_wt + C_act decomposition; both components min-max normalized to [0,1].) `C_act` is the per-neuron mean activation L2 distance, where activation vectors H̄ are normalized before computing the distance (normalization method unspecified in the paper). Implementation: gate-output rows are L2-normalized per-row before computing pairwise Euclidean distances; equivalent to cosine distance on the normalized rows. `freq_i` is the count of calibration tokens for which expert i is in the top-k active set, equivalent to `S_i^freq × |X|` in the paper's notation (REAM Eq. 2).

#### Step 5: Router Resize

Remove merged non-centroid experts' rows from `gate.weight`. Update `num_experts` on the MoE block. SE rows are **not removed** — they remain in the router and expert list unchanged.

### Covariance Side-Collection

During the profiling forward pass, two covariance matrices are accumulated per (layer, expert):
- **A_gate_up** (`gate_proj`): Input covariance for gate_proj and up_proj (shared tensor)
- **A_down** (`down_proj`): Input covariance for down_proj (intermediate activations)

Stored in `_stage2_input_covariance.pt` (fp16 persisted dtype per [D-cov-storage-fp16]; eigendecomposition still runs in fp64 in-memory in Stage 3, so numerical conditioning is preserved). On H200 with `batch_size=6`, the covariance accumulates signal across all 4000 calibration samples, providing well-conditioned A matrices for Stage 3.

### Budget Bump Loop

Two safety gates can raise the effective target if merge quality is poor (project-original feasibility/quality gate; see [D-ream-budget-bump]):
- **`max_merge_group_size=8`** [D5a]: If any group exceeds this, bump target. The REAM paper uses C=16 at 25% reduction (128→96 experts) and C=32 at 50% reduction (128→64 experts) on a 128-expert pool — in both cases C is far larger than the average absorption per centroid. Our pipeline targets ~30% expert reduction on a 256-expert pool; at the floor budget (256→128), each centroid absorbs an average of 1.0 non-centroid, so C=8 provides 8× headroom above the average. The budget-bump fallback catches any groups that do exceed the cap.
- **`ream_cost_sigma_threshold=1.5`** [D-ream-budget-bump]: If mean cost exceeds `running_mean × (1 + 1.5)`, bump target (inactive for the first 4 layers that contribute valid mean-cost samples — layers without merges or with all-zero pair costs are excluded from the running history)

### Resume

Per-layer atomic checkpointing to `_stage2_partial/` (see §11 for the `.tmp + os.replace` idiom and `.pt`-before-`.json` ordering invariant):
- `merge_{layer_idx}.json`: centroid IDs, groupings, frequencies, merge map
- `layer_{layer_idx}.pt`: covariance snapshot for this layer

On resume, completed layers are replayed from partial files (fast, no forward pass). The model must be passed in pre-merge state (Stage 1 output) — a guard checks `num_routed_experts` matches the pre-merge count.

**Critical invariant:** Covariance remapping (`_remap_covariance_for_layer`) must happen BEFORE the snapshot. Snapshotting before remapping persists pre-merge expert keys, corrupting Stage 3 inputs on resume.

### Stage 2 v2 (revision spec: `max_quality/docs/stage2_assignment_revision.md`)

The **assignment + cost-matrix + merge** pipeline above is the v1 baseline (greedy + symmetric δ_REAM cost + freq-weighted merge, all preserved bit-identically when the v2 flags below are at their defaults). Stage 2 v2 layers the following opt-in features on top, gated by config flags under `stage2_reap_ream:`. Defaults are baseline-off; flipping them on activates each feature in isolation for ablation. See `max_quality/docs/stage2_assignment_revision.md` for the full design and the § 8 ablation matrix that locks in production defaults.

- **`assignment_solver`** (default `"greedy"`) — replaces the legacy descending-saliency greedy with a configurable solver: `hungarian` (rectangular `scipy.linear_sum_assignment` for slack-capacity 1-1 assignment), `mcf` (capacitated min-cost flow via OR-Tools `SimpleMinCostFlow`), `auto` (dispatch hungarian↔mcf based on `n_NC ≤ N'_l`), or `sinkhorn` (capacitated entropy-regularized OT via log-domain Sinkhorn-Knopp with linear ε-annealing and a slack-child dummy-row construction). [D-mcf-assignment, D-sinkhorn-soft-assign]
- **`cost_alignment`** (default `"pre"`) — when `"post"`, replaces the symmetric δ_REAM cost with the per-pair **Hungarian-aligned whitened residual** `‖(W_c − P_cm·W_m) · A^{1/2}‖_F` (sum over gate/up/down per the AA-SVD lineage; `A^{1/2}` multiplies ΔW on the **right**, input axis). The Hungarian permutation `P_cm` is cached and reused by the merge step, so each pair is aligned exactly once. [D-whitened-cost]
- **`cost_whitening`** (default `"none"`) — `"diag"` uses `sqrt(diag(A))` (cheap fallback), `"full"` uses the eigen-sqrt `V·diag(sqrt(λ_clamped))·V^T` (AA-SVD form, mirrors `stage3_svd._precompute_eigh`).
- **`cost_asymmetric`** (default `false`) — multiplies the post-alignment residual by `freq_m / (freq_c + freq_m)` so high-frequency non-centroids are penalized when they would dominate (wash out) a low-frequency centroid. Valid only with `ream.frequency_weighted_merge=true` (rejected at run-time otherwise). [D-asymmetric-freq]
- **`cost_topk_filter`** (default `48`, only used when `cost_alignment="post"`) — the K-prefilter: per non-centroid m, only the top-K candidate centroids by cheap symmetric δ_REAM get the expensive whitened residual computed; the rest get `+∞` and the assignment solver treats them as forbidden arcs.
- **`capacity_util_threshold`** (default `0.25`) — capacity-utilization gate (M3): `u = n_NC / (N'_l × C_max)`. When `u < threshold` the layer falls back to the cheap `"pre"` path regardless of `cost_alignment`, since slack capacity makes the heavy machinery unlikely to change the assignment. [D-capacity-util-gate]
- **`em_refinement_rounds`** (default `0`) — EM-style assignment refinement (M4 / Sub-MoE): tentatively merge each non-singleton group with current assignment, recompute the cost matrix against the merged centroid, re-solve the assignment, repeat. Only meaningful under `cost_alignment="post"` (the cheap symmetric cost doesn't depend on centroid weights). Stops early if `em_convergence_break=true` (default) and the assignment stops changing. [D-em-refinement]
- **`expert_distill_steps`** (default `0`) — per-merge-group MSE distillation (M8): for each non-singleton group, snapshot the pre-merge expert weights, then run AdamW (default 500 steps, lr=1e-4, betas=(0.9, 0.95)) to optimize the merged centroid against a freq-weighted target of pre-merge group-member outputs on reservoir-sampled layer-input tokens. Plateau early-break, fp32 optimizer with bf16 forward, bank weights restored to original dtype on writeback. [D-expert-distill-mse, D-expert-distill-mse-v1]

**Resume schema (v2).** `_stage2_partial/merge_{layer_idx}.json` bumped from `format_version: 1` to `format_version: 2` to carry the new forensic / resume fields: `assignment_solver_used`, `cost_alignment_used`, `em_rounds_completed`, `distill_state` (per merged-group dict). **No backward-compat shim** — operators upgrading mid-pipeline must finish a stage on one version or restart cleanly (per § 11 strict version match).

---

## 5.5. Stage 2.5 — Post-Merge Router Calibration

**File:** [`stage5_router_kd.py`](src/moe_compress/stage5_router_kd.py) (same code as Stage 5)
**Paper:** Router Knowledge Distillation for MoE Compression (2603.02217), Eq. 3, Table 1, §F.3
**Hardware:** H200 required. VRAM accounting: teacher BF16 ~70 GB + student BF16 ~50 GB + logits/activation buffer ~6 GB → ~126 GB live; ~15 GB headroom on H200's 141 GB. (At Stage 5 the student is post-Stage-4 FactoredExperts and is smaller than 50 GB, so the ~120 GB / ~126 GB / ~15 GB headroom is conservative for Stage 5; exact at Stage 2.5 where the student is the pre-SVD pruned model.)

### What

Runs the Router KD algorithm (identical to Stage 5) on the Stage 2 output — before SVD factorization. Only `mlp.gate.weight` is trainable; all expert, backbone (attention, embeddings, lm_head, RMSNorm) parameters remain frozen, per Router-KD paper §4. Gradients flow through the soft-gating weights of the selected experts during the forward pass (so the router's gate signal reaches the loss); only `mlp.gate.weight` parameter values are updated.

### Why

After Stage 2, the router has been **resized** (rows for deleted experts removed) but never retrained. The surviving router weights were calibrated for the original 256-expert landscape. They now route among ~180–200 merged experts whose weight distributions have shifted. Stage 3's covariance collection runs on this degraded routing — better routing at this point means the cross-covariance and B-covariance collected by Stage 3 are more representative of actual inference-time token distribution per expert.

**Stage 2 v2 interaction (revision spec).** When Stage 2's per-merge-group expert distillation (`expert_distill_steps > 0`) is enabled, Stage 2.5 receives a model whose merged-centroid weights have **already been distilled** to match a freq-weighted target of pre-merge group-member outputs. Stage 2.5's job is therefore purely router calibration on top of pre-distilled experts — not expert recovery. The freeze pattern is unchanged: only `mlp.gate.weight` is trainable; expert/backbone weights stay frozen.

Stage 2.5 is distinct from Stage 5: Stage 5 recalibrates routers after SVD factorization and EoRA. Stage 2.5 recalibrates after merging only. Both are needed: the model changes again in Stages 3 and 4, making Stage 2.5's routers stale again — Stage 5 corrects this. The full chain is: merge → heal routers (2.5) → factorize → compensate → heal routers again (5).

### How

Identical to Stage 5 (§8), with two differences:

| Parameter | Stage 2.5 | Stage 5 |
|---|---|---|
| Input model | Stage 2 output (dense merged experts) | Stage 4 output (FactoredExperts + EoRA) |
| Teacher precision | BF16 (both stages) | BF16 (both stages) |
| Checkpoint prefix | `_stage2p5_partial/` | `_stage5_partial/` |
| Hub artifact | `<base>-stage2p5` | `<base>-stage5` |

All other hyperparameters (lr=5e-5, bs=8, τ=1.0, epochs=1, max_samples=3000, seq_len=512) inherit from §8. (See §8's batch-size hyperparameter table for the bs=8/accum=1 ↔ paper's bs=2/accum=4 equivalence rationale.)

**Calibration data:** Nemotron weighted subsets per §2 (D11) — same calibration source as Stage 2 and Stage 5; canonical reference is §2.

**Teacher loading fallbacks:** Stage 2.5 may use the same 4-bit / cache fallbacks as Stage 5 if VRAM is tight — teacher loadable in 4-bit via bitsandbytes, or precomputed teacher vocabulary logits loaded from a sidecar cache file (`teacher_logits_cache` config key) to skip the live teacher entirely (cache wins on conflict — see §8 Teacher Loading for the precedence rule). See §8 "Teacher Loading" for details.

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
**Hardware:** H200 (141 GB). Pruned student (~45 GB post-Stage-2.5) stays resident from Stage 2.5. Original BF16 teacher (~70 GB) is loaded for the Phase A cross-covariance dual-forward (~115 GB combined) and **stays resident through Phase C.5** so its block forwards can be invoked on-demand during block refinement. Phase C.5 trains one block at a time; per-block trainables ~525 M params (~1 GB BF16; fp32 AdamW state ~6.3 GB at 12 B/param) plus per-block activations + grads (~5–10 GB at batch 32, seq 2048) leaves ~9–15 GB headroom. The previous spec variant freed the teacher after Phase A and fed Phase C.5 from a 215 GB on-disk teacher-block-output cache (Phase A.X) — that path was an A100-era constraint and is **removed** on H200. (See Phase C.5 "Optimizer-precision rationale" for the per-component breakdown.)

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

The teacher model is **not** freed after Phase A. It remains resident through Phase C.5 so that the original block forwards `ℒ_i(X)` required by Phase C.5's anchored objective can be invoked on-demand during refinement (see Phase C.5 below). The teacher is freed only after Phase C.5 of the final block completes; the factoring step (Phase B/B.2/C) does not use the teacher and could free it earlier on hardware tighter than H200, but on H200 the cost of keeping it resident is zero relative to the simpler control flow.

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

Then compute the whitened weight matrix `S_g · W_g^T` (where `W_g^T` transposes from PyTorch's stored `[d_out × d_in]` to `[d_in × d_out]`, giving a `[d_in × d_out]` result; applied per-expert). The singular values used for effective rank are those of `S_g · W_g^T`, not of raw `W_g`. The per-group covariance `X_g^T X_g` is the average over all experts in the group (shared input distribution for the same matrix type within a layer). (Equivalent to paper Eq. 1; the transpose is a storage-convention transform — singular values identical.)

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

Each expert is factored independently with no shared basis; the D-Rank ω adaptation is per-expert (D7). **Per-projection bias** (approximately budget-neutral): `gate_proj=1.33`, `up_proj=0.67`, `down_proj=1.0`. The ratio `gate:up:down = 4:2:3` is adopted from jangq's MLP-asymmetry analysis for SwiGLU quantization (`397B-MLP-ASYMMETRY.md` §3.1), translated from bit space to rank space. Rationale: gate errors are amplified multiplicatively through `SiLU(gate)·up`; down errors propagate to the residual stream of every downstream layer; up errors are bounded and linear. The multipliers sum to 3.0 across the three projection types, approximately parameter-budget-preserving: exactly preserved when gate/up/down receive the same `k_g` and share the same `ω_g` (which holds under SwiGLU symmetry where gate/up have identical input dimensions). In the general case, the multipliers redistribute rank between projection types and may shift the post-bias parameter total by a few percent. See [D7a](#12-known-deviations-from-papers). The mean rank `k̄` used in `ε*` is the bias-adjusted group rank (i.e., after applying the gate/up/down multipliers from Step B.3 — deviation from paper: the paper defines k̄ as the plain uniform rank k̄ = (m×n)/(m+n) × ρ; see D7a).

#### Phase B.2: Swift-SVD Per-Expert Rank Redistribution (Paper 2604.01609, Algorithm 2)

Within each (layer, matrix_type) group, D-Rank gives a uniform rank `k_g` to every expert. Swift-SVD refines this by redistributing the group's total rank budget `k_g × N_experts` across individual experts using a blending score:

```
s_i = β_i^α · (log(e + ε*_i))^{1-α}
```

where e ≈ 2.718

- `β_i = σ_i² / Σ_j σ_j²` — spectral energy proportion (how much of the group's total spectral energy this expert contributes; see [D8](#12-known-deviations-from-papers))
- `ε*_i = √(Σ_{j>k̄} σ̃_j² / Σ_j σ̃_j²)` — activation-weighted reconstruction error at the group's mean rank `k̄`, where `σ̃_j` are the singular values of `A^{1/2}·W` (Stage 2 input auto-covariance from §5; see [D8](#12-known-deviations-from-papers) — ε* is now activation-weighted, not spectral-only). Higher = this expert needs more rank. This equals `‖A^{1/2}·W − A^{1/2}·Ŵ(k̄)‖_F / ‖A^{1/2}·W‖_F` = `‖A^{1/2}·(W − Ŵ(k̄))‖_F / ‖A^{1/2}·W‖_F` (relative reconstruction error in the activation-weighted norm, where A^{1/2} left-multiplies W as in ‖XW − XW_k‖_F = ‖A^{1/2}(W − W_k)‖_F), i.e., the activation-weighted analogue of the paper's spectral ε*. (Deviation from paper: the paper's ε* is absolute truncation error; the spec normalizes to a relative ratio for cross-expert comparability — see §12 D-eps-star.) (`k̄` is the bias-adjusted group rank from Step B.3, per D7a.)
- `α ∈ [0, 1]` — balances the two signals

**α selection (paper §3.2.2 — validation-based):** For each candidate α ∈ {0.0, 0.1, ..., 1.0}, the full model is factored at the corresponding per-expert ranks using the closed-form solution from Swift-SVD Eq. 3: W*_k = W V_k V_k^T, where V_k are the top-k right singular vectors of XW, equivalently the top-k eigenvectors of W^T A W (activation-weighted weight covariance, computed as W^T @ A_g @ W for each expert; A_g is the Stage 2 pre-prune input covariance from _stage2_input_covariance.pt — paper-exact per Theorem 3.1 / Eq. 3) and evaluated on WikiText-2 PPL (computed per the §9 'WikiText-2 PPL Protocol' subsection (corpus, chunking, NLL aggregation), with `validation_samples: 512` chunks; project-tunable — Swift-SVD §3.2.2 and Appendix do not specify a sample count for the WikiText-2 PPL grid search). The α yielding the lowest end-to-end perplexity is selected. This implements the paper's exact procedure: *"For each candidate corresponding to α_i, the optimal low-rank approximation of every layer is computed using the closed-form solution in (3). The resulting compressed models are then evaluated on a validation set, and the candidate that yields the best end-to-end performance is selected."*

The factoring reuses cached spectral components from Phase A's B-covariance collection; each candidate requires ~2 minutes for a full 40-layer factor pass and ~20 seconds for PPL evaluation on H200. No model copies are made — originals are snapshotted to CPU RAM (~50 GB; H200 has 256 GB host RAM) and restored after each evaluation. Total α search: ~33 minutes for 11 candidates.

**Paper-compliance contract.** The α search MUST complete the paper-exact end-to-end PPL validation (Swift-SVD §3.2.2). If host RAM headroom at α-search entry is insufficient (<15 GB available), Stage 3 raises `RuntimeError` immediately rather than degrade to a spectral proxy — silently producing a non-paper-compliant model is worse than failing fast. Operators must provision adequate host RAM (~50 GB for the Qwen3-30B snapshot plus working set) or reduce `validation_samples` to fit. The previously-shipped silent spectral fallback was deviation D9 and was removed from Ch. 12 specifically because the pipeline now refuses to run that path.

> **Implementation follow-up:** Replace the current OOM auto-fallback at `stage3_svd.py:285-301` with a hard `RuntimeError` to bring the code in line with this contract.

**Redistribution rule (paper 2604.01609 Algorithm 2 lines 4–9):** Every expert starts at the minimum rank `k̄ · δ`, and the remaining flexible pool `b = k̄ · L · (1 − δ)` (where `L = N_experts`) is distributed by score share:

```
k_i ← floor(k̄ · δ) + floor(b · s_i / Σ_j s_j),   b = k̄ · L · (1 − δ),   δ = 0.5
```

Paper line 4 sets `k_i ← k̄ · δ` real-valued and line 9 uses bracketed `[b · s_i / Σ_j s_j]` (paper convention typically denotes rounding); the spec applies `floor(·)` to both terms because per-expert ranks must be integers. The paper §A.1 warns that δ = 0 is unsafe — without a minimum-rank floor the score-weighted reallocation can drive low-score experts to rank zero and propagate reconstruction error, so the floor `δ = 0.5` is required for the redistribution to be well-behaved. After redistribution the per-expert rank is clamped above the floor `k_i ← max(k_i, floor(k̄ · δ))` (defensive against rounding residuals; structurally the redistribution rule already enforces it).

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

Substitutes pre-prune auto-covariance for cross-covariance. The two coincide when pre/post distributions are similar (light pruning). Active when `aa_svd.cross_covariance: false` in config. This hybrid (auto-cov substituted into the AA-SVD `B⁻¹·L_B` machinery) is not a paper-recognised variant — see [D6](#12-known-deviations-from-papers) for its rationale.

**Path 3 — Corollary 3.3 fallback (B only):**

```
M = W · L_B
```

Then: `SVD(M) = U Σ V^T`, `U_k = U[:,:k] · diag(Σ[:k])`, `V_k^T = V^T[:k,:] · L_B⁻¹`. The rank-k reconstruction is `W ≈ U_k · V_k^T`. **L_B convention:** the spec uses `L_B` for the **lower-triangular** Cholesky factor satisfying `L_B · L_B^T = B` (paper Algorithm 1 line 4 writes the same factor as `R` and uses the `R^T` transpose to compute `M = W·R^T = W·L_B^T`). Spec Path 3 writes `M = W·L_B` rather than `W·L_B^T` because under A=B the singular values (and hence the rank-k truncation) are invariant to right-multiplying the chol factor or its transpose, so the single-letter form drops the transpose for legibility. The back-projection `L_B⁻¹` in the V_k^T expression is the inverse of the **same** lower-triangular factor.

**Path selection rule:** Path 1 when full cross-cov C is available; Path 2 when only auto-cov A and B (input-side) are available (gate/up — D6 hybrid); Path 3 when only B is available (down_proj fallback under Corollary 3.3 with A=B).

**Eigendecomposition caching (gate_proj ↔ up_proj):** `gate_proj` and `up_proj` share the same input covariance (post-router pre-MLP hidden state) and the same eigendecomposition is reused for both. The eigendecomposition of B (`eigh`) and the derived right-hand-side product (CQ·diag(1/√λ), AQ·diag(1/√λ), or L_B depending on the path) are precomputed once per expert via `_precompute_eigh` and cached in an `_EighDecomp` dataclass. Both `gate_proj` and `up_proj` then call `_aa_svd_precomputed`, which skips directly to `M = W @ rhs`, SVD, and back-solve. `down_proj` has its own B (intermediate-dim covariance, 512×512) and goes through the full `_aa_svd` path.

This eliminates N_experts × N_layers redundant `eigh(2048×2048)` calls (~7,200 for 180 experts × 40 layers). The optimization is **mathematically identical** — same eigendecomposition, same rhs matrix, same floating-point operations on the same inputs; the only change is that the result is computed once and reused. Estimated wall-clock reduction: ~25% on Phase C.

**Numerical safeguards:**
- Eigendecomposition replaces Cholesky (handles rank-deficient B natively)
- Dtype-aware noise floor: `bf16→1e-2`, `fp16→1e-3`, `fp32→1e-6`
- `k_eff = min(k, r_eff)` — never allocates rank beyond B's effective rank
- Zero-padding when `k_eff < k` so FactoredExperts tensors stay shape-stable
- If `_precompute_eigh` raises (e.g. all-zero B), the per-matrix loop falls back to full `_aa_svd` which itself falls back to plain SVD

#### Phase C.5: Block-Level Iterative Refinement (Paper 2604.02119, Algorithm 2, §3.3)

After all linear sub-layers (attention projections and MLP gate/up/down projections for all experts) within a single decoder block have been individually factorized via Phase C (Paths 1/2/3), AA-SVD performs a **block-level joint refinement pass** that jointly optimizes the factorized weight factors and the block's normalization parameters to minimize the block's output error against the original model. This is the central contribution of AA-SVD over standard per-layer SVD.

**Block definition:** One transformer block `ℒ_i` = one decoder layer, comprising all its linear layers (attention projections + MLP gate/up/down projections for all routed experts in that layer), all non-linear operations, and all normalization layers. Attention projections in Stage 3 are **not** factorized (§10 Protected Components); the joint objective's factorized-factor variables `{U_j, V_j}` therefore range over the MoE expert matrices only. The refinement is applied sequentially, block by block, after each block's Phase C factorization — upstream blocks remain frozen while the current block is refined. For Qwen3's architecture, each block includes the sliding-window SDPA attention and the MoE MLP; both of their RMSNorm scales are updated during refinement (see "RMSNorm scope" below).

**Objective (Section 3.3):**

```
ℓ_i = E_{X∼𝒟_i}[‖ℒ_i(X) − ℒ'_i(X')‖²]
```

where:
- `ℒ_i(X)` — the original (unfactorized) block's output on calibration activations `X` (the hidden states arriving at block `i` from the original model). **Computed on-the-fly by invoking the still-resident teacher's block `i` forward on the cumulative teacher upstream activations.** No disk cache is used.
- `ℒ'_i(X')` — the compressed block's output on shifted calibration activations `X'` (the hidden states produced by the already-refined upstream compressed blocks)

In practice this is the mean over the calibration batch at each gradient step. The calibration tensor is split into full batches of `batch_size`; any trailing partial batch (when `N_calib_seqs % batch_size ≠ 0`) is dropped to keep the cached attention-mask / position-embedding kwargs shape-stable across the AdamW loop. With the default config (256 calibration sequences × batch_size 32) this is exact (256 = 8 × 32) and no sequences are dropped.

This anchors the compressed block to the original block's output while conditioning on the **actual shifted input** produced by compression of prior layers — precisely the anchored-adaptive formulation of Theorem 3.2 extended to the block level.

**Optimization procedure (Algorithm 2, line 9; Appendix B.2):**

Minimize `ℓ_i` jointly over:
1. All factorized weight factors `{U_j, V_j}` for every linear layer `j` in block `i`
2. Block-local parameters `θ_i` — the **RMSNorm scale parameters** (and biases, if any) within block `i`

Optimizer: **standard `torch.optim.AdamW`** (fp32 moments `m`/`v`, fp32 master weights — *not* 8-bit AdamW), learning rate `1×10⁻⁴`, cosine learning rate schedule with linear warmup, **25 epochs** over the calibration data, batch size 32. All parameters in items 1 and 2 are updated simultaneously in each gradient step — this is a joint optimization, not an alternating coordinate-descent loop.

**Optimizer-precision rationale (H200-specific).** The paper specifies "AdamW" without pinning precision. On H200 (141 GB), per-block trainables are ~525M params (~1 GB BF16); fp32 AdamW state at 12 B/param is ~6.3 GB. Combined with the resident teacher (~70 GB), compressed student (~45 GB), and per-block activations + grads (~5–10 GB at batch 32, seq 2048), the fp32 optimizer fits with ~9–14 GB headroom. 8-bit AdamW (e.g. `bitsandbytes.optim.AdamW8bit`) would save ~5 GB at the cost of stochastic-rounding noise on the moments — a fidelity hit at `lr = 1×10⁻⁴` over only 25 epochs that the headroom does not require. The 8-bit variant remains available as an A100-tier escape hatch but is **not** the default and must not be used silently.

**Convergence:** Fixed epoch count of **25 epochs**. No delta-objective threshold is specified by the paper; training always runs for the full 25 epochs.

**Interaction with Paths 1/2/3:** Phase C factorization (Paths 1/2/3) provides the initialization for `{U_j, V_j}`. Phase C.5 refines these initializations via gradient descent; it does not re-invoke the Theorem 3.2 closed form. The B/C covariances computed in Phase A are used only for the Phase C initialization — Phase C.5 uses the calibration activations directly via forward passes through the (partially compressed) student model, and computes `ℒ_i(X)` by invoking the still-resident teacher's block `i` forward.

**Teacher activation stream:** At block `i`, `X_i^{teacher}` is the cumulative teacher upstream — the teacher's hidden state at block `i`'s input, produced by running the teacher from layer 0 through layer `i−1` on the calibration sequence. Both `X_i^{teacher}` and `X'_i^{student}` (input to refined student block `i`, produced by the already-refined upstream student blocks) are **constant during block i's 25 inner AdamW epochs** — neither stream is recomputed per gradient step, since upstream layers are frozen for the duration of block `i`'s refinement. The streams are advanced **once per block transition** (`X_{i+1}^{teacher} = ℒ_i(X_i^{teacher})`, `X'_{i+1} = ℒ'_i(X'_i)`) using the teacher's block forward and the refined student block respectively. Per-batch streams live in CPU RAM (~2 GB per stream × 2 = 4 GB for 256 calib seqs × 2048 × 5120 × bf16) and are swapped to GPU per gradient step. All decoder layers participate in the stream advance (including any non-MoE dense interlayers); only MoE blocks receive AdamW refinement.

**RMSNorm scope:** Only the RMSNorm layers **within** block `i` have their scale parameters updated; norms in all other blocks remain frozen. The block-local set covers (a) the two block-level norms — `input_layernorm` (pre-attention) and `post_attention_layernorm` (pre-MLP) — and (b) any per-head attention norms the architecture defines, e.g. for Qwen3 the `self_attn.q_norm` and `self_attn.k_norm` modules. The model-level `norm` and any embedding norms are out of scope. See §10 for the updated protected-component policy.

**Sequential execution (Algorithm 2 lines 2–11):** After block `i`'s Phase C.5 completes, the refined compressed block `ℒ'_i` is used to produce `X'_{i+1}` (the input to block `i+1`) via a forward pass through `ℒ'_i`. This updated `X'` feeds into block `i+1`'s Phase C factorization and Phase C.5 refinement. Blocks are processed strictly in order 0 → (N_layers − 1). Non-MoE decoder layers (dense interlayers, if the architecture has any) participate in the stream advance but skip the AdamW refinement — they have no factorized `{U_j, V_j}` to update; their RMSNorm scales remain frozen. See [D-c5-moe-only](#12-known-deviations-from-papers).

### Resume

- B-cov spill files at `_stage3_bcov_partial/layer_{idx}.pt` — layers whose spill files already exist are skipped on re-entry
- C-cov spill files at `_stage3_ccov_partial/layer_{idx}.pt`
- Phase C.5 per-block checkpoint at `_stage3_phase_c5_partial/block_{i}.pt` — saves the refined `{U_j, V_j}` and updated RMSNorm scales for each completed block; on re-entry, blocks with existing checkpoints are skipped and Phase C.5 resumes at the first un-refined block. The teacher must still be resident on resume; a re-entry from before Phase A completes triggers a full Phase A re-run, while a re-entry between Phase C.5 of block `i` and block `i+1` only re-loads the teacher (Phase A spills already on disk).
- B/C spill directories are cleaned up on successful Stage 3 completion; the Phase C.5 checkpoint directory is removed after the final block.
- Original weights snapshot (`_stage3_original_weights.pt`) is saved for Stage 4 residual computation

---

## 7. Stage 4 — EoRA Residual Compensation

**File:** [`stage4_eora.py`](src/moe_compress/stage4_eora.py)
**Paper:** EoRA: Training-Free Compensation for Compressed LLMs (2410.21271), Algorithm 1
**Hardware:** H200. One calibration forward pass to collect per-expert input activation samples `X̃_expert ∈ ℝ^{N_e × d_in}` (rows = per-token activations for tokens routed to expert e). FactoredExperts model stays resident from Stage 3; `_stage3_original_weights.pt` remains in CPU RAM.

### What

For each factored expert matrix, computes the residual `ΔW = W_original − U·V` and adds a rank-r correction that concentrates on the **most important input directions** (as measured by the eigenspectrum of the per-expert input Gram matrix `X̃^T X̃`). The correction is appended to the existing factored representation by widening U and V along the rank dimension.

**Convention:** `X̃ ∈ ℝ^{N × d_in}` is token-major (rows = per-token activation samples for tokens routed to this expert during one calibration pass). `A = X̃^T X̃ ∈ ℝ^{d_in × d_in}` is the (un-normalized) input Gram matrix; its rank is at most `min(N, d_in)` and in practice is bounded by the number of routed tokens collected for the expert. Paper §3 (line 183) states "the average of the input activations over the calibration set"; this average is by construction rank-1 (a single d_in-vector). The spec deliberately reinterprets X̃ as the multi-sample stack ∈ ℝ^{N_e × d_in} so that A = X̃^T X̃ has rank ≤ min(N_e, d_in) (D10 deviation), restoring meaning to rank-128 corrections matching the paper's experimental results.

Note on notation: §7 Step 7's "B·A" uses the paper's EoRA Algorithm 1 step 7 convention where A is the EoRA correction-factor matrix V_corr (see Step 5), distinct from the Gram matrix A = X̃^T X̃ defined above. Both follow paper notation; rename to G for the Gram in mental models if needed.

### Why

EoRA recovers quality lost to rank truncation in Stage 3. The paper reports +10.84pp ARC-C on LLaMA3-8B (GPTQ-3-bit-class quantization residuals; Stage 3 SVD residuals at moderate ρ are several orders of magnitude smaller — uplift not directly portable to our BF16 pipeline; cited only as magnitude context for the method's ceiling on heavily-corrupted weights). The key innovation over naive SVD of the residual is the √Λ-weighted eigenspace projection, which concentrates the correction rank budget on directions the model actually uses.

### How — Paper Algorithm 1

For each (layer, expert, matrix):

1. **Residual:** `ΔW = W_orig − U_old · V_old` — shape `[d_out × d_in]`

2. **Build input Gram matrix and eigendecompose:** For each (layer, expert), collect the per-token activation samples `X̃_expert ∈ ℝ^{N_e × d_in}` for tokens routed to expert `e` during the calibration pass. Form the input Gram matrix `A = X̃_expert^T X̃_expert`, shape `[d_in × d_in]`, with rank up to `min(N_e, d_in)`. Eigendecompose: `A = Q Λ Q^T`. Sort eigenvalues in descending order and keep `n_keep = |{j : λ_j > τ_floor}|` eigenpairs, where `τ_floor` is a dtype-aware noise-floor threshold (relative to `λ_1`). Under typical calibration volumes (`N_e ≫ 128`), `n_keep` is bounded by the noise floor and routing volume; the rank cap then clamps `take_eff` (Step 5) below `n_keep` rather than vice-versa. Small-eigenvalue directions discarded by the floor are noise-dominated activation modes. (See D10 / D-S-H-1: under the multi-sample reading the importance signal is the full eigenspectrum of `X̃^T X̃`, weighting each eigendirection by its activation energy `λ_j`.)

3. **√Λ-scaled projection:** `Q' = Q_keep · √Λ_keep` — shape `[d_in × n_keep]`. This is the **full** signal eigenspace, NOT truncated to `r`. The √Λ scaling importance-weights each direction by its activation variance.

4. **Full projection:** `ΔW' = ΔW · Q'` — shape `[d_out × n_keep]`

5. **Rank-r SVD:** `SVD(ΔW') → U', Σ', V'^T`. Take top `take_eff = min(r, min(d_out, n_keep))`, where `r = min(rank_budget, eigenspace_rank_cap)` is the per-matrix EoRA rank from the budget step. Under the multi-sample reading the rank cap is operative: `n_keep` typically exceeds 128, so `take_eff` is set by `r` (and ultimately by `eigenspace_rank_cap`) rather than collapsing to 1.

6. **Correction factors:**
   - `U_corr = U'[:, :take_eff] · Σ'[:take_eff]` — shape `[d_out × take_eff]`
   - `V_corr = V'^T[:take_eff] · (√Λ_keep)⁻¹ · Q_keep^T` — shape `[take_eff × d_in]` (back-projected to original weight space)

7. **Widen:** `new_U = [U_old | U_corr]`, `new_V = [V_old; V_corr]` — algebraically equivalent to `Ŵ·x + B·A·x` (paper Algorithm 1 step 7; Eq. 4 of the paper consolidates `B·A` into a single `B′`), where `B = U_corr` and `A = V_corr` are the EoRA correction factors appended to the existing factorization.

### Budget

`compensation_budget_pct=3%` of Stage 3 per-matrix parameter savings (project-chosen: 3% empirically selected to keep Stage 4's parameter footprint small relative to Stage 3 savings; **not from paper** — see D-eora-budget-pct), capped at `eigenspace_rank_cap=128` rank per expert. Rank 128 is a common reporting rank in the EoRA paper (Tables 2–3) and lies within the paper's evaluated range {64, 128, 256, 512}; it is a project choice from the paper's range, not a unique "paper default".

### Correctness Notes

- The √Λ scaling is the **core** innovation of EoRA. Without it, the algorithm degenerates toward ZeroQuant-V2 (plain SVD on ΔW with no activation weighting) — Act-S is a separate method that uses per-channel L1-magnitude diagonal scaling, unrelated to eigenvector projection.
- Pre-truncating to `r` eigenvectors before SVD (the previous bug) eliminates the joint optimization that makes EoRA better than Act-S. The SVD must operate on the full `[d_out × n_keep]` projected error to optimally select the best `r` directions.
- The back-projection `V_corr = (V'^T · (√Λ)⁻¹) · Q^T` is critical — without the `(√Λ)⁻¹` term, the correction adapter operates in eigenspace rather than weight space, and the errors compound.

### Resume

Per-layer atomic checkpointing to `_stage4_partial/layer_{layer_idx}.pt` (format_version=1). Each checkpoint contains the full FactoredExperts U/V state, ranks, effective ranks, and parameter counts. On resume, completed layers are loaded directly; failed layers re-run from the Stage 3 output.

Stage 4 is the final consumer of both `_stage3_original_weights.pt` (Step 1, residual) and `_stage2_input_covariance.pt` (Step 2, input Gram for the √Λ projection); both are deleted on Stage 4 success. They remain durable on their per-stage Hub repos (`<base>-stage2`, `<base>-stage3`); on-bucket retention only inflates the entrypoint's job-exit aux upload to the aggregate result repo. On Stage 4 failure both are kept so a re-run can pick up cleanly.

---

## 8. Stage 5 — Router Knowledge Distillation (Final)

**File:** [`stage5_router_kd.py`](src/moe_compress/stage5_router_kd.py)
**Paper:** Router Knowledge Distillation for MoE Compression (2603.02217), Eq. 3, Table 1, §F.3
**Hardware:** H200. EoRA-compensated student model stays resident from Stage 4. Teacher loads in BF16 (~70 GB); combined VRAM ~70 GB teacher + ~50 GB student + logits/activation buffer ~6 GB → 126 GB total (with bs=8 logits).

### What

Trains **only** the router gate weights to match the original (uncompressed) teacher's output distribution via vocabulary-level KL divergence. Only `mlp.gate.weight` is trainable; all expert, backbone (attention, embeddings, lm_head, RMSNorm) parameters remain frozen, per Router-KD paper §4. Gradients flow through the soft-gating weights of the selected experts during the forward pass (so the router's gate signal reaches the loss); only `mlp.gate.weight` parameter values are updated.

### Why

After Stages 2–4, the expert weights have changed but the router weights still reflect the original expert set. The router sends tokens to suboptimal experts, degrading quality. Router KD recalibrates routing decisions to the new expert landscape by matching the teacher's next-token prediction distribution — the gradient signal flows backward through the routing decisions, naturally adapting them.

The paper explicitly uses vocabulary-level output distillation (not router-gate-level): "By distilling output logits rather than matching router gate values explicitly, Router KD avoids requiring the teacher and student to share identical expert sets or gate dimensionalities."

### How — Vocabulary-Level KD (Paper Eq. 3)

```
L_RKD = (τ² / N_x) × Σ_t m_{t+1} · KL(softmax(z_T^t / τ) ‖ softmax(z_S^t / τ))
```

where `z_T, z_S ∈ ℝ^{|V|}` are teacher/student vocabulary logits, `m_{t+1} ∈ {0,1}` is the padding mask, `N_x = Σ_t m_{t+1}` is the count of unmasked positions, and `τ=1.0` is the temperature. Calibration sequences are fully packed so `m_{t+1}=1` everywhere in practice. *Paper Eq. 3 includes a `+ ε` zero-mask safety constant; the spec drops it under the fully-packed-sequences invariant (no padding → m=1 everywhere → N_x ≥ 1 deterministically). If a future calibration source ever introduces padding, the `+ ε` must be restored.*

**Implementation:**
1. Teacher forward pass (no_grad) → vocabulary logits `[B, L, |V|]`
2. Student forward pass (with gradients) → vocabulary logits `[B, L, |V|]`
3. Shift logits: position `t` predicts token `t+1` (standard causal LM)
4. Chunked KL: process `chunk_size` sequence positions at a time to bound peak memory at `B × chunk × |V| × 4` bytes. On H200 with ~15 GB VRAM headroom at bs=8, `chunk_size=512` (the full sequence length) is safe — peak intermediate is ~2.4 GB. Chunking is retained as a configurable parameter (`kd_seq_chunk_size`) for smaller-VRAM hardware; on H200 the overhead of chunk-boundary Python loops is eliminated by setting chunk=seq_len.
5. `F.kl_div(log_softmax(student/τ), softmax(teacher/τ))` = KL(teacher ‖ student) — correct forward KL direction (here "forward KL" = teacher-as-reference, matching paper Eq. 3's `D_KL(p_T ‖ p_S)` convention). *Proof:* `F.kl_div(input=log_q, target=p)` computes `Σ p · (log p − log q) = KL(p ‖ q)`. With input=`log_softmax(student/τ)` and target=`softmax(teacher/τ)`, this yields KL(teacher ‖ student) = paper Eq. 3. Substituting `p = softmax(z_T/τ)` and `q = softmax(z_S/τ)` gives `Σ softmax(z_T/τ) · (log_softmax(z_T/τ) − log_softmax(z_S/τ)) = KL(softmax(z_T/τ) ‖ softmax(z_S/τ))` = paper Eq. 3 form.

**`torch.compile` acceleration (Stages 2.5 and 5):** When `torch_compile: true` in the stage config, both teacher and student models are compiled via `torch.compile(model, mode="reduce-overhead")` before the KD training loop. On H200 (Hopper architecture), this enables kernel fusion and reduced launch overhead across the MoE dispatch + expert matmul sequence. Expected speedup: 20–40% on forward pass throughput after a one-time ~2–5 min compilation cost. Compilation is skipped when the model uses custom `instrument_experts` hooks (Stage 2's profiling pass), since the monkey-patched forward breaks torch.compile's graph tracing. Quality impact: zero — `torch.compile` produces numerically identical outputs in default mode.

### Hyperparameters (Paper Table 1, §F.3)

| Parameter | Value | Source |
|-----------|-------|--------|
| Optimizer | AdamW (`weight_decay=0.0`) | Adapted (paper unspecified — paper §F.3 Table 1 lists no optimizer; AdamW is the project default. `weight_decay=0.0` overrides PyTorch's 0.01 default to avoid regularizing the small router-gate matrix toward zero, since only `mlp.gate.weight` is trainable and there are no other parameters absorbing the decay term.) |
| Learning rate | **5×10⁻⁵** | Paper Table 1 (implementation previously used 1e-5; corrected to match paper) |
| Epochs | 1 | Paper |
| Batch size | 8 | Adapted (paper: 2 with grad-accum=4 → effective 8; spec uses 8 with grad-accum=1, mathematically equivalent). (Loss is per-sequence-normalized by `N_x` per Eq. 3, so microbatch grouping does not rescale individual sequence contributions.) |
| Gradient accumulation | 1 | Adapted (paper: 4) |
| **Effective batch size** | **8** | Same as paper (8×1 = 2×4). **Equivalence requires the accumulation reduction policy be `sum`-then-divide-once-by-8 (not per-microbatch `mean` averaged across 4 steps).** Concretely: each microbatch loss is the per-sequence τ²/N_x-normalized KL summed over its sequences; under bs=2/accum=4, the four microbatch summed losses are accumulated by `loss.backward()` (which adds gradients in PyTorch) and the optimizer step divides the accumulated gradient by 8 (= total sequences). Under bs=8/accum=1, the eight summed losses are produced in one microbatch and divided by 8 once. Both produce the same gradient. **If the accumulation uses per-microbatch `mean` averaging, gradients differ by a factor of accum_steps and the equivalence breaks** — implementations must pin sum-style accumulation. |
| Max sequence length | 512 | Paper |
| KD temperature (τ) | 1.0 | Paper |
| Max calibration samples | 3000 | Paper |
| Teacher precision | BF16 | H200 (141 GB VRAM) fits teacher+student in BF16 with ~15 GB headroom at bs=8. Student is always BF16. |
| KL sequence chunk size | 512 | H200 (full sequence in one chunk — ~2.4 GB peak). Configurable via `kd_seq_chunk_size` for smaller hardware. |
| torch.compile | true | H200 Hopper. 20–40% forward pass speedup. Set false for debugging or hardware without compile support. |

### Teacher Loading

**Calibration data:** Nemotron weighted subsets per §2 (D11) — canonical reference is §2; same calibration source as Stages 2 and 2.5.

On H200 (141 GB VRAM), both teacher and student load in BF16 (~70 GB + ~50 GB = ~120 GB). At bs=8 with seq_len=512, vocabulary logits + small KV-cache headroom consume ~6 GB, leaving ~15 GB headroom. Alternatively, precomputed teacher vocabulary logits can be loaded from a cache file (`teacher_logits_cache` config key) to skip the live teacher entirely.

**4-bit teacher fallback (`teacher_load_in_4bit: true`):** loads the teacher via bitsandbytes NF4 quantization (~17 GB live VRAM vs ~70 GB BF16), `compute_dtype=bfloat16`. Use when host VRAM is insufficient for full BF16 teacher (e.g., A100-40G nodes). Marginal logits drift expected (NF4 quantization noise on the teacher logits is small in spot checks); acceptable when accompanied by per-batch teacher logits caching during Stage 5 to avoid running 4-bit forward repeatedly. Mutually exclusive with `teacher_logits_cache` — if both are configured, the cache wins.

Frozen-scope reminder: only `mlp.gate.weight` is trainable at Stage 2.5/5. No Phase C.5-style RMSNorm carve-out applies at Stage 2.5/5; that exception is scoped to Stage 3 only, per §10. Gradients flow through the soft-gating weights of the selected experts during the forward pass (so the router's gate signal reaches the loss); only `mlp.gate.weight` parameter values are updated.

### Resume

Step-boundary checkpointing to `_stage5_partial/step_{N}.pt` (every 100 optimizer steps). Each checkpoint contains router parameter state + optimizer state + the (epoch, batch_idx) cursor. On resume, the optimizer state is restored exactly; the training loop fast-forwards through `resume_batch_i` inclusive (the gradient signal of any batch that was already absorbed into the last checkpointed optimizer step is correctly skipped, not re-applied). (The (epoch, batch_idx) cursor is captured AFTER the optimizer.step() that consumed those batches, so fast-forward through resume_batch_i inclusive cannot drop or double-count any sequence.) Only the two most recent checkpoints are retained.

---

## 9. Stage 6 — Validation

**File:** [`stage6_validate.py`](src/moe_compress/stage6_validate.py)
**Hardware:** Runs on the same H200 instance as Stage 5 (student model stays resident).

### What

Evaluates the compressed model against the uncompressed teacher on 5 metrics, enforces hard quality gates, and produces an imatrix file for downstream GGUF quantization. On H200 the larger VRAM headroom allows larger batch sizes for all evaluation phases.

### Metrics

| Metric | Method | Threshold |
|--------|--------|-----------|
| WikiText-2 PPL | Per "WikiText-2 PPL Protocol" below (F-S-C-1: 2048-token non-overlapping chunks, drop last partial, micro-averaged shifted-position NLL) | ≤ +3% relative |
| ARC-C accuracy | lm-eval harness, 0-shot | ≤ 1.5pp absolute drop |
| HellaSwag accuracy | lm-eval harness, 0-shot | ≤ 1.5pp absolute drop |
| HumanEval pass@1 | Greedy pass@1 (do_sample=False, n=1) — NOT Chen et al. 2021 stochastic pass@1 (n=10, T=0.2, top_p=0.95). See [D-humaneval-greedy](#12-known-deviations-from-papers). Exec-based scoring (**in-process**, NOT Chen et al.'s subprocess sandbox — see [D-humaneval-greedy] for the in-process safety/correctness disclosure). | ≤ 3pp absolute drop |
| MATH-500 accuracy | `HuggingFaceH4/MATH-500` (revision pinned in run config under `dataset_revisions`); in-tree grader (`stage6_validate.py:_check_math`): `\boxed{}` extraction first; if missing, `_parse_latex` fallback; final SymPy `simplify(comp − ref) == 0` check with numeric (`_last_numeric`) tie-break. Grader is project-original (NOT the Hendrycks et al. 2021 `math_equivalence.py`); absolute MATH-500 numbers may differ from published Hendrycks-grader baselines but are consistent across teacher/student under the same project grader, sufficient for the relative-to-teacher gate | ≤ 3pp absolute drop |

#### WikiText-2 PPL Protocol (F-S-C-1)

This protocol is the canonical Stage 6 PPL definition; §6 Phase B.2's α-search PPL evaluation (line ~503) is required to use the **identical** protocol so that Stage 3 selection and Stage 6 gating are consistent. (§6 Phase D α-search must use this identical protocol — the back-reference in §6 line ~503 is added in Track C iter-3.)

- **Corpus.** `wikitext-2-raw-v1`, `test` split (HuggingFace dataset id `Salesforce/wikitext`, name `wikitext-2-raw-v1`). The exact dataset revision (commit sha) is recorded in `dataset_revisions` and folded into the teacher cache key.
- **Concatenation.** All test rows are concatenated into a single token stream. **BOS policy:** apply the tokenizer's default `add_special_tokens=True` once on the concatenated text (one BOS at stream head if the tokenizer adds one; otherwise no BOS — both are spec-compliant; the chosen policy is "delegate to tokenizer default" so the same tokenizer config produces identical chunking across runs). For Qwen3-30B-A3B, the default tokenizer does not add BOS, so the stream begins at the first wikitext token. **Cross-model portability:** if a future target model's tokenizer adds a BOS by default, the policy still applies — the BOS counts as one token in the first chunk; the chunking (non-overlapping 2048-token blocks, drop last partial) is unchanged. No per-row BOS injection. Row-to-row separator: the **two-newline (`"\n\n"`) join** is the project convention, matching the canonical HF `evaluate` / `lm-eval` WikiText-2 PPL recipe; the same join is also used by the imatrix calibration corpus build (see "imatrix Generation" below) so the PPL eval and imatrix activation statistics see comparable token distributions. (F-iter4-M-1.)
- **Chunking.** Non-overlapping fixed-length chunks of **2048 tokens**. Stride = chunk_len (no overlap, no sliding window). The seq_len choice matches Swift-SVD's 2048-token calibration-sample length (Appendix A.2 of 2604.01609); the project re-uses this length for non-overlapping evaluation chunks. The drop-last-partial and concatenation-with-BOS conventions are project-chosen (community-convention HuggingFace `evaluate` / `lm-eval` PPL recipes).
- **Last-chunk policy.** The incomplete final chunk is **dropped** (no padding, no shorter-chunk inclusion).
- **NLL aggregation.** Micro-average over all shifted token positions across all retained chunks:
  ```
  PPL = exp( Σ_{chunks} Σ_t NLL_t  /  Σ_{chunks} (chunk_len − 1) )
  ```
  Each retained chunk contributes exactly `chunk_len − 1 = 2047` shifted positions; the denominator is therefore `(num_chunks_retained) × 2047`.

### Measured Reduction

The actual parameter reduction is computed from live parameter counts (accounting for effective ranks in FactoredExperts). Must be ≥ 30.0%.

**Definition (per F-S-M-3).**
```
Measured Reduction = 1 − live_param_count(student) / live_param_count(teacher)
```
where `live_param_count(model)` includes **all** model parameters: token embeddings, attention projections (DeltaNet linear attention + full attention), MoE expert weights (routed experts, including FactoredExperts U/V factors counted at their per-expert effective ranks; routers; shared experts), all RMSNorm scale parameters (must be registered as `nn.Parameter`, not buffers — true for Qwen3-30B-A3B and the target model), and `lm_head`. Excludes optimizer state, KV cache, and activation buffers (these are not model parameters). Both numerator and denominator are computed via the same iteration over `model.parameters()` with `requires_grad`-agnostic counting; if a future architecture registers any of the listed components as a buffer, the iteration must be extended to `chain(model.parameters(), filtered_buffers)` to maintain ratio symmetry.

### Execution Model (Compute-Time Optimized)

Stage 6 runs the following phases. All optimizations are purely computational scheduling — larger batches, cached known-constants, overlapped I/O, and torch.compile. **No metric, formula, threshold, or evaluation methodology is changed.** Outputs are numerically equivalent to the batch_size=1 baseline up to torch.compile/CUDA-graph kernel-selection variance (optimizations #1–#4 are bit-identical under the protocol prerequisites listed below; #5 `torch.compile(mode="reduce-overhead")` introduces non-bit-level run-to-run variance that does not change algorithmic correctness — see #5 row in the Spec Compliance table for the precise scope).

#### Phase 1: Student Evaluation

| Sub-phase | Optimization | Config Key | Default |
|-----------|-------------|------------|---------|
| WikiText-2 PPL | **#1** — batch_size=1→8 | `ppl_batch_size` | 8 |
| ARC-C + HellaSwag | **#2** — lm-eval batch_size=1→auto:8 | `lm_eval_batch_size` | `"auto:8"` |
| HumanEval (164 prompts) | **#3** — batched model.generate() | `gen_batch_size` | 8 |
| MATH-500 (500 prompts) | **#4** — batched model.generate() | `gen_batch_size` | 8 |

**Pinned generation configuration for #3 / #4 (binding requirements for the bs=1 / batched argmax-identity claim per F-S-H-2):**
- `tokenizer.padding_side = "left"` (left-padding is mandatory; right-padding would break the KV-cache identity)
- `attention_mask` zeroed at pad positions (mandatory; keeps attention output unaffected by padded tokens)
- `tokenizer.pad_token_id` is set explicitly (HumanEval/MATH-500 prompts must have a deterministic pad token)
- `attn_implementation = "eager"` (forced by Stage 6 regardless of upstream config; see F-S-M-1)
- `do_sample = False`, `n = 1` (greedy single-sample; see D-humaneval-greedy)
- KV-cache reuse enabled (`use_cache=True`, the HF default)

**torch.compile** (**#5**, `torch_compile: true`): Before any evaluation begins, `model.forward` is compiled via `torch.compile(model.forward, dynamic=True, mode="reduce-overhead")`. `dynamic=True` handles variable-length padded batches from lm-eval. One-time compilation cost (~3–5 min on H200) is amortized across 1000+ forward passes. **Only** the forward pass is compiled — `model.generate()` is **not** wrapped with `torch.compile`, but each decoding step internally calls the compiled `model.forward`, so generative evals (HumanEval, MATH-500) DO benefit from the compiled forward; only the generate-loop control flow itself runs eagerly. Wrapping `model.generate()` directly was avoided because the prefill-vs-decode shape transition can still trigger one extra recompile; the per-step forward path is the dominant cost and is fully captured.

#### Phase 2 (Background): Teacher I/O Overlap — concurrent with Phase 1 (#6)

The teacher preload begins as early as Phase 1 entry (host-RAM-only, GPU-independent — no contention with student evals). In the most conservative scheduling, it begins during Phase 1's generative evals (after the zero-shot harness completes), but starting earlier (during PPL or zero-shot) is also safe and may better hide the load: a background thread loads the teacher model to **host RAM** (device_map="cpu") while the GPU runs HumanEval and MATH-500. When Phase 1 completes, the student is moved to CPU, and the pre-loaded teacher is moved to GPU — eliminating the ~3–5 min dead time that a blocking teacher load would cause.

#### Phase 3: Teacher Evaluation (or Cache Hit)

**Teacher eval caching** (**#7**, `teacher_eval_cache.enabled: true`): The teacher (uncompressed Qwen3-30B-A3B) is a fixed, known model. Every Stage 6 run re-evaluates the same teacher on the same benchmarks with the same results. When caching is enabled:

- **First run:** Teacher is evaluated normally (PPL + zero-shot + generative). Results and param counts are saved to `teacher_eval_cache.json` with a cache key composed of the following inputs (each component is either an HF revision-sha or a `pkg.__version__` string), joined and hashed via SHA-256 (per F-S-H-3):
  ```
  sha256(model_name + "\x1f" + revision + "\x1f" + tokenizer_revision
       + "\x1f" + dataset_revisions + "\x1f" + lm_eval_version
       + "\x1f" + transformers_version + "\x1f" + dtype
       + "\x1f" + attn_impl + "\x1f" + eval_config_subset)
  ```
  Implementation: components are assembled into a Python dict with the listed keys, then serialized via `json.dumps(payload, sort_keys=True, separators=(",",":"))` and SHA-256 hashed. The pseudocode `+ "\x1f" +` notation above is illustrative of the deterministic-byte-representation goal; the canonical-JSON serialization with sorted keys provides equivalent reproducibility (and is what the implementation uses). `eval_config_subset` is the run config's `wikitext2`, `zero_shot`, `generative` subdicts as-is (operators must avoid putting non-numerical-affecting keys like file paths or log levels inside those three blocks; the cache will invalidate if such a key is changed, which is conservative but harmless). `dataset_revisions` is the JSON-canonicalized mapping `{"wikitext_ppl": "<sha>", "humaneval": "<sha>", "math500": "<sha>"}` serialized with sorted keys before concatenation. **F-iter4-HIGH-1 limitation:** the lm-eval-managed datasets (HellaSwag, ARC-Challenge) are intentionally **not** pinned in this mapping — `lm-eval` resolves their revisions internally and our `simple_evaluate(...)` call cannot enforce a SHA at load time. The cache key compensates by folding in `lm_eval_version` and a SHA-256 of the lm-eval task config (task list + lm_eval_batch_size); precise per-dataset SHA control for those tasks requires editing lm-eval task YAMLs out-of-band.
  The cache file is atomically written only after ALL configured teacher metrics have been computed; partial caches are never persisted. See §11 for the full atomic-write contract.
- **Subsequent runs:** Cached teacher results are loaded directly. No teacher model load, no teacher evaluation. This eliminates ~50% of total Stage 6 wall-clock time.
- **Auto-invalidation:** If any cache-key component changes (model/tokenizer/dataset revision, lm-eval or transformers package version, dtype, attention implementation, or any parameter in the recorded eval-config subset), the cache key mismatches and the teacher is re-evaluated.

When the cache misses, teacher evaluation uses the same batch sizes and torch.compile as student evaluation. To remove cross-batch kernel variance, the gate run pins `attn_implementation='eager'` (per F-S-M-1; see "Spec Compliance of Optimizations" below).

#### Phase 4: GGUF Conversion Overlap (#8)

When the teacher is being evaluated on GPU, the GGUF conversion (`convert_hf_to_gguf.py`) runs simultaneously in a **background CPU thread**. This is safe because GGUF conversion reads from the saved Stage 5 checkpoint on disk (CPU-only, ~5–10 min) — the checkpoint is durable per §11's atomic-write contract before this thread starts — teacher evaluation runs on GPU, and CPU and GPU work are fully independent. When teacher eval finishes and the teacher is freed, the F16 GGUF is ready — `llama-imatrix` can start immediately. **On a teacher-cache HIT** (no teacher evaluation runs), there is no GPU work to overlap GGUF conversion with after Phase 3, so the GGUF conversion runs **sequentially** as part of Phase 4/5 (`_generate_imatrix` calls `convert_hf_to_gguf.py` then `llama-imatrix` in order on the main thread). The cache-hit path therefore loses the ~5–10 min overlap benefit but still completes Phase 4+5 in a single sequential pass after student evals.

**GGUF dtype path (F-S-L-3).** `convert_hf_to_gguf.py` reads BF16 / F32 weights from the HF checkpoint and writes an **F16 GGUF** (`model_f16.gguf`) as the conversion target — F16 is the source dtype for imatrix-guided quantization downstream; no quantization happens at this step.

#### Phase 5: imatrix Generation

After the teacher is freed from VRAM, `llama-imatrix` runs on the F16 GGUF (pre-built in Phase 4) with a **benchmark-independent calibration corpus** (per F-S-H-4 option (a); see "imatrix Generation" below).

### Spec Compliance of Optimizations

| Optimization | Why Numerically Identical |
|---|---|
| #1 PPL batch_size=8 | Numerically identical to bs=1 **under the F-S-C-1 protocol** (concatenated 2048-token chunks, drop last partial). The identity `out.loss × (chunk_len − 1) × num_chunks_in_batch` recovers the exact summed NLL because every chunk has the same shifted-position count and there is no padding. With ragged batches or padding, the identity does not hold. (per F-S-H-1) |
| #2 lm-eval batch_size=auto:8 | Task-equivalent within harness tolerance (lm-eval `auto` adapts to GPU memory; sdpa kernel may produce non-bitwise outputs across batch sizes). The Stage 6 gate run pins `attn_implementation='eager'` to remove this source of variance; under eager attention plus left-padding with a causal attention mask, loglikelihood scoring is batch-size-independent. (per F-S-M-1) |
| #3, #4 Batched generate | **Greedy bs=1 vs batched: argmax-identical when left-padded with attention-mask zeros, KV-cache reuse, AND `attn_implementation="eager"` (the same eager-pin used for #2) — true under do_sample=False.** Under SDPA/flash, batched-vs-bs=1 logits can drift by ~1e-5–1e-4 and flip argmax on near-tied tokens, breaking the claimed identity; eager attention removes this. This identity would NOT hold under stochastic sampling. (per F-S-H-2; eager pin per F-S-M-1) |
| #5 torch.compile | No algorithmic approximation in default/reduce-overhead modes. Bit-level equivalence is **not guaranteed across runs** in `reduce-overhead` mode because CUDA-graph capture is order-sensitive and selected kernels can vary. The spec already documents torch_compile as valid in Stage 5. (per F-S-L-2) |
| #6 Teacher I/O overlap | Computation unchanged — only I/O scheduling differs |
| #7 Teacher eval cache | Teacher is deterministic — same model + same eval = same numbers. Cache key composition is fully specified above (Phase 3) and includes model/tokenizer/dataset revisions, lm-eval and transformers versions, dtype, attn_impl, and eval-config subset. |
| #8 GGUF overlap | GGUF conversion reads from saved checkpoint, independent of GPU evaluation |

### vLLM Note

vLLM is **NOT viable** for this model. The compressed model uses a custom `FactoredExperts` nn.Module that replaces the standard `Qwen3_5MoeExperts`. vLLM requires its own model-specific forward implementation and weight loader — it cannot load arbitrary custom nn.Module subclasses. The weight names (`gate_proj_U`, `gate_proj_V`, etc.) don't match what vLLM's Qwen3MoE loader expects. Stick with HuggingFace `model.forward()` and `model.generate()`.

### imatrix Generation

**Calibration corpus (F-S-H-4, option (a) — community convention).** Stage 6 uses a **benchmark-independent calibration corpus** for `llama-imatrix`: the **WikiText-2 `train` split** (`wikitext-2-raw-v1`, dataset id `Salesforce/wikitext`, name `wikitext-2-raw-v1`). Eval-text reuse was rejected to match community convention and to avoid biasing imatrix statistics toward the eval distribution. After the teacher is freed, the final frozen model is converted to F16 GGUF and `llama-imatrix` runs on the wiki.train calibration text, producing `imatrix.gguf`. The concatenation of eval-text is still emitted to `eval_text_concat.txt` as a debugging side-channel (it is **not** the imatrix calibration input).

**Artifacts:**

| File | Description |
|------|-------------|
| `stage6_eval.json` | Quality gate results (metrics + pass/fail) |
| `teacher_eval_cache.json` | Cached teacher eval results + param counts (when caching enabled). Atomic-write contract per §11. |
| `calibration_wiki_train.txt` | wiki.train calibration corpus actually fed to `llama-imatrix` (the input that produced `imatrix.gguf`). |
| `eval_text_concat.txt` | Concatenated eval text from all benchmarks (debugging side-channel only; **not** the imatrix calibration input). |
| `model_f16.gguf` | Intermediate F16 GGUF of the compressed student (per F-S-L-3 path) |
| `imatrix.gguf` | Final importance matrix for GGUF quantization |

llama.cpp is built in the background by the job entrypoint (daemon thread, starts when Stage 1 begins) so the ~5-minute build does not add to wall-clock time. If llama.cpp is unavailable, `eval_text_concat.txt` is still written and Stage 6 passes normally.

### Expected Wall-Clock Impact

| Scenario | Student Evals (incl. compile) | Teacher Evals | imatrix | Total |
|---|---|---|---|---|
| Baseline (batch_size=1, no cache) | ~90–150 min | ~90–150 min | ~20 min | ~200–320 min |
| After P0 (#7 cache, #2 lm-eval) | ~30–50 min | 0 min (cached) | ~20 min | ~50–70 min |
| After P0 + P1 (#1 PPL, #3/#4 gen) | ~10–20 min | 0 min | ~20 min | ~30–40 min |
| After all optimizations | ~11–20 min (incl. ~3–5 min torch.compile) | 0 min | ~15 min (overlapped) | ~25–30 min |

Per F-S-N-2: the one-time torch.compile cost (~3–5 min on H200) is folded into the "Student Evals" line for honesty rather than reported as a separately amortized cost.

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
2. `fsync(<path>.tmp)` — flush file data to storage
3. `os.replace(<path>.tmp, <path>)` — atomic rename on POSIX
4. `fsync(parent_dir)` — make the new directory entry durable (survives power loss after rename)

This sequence survives SIGKILL, training-framework timeout, kernel panic, and power loss. A crash at any point before step 3 completes leaves at most a `.tmp` file, never a truncated or partially-visible final file. A crash between steps 3 and 4 may lose the rename on some POSIX filesystems, but step 4 is a best-effort durability seal — the file is never corrupted. Dangling `.tmp` files are cleaned up at stage startup.

**`--no-resume` flag:** When passed to `run_pipeline.py`, disables all within-stage resume behaviour. Each stage runs unconditionally from scratch with no partial-file I/O. Stage 1 and Stage 6 are unaffected (they have no resume files).

| Stage | Resume Mechanism | Granularity | `--no-resume` Effect |
|-------|-----------------|-------------|---------------------|
| 1 | None (stateless, ~5 min) | N/A | None — JSONs are outputs, not resume files |
| 2 | `_stage2_partial/merge_{i}.json` + `layer_{i}.pt` | Per MoE layer | Skip all partial I/O |
| 3 | `_stage3_bcov_partial/`, `_stage3_ccov_partial/` spills; `_stage3_alpha_result.json` | Per covariance phase + α search | Delete existing spills; skip α cache |
| 4 | `_stage4_partial/layer_{i}.pt` | Per MoE layer | Skip all partial I/O |
| 5 | `_stage5_partial/step_{N}.pt` (rolling window of 2) | Per optimizer step (every 100 steps) | Skip all checkpoint I/O |
| 6 | None (stateless by design) | N/A | None — teacher_eval_cache is a speedup cache, not resume |

**`teacher_eval_cache.json` atomic-write contract (F-S-M-2).** Although Stage 6 has no resume files, the teacher eval cache is durability-sensitive. Writes use the same atomic sequence (`tempfile + fsync + os.replace + parent fsync`) as the within-stage partial files defined above. **The cache file is written only after ALL configured teacher metrics (PPL + each lm-eval task + each generative task) have been computed; partial caches are never persisted.** A crash at any point during teacher evaluation leaves either no cache file (forcing a fresh teacher run on the next entry) or the previous run's intact cache file (unchanged), but never a partially populated cache. This makes the cache invariant: any `teacher_eval_cache.json` on disk is complete with respect to the recorded cache key.

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
<!-- D-SE-A — CONSUMED by stage1/plugins/three_way_and.py (commit pending). -->

| D3 | 1 | γ entropy tolerance | 2604.06542 Eq. 10: γ∈[0,1], no default given | γ=0.1 (project-chosen, not from paper) | Paper leaves γ unspecified; 0.1 chosen empirically |
| D4 | 1 | D^l update after merge | 2604.06542 Algorithm 1 lines 9–10: zero only pair entry D_{i*,j*} and D_{j*,i*} (line 9); update R^l ← R^l − 2·D_{i*,j*} (line 10) | Zeros the absorbed expert's entire row and column in D^l; recomputes R^l from updated matrix | Prevents the absorbed expert from influencing future pair-selection; the paper's update assumes the merged expert cannot be re-selected, but only zeroing the pair entry leaves stale similarity values that can distort R^l and layer selection in subsequent iterations |
| D5 | 1 | Floor without layer bonuses | GRAPE has no floor constraints | min_experts_per_layer = num_routed_experts // 2 (=128); no early/late layer bonuses; floor enforced during greedy by skipping any layer at-or-below floor when picking argmax/argmin (computed dynamically as num_routed_experts // 2 per model (=128 for 256-expert Qwen3-30B-A3B; would be 64 for 128-expert layers, etc.)) | 50% max removal per layer bounds the compression within the range where papers demonstrate results; bonuses removed — the floor alone is sufficient. Greedy-time enforcement (skipping floor-saturated layers in the argmin step) prevents the layer from being selected after it has reached its protected size. |
| D-cka-distance | 1 | CKA distance vs paper similarity; argmin vs argmax | 2604.06542 line 245 defines `D^l` as a CKA *similarity* matrix; Algorithm 1 lines 6–7 select pairs with `argmax D^l` and pick the most-redundant layer with `argmax R^l` (the sum of off-diagonal similarities) | `D^l_{ij} = 1 − CKA(f_i, f_j)` (distance form, 0 = identical, 1 = maximally different); the greedy uses `argmin D^l` and `argmin R^l` on this distance matrix | Distance form chosen so the redundancy criterion `R^l = Σ_{i≠j} D^l_{ij}` reads as "higher = more redundant by Σ-of-distances" while still producing the same expert pair under argmin. The transformation is a sign-flip relative to the paper: `argmin distance = argmax similarity` and `R^l_{distance} = N(N−1) − R^l_{similarity}` (constant offset; identity assumes the i=j diagonal is excluded (Σ_{i≠j}) — including it would break the offset because 1−CKA(f_i,f_i)=0 vs CKA(f_i,f_i)=1), so layer ranking, pair ranking, and final budget are mathematically equivalent. The polarity of the Eq. 3 normalization `R̃^l` is inverted but the cross-layer ranking is preserved. |
<!-- D-ma-detector items (1)–(2) — dynamic detector thresholds + 0.75-depth
     fallback — CONSUMED by stage1/plugins/ma_detection.py (commit pending).
     Items (3) CKA reservoir cap → SHARED with cka_distance; (4) phase_b /
     ablation batch sizes → SHARED with aimer / sink_token / three_way_and /
     ablation_filter; (5) num_calibration_samples → SHARED across Stage 1
     plugins. Resolve items (3)–(5) in Phase 9 when their owning plugins are
     processed. -->
| D-ma-detector (items 3-5 — sampling caps, batch sizes, num_calibration_samples) | 1 | <!-- SHARED — re-check at end --> | <!-- SHARED — items (3)-(5) of the original row remain; items (1)-(2) consumed by ma_detection --> | The 3.0 / 2.0 detector thresholds (item 1) and the 0.75 fallback (item 2) are now documented in the `ma_detection` plugin docstring; this row retains only the cross-plugin sampling items. | Resolve in Phase 9. |
| D-se-blacklist-merge | 1 | SE-blacklist integration into GRAPE greedy loop | 2507.23279 (SE detection) and 2604.06542 (GRAPE) describe the two phases independently; neither paper specifies how SEs interact with GRAPE's greedy merge | Spec-original integration: each SE's row and column in `D^l` are zeroed before the greedy loop so SEs never participate in pair selection and contribute zero to `R^l`; SE cluster slots are subtracted from `cluster_counts` and the global budget (`effective_budget = global_budget − total_SEs`); per-layer floor is applied to the non-SE pool only (`floor_l = max(min_experts − \|SE_l\|, 0)`) | SEs must be preserved exactly (per 2507.23279 Table 3 catastrophic-collapse evidence), so they must not be merge candidates; running GRAPE on the full `D^l` would let an SE be absorbed into another cluster, defeating the blacklist. Subtracting SE slots from the budget keeps the post-Stage-1 surviving expert count consistent with the user-specified `expert_prune_ratio`; applying the floor only to non-SE experts ensures floor protection scales with the available redundancy pool, not with the protected count. |
| D-grape-restart-merge | 1 | Lag-corrected post-selection restart variant (second restart path) | 2604.06542 Algorithm 1 lines 4–5: `if F = {1,...,L} then F ← ∅` falls through to lines 6–13 within the same `while` iteration — the paper's pseudocode unambiguously executes one merge after restart. The paper has only **one** restart path (the all-frozen check at the top of the loop). | Two restart paths: (1) the **all-frozen restart at the top of the loop** (`frozen.clear()` then continue the same iteration with argmin layer/pair selection and one merge) — this matches the paper's literal pseudocode lines 4–13. (2) Spec-added **lag-corrected post-selection restart** (code line ~911): if argmin layer/pair selection fails *after* the top-of-loop check (e.g., all unfrozen layers are at-floor), the spec clears `frozen` and immediately runs argmin layer/pair selection and one merge — a second restart path the paper does not describe. | Path (1) is paper-literal, not a deviation. Path (2) is the actual project-original choice: it covers the case where freezing/floor-saturation interact such that the top-of-loop all-frozen check did not trigger but no merge candidate is selectable in the current iteration. Allows GRAPE to escape local optima created by mid-iteration saturation; bounded (≤ one extra merge per lag-corrected restart-cycle, and the budget-termination check still gates the merge). |
| D5a | 2 | Cap on the number of experts merged into one survivor (max_merge_group_size) | 2604.04356 §4 / experiments: C=16 at 25% reduction (Qwen3-30B-A3B 96 centroids on 128 experts); C=32 at 50% reduction (64 centroids on 128 experts) | `max_merge_group_size = 8`; if any group exceeds the cap, the budget-bump loop raises `effective_target` until feasibility holds (or falls back to zero-merge for the layer) | Smaller groups reduce destructive averaging on long-tail experts: at our floor budget (256→128, ~30% reduction on a 256-expert pool) the per-centroid average absorption is 1.0 non-centroid, so C=8 provides 8× headroom above the average while still bounding any single survivor's merge breadth. The budget-bump loop catches feasibility violations (D-ream-budget-bump) so no expert weights are silently dropped. |
| D5b | 2 | Cost matrix choice for neuron permutation alignment in merge | 2604.04356 Eq. 6: frequency-weighted average with neuron permutation alignment (Ainsworth et al., 2023) w.r.t. the centroid expert; cost matrix C unspecified | Hungarian permutation `P_i` with cost matrix `C = C_wt + C_act` (gate+up Frobenius weight distance + per-neuron mean-activation L2 distance) | Paper prescribes permutation but leaves the cost form open. Spec uses `C_wt + C_act`: weight-space Frobenius distance captures structural similarity; activation-weighted neuron L2 distance captures functional importance. *TODO: Ablation of cost matrix choice (C_wt only vs. C_wt + C_act vs. activation-only) pending Stage 6 evals.* |
| D-ream-aggregation | 2 | δ_REAM cross-component aggregator | 2604.04356 Eq. 7: `δ_REAM = δ_g + δ̃_E` (sum, not normalized) | `δ_REAM(i,j) = (δ_gate(i,j) + δ̃_expert(i,j)) / 2` (mean, ∈ [0, 1]) | Monotone in each component cosine; the joint greedy ranking matches the paper's exactly only when components agree on the pair, and is a project-original re-weighting otherwise. The /2 normalization keeps cost values in [0,1] for cross-stage diagnostic comparability (Stage 3 also uses [0, 1] similarities) and lets the cost-threshold logic (`ream_cost_sigma_threshold`) operate on bounded, mean-relative quantities. |
| D-ream-similarity-rescale | 2 | Component similarities rescaled to [0, 1] | 2604.04356 Eqs. 4–5: raw cosine similarity used directly | δ_gate: L2-row-normalize profiles → pairwise Euclidean distance → `dist2sim` (1 − d/max(d)). δ̃_expert: raw cosine ∈ [−1, 1] rescaled as `(cos + 1) / 2` ∈ [0, 1] | Both transforms are monotone in the underlying cosine, so greedy ranking (and therefore centroid→non-centroid assignment) is preserved. The [0, 1] range is cross-stage-comparable, lets δ_gate and δ̃_expert be averaged on the same footing in [D-ream-aggregation], and matches the bounded similarity scales used elsewhere (e.g., Stage 3). |
| D-ream-sparse-routing | 2 | Eq. 8 numerator under sparse top-k routing | 2604.04356 Eq. 8: defined over expert outputs `E_e(x)`; paper does not specify behavior when an expert is not dispatched on token x | For jointly-active tokens use `σ(x)_e · E_e(x)` with full-softmax `σ(x)_e`; for non-jointly-active tokens, contribute zero to the numerator (expert output not computed) while keeping the full |X| in the denominator | The paper evaluates on dense-style inner products and does not prescribe a sparse case. Treating skipped tokens as zero contributions is the natural interpretation under top-k dispatch (the expert output is genuinely absent), and matches the implementation. The convention deflates δ̃_expert in proportion to the jointly-active fraction, biasing greedy assignment toward expert pairs that co-fire — desirable, since pairs that rarely co-activate carry little merge signal. |
| D-ream-budget-bump | 2 | Per-layer feasibility / quality gate around REAM target | 2604.04356: no feasibility-bump loop or cost-threshold gate described | Two project-original gates raise the layer's effective centroid count: (1) feasibility — if `N'_l × max_merge_group_size < N_l − N'_l` (i.e., the per-centroid cap, which counts non-centroids only per §5 Step 4, cannot absorb every non-centroid), bump `effective_target` by `max(1, ceil(effective_target × cost_bump_ratio))` and retry; falls back to zero-merge if `effective_target` reaches `n_experts` without feasibility. The §5 Step 3 form (`N'_l × max_merge_group_size ≥ N_l − N'_l`) is canonical; the implementation matches it (`stage2_reap_ream.py` line 381: `n_ream_nc > n_ream_c * max_group_cap`, where `n_ream_nc = N_l − N'_l`). (2) quality — if mean assigned cost exceeds `running_mean × (1 + ream_cost_sigma_threshold)` with `ream_cost_sigma_threshold = 1.5` (mean-relative multiplier; inactive for the first 4 layers that contribute valid mean-cost samples while the running mean stabilizes), bump target. | Feasibility gate guarantees every non-centroid is assignable under the D5a cap — without it, a large `max_merge_group_size` violation would silently drop expert weights. Quality gate prevents a layer from being forced through a high-cost (poor-similarity) merge configuration when the REAP/REAM signal indicates the target is too aggressive for that layer; the threshold value 1.5 is mean-relative (post-warm-up) and was tuned to fire only on outlier layers, leaving most layers at their GRAPE-allocated target. **Quality-gate exhaustion:** distinct from the feasibility-fallback above, if the quality gate is still active when `effective_target = n_experts`, the spec applies the most recent above-threshold assignment (last-resort apply-anyway) instead of zero-merging; rationale: quality-gate failures often coincide with naturally high-cost layers where any single-cap-respecting assignment is the best available, and zero-merging would unnecessarily cost compression. **Orphan-singleton promotion:** if the capped greedy assignment leaves any non-centroid unassigned (rare edge case where every centroid's cap is saturated by lower-cost candidates), the orphan is promoted to a singleton centroid (no merge) for that layer — defensive safety net; the Step-3 feasibility check is designed to prevent this. |
| D-reap-routing-weight | 2 | REAP routing weight g_j(x) renormalization | 2510.13999 Eq. 9: `S_j = (1/|X_j|) Σ g_j(x)·‖f_j(x)‖₂` with `g_j(x)` defined as the post-softmax routing weight for expert j (paper does not specify whether g_j(x) is renormalized after top-k masking) | Spec uses the **dispatched** weight as the model's forward pass applies it — for Qwen3-MoE this is `softmax(router_logits)[j] / Σ_{k∈top-k} softmax(router_logits)[k]` (top-k softmax outputs renormalized to sum=1 over the top-k set), NOT the un-renormalized masked softmax | The model's actual forward output uses the renormalized top-k weight, so REAP's "expert importance" S_j is most faithful to the model's behavior when computed against the same weight the experts actually receive. The paper is silent on this choice; both readings (renormalized vs un-renormalized) are defensible from Eq. 9 alone. The spec follows the runtime-faithful reading. Note: the un-renormalized reading would yield the same expert *ranking* under top-k routing only if the per-token sum is constant, which it is not (varies per token); so the choice is empirically distinguishable, just not in a paper-prescribed direction. |
| D-ream-resume-fallback | 2 | Resume-path C_act omission for legacy partial directories | The current spec mandates `C = C_act + C_wt` (D5b) for the Hungarian intermediate-neuron alignment in merge. Resume from a partial directory created **before** the B-iter5-M-2 fix (which began persisting `_neuron_means_layer{li}.pt` per merged layer) cannot reconstruct `C_act` from disk | When `_neuron_means_layer{li}.pt` is missing on resume, the implementation logs ERROR and falls back to weight-only `C = _safe_norm(C_gate + C_up)` (no `C_act`), producing merged weights that **differ from a fresh run** for the affected layer. New runs from scratch always persist neuron-means and reconstruct `C_act` correctly | Hard-error on missing legacy artifacts would force operators to discard partial progress on a long-running stage; the spec instead documents the degraded path so users can decide. The deviation only affects merges produced from pre-2026-05-07 partial directories; any partial directory written by the current code includes the neuron-means artifact and resumes spec-compliantly. **Operators resuming legacy partials should expect cosmetic divergence from a fresh run on the affected layers; downstream Stage 3+ is not invalidated by this divergence (the merged weights remain valid expert weights, just not the C_act-optimized assignment).** |
| D-mcf-assignment | 2 | Optimal capacitated assignment via min-cost flow / Hungarian instead of greedy | 2604.04356 §4: descending-saliency single-pass greedy with per-centroid cap (`group_size`) | Stage 2 v2 adds `assignment_solver: "greedy" \| "hungarian" \| "mcf" \| "auto" \| "sinkhorn"` (default `"greedy"` reproduces v1 bit-identically). `mcf` uses OR-Tools `SimpleMinCostFlow` with cost-range normalization to int. `hungarian` uses `scipy.linear_sum_assignment` with `+∞` → large-finite sentinel for forbidden arcs; `auto` picks hungarian for slack-capacity (`n_NC ≤ N'_l`) and mcf otherwise. | Greedy is biased toward the highest-saliency centroid (it picks first); MCF is integer-optimal under LP relaxation (transportation polytope is totally unimodular, Ahuja–Magnanti–Orlin §9). Synthetic counterexamples (spec § 2 reviewer report) show greedy 28–34% above the MCF optimum on tight-capacity instances. At our N=256, N'_l ∈ [128, 200], C_max=7 the gap is expected smaller (loose capacity), but MCF is essentially free (~10 ms / layer × 40 layers). |
| D-whitened-cost | 2 | Activation-aware whitened post-alignment cost matrix (AA-SVD lineage) | 2604.04356 Eq. 7: cost is symmetric `1 − δ_REAM` with `δ_REAM = δ_gate + δ̃_expert`, no whitening, no alignment | Stage 2 v2 adds `cost_alignment: "post"` mode where the cost is `‖(W_c − P_cm·W_m) · A^{1/2}‖_F` summed over gate/up/down. `A^{1/2}` multiplies ΔW on the **right** (input axis), per `E_x ‖ΔW · x‖² = tr(ΔW · A · ΔW^T) = ‖ΔW · A^{1/2}‖_F²`. The Hungarian permutation `P_cm` is computed once per (c, m) pair via `_permutation_align_to_centroid` and cached for the merge step (single Hungarian, two consumers). `cost_whitening: "diag"` uses `sqrt(diag(A))`; `"full"` uses `V·diag(sqrt(λ_clamped))·V^T` from `torch.linalg.eigh` mirroring `stage3_svd._precompute_eigh`. Default `"none"` reproduces v1. | The pre-alignment δ_REAM cost is alignment-invariant (output cosine and gate-logit cosine don't depend on neuron permutations) but lacks a weight-space residual term. The post-alignment whitened residual measures merge error in the directions that actually carry calibration signal (AA-SVD lineage, arXiv 2604.02119, already used by Stage 3) — explicitly **not** AIM (arXiv 2502.02421), whose actual formulation uses per-channel diagonal scaling by `mean(\|x_i\|)`, a different scheme. The K-prefilter (`cost_topk_filter`, default 48) bounds the per-pair Hungarian compute. |
| D-asymmetric-freq | 2 | Asymmetric freq-weighted cost factor `freq_m / (freq_c + freq_m)` | 2604.04356: cost is symmetric (`d[i,j] = d[j,i]`); the merge formula (Eq. 6) is freq-weighted but the assignment cost does not encode the direction asymmetry | Stage 2 v2 adds `cost_asymmetric: true` (default `false`), which multiplies the post-alignment whitened residual by `freq_m / (freq_c + freq_m)`. Both-zero edge case → 0.5 neutral. Valid only with `ream.frequency_weighted_merge=true` (rejected at run-time otherwise — fail-fast at the top of `run()`). | The merge formula `W_merged = Σ (freq_e / Σ freq) · P_e(W_e)` weights each member by its freq share. A high-freq non-centroid merged into a low-freq centroid dominates the merged weight (freq washout); the symmetric cost matrix cannot distinguish merge direction. The asymmetric factor is the per-pair version of the merge weight: `freq_m / (freq_c + freq_m)` is exactly the share of `freq_m` in a 2-element merge group {c, m}, so the cost penalizes pairs where m would dominate c. Only valid under freq-weighted merge — under saliency-weighted merge the analogous factor would be `sal_m / (sal_c + sal_m)`. |
| D-capacity-util-gate | 2 | Per-layer SLACK/TIGHT path selection by capacity utilization | No paper precedent — project-original | Stage 2 v2 adds `_pick_effective_alignment(n_nc, n_c, max_group_cap, threshold, configured)`: when `u = n_nc / (n_c × max_group_cap) < capacity_util_threshold` (default 0.25), the layer falls back to `cost_alignment="pre"` (cheap symmetric δ_REAM) regardless of the configured value. Uncapped (`max_group_cap == 0`) is treated as fully slack (`u = 0`). | The post-alignment whitened cost is expensive (per-pair Hungarian + 3 Frobenius norms × K candidates per non-centroid). At low utilization (slack capacity), most centroids have many obvious good matches and the heavyweight cost matrix is unlikely to change the assignment meaningfully — gating the heavy machinery on `u ≥ 0.25` saves ~50% of the per-layer compute on GRAPE-allocated heterogeneous-budget runs. Layers near the floor (50% reduction → `u ≈ 0.6–0.9`) still get the full machinery; high-budget layers (low u) skip it. |
| D-em-refinement | 2 | EM-style assignment refinement (Sub-MoE) | 2604.04356 / 2510.13999 / 2604.06542: single-shot greedy or Hungarian assignment, no iterative refinement | Stage 2 v2 adds `em_refinement_rounds` (default 0) iterations of: tentatively merge each non-singleton group with current assignment (no model mutation; freq-weighted weights computed in-memory), recompute the cost matrix against the tentative merged centroids, re-solve the assignment. Stops early on `em_convergence_break=True` (default) and assignment stability. EM is a no-op under `cost_alignment="pre"` (cheap symmetric cost doesn't depend on centroid weights). | The merge formula is non-linear in inputs but linear in weights: `forward(linear_combo(W_e)) ≠ linear_combo(forward(W_e))`. After one merge, the centroid's weights are no longer the original — a new assignment under the new centroid weights may produce a lower-cost matching. Sub-MoE (arXiv 2506.23266) demonstrates this iterative refinement on K-means-style merging; the Stage 2 v2 EM round is the same idea applied to capacitated assignment. The cached perm becomes stale under tentative weights, so the inner cost recomputes the perm; the cache is **not** updated with tentative residuals so the merge step's perm-cache reuse is preserved. |
| D-expert-distill-mse | 2 | Per-merge-group expert distillation against routing-gated original outputs | 2604.04356: one-shot weighted average; no post-merge expert refinement. SlimMoE (2506.18349) and MoE-Pruner (2410.12013): distill the full MoE-block output, with router updates concurrent in some phases. | Stage 2 v2 adds `expert_distill_steps` (default 0) of AdamW MSE distillation per non-singleton group: target = `Σ_{e∈g, e∈TopK(σ_orig(x))} g_e^orig(x) · E_e^orig(x)` (routing-gated additive contribution of pre-merge group members on tokens from `X_g`); student = `g_g^merged(x) · E_g^merged(x)` with the post-resize router row frozen. Trainable: only the merged centroid's gate/up/down. Plateau early-break, fp32 optimizer with bf16 forward, bank dtype preserved on writeback. | Two project-original deviations from SlimMoE / MoE-Pruner: (a) we distill the per-merge-group additive contribution (only the merged centroid changes; other experts and the rest of the MoE block are untouched) so the loss attributes cleanly to the centroid being trained, rather than the full MoE-block output where errors compound across all experts; (b) expert-only training is strictly separated from router-only training (Stage 2.5) for resume-isolation and stage-boundary clarity, where SlimMoE's distillation phases can update both concurrently. The pre-merge router row is carried over verbatim to the post-resize router (centroid expert's original row), so `g_g^merged(x)` is the original centroid's routing weight evaluated under the new (smaller) softmax denominator — Stage 2.5 retrains it. **Stage 2.5 consequently sees a model whose merged centroids are already distilled — its job becomes purely router calibration on top of pre-distilled experts, not expert recovery (see § 5.5).** |
| D-expert-distill-mse-v1 | 2 | v1 simplifications layered on D-expert-distill-mse | The contract above is the *target* — full routing-gated target on `X_g`. v1 implementation simplifies for engineering tractability. | The v1 implementation in `_distill_merged_group` differs in two ways: (i) target uses **freq-weighted-only** mixing `Σ (freq_e / Σ freq) · E_e^orig(x)` (no per-token routing weight `g_e^orig(x)`); (ii) input tokens are the reservoir-sampled layer-input captured during profile (cap at `expert_distill_token_cap=8192`, seeded per-layer for reproducibility), not the routing-restricted `X_g` set. | The full routing-gated form requires storing `g_e^orig(x)` per (expert, token) pair (additional memory) and reconstructing `X_g` from `ReamCostAccumulator.gate_logit_profiles` keys (additional plumbing). v1 produces a correctly-signed merge-error gradient on a uniform-token sample — the merged centroid is still pulled toward a freq-weighted average of original-expert outputs. Phase 3 v2 will lift both simplifications; the § 8 ablation matrix row A8 measures v1, A8' (planned) measures the spec form. |
| D-sinkhorn-soft-assign | 2 | Capacitated entropy-regularized OT alternative to MCF | 2604.04356: greedy assignment, no OT formulation | Stage 2 v2 adds `assignment_solver: "sinkhorn"` (opt-in, default off): solve `min Σ T_cm·d_cm + ε·Σ T_cm log T_cm` with Sinkhorn-Knopp iterations (linear ε-anneal `1.0 → 0.01` over `sinkhorn_iters` ≈ 200), then argmax over real centroids per non-centroid for the hard assignment. The capacity inequality `Σ_m T_cm ≤ C_max` is converted to equality via a **dummy slack child** with marginal `n_C × C_max − n_NC` and uniform high cost — standard partial-OT trick (Cuturi 2013 + dummy-marginal). Cost matrix normalized to `[0, 1]` before Sinkhorn iterations so ε values are scale-invariant (positive affine transformation invariance of OT). | This is **not** Sparsity-Constrained OT (arXiv 2209.15466), which uses quadratic regularization with a first-order semi-dual solver and cardinality (`\|T\|_0 ≤ k`) constraints — different scheme entirely. Spec § 5 step 4d frames the construction as a dummy *centroid*; the implementation uses a dummy *child* (rows-side dummy). The two are dual under argmax over real centroids and produce the same hard assignment; the slack-child form is simpler because the real-children argmax never has to filter out a dummy column. Sinkhorn falls back to greedy on infeasibility (`n_C × C_max < n_NC`) with a clear warning. Currently Tier-3-opt-in; gated default flip on `A9 vs A8 ≥ +0.1 GEN-avg` per § 8 ablation. |
| D-protocol-blend | 2.5 | Protocol combination: REAM + Router KD in sequence | 2604.04356 (REAM): explicitly evaluates "without any fine-tuning after compression"; 2603.02217 (Router KD): designed as a standalone step, not as a post-REAM patch | Spec applies Router KD (Stage 2.5) immediately after the REAM merge | Router KD restores routing accuracy degraded by weight averaging; REAM's static evaluation does not cover post-merge routing drift. Combined protocol not ablated against REAM-static-only baseline: empirical_pending |
| D6 | 3 | AA-SVD cross-covariance scope and Path 2 auto-cov substitution | 2604.02119 Theorem 3.2 requires cross-covariance for all linear layers; the paper recognises Path 1 (`M = W·C·B⁻¹·L_B`, Theorem 3.2) and the Path 3 special case A=B (Corollary 3.3, `M = W·L_B`). Path 2 (`M = W·A·B⁻¹·L_B` with A ≠ B, A = pre-prune auto-cov) is *not* a paper-recognised variant. | (a) Cross-covariance C collected for gate_proj/up_proj (input-side) via dual-forward; down_proj falls back to Corollary 3.3 (B-only) because the teacher's per-expert intermediate activations require full expert dispatch instrumentation. (b) Path 2 (auto-cov-for-cross-cov substitution, A from Stage 2) is enabled by `aa_svd.cross_covariance: false` for runs where C is unavailable but A is. | Gate/up inputs share the same hidden state (pre-routing) so one capture covers both; down_proj inputs are expert-internal (post gate+up) and differ between teacher and student expert sets. Path 2 is a project-original hybrid: it slots the pre-prune auto-covariance into the Theorem 3.2 machinery as a strict generalisation of Corollary 3.3 (which uses A = B = X_post). The substitution is consistent (the two coincide when pre/post distributions are similar — light pruning) and degrades gracefully toward Path 3 as the auto-cov departs from the cross-cov; quality vs. Path 3 not separately ablated. |
| D-AASVD-objective | 3 | AA-SVD primary objective variant | 2604.02119 §4.3 Table 5 recommends input-aware (A=B=X, Corollary 3.3 with pre-prune covariance) + block refinement as primary recipe (PPL 6.89 at ρ=0.8 LLaMA-7B) | Spec uses anchored-adaptive (A=X_pre, B=X_post, Theorem 3.2) + block refinement (Path 1). Quality gap ~0.2 PPL at ρ=0.8 on LLaMA-7B; Qwen3-30B-A3B comparison empirical_pending. | Anchored-adaptive is the paper's central theoretical contribution and expected to outperform in high-compression regimes where upstream drift is larger; empirical validation on Qwen3-30B-A3B pending |
| D7 | 3 | D-Rank ω adapted for MoE | 2509.25622 Eq. 7: ω = d₁ + n·d₂ (layers per group × dimensions) | ω = n_experts × (d_out + d_in) | D-Rank targets shared-basis layer groups; adapted for MoE expert groups |
| D7a | 3 | Per-projection rank bias and `k̄` semantics for ε* | 2509.25622: D-Rank Eq. 7 produces a single `k_g` per (layer, matrix_type) group; no per-projection-type multiplier. Swift-SVD 2604.01609 defines `k̄ = (m·n)/(m+n)·ρ` as the plain uniform rank entering ε*. | Group ranks from Eq. 7 are scaled by `gate=1.33, up=0.67, down=1.0` (sum=3.0; the multipliers are approximately parameter-budget-preserving: exactly preserved when gate/up/down receive the same `k_g` and share the same `ω_g` — which holds under SwiGLU symmetry where gate/up have identical input dimensions; in the general case, the multipliers redistribute rank between projection types and may shift the post-bias parameter total by a few percent) before per-expert redistribution; the bias-adjusted `k̄` also flows into the Swift-SVD ε* computation (§6 Phase D). | Adapted from jangq's GGUF bit-allocation insight (`gate:up:down ≈ 4:2:3`, see `397B-MLP-ASYMMETRY.md` §3.1): SwiGLU forward couples gate errors multiplicatively via SiLU, while down errors propagate to the residual stream. The ratio translates the same physical asymmetry from bit space to rank space. *TODO: empirical re-tune from clean per-projection `recon_rel_err` once Stage 6 evals are available; current values inherited unchanged from a prior bf16-bug-tainted run and are theoretically- (not empirically-) grounded.* |
| D8 | 3 | Swift-SVD β | 2604.01609 Alg. 2: β = end-to-end layer importance, min-max normalized to [1,2] | β = per-expert spectral energy share (σ_i² / Σ σ_j²) in range (0,1] | Paper's β is per-layer end-to-end importance (requires 40 extra forward passes), min-max normalized to [1,2]; adapted to per-expert spectral energy share (σ_i²/Σσ_j²) in range (0,1]. The range difference changes blending behavior: paper's β∈[1,2] means β^α always amplifies; spec's β∈(0,1] can suppress low-energy experts. This is intentional — per-expert spectral energy within a group is the natural adaptation of per-layer importance for MoE expert redistribution. ε* is now activation-weighted via Stage 2 A-covariance (no longer a deviation) |
| D-eps-star | 3 | Swift-SVD ε* normalization | 2604.01609 Eq. 4: ε*_k = (Σ_{j>k} σ_j²)^{1/2} — absolute truncation error | ε*_i = √(Σ_{j>k̄} σ̃_j² / Σ_j σ̃_j²) — relative ratio (normalized by total spectral energy) | Normalization makes ε* scale-invariant across experts with different total spectral energy, enabling meaningful cross-expert comparison within the redistribution step; the log() in the blending score damps large outliers regardless. Additionally, σ̃_j = sv(A^{1/2}·W), not sv(W·A^{1/2}) — A^{1/2} left-multiplies W to match the activation-weighted output error ‖XW − XW_k‖_F. |
| D10 | 4 | Eigenspace noise-floor truncation | 2410.21271 Alg. 1: full Q ∈ ℝ^{k×k} used; QQ^T = I guarantees Theorem 1 exactness | Multi-sample reading: `X̃ ∈ ℝ^{N×d_in}` is the matrix of per-token activation samples for tokens routed to the expert, so `A = X̃^T X̃ ∈ ℝ^{d_in × d_in}` has rank ≤ min(N, d_in) (typically ≫ 1 under our calibration volume). The noise-floor threshold keeps only eigenvalues above a dtype-aware floor; small-eigenvalue directions below the floor are discarded. | This is a **real, deliberate deviation** from Theorem 1 exactness — not exact. When A has rank > 1, noise-floor truncation discards small-eigenvalue directions (noise-dominated activation modes). The discarded **activation-weighted reconstruction-error component** is upper-bounded by `‖ΔW‖_2^2 · Σ_{j > n_keep} λ_j` (where `ΔW = W_orig − Ŵ`). (Loewner-trace majorant: the discarded contribution to tr(ΔW · A_tail · ΔW^T) is bounded by ‖ΔW‖_2^2 · Σ_{j > n_keep} λ_j where A_tail = Σ_{j > n_keep} λ_j q_j q_j^T.) For the eigendirections kept, Theorem-1-style exactness holds in the kept subspace. This residual is dominated by Stage 3 SVD residual / quantization residual at moderate-to-high compression ratios. The trade-off is intentional — preserving every tiny eigendirection would waste rank budget on noise; the rank cap (`eigenspace_rank_cap=128`) further bounds `take_eff` so the correction concentrates on the highest-energy directions. See §7 Step 2 and the D-S-H-1 / Stage 4 spec rewrite (2026-05-06) for resolution of the prior rank-1 framing. |
| D-eora-budget-pct | 4 | EoRA per-matrix rank budget = 3% of Stage 3 savings | 2410.21271: paper sweeps fixed correction ranks {64, 128, 256, 512} per matrix in its experiments; no "% of savings" rule | `compensation_budget_pct=3%` of Stage 3 per-matrix parameter savings, then capped at `eigenspace_rank_cap=128` rank | Project-chosen, **not from paper**. 3% empirically selected to keep Stage 4's added parameter footprint small relative to Stage 3 savings (net compression remains favorable while still recovering quality on the most-truncated matrices). The cap at 128 keeps per-matrix EoRA rank within the paper's evaluated range {64, 128, 256, 512}. *TODO: ablate 1% / 3% / 5% once Stage 6 evals are available.* |
| D11 | 2, 2.5, 5 | Calibration data source | 2603.02217 §F.3 Table 1: c4; 2510.13999 §4: c4 + evol-codealpaca (used identically across all experiments) | Multi-domain Nemotron-Cascade-2-SFT-Data with weighted subsets (chat 0.56, math 0.21, science 0.11, etc.) | Task-aware calibration better matches target deployment distribution; c4 and evol-codealpaca are general pre-training / instruction-tuning data with limited reasoning/code coverage relative to the target deployment mix |
| D-cal-size | 2 | Calibration sequence count | 2604.04356 §4: 3072 sequences × 512 tokens (1.57M tokens total); 2510.13999: 1024 sequences × 2048 tokens (2.1M tokens total) | 4000 sequences × 2048 tokens (8.19M tokens total) (Nemotron weighted subsets) | Exceeds both papers' calibration volumes (5.2× REAM in tokens, 3.9× REAP in tokens); longer 2048-token sequences match the deployment context length and capture more inter-token routing patterns per sequence. Task-aware Nemotron dataset documented in D11 |
<!-- D-aimer-cross-check — CONSUMED by stage1/plugins/aimer.py (commit pending). -->

<!-- D-sink-token-routing — CONSUMED by stage1/plugins/sink_token.py (commit pending). -->
| D-causal-ablation-validation | 1 | Phase D ablation filter (project-original; load-bearing — produces final blacklist) | None — paper validates SE detection via global ablations (Table 3) but not as a per-expert filter | Phase D ablates every Phase C candidate over a held-out 100-sample slice (deterministic seed offset distinct from Phase A/B; cached at `_calibration_cache_phase_d/`). For each `(l, e)` candidate, install a forward hook that zeros `down_proj` output, measure ΔNLL = ablated_nll − baseline_nll, and include `(l, e)` in the final blacklist iff `ΔNLL > ablation_filter_threshold` (=0.001 ≈ 0.1% PPL impact). Per-candidate ΔNLL retained in `stage1_ablation_filter.json` for audit. Default `ablation_filter_batch_size = 8` over 100 samples (~13 batches/candidate) → ~15–30 min for 60–100 candidates on H200. (bs=32 was the v6-prep default; OOM'd on 2026-05-10 job 6a00caf0 because the bf16→fp32 logits upcast wanted 38 GB at bs=32 vs ~9.5 GB at bs=8.) | v4 produced a 158-expert blacklist of which only 5 had measurable ΔNLL; 144 were dead-weight, 9 were active false positives that *hurt* PPL when protected. Static-threshold detection is fragile across architectures (each new architecture shifts the right thresholds and the right values are unknown until the model is run); ablation evidence is ground truth. v6 promoted ablation from report-only to the load-bearing final filter and rewrote Phase C as a candidate-pool generator gated by Phase D evidence. |
| D-magnitude-topk-candidates | 1 | Magnitude top-K candidate source for Phase C | None — magnitude top-K not in 2507.23279 (paper's three-way AND is the sole criterion); GRAPE 2604.06542 is unrelated | Phase C augments the candidate pool with the top-`magnitude_topk_per_l_layer` (=16) experts per `l ∈ L` by `per_expert_max(l, ·)`, regardless of whether they pass the three-way AND threshold. Provenance tagged "magnitude_topk". Final blacklist requires Phase D ablation evidence. | Catches SEs whose magnitude doesn't quite cross the three-way AND but is still large within the layer. K=16 = 2× the model's active-experts-per-token (top-8 routing) — broad enough to pick up architecture-shifted SEs that static thresholds miss. Motivated by v3 Phase F surfacing non-blacklisted experts with measurable ΔNLL (e.g., `L34E85` with ΔNLL = −0.025) that a magnitude-top-K source would have caught. False candidates cost ablation time but cannot reach the final blacklist without Phase D evidence. |
| D-reap-min-active-tokens | 2 | Centroid-candidacy filter on min active-token count | REAM/REAP: no minimum-active-tokens filter described | `reap_min_active_tokens=32` (configurable; default 0 in code, set to 32 in production config) excludes experts with fewer than 32 active calibration tokens from REAM-centroid candidacy. Filtered experts become non-centroids and are merged. If `len(ream_centroid_ids) < ream_target` after filtering, a WARNING is logged but no compensating bump fires | Low-frequency experts have noisy gate/expert profiles (averaged over <32 tokens), so promoting them to centroids would propagate that noise into the merged weights. Filtering them to non-centroid status routes them through the Hungarian alignment (which projects them onto a higher-frequency centroid's neuron space) instead. Compression target may shrink slightly when many low-frequency experts are filtered; the WARNING surfaces this without silent silent compression shortfall. |
| D-cov-storage-fp16 | 2, 3 | Stage 2 covariance + Stage 3 B-cov persisted in fp16 (not fp32) | Spec §5 "Covariance Side-Collection" originally stated fp32 storage citing Swift-SVD certification | Persisted as fp16 (10 mantissa bits) for both `_stage2_input_covariance.pt` and `_bcov_*.pt`; eigendecomposition still runs in fp64 in-memory | fp16 mantissa precision (10 bits) is strictly higher than bf16 (7 bits) and produced cleaner Stage 3 rank-deficiency outcomes than bf16 in spot checks. Halves the persisted-covariance disk footprint vs fp32 (~2× saving on the gigabyte-scale covariance artifact) without measurable downstream PPL/zero-shot drift. Switching back to fp32 is a one-line config flip if a future model exposes precision sensitivity. |
<!-- D-a-max-fraction — CONSUMED by stage1/plugins/three_way_and.py (commit pending). -->

| D-per-type-alpha | 3 | Per-projection-type α refinement after paper-exact global α-search | 2604.01609 §3.2.2 selects a single global α via end-to-end WikiText-2 PPL validation; the chosen α is used for every projection. | When `swift_svd_plus.per_group_type: true` (production default), Stage 3 first runs the paper-exact validation search to pick `best_global_alpha`, then runs a per-projection-type spectral-proxy refinement (`_swift_svd_plus_alpha_search`) seeded from that global α. The final factoring uses the per-type `alpha_by_type` map, **not** the validation-winning global α directly. | Project extension. Paper's single-α assumption pools gate/up/down into one allocation regime; per-type allows the gate-vs-up-vs-down spectral asymmetry (separately documented as the per-projection bias multipliers) to also influence redistribution. The seed from global validation keeps the search anchored near the paper-compliant region, and the per-type refinement is bounded by the spectral proxy (cheap, no model forward). Disable by setting `per_group_type: false` to restore strict paper compliance — operator choice, opt-in. |
| D-no-intra-block-cascade | 3 | Phase C factorizes all Wj in a block from un-cascaded covariances | 2604.02119 Algorithm 2 lines 4–8 + Algorithm 1 line 5: within each block, when compressing Wj, the input X′_j must be produced by a forward pass through L′_i with the **already-compressed** Wj′ (j′<j) of the same block. Compression of Wj+1 should therefore see the post-compression activations of Wj. | Phase A (lines 461–477) collects every layer's B-cov and C-cov once in a single pre-factorization dual-forward against the **un-cascaded** student. Phase C then factorizes every Wj in the model from those static covariances; the within-block cascade Algorithm 1 line 5 prescribes is skipped. Phase C.5 (lines 590–624) restores **cross-block** sequential consistency (after block i is refined, X′_{i+1} reflects the refined block i) but does **not** revisit the intra-block layer-j cascade. | Project-pragmatic. Doing the paper's per-Wj cascade requires per-(layer, sublayer-j) targeted dual-forwards (40 × 5 ≈ 200 forward passes through partial models, with covariance recollection at each), versus our one-shot Phase A. The activation-aware AA-SVD weighting (B-cov from un-cascaded student is a close approximation of the cascade B-cov for moderate compression ratios; cross-block cascade restoration in Phase C.5 absorbs most of the residual) and the joint Phase C.5 AdamW refinement together minimise the practical gap; the paper itself reports that Phase C.5 (Algorithm 2 line 9) is the dominant quality lever over Algorithm 1's line 5 cascade. Trade-off accepted; revisit if Stage 6 PPL regresses on a future architecture port. |
| D-drank-premerge-A | 3 | D-Rank whitening reuses Stage 2 pre-merge A-covariance on post-merge weights | 2509.25622 §3.2.1 assumes the whitening factor `S_g` is computed from activations of the model being compressed (post-merge for our pipeline). | Phase B uses `A_gate_up` and `A_down` from `_stage2_input_covariance.pt`, collected during Stage 2 calibration on the pre-merge expert population. After REAM merging, the surviving experts produce slightly different intermediate activations than the pre-merge experts they replaced; the down_proj input distribution shift is the larger of the two. | Project-pragmatic. Re-running a Stage 3-specific calibration pass to collect post-merge A would cost a full teacher+student forward and ~140 GB of new covariance on disk, with marginal expected impact: REAM's frequency-weighted merge (paper Eq. 6) preserves expected activations by construction, so the pre/post-merge A on a per-(layer, matrix-type) average basis is close to identity under expected-merge invariance. Stage 4's EoRA residual compensation (paper 2410.21271) absorbs any residual whitening mismatch via the activation-aware √Λ projection on the **post-merge** Stage 4 covariance reuse. Trade-off accepted; revisit if Stage 6 PPL regresses unexpectedly on a future architecture port. |
| D-c5-moe-only | 3 | Phase C.5 refines MoE blocks only, not every transformer block | 2604.02119 Algorithm 2 (lines 2–11) iterates the block refinement over **every** block `Lᵢ ∈ M`; paper applies the joint AdamW objective to each block's full parameter set (factorized factors + RMSNorm scales). | Stage 3 only factorizes MoE expert matrices (`gate_proj` / `up_proj` / `down_proj`) per §10 Protected Components; attention projections, embeddings, lm_head, shared experts, and dense (non-MoE) decoder layers are untouched. Phase C.5 therefore only fires on MoE decoder layers, where there are factorized `{U_j, V_j}` to update. Non-MoE decoder layers (if any in the architecture) participate in the stream-advance forward but skip the AdamW refinement; their RMSNorm scales remain frozen. | The skipped quality lift is bounded: dense interlayers contribute only RMSNorm scale corrections to the paper's objective. For Qwen3-30B-A3B (the project's target model) every non-shared decoder layer is MoE, so the deviation is **vacuous in practice** — every layer with refinable factors is refined. The deviation is named explicitly so a future port to a mixed dense/MoE architecture cannot silently drop the dense-block refinement step. |
| D-humaneval-greedy | 6 | HumanEval pass@1 protocol (greedy + in-process exec) | Chen et al. 2021 (canonical HumanEval): stochastic pass@1 estimated from n=10 samples per problem at temperature T=0.2 with top_p=0.95, then unbiased pass@k formula; **exec-based scoring runs each problem in a subprocess sandbox** | (a) Greedy decoding pass@1 (do_sample=False, n=1, no temperature, no top_p), single sample per problem. (b) Exec-based scoring runs **in-process**, NOT in a subprocess sandbox: tests can leak `sys.modules` / signal handlers / `os.environ` mutations across problems. Both teacher and student see the same in-process leakage so the relative-to-teacher gate is not biased. | Greedy is lower-variance and reproducible across runs without seed plumbing, sufficient for **relative-to-teacher gating** (the gate is a 3pp absolute drop vs the same-protocol teacher score, not against published baselines). **Absolute pass@1 numbers will not match published Chen et al. 2021 baselines** and must not be compared to them. The gate's batched-vs-bs=1 numerical-identity claim (#3, #4) holds under greedy decoding only. In-process exec is documented because subprocess isolation would slow eval substantially with no signal-quality benefit for the relative gate; see `stage6_validate.py` module docstring "Known limitations" subsection for the operational caveats (daemon-thread leakage, no syscall interruption, no seccomp/landlock). |

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
| 2508.03616 | Hidden Dynamics of Massive Activations in Transformer Training | 2025 | Stage 1 §4 Phase A L31 reset rationale |
| 2603.17771 | Attention Sinks Induce Gradient Sinks | 2026 | Stage 1 §4 Phase A gated-architecture rationale |
| 2603.18492 | AIMER: Activation-Independent Magnitude-Energy Ratio for Expert Pruning | 2026 | Stage 1 §4 Phase C AIMER candidate source (D-aimer-cross-check), §4 Phase A calibration-sensitivity discussion |

---

*This document was generated from a full algorithmic review of the max_quality codebase on 2026-04-28; §12 updated 2026-04-29 after a per-stage paper compliance audit including full methodology-section cross-reference of all 10 cited papers; further per-stage spec-only paper-compliance review on 2026-05-01 added D5a (REAM merge-group cap) and D5b (intermediate-neuron Hungarian alignment in merge), corrected the D-Rank citation (Eq. 7, not Eq. 6) and the §6 ε* formula to reflect activation weighting per D8, fixed the δ_gate similarity/distance notation in §5, clarified A-covariance reuse from Stage 2 in §6 Phase A, clarified the A-vs-B weighting roles in §6 Phase D, fixed the SVD reconstruction notation (`diag(Σ[:k])` instead of `S[:k]`), and corrected the §4 R^l-update rationale. Spec redesign on 2026-04-29: merged Stage 0 into Stage 1 (CKA + SE detection), floor=n//2, max_merge_group=8, Router KD bs=8 (equivalent to paper's bs=2 × accum=4 = 8; not a §12 deviation). D9 resolved on 2026-04-30: Swift-SVD α selection now uses paper-exact WikiText-2 PPL validation (§3.2.2 of 2604.01609) instead of spectral proxy; D9 removed from §12. Phase C eigh caching added 2026-04-30: gate_proj/up_proj share the same B and C covariance; eigendecomposition is now precomputed once per expert and reused for both projections, eliminating ~7,200 redundant eigh(2048×2048) calls. Compute-time optimizations 2026-04-30: (1) Stage 2 sequential profiling with early-exit forward — **implemented**; (2) vectorized REAM accumulators — **planned, not yet implemented**; (3) Stage 5 KL chunk size increased to full sequence length on H200 — **implemented**; (4) torch.compile support for Stages 2.5/5 KD forward passes — **implemented**. Stage 6 compute-time optimizations 2026-04-30 — **all implemented**: (5) WikiText-2 PPL batch_size 1→8; (6) lm-eval batch_size=auto:8; (7) batched model.generate() for HumanEval and MATH-500; (8) torch.compile for prefill-dominant forward paths; (9) teacher eval caching with sha256 cache key auto-invalidation (~50% total time eliminated); (10) teacher I/O overlap via background CPU preload; (11) GGUF conversion overlap with teacher eval. All Stage 6 optimizations are purely computational scheduling — numerically identical to batch_size=1 baseline. Expected total Stage 6 speedup: ~8–12× (from ~3–5 hours to ~25–30 minutes on H200). SE detection rewritten 2026-05-05 (audit triage findings F-0006, F-0037, F-0016, F-0015, F-0012): §4 Phase B replaced with paper-exact Algorithm 1 criterion (three-way AND: a_{l,e} > P99.5(A) AND > 0.1·a_max AND l ∈ L); Phase A.5 added for MA-formation layer detection pre-pass; empirical SE scale stated as < 0.5% (no hard cap); canonical SE verification reference added. D1 (per-layer z-score deviation) and D2 (blacklist caps deviation) removed from §12 — spec now complies with the paper on these points. Stage 3 spec updated 2026-05-05 (audit triage findings F-ch12-missing-0001 CRITICAL and F-ch12-missing-0004 HIGH): (1) §6 Phase B rewritten to include FP64 Cholesky whitening per D-Rank paper 2509.25622 Eq. 1 — effective rank is now computed from SVD of `S_g · W_g` (whitened) not raw `W_g`; covariance `X_g^T X_g` sourced from Stage 2 `_stage2_input_covariance.pt` (A_gate_up for gate/up, A_down for down_proj); (2) §6 Phase C.5 added — AA-SVD block-level joint refinement per paper 2604.02119 Algorithm 2 §3.3: after each block's Phase C factorization, all factorized weight factors `{U_j, V_j}` and block-local RMSNorm scale parameters `θ_i` are jointly optimized via AdamW (lr=1e-4, 25 epochs, cosine schedule, batch 32) to minimize block output MSE against the original model; blocks processed sequentially 0→N-1; (3) §10 Protected Components updated to carve out RMSNorm scale parameters `θ_i` during Stage 3 Phase C.5 block refinement only (paper-required exception to the general RMSNorm protection rule). No §12 D-rows added or removed for these two changes — the spec now describes the paper-exact algorithms. All formulas were verified against the cited papers. All deviations are deliberate and documented. For the original validation audit, see the archived [VALIDATED_STRATEGIES.md](https://huggingface.co/pirola/moe-compression-workflow/blob/main/VALIDATED_STRATEGIES.md). Stage 2.5/5 spec touch-up 2026-05-06: widened D11 to cover Stage 2.5, added teacher-fallback options to §5.5, tightened frozen-scope wording, KL direction proof, batch-size equivalence note. Stage 6 spec rewrite 2026-05-06: pinned PPL protocol, declared HumanEval greedy pass@1 deviation (D-humaneval-greedy), expanded cache-key spec, switched imatrix to wiki.train, atomic cache writes, fixed Qwen3-30B-A3B typo. Stage 4 spec rewrite 2026-05-06: committed to multi-sample X̃ reading, amended D10 to honestly describe noise-floor truncation as a real bounded deviation, added D-eora-budget-pct, corrected 'paper default' wording. Stage 3 spec touch-up 2026-05-06: added Phase A.X teacher block-output cache (cache approach chosen over keep-resident — preserves VRAM economy), tightened D7a budget-neutrality wording, extended D6 to cover Path 2, marked §3 Budget Solver as project-original, plus minor citation/wording fixes. Stage 1 spec touch-up 2026-05-06: added D-rows D-cka-distance, D-ma-detector, D-se-blacklist-merge; amended D5 to capture floor-during-greedy enforcement detail; corrected Algorithm 1 line citations (lines 4–12 / 14–32); added cross-references in §4 Phases A/D/E and the Key Formulas block; tightened MA-detector wording, Tables 6/7 inconsistency note, and the CKA/Eq. 11/Eq. 3/Eq. 10 paper-vs-spec polarity notes. Stage 2 spec touch-up 2026-05-06: added D5a, D-ream-aggregation, D-ream-similarity-rescale, D-ream-sparse-routing, D-ream-budget-bump; corrected Eq. 7 reference; minor polish. Stage 1 spec touch-up 2026-05-07 (post-independent-verification): [SUPERSEDED — see same-day continued entry below] corrected paper-citation §3.2.1 → §3.2.2 for the input-stable MA quote (lines 405–406); updated Algorithm 1 line range 15–23 → 15–22 (line 23 is `end for`); added "(spec inference; paper does not explicitly classify why L1E8 fails the SE criterion)" parenthetical to the Tables 6/7 inconsistency note; rewrote the §4 Key-Formulas R̃^l polarity note as the explicit identity `R̃^l_dist = 1 − R̃^l_sim`; added D-grape-restart-merge to §12 documenting the one-extra-merge-per-restart-cycle choice and cross-linked it from §4 Phase E. Stage 2.5/5 spec touch-up 2026-05-07 (Track E independent-verification fixer pass): §8 N_x defined without `+ ε` under fully-packed-sequences invariant with restoration note; KL-direction proof pinned to teacher-as-reference convention; §5.5 and §8 frozen-scope clauses extended with paper's gradient-flow language; §5.5 cache-precedence cross-reference added; §8 vocab-logits VRAM accounting unified at ~6 GB (~15 GB headroom) across §5.5 trailer / §8 hyperparameter table / §8 Teacher Loading; §5.5 hyperparameter inheritance cross-references §8 batch-size table; §8 resume rationale tied to optimizer-step boundary; §5.5 differences table teacher-precision row clarified; §8 4-bit teacher TVD bound replaced with qualitative spot-check claim. Stage 6 spec touch-up 2026-05-07 (Track F iter-4 post-independent-verification): WikiText-2 PPL row-join codified as `"\n\n"` (project convention matching HF / lm-eval recipe and the imatrix calibration corpus build) — F-iter4-M-1; teacher cache-key dataset_revisions mapping reduced to {wikitext_ppl, humaneval, math500} — hellaswag/arc_challenge are lm-eval-managed and cannot be enforced at load time, the cache key compensates by including `lm_eval_version` and a SHA-256 of the lm-eval task config (F-iter4-HIGH-1). Stage 1 spec polish 2026-05-07: tightened Algorithm 1 line citations, made D-SE-A external-code claim explicit, reworked R̃^l polarity derivation, scoped D-grape-restart-merge to the lag-corrected variant, plus minor wording fixes. 2026-05-07: §4 Phase A line 161 citation corrected back to paper §3.2.1; the previous 2026-05-05 update went the wrong way — lines 405–406 are §3.2.1, not §3.2.2; Algorithm 1 is in Appendix L. Stage 1 spec touch-up 2026-05-07 (continued): reverted §4 Phase A citation to paper §3.2.1 (the prior swap was incorrect; lines 405-406 are §3.2.1, Algorithm 1 is in Appendix L); tightened Phase E entropy-re-eval wording to match paper line 11–12 same-iteration order. Stage 1 spec touch-up 2026-05-07 (iter+2): rewrote Phase E merge bullet to mirror paper Algorithm 1 lines 8-12 control flow; tightened §4 Phase A citation with verbatim quote; clarified Phase B scope and running-max equivalence; added forward-reference to D-grape-restart-merge path-(2); generalized D5 floor wording to non-256-expert architectures. Stage 1 spec touch-up 2026-05-07 (iter+4): Phase E argmin/argmax inline cross-link to D-cka-distance; Qwen3-30B-A3B SE-set citation tightened to Table 1 row reference; Phase D verbatim quote restored from paper §3.3 line 249. Stage 1 spec touch-up 2026-05-07 (iter+5): Phase A first-MoE-layer fall-through clarified; Algorithm 1 condensation note explicit about paper line 8 'if MA pattern detected' replacement; Eq numbering OCR caveat; Eq. 2/11 cross-reference; Tables 6/7 typo annotation. Stage 1 spec touch-up 2026-05-07 (iter+6): dropped inaccurate Eq. 2 cross-reference (Eq. 2 is the normalized average; Eq. 11 is the unnormalized sum); 0.75-fallback exactness caveat for non-integer total_layers. Stage 3 spec touch-up 2026-05-07: removed Phase A.X (215 GB on-disk teacher block-output cache). Hardware target is H200 (141 GB), where the teacher (~70 GB) and student (~50 GB) both fit during Phase C.5 alongside the per-block AdamW optimizer state, gradients, and activations (~10–15 GB) — Phase C.5 now invokes the still-resident teacher's block forward on-the-fly. The cache approach was an A100-era artifact and has been removed end-to-end (§6 Hardware, Phase A trailer, Phase C.5 objective+interaction, Resume). Resume scheme replaced the per-layer teacher-output cache with a per-block Phase C.5 checkpoint (`_stage3_phase_c5_partial/block_{i}.pt`) covering refined `{U_j, V_j}` and updated RMSNorm scales.*
