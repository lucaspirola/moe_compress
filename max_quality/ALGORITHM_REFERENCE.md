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

<!-- 4. Magnitude top-K in l ∈ L (project-original; no paper) — CONSUMED
     by stage1/plugins/magnitude_topk.py (commit pending). Full
     project-original rationale + K=16=2×top-routing derivation + v3
     Phase F motivating evidence (L34E85 ΔNLL ≈ −0.025) now live in the
     plugin docstring. -->

**De-duplication** is by `(layer, expert)` pair; provenance is the union of all sources that flagged the candidate. Final filter is Phase D's ablation pass.

**Empirical scale (paper 2507.23279):** SEs account for fewer than 0.5% of all experts across the MoE models studied (Table 1: 0.05% for Qwen3-30B-A3B, 0.06% for DeepSeek-R1, 0.11% for DeepSeek-V2-Lite-Chat, 0.39% for Mixtral-8x7B-Instruct-v0.1). Source 1 alone reproduces the paper's canonical SE set on Qwen3-30B-A3B (Table 2: L1E68, L2E92, L3E82); v6 broadens the candidate pool with sources 2-4 to catch architecture-shifted SEs that the static three-way AND threshold misses on Qwen3.6.

<!-- §4 Phase D (Ablation Filter) — CONSUMED by stage1/plugins/ablation_filter.py
     (commit pending). Full procedure (held-out slice / baseline /
     per-candidate ablation / threshold filter), cost estimate, the v4
     158→5 motivation, and the bs=32→8 job-6a00caf0 OOM archaeology now
     live in the plugin module docstring. -->

<!-- §4 Phase E (CKA Distance Matrices) — CONSUMED by
     stage1/plugins/cka_distance.py (commit pending). Full distance-form
     definition (D^l_{ij} = 1 − CKA), GRAPE-line-245 similarity-vs-distance
     citation, D-cka-distance sign-flip derivation, Kornblith CKA primitive
     citation + official-code SHA, GPU/CPU implementation paths, and the
     reservoir-sampling rationale (256/expert from ~16K routed tokens) now
     live in the plugin module docstring. SHARED notes: Phase F's downstream
     argmin / R^l consumption of these matrices remains documented under
     §4 Phase F + the D-cka-distance §12 row is also consumed below. -->

<!-- §4 Phase F (GRAPE Budget Allocation) — CONSUMED by
     stage1/plugins/grape_merge.py (commit pending). Full Algorithm 1
     transcription, all five GRAPE-specific deviations (D3 γ default,
     D4 D^l update, D5 floor, D-grape-restart-merge two-restart-paths,
     D-se-blacklist-merge SE integration), the key-formulas block (R^l
     / R̃^l polarity derivation / Ê / E), the correctness notes
     (full-row/col-zero vs paper-line-9 derivation), and the Resume
     section now live in the plugin module docstring. -->

### Blacklist Output (`stage1_blacklist.json`)

Stage 1 emits `stage1_blacklist.json` containing the ablation-validated Super Expert blacklist — `(layer, expert)` index pairs whose Phase D ΔNLL exceeded `ablation_filter_threshold`. The candidate pool from Phase C (which can include three-way AND, AIMER, sink-token, and magnitude-top-K provenance) is recorded separately under `aimer.candidates`, `sink_token.candidates`, and `magnitude_topk.candidates` for audit; the per-candidate ΔNLL is in the companion `stage1_ablation_filter.json`. Final blacklist size is bounded by both the candidate pool and the threshold cut — typical Qwen3.6 runs produce 10–30 ablation-validated SEs out of 50–100 candidates, well below the paper's < 0.5% empirical scale.

Shared experts (`mlp.shared_expert`) are **not in the blacklist** and are not processed by Stage 1 at all. They live in a separate model attribute, distinct from the routed `mlp.experts` list, and are architecturally invisible to `iter_moe_layers`, GRAPE, and REAM. No explicit exclusion is needed — they are simply never candidates.

**What GRAPE contributes to Stage 2 is per-layer budgets (N'_l), not individual expert blacklists.** N'_l ≥ `min_experts_per_layer` (128) for every layer due to the floor constraint. Stage 2 uses N'_l as the target centroid count for REAM per layer; REAP scores then determine which N'_l routed non-blacklisted experts become centroids.

> Stage 1 spec touch-up 2026-05-07 (iter+3): clarified Phase F entropy-recompute scope (only c_{l*} changes per iter); §4 Phase E citation §3.2 → §3.3; GRAPE K vs effective_budget cross-link in Phase F step 2; minor wording polish on Mixtral percentage and first-MoE-layer scope. (Phase letters updated 2026-05-10 v6 rename: pre-v6 D/E referred to CKA/GRAPE, now E/F.)

---

<!-- §5 Stage 2 (REAP Scoring + REAM Pseudo-Pruning) — CONSUMED by
     stage2/plugins/* (commits pending). Full §5 narrative —
     papers (REAP arXiv:2510.13999 + REAM arXiv:2604.04356),
     hardware framing, sequential profiling with early-exit
     optimization, blacklisted-expert exclusion, Step 1
     (REAP Eq. 9 scoring), Step 2 (REAM Eqs. 5/7/8 cost matrix),
     Step 3 (greedy pseudo-pruning + feasibility/quality bumps),
     Step 4 (Eq. 6 freq-weighted merge + Hungarian alignment),
     Step 5 (router resize), covariance side-collection, budget
     bump loop, resume, and Stage 2 v2 opt-in feature catalog —
     all relocated into the per-plugin module docstrings under
     stage2/plugins/. -->

---

<!-- §5.5 Stage 2.5 (Post-Merge Router Calibration) — CONSUMED by
     router_kd/plugins/* (commits pending). Full §5.5 narrative —
     paper (Router-KD arXiv:2603.02217), Stage-2.5-vs-Stage-5
     parameter table, calibration deviation D11 (SHARED — owner
     stage2/plugins/reap_scoring.py), teacher loading fallbacks —
     all relocated into router_kd/plugins/* module docstrings. -->

---

<!-- §8 Stage 5 (Router Knowledge Distillation — Final) — CONSUMED by
     router_kd/plugins/* (commits pending). Full §8 narrative —
     paper (Router-KD arXiv:2603.02217 §F.3 Eq. 3 Table 1), 4-bit
     teacher fallback, teacher-logits cache (SHA-256 key), KL
     direction proof, frozen-scope clauses, batch-size table, vocab
     guard, hyperparameter table, calibration deviation D11 (SHARED)
     — all relocated into router_kd/plugins/* module docstrings. -->

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

<!-- D3 (γ entropy tolerance default) — CONSUMED by stage1/plugins/grape_merge.py (commit pending). -->
<!-- D4 (D^l update zeros full row/col vs paper line 9 pair-entry) — CONSUMED by stage1/plugins/grape_merge.py (commit pending). -->
<!-- D5 (per-layer floor with no layer bonuses) — CONSUMED by stage1/plugins/grape_merge.py (commit pending). -->
<!-- D-cka-distance — CONSUMED by stage1/plugins/cka_distance.py (commit
     pending). Full sign-flip derivation (argmin distance ↔ argmax
     similarity; R^l_dist = N(N−1) − R^l_sim under i ≠ j; R̃^l polarity
     inversion) now lives in the plugin module docstring. SHARED note:
     Phase F's grape_merge plugin also depends on this deviation for the
     argmin layer/pair selection; re-check at the §4 Phase F consumption
     step. -->
<!-- D-ma-detector items (1)–(2) — dynamic detector thresholds + 0.75-depth
     fallback — CONSUMED by stage1/plugins/ma_detection.py (commit pending).
     Items (3) CKA reservoir cap → SHARED with cka_distance; (4) phase_b /
     ablation batch sizes → SHARED with aimer / sink_token / three_way_and /
     ablation_filter; (5) num_calibration_samples → SHARED across Stage 1
     plugins. Resolve items (3)–(5) in Phase 9 when their owning plugins are
     processed. -->
| D-ma-detector (items 3-5 — sampling caps, batch sizes, num_calibration_samples) | 1 | <!-- SHARED — re-check at end --> | <!-- SHARED — items (3)-(5) of the original row remain; items (1)-(2) consumed by ma_detection --> | The 3.0 / 2.0 detector thresholds (item 1) and the 0.75 fallback (item 2) are now documented in the `ma_detection` plugin docstring; this row retains only the cross-plugin sampling items. | Resolve in Phase 9. |
<!-- D-se-blacklist-merge — CONSUMED by stage1/plugins/grape_merge.py (commit pending). -->
<!-- D-grape-restart-merge (lag-corrected post-selection restart) — CONSUMED by stage1/plugins/grape_merge.py (commit pending). -->
<!-- D5a — CONSUMED by stage2/plugins/layer_merge.py (commit pending). -->
<!-- D5b — CONSUMED by stage2/plugins/layer_merge.py (commit pending). -->
<!-- D-ream-aggregation — CONSUMED by stage2/plugins/ream_cost.py (commit pending). -->
<!-- D-ream-similarity-rescale — CONSUMED by stage2/plugins/ream_cost.py (commit pending). -->
<!-- D-ream-sparse-routing — CONSUMED by stage2/plugins/ream_cost.py (commit pending). -->
<!-- D-ream-budget-bump — CONSUMED by stage2/plugins/layer_merge.py (commit pending). -->
<!-- D-reap-routing-weight — CONSUMED by stage2/plugins/reap_scoring.py (commit pending). -->
<!-- D-ream-resume-fallback — CONSUMED by stage2/plugins/ream_cost.py (commit pending). -->
<!-- D-mcf-assignment — CONSUMED by stage2/plugins/solver_hungarian.py + solver_mcf.py + solver_auto.py (commits pending). -->
<!-- D-whitened-cost — CONSUMED by stage2/plugins/ream_cost_post.py (commit pending). -->
<!-- D-asymmetric-freq — CONSUMED by stage2/plugins/ream_cost_post.py (commit pending). -->
<!-- D-capacity-util-gate — CONSUMED by stage2/plugins/capacity_gate.py (commit pending). -->
<!-- D-em-refinement — CONSUMED by stage2/plugins/em_refine.py (commit pending). -->
<!-- D-expert-distill-mse — CONSUMED by stage2/plugins/expert_distill.py (commit pending). -->
<!-- D-expert-distill-mse-v1 — CONSUMED by stage2/plugins/expert_distill.py (commit pending). -->
<!-- D-sinkhorn-soft-assign — CONSUMED by stage2/plugins/solver_sinkhorn.py (commit pending). -->
| D-protocol-blend | 2.5 | Protocol combination: REAM + Router KD in sequence | 2604.04356 (REAM): explicitly evaluates "without any fine-tuning after compression"; 2603.02217 (Router KD): designed as a standalone step, not as a post-REAM patch | Spec applies Router KD (Stage 2.5) immediately after the REAM merge | Router KD restores routing accuracy degraded by weight averaging; REAM's static evaluation does not cover post-merge routing drift. Combined protocol not ablated against REAM-static-only baseline: empirical_pending |
<!-- D6 — CONSUMED by stage3/plugins/covariance_collection.py (commit pending). -->
| D-AASVD-objective | 3 | AA-SVD primary objective variant | 2604.02119 §4.3 Table 5 recommends input-aware (A=B=X, Corollary 3.3 with pre-prune covariance) + block refinement as primary recipe (PPL 6.89 at ρ=0.8 LLaMA-7B) | Spec uses anchored-adaptive (A=X_pre, B=X_post, Theorem 3.2) + block refinement (Path 1). Quality gap ~0.2 PPL at ρ=0.8 on LLaMA-7B; Qwen3-30B-A3B comparison empirical_pending. | Anchored-adaptive is the paper's central theoretical contribution and expected to outperform in high-compression regimes where upstream drift is larger; empirical validation on Qwen3-30B-A3B pending |
<!-- D7 — CONSUMED by stage3/plugins/d_rank_allocate.py (commit pending). -->
<!-- D7a — CONSUMED by stage3/plugins/d_rank_allocate.py (commit pending). -->
<!-- D8 — CONSUMED by stage3/plugins/swift_svd_alpha.py (commit pending). -->
<!-- D-eps-star — CONSUMED by stage3/plugins/swift_svd_alpha.py (commit pending). -->
<!-- D10 — CONSUMED by stage4/plugins/eora_compensation.py (commit pending). -->
<!-- D-eora-budget-pct — CONSUMED by stage4/plugins/eora_compensation.py (commit pending). -->
| D11 | 2, 2.5, 5 | Calibration data source | 2603.02217 §F.3 Table 1: c4; 2510.13999 §4: c4 + evol-codealpaca (used identically across all experiments) | Multi-domain Nemotron-Cascade-2-SFT-Data with weighted subsets (chat 0.56, math 0.21, science 0.11, etc.) | Task-aware calibration better matches target deployment distribution; c4 and evol-codealpaca are general pre-training / instruction-tuning data with limited reasoning/code coverage relative to the target deployment mix |
| D-cal-size | 2 | Calibration sequence count | 2604.04356 §4: 3072 sequences × 512 tokens (1.57M tokens total); 2510.13999: 1024 sequences × 2048 tokens (2.1M tokens total) | 4000 sequences × 2048 tokens (8.19M tokens total) (Nemotron weighted subsets) | Exceeds both papers' calibration volumes (5.2× REAM in tokens, 3.9× REAP in tokens); longer 2048-token sequences match the deployment context length and capture more inter-token routing patterns per sequence. Task-aware Nemotron dataset documented in D11 |
<!-- D-aimer-cross-check — CONSUMED by stage1/plugins/aimer.py (commit pending). -->

<!-- D-sink-token-routing — CONSUMED by stage1/plugins/sink_token.py (commit pending). -->
<!-- D-causal-ablation-validation — CONSUMED by stage1/plugins/ablation_filter.py (commit pending). -->
<!-- D-magnitude-topk-candidates — CONSUMED by stage1/plugins/magnitude_topk.py (commit pending). -->
<!-- D-reap-min-active-tokens — CONSUMED by stage2/plugins/reap_scoring.py (commit pending). -->
<!-- D-cov-storage-fp16 — CONSUMED by stage3/plugins/covariance_collection.py (commit pending). -->
<!-- D-a-max-fraction — CONSUMED by stage1/plugins/three_way_and.py (commit pending). -->

<!-- D-per-type-alpha — CONSUMED by stage3/plugins/swift_svd_alpha.py (commit pending). -->
<!-- D-no-intra-block-cascade — CONSUMED by stage3/plugins/aa_svd_factor.py (commit pending). -->
<!-- D-drank-premerge-A — CONSUMED by stage3/plugins/d_rank_allocate.py (commit pending). -->
<!-- D-c5-moe-only — CONSUMED by stage3/plugins/block_refine.py (commit pending). -->
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
