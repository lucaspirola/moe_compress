# Comprehensive Stage-1 + Stage-2 Plan — Optimizing for Ablation SC

**Repo**: `moe_compress` at `/home/lucas/ai/moe_compress`
**Branch**: `feat/calibration-v2`
**Status**: Spec — produced under ml-intern protocol. NOT implemented.
**Hermetic**: Yes. Any fresh agent can execute this plan from this file alone — all background, paper details, code-inspection findings, prior-conv claims, and companion-plan summaries are embedded inline.
**Date**: 2026-05-27

---

## Table of contents

1. TL;DR
2. Background (you need this if you've never seen `moe_compress` before)
3. The prior ml-intern conversation (2026-05-18) — verbatim excerpts
4. Gap analysis — prior recommendations vs current code
5. 2026 literature anchors — full paper details (R1–R8 + reference papers)
6. The S-series ablation matrix
7. Build order with gates
8. Success criteria
9. Risks + halt-triggers
10. What this plan does NOT do
11. Companion plans, summarized inline
12. Open questions raised to user
13. Code citations (file:line index)
14. Appendix A — Considered-but-rejected directions
15. Appendix B — HF Hub artifact paths

---

## 1. TL;DR

The prior ml-intern conversation (2026-05-18) made **4 recommendations** to improve Stage 1 (per-layer budget allocation) and Stage 2 (merge cost + merge step). **None have been implemented.** Three SC-related plans exist in `tasks/` but they only address Stage 2 *speed* (`SC_FAST_PLAN_V3.md`, `SC_BOTTLENECK_PLAN.md`), a *substrate* (`L1_FOR_SC_PLAN.md`'s vLLM forward), or *data writers* (`CALIBRATION_MIX_V2_PLAN.md`) — not the algorithmic recommendations. A fourth plan (`RKD_AB_PLAN.md`) audits Stage 2.5 specifically and **runs before this plan's S-series**.

A 2026 literature crawl strengthens 4 of the original anchors with strict improvements, validates the 5th, and adds 3 more. Key new findings:

| Prior conv anchor | 2026 strict-improvement | Why strictly better |
|---|---|---|
| EvoESAP (arxiv 2603.06003) evolutionary search | **RCO** (arxiv 2605.00649, IST-DASLab) | +4.5 avg pts at 25% compression on Qwen3-30B; 4× faster; open-source @ github.com/IST-DASLab/RCO |
| MergeMoE (arxiv 2510.14436) closed-form merge | still SOTA — T₁=Q·P† math verified | The per-cluster pseudo-inverse merge step is what Rec 3 proposed |
| Sequential greedy w/ propagation (Rec 4a paraphrased) | **REAM** (arxiv 2604.04356, Samsung SAIL Montreal, Apr 2026) | Open-source @ github.com/SamsungSAILMontreal/ream; **our code already uses REAM terminology (`_ream_cost_matrix`, etc.) but does NOT implement REAM's sequential merging — this is the single highest-leverage gap** |
| HC-SMoE Appendix B.1 (arxiv 2410.08589, ICML 2024) | non-uniform-budget precedent | Validates the per-layer-varying budget premise underlying Rec 2 |
| (untested in prior conv) | **Router KD** (arxiv 2603.02217) | ~2h fine-tune; works particularly well for fine-grained 256-expert MoEs; orthogonal post-merge stage |
| (untested in prior conv) | **HodgeCover** (arxiv 2605.13997) | Tested on **Qwen3.5-35B-A3B (EXACT same arch as our project)** at 66% compression |

This plan proposes a **9-row S-series ablation matrix** that bisects Stage 1 × Stage 2 × Stage 2.5 along the prior recommendations + new findings, ordered by ROI (cheapest-first), gated so cheap rows precede expensive ones, with each row anchored to specific paper(s) or marked as a stack/baseline test.

**Total cost estimate**: ~9 H200-rows at ~3h each ≈ 27h × $3.39/h ≈ **$92** for the S-series proper. Plus ~100h of human integration time (mostly for vendoring RCO, REAM, and writing the MergeMoE merge step). `S_CALIB` adds 1 row (~$10). `S2_GLOBAL` adds ~5h (~$17). **Headline total: ~$120 GPU + ~100h dev.**

---

## 2. Background

### 2.1 Project overview

`moe_compress` is a **model-agnostic post-training compression tool** for sparse Mixture-of-Experts (MoE) Large Language Models. The current test case is **Qwen3.6-35B-A3B**, a fine-grained MoE from the Qwen team. The goal: reduce the parameter count by ~35% via **expert merging** (folding redundant experts into kept "centroid" experts), while preserving generation quality.

The output of the pipeline is a smaller MoE checkpoint that still routes through the standard MoE-block interface (no architectural changes — just fewer experts per layer). The compressed model is then evaluated on a cheap "thermometer" benchmark (WikiText bits-per-token + a few MC tasks) to compare strategies.

### 2.2 Model architecture (Qwen3.6-35B-A3B)

- **Total parameters**: ~35 B
- **Active parameters per token**: ~3 B (the "A3B" suffix)
- **Layers**: 40 MoE layers (each with attention + MoE-FFN block)
- **Experts per layer**: 256
- **Top-k routing**: 8 (each token activates 8 experts per layer)
- **Hidden dim**: 2048
- **MoE intermediate size**: 512 (per expert)
- **Expert structure** (per the HF `Qwen3MoeExperts` layout in `transformers >= 5.x`):
  - `gate_up_proj[E, 2I, H]` — fused gate+up projection (`I=512`, `H=2048`, `E=256`)
  - `down_proj[E, H, I]` — down projection
- **SwiGLU activation**: `down(silu(gate(x)) * up(x))`
- **Routing**: linear router → softmax/top-k → weighted sum of activated experts

At 256 experts × 40 layers, naive iteration is expensive — this informs many of the cost/budget decisions below.

### 2.3 Compression pipeline (4 stages)

```
Stage 1 (GRAPE)        →   Stage 2 (REAP+REAM)      →   Stage 2.5 (router KD)   →   Eval (thermometer)
budget per layer +          centroid selection +          fine-tune router only        bpt_gap on WikiText
super-expert detection      child→centroid merging        on calibration data          + ARC-E + HSwag + top1
+ ablation filter
```

#### Stage 1 — GRAPE (`stage1/orchestrator.py`, `stage1/plugins/grape_merge.py`)

- **Phase A** — MA-formation layer detection (which layers are merge-amenable)
- **Phase B** — profiling (per-expert outputs + routing stats from calibration forward)
- **Phase C** — redundancy/similarity computation (CKA on expert outputs)
- **Phase D** — ablation filter (`stage1_ablation_filter.py`): zeroes each candidate expert's `down_proj` output and measures ΔNLL on a held-out corpus — a behavioral damage measure
- **Output**:
  - `stage1_blacklist` (super-experts protected from merging, via three-way AND of activation + magnitude + sink-token criteria)
  - `per_layer_target_experts` (how many experts to keep per layer)
  - The current allocator (`_grape_greedy_merge` in `stage1/plugins/grape_merge.py`) is a **greedy knapsack** over `R[li]` = sum of off-diagonal CKA distances in layer li, modulated by an entropy gate (`gamma`, `_entropy`) that trades merge damage for routing-distribution balance
- **Inert hook**: `merge_cost_prior` (`grape_merge.py:171–334`) accepts `{layer_idx: float}` and would bias selection toward `R[li] · merge_cost_prior[li]` instead of `R[li]` alone — but defaults to `None` and no caller populates it. **This is where Rec 2's damage-curve DP would land.**

#### Stage 2 — REAP + REAM (`stage2/orchestrator.py`, `stage2/plugins/`)

- **REAP scoring** (`stage2/plugins/reap_scoring.py`) — within-layer importance signal `g_j(x) · ‖f_j(x)‖₂` (router-weighted activation norm); selects top-N "centroid" experts to keep
- **REAM cost matrix** (`stage2/plugins/ream_cost.py`, `ream_cost_post.py`, `output_space_cost.py`) — per-pair cost between each non-centroid (child) and each centroid candidate. Three modes via `cost_alignment`:
  - **`pre`** (default) — weight-space symmetric δ_REAM cost `1 − (δ_gate + δ̃_expert)/2`, blending gate-logit similarity and gated-output cosine similarity from profiling stats. **Cheap** (closed-form, no per-pair forward).
  - **`post`** — post-alignment whitened residual: per-child, top-K cheap candidates, then per-pair Hungarian neuron-permutation alignment + whitened Frobenius residual. **Medium cost.**
  - **`output`** (**= SC**) — for each child m and its top-K cheap-cost candidate centroids, tentatively merge, run the merged expert on calibration tokens, and measure the routing-weighted MSE of the layer's gated routed-output change: `mean_t σ_m(x_t)·‖E_m − E_merged‖²`. **Expensive** (~3 h/row on Qwen3.6-35B-A3B).
- **Assignment solvers** (`stage2/plugins/solver_*.py`) — once the cost matrix is built, assign each child to one centroid:
  - `greedy` (default) — centroid-order single pass, each slot takes cheapest unassigned child
  - `hungarian` — rectangular optimal 1-1 via `scipy.linear_sum_assignment`
  - `mcf` — min-cost-flow with true per-centroid capacity (ortools)
  - `auto` — `hungarian` if `n_children ≤ n_centroids`, else `mcf`
  - `sinkhorn` — capacitated entropy-regularized OT, ε-annealed
- **Merge step** (`stage2/merging.py:_merge_experts_inplace`) — given the assignment, compute the merged centroid weights. Currently: **frequency-weighted averaging + permutation alignment** (via `_tentative_merged_weights` in `output_space_cost.py:212`). **This is where Rec 3 (MergeMoE T₁=Q·P†) would land.**
- **Refinements** (off by default; mostly investigated as Directions B/D/E in the historical S-sweep):
  - Direction B — skip-merge floor (mask costliest merges)
  - Direction D — 2-opt refinement (`two_opt_refine.py`)
  - EM refinement (`em_refine.py`) — re-solve after substituting tentative merged centroids
  - Expert distillation per-merge-group (`expert_distill.py`)

#### Stage 2.5 — Router KD (`router_kd/`, `stage5_router_kd.py`)

- **Trainable**: only router weights (`_freeze_non_routers`, `trainable_name_patterns`); all expert weights frozen
- **Loss**: temperature-scaled vocab-KL between student and teacher final logits, computed in sequence chunks (`_chunked_vocab_kl`, `kd_seq_chunk_size=32` post the 2026-05-19 OOM fix at `kd_seq_chunk_size=512`)
- **Optimizer**: AdamW, weight_decay=0.01
- **Temperature τ**: constant 1.0 (a ramp 4.0 → 1.0 was removed on 2026-05-19 because under a ramp the logged raw_kl drifts with τ and is no longer a faithful teacher↔student signal — it corrupted save-best)
- **Recipe** (current config): epochs=1, ~375 steps, calibration from `nvidia/Nemotron-Cascade-2-SFT-Data`, save-best, teacher-logits cache, early_stop_patience=8
- **Caveat** (per `MOE_COMPRESS_REPORT.md` §3): `epochs=1` / `weight_decay=0.01` / `early_stop_patience=8` were tuned against the now-discredited ramp-era reading; a re-audit (epochs=2 diagnostic at T=1, watching the honest KL) is pending. **`RKD_AB_PLAN.md` does this audit.**
- **Direction E (`merge_repair`)**: optional path that ALSO unfreezes the merged centroid experts and adds a per-layer MSE term between student/teacher MoE-block outputs on merge-affected layers. Failed in the historical SE row due to optimizer-state OOM on H200.

#### Eval — thermometer (`stage6alt/`, `stage6alt_thermometer.py`)

- **`bpt_gap`** = student_bpt − teacher_bpt on WikiText (lower is better; lower = less compression damage)
- **`top1_agreement`** — fraction of tokens where student top-1 matches teacher top-1
- **ARC-Easy** (multiple choice, commonsense reasoning, ~2400 samples)
- **HellaSwag** (multiple choice, situation completion, ~10k samples)
- All metrics cached against a constant teacher (`teacher_bpt = 2.7242` for Qwen3.6-35B-A3B)

### 2.4 Terminology cheat sheet

| Term | Meaning |
|---|---|
| **MoE** | Mixture of Experts — each layer has N "experts" and a "router" that picks top-k per token |
| **Expert** | A small FFN sub-network (gate / up / down projections + SwiGLU) |
| **Router** | The linear+softmax+top-k selector that routes each token to k experts |
| **Centroid** | An expert that is **kept** after compression |
| **Child** | An expert that is **merged into** a centroid (loses its identity) |
| **GRAPE** | The project's Stage 1 budget allocator (CKA-redundancy + entropy gate). Acronym source not material; treat as a proper noun. |
| **REAP** | Router-weighted Expert Activation Pruning — within-layer importance signal (arxiv 2510.13999). The paper itself proposes pruning, not merging; the project uses the SCORING from REAP for centroid selection then folds the rest via REAM. |
| **REAM** | The merge cost framework in this project (`pre`/`post`/`output` cost modes) + assignment solver dispatch. Project's internal name; aligned with the Samsung "REAM" paper (arxiv 2604.04356) which adds sequential merging on top. |
| **SC** | Strategy C = `cost_alignment="output"` — the prior best result at bpt_gap = 0.1293 |
| **SCD** | SC + Direction D (2-opt refinement) — regressed to 0.1868 (the SCD regression evidences local-vs-global cost gap) |
| **SCD regression** | The observation that 2-opt refinement *lowers* total assignment cost on the cost matrix yet the *model* gets worse — evidence that even the output-space cost is an imperfect proxy for end-to-end loss |
| **Direction A/B/C/D/E** | Historical strategy-sweep directions: A = damage-aware budget retune, B = skip-merge floor, C = output-space cost, D = 2-opt refinement, E = merge-repair |
| **S-rows** | S0, SA, SAB, SC, SCD, SE — the historical strategy sweep rows. **New S-rows in this plan** = the proposed S1_DP / S1_RCO / S2_MM / S2_SEQ / S2_MM_SEQ / S2_GLOBAL / S5_RKD / S_BEST / S_CALIB. |
| **`bpt_gap`** | Bits-per-token gap (student − teacher) on WikiText — the primary thermometer metric |
| **MA-formation** | "Merge-amenable formation" — Stage 1's classification of which decoder layers are amenable to merging |

### 2.5 Current production state — strategy sweep S0–SE

From `MOE_COMPRESS_REPORT.md` §5.2 (results in HF bucket `pirola/moe-strategy-35pct`):

| rank | row | direction | bpt_gap | top1_agree | student_bpt | ARC-E | HSwag |
|---|---|---|---|---|---|---|---|
| 1 | **SC** | **C — output-space merge cost** | **0.1293** | 0.843 | 2.8535 | 0.74 | 0.735 |
| 2 | SCD | C + D (2-opt refinement) | 0.1868 | 0.825 | 2.9110 | 0.74 | 0.725 |
| 3 | S0 | baseline (greedy, all-off) | 0.5767 | 0.735 | 3.3009 | 0.71 | 0.715 |
| 4 | SA | A — damage-aware budget retune | 0.7688 | 0.686 | 3.4930 | 0.71 | 0.590 |
| 5 | SAB | A + B (skip-merge floor) | 0.7808 | 0.684 | 3.5050 | 0.66 | 0.605 |
| — | SE | E — merge-repair | **FAILED (OOM)** | — | — | — | — |

(teacher_bpt = 2.7242 for all rows. S0/SA used `floor_divisor=4` for the budget retune; S0 itself is plain greedy. SA/SAB share Stage 1 with S0 and consume the budget-retune output.)

**Key findings from this sweep:**

1. **Direction C (output-space merge cost) is the decisive win**: bpt_gap 0.5767 → 0.1293 (~4.5×), and it beats baseline on *every* metric (top1_agreement, ARC-E, HellaSwag all up together). The other rows all use the same greedy solver — **the win is the cost function, not the solver.**
2. **Directions A and A+B HURT** — worse than the trivial greedy baseline (0.77/0.78 vs 0.58). The damage-aware budget retune (against *weight-space* damage) and the skip-merge floor both regressed.
3. **Direction D adds nothing / slightly hurts**: SCD (C+D) 0.1868 vs SC 0.1293. 2-opt provably lowers total assignment cost yet the *model* got worse — the **SCD regression**.
4. **Direction E could not be measured** — `_chunked_vocab_kl` initially OOM'd at the original `kd_seq_chunk_size=512`; chunk size was reduced to 32 to recover SC and SCD, but SE additionally OOMs on the Adam optimizer state (unfrozen merged centroids).

**Cross-cutting conclusion (per `MOE_COMPRESS_REPORT.md` §5.4):**
- **The lever is the cost, not the solver.** A faithful (output-space) merge cost beat the baseline ~4.5×; harder solvers/refinements on top did not help.
- **The output-space cost is still an imperfect proxy** — *locally* faithful (single-layer routed-output residual) but the SCD regression suggests it is not *globally* faithful (end-to-end). A downstream/end-to-end damage measure is the principled next step.
- **The same weight/structural-proxy weakness exists in Stage 1's per-layer budget allocation**: GRAPE allocates budget by CKA-redundancy + entropy gate that is explicitly *not* damage-minimizing. The `merge_cost_prior` hook to feed measured damage in already exists but is inert.

### 2.6 Calibration data

Currently: `nvidia/Nemotron-Cascade-2-SFT-Data` (instruction-format SFT corpus). This is what feeds the calibration forward pass in Stages 1 and 2 (and Stage 2.5's KD).

`MOE_COMPRESS_REPORT.md` notes the prior runs may have been confounded by calibration choice. The 2026 SOTA literature uses different mixtures (notably REAM's best result was on `C4 : NuminaMath : The-Stack-Smol = 0:0.5:0.5`). **This plan deliberately holds calibration data fixed across the S-series** to isolate the algorithmic variable; the calibration probe is `S_CALIB`, run only after the algorithmic winner is locked.

---

## 3. The prior ml-intern conversation (2026-05-18) — verbatim excerpts

The prior conversation is at `/tmp/ml-intern-sessions/sessions/2026-05-18/b9ab0480-output-space-only.jsonl` (33 messages, ~160 KB). The full synthesis is at msg[30]; a clarifying follow-up at msg[32].

### 3.1 The user's framing (msg[0])

> *"everything in your memory now is old. I worked a lot on our repo. read https://huggingface.co/datasets/pirola/moe-compress/blob/main/tasks/MOE_COMPRESS_REPORT.md to catch up. After many tests, we found out that the standard merging decision is not the best: we got much better results when the cost is the output-space. The topic of research now is how do we improve stage 1 and stage 2?"*

### 3.2 The 4 recommendations (verbatim from msg[30])

> ### The Core Insight (confirmed by literature)
>
> Your finding that **output-space cost is the decisive lever** is strongly validated by multiple recent papers:
> - **HC-SMoE** (ICML 2024): showed output-based clustering massively outperforms weight-based and router-logit-based clustering
> - **MergeMoE** (2025): formalizes expert merging as an **output-space optimization** with compression matrices T₁, T₂, T₃ solved via least-squares — and beats all baselines on Qwen3-30B
> - **REAP** (2025): shows that `g_j(x) · ‖f_j(x)‖₂` (router-weighted activation norm) is the right importance signal — not frequency, not weight-space metrics
> - **EvoESAP** (June 2026): decouples **within-layer ranking** from **across-layer budget allocation** and uses evolutionary search with a global proxy (ESAP) to find non-uniform budgets — up to +19.6% on MATH-500
>
> ---
>
> ### Stage 1 Improvements: Budget Allocation
>
> **Problem**: GRAPE allocates per-layer budgets using CKA-redundancy + entropy gate. This is a structural/weight-space proxy that doesn't reflect actual merge damage. The `merge_cost_prior` hook exists but is inert.
>
> **Recommendation 1: EvoESAP-style evolutionary budget search with output-space fitness**
>
> The key idea from EvoESAP: decouple the problem into:
> 1. **Within-layer merge ordering** (your output-space cost already handles this well)
> 2. **Across-layer budget allocation** (currently GRAPE's CKA knapsack — the weak link)
>
> Replace GRAPE's greedy CKA knapsack with an evolutionary search:
> - **Search space**: integer vectors `r = (r₁, ..., r_L)` where `rₗ` = experts to keep in layer ℓ, subject to `Σrₗ` = global budget
> - **Mutation**: budget-preserving "level-switch" — move Δ merge slots from layer a to layer b
> - **Fitness function**: Instead of EvoESAP's ESAP (designed for pruning), use **bpt_gap on a small held-out set** (you already have this as your thermometer). Or cheaper: **sum of per-layer output-space merge costs** for the top-K merges in each layer under allocation r — this is fast because you can precompute per-layer merge-cost curves.
>
> This directly addresses the `merge_cost_prior` hook: rather than biasing the greedy knapsack by a prior, you **search** for the budget that minimizes actual merge damage.
>
> **Recommendation 2: Per-layer merge-damage curves as GRAPE replacement**
>
> Simpler than full evolutionary search:
> 1. For each layer, compute the output-space cost of merging the cheapest 1, 2, 3, ... K experts (using your existing Direction C `_output_space_cost`)
> 2. This gives you a **damage curve** `D_ℓ(k)` per layer — how much output damage from merging k experts in layer ℓ
> 3. The optimal budget is then a knapsack: minimize `Σ D_ℓ(k_ℓ)` subject to `Σ k_ℓ = global_budget`
> 4. This is a **1D dynamic programming** problem — trivial to solve exactly
>
> This is essentially what `merge_cost_prior` was designed for, but computed with the **output-space** cost rather than weight-space.
>
> ---
>
> ### Stage 2 Improvements: The Cost Function
>
> **Problem**: Output-space cost works great but: (a) it's expensive (~3h/row), (b) it's still locally imperfect (SCD regression when optimized harder).
>
> **Recommendation 3: MergeMoE-style output-merging optimization**
>
> Instead of: merge by weighted averaging then measure output MSE...
> MergeMoE's insight: solve for **optimal compression matrices** T₁ via least-squares that directly minimize output reconstruction error. Specifically:
> - Concatenate group experts' weights to form the "wide" merged expert
> - Solve T₁ = Q·P† (pseudo-inverse) where Q = original expert outputs, P = concatenated merged outputs
> - This is a **closed-form per-group optimization** that's faster than your current Hungarian permutation approach
>
> The key difference from your current Direction C: you currently measure the *damage* of a naive merge and use that as a cost signal for assignment. MergeMoE instead **optimizes the merge itself** to minimize output error. These are complementary — you could:
> 1. Use output-space cost for assignment (Direction C, keep as-is)
> 2. Then use MergeMoE-style least-squares merge instead of frequency-weighted averaging for the actual merge step
>
> **Recommendation 4: Address the local→global gap (SCD regression)**
>
> The SCD regression tells you: minimizing per-layer output MSE doesn't minimize end-to-end loss. Papers suggest two approaches:
>
> **(a) Sequential greedy with global feedback** (cheapest):
> - Merge layer-by-layer in order (or by damage-ascending order)
> - After merging each layer, **re-evaluate downstream layers' costs** on the updated model
> - This captures cascading effects — a merge in layer 5 changes the input distribution to layer 6
>
> **(b) EvoESAP-style ESAP as a global fitness**:
> - After tentative assignment per layer, compute ESAP (speculative acceptance proxy) or just bpt_gap on a small sample
> - Use this as the **acceptance criterion** for whether a proposed merge improves things globally
> - This is more expensive but directly measures end-to-end quality
>
> **(c) Iterative merge-and-measure** (most expensive but principled):
> - Do one merge pass (greedy with output-space cost)
> - Measure global damage
> - Re-run cost computation on the post-merge model for remaining layers
> - Repeat until budget exhausted

### 3.3 HC-SMoE caveat (verbatim from msg[32])

> *"None of them directly. HC-SMoE's contribution is using **hierarchical clustering on expert outputs** as the similarity metric for grouping — which validates the same principle you already discovered (output-space > weight-space), but their actual method (agglomerative clustering with average linkage, frequency-weighted merging) is less sophisticated than what you already have in Stage 2.*
>
> *Their non-uniform budget experiment (Appendix B.1) is relevant but crude: they just keep the top-r% most frequently activated experts globally and let that determine per-layer counts. EvoESAP's evolutionary search (Rec 1) is a much more principled version of the same idea.*
>
> *So HC-SMoE's role in my synthesis was **validation** ("output-based similarity is the right signal") rather than a source of novel technique. Your Direction C already subsumes their core insight and goes further (routing-weighted MSE > raw output cosine similarity)."*

---

## 4. Gap analysis — prior recommendations vs current code

Verified by direct code inspection on the current `feat/calibration-v2` branch:

| Prior conv recommendation | Status in current code | Inert-hook check | Verdict |
|---|---|---|---|
| **Rec 1** EvoESAP evolutionary budget search (Stage 1) | NOT implemented | n/a | Open. Now superseded by **RCO** per arxiv 2605.00649. |
| **Rec 2** Per-layer damage curve `D_ℓ(k)` + DP knapsack (Stage 1) | NOT implemented; `merge_cost_prior` hook at `stage1/plugins/grape_merge.py:171-334` defaults to `None` | **Inert** — code accepts the dict but no caller populates it | Open. The hook will accept `{layer_idx: prior}` from a precomputed curve; DP is a tiny addition. |
| **Rec 3** MergeMoE T₁=Q·P† merge step (Stage 2) | NOT implemented; `_tentative_merged_weights` at `stage2/plugins/output_space_cost.py:212` and `merging.py:_merge_experts_inplace` at `:33` both do **frequency-weighted averaging + permutation alignment** | n/a | Open. Closed-form math; per-cluster O(d²); no training. |
| **Rec 4a** Sequential greedy with downstream propagation (Stage 2) | NOT implemented. The current driver does **one** profile-pass forward pre-Stage-2 and never re-runs after a layer's merge | `_profile_layer` at `stage2/profiling.py:86-286` runs against the unmerged base for all 40 layers | Open. **This is exactly REAM (arxiv 2604.04356)** — and our code already uses REAM terminology, just stops short of the sequential loop. |
| **Rec 4b** ESAP-style global fitness as acceptance gate (Stage 2) | NOT implemented | n/a | Open. ESAP designed for pruning; need a merging-flavoured variant. **`L1_FOR_SC_PLAN.md` Phase 2.5 covers the same surface via vLLM rollout cost — that plan is the right vehicle.** |
| **Rec 4c** Iterative merge-and-measure (Stage 2) | NOT implemented | n/a | Open. **Subsumed by REAM sequential merging (Rec 4a's full form).** Treat as a single line of work. |

**Conclusion**: every prior-conv recommendation remains a real open lever. **None** of the three other in-progress SC plans cover them — those plans are about speed (`SC_FAST_V3`, `SC_BOTTLENECK`), data writers (`CALIBRATION_MIX_V2`), or the vLLM substrate (`L1_FOR_SC`). They are **complements**, not substitutes, to this plan.

---

## 5. 2026 literature anchors — full paper details

Each anchor below is grounded in either the prior-conv synthesis or the 2026-05-27 ml-intern research crawl (50+ tool calls via the Tier-B research subagent). Numbers are paper-reported; "Qwen3-30B" in these results means Qwen3-30B-A3B (128 experts), which is the closest publicly-evaluated relative to our Qwen3.6-35B-A3B (256 experts). HodgeCover (R6) is the exception — it benchmarks on Qwen3.5-35B-A3B (the immediate predecessor with 256 experts).

### 5.1 R1 — MergeMoE (arxiv 2510.14436, Oct 2025)

- **Title**: *MergeMoE: Efficient Compression of MoE Models via Expert Output Merging*
- **Slot**: Stage 2 merge step
- **Used by**: row **S2_MM** (and stacked in **S2_MM_SEQ**, **S_BEST**)
- **Method (§3-4 of paper)**: Recast expert merging as **output-space optimization** via compression matrices T₁ (down-projection compression), T₂ (gate-projection compression), T₃ (up-projection compression).
  - **T₂, T₃** = frequency-weighted parameter average within cluster (closed form from Theorem 1: `w_{ij} = f_j / Σ_{k∈Ci} f_k`)
  - **T₁ = Q · P†** (Moore-Penrose pseudoinverse) where:
    - `P = σ(T₂·W'_G·X̂) ⊙ (T₃·W'_U·X̂)` (the activations *after* the new T₂/T₃ on calibration sample X̂)
    - `Q = σ(W'_G·X̂) ⊙ (W'_U·X̂)` (the original activations)
  - Single-batch closed-form per cluster; no training
- **Clustering**: cosine similarity of concatenated `[W_U; W_G]` matrices for expert grouping (this plan does not use MergeMoE's clustering — we use SC's output-cost assignment, which is stronger per §2.5)
- **Result on Qwen3-30B-A3B at 25% compression (128→96 experts)** (paper Table 1):
  - **+1.71 pt avg over M-SMoE** (73.95 vs 72.24)
  - WinoGrande: 73.72 vs 74.27 (full)
  - ARC-c: 63.48 vs 67.49 (full)
  - HellaSwag: 74.93 vs 76.38 (full)
  - PIQA: 81.34 vs 81.72 (full)
- **Validation for 256-expert scale**: paper does not test 256-expert models, but HodgeCover (R6) uses exactly Qwen3.5-35B-A3B (256 experts) and benchmarks REAP/REAM/MC-SMoE on it — confirming the 256-expert setting is tractable for calibration-based merging. MergeMoE's math generalizes directly: T₁/T₂/T₃ framework applies to any expert count.
- **Compute cost (paper)**: per-cluster pseudoinverse is O(d²) where d = `moe_intermediate_size = 512` for our model. Per-layer adds ~5 min on top of the existing SC cost-matrix loop.
- **Code**: No public release. Math is fully specified in paper §4.
- **HF Hub**: No dataset linked.
- **Why "strict improvement" for Rec 3**: the prior conv's Rec 3 *was* exactly this — paper math fully verified by the research subagent.

### 5.2 R2 — REAM (arxiv 2604.04356, Apr 2026)

- **Title**: *REAM: Merging Improves Pruning of Experts in LLMs*
- **Authors**: Samsung SAIL Montreal
- **Slot**: Stage 2 propagation — the **sequential** merge-and-re-profile loop
- **Used by**: row **S2_SEQ** (and stacked in **S2_MM_SEQ**, **S_BEST**)
- **Method (paper §3-4)**:
  - **Similarity metric**: δ_REAM(i,j) = δ_gate(i,j) + gated-output cosine similarity
  - **Pseudo-pruning**: top-N' experts become centroids (by REAP saliency); non-centroids absorbed (≤C=16 or 32 per centroid)
  - **Permutation alignment**: combined cost = `C_act + C_wt` (activation + weight L2 distances), Hungarian matching
  - **Sequential merging**: after merging layer ℓ, run a second forward pass to recompute activations for layer ℓ+1 — this is the **key delta vs everything else in 2025-26**. Adds ~50% overhead (1.5h vs 1h for 30B model per their wall-clock report).
  - **Calibration data sensitivity is the single biggest lever**: C4-heavy → MC gain, GEN collapse; code-heavy → code GEN +40 pt
- **Result on Qwen3-30B-A3B-Instruct-2507 at 25% compression (128→96)** (paper Table 2):
  - **REAM (best calib: C4:Math:Code = 0:0.5:0.5) GEN avg 69.8** vs HC-SMoE 67.4 vs REAP 68.6 vs Freq 67.6
  - IFEval: 89.9
  - AIME25: 60.0
  - GSM8K: 86.3
  - GPQA: 38.4
  - HumanEval: 93.3
  - LiveCodeBench: 51.0
  - At 50% compression (128→64): REAM best GEN 57.1 with 0:0.3:0.7 mixture
- **What made it work** (per paper §5):
  1. Sequential merging closes local→global gap by propagating compressed layer outputs to next layer **before** computing its statistics
  2. Combined activation+weight permutation alignment outperforms either alone
  3. Calibration data mixture is a task-alignment lever HC-SMoE can't use (clustering is data-invariant)
- **Code**: **OPEN-SOURCE** at https://github.com/SamsungSAILMontreal/ream (15 stars at time of crawl). Look for `sequential_merging` flag.
- **Why "strict improvement" for Rec 4a/4c**: prior conv's Rec 4a was paraphrased "sequential greedy with global feedback"; REAM is exactly that with an open codebase to vendor from.
- **Project note**: our codebase **already uses REAM terminology** (`ream_cost_matrix`, `ream_cost.py`, `ream_cost_post.py`, `δ_REAM` cost) but does NOT yet implement REAM's sequential merging itself. This is the **single highest-leverage algorithmic gap.**

### 5.3 R3 — RCO (arxiv 2605.00649, May 2026)

- **Title**: *Model Compression with Exact Budget Constraints via Riemannian Manifolds*
- **Authors**: IST-DASLab
- **Slot**: Stage 1 budget allocation
- **Used by**: row **S1_RCO** (and stacked in **S_BEST**)
- **Method (paper §3, Algorithm 1)**:
  - Per-group logits α_ik in softmax-relaxed space on budget manifold `C(α) = B` (smooth constraint surface)
  - **Forward**: Gumbel-STE + DP solves multiple-choice knapsack per step → exact budget **discrete** assignment
  - **Backward**: tangent projection removes constraint-normal gradient component → budget stays *exactly* satisfied at every step
  - **Momentum transport**: Adam step + binary-search retraction back onto manifold
  - **Optimizer**: Adam with cosine temperature annealing on Gumbel τ
  - **Initialization**: from REAP saliency scores → fine-tunes the allocation jointly
- **Result on Qwen3-30B-A3B at 25% pruning**:
  - **RCO 71.0 Avg** vs EvoESAP 66.5 vs REAP-uniform baseline
  - At 50% pruning: RCO maintains +2.0 pt advantage
  - **Wall-clock: ~85 min vs 5.2h for EvoESAP at same compute** (4× speedup)
- **On Qwen3-Coder-Next (512 experts) at 50% sparsity**: 97% HumanEval (vs 55% for uniform allocation) — demonstrates the gradient-based approach scales to bigger expert pools.
- **Why "strict improvement" for Rec 1**: prior conv's Rec 1 anchor was EvoESAP (arxiv 2603.06003); RCO **literally benchmarks against EvoESAP** and shows +4.5 avg pts at 4× lower compute. It is the same conceptual move (jointly optimize per-layer budget under a global constraint) with strictly better solver geometry.
- **Code**: **OPEN-SOURCE** at https://github.com/IST-DASLab/RCO. Algorithm 1 is the full implementation.
- **Fitness signal we'll use**: output-space MSE on calibration (the SC cost we already compute), OR — per paper §4.2 — actual model loss (which requires the L1/vLLM substrate; see `L1_FOR_SC_PLAN.md`).

### 5.4 R4 — Additivity theorem (arxiv 2308.10438, Aug 2023)

- **Title**: *Efficient Joint Optimization of Layer-Adaptive Weight Pruning in Deep Neural Networks*
- **Slot**: formal basis for **DP-knapsack** approach to Stage 1 budget allocation
- **Used by**: row **S1_DP**
- **Theorem** (paper Theorem 1): under Taylor + i.i.d. perturbation assumptions,
  ```
  E[‖f(x; W) − f(x; W̃)‖²] ≈ Σᵢ E[δᵢ]
  ```
  where `δᵢ` = output distortion from pruning layer i **alone**.
- **Additivity** means: total distortion decomposes across layers. This enables:
  1. Precompute `D_ℓ(k)` = output-space MSE from merging to k experts at layer ℓ (independent per layer)
  2. DP solve: `min Σ D_ℓ(k_ℓ)` s.t. `Σ k_ℓ = budget`
  3. Runtime: O(L · K · B') (L layers, K options per layer, B' budget units) — trivially tractable
- **Result on ImageNet ResNet-50**: +4.7% top-1 over baselines at the same sparsity.
- **Critical caveat**: additivity assumes *small perturbations*. The SCD regression in our project (`SC + 2-opt → bpt_gap up`) suggests **this assumption breaks at 35% compression**. Therefore S1_DP is positioned as a **cheap baseline** against S1_RCO, not the headline Stage 1 method.

### 5.5 R5 — Router KD (arxiv 2603.02217, Mar 2026)

- **Title**: *Is Retraining-Free Enough? The Necessity of Router Calibration for Efficient MoE Compression*
- **Slot**: Stage 2.5 fine-tune
- **Used by**: row **S5_RKD** (and audited head-to-head against current production Stage 2.5 in `RKD_AB_PLAN.md`)
- **Method (paper §4)**:
  - Fix all expert and backbone weights **after** merging
  - Optimize **only θ_R** (router parameters) via KL divergence of student vs teacher next-token distributions
  - Loss: `L_RKD = (τ² / N_x) · Σ_t m_{t+1} · D_KL(p_T ‖ p_S)` at temperature τ
  - Forward-KL (teacher → student) direction, NOT reverse
  - Explicit padding mask `m_{t+1}` (skip pad/special tokens)
- **Training time**: ~2h for 30B model (the paper's setup; ~2–3k steps on H200)
- **Result**: consistent recovery across Expert Pruning, Editing, and Merging paradigms on Qwen3-30B-A3B after Router KD. Router is only 0.04% of Qwen3's parameters.
- **Particularly effective**: fine-grained MoEs (128+ experts) — our setup (256) sits even further in this regime
- **Why this is anchored even though our current Stage 2.5 looks similar**: the prior conv did NOT propose this as a separate row. The cross-check (see `RKD_AB_PLAN.md` §5) reveals 4–5 specific recipe deltas vs our current implementation: τ value, training duration, calibration corpus, weight-decay, loss masking. The audit-as-S5_RKD path is: run a *canonical* Stage 2 (cheap S0-style cost), produce ONE checkpoint, then run **two** Stage 2.5 variants on it — current vs paper. Whichever wins becomes the Stage 2.5 default for the rest of the S-series. **Run `RKD_AB_PLAN.md` BEFORE this plan's S-series.**

### 5.6 R6 — HodgeCover (arxiv 2605.13997, May 2026)

- **Slot**: validation / sanity-check on our exact architecture
- **Why anchored**: the only 2026 paper that benchmarks on **Qwen3.5-35B-A3B (256 experts, 40 MoE layers — our immediate predecessor architecture)**. Method uses topology-aware "Hodge covers" for aggressive (66%) expert compression.
- **Used by**: not directly an S-row — used as **validation target** (compare our SC/SC+REAM/SC+RCO numbers to HodgeCover's at the same compression class to confirm we are in the right ballpark)
- **Result**: at 66% compression on Qwen3.5-35B-A3B, HodgeCover matches REAP and beats MC-SMoE — the 256-expert + 40-layer regime is empirically tractable for calibration-based methods.

### 5.7 R7 — SlimQwen (arxiv 2605.08738, May 2026)

- **Authors**: Qwen team
- **Slot**: constraint-awareness for the plan
- **Key finding**: *"different one-shot expert compression methods converge to similar final performance after large-scale continual pretraining"* (after 120B-400B token KD pretraining the merge algorithm differences wash out).
- **Implication for our scenario**: we do NOT continual-pretrain at scale. We do Stage 2.5 router KD (~375 steps current; ~2-3k steps post-RKD audit). **In our regime, merge algorithm quality DOES matter** — i.e., investment in Rec 3/4 is justified.
- **Used by**: cited as the justification for investing dev hours in algorithmic merging quality rather than pretraining-style heal.

### 5.8 R8 — HC-SMoE (arxiv 2410.08589, ICML 2024)

- **Title**: *Retraining-Free Merging of Sparse MoE via Hierarchical Clustering*
- **Slot**: **non-uniform-budget precedent** for Rec 2 (DP knapsack)
- **Used by**: row **S1_DP** (as conceptual precedent, alongside the additivity theorem R4)
- **Method**: agglomerative clustering on expert *outputs* with average linkage + frequency-weighted merging. **Appendix B.1** explores non-uniform per-layer budgets by globally keeping top-r% most-frequently-activated experts.
- **Why this matters for S1_DP**: HC-SMoE Appendix B.1 was the prior conv's earliest cited precedent for "vary the per-layer budget instead of holding it uniform". HC-SMoE's implementation is crude (global frequency threshold determines per-layer counts as a side-effect) — both Rec 2 (output-cost DP) and R3 RCO are principled refinements of the same idea.
- **Project relationship**: per the prior conv's msg[32] caveat, **our Direction C already subsumes HC-SMoE's core insight (output-based similarity)** and goes further (routing-weighted MSE > raw output cosine sim). HC-SMoE's role here is as a **non-uniform-budget precedent**, not a merge-step anchor.
- **Code**: not publicly released as of the crawl.

### 5.9 Reference papers — used in framing but not as row anchors

| Paper | arxiv | Why mentioned | Why not a row anchor |
|---|---|---|---|
| **EvoESAP** | 2603.06003 | Prior conv's Rec 1 anchor; ESAP proxy concept also feeds Rec 4b | Superseded by RCO (R3) per RCO's own benchmark |
| **MoE-I²** | 2411.01016 | LOO importance is a precursor to per-layer damage curves | Expensive at 256 experts × 40 layers (256·40 = 10,240 LOO forward passes); the output-space cost we already compute is a cheaper drop-in for `D_ℓ(k)` |
| **DiEP** | 2509.16105 | Differentiable expert pruning — alternative to RCO | Less principled than RCO's manifold construction; not selected to avoid running two similar Stage-1 variants |
| **REAP** (the paper, not our scoring use) | 2510.13999 | Pruning-first argument; importance signal `g_j · ‖f_j‖₂` | The paper argues *pruning > merging* in principle; we use its scoring (REAP) but reject its premise (our SC = 0.1293 demonstrates merging works when the cost is right) |

See **Appendix A** for considered-but-rejected papers (RFID-MoE, LightMoE, Sub-MoE).

---

## 6. The S-series ablation matrix

**Naming convention**: `S<digit>_<algo-tag>` where the digit is the optimization stage primarily exercised and the tag identifies the variant.

**Fixed across all rows** (do not vary these — confounds): model = Qwen3.6-35B-A3B; compression target = 35%; calibration data = `nvidia/Nemotron-Cascade-2-SFT-Data` (varies only in deferred `S_CALIB`); `min_experts_per_layer` floor = current default (do not reopen — see §10 / `MOE_COMPRESS_REPORT.md` §5.1).

**Anchors recap** for cross-reference in the matrix:
- **R1** = MergeMoE 2510.14436
- **R2** = REAM 2604.04356
- **R3** = RCO 2605.00649
- **R4** = Additivity 2308.10438
- **R5** = Router KD 2603.02217
- **R6** = HodgeCover 2605.13997 (validation)
- **R7** = SlimQwen 2605.08738 (framing)
- **R8** = HC-SMoE 2410.08589

### 6.1 Stage 1 (budget allocation) rows

| Row | Anchor | Stage 1 algorithm | Stage 2 cost | Stage 2.5 | Hypothesis | Cost (rough) |
|---|---|---|---|---|---|---|
| **`S0_GRAPE`** | none (variance/baseline) | GRAPE (CKA-redundancy + entropy gate) — **current baseline** | SC (output cost) | current KL | Replicates SC=0.1293. Variance check. | 1 row (~3h) |
| **`S1_DP`** | **R4** (additivity); precedent **R8** (HC-SMoE Appendix B.1) | **Rec 2**: per-layer `D_ℓ(k)` curve from `_output_space_cost` + DP knapsack into `merge_cost_prior` | SC | current KL | Cheap budget reallocation; expects **0.10–0.12** if additivity holds. If `bpt_gap` ≥ 0.13 — additivity fails (SCD-style); pivot to S1_RCO. | +30 min (DP) + 1 row |
| **`S1_RCO`** | **R3** (RCO) | **Rec 1 upgrade**: RCO Riemannian manifold optimization, initialized from GRAPE logits, fitness = output-space MSE on calibration | SC | current KL | Expected **0.08–0.11** based on RCO reporting +4.5 pts avg over uniform at 25%. Open-source code → integration ~16h. | +1 RCO run (~85 min) + 1 row |

### 6.2 Stage 2 (cost + merge step) rows

| Row | Anchor | Stage 1 | Stage 2 cost / merge | Stage 2.5 | Hypothesis | Cost |
|---|---|---|---|---|---|---|
| **`S2_MM`** | **R1** (MergeMoE) | GRAPE (= S0) | SC cost + **MergeMoE `T₁ = Q·P†` merge step (Rec 3)** instead of freq-weighted | current KL | Replaces the merge step only. Closed-form pseudo-inverse per cluster (~5 min/layer add on top of SC). Expected **0.09–0.11** — the local cost is faithfully realized by an optimal merge. | +closed-form solve per cluster, +1 row |
| **`S2_SEQ`** | **R2** (REAM) | GRAPE | SC cost + **REAM sequential per-layer re-eval (Rec 4a/4c)** | current KL | Closes local→global gap. ~50% Stage 2 overhead (per REAM paper: 1.5h vs 1h on 30B). Expected **0.08–0.11**; the SCD regression should disappear because each layer's cost reflects the actual compressed upstream context. | +50% Stage 2 time, +1 row |
| **`S2_MM_SEQ`** | **R1+R2** (combo) | GRAPE | SC + MergeMoE merge + REAM sequential | current KL | Combines the two best Stage 2 levers. Expected **0.06–0.09** if both are additive; risk: REAM's sequential pass amplifies any merge-step bias. | +1 row at +50% Stage 2 |
| **`S2_GLOBAL`** | synthesis (no single paper); leverages ESAP concept from prior conv Rec 4b + RCO's full-loss fitness (R3) + `L1_FOR_SC_PLAN.md` Phase 2.5 substrate | GRAPE | **SC with multi-layer rollout cost via L1/vLLM (Rec 4b)** | current KL | Replaces single-layer MSE with end-to-end-residual cost. **GATED on L1_FOR_SC Phase 2.5 landing** — that plan is the right vehicle. Expected **0.05–0.10** but the win is uncertain — the rollout is noisier and might not beat REAM sequential. | +1 row at L1 wall-clock (~5h) |

### 6.3 Combination + Stage 2.5 rows

| Row | Anchor | Stage 1 | Stage 2 | Stage 2.5 | Hypothesis | Cost |
|---|---|---|---|---|---|---|
| **`S5_RKD`** | **R5** (Router KD) | GRAPE | SC | **Router KD — only router params, KL@τ=4, ~2-3k steps, wd=0, wikitext-103-raw calibration** (paper recipe; see `RKD_AB_PLAN.md` for the head-to-head audit) | Orthogonal recovery layer. Particularly effective for 256-expert fine-grained per R5. Expected **0.08–0.11** on top of SC. **Only run AFTER `RKD_AB_PLAN.md` resolves which recipe wins.** | +2h router-only train |
| **`S_BEST`** | stack | best of {GRAPE, S1_DP, S1_RCO} | best of {SC, S2_MM, S2_SEQ, S2_MM_SEQ} | best of {current KL, S5_RKD-winner from `RKD_AB_PLAN.md`} | **The headline number**: stack the winners. Only ship after individual rows have been validated to avoid stacking failures. | +1 row |

### 6.4 Calibration-data probe (deferred — out of S-series)

| Row | Anchor | Algorithm | Calibration data | Hypothesis | Cost |
|---|---|---|---|---|---|
| **`S_CALIB`** | **R2** (REAM's best mix) | S_BEST winner | `C4 : NuminaMath : The-Stack-Smol = 0:0.5:0.5` (REAM's best GEN mixture) | Tests whether calibration mismatch is the residual gap. **Run only after algorithmic winner is locked.** | +1 row, +calibration prep |

---

## 7. Build order (gates + dependencies)

Pre-flight: `RKD_AB_PLAN.md` runs **first**. Its result locks the Stage 2.5 recipe used in every S-row below. Cost: ~$15 GPU + ~1h human pre-flight (see `RKD_AB_PLAN.md` §9).

```
  ┌─ A0. Replicate S0_GRAPE (variance check on SC=0.1293)         [1 row, ~3h]
  │
  ├─→ A1. Implement Rec 2 damage-curve + DP                        [16h dev + 1 row]
  │       → ship S1_DP
  │       Gate: bpt_gap < 0.13. If failure, additivity broken — skip to A5.
  │       Code: write a new `stage1/plugins/damage_curve_dp.py`;
  │       populate `merge_cost_prior` from per-layer SC cost curve;
  │       DP solve in `budget/solver.py` extension (~200 LoC).
  │
  ├─→ A2. Implement Rec 3 MergeMoE T₁=Q·P† merge step              [12h dev + 1 row]
  │       → ship S2_MM
  │       Gate: bpt_gap < S0. Cheapest Stage 2 win. Risk: numerical
  │       stability of the pseudo-inverse on rank-deficient P matrices.
  │       Code: extend `stage2/plugins/output_space_cost.py:_tentative_merged_weights`
  │       AND `stage2/merging.py:_merge_experts_inplace` with a
  │       `merge_step="mergemoe"` config knob.
  │
  ├─→ A3. Integrate REAM sequential merging                         [24h dev + 1 row]
  │       Vendor github.com/SamsungSAILMontreal/ream under
  │       max_quality/src/moe_compress/stage2/plugins/ream_sequential.py;
  │       wire into `stage2/orchestrator.py` re-profile loop.
  │       → ship S2_SEQ
  │       Gate: bpt_gap < S0 AND SCD-style regression disappears in
  │       a 2-opt diagnostic. Risk: 50% Stage 2 overhead.
  │       Note: requires invalidating cov_acc, ream_acc, and
  │       layer_input reservoir caches after each layer's merge —
  │       add `on_post_merge` plugin hook.
  │
  ├─→ A4. Stack: S2_MM_SEQ                                          [1 row]
  │       Pure ablation of A2+A3.
  │
  ├─→ A5. Integrate RCO (github.com/IST-DASLab/RCO)                 [40h dev + 1 row]
  │       Initialize logits from GRAPE; fitness = output-space MSE.
  │       Gated on A2 & A3 working (the loss signal RCO optimizes
  │       will reflect the post-A2+A3 cost function).
  │       → ship S1_RCO
  │
  ├─→ A6. Run S5_RKD with the RKD_AB_PLAN.md winner                 [1 row]
  │       (If `RKD_AB_PLAN.md` Row P won: this is "paper recipe applied
  │       on top of SC". If Row C won: S5_RKD is dropped — current
  │       Stage 2.5 already applies it, no new row needed.)
  │
  ├─→ A7. Stack winners: S_BEST                                     [1 row]
  │       Only after every component has individually beaten S0.
  │
  └─→ A8. (deferred) Calibration data probe                          [+1 row]
          Run S_CALIB only after S_BEST locks. Do NOT confound with
          earlier rows.
```

**Concurrent track** (independent of A0–A8): `L1_FOR_SC_PLAN.md` Phase 2.5 (vLLM rollout cost). When that lands, run `S2_GLOBAL` as the cross-check on whether end-to-end cost beats REAM sequential.

---

## 8. Success criteria

| Row | Pass | Marginal pass | Hard fail |
|---|---|---|---|
| `S0_GRAPE` | bpt_gap ∈ [0.118, 0.135] (replicates 0.1293 ± 0.005 noise) | ∈ [0.135, 0.150] (variance higher than expected — investigate seed sensitivity) | > 0.150 (something regressed) |
| `S1_DP` | bpt_gap ≤ S0 − 0.01 | bpt_gap ≤ S0 (no harm; useful interpretive baseline for additivity) | bpt_gap > S0 + 0.02 (additivity broken — pivot to RCO) |
| `S1_RCO` | bpt_gap ≤ S0 − 0.02 | bpt_gap ≤ S0 − 0.005 | bpt_gap > S0 (RCO underperforms — likely bad fitness signal; revisit) |
| `S2_MM` | bpt_gap ≤ S0 − 0.02 + ARC-E + HSwag both ≥ S0 | bpt_gap ≤ S0 − 0.005 with no metric regression | bpt_gap > S0 OR pseudo-inverse numerical instability |
| `S2_SEQ` | bpt_gap ≤ S0 − 0.02; SCD-style 2-opt diagnostic shows local-vs-global agreement | bpt_gap ≤ S0 − 0.005 | bpt_gap > S0 OR Stage 2 wall-clock > 1.6× S0 |
| `S2_MM_SEQ` | bpt_gap ≤ min(S2_MM, S2_SEQ) − 0.01 | bpt_gap ≤ min(S2_MM, S2_SEQ) | not strictly better than either alone — record and move on |
| `S5_RKD` | bpt_gap ≤ S0 − 0.02; routing-entropy stable | bpt_gap ≤ S0 − 0.005 | bpt_gap > S0 |
| `S_BEST` | bpt_gap ≤ 0.05 AND ARC-E ≥ 0.76 AND HSwag ≥ 0.745 | bpt_gap ≤ 0.08 | bpt_gap > S0 stacked — investigate component conflicts |

**Cross-metric requirement** (everywhere): no S-row may ship as "winner" if any of `top1_agreement`, ARC-E, HellaSwag drops by more than 0.01 vs S0. The prior conv's "lever is the cost function" framing only holds when all metrics move together (SC's headline 4.5× win came with simultaneous ARC-E + HSwag gains; SCD's regression came with monotonic damage). Per-metric guardrails block bpt_gap-only optimization.

---

## 9. Risks + halt-triggers

| Risk | Catch | Halt action |
|---|---|---|
| **R1** MergeMoE `T₁=Q·P†` produces rank-deficient solves at small `n_calibration_tokens` / per-cluster size. | Add per-cluster `cond(P)` check before solve. Threshold at 1e8. | If cond > 1e8 for > 5% of clusters, fall back to bf16 truncated SVD pseudoinverse with rank cutoff = 0.95 of energy. |
| **R2** REAM sequential merging exposes hidden per-layer state that the current `output_space_cost` plugin doesn't refresh (cov_acc, ream_acc, layer_input reservoir all may be stale after upstream merges). | Diff the per-layer cost matrix between (a) pre-merge profile and (b) post-merge re-profile on a synthetic 3-layer toy. Any cost-matrix drift > 1e-3 relative means the plugin's state caches are stale. | If drift detected, add an `on_post_merge` plugin hook that invalidates layer-input + ream + cov caches. |
| **R3** RCO fitness on output-space MSE diverges from `bpt_gap` (the loss-signal mismatch the SCD regression already showed at the cost-matrix level). | Tracking-set: per RCO step, snapshot the current logits and report the implied per-layer budget vector. Run 3 spot-check budget vectors through end-to-end Stage 2 → bpt_gap. If RCO's fitness ranking disagrees with end-to-end ranking, halt. | Switch fitness signal to KL-divergence vs teacher final logits (cheap because we already cache them at Stage 2.5). |
| **R4** Stacking failures — `S_BEST` is worse than the best single component. | Mandatory A7 acceptance: each component must demonstrate non-conflicting gains in the pairwise stacks (S2_MM_SEQ, S1_RCO+S2_SEQ on a partial sub-sweep) before the full S_BEST run. | If S_BEST < best single, debug which pair regressed via the pairwise rows. |
| **R5** Stage 2 wall-clock balloons under sequential merging. SC is already ~3h/row; sequential adds ~50% → ~4.5h. Combined with stacking, S_BEST could approach 6h. | Track Stage 2 wall-clock per row. If > 5h, gate the next row on `SC_FAST_PLAN_V3.md`'s reservoir + per-pair micro-opts having landed (those plans target ~10–15 min savings; for a ~3h baseline they don't break the bank but for stacked rows they matter). | If > 6h, halt and prioritize `SC_FAST_PLAN_V3.md` work before continuing S-series. |
| **R6** RKD_AB_PLAN.md inconclusive (paper recipe ≈ current). | Drill-down rows in `RKD_AB_PLAN.md` §8. | If still inconclusive, treat Stage 2.5 as a free parameter and ship S_BEST with both variants; pick the winner empirically. |

---

## 10. What this plan does NOT do

- It does **not** include speed optimizations for SC — those are covered by `SC_FAST_PLAN_V3.md` (reservoir vectorization, per-pair micro-opts, profile-pass sidecar). This plan inherits row times from `SC_FAST_PLAN_V3.md`'s landing or non-landing.
- It does **not** propose the vLLM L1 substrate — `L1_FOR_SC_PLAN.md` is the right plan for that. This plan integrates with L1 only for `S2_GLOBAL`.
- It does **not** touch the calibration-v2 sidecar writers — `CALIBRATION_MIX_V2_DESIGN.md` / `_PLAN.md` is the right plan for that.
- It does **not** reopen the `min_experts_per_layer: 128` floor question. That was settled in `MOE_COMPRESS_REPORT.md` §5.1 (the floor + GRAPE classification combined to silently ship 22%; the current run uses the budget_retune Direction A path). `S1_DP` and `S1_RCO` interact with this floor — both must respect it (RCO via a projection step; DP via the knapsack bound).
- It does **not** propose RFID-MoE SVD or LightMoE — those are orthogonal paradigms (within-expert SVD; LoRA-replace) more relevant for 60%+ compression. See Appendix A.
- It does **not** revisit `min_experts_per_layer` semantics; treats it as fixed across all rows.
- It does **not** run the Stage 2.5 head-to-head audit itself — that's `RKD_AB_PLAN.md`. This plan **consumes** the audit's result.

---

## 11. Companion plans, summarized inline

### 11.1 `RKD_AB_PLAN.md` — Stage 2.5 audit (runs FIRST)

**Goal**: head-to-head A/B of current Stage 2.5 (production code) vs paper recipe (arxiv 2603.02217), on a single canonical Stage 2 substrate (S0 config). Holds Stage 2 byte-identical between the two rows, so the only variable is the Stage 2.5 recipe.

**Setup**: Stage 2 = S0's cheap `pre`-cost configuration (`cost_alignment="pre"`, `capacity_util_threshold=0`, `assignment_solver="greedy"`, all other knobs off). Save the resulting `stage2_pruned/` checkpoint and hardlink into both Stage 2.5 rows.

**Rows**:
- **Row 0a, 0b** (variance baseline): canonical Stage 2 + current Stage 2.5, two seeds — establishes `σ_C` noise floor
- **Row P** (paper): canonical Stage 2 + S5_RKD recipe (τ=4, epochs=2, wd=0, early-stop off, `wikitext-103-raw` calibration, explicit padding mask, forward-KL with τ² scaling)

**Pre-flight**: read `stage5_router_kd.py` and `_chunked_vocab_kl`. Verify (a) forward-KL direction, (b) padding-mask application, (c) τ² scaling. Any mismatch is the win — fix before running rows.

**Decision rule**: paper wins iff `m_C − Row P > 2·σ_C` AND no side-metric (ARC-E / HellaSwag / top1_agreement) regresses by >0.01.

**Cost**: ~4 H200-hours wall + ~$15 GPU.

**Output for this plan**: locks the Stage 2.5 recipe used in every S-row below.

### 11.2 `SC_BOTTLENECK_PLAN.md` — initial speed diagnostic

Phase-1 (`feat/calibration-v2` HEAD `6ff3636`). Read-only code-inspection diagnosis.

**Claimed top-3 bottlenecks** (later partially refuted by `SC_FAST_PLAN_V3.md`):
- #1 `_LayerInputAccumulator.add` per-token Python reservoir loop (claimed ~50–80% of row)
- #2 Per-pair Hungarian + fp32 weighted-merge inside `_output_space_cost` (claimed ~20–35%)
- #3 Universal profile-pass forward (claimed ~15–25%)

**Proposed phases**:
- Phase A — vectorize the reservoir (Design C: pure-tensor Algorithm R)
- Phase B — perm cache write from output path (B1), bf16 weighted merge (B2), GPU LAP via `auction-lap` (B3)
- Phase C — profile-pass sidecar cache (extends calibration-v2 writers)

**Status**: this plan was **superseded by `SC_FAST_PLAN_V3.md`**, which measured the bottlenecks instead of estimating. Keep as historical context.

### 11.3 `SC_FAST_PLAN_V3.md` — corrected speed diagnostic (empirical)

**Methodology**: actual CPU timing of production code at SC config shapes; CPU↔GPU ratios for extrapolation; tagged confidence levels.

**Corrected top-3 bottlenecks** (refutes both prior estimates):
- **#1 Universal profile-pass forward** (~28–47% of 3h row at ~30–50 ms/layer-forward × 102,500 layer-forwards). NOT SC-specific.
- **#2 Per-pair output-cost loop** (~9–13% of row). SC-specific. ~98,000 pairs × ~12 ms GPU.
- **#3 Bump-loop multiplier** (conjecture, 0–60 min). Needs GPU log inspection.

**Refutations**:
- `_LayerInputAccumulator.add` is **~4% of row, not 50–80%** (measured 85.9 ms/batch × 4960 hot calls = 7.1 min).
- Per-pair Hungarian is **~30% within per-pair, ~11% of row, not 96%** (scipy LAP measured at 4.66 ms/solve, not 25 ms).

**Optimization plan**:
- Optimization A — profile-pass sidecar cache (top-1 lever, ~30–50 min saved per row, universal)
- Optimization B — per-pair micro-opts (B1: perm cache write ~1 min; B2: bf16 weighted merge ~2–3 min; B3: vectorize argpartition ~1 min; B4: hoist build_banks ~1 min)
- Optimization C — vectorized reservoir (~7 min saved)
- Optimization D — bump-loop reuse (conditional, 0–60 min)

**Realistic post-optimization SC row times**: baseline 3h → conservative 2h53m (C only) → likely 2h45m (B+C) → optimistic 1h50m (A+B+C).

**Implication for this plan**: row times we project (~3h SC) hold as the baseline. If `SC_FAST_PLAN_V3.md` Phase A lands, every row in this plan drops by ~30–50 min.

### 11.4 `L1_FOR_SC_PLAN.md` — vLLM substrate for global-cost extension

**Scope**: vLLM-backed forward as a Stage 2 cost backend (NOT a full driver refactor). Three primitives:
1. `vllm_model(llm)` — single-worker reach into the live model via `LLM.apply_model`
2. `update_weights_inplace(llm, name_to_tensor)` — push named tensors via `replace_parameter(..., prefer_copy=True)` (CUDA-graph-safe)
3. `hf_to_vllm_experts(hf_layer, layer_idx)` — translate HF `Qwen3MoeExperts` → vLLM `w13_weight`/`w2_weight`

**Pivoted from "speed up SC's Hungarian" to two real use cases**:
- **Phase 2.5 — Global-cost variant (Direction-C upgrade)**: replace single-expert local SwiGLU with multi-layer rollout cost. **This is the substrate `S2_GLOBAL` depends on.**
- **Phase 2.6 — Optimizer/heal ablation**: drive A/B variants of Stage 2.5 router KD or per-layer heal in a single persistent vLLM session.

**Cumulative GPU budget**: ~$210–330 across Phases 2.1–2.6.

**Implication for this plan**: `S2_GLOBAL` is GATED on Phase 2.5 landing. If `L1_FOR_SC_PLAN.md` does not land within the S-series timeframe, drop `S2_GLOBAL` or replace with a low-fidelity HF-only multi-layer rollout (slower but workable).

### 11.5 `CALIBRATION_MIX_V2_*.md` — V1+V2 sidecar writers (orthogonal)

These plans extend the patched vLLM wheel's V1+V2 writers to capture REAP scores + covariance + routing stats. The Stage 2 driver consumes them via `Stage2ReapScoresCacheProvider` / `Stage2RoutingStatsCacheProvider`.

**Orthogonal to this plan**: the algorithmic S-series doesn't change what data the writers emit. If the V1+V2 writers gain a `cov_acc` + `layer_input_acc` sidecar (per `SC_FAST_PLAN_V3.md` Optimization A), every S-row gets faster — but the algorithmic ablations themselves don't change.

### 11.6 Other supporting docs

- **`algorithm_reference_retirement.md`** — historical record of which algorithms were dropped from the project (the A0–A11 historical sweep's invalidated rows). Not relevant for this plan beyond explaining why the historical numbers in `MOE_COMPRESS_REPORT.md` §5.1 are flagged as "please cross-check raw artifacts".
- **`kd_fix_plan.md`** — historical Stage 2.5 fixes including the 2026-05-19 ramp removal that triggered the pending hyperparameter audit. Background context for `RKD_AB_PLAN.md`.
- **`lessons.md`** — accumulated lessons from prior failures. Worth scanning before A1; not load-bearing for this plan.

---

## 12. Open questions raised to user

1. **Calibration data confounding**: REAM (R2) reports its best GEN result on `C4:NuminaMath:The-Stack-Smol = 0:0.5:0.5`. We currently use `nvidia/Nemotron-Cascade-2-SFT-Data` (instruction-format SFT). Whether to fold the calibration-data probe `S_CALIB` into the S-series (and risk confounding) or run it only after S_BEST.

2. **Eval suite**: 2026 papers report on **IFEval, AIME25, GPQA-Diamond, HumanEval, LiveCodeBench** (REAM's GEN suite). We currently report on `bpt_gap`, ARC-E, HellaSwag, MMLU. MC suite and bpt_gap correlate; GEN suite is more sensitive to calibration-data choice and merge quality. Whether to extend `stage6alt_thermometer.py` to add 1-2 GEN benchmarks (IFEval + GSM8K are cheap).

3. **GPU budget**: A0–A7 implies ~9 H200 rows × ~3h ≈ 27h × $3.39/h ≈ **$92** for the S-series proper. Plus ~100h human integration time (mostly for vendoring RCO + REAM + writing MergeMoE merge step). `S_CALIB` adds 1 row (~$10). `S2_GLOBAL` adds ~5h (~$17). **Total: ~$120 + dev**.

4. **REAM vendoring vs reimplementing**: the REAM repo is small and license-permissive; preferred path is to vendor the relevant pieces under `max_quality/src/moe_compress/stage2/plugins/ream_sequential.py` rather than add a runtime dependency.

5. **Order of A2 vs A3**: A2 (MergeMoE merge step) is cheaper to ship; A3 (REAM sequential) is the higher-leverage win per the recipe table. Both can be parallelized but A4 (stacked) gates on both.

---

## 13. Code citations (file:line index)

### Files this plan would modify

- **Stage 1 budget allocation** (Recs 1, 2):
  - `max_quality/src/moe_compress/stage1/plugins/grape_merge.py:171–334` — `merge_cost_prior` hook (inert by default; populate from S1_DP damage curve or S1_RCO logits)
  - `max_quality/src/moe_compress/budget/solver.py:154` — `solve()` entrypoint; extend for DP variant (S1_DP)
  - `max_quality/src/moe_compress/budget_retune.py` — Direction A retune (existing); reference for the RCO integration shape

- **Stage 2 cost matrix** (Rec 3, 4a):
  - `max_quality/src/moe_compress/stage2/plugins/output_space_cost.py:212` — `_tentative_merged_weights` (the merge step Rec 3 replaces); the Hungarian permutation alignment lives at `:256`
  - `max_quality/src/moe_compress/stage2/plugins/output_space_cost.py:276` — `_output_space_cost` (the cost-matrix main loop)
  - `max_quality/src/moe_compress/stage2/plugins/output_space_cost.py:157` — `_swiglu_forward` (per-pair forward; ~0.4 ms on GPU)
  - `max_quality/src/moe_compress/stage2/permutation_align.py:114` — `_permutation_align_to_centroid` (Hungarian wrapper)
  - `max_quality/src/moe_compress/stage2/merging.py:33` — `_merge_experts_inplace` (final merge step; second call site for Rec 3)
  - `max_quality/src/moe_compress/stage2/plugins/ream_cost.py`, `ream_cost_post.py` — the `pre`/`post` cost paths (reference for the API shape)

- **Stage 2 propagation** (Rec 4a):
  - `max_quality/src/moe_compress/stage2/profiling.py:86–286` — `_profile_layer` (REAM Rec 4a target; needs sequential re-profile after each merge)
  - `max_quality/src/moe_compress/stage2/profiling.py:58–80` — `_LayerInputAccumulator.add` (the reservoir bottleneck per `SC_FAST_PLAN_V3.md`; orthogonal to this plan but interacts with S2_SEQ's re-profile)
  - `max_quality/src/moe_compress/stage2/profiling.py:233` — `register_forward_pre_hook` callback registration
  - `max_quality/src/moe_compress/stage2/orchestrator.py:302–469` — `_run_assignment` bump-loop (interacts with S2_SEQ)
  - `max_quality/src/moe_compress/stage2/plugins/layer_merge.py:448–464` — `_need_layer_inputs` (only-on-output gate)

- **Stage 2 assignment + refinement** (reference, not modified):
  - `max_quality/src/moe_compress/stage2/plugins/solver_greedy.py`
  - `max_quality/src/moe_compress/stage2/plugins/solver_hungarian.py`
  - `max_quality/src/moe_compress/stage2/plugins/solver_mcf.py`
  - `max_quality/src/moe_compress/stage2/plugins/solver_sinkhorn.py`
  - `max_quality/src/moe_compress/stage2/plugins/em_refine.py:277` — EM early-return for non-post (confirms EM is a no-op under SC)
  - `max_quality/src/moe_compress/stage2/plugins/two_opt_refine.py:73` — 2-opt entrypoint (Direction D; SCD regression diagnostic)

- **Stage 2.5 router KD** (consumes `RKD_AB_PLAN.md` decision):
  - `max_quality/src/moe_compress/router_kd/plugins/stage5_router_kd.py` (or similar — see `RKD_AB_PLAN.md` §12)
  - `_chunked_vocab_kl` (verify forward-KL direction + padding mask + τ² scaling)
  - `max_quality/src/moe_compress/router_kd/plugins/merge_repair.py` — Direction E (out of scope here)

- **Ablation matrix**:
  - `max_quality/src/moe_compress/run_ablations.py:192–197` — current S-rows definition; ADD new rows here

- **Eval**:
  - `max_quality/src/moe_compress/stage6alt/...` — thermometer entrypoint
  - `max_quality/src/moe_compress/stage6alt_thermometer.py`

### Existing plans referenced

- `tasks/MOE_COMPRESS_REPORT.md` (canonical project state)
- `tasks/SC_FAST_PLAN_V3.md` (corrected speed diagnostic)
- `tasks/SC_BOTTLENECK_PLAN.md` (initial speed diagnostic — superseded)
- `tasks/L1_FOR_SC_PLAN.md` (vLLM substrate)
- `tasks/RKD_AB_PLAN.md` (Stage 2.5 audit — runs first)
- `tasks/CALIBRATION_MIX_V2_DESIGN.md`, `_PLAN.md` (sidecar writers)
- `tasks/calib_v2_writers_todo.md` (writer status)
- `tasks/algorithm_reference_retirement.md` (historical context)
- `tasks/kd_fix_plan.md` (Stage 2.5 history)
- `tasks/lessons.md` (accumulated lessons)

### Literature (full arxiv index)

| arxiv ID | Title | Role |
|---|---|---|
| **2510.14436** | MergeMoE: Efficient Compression of MoE Models via Expert Output Merging | R1 — Rec 3 merge step |
| **2604.04356** | REAM: Merging Improves Pruning of Experts in LLMs | R2 — Rec 4a/4c sequential merge |
| **2605.00649** | Model Compression with Exact Budget Constraints via Riemannian Manifolds (RCO) | R3 — Rec 1 budget allocation |
| **2308.10438** | Efficient Joint Optimization of Layer-Adaptive Weight Pruning in Deep Neural Networks (additivity theorem) | R4 — Rec 2 DP formal basis |
| **2603.02217** | Is Retraining-Free Enough? The Necessity of Router Calibration for Efficient MoE Compression (Router KD) | R5 — Stage 2.5 anchor |
| **2605.13997** | HodgeCover (topology-aware MoE compression) | R6 — Qwen3.5-35B-A3B validation target |
| **2605.08738** | SlimQwen | R7 — post-training scenario validation |
| **2410.08589** | HC-SMoE: Retraining-Free Merging of Sparse MoE via Hierarchical Clustering (ICML 2024) | R8 — non-uniform budget precedent |
| 2603.06003 | EvoESAP | Reference — superseded by R3 |
| 2411.01016 | MoE-I² | Reference — LOO importance precursor |
| 2509.16105 | DiEP | Reference — alternative to R3 (not selected) |
| 2510.13999 | REAP (paper itself) | Reference — provides our centroid-selection scoring |
| 2506.23266 | Sub-MoE | Reference — orthogonal paradigm (see Appendix A) |
| 2602.09316 | RFID-MoE | Reference — orthogonal paradigm (see Appendix A) |
| 2603.12645 | LightMoE | Reference — orthogonal paradigm (see Appendix A) |

### GitHub repositories to vendor or reference

- `github.com/SamsungSAILMontreal/ream` — REAM implementation (vendor for S2_SEQ)
- `github.com/IST-DASLab/RCO` — RCO implementation (vendor for S1_RCO)
- `github.com/lliai/MoERazor` — Sub-MoE (reference, not used)
- `github.com/ZongfangLiu/EvoESAP` — EvoESAP (reference, not used — superseded by RCO)

---

## 14. Appendix A — Considered-but-rejected directions

These appeared in the 2026-05-27 research crawl but were not selected as S-row anchors. They are documented here so a future agent doesn't redo the analysis.

### A.1 RFID-MoE (arxiv 2602.09316, Feb 2026)

- **Title**: *Effective MoE-based LLM Compression by Exploiting Heterogeneous Inter-Group Experts Routing Frequency and Information Density*
- **Method**: SVD low-rank decomposition within concatenated expert groups, with adaptive rank per group via hybrid frequency + effective rank (spectral entropy) metric
- **Result**: at 60% compression on Qwen3-30B-A3B, PTB perplexity 16.92 vs MoBE 24.93
- **Why rejected**: orthogonal paradigm (within-expert SVD compression, not expert merging). Most relevant at very high compression (60%+); our target is 35% where expert merging dominates. Could be a complementary post-merge step in a future high-compression variant.

### A.2 LightMoE (arxiv 2603.12645, Mar 2026)

- **Title**: *LightMoE: Reducing Mixture-of-Experts Redundancy through Expert Replacing*
- **Method**: "Expert replacing" — low-ranked expert groups become a shared base matrix + expert-specific LoRA adapters (rank 16). Annealed recovery via fine-tuning.
- **Result**: at 30% compression on OLMoE-1B-7B matches LoRA fine-tuning quality; at 50% beats MC-SMoE/HC-SMoE by 5.6% avg.
- **Why rejected**: requires fine-tuning (AdamW, lr=1e-4, ~2000 steps). Our scenario is post-training without large-scale recovery. The LoRA-adapter approach is a different paradigm than the clean expert-merge → router-KD pipeline.

### A.3 Sub-MoE (arxiv 2506.23266, Jun 2025)

- **Title**: *Sub-MoE: Efficient Mixture-of-Expert LLMs Compression via Subspace Expert Merging*
- **Method**: subspace expert merging via low-rank decomposition
- **Why rejected**: incremental over MergeMoE; the T₁=Q·P† closed-form in MergeMoE (R1) already captures the optimal output-reconstruction in the full output space without restricting to a subspace.

### A.4 EvoESAP (arxiv 2603.06003, Jun 2026)

- **Why mentioned but not anchored**: prior conv's Rec 1 anchor. Superseded by RCO (R3) which strictly dominates on benchmark + wall-clock.

---

## 15. Appendix B — HF Hub artifact paths

- **Model**: `Qwen/Qwen3.6-35B-A3B` (base) / `Qwen/Qwen3.6-35B-A3B-FP8` (teacher; faster inference for KD)
- **Calibration data (current)**: `nvidia/Nemotron-Cascade-2-SFT-Data` (instruction-format SFT)
- **Calibration data (proposed for `S_CALIB`)**: REAM's mix
  - `allenai/c4` (subset)
  - `AI-MO/NuminaMath-CoT` or `AI-MO/NuminaMath-1.5`
  - `bigcode/the-stack-smol`
- **Calibration data (proposed for `RKD_AB_PLAN.md` Row P)**: `wikitext` config `wikitext-103-raw-v1`
- **Results buckets**:
  - `pirola/moe-strategy-35pct` (current S0/SA/SAB/SC/SCD/SE sweep)
  - `pirola/moe-ablations` (historical A0–A11, R0–R7)
  - `pirola/moe-compress` (canonical reports; the prior conv's source for `MOE_COMPRESS_REPORT.md`)

---

*Generated 2026-05-27 under ml-intern protocol active state. Hermetic spec — a fresh agent with zero prior context can execute this plan from this file alone. No code changes accompany this plan. Build-order in §7 is the recommended landing sequence; each gate must close before the next opens. The ablation matrix in §6 is the deliverable contract for cross-checking against prior conv + 2026 literature. Run `RKD_AB_PLAN.md` BEFORE this plan's S-series.*
