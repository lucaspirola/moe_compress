# MoE-Compression Pipeline — Architecture & Findings Report

*Prepared for external expert cross-check. Repo: `moe_compress` (model-agnostic
tool; current test case Qwen3.6-35B-A3B, a fine-grained MoE). Scope: what each
stage can do, and every result obtained so far. Caveats are marked explicitly —
please cross-check numbers against the raw artifacts in HF bucket
`pirola/moe-strategy-35pct` and the per-row `stage6alt_eval.json` files.*

---

## 0. Pipeline overview

The compressor reduces an MoE model's parameter count (target here: ~35%) by
**merging routed experts**, in four stages:

1. **Stage 1 — GRAPE**: decide *how many* experts to keep per layer (per-layer
   budgets), detect "super-experts" to protect, and screen layers/experts by
   measured ablation damage.
2. **Stage 2 — REAP + REAM**: within each layer, select which experts survive
   (centroids) and merge the rest (children) into them.
3. **Stage 2.5 — Post-Merge Router KD**: knowledge-distil the routers (and,
   optionally, the merged experts) so routing adapts to the merged expert set.
4. **Evaluation — "thermometer"**: a cheap per-row metric, `bpt_gap` =
   student bits-per-token − teacher bits-per-token on WikiText (lower = less
   compression damage), plus `top1_agreement`, ARC-Easy, HellaSwag.

Primary metric in all tables below: **`bpt_gap`** (lower is better).

---

## 1. Stage 1 — GRAPE (`stage1_grape.py`, `budget/solver.py`, `stage1_ablation_filter.py`)

GRAPE = the budgeting/screening stage. It runs in phases (A–D observed in logs):

- **Phase A — MA-formation layer detection**: identifies which decoder layers
  are merge-amenable.
- **Phase B — profiling**: a calibration forward pass collects per-expert
  output representations and routing statistics.
- **Phase C** — redundancy / similarity computation.
- **Phase D — ablation filter** (`stage1_ablation_filter.py`): physically
  zeroes each candidate expert's `down_proj` output and measures **ΔNLL on a
  held-out corpus** — an output/behavioral damage measure (not a weight norm).
  Replaced an earlier static-threshold blacklist that was ~91% dead weight.

What Stage 1 produces / can do:

- **Super-expert detection (`stage1_blacklist`)**: experts protected from
  merging, selected by a three-way AND criterion (P99.5 activation + 0.1·a_max
  …). Earlier config keys (`zscore_threshold`, `max_blacklisted_per_layer`,
  `global_blacklist_cap_pct`) are **deprecated and ignored** — the code warns.
- **Per-layer expert-budget allocation (`_grape_greedy_merge`)**: a greedy
  knapsack that, at each step, merges in the layer with the smallest
  `R[li]` — the **sum of off-diagonal CKA distances** in that layer — modulated
  by an **entropy gate** (`gamma`, `_entropy`) that deliberately trades merge
  damage for routing-distribution balance. Pairwise merge candidates use a
  `1 − CKA` distance matrix (`_cka_distance_matrix`), where CKA is computed on
  expert *outputs* from the calibration forward.
- **Global budget** (`budget/solver.py`, `BudgetDecomposition`): sets the
  overall keep-ratio; the per-layer split is GRAPE's job.
- **`min_experts_per_layer` floor** (config; 128 in the current config): a hard
  lower bound on experts kept per layer. NOTE — see Findings §5.1: this floor,
  combined with Stage-1 classifying many layers non-redundant, made the
  *historical* runs unable to reach 35% and silently ship ~22%.
- **`merge_cost_prior` hook** (`stage1_grape.py:626, 1549`): an *implemented
  but inert* (defaults to `None`) mechanism to bias per-layer budget allocation
  by *measured* merge damage (`R[li] * merge_cost_prior[li]`) instead of pure
  CKA redundancy. Not currently used.
- **Direction A — `budget_retune`** (`budget_retune.py`): a post-Stage-2,
  damage-aware re-solve of the per-layer budget using S0's measured Stage-2
  merge damage, with a configurable `N//K` floor (`floor_divisor`). With
  `floor_divisor=2` it is a byte-identical no-op (matches GRAPE's hard N//2
  floor); `floor_divisor>2` lets donor layers drop below N//2.

Stage 1 is **model-dependent, not strategy-dependent** — it is computed once
and shared (hard-linked) across all sweep rows.

---

## 2. Stage 2 — REAP + REAM expert merging (`stage2_reap_ream.py`)

Per layer: keep `budget` experts (**centroids**), fold the remaining experts
(**children**) into them. The child→centroid pairing is a **cost matrix** fed
to an **assignment solver**.

### 2.1 Cost matrices — `cost_alignment` (`_ream_cost_matrix:1544`)

| mode | what the cost is | expense |
|---|---|---|
| `pre` (default) | **weight-space** symmetric δ_REAM cost `1 − (δ_gate + δ̃_expert)/2` — a blend of gate-logit similarity and gated-output cosine similarity, from accumulated profiling stats. | cheap, closed-form |
| `post` | **post-alignment whitened residual**: per-child, top-K cheap candidates, then a per-pair Hungarian neuron-permutation alignment + whitened Frobenius residual `‖(W_c − P·W_m)·A^½‖_F`. | moderate |
| `output` (**Direction C**) | **output-space**: for each child m and its top-K cheap-cost candidate centroids, tentatively merge, run the merged expert on calibration tokens, and measure the **routing-weighted MSE of the layer's gated routed-output change**, `mean_t σ_m(x_t)·‖E_m − E_merged‖²`. (`_output_space_cost:1976`) | expensive — see §5.3 |

Cost-shaping knobs: `cost_whitening` (none/diag), `cost_asymmetric` (freq-weighted
residual), `cost_topk_filter` (candidate shortlist size, default 48).

A **capacity gate** (`capacity_util_threshold`, `_pick_effective_alignment`)
decides whether the configured `cost_alignment` actually runs. **Caveat**: at
the default `0.25`, observed expert utilisation (0.0) fell below threshold and
forced `pre` — so in the historical A0–A11 ablations the `post`/whitening/
asymmetric/topk knobs were **dead code**. `capacity_util_threshold: 0` opens the
gate.

### 2.2 Assignment solvers — `assignment_solver` (`SolverName:51`, dispatch `_assign_children_to_centroids:2734`)

| solver | function | behavior |
|---|---|---|
| `greedy` (default) | `_assign_greedy:2817` | Capped path: centroid-order single pass, each slot takes its cheapest unassigned child, no backtracking. Uncapped path: per-child argmin (optimal). |
| `hungarian` | `_assign_hungarian:2901` | Rectangular optimal 1-1 (`scipy.linear_sum_assignment`). |
| `mcf` | `_assign_mcf:2982` | Min-cost-flow with true per-centroid capacity (ortools); optimal capacitated. Falls back to greedy if ortools missing. |
| `auto` | `:2796` | `hungarian` if `n_children ≤ n_centroids`, else `mcf`. |
| `sinkhorn` | `_assign_sinkhorn:3072` | Capacitated entropy-regularized OT, ε-annealed. |

Merging is **capped** (`max_merge_group_size: 8`) — so `greedy` here is a
genuine heuristic, not optimal.

### 2.3 Refinements / extras

- **Direction B — skip-merge floor** (`_apply_skip_merge_floor`,
  `skip_merge_percentile`): masks the costliest merges (cost above the Kth
  percentile) to `+inf`; affected children become standalone "orphan-promoted"
  kept experts. (Note: keeps extra experts → can land below the target
  compression.)
- **Direction D — 2-opt refinement** (`_two_opt_refine:2466`, `two_opt_refine`):
  a post-assignment pass that accepts only strictly cost-lowering swaps/moves.
- **EM refinement** (`_em_refine_assignment:2585`, `em_refinement_rounds`):
  re-solve the assignment after substituting the tentative merged centroids
  (the cost is otherwise computed against *un-merged* centroids).
- **Per-merge expert distillation** (`expert_distill_steps`,
  `expert_distill_min_freq_sum`): optional per-merge-group fine-tune.

---

## 3. Stage 2.5 — Post-Merge Router KD (`stage5_router_kd.py`)

Recalibrates routing after Stage-2 merging via knowledge distillation from the
(unmerged, FP8) teacher.

- **Trainable by default**: *only the router weights* (`_freeze_non_routers`,
  `trainable_name_patterns`); all expert weights frozen.
- **Loss**: temperature-scaled **vocab-KL** between student and teacher final
  logits, computed in sequence chunks (`_chunked_vocab_kl`; `kd_seq_chunk_size`
  — see §5.3). Optimizer **AdamW** (weight decay regularizes routers).
- **Temperature ramp**: `kd_temperature_start` (4.0) → `kd_temperature_end`
  (1.0) over `kd_temperature_ramp_fraction` of training; legacy scalar
  `kd_temperature` fallback.
- **Recipe**: `epochs` (1 in current config — reduced from a 2-epoch overfit),
  ~375 steps, calibration from `nvidia/Nemotron-Cascade-2-SFT-Data`, save-best
  router state, teacher-logits cache.
- **Direction E — `merge_repair`**: ALSO unfreezes the **merged centroid
  experts** and adds a **per-layer MSE** term between student and teacher
  MoE-block outputs on merge-affected layers (averaged over layers; `mse_weight`
  the only knob). Total loss = `kl_loss + mse_weight · mean(per-layer MSE)`.
  Expert and router params go in separate AdamW groups.

---

## 4. Evaluation — thermometer (`stage6alt_thermometer.py`)

A cheap per-row eval: `bpt_gap` = student_bpt − teacher_bpt on WikiText
(absolute teacher-vs-student gap), `top1_agreement`, ARC-Easy, HellaSwag, with a
cached teacher baseline. (`teacher_bpt` is constant across rows.)

---

## 5. Findings

### 5.1 Historical sweep A0–A11 + R0–R7 — **~22% compression** (NOT 35%)

The original 12-row ablation (A0–A11) and the redesigned R-rows. Key results
(from `tasks/FINAL_FINDINGS.md` / project memory — **please cross-check raw
artifacts**):

- **A0 baseline (greedy, all knobs off): `bpt_gap ≈ 0.2507`.**
- **This 0.2507 is a ~22% compression number, not 35%.** Stage 1 classified
  ~21/40 layers non-redundant; with `min_experts_per_layer: 128`, 35% was
  unreachable, so the runs **silently shipped ~22%**.
- **A1–A11 cost-knob rows were largely invalid**: the capacity gate
  (`capacity_util_threshold: 0.25`) forced `pre`, making `cost_alignment`/
  `whitening`/`asymmetric`/`topk` **dead code** for A0–A11.
- **R-rows (gate opened, `capacity_util_threshold: 0`)** give the trustworthy
  solver×cost grid:
  - Under the `pre` cost: `greedy 0.2507 ≪ sinkhorn 0.3353 ≪ auto 0.4028` —
    harder solvers strictly *worse*.
  - Under the `post` cost: `greedy 0.2668, sinkhorn 0.2725, auto 0.2747` — all
    three **collapse into a ~0.008 band**.
- **Conclusion of the historical sweep**: greedy + all-knobs-off won; the
  optimal-assignment machinery never beat greedy. The `greedy≪sinkhorn≪auto`
  ordering under `pre` was a **mis-specified-cost artifact** — an optimizing
  solver faithfully optimizing the wrong objective amplifies damage.

### 5.2 Strategy sweep S0–SE — **genuine ~35% compression** (this run)

Six rows: baseline + the five "beyond-greedy" directions. Final leaderboard
(thermometer `bpt_gap`, WikiText; results in bucket `pirola/moe-strategy-35pct`):

| rank | row | direction | bpt_gap | top1_agree | student_bpt | ARC-E | HSwag |
|---|---|---|---|---|---|---|---|
| 1 | **SC** | **C — output-space merge cost** | **0.1293** | 0.843 | 2.8535 | 0.74 | 0.735 |
| 2 | SCD | C + D (2-opt refinement) | 0.1868 | 0.825 | 2.9110 | 0.74 | 0.725 |
| 3 | S0 | baseline (greedy, all-off) | 0.5767 | 0.735 | 3.3009 | 0.71 | 0.715 |
| 4 | SA | A — damage-aware budget retune | 0.7688 | 0.686 | 3.4930 | 0.71 | 0.590 |
| 5 | SAB | A + B (skip-merge floor) | 0.7808 | 0.684 | 3.5050 | 0.66 | 0.605 |
| — | SE | E — merge-repair | **FAILED (OOM)** | — | — | — | — |

(teacher_bpt = 2.7242 for all rows. S0/SA used `floor_divisor=4` for the
budget retune; S0 itself is plain greedy. SA/SAB share Stage 1 with S0 and
consume the budget-retune output.)

**Findings:**

1. **Direction C (output-space merge cost) is the decisive win**: `bpt_gap`
   0.5767 → 0.1293 (~4.5×), and it beats baseline on *every* metric
   (top1_agreement, ARC-E, HellaSwag all up together). The other rows all use
   the same `greedy` solver — **the win is the cost function, not the solver.**
2. **Directions A and A+B HURT** — worse than the trivial greedy baseline
   (0.77/0.78 vs 0.58). The damage-aware budget retune (against weight-space
   damage) and the skip-merge floor both regressed.
3. **Direction D adds nothing / slightly hurts**: SCD (C+D) 0.1868 vs SC 0.1293.
   2-opt provably *lowers total assignment cost* yet the *model* got worse —
   evidence that even the output-space cost is still an imperfect proxy
   (optimizing it harder overfits the residual proxy error).
4. **Direction E could not be measured** — see §5.3.

**Caveat for cross-check**: Direction C changes *which* experts merge, not *how
many* — SC should be at the same ~35% compression as S0. We have **not yet
independently confirmed SC's compression level**; experts should verify SC's
result is a genuine quality gain at equal compression, not partly a
compression-level difference.

### 5.3 Operational failures encountered this run

- **Stage 2.5 CUDA OOM (SC, SCD, SE)**: `_chunked_vocab_kl` OOM'd — the
  full-sequence vocab-KL chunk (`kd_seq_chunk_size: 512`) needed ~3.78 GiB and
  did not fit (≈0.6–2 GiB free of 139.8 GiB on H200). **Fix**: `kd_seq_chunk_size
  512 → 32` (correctness-neutral — KL is summed over the token axis identically
  at any chunk size). Recovered SC and SCD by resuming from their saved
  `stage2_pruned` and re-running Stage 2.5.
- **SE remains unrecovered**: SE additionally OOMs on **Adam optimizer state** —
  Direction E unfreezes the merged centroid experts, so AdamW must allocate
  momentum/variance buffers for all of them, which does not fit alongside
  student + FP8 teacher on a single H200. Not addressed by the chunk fix.
  Fixing SE needs a Direction-E optimizer-memory change (CPU-offloaded Adam =
  numerically faithful; or 8-bit Adam / partial unfreeze = changes what SE
  measures).
- **Direction C is expensive**: `_output_space_cost` is ~3 h/row in Stage 2,
  dominated by a per-pair Hungarian neuron-permutation solve inside
  `_tentative_merged_weights` (~25 ms/pair, ~96% of per-pair cost; the SwiGLU
  forward is ~0.4 ms). It does not GPU-vectorize (no batched Hungarian); only
  process-level parallelism would speed it up.

### 5.4 Cross-cutting conclusion (for discussion)

- **The lever is the cost, not the solver.** A faithful (output-space) merge
  cost beat the baseline ~4.5×; harder solvers/refinements on top did not help
  (SCD regression; the historical solver collapse under a good cost). Greedy is
  not the bottleneck.
- **The output-space cost is still an imperfect proxy** — it is *locally*
  faithful (single-layer routed-output residual) but the SCD regression
  suggests it is not *globally* faithful (end-to-end). A downstream/end-to-end
  damage measure is the principled next step.
- **The same weight/structural-proxy weakness exists in Stage 1's per-layer
  budget allocation**: GRAPE allocates budget by CKA-redundancy + an entropy
  gate that is explicitly *not* damage-minimizing. The `merge_cost_prior` hook
  to feed measured damage in already exists but is inert. (Stage 1's ablation
  filter, by contrast, already uses a behavioral ΔNLL measure — no gap there;
  GRAPE pair selection uses CKA on outputs — a moderate gap.)

---

## 6. Open questions for the expert team

1. Is SC's 0.1293 a genuine equal-compression quality gain? Verify SC's actual
   per-layer expert counts / compression ratio vs S0.
2. Is the output-space cost's local-vs-global proxy gap (the SCD regression)
   best closed by an end-to-end (logit/KL) damage measure?
3. Should Stage-1 budget allocation move off CKA-redundancy onto a measured
   merge-damage prior (`merge_cost_prior`)?
4. Direction E: is a numerically-faithful memory fix (CPU-offload Adam) worth
   it, or should the merge-repair design itself change (alternating
   freeze-phases — see `tasks/`/memory notes)?
5. Cross-check the historical numbers (§5.1) against the raw `pirola/moe-ablations`
   bucket — they are reported here from project notes, not re-derived.

---

*Generated 2026-05-18. Code references are file:line in
`max_quality/src/moe_compress/`. Raw results: HF buckets `pirola/moe-strategy-35pct`
(this sweep) and `pirola/moe-ablations` (historical).*
