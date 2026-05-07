# Stage 2 Assignment Revision — Workflow Specification

**Status:** design, pending implementation
**Date:** 2026-05-07
**Scope:** Stage 2 (REAP scoring + REAM pseudo-pruning) only. Stages 1, 2.5, and 3+ are not modified by this revision.
**Authoritative spec:** `max_quality/ALGORITHM_REFERENCE.md` § 5. This document defines the deltas; once landed, § 5 absorbs the relevant subsections and gains new D-rows.

---

## 1. Why this revision exists

A reviewer audited the upstream REAM reference (`SamsungSAILMontreal/ream`, `ream/ream.py`) without seeing our pipeline and produced a three-part proposal:

- **P1.** Replace greedy assignment with min-cost flow (MCF).
- **P2.** Add CLARANS local search to refine centroid selection.
- **P3.** Replace the symmetric pre-alignment cost with an asymmetric, frequency-weighted, post-Hungarian-alignment cost (with a top-K two-stage filter).

Cross-check against our Stage 2 implementation (`src/moe_compress/stage2_reap_ream.py`) and the upstream code shows several of the reviewer's premises are weaker for our setup than the report implies. This document records the cross-check, adds improvements the reviewer missed, and pins down the workflow we will implement and ablate.

---

## 2. Cross-check of the reviewer's report

### 2.1 Verified
- **Greedy iterates centroids in descending saliency** with a per-centroid cap (`ream.py:74–87`; our impl: `stage2_reap_ream.py:1113–1135`).
- **Feasibility check is `n_C × C_max ≥ n_NC`** (`ream.py:71`; our impl: `stage2_reap_ream.py:422`).
- **Cost matrix is symmetric** in upstream and ours.
- **Cost matrix used for assignment ≠ cost matrix used for merge alignment.** Merge step uses Hungarian on `C = C_act + C_wt` (`ream/merger.py:580–592`; ours: `stage2_reap_ream.py:1252–1277`).
- **MCF over the transportation polytope is integer-optimal under LP relaxation** (total unimodularity).

### 2.2 Wrong or overstated for our pipeline
- **"Cost = 1 − output similarity, output sim only."** REFUTED. Upstream averages output sim with router-logit sim when `gate_logits` is provided (`ream.py:56–57`). Our impl always combines both: `cost = 1 − (sim_gate + sim_expert)/2` at `stage2_reap_ream.py:1043`. The "routing behavior is ignored" framing does not apply to us.
- **"Merge formula is freq-weighted."** PARTIAL. Upstream's **default** is **saliency-weighted** averaging (`merger.py:563`, default `saliency='reap'`). Upstream `merger.py:597` also exposes a `merging='avg_freq'` mode that matches our convention, so freq-weighted averaging is a configurable upstream mode rather than a project-original deviation. Our spec ships freq-weighted as the only mode (literal REAM Eq. 6 reading). Flaw 2 (asymmetric merge direction) applies to us with `freq` as the weighting variable, and the reviewer's asymmetric-cost formula `freq_m/(freq_c+freq_m)` is correct in form *only* under our freq-weighted merge — under upstream's default saliency-weighted path the analogous factor would be `sal_m/(sal_c+sal_m)`.
- **Flaw 1 — "pre-alignment cost is bad."** WEAKER than claimed. δ_gate and δ̃_expert are both alignment-invariant (Hungarian acts on the intermediate-neuron axis; outputs and routing logits are not permuted). What is missing from the assignment cost is a **weight-space residual term** — and we already compute one (`C_act + C_wt`) but only consume it during merge.
- **The 29% greedy-vs-optimal gap on synthetic data.** Tested at N=32–64 with tight capacity. Our regime is N=256 with N'_l ∈ [128, 200] and `C_max = 7`, so total capacity is **7–25× slack**. Greedy/MCF gap shrinks dramatically when capacity is loose; for our reduction ratio the gap is plausibly <5%.
- **The 178% "cross-validation gap."** Metric-vs-metric, not benchmark-vs-benchmark. Reviewer admits this in §5.5; the headline number is misleading until end-to-end measured.
- **Compute estimates.** Reviewer assumed `d_intermediate = 768`. Our target is **Qwen3.6-35B-A3B with `moe_intermediate_size = 512`** — Hungarian is ~3× cheaper. Reviewer's 42-min estimate becomes ~20 min for K=16.
- **K=16 in the two-stage filter.** Reviewer's own measurement gives rank correlation 0.332 between cheap and expensive cost. K=16 is too small at that correlation — bumped to **K=24** here, and further to **K=48** as the Tier 2 production default (see § 6 `cost_topk_filter`).
- **CLARANS' "1–2 sweep convergence" claim.** Unvalidated in the report. For us it is more speculative because we use REAP for centroid selection, and CLARANS evaluates assignment cost only — it can swap out a high-REAP centroid that is functionally critical but expensive to assign to.

---

## 3. Improvements the reviewer missed

| ID | Improvement | Source | Cost | Applied? |
|----|-------------|--------|------|----------|
| **M1** | Reuse merge-time Hungarian for the assignment cost (one alignment, two consumers) | own observation | free | yes |
| **M2** | Activation-aware whitened residual: `‖ΔW · A^{1/2}‖_F` (input-cov on the right of ΔW) | AA-SVD (arXiv 2604.02119), already cited by upstream § 6; lineage GPTQ / OBS / RegMean | free (cov already collected) | yes |
| **M3** | Heterogeneous per-layer regime: gate heavy machinery on capacity-utilization threshold | own observation | saves 50–70% overhead | yes |
| **M4** | EM refinement: re-cost from merged centroid, reassign, repeat | Sub-MoE (arXiv 2506.23266) | ~10–15 min/round | yes (3 rounds default) |
| **M5** | Stage 2.5 router KD partially absorbs assignment errors → bounded ROI on assignment polish | spec § 5.5 | reasoning | shapes risk tolerance |
| **M6** | Do **not** swap to subspace/SVD merging in Stage 2 — Stage 3 owns SVD | spec § 6 | n/a | rejected |
| **M7** | Per-layer solver choice: Hungarian when `n_NC ≤ N'_l`, MCF when capacitated | own observation | sub-ms per layer | yes |
| **M8** | Per-merge-group expert distillation: 500-step MSE on merged centroid weights vs **routing-gated** original-expert outputs on calibration activations | SlimMoE (arXiv 2506.18349) / MoE-Pruner (arXiv 2410.12013), spec deviation noted in § 10 | ~40–80 min sequential | yes (Tier 1) |
| **M9** | Capacitated entropy-regularized OT (dummy-row construction) as inner-loop alternative to MCF | Cuturi 2013 (Sinkhorn), partial-OT line; **not** Sparsity-Constrained OT (which uses a quadratic-reg + first-order semi-dual solver, different scheme) | ~5–10 min | yes (Tier 3, behind flag) |

---

## 4. Decisions

Locked at the design checkpoint with the user:

- **Bundle:** maximum-quality bundle — M1+M2+M3+M4+M7+M8+M9, plus reviewer's P1 and the asymmetric-freq part of P3, **no centroid swapping**. Full Tier 1 + Tier 2 + Tier 3 from § 4.4 of the design discussion.
- **REAP top-N is strictly preserved.** No CLARANS, no swap heuristics, no saliency-aware reweighting of centroid selection.
- **All changes ship behind flags.** Defaults are set after the ablation in § 8.
- **Compute budget:** target ≤ 2 h added to Stage 2; **may exceed 2 h** if measured quality justifies it. Stage 2 is one-time per model, so wall-time matters less than final quality.
- **Out of scope:** Sub-MoE SVD merging (Stage 3 owns SVD), CLARANS centroid swaps, alternative saliency metrics (HEAPr would also touch Stage 1), GPU MCF (CPU MCF is not the bottleneck).

---

## 5. Per-layer workflow (revised)

Sequential merging is preserved (REAM §4 / spec § 5 Step 3). Each layer is processed in order.

```
LAYER L (executes after layers 0..L-1 are merged):

  1. Profile L (current behavior, unchanged):
     - δ_gate logits, expert outputs (gated), input covariance A_gate_up / A_down,
       per-expert frequency freq_e, REAP score S_e.

  2. Centroid selection (current behavior, unchanged):
     - Exclude SE blacklist + min-active-tokens filter.
     - centroid_ids = argsort(REAP)[::-1][:N'_l]   (top-N'_l by REAP score).

  3. Capacity-utilization gate (M3):
     - u = (n_NC) / (N'_l * C_max)        # both already exclude SEs.
     - If u < cap_util_threshold (default 0.25): SLACK regime — go to step 4S.
     - Else: TIGHT regime — go to step 4T.

  4S. SLACK path (cheap, optimal under symmetric δ_REAM):
     - Cost matrix = current δ_REAM (no recomputation, no whitening).
     - Solver: rectangular Hungarian (scipy linear_sum_assignment) on the
       (n_NC × n_C) cost matrix when n_NC ≤ N'_l (1-to-1, optimal in ms).
       Otherwise MCF.
     - Skip EM (M4) — slack regime cannot benefit measurably.
     - Goto step 7.

  4T. TIGHT path (full machinery):
     a. Cheap pre-filter: build δ_REAM(c, m) for all pairs (current code).
     b. For each non-centroid m: candidate set K_m = top-K by δ_REAM
        (default K = 48; Tier 2).
     c. For each (c, m) ∈ candidate set:
          (i)  P_cm = Hungarian on C_act + C_wt   # reuses
                _permutation_align_to_centroid; cache the permutation.
          (ii) Activation-aware whitened residual (M2, AA-SVD lineage):
                Convention — PyTorch nn.Linear weight shapes:
                  W_gate, W_up : (d_int × hidden)
                  W_down       : (hidden × d_int)
                Input covariances (already collected in Stage 2):
                  A_gate_up    : (hidden × hidden)
                  A_down       : (d_int × d_int)
                The activation-aware Frobenius cost minimizes
                  E_x ‖ΔW · x‖² = tr(ΔW · A · ΔW^T) = ‖ΔW · A^{1/2}‖_F²
                so A^{1/2} multiplies ΔW on the **right** (input axis), not
                the left:
                R_cm = ‖(W_c_gate − P_cm · W_m_gate)         · A_gate_up^{1/2}‖_F
                     + ‖(W_c_up   − P_cm · W_m_up)           · A_gate_up^{1/2}‖_F
                     + ‖(W_c_down − W_m_down · P_cm^T)       · A_down^{1/2}    ‖_F
                Default A_*^{1/2} = full eigen-sqrt of the input covariance
                (Tier 2). Diagonal proxy `diag(A).sqrt()` (a column-scaling
                vector applied right-of-ΔW element-wise) retained as a
                fallback flag for memory-pressured runs.
                Re-citation note: the eigen-sqrt-of-input-covariance whitening
                is the AA-SVD form (arXiv 2604.02119, already cited by upstream
                § 6) / GPTQ–OBS–RegMean lineage. **Not** AIM (arXiv 2502.02421),
                whose actual formulation is per-channel diagonal scaling by
                `mean(|x_i|)` — see § 11 for why we picked the AA-SVD form.
          (iii) Asymmetric freq weighting (P3 form for our freq-weighted merge):
                d_cm = (freq_m / (freq_c + freq_m)) * R_cm.
        Off-candidate entries: d_cm = +∞.
     d. Solve assignment (default solver = "auto"):
        - If problem is 1-to-1 (n_NC ≤ N'_l with C_max ≥ 1): rectangular
          Hungarian. (P1's MCF reduces to Hungarian here.)
        - Else: MCF (OR-Tools SimpleMinCostFlow). +∞ entries are dropped.
        - If `assignment_solver: "sinkhorn"` (Tier 3, M9): solve the
          entropy-regularized OT relaxation with a dummy-row construction
          so standard Sinkhorn-Knopp (equality marginals only) applies to
          our `Σ_m T_cm ≤ C_max` capacity bound:
                Add one virtual centroid c_∅ with cost d_{c_∅, m} = +∞·(1−tiny)
                  (large enough to be never preferred over a real centroid
                   with finite cost; finite so Sinkhorn-Knopp does not
                   underflow);
                Set capacity of c_∅ = (n_NC − Σ_real C_max) so total supply
                  matches demand;
                Solve
                  min Σ T_cm · d_cm + ε · Σ T_cm log T_cm
                  s.t. Σ_c T_cm = 1   (each NC fully assigned)
                       Σ_m T_cm = C_c (per-centroid capacity, equality after
                                       dummy-row balances supply/demand)
                via Sinkhorn-Knopp iterations on GPU (ε annealed from
                1.0 → 0.01 over `sinkhorn_iters` ≈ 200); take argmax over
                real centroids per non-centroid as the hard assignment.
                Citation note: this is a standard capacitated-OT
                construction (Cuturi 2013 + dummy-marginal trick / partial
                OT line). It is **not** the formulation of Sparsity-Constrained
                OT (arXiv 2209.15466), which uses quadratic regularization
                with a first-order semi-dual solver and cardinality
                (`||T||_0 ≤ k`) constraints — different scheme entirely.
     e. EM refinement (M4), em_refinement_rounds iterations, default 3 (Tier 2):
        - Tentatively merge groups with current assignment using cached P_cm.
        - Recompute d_cm only for (m, current_centroid_of_m'_neighbors_in_K)
          — i.e., update only the affected columns, not the full matrix.
        - Re-solve. Break if assignment is unchanged (em_convergence_break).

  5. Quality / feasibility gates (current behavior, unchanged):
     - max_merge_group_size cap, ream_cost_sigma_threshold cap.
     - Bump effective_target if either gate fires; restart from step 2
       with new N'_l. Cache survival rule on a bump:
         - The new centroid set is a superset of the old one (top-(N'_l+Δ)
           by REAP), so any (c, m) cache entry where c remains a centroid
           AND m remains a non-centroid carries over without recomputation.
         - Promoted experts (formerly NCs, now centroids) leave their old
           rows in the cache effectively orphaned (no harm, just unused);
           they need fresh P_cm / R_cm computation against their new
           candidate sets when ranked as centroids.

  6. Final assignment is the output of step 4S/4T.

  7. Merge step (current behavior, with one optimization):
     - Use cached P_cm for the chosen group memberships (no recomputation).
     - Frequency-weighted average per Eq. 6 (group-renormalized form,
       spec § 5 Step 4 — unchanged).

  7b. Per-merge-group expert distillation (Tier 1, M8):
     For each merged group g with |g| ≥ 2 (singletons skipped):
       - Trainable: only the merged centroid's gate_proj, up_proj, down_proj.
         Stage 2.5 router KD has NOT yet run — this distillation step runs
         inside Stage 2, BEFORE Stage 2.5. Routers are frozen here; they will
         be tuned by Stage 2.5.
       - Frozen: every other parameter in the model.
       - Active token set, pinned definition:
             X_g = { x ∈ calibration : ∃ e ∈ g, e ∈ TopK(σ_orig(x)) }
         where σ_orig is the **original (pre-merge) router softmax** captured
         during step 1's profile. X_g is the union of original-routing top-k
         membership across the group; this matches the support over which
         the freq-weighted mixture is defined.
       - Target output on x ∈ X_g, pinned definition (routing-gated mixture):
             y_target(x) = Σ_{e ∈ g, e ∈ TopK(σ_orig(x))}
                                 g_e^{orig}(x) · E_e^{orig}(x)
         where g_e^{orig}(x) is the post-softmax routing weight as dispatched
         in the original model (renormalized top-k for Qwen3-MoE per upstream
         spec § 5 Step 1) and E_e^{orig}(x) is the original expert's
         pre-routing FFN output (gate→silu→up→down). This is the *additive
         contribution* expert e made to the original MoE block's output on x,
         summed over members of g that were dispatched to x. Note that this
         **is** routing-gated (not bare-mixture); it is the quantity the
         merged centroid will replace in the post-merge MoE forward.
         (Spec deviation note: SlimMoE / MoE-Pruner distill against the full
         MoE-block output rather than the per-expert-group additive contribution.
         We use the additive form because we want to repair *only* the
         merged centroid; the rest of the layer's experts remain unchanged.
         Documented in § 10 D-row D-expert-distill-mse.)
       - Student forward on x ∈ X_g:
             y_student(x) = g_g^{merged}(x) · E_g^{merged}(x)
         where g_g^{merged}(x) is the **frozen, post-resize** router's weight
         for the merged centroid, and E_g^{merged}(x) is the trainable
         centroid's FFN output. The router is frozen during this step
         (Stage 2.5 will fix it later); we are not asking the router to
         match — we are asking the centroid's FFN to produce the right
         contribution given whatever the post-resize router currently sends.
         Note on the post-resize router row: by the router-resize convention
         (step 8 below, mirroring upstream § 5 Step 5), the router row for
         the merged centroid is the centroid's **original (pre-merge) router
         row carried over verbatim** — non-centroid rows are deleted, the
         centroid row is unchanged. So `g_g^{merged}(x)` is the *original*
         centroid's routing weight evaluated under the new (smaller) softmax
         denominator, not a freshly-trained value. Stage 2.5 retrains this row.
       - Loss: `MSE(y_student(x), y_target(x))` averaged over X_g.
       - Originals retention: `E_e^{orig}` and `g_e^{orig}` come from the
         snapshot taken at the start of layer L's profile (step 1) — kept in
         CPU RAM until step 7b finishes for layer L, then released.
       - Steps: 500 (default; Tier 1). Optimizer: AdamW lr=1e-4, β=(0.9,0.95),
         wd=0.0, bf16 forward, fp32 master/optimizer state. Batch: tokens of
         X_g sharded to fit ~5–8 GB activation budget per step.
       - Token budget per group: capped at min(|X_g|, 8192) tokens; if |X_g|
         is smaller than a microbatch, use the full set.
       - Convergence break: stop early if the moving-average loss falls
         below 1e-4 of the initial value or plateaus for 50 steps.
       - Parallelism: groups within a layer are independent; can run
         sequentially OR in parallel up to VRAM headroom (initial impl
         sequential, parallelism is an optimization deferred behind
         `expert_distill_parallel_groups`).

  8. Router resize, covariance remap, atomic checkpoint (unchanged).
```

---

## 6. New config knobs

All under `stage2:` in `configs/qwen36_35b_a3b_30pct.yaml`. Defaults match step 5 above; setting `assignment_solver: "greedy"`, `cost_alignment: "pre"`, `cost_whitening: "none"`, `cost_asymmetric: false`, `em_refinement_rounds: 0`, and `expert_distill_steps: 0` reproduces today's behavior bit-identically.

```yaml
stage2:
  # ---- Solver ----
  assignment_solver: "auto"        # "greedy" | "hungarian" | "mcf" | "auto" | "sinkhorn"
                                   # auto: hungarian when n_NC ≤ N'_l, MCF otherwise
                                   # sinkhorn: M9 / Tier 3 — entropy-regularized OT on GPU
  sinkhorn_epsilon_init: 1.0       # Tier 3 only — initial regularization
  sinkhorn_epsilon_final: 0.01     # annealed to this over sinkhorn_iters iterations
  sinkhorn_iters: 200              # Sinkhorn-Knopp iteration count
  # ---- Cost matrix variants ----
  cost_alignment: "post"           # "pre" (current δ_REAM) | "post" (Hungarian-aligned residual via M1)
  cost_whitening: "full"           # "none" | "diag" (cheap fallback) | "full" (eigen-sqrt; Tier 2 default)
  cost_asymmetric: true            # multiply by freq_m/(freq_c+freq_m)
  cost_topk_filter: 48             # K candidates per non-centroid for cost_alignment="post" (Tier 2: 48; was 24)
  # ---- Per-layer regime gating (M3) ----
  capacity_util_threshold: 0.25    # u < this → SLACK path; >= this → TIGHT path
  # ---- EM refinement (M4) ----
  em_refinement_rounds: 3          # 0 disables; Tier 2 default 3
  em_convergence_break: true       # stop early when assignment stops changing
  # ---- Per-expert distillation (M8 / Tier 1) ----
  expert_distill_steps: 500        # 0 disables; Tier 1 default 500 MSE steps per merged group
  expert_distill_lr: 1.0e-4        # AdamW learning rate
  expert_distill_betas: [0.9, 0.95]
  expert_distill_token_cap: 8192   # max calibration tokens used per group (subsampled if X_g larger)
  expert_distill_skip_singletons: true   # |g|==1 has no merge to repair; skip
  expert_distill_loss_plateau_steps: 50  # early-break window
  expert_distill_loss_plateau_eps: 1.0e-4 # relative; loss/loss_init below this triggers break
  expert_distill_parallel_groups: 1      # 1 = sequential; >1 = parallelize within layer (deferred)
  expert_distill_optimizer_dtype: "fp32" # AdamW master/state dtype; bf16 forward
```

Existing knobs (`max_merge_group_size`, `ream_cost_sigma_threshold`, `cost_bump_ratio`, `min_active_tokens`) are unchanged.

**Snapshot of pre-merge expert weights:** Step 7b (per-group distillation) needs the **original** down_proj outputs as its target, so a CPU-side snapshot of each non-centroid's `gate_proj/up_proj/down_proj` for the current layer must be held from step 1 through step 7b. At ~512 × 2048 × 3 × 2 bytes per expert ≈ 6 MB, holding 256 experts costs ~1.5 GB host RAM — trivially affordable. The snapshot is released the moment step 7b finishes for layer L; it is **not** persisted across layers.

---

## 7. Compute budget (Qwen3.6-35B-A3B)

`moe_intermediate_size = 512`, N = 256, N'_l ∈ [128, 200], 40 MoE layers, top-k = 8, hidden_size = 2048.

### 7.1 Per-layer machinery (assignment + cost matrix)

| Component | Per layer | × 40 layers |
|---|---|---|
| Cheap δ_REAM cost matrix (current) | ~1 s | ~40 s |
| Hungarian on 512×512 (`scipy`) | ~5–10 ms / pair | — |
| Two-stage filter, **K=48**, TIGHT path | ~48 × n_NC × 10 ms / 2 ≈ 60 s | ~20 min (TIGHT layers only) |
| Full eigen-sqrt of A_gate_up + A_down (Tier 2) | ~10 s | ~7 min |
| MCF / Hungarian solve | <10 ms | <0.5 s |
| EM round (incremental, K-neighborhood updates) | ~10–15 s | ~6–10 min / round |
| With M3 gating (assume ~50 % layers TIGHT) | — | halves the TIGHT-only rows |

Subtotal Tiers 0+2 (cost-matrix + 3 EM rounds + full whitening, M3 gating active): **~30–45 min**.

### 7.2 Per-merge-group distillation (Tier 1 / M8)

For each merged group with |g| ≥ 2:
- Forward pass through the merged centroid's gate/up/down (~512 × 2048 × 3 ops, bf16) on `min(|X_g|, 8192)` tokens.
- Target forward through |g|−1 absorbed expert snapshots (CPU→GPU streamed) on the same tokens.
- 500 AdamW steps. With reasonable batching (~512 tokens/microbatch), ~10–20 ms/step → ~5–10 s per group.

Group counts (derivation from layer geometry):
- A non-singleton group is a centroid that absorbs ≥ 1 non-centroid; equivalently, every non-centroid sits in exactly one non-singleton group with its centroid. So the number of non-singleton groups per layer is bounded by:
  - **Lower bound:** `ceil(n_NC / C_max)` (every non-centroid maximally co-located).
  - **Upper bound:** `min(N'_l, n_NC)` (every non-centroid in its own pair with a distinct centroid).
- Concrete envelope at N=256:
  - Floor budget (`N'_l = 128`, `n_NC = 128`, `C_max = 7`): 19 to 128 non-singleton groups per layer.
  - Mid budget (`N'_l = 180`, `n_NC = 76`): 11 to 76 per layer.
  - Light budget (`N'_l = 200`, `n_NC = 56`): 8 to 56 per layer.
- For compute-budget headroom calculations below we use **80 non-singleton groups per layer** (worst-case-ish across the GRAPE-allocated distribution). At 40 layers: ~3200 groups total. At 7 s per group: ~6.2 h sequential. **Use this as the upper bound; loss-plateau early-break is expected to cut it 30–50 %.**

This is the dominant cost line. Mitigations:
- **`expert_distill_parallel_groups`** (deferred, but spec'd): each merged group's distillation is local and independent, so |g| groups can run in parallel up to VRAM. At 4-way parallelism, drops to ~1 hour per the M8 line.
- **Group selection by impact:** `expert_distill_min_freq_sum` flag (add to the knob list above) — only distill groups whose Σ freq_e ≥ threshold (high-traffic groups). Skipping the bottom 50 % of groups by freq mass loses < 5 % of the distillation benefit (per SlimMoE Fig. 4 analog).
- **Loss-plateau early break** is already in: most groups converge well before 500 steps once `expert_distill_loss_plateau_eps` is met.

### 7.3 Realistic total

| Configuration | Added wall-time (worst-case 3200 groups, before plateau-break) |
|---|---|
| Tiers 0+2 only (cost matrix + EM, no distillation) | ~30–45 min |
| Tiers 0+2 + Tier 1 (parallel_groups=1, all groups) | **~6 h** |
| Tiers 0+2 + Tier 1 (parallel_groups=4, all groups) | ~1.5–2 h |
| Tiers 0+2 + Tier 1 (parallel_groups=4, freq-sum top 50 %) | ~1 h |
| + Tier 3 (Sinkhorn) | + ~5–10 min if used in place of MCF in EM rounds |

Plateau-break is expected to reduce these by 30–50 %. Defaults ship with **parallel_groups=1, all groups, plateau-break enabled** — worst case ~6 h before the plateau-break cut, ~3–4 h likely. Going past the user's 2-hour soft target is acceptable per § 4 because Stage 2 is one-time per model. The parallel_groups optimization is a follow-up; it's spec'd here so the data flow is set up to support it without refactor.

---

## 8. Ablation plan

Required before defaults are locked in spec § 5. Each row is a single full Stage 2 run + Stage 2.5 + Stage 6 evaluation.

| # | Config | Hypothesis tested |
|---|--------|------|
| A0 | Current Stage 2 (all new flags off) | Baseline. |
| A1 | `assignment_solver: auto`, rest off | P1 alone. Expected near-zero quality delta given slack capacity; sanity check. |
| A2 | A1 + `cost_alignment: post`, `cost_whitening: diag` | M1 + M2 (cheap whitening). Isolates AIM-style cost gain. |
| A3 | A2 + `cost_asymmetric: true` | Adds Flaw 2 fix. Asymmetric direction. |
| A4 | A3 + `em_refinement_rounds: 2`, `capacity_util_threshold: 0.25`, `cost_topk_filter: 24` | Cost-matrix bundle ceiling at the original (pre-Tier-2) settings. |
| A5 | A4 with `cost_topk_filter: 16` | Whether K = 24 was already too aggressive. |
| A6 | A4 with `em_refinement_rounds: 3` | Whether 3rd EM round adds anything (diminishing returns check). |
| A7 | A4 with `cost_whitening: full`, `cost_topk_filter: 48`, `em_refinement_rounds: 3` | **Tier 2 ceiling** (no expert distillation yet). Isolates the Tier 2 gain. |
| A8 | A7 + `expert_distill_steps: 500` (Tier 1) | Full Tier 0+1+2 bundle. **Production-default candidate.** Measures distillation lift on top of the best cost matrix. |
| A9 | A8 with `assignment_solver: sinkhorn` (Tier 3) | Whether Sinkhorn OT inside EM beats hard MCF. |
| A10 | A8 with `expert_distill_steps: 200` | Whether 200 steps suffice (cuts Tier 1 wall-time by 60 %). |
| A11 | A8 with `expert_distill_min_freq_sum` selecting top 50 % of groups by freq mass | Whether selective distillation captures most of the gain at half the wall-time. |

**Acceptance criteria:**
- **A4 vs A0** ≥ **+0.3** GEN-average → green-light the cost-matrix bundle as default (Tiers 0+2 minus full whitening / K=48 polish).
- **A8 vs A4** ≥ **+0.3** GEN-average → green-light per-expert distillation as default (Tier 1).
  (The first two criteria together imply A8 vs A0 ≥ +0.6, so no separate third bound is needed.)
- **No regression > 3 %** relative WikiText-2 PPL vs A0 in any green-lit row.
- **A9 must beat A8 by ≥ +0.1** to keep Sinkhorn enabled by default; otherwise leave it as an opt-in flag.
- **A10 / A11**: if either matches A8 within 0.1 GEN-avg, prefer the cheaper variant for production defaults.

If A4 clears its bar but A8 does not, ship the Tier 0+2 bundle and keep distillation as an opt-in flag for follow-up investigation. If A8 also fails to clear, the SlimMoE-style recovery hypothesis is wrong for our setup; re-investigate before continuing.

---

## 9. Tests (new + retained)

### 9.1 New unit tests (`tests/test_stage2_assignment_v2.py`)
- `test_mcf_matches_hungarian_when_capacity_one_to_one` — sanity.
- `test_mcf_matches_or_beats_greedy_on_synthetic_tight` — proves solver upgrade.
- `test_whitened_cost_recovers_planted_groups` — synthetic experts with known structure.
- `test_full_eigen_sqrt_matches_diag_when_covariance_diagonal` — sanity bridging Tier 0 and Tier 2 whitening.
- `test_asymmetric_cost_prefers_high_freq_centroid` — direction check.
- `test_em_refinement_monotone_decreasing` — assignment cost cannot increase across rounds.
- `test_em_refinement_converges_within_3_rounds_synthetic` — early-break check.
- `test_capacity_util_gate_routes_correctly` — slack vs tight branching.
- `test_topk_filter_raises_when_k_lt_group_size` — must raise, not silently truncate.
- `test_sinkhorn_solution_converges_to_mcf_at_low_epsilon` — Tier 3 sanity: as ε → 0 the soft assignment hardens to MCF's solution.

### 9.2 New tests for per-expert distillation (`tests/test_stage2_expert_distill.py`)
- `test_distill_singleton_group_is_skipped` — |g|==1 → no-op, no optimizer instantiated.
- `test_distill_loss_strictly_decreases_then_plateaus` — convergence sanity on a synthetic 2-expert merge.
- `test_distill_respects_token_cap` — `expert_distill_token_cap` truncates X_g deterministically (subsample seed = layer_idx).
- `test_distill_only_trains_merged_centroid` — every other parameter's grad must be None / zero throughout.
- `test_distill_preserves_dtype_invariant` — bf16 forward, fp32 optimizer state; merged weights end as bf16.
- `test_distill_loss_plateau_break_fires` — early termination when loss/loss_init < `expert_distill_loss_plateau_eps` for `expert_distill_loss_plateau_steps` consecutive steps.
- `test_distill_target_is_freq_weighted_original_outputs` — exact match against a hand-computed reference on a 3-expert toy group.
- `test_distill_resume_picks_up_at_correct_step_within_layer` — step-counter persisted in `merge_{layer_idx}.json`.

### 9.3 Existing tests must continue to pass
- `tests/test_stage2_merge.py` (assignment, alignment, router resize) — verify with all new flags **off** and **on**.
- `tests/test_smoke_stage2_resume.py` — resume invariant: cached P_cm, EM-round state, AND distillation step counter / optimizer state must round-trip through a forced crash without leaking partial computation. Distillation in particular must be resumable mid-group, not just mid-layer — checkpoint every N steps (default N = 100).

### 9.4 Compatibility invariant
With all new flags at their baseline values (`assignment_solver: "greedy"`, `cost_alignment: "pre"`, `cost_whitening: "none"`, `cost_asymmetric: false`, `em_refinement_rounds: 0`, `expert_distill_steps: 0`), the output of Stage 2 is bit-identical to the current implementation. Add a CI guard test that diff-compares centroid IDs, group memberships, and final merged-expert weights against a small reference fixture.

---

## 10. Spec changes once landed

`max_quality/ALGORITHM_REFERENCE.md` § 5 absorbs:
- **Step 3 update:** the assignment-solver choice (auto / hungarian / mcf / sinkhorn) and the capacity-utilization gate (M3).
- **Step 2 update:** new "Cost matrix variants" subsection covering symmetric δ_REAM (legacy), post-alignment whitened residual (diag and full eigen-sqrt), and asymmetric freq weighting.
- **New "Phase E: EM refinement" subsection** under § 5 Step 3, covering M4.
- **New "Phase F: Per-merge-group distillation" subsection** between § 5 Step 4 (merge) and § 5 Step 5 (router resize), covering M8 (Tier 1).
- **§ 5.5 (Stage 2.5) note:** clarify that Stage 2.5 router KD now runs on a model whose experts have already been distilled to their freq-weighted target — Stage 2.5's job is purely router calibration, not expert recovery.

§ 12 (Known Deviations from Papers) gains:
- **D-mcf-assignment** — MCF replaces greedy, integer-optimal under LP relaxation.
- **D-whitened-cost** — Activation-aware whitening of the alignment residual: `‖ΔW · A^{1/2}‖_F` with input covariance multiplied on the **right** of ΔW (input axis), matching `E_x ‖ΔW · x‖² = ‖ΔW · A^{1/2}‖_F²`. Uses `A_gate_up` (size hidden×hidden) and `A_down` (size d_int×d_int) already collected for Stage 3. Lineage: AA-SVD (arXiv 2604.02119, already cited by upstream § 6) / GPTQ–OBS–RegMean. Explicitly **not** AIM (arXiv 2502.02421) — AIM uses per-channel diagonal scaling by `mean(|x_i|)`, a different formulation.
- **D-asymmetric-freq** — `freq_m / (freq_c + freq_m)` weighting, paired with the freq-weighted merge of Eq. 6.
- **D-em-refinement** — Sub-MoE-style EM iteration on the (assignment, merge) pair.
- **D-capacity-util-gate** — heavy machinery gated to layers where capacity utilization u ≥ 0.25.
- **D-expert-distill-mse** — 500-step MSE distillation per merged group, target = routing-gated additive contribution `Σ_{e∈g, e∈TopK(σ_orig(x))} g_e^{orig}(x)·E_e^{orig}(x)` over the original-routing union token set X_g, student = `g_g^{merged}(x)·E_g^{merged}(x)`. Two project-original deviations from SlimMoE / MoE-Pruner: (a) we distill the per-merge-group additive contribution rather than the full MoE-block output (other experts are unchanged), and (b) we strictly separate expert-only training (Stage 2, this step) from router-only training (Stage 2.5), where SlimMoE's distillation phases can update routers and experts concurrently. The split is chosen for resume-isolation and stage-boundary clarity.

- **D-expert-distill-mse-v1** — v1 implementation simplifications layered on top of `D-expert-distill-mse`. The full spec definition above is the *target* contract; the v1 implementation in `_distill_merged_group` differs in two ways: (i) the target uses **freq-weighted-only** mixing `Σ (freq_e / Σ freq) · E_e^{orig}(x)` (no per-token routing weight `g_e^{orig}(x)`); (ii) the input token set is the **reservoir-sampled layer-input** captured during profile, not the original-routing union `X_g`. Both simplifications avoid a more invasive instrumentation pass: implementing the routing-gated form requires storing `g_e^{orig}` per (expert, token) pair and reconstructing `X_g` from `ReamCostAccumulator.gate_logit_profiles` keys, which is bounded but adds memory pressure. The v1 deviation produces a weaker but correctly-signed merge-error gradient — the merged centroid is still pulled toward a freq-weighted average of original-expert outputs, just on a uniform-token sample instead of the routing-restricted set. Phase 3 v2 will lift both simplifications; the ablation in § 8 row A8 measures the v1 form, A8' (planned) will measure the spec form.
- **D-sinkhorn-soft-assign** — Capacitated entropy-regularized OT via Sinkhorn-Knopp with a dummy-row construction that converts the inequality capacity bound into equality marginals. Standard partial-OT / capacitated-OT trick (Cuturi 2013 + dummy-marginal). Opt-in via `assignment_solver: "sinkhorn"`. Explicitly **not** Sparsity-Constrained OT (arXiv 2209.15466), which uses quadratic regularization with a first-order semi-dual solver and cardinality (`||T||_0 ≤ k`) constraints.

---

## 11. Rejected proposals and rationale

| Proposal | Source | Why rejected |
|---|---|---|
| CLARANS centroid swaps | reviewer P2 | REAP top-N is the pipeline's importance contract. Swapping based on assignment cost alone risks losing functionally critical experts; SE-blacklist and GRAPE budget interact with it. |
| Sub-MoE SVD merge | arXiv 2506.23266 (M6) | Stage 3 owns SVD. Cascading low-rank approximations across stages is hard to reason about. |
| Replace REAP with HEAPr (OBS) | arXiv 2509.22299 | Out of scope for this revision; would also touch Stage 1 and the budget solver. |
| GPU MCF | ScaleOPT 2510.20499 | Total assignment compute < 1 s with CPU MCF. Not the bottleneck. |
| K = 16 in two-stage filter | reviewer P3 | Reviewer's own rank correlation 0.332 implies K = 24 minimum, and we now ship K = 48 by default. |
| Multi-start REAP-perturbed centroid selection | own brainstorm | Self-contradictory with "preserve REAP top-N strictly". |
| Larger calibration set for cost-matrix profiling | own brainstorm | 4000 sequences already suffice for cost stability; gain too small to justify a 2× profile-time hit. |
| Cross-layer backprop of merge errors | own brainstorm | High implementation complexity, breaks sequential merging guarantee. |

---

## 12. Implementation handoff (for the next pass)

This document is the contract. The implementation pass should:

**Phase 1 — flags + cost matrix (Tier 0+2):**
1. Land the flags in `configs/qwen36_35b_a3b_30pct.yaml` and the loader.
2. Refactor `_assign_children_to_centroids` to dispatch on `assignment_solver` (`greedy` → existing code path; `hungarian` → `scipy.optimize.linear_sum_assignment` with rectangular padding; `mcf` → OR-Tools `SimpleMinCostFlow`; `auto` → routing rule).
3. Extract `_permutation_align_to_centroid` so it can be called from both the cost-matrix builder (cache `P_cm` and `R_cm`) and the merge step (consume cache).
4. Add the M2 whitening helper. Diagonal path uses `A.diag().sqrt()`; full path uses `torch.linalg.eigh(A)` with eigenvalue clamping (cache the result — Stage 3 reuses it).
5. Wire the SLACK/TIGHT capacity gate into the per-layer driver.
6. Add the asymmetric `freq_m / (freq_c + freq_m)` factor.

**Phase 2 — EM refinement (Tier 0+2):**
7. Implement EM refinement as a thin loop over steps 4c–4d with incremental column updates (only re-cost columns whose centroid identity changed since the last round).
8. Persist EM round counter in the per-layer partial JSON for resume.

**Phase 3 — per-expert distillation (Tier 1):**
9. Add a CPU-side snapshot of pre-merge gate/up/down weights at the start of each layer's profile pass; release after step 7b finishes.
10. Build the `_distill_merged_group` routine: forward target through original snapshots (streamed CPU→GPU), forward student through merged centroid, MSE loss, AdamW step.
11. Wire `expert_distill_loss_plateau_*` early-break.
12. Persist distillation step counter and AdamW state in `merge_{layer_idx}.json` (extend the partial schema; bump `format_version`).
13. Defer `expert_distill_parallel_groups > 1` to a follow-up — initial impl runs groups sequentially.

**Phase 4 — Sinkhorn (Tier 3):**
14. Add `_assign_sinkhorn` using a GPU Sinkhorn-Knopp inner loop with capacity normalization and ε annealing. Reuse the same cost matrix `d_cm`. Hardening at the end via argmax over centroids.

**Phase 5 — tests + ablation:**
15. Add the new tests (§ 9.1, § 9.2). Confirm the compatibility invariant (§ 9.4) on a small fixture before flipping any defaults.
16. Run the § 8 ablation matrix on Qwen3.6-35B-A3B end-to-end (Stage 1 → Stage 2 → Stage 2.5 → Stage 6). Update § 10 spec edits with the green-lit configuration.

No production defaults flip until the ablation has cleared the acceptance criteria in § 8.

### 12.1 Resume schema bump

The per-layer partial JSON (`_stage2_partial/merge_{layer_idx}.json`) gains:
- `assignment_solver_used` — for forensic verification on resume.
- `em_rounds_completed` — replay safe.
- `distill_state.{group_id}` — `{step, optimizer_state_b64, loss_history, plateau_counter}` per merged group; allows mid-group resume.
- `cost_matrix_hash` — sha256 of the `d_cm` matrix; resume aborts loudly if the rebuilt matrix mismatches (catches a calibration-data or seed drift between the original run and the resume).

`format_version` bumps from 1 → 2. **No backward-compat shim** — this follows the upstream contract in `ALGORITHM_REFERENCE.md` § 11 ("Format Version Enforcement"): on a version mismatch, raise a clear error instructing the operator to delete `_stage2_partial/` and re-run. Loading a v1 partial under v2 code (or vice versa) is not supported. This is intentional — silently reinterpreting old partials risks resuming with stale or incompatible cached state, especially when the new run has different `assignment_solver` / `cost_alignment` / `expert_distill_steps` settings. Operators upgrading the codebase mid-pipeline are expected to either finish a stage on the old version or restart it cleanly on the new one.
