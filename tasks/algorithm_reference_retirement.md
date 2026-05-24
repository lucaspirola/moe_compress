# Retire `ALGORITHM_REFERENCE.md` — transfer info into plugin docstrings

Branch: `feat/algorithm-reference-retirement` (off `main` @ `febc13f`)
Source doc: `max_quality/ALGORITHM_REFERENCE.md` (~1050 lines) — **consumable**
Goal: every plugin file becomes **self-contained** — a reviewer reading only
the plugin file (module docstring + class docstring + `paper` field) knows:
  1. What paper(s) the plugin implements (with arXiv ID + section/equation),
     **cross-verified against the local paper text in `audit/spec_compliance/01_papers/<arxiv_id>/source.md`**.
  2. When the paper has officially-released code, the **golden implementation
     reference** with the **exact commit SHA**.
  3. What deviations from the paper the implementation makes, with the spec's
     justification preserved.
  4. The key formulas / hyperparameters the plugin uses.

After this task: `ALGORITHM_REFERENCE.md` is **deleted**. No surviving file may
link to it.

## Policy — consumable source doc

As each plugin is updated, the corresponding text in `ALGORITHM_REFERENCE.md` is
**surgically removed**. The doc's residual size shrinks monotonically; an empty
file at the end of Phase 7 = 100% of the information was transferred.

When a passage is **shared by multiple plugins** (e.g., the §5 covariance side-
collection block consumed by both Stage 2 plugins and Stage 3 plugins), it is
**marked** rather than deleted, with a `<!-- SHARED — re-check at end -->` HTML
comment, and resolved in Phase 9 once we know which plugin "owns" it.

## Policy — paper cross-check

For each plugin's citations:
  * **Paper line refs** (e.g., "Algorithm 1 line 8") MUST be verified against
    `audit/spec_compliance/01_papers/<arxiv_id>/source.md`. The source.md
    line numbers and the paper's PDF line/§ numbers may differ — use the
    `source.md` (project-canonical) numbering.
  * **Claims attributed to the paper** that are NOT in `source.md` must be
    treated as suspect — either drop them or re-attribute to the actual source
    (often the paper's official code).
  * When a paper has officially-released code, fetch the **default-branch HEAD
    SHA** via `gh api repos/<owner>/<repo>/commits/HEAD --jq .sha` and pin it
    in the docstring. Re-verify the SHA only if the upstream repo's behaviour
    changes; otherwise the pinned SHA is the golden reference.

## Policy — Phase A/B/C/D/E/F labels

**Dropped from new docstrings.** The Phase taxonomy was a project-internal
organising scheme grounded in `ALGORITHM_REFERENCE.md`'s sections; once the
doc is deleted, the taxonomy loses its centralised definition. The orchestrator's
dependency ordering is the real load-bearing structure, and it is already
encoded in plugins' `reads`/`writes` declarations.

* **Existing code** (variable names like `_PHASE_A_BATCH_SIZE`, log messages
  like `"Stage 1 Phase A: ..."`) is left alone — renaming is a separate
  refactor with operational risk (Trackio dashboards).
* **New docstrings** describe plugins by their concern (e.g., "MA-formation
  layer detection", not "Phase A").

## Surface area

**Files that link to `ALGORITHM_REFERENCE.md` directly** (must be cleansed):
- `stage1/plugins/ma_detection.py` ← also has the wrong `paper` field
- `stage1/plugins/ablation_filter.py`
- `stage1/plugins/magnitude_topk.py`
- `stage2/resume.py`
- `budget_retune.py`
- `budget/solver.py`
- `utils/cov_sqrt.py`

**Plugin files that need docstring enrichment** (full per-plugin spec inline):
- Stage 1: 8 plugins
- Stage 2: 17 plugins
- Stage 3: 5 plugins
- Stage 4: 2 plugins
- Router-KD: 6 plugins
- Stage 6: 8 plugins
- Stage 6alt: 6 plugins
- Total: 52 plugin files

## Local paper archive

`audit/spec_compliance/01_papers/<arxiv_id>/source.md` holds the paper text for:

| arXiv ID | Title | Stage(s) | Official code (SHA pinned at first plugin) |
|---|---|---|---|
| 2507.23279 | Super Experts in MoE Models | 1 | `ZunhaiSu/Super-Experts-Profilling` @ `573aead3127ae593ba267758b832944f8fed1485` (2025-09-25) |
| 2604.06542 | GRAPE: Greedy Redundancy-Aware Pruning for MoE | 1 | TBD (lookup at grape_merge) |
| 2603.18492 | AIMER | 1 | not in local archive — **download required** |
| 1905.00414 | CKA (Kornblith et al.) | 1 | not in local archive — math identity, no code dep |
| 2510.13999 | REAP | 2 | TBD (lookup at reap_scoring) |
| 2604.04356 | REAM | 2 | TBD (lookup at ream_cost) |
| 2509.25622 | D-Rank | 3 | TBD |
| 2604.02119 | AA-SVD | 3 | TBD |
| 2604.01609 | Swift-SVD | 3 | TBD |
| 2503.12340 | SVD-LLM V2 | 3 | TBD |
| 2410.21271 | EoRA | 4 | TBD |
| 2603.02217 | Router-KD | 2.5 / 5 | TBD |
| 2107.03374 | HumanEval (Chen et al.) | 6 | not in local archive — citation only |
| 2506.23266 | Sub-MoE | 2 (em_refine) | not in local archive — em_refine cites only |
| 2506.18349 | SlimMoE | 2 (expert_distill) | not in local archive — citation only |
| 2410.12013 | MoE-Pruner | 2 (expert_distill) | not in local archive — citation only |

## Plugin × paper × deviation map (Stage 1)

Source paper: arXiv:2507.23279 (Super Experts in MoE Models) — primary source for the SE-detection pipeline.

| Plugin | Paper(s) | Applicable deviations (from §12) |
|---|---|---|
| `ma_detection` | arXiv:2507.23279 Algorithm 1 (Appendix L) Stage 1 — MA-formation layer detection. Official code @ `573aead3127ae593ba267758b832944f8fed1485`: `run.py:28` + `eval_utils.py:470-471`. | D-ma-detector |
| `three_way_and` | arXiv:2507.23279 Eq. 6 — three-way AND SE criterion | D-SE-A, D-a-max-fraction |
| `aimer` | arXiv:2603.18492 — AIMER weight-only score | D-aimer-cross-check (project-original candidate-source role) |
| `sink_token` | arXiv:2507.23279 Figures 20-21 (descriptive observation) | D-sink-token-routing (project-original detector criterion) |
| `magnitude_topk` | (no paper — project-original) | D-magnitude-topk-candidates |
| `ablation_filter` | (no paper — project-original) | D-causal-ablation-validation |
| `cka_distance` | Kornblith et al. ICML 2019 (arXiv:1905.00414) CKA; cited by GRAPE §3.2 | D-cka-distance (sign-flip to distance form) |
| `grape_merge` | arXiv:2604.06542 §3.2-3.3, Algorithm 1 — GRAPE budget allocation | D3, D4, D5, D-cka-distance, D-se-blacklist-merge, D-grape-restart-merge |

## Plugin × paper × deviation map (Stage 2)

Source papers: REAP (arXiv:2510.13999) §Eq. 9 + REAM (arXiv:2604.04356) §3–4, Eq. 4–8 (Eq. 7 the aggregator).

| Plugin | Section in §5 | Paper(s) | Applicable deviations |
|---|---|---|---|
| `reap_scoring` | §5 Step 1 | arXiv:2510.13999 Eq. 9 (REAP) | D-reap-routing-weight, D-reap-min-active-tokens |
| `ream_cost` | §5 Step 2 (pre-alignment δ_REAM, symmetric) | arXiv:2604.04356 §3, Eq. 4–5, 7 | D-ream-aggregation, D-ream-similarity-rescale, D-ream-sparse-routing |
| `ream_cost_post` | §5 Step 2 v2 (post-alignment whitened residual) | arXiv:2604.02119 AA-SVD lineage (cost variant) | D-whitened-cost, D-asymmetric-freq, D-capacity-util-gate |
| `output_space_cost` | §5 Step 2 v2 (output-space cost variant) | arXiv:2604.04356 §3 (output-space cost) | D-whitened-cost (output-space mode) |
| `solver_greedy` | §5 Step 3 (single-pass greedy) | arXiv:2604.04356 §4 | (v1 baseline — no deviation) |
| `solver_hungarian` | §5 Step 3 v2 | (no paper — `scipy.linear_sum_assignment`) | D-mcf-assignment |
| `solver_mcf` | §5 Step 3 v2 | OR-Tools `SimpleMinCostFlow` (Ahuja–Magnanti–Orlin §9) | D-mcf-assignment |
| `solver_sinkhorn` | §5 Step 3 v2 | Cuturi 2013 + dummy-marginal partial-OT | D-sinkhorn-soft-assign |
| `solver_auto` | §5 Step 3 v2 | (dispatcher heuristic) | D-mcf-assignment (auto-dispatch rule) |
| `solver_dispatch` | §5 Step 3 v2 | (dispatcher harness) | — |
| `skip_merge_floor` | §5 feasibility / quality gates | (no paper — project-original) | D-ream-budget-bump |
| `capacity_gate` | §5 Step 3 capacity-util gate | (no paper — project-original) | D-capacity-util-gate |
| `em_refine` | §5 Step 3 v2 EM-refinement | Sub-MoE (arXiv:2506.23266) iterative refinement | D-em-refinement |
| `two_opt_refine` | §5 Step 3 refinement | (no paper — classic 2-opt local search) | — |
| `expert_distill` | §5 Step 4 v2 per-merge-group distillation | SlimMoE (arXiv:2506.18349), MoE-Pruner (arXiv:2410.12013) | D-expert-distill-mse, D-expert-distill-mse-v1 |
| `layer_merge` | §5 Step 4 frequency-weighted merge | arXiv:2604.04356 §4 Eq. 6 | D5a, D5b, D-ream-resume-fallback |
| `merge_heal` | §5 Step 5 (router resize healing) | (no paper — project-original) | — |

## Plugin × paper × deviation map (Stage 3)

Source papers: D-Rank (arXiv:2509.25622), AA-SVD (arXiv:2604.02119), SVD-LLM V2 (arXiv:2503.12340), Swift-SVD (arXiv:2604.01609).

| Plugin | Paper(s) | Applicable deviations |
|---|---|---|
| `covariance_collection` | arXiv:2604.02119 Theorem 3.2 (dual-forward B and cross-covariance C) | D6, D-no-intra-block-cascade, D-cov-storage-fp16 |
| `d_rank_allocate` | arXiv:2509.25622 §3.1, Eq. 1-2 + Eq. 7 (D-Rank) | D7, D7a, D-drank-premerge-A |
| `swift_svd_alpha` | arXiv:2604.01609 Algorithm 2 (Swift-SVD per-expert rank redistribution) | D8, D-eps-star, D-per-type-alpha |
| `aa_svd_factor` | arXiv:2604.02119 Theorem 3.2 (Path 1) / Corollary 3.3 (Path 3) | D6, D-AASVD-objective |
| `block_refine` | arXiv:2604.02119 Algorithm 2 line 9 + §3.3 (block-level AdamW refinement) | D-c5-moe-only |

## Plugin × paper × deviation map (Stage 4)

Source paper: EoRA (arXiv:2410.21271) Algorithm 1.

| Plugin | Paper(s) | Applicable deviations |
|---|---|---|
| `eora_inputs` | arXiv:2410.21271 Algorithm 1 step 2 (with multi-sample X̃ ∈ ℝ^{N×d_in}) | D10 |
| `eora_compensation` | arXiv:2410.21271 Algorithm 1 (full procedure) | D-eora-budget-pct |

## Plugin × paper × deviation map (Router-KD — Stages 2.5 & 5)

Source paper: Router-KD for MoE Compression (arXiv:2603.02217) Eq. 3, Table 1, §F.3.

| Plugin | Concern | Paper(s) | Applicable deviations |
|---|---|---|---|
| `trainable_scope` | trainable-param scope (only `mlp.gate.weight`; teacher freeze) | arXiv:2603.02217 §4 | — |
| `kd_optimizer` | Table 1 hyperparameters (AdamW, lr=5e-5, bs=8) | arXiv:2603.02217 Table 1 / §F.3 | (`weight_decay=0.0` override; documented inline) |
| `vocab_kd` | Eq. 3 (vocab-level forward-KL) | arXiv:2603.02217 Eq. 3 | (fully-packed `+ ε` drop; documented inline) |
| `teacher` | Teacher Loading (BF16 + 4-bit fallback + cache-wins precedence) | arXiv:2603.02217 §F.3 | — |
| `merge_repair` | Stage-2.5 vs Stage-5 distinction; merge-time repair | arXiv:2603.02217 §4 + arXiv:2604.04356 (REAM) interaction | D-protocol-blend |
| `early_stop` | step-boundary checkpointing + plateau early-break | (no paper — project-original) | — |

## Plugin × paper × deviation map (Stage 6 — validation)

Source: §9 of `ALGORITHM_REFERENCE.md`. Stage 6 is a validation harness; per-plugin paper attribution varies.

| Plugin | Concern | Paper(s) | Applicable deviations |
|---|---|---|---|
| `eval_environment` | env setup (revision pinning, cu130/Hopper patches, torch.compile setup, masking_utils patch) | arXiv:2603.17771 (gated attn / MA-sink loop) | (F-S-M-1 / F-S-M-3 / F-S-H-3 internal pins) |
| `wikitext_ppl` | WikiText-2 PPL Protocol (F-S-C-1) | (canonical HF `evaluate` / `lm-eval` PPL recipe) | — |
| `zero_shot_lm_eval` | ARC-C + HellaSwag via lm-eval-harness | EleutherAI lm-evaluation-harness | — |
| `humaneval` | HumanEval pass@1 | Chen et al. 2021 (arXiv:2107.03374) | D-humaneval-greedy |
| `math500` | MATH-500 accuracy | Hendrycks et al. 2021 (MATH) + HuggingFaceH4/MATH-500 | (in-tree project-original grader) |
| `teacher_provider` | teacher I/O overlap + teacher eval cache | (no paper — project-original) | — |
| `imatrix_export` | GGUF convert + llama-imatrix + eval-text-concat | llama.cpp imatrix-guided quantisation | (F-S-L-3 GGUF dtype path) |
| `validation_report` | final JSON-assembly + Trackio flatten + threshold gating | (no paper — project-original) | — |

## Plugin × paper × deviation map (Stage 6alt — thermometer)

**Not in `ALGORITHM_REFERENCE.md`.** Stage 6alt's spec is self-contained in
`stage6alt_thermometer.py`'s module docstring. Phase 7 is a sanity sweep — no
info-transfer needed.

| Plugin | Concern |
|---|---|
| `thermo_environment` | env setup (pure Pattern B reproducing Stage 6 helpers) |
| `thermo_corpus` | calibration-corpus build (nemotron + wikitext) |
| `bpt_metric` | BPT (bits-per-token) NLL forward pass |
| `zero_shot_subset` | cheap ARC-Easy + HellaSwag via lm-eval |
| `thermo_teacher_provider` | sweep-shared teacher cache |
| `thermo_report` | final JSON assembly + top-1 agreement + gap |

## Non-plugin files to cleanse (Phase 8)

These reference `ALGORITHM_REFERENCE.md` in comments / docstrings but are not
plugins — their cleansing is straightforward:

- `stage2/resume.py`
- `budget_retune.py`
- `budget/solver.py`
- `utils/cov_sqrt.py`

## Execution plan

- [x] Phase 0: branch `feat/algorithm-reference-retirement` from `main` @ `febc13f`
- [ ] Phase 1 (Stage 1, 8 plugins) — one commit per plugin; consume `ALGORITHM_REFERENCE.md` per plugin
- [ ] Phase 2 (Stage 2, 17 plugins)
- [ ] Phase 3 (Stage 3, 5 plugins)
- [ ] Phase 4 (Stage 4, 2 plugins)
- [ ] Phase 5 (Router-KD, 6 plugins)
- [ ] Phase 6 (Stage 6, 8 plugins)
- [ ] Phase 7 (Stage 6alt, 6 plugins — sanity sweep only)
- [ ] Phase 8 — cleanse non-plugin files
- [ ] Phase 9 — resolve any `SHARED — re-check at end` markers in
       `ALGORITHM_REFERENCE.md`; confirm doc is empty modulo title/footer
- [ ] Phase 10 — delete `ALGORITHM_REFERENCE.md` (own atomic commit); sweep for
       any remaining `ALGORITHM_REFERENCE` reference
- [ ] Phase 11 — full test suite green

## Verification gate per commit

* Plugin's own tests + the stage golden test green.
* `grep -r ALGORITHM_REFERENCE max_quality/src/<plugin-stage>/` returns nothing
  for that plugin.
* `grep -F "<consumed text>" max_quality/ALGORITHM_REFERENCE.md` returns
  nothing (proves the surgical deletion landed).

## End-of-task verification

* `ls -la max_quality/ALGORITHM_REFERENCE.md` returns "No such file or directory".
* `grep -r ALGORITHM_REFERENCE max_quality/` returns nothing.
* `pytest max_quality/tests/` green.
