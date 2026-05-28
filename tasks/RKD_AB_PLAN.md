# Router-KD A/B — Stage 2.5 Audit on Canonical REAP Stage 2

**Branch**: `feat/calibration-v2`
**Status**: Spec. NOT implemented. **Runs before the rest of the S-series in `SC_STAGE12_COMPREHENSIVE_PLAN.md`.**
**Author**: ml-intern protocol (session 2d4987aa, continuation of `SC_STAGE12_COMPREHENSIVE_PLAN.md`)
**Date**: 2026-05-27

---

## 0. TL;DR

Hold Stage 2 constant at the cheap, fast REAP+pre-cost configuration (the existing S0 baseline — no per-pair Hungarian, no output-space cost, ~1h). Produce one Stage 2 checkpoint. Then run **two** Stage 2.5 recipes against that exact same checkpoint:

- **Row C** (control): the current production Stage 2.5 (`stage5_router_kd.py` at its present config).
- **Row P** (paper): the recipe from "Is Retraining-Free Enough? The Necessity of Router Calibration for Efficient MoE Compression" (arxiv 2603.02217, Mar 2026).

The only variable across the two rows is the Stage 2.5 recipe. The Stage 2 output is byte-identical (hardlinked from one save). This isolates the router-KD question cleanly: **does the paper's recipe beat ours, at all, on our model + data, at fixed merge substrate?**

This row is also the long-pending **Stage 2.5 hyperparameter re-audit** that `MOE_COMPRESS_REPORT.md` §3 flags as overdue (the current `epochs=1` / `weight_decay=0.01` / `early_stop_patience=8` were tuned against the discredited pre-2026-05-19 ramp-era reading).

**Total cost**: ~4 H200-hours wall + ~$15 GPU. Runs before any S-series row.

---

## 1. Why this runs first

1. **Result feeds forward**: every S-series row in `SC_STAGE12_COMPREHENSIVE_PLAN.md` ends with Stage 2.5 router-KD. If the paper's recipe wins here, every subsequent row inherits that improvement for free. If our current recipe wins, we stop spending compute investigating the paper's recipe in any downstream stack.
2. **It's the cheapest meaningful experiment we can run**: Stage 2 substrate is the cheap-cost S0 path (~1h), Stage 2.5 is ~1–2h each row. Total ~$15. The S-series's expensive rows (SC = 3h Stage 2 alone) can wait.
3. **It pays off a known debt**: the Stage 2.5 hyperparameters are explicitly flagged in `MOE_COMPRESS_REPORT.md` as un-audited since the temperature-ramp removal on 2026-05-19. This row IS that audit.
4. **It isolates one variable**: by fixing Stage 2 to a single byte-identical checkpoint, every metric difference between Row C and Row P is attributable to the Stage 2.5 recipe. None of the existing S-series rows decouple Stage 2 from Stage 2.5 this cleanly.

---

## 2. The canonical Stage 2 substrate

Run once, save the `stage2_pruned/` artifact, hardlink it into both Stage 2.5 row directories.

**Config block** (delta from the production `qwen36_35b_a3b_30pct.yaml` Stage 2 defaults):

```yaml
stage2_reap_ream:
  cost_alignment: "pre"               # weight-space delta_REAM cost — cheap, closed-form
  capacity_util_threshold: 0          # gate open (otherwise it would force `pre` anyway,
                                      # but pinning it makes the substrate strategy-agnostic)
  assignment_solver: "greedy"         # no Hungarian/MCF/sinkhorn — cheapest solver
  cost_whitening: "none"
  cost_asymmetric: false
  cost_topk_filter: 48                # default
  two_opt_refine: false
  em_refinement_rounds: 0
  expert_distill_steps: 0
  skip_merge_percentile: null         # no skip-merge floor
  merge_heal_enabled: false
```

This is literally the **S0** baseline configuration (per `run_ablations.py:192` — `("S0", {})` with all defaults). Expected wall-clock: ~1h Stage 2 on H200. Stage 2 output `stage2_pruned/` is the canonical substrate for both Row C and Row P.

**Why this Stage 2 instead of SC** (output-space cost):
- It's 3× faster (~1h vs ~3h).
- It produces a model with a *bigger* bpt_gap (SC's quality came from a better cost), which gives Stage 2.5 more headroom to recover — so any difference between recipes is more visible.
- It's byte-deterministic given the same seed and inputs (no per-pair Hungarian non-uniqueness; greedy is fully ordered).

**Why not REAP-only (no merging at all)**:
- Would require a new code path (`merging.py:_merge_experts_inplace` always runs after Stage 2's assignment).
- Adding that path is net-new code — contradicts "before anything else."
- The `pre`-cost merge gives a closer apples-to-apples comparison with the actual production pipeline.

**Stage 1**: GRAPE at default. Hardlinked from a single Stage 1 run; same Stage 1 inputs for both rows.

---

## 3. Row C — current Stage 2.5 (control)

The production `stage5_router_kd.py` configuration as currently committed to `feat/calibration-v2`. No changes to code or config.

Specifically (per `MOE_COMPRESS_REPORT.md` §3 and the live config):

| Knob | Value |
|---|---|
| Trainable params | router weights only (`_freeze_non_routers`, `trainable_name_patterns`) |
| Loss | temperature-scaled vocab-KL between student/teacher final logits, computed in sequence chunks (`_chunked_vocab_kl`, `kd_seq_chunk_size=32` post the 2026-05-19 OOM fix) |
| Optimizer | AdamW |
| Weight decay | 0.01 |
| Temperature τ | 1.0 constant (post the 2026-05-19 ramp removal) |
| Epochs | 1 |
| Steps | ~375 |
| Calibration data | `nvidia/Nemotron-Cascade-2-SFT-Data` |
| Save-best | yes |
| Early-stop patience | 8 |
| Teacher-logits cache | yes (existing pipeline) |
| `merge_repair` (Direction E) | OFF |

Goal: reproduce the current production result on the canonical Stage 2 substrate. This is also half of the variance baseline (see §6).

---

## 4. Row P — paper recipe (S5_RKD)

The recipe from arxiv 2603.02217 ("Is Retraining-Free Enough? The Necessity of Router Calibration for Efficient MoE Compression"). Same module (`stage5_router_kd.py`), same trainable set (router only), same loss family (vocab-KL) — but the four bundled paper-recipe-deltas applied together.

| Knob | Value | Why this value |
|---|---|---|
| Trainable params | router weights only | identical to Row C |
| Loss form | τ² · KL(p_T \|\| p_S) with explicit padding mask m_{t+1} | The paper writes the loss as `L_RKD = (τ² / N_x) · Σ_t m_{t+1} · D_KL(p_T \|\| p_S)`. The τ² scaling is the standard Hinton-distillation form. We must confirm our current `_chunked_vocab_kl` already (a) does forward-KL (teacher→student, not reverse), and (b) applies a padding mask. If not, fix to match before running Row P. |
| Optimizer | AdamW | identical to Row C |
| Weight decay | 0.0 (the paper does not specify; the standard distillation prior is 0) | Tests our current 0.01 decay assumption from the discredited ramp-era tuning |
| Temperature τ | 4.0 (canonical distillation choice, the value the original Hinton 2015 distillation paper uses; the 2603.02217 paper doesn't pin τ, so default to the field convention) | Tests whether the post-ramp-removal `τ=1` is actually undertuned |
| Epochs | 2 (sufficient to expose overfit if it happens; matches the "epochs=2 diagnostic at T=1" the report says is pending anyway) | Tests whether `epochs=1` was undertraining |
| Calibration data | `wikitext/wikitext-103-raw-v1` (raw unlabeled text, the standard distillation calibration corpus; verified accessible on HF Hub) | Tests whether the chat-format SFT data is a distribution mismatch for the next-token KL objective |
| Save-best | yes | identical |
| Early-stop patience | OFF | The current early_stop=8 was tuned against discredited metrics; removing it lets the longer training run to completion |
| `merge_repair` | OFF | Same as Row C — isolates the router-KD recipe variable, not the merge-repair variable |
| Mask | explicit non-pad mask | Match paper's `m_{t+1}`. Verify our chunked KL applies it; fix if not. |

**The bundle**: Row P changes 4 things at once vs Row C — `τ=4` not 1, `wd=0` not 0.01, `epochs=2` not 1, calibration data swap, no early-stop. This is the **paper's recipe** applied as a whole. If Row P wins, we then optionally drill down (§8) to attribute the win to specific deltas. If Row P loses, we already know the paper's recipe doesn't beat ours on our setup and we move on.

**Pre-flight check before Row P**: read `stage5_router_kd.py` and verify:
1. The KL direction in `_chunked_vocab_kl` (forward-KL vs reverse-KL — the paper specifies forward, teacher→student).
2. Whether the padding/special-token mask is applied to the per-token KL contributions.
3. Where `kd_temperature` is consumed; confirm it scales both logits AND the loss by `τ²` (Hinton form).

Any of these three not matching the paper → fix the production code BEFORE Row P. Surfacing such a mismatch is itself a useful outcome.

---

## 5. The recipe-delta table (the core comparison)

| Detail | Row C (current) | Row P (paper) |
|---|---|---|
| Trainable params | router only | router only |
| Loss family | vocab-KL chunked | vocab-KL chunked |
| KL direction | (verify in code; should be forward — teacher → student) | forward (D_KL(p_T \|\| p_S)) |
| Loss τ² scaling | (verify in code) | yes, explicit |
| Padding mask | (verify in code) | yes, explicit |
| Temperature τ | 1.0 | 4.0 |
| Epochs | 1 | 2 |
| Steps (approx) | ~375 | ~750 |
| Calibration data | `nvidia/Nemotron-Cascade-2-SFT-Data` | `wikitext/wikitext-103-raw-v1` |
| Weight decay | 0.01 | 0.0 |
| Early-stop patience | 8 | OFF |
| save-best | yes | yes |
| teacher-logits cache | yes | yes |
| `merge_repair` | OFF | OFF |

---

## 6. Variance baseline (Row 0)

Before running Row C vs Row P, establish a noise floor: re-run **Row C twice with different seeds** on the same Stage 2 substrate. Call these Row 0a and Row 0b.

| Row | Stage 2 | Stage 2.5 | Seed |
|---|---|---|---|
| Row 0a | canonical (hardlinked) | current | seed A |
| Row 0b | canonical (hardlinked) | current | seed B |

Compute the bpt_gap difference `|Row 0a − Row 0b|`. This is the seed-to-seed noise floor for Stage 2.5 alone. Any Row P − Row C delta must be larger than ~2× this floor to be called a real effect.

Row 0a doubles as Row C. So total run count is **3 Stage 2.5 runs** (Row 0a/C, Row 0b, Row P), not 4.

---

## 7. Success criteria

Let `m_C = mean(Row 0a, Row 0b) bpt_gap` and `σ_C = |Row 0a − Row 0b| / 2`.

| Outcome | Bar | Action |
|---|---|---|
| **Decisive paper win** | `m_C − Row P > 2·σ_C` AND no side-metric (ARC-E, HellaSwag, top1_agreement) regresses by >0.01 | Promote Row P's recipe to the new Stage 2.5 production default. Re-run the S-series in `SC_STAGE12_COMPREHENSIVE_PLAN.md` on top of it. |
| **Decisive current win** | `Row P − m_C > 2·σ_C` OR side-metric regression on Row P | Stage 2.5 stays as-is. Do not re-investigate the paper's recipe in the S-series. Mark `R5` in the comprehensive plan as resolved. |
| **Inconclusive** | `|Row P − m_C| ≤ 2·σ_C` | Fire the §8 drill-down. |
| **Bug surfaced** | The pre-flight check (§4) finds the KL direction, mask, or τ²-scaling is wrong in `_chunked_vocab_kl` | Fix the production code first. Then re-run Row 0a/0b/Row P. The fix is the win. |

---

## 8. Drill-down rows (only on inconclusive)

If `|Row P − m_C| ≤ 2·σ_C`, the bundle didn't move the needle — but that could be because the deltas cancel. Run these single-variable rows to attribute:

| Row | Vs Row C, changes | Tests |
|---|---|---|
| P-τ | only `τ=4` (not 1) | Is τ the lever? |
| P-data | only swap to wikitext-103-raw | Is calibration data the lever? |
| P-time | only `epochs=2` + early-stop off (longer training) | Is training-length the lever? |
| P-wd | only `weight_decay=0.0` | Is regularization the lever? |

4 rows × ~1.5h Stage 2.5 each ≈ 6h × $3.39/h = ~$20. Don't run these unless the bundle was inconclusive.

---

## 9. Build order

```
  Pre-flight (read-only — no GPU)
  ─────────────────────────────────
  1. Read stage5_router_kd.py + _chunked_vocab_kl. Verify:
       (a) KL direction is forward (teacher → student)
       (b) Loss applies a padding/special-token mask
       (c) τ² scaling is correct
     If any mismatch — open a fix-it ticket before any GPU spend.
     Estimated time: 1h human, $0 GPU.

  Phase 1 — canonical Stage 2 substrate          (1 × H200 row)
  ──────────────────────────────────────────────────────────────
  2. Run S0-config Stage 1 + Stage 2 once. Save `stage2_pruned/`
     to a stable artifact path. ~1h GPU, ~$3.

  Phase 2 — variance baseline                    (2 × H200 rows)
  ──────────────────────────────────────────────────────────────
  3. Row 0a: hardlink stage2_pruned/ + current Stage 2.5, seed A. ~1h, ~$3.
  4. Row 0b: same as 0a, seed B. ~1h, ~$3.
     Computes σ_C.

  Phase 3 — paper recipe                         (1 × H200 row)
  ─────────────────────────────────────────────────────────────
  5. Row P: hardlink stage2_pruned/ + paper recipe per §4.
     ~1.5–2h, ~$6.

  Phase 4 — verdict                              (no GPU)
  ──────────────────────────────────────────────────────────
  6. Apply §7 decision rules.
  7. If decisive: update `SC_STAGE12_COMPREHENSIVE_PLAN.md` §3 to
     reflect the resolved Stage 2.5 recipe. The full S-series
     proceeds with the winning recipe baked in.
  8. If inconclusive: fire §8 drill-down rows.

  TOTAL: ~4–5h GPU, ~$15 across 4 rows.
```

---

## 10. Risks + halt-triggers

| Risk | Catch | Halt action |
|---|---|---|
| **R1** Pre-flight finds KL direction is reversed (we may be doing reverse-KL, student → teacher). | Read `_chunked_vocab_kl` carefully; the order of `F.kl_div(input, target)` matters in PyTorch. | Fix before any GPU spend. The fix itself may be the entire win — re-run Row 0a/0b/Row P after. |
| **R2** σ_C (seed noise) is large — say 0.05+. Means we can't tell apart any reasonable recipe difference. | Compute σ_C after Phase 2. | If σ_C > 0.05, the Stage 2.5 outcome is noise-dominated and the audit is structurally unanswerable without averaging more seeds. Surface to user; consider extending to 4-seed Row 0 (+2 rows, +~$6). |
| **R3** Stage 2.5 OOMs on Row P. The longer epochs + bigger calibration corpus may exceed the `kd_seq_chunk_size=32` budget. | Watch the first 50 steps of Row P for memory growth. | If OOM: drop `kd_seq_chunk_size` to 16. Numerically identical (KL sums over chunks); doubles I/O overhead. |
| **R4** Row P calibration data (wikitext-103-raw) is structurally too different — model can't learn anything in 750 steps because tokens aren't in distribution. | Watch the raw_kl trajectory; expect monotone decrease. If it stalls flat from step 100, the data is wrong for this loss. | Fall back to a closer corpus: `c4` or a mix `c4:wikitext = 0.7:0.3`. The point is "unlabeled raw text," not specifically wikitext. |
| **R5** The paper recipe wins on bpt_gap but loses on side metrics (ARC-E or HellaSwag drops >0.01). | Side metrics tracked at every Row 0/C/P. | Per §7's cross-metric guardrail: do not promote. Run drill-down §8 to find which delta caused the side-metric drop (most likely the calibration-data swap moving the routing distribution off the eval-style domain). |
| **R6** The canonical Stage 2 substrate has higher bpt_gap than expected (S0 was 0.5767 historically; if we get 0.7+, something regressed). | Compare against S0's historical bpt_gap (0.5767) as the substrate floor. | If substrate is broken, halt — fix Stage 2 before anything else. This would invalidate the entire premise. |

---

## 11. What this plan does NOT do

- Does NOT use SC's output-space cost for Stage 2. Substrate is the cheap `pre` cost (= S0 baseline). The Stage 2.5 audit is invariant to Stage 2 cost choice; cheap is faster.
- Does NOT test the Direction E (`merge_repair`) path. That's a different question (unfreeze merged experts + per-layer MSE) — orthogonal to the router-KD recipe audit.
- Does NOT attempt to beat the current production bpt_gap. The substrate is intentionally worse than SC so Stage 2.5 effects are visible. The headline number stays SC = 0.1293.
- Does NOT change `min_experts_per_layer`, `floor_divisor`, or any other Stage 1 knob. Stage 1 = GRAPE default, hardlinked.
- Does NOT touch the `_LayerInputAccumulator` reservoir, the perm cache, or anything in `SC_FAST_PLAN_V3.md`. Those speed optimizations are not on the critical path of this audit.

---

## 12. Citations (file:line)

### Codebase
- Stage 2.5 entrypoint: `max_quality/src/moe_compress/router_kd/...` (the `stage5_router_kd.py` referenced in the report)
- `_chunked_vocab_kl`: search in `max_quality/src/moe_compress/router_kd/` — function that pre-flight check audits
- S0 baseline config: `max_quality/src/moe_compress/run_ablations.py:192`
- Stage 2 cheap-cost path: `cost_alignment: "pre"` → `max_quality/src/moe_compress/stage2/plugins/ream_cost.py`
- Stage 2 artifact: `stage2_pruned/` (per `MOE_COMPRESS_REPORT.md` §5.3 — recovered SC/SCD by resuming from this)
- `merge_repair` (Direction E, OFF in this plan): `max_quality/src/moe_compress/router_kd/plugins/merge_repair.py`

### Existing plans
- `tasks/MOE_COMPRESS_REPORT.md` §3 (Stage 2.5 description + un-audited hyperparameter caveat)
- `tasks/SC_STAGE12_COMPREHENSIVE_PLAN.md` (the larger plan this row sits in front of)
- `tasks/kd_fix_plan.md` (the historical KD fixes — context for why current hyperparameters are suspect)

### Literature
- arxiv **2603.02217** — Router KD source paper (Section 4 has the L_RKD loss form)
- Hinton et al. 2015 — original distillation with τ² scaling (the canonical convention Row P inherits)

### Datasets
- `nvidia/Nemotron-Cascade-2-SFT-Data` (current calibration) — already on HF Hub, in production pipeline
- `wikitext/wikitext-103-raw-v1` (proposed Row P calibration) — standard distillation corpus, on HF Hub, ~500MB raw text

---

*Generated 2026-05-27 under ml-intern protocol. Read-only spec; no code changes accompany this plan. This row runs BEFORE the S-series in `SC_STAGE12_COMPREHENSIVE_PLAN.md`. Cost ~$15 GPU + 1h human pre-flight. Resolves the Stage 2.5 audit debt and pins the recipe for downstream rows.*
