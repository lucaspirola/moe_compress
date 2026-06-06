# Design — REAP vs REAM × healing-recipe probe (pre-ablation)

**Date:** 2026-06-06
**Status:** design, awaiting implementation plan
**Author:** Lucas + Claude

## 1. Goal

Cheaply decide the Stage-2 direction before committing to a full Stage-2
ablation. Produce **6 small models** and score them with the cheap Stage-6-alt
"thermometer" eval, then pick:

- the better **base compression**: REAP (prune experts) vs REAM (merge experts), and
- the better **repair recipe**: our current dials vs the paper's dials.

This is a probe, not the ablation — the output is a *direction*, not a final model.

## 2. The 6 models

All built from `Qwen/Qwen3.6-35B-A3B`, all using our calibration data
`qwen3-pretrain-mix-v2`. "35%" = **35% expert reduction at Stage 2 only**
(no SVD, no Stage 3/4/5-beyond-healing).

**REAP branch** (prune 35% of experts by saliency):
1. REAP base — no repair
2. REAP + our-recipe healing
3. REAP + paper-dials healing

**REAM branch** (merge experts down to ~65% of original count):
4. REAM base — no repair
5. REAM + our-recipe healing
6. REAM + paper-dials healing

**Eval:** Stage-6-alt thermometer on **wikitext** for all 6 → compare → decide.

## 3. "Paper-faithful" Stage-2 plugin sets

Both bases run only the core paper algorithm; all project refinements OFF.

**REAP base** — ON:
- `reap_scoring` (Eq. 9 saliency) + the prune action (drop lowest-saliency
  experts to hit 35%).

**REAM base** — ON:
- `ream_cost` (similarity cost, Eqs. 5/7/8)
- `solver_greedy` (paper-faithful descending-saliency greedy assignment)
- `layer_merge` (REAM Eq. 6 merge + sequential orchestration)
- `ream_sequential` (the §4 sequential propagation — re-forward each merged
  layer before deciding the next)

**OFF for both (project add-ons / non-paper variants):**
`merge_heal`, `expert_distill`, `em_refine`, `two_opt_refine`,
`capacity_gate`, `skip_merge_floor`, `output_space_cost`, `ream_cost_post`,
`regmean_merge` (alternative merge op), and the advanced solvers
`solver_hungarian` / `solver_mcf` / `solver_sinkhorn` / `solver_auto`
(paper uses greedy).

> The exact config keys that toggle each plugin are to be confirmed in the
> implementation plan (the `stage2_reap_ream` config block + plugin registry).

## 4. The repair (healing) step

Same mechanism for both recipes — vocab-KL distillation teaching the
compressed model to imitate the original (`router_kd` / `merge_repair`). The
two recipes differ only in dials, **both on `qwen3-pretrain-mix-v2`** (our
data — no Wikipedia, per the "our dataset everywhere" rule):

| Dial | our-recipe | paper-dials |
|---|---|---|
| kd_temperature (output "blur") | 1.0 | 4.0 |
| weight_decay | 0.01 | 0.0 |
| epochs (passes) | 1 | 2 |
| early_stop_patience | 8 | 0 (off) |
| calibration source | qwen3-pretrain-mix-v2 | **qwen3-pretrain-mix-v2** (NOT wikitext) |

## 5. Dataset rule (project-wide)

The old Nemotron-Cascade dataset is retired everywhere. Stage 2 and the repair
step already read the shared `config.calibration.source = qwen3-pretrain-mix-v2`.
Two spots must change to honor the rule:
- `rkd_paper_recipe` currently force-overrides the source to `wikitext-103`;
  the probe needs a **dials-only mode** that keeps our source.
- Stage-6-alt eval corpus defaults to a Nemotron held-out slice; the probe
  sets it to **wikitext** (already supported) as a neutral yardstick.

## 6. Dependencies & sequencing

- **Gated on the calibration run** (the vLLM self-traces job on the spot
  H200). REAP consumes the `reap_scores` sidecar; REAM consumes the
  gate-logit / input-cov / expert-output sidecars. Full sidecars are final at
  calibration end (ETA ~21:30 UTC 2026-06-06).
- **Run right after calibration, reusing the same volume** (`0a2fda41`) — it
  already holds the teacher, the v2 data, the venv, and all sidecars. No
  re-staging.
- **Hardware:** spot 1×H200 is viable — Stage 2 checkpoints **per layer**
  (`stage2/resume.py`, atomic `.tmp`+`os.replace`), so a preemption resumes
  from the last completed layer, exactly like calibration's per-chunk resume.
  Healing checkpoints per epoch/step. Stage-6-alt is cheap and re-runnable.

## 7. Code changes required

1. **dials-only RKD mode** — `rkd_paper_recipe`: apply the 4 numeric dials but
   keep `calibration.source` unchanged (don't force wikitext).
2. **Stage-2-only 35% config** — a config variant: 35% expert reduction, SVD
   off, Stages 3/4 off, Stage 5 = the healing step only, Stage-6-alt on
   wikitext. Paper-faithful plugin sets per §3.
3. **Stage-6-alt corpus = wikitext** (config flag, already supported).
4. **Orchestration** — produce the 6 models (2 bases → each + 2 healed) and
   run Stage-6-alt on each, writing a comparison table.

## 8. Decision criteria

From the Stage-6-alt thermometer metrics on wikitext (bits-per-token / the
zero-shot subset), pick the winning base (REAP vs REAM) and repair recipe.
That choice seeds the full Stage-2 ablation.

## 9. Out of scope

SVD / Stage 3-4, the full ablation grid, any merge of these probe models into
`main`, and FP8 teacher variants.
