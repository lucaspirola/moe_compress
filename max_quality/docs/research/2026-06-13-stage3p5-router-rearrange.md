# Would a "Stage 3.5" Router-KD (between Stage 3 and Stage 4) make EoRA more effective?

Read-only architecture analysis. Code root: `max_quality/src/moe_compress/`.
All claims cite `file:line` against the actual implementation.

**Question.** Insert a Router-KD step (identical to Stage 5) BETWEEN Stage 3
(AA-SVD) and Stage 4 (EoRA) — re-train/re-arrange the routers right after the
SVD factorization and before the EoRA low-rank compensation — instead of only
at the end (Stage 5). Does that make Stage 4 more effective?

---

## VERDICT: (b) NO — it would likely HURT or be redundant, AND (c) it is the wrong cheap lever

Short version: **EoRA's compensation is keyed to a per-expert input covariance
`A_cov` that was captured ONCE, under the ORIGINAL (uncompressed) model's
routing, during vLLM calibration.** Stage 3.5 would change which tokens each
expert receives, but it would NOT (and cannot, without re-capture) change
`A_cov`. So Stage 3.5 makes `A_cov` *stale for the new routing* — the exact
opposite of "let EoRA target the tokens that will actually be routed there." And
the current `3→4→5` order is the principled one: EoRA fixes the experts first,
then the router (Stage 5) adapts to the FINAL, compensated experts. Putting the
router adapt in the middle creates a router-adapted-to-pre-EoRA-experts
mismatch that Stage 5 then has to undo anyway.

If anyone wants to test the steelman, the cheap ablation is in §6 — but the
mechanism strongly predicts neutral-to-negative, and the cost is a full extra
Router-KD train (the pipeline runtime long pole).

---

## 1. What each stage actually consumes/produces (code-grounded)

### Stage 3 (AA-SVD) — `stage3/plugins/aa_svd_factor.py`, `covariance_collection.py`
- Factorizes every surviving expert weight `W` into rank-k `U·V` via
  `_aa_svd` / `_aa_svd_precomputed` (`aa_svd_factor.py:265-405`). It is pure
  linear algebra over **covariances**, not a training loop.
- Covariances used:
  - **B / S** = post-prune input Gram `X_post^T X_post` per
    `(layer, expert, matrix)` — collected by hooking the *student's* expert
    inputs (`covariance_collection.py:486-499`, `input_cb` :639-661).
  - **C** = cross-cov `X_pre^T X_post` (gate_proj only) from a dual-forward
    against the teacher (`covariance_collection.py:496-503`, `:662-759`).
  - **A** = pre-prune input auto-cov from Stage 2 (`_stage2_input_covariance.pt`).
    Explicitly **reserved for Stage-4 refinement only** and NOT used in the
    Stage-3 rank-k step (`aa_svd_factor.py:221-230`, `_precompute_eigh` `del A`
    `:230`).
- Stage 3 does NOT touch the router gates. It swaps `ref.mlp.experts` for a
  `FactoredExperts` (`aa_svd_factor.py:688-689`); the `mlp.gate` (router) is
  untouched.

### Stage 4 (EoRA) — `stage4/plugins/eora_compensation.py`, `eora_inputs.py`, `input_cov_cache.py`
- For each `(layer, expert, matrix)`: build the residual
  `ΔW = W_orig − U_e·V_e` (`eora_compensation.py:438-442`), whiten it through
  the eigenbasis of the **input covariance `A`** (`_eigh_spectrum` :182-232,
  `_compute_eora_factors` :235-386), rank-r SVD, back-project, and append the
  correction columns via `fe.widen_rank` (`:854`).
- **The √Λ scaling on `A`'s eigenbasis is the entire point of EoRA**
  (`eora_compensation.py:256-258`, :327-335): it importance-weights the
  correction toward the input directions the expert actually sees. Without `A`
  it degrades to a plain SVD of the residual (`_plain_svd_padded` :290-300).
- `A` per expert is `A_cov.get((layer, e, matrix))`
  (`eora_compensation.py:765`), where up_proj reuses gate_proj's `A`
  (`:763-764`).

### Stage 5 (Router-KD) — `router_kd/orchestrator.py`, `vocab_kd.py`, `trainable_scope.py`
- Trains **routers only**: `_freeze_non_routers` sets `requires_grad` true only
  for params matching `trainable_name_patterns` (`trainable_scope.py:78-80`,
  :189), i.e. `mlp.gate`. Every other param is frozen.
- Loss is vocab-level KL(teacher‖student) (`vocab_kd.py:72-132`,
  orchestrator `:725-798`). It changes the gate weights → changes the
  token→expert routing — but it does NOT change any expert weight (`U/V`).

**Net dependency chain that matters here:**
`A_cov` (per-expert input Gram, fixed at calibration) → EoRA √Λ whitening →
correction. The router determines *which tokens land in each expert's Gram*.

---

## 2. Sub-question 1 — Is EoRA's effectiveness router-dependent? Is `A_cov` stale after Stage 3.5?

**Yes, router-dependent; and yes, Stage 3.5 makes it stale.**

`A_cov` is a per-`(layer, expert, matrix)` Gram. The accumulator keys exactly
that tuple and ingests only the rows for tokens routed to that expert:
- `InputCovarianceAccumulator.update(layer_idx, expert_idx, matrix_name, x)`
  (`utils/activation_hooks.py:983-1023`), where `x` = "tokens routed to expert"
  (`:752` "T is the number of tokens routed to" the expert).
- The cached/V2 form is the same dict shape, captured during vLLM calibration
  on the **original uncompressed model** via `--capture-input-covariance`
  (`stage4/plugins/input_cov_cache.py:1-15`, `:78-103`;
  `stage3/plugins/input_cov_cache.py:1-17`). `sigma_in` carries a `top_k`
  routing field baked in at capture time (`cached_calibration_signals.py:474`,
  `:533-541`, `:697`, `:752-755`).

So `A_cov` is, by construction, **the input distribution of the tokens the
ORIGINAL router sent to expert `e`**. It is computed ONCE, upstream, and only
*loaded* by Stage 4 (`eora_inputs.py:114-217`; `input_cov_cache` short-circuit
`eora_inputs.py:145-156`). Stage 4 never re-derives it.

Now the mechanism:
- If Stage 3.5 re-trains the router, the set of tokens routed to expert `e`
  changes → the *true* input distribution per expert shifts → the matrix EoRA
  *should* whiten against is a different `A_e`.
- But EoRA still reads the **stale** `A_cov[e]` captured under the old routing.
  It would importance-weight the residual correction toward directions that
  matter for the OLD token set, not the NEW one. That is precisely a
  whitening/objective mismatch.

**Quantifying the harm direction.** EoRA solves, per expert,
`min_{rank-r} ‖(ΔW − Corr)·√A_e‖_F` (the `A`-weighted residual; the trackio note
at `eora_compensation.py:881-886` flags that the optimized objective is
`tr(ΔW·A·ΔW^T)`). The correction is optimal for the `A` you feed it. Feeding a
routing-mismatched `A` makes the rank-r budget concentrate on the wrong
eigendirections. Robustness is only partial:
- The **noise-floor truncation** keeps high-energy directions and discards the
  tail (`_eigh_spectrum` :217-232; bounded residual `‖ΔW‖²·Σλ_tail`, docstring
  :43-59). The dominant input subspace of a given expert is somewhat stable to
  modest routing perturbations, so a *small* router move ⇒ a *small* `A` move ⇒
  EoRA degrades gracefully (the whitening is on a similar eigenbasis).
- But Router-KD is not a small move when it actually does work: its whole value
  is re-arranging which experts fire (it is the post-merge/post-compression
  *router calibration* paper, `trainable_scope.py:1-21`). A meaningful routing
  change is exactly the regime where `A_cov` staleness bites.

**Conclusion:** `A_cov` is router-dependent and was captured under the original
routing. Stage 3.5 cannot refresh it (no covariance re-capture is in the
pipeline — Stage 4 only loads). So Stage 3.5 strictly *de-aligns* EoRA's
whitening from the routing EoRA's output will be served under. This is a
headwind, not a tailwind.

---

## 3. Sub-question 2 — Which order is more self-consistent? Is "adapt router last" principled?

**The current `3→4→5` order is the principled one.**

Two competing consistencies:

- **Proposed `3→3.5→4→5`:** the router (3.5) adapts to the SVD-degraded,
  *un-compensated* experts. Then Stage 4 EoRA *changes those experts*
  (`widen_rank` appends correction columns, `eora_compensation.py:854`). The
  router is now tuned to a model state that no longer exists. Stage 5 would then
  have to re-adapt the router again to the post-EoRA experts — so 3.5 is wasted
  motion at best, and at worst it biases the (stale) `A_cov`-keyed routing that
  EoRA then compensates against, baking the mismatch into the corrections.

- **Current `3→4→5`:** EoRA fixes the experts to their final form FIRST (with
  `A_cov` matching the routing that produced the originals — i.e. the routing
  EoRA was designed around). THEN the router (Stage 5) adapts to the FINAL,
  fully-compensated experts. The thing the router calibrates against is the
  deployed artifact. There is no "router adapted to a model state we then
  overwrite."

EoRA itself assumes a fixed routing context: it consumes `originals` (the
pre-factorization weights, `eora_inputs.py:296`) and the calibration `A_cov`,
and its double-widen guard pins it to the Stage-3 ranks
(`eora_compensation.py:848-853`). It is a one-shot closed-form correction with
no notion of "the router will change underneath me." Adapting the router
*before* it, then changing the experts, violates the assumption EoRA is built
on. Adapting the router *after* (Stage 5) respects it. **"Adapt the router last"
is correct: the router should calibrate to the final experts, and the final
experts depend on EoRA, so EoRA must come first.**

---

## 4. Sub-question 3 — Steelman: could Stage 3.5 let EoRA target the tokens that will actually be routed there?

This is the strongest pro-argument, and it fails on the implementation as built.

The steelman: "post-compression the router will send different tokens to expert
`e`; if we re-route *first*, then compute EoRA against the tokens that will
actually arrive, EoRA stops wasting rank budget on tokens the final router won't
send there."

Why it does not hold here:
1. **EoRA does not re-measure `A` after any routing change.** For the steelman
   to work, Stage 3.5 would have to be followed by a *covariance re-capture*
   under the new routing, and EoRA would have to consume THAT. The pipeline has
   no such step — `eora_inputs` only *loads* a pre-existing `A_cov`
   (`eora_inputs.py:114-217`); there is no hook that re-runs calibration after
   Stage 3.5. So "target the tokens that will actually be routed" is impossible
   without also adding a (very expensive) re-capture. Stage 3.5 alone gives EoRA
   a router it can't see and a covariance it can't refresh.
2. **EoRA does not currently "waste" capacity on wrong-token directions in a way
   3.5 fixes.** `A_cov` is *already* the per-expert Gram of the tokens routed to
   `e` (under the original model). The correction is already concentrated on the
   directions those tokens span. The only mismatch is original-routing vs
   final-routing; 3.5 *increases* that mismatch (it moves the routing away from
   the one `A_cov` encodes) rather than closing it.
3. The genuinely correct version of the steelman is **iterative joint
   refinement**: (a) re-route, (b) RE-CAPTURE per-expert covariance under the new
   routing, (c) re-solve EoRA, (d) repeat. That is a real research direction
   (alternating router-adaptation ↔ activation-aware compensation) — but it is a
   different, much heavier algorithm than "drop a Stage 5 clone in the middle,"
   and the pipeline provides none of the re-capture plumbing.

So the steelman, on this codebase, reduces to "add Stage 3.5 AND a full
covariance re-capture AND re-run EoRA" — i.e. a joint/iterative scheme, not the
proposed single insertion.

---

## 5. Sub-question 4 — Paper / theory: does EoRA or AA-SVD prescribe an ordering?

- **EoRA (arXiv:2410.21271, `eora_compensation.py:3-29`)** is a *post-training*
  one-shot eigenspace correction of a fixed compressed model against a fixed
  input covariance `A = X̃^T X̃`. It assumes the activation statistics (hence the
  routing that produced them) are FIXED. It has no concept of adapting routing
  between capture and correction; doing so violates the `A = X̃^T X̃` premise the
  √Λ whitening rests on (Theorem 1 exactness is stated for the captured `A`,
  docstring :31-52). EoRA assumes a fixed routing — full stop.
- **AA-SVD (arXiv:2604.02119, `aa_svd_factor.py:3-58`)** prescribes
  factorize-then-optionally-refine; its quality lever is the block-refinement
  (Algorithm 2 line 9), and it likewise treats the calibration covariances as
  fixed inputs. It says nothing about re-training routers mid-decomposition.
- **Router-KD (arXiv:2603.02217, `trainable_scope.py:1-21`)** is explicitly
  framed as a *post-compression* router calibration — i.e. it belongs AFTER the
  weights are in their final (compressed + compensated) form. That is Stage 5.
- **Precedent for alternating router-adapt ↔ compensation:** none of the three
  papers prescribes it. It would be a novel joint/iterative method (see §4.3),
  not something any of these methods endorses as a drop-in reorder.

No paper supports inserting a router train between factorization and
compensation; two of the three (EoRA, Router-KD) actively imply the opposite
ordering.

---

## 6. Sub-question 5 — Cost / feasibility, and the cheap ablation if you must test it

**Cost.** Stage 3.5 = a full Router-KD train. Router-KD is *the pipeline runtime
long pole* — the only real training stage, default `epochs=2`, ~3000-sample
calibration with teacher forwards (`docs/multigpu_analysis/router_kd.md` §1;
orchestrator `:332`, `:665-1028`; ~70 GB BF16 teacher load
`router_kd.md` §5). Inserting 3.5 and KEEPING Stage 5 ≈ **doubles the most
expensive stage's wall-clock and the teacher-load VRAM churn.**

- Would you keep Stage 5? You'd HAVE to: EoRA changes the experts after 3.5
  (§3), so the router still needs a final calibration to the post-EoRA model.
  So `3→3.5→4→5` pays for TWO router trains where `3→4→5` pays for one — and the
  first one (3.5) trains against a transient pre-EoRA state that gets
  overwritten. A single Router-KD after EoRA (current) is **strictly cheaper and
  expected at-least-as-good** (it adapts to the final model; 3.5 adapts to an
  intermediate one and then needs redoing).
- Replacing Stage 5 with Stage 3.5 (only one train, in the middle) is worse:
  the final router would be calibrated to the pre-EoRA experts, leaving the
  deployed model's router mismatched to its own (compensated) experts.

**The minimum experiment, if you want to falsify the mechanism anyway:**
- **Arms:** A = baseline `3→4→5`; B = `3→3.5→4→5` (3.5 = a Stage-5-identical
  Router-KD via `router_kd.run(..., stage_key=...)`, run on the post-Stage-3 /
  pre-Stage-4 model). Keep Stage 5 in BOTH arms so the only delta is the inserted
  3.5.
- **Critical control:** because EoRA cannot refresh `A_cov`, arm B inherits the
  *same* stale `A_cov` as arm A. To even give the steelman a chance you'd add a
  third arm B′ = `3→3.5→(re-capture A_cov under new routing)→4→5`. B′ requires a
  covariance re-capture (a calibration forward pass producing a fresh
  `_stage2_input_covariance.pt` / V2 sidecar) — that is the expensive part, and
  it is what actually tests §4's real hypothesis.
- **Metric:** Stage-6-alt thermometer score (the project's standing comparator,
  per `reference_pipeline_concepts_solver_bythebook`), plus the EoRA log-only
  residual drop (`stage4_eora.log_residuals=true`,
  `eora_compensation.py:860-873`) per arm to see whether 3.5 helps or hurts the
  `A`-weighted residual EoRA reports.
- **Smallest scale:** one or two MoE layers / a reduced-rank config is enough to
  see the *direction* of the effect on the EoRA residual; you do not need a full
  net-35% run to detect "3.5 worsened the whitening match."
- **Cost gate:** arm B is one extra Router-KD train (long pole ×2). Arm B′ adds a
  full covariance re-capture on top. Given the mechanism predicts B ≤ A and only
  B′ could conceivably beat A, **B′ is the only arm worth the spend — and B′ is
  no longer "insert a Stage 5 clone," it is a joint/iterative method.**

---

## 7. Bottom line

- EoRA's correction is whitened by `A_cov` = the per-expert input Gram captured
  ONCE under the **original** model's routing (`activation_hooks.py:983-1023`,
  `input_cov_cache.py`; loaded-not-recomputed in `eora_inputs.py:114-217`).
- Stage 3.5 changes the routing but cannot refresh `A_cov`, so it makes EoRA's
  whitening **stale for the new routing** — a headwind (§2).
- The current `3→4→5` "adapt the router LAST" order is principled: EoRA produces
  the final experts; the router should calibrate to those final experts, which
  is exactly Stage 5 (§3). Stage 3.5 calibrates the router to a transient
  pre-EoRA state that EoRA then overwrites, so it must be redone anyway.
- No paper (EoRA, AA-SVD, Router-KD) prescribes a mid-pipeline router train;
  EoRA and Router-KD imply the opposite (§5).
- The only version of the user's idea that *could* help is full
  **joint/iterative** router-adapt ↔ covariance-recapture ↔ re-compensate — a
  different, heavier algorithm the pipeline has no plumbing for — and even that
  is speculative (§4).
- Cost: Stage 3.5 ≈ doubles the runtime long pole and still needs Stage 5.
  **Not worth it.** If tested at all, test arm B′ (with re-capture); plain
  arm B is predicted neutral-to-negative.

---

# Addendum: does Stage 2 / Stage 2.5 already stale `A_cov`?

*Follow-up audit (2026-06-13). Tests the prior verdict's consistency: by the
same staleness argument used against Stage 3.5, do the EXISTING Stage 2
(REAP/REAM prune-merge) and Stage 2.5 (Router-KD heal) — both of which run AFTER
calibration and BEFORE Stage 3/4 — already stale the very `A_cov` that EoRA
whitens with? Every claim below is re-grounded against the actual code.*

## A0. Verdict up front (honest, not a defense of the prior doc)

**Yes. The current `2 → 2.5 → 3 → 4 → 5` pipeline ALREADY feeds Stage-4 EoRA a
routing-stale `A_cov`, and the staleness is larger than the prior doc let on.**
There are *two* independent staleness sources baked into the production path,
both upstream of Stage 3.5:

1. **Stage 2 (prune AND merge) stales `A_cov`** — even with no router training.
   The surviving experts' *token membership* changes (dropped/merged experts'
   tokens redistribute; top-k renormalizes over survivors; the residual stream
   shifts downstream `X` for ALL experts). The covariance is never re-measured.
2. **Stage 2.5 (Router-KD heal) stales `A_cov` again** — it re-trains the routers
   on the pruned/merged model (routers-only, experts frozen), explicitly to
   *change* routing, and writes **no** covariance artifact.

So the prior doc's central mechanism ("a router change stales `A_cov` and EoRA
can't refresh it") is **already true of the shipping pipeline before Stage 3.5
is ever considered.** The staleness ship has sailed. This does *partially* weaken
the absolutist framing of the anti-3.5 verdict — but, as §A5 shows, there is a
real asymmetry that keeps the *bottom-line recommendation* (don't add 3.5)
intact, while exposing a genuine latent inefficiency in the current pipeline.

## A1. Pipeline order + what `A_cov` Stage 3/4 actually load (confirmed)

Order is `2 → 2.5 → 3 → 4 → 5`, all in one process for the in-run case:
- `run_pipeline.py:69` — `STAGE_REGISTRY[3] = ("stage3_svd", "stage2p5_final")`:
  Stage 3's declared input is `stage2p5_final`, the **post-Router-KD-heal** model.
- `run_pipeline.py:290-296` — Stage 2.5 invoked as
  `stage5_router_kd.run(..., stage_key="stage2p5")` immediately after Stage 2,
  producing `stage2p5_final`. The comment at `run_pipeline.py:272-273` states the
  *intent* outright: recalibrate routers "so Stage 3 covariance collection sees
  already-adapted routing decisions."
- `run_pipeline.py:511-522` — Stage-3 loader prefers `stage2p5_final` over
  `stage2_pruned`.

**What `A_cov` Stage 3 and Stage 4 load — cache-first, calibration sidecar wins.**
Both stages are two-tier: try the **calibration-V2 sidecar `covariance.pt` FIRST**,
fall back to the Stage-2 remapped `_stage2_input_covariance.pt` only on miss.
- Stage 3: `stage3/orchestrator.py:170-181` dispatches `Stage3InputCovCacheProvider`;
  on hit logs **"Stage 3: V2 input-cov cache HIT (%d keys) — skipping
  `_stage2_input_covariance.pt` load"** (`stage3/orchestrator.py:178-181`); only
  if `A_cov is None` does it call
  `_load_stage2_covariance(artifacts_dir / "_stage2_input_covariance.pt")`
  (`stage3/orchestrator.py:190-191`).
- Stage 4: `stage4/plugins/eora_inputs.py:145-156` short-circuits on the same
  cache hit (log **"Stage 4: V2 input-cov cache HIT … — skipping
  `_stage2_input_covariance.pt` load"**, `:146-148`); else loads
  `artifacts_dir / "_stage2_input_covariance.pt"` (`:162`).
- The sidecar is written by vLLM `--capture-input-covariance` during calibration
  on the **loaded model** and keyed `(layer_idx, expert_idx, matrix_name)` over
  that model's **full expert set** (`cached_calibration_signals.py:676-696`;
  writer keys `(int(li), int(e), name)` over `range(cov.shape[0])`). In our run,
  calibration ran on the **original uncompressed 256-expert model**, so the
  sidecar's expert indices are the **original 0..255**, captured under the
  **original router**.

**In our actual run, Stage 3 logged "V2 input-cov cache HIT (20480 keys)"** — i.e.
the calibration `covariance.pt` (original-model, original-routing) WON the cache.
So both Stage 3 and Stage 4 whiten with the **raw original-model calibration
covariance**, which the prune/merge remap never even touched (the remap only
affects the *fallback* `_stage2_input_covariance.pt`, which was bypassed).

**Is there ANY re-capture/refresh of `A_cov` after Stage 2 or 2.5?** No.
- Stage 2 only ever does a **pure index remap** of the pre-merge accumulator
  (§A2). No forward pass re-measures covariance post-merge.
- Stage 2.5 / Router-KD writes **zero** covariance: a full grep of `router_kd/`
  for cov/Gram/SYRK/`InputCovarianceAccumulator`/`_stage2_input_covariance`
  returns nothing functional; its only output is the model checkpoint
  (`router_kd/orchestrator.py:1052-1058` → `{stage_key}_final`). Confirmed.

## A2. Does Stage 2 (merge) remap `A_cov`? — pure index relabel, routing-blind

When Stage 2 **merges**, the covariance is **re-keyed by a pure index remap**,
NOT re-captured:
- `stage2/plugins/layer_merge.py:680` — `write_artifacts` calls
  `_remap_covariance_for_layer(cov_acc, layer_idx, final_kept_ids)` before the
  snapshot.
- `stage2/shared_io.py:308` builds `id_to_new = {old: new for new, old in
  enumerate(kept_ids)}`; `:324-326` relabels each surviving key
  `(li, eidx, name) → (li, id_to_new[eidx], name)` and **carries the SAME tensor
  object `val` verbatim** (`new_cov[new_key] = val`); `:320-323` **drops** pruned
  experts' Grams entirely.
- Written to `_stage2_input_covariance.pt` via `stage2/orchestrator.py:1728` →
  `_save_covariance` (`shared_io.py:216-298`) — a snapshot of the in-memory
  accumulator, **no forward pass**.

So the merged expert's stored `A` is the **original calibration Gram of the
single original expert that now occupies that survivor slot, copied verbatim** —
NOT averaged across the merge group's constituents, and with **zero accounting**
for the fact that post-merge routing sends a *different* token set to that
merged slot. This is option **(a)** from the question: a pure index remap that
ignores the routing change. And in the production V2-cache path it's worse than
even that — the relabeled file is **bypassed** (§A1), so Stage 3/4 use the raw
original-expert calibration Grams keyed positionally against the current
(post-merge) expert index, with **no original↔survivor reconciliation in the
consumer at all** (`eora_compensation.py:764-765` does a silent
`A_cov.get(cov_key)`; a key miss degrades silently to plain SVD).

## A3. Does Stage 2 prune ALONE already stale `A_cov`? — yes

REAP faithful-prune drops experts and the forward renormalizes top-k over
survivors (`reap_prune.py:394-410`; the router rows for dropped experts are
removed). The router is *not* retrained, but:
- Tokens that the original router sent to a now-dropped expert are **reassigned**
  to survivors (top-k over the surviving set), so each survivor's true token
  membership grows/shifts.
- Upstream pruning changes the **residual stream**, so downstream layers' hidden
  states `X` — the inputs to *every* surviving expert's Gram — shift too.

The calibration `A_cov` encodes neither: it is the original 256-expert routing's
per-expert Gram. So **even before Stage 2.5, `A_cov` is already mismatched to the
post-prune routing.** (Note: faithful-prune itself writes only an *empty*
sentinel covariance, `reap_prune.py:473-484`, and in pure-prune mode Stages 3+
are typically skipped — but in the REAP/REAM *compression* arms that DO run
Stage 3/4, the consumed `A_cov` is the original-model sidecar, which the prune
never refreshed.)

## A4. Does Stage 2.5 (Router-KD heal) stale `A_cov` further? — yes

- Stage 2.5 trains **routers only**: `_freeze_non_routers`
  (`router_kd/plugins/trainable_scope.py:78-80`) sets `requires_grad=True` only
  for names matching the trainable patterns (`mlp.gate`); everything else frozen
  (`trainable_scope.py:189`, orchestrator comment `:518-520`). It runs a full
  optimizer loop over those router params (`orchestrator.py:761` fwd, `:850` bwd,
  `:876` step), so the gate weights move → **token→expert routing changes** on the
  pruned/merged model. (Caveat: an optional, default-OFF `merge_repair` flag can
  also unfreeze merged-centroid expert rows at `stage2p5` only,
  `merge_repair.py:489-494` — still routers-plus-centroids, never the general
  experts, and off by default.)
- Stage 2.5 writes **no covariance** (§A1). So the routing shift it deliberately
  induces is **never reflected** in the `A_cov` that Stage 3/4 subsequently load.
- It runs in window-1 (`2 → 2.5 → stage2p5_final`) before Stage 3 (window-2,
  `run_pipeline.py:290-296`, `:69`, `:511-522`), exactly as the question states.

So Stage 2.5 adds a **second** routing perturbation on top of Stage 2's, and like
Stage 2 it cannot and does not refresh `A_cov`.

## A5. Reconciliation — does this break the anti-3.5 verdict?

**(a) Is the current 3→4→5 pipeline already consuming a 2- and 2.5-stale `A_cov`
for EoRA?** Yes, unambiguously. Both the prune/merge routing change (§A2-A3) and
the Router-KD heal routing change (§A4) occur before Stage 3, and the EoRA
whitening covariance is the original-model calibration sidecar, never re-measured
(§A1). The pipeline tolerates *exactly* the staleness the prior doc warns about
for 3.5.

**(b) Is this a KNOWN/accepted approximation?** It is **explicitly intended**, and
that intent reveals the design's own theory of the staleness. The comment at
`run_pipeline.py:272-273` says Stage 2.5 is ordered before Stage 3 *"so Stage 3
covariance collection sees already-adapted routing decisions."* That sentence is
about **Stage 3's B/S/C covariances** — the *fresh, re-collected* Grams that
Stage 3 computes by hooking the live (post-2.5) student
(`covariance_collection.py:486-503`). It is **NOT** about EoRA's `A` — which is
the stale calibration sidecar. So the pipeline's own design *accepts* `A`-staleness
as a deliberate approximation: it treats the **per-expert input covariance `A` as
approximately routing-invariant**, while taking care to make the **B/S/C**
covariances routing-current. That is the principled asymmetry, and it is the key
to reconciling the two analyses.

**(c) Does this make the anti-3.5 verdict INCONSISTENT?** Partially — the prior
doc *overstated* the novelty of the staleness. "Stage 3.5 makes `A_cov` stale" is
not a fresh harm 3.5 introduces; `A_cov` is *already* stale by the time Stage 3
runs. To that extent the prior doc's §2 framing ("the exact opposite of letting
EoRA target the tokens that will actually be routed") is too strong: EoRA never
sees the routing-current tokens *anyway*, with or without 3.5. **Honest
correction: the marginal staleness 3.5 adds is incremental, not categorical.**

**BUT there is a real, load-bearing asymmetry that the prior verdict's bottom
line survives on — and it is NOT the one the prior doc emphasized:**

1. **`A`-staleness is the design's accepted regime; the prior doc should have
   argued "3.5 adds MORE of an already-tolerated error," not "3.5 introduces a
   new error."** The correct anti-3.5 argument is *monotonicity*: Stage 2.5 is
   one router move the design already eats; inserting Stage 3.5 stacks a *second*
   router move whose drift compounds — and crucially it lands **after** the
   Stage-3 SVD factorization that EoRA must compensate. The residual
   `ΔW = W_orig − U·V` that EoRA corrects is computed against the Stage-3 factors;
   if 3.5 moves routing *between* Stage 3 and Stage 4, the `A` that EoRA whitens
   `ΔW` with is stale **relative to the very factorization it is repairing**,
   whereas 2/2.5 staleness is "upstream" — it perturbs routing *before* the
   factorization is even formed, so Stage 3 at least builds its B/S/C and its
   factors *under the post-2.5 routing*. In short: **2.5's drift is absorbed into
   the factors EoRA repairs; 3.5's drift is injected after them, so EoRA's
   correction targets a routing that no longer matches either its factors or its
   `A`.** That asymmetry is genuine and keeps 3.5 on the wrong side.

2. **Stage 5 still has to undo a mid-pipeline router move (§3 of the main doc
   stands).** 3.5 calibrates the router to pre-EoRA experts that Stage 4 then
   overwrites; the cost (one extra Router-KD train, the runtime long pole) buys a
   router state that must be redone. That argument never depended on `A`-staleness
   and is unaffected by this addendum.

So: **the anti-3.5 recommendation stands, but for a corrected reason.** Drop the
"3.5 uniquely stales `A`" framing (false — it's already stale). Keep "3.5 adds a
post-factorization router move that (i) compounds an already-tolerated `A`-drift
at the worst possible point in the chain and (ii) must be re-undone by Stage 5."

**(d) Does this reveal a latent inefficiency worth fixing in the CURRENT
pipeline, independent of 3.5?** **Yes — and this is the most actionable finding.**
The pipeline deliberately refreshes B/S/C for Stage 3 under post-2.5 routing
(`run_pipeline.py:272-273`) but leaves EoRA's `A` frozen at original-model
calibration. That is an *internal inconsistency*: Stage 3 is fed routing-current
second-moments while Stage 4 — which runs on the *same* post-2.5 model moments
later — is fed routing-stale ones. The candidate fix is to **re-capture (or
remap-with-routing) `A` under the post-2.5 model before Stage 4**, reusing the
Stage-3 hook infrastructure that already collects per-expert input Grams on the
live student (`covariance_collection.py:486-499`, `input_cb` `:639-661`) — Stage 3
*already* walks every token through the post-2.5 model and accumulates per-expert
input covariance; EoRA's `A` could be sourced from THAT pass instead of the
calibration sidecar, at **near-zero extra cost** (the forward is already
happening). This would make EoRA's whitening consistent with the routing the
model is actually deployed under, closing the same gap the prior doc worried 3.5
would *open* — but doing it *without* the extra Router-KD train, and *without*
reordering the pipeline.

   - **Caveat / why it's "candidate," not "obviously a win":** EoRA's `A` is the
     **pre-prune input auto-cov** in the AA-SVD formulation (the main doc §1 notes
     `A` is "reserved for Stage-4 refinement," `aa_svd_factor.py:221-230`), and the
     EoRA paper's Theorem-1 exactness is stated for the *captured* `A`
     (`eora_compensation.py:31-52`). Whether "`A` = post-2.5 re-collected per-expert
     Gram" strictly matches the EoRA objective, or whether the calibration `A` is
     intentionally the *broader* original-routing distribution (more robust, less
     overfit to the post-2.5 token assignment), is a real question — it needs an
     A/B: arm A = current (calibration `A`); arm A* = EoRA whitened by the
     Stage-3-collected post-2.5 `A`. This is **strictly cheaper than any 3.5 arm**
     (no extra train, reuses an existing forward pass) and tests the actually-useful
     hypothesis: *does making EoRA's whitening routing-current help?* If A* ≥ A, the
     current pipeline has a free quality win and the 3.5 idea is doubly moot; if
     A* ≤ A, the calibration `A`'s routing-invariance is a *feature* and the prior
     doc's instinct (don't chase routing-current `A`) is vindicated empirically.

## A6. Addendum bottom line

- The shipping `2 → 2.5 → 3 → 4 → 5` pipeline **already feeds EoRA a routing-stale
  `A_cov`** — staled by Stage 2 prune/merge (pure index remap, §A2-A3) and again
  by Stage 2.5 router heal (no cov written, §A4); both refresh nothing, and the
  consumed `A` is the original-model calibration sidecar that won the V2 cache
  ("HIT 20480 keys" in our run, §A1).
- This **weakens the prior doc's *framing*** ("3.5 uniquely makes `A` stale" — it's
  already stale) **but not its recommendation.** The corrected anti-3.5 argument:
  3.5 stacks a *second*, *post-factorization* router move that compounds an
  already-tolerated `A`-drift at the worst point in the chain and still needs
  Stage 5 to undo it.
- The pipeline's own intent (`run_pipeline.py:272-273`) reveals the principled
  asymmetry: it makes **B/S/C** routing-current for Stage 3 but accepts **`A`** as
  approximately routing-invariant for Stage 4. That asymmetry is the real reason
  the two analyses reconcile.
- **The actionable finding is NOT about 3.5 at all:** there is a latent
  inconsistency — Stage 4's `A` is routing-stale while Stage 3's B/S/C (collected
  on the *same* post-2.5 model) are routing-current. The cheap, no-extra-train fix
  is to source EoRA's `A` from the Stage-3 live forward pass that already runs.
  Worth one A/B (arm A* = post-2.5-collected `A`) — strictly cheaper than any 3.5
  experiment and it tests the genuinely useful question.
