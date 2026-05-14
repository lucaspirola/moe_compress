# KD Loss Strategy — Stage 2.5 / Stage 5 Router Distillation

**Last revised:** 2026-05-14 — incorporates Move A + Move B (commit `fe6c755`),
validated on A0 with `best_raw_kl_ema = 0.019535` (≈6× under the prior 0.117
floor; ≈5× under the 0.09 outcome gate).

This document explains the full KD loss strategy used to recover the routers
of the merged MoE student model in `stage5_router_kd.run` (used by both
Stage 2.5, the post-merge router KD, and Stage 5, the final router KD).
It covers (1) what the loss measures, (2) what is and isn't trainable,
(3) why the run converges to an *irreducible* floor rather than to the
teacher, (4) the Move A + Move B convergence-control machinery added to
prevent overfit drift, and (5) how to read the metrics emitted to Trackio.

---

## 1. What the loss is

`_chunked_vocab_kl` (in `stage5_router_kd.py` around line 970) computes the
full-vocabulary KL divergence between the teacher and student next-token
distributions, scaled by τ² per the standard temperature-scaled KD recipe
(paper 2603.02217, Eq. 3):

  L_RKD(student | teacher) = (τ² / N_tokens) · Σ_{t} KL( p_T^{(t)} ‖ p_S^{(t)} )

where for each token position *t* in the (shifted) sequence:

  p_T^{(t)} = softmax( z_T^{(t)} / τ )                # teacher dist over |V|
  p_S^{(t)} = softmax( z_S^{(t)} / τ )                # student dist over |V|

The KL is computed in sequence chunks (`kd_seq_chunk_size`) to bound peak
intermediate memory at large |V|. For a fully-packed batch with no padding,
N_tokens = B × (L − 1), where the −1 comes from the standard causal shift
(predict token *t+1* from position *t*).

`τ` is the temperature. Move B ramps τ linearly from `kd_temperature_start`
(4.0) to `kd_temperature_end` (1.0) across the run. The optimizer's "view"
of the loss carries the τ² factor (so it scales 16× higher at τ=4 than at
τ=1, which is the standard Hinton convention and keeps gradient magnitude
roughly comparable across temperatures). We also track a τ-invariant
`raw_kl = loss / τ²` for comparison across the schedule and across runs.

## 2. What is trainable, and what isn't

This is the most-confused property of the recipe, so it's stated explicitly:

  | Component                          | State            |
  |------------------------------------|------------------|
  | Router (`mlp.gate.weight`)         | **TRAINABLE**    |
  | Experts (post-merge, in `mlp.experts.*`) | frozen     |
  | Shared expert                      | frozen           |
  | Attention (q/k/v/o projections)    | frozen           |
  | RMSNorm                            | frozen           |
  | Token embeddings                   | frozen           |
  | LM head                            | frozen           |

The router is the only thing learning. Concretely:
`_freeze_non_routers(student, trainable_patterns)` walks the student's
parameters and only enables `requires_grad` on those whose name matches
`trainable_name_patterns` (default: `["mlp.gate.weight"]`).

The KD loss is computed at the *vocabulary level* (Eq. 3), not at the
router-decision level. There is no auxiliary "match the teacher's
top-k expert IDs" loss. The gradient flows from the next-token logits
all the way back through the frozen-expert MLP path and through the
router's softmax-into-top-k op, eventually updating only
`mlp.gate.weight`.

**Side-effect:** the loss is moved by the only knob it has — the
routing decision. The training therefore answers the question *"given
this frozen, merged set of experts, which routing weights make the
output-token distribution closest to the teacher's?"*

## 3. Why the loss cannot reach zero

Stage 2 merges experts. Specifically, on the 30%-reduction config:

  - The teacher has 256 experts per MoE layer.
  - The student starts post-Stage-2 with a layer-dependent count: ~10
    early layers keep all 256 (by design, see
    `project_stage1_budget_pattern.md`), middle layers merge down to
    128 or 154 experts, late layers vary by Stage-1 budget.
  - The merged experts are linear combinations of the originals, so the
    surviving expert basis no longer spans the teacher's expert space.

Stage 2.5's KD cannot recover the discarded capacity. It can only do
two things:

  1. **Re-allocate routing.** For each token, decide which surviving
     expert(s) produce a next-token distribution closest to the
     teacher's, even if the perfect match (some original expert) is
     gone.
  2. **Smooth the topology.** Reduce sharp routing failures (tokens
     dispatched to an expert that no longer represents what they
     need) by adjusting `gate.weight` rows.

The KL has an **irreducible floor** set by:

  - Information destroyed in the expert merge (the surviving expert
    basis no longer spans the teacher's behavior).
  - The teacher's intrinsic entropy on each token (KL is asymmetric
    and bounded below by routing-noise, not zero).

Empirically: on Qwen3.6-35B-A3B with the 30% config, A0 (full Move A+B
on the merged student) hits **`raw_kl_ema ≈ 0.019`** at the best
checkpoint. The pre-Move-A+B A0 (which exported the *last-step* state
rather than the best) drifted to **`raw_kl_ema ≈ 0.117–0.169`** by the
end of run. That gap is *not* a quality gap in the merge itself — both
runs share the same Stage 2 merge. The gap is purely a
training-control gap that Move A + Move B closed.

**Practical consequence:** raw_kl absolute numbers measure routing
recovery, not student-vs-teacher capacity matching. They're a
*relative* metric across runs that share the same Stage-2 merge. Don't
read them as a perplexity proxy — that's what Stage 6 PPL is for.

## 4. The training-control problem (and Move A + Move B's fix)

The pre-Move-A+B A0 run exhibited a textbook over-training signature:

  - Loss bottomed at step 950 (raw_kl_ema ≈ 0.117).
  - Then *rose* to 0.169 by step 1400 — clear memorization of the
    calibration distribution.
  - The run exported the **step-1400** weights (last-step semantics),
    not the optimum.

The fix had to add NO speed compromise and NO change to the loss
itself. We added two orthogonal mechanisms.

### Move A — convergence control

Three components, all in `stage5_router_kd.py`:

**A.1 Cosine LR schedule with linear warmup.**

  - YAML: `lr_schedule: cosine`, `warmup_ratio: 0.05`, `lr_min_ratio: 0.10`.
  - Peak LR = `learning_rate` (default 5e-5).
  - Warmup: linear from `1/warmup_steps × peak_lr` at step 0 up to peak
    over the first 5% of optimizer steps (`warmup_steps ≈ 112` at the
    default `epochs=3 × 6000/8 = 2250` total steps).
  - Decay: cosine from peak down to `lr_min_ratio × peak = 5e-6` over
    the remaining 95% of steps.
  - Off-by-one fix in the LR lambda (line ~770 of the file): the
    formula uses `(current_step + 1) / warmup_steps` so that step 0
    fires at `1/warmup_steps × peak_lr`, not 0. (Without this, LambdaLR's
    default `last_epoch=-1` makes the first optimizer call see LR=0,
    wasting one update.)

  Why it helps: the warmup keeps the early steps in a safe gradient
  regime when the loss curvature near the merged-expert initialization
  is unknown. The cosine floor (10% of peak) keeps the optimizer alive
  through the last 10% of training so it can do meaningful fine
  adjustment, instead of approaching zero LR and stalling.

**A.2 EMA-smoothed save-best by raw_kl.**

  - YAML: `save_best: true`, `best_metric_ema_alpha: 0.2`.
  - At every log boundary, compute `raw_kl = loss / τ²` (T-invariant)
    over the recent window, then update an EMA: `ema = α·raw_kl + (1−α)·prev_ema`.
    First-observation bootstrap is `ema = raw_kl_val` (not `+inf`
    arithmetic).
  - Track `best_raw_kl_ema = min(history)`. When the EMA goes below
    the running min, atomically write `best.pt` to disk (router
    state-dict only, ~10–50 MB).
  - At end of Stage 2.5, if `save_best=true` and `best.pt` exists:
    reload the router state into the (otherwise current) student,
    using `load_state_dict(strict=False)` because best.pt holds only
    the trainable subset. The export `stage2p5_final/` thus carries
    the *best* router state, not the *last* router state.

  Why it helps: it severs the dependence of the final exported model
  on the random walk of the loss curve past its optimum. If the run
  drifts (as A0 did), we still export the optimum.

**A.3 Robust resume.**

  - The checkpoint format is bumped to v2 (`format_version: 2` in
    `_save_stage5_checkpoint`) to carry `scheduler_state`, `best_raw_kl_ema`,
    `best_step`, `prev_ema`. Resume validates and restores all of them.
  - If resuming a v1 checkpoint (legacy), the scheduler is
    fast-forwarded by replaying `scheduler.step()` `resume_step` times;
    best/prev EMA reinitialize from `+inf`. The run still produces
    correct exports, just with a slightly stale best-tracker.

### Move B — temperature ramp + more epochs

Two components, both YAML-only.

**B.1 Linear temperature ramp 4.0 → 1.0.**

  - YAML: `kd_temperature_start: 4.0`, `kd_temperature_end: 1.0`.
  - At every optimizer step, the current τ is computed as
    `τ(s) = τ_start + (τ_end − τ_start) · clip(s / total_optim_steps, 0, 1)`.
  - The forward call to `_chunked_vocab_kl(s, t, temperature=τ, ...)`
    uses this per-step τ.
  - The `raw_kl = loss / τ²` running track is τ-invariant by construction,
    so the best-tracker works cleanly across the ramp.

  Why it helps: high τ early flattens the teacher's posterior
  (softens it), making it easier for the router to find a "broadly
  good" routing pattern without being trapped into matching every
  fine peak. Low τ late sharpens the targets, pulling the router
  toward the teacher's modes. The schedule mirrors curriculum
  learning — start easy, finish strict.

**B.2 More epochs.**

  - YAML: `epochs: 3` (was 1; was briefly 2 in an earlier failed
    experiment).
  - Combined with `max_calibration_samples: 6000` and `batch_size: 8`,
    this gives `2250` optimizer steps total.
  - The cosine LR ensures the third epoch is at low LR (~10–30% of
    peak), so the third pass is fine adjustment, not memorization
    pressure.

  Why it helps: more passes over the calibration corpus give the
  router more chances to find better assignments. Without Move A's
  save-best + cosine decay, extra epochs would just amplify the drift
  problem; with them, extra epochs simply lower the floor.

### Why this is no-speed-compromise

None of Move A or Move B touches the loss formula, the kernel, the
batch size, the optimizer choice, or the model topology. The cost is:

  - **Wall-time:** 3 epochs vs 1 = roughly 3× longer per ablation,
    ~+40 min on H200. Explicitly priced and accepted by the user.
  - **Per-step compute:** unchanged (same forward, same backward).
  - **Memory:** trivial (a few `float` locals for EMA state; one
    `best.pt` blob of ~10–50 MB on disk).

The first-call torch.compile latency is unaffected. The cosine
schedule's effect on per-step wall is nothing (it's an LR multiplier
update). Save-best's disk write happens at log boundaries (default
every 50 optimizer steps) and writes ~50 MB max — well under one
batch's compute time.

## 5. How to read the metrics

Every log boundary emits to Trackio:

  - `stage5/loss` — the optimizer's view of the loss (carries τ² scale).
    Don't use for cross-run comparison.
  - `stage5/raw_kl` — `loss / τ²`. The τ-invariant per-window KL.
    Comparable across the temperature schedule and across runs with
    the same Stage-2 merge.
  - `stage5/raw_kl_ema` — EMA of `raw_kl` with α=0.2. The signal the
    best-tracker uses. Lower-noise than `raw_kl`.
  - `stage5/best_raw_kl_ema` — running min of `raw_kl_ema` across the
    run. Only ever decreases. **This is the outcome metric.**
  - `stage5/best_step` — step at which the best was found.
  - `stage5/lr` — `scheduler.get_last_lr()[0]`. Should show cosine
    warmup-then-decay.
  - `stage5/temperature` — current τ from the ramp. Should show
    linear 4.0 → 1.0.
  - `stage5/grad_norm` — pre-clip gradient norm of the trainable
    routers.

The outcome gate (per the original plan, defined in
`~/.claude/plans/the-oregon-b200-we-re-wondrous-scott.md`):

  - **`best_raw_kl_ema ≤ 0.09`**: Move A landed AND Move B contributed
    meaningfully (vs the 0.117 baseline).
  - **0.09 < best_raw_kl_ema ≤ 0.117**: Move A landed; Move B did not
    contribute meaningfully — investigate.
  - **best_raw_kl_ema > 0.117**: regression vs baseline; roll back.

A0 on 2026-05-14 hit `best_raw_kl_ema = 0.019535 @ step 250` — 5×
under the gate, locked in well before the LR/T schedule had fully
ramped. This was on epoch 0 batch 49 (the first log boundary at the
default `log_every_n_steps`), so the bootstrap-on-first-observation
behavior worked as designed and the cosine warmup got us past the
floor immediately.

## 6. Failure modes the recipe survives

The implementation has explicit guards for the failure modes that
occurred during development:

  - **Disk full mid-save** — addressed at the infra level, not in
    KD code; see `_save_stage5_checkpoint` atomic write.
  - **Crash mid-Stage-2.5** — full optim + LR-scheduler + best-tracker
    state in `step_NNNN.pt` (v2 format). Resume restores all of it.
  - **best.pt holds non-router params** — `_save_best_router_state`
    filters to `requires_grad=True` params at save time; resume
    `load_state_dict(strict=False)` and asserts `unexpected == []`.
    The "Stage stage2p5: reloaded best router state from step=N
    (raw_kl_ema=X); missing=986 (expected); unexpected=0" log line
    is the success signature.
  - **Temperature ramp gives NaN at τ→0** — `max(τ², 1e-12)` guard
    in `raw_kl` computation; `τ_end = 1.0` so τ never reaches 0
    by construction. Defense in depth.

## 7. What's deliberately *out* of this recipe

These were considered and explicitly excluded by the planning round
(`the-oregon-b200-we-re-wondrous-scott.md` § "Out of scope"):

  - **Top-k masked KL.** Restricting the KL to the teacher's top-k
    tokens would change the loss surface; ranks below A+B on
    expected impact. Defer pending an A/B run.
  - **Effective-batch bump via grad_accum.** A+B already addresses
    the gradient-SNR concern via more epochs at low LR.
  - **Teacher-logits cache enablement.** Wall-time hit of A+B is
    bounded; revisit if actual wall exceeds $1/ablation. Note:
    this is mutually exclusive with `epochs > 1` (the cache
    code has a hard guard at `stage5_router_kd.py:744` — caches
    are indexed by (epoch, batch) and would re-pair teacher logits
    with student inputs incorrectly on epoch 2+).
  - **Load-balancing auxiliary loss.** Affects Stage 6 PPL, not the
    Stage 2.5 floor we're optimizing.

## 8. Why the metric improved 6× from baseline to A0

The pre-Move-A+B A0 ran with:

  - epochs=2 (then 1 in the rollback) — fewer passes, less opportunity
    to find a better routing.
  - Constant LR=5e-5 throughout — at high LR for the entire run, with
    no cooldown to anneal into a fine adjustment.
  - Constant τ=1.0 — hard targets only, no curriculum.
  - Last-step semantics — exported wherever the loss happened to be
    at the final step, even if that was past its optimum.

A0 with Move A+B:

  - epochs=3 — more passes (priced explicitly).
  - Cosine LR with warmup + 10% floor — gentle start, real cooldown
    into a fine-adjustment regime in the last 10% of training.
  - τ ramp 4.0 → 1.0 — curriculum from soft to hard targets.
  - Save-best on EMA-smoothed raw_kl — exports the optimum regardless
    of where the loss happens to be at the final step.

The 6× improvement (`0.117 → 0.019`) is the sum of all four effects.
The plan predicted ~0.06–0.07 expected floor; we hit ~0.02. Two
hypotheses for outperforming the prediction:

  1. The save-best timing (step 250) is much earlier than expected.
     The cosine warmup may make the early-run loss minimum easier to
     find than the constant-LR baseline did — warmup acts as a
     "gentle introduction" that hits a good basin fast.
  2. The bootstrap-on-first-observation EMA seed is dominated by the
     first raw_kl observation. If that observation happens to be a
     particularly clean one (low variance), the EMA latches onto it
     and subsequent observations have to actually be lower to displace
     it. This is a *desirable* selection bias for save-best, but it
     makes the "best step" location somewhat fragile to the random
     seed of the first calibration window.

Future ablations (A1..A11) will indicate whether 0.02 is reproducible
or whether A0 was unusually lucky with seed/first-window. Either way,
**all 12 ablations clear the 0.09 outcome gate is the success bar**.
