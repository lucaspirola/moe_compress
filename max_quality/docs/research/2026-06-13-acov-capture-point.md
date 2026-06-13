# Where should `A_cov` be captured? — paper- and code-grounded verdict

Read-only research. Code root `max_quality/src/moe_compress/`. Every code claim
cites `file:line` against the actual implementation; every paper claim is marked
**VERIFIED** (I retrieved and read the text) or **INFERRED** (my reasoning, not a
paper quote).

> **STATUS — round 2 (CONVERGED on substance).** The verdict on Q1/Q2/Q3 HELD;
> the reviewer cloned BOTH upstreams and confirmed every paper quote + AA-SVD's
> original-model anchor. ONE conceptual attribution was wrong and is now
> corrected throughout: **round 1 INFERRED that *EoRA's* whitening covariance is
> the original anchor — it is NOT.** Upstream EoRA whitens with the
> **sequentially-compressed SHIFT** `X'` (verified firsthand in `eora.py` @
> `6a42e2e`, see § Researcher round 2). The "original anchor `A=X`" property
> belongs to **AA-SVD objective ④**, not EoRA. This *strengthens* the
> recommendation: feeding Stage-4 EoRA a post-2.5 shift covariance moves it
> TOWARD upstream-EoRA fidelity. Body corrected in place; full reconciliation +
> both upstreams (read at their cited commits) in **§ Researcher round 2**.

---

## TL;DR verdict

**Two distinct covariances, two distinct papers — keep them straight:**
- **AA-SVD anchor** `A = X` = the ORIGINAL model's per-expert input Gram. Used by
  **Stage 3's AA-SVD factorization** as the anchor in objective ④
  `‖W·X − W'·X'‖²`. Capturing this on the original model is **paper-correct**
  (verified in AA-SVD upstream `1fa1b68`: `gram_xorig` from `x_orig`).
- **EoRA whitening cov** = the SHIFT `X'` = the input each layer receives in the
  **sequentially-compressed** model. Verified firsthand: upstream EoRA
  (`eora.py` @ `6a42e2e`) accumulates its `√Λ` whitening matrix on activations
  propagated through the compressed+compensated stack. **NOT the original.**

| Q | Verdict |
|---|---------|
| **Q1. Capture on original, or post-2.5?** | **BOTH — but for different consumers.** Keep the ORIGINAL `A` (it is AA-SVD's anchor, consumed by Stage-3 factorization — removing it breaks AA-SVD objective ④). AND add a post-2.5 SHIFT cov (it is what upstream EoRA actually whitens with, and Stage-4 EoRA currently lacks it). The user's "if we don't need original-A, why capture it" is answered: AA-SVD *does* need the original anchor. The real gap is a **missing** shift capture for EoRA, not a misplaced original. |
| **Q2. Where finally?** | **Original anchor `A` → stays at calibration (Stage 3 anchor). SHIFT cov → captured post-2.5, sequentially per-layer through the compressed model — exactly upstream-EoRA's regime, and exactly what Stage 3's live hook already collects (B/S).** So the shift cov is one near-zero-cost ride-along away. Feed it to Stage-4 EoRA as its whitening cov (or A/B it first). |
| **Q3. Paper-faithful ordering?** | **Order `calibrate(orig) → 2 → 2.5 → 3 → 4 → 5` is correct; the fidelity gap is in Stage 4's INPUT, not the ordering.** Upstream EoRA whitens with the sequential-compressed shift; our Stage-4 whitens with the frozen original calibration `A` → **non-faithful to upstream EoRA**. Stage 3 already holds the post-2.5 shift cov; routing it into Stage 4 makes EoRA MORE paper-faithful. No reorder, no removal of the original capture. |

**One-line bottom line:** The expensive 91 GB original-model `A` is **not wasted**
— it is AA-SVD's anchor (Stage 3). The actual defect is that **Stage-4 EoRA
whitens with that original anchor instead of the sequential-compressed shift that
upstream EoRA uses** — so the cheap fix (route Stage-3's already-collected post-2.5
shift cov into Stage 4) makes EoRA MORE paper-faithful, not less. Gated by one A/B.

---

## 1. What `A_cov` is and where it is captured (CODE, verified)

### 1.1 The capture point — calibration, on the loaded (original) model

- `A_cov` is a per-`(layer_idx, expert_idx, matrix_name)` input Gram
  `XᵀX ∈ ℝ^{d_in×d_in}`, X = the rows of tokens routed to that expert.
  Accumulated by `InputCovarianceAccumulator.update(layer, expert, matrix, x)`
  via `cov = flat_f32.transpose(0,1) @ flat_f32`
  (`utils/activation_hooks.py:1018-1029`). X is "tokens routed to this expert"
  — the per-expert input under whatever router fired
  (`activation_hooks.py:752` docstring; `update` keys the exact tuple
  `:1023`).
- Production capture is the **calibration-V2 sidecar** `covariance.pt`, written
  by the `--capture-input-covariance` flag → `vllm.calibration_input_cov`
  (`scripts/build_self_traces_calib_vllm.py:198` maps
  `"capture_input_covariance": "vllm.calibration_input_cov"`). The payload is
  keyed `(layer_rank, expert_idx, matrix_name) → Tensor[d,d]` over the model's
  FULL expert set, with a `top_k` routing field baked in
  (`utils/cached_calibration_signals.py:474-482` `CalibrationPayload`;
  `format_version=4`, `top_k`, `cov_acc` dict).
- In the production run this calibration ran on the **original uncompressed
  256-expert model** (prior doc confirmed "V2 input-cov cache HIT (20480 keys)"
  — 20480 = 40 layers × 256 experts × 2 matrices), so the sidecar's expert
  indices are the original 0..255 under the original router's top_k.

### 1.2 The consumers — Stage 3 and Stage 4 only LOAD it; neither re-captures

- **Stage 4 (EoRA)** loads `A_cov` and never re-derives it:
  `EoraInputsPlugin.load_eora_inputs` either short-circuits on the V2 cache hit
  (`ctx.has("A_cov")` → "Stage 4: V2 input-cov cache HIT … skipping
  `_stage2_input_covariance.pt` load", `stage4/plugins/eora_inputs.py:145-156`)
  or loads `_stage2_input_covariance.pt` from disk (`:162-217`). The plugin's
  `provides = ()` and its docstring states every input is "a precomputed
  on-disk artifact" (`eora_inputs.py:97-99`). **There is no forward pass in
  Stage 4.** The compensation kernel reads `A = A_cov.get(cov_key)`
  (`stage4/plugins/eora_compensation.py:765`) and computes
  `ΔW = W_orig − Ŵ`, whitens by `Q√Λ` of `A` (`_compute_eora_factors`
  `:235-386`).
- **Stage 3** likewise prefers the same calibration sidecar (V2 cache HIT log)
  and only falls back to `_stage2_input_covariance.pt`
  (`stage3/orchestrator.py:170-191`). But Stage 3 *also collects fresh B/S/C*
  on the live post-2.5 student (`covariance_collection.py:486-503`) — see §4.

### 1.3 The docstring that CONTRADICTS the code — flagged

`eora_compensation.py:79-85` (under the "Activation-cov reuse" header at `:77`)
claims:

> "This plugin reads the **post-merge** A-covariance from Stage 2's sidecar …
> the EoRA √Λ projection is computed on the **post-merge** A re-collected for
> Stage 4, so the activation-aware projection sees the true post-merge
> distribution."

**This is false in the production V2-cache path.** Nothing in Stage 4
re-collects A (§1.2); the V2 sidecar that wins the cache is the **original
256-expert calibration** Gram, not a "post-merge re-collected" one. Even the
fallback `_stage2_input_covariance.pt` is a pure index *remap* of the original
Grams (`stage2/shared_io.py:308-326` relabels keys and carries the SAME tensor
`val` verbatim; pruned experts dropped at `:320-323`) — never re-measured. So
"post-merge A re-collected for Stage 4" describes an intent the code does not
implement. **This docstring should be corrected regardless of the Q1/Q2
outcome** (it currently misleads any reader into thinking EoRA's whitening is
routing-current; it is not).

---

## 2. What the papers actually prescribe (VERIFIED quotes)

I retrieved the project's local truth-ref copies under
`audit/spec_compliance/01_papers/` (the audit's own retrieved source text) AND
cross-checked against the live arXiv HTML. Quotes below are **verbatim from the
retrieved text** unless marked INFERRED.

### 2.1 EoRA — arXiv:2410.21271 (VERIFIED)

The layer-wise compression loss (their Eq. 1) is
(`2410.21271/source.md:107-118`):

> "aiming to minimize the layer-wise output difference between the original
> weight W ∈ ℝ^{…} and the compressed weight Ŵ … `||W X − Ŵ X||` … where X
> is the input activation of layer ℓ"

**Decisive point: the SAME `X` multiplies both `W` and `Ŵ`.** EoRA's setup is
quantization/2:4-sparsity, where compression does NOT change which inputs a
layer receives — so there is a single `X` and a single covariance. The
eigenspace is built from that one `X` (`2410.21271/source.md:182-184`):

> "we perform the eigendecomposition on `X̃X̃ᵀ` where `X̃ ∈ ℝ^{…}` is the average
> of the input activations over the task-specific calibration set. The
> eigendecomposition `X̃X̃ᵀ = QΛQᵀ` is then used to derive the eigenspace
> projection matrix … `Q' = Q√Λ` … the projected error `ΔW' = ΔW Q'`."

The compensation target is `ΔW = W − Ŵ` projected and rank-r SVD'd
(`:200-269`, their Eq. 3-4). **The code matches this exactly**
(`_compute_eora_factors` `eora_compensation.py:244-386`).

- **Original vs compressed? — the PAPER is silent, the upstream CODE is not.**
  EoRA's text says "input activation of layer ℓ" and uses ONE `X` symbol for both
  `W` and `Ŵ` in Eq.1. **VERIFIED (paper):** the prose never distinguishes
  original-X from compressed-X. **VERIFIED (upstream code, round 2 — see §
  Researcher round 2):** the reference impl `NVlabs/EoRA` @ `6a42e2e`
  (`eora.py::llama_sequential_eigen`) accumulates the `√Λ` whitening covariance
  on activations that are propagated through the **sequentially-compressed +
  already-compensated** stack (weights overwritten at `eora.py:512`, re-forwarded
  `:518-519`, swapped into the next layer's input `:524`). **So upstream EoRA's
  whitening `X` is the SHIFT `X'`, NOT the original.** Only the target
  `delta = original_weight − compressed_weight` (`eora.py:478`) uses original
  weights. ⟹ **Round-1 correction:** I had attributed "original-anchor whitening"
  to EoRA; that was wrong. The original anchor belongs to **AA-SVD ④** (§2.2);
  EoRA whitens with the shift.
- **Sequential or one-shot?** **VERIFIED (paper + code):** EoRA is one-shot per
  compressed layer ("without any need for backpropagation", `:276-280`), but the
  reference impl runs it **layer-sequentially** down the stack
  (`for i in range(len(layers))`, `eora.py:431`), so each layer's whitening cov
  is conditioned on the compressed upstream — i.e. it IS shift-aware in the
  AA-SVD sense, despite the per-layer step being closed-form.

### 2.2 AA-SVD — arXiv:2604.02119, "Anchored and Adaptive SVD" (VERIFIED — decisive)

This is the paper that **explicitly answers the original-vs-compressed
question** the project's Stage 3 cites. Its four objectives
(`2604.02119/source.md:370-385`, Figure 2 left):

> "② Input-aware: `‖W X − W′X‖²_F` — matches outputs on **original inputs X**.
> ③ Shift-aware: `‖W X′ − W′X′‖²_F` — matches outputs on the **shifted inputs
> X′** seen after upstream compression. ④ **Anchored adaptive (ours):
> `‖W X − W′X′‖²_F` — anchors the target to the original output while
> conditioning on the shifted input**, combining an uncorrupted reference with
> distribution-shift awareness."

The closed form (Theorem 3.2, `:625-660`) and Algorithm 1 (`:761-766`):

> "**1: Set A = X, B = X′** {shift-aware: A=B=X′; input-aware: A=B=X}
> 2: Compute C = A Bᵀ and S = B Bᵀ
> 3: Factorize S = R Rᵀ …
> 4: Compute M = W C S⁻¹ R …"

And the definitions (`:453-455`, `:474-488`):

> "X ∈ ℝ^{n×l} collects intermediate activations at the input of f **from the
> original, uncompressed network** on calibration samples" … "X′ … produced by
> running the **partially compressed network** on the same calibration samples."

**VERIFIED conclusions from AA-SVD:**
1. The **auto/anchor covariance uses the ORIGINAL model's input activations**
   (`A = X`, X from the uncompressed network). Capturing `A` on the original
   model is **paper-correct, not a bug** — it is the *anchor* `W·X`.
2. The **shift covariance uses the COMPRESSED model's input** (`X'`, run through
   the partially-compressed network). The cross-cov `C = XX'ᵀ` and shift auto-cov
   `S = X'X'ᵀ` need the compressed-model forward.
3. **Ordering for the shift side is a HARD requirement, sequential/topological**
   (`:506-511`): "shift-aware compression must follow a valid topological
   order … compressing out of order yields features X′ inconsistent with any
   valid partial compression state." So the shift `X'` (and S, C) must be
   collected by running the compressed model in layer order — exactly what
   Stage 3's live hook does.
4. **Anchoring solely to X′ is dangerous** (`:511-514`): "when upstream
   compression has degraded representations, anchoring solely to X′ risks
   amplifying divergence from the original network's behavior." This is the
   theoretical warning AGAINST throwing away the original `A`.

### 2.3 Router-KD — arXiv:2603.02217 (VERIFIED)

> "persistent post-compression degradation largely stems from … router–expert
> mismatch when experts are changed but the router is [left unchanged]"
> (`2603.02217/source.md:19-24`); Router-KD "updates only the router
> parameters of the compressed model, distilling knowledge from the original
> model" (`:171-176`).

- **Does Router-KD assume covariances are refreshed?** **VERIFIED: it says
  NOTHING about covariances.** It trains routers only; it neither writes nor
  consumes any activation covariance. The pipeline confirms this — a grep of
  `router_kd/` for cov/Gram returns nothing functional (prior doc §A1, re-checked).
  **INFERRED:** Router-KD therefore neither requires nor forbids an `A` refresh;
  it is orthogonal. But it is the stage that *deliberately changes routing*, so
  it is the reason `A'` (shift) would differ from `A` (anchor).

### 2.4 Convention — where each method sits (VERIFIED, corrected round 2)

AA-SVD's related-work (`2604.02119:453-476`, VERIFIED) classifies the field:
**input-aware methods (SVD-LLM, ASVD) use the ORIGINAL network's fixed X**;
**shift-aware methods (GPTQ, SparseGPT, Dobi-SVD) use the sequentially-compressed
X'**. Round 1 mis-placed EoRA as "input-aware → original `X`". The upstream-code
read (§2.1, § Researcher round 2) corrects this: **upstream EoRA is
shift-aware** — its whitening cov is the sequentially-compressed input. So the
faithful-fidelity convention for EoRA's whitening is the **compressed shift**,
which is precisely what our Stage-4 does NOT use today (it uses the original
calibration anchor). The ORIGINAL anchor's correct home is **AA-SVD's `A=X`**
(Stage 3 factorization), not EoRA's whitening.

---

## 3. The crux: which covariance whitens EoRA's residual — and on which model?

This decides Q1/Q2. Round 1 conflated two distinct roles; round 2 (upstream code)
separates them cleanly:

- **EoRA's WHITENING covariance is the SHIFT `X'`** — the input each layer
  receives in the **sequentially-compressed** model. **VERIFIED (upstream code):**
  `eora.py::llama_sequential_eigen` builds `√Λ` from a hook over `inps` that have
  been propagated through the compressed+compensated upstream stack
  (`eora.py:512` overwrite, `:518-519` re-forward, `:524` swap). EoRA whitens
  `ΔW` by the eigenspace of `X'X'ᵀ`. ⟹ **the faithful EoRA whitening cov is the
  compressed-shift, NOT the original.** Our Stage-4 currently whitens with the
  **original calibration anchor** `A` → **non-faithful to upstream EoRA.**
- **EoRA's TARGET** `ΔW = W_orig − Ŵ` uses **original** weights (`eora.py:478`).
  This is the ONLY place "original" legitimately enters EoRA — the *target*, not
  the *whitening cov*. Our Stage-4 gets this right (`ΔW = W_orig − U·V`,
  `eora_compensation.py:438-442`).
- **The ANCHOR `A=X` (original input) belongs to AA-SVD, not EoRA.** AA-SVD
  objective ④ `‖W·X − W'·X'‖²` anchors the *output target* to the original input
  `X` (VERIFIED: AA-SVD Alg 1 `A=X`; upstream `gram_xorig` from `x_orig`,
  `compress.py:601,632`). Stage 3's AA-SVD factorization is the consumer that
  needs the original anchor `A`. So the 91 GB original-model `A` is **not wasted
  — it is Stage 3's anchor.** But Stage 4 EoRA is the WRONG consumer to feed it
  to as a whitening cov.

**Net:** the original `A` is correct *for AA-SVD/Stage 3*; the fidelity gap is
that *Stage 4 EoRA* reuses that same original `A` as its whitening cov instead of
the compressed-shift `X'` upstream EoRA uses. Routing Stage-3's already-collected
post-2.5 shift cov into Stage 4 moves EoRA TOWARD upstream fidelity.

**Does prune/merge break expert IDENTITY so the original-per-expert `A` no
longer corresponds to a survivor?** Partially yes (CODE):

- **Prune (REAP faithful):** survivors keep their original index identity; the
  original-expert `A` for a surviving expert `e` is the right *anchor* Gram for
  that same expert's weight `W_e` (which is unchanged in identity). The only
  staleness is that the survivor now *also* receives tokens reassigned from
  dropped experts (top-k renormalizes over survivors,
  `reap_prune.py:394-410`) — that is a **shift** effect (`X'≠X`), captured by
  AA-SVD's `X'`, not a corruption of the anchor `A`.
- **Merge:** a survivor slot is a centroid of a merge GROUP, but the stored `A`
  is the **single original expert's** Gram copied verbatim into that slot
  (`stage2/shared_io.py:324-326`), **not** averaged over the group. **INFERRED:**
  here the anchor `A` is genuinely mis-attributed — the merged weight `W_merged`
  is anchored to ONE constituent's input distribution, not the merged slot's
  true (group-union) distribution. This is a real correctness wart for the merge
  arms (REAM), independent of routing-shift, and is **not fixed by re-capturing
  post-2.5** either unless the post-2.5 capture is per-survivor-slot (which a
  Stage-3 ride-along WOULD give — see §5).

---

## 4. The internal inconsistency the code already exhibits (CODE, verified)

The pipeline is `calibrate(orig) → 2 → 2.5 → 3 → 4 → 5`
(`run_pipeline.py:63-71` STAGE_REGISTRY; Stage 2.5 inserted at `:289-296`).

- Stage 2.5's stated intent (`run_pipeline.py:272-273`): recalibrate routers "so
  **Stage 3 covariance collection sees already-adapted routing decisions**."
  → Stage 3's B/S/C ARE routing-current (collected on the live post-2.5 student,
  `covariance_collection.py:486-503`).
- Stage 4's EoRA whitening `A`, "collected" on the original model at calibration,
  is **NOT** routing-current — it is the frozen original-calibration sidecar
  (§1.2). Upstream EoRA would whiten with the post-2.5 **shift** here.

So Stage 3 **correctly** uses the original anchor `A` (for AA-SVD factorization)
AND a post-2.5 shift `X'` (its B/S/C). That half is AA-SVD-faithful. The defect
is in **Stage 4**: it whitens EoRA's residual with the SAME original anchor `A`,
whereas upstream EoRA whitens with the compressed **shift** `X'`. Stage 3 already
holds that shift cov (its B/S, collected on the live post-2.5 student) — Stage 4
just never receives it. **EoRA is run with the wrong whitening cov when the right
one (the shift) is one ride-along away.**

---

## 5. Recommendation + cheapest correct implementation

### 5.1 Do NOT remove the original-model `A` capture (answers Q1)

It is **AA-SVD's anchor** (Alg 1 `A=X` original; upstream `gram_xorig`,
`compress.py:601,632`), consumed by **Stage 3's factorization** in objective ④
`‖W·X − W'·X'‖²`. Removing it would break the anchor side of AA-SVD and leave
only shift-aware factorization, which AA-SVD explicitly warns "risks amplifying
divergence from the original network's behavior" (`2604.02119:511-514`, VERIFIED).
So the user's "why capture it at all" premise is answered: **the original `A` is
Stage 3's anchor — it is not a redundant copy and not wasted.** (Note: the
original `A` being *also* reused as Stage-4 EoRA's whitening cov is the separate
fidelity gap — §5.3 — but that argues for ADDING a shift cov for EoRA, not for
removing the anchor.)

### 5.2 ADD a cheap post-2.5 shift cov ride-along on the Stage-3 forward (answers Q2)

Stage 3 **already** walks every calibration token through the post-2.5
compressed model and accumulates per-`(layer,expert,matrix)` input Grams for
B/S (`covariance_collection.py:486-499`, `input_cb` `:639-661`). That live B/S
**is** the post-2.5 per-survivor-slot input Gram `X'X'ᵀ` — i.e. EoRA's missing
shift covariance, already computed, for free. The cheapest correct change:

- Persist Stage-3's post-2.5 per-expert input Gram as `A'` (it is B, modulo the
  gate/up alias and dtype) and hand it to Stage 4 alongside the anchor `A`.
- **Near-zero extra cost:** the forward is already running; this is a save +
  re-key, no new pass.
- Sequential/topological correctness is satisfied automatically — Stage 3
  collects layer-by-layer through the compressed model (AA-SVD's hard ordering
  requirement, §2.2 pt 3, is already met by the existing pass).

### 5.3 Feed Stage-4 EoRA the SHIFT cov (upstream-faithful), or A/B it (answers Q3)

Upstream EoRA whitens with the sequential-compressed shift `X'` (§3, verified).
Our Stage-4 whitens with the original anchor `A`. To restore fidelity, swap (or
A/B) the whitening cov:
- **A (current):** whiten ΔW by the original calibration `A`. **Non-faithful** to
  upstream EoRA (upstream uses the shift), and `A` may also be merge-mis-attributed
  (§3, merge case).
- **A\* (upstream-faithful, cheap):** whiten ΔW by the post-2.5 **shift** cov
  (Stage-3's already-collected per-expert input Gram on the live post-2.5
  student). This is what upstream EoRA actually does. Tests "does the faithful
  shift whitening help?"
- **AA (most faithful to the project's stack):** full Anchored-Adaptive EoRA —
  AA-SVD closed form with anchor `A=X` (original) and shift `B=X'` (post-2.5),
  i.e. `C=XX'ᵀ`, `S=X'X'ᵀ`. (`C` for gate_proj is *already collected* in Stage 3
  — `covariance_collection.py:496-503` — so even AA is mostly plumbing.) Note
  this exceeds vanilla upstream EoRA (which is shift-only); it is the
  AA-SVD-consistent generalization.

**The A/B that resolves it** (cheapest first):
- Arm A = current (original `A`). Arm A\* = post-2.5 shift cov from the Stage-3
  ride-along. Same Stage-3 forward, no extra train, no extra calibration. Metric:
  Stage-6-alt thermometer + the log-only EoRA residual drop
  (`stage4_eora.log_residuals=true`, `eora_compensation.py:860-868`).
- **Expectation (strengthened by round 2):** A\* should be **≥** A, because A\* is
  what upstream EoRA prescribes. If A\* ≥ A → promote to AA. If A\* ≤ A → the
  original `A`'s routing-invariance is empirically a feature here (AA-SVD's
  "noisy X′ from finite calibration" caveat, `2604.02119:514`) — keep current,
  but the result is then a *measured* deviation from upstream EoRA, documented.

### 5.4 Fix the misleading docstring (independent of the A/B)

`eora_compensation.py:79-85` ("the EoRA √Λ projection is computed on the
**post-merge** A re-collected for Stage 4") is factually wrong (§1.3): nothing in
Stage 4 re-collects, and the cov it whitens with is the **original-model
calibration anchor** (or, on the fallback path, a verbatim index-remap of it),
NOT a post-merge re-collection. It should be corrected to state that Stage-4 EoRA
currently whitens by the **original-calibration anchor `A`** and does NOT consume
a post-2.5 shift cov — which (per §3) is itself the fidelity gap vs upstream EoRA.
Documentation correctness fix regardless of whether 5.2/5.3 land.

---

## 6. Answers to the three questions (explicit)

1. **Do we need post-2.5 `A_cov`?** **YES — for Stage-4 EoRA, which is missing
   it.** Keep the original `A` too: it is **AA-SVD's anchor**, consumed by Stage 3
   (drop it and you break AA-SVD objective ④). The user's "if we don't need
   original-A, why capture it" is answered: AA-SVD *does* need the original anchor.
   What's missing is a **post-2.5 shift cov** for EoRA — which upstream EoRA
   actually whitens with, and which Stage 3 already collects for free.

2. **Where to capture finally?** Original anchor `A` → original-model calibration
   (unchanged; it is Stage 3's anchor). Shift cov → **post-2.5, sequentially
   per-layer through the compressed model** — exactly upstream EoRA's regime
   (`eora.py:431,512,518,524`) and exactly what Stage 3's live hook already
   produces (its per-expert B/S on the post-2.5 student,
   `covariance_collection.py:486-499`). Route that into Stage 4 (no new pass).

3. **Paper-faithful pipeline?** Order `calibrate(orig) → 2 → 2.5 → 3 → 4 → 5` is
   correct and needs no reorder. The fidelity gap is in **Stage 4's INPUT, not the
   ordering**: upstream EoRA whitens with the sequential-compressed **shift** `X'`,
   but our Stage 4 whitens with the frozen **original** anchor `A`. Feeding Stage 4
   the post-2.5 shift cov (already in Stage 3's hand) makes EoRA **more** faithful
   to upstream. Stage 3 itself is already AA-SVD-faithful (anchor `A` original,
   B/S/C post-2.5). Fix = add the shift cov to Stage 4; do NOT remove the original.

---

## 7. VERIFIED vs INFERRED ledger (per project rules)

**VERIFIED — papers (read the retrieved text + cite line):**
- EoRA Eq.1 `‖WX − ŴX‖`, X = "input activation of layer ℓ"
  (`2410.21271/source.md:107-118`); eigenspace `X̃X̃ᵀ=QΛQᵀ`, `Q'=Q√Λ`,
  `ΔW'=ΔWQ'` (`:182-193`); one-shot per layer, no backprop (`:276-280`).
  **The PROSE does not say whether the whitening `X` is original or compressed.**
- AA-SVD Objective ④ `‖WX − W'X'‖`, Alg 1 `A=X (original), B=X' (shifted)`,
  `C=ABᵀ, S=BBᵀ` (`2604.02119/source.md:370-385, 625-660, 761-766`);
  `X` from "the original, uncompressed network", `X'` from "the partially
  compressed network" (`:453-455, 474-488`); shift ordering is a hard
  topological requirement (`:506-511`); anchoring solely to X′ "risks amplifying
  divergence" (`:511-514`).
- Router-KD trains routers only, distilling original model; says nothing about
  covariances (`2603.02217/source.md:19-28, 171-176`).

**VERIFIED — upstream code (cloned + read at the cited commits, round 2):**
- **NVlabs/EoRA @ `6a42e2e`** (`eora.py::llama_sequential_eigen`): whitening cov
  `subset_eigen_scaling_diag_matrix` accumulated by a fwd hook over `inps`
  (`eora.py:447-466`); `inps` propagated through the compressed+compensated stack
  (overwrite `:512`, re-forward `:518-519`, swap `:524`) ⟹ **whitening cov is the
  SHIFT `X'`**. Target `delta = original_weight − compressed_weight` (`:478`) ⟹
  only the *target* uses original weights. **Settles round 1's open gap: EoRA
  whitens with the shift, not the original anchor.**
- **atulkumarin/AA-SVD @ `1fa1b68`** (`aa_svd/compression/compress.py:589-651`):
  dual-forward — `gram_xorig` from `x_orig` (original-model input = ANCHOR),
  `gram_x` from `x` (compressed input = SHIFT), `gram_cross` from `(x_orig, x)`.
  Confirms `A=X` anchor is collected on the **original** model. **The
  original-anchor property belongs to AA-SVD, as re-framed.**

**VERIFIED — repo code (re-opened every cite):**
- Capture `activation_hooks.py:1018-1029`; Stage 4 load-only (no forward)
  `eora_inputs.py:145-217`; EoRA kernel + `A=A_cov.get(...)`
  `eora_compensation.py:235-386, 765`; ΔW=W_orig−U·V `:438-442`; Stage 3 live
  B/S/C `covariance_collection.py:486-503`; pipeline order
  `run_pipeline.py:63-71, 272-296`; Stage 2 cov index-remap
  `stage2/shared_io.py:308-326`; **docstring contradiction
  `eora_compensation.py:79-85`** (cite corrected from round-1's 77-85).

**INFERRED (my reasoning, NOT a quote):**
- That the merge case mis-attributes the anchor `A` (one constituent's Gram, not
  the group union) — derived from the index-remap code, not stated in any paper.
- The A/B prediction directions in §5.3 (mechanism-based, not measured) — though
  now *strengthened* by the upstream-EoRA finding (A\* = the upstream regime).
- The "near-zero cost" of the Stage-3 ride-along (architectural inference from
  the existing hook; not benchmarked).

**No remaining un-read upstreams.** Both reference impls (EoRA, AA-SVD) read
firsthand at their cited commits; round-1's flagged gap is closed.

---

## Reviewer round 1 (adversarial verification)

Read-only adversarial review. I (a) re-grepped every paper quote against the
local retrieved sources, (b) **cloned and read** the actual upstream `.py`
(NVlabs/EoRA and atulkumarin/AA-SVD `@1fa1b68`) — the round-2 gap the researcher
flagged in §7 — and (c) re-opened every cited `file:line` in the repo under
review. **Verdict: CHANGES REQUESTED** — the verdict's *recommendations* (keep
the original `A`; the gap is the missing shift; fix the docstring; merge wart is
real) are sound and survive, but the **EoRA-specific anchor claim is partly
wrong against the actual EoRA upstream code**, which the researcher never read.

### Paper-quote verification — ALL VERIFIED (no hallucinations)

Every quoted phrase appears verbatim in the project's retrieved sources:

- **AA-SVD (2604.02119)** — `source.md`: Objective ④ `‖WX − W′X′‖²_F`
  "anchors the target to the original output while conditioning on the shifted
  input" (L382-385); Alg 1 "Set A = X, B = X′ {shift-aware: A=B=X′; input-aware:
  A=B=X}", "C = AB⊤ and S = BB⊤" (L761-762); X "from the original, uncompressed
  network" (L453); X′ "running the partially compressed network" (L488);
  topological hard requirement (L508); "anchoring solely to X′ risks amplifying
  divergence" (L511-512). **All confirmed.**
- **EoRA (2410.21271)** — Eq.1 `‖W − Ŵ‖`·X, "input activation of layer ℓ"
  (L118); eigendecomp `X̃X̃ᵀ=QΛQᵀ`, "projection matrix Q′ = Q√Λ", "ΔW′ = ΔW Q′"
  (L181-194); "without any need for backpropagation … few minutes" (L72,279).
  **All confirmed.**
- **Router-KD (2603.02217)** — "router–expert mismatch when experts are changed
  but the router is left [untouched]" (L19-24); "Router KD updates only the
  router parameters" (L173). Grep of the source for `covariance`/`Gram`
  returns **zero functional hits** (only an OCR collision on "Program"). The
  "says nothing about covariances" claim is **confirmed.**

No Critical finding on the quotes — the feedback_webfetch_quotes_not_verified
risk does not materialize here.

### Upstream code — AA-SVD CONFIRMS the anchor; EoRA PARTLY REFUTES it

**AA-SVD (`atulkumarin/AA-SVD@1fa1b68`) — confirms the researcher.**
`aa_svd/compression/compress.py::collect_gram_matrix_parallel` (L574-655) runs a
dual forward: one hook on the **original** layer adapter, one on the **clone
(compressed)** adapter, and builds three Grams — returned (L655) and unpacked by
the caller (L786) as `xhatTxhat, xTxhat, xTx`. The caller feeds
`compress_module_obj4(target, xTx, xTxhat, xhatTxhat, …)` (L808). In
`decompose.py::_compress_module_obj34` (L161-212), `xTx` is the **anchor**
(original-model auto-cov, `A=X`), `xhatTxhat` the **shift** (`S=X′X′`),
`xTxhat` the **cross** (`C`). **So AA-SVD's anchor `A=X` IS collected on the
original uncompressed model — the verdict's central claim is upstream-confirmed.**
(Note a local naming swap inside `collect_gram_matrix_parallel`: the var
`gram_x` is fed by the *clone* path and returned as `xhatTxhat`; semantics are
nonetheless unambiguous from the caller. Does not affect the conclusion.)

**EoRA (`NVlabs/EoRA`) — REFUTES "EoRA's anchor is the original X". (High.)**
The researcher explicitly did NOT clone NVlabs/EoRA (§7: "NOT independently
verifiable here") and *inferred* that EoRA's whitening `X` is the original
anchor. The actual upstream code says otherwise:
- In `eora.py::llama_sequential_eigen` (L385-530), the eigenspace covariance is
  accumulated by a forward hook (`tmpp`, L447-456: `inp.T @ inp`) **while
  walking layers sequentially**. After compensating layer *i*, L512 overwrites
  `subset[name].weight.data = comp_weight` (compressed + low-rank), then L518-519
  recompute `outs` through the now-compensated layer and L526 swaps `inps,outs`.
- **Therefore EoRA's `X̃` for layer i+1 is propagated through the
  compressed-AND-compensated upstream layers — a *shift-aware* X′, NOT the pure
  original anchor.** Only the *target* `ΔW = original_weight − compressed_weight`
  (L478) uses the original weights; the *whitening covariance* is the
  sequentially-compressed model's input.
- The main block confirms a fresh original model is reloaded before eigen
  (`del model; model = get_llama(...)`, L1451-1453), but that only sets the
  *layer-0* input to original; downstream layers are progressively compressed
  in-loop by L512.

**Consequence for the verdict's logic:** EoRA-the-reference-impl whitens with a
*shift* covariance (collected sequentially on the compressing model), which is
the **opposite** of the doc's claim that "EoRA's `A` = the original anchor, and
the shift is the missing piece." For *EoRA specifically*, upstream uses the
shift; it is AA-SVD's *objective ④* (not EoRA) that anchors `W·X` to the
original while conditioning on `X′`. The doc **conflates** EoRA's single-cov
whitening (which upstream makes routing-/compression-current) with AA-SVD's
anchor term. This does not overturn "keep the original `A`" (AA-SVD's anchor
genuinely wants original X), but it **inverts the doc's framing of which method
wants which covariance**, and it means the §5.3 "A\* (shift-only) tests does
routing-current whitening help" arm is actually the EoRA-upstream-faithful
configuration — not the deviation the doc presents it as. The doc must be
corrected to say: *the project's EoRA port whitens with the original-calibration
cov, whereas upstream EoRA whitens with the sequentially-compressed cov; making
EoRA routing-current (arm A\*) moves TOWARD upstream-EoRA fidelity, not away
from it.*

### Code claims in the repo under review — VERIFIED

- Stage 4 loads `A` and never re-collects it: `eora_inputs.py:145-156` cache
  short-circuit + `:217` `torch.load(...)`; `provides=()` (`:99`);
  `A = A_cov.get(cov_key)` (`eora_compensation.py:765`). **Confirmed — no forward
  pass in Stage 4.**
- False docstring `eora_compensation.py:79-85` ("post-merge A re-collected for
  Stage 4"): the code re-collects nothing (above), so the docstring is
  **factually wrong — confirmed (High, doc-correctness).** (Researcher cited
  `77-85`; the false sentence is precisely `79-85`. Minor cite slip.)
- REAM merge anchor mis-attribution: `shared_io.py:308-326` is a pure index
  relabel — `new_cov[new_key] = val` carries the **same tensor** verbatim
  (`:325`), pruned experts dropped (`:320-323`), **never averaged over the merge
  group.** **Confirmed.**
- Stage 3 collects S (shift) + C (cross) on the live student:
  `covariance_collection.py:486-503`. **Confirmed** — but the file is at
  `stage3/plugins/covariance_collection.py`; the doc drops the `plugins/`
  segment throughout (Nitpick).
- AA-SVD `A` reserved, unused in the Stage-3 rank step:
  `aa_svd_factor.py:230` `del A`. **Confirmed.**

### Findings by category

- **Critical:** none. (No fabricated quotes; the load-bearing AA-SVD anchor
  claim is upstream-confirmed.)
- **High — EoRA anchor claim contradicts EoRA upstream.** The doc's §2.1/§3/§5
  framing that "EoRA's `X`/`A` is the original anchor" is **not** what
  `NVlabs/EoRA::llama_sequential_eigen` does (it whitens with a sequentially-
  compressed/compensated `X′`, L512+L518-526). Fix the framing: the *anchor =
  original X* result is AA-SVD's (Alg 1 `A=X`), confirmed; EoRA-upstream itself
  uses the shift. This re-labels arm A\* as the EoRA-faithful arm.
- **High — false docstring** `eora_compensation.py:79-85` (confirmed; the doc
  already flags it — keep, fix the cite to `79-85`).
- **Medium — "EoRA needs the shift" is stated as an EoRA gap but is really an
  AA-SVD-objective-④ gap.** Per upstream, EoRA *already* uses a shift cov; the
  project's port is the outlier in using original-calib `A`. Re-attribute the
  argument to AA-SVD ④ (which the project's Stage 3 already implements with
  S/C); the EoRA Stage-4 step using only `A` is a *project choice*, and whether
  it should consume the shift is exactly the A\* ablation — correctly proposed,
  wrongly motivated.
- **Low — cite slips:** `covariance_collection.py` missing `plugins/` path;
  docstring range `77-85`→`79-85`.
- **Nitpick:** the doc's "91 GB anchor is vindicated/wasted" rhetoric is fine,
  but should note that *upstream EoRA would have captured a (cheaper, sequential)
  shift cov instead of the 91 GB original* — so "the original is the
  paper-mandated anchor" is true for AA-SVD but NOT a vindication of EoRA's cost
  profile.

### What the researcher must fix/re-verify

1. **Re-frame the EoRA claim against the now-cloned NVlabs/EoRA code.** State
   explicitly that upstream EoRA whitens with the sequentially-compressed `X′`
   (`eora.py:512,518-526`), so "original anchor" is AA-SVD's property, not
   EoRA's. The conflation in §2.1/§3/§5 must be corrected.
2. **Re-attribute the "missing shift" argument** from EoRA to AA-SVD objective
   ④; clarify that the project's Stage-4 EoRA-on-original-`A` is a deliberate
   port choice, and arm A\* (shift cov) moves toward upstream-EoRA fidelity.
3. Keep (verified): the anchor `A=X` original (AA-SVD upstream-confirmed), the
   false docstring, the REAM merge wart, the cheap Stage-3 ride-along for the
   shift. Fix the two cite slips.

**Bottom line:** quotes all real; AA-SVD upstream confirms the original-anchor
claim; **EoRA upstream contradicts the doc's EoRA-specific framing** (it uses a
shift cov, not the original). Recommendations stand; the EoRA attribution must
be corrected before approval.

---

## Researcher round 2 (reconciliation — CONVERGED)

I read **both** upstreams firsthand at the exact commits the repo docstrings cite
(per project rule: compare against actual upstream `.py`, not the paper). The
reviewer is correct on the one substantive point; I have corrected the EoRA
attribution throughout the body above. Reconciliation:

### What I verified firsthand this round

- **NVlabs/EoRA `@6a42e2edcc7559422d14ccf79b0105b2d8a78c76`** (HEAD matched the
  docstring commit), `eora.py::llama_sequential_eigen` (L385-524):
  - Whitening cov `subset_eigen_scaling_diag_matrix[name]` is built by a
    `register_forward_hook` (`tmpp`, L447-466) computing `inp.transpose@inp` over
    `inps` — the **inputs to the current layer**.
  - Those `inps` are the **sequentially-compressed** activations: after solving a
    layer, `comp_weight = compressed_weight + B@A` is written back
    (`subset[name].weight.data = comp_weight`, **L512**), the layer is re-run
    (**L518-519**), and `inps, outs = outs, inps` (**L524**) feeds the compensated
    output as the next layer's input. ⟹ **EoRA's whitening cov = SHIFT `X'`.**
  - The target `delta = original_weight − compressed_weight` (**L478**) is the
    only place original weights enter. ⟹ original is the *target*, not the
    whitening cov.
- **atulkumarin/AA-SVD `@1fa1b686cd9b13a77607a676564e37d438a176c8`**,
  `aa_svd/compression/compress.py` (L589-651): dual-forward captures
  `gram_xorig` from `x_orig` (original-model input → **anchor `A`**), `gram_x`
  from `x` (compressed input → **shift**), `gram_cross` from `(x_orig, x)`.
  ⟹ AA-SVD's `A=X` anchor is collected on the **original** model. **This is where
  the original-anchor property lives — AA-SVD, not EoRA.**

### The correction (what was wrong in round 1)

Round 1 wrote "EoRA's whitening covariance X is the original anchor" as an
**INFERRED** claim (the EoRA paper prose is genuinely silent, and I had not read
their code). That inference was **wrong**. Upstream EoRA whitens with the
sequential-compressed **shift**. The "original anchor `A=X`" is **AA-SVD
objective ④**'s property. Corrected in: TL;DR table, §2.1, §2.4, §3, §4, §5.1,
§5.3, §6, §7.

### Why this STRENGTHENS the verdict (does not weaken it)

The recommendation was "add a post-2.5 shift cov and feed/A-B it into Stage 4."
Round 1 justified arm A\* as "tests whether routing-current whitening helps."
Round 2 sharpens it: **arm A\* (post-2.5 shift cov) is precisely the regime
upstream EoRA uses.** So A\* is not a speculative variant — it is the
**upstream-faithful** EoRA. The current Stage-4 (original `A` whitening) is the
**deviation**. Hence:
- The expected A/B direction is now **A\* ≥ A** with a *paper-fidelity* rationale,
  not just a mechanism guess.
- "Keep the original `A`" survives — but its justification narrows from "EoRA's
  anchor" (wrong) to "**AA-SVD's** anchor, consumed by Stage 3" (right).
- The internal inconsistency is sharper: Stage 3 is fully AA-SVD-faithful (anchor
  original + shift post-2.5), while Stage 4 EoRA whitens with the original anchor
  instead of the shift upstream EoRA mandates — a one-ride-along fix.

### Corrected verdict on the three questions

1. **Do we need post-2.5 `A_cov`?** YES, for Stage-4 EoRA (its whitening cov
   should be the post-2.5 shift, matching upstream EoRA; today it wrongly reuses
   the original anchor). Keep the original `A` too — it is AA-SVD's anchor for
   Stage 3, not redundant.
2. **Where to capture finally?** Original anchor `A` → original-model calibration
   (Stage 3's anchor, unchanged). Shift cov → post-2.5, sequential per-layer
   through the compressed model — already collected by Stage 3's live hook; route
   it into Stage 4. No new forward pass.
3. **Paper-faithful pipeline?** Order is correct; no reorder. The fix is Stage-4's
   *input*: whiten EoRA's ΔW with the post-2.5 **shift** cov (upstream-EoRA
   faithful), optionally the full Anchored-Adaptive form (anchor + shift,
   AA-SVD-consistent). Plus two doc fixes: correct the false
   `eora_compensation.py:79-85` "re-collected" claim, and (separately) the REAM
   merge-slot anchor mis-attribution.

### Precise faithful-pipeline recommendation

```
calibrate(original)  → captures ANCHOR A = XᵀX  (original model, original routing)
  → Stage 2 (prune/merge)
  → Stage 2.5 (Router-KD heal)            [routing now post-2.5]
  → Stage 3 (AA-SVD):  factorize using ANCHOR A (original)  ← unchanged, faithful
              + collect SHIFT cov X'X'ᵀ on the live post-2.5 student  ← ALREADY HAPPENS (B/S)
              → PERSIST that shift cov as an EoRA input          ← NEW (cheap ride-along)
  → Stage 4 (EoRA):    ΔW = W_orig − Ŵ           (original target — unchanged)
              whiten ΔW by the SHIFT cov X'       ← CHANGE from original A (upstream-faithful)
              [or AA form: anchor A + shift X' per AA-SVD ④]
  → Stage 5 (Router-KD final)
```
Gate the whitening-cov change on arm A (original `A`) vs arm A\* (post-2.5 shift),
metric = Stage-6-alt thermometer + log-only EoRA residual drop. Expectation
A\* ≥ A (A\* is the upstream-EoRA regime). No extra train, no extra calibration
pass, no reorder, and the original anchor capture stays.
