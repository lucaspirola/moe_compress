# Triage Sheet — Spec Compliance Audit

**How to use:** For each card write your disposition in the `DECISION` line.
Choices: `accept-fix` · `accept-but-different-fix` · `reject` · `defer`

After triaging, run the fix-up session in `INSTRUCTIONS_AUDIT_MAX_QUALITY.txt §2`.

---

## Stage 1 — SE Detection (§4)

> **Dependency note — three findings share one root:**
> F-0006 (detection criterion), F-0037 (MA-layer pass skipped), and F-0016
> (late-layer false-positive risk) are three facets of the same underlying
> architectural choice: the spec uses a per-layer z-score instead of the paper's
> two-pass (P99.5 + L-filter) criterion.  Decisions on these three should be
> consistent:
> - Accept F-0006 ↔ accept F-0037 (they describe the same deviation from
>   different angles; splitting them is not meaningful).
> - F-0016 is a *consequence* of the missing L-filter (F-0037). If you keep
>   F-0037's deviation, F-0016 becomes an active risk to manage, not just a
>   theoretical concern.
> - F-0015 (the 5% global cap) is partly a mitigation for F-0016. Its value as
>   a guardrail depends on whether F-0016's risk is accepted as real.

---

### F-paper-2507.23279-0006 · HIGH · uncovered_deviation
**§4, spec line 142**

**Problem**
The paper's SE detection criterion is a three-way AND: activation above global
P99.5 percentile, above 1/10 · a_max, AND the layer must be in the MA-formation
set L. The spec replaces this entire criterion with a per-layer z-score (z > 2.5).
§12 D1 sanctions this deviation, but D1's robustness argument references Table 9's
(P95–P99.9) × (0.07–0.10) band — a different parameter space than z-scores.
The sanction exists but has no empirical backing for the specific z-score choice.

**Spec currently does**
`a_{l,e} > mean_l + 2.5 · std_l`

**Paper requires (2507.23279 Algorithm 1)**
`a_{l,e} > P99.5(D) AND > 0.1·a_max AND l ∈ L`

**Proposed fix**
D1 already sanctions this. Tag D1 as `empirical_pending` and add a note that
the z-score criterion has not been empirically compared against the paper's
P99.5 criterion on Qwen3-30B-A3B.

> DECISION: reject
> NOTES: our spec should completely match the papers for all the findings in this group

---

### F-paper-2507.23279-0037 · HIGH · uncovered_deviation
**§4, spec line 730**
*Root of the same deviation as F-0006 above.*

**Problem**
Paper Algorithm 1 Stage 1 is a dedicated calibration pass that builds the set L
of MA-formation layers by detecting the MA pattern in each layer's hidden states.
The spec skips this pass entirely — L is never constructed and all layers are
treated as candidates. D1 sanctions this with the argument that z > 2.5 is
conservative enough in practice (true SEs typically hit z > 10), but the L-filter
is described in the paper as fundamental, not optional.

**Spec currently does**
Skips Algorithm 1 Stage 1; no L set; all layers are candidates for SE flagging.

**Paper requires (2507.23279 Algorithm 1 Stage 1)**
Build `L = { l : MA pattern detected in H_l(x) }`; only flag SEs where `l ∈ L`.

**Proposed fix**
Same action as F-0006: D1 already sanctions this. Tag D1 as `empirical_pending`
noting that without L, late-layer false positives are theoretically possible
(see F-0016).

> DECISION: reject
> NOTES: our spec should completely match the papers for all the findings in this group

---

### F-paper-2507.23279-0016 · HIGH · uncovered_deviation
**§4**
*Downstream consequence of the missing L-filter (F-0037).*

**Problem**
Because all layers are candidates, late-layer outlier experts (Table 7 shows
L47E48 and L47E100 on Qwen3) could satisfy z > 2.5 and be incorrectly added to
the SE blacklist. The paper explicitly notes these late-layer outliers "do not hold
the same level of significance as SEs." D1's defence is that z > 10 in practice —
but Table 7 outliers may approach that threshold. Sanctioned by D1 + D2, but
without empirical verification on the target model.

**Spec currently does**
No L-filter; per-layer cap of 4 and global 5% cap (D2) bound worst-case damage.

**Paper requires**
Late-layer experts excluded via the L-filter before SE flagging.

**Proposed fix**
Already covered by D1 + D2. Tag both rows `empirical_pending` and note that a
verification run should compare the spec's SE detector output against the paper's
canonical SE set (L1E68, L2E92, L3E82 for Qwen3-30B-A3B) to confirm zero
false positives on the target model.

> DECISION: reject
> NOTES: our spec should completely match the papers for all the findings in this group

---

### F-paper-2507.23279-0015 · MEDIUM · paraphrase_drift
**§4, spec line 142**
*Partially mitigates F-0016 (the 5% cap limits false-positive damage).*

**Problem**
The paper reports SEs empirically account for < 0.5% of all experts. The spec sets
a global cap of 5% — 10× looser. D2 frames this as a safety guardrail (the cap
bounds worst-case damage, not an expected count). The 5% figure is not wrong, but
if read as an approximation of the paper's empirical bound it is misleading.

**Spec currently does**
Global SE cap: ≤ 5% of experts. Per-layer cap: ≤ 4.

**Paper states**
SEs account for < 0.5% of experts empirically.

**Proposed fix**
Already sanctioned by D2. Add a clarifying note to D2 stating: "5% is a
worst-case safety cap, not an approximation of the paper's 0.5% empirical count."
No algorithm change needed.

> DECISION: reject
> NOTES: our spec should completely match the papers for all the findings in this group

---

### F-paper-2507.23279-0012 · MEDIUM · silent_omission
**§4**

**Problem**
Paper Table 2 names the canonical SE set for Qwen3-30B-A3B (Layer 1 Expert 68,
Layer 2 Expert 92, Layer 3 Expert 82, among others). The spec contains no mention
of this ground truth. Given the D1 z-score deviation, the spec's detector output
may differ from the paper's — but there is no regression anchor to catch that.

**Spec currently does**
No canonical SE set cited; no regression check.

**Paper provides**
Named SEs for the target model.

**Proposed fix**
Add a note in §4 Stage 1 (or §12) naming the paper's canonical SE set for
Qwen3-30B-A3B as the expected verification output. Documentation addition only,
no algorithm change.

> DECISION: reject
> NOTES: this tool is indentend to be run on any model, so our spec should completely match the papers for all the findings in this group

---

### F-paper-2507.23279-0038 · LOW · paraphrase_drift
**§4, spec line 135**

**Problem**
Minor notation mismatch. Paper writes the SE activation signal as
`max_{x∈D} |h_{l,e}(x) · W^{l,e}_{down_proj}|`, making explicit that the signal
is computed after the down-projection weight multiplication. Spec writes
`max(|down_proj_output|)`, which is equivalent but obscures whether this is
pre- or post-weight multiplication.

**Spec currently does**
`max(|down_proj_output|)` — correct but ambiguous.

**Paper writes**
`max_{x∈D} |h_{l,e}(x) · W^{l,e}_{down_proj}|`

**Proposed fix**
Update spec notation at line 135 to match the paper's explicit form. One-line
text change.

> DECISION: accept-fix
> NOTES: No comments.

---

## Stage 2 — Calibration Data & REAM Merge (§5)

> **Dependency note — two findings share one §12 row (D11):**
> F-ch12-missing-0007 (REAM calibration size) and F-ch12-missing-0011 (REAP
> calibration source) are independent deviations against different papers, but
> both are fixed by extending the same §12 D11 row. If you accept either one,
> edit D11 once; if you accept both, it is still one edit.

---

### F-ch12-missing-0007 · MEDIUM · silent_omission
**§5, spec line 274**

**Problem**
REAM (paper 2604.04356) uses 3072 sequences × 512 tokens for calibration. The
spec uses 1024 sequences. §12 D11 sanctions the calibration *source* change
(c4 → Nemotron) but says nothing about calibration *size*. The size is a separate
unsanctioned deviation against the same REAM paper.

**Spec currently does**
Calibration: 1024 sequences.

**Paper requires (2604.04356)**
3072 sequences × 512 tokens.

**Proposed fix**
Extend §12 D11 to add: "REAM calibration size: spec uses 1024 sequences vs
paper 3072 × 512. Justification: reduces Phase A wall-time by ~3×; empirical
PPL impact is empirical_pending."

> DECISION: reject
> NOTES: Bump the calibration sequences to 4000, using our calibration set from nemotron with the already defined percentages

---

### F-ch12-missing-0011 · LOW · silent_omission
**§5, spec line 89**
*Extends the same D11 row as F-ch12-missing-0007 above.*

**Problem**
REAP (paper 2510.13999) uses c4 + evol-codealpaca as calibration source. The spec
uses Nemotron-Cascade-2-SFT-Data. D11 sanctions this change but cites only
2603.02217 (Router KD). The same deviation applies against REAP (2510.13999) and
is not covered.

**Spec currently does**
D11 covers calibration-source change for paper 2603.02217 only.

**Also applies against**
Paper 2510.13999 (REAP).

**Proposed fix**
Add `2510.13999` to D11's "Paper" column. One-word edit in the §12 table.

> DECISION: accept-fix
> NOTES: We should stick to our calibration data from Nemotron

---

### F-cross-h200-0001 · LOW · unsupported_justification
**§5, spec line 274**

**Problem**
The spec stores Stage 2 covariance matrices in FP16 (paper 2604.01609 certifies
FP32 only). The paper crossref records `sanction_justification = h200`, pointing
to backing claim H200-0008. But H200-0008 audits as *unverifiable* — H200 truth
refs cover hardware floating-point support in general, not the specific trade-off
of FP16 vs FP32 covariance storage. The sanction is hanging in the wrong regime;
this is an algorithmic precision choice that belongs in §12.

**Spec currently does**
FP16 covariance storage, tagged as `h200`-sanctioned (claim not verifiable).

**Correct sanction regime**
§12 with an explicit numerical-precision row.

**Proposed fix**
Add a §12 D-row:

> **D-cov-fp16** | Paper 2604.01609 Table 5 | Spec stores covariance in FP16
> to halve Stage 2 peak VRAM. Justification: covariance is used only to compute
> eigenvectors; orientation is preserved at FP16 for well-conditioned matrices.
> Empirical PPL delta FP16 vs FP32: pending.

Remove or correct the `h200` sanction tag from the paper crossref entry.

> DECISION: reject
> NOTES: we should stick to the same precision as per paper

---

### F-ch12-0001 · MEDIUM · misdescribed_ch12
**§12 D5b, spec line 736**

**Problem**
D5b "Paper Says" currently states "Eq. 6: frequency-weighted average (no
permutation)." This is factually wrong. Paper 2604.04356, immediately after Eq. 6,
states: *"where W_i are expert i's weight matrices with neuron permutation
alignment (Ainsworth et al., 2023) applied w.r.t. the dominant (centroid) expert."*
The paper prescribes permutation. The spec's Hungarian alignment is therefore
compliant, not a deviation. The real (minor) deviation is the specific cost matrix
C = C_wt + C_act, which the paper leaves unspecified.

**Spec currently does**
D5b frames the deviation as "permutation vs no permutation" — the wrong axis.

**Paper actually says (2604.04356 Eq. 6)**
Permutation alignment is prescribed; specific cost matrix C is unspecified.

**Proposed fix**
Rewrite D5b "Paper Says" column to:

> *2604.04356 Eq. 6: frequency-weighted average with neuron permutation alignment
> (Ainsworth et al., 2023) w.r.t. the centroid expert; cost matrix C unspecified.*

Rewrite "Deviation" column to:

> *Spec uses cost matrix C = C_wt + C_act. Paper prescribes permutation but leaves
> cost form open. Ablation pending (empirical_pending).*

> DECISION: accept-fix
> NOTES: spec must match paper — rewrite D5b "Paper Says" to correctly state that permutation alignment is prescribed, and reframe the deviation as the unspecified cost matrix C = C_wt + C_act

---

### F-resume-0001 · LOW · silent_omission
**§5, spec line 284**

**Problem**
Stage 2's "atomic checkpointing" phrase at line 284 does not explain what "atomic"
means or point to where it is defined. §11 (lines 695, 710) contains the full
definition (`.tmp + os.replace`, payload-before-manifest ordering). A reader of
§5 alone has no way to know how the atomicity guarantee is implemented.

**Spec currently does**
Uses "atomic checkpointing" without definition or forward-reference.

**§11 provides**
Full atomic-write idiom.

**Proposed fix**
Add a parenthetical at line 284: *(see §11 for the `.tmp + os.replace` idiom and
`.pt`-before-`.json` ordering invariant).*

> DECISION: accept-fix
> NOTES:

---

## Stage 2.5 — Router KD (§5)

---

### F-ch12-missing-0008 · LOW · silent_omission
**§5, spec line 295**

**Problem**
REAM (paper 2604.04356) is designed as a static protocol and explicitly evaluates
"without any fine-tuning after compression." The spec inserts Stage 2.5 (Router KD
from paper 2603.02217) immediately after the REAM merge. Neither paper anticipates
or accounts for this combination. No §12 row covers the protocol blend.

**Spec currently does**
REAM merge → Router KD recalibration in sequence.

**Papers prescribe**
REAM: merge only, no subsequent training. Router KD: a separate standalone step,
not designed as a post-REAM patch.

**Proposed fix**
Add a §12 D-row:

> **D-protocol-blend** | Papers 2604.04356 (REAM) + 2603.02217 (Router KD) |
> Spec applies Router KD after REAM merge. Justification: Router KD restores
> routing accuracy degraded by weight averaging; REAM's static evaluation
> does not cover post-merge routing drift. Combined protocol not ablated against
> REAM-static-only baseline: empirical_pending.

> DECISION: accept-fix
> NOTES:

---

## Stage 3 — SVD Compression (§6)

### Phase A — AA-SVD

---

### F-ch12-missing-0001 · CRITICAL · silent_omission
**§6, spec line 332**

**Problem**
AA-SVD's central contribution (paper 2604.02119) is an iterative *block-level*
refinement loop: given a block of layers, alternately update the weight matrices
W_i and the scale parameters θ_i until convergence (Algorithm 2). The spec
implements only the simpler per-layer Path 1 (anchored-adaptive objective) and
never enters this loop. Additionally, §10 forbids modifying RMSNorm, which directly
contradicts the paper's requirement that θ_i (the RMSNorm scale) be tuned during
refinement. The paper's own ablation (§6) shows block refinement dominates quality;
it is the single largest uncovered deviation in the spec.

**Spec currently does**
Per-layer anchored SVD (Path 1); no block refinement; RMSNorm in Protected
Components list.

**Paper requires (2604.02119 Algorithm 2)**
Block-level loop: solve for W_i, then update θ_i, repeat until convergence.
θ_i is part of the algorithm, not a protected component.

**Proposed fix**
Add a §12 D-row explicitly disclaiming the omission:

> **D-AA-SVD-refine** | Paper 2604.02119 Algorithm 2 | Spec omits block-level
> (W_i, θ_i) iterative refinement. Justification: H200 wall-time and VRAM cost
> of running a per-block convergence loop across 256 experts is prohibitive at
> current calibration set size; per-layer Path 1 is used instead. Quality gap:
> acknowledged; empirical comparison vs full Algorithm 2 is empirical_pending.

*(Alternative: implement Algorithm 2 fully — large scope, separate milestone.)*

> DECISION: reject
> NOTES: spec must match paper, we have to implement the right thing

---

### Phase B — D-Rank + Swift-SVD+

---

### F-ch12-missing-0004 · HIGH · silent_omission
**§6 Phase B (D-Rank), spec line 366**

**Problem**
Paper 2509.25622 computes effective rank from the *whitened* weight matrix:
S_g = cholesky(X^T X) in FP64, then R_eff from σ_i(S_g · W_g)². This makes the
rank allocation input-distribution-aware. The spec skips whitening entirely and
computes R_eff from the singular values of raw W in FP32. No §12 row covers this.
H200 supports FP64 Tensor Cores, so hardware is not the reason.

**Spec currently does**
R_eff from SVD of raw W, FP32.

**Paper requires (2509.25622 Eq. 1)**
R_eff from SVD of `S_g · W_g` where `S_g = cholesky(X^T X)` in FP64.

**Proposed fix**
Add a §12 D-row:

> **D-Drank-whiten** | Paper 2509.25622 Eq. 1 | Spec uses raw W singular values
> rather than whitened S·W. Justification: per-expert spectral share of raw W is
> a viable proxy; FP64 Cholesky per expert increases Phase B runtime ~2× for
> marginal rank-allocation improvement. Empirical comparison pending.

*(Alternative: implement the FP64 Cholesky whitening — moderate scope change.)*

> DECISION: reject
> NOTES: spec must match paper, we have to implement the right thing

---

### F-ch12-missing-0002 · HIGH · silent_omission
**§6 Phase B (Swift-SVD+), spec line 384**

**Problem**
Swift-SVD+ Algorithm 2 defines a minimal-rank floor:
`k_i ← max(k_i, floor(k̄ · δ))` with δ = 0.5 recommended. The paper explicitly
warns that δ = 0 is numerically unstable. The spec has no floor, which means any
expert can receive rank 0 after redistribution — precisely the regime the paper
warns against. No §12 row sanctions running at δ = 0.

**Spec currently does**
No minimal-rank floor; δ = 0 effectively.

**Paper requires (2604.01609 Algorithm 2)**
`k_i ← max(k_i, floor(k̄ · 0.5))` after rank redistribution.

**Proposed fix**
Accept-fix: add `k_i = max(k_i, floor(k_bar * 0.5))` in Phase B after rank
redistribution. Small, targeted change that directly prevents the paper-warned
instability. No §12 row needed once implemented.

*(Alternative: add a §12 D-row sanctioning δ = 0 with empirical evidence that
the instability does not manifest in per-expert scope.)*

> DECISION: accept-fix
> NOTES: 

---

### F-ch12-missing-0009 · MEDIUM · misdescribed_ch12
**§12 D7a, spec line 739**

**Problem**
D7a justifies the per-projection rank bias by noting that paper 2509.25622
(D-Rank) prescribes uniform ranks with no per-projection-type loop. But Swift-SVD+
Algorithm 2 (paper 2604.01609) also loops only over layers `i ∈ {1..L}` — it too
prescribes uniformity. D7a's deviation therefore applies against two papers but
cites only one. The sanction is not wrong, just incomplete.

**Spec currently does**
D7a "Paper Says" cites only 2509.25622.

**Both papers prescribe**
Per-layer uniformity; no per-projection-type rank variation.

**Proposed fix**
Extend D7a "Paper Says" column to read:

> *Both 2509.25622 Eq. 7 (D-Rank) and 2604.01609 Algorithm 2 (Swift-SVD+)
> allocate ranks per layer only, with no per-projection-type variation.*

Implementation and Justification columns are unchanged.

> DECISION: accept-fix
> NOTES: spec must match paper — D7a must cite both 2509.25622 (D-Rank) and 2604.01609 (Swift-SVD+) as both prescribe per-layer uniformity with no per-projection-type variation

---

## Stage 4 — EoRA (§7)

> **Dependency note — F-ch12-missing-0005 and F-ch12-missing-0006 are weakly
> coupled:**
> F-0005 is about what matrix is decomposed (full second-moment vs rank-1 mean
> outer product). F-0006 is about the rank cap (64 vs paper default 128).
> If you switch to the mean-vector input (F-0005), the useful rank per expert
> drops sharply (rank-1 input can only produce rank-1 signal), which would make
> a rank cap of 64 — or even lower — more than adequate. If you keep the
> second-moment input (reject/defer F-0005), the rank-cap justification in F-0006
> needs to stand on its own VRAM budget argument.

---

### F-ch12-missing-0005 · HIGH · silent_omission
**§7, spec line 463**

**Problem**
EoRA Algorithm 1 step 2 (paper 2410.21271) takes X̃ defined as the *average* of
input activations over the calibration set — a mean vector whose outer product is
rank-1. The spec reuses A = X^T X (full second-moment covariance, up to rank d_in)
from Stage 2. These produce fundamentally different eigenspaces: second-moment
captures activation spread, mean-outer captures only the mean direction. No §12
row covers this substitution. D10 covers noise-floor truncation of Q, not this
upstream matrix choice.

**Spec currently does**
`A = X^T X` (full second-moment covariance, Stage 2 reuse).

**Paper requires (2410.21271 Algorithm 1)**
`X̃ = mean(X)` → outer product X̃ X̃^T (rank-1).

**Proposed fix**
Add a §12 D-row:

> **D-EoRA-input** | Paper 2410.21271 Algorithm 1 | Spec uses second-moment
> covariance A = X^T X instead of mean-activation outer product X̃ X̃^T.
> Justification: second-moment covariance captures activation spread; rank-1
> mean outer product collapses to a single direction, likely losing residual
> structure. Stage 2 covariance is available at zero marginal cost. Empirical
> comparison pending.

> DECISION: reject
> NOTES: spec must match paper — implement mean-activation outer product X̃X̃^T as prescribed by EoRA Algorithm 1 step 2; remove the X^T X substitution

---

### F-ch12-missing-0006 · MEDIUM · silent_omission
**§7, spec line 479**
*See dependency note above — this decision depends on F-ch12-missing-0005.*

**Problem**
The EoRA paper (2410.21271) uses a default rank of ~128 per expert. The spec caps
rank at `eigenspace_rank_cap = 64`. No §12 row explains why. The cap is likely
VRAM-driven but no hardware-budget arithmetic is documented.

**Spec currently does**
`eigenspace_rank_cap = 64` (half the paper default).

**Paper default**
~128 rank per expert.

**Proposed fix**
Add a §12 D-row with the budget arithmetic:

> **D-EoRA-rank** | Paper 2410.21271 default ~128 | Spec caps at 64.
> Justification: VRAM budget — 256 experts × 64 rank × 3 matrices × 2 bytes
> (BF16) × 2 (A+B) ≈ 12 GB adapter storage; rank 128 doubles this to ~24 GB
> additional VRAM during merge. Empirical quality gap rank 64 vs 128: pending.

*(Note: if F-ch12-missing-0005 is accepted-fix and the input switches to rank-1
mean outer product, this row's justification should reference that change.)*

> DECISION: reject
> NOTES: spec must match paper — implement rank ~128 per expert as per paper default; given F-ch12-missing-0005 is also rejected (mean outer product adopted), a rank cap lower than 128 may be revisited empirically, but default must start at paper value

---

## §11 — Crash-Resume

---

### F-resume-0002 · MEDIUM · durability_gap
**§11, spec line 695**

**Problem**
§11 describes atomic checkpointing via `.tmp + os.replace`. The crash-resume
truth reference states that `fsync(file)` followed by `fsync(parent_dir)` is
required before the rename to survive power-loss or kernel-panic. The spec omits
both fsyncs and does not state that its guarantee is limited to SIGKILL/timeout.
A reader may assume broader durability guarantees than the spec actually provides.

**Spec currently does**
`.tmp + os.replace` with no threat-model scope statement and no fsync.

**Truth requires**
Either fsync both file and parent directory before rename (full durability), or
an explicit scope statement that limits the guarantee to SIGKILL/timeout only.

**Proposed fix**
Accept-fix (scope-narrow variant): add one sentence to §11 after the `.tmp +
os.replace` description:

> *Durability scope: SIGKILL and training-framework timeout only. Power-loss and
> kernel-panic are out of scope; fsync-before-rename is not implemented.*

This is honest and avoids implying durability the spec does not provide.

> DECISION: reject
> NOTES: spec must match the truth reference — implement fsync(file) + fsync(parent_dir) before os.replace to provide full durability against power-loss and kernel-panic, not just SIGKILL/timeout

---

## H200 Performance Claims — Info Batch (19 findings)

**Findings covered:**
F-h200-0001 through F-h200-0020 (all 19 H200-regime info findings)

**Pattern**
Every finding here is the same: the spec states a timing or speedup claim
(e.g., "~5 min Stage 1", "8–12× speedup for Stage 6", "~50% saving via teacher
cache") and the audit marks it *unverifiable* because no H200 truth reference
provides a measured figure for that specific claim. These are engineering
projections, not measured benchmarks.

**Affected spec lines**
62, 138, 224, 274, 426, 533, 539, 568, 587, 588 (×2), 595, 602, 619, 620, 621,
651, 656, 712.

**Proposed fix — two options, choose one:**

*Option A — Defer:* Accept the claims as projections. Add a single note in the
spec's opening section (or a §13 "Performance Estimates" callout) stating that
all timing/speedup figures are projected estimates based on algorithmic analysis
and that actual H200 benchmarks are pending. No individual lines need editing.

*Option B — Label in-line:* Prefix each affected claim with "est." or
"(projected)" in the spec text. More granular, but touches ~15 lines.

> DECISION: accept-fix (Option A)
> NOTES: add a single global note in the spec's opening section or a §13 "Performance Estimates" callout stating all timing/speedup figures are projected estimates and actual H200 benchmarks are pending; no per-line edits

---

## Summary table

| ID | Stage | Severity | Decision |
|---|---|---|---|
| F-paper-2507.23279-0006 | Stage 1 | HIGH | |
| F-paper-2507.23279-0037 | Stage 1 | HIGH | |
| F-paper-2507.23279-0016 | Stage 1 | HIGH | |
| F-paper-2507.23279-0015 | Stage 1 | MEDIUM | |
| F-paper-2507.23279-0012 | Stage 1 | MEDIUM | |
| F-paper-2507.23279-0038 | Stage 1 | LOW | |
| F-ch12-missing-0007 | Stage 2 | MEDIUM | |
| F-ch12-missing-0011 | Stage 2 | LOW | |
| F-cross-h200-0001 | Stage 2 | LOW | |
| F-ch12-0001 | Stage 2 | MEDIUM | |
| F-resume-0001 | Stage 2 | LOW | |
| F-ch12-missing-0008 | Stage 2.5 | LOW | |
| F-ch12-missing-0001 | Stage 3 Phase A | CRITICAL | |
| F-ch12-missing-0004 | Stage 3 Phase B | HIGH | |
| F-ch12-missing-0002 | Stage 3 Phase B | HIGH | |
| F-ch12-missing-0009 | Stage 3 Phase B | MEDIUM | |
| F-ch12-missing-0005 | Stage 4 | HIGH | |
| F-ch12-missing-0006 | Stage 4 | MEDIUM | |
| F-resume-0002 | §11 | MEDIUM | |
| F-h200-* batch (19) | H200 claims | INFO | |
