# Spec Compliance Audit — Findings

## Executive Summary

Total findings (after dedup): **38** (from 97 raw rows; 59 duplicates merged).

### Tally by regime

| Regime | Count |
|---|---|
| paper | 6 |
| h200 | 19 |
| resume | 2 |
| cross | 11 |

### Tally by severity

| Severity | Count |
|---|---|
| critical | 1 |
| high | 6 |
| medium | 7 |
| low | 5 |
| minor | 0 |
| info | 19 |

### Tally by regime × severity

| Regime | critical | high | medium | low | minor | info |
|---|---|---|---|---|---|---|
| paper | 0 | 3 | 2 | 1 | 0 | 0 |
| h200 | 0 | 0 | 0 | 0 | 0 | 19 |
| resume | 0 | 0 | 1 | 1 | 0 | 0 |
| cross | 1 | 3 | 4 | 3 | 0 | 0 |

### Headline issues (top 5 by severity)

- **F-ch12-missing-0001** [critical] silent_omission — spec line 332: AA-SVD's most critical contribution — block-level refinement of (W_i, θ_i) — is wholly absent from Stage 3 and uncovered by §12. Spec implements only per-layer Path 1 (anchored-adaptive objective 4) w
- **F-paper-2507.23279-0006** [high] uncovered_deviation — spec line 142: Major deviation from canonical SE detection criterion. Justification is paper-internal/algorithmic (avoids two-pass global stats and MA-formation layer detection); sanctioned as ch12. Note: paper Tabl
- **F-ch12-missing-0004** [high] silent_omission — spec line 366: D-Rank's effective rank in the paper is computed from singular values of the WHITENED matrix S·W (Eq. 1: λ^i_g = σ_i(S_g·W_g)²) where S_g = cholesky(X^T X) in FP64. Spec computes p_i and R_eff directl
- **F-ch12-missing-0002** [high] silent_omission — spec line 384: Swift-SVD+ Algorithm 2's minimal-rank floor (k_i ← k̄·δ, recommended δ=0.5; paper warns δ=0 is unstable) is not implemented. Spec runs the paper's explicitly-warned-against unstable regime. No §12 row
- **F-ch12-missing-0005** [high] silent_omission — spec line 463: EoRA Algorithm 1 step 2 inputs 'X̃: Average of the input activations' — i.e. mean vector outer product (rank-1). Spec uses A = X^T X full second-moment covariance reused from Stage 2 (rank up to d_in)

## Paper-regime findings

### High

### F-paper-2507.23279-0006 [HIGH · uncovered_deviation]
**Spec**: §§4 Stage 1, line 142
**Summary**: Major deviation from canonical SE detection criterion. Justification is paper-internal/algorithmic (avoids two-pass global stats and MA-formation layer detection); sanctioned as ch12. Note: paper Table 9 (Appendix I) shows SE counts robust across (P95..P99.9, 0.07..0.10·amax), but the per-layer z-score is a different statistic entirely — not within the paper's robustness band.
**Evidence**:
- **paper**: "a_{l,e} > P99.5 and a_{l,e} > (1/10) * a_max and l in L" _(source: 2507.23279)_
- **ch12**: "{"row": "D1"}" _(source: ch12)_
**Recommended fix**: Already sanctioned in §12 D1; track for empirical verification.

### F-paper-2507.23279-0037 [HIGH · uncovered_deviation]
**Spec**: §§4 Stage 1, line 730
**Summary**: Spec explicitly skips Algorithm 1 Stage 1 (MA-formation layer detection). D1 sanctions this on grounds of avoiding two-pass collection and conservative behavior of per-layer z-score. Same deviation as #eq6_se_def.
**Evidence**:
- **paper**: "Stage 1: Calibration of MA-formation Layers. L <- empty; ... if MA pattern detected in Hl(x): L <- L union {l}" _(source: 2507.23279)_
- **ch12**: "{"row": "D1"}" _(source: ch12)_
**Recommended fix**: Already sanctioned in §12 D1; track for empirical verification.
**Dup IDs**: F-paper-2507.23279-0039, F-paper-2507.23279-0047

### F-paper-2507.23279-0016 [HIGH · uncovered_deviation]
**Spec**: §Stage 1 / §4, line n/a
**Summary**: Without l∈L filter, late-layer outlier experts (Table 7: L47E48, L47E100 on Qwen3) could match per-layer z>2.5 and be added to the blacklist. The 4-per-layer cap and 5% global cap (D2) bound the damage, but this is a real false-positive risk. Sanctioned via D1's 'per-layer z-score is conservative in practice (z>10 typically)' — but Table 7 outlier experts may approach that threshold.
**Evidence**:
- **paper**: "some experts in the final layers also exhibit extreme activation outliers. However ... they do not hold the same level of significance as SEs." _(source: 2507.23279)_
- **ch12**: "{"row": "D1"}" _(source: ch12)_
**Recommended fix**: Already sanctioned in §12 D1; track for empirical verification.
**Dup IDs**: F-paper-2507.23279-0027, F-paper-2507.23279-0035

### Medium

### F-paper-2507.23279-0015 [MEDIUM · paraphrase_drift]
**Spec**: §§4 Stage 1, line 142
**Summary**: Spec's 5% global cap is 10× the paper's <0.5% empirical bound — generous but loose. D2 frames this as a safety guardrail; treated as ch12-sanctioned. Per-layer cap of 4 is empirical/conservative. Drift in stated bound, not a violation.
**Evidence**:
- **paper**: "SEs ... accounting for less than 0.5% of all experts" _(source: 2507.23279)_
- **ch12**: "{"row": "D2"}" _(source: ch12)_
**Recommended fix**: Already sanctioned in §12 D2; track for empirical verification.
**Dup IDs**: F-paper-2507.23279-0052

### F-paper-2507.23279-0012 [MEDIUM · silent_omission]
**Spec**: §Stage 1 / §4, line n/a
**Summary**: This is the canonical SE ground truth for the spec's target model. Spec does not name these explicitly; with D1's per-layer z-score, output may differ. Reproducibility check should compare detector output to this set. Logged as info — spec omission is not a violation, but a regression test would help.
**Evidence**:
- **paper**: "Qwen3-30B-A3B  Layer 1 Expert 68, Layer 2 Expert 92, Layer 3 Expert 82" _(source: 2507.23279)_
**Recommended fix**: Consider whether spec should ingest this paper detail.
**Dup IDs**: F-paper-2507.23279-0029, F-paper-2507.23279-0050, F-paper-2507.23279-0001, F-paper-2507.23279-0002, F-paper-2507.23279-0003, F-paper-2507.23279-0004, F-paper-2507.23279-0005, F-paper-2507.23279-0007, F-paper-2507.23279-0008, F-paper-2507.23279-0009, F-paper-2507.23279-0010, F-paper-2507.23279-0011, F-paper-2507.23279-0013, F-paper-2507.23279-0014, F-paper-2507.23279-0017, F-paper-2507.23279-0018, F-paper-2507.23279-0019, F-paper-2507.23279-0020, F-paper-2507.23279-0021, F-paper-2507.23279-0022, F-paper-2507.23279-0023, F-paper-2507.23279-0024, F-paper-2507.23279-0025, F-paper-2507.23279-0026, F-paper-2507.23279-0028, F-paper-2507.23279-0030, F-paper-2507.23279-0031, F-paper-2507.23279-0032, F-paper-2507.23279-0033, F-paper-2507.23279-0034, F-paper-2507.23279-0036, F-paper-2507.23279-0040, F-paper-2507.23279-0041, F-paper-2507.23279-0042, F-paper-2507.23279-0043, F-paper-2507.23279-0044, F-paper-2507.23279-0045, F-paper-2507.23279-0046, F-paper-2507.23279-0048, F-paper-2507.23279-0049, F-paper-2507.23279-0051, F-paper-2507.23279-0053, F-paper-2507.23279-0054, F-paper-2507.23279-0055, F-paper-2507.23279-0056, F-paper-2507.23279-0057, F-paper-2507.23279-0058, F-paper-2507.23279-0059, F-paper-2507.23279-0060, F-paper-2507.23279-0061, F-paper-2507.23279-0062

### Low

### F-paper-2507.23279-0038 [LOW · paraphrase_drift]
**Spec**: §§4 Stage 1, line 135
**Summary**: Signal collection matches exactly. The paraphrase drift is minor — paper writes |h·W_d| (the down_proj output, post-multiplication); spec writes max(|down_proj_output|). These are equivalent; no real deviation.
**Evidence**:
- **paper**: "a_{l,e} <- max_{x in D} | h_{l,e}(x) * W^{l,e}_{down_proj} |" _(source: 2507.23279)_
**Recommended fix**: Resolve paraphrase drift in spec text.

## H200-regime findings

### Info

### F-h200-0001 [INFO · missing_citation]
**Spec**: §n/a, line 62
**Summary**: H200-0002: ~5 min Stage 1 wall-time on H200 has no direct truth ref; truth file does not contain pipeline benchmark numbers.
**Evidence**:
- **h200_truth**: "{"verdict": "unverifiable", "detail": "~5 min Stage 1 wall-time on H200 has no direct truth ref; truth file does not contain pipeline benchmark numbers."}" _(source: h200_truth)_
**Recommended fix**: Empirical benchmark or add truth-ref citation; spec claim lacks independent verification source.

### F-h200-0002 [INFO · missing_citation]
**Spec**: §n/a, line 138
**Summary**: H200-0004: Wall-time ~5 min for Phase A single forward pass not covered by truth refs (no benchmark facts in truth).
**Evidence**:
- **h200_truth**: "{"verdict": "unverifiable", "detail": "Wall-time ~5 min for Phase A single forward pass not covered by truth refs (no benchmark facts in truth)."}" _(source: h200_truth)_
**Recommended fix**: Empirical benchmark or add truth-ref citation; spec claim lacks independent verification source.

### F-h200-0003 [INFO · missing_citation]
**Spec**: §n/a, line 224
**Summary**: H200-0006: 10–100x vectorization speedup is a planned-not-implemented projection; no truth ref provides empirical bound.
**Evidence**:
- **h200_truth**: "{"verdict": "unverifiable", "detail": "10–100x vectorization speedup is a planned-not-implemented projection; no truth ref provides empirical bound."}" _(source: h200_truth)_
**Recommended fix**: Empirical benchmark or add truth-ref citation; spec claim lacks independent verification source.
**Dup IDs**: F-h200-0004

### F-h200-0005 [INFO · missing_citation]
**Spec**: §n/a, line 274
**Summary**: H200-0008: fp16 covariance storage choice is implementation detail; not contradicted by H200 truth refs (truth covers FP8/BF16/FP16 hardware support generally).
**Evidence**:
- **h200_truth**: "{"verdict": "unverifiable", "detail": "fp16 covariance storage choice is implementation detail; not contradicted by H200 truth refs (truth covers FP8/BF16/FP16 hardware support generally)."}" _(source: h200_truth)_
**Recommended fix**: Empirical benchmark or add truth-ref citation; spec claim lacks independent verification source.

### F-h200-0006 [INFO · missing_citation]
**Spec**: §n/a, line 426
**Summary**: H200-0016: ~25% Phase C speedup from eigh caching not in truth refs (algorithmic projection).
**Evidence**:
- **h200_truth**: "{"verdict": "unverifiable", "detail": "~25% Phase C speedup from eigh caching not in truth refs (algorithmic projection)."}" _(source: h200_truth)_
**Recommended fix**: Empirical benchmark or add truth-ref citation; spec claim lacks independent verification source.

### F-h200-0007 [INFO · missing_citation]
**Spec**: §n/a, line 533
**Summary**: H200-0023: Hyperparameter values (lr/bs/grad_accum/seq_len) are paper-derived; H200 truth refs only constrain hardware. See paper crossref 2603.02217#tab1-* for paper consistency.
**Evidence**:
- **h200_truth**: "{"verdict": "unverifiable", "detail": "Hyperparameter values (lr/bs/grad_accum/seq_len) are paper-derived; H200 truth refs only constrain hardware. See paper crossref 2603.02217#tab1-* for paper consistency."}" _(source: h200_truth)_
**Recommended fix**: Empirical benchmark or add truth-ref citation; spec claim lacks independent verification source.

### F-h200-0020 [INFO · missing_citation]
**Spec**: §n/a, line 539
**Summary**: H200-0043: max_calib_samples=3000 and τ=1.0 are paper-derived hyperparameters; H200 truth does not constrain them.
**Evidence**:
- **h200_truth**: "{"verdict": "unverifiable", "detail": "max_calib_samples=3000 and τ=1.0 are paper-derived hyperparameters; H200 truth does not constrain them."}" _(source: h200_truth)_
**Recommended fix**: Empirical benchmark or add truth-ref citation; spec claim lacks independent verification source.

### F-h200-0008 [INFO · missing_citation]
**Spec**: §n/a, line 568
**Summary**: H200-0027: WikiText-2 PPL eval at seq_len=2048 is a methodology choice; not constrained by H200 truth.
**Evidence**:
- **h200_truth**: "{"verdict": "unverifiable", "detail": "WikiText-2 PPL eval at seq_len=2048 is a methodology choice; not constrained by H200 truth."}" _(source: h200_truth)_
**Recommended fix**: Empirical benchmark or add truth-ref citation; spec claim lacks independent verification source.

### F-h200-0009 [INFO · missing_citation]
**Spec**: §n/a, line 586
**Summary**: H200-0028: PPL batch size raise is methodology; truth refs do not cover eval batching. Numerical equivalence covered by H200-0035.
**Evidence**:
- **h200_truth**: "{"verdict": "unverifiable", "detail": "PPL batch size raise is methodology; truth refs do not cover eval batching. Numerical equivalence covered by H200-0035."}" _(source: h200_truth)_
**Recommended fix**: Empirical benchmark or add truth-ref citation; spec claim lacks independent verification source.

### F-h200-0010 [INFO · missing_citation]
**Spec**: §n/a, line 587
**Summary**: H200-0029: lm-eval auto:8 batching is standard; numerical equivalence in H200-0036.
**Evidence**:
- **h200_truth**: "{"verdict": "unverifiable", "detail": "lm-eval auto:8 batching is standard; numerical equivalence in H200-0036."}" _(source: h200_truth)_
**Recommended fix**: Empirical benchmark or add truth-ref citation; spec claim lacks independent verification source.

### F-h200-0011 [INFO · missing_citation]
**Spec**: §n/a, line 588
**Summary**: H200-0030: Generative batched eval is methodology; numerical equivalence in H200-0037.
**Evidence**:
- **h200_truth**: "{"verdict": "unverifiable", "detail": "Generative batched eval is methodology; numerical equivalence in H200-0037."}" _(source: h200_truth)_
**Recommended fix**: Empirical benchmark or add truth-ref citation; spec claim lacks independent verification source.

### F-h200-0012 [INFO · missing_citation]
**Spec**: §n/a, line 595
**Summary**: H200-0032: Teacher I/O overlap savings ~3-5 min is empirical projection; not in truth refs.
**Evidence**:
- **h200_truth**: "{"verdict": "unverifiable", "detail": "Teacher I/O overlap savings ~3-5 min is empirical projection; not in truth refs."}" _(source: h200_truth)_
**Recommended fix**: Empirical benchmark or add truth-ref citation; spec claim lacks independent verification source.

### F-h200-0013 [INFO · missing_citation]
**Spec**: §n/a, line 602
**Summary**: H200-0033: ~50% wall-time saving via teacher cache is empirical; not in truth refs.
**Evidence**:
- **h200_truth**: "{"verdict": "unverifiable", "detail": "~50% wall-time saving via teacher cache is empirical; not in truth refs."}" _(source: h200_truth)_
**Recommended fix**: Empirical benchmark or add truth-ref citation; spec claim lacks independent verification source.

### F-h200-0014 [INFO · missing_citation]
**Spec**: §n/a, line 619
**Summary**: H200-0035: Per-token NLL recovery formula is mathematical, not H200-specific. Numerical-identity claim is correct in principle (sum over tokens batch-invariant) but empirical exactness on H200 not in truth.
**Evidence**:
- **h200_truth**: "{"verdict": "unverifiable", "detail": "Per-token NLL recovery formula is mathematical, not H200-specific. Numerical-identity claim is correct in principle (sum over tokens batch-invariant) but empirical exactness on H200 not in truth."}" _(source: h200_truth)_
**Recommended fix**: Empirical benchmark or add truth-ref citation; spec claim lacks independent verification source.

### F-h200-0015 [INFO · missing_citation]
**Spec**: §n/a, line 620
**Summary**: H200-0036: lm-eval determinism with left-padding is library behavior; truth refs do not cover lm-eval. Plausible.
**Evidence**:
- **h200_truth**: "{"verdict": "unverifiable", "detail": "lm-eval determinism with left-padding is library behavior; truth refs do not cover lm-eval. Plausible."}" _(source: h200_truth)_
**Recommended fix**: Empirical benchmark or add truth-ref citation; spec claim lacks independent verification source.

### F-h200-0016 [INFO · missing_citation]
**Spec**: §n/a, line 621
**Summary**: H200-0037: Greedy + left-pad + attention_mask preserves argmax in principle; not directly in truth refs.
**Evidence**:
- **h200_truth**: "{"verdict": "unverifiable", "detail": "Greedy + left-pad + attention_mask preserves argmax in principle; not directly in truth refs."}" _(source: h200_truth)_
**Recommended fix**: Empirical benchmark or add truth-ref citation; spec claim lacks independent verification source.

### F-h200-0017 [INFO · missing_citation]
**Spec**: §n/a, line 651
**Summary**: H200-0039: Baseline Stage 6 wall-time projection (~200-320 min) not in truth refs.
**Evidence**:
- **h200_truth**: "{"verdict": "unverifiable", "detail": "Baseline Stage 6 wall-time projection (~200-320 min) not in truth refs."}" _(source: h200_truth)_
**Recommended fix**: Empirical benchmark or add truth-ref citation; spec claim lacks independent verification source.

### F-h200-0018 [INFO · missing_citation]
**Spec**: §n/a, line 656
**Summary**: H200-0040: 8-12x speedup projection (3-5 hr → 25-30 min) is empirical; not in truth refs.
**Evidence**:
- **h200_truth**: "{"verdict": "unverifiable", "detail": "8-12x speedup projection (3-5 hr → 25-30 min) is empirical; not in truth refs."}" _(source: h200_truth)_
**Recommended fix**: Empirical benchmark or add truth-ref citation; spec claim lacks independent verification source.

### F-h200-0019 [INFO · missing_citation]
**Spec**: §n/a, line 712
**Summary**: H200-0042: Resume-savings numbers (60s teacher load, 33 min α save) are empirical; not in truth refs.
**Evidence**:
- **h200_truth**: "{"verdict": "unverifiable", "detail": "Resume-savings numbers (60s teacher load, 33 min α save) are empirical; not in truth refs."}" _(source: h200_truth)_
**Recommended fix**: Empirical benchmark or add truth-ref citation; spec claim lacks independent verification source.

## Resume-regime findings

### Medium

### F-resume-0002 [MEDIUM · durability_gap]
**Spec**: §§11 Within-Stage Crash-Resume, line 695
**Summary**: Spec describes the .tmp+os.replace idiom but omits the fsync(file)+fsync(parent_dir)-before-rename invariant. For SIGKILL-only threat models (the spec's stated scope) os.replace alone suffices; for power-loss/kernel-panic the missing fsync is a durability gap.
**Evidence**:
- **resume_truth**: "{"truth_ref": "resume_truth.md#L18", "fact": "fsync of file (and parent directory) is REQUIRED BEFORE rename for crash-durability; without fsync, rename can be reordered with the data write and a power-loss / kernel-panic crash can leave the renamed file pointing at zero-length or stale data."}" _(source: resume_truth)_
**Recommended fix**: Either narrow §11 threat model to SIGKILL/timeout explicitly, or extend the atomic-write description to include fsync of the file and parent directory before os.replace.

### Low

### F-resume-0001 [LOW · silent_omission]
**Spec**: §§5 Stage 2 Resume, line 284
**Summary**: Stage 2 'atomic checkpointing' line does not state the .tmp+os.replace mechanism nor the .pt-before-.json ordering at point of use; both invariants exist in §11 (lines 695, 710) so the spec is internally complete, but this section does not self-contain the atomicity definition.
**Evidence**:
- **resume_truth**: "{"truth_ref": "resume_truth.md#L15,L20", "fact": "Atomicity = .tmp+os.replace; ordering = payload(.pt) before manifest(.json)."}" _(source: resume_truth)_
**Recommended fix**: Add a one-line forward-reference to §11 atomic-write idiom and §11 layer-completion ordering at line 284.

## Cross-cutting findings

### Critical

### F-ch12-missing-0001 [CRITICAL · silent_omission]
**Spec**: §§6 Stage 3, line 332
**Summary**: AA-SVD's most critical contribution — block-level refinement of (W_i, θ_i) — is wholly absent from Stage 3 and uncovered by §12. Spec implements only per-layer Path 1 (anchored-adaptive objective 4) without the iterative block refinement that the paper's §6 ablation shows dominates the layer objective. §10 'Protected Components' explicitly forbids RMSNorm modification, directly contradicting the paper's prescription that θ_i be tuned during refinement. Paper crossref flags 8 separate deviation_uncovered claims around this single architectural omission.
**Evidence**:
- **paper**: "{"paper_id": "2604.02119", "claims": ["fig2_block_loss", "alg2_refine", "ablation_refine_dominant", "paper_recommendation", "sec3_3_theta", "alg2_inner", "alg2_advance", "contributions"]}" _(source: 2604.02119)_
- **ch12**: "{"row_id": null, "covered_by": "D6 only (cross-covariance scope)"}" _(source: ch12)_
**Recommended fix**: Either (a) implement Stage 3 block refinement per Algorithm 2 (with θ_i optimization scoped narrowly enough to coexist with §10), and remove RMSNorm from Protected Components for that step, or (b) add a §12 D-row explicitly disclaiming the omission with a justification (e.g. wall-time/VRAM cost of block refinement on H200) and a quantitative quality-gap acknowledgement.

### High

### F-ch12-missing-0004 [HIGH · silent_omission]
**Spec**: §§6 Phase B (D-Rank), line 366
**Summary**: D-Rank's effective rank in the paper is computed from singular values of the WHITENED matrix S·W (Eq. 1: λ^i_g = σ_i(S_g·W_g)²) where S_g = cholesky(X^T X) in FP64. Spec computes p_i and R_eff directly from raw W singular values (no S, no whitening) and uses fp32 max precision. D7 only sanctions the ω parameter-cost form — the upstream whitening pipeline is wholly absent. H200 supports FP64 Tensor Core (h200_truth.md), so there is no hardware reason to drop FP64.
**Evidence**:
- **paper**: "{"paper_id": "2509.25622", "claims": ["prelim_whitening", "post_eq7_pipeline", "hp_hardware", "whitening_S_definition"]}" _(source: 2509.25622)_
- **ch12**: "{"row_id": null, "covered_by": "D7 (only ω form)"}" _(source: ch12)_
**Recommended fix**: Either (a) implement S = cholesky(X^T X) in FP64 and compute R_eff from σ_i(S·W)² per paper Eq. 1, or (b) add a §12 D-row sanctioning the unwhitened R_eff with empirical evidence that per-expert spectral share is a viable proxy.

### F-ch12-missing-0002 [HIGH · silent_omission]
**Spec**: §§6 Phase B, line 384
**Summary**: Swift-SVD+ Algorithm 2's minimal-rank floor (k_i ← k̄·δ, recommended δ=0.5; paper warns δ=0 is unstable) is not implemented. Spec runs the paper's explicitly-warned-against unstable regime. No §12 row sanctions this.
**Evidence**:
- **paper**: "{"paper_id": "2604.01609", "claims": ["minimal_rank", "delta_role", "alg2_minimal_rank_def", "flex_pool"]}" _(source: 2604.01609)_
- **ch12**: "{"row_id": null}" _(source: ch12)_
**Recommended fix**: Implement δ=0.5 floor in Phase B per-expert redistribution, OR add a §12 D-row sanctioning δ=0 with empirical evidence that on the spec's per-expert (rather than per-layer) scope the instability does not manifest.
**Dup IDs**: F-ch12-missing-0003

### F-ch12-missing-0005 [HIGH · silent_omission]
**Spec**: §§7 Stage 4, line 463
**Summary**: EoRA Algorithm 1 step 2 inputs 'X̃: Average of the input activations' — i.e. mean vector outer product (rank-1). Spec uses A = X^T X full second-moment covariance reused from Stage 2 (rank up to d_in). The eigenspace differs fundamentally; this is a material numerical deviation not covered by D10 (D10 only addresses noise-floor truncation of Q, not the matrix being decomposed).
**Evidence**:
- **paper**: "{"paper_id": "2410.21271", "claims": ["claim_xtilde_mean"]}" _(source: 2410.21271)_
- **ch12**: "{"row_id": null}" _(source: ch12)_
**Recommended fix**: Add a §12 D-row sanctioning second-moment covariance over mean-outer-product. Justification likely 'second-moment is strictly more informative; mean-outer rank-1 collapses too aggressively' — but this needs explicit acknowledgement.

### Medium

### F-ch12-missing-0007 [MEDIUM · silent_omission]
**Spec**: §§5 Stage 2, line 274
**Summary**: REAM uses 3072 sequences × 512 tokens for calibration; spec uses 1024 sequences. D11 sanctions calibration data SOURCE for Router KD (2603.02217 c4 → Nemotron) but is silent on REAM's calibration SIZE.
**Evidence**:
- **paper**: "{"paper_id": "2604.04356", "claims": ["hp_calib_size", "claim_calib_seqs_3072_x_512"]}" _(source: 2604.04356)_
- **ch12**: "{"row_id": null, "covered_by": "D11 covers a different paper (2603.02217)"}" _(source: ch12)_
**Recommended fix**: Extend D11 to cover REAM's calibration size, or add a separate D-row.
**Dup IDs**: F-ch12-missing-0010

### F-ch12-missing-0006 [MEDIUM · silent_omission]
**Spec**: §§7 Stage 4, line 479
**Summary**: Spec caps EoRA rank at eigenspace_rank_cap=64 per expert per matrix; paper default is ~128. No §12 row. Likely VRAM-driven (256 experts × 64 rank × 3 matrices is already large) but no h200 fact backs it.
**Evidence**:
- **paper**: "{"paper_id": "2410.21271", "claims": ["hp_default_rank"]}" _(source: 2410.21271)_
- **ch12**: "{"row_id": null}" _(source: ch12)_
**Recommended fix**: Add a §12 D-row citing h200 VRAM bound, or document the budget arithmetic that gives 64 as the maximum compatible cap.

### F-ch12-0001 [MEDIUM · misdescribed_ch12]
**Spec**: §§12 D5b, line 736
**Summary**: D5b 'Paper Says' incorrectly states REAM Eq. 6 has 'no permutation'. The paper text immediately following Eq. 6 explicitly prescribes 'neuron permutation alignment (Ainsworth et al., 2023) applied w.r.t. the dominant (centroid) expert' — citing the same Git Re-Basin reference the spec uses. The implementation's Hungarian alignment is therefore NOT a deviation from the paper; both prescribe permutation alignment. The genuine (minor) deviation is the specific cost matrix C = C_wt + C_act, which the paper does not specify.
**Evidence**:
- **paper**: "W_merged = (Σ S^freq_i W_i) / (Σ S^freq_i), (6), where W_i are expert i's weight matrices with neuron permutation alignment (Ainsworth et al., 2023) applied w.r.t. the dominant (centroid) expert." _(source: 2604.04356)_
- **ch12**: "{"row_id": "D5b", "claimed": "Eq. 6: frequency-weighted average ... (no permutation)"}" _(source: ch12)_
**Recommended fix**: Rewrite D5b 'Paper Says' to: '2604.04356 Eq. 6: frequency-weighted average with neuron permutation alignment (Ainsworth et al., 2023) applied to the centroid; cost matrix unspecified'. Reframe deviation as 'specific cost form (C_wt + C_act)' rather than 'permutation vs no permutation'. TODO ablation language is appropriate; tag empirical_pending.

### F-ch12-missing-0009 [MEDIUM · misdescribed_ch12]
**Spec**: §§12 D7a, line 739
**Summary**: D7a's 'Paper Says' cites only 2509.25622 (D-Rank) for the per-projection uniformity claim, but Swift-SVD+ Algorithm 2 (2604.01609) also indexes only over layers i ∈ {1..L} with no per-projection-type loop — i.e. Swift-SVD+ also prescribes uniformity that D7a violates. The deviation thus applies against TWO papers, not one.
**Evidence**:
- **paper**: "{"paper_id": "2604.01609", "claims": ["per_proj_uniform"]}" _(source: 2604.01609)_
- **ch12**: "{"row_id": "D7a"}" _(source: ch12)_
**Recommended fix**: Extend D7a's 'Paper Says' to cite both 2509.25622 Eq. 7 AND 2604.01609 Algorithm 2, both of which prescribe per-projection uniformity. The Implementation/Justification columns are unchanged.

### Low

### F-ch12-missing-0011 [LOW · silent_omission]
**Spec**: §§5 Stage 2 calibration, line 89
**Summary**: REAP's calibration source is c4 + evol-codealpaca; spec uses Nemotron-Cascade-2-SFT-Data. D11 only sanctions the calibration-source change against 2603.02217 (Router KD), not against REAP. Same change, different paper citation.
**Evidence**:
- **paper**: "{"paper_id": "2510.13999", "claims": ["hp-calib-small"]}" _(source: 2510.13999)_
- **ch12**: "{"row_id": null, "covered_by": "D11 covers a different paper"}" _(source: ch12)_
**Recommended fix**: Extend D11's paper citation list to include 2510.13999, or add a sibling row.

### F-cross-h200-0001 [LOW · unsupported_justification]
**Spec**: §§5 Covariance Side-Collection, line 274
**Summary**: Paper deviation 2604.01609#fp_precision_table5 is sanctioned by 'h200' but the backing H200 claim H200-0008 audits unverifiable, not consistent. The FP16 covariance storage choice is a precision/storage trade-off independent of H200 hardware (any GPU could store FP16); 'h200' is not the right sanction regime.
**Evidence**:
- **paper**: "{"claim_id": "2604.01609#fp_precision_table5", "sanction_justification": "h200", "note": "Spec stores Stage 2 covariance in FP16 (paper certifies FP32 only); paper crossref records sanction_justification=h200."}" _(source: paper)_
- **h200_truth**: "{"backing_h200_claim": "H200-0008", "audit_verdict": "unverifiable", "note": "H200-0008 audits as unverifiable — H200 truth refs do not justify FP16 storage choice (truth covers hardware FP support generally; FP16 vs FP32 covariance storage is an algorithmic precision decision, not an H200-specific affordance). The 'h200' sanction is therefore not anchored to a consistent H200 claim."}" _(source: h200_truth)_
**Recommended fix**: Either (a) re-sanction the FP16 covariance deviation under Ch.12 with an explicit numerical-precision row, or (b) add an H200-specific empirical justification (e.g. measured FP16 vs FP32 PPL on H200) so that an H200 claim can audit consistent.

### F-ch12-missing-0008 [LOW · silent_omission]
**Spec**: §§5 Stage 2.5, line 295
**Summary**: REAM evaluates 'without any fine-tuning after compression'. Spec adds Stage 2.5 (post-merge router KD), departing from REAM's static protocol. Paper protocol mixing (REAM static + 2603.02217 Router KD) is uncovered by §12.
**Evidence**:
- **paper**: "{"paper_id": "2604.04356", "claims": ["claim_no_finetune", "claim_static_only"]}" _(source: 2604.04356)_
- **ch12**: "{"row_id": null}" _(source: ch12)_
**Recommended fix**: Add a §12 D-row explicitly noting the protocol blend: REAM merge + Router KD recalibration, citing both papers.

## Watch items (empirical_pending)

_No findings tagged `empirical_pending` in current dataset. Tracked watch items per playbook (Ch.12 D5b, D7a etc.) are sanctioned in-spec and remain pending empirical confirmation:_

- **D5b** — Git Re-Basin permutation symmetry, ablation pending
- **D7a** — per-projection rank bias, ablation pending
- **EoRA V_corr** — correction term effectiveness, empirical pending
- **Stage 3 α search** — paper-exact α search, empirical confirmation pending
- **calibration-data sizing** — fresh-sample vs cycling, empirical pending
