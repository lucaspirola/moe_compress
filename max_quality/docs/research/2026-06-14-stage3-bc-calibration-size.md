# How much calibration data do Stage-3 B and C actually need? — paper- & code-grounded

Read-only research. Every paper claim is **VERIFIED** (I read the retrieved
primary text and/or the official repo `.py` myself) or **INFERRED** (my
reasoning, flagged). Code claims cite `file:line` against the live tree. Upstream
AA-SVD was cloned at the commit our docstrings cite and read firsthand.

---

## TL;DR verdict

**VERDICT: CAN-SAFELY-REDUCE-TO ~256–512 sequences for the AA-SVD B/C
covariances — but make it a *separate* knob and validate the spectrum, not a
blind cut of the global `num_sequences`.**

- The method behind both **B** (post-prune input covariance `S = X′ᵀX′`) and
  **C** (cross-covariance `X ᵀX′`, gate-only) is **AA-SVD** ("Anchored and
  Adaptive SVD", arXiv:2604.02119). The factorization our code runs is literally
  its Theorem 3.2 (`M = W·C·S⁻¹·R`, our `R = L_B`). **VERIFIED.**
- AA-SVD's **default calibration is 256 samples × 2048 tokens ≈ 0.52M tokens**
  (paper §4 L843; Appendix B.1 L1691; official repo
  `config/compression/compress.yaml:9` `num_calibration_samples: 256`,
  `config/data/*.yaml` `block_size: 2048`). **VERIFIED.**
- AA-SVD's own ablation (Figure 3, §C L1089–1097) reports the covariance
  **saturates by ~64 samples for PPL** and accuracy plateaus near **128–256**;
  256 is their chosen "perplexity has saturated, accuracy near plateau" point.
  **VERIFIED** (the saturation prose is line-cited; "512 is the largest swept
  point" is an **INFERRED** read of the Fig-3 x-axis grid, not line-cited prose).
- EoRA (the whitening method our Stage-4 shares this cov family with,
  arXiv:2410.21271) uses **128 samples × 2048** for LM and reports working "even
  with as few as **32**" (§4.1 L504–505, §4.3 L612–615). **VERIFIED.**
- **Our run uses 4000 sequences × 4096 tokens ≈ 16.4M tokens** for B/C — about
  **31× the AA-SVD default token budget** (16.4M vs 0.52M). **VERIFIED** (config
  `num_sequences: 4000`, `sequence_length: 4096`; orchestrator wires B/C off the
  *same* `cal` block, `stage3/orchestrator.py:235`).
- The 4000 is **inherited, not chosen for B/C.** It is the global calibration-set
  size sized for the **Stage-2 reservoir / Phase-B profiling** ("256 is enough
  for layer_max; production used 4000 because Phase B's reservoir benefits",
  `scripts/phase_a_diagnostic.py:201–202`). Stage-3 cov reads `cal["num_sequences"]`
  verbatim (`orchestrator.py:232–235`) — there is **no B/C-specific basis** for
  4000. **VERIFIED.**

**Bottom line:** by the source paper and its official repo, B and C are
**massively over-calibrated** (~31× the prescribed token budget; ~16× more
*sequences* than AA-SVD's 256, ~125× more than EoRA's floor of 32). Cutting the
B/C cov pass to **256–512 sequences** (512 the safe default, 256 defensible with
the spectrum check below) is squarely inside what both source papers prescribe
and ablate as saturated. The MoE per-expert split does **not** make this unsafe:
with the real Qwen3.6-35B-A3B dims (`hidden_size = 2048`, `num_experts = 256`,
`top_k = 8`), the gate/up covariance is `d = 2048` and each expert still sees
`seqs × 128` tokens, so even 256 sequences gives per-expert `n/d ≈ 16` (past
full-rank) and 512 gives `≈ 32` (solid); down_proj's `d = 512` gives 4× higher
`n/d` still. The spectrum check (§5/§6) is therefore a cheap *confirmation*, not
a risk gate — the corrected math makes the cut **safer**, not riskier.

---

## 1. The exact methods/papers behind B and C (VERIFIED)

From the code docstrings and the live factorization path:

- **B** = the post-prune student input covariance. Code calls it `B_acc` /
  `B_cov` / `_bcov_*.pt`; the docstring's "Naming bridge" states **the code `B`
  IS the paper's `S = BᵀB = E[X_postᵀX_post]`**
  (`stage3/plugins/covariance_collection.py:8–24`). It is `S = X′ᵀX′`, the
  shift (post-prune) auto-covariance.
- **C** = the cross-covariance `E[X_preᵀX_post] = XᵀX′`, gate_proj only
  (`covariance_collection.py:14–16, 531–539`; deviation D6 at `:69–95`).
- The factorization that consumes them is **Theorem 3.2 / Corollary 3.3 of
  AA-SVD** — `M = W·C·S⁻¹·L_B` (Path 1) or `M = W·L_B` (Path 3, C unavailable)
  (`aa_svd_factor.py:12–17, 350–379`; `_precompute_eigh` builds exactly the
  paper's `R = S = R Rᵀ` whitening + `C·Q·Λ^{-1/2}` rhs, `:217–263`).

**Paper:** *AA-SVD: Anchored and Adaptive SVD* — **arXiv:2604.02119**.
**VERIFIED**: retrieved source `audit/spec_compliance/01_papers/2604.02119/source.md`
(sha256 `f6383bc…`, 1875 lines, `extraction_log.txt`). Algorithm 1 (L753–768):

> "1: Set A = X, B = X′ … 2: Compute **C = ABᵀ and S = BBᵀ** 3: Factorize
> S = RRᵀ … 4: Compute **M = W C S⁻¹ R**" — our code's `B = X′ᵀX′ = S`,
> `C = XᵀX′`, `M = W·C·S⁻¹·L_B`. Exact match.

**Official code:** `atulkumarin/AA-SVD` @ `1fa1b686cd9b13a77607a676564e37d438a176c8`
(repo HEAD verified to equal the cited commit via `git ls-remote`; cloned and
read). This is the source repo our docstrings already cite.

> Note on the EoRA overlap: the **whitening** machinery (eigendecompose a `d×d`
> input Gram, project, truncated-SVD) is the EoRA construction (arXiv:2410.21271,
> §3.2). In *our* stack the original-anchor / shift split belongs to AA-SVD
> objective ④, and EoRA is the **Stage-4** consumer of the shift cov (see the
> companion doc `2026-06-13-acov-capture-point.md`). For the **Stage-3 B/C
> calibration-size question both relevant papers point the same way: a few
> hundred samples.** I cite EoRA only as a second independent data point.

---

## 2. Prescribed B/C calibration size per source (VERIFIED, with citations)

### 2.1 AA-SVD (the method our B/C implement) — 256 × 2048 ≈ 0.52M tokens

- **Paper §4, L843–844 (VERIFIED):**
  > "For calibration, we follow prior work and use **256 samples drawn from the
  > WikiText2 … training split** unless otherwise stated; our ablations show this
  > modest budget is sufficient for stable compression."
- **Paper Appendix B.1, L1691–1693 (VERIFIED):** the cost-independence argument
  states the regime explicitly —
  > "in our setting with **256 samples of length 2048, corresponding to over half
  > a million effective columns** … the covariance matrices are fixed-size d × d
  > regardless of the batch length."

  i.e. 256 × 2048 ≈ **524,288 tokens** is the *entire* B/C/S calibration budget.
- **Official repo defaults (VERIFIED, read firsthand):**
  - `config/compression/compress.yaml:9` → `num_calibration_samples: 256`
  - `config/data/wikitext2.yaml:9` / `c4.yaml:7` / `ptb.yaml:7` →
    `block_size: 2048` (the comment literally says "# sequence length")
  - The covariance is accumulated additively over these samples
    (`aa_svd/compression/compress.py` `GramAccumulator`, the
    `self.num_samples`/`current_num_samples` running sum at `:527,546`).
  - (The `nsamples = 2048` / `512` presets in `aa_svd/data/utils.py:51,82` are
    the **eval/PPL** data path — `compression_default` / `compression_v2` —
    **not** the calibration covariance, which is driven by
    `num_calibration_samples` above. Flagged so the reader doesn't mistake them.)

**AA-SVD prescribed B/C token budget: ~0.52M tokens (256 seq × 2048).**

### 2.2 EoRA (independent second source) — 128 × 2048, floor 32

- **Paper §4.1, L504–505 (VERIFIED):**
  > "We sample **128 concatenated sentences of length 2048** from the WikiText2
  > training set as the calibration set for EoRA …"
- **Paper §4.3 ablation, L612–615 (VERIFIED):**
  > "We vary the calibration size … using as few as **32 calibration samples** to
  > [compute the eigenspace still works], as shown in Table 8."
- **Abstract, L34 (VERIFIED):** results "achieved within minutes using just **64
  calibration** [samples]".

**EoRA prescribed whitening-cov budget: 128 × 2048 ≈ 0.26M tokens; floor 32 ×
2048 ≈ 0.066M.**

---

## 3. Saturation / ablation evidence (VERIFIED — this is the decisive part)

AA-SVD ran the **exact ablation the question asks for** (Figure 3 + §C):

- **L1089–1097 (VERIFIED):**
  > "Figure 3 shows the effect of **calibration set size** on WikiText2
  > perplexity … at ratios 0.8 and 0.6. **Perplexity drops sharply with the
  > first ∼64 samples and largely saturates thereafter** … indicating a small
  > calibration set is sufficient … Accuracy … continues to improve … between 64
  > and 128 samples before gradually plateauing … Our **default of 256 samples**
  > strikes a practical balance: **perplexity has saturated and accuracy is near
  > its plateau**."
- **Figure 3 x-axis (L1123–1130 — INFERRED read of the axis tick labels, not
  line-cited prose):** the swept grid is `8, 16, 32, 64, 128, 256, 512` — **512
  is the largest point they test.** There is no evidence in the paper that
  anything beyond ~256–512 helps the covariance; the curves are flat there.

**Saturation point: PPL saturates ≈64 samples; accuracy plateaus ≈128–256; 256 is
the chosen knee** (all VERIFIED, line-cited L1089–1097). **512 is the largest
swept point with no further gain shown** (INFERRED from the Fig-3 axis grid, not
prose). This is a direct, primary-source answer: **more calibration beyond a few
hundred samples does not improve the AA-SVD covariance estimate.**

EoRA's Table 8 ablation (§4.3) independently corroborates: degrades gracefully
down to 32 samples (**VERIFIED** L612–615).

---

## 4. Our current usage vs the papers — the gap (VERIFIED)

| Quantity | AA-SVD default | EoRA default | **Our Stage-3 B/C** |
|---|---|---|---|
| Sequences | 256 | 128 (floor 32) | **4000** |
| Seq length | 2048 | 2048 | **4096** |
| **Total tokens** | **~0.52M** | ~0.26M | **~16.4M** |
| Ratio vs AA-SVD | 1× | 0.5× | **~31×** |

- **Our config (VERIFIED):** `configs/qwen36_35b_a3b_30pct.yaml:71–72`
  `num_sequences: 4000`, `sequence_length: 4096`. The ablation runner passes
  `--num-sequences 4000` (`scripts/box_run_ablation.sh:26`).
- **B/C reads the global calib block verbatim (VERIFIED):**
  `stage3/orchestrator.py:231–238` —
  > "B covariance + cross-covariance: fresh calibration through both models. Use
  > **cal["num_sequences"] directly** …" → `spec = spec_from_config(cal,
  > seed_offset=2)` → `build_calibration_tensor(...)`. So B/C inherit the global
  > `num_sequences=4000`. **There is no Stage-3-cov-specific size override.**

**Is 4000 justified for B/C? NO — it is inherited (VERIFIED).** The only documented
rationale for 4000 is the Stage-2 / Phase-B reservoir, *not* Stage-3 covariance:

- `scripts/phase_a_diagnostic.py:200–202` (VERIFIED):
  > `--num-samples … default 256 … help="… **256 is enough** for layer_max;
  > **production used 4000 because Phase B's reservoir benefits**."`

That is the smoking gun: 4000 was sized for a *different* consumer (Stage-2
reservoir sampling / profiling), and Stage-3 B/C rides the same calibration
tensor purely by inheritance. **No config comment, plan, or prior finding ties
4000 to a B/C covariance-quality requirement.** (Searched `stage3/`, `scripts/`,
`configs/` — only the reservoir rationale appears.)

---

## 5. Statistical sufficiency for the AA-SVD *spectrum* (honest assessment)

The question rightly pushes past "papers use 256": AA-SVD's **rank selection +
EoRA-style whitening depend on the eigenvalue spectrum of a `d×d` Gram**, not
just the top-k directions. Does 256–512 samples estimate that spectrum well?

**Real model dims (VERIFIED — Qwen3.6-35B-A3B `config.json`, fetched from HF Hub
this round and cross-checked against the reviewer):** `hidden_size = 2048`,
`moe_intermediate_size = 512`, `num_experts = 256`, `num_experts_per_tok = 8`.
So the **gate_proj / up_proj covariance is `d = 2048`** (not the 5120 used in the
first draft — that was wrong), and the **down_proj covariance is `d = 512`**.

**The dimension/sample arithmetic (INFERRED reasoning, standard covariance
estimation theory):**

- A `d×d` sample covariance from `n` token-vectors is full-rank only once
  `n ≥ d`; its *tail* eigenvalues (which drive rank-truncation decisions) need
  `n ≫ d` — the Marčenko–Pastur edge widens like `√(d/n)`, so to pin tail
  eigenvalues to ~10% relative error you want `n/d` of order tens.
- **Whole-model (un-routed) regime:** for `d = 2048`, AA-SVD's 256×2048 ≈ 0.52M
  tokens ⇒ `n/d ≈ 256`; our 4000×4096 ≈ 16.4M ⇒ `n/d ≈ 8000`. Both deep in the
  well-conditioned regime; 4000 buys ~31× margin the spectrum doesn't need.
- **MoE per-expert split (the subtlety — now correctly quantified, INFERRED).**
  B/C are keyed **per `(layer, expert, matrix)`** (`covariance_collection.py:
  531–545`). Routing sends each expert only `top_k / n_experts = 8/256 = 1/32`
  of the token stream, so **per-expert tokens ≈ seqs × 4096 / 32 = seqs × 128**.
  For the gate/up `d = 2048`:
  - **256 seqs ⇒ ~32K tok/expert ⇒ `n/d ≈ 16`** — **past full-rank, FINE** (NOT
    the "≈6 borderline" the first draft claimed; that error came from d=5120).
  - **512 seqs ⇒ ~65K/expert ⇒ `n/d ≈ 32`** — solid, into the stable-tail band.
  - **1000 seqs ⇒ ~128K/expert ⇒ `n/d ≈ 62`** — overkill.
  - 4000 (current) ⇒ ~512K/expert ⇒ `n/d ≈ 250` — far past saturation.
  - **down_proj `d = 512`** gives **4× higher `n/d`** at every point (256 seqs ⇒
    `n/d ≈ 64`) — even less of a concern than gate/up.

**So the honest picture:** the MoE per-expert split lowers the per-key sample
count to `seqs × 128`, but with the **real `d = 2048`** even 256 sequences keeps
per-expert `n/d ≈ 16` (gate/up) / `≈ 64` (down) — past full-rank and into the
regime where AA-SVD/EoRA report saturation. The per-expert keying therefore does
**not** make a few-hundred-sequence cut unsafe; it only argues for confirming the
spectrum rather than assuming it. 4000 is over-calibrated both whole-model and
per-expert.

- Target: per-expert `n/d ≳ 20–30` (comfortably into stable-tail).
- 512 sequences already clears this for gate/up (`n/d ≈ 32`) and down (`≈ 128`);
  256 sits at `n/d ≈ 16` (gate/up) — defensible, confirm with the §6 check.

---

## 6. Recommendation & verdict

**VERDICT: CAN-SAFELY-REDUCE — to ~256–512 sequences for the Stage-3 B/C cov
pass specifically (512 the safe default; 256 defensible with the check below) —
with (a) a *dedicated* knob, not a cut of the global `num_sequences` (which the
Stage-2 reservoir needs at 4000), and (b) a one-time spectrum-equivalence check on
**both B and C** plus a single end-to-end PPL spot-check at the chosen N to
confirm rank-flip stability across the 40×~200 expert grid.**

Reasoning:

1. **Paper-faithful floor is far below 4000.** AA-SVD prescribes and *ablates*
   256 (saturated by 64–256, ceiling tested 512); EoRA 128 (floor 32). 4000×4096
   is ~31× the AA-SVD token budget — **inherited from the Stage-2 reservoir, with
   no B/C-specific justification** (§4). Reducing the B/C pass is paper-faithful,
   not a deviation.
2. **The closed-form result is calibration-size-robust by construction.** AA-SVD
   Appendix B.1 (L1690–1693, VERIFIED): the solution "operates only on the
   covariance matrices … fixed-size d×d regardless of the batch length" and its
   "cost is independent of the number of calibration tokens" (L717–719). The
   *output* (rank-k factor) is a function of the converged `S`,`C` — once the
   spectrum has saturated (§3), more tokens change the factor negligibly. This is
   why a smaller B/C pass should be **quality-neutral**, not just cheaper.
3. **256 vs 512 — the per-expert math (§5) now favors the lower band.** With the
   real `d = 2048` (gate/up) the per-expert sample count is `seqs × 128`, so
   **256 seqs ⇒ `n/d ≈ 16`** (past full-rank; defensible) and **512 seqs ⇒
   `n/d ≈ 32`** (solid into the stable-tail the AA-SVD rank threshold —
   `_precompute_eigh` noise floor, `aa_svd_factor.py:221–227` — keys on). down_proj
   `d = 512` is 4× safer still. So **512 is the safe default, 256 the aggressive-
   but-defensible floor** — and even 512 cuts the dual-forward ~8×.
4. **Mechanism for a clean cut:** add a Stage-3-cov-specific
   `stage3_svd.cov_num_sequences` (slice the first N rows of the calib tensor in
   `orchestrator.py:235–238`, or a dedicated `spec_from_config` override),
   defaulting to the global value for back-compat. This is a ~5-line change that
   leaves Stage-2's 4000-seq reservoir untouched and the golden byte-identical
   when the override is absent.

**This is NOT "KEEP-CURRENT"** (current is demonstrably over-calibrated vs both
source papers) **and NOT a pure "NEEDS-EMPIRICAL-TEST"** (the papers already ran
the saturation ablation; the per-expert math now confirms 256–512 is past
full-rank). It is **CAN-SAFELY-REDUCE**, with the *exact* number (256 vs 512) the
only open knob — pin it with a cheap confirmation, in two parts:

1. **Spectrum-equivalence on BOTH B and C** (not B alone). Collect at the
   candidate N (256 and/or 512) and at 4000 for a couple of layers and compare,
   per-expert: (i) retained-rank counts `r_eff` and (ii) the top-`d/2` eigenvalue
   spectrum. **Include C** as well as B — C is the teacher×student position-join
   (`covariance_collection.py:754–795`), and the join can keep *fewer* matched
   rows per expert than B's own routed count, so C's per-expert `n` may be lower
   than B's and is the binding constraint. Agreement within the storage-dtype
   noise floor ⇒ that N is safe.
2. **One end-to-end PPL spot-check at the chosen N.** Per-layer `r_eff` rank flips
   (`aa_svd_factor.py:221–227`) are individually tiny but **compound across
   40 layers × ~200 experts**, so a per-layer spectrum match does not by itself
   guarantee the assembled model is unchanged. Run a single WikiText2 PPL eval of
   the fully-factored model at N vs 4000 before committing the cut to the full
   ablation. If PPL matches, ship N; if it drifts, step up (256→512→1000).

Both parts together are a ≤2-layer dual-forward plus one PPL eval — a small
fraction of the cost the cut saves.

**Expected speedup:** the Stage-3 dual-forward cost is linear in sequences, so
4000→512 is ~**8× faster** B/C collection and 4000→256 ~**16×**. Quality-neutral
by the paper's saturation result and the corrected per-expert `n/d`, modulo the
two-part confirmation above.

---

## 7. VERIFIED vs INFERRED ledger

**VERIFIED — primary papers (read retrieved text, line-cited):**
- AA-SVD default 256 samples (§4 L843); "256 samples of length 2048, over half a
  million columns" (App. B.1 L1691); cost independent of #tokens (L717–719);
  saturation ablation "sharp drop by ~64, saturates thereafter; 256 is the knee"
  (§C L1089–1097); Algorithm 1 `C=ABᵀ, S=BBᵀ, M=W C S⁻¹ R` (L753–768).
- EoRA 128×2048 calibration (§4.1 L504–505); floor 32 (§4.3 L612–615); 64 in
  abstract (L34).

**VERIFIED — target model dims (Qwen3.6-35B-A3B `config.json`, fetched from HF
Hub this round, cross-checked vs reviewer):** `hidden_size = 2048`,
`moe_intermediate_size = 512`, `num_experts = 256`, `num_experts_per_tok = 8`.
⇒ gate/up cov `d = 2048`, down_proj cov `d = 512`. (The first draft's `d = 5120`
was wrong; corrected throughout §5/§6.)

**VERIFIED — official upstream code (cloned `atulkumarin/AA-SVD @ 1fa1b686cd`,
HEAD == cited commit, read firsthand):**
- `config/compression/compress.yaml:9` `num_calibration_samples: 256`;
  `config/data/{wikitext2,c4,ptb}.yaml` `block_size: 2048`;
  `aa_svd/data/iterable_text_dataset.py:17` `block_size=512` default (overridden
  to 2048 by the data configs); additive Gram accumulation
  `compress.py:527,546`. The `data/utils.py:51,82` `nsamples 2048/512` are the
  eval-data presets, not the calibration cov (flagged).

**VERIFIED — repo under review (re-opened each cite):**
- B = paper-`S` naming bridge `covariance_collection.py:8–24`; C gate-only
  `:531–539`, D6 `:69–95`; per-`(layer,expert,matrix)` keying `:531–545`; C's
  teacher×student position-join `:754–795`; Theorem-3.2 factor
  `aa_svd_factor.py:217–263, 350–379`; noise-floor rank threshold `:221–227`.

**FLAGGED — codebase docstring title error (separate cleanup, not blocking):**
`covariance_collection.py:5` and `aa_svd_factor.py:5` label arXiv:2604.02119 as
*"Activation-Aware SVD with Cross-Covariance Calibration"*. The paper's real
title is **"Anchored and Adaptive SVD"** (VERIFIED, retrieved source +
the four-objective taxonomy at L370–385). This doc uses the correct title; the
**code comments are wrong** and should be corrected in a separate docstring pass.
- B/C inherit global `num_sequences` `orchestrator.py:231–238` (no cov-specific
  override); config `qwen36_35b_a3b_30pct.yaml:71–72` (4000 × 4096); runner
  `box_run_ablation.sh:26`; **4000-is-for-Phase-B-reservoir rationale**
  `phase_a_diagnostic.py:200–202`.

**INFERRED (my reasoning, NOT a paper quote):**
- The `n/d` / Marčenko–Pastur tail-eigenvalue argument and per-expert token
  arithmetic in §5, now using the **VERIFIED** dims (`d=2048` gate/up, `d=512`
  down, `n_experts=256`, `top_k=8`): per-expert tokens ≈ `seqs × 128`, so 256
  seqs ⇒ gate/up `n/d ≈ 16`, 512 ⇒ `≈ 32`. The Marčenko–Pastur/tail-saturation
  *reasoning* is inferred; the dims feeding it are verified. (Post-prune
  `n_experts` for the merge arms may be < 256, which only *raises* per-expert
  `n/d` — strengthening the cut.)
- The "256–512 is the safe band, 512 default" and the two-part confirmation
  (spectrum on B+C plus one end-to-end PPL) — mechanism-based; not yet run on this
  model. Picking 256 vs 512 is the only remaining empirical knob.
- The ~5-line `cov_num_sequences` implementation sketch (architectural, not
  implemented).

**Could NOT verify from primary sources:** an AA-SVD/EoRA ablation specifically on
a **MoE per-expert** covariance (both papers ablate dense models). The per-expert
caveat in §5 is therefore *the* thing to empirically pin before committing the cut
— flagged honestly as the load-bearing unknown.
