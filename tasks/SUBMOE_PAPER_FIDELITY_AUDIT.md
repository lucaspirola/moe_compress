# Sub-MoE paper-fidelity audit (arXiv:2506.23266)

## Setup

- Paper extract: `/tmp/submoe_paperfid/paper.txt` (1865 lines)
- Paper PDF: `/tmp/submoe_paperfid/paper.pdf` (2.86 MB, downloaded 2026-05-28)
- Our impl: `max_quality/src/moe_compress/stage2/plugins/em_refine.py`
- Date: 2026-05-28
- Authority: paper text only — **no upstream code exists** (verified live via GitHub API)
  - Paper-cited URL `github.com/lliai/MoERazor` → HTTP 404 (verified `curl -o /dev/null -w "%{http_code}"` on 2026-05-28)
  - Placeholder repo `github.com/siruihan2024/Sub-MoE` → exists with only `LICENSE` (MIT) + `README.md` (size=1 KB, `default_branch=main`, `updated_at=2026-03-14T15:46:59Z`) — NO source code
- Authors: Lujun Li, Qiyuan Zhu, Jiacheng Wang, Wei Li, Hao Gu, Sirui Han, Yike Guo (HKUST + Xi'an Jiaotong + Birmingham)

## Background — what Sub-MoE actually proposes

Per the paper, Sub-MoE has **two stages**, neither of which is an "EM loop over tentative merges":

1. **Adaptive Expert Clustering** — classical K-means over expert *output vectors* `Y_i = {E_i(x_1), …, E_i(x_m)}`, with:
   - Init: k-means++ seeding (line 245)
   - Assignment: nearest centroid by cosine similarity of outputs (Eq. 2, line 232)
   - Update: arithmetic mean of assigned experts' output vectors `C_i = (1/|Q_i|) · Σ Y_j` (line 250); objective J in Eq. 3 at line 261
   - Convergence: "assignments stabilize or maximum iterations" (line 251) — no specific max_iter is published
2. **Subspace Expert Merging** — single-shot SVD + frequency-weighted V-matrix merge (Eqs. 5–8):
   - Concatenate weights of clustered experts vertically
   - Joint SVD → shared `U·Σ` + per-expert `V^(i)`
   - Frequency-weighted average of `V^(i)` (Eq. 7)
   - Reconstruct: `W_merged = U · Σ · V_merged^T` (Eq. 8)

**Crucial paper fact**: the K-means iteration operates on **output vectors** (functional similarity), is **completely separate** from the merge stage, and the merge stage is **single-shot** — there is no iterative refinement that re-clusters AFTER computing tentative merged weights. The paper does not propose an EM-style loop over tentative merges; it proposes K-means (which is itself EM-like) followed by single-shot subspace merging.

## What our plugin actually does

`em_refine.py` runs an **iterative re-assignment loop under capacitated REAM cost** in the Stage 2 v2 pipeline:

- **Init**: REAP-saliency-based centroid selection (top-K by REAP score), NOT k-means++ — done upstream in `orchestrator.py` line 344 via `select_centroids_by_reap`.
- **E-step** (`_em_refine_assignment` step 4): re-solve the assignment via `_assign_children_to_centroids` against a freshly computed REAM cost matrix (whitened Frobenius residual under post-alignment, NOT cosine similarity of outputs).
- **M-step** (`_em_compute_tentative_weights`): freq-weighted average of permutation-aligned **weight matrices** (gate/up/down) of current group members — this is essentially Sub-MoE's Eq. 7 applied to RAW WEIGHTS (not V-matrices) with no SVD.
- **Convergence**: `new_assignment == assignment` strict equality OR `em_rounds` exhausted (default `em_refinement_rounds=0`).
- **Guard**: no-op when `cost_alignment != "post"`.

## Per-surface comparison

| # | Surface | Paper §/Eq. (verified line) | Our code line | Verdict | Evidence |
|---|---|---|---|---|---|
| 1 | Iteration structure (E→M→check→repeat) | §3.2 K-means steps 1–4 (paper lines 244–263) | `em_refine.py` lines 286–331 | DEVIATE-WITH-JUSTIFICATION | Paper iterates assignment + mean-of-OUTPUTS; we iterate assignment + tentative-MERGE-of-WEIGHTS. Both follow E-step→M-step→convergence, but the M-step's quantity is fundamentally different. Plugin docstring (lines 38–60) acknowledges this is "adapted to the freq-weighted merge formula" and acknowledges the merge formula is non-linear in inputs but linear in weights — a defensible adaptation but it is an adaptation, not a re-implementation. |
| 2 | E-step assignment metric | Eq. 2 (paper line 232: cosine similarity of outputs) | `_ream_cost_matrix` via REAM δ-residual (line 298) | DEVIATE-WITH-JUSTIFICATION | Sub-MoE: `Sim(E_i, E_j) = (1/m) Σ ⟨E_i(x_l), E_j(x_l)⟩ / (‖E_i(x_l)‖·‖E_j(x_l)‖)`. Ours: REAM whitened Frobenius residual `‖(W_c − P_cm·W_m)·A^{1/2}‖_F` per gate/up/down (REAM arXiv:2604.04356, NOT Sub-MoE). The docstring correctly attributes this to REAM and labels Sub-MoE as inspiration only for the iteration shell. |
| 3 | E-step assignment shape (uncapacitated vs capacitated) | §3.2 step 2 (paper line 247): "Each expert E_j is assigned to the nearest cluster centroid" | `_assign_children_to_centroids` with `max_group_cap` (lines 317–323) | DEVIATE-WITH-JUSTIFICATION | Paper does pure nearest-centroid (no capacity constraints). Ours imposes a `max_group_cap` constraint and supports Sinkhorn/Hungarian-style solvers. This is what the existing docstring (line 8) calls the "Stage 2 v2 capacitated-assignment setting" — acknowledged D-tag. |
| 4 | M-step quantity updated | §3.2 step 3 (paper line 249–250): `C_i = (1/\|Q_i\|) Σ_{E_j ∈ Q_i} Y_j` — mean of expert OUTPUT VECTORS | `_em_compute_tentative_weights` lines 178–213: freq-weighted mean of permutation-aligned WEIGHT MATRICES | DEVIATE-WITH-JUSTIFICATION | Two independent deviations: (a) **outputs vs weights** — Sub-MoE's K-means clusters in output-vector space and centroids are arithmetic mean of outputs; ours uses the merge formula on weights as an M-step proxy; (b) **uniform vs freq-weighted** — Sub-MoE's K-means uses uniform mean (paper line 250 formula, Eq. 3 objective at line 261); ours uses frequency weighting `(freq_e / Σ freq) · perm_e(W_e)` (em_refine.py line 160). Note: Sub-MoE *does* use freq-weighting at the SVD V-matrix merge step (Eq. 7, paper line 326), so the freq-weighting itself is paper-attested but for the merge, NOT the clustering. **Docstring conflates these.** See Finding M-1 below. |
| 5 | Convergence criterion | §3.2 step 4 (paper line 251): "assignments stabilize OR maximum iterations" | `em_refine.py` lines 329–330: strict `new_assignment == assignment` OR `em_rounds` exhausted | MATCH (intent) — Nitpick on strictness | Paper does not name a tolerance. Plugin's strict equality is a valid interpretation of "stabilize"; the in-code docstring (lines 82–88) already flags the Sinkhorn flip-flop issue as future-work (MEDIUM-2 audit history) — acceptable. |
| 6 | Initialization | §3.2 step 1 (paper line 244–246): "k-means++ [16]" advanced seeding | REAP-saliency-based centroid selection (`select_centroids_by_reap` in `orchestrator.py:344`) | DEVIATE-WITH-JUSTIFICATION (UNDOCUMENTED for Sub-MoE) | Paper explicitly uses k-means++. Ours uses REAP saliency (REAP arXiv:2510.13999). The plugin docstring acknowledges REAP is the baseline (line 11) but does NOT explicitly call out that k-means++ initialization is being replaced. Note: the plugin OPERATES on a pre-initialized assignment (it's a *refiner*, not an initializer), so this deviation belongs to the wider Stage 2 v2 design, not to em_refine.py specifically — see Finding L-1 below. |
| 7 | Hyperparameters: max iterations | Paper line 251: "maximum iterations" — no published default | `em_refinement_rounds: int = 0` (em_refine.py line 372) | OPEN-QUESTION (paper-side) | Sub-MoE paper doesn't publish a max_iter default. Our default of 0 (EM disabled) is conservative and reasonable, but cannot be paper-matched. Acceptable. |
| 8 | Hyperparameters: convergence break | Paper line 251: implicit (assignments stabilize) | `em_convergence_break: bool = True` (line 373) | MATCH (intent) | The plugin's `em_convergence_break=True` is the paper's "stabilize" branch; `em_convergence_break=False` runs the full max-iter — both branches are paper-attested. |
| 9 | Frequency-weighting in EM M-step | Paper Eq. 7 (line 326) is FREQ-WEIGHTED only for V-matrix MERGE, NOT clustering centroid update; paper line 250 centroid formula `C_i = (1/\|Q_i\|) Σ Y_j` is **uniform** | `_em_compute_tentative_weights` lines 160–171: freq-weighted mean | DEVIATE-WITH-JUSTIFICATION (collapsed into Finding M-1) | Plugin uses freq-weighting at the M-step. This is a (defensible) hybrid: Sub-MoE K-means iteration style + Sub-MoE merge-step weighting formula. The current docstring does not flag this hybrid clearly. |
| 10 | Pattern H verification stamp (per RegMean precedent) | n/a (no upstream) | NO stamp present in em_refine.py | LOW — pre-existing | RegMean (`regmean_merge.py:83-89`) has a `Pattern H: clean-room re-implementation... Paper re-verification stamp: 2026-05-28` block. em_refine.py has no equivalent. Since there IS no upstream code to clean-room from for Sub-MoE, the Pattern H statement would be slightly different (paper-only + license-null + 404 acknowledgment), but the absence of any 2026-05-28 stamp is a docstring-discipline gap. |

## Findings

### Critical: 0

(No paper-eligible critical findings. The plugin **never claims** to re-implement Sub-MoE — its docstring labels Sub-MoE as "inspiration" for the iterative-refinement *shell*, and explicitly tags itself as a `D-em-refinement` deviation. There is no algorithmic divergence presented as paper-faithful that is actually not.)

### High: 0

(No documentation-vs-behavior mismatches that materially mislead a reader.)

### Medium: 1

**M-1 — Docstring conflates Sub-MoE's K-means M-step (uniform mean of outputs) with Sub-MoE's V-matrix merge step (freq-weighted)**

- Location: `em_refine.py` lines 40–46:
  > "The merge formula is non-linear in inputs but linear in weights … Sub-MoE demonstrates this iterative refinement on K-means- style merging; the Stage 2 v2 EM round adapts it to the freq-weighted merge formula used by capacitated assignment."
- Issue: This passage implies Sub-MoE's K-means iteratively re-runs *merged-centroid* assignments. Verified: Sub-MoE's K-means iterates over **expert output vectors** (Eq. 2 cosine sim, Eq. 3 uniform mean — paper lines 232, 252), not over merged weight matrices. The merge step (SVD+freq) runs **after** K-means converges and is single-shot.
- Risk: A reader inheriting this codebase will believe Sub-MoE proposes the *exact* loop we run; they may then defend a paper-faithfulness claim that isn't true.
- Remediation: tighten the language. Suggested edit:
  ```
  Inspiration: Sub-MoE (arXiv:2506.23266) applies classical K-means
  (cosine-sim of expert outputs, uniform centroid mean, k-means++ init)
  to the clustering step. Our plugin runs a *different* iterative loop
  — re-assignment under tentative WEIGHT-space merges with freq weighting
  — that borrows only the E→M→converge shape from Sub-MoE's clustering.
  The freq weighting itself is paper-attested but only at Sub-MoE's
  V-matrix merge step (Eq. 7), not at its centroid update (Eq. 3, which
  is uniform).
  ```

### Low: 2

**L-1 — Initialization deviation (REAP vs k-means++) is undocumented at the em_refine call-site**

- Location: `em_refine.py` module docstring overall; this plugin is a *refiner* and the initialization is upstream in `orchestrator.py:344`.
- Issue: The plugin docstring does not surface that the EM loop starts from REAP-saliency centroids, NOT from k-means++ seeded centroids. A standalone reader of em_refine.py cannot trace this.
- Risk: Low — anyone wanting to track down init will find `select_centroids_by_reap` quickly via the orchestrator. But the paper-fidelity story is incomplete in this file.
- Remediation: add a one-line note to the module docstring under a new "Initialization context" subhead pointing at `orchestrator.py:select_centroids_by_reap` and noting "Sub-MoE uses k-means++ on expert outputs (paper line 245); the Stage 2 v2 pipeline initializes from REAP-saliency-ranked centroids (REAP arXiv:2510.13999)."

**L-2 — Missing Pattern H verification stamp (RegMean precedent)**

- Location: `em_refine.py` module docstring (lines 1–89).
- Issue: Per RegMean's Pattern H block (`regmean_merge.py:83–89`), Stage 2 plugins that cite upstream paper code carry a "clean-room re-implementation … Paper re-verification stamp: 2026-05-28" footer. em_refine.py predates that convention and has no stamp. The deviation case is slightly novel: Sub-MoE has NO upstream code at all (paper-cited URL is 404, placeholder repo is empty), so the Pattern H statement should be "paper-only re-implementation" not "clean-room re-implementation of <upstream URL>".
- Risk: Low — the docstring already disclaims "Official code: None for this specific Sub-MoE-inspired EM loop" (lines 14–18). Missing only the dated re-verification stamp.
- Remediation: add a "Pattern H: paper-only re-implementation" block with the 2026-05-28 stamp + the two URL verifications (lliai/MoERazor 404; siruihan2024/Sub-MoE README+LICENSE only, MIT license, no source).

### Nitpick: 1

**N-1 — "M-step" / "E-step" terminology is informal**

- Location: throughout `em_refine.py` and the audit prompt's E/M-step framing.
- Note: Sub-MoE's paper never uses "EM"/"E-step"/"M-step" terminology. The paper says "K-means clustering" (paper line 242) with steps 1–4 labeled "Means Initialization / Clusters Assignment / Means Update / Convergence". Our naming convention ("EM refinement") is internally consistent with the historical "M4 / step-4T(e)" labels (em_refine.py lines 76–80) but is not a paper-traceable term. This is purely a stylistic note — no remediation needed unless future docs want to mirror the paper's naming.

## Severity counts

| Severity | Count |
|---|---|
| Critical | 0 |
| High | 0 |
| Medium | 1 |
| Low | 2 |
| Nitpick | 1 |

## Most consequential finding

**M-1** (above) — the docstring's "Sub-MoE demonstrates this iterative refinement on K-means-style merging" sentence is misleading. Sub-MoE's K-means refines on **expert outputs with uniform mean**, not on **tentatively merged weight matrices with freq weighting**. The plugin is a sound *adaptation*, but a reader treating the current docstring as paper-faithful documentation will misremember what Sub-MoE actually says.

Concrete remediation: replace `em_refine.py` lines 38–46 with the corrected docstring text in M-1 above. ~6-line surgical change. Add Pattern H stamp at the same time (L-2). Total: ~15 lines edited.

## Recommended D-tags to add (or revise)

1. **D-em-refinement** (existing, line 23): tighten the citation per M-1. The current text claims Sub-MoE demonstrates the iterative refinement on K-means-style merging; this should be reframed to say Sub-MoE demonstrates iterative K-means clustering on outputs, and our adaptation moves the iteration to weight-space tentative merges under capacitated assignment.

2. **D-em-init-reap** (new): explicit acknowledgment that EM starts from REAP-saliency centroids, not k-means++ seeded centroids (L-1). One-line cross-ref to `select_centroids_by_reap`.

3. **D-em-m-step-freq** (new — could be subsumed under D-em-refinement): the M-step uses freq weighting (`f(V_i)` from Sub-MoE Eq. 7) NOT the uniform mean from Sub-MoE Eq. 3 — a deliberate hybrid borrowing the merge-step's weighting for the iteration's centroid update.

## Recommended changes

Two surgical edits to `em_refine.py` module docstring:

1. **Fix M-1** by rewriting lines 38–46 to draw a sharper line between Sub-MoE's actual K-means (outputs, uniform, k-means++) and our adapted loop (weights, freq-weighted, REAP-initialized, capacitated assignment).
2. **Address L-1 + L-2** by adding a single "Pattern H + initialization context" block near the top of the docstring with:
   - `Pattern H` label
   - `Paper re-verification stamp: 2026-05-28`
   - Upstream URL audit: `lliai/MoERazor` → HTTP 404 (verified 2026-05-28); `siruihan2024/Sub-MoE` → MIT-licensed README+LICENSE placeholder, no source code (verified 2026-05-28 via GitHub API).
   - One-line note that initialization is REAP-saliency-based (orchestrator.py), NOT k-means++.

**No behavioral / numeric / test changes are needed.** The implementation itself is defensibly correct against the paper-as-inspiration framing — only the docstring needs a precision pass.

## Decision: fix divergences or document them?

**Document them.** All deviations are defensible adaptations to the Stage 2 v2 capacitated-assignment setting (which Sub-MoE does not address), and the plugin already labels itself a Sub-MoE-*inspired* loop (not a re-implementation). The fixer task is a pure docstring polish — spawn a **docstring fixer** for the two surgical edits above. No code fixer is warranted.
