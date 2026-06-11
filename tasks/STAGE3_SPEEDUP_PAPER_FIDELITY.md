# Stage 3 Speed-Up — PAPER-FIDELITY Audit

Audits whether the proposed Stage-3 speedups (A1/A2/A4/A6, from
`tasks/STAGE3_SPEEDUP_ANALYSIS.md` on `analysis/stage3-speedup`) change **what**
the cited papers' algorithms must compute, or only **how fast** it is computed.

Read-only on code. Every paper quote below was extracted from the **actual
PDF** via `curl https://arxiv.org/pdf/<id> | pdftotext | grep/sed` and the line
numbers are from that `pdftotext` output (verified, not from memory/WebFetch).

## Papers in scope (verified from code docstrings + the real PDFs)

| arXiv | Name | What Stage-3 cites it for | Touched by |
|-------|------|---------------------------|-----------|
| **2604.02119** | AA-SVD: Activation-Aware SVD | Theorem 3.2 cross-cov `C = X_pre^T X_post`; Algorithm 1 (CompressLayer); Algorithm 2 §3.3 (block-wise + refinement) | A1, A4, A6 |
| **2604.01609** | Swift-SVD / Swift-SVD+ | §3.2.2 + Algorithm 2 — dynamic-rank candidate **grid search**, select by validation PPL | A2 |

Code cites verified with `grep -n`:
`aa_svd_factor.py:1,6,12,166,222,244` (Thm 3.2 / arXiv:2604.02119);
`covariance_collection.py:1,6,8,197,300-304,806` (`C = X_pre^T X_post`);
`swift_svd_alpha.py:6,7,187,576,695,1053` (Swift-SVD §3.2.2 / Alg 2).

### Verified paper definitions relied on below

**Theorem 3.2** (2604.02119 PDF, pdftotext lines 523-555, verbatim):

> "Theorem 3.2. Let W ∈ R^{m×n} be a fixed weight matrix and A, B ∈ R^{n×l} be
> any two matrices. … an optimal solution … is W'⋆ = SVDk(W A B^⊤ (B B^⊤)^{-1}
> L_B) L_B^{-1}"

and (lines 575-577):

> "The solution operates only on the covariance matrices XX'^⊤ and X'X'^⊤, not
> on raw activations, so its cost is independent of the number of calibration
> tokens."

→ The factorization is a **pure function of the two covariance matrices**
`C = X X'^⊤` (cross) and `S = X' X'^⊤` (B-cov). It does **not** depend on token
batching, collection order, or how many layers were instrumented at once. This
is the load-bearing fact for A1/A4/A6.

**Algorithm 1 / Algorithm 2** (2604.02119 PDF, lines 597-616, verbatim):

> Alg 1 line 1: "Set A = X, B = X'  {shift-aware: A = B = X'; input-aware: A =
> B = X}"; line 2: "Compute C = A B^⊤ and S = B B^⊤".
> Alg 2 line 5: "Collect Xj from Li and X'j from L'i by forward pass up to
> layer j"; line 10: "Update inputs for next block: X ← Li(X), X' ← L'i(X')".

→ X = activation in the **original** model, X' = activation in the **compressed**
model. In the repo's naming `X_pre = X` (teacher/original), `X_post = X'`
(student/pruned). (Pre-existing deviation, NOT under audit: the repo collects X'
from a *fully*-pruned student over full depth in a single two-model snapshot,
rather than Alg 2's progressive block-by-block `X' ← L'i(X')`. Documented as
legacy "D9". A1 does not change this; it is the backdrop the audit assumes.)

**Swift-SVD grid search** (2604.01609 PDF, lines 358-361, verbatim):

> "Swift-SVD generates a set of candidate dynamic rank allocation schemes …
> A lightweight grid search is then performed over these candidates—each model
> is compressed using the optimal solution in a) and evaluated on a validation
> set—to select the configuration that yields the best end-to-end performance."

→ The paper *defines the output* as `argmin_candidate (validation PPL)`. Any
optimization that does not change the candidate set, the per-candidate compressed
model, or the validation metric is fidelity-preserving.

---

## Per-optimization verdicts

### A1 — single-pass / windowed covariance collection — **FIDELITY-RISK / NEEDS CARE**

**Claim:** hook all (or a window of G) MoE layers in ONE dual-forward instead of
N separate full-depth passes; "byte-identical" because the Gram is keyed by
`(layer,expert,matrix)` and is a pure additive sum.

**What the paper fixes.** Theorem 3.2 needs, *per layer L*, the pair
`(X_pre^{(L)}, X_post^{(L)})` = (original-model input to L's experts,
pruned-model input to L's experts), reduced into `C^{(L)} = X_pre^T X_post` and
`S^{(L)} = X_post^T X_post`. The *value* of these covariances is all that matters
(PDF lines 575-577); collection order is irrelevant **if the captured
activations are the same numbers**.

**Accumulator side — byte-identical (verified).** `update_cross` is
`cur.add_(cross_f32)` keyed by `(layer,expert,matrix)`
(`activation_hooks.py:1042-1052`); `update` (B-cov) is the analogous additive sum
(`:1016-1018`). Teacher/student token matching uses a **batch-local**
`_teacher_hidden` dict that is `.clear()`-ed every batch
(`covariance_collection.py:469,492`), so there is no cross-layer or cross-batch
token-id collision when many layers are hooked at once. Per key, the additive
order over (batch, token) is identical whether 1 or N layers are hooked, and
`finalize_layer` casts to storage-dtype once per key in both schemes
(`activation_hooks.py:1054-1070`). **At the accumulator level the doc's
byte-identical claim holds.**

**The real risk — instrumentation changes the forward numerics, and A1 changes
WHICH layers are instrumented.** `instrument_experts` does not merely observe; it
**replaces** the experts' `forward` with a per-expert Python loop
(`activation_hooks.py:1491-1521`) that recomputes gate/up/down via `F.linear`
and re-assembles the output with `index_add_` (`:1518-1519`). This is
*mathematically* the same map as the native fused/grouped GEMM but with a
**different fp reduction order**, so its output is equal only up to fp rounding.

- **Current N-pass design:** when collecting layer L, *only* layer L is
  instrumented (`covariance_collection.py:474-486`). Layers `0..L-1` — whose
  outputs flow through the residual stream into L's expert input — run the
  **native fused** path. So `X_pre^{(L)}`/`X_post^{(L)}` are captured downstream
  of *native* upstream MoE forwards.
- **A1 (all/windowed layers hooked):** layers `0..L-1` now also run the
  **Python-loop** path. The residual-stream hidden state arriving at L shifts by
  fp noise relative to the N-pass design ⇒ `X_pre^{(L)}`/`X_post^{(L)}` (and thus
  `C^{(L)}`, `S^{(L)}`) are **not byte-identical** to today's outputs.

This is *not* a paper-definition break — both schemes still compute
`C = X_pre^T X_post` for the correct per-layer pre/post pair (no cross-layer
contamination: the keying + per-batch teacher clear guarantee each captured row
is matched to its own layer and token). It is a **numerical-reproducibility**
risk: A1's covariances differ from the current pipeline's at the fp-rounding
level, which can perturb the downstream `eigh`/rank decision (the project's own
notes flag fp16-vs-bf16 cov storage as able to *flip ranks* — i.e. this pipeline
is rank-sensitive to small cov perturbations).

**Per-layer pre/post pairing is preserved (the question asked).** Yes: hooking
many layers simultaneously still captures, for each layer L, that layer's own
expert input on each model. `_teacher_hidden` is keyed by `layer` then `tidx`
(`covariance_collection.py:354-360`) and read back under the same `li`
(`:373-375`); the teacher forward fully completes before the student forward
within a batch (`:493-499`), and the dict is cleared per batch. Hooking more
layers does not let layer L's teacher rows leak into layer L′'s cross term. The
N-pass design exists for **peak-RAM**, not for pre/post correctness
(`covariance_collection.py:438-446` documents it as a memory tradeoff, "Wall-clock
cost is ~N× the simultaneous design"). So the *pairing* is safe; the *fp value*
is the caveat.

**Condition to make A1 safe.**
1. Either keep upstream layers on the **native** forward (capture-only hook that
   does not replace the expert forward with the Python loop — i.e. land A7's
   capture-only path first, then the only instrumented-vs-native delta is gone
   and A1 becomes byte-identical), **or**
2. Accept A1 as equal only **within fp tolerance** (which is all the analysis doc
   itself claims — "within fp tolerance", not bit-exact) and **re-validate the
   selected ranks** on one real arm (the rank map is the sensitive output). Do
   NOT advertise A1 as bit-identical to the current artifacts.

Windowing (G layers/pass) has the *same* risk for the same reason; smaller G does
not remove it (any pass with ≥2 instrumented layers changes upstream numerics for
the later one).

---

### A2 — cache `eigh`/rhs across the 11 alpha candidates — **PAPER-FAITHFUL**

**Claim:** `_precompute_eigh(B,A,C)` is rank/alpha-independent; cache it once per
`(layer,expert)` and reuse across all 11 PPL candidates.

**Verified.** `_precompute_eigh` (`aa_svd_factor.py:187-262`) consumes only the
covariances: it eigendecomposes `B` (`:207-209`), explicitly `del A` so A cannot
influence the factorization (`:230`), and builds `rhs` from `C` and the B-eigen
basis (`:242-253`). Its return `_EighDecomp(eigvals,eigvecs,inv_sqrt,rhs,
rhs_pinv,r_eff)` (`:255-262`) contains **no `k`/alpha term**. `k` enters *only*
in `_aa_svd_precomputed` (`:265+`, called at `swift_svd_alpha.py:462-464`).
Across the 11 candidates B/A/C are the identical spilled covariances
(`swift_svd_alpha.py:438-442` reload the same `B_acc`/`A_cov`/`C_acc`), so the
cached decomp is bit-identical for every candidate, and the per-candidate factors
`U_k,V_k` are therefore bit-identical to recomputing the eigh each time.

**Paper alignment.** Swift-SVD §3.2.2 defines selection as `argmin` validation
PPL over the candidate set (PDF lines 358-361, quoted above). A2 changes neither
the candidate set, the per-candidate compressed model, nor the PPL metric — it
only removes a redundant recomputation of an alpha-invariant quantity. The
repo *already* applies exactly this caching on the spectral-proxy path
(`grouped_svs_cache`); the PPL path simply never got it. **Same selected
configuration, same factors. Pure speed.**

---

### A4 — vectorize the per-token cross-cov Python loop — **PAPER-FAITHFUL (with one alignment check)**

**Claim:** replace the `{tidx: row}` dict-build + per-row `torch.stack`
(`covariance_collection.py:359-360, 397-403`) with a single `index_select` of the
teacher tensor by the student's `token_idx`. Pure reshape; the matmul
`cross = X_pre.T @ X_post` (`:411`) is unchanged.

**Why faithful.** Theorem 3.2 only needs the value `C = X_pre^T X_post` over the
matched token positions. The current loop builds `X_pre` by, per student token
`tidx`, looking up `teacher_store[tidx]` and stacking (`:397-403`); an
`index_select(0, token_idx)` over a dense `[n_pos, d]` teacher tensor yields the
**same rows in the same order**, hence the same `X_pre`, hence the same matmul
and the same accumulated Gram. No paper quantity changes — the analysis doc
correctly calls it "pure reshaping, identical fp result."

**The one detail to honor (NEEDS CARE within the refactor, not a paper break).**
Correctness hinges on **token-position alignment** surviving the dict→dense-tensor
change. Two invariants the vectorized version must preserve, both already
load-bearing in the current code:
1. **Missing-token skip.** The current loop appends a pair *only if*
   `tidx in teacher_store` (`:398`) — i.e. tokens the student routes to an expert
   but the teacher did not place at that position are dropped. A dense
   `index_select` must replicate this membership filter (mask to the intersection),
   or it would fabricate rows the current code excludes → different `C`.
2. **Per-row device coercion** (`:399` `.to(tgt_device)`) and the
   `n_tokens=len(pre_vecs)` count fed to `update_cross` (`:418`) must equal the
   number of *matched* rows, not the raw `token_idx` length.

If those two are honored, A4 is bit-identical. Flag as paper-faithful; the
"care" is an implementation invariant, not a fidelity trade.

---

### A6 — bump covariance `batch_size` on the sharded run — **PAPER-FAITHFUL**

**Claim:** the Gram is `sum over tokens of outer products`; batch grouping does
not change the sum (associative), so a larger `batch_size` yields the identical
covariance, just fewer/larger kernel launches.

**Verified.** Theorem 3.2 / PDF lines 575-577: the solution "operates only on the
covariance matrices … its cost is independent of the number of calibration
tokens" — the covariance is a per-token sum, and `batch_size` only regroups which
tokens are summed together per forward. The accumulator adds per-batch partials
into the same `(layer,expert,matrix)` key (`activation_hooks.py:1016-1018`,
`1044-1051`); B-cov forward runs under `no_grad` (`covariance_collection.py:495,
498`), so there is no batch-coupled state (no batchnorm, no dropout). The cov
forward is `no_grad` and the per-token contribution is independent of its batch
neighbours.

**The fp caveat is benign and already accepted.** Changing `batch_size` changes
the **summation order** of fp32 partials within a key (more tokens per matmul =
one bigger GEMM reduction vs several smaller ones), so the result is equal up to
fp rounding, *not* literally bit-identical — exactly the same class of fp-order
nondeterminism the pipeline already tolerates between runs. This does not change
any paper-defined quantity. The orchestrator already anticipates this bump as a
MEASURED VRAM decision (`orchestrator.py:197-206`). **Paper-faithful; treat as
fp-tolerance equal, not bit-exact.**

---

## Bottom line

| ID | Verdict | One-line reason |
|----|---------|-----------------|
| **A1** | **FIDELITY-RISK / NEEDS CARE** | Pairing & accumulator are correct, but hooking upstream layers swaps their forward to the Python expert-loop, perturbing the residual stream → cov values differ from today's at fp level (rank-sensitive). Make safe by capturing-only (A7 first) OR re-validating ranks and not claiming bit-identical. |
| **A2** | **PAPER-FAITHFUL** | `_precompute_eigh` is alpha/k-independent; caching it across the 11 candidates gives bit-identical factors and the same argmin-PPL configuration the paper defines. |
| **A4** | **PAPER-FAITHFUL** (honor 2 invariants) | `index_select` reproduces the same matched `X_pre` rows → same `C`, **iff** the missing-token membership filter and matched-row `n_tokens` count are preserved. |
| **A6** | **PAPER-FAITHFUL** | Covariance is a per-token sum (Thm 3.2: cost independent of token count); larger batch only regroups the sum. fp-order equal, not bit-exact — already accepted. |

**Paper-safe now:** A2 (bit-identical), A4 (with the two invariants), A6 (fp-tol).
**Needs care:** A1 — safe as a *fp-tolerance* optimization, NOT as a
bit-identical one; the clean fix is to land A7's capture-only hook so upstream
layers keep the native forward, after which A1 also becomes byte-identical.

**Single biggest fidelity risk:** A1's switch of upstream MoE layers from the
native fused forward to the per-expert Python-loop forward
(`activation_hooks.py:1491-1521`). It does not break Theorem 3.2's per-layer
pre/post pairing (that is preserved), but it perturbs the captured covariances at
the fp level in a pipeline whose own rank decisions are documented to flip under
small covariance perturbations — so A1 must not be shipped as "byte-identical"
without either the capture-only hook or a rank re-validation on a real arm.
