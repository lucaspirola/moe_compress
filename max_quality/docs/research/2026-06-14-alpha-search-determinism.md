# Stage-3 AA-SVD α-search cross-host non-determinism — root cause + fix design

**Status:** research / design only (no implementation). A reviewer checks this,
then a separate impl loop builds it.
**Branch:** `research/alpha-search-determinism`
**Repo HEAD at investigation:** `d967b16` (max_quality)
**Golden minted at:** `588ec5e` ("test(stage3): mint Tier-2 additive [0,0.5,1] alpha-variant golden")

All file:line citations are against the working tree at `d967b16` unless noted.

---

## 0. TL;DR

* **Root cause (one-liner):** the Stage-3 Swift-SVD+ α-path whitens with an
  **`eigh`-based factor** `L_A = eigvecs[:, keep] · √eigvals[keep]`
  (`swift_svd_alpha.py:1205-1208` and `:1403-1406`) gated by a **relative keep
  threshold** `eigvals_a > eigvals_a.max() * 1e-6` (`:1206`, `:1404`). That
  threshold is a **hard discrete boundary**: an eigenvalue near `1e-6·λ_max` is
  kept on one LAPACK/BLAS build and dropped on another, changing `L_A`'s column
  count → a discrete jump in the spectra `svdvals(W @ L_A)`, the blending scores,
  the α selection, AND the integer rank rounding ⇒ **non-reproducible across
  hosts**. (Eigenvector sign/rotation choice is NOT the cause — `svdvals(W @ L)`
  depends only on `L·Lᵀ`; see §2.1.) The fix replaces the eigh-factor with a
  **full-rank Cholesky factor** `L_C = cholesky(A64 + jitter)` (no threshold,
  unique), preserving the `W @ L_C` order — NOT the symmetric-sqrt / `L @ W.T`
  form (§3.1).
* **Two failure modes, same source:** (a) a **near-tied α** where the selection
  `if err < best_err` flips the winner on last-bit jitter
  (`swift_svd_alpha.py:1326`, `:1337`); (b) **score-value drift** that shifts
  even well-separated allocations across the integer-rounding boundary in
  `_redistribute_ranks_swift_svd_plus` (`:1449-1452`) — this one is NOT fixed by
  any α-selection tie-break.
* **Is the real 35B run at risk?** **YES, but indirectly / mildly for the
  default config, and the test is the canary.** The real config
  (`qwen36_35b_a3b_reap_faithful.yaml:325-334`) runs `validation_samples: 512` +
  `per_group_type: true`. Under `per_group_type=True` the validation-PPL α is
  **discarded** (H-α2, confirmed in code at `swift_svd_alpha.py:1629-1635`) and
  the final `alpha_by_type` comes from the **same spectral proxy** that the tiny
  test exercises. So the real run's rank_map is decided by the exact unstable
  code path. On the real 35B the α landscape is mostly well-separated (strong
  signal), so α flips are rarer — but the **score-value drift → rank-rounding
  boundary** crossings are NOT rare and are the same mechanism that shuffles the
  tiny model's gate_proj ranks despite an unchanged α.
* **Where the fix goes:** make the α-path whitening deterministic by switching
  the **factor** from `eigh` to `L_C = cholesky(A64 + jitter)` while KEEPING the
  `svdvals(W @ L_C)` order (LOAD-BEARING: `L_Cᵀ L_C ≠ A`, so `svdvals(L_C @ W.T)`
  is a different, paper-incorrect quantity — §3.1), in BOTH spectral producers
  (`_swift_svd_plus_alpha_search` `:1199-1222` and
  `_redistribute_ranks_swift_svd_plus` `:1397-1418`). Apply the IDENTICAL swap to
  both so the Tier-1 `torch.equal` cache precondition (`test_stage3_tier1.py:147`)
  holds. Add an explicit ε-tolerant α tie-break in the two proxy selection loops
  (`:1320-1329`, `:1332-1340`) as defence-in-depth. The golden MUST be
  regenerated after.

---

## 1. The exact α-selection code (file:line + quotes)

### 1.1 Two α paths; only the *proxy* drives the deployed ranks

`SwiftSvdAlphaPlugin.select_alpha` (`swift_svd_alpha.py:1532-1658`) dispatches on
`alpha_grid` length and `validation_samples`:

```
1569  if alpha_grid and len(alpha_grid) > 1:
1578      if validation_samples > 0:
...               best_global_alpha = _swift_svd_plus_alpha_search_validation(...)   # PPL grid
1629          if per_group_type:
1631              alpha_by_type, grouped_svs_cache = _swift_svd_plus_alpha_search(   # SPECTRAL PROXY
1632                  moe_layers, group_stats, ranks, alpha_grid,
1633                  per_group_type=True, A_cov=A_cov, return_svs=True,
1634              )
1635              cache_was_built = True
...
1639      else:                                                                       # validation_samples == 0
1641          alpha_by_type, grouped_svs_cache = _swift_svd_plus_alpha_search(...)     # SPECTRAL PROXY
1646      per_expert_ranks = _redistribute_ranks_swift_svd_plus(...)                   # uses alpha_by_type
```

The crucial fact (memory note **H-α2**, **confirmed true in code**): when
`per_group_type=True` (the production default, `yaml:334`), `best_global_alpha`
from the WikiText-2 PPL grid is computed (line ~1612) but **never flows into
`alpha_by_type`** — branch (i) overwrites it at `:1631`. The deployed
`alpha_by_type` is decided **only** by the spectral proxy
`_swift_svd_plus_alpha_search`. The docstring states this verbatim at
`swift_svd_alpha.py:113-118` and `:130-138` ("its winning α is discarded for the
factoring step").

⇒ **The validation-PPL `_argmin_alpha` (`:697-719`) is NOT the bug site for the
default config.** The brief's hypothesised "round the PPL metric before argmin"
would not fix the failing test (which runs `validation_samples=0`, branch iii)
nor the default 35B config (branch i, proxy decides).

### 1.2 The proxy's α selection (the real selection site)

`_swift_svd_plus_alpha_search` (`swift_svd_alpha.py:1129-1342`), `per_group_type`
branch:

```
1320  best_alphas: dict[str, float] = {}
1321  for name in MATRIX_NAMES:
1322      best_alpha = 0.5
1323      best_err = float("inf")
1324      for alpha in alpha_grid:
1325          err = _evaluate_alpha(name, alpha)
1326          if err < best_err:
1327              best_err = err
1328              best_alpha = alpha
1329      best_alphas[name] = best_alpha
```

and the `per_group_type=False` branch (`:1332-1340`) uses the identical
`if err < best_err` rule. **Tie-break = raw float ordering with strict `<`**:
the FIRST (lowest-grid-index) α wins a tie. There is **no rounding and no
explicit ε-tolerance** — the winner depends on the last bits of `err`.

`err` is `_evaluate_alpha(name, alpha)` (`:1224-1309`): a **spectral** quantity
(`total_err = Σ tail-energy at the allocated per-expert ranks`), computed from
`grouped_svs` — NOT a forward-pass PPL. So the fp non-determinism that drives it
is the **`svdvals`/`eigh` linear-algebra**, not a model forward.

### 1.3 The deployed redistribute also rounds on drifting scores

`_redistribute_ranks_swift_svd_plus` (`:1345-1476`) recomputes the same scores
and does integer allocation:

```
1449  per_e = [
1450      max(rank_floor, min(cap, rank_floor + int(math.floor(flexible_pool * (sc / total_score)))))
1451      for sc in scores
1452  ]
1454  diff = total_group_rank - sum(per_e)
1455  if diff != 0:
1456      order = sorted(range(gs.n_experts), key=lambda i: scores[i], reverse=(diff > 0))
```

`int(math.floor(...))` is a hard boundary: an arbitrarily small score change can
move `flexible_pool * share` across an integer and change a rank by 1. The
`sorted(..., key=scores[i])` residual loop is ALSO score-ordering-dependent.
This is why the tiny model's `gate_proj` ranks differ cross-host **even though
gate_proj's α is identical (1.0 on both hosts)** — see §2.3.

---

## 2. Where the fp non-determinism enters (measured)

### 2.1 The metric is spectral, not a forward PPL

For the failing test and the default config's deployed-α path, the per-α metric
is `_evaluate_alpha` → tail spectral energy over `grouped_svs`. `grouped_svs` is
built once at `:1190-1222`:

```
1199  W = banks[name].get(e).detach().to(device="cpu", dtype=torch.float64)
1201  A = _cov_lookup(A_cov, li, e, name) if A_cov else None
1203      A64 = A.to(device="cpu", dtype=torch.float64)
1204      A64 = 0.5 * (A64 + A64.T)
1205      eigvals_a, eigvecs_a = torch.linalg.eigh(A64)
1206      keep_a = eigvals_a > eigvals_a.max() * 1e-6
1208      L_A = eigvecs_a[:, keep_a] * eigvals_a[keep_a].clamp_min(1e-12).sqrt().unsqueeze(0)
1209      M_A = W @ L_A
1210      svs = torch.linalg.svdvals(M_A)
```

The same `eigh`-factor block is duplicated in the deployed recompute at
`:1400-1407`. **This `eigh` whitening is the host-unstable component**, and the
instability is **discrete**, not continuous:

1. **Keep-threshold membership (discrete) — THE DOMINANT INSTABILITY.**
   `keep_a = eigvals_a > eigvals_a.max() * 1e-6` (`:1206`, `:1404`) is a HARD
   boundary. An eigenvalue sitting near `1e-6 · λ_max` can be kept on one
   LAPACK/BLAS build and dropped on another (the eigenvalue itself differs by
   round-off across builds, enough to cross the threshold), changing `L_A`'s
   **column count**. That is a discrete jump in `L_A` → a discrete jump in
   `svdvals(W @ L_A)` → a score change far larger than fp64 round-off, which then
   crosses the integer rank boundary (§2.3). This is the mechanism that produces
   the observed cross-host divergence.
2. **Eigenbasis choice is NOT the cause (correction).** It is tempting to blame
   `eigh`'s sign/rotation ambiguity in near-degenerate subspaces, but
   `svdvals(W @ L)` depends ONLY on `L · Lᵀ` (the symmetric Gram), not on the
   eigenbasis representative: a sign-flip or in-subspace rotation of `eigvecs_a`
   leaves `L_A · L_Aᵀ` — and hence the singular values — invariant. The reviewer
   measured `~5e-15` agreement under deliberate sign/rotation perturbation, so
   eigenvector-mixing is a red herring. The instability is entirely the discrete
   `keep_a` membership flip above.

**Why Cholesky removes it:** `cholesky(A64 + jitter)` produces a **full-rank**
factor with **no keep-threshold** and a **unique** lower-triangular form
(positive diagonal). There is no discrete membership boundary to flip across
builds, so `svdvals(W @ L_C)` is cross-host reproducible. The fix's justification
is precisely *discrete-threshold removal + Cholesky full-rank uniqueness* — NOT
any eigenvector-mixing argument.

By contrast the **stable** non-α D-Rank path
(`d_rank_allocate.py:_group_stat`, ~`:362-396`) whitens with **Cholesky**:

```
jitter = 1e-6 * A64.diag().mean().clamp_min(1e-12) * torch.eye(...)
L_A = torch.linalg.cholesky(A64 + jitter)
...
s = torch.linalg.svdvals(L_A @ W.T)
```

Cholesky of an SPD matrix is **unique** (lower-triangular, positive diagonal) and
has no keep-threshold, so it is cross-host reproducible. This is the asymmetry
that explains why `test_stage3_rank_map_byte_identical` (non-α) passes
byte-identically on this host while the α-variant fails. **Note:** D-Rank's
`L_A @ W.T` order shown above is correct *for D-Rank's own objective*; the
α-path's fix borrows only the Cholesky *factor*, NOT this order — it keeps
`W @ L_C` (§3.1, LOAD-BEARING), since the two orders are different quantities for
a triangular factor.

### 2.2 Why near-tied on the tiny model but (mostly) not on the 35B

On the tiny synthetic model the spectra are nearly flat (random small W, tiny
A_cov), so for several (name) the three α give `err` values that are **equal to
fp64 round-off**. Measured on this RTX 5080 host (instrumented run, then
reverted):

```
PROBE name=up_proj   alpha=0.0 err=0.5768380094804213
PROBE name=up_proj   alpha=0.5 err=0.5768380094804213
PROBE name=up_proj   alpha=1.0 err=0.5768380094804213     # all three EQUAL to printed precision
PROBE name=gate_proj alpha=0.0 err=0.3647673787946245
PROBE name=gate_proj alpha=0.5 err=0.3647673787946245
PROBE name=gate_proj alpha=1.0 err=0.3598613386025552
PROBE name=down_proj alpha=0.0 err=5.635943754342588e-06
PROBE name=down_proj alpha=0.5 err=4.6538904104479425e-06
PROBE name=down_proj alpha=1.0 err=4.6538904104479425e-06
```

`up_proj` is a perfect three-way tie ⇒ `if err < best_err` keeps the first
(α=0.0) here; the golden host kept α=0.5. On a real 35B with a strong spectral
signal these landscapes are usually well-separated, so an α *flip* is much less
likely. **But the run is still at risk** via §2.3.

### 2.3 The deeper risk that survives any α tie-break

The golden (`rank_map.alpha.fp32.json` @ 588ec5e) vs this host's produced
rank_map differ in **gate_proj and down_proj ranks too, with the SAME α**:

```
golden  alpha_by_type: gate=1.0, up=0.5, down=0.5
host    alpha_by_type: gate=1.0, up=0.0, down=0.5      # only up flips
golden  L0 gate=(5,5)  host L0 gate=(4,6)              # gate α identical yet ranks differ
```

Instrumented, this host computes `gate_proj@α=1.0` scores `[0.361, 0.639]` →
deterministically `(4,6)`. For the golden to land `(5,5)` the minting host's
gate scores must have been ≈`(0.5,0.5)` — a ~0.14 absolute gap, **far larger
than fp64 round-off** on identical inputs. Since the non-α path proves W and
`group_stats` are byte-identical cross-host, the only host-dependent input is the
**`eigh`-whitened spectrum** (§2.1). ⇒ the score *values* themselves drift
cross-host by enough to cross the `int(math.floor(...))` boundary at
`:1450`. **An α-selection tie-break alone does not fix this** — gate's α never
changed. The whitening must be made deterministic.

(Self-reproducibility confirmed: two consecutive regen runs on this host produce
byte-identical rank_maps, so the divergence is purely cross-host, not run-to-run.)

### 2.4 `per_group_type: true` interaction — confirmed

* H-α2 ("validation α discarded under `per_group_type=True`") is **TRUE in
  code** (`:1629-1635`). The validation-PPL α never reaches the rank_map for the
  default config.
* The deployed `alpha_by_type` comes from `_swift_svd_plus_alpha_search`
  (proxy). Therefore the fix belongs in the **proxy spectral path**
  (whitening + α-selection), NOT in the validation-PPL `_argmin_alpha`.
* `validation_samples=0` (failing test) and `validation_samples=512 +
  per_group_type=True` (real config) **both** route through the same proxy ⇒ one
  fix covers both.

---

## 3. The fix (minimal, byte-safe, host-stable)

### 3.1 Primary fix — deterministic whitening (root cause)

Replace the **`eigh`-factor whitening** in the α-path with a **Cholesky
whitening**, in BOTH spectral producers:

* `_swift_svd_plus_alpha_search`, `swift_svd_alpha.py:1203-1210`
* `_redistribute_ranks_swift_svd_plus`, `swift_svd_alpha.py:1401-1407`

**LOAD-BEARING impl constraint — keep the `W @ L_C` order. Do NOT mirror
`d_rank_allocate` / do NOT use `svdvals(L_C @ W.T)`.** The exact swap, applied
IDENTICALLY at both `:1209-1210` and `:1407`:

```python
A64 = A.to(device="cpu", dtype=torch.float64)
A64 = 0.5 * (A64 + A64.T)
jitter = 1e-6 * A64.diag().mean().clamp_min(1e-12) * torch.eye(
    A64.shape[0], dtype=torch.float64
)
L_C = torch.linalg.cholesky(A64 + jitter)          # full-rank, unique, host-stable
svs = torch.linalg.svdvals(W @ L_C)                # MUST be W @ L_C — see below
```

with the existing Cholesky-failure fallback to `svdvals(W)` + the warn-once
(`_warn_raw_svd_fallback_once`).

**Why `W @ L_C` and NOT `L_C @ W.T` (the d_rank_allocate convention):** for a
Cholesky factor `L_C` of `(A + jitter)`, `L_Cᵀ L_C ≠ A` — Cholesky is a
*triangular* factor, not the symmetric square root, so the two orderings are
genuinely DIFFERENT, non-equal quantities (the reviewer measured a **3.95**
divergence between them, not round-off). The activation-weighted spectrum the
α-path's docstring specifies (`:94-101`, `‖XW − XW_k‖_F ↔ ‖W − W_k‖_{A,F}`) is

```
svdvals(W @ L_C) = √eig(W · (A + jitter) · Wᵀ)
```

which is the CORRECT activation-weighted error spectrum (it depends on `A` only
through `L_C L_Cᵀ = A + jitter`, so the Cholesky-vs-symmetric-sqrt choice does
not matter for `W @ L_C`). `svdvals(L_C @ W.T)` would instead compute
`√eig(Wᵀ · (A + jitter) · W)` over the wrong axes — a paper-INCORRECT quantity.
So the impl loop must NOT "mirror d_rank_allocate exactly"; it keeps the existing
`W @ L` order and only swaps the *factor* (`eigh`-factor → Cholesky factor). The
prior draft's "mirror d_rank_allocate / `(or L_A @ W.T)`" wording was wrong and
is removed here.

Because the IDENTICAL swap is applied to both producers, the Tier-1 cache-reuse
precondition (`test_stage3_tier1.py:147`, the `torch.equal` between the producer
in `_swift_svd_plus_alpha_search` and the recompute in
`_redistribute_ranks_swift_svd_plus`) still holds — same `W @ L_C`, same dtype,
same device on both sides. Document the swap as a tightening of the existing
`D-eps-star` deviation. The numeric *values* change (Cholesky factor ≠
eigh-factor), which is exactly why the golden must be regenerated (§4).

**Why this is the right fix:** Cholesky of `(A + jitter)` is **full-rank** (no
`keep_a` threshold to flip cross-build) and **unique** (lower-triangular,
positive diagonal), removing the discrete-membership non-determinism that is the
dominant instability (§2.1). All downstream quantities (scores, α selection,
integer ranks) become host-stable — fixing both failure modes (§2.2 α tie AND
§2.3 score-value drift), while keeping the correct `W @ L_C` activation-weighted
spectrum.

### 3.2 Secondary fix — ε-tolerant α tie-break (defence in depth)

Even with deterministic whitening, an exact/near tie in `err` is possible
(e.g. `up_proj` all-equal on the tiny model). Add an explicit canonical
tie-break to the two proxy selection loops so a tie can never depend on float
ordering:

* `_swift_svd_plus_alpha_search` per-type loop `:1324-1328`
* `_swift_svd_plus_alpha_search` global loop `:1335-1339`

Rule (keep `best_alpha` only on a **strictly-better-beyond-tolerance**
improvement; ties resolve to the lowest grid index / lowest α, which is the
current strict-`<` behaviour and matches `_argmin_alpha`'s documented tie policy
at `:697-719`):

```python
ALPHA_ERR_REL_TOL = 1e-9       # relative; see justification below
best_err = float("inf"); best_alpha = alpha_grid[0]
for alpha in alpha_grid:
    err = _evaluate_alpha(name, alpha)
    if err < best_err * (1.0 - ALPHA_ERR_REL_TOL):    # strictly better beyond tol
        best_err = err
        best_alpha = alpha
```

(Equivalently: round `err` to `ALPHA_ERR_SIG = 9` significant digits before the
strict `<`. Relative form is preferred because `err` spans `~5e-6`…`~0.58`
across matrix types — an absolute epsilon can't serve both. Note: keep `best_err`
as the *unrounded* value once selected, comparing the next raw `err` against the
tolerance band, so the comparison is associative and order-stable.)

**Justification of `1e-9` relative:**
* **Larger than cross-host jitter:** with deterministic Cholesky whitening the
  residual cross-host difference is pure fp64 round-off in `svdvals` + the score
  arithmetic, `O(1e-13…1e-15)` relative. `1e-9` is ~4–6 orders of magnitude
  above that ⇒ swallows all jitter.
* **Smaller than any meaningful spectral gap:** a real α difference that should
  change the allocation moves `err` by a fraction of a singular-value² — on the
  35B that is `O(1e-3)` relative or larger (the `gate_proj` real gap above is
  `~1.4%`). `1e-9` is ~6 orders below that ⇒ never masks a genuine winner.
* It is the SAME order as established golden-stability epsilons elsewhere in the
  stage-3 fp64 spectra work (the D-Rank "0 rank flips to ~1e-14" claim,
  `d_rank_allocate.py` D-drank-fp64-spectrum). `1e-9` is a deliberately
  conservative buffer above that.

**Do NOT** apply rounding/epsilon to the residual `sorted(..., key=scores[i])`
loop or to `int(math.floor(...))`: once §3.1 makes the scores host-stable those
become deterministic for free, and perturbing them risks changing clear-winner
allocations. Keep them as-is.

### 3.3 What the fix must NOT change

* **Spectral-proxy `validation_samples=0` semantics:** unchanged except the
  whitening recipe + the explicit tie-break; the path still runs offline, no
  forward passes.
* **Validation-PPL path (`_argmin_alpha`, `:697-719`):** untouched. It is not
  the deployed-α decider under `per_group_type=True` and the brief's
  "round-the-PPL-metric" idea is explicitly NOT adopted (wrong site). The DP
  merge equivalence (`run_dp_alpha_search`) is therefore also untouched.
* **Clear-winner selections:** with `1e-9` rel-tol, any α whose `err` is better
  by more than `1e-9` still wins ⇒ real-data winners are preserved.
* **Non-α golden (`test_stage3_rank_map_byte_identical`):** the uniform path
  (`alpha_grid` length ≤ 1) never enters `_swift_svd_plus_alpha_search` /
  `_redistribute_ranks_swift_svd_plus` (`select_alpha` `:1651-1655`), so that
  golden is **unaffected** — it must stay byte-identical and must NOT be
  regenerated.

---

## 4. Golden-regeneration plan

The fix **changes the produced bytes** for the α-variant goldens (the Cholesky
whitening yields different spectra → different scores/ranks; this is intended).
Sequence:

1. Land §3.1 + §3.2 on the impl branch.
2. Regenerate ONLY the α-variant goldens:
   ```
   MOE_REGEN_GOLDEN=1 python3 -m pytest \
     max_quality/tests/test_stage3_golden_snapshot.py::test_stage3_rank_map_alpha_variant_byte_identical -v
   ```
   (writes `tests/golden/stage3/rank_map.alpha.fp32.json` and `…bf16.json`).
3. `git diff` the two regenerated files; sanity-check the new `alpha_by_type` +
   `rank_map` for plausibility (budget conservation per group still holds).
4. Commit the regenerated goldens **with the code fix in the same PR** so the
   bless is human-gated and atomic.
5. **Do NOT** regenerate the non-α goldens (`rank_map.{fp32,bf16}.json`). Verify
   `test_stage3_rank_map_byte_identical` still passes against the *unchanged*
   committed bytes (it must).
6. **Cross-host reproducibility is the acceptance gate:** the NEW golden must be
   reproducible on a second host/build. Verify by running the regen on at least
   one host with a different BLAS (e.g. the H200/CUDA box) and confirming
   byte-identity to the RTX 5080 mint. (This is the whole point — a golden that
   only reproduces on its mint host is the bug we are removing.)

---

## 5. Test plan

### 5.1 Host-stability unit test (new — the core acceptance test)

Add a focused test (e.g. `tests/test_stage3_alpha_determinism.py`) that asserts
α selection is invariant under fp-epsilon perturbation of the spectra:

* Build a small `group_stats` / `grouped_svs` fixture (or reuse the tiny
  pipeline) where one matrix type is a near-tie.
* Compute `alpha_by_type` once. Then perturb each `svs` (or each `A_cov`) by a
  relative `±1e-12` and recompute. **Assert the selected `alpha_by_type` is
  identical** across the perturbation (this is what cross-host jitter looks
  like).
* Separately, assert the perturbation does NOT change a **clear-winner** case
  (construct a fixture with a real `>1e-6` err gap; the winner must be stable AND
  must equal the un-perturbed winner) — proves the tolerance does not mask real
  signal.
* Assert `_redistribute_ranks_swift_svd_plus` produces identical integer ranks
  under the same perturbation (covers §2.3).

### 5.2 Whitening-determinism micro-test

Target the ACTUAL instability — the discrete `keep_a` threshold flip, not
eigenbasis choice:

* **Demonstrate the old eigh path WAS threshold-sensitive:** build an `A` with an
  eigenvalue sitting just at `1e-6 · λ_max`; perturb it by `±ε` so it crosses the
  `keep_a` boundary; show `svdvals(W @ L_A_eigh)` jumps discontinuously (column
  count changes).
* **Demonstrate the new Cholesky path is NOT:** under the same `±ε` perturbation,
  `svdvals(W @ L_C)` with `L_C = cholesky(A + jitter)` changes only by round-off
  (no threshold to cross), and `cholesky(A + jitter)` run twice is byte-equal.
* **Do NOT** assert anything about eigenvector sign/rotation invariance —
  `svdvals(W @ L)` is invariant to it by construction (depends only on `L·Lᵀ`;
  reviewer measured `~5e-15`), so such a test would prove nothing about the bug.

### 5.3 Golden regression

* `test_stage3_rank_map_alpha_variant_byte_identical[fp32,bf16]` passes against
  the regenerated goldens (post-fix).
* `test_stage3_rank_map_byte_identical` still passes against the UNCHANGED non-α
  goldens.

### 5.4 Cross-host bless (manual, gated)

Per §4.6 — regen on a second BLAS/host and diff to zero bytes.

---

## 6. Evidence log (commands run during investigation)

* `pytest …test_stage3_rank_map_alpha_variant_byte_identical[fp32]` → **FAIL**
  (drift) on RTX 5080; `…byte_identical` (non-α) → **PASS**. Confirms the
  divergence is confined to the α-path.
* Instrumented `_evaluate_alpha` / `_redistribute_ranks_swift_svd_plus` (prints,
  then reverted) → captured the `up_proj` 3-way `err` tie and the `gate_proj`
  same-α rank divergence (§2.2, §2.3).
* `git diff 588ec5e..HEAD -- …/swift_svd_alpha.py` over the scoring functions →
  **empty**; the scoring/whitening math is byte-identical between the golden
  commit and HEAD ⇒ the divergence is host/BLAS, not a code change.
* Two consecutive `MOE_REGEN_GOLDEN=1` runs on this host → **byte-identical** ⇒
  self-reproducible; cross-host only.
* Confirmed the non-α D-Rank path uses Cholesky (`d_rank_allocate.py`,
  `cholesky(A64 + jitter)`) vs the α-path's `eigh` factor — the asymmetry that
  explains the pass/fail split.
