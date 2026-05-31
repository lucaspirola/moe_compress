# Stage-4 (EoRA) Optimization Plan

**Base:** `origin/main` @ `588ec5e` · **Branch:** `plan/stage4-opt` · **Scope:** PLAN ONLY (no production edits in this branch).

All file:line citations below were re-verified against the real `origin/main` blob
`max_quality/src/moe_compress/stage4/plugins/eora_compensation.py` (read via `git show`), not the research-pass numbers.

---

## 0. Hot path — verified anchors

Single file: `max_quality/src/moe_compress/stage4/plugins/eora_compensation.py`.

| Anchor | Verified line | What it is |
|---|---|---|
| `_compute_eora_factors` def | **182** | the EoRA Algorithm-1 kernel (relocated verbatim from monolith; re-exported by `stage4_eora`) |
| `torch.linalg.eigh(A)` | **247** | eigendecomp of whitening covariance `A` (per `(layer,expert,name)`) |
| `torch.linalg.svd(delta_prime)` | **275** | full SVD of projected residual `ΔW'` (`take_eff = min(r, U_p.shape[1])` at **276**) |
| `compensate_layer` def | **363** | the S4-4 per-layer hook (reproduces the monolith `run()` inline block) |
| matrix loop `for name in MATRIX_NAMES` | **409** | outer per-matrix-type loop |
| per-expert loop `for e in range(N)` | **433** | inner per-expert loop |
| `res_before_sum += float(...item()**2)` | **446** | host sync #1 (Lever B) |
| `cov_key` (up_proj reuses gate_proj A) | **448** | `cov_key = (layer_idx, e, "gate_proj") if name == "up_proj" else key` |
| `res_after_sum += float(...item()**2)` | **458** | host sync #2 (Lever B) |

`MATRIX_NAMES = ("gate_proj", "up_proj", "down_proj")` — verified `utils/model_io.py:330`.
gate_proj and up_proj are **adjacent** in iteration order, which Lever A exploits.

**Device contract (confirmed, do NOT change):** the kernel moves operands to `dev` itself —
`delta = delta.to(device=device, dtype=torch.float32)` (`:221`) and `A = A.to(device=device, dtype=torch.float32)` (`:239`).
In the hook, `W_orig_f = W_orig.to(device=dev, ...)` (`:438`) and the `U_e/V_e` come from `fe.*_U.data[e]` already on `dev`.
There is **no device-mismatch crash surface** in Stage 4; all three levers preserve this — keep every new tensor on `dev`.

**Cov-reuse fact for Lever A (verified `:447-449`):** for `name == "up_proj"`, `cov_key` is rewritten to the
gate_proj key `(layer_idx, e, "gate_proj")`, so `A = A_cov.get(cov_key)` is the **identical object** for the
gate_proj and up_proj passes of the same `(layer, expert)`. Hence `eigh(A)` at `:247` is computed on
bit-identical input twice → redundant.

---

## 1. Lever A — eigh-reuse for the {gate_proj, up_proj} group (Tier-1, byte-identical)

### Why
For each `(layer, expert)`, the up_proj pass feeds the *same* covariance `A` as gate_proj (`:447-449`),
so `torch.linalg.eigh(A)` (`:247`) runs twice on bit-identical input. eigh is the dominant cost on the
gate/up pair (measured **1.70×** on that pair, ≈ −154 s over 48 layers). Reusing the spectrum
`(Q, √Λ, 1/√Λ)` from the first computation is **bit-identical** to recomputing it (same input, same op, same
device, same dtype) — therefore byte-identical golden.

### Structure decision: expert-outer for the {gate,up} group ONLY
Two ways to reuse:
- **(rejected) cache-all-experts:** precompute every expert's spectrum for a matrix → ≈0.13 GB/layer of cached
  eigenvectors (N × d_in × d_in fp32). Wasteful and changes the loop topology more than necessary.
- **(chosen) expert-outer micro-group for {gate,up}:** restructure so that, per expert `e`, gate_proj and
  up_proj are processed back-to-back sharing one eigh result. Cache footprint ≈ one spectrum (`Q` d_in×n_keep +
  two vectors) ≈ **~2 MB**, freed per expert. down_proj is left exactly as-is (it has its own distinct `A`).

### Iteration-order / ordering caveat (must verify before coding)
The current code is **matrix-outer, expert-inner**: it fully processes gate_proj for all N experts (building
`U_corr[N]`, calling `fe.widen_rank("gate_proj", ...)`, emitting trackio, spilling contributions) **before**
touching up_proj. A naive "expert-outer for {gate,up}" reorders these side effects:
`widen_rank` is called per-matrix on a `[N, d_out, r]` tensor assembled across all experts, and trackio emits
once per matrix. **Therefore the eigh-reuse must NOT collapse the two matrix passes into one expert loop at the
`widen_rank`/trackio granularity.** The safe restructure keeps the matrix-level aggregation intact and only
shares the *spectrum computation* inside the kernel.

### Recommended before/after sketch (kernel-level reuse, order-preserving)
Split `_compute_eora_factors` so the eigh step is separable, and have the hook compute the spectrum once per
`(layer, expert)` for the gate/up group, passing it into both calls. Concretely:

**Add** an internal helper (or an optional precomputed-spectrum arg) to `_compute_eora_factors`:
```
# pseudo — eigh extracted to a reusable spectrum object
def _eigh_spectrum(A, d_in, storage_dtype, device):
    # MUST perform the SAME prologue the kernel does at :239-:240 BEFORE eigh:
    #   A = A.to(device=device, dtype=torch.float32)   # :239
    #   A = 0.5 * (A + A.T)                              # :240
    # then torch.linalg.eigh(A) (:247) on that exact post-cast, post-symmetrize
    # input. The memoized spectrum must be computed strictly AFTER this
    # .to()+symmetrize, or the up_proj reuse will not be bit-identical to the
    # gate_proj eigh input. (eigh is keyed on the symmetrized fp32 matrix, not
    # the raw stored `A`.)
    # returns None  → caller falls back to _plain_svd_padded()
    # returns (eigvecs_keep, sqrt_lambda, inv_sqrt_lambda)  on success
    ...  # exactly the :239-:286 prologue, no behavioral change

def _compute_eora_factors(delta, A, r, device, *, storage_dtype=None, spectrum=None):
    # if spectrum is None and A is not None: spectrum = _eigh_spectrum(...)
    # then reuse spectrum for Q_prime / delta_prime / back-projection
```
**A1 memo key caveat (byte-identity):** the cached spectrum that the up_proj pass
reuses MUST be the spectrum of the *post-cast, post-symmetrize* matrix — i.e. the
gate_proj pass memoizes AFTER running `A = A.to(device, fp32); A = 0.5*(A + A.T)`
(`:239-:240`) and only then `eigh` (`:247`). Memoizing the raw stored `A` (or its
spectrum computed before the symmetrize) would let FP rounding in the `.to()`/`+A.T`
diverge from the gate pass and break the byte-identical golden. Memoize strictly
after the `:239-:240` step.

In the hook, restructure the gate/up handling so per expert `e` we compute the spectrum once from
`A_cov.get((layer_idx, e, "gate_proj"))` and pass it to **both** the gate_proj and up_proj
`_compute_eora_factors` calls. Implementation options (pick at code time, both order-preserving):
- **(A1)** keep matrix-outer loops but memoize the spectrum in a small per-layer dict keyed by `(e)` that the
  up_proj pass reads (built during the gate_proj pass; ≈2 MB × transient). Simplest diff; preserves exact
  `widen_rank`/trackio order.
- **(A2)** introduce a dedicated `{gate_proj, up_proj}` co-processing block that builds both `U_corr`/`V_corr`
  tensors in one expert loop, then calls `widen_rank("gate_proj", …)` and `widen_rank("up_proj", …)` **in that
  exact order** afterwards, then trackio for gate then up **in that exact order**, then continues to down_proj.
  This is the cleaner end state but must reproduce the current emit/widen order precisely.

**Preferred: A1** — minimal diff, zero risk of reordering `widen_rank`/trackio/spill, and the byte-identical
claim is trivially auditable (the spectrum object is bit-identical to what the second eigh would have produced).

### Byte-identical justification
The only changed computation is *not recomputing* `eigh` on the up_proj pass; everything downstream consumes the
same `(eigvecs_keep, sqrt_lambda, inv_sqrt_lambda)` floats. No FP reordering, no change to `take_eff`, no change
to `widen_rank` inputs → `eora_ranks.{bf16,fp32}.json` unchanged.

### Test that proves it
`test_stage4_golden_snapshot.py` (both fp32 + bf16 params) must remain byte-identical with **no re-bless**.
Plus `test_eora_bf16_A.py` (kernel numerics) and `test_stage4_plugin_compensation.py` (shape/plumbing) pass
unchanged.

---

## 2. Lever B — defer `.item()` host syncs (Tier-1, golden-safe)

### Why
`:446` `res_before_sum += float(delta.norm().item() ** 2)` and `:458`
`res_after_sum += float(res_after.norm().item() ** 2)` each force a device→host sync **per expert** (N syncs ×2
per matrix). These feed ONLY the `log.info` residual line (`:478-480`) and the trackio
`*_residual_unweighted_*` keys (`:491-493`). They do **NOT** feed the golden: `eora_ranks.json` carries only
`rank_map` (ints), `compensated_params` (int), and the literal config block — confirmed by reading the golden
JSON top-level keys `['compensated_params','config','rank_map']` and the snapshot test docstring
(`test_stage4_golden_snapshot.py:19-25`).

### Before/after sketch
Accumulate the squared norms on-device and sync once per matrix (after the expert loop), not per expert:
```
# before (per-expert host sync, :446 / :458)
res_before_sum += float(delta.norm().item() ** 2)
...
res_after_sum  += float(res_after.norm().item() ** 2)

# after — GPU accumulators, single sync per matrix
res_before_acc = torch.zeros((), device=dev, dtype=torch.float32)   # init before expert loop
res_after_acc  = torch.zeros((), device=dev, dtype=torch.float32)
...
res_before_acc += delta.norm() ** 2               # stays on dev (delta already fp32 at :445/:221)
res_after_acc  += res_after.norm() ** 2
...
# after the expert loop (once):
res_before_sum = float(res_before_acc.item())
res_after_sum  = float(res_after_acc.item())
```
Note: `norm()**2` == `(x**2).sum()` in fp32; either form is fine as long as it stays on `dev` until one sync.
**Preferred: keep `.norm()**2` and defer only the `.item()`** (accumulate the 0-d tensor, sync once) — the most
literal, lowest-drift variant. The earlier `(delta.float() ** 2).sum()` sketch has a **no-op `.float()`** —
`delta` is already fp32 (cast at `:221`, in scope at `:445`), so the `.float()` does nothing and is dropped
above; the deferred-`.item()` `.norm()**2` form is what stands.

### Golden / thread-safety confirmation
- **Golden:** unchanged (residual is log/trackio only; not in `eora_ranks.json`).
- **Thread-safety:** the hook is single-threaded per layer (no `concurrent.futures`, no threads in this file);
  the accumulators are plain locals. Log-only.
- **Drift:** summation reordering across experts can shift the *logged/trackio* residual at the ~ULP level —
  acceptable, not golden-pinned. (Lever C will move it ~1e-1 anyway; see §3.)

### Test that proves it
`test_stage4_golden_snapshot.py` byte-identical (no re-bless). The residual value is not asserted by any test
(grep confirmed: only `test_eora_bf16_A.py` asserts residual *inequalities*, which are computed independently
inside that test, not read from the hook).

---

## 3. Lever C — Gram-side SVD (Tier-2, rank-only golden, HUMAN-GATED)

### Why
`torch.linalg.svd(delta_prime)` (`:275`) computes the full SVD of `ΔW' ∈ ℝ^{d_out × n_keep}` (n_keep up to
~2048) but we only consume `take_eff ≤ r ≤ 128` triplets (`:276`, `U_p[:, :take_eff]`, `S_p[:take_eff]`,
`Vh_p[:take_eff]` at `:279,:286`). Replace with an eigendecomposition of the **smaller Gram**:
`G = ΔW'ᵀΔW'` (n_keep×n_keep) if `n_keep ≤ d_out`, else `G = ΔW'ΔW'ᵀ` (d_out×d_out), then extract the top-`r`
singular triplets. Measured: **1.84×** gate/up, **3.24×** down end-to-end; SVD step alone **5–5.6×**.

### Before/after sketch
```
# before (:275-:286)
U_p, S_p, Vh_p = torch.linalg.svd(delta_prime, full_matrices=False)
take_eff = min(r, int(U_p.shape[1]))       # U_p.shape[1] == min(d_out, n_keep); SVD keeps ALL σ incl. ~0
U_corr = U_p[:, :take_eff] * S_p[:take_eff]
inv_sqrt_lambda = eigvals_keep.clamp_min(1e-30).rsqrt()
V_corr = (Vh_p[:take_eff, :] * inv_sqrt_lambda.unsqueeze(0)) @ eigvecs_keep.T

# after — Gram-side, top-r triplets (fp32 throughout; see conditioning note)
# CRITICAL: take_eff MUST mirror production EXACTLY. Production (:276) is
#   take_eff = min(r, U_p.shape[1])  with U_p from torch.linalg.svd(..., full_matrices=False)
# so U_p.shape[1] == min(d_out, n_keep) and the SVD returns ALL singular values,
# including exact/near-zero/Gram-negative ones (SVD does NOT drop zeros). Do NOT
# add an `(evals > 0).sum()` rank filter here — that would make Gram take_eff
# STRICTLY SMALLER than production whenever a kept direction has a zero/near-zero/
# Gram-negative eigenvalue, a DETERMINISTIC rank reduction the production SVD never
# does → flips rank_map / compensated_params → breaks the byte-identical golden BY
# CONSTRUCTION (not merely at a numerical tie). Instead, keep take_eff identical to
# production and clamp the singular VALUES; the negligible directions then sit in the
# structurally-zero low-rank tail and contribute ~nothing.
d_out_, n_keep_ = delta_prime.shape
take_eff = min(r, min(d_out_, n_keep_))    # == production min(r, U_p.shape[1]); NO (evals>0) filter
if n_keep_ <= d_out_:                      # right-Gram smaller
    G = delta_prime.T @ delta_prime        # [n_keep, n_keep], fp32
    evals, evecs = torch.linalg.eigh(G)    # ascending
    idx = torch.arange(n_keep_-1, n_keep_-1-take_eff, -1, device=dev)  # top-take_eff descending
    s = evals[idx].clamp_min(0).sqrt()     # singular values (clamp the VALUES, not the count)
    Vh = evecs[:, idx].T                   # right singular vecs (== Vh_p[:take_eff])
    # guard zero-σ columns: those left vectors are in the structurally-negligible
    # tail and contribute ~nothing; eps/clamp tensor lives on `dev` (no device mismatch).
    U  = (delta_prime @ evecs[:, idx]) / s.clamp_min(eps)  # left singular vecs
else:                                      # left-Gram smaller
    G = delta_prime @ delta_prime.T        # [d_out, d_out]
    ...                                    # symmetric construction, V = (Δ'^T U)/s,
                                           # same take_eff = min(r, min(d_out_, n_keep_)),
                                           # same s = evals.clamp_min(0).sqrt(), same
                                           # s.clamp_min(eps) guard on the divide
U_corr = U * s                             # == U_p[:,:take_eff] * S_p[:take_eff]
V_corr = (Vh * inv_sqrt_lambda.unsqueeze(0)) @ eigvecs_keep.T
```
**`eps` device note:** the `s.clamp_min(eps)` guard and any Gram-side `eps`/clamp
must use a scalar/tensor already on `dev` (a Python float literal is fine; a tensor
must be created with `device=dev`). No new device-mismatch surface — the kernel's
device contract (`§0`) is preserved.
Keep zero-padding logic (`:289-296`) **unchanged** — `test_eora_zero_pad_path_used_when_take_lt_r` asserts
`torch.equal(U[:, take:], zeros)` / `torch.equal(V[take:, :], zeros)`, so the pad region must stay exactly zero.

### Critical numerical constraints
- **Keep the `eigh(A)` at `:247` in fp32.** Lever C only changes the *inner* SVD on `delta_prime`. Do NOT
  Gram-square `A` itself — squaring `A` worsens conditioning of the whitening spectrum (the Stage-3 fp64-trick
  does not transfer to EoRA; fp64 spectra are 9× slower and rejected).
- **`take_eff` rank-boundary tie:** Gram eigh reorders FP vs direct SVD; reconstruction rel-Frobenius ≈ **4e-4**.
  At a near-degenerate singular-value tie this could flip `take_eff` by ±1 at a boundary → a **rank flip** that
  *would* change `eora_ranks.json`. This near-tie reorder (~4e-4) is the ONLY golden-touch surface, and the plan
  already human-gates it (see "Golden gate" below).
- **Define `take_eff` to match production EXACTLY — `take_eff = min(r, min(d_out, n_keep))`.** Production (`:276`)
  is `min(r, U_p.shape[1])` where `U_p` comes from `torch.linalg.svd(delta_prime, full_matrices=False)` (`:275`),
  so `U_p.shape[1] == min(d_out, n_keep)` and the SVD returns **all** singular values including exact/near-zero
  ones (SVD does NOT drop zeros). Do **NOT** add a `(number of strictly positive singular values)` /
  `(evals > 0).sum()` term — that filter would make Gram `take_eff` strictly smaller than production on any kept
  direction with a zero/near-zero/Gram-negative eigenvalue, a deterministic rank reduction that breaks the
  byte-identical golden BY CONSTRUCTION. Clamp the singular **values** (`s = evals.clamp_min(0).sqrt()`) and
  guard the left-vector divide (`/ s.clamp_min(eps)`) instead of reducing the count.

### Golden gate — HUMAN-GATED rank-diff (NO blind regen)
1. Implement Lever C behind the change.
2. Run a **rank-diff harness** (plan-only describes it; implementer builds it): run Stage 4 on the
   `tiny_model` fixture (both fp32+bf16) with Gram-side SVD, dump the produced `eora_ranks.{case}.json`, and
   `diff` its `rank_map` + `compensated_params` against the **current committed golden**.
   - The tiny fixture's golden is empty (`rank_map: {}`, `compensated_params: 0`, `eigenspace_rank_cap: 4`), so
     the tiny golden almost certainly shows **0 flips**. To meaningfully exercise the rank-boundary, the
     harness should ALSO run a larger synthetic case (production-like d_in≈4096, r≈128) and report any
     `take_eff` deltas vs the production SVD path on the same inputs.
3. **Re-bless ONLY if a flip is observed** (expected outcome: 0 flips, mirroring Stage-3's 0-flip Gram-side
   result). If 0 flips → golden is byte-identical, no re-bless, commit code only.
4. **Never** run a blind `MOE_REGEN_GOLDEN=1`. If a flip is real and intended, re-bless with the explicit human
   sign-off recorded in the commit message + a one-line note here.

### Non-golden float drift — REQUIRED pre-merge confirmation
Saved checkpoint U/V floats and the trackio residuals **will drift ~1e-1** under Gram-side SVD. These are NOT
golden-pinned. **Before merging Lever C, the implementer MUST confirm no downstream test pins post-Stage-4
checkpoint float tensors via tolerance compare.**

Verified during planning (so the implementer can re-confirm fast):
- `grep` of stage-4 tests for `allclose`/`torch.equal`/`effective_ranks` found **one** float compare
  **ON THE POST-STAGE-4 U/V OUTPUT PATH**:
  `test_smoke_stage4_resume.py:233-235` — `torch.allclose(U_actual, U_expected, atol=1e-5)`.
  A second `torch.equal` exists but is verified-irrelevant: `test_stage4_input_cov_cache.py:257`
  `torch.equal(A_cov[(0,0,"gate_proj")], torch.eye(3, dtype=fp16) * 2)`. That assertion is on the **INPUT
  covariance** `A_cov` (a cache-load roundtrip), which is **UPSTREAM of `_compute_eora_factors`** — no lever
  (A, B, or C) touches it, so it cannot drift. It is omitted from the "output path" count deliberately.
  **`test_smoke_stage4_resume.py:233-235` is SAFE for Lever C.** Reading `test_smoke_stage4_resume.py:189-241`: `saved` is captured from a clean
  run, written *into* the spill file, and the resumed run **loads layer 0 from spill** (not recomputed). The
  assertion compares spill-write bytes vs spill-read bytes on the same compute path — it does NOT compare a
  recompute against a frozen expectation. Lever C changes the compute path uniformly, so both sides move
  together. **Re-run this test to confirm; do not pre-emptively touch it.**
- `test_eora_bf16_A.py` asserts only *inequalities* with margins (`res_a < res_iso - 1e-6`,
  `residual < ‖δ‖`) and shape/zero-pad equality. A 4e-4 rel-Frobenius drift preserves all of these. Re-run to
  confirm.

### Test that proves it
- `test_stage4_golden_snapshot.py` byte-identical (expected 0-flip) — the hard gate.
- `test_eora_bf16_A.py` (inequalities + zero-pad `torch.equal`) pass.
- `test_smoke_stage4_resume.py` (spill roundtrip allclose) pass.
- The new rank-diff harness output (committed as an artifact under `tasks/` or printed in CI log) showing
  flip count.

---

## 4. Ordering

Land in this order; each step independently shippable:

1. **Lever A** (eigh-reuse, A1 memoize variant) — byte-identical, smallest diff, biggest single win on gate/up.
2. **Lever B** (deferred sync) — byte-identical, independent of A, trivial.
   *(A and B can be one commit since both are byte-identical and touch the same hook; keep them separate commits
   for clean rollback.)*
3. **Lever C** (Gram-side SVD) — last, because it is the only golden-touching lever and requires the human
   rank-diff gate. Build on top of A (A's extracted `_eigh_spectrum` helper leaves `delta_prime` construction
   intact, so C slots cleanly into the post-`delta_prime` SVD block).

Rationale: A+B are zero-risk and bank ~1.7× on gate/up + sync savings immediately; C is gated and reviewed
separately so a rank-flip surprise never blocks the safe wins.

---

## 5. Risks + rollback

| Lever | Risk | Mitigation | Rollback |
|---|---|---|---|
| A | Accidentally reorders `widen_rank`/trackio/spill if the {gate,up} block is collapsed wrong | Use variant A1 (memoize spectrum, keep matrix-outer loops); golden byte-check catches any reorder that touches ranks | revert single commit; loops restored |
| A | Memoized spectrum holds GPU memory across experts | Spectrum is ~2 MB and freed each expert (keyed by `e`, dropped after up_proj pass); never cache all experts | n/a |
| B | Summation reorder shifts logged residual at ULP/1e-1 | Log/trackio only, not golden; documented | revert single commit |
| C | `take_eff` rank flip at a singular-value tie → golden drift | Human-gated rank-diff before merge; re-bless only on real flip; expected 0 flips | revert single commit; golden untouched if 0 flips |
| C | Gram-squaring worsens conditioning | Square only `delta_prime` (small), keep `eigh(A)` fp32; do NOT square `A` | n/a (design constraint) |
| C | Downstream float-tensor pin breaks | Confirmed only `test_smoke_stage4_resume.py` (spill roundtrip, safe); re-run before merge | revert single commit |

---

## 6. Explicitly OUT of scope (do NOT plan/implement)

- **CPU per-expert threading** — only helps if Stage 4 ran on CPU; it runs GPU-resident where per-call is faster.
- **Batched eigh/svd** — measured a no-op/regression.
- **fp64 spectra** — 9× slower; the Stage-3 fp64 trick does not transfer to EoRA.

### Measurement discrepancy to FLAG (implementer must re-confirm, do NOT cite as fact)
One research pass measured CPU `eigh(2048)` ≈ **59 ms**, another ≈ **87 s**. The 87 s figure is suspect (likely
a thread-thrash / first-call import artifact). It does not change the "Stage 4 stays GPU-resident" conclusion,
but **must not be quoted as a fact**. Re-measure on the host before relying on any absolute CPU eigh timing.

---

## 7. Testing plan (host RTX 5080, ~112 s total)

Run from repo root `/home/lucas/ai/moe_compress`:

```
pytest max_quality/tests/test_stage4_golden_snapshot.py \
       max_quality/tests/test_stage4_plugin_compensation.py -v
```
(~112 s per the research pass; both fp32+bf16 golden params exercised.)

Plus the numerics + roundtrip pins touched by Lever C:
```
pytest max_quality/tests/test_eora_bf16_A.py \
       max_quality/tests/test_smoke_stage4_resume.py -v
```

**Gate per lever:**
- A, B: golden byte-identical, all four files green, **no re-bless**.
- C: run all four; produce the rank-diff harness output; re-bless ONLY on a confirmed flip with human sign-off.

**Determinism caveat:** regen+verify must run on the same machine/wheel/venv (golden test docstring
`test_stage4_golden_snapshot.py:8-16`). The RTX 5080 host is the canonical bless machine.

---

## 8. Plan provenance
- Code read from real blob `git show origin/main:max_quality/src/moe_compress/stage4/plugins/eora_compensation.py`.
- Golden read from `max_quality/tests/golden/stage4/eora_ranks.{bf16,fp32}.json` (top keys
  `compensated_params`, `config`, `rank_map`; tiny golden is empty/zero).
- Tests read from `max_quality/tests/{test_stage4_golden_snapshot,test_stage4_plugin_compensation,test_eora_bf16_A,test_smoke_stage4_resume}.py`.
- All §0 line numbers re-verified against the blob (not the research-pass numbers).
