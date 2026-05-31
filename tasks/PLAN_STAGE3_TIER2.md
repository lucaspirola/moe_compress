# Stage-3 Tier-2 Implementation Plan

**Status:** PLAN ONLY — no production code is touched by this document.
**Base:** `origin/main` @ `f5d7c1e600213b9eb8591baca88b62a5aba7c09c`
**Branch:** `plan/stage3-tier2`
**Paths:** all under `max_quality/src/moe_compress/...` unless noted.

All file:line citations below were read directly from `origin/main` blobs
(`git show origin/main:<path>`), not the local tree. Where the user's brief
cited an approximate line, the verified exact line is given and the drift is
noted.

---

## 0. Problem statement (fixed constraints — not open questions)

Stage 3 will run **GPU-resident**: the model weights `W` live on GPU during
factorization. Today the whitened-spectrum decompositions read a
**CPU-resident** `A_cov` / `A_g` (loaded with `map_location="cpu"` and never
moved) and combine it with a **GPU-resident** `W`. `eigh` / `cholesky` /
`svdvals` / matmul across two devices **raises** (`RuntimeError: Expected all
tensors to be on the same device`). On a GPU model this is a hard **crash**,
not a slowdown.

Tier-2 delivers four changes:

1. **Crash-fix / device move** — run the whitened decomps on the model's device.
2. **Precision split** — rank-deciding *spectra* in **fp64 (CPU)**; bulk
   factor matrices `U_k`/`V_k` in **fp32 on GPU**.
3. **Zero-pad fix** — `factor_layer` must pad each expert's `U_k`/`V_k` to the
   layer slot width before `set_factors` (pre-existing crash on non-uniform
   per-expert ranks).
4. **Golden re-bless** — diff fp64-spectra ranks vs the current golden, human
   review, then re-bless byte-identical goldens + unblock the α-grid xfail +
   add a device-independence assertion.

---

## 1. Verified citations (origin/main)

### 1.1 swift path — `stage3/plugins/swift_svd_alpha.py`
`_swift_svd_plus_alpha_search` (def at **685**):

| What | Verified line | Brief cited | Drift |
|---|---|---|---|
| `W = banks[name].get(e).detach().to(torch.float32)` | **749** | — | — |
| `A_f32 = A.to(torch.float32)` ; symmetrize | **753-754** | ~743-761 | ok |
| `eigvals_a, eigvecs_a = torch.linalg.eigh(A_f32)` | **755** | ~743-761 | ok |
| `keep_a = eigvals_a > eigvals_a.max() * 1e-6` | **756** | ~743-761 | ok |
| `L_A = eigvecs_a[:, keep_a] * eigvals_a[keep_a].clamp_min(1e-12).sqrt()...` | **758** | — | — |
| `M_A = W @ L_A` | **759** | ~761 | ok |
| `svs = torch.linalg.svdvals(M_A)` | **760** | ~761 | ok |
| `tail = float(s2[k_group:].sum().item()) if k_group < len(s2) else 0.0` | **807** | ~796 | **+11 lines** |

`_redistribute_ranks_swift_svd_plus` (def at **895**) — the twin path:

| What | Verified line | Brief cited | Drift |
|---|---|---|---|
| `eigvals_a, eigvecs_a = torch.linalg.eigh(A_f32)` | **949** | — | — |
| `L_A = eigvecs_a[:, keep_a] * ...sqrt()...` | **952** | — | — |
| `svs = torch.linalg.svdvals(W @ L_A)` | **953** | **953** | exact |
| `tail = ...` | **972** | — | — |
| `cached_svs` reuse branch (item-2 cache; byte-identical to recompute *once both producer and recompute are CPU-fp64* — see §2.3 / C1) | **936-945** | — | — |

Note: the redistribute path has a `grouped_svs_cache` fast path (**936**) that
reuses the proxy's spectrum when threaded; the recompute branch (**943-961**)
is the device-mixed one. **Both** the cached value (built in `_swift_..._search`)
and the recompute must be on the policy device, or `torch.equal` precondition
(tier1 test) and the energy math break.

### 1.2 d_rank path — `stage3/plugins/d_rank_allocate.py`
`_group_stat` (def at **338**):

| What | Verified line | Brief cited | Drift |
|---|---|---|---|
| `A64 = A_g.to(torch.float64)` ; symmetrize | **355-356** | ~355-377 | ok |
| `jitter = 1e-6 * A64.diag().mean()... * eye(..., float64, device=A64.device)` | **357-358** | — | — |
| `L_A = torch.linalg.cholesky(A64 + jitter).to(torch.float32)` | **359** | ~355-377 | ok — **already fp64 chol → fp32 cast** |
| `W = bank.get(e).detach().to(torch.float32)` | **367** | — | — |
| `M = L_A @ W.T` ; `s = torch.linalg.svdvals(M)` | **373-374** | ~355-377 | ok |
| `eff_rank = float(torch.exp(-(p*p.clamp(min=1e-12).log()).sum()).item())` | **392** | **392** | exact |

`_d_rank_allocate` (the allocator):

| What | Verified line | Brief cited | Drift |
|---|---|---|---|
| `_weight(g,s): return math.sqrt(s.effective_rank / s.omega) * pw.get(...)` | **476** | **476** | exact |
| `out = {g: max(1, min(int(round(raw[g])), _cap(s))) ...}` | **502** | **502** | exact |

The fp64-chol→fp32-cast at **359** is the existing precision pattern; the
Tier-2 policy folds in consistently (§3).

### 1.3 covariance load — `stage3/plugins/covariance_collection.py`
- `_load_stage2_covariance` def at **467**.
- `payload = torch.load(path, map_location="cpu")` at **512** (brief: ~512,
  exact). Returns `payload.get("covariance", {})` at **513**. The covariance
  dict is **never** `.to(device)`'d anywhere — it is the CPU source of the
  device mismatch. (Stage-2 fp16-persist / fp64-in-memory deviation is noted in
  the plugin `paper` string at **541-543**; storage stays fp16, that is
  unrelated to the residency fix.)

### 1.4 zero-pad bug — `stage3/plugins/aa_svd_factor.py` + `utils/model_io.py`
`factor_layer` (the verbatim per-layer loop):

| What | Verified line | Brief cited | Drift |
|---|---|---|---|
| `# Experts with lower rank will be zero-padded; effective_ranks tracks...` | **519-521** | ~515 | ok (comment is a **lie** today) |
| `ranks_layer = {name: max(per_expert_ranks.get((li,name,e), ranks[...]) for e in range(...))}` | **522-528** | ~514-524 | ok — slot width = **max_e** per-expert rank |
| else-branch `ranks_layer = {name: ranks[(li,name)]}` (uniform) | **531-533** | — | — |
| `FactoredExperts(..., ranks=ranks_layer, dtype, device=dev)` | **592-597** | — | slot params built at `ranks_layer[name]` |
| per-expert `k = per_expert_ranks.get((li,name,e), ranks_layer[name])` | **632-635** | ~608-633 | ok |
| `U_k,V_k,rel_err,k_eff = _aa_svd_precomputed(W, ..., k, ...)` / `_aa_svd(...)` | **637-639 / 648-650** | — | `U_k`=`(d_out,k)`, `V_k`=`(k,d_in)` |
| `if k_eff < k: k_eff_clip_count[name]+=1` | **651-652** | — | — |
| `new_factored.set_factors(e, name, U_k, V_k, effective_rank=k_eff)` | **653** | ~608-633 | ok |
| `rank_map[f"L{li}_E{e}_{name}"] = k` | **654** | — | rank_map records **k**, not k_eff |

`_aa_svd` / `_aa_svd_precomputed` already zero-pad **`k_eff → k`** internally
(`U_k = torch.zeros(d_out, k...)`; `U_k[:, :k_eff] = U_eff`) at **310-316**,
**326-332**, **394-400**. They do **NOT** pad **`k → k_max`** (the slot width
`ranks_layer[name]`). That second pad is the missing step.

`FactoredExperts.set_factors` — `utils/model_io.py`:

| What | Verified line | Brief cited | Drift |
|---|---|---|---|
| `def set_factors(self, expert_idx, name, U, V, *, effective_rank=None)` | **704-707** | ~704-746 | ok |
| docstring: "important for honest parameter counting when callers zero-pad to a fixed slot width" | **712-715** | — | the arg exists **for exactly this** |
| `if effective_rank is None: effective_rank = self.ranks[name]` | **724-725** | — | — |
| range check `0 <= effective_rank <= self.ranks[name]` | **726-731** | — | — |
| `exp_U = (U_param.shape[1], U_param.shape[2])` (= `(d_out, slot)`) | **734** | — | — |
| `if tuple(U.shape) != exp_U: raise ValueError("U.shape=... expected ...")` | **737-740** | ~704-746 | **the crash** |
| `if tuple(V.shape) != exp_V: raise ValueError(...)` | **741-744** | — | — |
| `U_param.data[e].copy_(U.to(device=U_param.device, dtype=...))` | **745** | — | **set_factors already device-coerces** |
| `self.effective_ranks[name][e] = int(effective_rank)` | **746** | — | — |

`FactoredExperts.forward` — `utils/model_io.py` **872-960**:
- factors indexed per active expert (**940-946**), then
  `gate = bmm(bmm(gathered, V_g.T), U_g.T)` etc. (**949-952**).
- A trailing **zero row of V** produces a zero in the rank-`k` intermediate,
  which a **zero column of U** then maps to zero — the padded directions
  contribute exactly 0. **Forward tolerates trailing zero rank.** (Confirmed by
  the matmul structure at **949-952**; verified additionally because EoRA's
  `widen_rank` path at **838-867** relies on the same zero-pad-is-inert
  property and is already shipped.)

### 1.5 golden snapshot — `tests/test_stage3_golden_snapshot.py`
- byte-identical test `test_stage3_rank_map_byte_identical` at **155**, runs
  `device=None` → **CPU** (param at **159**), `alpha_grid=[0.5]` (length 1,
  uniform path, never enters `_swift_..._search`).
- existing goldens on disk: `tests/golden/stage3/rank_map.fp32.json`,
  `rank_map.bf16.json` (verified via `git ls-tree`). **No** `rank_map.alpha.*`
  goldens exist yet.
- α-variant fixture `patched_stage3_alpha` sets `alpha_grid=[0.0,0.5,1.0]`,
  `validation_samples=0` (offline spectral proxy) at **216**.
- α-variant test `test_stage3_rank_map_alpha_variant_byte_identical` at **273**,
  wrapped in `@pytest.mark.xfail(..., strict=False, raises=ValueError)` at
  **251-272**. The xfail `reason` (**253-269**) **explicitly names this Tier-2
  ticket**: "file a Tier-2 / re-bless ticket to zero-pad in factor_layer, then
  this xfail flips to a real bless via MOE_REGEN_GOLDEN=1." It cites the byte-
  safe cache proof `test_stage3_tier1.py::test_grouped_svs_cache_equals_recompute`
  (verified to exist at `test_stage3_tier1.py:150`).
- `MOE_REGEN_GOLDEN=1` regen branch at **50** / **172** / **292**.

---

## 2. Change 1 — device move (the crash-fix)

### 2.1 Decision: where each operand lives
The model device is `dev` (already computed in `factor_layer` at
`aa_svd_factor.py:578` as `ex.gate_up_proj.device`). For the **swift** and
**d_rank** spectra paths the equivalent device is the bank/weight device. The
covariance dict from `_load_stage2_covariance` is **CPU**.

**Policy (combined with §3 precision):** rank-deciding spectra are computed on
**CPU in fp64** (see §3 for why not GPU-fp64). Therefore the device move is:
bring `W` (and the small `A`) to **CPU** for the *spectrum* computation, and
keep the *factor* construction on **GPU-fp32**. This both fixes the mismatch
(everything that touches `A_cov`/`A_g` is co-located on CPU-fp64) and satisfies
the device-independence guarantee for free.

> Rationale for moving the *spectrum* to CPU rather than `A` to GPU: the
> measurement showed fp64 `svdvals` on consumer Blackwell (RTX 5080) is ~14×
> slower than fp32-GPU and slower than CPU. We must NOT put fp64 `svdvals` on a
> consumer GPU. The spectrum matrices are tiny (one expert, one matrix type at
> a time — `M_A` is `[d_out, r_A]`, `M = [d_in, d_out]`), so CPU-fp64 svdvals is
> cheap and bounded. eigh/cholesky fp64 on H200 is only ~5–7× and acceptable,
> but to keep ONE device-independent code path we standardize the
> rank-deciding spectra on CPU-fp64 across both H200 and the 5080 host.

### 2.2 swift — `_swift_svd_plus_alpha_search` (~749-760)
**Before** (device-mixed, fp32):
```python
W   = banks[name].get(e).detach().to(torch.float32)          # GPU
A_f32 = A.to(torch.float32)                                  # CPU
A_f32 = 0.5 * (A_f32 + A_f32.T)
eigvals_a, eigvecs_a = torch.linalg.eigh(A_f32)              # CPU op
L_A = eigvecs_a[:, keep_a] * eigvals_a[keep_a]...sqrt()...   # CPU
M_A = W @ L_A                                                # CRASH: GPU @ CPU
svs = torch.linalg.svdvals(M_A)
```
**After** (co-located on CPU, fp64 spectrum — see §3 for the dtype helper):
```python
# spectrum decision is device-independent: do it on CPU in fp64.
W64 = banks[name].get(e).detach().to(device="cpu", dtype=torch.float64)
A64 = A.to(device="cpu", dtype=torch.float64)
A64 = 0.5 * (A64 + A64.T)
eigvals_a, eigvecs_a = torch.linalg.eigh(A64)
keep_a = eigvals_a > eigvals_a.max() * 1e-6
if keep_a.any():
    L_A = eigvecs_a[:, keep_a] * eigvals_a[keep_a].clamp_min(1e-12).sqrt().unsqueeze(0)
    M_A = W64 @ L_A
    svs = torch.linalg.svdvals(M_A)        # fp64, CPU — feeds the rank cutoff
else:
    ... svs = torch.linalg.svdvals(W64)
grouped_svs[name][(li, e)] = svs           # store fp64 spectrum
```
- `keep_a` threshold `>max·1e-6` is **kept verbatim** (it gates the whitening
  mask; in fp64 it is more stable, never less).
- the downstream `s2 = svs*svs`, `tail = s2[k_group:].sum()` energy/tail cutoff
  at **807** now consumes an fp64 spectrum → deterministic ranks.

**Why:** removes the `GPU @ CPU` crash AND pins the rank decision to fp64.

### 2.3 swift redistribute twin — `_redistribute_ranks_swift_svd_plus` (~943-961)
Mirror 2.2 exactly in the **recompute** branch (**949-953**). The **cached**
branch (**936-941**) reuses `grouped_svs_cache[name][(li,e)]`, which is now an
fp64-CPU tensor built in 2.2 — so the cache producer and this consumer remain
identical dtype/device (both CPU-fp64). **Verify** that identity holds, else any
downstream `torch.equal` comparison fails on a dtype mismatch.

> **Test-fixture coupling (CRITICAL — see C1):** the producer↔consumer identity
> is not the only `torch.equal` consumer. The tier-1 precondition test
> `test_stage3_tier1.py::test_grouped_svs_cache_precondition_torch_equal`
> (**test_stage3_tier1.py:107-147**) asserts the producer's spectrum is
> `torch.equal` to an **inline recompute that hardcodes fp32**
> (`W…to(torch.float32)` at **:138**, `A…to(torch.float32)` at **:139**,
> `svdvals(W @ L_A)` at **:144**, the `torch.equal(...)` assert at **:145**).
> Once §2.2 makes the producer emit **fp64-CPU** spectra, that test FAILS on
> dtype+value (producer fp64 vs inline fp32). It **must** be patched in lockstep
> with §2.2: edit the inline recompute at **tier1:138-144** to CPU-fp64
> (`W…to(device="cpu", dtype=torch.float64)`, `A…` likewise, keep the
> `eigh`/`keep_a`/`L_A`/`svdvals` structure) so producer and inline are both
> CPU-fp64 and the assert holds bit-exact again. This is a **test-fixture edit**,
> not a production producer/consumer concern (see §9 risk row, §10 files-touched).
> Note: the *other* tier-1 test `test_grouped_svs_cache_equals_recompute`
> (**:150**) only compares integer rank dicts via `==` and survives the dtype
> change unchanged — do not confuse the two.

### 2.4 d_rank — `_group_stat` (~355-374)
**Before:**
```python
A64 = A_g.to(torch.float64)                       # CPU (A_g is CPU)
... L_A = torch.linalg.cholesky(A64 + jitter).to(torch.float32)   # CPU, → fp32
W = bank.get(e).detach().to(torch.float32)        # GPU
M = L_A @ W.T                                      # CRASH: CPU @ GPU
s = torch.linalg.svdvals(M)
```
**After:**
```python
A64 = A_g.to(device="cpu", dtype=torch.float64)
A64 = 0.5 * (A64 + A64.T)
jitter = 1e-6 * A64.diag().mean().clamp_min(1e-12) * torch.eye(
    A64.shape[0], dtype=torch.float64, device="cpu")
L_A = torch.linalg.cholesky(A64 + jitter)         # fp64, CPU — KEEP fp64 (no cast)
...
W64 = bank.get(e).detach().to(device="cpu", dtype=torch.float64)
M = L_A @ W64.T
s = torch.linalg.svdvals(M)                        # fp64, CPU — feeds eff_rank
```
- **Drop the `.to(torch.float32)` cast at line 359** for the spectrum path: the
  cast was the half-measure that loses fp64 in the svdvals. `eff_rank` at
  **392** and the `_weight`/`round()` at **476**/**502** then derive from fp64.
- `mean_s` is stored fp64; downstream swift ε* consumes the full spectrum — keep
  it fp64 there too (consistency).

**Why:** removes `CPU @ GPU` crash; makes `eff_rank` → `round()` deterministic.

### 2.4.1 item-3 disproof test — re-confirm it still holds (M2)
`test_stage3_tier1.py::test_group_stat_vs_swift_spectra_differ`
(**test_stage3_tier1.py:216-245**) shadows the exact two operators §2/§3 change.
It hardcodes precision —
`cholesky(A_g.double()+1e-6·eye.double()).float()` (**:232**),
`svdvals(L_chol @ W.T)` (**:233**) for the d_rank `_group_stat` side, and
`svdvals(W @ L_eigh)` (**:240**) for the swift side — then asserts
`not torch.allclose(s_group, s_swift)` (**:242**): the group-averaged
Cholesky-whitened spectrum is NOT close to the per-expert eigh-whitened spectrum.
- **Does the disproof still hold after §2/§3?** Yes — it compares two
  *structurally different operators* (group-avg Cholesky `L_chol @ W.T` vs
  per-expert eigh `W @ L_eigh`), and that inequality is independent of fp32 vs
  fp64 or CPU vs GPU. The precision/device change does not make them coincide;
  the test stays green and its load-bearing finding is unaffected. The plan
  states this explicitly so the change isn't assumed to silently break it.
- **But the hardcoded precision is now a latent desync (action required):** this
  test inlines `.float()` on the Cholesky factor (**:232**) and fp32 `svdvals`,
  i.e. it still models the **old** `_group_stat` fp32 path that §2.4/§3.4
  *removes*. It is a disproof test (asserts inequality), so the fp32-vs-fp64
  delta cannot flip its result — but to keep the test an honest mirror of the
  shadowed production path, **update its `_group_stat`-side recompute to CPU-fp64
  in lockstep** (drop the `.float()` at :232; keep `.double()`), matching §2.4.
  Re-run after the change and confirm `not torch.allclose` still holds. Tracked
  in §10.

### 2.5 covariance load — leave on CPU
`_load_stage2_covariance` at **512** keeps `map_location="cpu"`. **No change** —
the covariances stay CPU-resident (that is correct for the one-layer-resident
invariant, §6), and the spectra now run on CPU too, so nothing crosses devices.
Do **not** add a blanket `.to(dev)` on the covariance dict — that would pin a
GPU copy of every layer's cov (§6 memory risk).

---

## 3. Change 2 — precision split (fp64 spectra / fp32-GPU factors)

### 3.1 The empirical basis (user decision, fixed)
3-seed measurement on real shapes:
- **fp64**: CPU and GPU agree to ~1e-14 → **0 rank flips**, device-independent.
- **fp32-GPU**: flips **2–3 / 216** ranks vs the CPU golden (~1% boundary
  flips) → fragile per-device re-bless. Rejected.
- fp64 `svdvals` on consumer Blackwell ~14× slower than fp32-GPU (and slower
  than CPU) → fp64 svdvals must **not** run on the 5080. eigh/cholesky fp64 on
  H200 ~5–7× → acceptable, but we standardize on CPU-fp64 spectra for one path.

### 3.2 The clean split
| Quantity | Decides ranks? | Precision | Device |
|---|---|---|---|
| swift `eigh(A)` + `svdvals(W@L_A)` → `tail`/energy cutoff (`:807`) | YES | **fp64** | **CPU** |
| d_rank `chol(A_g)` + `svdvals(L_A@W.T)` → `eff_rank` (`:392`) → `round()` (`:502`) | YES | **fp64** | **CPU** |
| `U_k` / `V_k` low-rank factors for the chosen `k` | NO | **fp32** | **GPU** |

The factor matrices are built in `aa_svd_factor.factor_layer` (the `_aa_svd*`
calls at **637-650**), which already runs on `dev` in fp32
(`W = originals[...].to(device=dev, dtype=torch.float32)` at **629**). **That
path is unchanged by §3** — it already does fp32-GPU. The only thing the rank
decision feeds into `factor_layer` is the integer `k` (via `per_expert_ranks` /
`ranks`), which is now fp64-derived upstream.

### 3.3 Redundant-cost note (svdvals-for-k vs SVD-for-UV)
The rank decision needs only **singular values** (`svdvals`, fp64-CPU). The
factor construction needs the **full SVD** (`U,S,Vh`, fp32-GPU) at the chosen
`k`. These are two separate decompositions of (related but not identical)
operators:
- rank decision: `svdvals(W @ L_A)` (whitened, CPU-fp64) — already computed in
  the swift/d_rank allocation phase, **before** `factor_layer`.
- factor build: `_aa_svd` solves the AA-SVD problem `W·C·S⁻¹·L_B^T` (Path 1) on
  GPU-fp32 — a *different* matrix than the whitened spectrum operator, so this
  is not literally redundant work; the two phases already exist independently
  on `origin/main`. **No new redundant SVD is introduced** by this plan — the
  spectra phase and the factor phase are already distinct. The only added cost
  is fp64 (vs fp32) on the **spectra** matrices, bounded to one
  expert×matrix-type at a time. Estimated added wall-time for the full 40-layer
  run: **a few minutes** (fp64 CPU svdvals on ~`[2048×r]` operands × ~7200
  expert·matrix calls). Acceptable for a one-time factorization.

> If a future refactor ever fuses "svdvals for k" and "full SVD for U/V" into
> one decomposition, the plan would need to split them (fp64-CPU svdvals +
> fp32-GPU full-SVD) and the redundant svdvals cost would apply. That fusion
> does **not** exist on origin/main, so it is out of scope here. Flag it.

### 3.4 d_rank fp64 consistency (the `:359` cast)
`_group_stat` already does fp64 cholesky then **casts to fp32** at **359**
before `svdvals`. §2.4 **removes that cast** so the spectrum stays fp64 through
`svdvals` → `eff_rank`. This is the "fold the precision policy in consistently"
item: the cholesky was always fp64; we now stop throwing the precision away one
line later.

**MANDATORY paper-fidelity follow-up — rewrite the deviation docstring (M1).**
The module-level deviation block **"Deviation: D-drank-fp64-mixed — FP64
Cholesky, FP32 SVD"** (`d_rank_allocate.py:193-210`, with the inline narrative
at **:199-201** that the cast happens "immediately… before forming `L_A @ W.T`")
codifies the fp32 cast as a *deliberate* choice — "FP32 there is adequate and
~3× faster… Documented for spec-compliance; no refactor planned" (**:207-210**).
Once §2.4 drops the cast, that block becomes a **lie**. Per paper-fidelity
discipline, the cast must NOT be silently deleted: the deviation entry at
**:193-210** (and the inline comment at **:199-201**) MUST be **rewritten** in
the same commit, restating the new rationale — the spectrum now stays fp64
through `svdvals` to make `eff_rank`/`round()` **device-independent** (CPU-fp64
== GPU-fp64 to ~1e-14, 0 rank flips; the prior fp32-GPU path flipped 2–3/216),
which is the whole point of Tier-2 §3.1. Relabel accordingly (no longer "mixed
precision"; now "FP64 Cholesky + FP64 SVD, CPU-resident, device-independent").
This docstring rewrite is in scope for §4/§10.

---

## 4. Change 3 — zero-pad fix in `factor_layer`

### 4.1 Root cause (verified)
`factor_layer` allocates `FactoredExperts(ranks=ranks_layer)` where
`ranks_layer[name] = max_e per_expert_rank` (slot width, **522-528**). Each
expert's factors are built at its **own** `k` (≤ slot), so `U_k=(d_out,k)`,
`V_k=(k,d_in)`. `set_factors` hard-checks `U.shape == (d_out, slot)` and
`raise ValueError` (**737-744**). When per-expert ranks are non-uniform (any
`alpha_grid` length>1), `k < slot` for some expert → **crash**. The
`_aa_svd*` functions pad `k_eff→k` internally (`_aa_svd_precomputed` try-branch
**310-316** and fallback **326-332**; `_aa_svd` non-precomputed **394-400**) but
never `k→slot`. (Note the §4.2 caveat: the internal pad targets the *clamped* k,
so the returned width can itself be `< k` — see L1 handling below.)

### 4.2 Fix — pad in `factor_layer` (chosen over padding in `set_factors`)
**Decision: pad in `factor_layer`, NOT inside `set_factors`.** Reasons:
- `set_factors` is a **shared low-level contract** (EoRA `widen_rank`, Stage-2,
  golden snapshot all call it). Its hard shape-check is a deliberate guardrail;
  silently auto-padding there would mask real shape bugs in other callers
  (per the user's no-monkey-patch / minimal-impact discipline).
- the slot width `ranks_layer[name]` is local to `factor_layer`; the pad belongs
  where the slot is known.
- `effective_rank=k_eff` already flows correctly through `set_factors` (**746**)
  for honest param counting — we keep passing `k_eff`, just reshape U/V to slot.

**Before** (`aa_svd_factor.py:637-653`):
```python
U_k, V_k, rel_err, k_eff = _aa_svd(W, A, B, k, C=C, device=dev, ...)   # (d_out,k)/(k,d_in)
if k_eff < k: k_eff_clip_count[name] += 1
new_factored.set_factors(e, name, U_k, V_k, effective_rank=k_eff)      # CRASH if k<slot
```
**After:**
```python
U_k, V_k, rel_err, k_eff = _aa_svd(W, A, B, k, C=C, device=dev, ...)
if k_eff < k: k_eff_clip_count[name] += 1
slot = ranks_layer[name]
# Index by the ACTUAL returned factor width, not the requested k. `_aa_svd*`
# internally re-clamps the requested k — `k = max(1, min(k, min(d_out,d_in)))`
# (aa_svd_factor.py:285) and `k_eff = max(1, min(k, decomp.r_eff))` (:290) — so
# the returned `U_k` width can be < the requested k when the internal clamp
# fires. Using `k` in the assignment would raise on a shape mismatch in that
# edge case.
u_w = U_k.shape[1]   # actual returned column count of U_k
v_w = V_k.shape[0]   # actual returned row count of V_k (== u_w in practice)
if u_w < slot:
    # zero-pad the per-expert factors up to the layer slot width.
    # trailing zero col(U)/row(V) are inert in the forward (bmm: V row=0 → 0 → U col=0).
    U_pad = torch.zeros(U_k.shape[0], slot, device=U_k.device, dtype=U_k.dtype)
    V_pad = torch.zeros(slot, V_k.shape[1], device=V_k.device, dtype=V_k.dtype)
    U_pad[:, :u_w] = U_k
    V_pad[:v_w, :] = V_k
    U_k, V_k = U_pad, V_pad
new_factored.set_factors(e, name, U_k, V_k, effective_rank=k_eff)
```
- `effective_rank=k_eff` (NOT `slot`, NOT `k`): `k_eff` is the count of columns
  with genuine signal. The padded `[k_eff:slot)` columns are zero; honest param
  counting (`effective_ranks` at **746**) stays correct. (Note: today
  `effective_rank=k_eff` and `rank_map[...] = k` at **654** — the rank_map keeps
  recording `k`, the *requested* per-expert rank, which is what the golden pins.
  Leave `rank_map` recording `k`; do **not** change it to `slot` or `k_eff`, or
  the golden diff in §5 conflates the zero-pad fix with a rank-semantics change.)

### 4.3 Forward inertness (re-confirmed)
`FactoredExperts.forward` (`model_io.py:949-952`) computes
`bmm(bmm(x, V.T), U.T)`. A zero row in V → zero entry in the rank-`slot`
intermediate; a zero column in U → that entry maps to zero output. Padded
directions contribute **exactly 0**. The same property already underpins
EoRA `widen_rank` (**838-867**, shipped). No forward change needed.

---

## 5. Change 4 — golden re-bless (a QUALITY gate, not a blind regen)

**MUST NOT** run a bare `MOE_REGEN_GOLDEN=1` and commit. The fp64-spectra ranks
may differ from the current fp32-CPU goldens. Sequence:

### 5.1 (a) Produce + review the rank-map DIFF FIRST
- Run the byte-identical case (`test_stage3_rank_map_byte_identical`,
  `device=None` = CPU, `alpha_grid=[0.5]`) **with the §2-§3 fp64 spectra
  applied** but BEFORE overwriting the golden.
- Compare produced `rank_map.json` vs the committed
  `tests/golden/stage3/rank_map.{fp32,bf16}.json`.
- Emit a structured diff: **flip count** + every flipped `L{layer}_E{expert}_{name}`
  with `old_k → new_k`. Note: the `[0.5]` uniform path does **not** enter
  `_swift_..._search` (per test docstring **194-196**), so this case's ranks
  come from the d_rank / group-uniform allocator — the fp64 change here is the
  `eff_rank`/`round()` boundary. Expect **small** flip count (boundary cases).
- **Present this diff to a human for review BEFORE re-blessing.** Do not
  proceed on flips without sign-off.

### 5.2 (b) Re-bless the pinned byte-identical golden ONCE
After human sign-off on the diff:
- `MOE_REGEN_GOLDEN=1 pytest .../test_stage3_golden_snapshot.py::test_stage3_rank_map_byte_identical -v`
- commit the new `rank_map.fp32.json` / `rank_map.bf16.json` bytes with the diff
  summary in the commit message.

### 5.3 (c) Bless the α-grid variant (unblock the xfail)
The zero-pad fix (§4) makes the non-uniform path stop raising `ValueError`, so
`test_stage3_rank_map_alpha_variant_byte_identical` (xfail at **251-272**) now
runs.
- Flip the decorator: remove `@pytest.mark.xfail(...)` (the reason string itself
  says "then this xfail flips to a real bless").
- `MOE_REGEN_GOLDEN=1 pytest ...::test_stage3_rank_map_alpha_variant_byte_identical`
  to mint `rank_map.alpha.fp32.json` / `rank_map.alpha.bf16.json` (new files).
- Run the diff-review step on these too (these DO exercise `_swift_..._search`
  at `alpha_grid=[0,0.5,1]` → the fp64 swift cutoff at **807**).

### 5.4 (d) Device-independence assertion (the guarantee)
Add a test asserting **fp64-CPU and fp64-GPU produce identical integer ranks**
(so a future device move can't silently re-flip). Form:
- run the spectra/allocation phase on CPU and (if `torch.cuda.is_available()`)
  on GPU, assert the resulting integer rank_maps are **equal** (not byte-equal
  artifacts — the integer rank dict).
- skip the GPU leg with `pytest.mark.skipif(not cuda)` so it's green on
  CPU-only CI; on the 5080 host it runs and proves the invariant.
- This is fp64-spectra on both devices; per §3.1 fp64 agrees to ~1e-14 → 0
  flips. Because §2 standardizes spectra on **CPU**, the GPU leg here is the
  *guard* against a regression that moves spectra back onto the GPU.

---

## 6. Memory / residency analysis

- **One-layer-resident invariant preserved.** `factor_layer` lazy-loads exactly
  one layer's B-cov (`aa_svd_factor.py:534-557`, "Keeps in-memory cov bounded to
  ~one layer (~3-5 GB at bf16)") and `_load_stage2_covariance` keeps everything
  `map_location="cpu"` (**512**). §2 does **NOT** add any `.to(dev)` on the cov
  dict, so no GPU pin of all layers' covs. **Do not** broadcast the covariance
  to GPU.
- **GPU VRAM impact bounded.** The decomp is **per-(expert, matrix-type)
  transient**: `M_A=[d_out, r_A]`, `M=[d_in, d_out]`, `W=[d_out, d_in]` — never
  model-sized. With spectra on CPU (§2 decision), the GPU sees only the fp32
  factor build that already exists on origin/main (`factor_layer` offloads the
  dense expert to CPU before allocating `FactoredExperts`, **579-580**, to avoid
  double-occupancy). **Net new GPU VRAM from Tier-2: ~0** (spectra moved to CPU,
  factors unchanged).
- **CPU RAM impact.** fp64 spectra double the transient spectrum tensor vs fp32,
  but these are tiny (one expert·matrix at a time, ≤ `[2048×2048]` fp64 ≈ 32 MB
  worst case, freed each iteration). Bounded; no accumulation.

---

## 7. Ordering / sequencing

Strict order (each step independently testable):

1. **Crash-fix + device move (§2).** swift `_swift_..._search` (749-760),
   swift redistribute twin (949-953) incl. cache-dtype check, d_rank
   `_group_stat` (355-374). Verify: a GPU-resident smoke run no longer raises
   the device-mismatch `RuntimeError`.
2. **Precision split (§3).** fp64 spectra (drop the `:359` fp32 cast; fp64 in
   swift). Factors stay fp32-GPU (no change). Verify: rank decision derives from
   fp64; CPU vs GPU spectra agree to ~1e-14.
3. **Zero-pad fix (§4).** pad `k→slot` in `factor_layer` before `set_factors`.
   Verify: non-uniform per-expert path no longer raises; forward output matches
   the unpadded reference to fp tolerance (padded dirs inert).
4. **Re-bless (§5)** — LAST, and only after 1-3 land:
   (a) diff + human review → (b) re-bless byte goldens → (c) unblock α xfail +
   mint α goldens → (d) add device-independence assertion.

Re-bless must be last because the golden bytes depend on the fp64 ranks (§3) AND
the α-grid goldens depend on the zero-pad fix (§4) not crashing.

---

## 8. Testing plan (host = RTX 5080)

Constraints: fp64-GPU `svdvals` is slow on the 5080 → **CPU-fp64 spectra tests
are the fast path**; byte-identity/rank-diff goldens run on CPU (~90s).

| Test | What it proves | Where | Cost |
|---|---|---|---|
| `test_stage3_golden_snapshot.py::test_stage3_rank_map_byte_identical` (fp32,bf16) | fp64-spectra byte-identical golden | CPU | ~90s |
| `test_stage3_golden_snapshot.py::test_stage3_rank_map_alpha_variant_byte_identical` (xfail→pass) | zero-pad fix + α-grid swift-path golden | CPU | ~90s |
| `test_stage3_tier1.py::test_grouped_svs_cache_precondition_torch_equal` (inline recompute patched to CPU-fp64, §2.2/C1) | producer fp64-CPU spectrum `torch.equal` to fp64-CPU inline recompute | CPU | fast |
| `test_stage3_tier1.py::test_grouped_svs_cache_equals_recompute` | cache vs recompute rank dicts still `==` after dtype/device change (no edit needed — integer-dict compare) | CPU | fast |
| `test_stage3_tier1.py::test_group_stat_vs_swift_spectra_differ` (recompute patched to CPU-fp64, M2) | item-3 disproof `not allclose` still holds; test mirrors the changed `_group_stat` path | CPU | fast |
| NEW device-independence test (§5d) | fp64-CPU rank_map == fp64-GPU rank_map | CPU + 5080 (skipif no cuda) | CPU fast; GPU slow-but-bounded |
| NEW non-uniform zero-pad unit test | `factor_layer` with non-uniform per-expert ranks no longer raises; forward inert | `test_stage3_plugin_aa_svd.py` | fast |
| GPU-resident smoke (model on cuda) | §2 crash-fix: no device-mismatch RuntimeError | 5080 | minutes |
| existing `test_stage3_plugin_swift_svd.py` / `_d_rank.py` / `_plugin_aa_svd.py` | no regression in spectra/alloc/factor plugins | CPU | fast |

Fast iteration loop on the 5080: run all **CPU** tests first (the rank decision
+ goldens are CPU-fp64). Only the GPU-resident smoke and the GPU leg of the
device-independence test touch cuda; keep those minimal (1 layer, few experts).

---

## 9. Risks + rollback

| Risk | Mitigation | Rollback |
|---|---|---|
| fp64 ranks flip more than expected vs golden | §5a human-reviewed diff BEFORE bless; flip count surfaced | revert the spectra dtype change (back to fp32) — goldens unchanged |
| tier-1 `torch.equal` precondition **test** breaks: `test_grouped_svs_cache_precondition_torch_equal` (tier1:107-147) hardcodes an fp32 inline recompute (:138-144) and `torch.equal`-asserts (:145) against the producer, which §2.2 now emits as fp64-CPU → dtype+value mismatch | C1 mandates editing the test's inline recompute to CPU-fp64 (tier1:138-144) **in the same commit as §2.2**; this is a test-fixture edit, not just a production producer/consumer concern | revert the spectra dtype change restores fp32 producer ↔ fp32 inline; test fixture and producer move in lockstep either way |
| production cache producer↔consumer dtype/device drift apart (within `swift_svd_alpha.py`) | §2.3 explicit verify both build/read CPU-fp64 | the cache reuse is an optimization; recompute branch is the fallback (identical once both are CPU-fp64) |
| zero-pad perturbs forward | §4.3 proves inertness (same as shipped EoRA widen_rank); add forward-equivalence unit test | pad fix is local to `factor_layer`; revert restores xfail |
| someone broadcasts cov to GPU "to fix the crash" | §6 forbids it; spectra-on-CPU is the sanctioned fix | n/a |
| fp64-GPU svdvals accidentally introduced on 5080 (slow) | §2.1 mandates CPU-fp64 spectra; device-independence test guards | revert the offending `.to(dev)` |
| α-grid goldens minted on a flaky/non-deterministic run | mint on CPU (deterministic), commit bytes with diff summary | delete the new `rank_map.alpha.*` files, restore xfail |

Each of §2/§3/§4 is independently revertable. §5 (goldens) is a separate commit;
reverting it restores the prior golden bytes and the xfail decorator. No
production behavior outside Stage-3 factorization is touched.

---

## 10. Files touched (summary)

| File | Change |
|---|---|
| `stage3/plugins/swift_svd_alpha.py` | §2.2/§2.3 device+fp64 in `_swift_..._search` (749-760) & `_redistribute_..._swift_svd_plus` (949-953) |
| `stage3/plugins/d_rank_allocate.py` | §2.4/§3.4 device+fp64 in `_group_stat` (355-374), drop fp32 cast at 359, **AND rewrite the D-drank-fp64-mixed deviation docstring (193-210, inline comment 199-201) — M1** |
| `stage3/plugins/aa_svd_factor.py` | §4 zero-pad `k→slot` before `set_factors` (after 650, before 653); index by actual returned width `U_k.shape[1]`/`V_k.shape[0]` — L1 |
| `stage3/plugins/covariance_collection.py` | **no change** (cov stays CPU — §2.5/§6) |
| `utils/model_io.py` | **no change** to `set_factors`/forward (pad in caller — §4.2) |
| `tests/test_stage3_tier1.py` | **C1** edit `test_grouped_svs_cache_precondition_torch_equal` inline recompute (138-144) to CPU-fp64 in lockstep with §2.2 (so `torch.equal` at :145 holds); **M2** update `test_group_stat_vs_swift_spectra_differ` `_group_stat`-side recompute (drop `.float()` at :232 → CPU-fp64) to mirror §2.4 (disproof at :242 still holds) |
| `tests/test_stage3_golden_snapshot.py` | §5c remove xfail on α-variant; (re-bless via env var, not code) |
| `tests/golden/stage3/rank_map.{fp32,bf16}.json` | §5b re-blessed bytes |
| `tests/golden/stage3/rank_map.alpha.{fp32,bf16}.json` | §5c new α goldens |
| `tests/test_stage3_plugin_aa_svd.py` (or new) | §8 non-uniform zero-pad unit test |
| new test (§5d) | fp64-CPU == fp64-GPU rank_map device-independence |
