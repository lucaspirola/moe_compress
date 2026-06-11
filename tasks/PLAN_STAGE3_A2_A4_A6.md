# PLAN — Stage 3 A2 + A4 + A6 (quality-neutral speedups)

Branch: `plan/stage3-a2-a4-a6` (off `main`). Loop 1 (A7 capture hook + A1 windowed single-pass cov) already LANDED on `main` ad849a2; this builds on it. All three opts are pure perf refactors — 1-GPU output **byte-identical** (or within documented fp tolerance). B-list (B1 proxy-alpha-selection, B2 coarser grid) is OFF-LIMITS.

## Invariants / non-goals (from the fidelity audit — apply to all three)
- **N-1** Do NOT drop fp32 `eigh` to bf16/fp16 (A5 guardrail). `_precompute_eigh` casts B→fp32 at `aa_svd_factor.py:207`; A2 caches the *result object*, never changes its dtype.
- **N-2** Do NOT change alpha SELECTION. A2 must not alter which alpha wins, only how fast each candidate is factored. No B1/B2.
- **N-3** 1-GPU path stays byte-identical: `test_stage3_golden_snapshot.py` must pass unchanged for all three.
- **N-4** Multi-GPU: correct for ANY N (consistent with `project_multigpu_stage3_landed`). A2 is N-agnostic math hoisting; A4 must work under DP cov replicas (`_cov_replica_worker`); A6 interacts directly with sharding.
- **N-5** Every opt ships an equivalence test (see Test Plan).

---

## A2 — cache `_EighDecomp` across the 11 alpha candidates

### (1) Current code (verified)
- `_swift_svd_plus_alpha_search_validation` candidate loop: `swift_svd_alpha.py:636-674` — `for idx, alpha in enumerate(alpha_grid):` calls `_factor_model_at_ranks(...)` once per alpha at **line 650**.
- `_factor_model_at_ranks` per-(layer,expert) eigh: `swift_svd_alpha.py:436-451` — inside `for e in range(...)` it calls `_precompute_eigh(B_shared, A_shared, C_shared, ...)` at **446-449**, recomputed on every candidate.
- `_precompute_eigh` depends only on (B, A, C); `A` is `del`-eted at `aa_svd_factor.py:230`; result is W-independent and k-independent (k is consumed only by `_aa_svd_precomputed`, `aa_svd_factor.py:265-285`).
- Across all 11 candidates, `B_acc`/`A_cov`/`C_acc` are the same spill files for a given layer (loaded at `swift_svd_alpha.py:404-406`); only `k`/`per_expert_ranks` changes. So `gate_up_decomp` is identical across candidates.
- Precedent: spectral-proxy path already caches via `grouped_svs` (`swift_svd_alpha.py:737, 778`) threaded into `_redistribute_ranks_swift_svd_plus` (`grouped_svs_cache`, line 907) — mirror that pattern.

### (2) Precise change — chosen design: per-layer eigh-decomp SPILL cache
The candidate loop is **alpha-major** (outer) and factors the whole model per alpha (PPL needs the full model at one alpha), so we cannot reorder to layer-major. Holding every layer's decomps in RAM across candidates does NOT fit (≈16 MB/expert × ~200 experts × 40 layers → hundreds of GB). Therefore **spill the decomps the same way cov is spilled** — one layer resident at a time, matching the existing cov-spill residency model (`swift_svd_alpha.py:404/484`):

1. Add optional param `gate_up_decomp_cache_dir: str | None = None` to `_factor_model_at_ranks` (`swift_svd_alpha.py:373`). Default `None` ⇒ today's behavior (recompute) for any direct caller / test.
2. In the per-expert loop, replace the unconditional `_precompute_eigh` (444-451) with a cache-aware path keyed `(layer_idx, expert_idx)`:
   - On the FIRST candidate (cache miss): compute `_precompute_eigh`, then serialize its small back-solve tensors (`rhs`, `rhs_pinv`, `eigvals_keep`, `eigvecs_keep`, `inv_sqrt`, `r_eff`) to `{cache_dir}/layer_{L}.pt` (one file per layer, holding that layer's `{expert: decomp}`).
   - On candidates 1..10 (cache hit): load that layer's `.pt` and reuse. Only one layer's decomps are resident at a time (loaded when `_factor_model_at_ranks` loads that layer's cov, freed when it unloads — `swift_svd_alpha.py:404/484-486`).
   - **ValueError sentinel:** an expert whose B has no eigenvalue above the floor raises `ValueError` at `_precompute_eigh` (`aa_svd_factor.py:216`); current code `pass`es (451) → full `_aa_svd` fallback. The cache MUST persist `None` for that (layer,expert) key (computed-but-failed) so candidates 1..10 take the same fallback and do NOT retry the eigh. Distinguish "absent" (not computed) from "present→None" (computed, failed).
3. In `_swift_svd_plus_alpha_search_validation`, create a temp `cache_dir` (e.g. under the existing Stage-3 spill root) BEFORE the `for alpha` loop (~636) and thread it into the `_factor_model_at_ranks` call at 650. `rmtree` it in a `finally` and from the orchestrator Stage-3 spill cleanup (`orchestrator.py:862-868`). Place it OUTSIDE the path the cov-spill resume logic scans.

*(A2 touches ONLY the candidate-validation path. The main factoring loop `AaSvdFactorPlugin.factor_layer` already computes each decomp exactly once — untouched. A2 is inert when `validation_samples == 0`, where the spectral-proxy path already caches via `grouped_svs`.)*

Fallback if profiling shows disk-load dominates: an in-RAM single-layer cache (hold only the current layer's decomps) — zero cross-candidate reuse, so prefer the spill design which delivers the ~10×.

### (3) Why quality-neutral (math)
`_precompute_eigh(B,A,C)` is a deterministic function of (B,A,C) only (`aa_svd_factor.py:187-262`); `A` is `del`'d at 230 so it cannot influence output; storage_dtype/noise-floor are identical across candidates. The eigh of a fixed fp32-symmetrized B is deterministic on a fixed device. fp32 serialize→deserialize is lossless. Therefore cached decomp == recomputed decomp bit-for-bit ⇒ `_aa_svd_precomputed(W, decomp, k)` returns identical U_k/V_k ⇒ identical factored model ⇒ identical PPL per candidate ⇒ **identical winning alpha** (N-2 satisfied).

### (4) Equivalence test
`test_a2_eigh_cache_byte_identical` (new): build a tiny fused-experts model; run `_factor_model_at_ranks` twice for two different rank allocations — once with `gate_up_decomp_cache_dir=None` (recompute) and once threading a shared spill dir — and assert the installed `FactoredExperts` U/V tensors are `torch.equal` per (layer,expert,matrix). Plus a `_EighDecomp` round-trip test: serialize→load, assert every field `torch.equal`. Mirror `test_a1_windowed_equals_perlayer` (`test_multigpu_stage3.py:578-602`).

### (5) Risks / edge cases
- ValueError sentinel (above) — must cache `None`, not skip the key.
- **storage_dtype consistency:** a single search uses one storage_dtype (sourced from the same `B_cov_dtype`); the cache key need not include it but assert/document the single-dtype assumption.
- **Resume/crash:** the eigh cache dir is ephemeral, rebuilt on the candidate-0 pass; safe to `rmtree` on any exit; keep it clear of the cov-spill resume scan path.

---

## A4 — vectorize the per-token cross-cov Python loop

### (1) Current code (verified)
- Teacher hook builds `{tidx: row}` dict: `covariance_collection.py:439-463` — `_teacher_input_cb`, inner `for i, tidx in enumerate(token_idx.tolist()): _teacher_hidden[key][tidx] = det[i]` (462-463).
- Student `input_cb` stacks rows one-by-one: `covariance_collection.py:465-522` — the `for i, tidx in enumerate(token_idx):` loop (500-503) appends to `pre_vecs`/`post_vecs`, then `torch.stack` (505-506), then `cross = X_pre.T @ X_post` (514), then `C_acc.update_cross(...)` (520-522).
- The PERF(MEDIUM-2) comment naming this the dominant CPU cost is at **481-485**.
- `token_idx` semantics (verified in `activation_hooks.py:1624, 1663-1669`): per-(layer,expert) flattened token positions from `torch.where(mask[e])`; `sel = hidden_states[token_idx]` is the row gather. The teacher's layer-input hidden state is identical across that layer's experts pre-routing, so the dict is effectively `position → layer_input_row`.
- `_teacher_hidden` is keyed by layer, cleared per BATCH (`covariance_collection.py:621`), holding all G window layers for the current batch.

### (2) Precise change — one dense per-layer tensor per batch
1. In `_teacher_input_cb`, on first dispatch for a layer in a batch, lazily allocate a dense `teacher_dense[li]` of shape `[T, d_in]` (pre-sized; `T = batch.shape[0]*batch.shape[1]`, known in `_collect_covariances`'s batch loop at `covariance_collection.py:615-626`, threaded into the closure), fp32, on `det.device`, plus a per-layer boolean `filled` mask `[T]` (default False). Scatter: `teacher_dense[li].index_copy_(0, token_idx.to(device), det)` and set `filled[token_idx]=True`. Replaces the Python `for i, tidx` loop (462-463).
2. In `input_cb`, replace the per-token gather (500-506) with a single `index_select`. To exactly reproduce the current `if tidx in teacher_store` skip (501), select only positions present in `filled`: `keep = filled[token_idx]; sel_idx = token_idx[keep]`; `X_pre = teacher_dense[li].index_select(0, sel_idx.to(tgt_device))`; `X_post = det_post[keep]`. `cross = X_pre.T @ X_post` (514) is **unchanged**.
3. Clear `teacher_dense` (and `filled`) per batch at the same point as `_teacher_hidden.clear()` (621).

### (3) Why quality-neutral (math)
The accumulated quantity is `C += X_pre.T @ X_post` over matched positions (514, 520-522). The dict path gathers matched rows in student-`token_idx` order; the dense path `index_select(0, sel_idx)` gathers the **same rows in the same order**. `X_post` rows are byte-identical. The matched-positions set is preserved by the `filled` mask reproducing `if tidx in teacher_store`. Gram is order-independent anyway, but we preserve order for bit-identity ⇒ **C identical (atol=0 expected; contractual bound rtol≤1e-6 to allow GPU `index_select` fp re-association).**

### (4) Equivalence test
`test_a4_cross_cov_dense_equals_dict` (new, `test_multigpu_stage3.py`, reusing the cross-cov harness): run `_collect_covariances` with cross-cov on a tiny model twice — legacy dict path (gate behind a `cov_cross_impl="dict"` kwarg, or the `cov_capture_mode="instrument"` legacy hook that still uses the dict) vs the new dense path — assert `torch.equal` on every C key (mirror lines 598-602). Assert for G∈{1,2,N} (window independence) and under **multiple simultaneously-hooked layers (G≥2)** — the explicit "multiple layers hooked" requirement, covered by per-layer `teacher_dense[li]`.

### (5) Risks / edge cases
- **Token-position alignment is THE correctness risk.** Both `token_idx` (teacher scatter, student gather) come from `torch.where(mask[e])` over the same `[T]` flattened axis (`activation_hooks.py:1624`), so they share the position space — the test is the proof, not the argument.
- **Unfilled positions:** the `filled` mask matches the current skip-if-absent (501); zero rows are never selected.
- **DP replicas (N-4):** `_cov_replica_worker` (`covariance_collection.py:708-816`) calls `_collect_covariances` per replica over a disjoint shard; `teacher_dense` is replica-local (same contract as `_teacher_hidden`). The cross-replica reduce (`_reduce_spilled_cov_dirs`) sums finalized Grams, untouched. A4 is replica-safe by construction.
- **Cross-device (sharding):** dense path gathers on the teacher device then a single `.to(tgt_device)` on the `[n_tok, d]` block — one D2D copy instead of n_tok, same values.
- **Memory:** dense `[T, d_in]` fp32 per window layer per batch, cleared per batch; for top-k MoE nearly all positions are dispatched so footprint ≈ the sparse dict. Document the G×T×d bound.

---

## A6 — bump cov `batch_size` on the sharded run (config-level, opt-in)

### (1) Current code (verified)
- In-process path: `bcov_batch_size = int(s3.get("batch_size", 1))` at `orchestrator.py:194`; used to build `batches` (195) and logged (461-468).
- DP replica path: `batch_size = int(config["stage3_svd"].get("batch_size", 1))` at `covariance_collection.py:761`, used at 762.
- The sharding "frees activation headroom for a larger batch_size … MEASURED on the first sharded run" comment is at `orchestrator.py:201-210`.
- `_resolve_cov_window` already has the VRAM-probe precedent (`covariance_collection.py:297-348`) — the template for any auto/measured sizing.

### (2) Precise change — expose the knob, default preserves today's value
1. Add a dedicated config key `stage3_svd.cov_batch_size` (distinct from `stage3_svd.batch_size`). Resolution: `cov_batch_size = int(s3.get("cov_batch_size", s3.get("batch_size", 1)))`. **Default == today's `batch_size`** ⇒ existing golden byte-identical (N-3). Apply at both sites: `orchestrator.py:194` and the replica worker `covariance_collection.py:761`.
   - Why a separate key: `stage3_svd.batch_size` semantics are overloaded across the stage; a cov-specific key lets operators raise cov throughput on a sharded box without touching any other batched path. Absent ⇒ inherits `batch_size` ⇒ no behavior change.
2. **Auto/measured raise — opt-in and gated.** `stage3_svd.cov_batch_size: "auto"` handling that, ONLY when >1 GPU is visible AND the model is sharded, probes free VRAM via `torch.cuda.mem_get_info()` on the hot device and picks a **conservative** bs from a measured per-sample activation cost (GB/sample, not a fixed multiplier — `feedback_batch_size_tune_by_unit`), keeping the same 25% headroom reserve as `_resolve_cov_window` (`covariance_collection.py:341`). On a single GPU or CPU, `"auto"` degrades to the inherited `batch_size` (NO raise) — 1-GPU golden untouched.
3. **Lowest-risk landing:** ship step 1 (explicit, default-preserving knob) plus the `"auto"` resolver **wired but returning the inherited value until a real ≥2-GPU box measures the per-sample cost** (Deferred section). Do NOT hardcode an aggressive bs; do NOT silently change any golden.

### (3) Why quality-neutral (math)
`InputCovarianceAccumulator.update` accumulates `Σ_token x x^T` as a running fp32 sum (`covariance_collection.py:206-210` + `activation_hooks.py` finalize). The sum is over per-token outer products; batch grouping changes neither which tokens nor their per-key reduction order (the same fp32-sum-stability property A1's windowing relies on — `test_a1_window_sizes_consistent`). ⇒ **covariance identical across batch_size values.**
- Document the caveat: this is the SAME fp32-sum-stability claim as A1. If a future accumulator switches to a pairwise/tree reduction tracking batch boundaries, re-check. Today it holds.

### (4) Equivalence test
`test_a6_cov_batch_size_invariant` (new, `test_multigpu_stage3.py`): run `_collect_covariances` (B-only and cross) on a tiny model with the same calibration batched at bs∈{1,2,4}, assert `torch.equal` (or `allclose rtol≤1e-6`) on every B and C key — the kind of test A1 has (`test_a1_window_sizes_consistent`, 605-623). Plus a resolver unit test: `cov_batch_size` absent ⇒ inherits `batch_size`; explicit int passes through; `"auto"` on CPU/1-GPU ⇒ inherited value (no raise) — mirror `test_resolve_cov_window_config` (626-643).

### (5) Risks / edge cases
- **Golden drift (N-3):** the ONLY way A6 changes the golden is a default change — it does not (`cov_batch_size` defaults to `batch_size`). Resolver test asserts this.
- **Different cov per bs would be a BUG:** the invariance test is the gate; if `allclose` fails beyond rtol≤1e-6, A6 is not quality-neutral and the auto-raise must NOT ship.
- **Sharding interaction:** bs>1 grows the hot-layer activation peak linearly; the conservative auto-resolver caps it; off-box it returns the inherited value (Deferred).
- **DP replica parity:** the replica worker reads the SAME `cov_batch_size` key so in-process and DP agree; the cross-replica Gram sum is bs-independent.

---

## Build sequence (checklist)
- [ ] **A4 first** (most self-contained, no config surface): dense-tensor teacher capture + `index_select` student path in `covariance_collection.py:439-522`, keep dict path reachable behind a kwarg for the equivalence test. Add `test_a4_cross_cov_dense_equals_dict` (+ G∈{1,2,N}, multi-layer). Run `test_multigpu_stage3.py` + cross-cov tests + golden.
- [ ] **A6 second** (pure config): `cov_batch_size` resolver (default-preserving) at `orchestrator.py:194` + `covariance_collection.py:761`; wire `"auto"` resolver returning inherited value off-box. Add `test_a6_cov_batch_size_invariant` + resolver unit test. Run golden.
- [ ] **A2 last** (most involved — spill-cache): `gate_up_decomp_cache_dir` param to `_factor_model_at_ranks` (`swift_svd_alpha.py:373/444-451`) + per-layer eigh-spill in `_swift_svd_plus_alpha_search_validation` (~636/650) + cleanup in `finally` and `orchestrator.py:862-868`. Add `test_a2_eigh_cache_byte_identical` + `_EighDecomp` round-trip test. Run swift_svd + golden.
- [ ] **Full suite** + `test_stage3_golden_snapshot.py` byte-identical gate after all three.

## Test plan (summary)
| Opt | New test | Assertion | Harness mirror |
|-----|----------|-----------|----------------|
| A2 | `test_a2_eigh_cache_byte_identical`, `_EighDecomp` round-trip | factored U/V `torch.equal` cache-on vs cache-off; decomp fields `torch.equal` after serialize/load | `test_a1_windowed_equals_perlayer` |
| A4 | `test_a4_cross_cov_dense_equals_dict` (B-only + cross, G∈{1,2,N}, multi-layer) | C keys `torch.equal` (atol=0; contract rtol≤1e-6) dict vs dense | `test_a1_window_sizes_consistent` / `_collect_windowed` |
| A6 | `test_a6_cov_batch_size_invariant` + resolver unit | B/C keys `torch.equal` across bs∈{1,2,4}; default inherits `batch_size` | `test_a1_window_sizes_consistent` / `test_resolve_cov_window_config` |
| all | existing `test_stage3_golden_snapshot.py` | unchanged, byte-identical | — |

## Deferred to GPU (real ≥2-GPU box only)
- **A6 measured auto-raise:** on the first live sharded run, measure per-sample activation GB on the hot-layer device (`torch.cuda.mem_get_info` before/after a known bs), then enable `"auto"`/set `cov_batch_size` to the conservative value. CPU/CI tests prove bs-invariance; only the VRAM-bound ceiling needs a real multi-GPU box. Do NOT hardcode an aggressive bs before this.
- **A4 under live DP replicas + cross-device sharding:** tiny-model tests cover the dense-vs-dict math and multi-layer windows on CPU; the cross-GPU `index_select(...).to(tgt_device)` path under `device_map=balanced` and per-replica `teacher_dense` lifetime validate end-to-end only on ≥2 GPUs.
- **A2 wall-clock ~10× on the alpha search:** byte-identity is CPU-testable; the actual ~10× (≈31 min → few min) is observable only on the real H200 alpha search. Time the disk-load-vs-eigh tradeoff there; if load dominates, switch to the in-RAM single-layer variant.
