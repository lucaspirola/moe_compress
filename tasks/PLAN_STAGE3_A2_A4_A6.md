# PLAN — Stage 3 A2 + A4 + A6 (quality-neutral speedups)

Branch: `plan/stage3-a2-a4-a6` (off `main`). Loop 1 (A7 capture hook + A1 windowed single-pass cov) already LANDED on `main` ad849a2; this builds on it. All three opts are pure perf refactors — 1-GPU output **byte-identical** (or within documented fp tolerance). B-list (B1 proxy-alpha-selection, B2 coarser grid) is OFF-LIMITS.

**File-path key** (all cites below use the bare filename + line; full paths under `max_quality/src/moe_compress/`):
- `swift_svd_alpha.py`, `aa_svd_factor.py`, `covariance_collection.py` → `stage3/plugins/`
- `orchestrator.py` → `stage3/orchestrator.py`
- `activation_hooks.py`, `calibration.py` → `utils/`
- tests → `max_quality/tests/test_multigpu_stage3.py`

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

1. Add optional param `gate_up_decomp_cache_dir: str | None = None` to BOTH `_factor_model_at_ranks` (`swift_svd_alpha.py:373`) AND `_swift_svd_plus_alpha_search_validation` (`swift_svd_alpha.py:556-573` — its signature currently has NO `artifacts_dir`, so the cache path must be THREADED IN; the caller `select_alpha` at `:1136` supplies it). Default `None` ⇒ today's behavior (recompute) for any direct caller / test.
2. In the per-expert loop, replace the `if B_shared is not None:` guard + `try/except ValueError: pass` block (444-451 — the cache-aware path wraps the guard+try, not line 451 literally) with a cache-aware path keyed `(layer_idx, expert_idx)`:
   - On the FIRST candidate (cache miss): compute `_precompute_eigh`, then serialize the COMPLETE `_EighDecomp` field set — verified `aa_svd_factor.py:178-184` = exactly `eigvals_keep, eigvecs_keep, inv_sqrt, rhs, rhs_pinv, r_eff` — to `{cache_dir}/layer_{L}.pt` (one file per layer, holding that layer's `{expert: decomp_or_None}`). gate_proj and up_proj SHARE one `gate_up_decomp` per expert (444-451), so the `(layer,expert)` key is correct (one decomp covers both).
   - On candidates 1..10 (cache hit): load that layer's `.pt` and reuse. Only one layer's decomps are resident at a time (loaded when `_factor_model_at_ranks` loads that layer's cov, freed when it unloads — `swift_svd_alpha.py:404/484-486`).
   - **ValueError sentinel:** an expert whose B has no eigenvalue above the floor raises `ValueError` at `_precompute_eigh` (`aa_svd_factor.py:216`); current code `pass`es (451) → full `_aa_svd` fallback. The cache MUST persist `None` for that (layer,expert) key (computed-but-failed) so candidates 1..10 take the same fallback and do NOT retry the eigh. Distinguish "absent" (not computed) from "present→None" (computed, failed).
   - **Cache-validity invariant (review H1):** `rhs` is `(B,C)`-dependent (Path-1 `C@eigvecs` vs Path-3). The cache is valid ONLY because `(B,C)`-PRESENCE is identical across all 11 candidates (same spill files, `swift_svd_alpha.py:405-406`). ASSERT/document this invariant at the cache site; a future change that made C-presence candidate-dependent would silently corrupt the cache.
3. **Cache-dir location + cleanup (review H2):** create the cache dir under `artifacts_dir` at a FIXED name (e.g. `{artifacts_dir}/_stage3_alpha_eigh_cache`) BEFORE the `for alpha` loop (~636), thread it into the `_factor_model_at_ranks` call at 650. `artifacts_dir` is available via `run_ctx` (orchestrator:252) so `select_alpha` (`swift_svd_alpha.py:1136`) supplies it through the threaded signature. Cleanup at THREE sites with the shared fixed path:
   - (a) **Stale-cache guard at search ENTRY (review M-new) — the load-bearing one:** UNCONDITIONALLY `rmtree`+recreate the cache dir at the top of `_swift_svd_plus_alpha_search_validation`, BEFORE the alpha loop, so a stale `_stage3_alpha_eigh_cache` left by a hard-killed prior run (SIGKILL/OOM where the `finally` never ran) can NEVER be read as a false hit. This is critical because the H1 cache-validity invariant only holds WITHIN one run; a cross-run stale file could carry decomps from a different rank/dtype/model config and would be silently loaded. Entry-rmtree closes this regardless of `--no-resume`.
   - (b) `rmtree` in a `finally` inside `_swift_svd_plus_alpha_search_validation` (covers normal+exception exit of the search).
   - (c) add the fixed name to the orchestrator Stage-3 spill-cleanup block (`orchestrator.py:862-868`, today only `_stage3_bcov_partial`/`_stage3_ccov_partial`) AND to the `no_resume` early-cleanup block (`orchestrator.py:227-232`) as crash backstops. Place the dir OUTSIDE the path the cov-spill resume logic scans.

*(A2 touches ONLY the candidate-validation path. The main factoring loop `AaSvdFactorPlugin.factor_layer` already computes each decomp exactly once — untouched. A2 is inert when `validation_samples == 0`, where the spectral-proxy path already caches via `grouped_svs`.)*

**Break-even gating (review M3):** the spill writes ~`eigvecs_keep` 2048×2048×4 ≈ 16 MB/expert × ~200 experts ≈ 3.2 GB/layer on candidate 0, re-read 10× → ~32 GB I/O per layer × 40 ≈ ~1.3 TB reads across the search. The "~10×" assumes eigh cost ≫ I/O; on H200 fp32 eigh this is plausible but NOT guaranteed. So gate the spill on a MEASURED break-even (Deferred). The in-RAM single-layer fallback gives ZERO cross-candidate reuse here (the loop is alpha-major and cannot be reordered — PPL needs the whole model factored per alpha), so it is not a real alternative; if the spill loses on I/O the honest answer is A2 yields little and should be dropped, not silently shipped.

### (3) Why quality-neutral (math)
`_precompute_eigh(B,A,C)` is a deterministic function of (B,A,C) only (`aa_svd_factor.py:187-262`); `A` is `del`'d at 230 so it cannot influence output; storage_dtype/noise-floor are identical across candidates. The eigh of a fixed fp32-symmetrized B is deterministic on a fixed device. fp32 serialize→deserialize is lossless. Therefore cached decomp == recomputed decomp bit-for-bit ⇒ `_aa_svd_precomputed(W, decomp, k)` returns identical U_k/V_k ⇒ identical factored model ⇒ identical PPL per candidate ⇒ **identical winning alpha** (N-2 satisfied).

### (4) Equivalence test
`test_a2_eigh_cache_byte_identical` (new): build a tiny fused-experts model; run `_factor_model_at_ranks` twice for two different rank allocations — once with `gate_up_decomp_cache_dir=None` (recompute) and once threading a shared spill dir — and assert the installed `FactoredExperts` U/V tensors are `torch.equal` per (layer,expert,matrix). Plus a `_EighDecomp` round-trip test: serialize→load, assert every field `torch.equal`. Mirror `test_a1_windowed_equals_perlayer` (`test_multigpu_stage3.py:578-602`).

### (5) Risks / edge cases
- ValueError sentinel (above) — must cache `None`, not skip the key.
- **storage_dtype consistency:** a single search uses one storage_dtype (sourced from the same `B_cov_dtype`); the cache key need not include it but assert/document the single-dtype assumption.
- **Resume/crash:** the eigh cache dir is ephemeral. Candidate 0 only WRITES on cache-miss, so a stale cross-run file would otherwise be read as a false hit — the entry-rmtree (step 3a) is what makes it safe. Keep it clear of the cov-spill resume scan path. Backstopped by the orchestrator 862-868 + no_resume 227-232 cleanups.

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
2. In `input_cb`, replace the per-token gather (500-506) with a single `index_select`. To exactly reproduce the current `if tidx in teacher_store` skip (501), select only positions present in `filled`: `keep = filled[token_idx]; sel_idx = token_idx[keep]`; `X_pre = teacher_dense[li].index_select(0, sel_idx.to(tgt_device))`; `X_post = det_post[keep]`. Boolean masking is order-preserving and `index_select(0, sel_idx)` gathers in `sel_idx` order, so rows align exactly with the current in-order loop. `cross = X_pre.T @ X_post` (514) is **unchanged**. Thread the token count as `n_tokens=int(keep.sum().item())` into `update_cross` (review L1 — current code passes `len(pre_vecs)` at 521, which no longer exists; the count feeds persisted `_gpu_token_count` metadata, not the Gram, but must stay correct).
3. Clear `teacher_dense` (and `filled`) per batch at the same point as `_teacher_hidden.clear()` (621).
   - **`index_copy_` repeated-index note (review M1):** scatter via `index_copy_(0, token_idx, det)`; PyTorch leaves repeated-index behavior unspecified, but `token_idx` from `torch.where(mask[e])` is UNIQUE within a single expert dispatch, and across experts the teacher's pre-routing layer-input row at a position is identical — so no real duplicate-write hazard. Document the reliance on per-expert index uniqueness.

### (3) Why quality-neutral (math)
The accumulated quantity is `C += X_pre.T @ X_post` over matched positions (514, 520-522). The dict path gathers matched rows in student-`token_idx` order; the dense path `index_select(0, sel_idx)` gathers the **same rows in the same order**. `X_post` rows are byte-identical. The matched-positions set is preserved by the `filled` mask reproducing `if tidx in teacher_store`. Gram is order-independent anyway, but we preserve order for bit-identity ⇒ **C identical (atol=0 expected; contractual bound rtol≤1e-6 to allow GPU `index_select` fp re-association).**

### (4) Equivalence test
`test_a4_cross_cov_dense_equals_dict` (new, `test_multigpu_stage3.py`, reusing the cross-cov harness): run `_collect_covariances` with cross-cov on a tiny model twice — legacy dict path vs the new dense path — assert `torch.equal` on every C key (mirror lines 598-602). **The dict path MUST be retained behind a `cov_cross_impl="dict"` kwarg purely for this test (review H3):** the `cov_capture_mode="instrument"` branch (`covariance_collection.py:586-591`) calls the SAME `input_cb`/`_teacher_input_cb` closures as `capture` — there is NO separate dict implementation in the instrument branch, so it would exercise the NEW dense code, not a legacy dict. The ONLY valid equivalence harness is keeping the old loop reachable behind the kwarg. Assert for G∈{1,2,N} (window independence) and under **multiple simultaneously-hooked layers (G≥2)** — the explicit "multiple layers hooked" requirement, covered by per-layer `teacher_dense[li]`.

### (5) Risks / edge cases
- **Token-position alignment is THE correctness risk.** Both `token_idx` (teacher scatter, student gather) come from `torch.where(mask[e])` over the same `[T]` flattened axis (`activation_hooks.py:1624`), so they share the position space — the test is the proof, not the argument.
- **Unfilled positions:** the `filled` mask matches the current skip-if-absent (501); zero rows are never selected.
- **DP replicas (N-4):** `_cov_replica_worker` (`covariance_collection.py:708-816`) calls `_collect_covariances` per replica over a disjoint shard; `teacher_dense` is replica-local (same contract as `_teacher_hidden`). The cross-replica reduce (`_reduce_spilled_cov_dirs`) sums finalized Grams, untouched. A4 is replica-safe by construction.
- **Cross-device (sharding):** dense path gathers on the teacher device then a single `.to(tgt_device)` on the `[n_tok, d]` block — one D2D copy instead of n_tok, same values.
- **Memory — A1×A4×A6 COMPOUND (review M2):** dense `[T, d_in]` fp32 is allocated for EVERY one of the G window layers simultaneously (`_teacher_hidden` holds all G layers, 437/621), and `T = batch_tokens` grows linearly with A6's raised `cov_batch_size`. So peak = `G · T · d_in · 4 = G · cov_bs · seq · d_in · 4`. Unlike the dict (which stored only DISPATCHED positions), the dense tensor allocates ALL T rows per layer even where only a subset is dispatched. The three opts compound: A1 raises G, A6 raises bs. The A6 auto-resolver's VRAM check MUST account for this dense-teacher peak (not just the forward activation), or A6 can OOM the hot device precisely when it raises bs. Cross-check the compounded peak against the same headroom A6 reserves; document the `G·cov_bs·seq·d_in·4` bound as a constraint the resolver enforces.

---

## A6 — bump cov `batch_size` on the sharded run (config-level, opt-in)

### (1) Current code (verified)
- In-process path: `bcov_batch_size = int(s3.get("batch_size", 1))` at `orchestrator.py:194`; used to build `batches` (195) and logged (461-468).
- DP replica path: `batch_size = int(config["stage3_svd"].get("batch_size", 1))` at `covariance_collection.py:761`, used at 762.
- The sharding "frees activation headroom for a larger batch_size … MEASURED on the first sharded run" comment is at `orchestrator.py:201-210`.
- `_resolve_cov_window` already has the VRAM-probe precedent (`covariance_collection.py:297-348`) — the template for any auto/measured sizing.

### (2) Precise change — expose the knob, default preserves today's value
1. Add a dedicated config key `stage3_svd.cov_batch_size` (distinct from `stage3_svd.batch_size`). Resolution: `cov_batch_size = int(<s3>.get("cov_batch_size", <s3>.get("batch_size", 1)))`. **Default resolves to today's `batch_size` value** ⇒ golden untouched (N-3, scoped per (5)). Apply at both sites, each with its OWN local var (review L2): `orchestrator.py:194` uses `s3` (already `config["stage3_svd"]`); the replica worker `covariance_collection.py:761` uses `config["stage3_svd"]`. Write the resolver expression per-site, do not assume a shared `s3`.
   - Why a separate key: `stage3_svd.batch_size` semantics are overloaded across the stage; a cov-specific key lets operators raise cov throughput on a sharded box without touching any other batched path. Absent ⇒ inherits `batch_size` ⇒ no behavior change.
2. **Auto/measured raise — opt-in and gated.** `stage3_svd.cov_batch_size: "auto"` handling that, ONLY when >1 GPU is visible AND the model is sharded, probes free VRAM via `torch.cuda.mem_get_info()` on the hot device and picks a **conservative** bs from a measured per-sample activation cost (GB/sample, not a fixed multiplier — `feedback_batch_size_tune_by_unit`), keeping the same 25% headroom reserve as `_resolve_cov_window` (`covariance_collection.py:341`). On a single GPU or CPU, `"auto"` degrades to the inherited `batch_size` (NO raise) — 1-GPU golden untouched.
3. **Lowest-risk landing:** ship step 1 (explicit, default-preserving knob) plus the `"auto"` resolver **wired but returning the inherited value until a real ≥2-GPU box measures the per-sample cost** (Deferred section). Do NOT hardcode an aggressive bs; do NOT silently change any golden.

### (3) Why quality-neutral (NOT byte-identical — fp-reassociation tolerance) — CORRECTED per review C1
**A6 is quality-neutral in the fp-tolerance sense, NOT byte-identical. A1's `torch.equal` invariance does NOT transfer to A6.** Why the distinction:
- A1 holds `batches` FIXED and only changes how many layers are hooked per forward, so each expert's per-forward `flat` token matrix is byte-identical → `flat.T @ flat` is the identical GEMM → `torch.equal` (that is why `test_a1_window_sizes_consistent` can assert equality).
- A6 changes `batch_size`, which re-partitions which tokens land in the SAME `flat.T @ flat` GEMM vs separate `cur.add_(cov)` partials (`InputCovarianceAccumulator.update`, `activation_hooks.py:1003-1020`; `iter_batches` partitions by sequence count, `calibration.py:317`). The token SET routed to an expert is unchanged, but the float SUMMATION GROUPING differs: more/fewer rows inside each GEMM's implementation-defined internal reduction, and a different number/order of `add_` chain steps. So `Σ x x^T` is the same MATHEMATICAL object but computed with different fp reassociation ⇒ differs at the fp32-rounding level. This is the same class of non-determinism as changing hardware/GEMM backend — far below the signal, but NOT bit-equal.

⇒ The contract for A6 is **`allclose` within a documented tolerance**, NOT `torch.equal`. The tolerance is justified empirically (measure the cross-bs delta on a real run) — it is fp-reassociation noise, not a quality trade, but it must be NAMED as tolerance not byte-identity.

### (4) Equivalence test
`test_a6_cov_batch_size_close` (new, `test_multigpu_stage3.py`): run `_collect_covariances` (B-only and cross) on a tiny model with the same calibration batched at bs∈{1,2,4}, assert `torch.allclose(rtol=..., atol=...)` (tolerance picked from a measured CPU-fp32 run — expected tiny; do NOT assert `torch.equal`) on every B and C key. NOTE this differs deliberately from the A1 `torch.equal` tests — A6 is reassociation-tolerant, not bit-exact. Plus a resolver unit test: `cov_batch_size` absent ⇒ inherits `batch_size`; explicit int passes through; `"auto"` on CPU/1-GPU ⇒ inherited value (no raise) — mirror `test_resolve_cov_window_config` (626-643).

### (5) Risks / edge cases
- **Golden safety (N-3) — SCOPED per review C2:** the golden is safe ONLY because the DEFAULT `cov_batch_size` resolves to the unchanged `batch_size` value (the golden was generated at that value). This is "default preserves the value", NOT "A6 is byte-neutral". Any operator who sets `cov_batch_size != batch_size` (or enables `"auto"` raising it on a multi-GPU box) WILL perturb covariance beyond byte-identity (per (3)) — that is an ACCEPTED production trade (fp-reassociation noise), not a golden regression, but it must be named so the auto-raise is never mistaken for neutral. Resolver test asserts the default resolves to `batch_size`.
- **Sharding interaction:** bs>1 grows the hot-layer activation peak linearly; the conservative auto-resolver caps it; off-box it returns the inherited value (Deferred). See M2 (A1×A4×A6 compounded peak).
- **DP replica parity:** the replica worker reads the SAME `cov_batch_size` key so in-process and DP agree; the cross-replica Gram sum is bs-independent (sums finalized per-key Grams).

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
| A6 | `test_a6_cov_batch_size_close` + resolver unit | B/C keys `torch.allclose(documented rtol/atol)` across bs∈{1,2,4} — NOT `torch.equal` (fp-reassociation, see C1); default resolves to `batch_size` | `test_resolve_cov_window_config` (resolver only) |
| all | existing `test_stage3_golden_snapshot.py` | unchanged, byte-identical | — |

## Deferred to GPU (real ≥2-GPU box only)
- **A6 measured auto-raise:** on the first live sharded run, measure per-sample activation GB on the hot-layer device (`torch.cuda.mem_get_info` before/after a known bs), then enable `"auto"`/set `cov_batch_size` to the conservative value. CPU/CI tests prove bs-invariance; only the VRAM-bound ceiling needs a real multi-GPU box. Do NOT hardcode an aggressive bs before this.
- **A4 under live DP replicas + cross-device sharding:** tiny-model tests cover the dense-vs-dict math and multi-layer windows on CPU; the cross-GPU `index_select(...).to(tgt_device)` path under `device_map=balanced` and per-replica `teacher_dense` lifetime validate end-to-end only on ≥2 GPUs.
- **A2 wall-clock ~10× on the alpha search:** byte-identity is CPU-testable; the actual ~10× (≈31 min → few min) is observable only on the real H200 alpha search. Time the disk-load-vs-eigh tradeoff there; if load dominates, switch to the in-RAM single-layer variant.
