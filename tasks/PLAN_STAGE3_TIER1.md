# Stage-3 "Tier 1" — Golden-Safe (Byte-Identical) Speedups — PLAN

Status: **PLAN ONLY** (not an implementation). Branch: `plan/stage3-tier1`.

All line references below are against `origin/main` blobs (the local tree on
`fix/svc-audit-script` is stale). They were read directly from the committed
blobs with `git show origin/main:<path>`.

## 0. Scope and the byte-identity contract

Tier 1 = "same math, computed fewer times" + pure I/O scheduling. Every item
here MUST leave `rank_map.json` (and every other on-disk Stage-3 artifact)
**byte-for-byte identical**. The pin is
`max_quality/tests/test_stage3_golden_snapshot.py::test_stage3_rank_map_byte_identical`
(parametrized `fp32` / `bf16`).

**EXCLUDED (Tier 2, separate track, NOT in this plan):** moving the
decomposition operands to GPU, or threading the per-expert eigh/svd that feeds
`rank_map.json`. Those reorder FP reductions (±1 rank flips) and break
byte-identity; they need a re-bless decision.

> **Item 1 was DROPPED.** A prior draft contained an "Item 1 — α-grid eigh
> caching" optimization (memoize the per-`(layer,expert)` `_EighDecomp` across
> α-candidates). It has been **removed** as memory-infeasible: to reuse a
> decomp across the α-candidate sweep the cache must survive the whole α-outer
> loop keyed `(layer, expert)`, i.e. hold a decomp for **every** expert at once.
> For the Qwen3.5-MoE target (`d_in ≈ 4096`, `r_eff ≈ 512`, `48` layers, `128`
> experts) that is `48 · 128 · 3 · 4096 · 512 · 4 B ≈ 144 GiB` resident — it
> OOMs the H200 and blows the ~128 GiB CPU headroom. Its only payoff (avoiding
> the eigh recompute per α) largely evaporates once Tier-2 moves eigh to GPU
> (~100× faster), so the cache buys little even where it fits. The item, its RAM
> analysis, and its unit test are gone; remaining items keep their original
> numbering (2, 3, 8, 9, 10) for stable cross-references.

### Architecture note (important — line refs in the brief were pre-S3-7)

Stage 3 is now plugin-driven. `moe_compress.stage3_svd.run` is a **thin shim**
that delegates to `moe_compress.stage3.orchestrator.run`
(`stage3_svd.py:103-134`). The "orchestrator ~:447,556,599,613,636" line
references in the task brief actually live in **two** files post-S3-7:

* `stage3/orchestrator.py` — the run-glue: `group_stats` loop (`:446-468`),
  `originals` snapshot + manifest (`:500-551`), α-cache resume + redistribute
  (`:586-615`), the `factor_layer` `loop_over` (`:653`).
* `stage3/plugins/swift_svd_alpha.py` — the relocated phase functions:
  `_factor_model_at_ranks` (`:373`), `_swift_svd_plus_alpha_search_validation`
  (`:556`), `_swift_svd_plus_alpha_search` (`:685`),
  `_redistribute_ranks_swift_svd_plus` (`:883`), and the live α-dispatch in
  `SwiftSvdAlphaPlugin.select_alpha` (`:1051-1130`).

The α-grid / redistribution code only runs when `alpha_grid` length > 1.
`select_alpha` dispatches: `validation_samples>0` → PPL grid
(`_swift_svd_plus_alpha_search_validation`) **plus** the spectral proxy when
`per_group_type` is on; else proxy only. Then `_redistribute_ranks_swift_svd_plus`.

---

## 1. Golden-fixture speed assessment (the verification budget)

**The golden is already a small/fast fixture — NO new fixture needed for the
end-to-end gate.**

* `tiny_model` / `tiny_config` (`max_quality/tests/conftest.py`):
  `hidden=16`, `intermediate=8`, `num_layers=2`, `num_experts=4`, `top_k=2`,
  calibration `num_sequences=8`, `sequence_length=16`.
* `tiny_config["stage3_svd"]`: `alpha_grid=[0.5]` (**length 1**),
  `validation_samples=2`, `block_refine.enabled=False`.
* Goldens on disk: `rank_map.fp32.json` (1155 B), `rank_map.bf16.json` (1193 B);
  both carry `"alpha_by_type": null`.

**Measured wall time on this box:** the full parametrized test
(`fp32`+`bf16`) ran in **~38–42 s** (≈19 s/param), peak RSS ~1.3 GB. Fast
enough to run on every iteration; no reduced-layer config or `STAGE3_FAST_TEST`
path is warranted.

> Environment caveat observed while timing: this box's PyTorch is built
> **without LAPACK**, so `torch.linalg.cholesky` / `svdvals` raise on CPU and
> the golden currently *fails to execute here* (it falls back to raw-SVD and
> then errors). This is exactly the determinism caveat the test header
> documents — goldens are bit-reproducible only on the same wheel/platform that
> seeded them. Implementers MUST run the golden on a LAPACK-enabled build
> (the seeding machine). It is **not** a code regression.

### The critical gap the golden does NOT cover

Because `alpha_grid=[0.5]` (length 1), the golden takes the **uniform path**
(`select_alpha` sets `alpha_by_type=None`, `per_expert_ranks=None`;
`stage3_orchestrator` α-cache branch is also gated on `len>1`). Therefore the
golden exercises **none** of:

* `_swift_svd_plus_alpha_search` / `_redistribute_ranks_swift_svd_plus` /
  `grouped_svs` (item 2's target),
* the α-search redistribution that would consume item 3's published spectra.

**Consequence:** items 2 and 3 are *invisible* to the golden. Their correctness
gate is the **focused unit test on small synthetic tensors** specified per item
below. The golden remains the regression backstop proving we did not perturb
the uniform path (and the factor path, which it does cover).

Optional belt-and-suspenders for items 2–3: a second tiny config with
`alpha_grid=[0.0,0.5,1.0]` + `validation_samples=2` would make the golden
actually traverse the α paths. Recommended as a **new golden variant**
(`rank_map.alpha.fp32.json`) rather than mutating the pinned one — additive,
keeps the existing pin immutable (use the additive `[0.0, 0.5, 1.0]` grid). This
is optional; the per-item unit tests are the primary gate.

---

## 2. Items

### Item 2 — Swift-SVD double-spectra (`grouped_svs_cache` plumbing)

**Where:** `_swift_svd_plus_alpha_search:726-761` builds `grouped_svs[name][(li,e)]`
= activation-weighted `svdvals(W @ L_A)` (with `L_A` from per-expert
`eigh(A)` + threshold). `_redistribute_ranks_swift_svd_plus:911-946` recomputes
the **identical** per-expert spectra (same `build_banks`, same
`_cov_lookup(A_cov,...)`, same eigh+threshold+`svdvals(W @ L_A)`).

**The dead parameter:** `_redistribute_ranks_swift_svd_plus(... ,
grouped_svs_cache=None, ...)` (param on `swift_svd_alpha.py:889`; `def` at
`:883`) is plumbed but never populated. Both call sites pass
`grouped_svs_cache=None` (`orchestrator.py:601` resume path, `swift_svd_alpha.py:1122`
in `select_alpha`).

**The fix:** have `_swift_svd_plus_alpha_search` **optionally return** its
`grouped_svs` so `select_alpha` can thread it into
`_redistribute_ranks_swift_svd_plus(grouped_svs_cache=grouped_svs)`. In
`_redistribute_*`, when `grouped_svs_cache` is provided, look up
`svs = grouped_svs_cache[name][(li, e)]` instead of recomputing the eigh+svd.

**Contracts to preserve (do NOT break the public surface — LOW finding):**
`_swift_svd_plus_alpha_search` is **re-exported** from the package shim at
`stage3_svd.py:77` (verified: it is one of the 8 names in the
`from .stage3.plugins.swift_svd_alpha import (...)` block) and is **name-pinned**
by `test_stage3_plugin_swift_svd.py:39` (the `_EXPECTED_NAMES`/imports tuple in
`test_swift_svd_module_imports`). Both must keep working unchanged. The pin-test
also calls `_redistribute_ranks_swift_svd_plus(...)` with its **current arity**
at `test_stage3_plugin_swift_svd.py:188` — the new `grouped_svs_cache`/return
plumbing must remain backward-compatible with that call.

**Backward-compatible signature (MANDATORY):** add an **opt-in** keyword
`return_svs: bool = False` to `_swift_svd_plus_alpha_search` (`def` at
`swift_svd_alpha.py:685`). When `return_svs=False` (the default) the function's
**base return type is unchanged** — it still returns just `alpha_by_type` — so
the re-export, the `:39` name-pin, and every existing caller keep their current
contract. Only when `return_svs=True` does it return the `(alpha_by_type,
grouped_svs)` tuple. Do **not** change the default return shape.

**Byte-identity argument & PRECONDITION (must verify before relying on it):**
the two code blocks must produce *bit-identical* `svs`. They look identical by
inspection (both `A_f32 = 0.5*(A+A.T)` → `eigh` → `keep = eigvals > max*1e-6`
→ `L_A = eigvecs[:,keep]*sqrt(clamp_min(eigvals,1e-12))` →
`svdvals(W @ L_A)`). **The implementer MUST add an assertion test that they are
`torch.equal`** before wiring the cache, because any latent difference (e.g.
`M_A = W @ L_A` then `svdvals(M_A)` vs `svdvals(W @ L_A)` — currently the proxy
materializes `M_A`, the redistribute does not; these are the same op but verify
the temporary doesn't change rounding) would silently shift `rank_map.json`.
The eigh itself is NOT reordered — it is the same call, just memoized.

**Caveat — the three dispatch branches (the proxy does NOT always run):**
`select_alpha` (`swift_svd_alpha.py:1089-1123`) has, when `len(alpha_grid) > 1`,
**three** branches that determine whether `grouped_svs` exists:

  * **(i)** `validation_samples > 0` **and** `per_group_type` → the proxy
    `_swift_svd_plus_alpha_search(..., per_group_type=True)` runs at `:1108` to
    produce `alpha_by_type`; `grouped_svs` **is** built here and can be threaded.
  * **(ii)** `validation_samples > 0` **and NOT** `per_group_type` →
    `alpha_by_type = {"all": best_global_alpha}` at `:1114`; the proxy does
    **NOT** run, so there is **no `grouped_svs`** in this branch.
  * **(iii)** `validation_samples == 0` → the proxy runs at `:1116` (fallback,
    spectral-proxy only); `grouped_svs` **is** built.

**Update BOTH in-tree call-sites of the proxy in lockstep** (verified against
`origin/main` blobs):

  * branch (i) proxy call at `swift_svd_alpha.py:1108` →
    `alpha_by_type, grouped_svs = _swift_svd_plus_alpha_search(..., per_group_type=True, A_cov=A_cov, return_svs=True)`.
  * branch (iii) proxy call at `swift_svd_alpha.py:1116` →
    `alpha_by_type, grouped_svs = _swift_svd_plus_alpha_search(..., per_group_type=per_group_type, A_cov=A_cov, return_svs=True)`.
  * branch (ii) at `swift_svd_alpha.py:1113`
    (`alpha_by_type = {"all": best_global_alpha}`) is **left unchanged** — the
    proxy is not called there, so there is no `grouped_svs` to thread.

The single `_redistribute_ranks_swift_svd_plus(...)` call shared by all three
branches is at `swift_svd_alpha.py:1120` (its `grouped_svs_cache=None` argument is
on `:1122`). Thread the cache into it **only in branches (i) and (iii)**; in
branch (ii) keep `grouped_svs_cache=None` so the redistribute recomputes the
spectra (correct — there is nothing to reuse). Implement this as an explicit
guard, **not** an unconditional pass:

```python
grouped_svs_cache = None                # default; branch (ii) keeps this
# branch (i)/(iii): proxy ran with return_svs=True and produced grouped_svs
#   alpha_by_type, grouped_svs = _swift_svd_plus_alpha_search(..., return_svs=True)
#   grouped_svs_cache = grouped_svs
...
per_expert_ranks = _redistribute_ranks_swift_svd_plus(
    moe_layers, group_stats, ranks, alpha_by_type,
    grouped_svs_cache=grouped_svs_cache if cache_was_built else None,
    A_cov=A_cov,
)
```

i.e. `cache if cache_was_built else None`, never an unconditional cache. The
**α-cache resume** path in the orchestrator — `_redistribute_ranks_swift_svd_plus`
call at `orchestrator.py:599` (its `grouped_svs_cache=None` on `:601`) — also did
NOT run the proxy this process, so it likewise stays `grouped_svs_cache=None` (it
correctly recomputes) — **do not touch that call site**.

**Expected gain:** ~2× (eliminates the second full pass of per-expert eigh+svd
over all `(layer, matrix, expert)`).

**Fast-test gate (no model):**
`test_grouped_svs_cache_equals_recompute` — build a tiny `moe_layers` +
`group_stats` + `A_cov` with small random tensors; assert
`_redistribute_ranks_swift_svd_plus(..., grouped_svs_cache=cache)` returns a
rank dict `==` to the `grouped_svs_cache=None` path, AND that the cache tensors
themselves are `torch.equal` to a fresh recompute. Seconds. LAPACK build req'd.

---

### Item 3 — `_group_stat` triple-pass — **NEEDS A PRECONDITION CHECK; likely
NOT byte-safe as literally described**

**Where the brief points:** `stage3_orchestrator.py:447-466` calls
`d_rank_allocate.py:338 _group_stat`, which does per-expert whitened svdvals;
the brief asserts Swift-SVD "redoes the same" and proposes `_group_stat`
publish per-expert spectra (it currently keeps only `mean_s`) for Swift-SVD to
consume.

**FINDING (must be respected — this is the load-bearing correctness note):**
the two spectra are **NOT the same** as currently written. They differ in three
independent ways:

1. **Different A matrix.** `_group_stat` whitens with the **group-averaged**
   covariance `A_g = mean_e(_cov_lookup(...))` (`stage3_orchestrator.py:463`,
   one `L_A` shared by all experts in the group). Swift-SVD whitens with the
   **per-expert** `A = _cov_lookup(A_cov, li, e, name)` (a different `L_A` per
   expert). Reusing the group-averaged-whitened spectrum where the per-expert
   one is required would change `epsilons`/`betas` and thus `rank_map.json`.
2. **Different factorization of A.** `_group_stat` uses
   `L_A = cholesky(A_g.float64 + jitter).float32` (`d_rank_allocate.py:355-359`).
   Swift-SVD uses `L_A = eigvecs * sqrt(eigvals)` from `eigh(A)` with a
   `>max*1e-6` threshold (`swift_svd_alpha.py:744-748`). Cholesky factor ≠
   eigh factor; `svdvals(L_chol @ W.T)` ≠ `svdvals(W @ L_eigh)` even for the
   same A.
3. **Different operator orientation.** `_group_stat` computes
   `svdvals(L_A @ W.T)` (`d_rank_allocate.py:373`); Swift-SVD computes
   `svdvals(W @ L_A)` (`:748`). (Singular values of `M` and `M.T` coincide, so
   this one alone is harmless — but combined with (1) and (2) the spectra
   diverge.)

**Therefore:** "have `_group_stat` publish its per-expert spectra for Swift-SVD
to consume" would change the numbers fed into the rank allocator unless the
allocator's spectra are *first* made identical to `_group_stat`'s — which is a
**semantic change** (re-bless territory), NOT a byte-identical cache reuse.

**Recommendation — DEMOTE Item 3 out of Tier 1.** Options for the implementer:

* **(A) Drop from Tier 1** (recommended). Item 3 cannot be done byte-identically
  by simple publish-and-consume; it requires unifying the whitening definition,
  which flips ranks. Move it to a Tier-2 / re-bless ticket.
* **(B) Narrow, byte-safe sub-optimization only:** the brief says `_group_stat`
  "keeps only `mean_s`" — if some *other* consumer recomputes the *group-
  averaged Cholesky-whitened* spectra `_group_stat` already produced internally
  (`svs` list at `d_rank_allocate.py:365-377`, discarded after `mean_s`), then
  publishing *those* exact tensors and having that exact consumer reuse them is
  byte-safe. But Swift-SVD is **not** such a consumer (it wants per-expert
  eigh-whitened spectra — see (1)/(2)). The implementer must first
  **grep for any consumer that re-derives the Cholesky-whitened per-expert
  spectra**; if none exists, Item 3(B) has no beneficiary and should be dropped.

**Action for the planner-to-implementer handoff:** treat Item 3 as
**blocked / re-scope**. Do NOT implement the literal "Swift-SVD consumes
`_group_stat`'s spectra" — it is not byte-identical.

**Fast-test gate (the disproof, cheap):**
`test_group_stat_vs_swift_spectra_differ` — on a tiny `(W, A_per_expert)` with a
non-trivial group-average, assert that `svdvals(cholesky(mean_A) @ W.T)` is
**not** `torch.allclose` to `svdvals(W @ eigh_factor(A_e))`. This codifies the
finding and prevents a future "obvious" reuse from silently regressing the
golden. If a real byte-safe beneficiary is found (Item 3(B)), add the standard
`cache == recompute` `torch.equal` assertion for *that* path.

---

### Item 8 — `_save_state_dict_sharded` overlap (I/O)

**Where:** `model_io.py:1217-1230 _save_state_dict_sharded`. The shard loop is
strictly serial: for each shard, `cpu_shard = {k: v.detach().cpu().contiguous()
...}` (the CPU clone, GPU→CPU copy) then `save_file(cpu_shard, ...)` (the disk
write), then `del cpu_shard`. The clone of shard N+1 cannot start until the disk
write of shard N returns.

**The fix:** overlap the CPU-clone of shard N+1 with the `save_file` of shard N,
using a **1–2 worker** `ThreadPoolExecutor` — mirror the established
`covariance_collection.py:344-350` `bcov-spill` executor pattern
(`max_workers=1`, `thread_name_prefix=...`, futures list, drained in a
`try/finally`). Submit `save_file` to the pool; produce the next `cpu_shard` on
the main thread while the prior write is in flight; bound in-flight to ≤1–2 so
peak CPU RAM stays ≤ 2 shards.

**Byte-identity argument:** shard *contents* and `model.safetensors.index.json`
(`weight_map`, `total_size`) are unchanged — only the *timing* of writes and the
order of clone-vs-write overlap changes. `save_file` writes each shard to a
distinct path; `weight_map` is populated deterministically by shard index in the
main-thread loop (keep that bookkeeping on the main thread, only the byte-write
is offloaded). The `model.safetensors.index.json` is written **after the pool is
drained** (move the existing `:1232-1236` block after a `for f in futures:
f.result()` join). Output bytes identical.

> Note: this artifact is the compressed checkpoint, not `rank_map.json`, so it
> is not the golden's pin — but it is still Tier-1 byte-identical by the same
> argument (contents unchanged). The golden stubs the saver (`_noop_save`,
> test L70-72/L114-117) so it neither covers nor is endangered by this change.

**Fast-test gate:**
`test_save_state_dict_sharded_overlap_identical` — build 3–4 tiny shards (small
CPU tensors), write once with the current serial code and once with the
overlapped pool into two temp dirs; assert every
`model-XXXXX-of-YYYYY.safetensors` is **byte-identical** (`read_bytes()` equal)
and `model.safetensors.index.json` is byte-identical. No GPU needed (`.cpu()` is
a no-op on CPU tensors). Milliseconds.

---

### Item 9 — `load_layer_from_disk` prefetch on the factor critical path (I/O)

**Where (scope: the MAIN factor loop ONLY):** `activation_hooks.py:1161
load_layer_from_disk` (in `InputCovarianceAccumulator`), called synchronously at
the top of each `factor_layer` iteration: `aa_svd_factor.py:534`
`B_acc.load_layer_from_disk(ref.layer_idx, bcov_spill_dir)` (and the C-cov twin
at `:543`), driven by the per-layer `loop_over` (`orchestrator.py:653`). The
factoring of layer N blocks on reading layer N's ~5 GB spill from disk first.

**Out of Tier-1 scope — the validation-path load:** there is a *second*,
distinct `B_acc.load_layer_from_disk` at `swift_svd_alpha.py:404` (inside
`_factor_model_at_ranks`, which is itself inside the per-α loop of the PPL
search). Prefetching THAT one is **NOT** part of this item — it sits under the
α-outer loop (so the same layer's spill is re-read once per α-candidate, a
different access pattern), and folding it into a prefetch interacts with the
α-search restructure. Tier-1 item 9 targets the main factor loop
(`aa_svd_factor.py:534` / driven by `orchestrator.py:653`) **only**; leave
`swift_svd_alpha.py:404` untouched.

**The fix:** prefetch layer N+1's spill (a `torch.load` to CPU) on a background
thread while layer N is being factored. The `load_layer_from_disk` body splits
cleanly into (a) `torch.load` + validation (no shared-state mutation,
`:1161-1186`) and (b) the under-`self._lock` accumulate (`:1188-1200`). Prefetch
only part (a) — load the payload dict for N+1 into a 1-slot cache; when the loop
reaches N+1, do the locked-accumulate from the prefetched payload instead of
re-reading disk. Use a single-worker executor (same
`covariance_collection.py:344` pattern).

**Byte-identity argument:** the accumulate math (`:1190-1199`,
`prev.float32 + disk_cov.float32 → storage_dtype`) is **unchanged** and still
runs under `self._lock` in loop order. Only the `torch.load` (a pure read of an
immutable on-disk file) is hoisted earlier in wall-clock time. The covariance
tensors consumed by `_aa_svd` are bit-identical; ranks unchanged.

**Caveats (must be in the implementation):**
* RAM: a 1-deep prefetch holds **two** layers' spill simultaneously (~10 GB).
  This is within the documented budget headroom but must be bounded to depth 1.
  Drain/cancel the prefetch in a `finally` so a factor-loop exception doesn't
  leak the executor.
* The spill file is immutable once written (Pattern-O manifest-last); reading it
  early is safe. Guard the resume case where a spill file is absent.
* `load_layer_from_disk` returns `True`/raises; preserve the loud
  RuntimeError-on-corrupt behavior (`:1162-1180`) — surface it from the
  prefetch future's `.result()` at consume time, not swallowed in the worker.

**Fast-test gate:**
`test_bcov_prefetch_matches_serial` — write 2–3 tiny per-layer spill files via
the accumulator's spill path, then load them (i) serially and (ii) via the
prefetch wrapper; assert the resulting `accumulator.covariance` /
`token_count` dicts are key-for-key `torch.equal`. No model. Milliseconds.

---

### Item 10 — drop the 50 GB SHA-256 on originals (I/O)

**Where:** `orchestrator.py:519-547` saves `_stage3_original_weights.pt`
(~50 GB) via `atomic_torch_save` (`:520-536`), then `write_manifest_last(...,
compute_sha256=True)` (`:536-547`). `write_manifest_last` calls
`atomic_io.py:343 _sha256_file`, streaming the **entire ~50 GB** through SHA-256
(`:345-352`). That is a full extra read of a 50 GB file on the critical path,
purely to populate a forensics field.

**The fix (cheapest, byte-safe — VERIFIED sufficient):** set
`compute_sha256=False` at the call site (`orchestrator.py:547`). The manifest
schema already supports `sha256=None` (`atomic_io.py:372-375` documents "large
artifacts may set this False ... rely on size + schema_version cross-checks").
`read_and_validate_manifest` defaults `require_sha256=False`
(`atomic_io.py:425`) and validates on size + schema_version; with `sha256=None`
it passes. There is an in-repo precedent shipping exactly this:
`wanda_intra_expert_score.py:799` calls `write_manifest_last(...,
compute_sha256=False)`.

**Byte-identity argument:** the **payload** `.pt` bytes are produced by
`atomic_torch_save` and are **completely unaffected** — only the *manifest
sidecar's* `sha256` field flips from a hex string to `null`. The manifest is not
a Stage-3 *math* artifact and `rank_map.json` is untouched. Strictly: this
changes the manifest JSON bytes (sha field), so the originals-manifest is *not*
byte-identical to a `compute_sha256=True` run — but it is **not pinned** by any
golden and carries no decompression-affecting data. If strict manifest
byte-identity is desired, prefer the alternative below.

**No reader needs a non-null sha (VERIFIED — not an open grep):** Stage 4 is the
sole and last consumer of `_stage3_original_weights.pt`
(`stage4/orchestrator.py:218`), and it only **deletes** the payload + manifest on
success (`:230`). Its one manifest read,
`eora_inputs.py:268 read_and_validate_manifest(originals_path,
originals_manifest_path, expected_schema_version=1)`, passes **no**
`require_sha256=True`, so it defaults to `False` and validates on size + schema
only — a `sha256=None` manifest passes cleanly. No other reader of this manifest
exists in the repo (the only `require_sha256=True` call sites are inside
`atomic_io`'s own unit tests, not the originals path). **Therefore the
hash-while-write alternative is unnecessary and is dropped** — do the one-line
`compute_sha256=False` at `orchestrator.py:547`.

**Fast-test gate:**
`test_originals_manifest_no_sha_validates` — `atomic_torch_save` a tiny dict,
`write_manifest_last(..., compute_sha256=False)`, then
`read_and_validate_manifest` and assert it passes with `sha256 is None` and the
payload `.pt` `read_bytes()` is identical to a `compute_sha256=True` write of the
same dict (payload identical; only the manifest sha differs). Milliseconds.

---

## 3. Summary table

| # | Item | Byte-safe? | Files touched | Fast-test gate |
|---|------|-----------|---------------|----------------|
| 1 | ~~α-grid eigh caching~~ | **DROPPED** — memory-infeasible (≈144 GiB resident to keep all `(layer,expert)` decomps across the α-outer loop; OOMs H200 + blows ~128 GiB CPU headroom); payoff evaporates once Tier-2 moves eigh to GPU. See §0 note. | none | none |
| 2 | swift-svd double-spectra (`grouped_svs_cache`) | YES (after `torch.equal` precondition check; opt-in `return_svs=False` default keeps base return type + re-export `stage3_svd.py:77` + pin-test `:39`/`:188` intact; thread cache ONLY in select_alpha branches (i)`:1108`+(iii)`:1116`, `cache if cache_was_built else None`; branch (ii)`:1113` + resume `orchestrator.py:599` stay `None`) | `stage3/plugins/swift_svd_alpha.py` (`_swift_svd_plus_alpha_search` opt-in return, `_redistribute_ranks_swift_svd_plus` consume, `SwiftSvdAlphaPlugin.select_alpha` thread both call-sites) | `test_grouped_svs_cache_equals_recompute` |
| 3 | group_stat triple-pass | **NO — re-scope/blocked** (group-avg+cholesky ≠ per-expert+eigh; see Finding) | none in Tier 1 | `test_group_stat_vs_swift_spectra_differ` (disproof) |
| 8 | sharded save overlap | YES (contents unchanged; index after drain) | `utils/model_io.py` (`_save_state_dict_sharded`) | `test_save_state_dict_sharded_overlap_identical` |
| 9 | bcov layer prefetch (MAIN factor loop only; `swift_svd_alpha.py:404` validation-path load OUT of scope) | YES (load hoisted; accumulate unchanged under lock) | `utils/activation_hooks.py` (`load_layer_from_disk` split), `stage3/plugins/aa_svd_factor.py` (`:534`) / `stage3/orchestrator.py` (`:653` prefetch driver) | `test_bcov_prefetch_matches_serial` |
| 10 | drop 50 GB originals SHA | YES for payload; manifest sha→null (no reader needs non-null sha — VERIFIED via Stage 4 `eora_inputs.py:268` + `require_sha256` default `False`) | `stage3/orchestrator.py:547` (`compute_sha256=False`) | `test_originals_manifest_no_sha_validates` |

## 4. Verification strategy (fast by design)

1. **Per-item unit tests (primary gate for 2,8,9,10; disproof for 3):**
   small synthetic tensors, no model load, `torch.equal` / byte-equal asserts of
   cached/overlapped path vs recompute/serial path. Each runs in
   ms–seconds. The unit test is the *only* coverage of item 2 because the golden's
   `alpha_grid=[0.5]` skips the α paths.
2. **End-to-end golden:**
   `pytest max_quality/tests/test_stage3_golden_snapshot.py` — ~38–42 s for both
   params, already small/fast, **no new fixture required**. Run on a
   LAPACK-enabled build. Proves the uniform + factor paths stay byte-identical.
3. **Optional additive α-golden variant** (`alpha_grid=[0.0, 0.5, 1.0]`) to give
   the golden suite real coverage of items 2–3 end-to-end. Additive new golden
   file; do **not** mutate the pinned `rank_map.{fp32,bf16}.json`.

## 5. Ordering / risk

* Lowest-risk first: **10** (1-line), **8** (self-contained), **9**
  (self-contained, RAM-bounded).
* **2** touches the α-search hot path — land with its unit test green, then the
  α-golden variant, then the pinned golden.
* **1** is **DROPPED** (memory-infeasible — see §0); no work.
* **3** is **not implemented in Tier 1** — file a Tier-2 / re-bless ticket;
  the disproof test documents why.
