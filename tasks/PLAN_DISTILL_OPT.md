# PLAN — Distillation-plugin Tier-1 CPU optimizations

**Branch:** `plan/distill-opt` (base `origin/main` @ `0ae66896942cd8ecbca1cf62c81aaec60f889590`)
**Scope:** PLAN ONLY. No production edits in this branch. All file:line citations verified against `origin/main` blobs on 2026-05-31.
**Recovery host:** H200 (~70 GB resident model; ample system RAM). This is where the levers run in production.
**Equivalence-test host:** RTX 5080, 16 GB VRAM (`nvidia-smi`: `16303 MiB`). Used ONLY to run the tiny (KB-scale) equivalence fixtures and CPU tests — it is NOT the recovery host and is NOT sized for the real working set. No lever's production budget is bound by the 5080.

## Tier-1 contract (applies to all four levers)

Every lever here is **mechanism-only / byte-identical**: it changes *when* / *how* a tensor is moved between host and device, or *how much* host RAM is held, but it does NOT change:
- the CPU bytes that are uploaded (same `.to(dtype=bf16)` source tensors),
- the order of GPU ops in the AdamW / forward loops,
- any RNG draw (no `randperm`/`Generator` touched),
- any numeric value entering a loss or an optimizer step.

The two affected plugins are **default-OFF** (`block_refine.enabled=False`; `expert_distill_steps=0`), but the user has confirmed the recovery run WILL enable both. So these are real hot paths, not dead code.

Threading / `lsa_pool` does NOT apply: neither path calls `linear_sum_assignment`; both are GPU-bound sequential loops. Do not propose it.

---

## Levers 1+2 (merged) — block_refine: pinned-memory + `non_blocking=True` per-step H2D

**File / function:** `max_quality/src/moe_compress/stage3/plugins/block_refine.py` → `_phase_c5_block_refine`.

> **Why merged / why NO resident-hoist.** An earlier draft proposed hoisting the per-step H2D into device-resident lists (`x_s_dev`/`target_dev`) once per block. **That is infeasible at recovery scale and is dropped.** From the REAL config (`qwen36_35b_a3b_30pct.yaml`: `calibration.num_sequences=4000`, `calibration.sequence_length=4096`, `block_refine.batch_size=32`) and the model (`hidden_size=2048`, Qwen3.6-35B-A3B — verified `stage2_assignment_revision.md:293`, `stage2/profiling.py:44`), each hoisted list spans ALL `ceil(4000/32)=125` batches:
>
> ```
> X_student (or teacher_targets) resident = 4000 × 4096 × 2048 × 2 bytes
>   = 67.1 GB (62.5 GiB) per list  →  ~134 GB (125 GiB) for BOTH lists
> ```
>
> On the H200 recovery host the model already occupies ~70 GB resident; adding ~134 GB of resident activations is a **guaranteed OOM**. The earlier "~4.3 GB resident" figure was ~30× too small and was internally inconsistent with the same draft's "~103 GB redundant H2D/block." The loop also CANNOT be reordered batch-outer to shrink the resident set: that would change the AdamW update order (a Tier-2 change), which is forbidden. So the per-step re-upload stays; we only make each upload faster.

### Current behavior (verified)
The per-block training loop (lines **509–512**) re-uploads the SAME CPU tensors on every one of the `epochs * len(batches)` steps:

```
509  for epoch in range(epochs):
510      for bi, _ in enumerate(batches):
511          x_s = X_student[bi].to(device=device, dtype=student_dtype)
512          target = teacher_targets[bi].to(device=device, dtype=student_dtype)
```

- `X_student[bi]` is a constant bf16 CPU tensor for the whole block (built once at line **314** via `_capture_block_input`, mutated only at block advance, line **598**).
- `teacher_targets[bi]` is built ONCE per block (lines **495–503** live path, or sliced from the cache at **473–476**) and is constant across all `epochs` (default 25) of that block.

These CPU bf16 tensors are **pageable** (`.to(device="cpu")` with no `.pin_memory()`):
- `_capture_block_input._hook` (line **294**): `captured[...] = t.detach().to(dtype=torch.bfloat16, device="cpu")`.
- live teacher target (line **503**): `teacher_targets.append(out.detach().to(dtype=torch.bfloat16, device="cpu"))`.
- `_advance_streams` (lines **624**, **629**): `new_s/new_t.append(out_*.detach().to(dtype=torch.bfloat16, device="cpu"))`.

Pageable→device copies are synchronous and ~3.25× slower than pinned async copies (measurement pass). Across 25 epochs × `n_batches`, the same pageable H2D is re-issued per step.

### The deliverable: pinned source + `non_blocking=True` per-step upload
This is a **single** change — pin and non_blocking land together (with pageable memory, `non_blocking=True` is silently synchronous, so neither helps alone):

1. Allocate the producer CPU tensors as **pinned**. Append `.pin_memory()` to the producer `.to(..., device="cpu")` calls (294, 503, 624, 629). The cache-slice path (lines 473–476) produces slices of one large cached tensor; pin the per-batch slice copy at the consumption site, not the whole cache.
2. At the per-step H2D site (511–512), add `non_blocking=True`:
   ```
   x_s    = X_student[bi].to(device=device, dtype=student_dtype, non_blocking=True)
   target = teacher_targets[bi].to(device=device, dtype=student_dtype, non_blocking=True)
   ```
The loop structure, op order, AdamW step order, LR schedule, and `step` increment are all UNCHANGED. The H2D is still issued per step (this is unavoidable without a resident hold we cannot afford); it is just ~3.25× faster per transfer and can overlap with compute on the same stream.

### Honest gain (recompute / scope at real dims)
This is a per-transfer speedup, NOT an elimination of bytes moved. At the reduced-dim measurement the H2D was ~20% of each step; the pinned+async path takes that ~20% to ~6% (≈3.25×). **At the real recovery dims (`batch_size=32 × seq_len=4096 × hidden=2048` bf16 per per-step tensor) the H2D share SHRINKS as a fraction of the step**, because the fwd+bwd through the full MoE block (with `epochs=25`, top-k routing) grows faster than the fixed-size transfer. So the realistic end-to-end gain is **smaller than the ~20%-of-step figure suggests** — recompute the actual H2D fraction at real dims during implementation (`nvidia-smi`/profiler on the tiny→scaled fixture) and report the measured share. Do NOT claim "~103 GB eliminated" — nothing is eliminated; each per-step transfer is just faster.

### Budget table (recomputed from REAL config dims)
| Quantity | Formula | Value |
|---|---|---|
| `n_batches` | `ceil(4000 / 32)` | **125** |
| per-step H2D tensor (`x_s` or `target`) | `32 × 4096 × 2048 × 2 B` | **0.5 GiB** |
| full-list resident (if hoisted — INFEASIBLE) | `4000 × 4096 × 2048 × 2 B` | **62.5 GiB / 67.1 GB per list; ~134 GB both** |
| pinned host RAM held (working set) | current block's `X_student`+`teacher_targets`+`X_teacher`, replaced at block advance (line 598) | bounded by `~3 × 62.5 GiB ≈ 188 GiB` worst-case if the full lists are pinned |

**Pinned-RAM caveat (recovery host, not GPU):** pinned host memory is non-pageable. Pinning the entire `X_student`/`teacher_targets`/`X_teacher` lists is ~188 GiB worst-case — verify the H200 host's system RAM headroom before pinning the full lists. If pinned-RAM pressure appears, pin only the per-step source slices that are about to upload (a rolling pin of the current batch), or pin `X_student`/`teacher_targets` and leave `X_teacher` pageable. This is a host-RAM budget concern only; it does NOT touch the 16 GB GPU budget (the 5080 is the equivalence-test host, never the recovery host).

**Byte-identity argument:** `.pin_memory()` returns a copy in pinned host memory with identical bytes; `non_blocking=True` only relaxes host-side synchronization of the copy. There are no custom CUDA streams and no RNG in this loop, so PyTorch orders the async copy before the consuming kernel on the same (default) stream — the consumed values are bit-identical. No numeric change; same op order; same `torch.equal` result.

---

## Lever 3 — block_hidden_cache: lazy / prefetch load

**File / function:** `max_quality/src/moe_compress/stage3/plugins/block_hidden_cache.py` → `Stage3BlockHiddenCacheProvider._load_layers` (lines **144–177**) and `on_load` (lines **179–291**).

### Current behavior (verified)
`_load_layers` (called from `on_load`, line **215**) eagerly `load_block_hidden(...)`s EVERY per-layer sidecar up-front (loop at **161–176**), and `on_load` reshapes ALL of them into `teacher_targets_cache` (loop at **240–283**), a `dict[layer_idx -> Tensor]` held for the whole Stage 3 pass. Measurement: ~21 s up-front stall + ~52 GB host RAM held for the duration.

`block_refine` consumes one layer's entry at a time (lines 463–476: `teacher_targets_cache.get(int(layer_idx))`), in `all_indices` order, with a full 25-epoch train between consecutive layer reads.

### Miss-detection contract (MUST preserve — this is the hard part)
Today the cache is **all-or-nothing at start**: `on_load` validates EVERY layer's prompt-count (I2 guard, lines 247–258) and token-count (lines 263–274) up-front and returns `None` (cache miss → live teacher forward) if ANY layer fails. The `block_refine` consumer therefore sees either a fully-hydrated dict or no dict at all. The orchestrator's `dispatch_first("on_load", ...)` either wins (all layers good) or the slot is absent (`ctx.has("teacher_targets_cache")` is False).

A lazy / per-layer load moves the miss point from "before Stage 3 starts" to "during layer `i`'s consumption" — and a malformed layer `i` discovered mid-pass would be discovered AFTER layers `0..i-1` already trained against cached targets, which is FINE numerically (each layer's target is independent), but the all-or-nothing miss-detection semantic must be preserved so a partial cache never silently mixes cached + live targets in a way the operator can't see.

### Why a "metadata-only peek" (design (a)) is infeasible as Tier-1
An earlier draft proposed validating each layer's `n_prompts_in_subset` + `n_tokens` from sidecar metadata WITHOUT materializing the tensor (a cheap "peek"). **That is not achievable in Tier-1 scope:**
- `load_block_hidden` (`cached_calibration_signals.py:1642`) does `torch.load(path, map_location="cpu", weights_only=False)` (line **1654**) — it deserializes the WHOLE `BlockHiddenPayload`, including the full `hidden_states: [n_tokens, hidden_dim]` tensor. There is no shape-only path.
- The manifest (written by `_write_payload_and_manifest`, ~line 303; `extra_meta={"artifact": signal_name}`, line 336) stores only `schema_version` + `size` + the artifact name — **NO `n_tokens`/`shape`/`n_prompts` field**. So a cheap n_tokens peek would require a WRITER-side manifest change (adding a shape header), which is out of Tier-1 scope.

So design (a) is dropped. We commit to design (b) below.

### After — design (b): presence-validate up-front, lazy-materialize per layer
1. **Presence validation stays eager and all-or-nothing**, but via file/manifest EXISTENCE, not tensor load: in `on_load`, for every expected `layer_idx`, resolve the sidecar path (`_resolve_sidecar_for_load`) and confirm the file + its manifest exist (and pass the existing `_validate_manifest_or_warn` schema-version check, which reads only the manifest, not the payload). If ANY layer's sidecar/manifest is absent, return `None` (full miss → live teacher forward) BEFORE any training — preserving the all-or-nothing semantic so a partial cache never silently mixes cached + live targets.
   - **Out of scope (accepted):** shape / `n_tokens` cannot be validated up-front without the writer-side manifest change above. The existing per-layer token-count and prompt-count guards (`block_refine` lines 466–492: a cached entry whose shape/token-count mismatches **falls through to the live teacher forward for that layer with a warning**) remain the safety net for a malformed-but-present sidecar. That per-layer fall-through is already in `origin/main` and is numerically safe (each layer's target is independent).
2. **Materialization becomes lazy**: populate `teacher_targets_cache` with a **lazy mapping** object (not an eager `dict`). On `get(layer_idx)` it `load_block_hidden`s + reshapes that one layer's sidecar on demand (`hs.reshape(n_prompts, seq_len, -1).contiguous()`, current line 278) and returns the `[n_prompts, seq_len, hidden]` tensor, then drops its own reference so the payload is freed after the consumer is done. Because `block_refine` reads each layer exactly once and never re-reads, only ~1 layer's tensor is resident at a time (~52 GB / n_layers). The consumer's `cached.get(int(layer_idx))` call site (block_refine line 465) is unchanged.

**What this wins:** removes the ~52 GB held-RAM (only one layer materialized at a time — the LARGER win), and because presence validation is cheap (stat + manifest read, no payload load) the ~21 s eager-load up-front stall is also largely eliminated.

### Lazy-mapping contract (MUST implement `__len__` and `.get`)
The orchestrator (`stage3/orchestrator.py:710`) logs `len(run_ctx.get("teacher_targets_cache"))` on a HIT, and `block_refine` (line 465) reads via `.get(int(layer_idx))` (NOT `[]`). The lazy mapping MUST therefore expose:
- `__len__()` → the validated layer count, computed from the presence scan WITHOUT materializing any tensor.
- `.get(layer_idx)` → the lazily-loaded `[n_prompts, seq_len, hidden]` tensor for a present key, and **`None` for an absent key** (matching `dict.get` semantics block_refine relies on).
Add both to the lazy-mapping unit assertions (see Verification gates): assert `len(mapping) == n_layers` before any `.get`, and `mapping.get(absent_idx) is None`.

### Optional prefetch (deferred)
Layer `i`'s `load_block_hidden` (~52 GB / n_layers ≈ ~1–2 GB) could be prefetched on a daemon thread during layer `i-1`'s 25-epoch train (train is GPU-bound; a CPU-side load overlaps cleanly). One-deep only (≤2 layers resident). KEEP off the byte-identity-critical path — prefetch changes load timing, never values. Defer to a follow-up unless the per-layer lazy stall is measured to matter; the lazy-materialize step alone removes the held-RAM and the up-front stall.

**Byte-identity argument:** the tensor returned by `get(layer_idx)` is the SAME reshaped `[n_prompts, seq_len, hidden]` bf16 tensor (same `hs.reshape(n_prompts, seq_len, -1).contiguous()`, line 278) whether materialized eagerly or lazily. `block_refine` slices it identically (lines 473–476). No value change; only allocation timing.

---

## Lever 4 — expert_distill: snapshot → members only

**File / function:** `max_quality/src/moe_compress/stage2/plugins/expert_distill.py` → `_snapshot_pre_merge_layer_experts` (lines **489–515**) and its caller `ExpertDistillPlugin.pre_merge_snapshot` (lines **987–1009**).

### Current behavior (verified)
`_snapshot_pre_merge_layer_experts` snapshots ALL `num_routed_experts` (loop `for eid in range(n)`, lines 510–514), each `.detach().cpu().clone()` of gate/up/down. But `_distill_merged_group` reads ONLY group members: `pre_merge_weights[m]["..."] for m in members` (v1 lines 677–679; v2 reads the same `pre_merge_weights[m]` per member). Non-member entries are NEVER read.

The v1-waste note in the docstring (lines 499–505) already flags this and prescribes the fix: narrow to `set().union(*grouped.values())`.

### `grouped` availability (verified)
- `grouped` is committed to ctx at `stage2/orchestrator.py` line **608** (inside `_run_assignment`, called at line **1567**).
- `pre_merge_snapshot` runs in `_STAGE2_POST_ASSIGN_PHASES` (`orchestrator.py` line **205**, first phase), walked at line **1568** — AFTER `_run_assignment`. So `grouped` IS in ctx when the snapshot hook fires. ✓
- `grouped` shape (verified `stage2/grouping.py:63` `_build_grouped_from_assignment`): `{centroid_id: [centroid_id, *absorbed_member_ids]}`. Orphan non-centroids are promoted to singleton groups (`_promote_orphans`, grouping.py). So `set().union(*grouped.values())` = every centroid + every assigned/promoted child = EXACTLY the experts that any `_distill_merged_group` call can read via `members`.

### `reads` declaration (advisory only — do NOT add a redundant entry)
The `reads`/`writes` tuples are **ADVISORY**: `pipeline/plugin.py:29–30` describes them as enabling "a future static check that every plugin reads keys some prior [plugin writes]" — there is NO static gate today. What actually enforces correctness at runtime is `ctx.get` raising a `KeyError` if a key was never `set`. So whether `grouped` appears in `reads` is **irrelevant to safety**; what matters is the runtime ordering: `grouped` is `set` (orch:608, inside `_run_assignment` called at orch:1567) strictly BEFORE it is read (the post-assign phase walk at orch:1568). That ordering IS safe.

`grouped` already happens to be in the plugin-level `reads` tuple (`expert_distill.py:895–897`: `"layer_ref", "pre_merge_weights", "grouped", "freq", "layer_input_acc"`), but do NOT treat that as load-bearing and do NOT add any further `reads` entry "for the gate" — there is no gate. The ONLY required change is the `pre_merge_snapshot` hook BODY (currently reads only `layer_ref`, line 999) calling `ctx.get("grouped")` and passing it down.

### After
1. Add an optional `members` param to the helper (keeps the existing 1-arg test call `_snapshot_pre_merge_layer_experts(layer_ref)` working — see test gate):
```
def _snapshot_pre_merge_layer_experts(
    layer_ref: MoELayerRef,
    members: "set[int] | None" = None,
) -> dict[int, dict[str, torch.Tensor]]:
    banks = build_banks(layer_ref)
    out: dict[int, dict[str, torch.Tensor]] = {}
    eids = range(layer_ref.num_routed_experts) if members is None else sorted(members)
    for eid in eids:
        out[eid] = {name: banks[name].get(eid).detach().cpu().clone() for name in MATRIX_NAMES}
    return out
```
2. In `pre_merge_snapshot` (line 1004–1008), compute the member set from `grouped` and pass it:
```
grouped = ctx.get("grouped")
needed = set().union(*grouped.values()) if grouped else None
pre_merge_weights = (
    _snapshot_pre_merge_layer_experts(layer_ref, members=needed)
    if self.expert_distill_steps > 0 else None
)
```
(Guard `if grouped` so an empty/None grouped degrades to full snapshot — defensive, never expected on the live path.)

**Byte-identity argument:** `_distill_merged_group` reads `pre_merge_weights[m]` only for `m in members`, and every such `m` is in `set().union(*grouped.values())`. The narrowed dict contains exactly those keys (same `.detach().cpu().clone()` bytes). Non-member keys were dead — removing them cannot change any distilled output. `skip_singletons` and the `len(members) <= 1` early-return mean some present keys go unread too; that's still a superset of what's read, so safe.

**Savings:** ~1.2 s + ~0.6 GB host RAM per layer (measurement pass) — the snapshot drops from `num_routed_experts` (e.g. 128/256) clones to the union of group members (≈ num_groups + absorbed, typically far fewer).

---

## Ordering

1. **Lever 4** first — fully self-contained, covered by behavioral tests, lowest risk, smallest diff. Lands and verifies independently.
2. **Lever 3** second — independent file, contract-sensitive but no interaction with Levers 1+2.
3. **Levers 1+2** last (single deliverable) — pinned-memory + `non_blocking=True` per-step H2D land together (pin+non_blocking are inseparable; pageable+non_blocking is silently sync). Needs the new block_refine bit-identity harness (build-from-scratch — the largest implementation cost) in place first.

Rationale: ascending risk / ascending need-for-new-gate. Lever 4 and Lever 3 are gated by existing/extendable behavioral tests; Levers 1+2 need the new from-scratch block_refine bit-identity harness, so build that once and land 1+2 against it.

---

## Verification gates

### Lever 4 (byte-identical; existing behavioral tests + one new equality test)
- Existing: `max_quality/tests/test_stage2_expert_distill.py`, `test_stage2_plugin_expert_distill.py`.
- **CAUTION:** the FULL-snapshot guard `set(snap.keys()) == set(range(n))` is in **`test_stage2_expert_distill.py`** (test `test_snapshot_pre_merge_layer_experts_makes_independent_cpu_clones`, def at line **106**, the `set(...)==set(range(n))` assert at line **115**) — NOT in `test_stage2_plugin_expert_distill.py`. The optional-`members`-defaults-to-None signature keeps it green (it calls `_snapshot_pre_merge_layer_experts(layer_ref)` with no `members`). Do NOT change that test's call; the narrowing only happens on the plugin path where `grouped` is passed.
- **MANY other bare callers:** `_snapshot_pre_merge_layer_experts(layer_ref)` is called with no `members` in numerous other tests across BOTH files — `test_stage2_expert_distill.py:185,224,267` and `test_stage2_plugin_expert_distill.py:196,292,305,336,370,381,456,469,509,556,567,614`. All rely on the default-None full-snapshot behavior; the optional-`members=None` default preserves every one of them unchanged.
- **NEW test** (`test_stage2_plugin_expert_distill.py`): drive `ExpertDistillPlugin.pre_merge_snapshot` + `merge` on the `tiny_model` fixture with a known `grouped` in ctx, once with the narrowed snapshot (production path) and once forcing a FULL snapshot (`members=None`), and assert the resulting distilled bank weights (`distill_state` + the written-back `gate/up/down` for each centroid) are **bit-identical** (`torch.equal`). This proves the narrowing is a pure dead-key removal. Seed RNG identically for both runs.

### Lever 3 (byte-identical; extend existing cache tests)
- Existing: `max_quality/tests/test_stage3_block_hidden_cache.py` (covers `test_load_miss_no_dir`, `test_load_hit_populates_teacher_targets_cache`, `test_token_count_mismatch_falls_through_to_miss`, `test_prompt_count_divergence_falls_through_to_miss`, `test_c1_batch_size_decoupled`).
- **NEW tests:**
  - `test_lazy_get_returns_same_tensor_as_eager`: build sidecars, call `on_load`, then `cache.get(layer_idx)` for each layer; assert each returned tensor `torch.equal` to the eager `hs.reshape(n_prompts, seq_len, -1).contiguous()`.
  - `test_lazy_mapping_len_and_get_contract`: after `on_load`, assert `len(cache) == n_layers` WITHOUT having materialized any tensor (the `len` must come from the presence scan, matching the orchestrator:710 HIT-log call), and assert `cache.get(absent_idx) is None` for a key not present (matching block_refine:465 `.get` semantics).
  - `test_partial_cache_still_all_or_nothing_miss`: remove ONE layer's sidecar/manifest; assert `on_load` returns `None` (full miss) BEFORE any training — i.e. presence validation stays eager/all-or-nothing. This is the contract-preservation gate.
  - (if prefetch is implemented) `test_prefetch_does_not_change_values`: same as the first test with prefetch enabled.
- These run on CPU; no GPU needed.

### Levers 1 + 2 (NO existing byte-identical golden AND no existing driver — build BOTH from scratch — LARGEST implementation cost)
The Stage 3 golden snapshot (`test_stage3_golden_snapshot.py`) runs with `block_refine` **OFF** (docstring lines 18–22: "captured with `block_refine` OFF"), so it CANNOT gate the block_refine changes. The mechanism-only argument (above) is necessary but NOT sufficient.

**There is NO existing harness to reuse.** `test_stage3_plugin_block_refine.py` is import/re-export/metadata/protocol/gate assertions ONLY (`test_block_refine_module_imports`, `test_monolith_reexports_block_refine_symbols`, `test_plugin_satisfies_protocol`, `test_plugin_metadata`, `test_plugin_is_enabled_gates`, `test_plugin_has_refine_blocks_hook`) — **NONE of them invoke `_phase_c5_block_refine` or build a runnable model**, there is no `tiny_model` fixture there, and no `model(input_ids=...)` forward. So "reuse the tiny_model fixture" is wrong. The harness must be **built from scratch** — acknowledge this as the single largest implementation cost of the whole plan.

- **NEW test** `max_quality/tests/test_stage3_block_refine_optimized_equiv.py`:
  1. **Build a synthetic student+teacher decoder stack** with `FactoredExperts` MoE layers that:
     - runs `model(input_ids=...)` forwards (required by `_capture_block_input` / `_capture_first_pass`, which register forward hooks and run the model),
     - supports `iter_decoder_layers` (the driver walks decoder layers),
     - satisfies the **14-arg `_phase_c5_block_refine` keyword contract** (`block_refine.py:165–181`: positional `student, teacher, moe_layers, teacher_moe_layers, calib_tensor` + keyword-only `batch_size, learning_rate, epochs, warmup_ratio, weight_decay, artifacts_dir, no_resume, device, teacher_targets_cache=None`).
     Use a small `calib_tensor`, `epochs > 1` (e.g. 2) so the per-step re-upload is exercised across epochs, and `batch_size` such that `n_batches >= 2`.
  2. Run `_phase_c5_block_refine` to completion under a deterministic seed and snapshot the trained `FactoredExperts` U/V slots + the four RMSNorm scales (`input_layernorm`, `post_attention_layernorm`, `self_attn.q_norm`, `self_attn.k_norm`).
  3. **Compare old vs new code path via captured golden (no production toggle/monkeypatch — project rule).** Capture a GOLDEN of the trained factors from `origin/main`'s pre-optimization code FIRST (run the harness against `origin/main` once, save the state_dict tensors under `max_quality/tests/golden/stage3_block_refine/`), then assert the post-optimization run reproduces it bit-identically via `torch.equal`. Do NOT add a production toggle/branch just for the test, and do NOT `monkeypatch.setattr` production code. Document the same determinism caveat as the existing golden (same wheel / same host for capture + verify).
  4. Assert `torch.equal` on every trained tensor (U/V for each MATRIX_NAME, all 4 norms). bf16 trained factors are restored to original dtype at lines 538–542; compare in that final dtype.
  5. **SELF-VERIFY the pinned path is actually exercised:** the harness MUST assert `epochs > 1` AND `n_batches >= 2` (so the per-step pinned+async H2D fires repeatedly), and MUST cover **BOTH the live-teacher-target path AND the cache-hit path** (`teacher_targets_cache` populated) so the pinned upload is verified for both target sources.
- **Determinism note:** PyTorch CPU/GPU op order is unchanged by Levers 1+2, so `torch.equal` (not `allclose`) is the correct assertion. If GPU non-determinism surfaces (it should not — op order is identical), that is a RED FLAG that the change is not actually mechanism-only; investigate, do not relax to `allclose`.
- Capture the golden from `origin/main` (this base) before touching block_refine.

### Full-suite regression
After all four levers: run the Stage 2 + Stage 3 test suites (`pytest max_quality/tests/test_stage2_expert_distill.py max_quality/tests/test_stage2_plugin_expert_distill.py max_quality/tests/test_stage3_block_hidden_cache.py max_quality/tests/test_stage3_plugin_block_refine.py max_quality/tests/test_stage3_golden_snapshot.py max_quality/tests/test_stage3_block_refine_optimized_equiv.py -v`) plus the existing Stage 3 golden (must remain byte-identical since block_refine is OFF there — proves no accidental change to the OFF path).

---

## Risks & rollback

| Lever | Risk | Mitigation / rollback |
|---|---|---|
| 1+2 | **Pinned host RAM pressure** on the H200 recovery host — pinned memory is non-pageable; pinning the full `X_student`/`teacher_targets`/`X_teacher` lists is ~188 GiB worst-case. | Verify the host's system-RAM headroom before pinning the full lists. If pressure appears, pin only the per-step source slice about to upload (rolling pin), or pin `X_student`/`teacher_targets` and leave `X_teacher` pageable. This is a HOST-RAM concern; it never touches the 16 GB GPU budget (the 5080 is the equivalence-test host only). Rollback: drop `.pin_memory()` + `non_blocking=True`, revert to the pageable per-step `.to()`. |
| 1+2 | `non_blocking=True` on pageable (un-pinned) memory is **silently synchronous** — pin + non_blocking must land TOGETHER or there is no gain. | They are a single commit; the equivalence harness's `epochs>1`+`n_batches>=2` self-check exercises the repeated per-step upload. |
| 1+2 | Resume-from-checkpoint path (lines 339–371) `continue`s before the train loop. | No change needed there; the pinned/async upload is only in the per-step train branch. Verify the resume test still passes. |
| 1+2 | Real end-to-end gain is **smaller than "~20% of step"** and shrinks at real dims (fixed H2D vs growing fwd+bwd). | Recompute the measured H2D fraction at real recovery dims during implementation; report the measured share, do NOT claim "~103 GB eliminated" (nothing is eliminated — each per-step transfer is just faster). |
| 3 | **Miss-detection contract regression** — a partial/lazy cache silently mixing cached + live targets. | Validation stays eager & all-or-nothing (design step 1). NEW test `test_partial_cache_still_all_or_nothing_miss` is the explicit gate. Rollback: revert to eager materialize (current code). |
| 3 | `load_block_hidden` cannot peek shape/`n_tokens` cheaply (it deserializes the whole payload; manifest has no shape field). | Design committed to (b): presence-validate up-front (stat + manifest, no payload load), lazy-materialize per layer (kills the 52 GB held-RAM AND largely kills the 21 s stall). Shape validation stays out of scope — the existing per-layer fall-through guard (block_refine 466–492) covers a malformed-but-present sidecar. |
| 3 | Lazy mapping missing `__len__`/`.get` → orchestrator:710 `len(...)` HIT-log crashes, or block_refine:465 `.get` mis-behaves. | Lazy-mapping contract mandates `__len__` (validated layer count, no materialize) + `.get(idx)` returning `None` for absent keys; unit-asserted. |
| 3 | Background prefetch thread races / holds an extra layer in RAM. | Prefetch is OPTIONAL and deferred; one-deep only (≤2 layers resident). Recommend shipping lazy-materialize without prefetch first. |
| 4 | `grouped` empty/None at snapshot time → narrowing drops everything. | `if grouped` guard falls back to full snapshot. Verified `grouped` is set (orch 608) before the phase (orch 1568); the guard is defensive only. |
| 4 | Existing `set(snap.keys()) == set(range(n))` test breaks. | Optional `members=None` default preserves the full-snapshot signature; the existing test passes unchanged. |

**Global rollback:** each lever is an independent commit; revert any one without affecting the others. The default-OFF gates mean a bad lever cannot affect a run that hasn't enabled the plugin.

---

## Testing plan on the equivalence-test host RTX 5080 (16 GB)

> The 5080 is the EQUIVALENCE-TEST host (tiny KB-scale fixtures only). The recovery run is on the H200. No lever's production budget is bound by the 5080.

1. **CPU-only tests** (Levers 3 + 4 + the equivalence harness's CPU mode): `pytest` the listed files. No VRAM concern.
2. **Levers 1+2 host-RAM sanity (BEFORE enabling on the real run):** the pinned working set is a HOST-RAM concern on the H200, not a GPU budget. Verify the recovery host's system-RAM headroom against the pinned lists (worst-case ~188 GiB if all of `X_student`/`teacher_targets`/`X_teacher` are pinned; the per-step GPU tensor is only ~0.5 GiB). There is NO resident-hoist, so there is no per-block GPU OOM gate to compute. This is a documented manual host-RAM check, not a unit test.
3. **Equivalence test on GPU** (if the host can run the tiny fixture on CUDA): run `test_stage3_block_refine_optimized_equiv.py` with `device=cuda` to exercise the real pinned+async H2D path; `torch.equal` must hold. The tiny fixture's working set is KB-scale, so no OOM risk there.
4. **Golden non-regression:** `test_stage3_golden_snapshot.py` (block_refine OFF) must stay byte-identical — proves the OFF path is untouched.
5. Capture the Levers-1+2 golden artifact from `origin/main` (this base) on THIS host before any block_refine edit, so capture and verify share wheel + platform (existing golden determinism caveat).

---

## Section-by-section outline (for the report)

1. Tier-1 contract
2. Levers 1+2 (merged) — block_refine pinned-memory + `non_blocking=True` per-step H2D (current 509–512, producers 294/503/624/629; NO resident-hoist — ~134 GB both-lists is a guaranteed H200 OOM; honest per-transfer ~3.25× gain that shrinks at real dims; budget table from real config)
3. Lever 3 — block_hidden_cache lazy-materialize (design (b): presence-validate up-front + lazy per-layer materialize; metadata peek infeasible — `torch.load weights_only=False` + no shape in manifest; lazy-mapping `__len__`/`.get` contract; preserves all-or-nothing presence miss; optional one-deep prefetch deferred)
4. Lever 4 — expert_distill snapshot→members (helper 489–515 + hook 999–1008; `reads` is ADVISORY only — runtime ordering orch:608<orch:1568 is what's safe; narrow via `set().union(*grouped.values())`)
5. Ordering (4 → 3 → 1+2)
6. Verification gates (Lever 4 equality test; Lever 3 lazy + `__len__`/`.get` + all-or-nothing presence tests; Levers 1+2 build-from-scratch bit-identity golden harness — the real gate AND the largest implementation cost; no toggle/monkeypatch, golden captured from origin/main)
7. Risks & rollback (Levers 1+2 host-RAM pinned pressure; Lever-3 presence miss + lazy-mapping contract; per-lever independent revert)
8. Equivalence-test host RTX 5080 testing plan
