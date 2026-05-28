# PLAN — W-1: Wanda `scalar_row` calibration sidecar

**Status**: planner-only. No code changed. Recommendation below.
**Repo**: `/home/lucas/moe_compress` (main @ `1fe4ba5`).
**Date**: 2026-05-29.
**Auditor finding tracked**: `tasks/AUDIT_CALIBRATION_COMPLETENESS_V2.md` §W-1.
**Predecessor TODO** (run-scope variant): `tasks/todo_wanda_compose_with_collect_covariances.md`.

---

## 1. Goal & context

`stage3/plugins/wanda_intra_expert_score.py:74-86` honestly discloses
**~2× Stage 3 calibration wall-clock when enabled** because the plugin
runs its own per-layer calibration sweep instead of composing with
`_collect_covariances`. The accumulator state — a per-(layer, expert,
matrix) running mean of `(x · g_e)²` per input channel — is
**deterministic given the same teacher inputs + routing weights**, so it
is a textbook calibration sidecar candidate.

A0..A11 ablation grids that exist precisely to compare intra-expert
pruning strategies pay this cost on EVERY row.

This plan picks between two competing strategies, both ~half-day lifts.

---

## 2. Two competing strategies

### (A) Compose with `_collect_covariances` — run-scope, zero new sidecar

* **What changes**: extend
  `stage3/plugins/covariance_collection.py::_collect_covariances` to
  accept `extra_callbacks: dict[str, list[Callable]] | None = None`;
  fold the Wanda accumulator's `input` + `intermediate` callbacks into
  the existing per-layer sweep. `instrument_experts` accepts a
  `dict[str, CallbackFn]` (one fn per name), so the composition either
  requires (i) a tiny fan-out wrapper that chains the cov callback +
  the wanda callback under each name, or (ii) lifting
  `instrument_experts` to accept `dict[str, list[CallbackFn]]` (cleaner
  but touches a hot utility).
* **Net effect**: Wanda's standalone per-layer sweep disappears; one
  forward pass covers both B-cov and scalar_row. ~half-day refactor
  including tests.
* **Pros**:
  - No new on-disk schema. No new manifest. No new flag.
  - Single source of truth: scalar_row only exists at run-time
    inside `WandaIntraExpertScorePlugin`.
  - Closes `D-zero-extra-forward` deviation outright (the brief's
    original promise).
  - Closes `tasks/todo_wanda_compose_with_collect_covariances.md`
    + audit W-2 in the same patch.
* **Cons**:
  - Saves the cost **once per run**, but every A0..A11 ablation row
    still recomputes scalar_row from scratch even though the inputs
    are identical across rows (the ablation varies the *score
    consumer*, not the calibration teacher or batches). The 11×
    repeat cost is the lever this plan targets.
  - Couples Wanda's lifecycle to `_collect_covariances` — a future
    refactor (e.g. moving cov collection into a different driver)
    has to drag Wanda along.

### (B) New `--capture-wanda-scalar-row` calibration flag + sidecar

* **What changes**: a new capture flag and sidecar that promotes
  `scalar_row` to a cross-run artifact, exactly parallel to
  `--capture-input-covariance` / `covariance.pt`.
  - New vLLM patch hook (or extension of `expert_in`) wires the
    per-(layer, expert, matrix) accumulator to vLLM's calibration
    dispatch.
  - New sidecar `wanda_scalar_row.pt` next to the JSONL.
  - New schema entry `SCHEMA_VERSIONS["wanda_scalar_row"] = 1`.
  - New dataclass + save/load pair in
    `utils/cached_calibration_signals.py`.
  - New `Stage3WandaScalarRowCacheProvider` (mirror of
    `Stage3InputCovCacheProvider`); on hit, populates
    `ctx["stage3.wanda_scalar_row"]` and `WandaIntraExpertScorePlugin`
    short-circuits its per-layer pass.
* **Net effect**: the first ablation row pays the capture cost (which
  is now FREE because it rides the existing calibration forwards
  inside vLLM); rows 2..N (A0..A11) skip the Stage-3 calibration
  sweep entirely.
* **Pros**:
  - Cross-run, cross-ablation amortization (the A0..A11 sweep is
    the explicit reason this plugin exists).
  - Aligns Wanda with the project's standing sidecar inventory
    (audit §6 lists 10+ existing capture flags following the same
    pattern).
  - Wanda calibration moves to the FREE side of the cost curve
    (lives inside vLLM's existing batched generate, not behind a
    separate moe_compress HF-forward sweep).
  - The math is unchanged — `scalar_row` is the SAME running mean
    upstream and downstream; only the producer moves.
* **Cons**:
  - Touches the vLLM calibration-hooks patch (one new
    `vllm.calibration_wanda_scalar_row` module, ~250 LOC mirroring
    `calibration_input_cov.py`).
  - **`expert_in` does not currently carry `topk_weights`** — see §3
    below. The patch must either (a) extend `expert_in` dispatch
    with `topk_weights`, or (b) cross-correlate `router` and
    `expert_in` on `layer_idx`.
  - Coexistence with the existing capture cohort — backed by F-H-7
    sidecar isolation (Audit Surface 7) — is well-trodden, so this
    is mechanical work, not novel risk.

### Recommendation

**Strategy B** — promote `scalar_row` to a calibration sidecar.

Rationale:

1. The auditor's W-1 finding (rated **HIGH**) explicitly upgrades the
   fix from "compose with `_collect_covariances`" (Strategy A's scope)
   to "promote to calibration sidecar". The upgrade reasoning is that
   Wanda exists primarily for the A0..A11 ablation grid; the
   amortization payoff is across rows, not within a single row.
2. The calibration phase ALREADY runs every forward Wanda needs (the
   teacher is identical, the batches are identical, the routing is
   deterministic per the existing `--capture-input-covariance`
   contract). Capturing scalar_row inside vLLM is FREE in wall-clock
   — it rides the same `expert_in` hook the input-cov writer already
   pays for.
3. Strategy A's saving is a one-time-per-row halving (~1× Stage 3
   cal). Strategy B's saving is ~2× Stage 3 calibration **per
   ablation row** plus the original cov-pass amortization, repeated
   across all 12 A0..A11 rows (Stage 3 cov cache HIT × 12 + Wanda
   scalar_row cache HIT × 12).
4. Disk cost is trivial (see §8).

**Strategy A is NOT abandoned** — it should land as a follow-up
patch only IF the realised A0..A11 cadence drops below ~3 rows
(otherwise Strategy B's cross-row amortization dominates). In
practice the cadence is 12 rows per teacher + mix combo, so the
crossover is irrelevant.

The D-tag updates in §6 below assume Strategy B; §6 also lists the
Strategy A variant for completeness.

---

## 3. Architectural question — how does Wanda see `topk_weights` inside the vLLM patch?

This is the only non-trivial design call. Wanda's `scalar_row` is

    scalar_row[c] = E_t [ (x[t, c] · g_{e, t})² ]

per channel `c`, per (layer, expert) `e`, where `g_{e, t}` is the
per-token routing weight assigned to expert `e`. vLLM's
`expert_in(layer_idx, hidden_states, topk_ids)` dispatch currently
carries `hidden_states` + `topk_ids` only — **no `topk_weights`**.

(Verified by reading `vllm_calibration_hooks.patch:642-647` for the
synthetic payload shape and `:7147` for `calibration_input_cov`'s
`_on_expert_in` signature — `topk_weights` is absent.)

The `router` dispatch DOES carry `topk_weights` + `topk_ids`
(`vllm_calibration_hooks.patch:633-639`).

**Two options inside Strategy B**:

* **B.1 — extend `expert_in` to carry `topk_weights`** (~10 LOC patch
  to the vLLM dispatch site for `expert_in`; backward-compatible
  because existing callbacks take `**kwargs`).
* **B.2 — observe `router` first, stash `topk_weights[layer_idx]` in a
  module-local dict, look it up inside `_on_expert_in`**. No vLLM
  patch dispatch change; ~5 LOC of bookkeeping inside the new
  `vllm.calibration_wanda_scalar_row` module. The
  `router` hook fires per MoE layer per forward already (it's how
  `calibration_routing_stats` already operates), so layer-keyed
  stashing is correct as long as the dispatch order is router →
  expert_in within one layer (it is, by the kernel structure).

**Recommend B.2** — local to the new module, no patch fan-out.
Document the layer-keyed stash + dispatch-order assumption in the
new module's docstring + a one-test smoke that simulates
out-of-order dispatch and asserts the error path is loud
(`assert router_observed_for_this_layer`).

---

## 4. Files to create / modify

### Create

| Path | Purpose | Approx LOC |
|---|---|---|
| `max_quality/patches/vllm_calibration_hooks.patch` (extend) — adds new module **`vllm/calibration_wanda_scalar_row.py`** | per-(layer, expert, "gate_proj"/"down_proj") scalar_row accumulator + `_on_router` stash + `_on_expert_in` reader + dump | ~280 |
| `max_quality/src/moe_compress/stage3/plugins/wanda_scalar_row_cache.py` | `Stage3WandaScalarRowCacheProvider` (mirror of `Stage3InputCovCacheProvider`) — on hit populates `ctx["stage3.wanda_scalar_row"]` and returns the payload | ~95 |
| `max_quality/tests/test_calibration_wanda_scalar_row_smoke.py` (inside patch) | smoke tests for the new vLLM module (env-gate, accumulation, dump, checkpoint, router-first ordering) | ~250 |
| `max_quality/tests/test_stage3_wanda_scalar_row_cache.py` | cache-provider tests: hit, miss, schema mismatch, dtype round-trip | ~150 |

### Modify

| Path | Change | Approx LOC delta |
|---|---|---|
| `max_quality/src/moe_compress/utils/cached_calibration_signals.py` | + `WandaScalarRowPayload` dataclass; + `SCHEMA_VERSIONS["wanda_scalar_row"] = 1`; + `save_wanda_scalar_row` / `load_wanda_scalar_row`; both call **`write_manifest_last`** + **`read_and_validate_manifest`** so this sidecar lands compliant with the audit's S-1 push from day one. | +80 |
| `max_quality/scripts/build_self_traces_calib_vllm.py` | + `--capture-wanda-scalar-row` arg group (mirrors `--capture-input-covariance` block at lines 760-783); + env-var `VLLM_CALIB_CAPTURE_WANDA_SCALAR_ROW=1` auto-enable in the pre-vllm-import section (line ~1041); + `dump_wanda_scalar_row` call at run-end (line ~2059 pattern); + `dump_wanda_scalar_row_checkpoint` in the periodic-checkpoint loop (line ~1856 pattern) | +60 |
| `max_quality/src/moe_compress/stage3/plugins/wanda_intra_expert_score.py` | (1) Add `reads += ("stage3.wanda_scalar_row",)`. (2) At top of `collect_wanda_scores`, check `ctx.has("stage3.wanda_scalar_row")` — on hit, hydrate `_WandaScalarRowAccumulator._cpu` from the payload and SKIP the per-layer calibration sweep entirely. (3) Update module docstring's `D-zero-extra-forward` block (lines 70-86) to point at the new cache provider — see §6. | +30 / -0 |
| `max_quality/src/moe_compress/stage3/orchestrator.py` | Register `Stage3WandaScalarRowCacheProvider()` in the same `_cache_only_plugins` list that holds `Stage3InputCovCacheProvider` (line ~133) | +1 |
| `max_quality/src/moe_compress/stage3/plugins/__init__.py` | Re-export `Stage3WandaScalarRowCacheProvider` for symmetry with the input-cov cache | +2 |
| `max_quality/patches/MANIFEST.md` | Bump entries: new module declared; SCHEMA_VERSIONS row added | +5 |

**Total**: ~950 LOC across 4 new files + 5 modifications, dominated
by the vLLM patch's accumulator + tests.

---

## 5. New sidecar contract

### 5.1 Pattern B — `format_version` in payload

```python
@dataclass
class WandaScalarRowPayload:
    """Per-(layer, expert, matrix) Wanda scalar_row running mean.

    On-disk shape mirrors the in-memory state of
    ``_WandaScalarRowAccumulator``:

        sigma_x_g_squared: dict[(layer_idx, expert_idx, matrix_name), Tensor[d_in] fp32]
        token_counts:      dict[(layer_idx, expert_idx, matrix_name), int]

    Two ``matrix_name`` values are present: ``"gate_proj"`` and
    ``"down_proj"`` — ``up_proj`` aliases ``gate_proj`` at compute time
    (per the existing D-gate-up-share deviation), so it is NOT stored.
    """
    schema_version: int   # = SCHEMA_VERSIONS["wanda_scalar_row"] = 1
    n_layers: int
    n_experts: int
    sigma_x_g_squared: dict  # (li, ei, name) -> Tensor[d_in] fp32 (storage)
    token_counts: dict       # (li, ei, name) -> int
```

The `schema_version` field IS the Pattern B `format_version` (the
existing sidecars in `cached_calibration_signals.py` use the name
`schema_version`; new sidecars MUST keep that name for `_check_schema`
to fire).

### 5.2 Pattern O — atomic write + manifest-last

* Writer (`save_wanda_scalar_row`):
  1. `tmp = path.with_suffix(".pt.tmp")`
  2. `torch.save(payload, tmp)` followed by `fsync(fd)` + `os.replace`
     + `fsync(parent)` — this is what `_atomic_torch_save` already
     does inside `cached_calibration_signals.py`.
  3. `write_manifest_last(path, manifest_path, schema_version=1,
     extra_meta={...})` — **new** for this sidecar. The 10 existing
     `save_*` pairs do NOT call this (audit §S-1), but this plan
     opts in unilaterally so the new sidecar lands compliant.
* Reader (`load_wanda_scalar_row`):
  1. `read_and_validate_manifest(path, manifest_path,
      schema_version=1)` — on missing manifest, emit a one-shot
     WARNING and fall back to bare `torch.load` (mirrors F-S3-1
     backward-compat shim).
  2. `_check_schema("wanda_scalar_row", loaded.schema_version, path)`.
  3. Return the typed payload.

### 5.3 Path layout (Surface 7 / F-H-7 compliant)

* New layout: `<jsonl.parent>/sidecars/<jsonl.stem>/wanda_scalar_row.pt`
* Manifest: `<jsonl.parent>/sidecars/<jsonl.stem>/wanda_scalar_row.pt.MANIFEST.json`
* Legacy fallback: `<jsonl.parent>/sidecars/wanda_scalar_row.pt` —
  resolved by the existing `_resolve_sidecar_for_load` (no extra
  code; `sidecar_path` + `_resolve_sidecar_for_load` already handle
  the namespaced layout for any signal name registered in
  `SCHEMA_VERSIONS`).

---

## 6. CLI flag wiring (`build_self_traces_calib_vllm.py`)

Mirror the `--capture-input-covariance` block exactly. Three insertion
points in the existing script:

### 6.1 New argparse args (insert after the `--capture-input-covariance`
group at line 783)

```python
p.add_argument("--capture-wanda-scalar-row", action="store_true",
               default=False,
               help="Capture per-(layer, expert, matrix) Wanda "
                    "scalar_row = E[(x · g_e)^2] per input channel "
                    "during calibration and write a moe_compress-side "
                    "sidecar at <jsonl>/sidecars/<stem>/"
                    "wanda_scalar_row.pt at run end (schema v1). "
                    "Consumed by Stage 3's WandaIntraExpertScorePlugin "
                    "via Stage3WandaScalarRowCacheProvider, which "
                    "short-circuits the plugin's own per-layer "
                    "calibration sweep (saving ~2x Stage 3 cal "
                    "wall-clock per A0..A11 ablation row). Requires "
                    "the vLLM calibration-hooks patch "
                    "(vllm.calibration_wanda_scalar_row). Auto-enables "
                    "VLLM_CALIB_CAPTURE_WANDA_SCALAR_ROW=1 + "
                    "VLLM_CALIB_CAPTURE_ROUTER=1 + "
                    "VLLM_CALIB_CAPTURE_EXPERT=1 BEFORE any vllm "
                    "import. Failures during dump are logged but do "
                    "NOT re-raise -- the JSONL is more valuable than "
                    "the wanda sidecar.")
p.add_argument("--wanda-scalar-row-checkpoint-every-chunks", type=int,
               default=1, help="...")
```

### 6.2 Env-var auto-enable (line ~1041 — same block as input_cov)

```python
if args.capture_wanda_scalar_row:
    os.environ["VLLM_CALIB_CAPTURE_WANDA_SCALAR_ROW"] = "1"
    os.environ["VLLM_CALIB_CAPTURE_ROUTER"] = "1"
    os.environ["VLLM_CALIB_CAPTURE_EXPERT"] = "1"
    log.info("wanda_scalar_row: capture ON")
```

### 6.3 Periodic checkpoint (line ~1856 block) + final dump (line ~2059 block)

Mirror the `_icov.dump_input_cov_checkpoint` / `_icov.dump_input_cov`
patterns exactly. Imports: `import vllm.calibration_wanda_scalar_row
as _wsr`.

---

## 7. Plugin code changes (`wanda_intra_expert_score.py`)

### 7.1 Reader-side short-circuit

At the top of `collect_wanda_scores` (line ~444), after config
validation but BEFORE the per-layer sweep at line 498:

```python
# CACHE HIT: hydrate _WandaScalarRowAccumulator from sidecar and
# skip the per-layer calibration pass entirely. Mirrors
# Stage3InputCovCacheProvider's contract — the cache provider has
# already validated schema + manifest by the time it lands on ctx.
if ctx.has("stage3.wanda_scalar_row"):
    payload = ctx.get("stage3.wanda_scalar_row")
    log.info(
        "wanda_intra_expert_score: cache HIT, %d entries — skipping "
        "calibration sweep",
        len(payload.sigma_x_g_squared),
    )
    acc = _WandaScalarRowAccumulator(scalar_row_dtype=scalar_row_dtype)
    for key, sigma in payload.sigma_x_g_squared.items():
        acc._cpu[key] = sigma.to(scalar_row_dtype)
        acc._nsamples[key] = int(payload.token_counts[key])
    # Skip directly to _compute_scores below.
else:
    # ... existing per-layer sweep at lines 498-528 ...
```

The `_cpu` private-dict write is intentional — the cache provider
short-circuits the accumulator's update path entirely; finalize
semantics are vacuous (the payload is already finalized at
calibration time).

### 7.2 D-tag updates (D-zero-extra-forward)

Replace the existing docstring block at lines 68-86 with:

```
Deviations from upstream
------------------------
**D-zero-extra-forward (RESOLVED via calibration sidecar — W-1)**.
The brief promised "zero extra forward cost" because the routing
weights are already collected during the covariance pass.
Resolution: ``scalar_row`` is captured as a calibration sidecar by
``vllm.calibration_wanda_scalar_row`` (gated on
``--capture-wanda-scalar-row``). On cache HIT, this plugin skips its
per-layer calibration sweep entirely and hydrates the accumulator
state directly from the sidecar — true zero extra forward.

On cache MISS (e.g. a sidecar was not captured), the plugin falls
back to the per-layer calibration sweep at lines ~498-528 (mirrors
``_collect_covariances`` structure) — the original ~2× Stage 3 cal
cost is retained as the fallback path so production runs that omit
the capture flag still succeed.

See:
* ``tasks/PLAN_W1_WANDA_SCALAR_ROW_CAPTURE.md`` for the plan
* ``stage3/plugins/wanda_scalar_row_cache.py`` for the cache provider
* ``vllm/calibration_wanda_scalar_row.py`` for the writer

**Honest cost** (cache MISS only): ~2× Stage 3 calibration
wall-clock when ``--capture-wanda-scalar-row`` was NOT set during
calibration. Cache HIT: zero extra forward.

A future patch may compose the fallback path with
``_collect_covariances`` (audit W-2 / Strategy A above) for the
no-sidecar path; this is independent of W-1 and tracked at
``tasks/todo_wanda_compose_with_collect_covariances.md``.
```

The Strategy-A variant of this docstring (if W-2 lands instead of
W-1) would keep the "deferred" wording but point at the run-scope
composition; included here for completeness only.

---

## 8. Cost analysis

### 8.1 Disk

Auditor estimate: `L × E × 2 matrices × d_in × 4 B ≈ 40 × 256 × 2 ×
2048 × 4 = 168 MB`.

Refining with actual Qwen3.6-35B-A3B dims (hidden=2048, moe_inter=768,
L_moe=40, E=256, fp32 storage as upstream does — matches the
in-memory `_WandaScalarRowAccumulator._cpu` dtype):

* `gate_proj` scalar_row: `40 × 256 × 2048 × 4 B = 80 MB`
* `down_proj` scalar_row: `40 × 256 × 768 × 4 B = 30 MB`
* token_counts: trivial (~80 KB)

**Total ≈ 110 MB on disk** (auditor's 168 MB was a worst-case;
realised will be lower because down_proj's d_in is smaller).

Storage policy: fp32 in the payload (per the `_WandaScalarRowAccumulator`
docstring rationale — the sum-of-squares numerically dominates and
underflows in fp16 on long calibration runs). The on-disk dtype is
controlled by a new arg `--wanda-scalar-row-storage-dtype`
(default `float32`; choices `float32`/`float16`/`bfloat16` mirror
`--stage2-profile-cov-storage-dtype`).

### 8.2 Calibration wall-clock delta

* **Capture side (one-time, inside vLLM forward)**: scalar_row
  accumulation is a per-(layer, expert) `(x * g)^2` reduce sum
  along the channel axis, on the CPU fp32 copy that the existing
  input_cov module already pays for at `vllm_calibration_hooks.patch:
  7187` (`hs.to("cpu", dtype=torch.float32, copy=True)`). The new
  module DOES NOT re-copy `hidden_states` — it RE-USES the cov
  module's CPU buffer pattern. Estimated overhead: **<5%** of the
  cov capture cost (a sum-of-squares reduce per expert vs a
  `d_in × d_in` matmul per expert).
* **Consumer side (per ablation row)**: cache HIT collapses
  ~2× Stage 3 calibration → 0. At ~5 min/Stage-3-cal-pass on H200
  (per the run cost tables in the OPT plans), this is **~10 min
  saved per A0..A11 row × 12 rows = ~2 h saved per calibration mix
  per teacher**.

### 8.3 Net cost reduction estimate

Assuming the A0..A11 sweep is run once per `(teacher, calibration
mix)` combo (current cadence):

| Quantity | Without W-1 | With W-1 |
|---|---|---|
| Stage 3 cal passes per A0..A11 row | 2 (cov + wanda) | 1 (cov only — cov cache HIT also reduces this to 0 once the operator captures it) |
| Wall-clock per A0..A11 row at Stage 3 cal | ~10 min | ~5 min |
| Wall-clock for 12-row sweep | ~120 min | ~60 min |
| Net saving | — | **~60 min per sweep** |
| Capture-side overhead | — | trivial (<30 s amortized across the calibration generate) |
| Disk | — | +110 MB |

The realised saving is **larger** when paired with the cov sidecar
HIT (`Stage3InputCovCacheProvider` already exists and is the same
pattern). In the fully cached state: A0..A11 row 1 pays the
calibration ONCE (in vLLM); rows 2..12 are pure Stage 3 factor +
score compute — no calibration sweep at all.

---

## 9. Test plan (4+ tests)

### T1. Calibration-side accumulation correctness (vLLM patch smoke test)

`test_calibration_wanda_scalar_row_smoke.py::test_accumulation_matches_reference`

Drive the new vLLM module with a synthetic `_on_router` then
`_on_expert_in` for one layer, one expert, two batches. Assert the
final dumped `sigma_x_g_squared[key]` equals the reference
running-mean computed by
`stage3.plugins.wanda_intra_expert_score._WandaScalarRowAccumulator`
on the same inputs. Bit-equality not required (CPU fp32 ↔ CPU
fp32); `torch.allclose(rtol=1e-6)` is the bar.

### T2. Router-before-expert_in ordering invariant (vLLM patch)

`test_calibration_wanda_scalar_row_smoke.py::test_expert_in_without_router_logs_error`

Dispatch `_on_expert_in` for a layer without first dispatching
`_on_router` for that layer. Assert (a) the accumulator does NOT
silently zero the row, (b) a one-shot ERROR log is emitted with
the layer index, (c) the dump still produces a valid (just
empty-for-that-layer) sidecar. This guards the B.2 router-stash
ordering assumption from §3.

### T3. Cache provider hit / miss / schema mismatch

`test_stage3_wanda_scalar_row_cache.py::test_cache_hit_populates_ctx`

* Write a valid sidecar with `save_wanda_scalar_row`.
* Run `Stage3WandaScalarRowCacheProvider().on_load(ctx, jsonl_path)`.
* Assert `ctx.has("stage3.wanda_scalar_row")` and the loaded
  `sigma_x_g_squared` round-trips bit-equality through fp32.

`test_stage3_wanda_scalar_row_cache.py::test_cache_miss_returns_none`

* No sidecar on disk → `on_load` returns `None`,
  `ctx.has("stage3.wanda_scalar_row")` is False.

`test_stage3_wanda_scalar_row_cache.py::test_schema_mismatch_raises`

* Write a sidecar then mutate its `schema_version` to 99.
* Assert `load_wanda_scalar_row` raises `ValueError` with the
  "Delete the sidecar to regenerate" actionable message
  (mirrors `_check_schema`'s contract).

### T4. End-to-end short-circuit in `WandaIntraExpertScorePlugin`

`test_stage3_wanda_intra_expert_score.py::test_plugin_consumes_cache_sidecar`

* Build a tiny model + populate
  `ctx["stage3.wanda_scalar_row"]` directly with a known
  `WandaScalarRowPayload`.
* Run `WandaIntraExpertScorePlugin().collect_wanda_scores(ctx)`.
* Spy on `instrument_experts` — assert it is NEVER called (the
  per-layer sweep is skipped).
* Assert the resulting `ctx["stage3.wanda_intra_expert_score"]`
  byte-equals the score map produced by the existing
  `test_score_map_matches_W_times_sqrt_scalar_row` path when the
  same `scalar_row` is computed live.

### T5. Manifest-last + atomic write (Pattern O)

`test_stage3_wanda_scalar_row_cache.py::test_writer_emits_manifest_after_payload`

* Mock `os.replace` to capture the order of `.pt` rename vs
  `.MANIFEST.json` rename.
* Assert the manifest rename is AFTER the payload rename
  (Pattern O write-order).
* Assert `_resolve_sidecar_for_load` preferring the namespaced
  layout over the legacy layout still works when both exist
  (F-H-7 backward-compat sanity).

### Existing tests that MUST stay green

* All 24 tests in `test_stage3_wanda_intra_expert_score.py` —
  the no-cache (miss) path is the existing behaviour and must
  remain byte-identical.

---

## 10. Risks & mitigations

1. **Router-before-expert_in dispatch order** (§3, B.2 assumption).
   Mitigation: T2 above + a defensive `assert layer_idx in
   _router_stash` inside `_on_expert_in` that logs once with the
   actionable "did you set VLLM_CALIB_CAPTURE_ROUTER=1?" message
   instead of silently producing wrong scalar_row.

2. **fp32 storage size on multi-TB calibration mixes**. 110 MB is
   trivial today, but if a future calibration mix scales `d_in` (e.g.
   a 8K-hidden model), the gate_proj entry grows quadratically.
   Mitigation: the storage dtype is operator-configurable
   (`--wanda-scalar-row-storage-dtype`); default fp32; fp16/bf16
   available with a docstring caveat about underflow risk on
   long-calibration runs (mirrors the cov sidecar's posture).

3. **Stage 3 cache provider miss-then-hit on partial captures**.
   If an operator runs calibration without
   `--capture-wanda-scalar-row` then re-runs Stage 3 with
   `stage3.wanda_intra_expert.enabled=True`, the cache provider
   misses cleanly and the existing per-layer fallback fires. This
   is the SAME contract as `Stage3InputCovCacheProvider`; no new
   risk.

4. **W-1 vs S-1 coupling** (audit §S-1 — 10 existing sidecars
   lack manifests). This plan adds an 11th sidecar that ALREADY
   has the manifest, which is the opposite asymmetry from today.
   Mitigation: the manifest-last code path is new code that lands
   compliant from day one; the audit's S-1 work item is
   orthogonal and can lift the 10 existing sidecars at its own
   cadence.

---

## 11. Out of scope

* **Strategy A** (compose with `_collect_covariances`). Tracked at
  `tasks/todo_wanda_compose_with_collect_covariances.md`. Independent
  of W-1; can land as a follow-up if the no-sidecar path becomes a
  bottleneck (it won't, given the production cadence).
* **Audit S-1** (manifest emission on the 10 existing
  `cached_calibration_signals.py` `save_*` pairs). Separate plan;
  W-1's manifest-emission lands as a "starts compliant" data point.
* **Audit S-2** (`_stage2_input_covariance.pt` manifest). Separate.
* **Audit W-2** (the run-scope variant). Tracked at the existing
  TODO doc; rendered redundant by W-1 on the cache-HIT path but
  retained as the cache-MISS fallback's improvement.
* **Stage 4 consumption of `wanda_scalar_row`**. Stage 4 does NOT
  currently consume this signal (the score map is a Stage 3
  research artifact for A0..A11 only). No reader-side wiring
  beyond `WandaIntraExpertScorePlugin` itself.

---

## 12. Recommended strategy

**Strategy B — calibration sidecar.** Estimated effort: ~half day
(~950 LOC across 4 new files + 5 modifications), dominated by the
new vLLM module + tests. Estimated net cost reduction: **~60 min
per 12-row A0..A11 sweep, per teacher × mix combo**, on top of
the existing cov-sidecar amortization. Disk cost: ~110 MB per
JSONL. Lands a compliant Pattern-O sidecar (manifest-last + atomic
write) from day one — establishes the posture audit §S-1 is
pushing toward.

---

## 13. Open questions for the user

1. **Storage dtype default**: fp32 (110 MB) vs bf16 (55 MB) vs fp16
   (55 MB, underflow risk). Recommendation: fp32; matches upstream
   fusion_bench's accumulator dtype + the plugin's existing
   in-memory choice.
2. **Whether to ALSO ship Strategy A** as a follow-up. Recommendation:
   defer until measured demand — Strategy B alone hits the
   ablation-grid use case, and Strategy A's run-scope saving is
   subsumed once Strategy B's sidecar exists.
3. **vLLM patch B.1 vs B.2** (extend `expert_in` dispatch with
   `topk_weights` vs in-module router stash). Recommendation: **B.2**
   — local to the new module, no fan-out to other patch consumers.

---

**Plan path**: `/home/lucas/moe_compress/tasks/PLAN_W1_WANDA_SCALAR_ROW_CAPTURE.md`
**Recommended strategy**: **B** (calibration sidecar)
**Net cost reduction estimate**: **~60 min saved per 12-row A0..A11
sweep, per (teacher × calibration mix) combo**, fpr ~110 MB disk
overhead per JSONL. Capture-side overhead negligible (<5% of the
existing input-cov capture cost — both ride the same CPU fp32
copy of `hidden_states`).
