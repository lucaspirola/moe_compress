# `vllm_calibration_hooks.patch` — manifest

Single source of truth for the patch's identity. Other places that reference
the patch (the HF Jobs build script, the README uploaded to
`pirola/vllm-patched-calib`) should match this file.

## Current

| Field | Value |
|---|---|
| Immutable tag | `calib-v2-input-cov-writer-chained-callbacks` |
| Branch (active) | `feat/calibration-v2` |
| vLLM upstream SHA | `ad7125a43e176d4161099480a66f0169609a690` (v0.21.0) |
| Patch line count | **5350** |
| Patch MD5 | **`c35dc497cd3e9268c7448410bdddf80c`** |
| HF model repo | `pirola/vllm-patched-calib` |
| Wheel filename pattern | `vllm-0.21.1.dev0+gad7125a43.d<YYYYMMDD>-cp312-cp312-linux_x86_64.whl` |
| Torch / CUDA pinned in build | `torch==2.11.0+cu130` |
| `TORCH_CUDA_ARCH_LIST` | `8.0;9.0a;10.0;12.0` (A100 / H100·H200 / B200 / RTX 6000 Pro) |

## Verifying locally

```bash
md5sum max_quality/patches/vllm_calibration_hooks.patch
# expect: c35dc497cd3e9268c7448410bdddf80c
wc -l max_quality/patches/vllm_calibration_hooks.patch
# expect: 5350

# Re-apply against a fresh v0.21.0 checkout (idempotency check):
git clone --depth 1 --branch v0.21.0 https://github.com/vllm-project/vllm /tmp/vllm-fresh
cd /tmp/vllm-fresh
git apply --check /path/to/vllm_calibration_hooks.patch && echo OK
```

## Change log

### `calib-v2-input-cov-writer-chained-callbacks` (current)
Review-fix follow-up to `calib-v2-input-cov-writer`. The single-slot
callback registry in `vllm/calibration_hooks.py` silently overwrote any
previous subscriber when a second writer registered for the same hook
name -- imatrix's `_on_expert_in` and input-cov's `_on_expert_in` could
not coexist (whichever ran second won; the other's accumulators stayed
empty without warning).

- `vllm/calibration_hooks.py`: `_callbacks: dict[str, list[Callable]]`
  (was `dict[str, Callable | None]`). `register_callback(name, fn)` now
  APPENDS `fn` to the list (identity-based de-dup so the same callable
  registered twice fires once), or CLEARS the list when `fn is None`.
  `dispatch(name, ...)` iterates the list in registration order and
  invokes every callback on the same payload.
- `tests/test_calibration_hooks_smoke.py`: +3 tests --
  `test_chained_callbacks_coexist_on_expert_in` pins both writers
  receiving every dispatch; `test_register_callback_dedup_same_callable`
  guards against double-fire on duplicate `register_callback` calls
  (resume / reinit paths); `test_register_callback_none_clears_list`
  pins the clear-on-None contract.
- `vllm/calibration_input_cov.py`: extended docstring on `_on_expert_in`
  explaining the CPU-residency choice (CUDA-graph capture safety), and
  on `dump_input_cov`'s log message clarifying that the sidecar stores
  the SHARED gate+up input covariance under `'gate_proj'` (consumers
  alias up_proj -> gate_proj via `_cov_lookup`).

### `calib-v2-input-cov-writer` (previous)
Adds the per-(layer, expert, "gate_proj") teacher input-covariance Σ_in
writer (Item 1 of the calibration-v2 writers campaign).
- `vllm/calibration_input_cov.py`: new module hooking ``expert_in`` to
  scatter-reduce per-(layer, expert) ``x^T x`` into a CPU fp32
  accumulator dict keyed by ``(layer_idx, expert_idx, "gate_proj")`` --
  the EXACT shape used by the Stage 2 writer's
  ``_stage2_input_covariance.pt``. Final ``dump_input_cov`` writes the
  payload via
  ``moe_compress.utils.cached_calibration_signals.save_covariance``
  (schema bumped to v2 to reflect the dict-valued payload).
  ``dump_input_cov_checkpoint`` / ``load_input_cov_checkpoint`` mirror
  the reap-scores / imatrix resumability cadence (atomic tmp+rename,
  schema-versioned, CPU-resident accumulators -> CUDA-graph-safe).
- `vllm/envs.py`: add ``VLLM_CALIB_CAPTURE_INPUT_COV`` to the
  ``TYPE_CHECKING`` block + the ``environment_variables`` dispatch dict
  for discoverability through ``vllm.envs``.
- `tests/test_calibration_input_cov_smoke.py`: +6 tests covering
  accumulator math, env-off short-circuit, checkpoint round-trip,
  two-segment additivity, dump-payload shape, and lazy-allocation of
  layers not pre-discovered by setup().

Driver-side companion (`build_self_traces_calib_vllm.py`): new flags
``--capture-input-covariance`` + ``--input-cov-checkpoint-every-chunks``,
parallel env-gating block (``VLLM_CALIB_CAPTURE_INPUT_COV=1``,
``VLLM_CALIB_CAPTURE_EXPERT=1``), post-``_load_teacher_vllm`` setup +
resume hydration, periodic in-loop checkpoint, and final dump that
removes the now-stale ckpt.

Stage 3 reader-side companion: ``Stage3InputCovCacheProvider``
(``max_quality/src/moe_compress/stage3/plugins/input_cov_cache.py``)
consulted by the Stage 3 orchestrator's run-glue BEFORE the legacy
``_load_stage2_covariance`` call. On hit the cached payload's
``sigma_in`` dict drops into ``A_cov`` directly; on miss the legacy
path runs. The provider is also registered FIRST in the Stage 3
``PluginRegistry`` for introspection parity with the Stage 2 / Stage 4
cache providers.

Stage 4 reader-side companion: ``Stage4InputCovCacheProvider``
(``max_quality/src/moe_compress/stage4/plugins/input_cov_cache.py``)
registered as the FIRST plugin in ``PluginRegistry``, BEFORE
``EoraInputsPlugin``. Its ``on_load`` populates ``ctx.A_cov`` +
``ctx.a_storage_dtype`` from the sidecar; the orchestrator dispatches
``on_load`` before ``walk_phases("load_eora_inputs", ...)``.
``EoraInputsPlugin.load_eora_inputs`` now starts with a
``ctx.has("A_cov")`` short-circuit guard so the on-disk
``_stage2_input_covariance.pt`` load is skipped on a cache hit.

Schema bump: ``SCHEMA_VERSIONS["covariance"]`` bumped from 1 to 2.
``CovariancePayload`` switched from a single 4-D tensor field (v1) to
the dict-valued layout (v2) -- mandatory because the live consumers in
Stage 3/4 (``_cov_lookup``, ``_compute_eora_factors``) need per-(layer,
expert, matrix) keying that a 4-D tensor cannot express without a
separate index mapping. v1 was never written by any production writer;
this is a forward-only bump.

### `calib-v2-reap-scores-writer` (previous)
Adds the REAP-scores writer (V1+V2 of the calibration-v2 writers campaign).
- `vllm/calibration_reap_scores.py`: new module pairing the `router` +
  `expert_out_unweighted` source-patched hooks to accumulate per-(layer,
  expert) `g_j · ‖f_j‖₂` contributions and per-expert token counts. Final
  `dump_reap_scores` normalizes to `S_j = (1/|X_j|)·Σ g_j·‖f_j‖₂` (REAP
  Eq. 9, arXiv:2510.13999) and writes the sidecar via
  `moe_compress.utils.cached_calibration_signals.save_reap_scores`. Periodic
  `dump_reap_scores_checkpoint` mirrors the imatrix resumability cadence;
  `load_reap_scores_checkpoint` hydrates on `--resume`. Atomic-write
  contract: tmp + `os.replace`, schema-versioned, CPU-resident
  accumulators (CUDA-graph-safe).
- `tests/test_calibration_reap_scores_smoke.py`: +6 tests covering
  accumulator math, env-off short-circuit, checkpoint round-trip,
  two-segment additivity, router-stash-miss guard, and dump
  normalization.

Driver-side companion (`build_self_traces_calib_vllm.py`): new flags
`--capture-reap-scores` + `--reap-scores-checkpoint-every-chunks`,
parallel env-gating block (`VLLM_CALIB_CAPTURE_REAP_SCORES=1`,
`VLLM_CALIB_CAPTURE_ROUTER=1`, `VLLM_CALIB_CAPTURE_EXPERT_UNWEIGHTED=1`,
`VLLM_USE_FLASHINFER_MOE_FP16=0`), post-`_load_teacher_vllm` setup +
resume hydration, periodic in-loop checkpoint, and final dump that
removes the now-stale ckpt.

Stage 2 reader-side companion: `Stage2ReapScoresCacheProvider`
(`max_quality/src/moe_compress/stage2/plugins/reap_scores_cache.py`)
registered as the FIRST plugin in `PluginRegistry`, BEFORE
`ReapScoringPlugin`. Its `on_load` hydrates `ctx.reap_scores_payload`
from the sidecar; its `on_score` populates `ctx.scores` + `ctx.freq`
from the cached row. `ReapScoringPlugin.on_score` now starts with a
`ctx.has("scores")` early-return guard so the live REAP forward is
skipped on a cache hit.

### `calib-v2-imatrix-resumable` (previous)
Adds spot-preemption resumability for the imatrix capture path.
- `vllm/calibration_imatrix.py`: new public API
  - `dump_imatrix_checkpoint(path)` — atomic `.imatrix.ckpt` via tmp+rename.
  - `load_imatrix_checkpoint(path) -> int` — hydrates accumulators in-place,
    preserving the CUDA-graph-pinned buffers; returns the loaded
    cumulative prompt count.
  - `get_n_prompts_accumulated() -> int` / `set_n_prompts_accumulated(n)` —
    driver-owned cumulative counter; source of truth for `m_last_chunk`.
- `vllm/calibration_imatrix.py`: `_write_dat` now uses tmp+rename, so the
  final `.imatrix.dat` is also atomic.
- `tests/test_calibration_imatrix_smoke.py`: +5 tests covering checkpoint
  round-trip, two-segment additivity, atomic-write crash safety, cumulative
  `chunk_count`, and the new final-dump atomicity contract.

### `calib-v2-patch-locked` (previous, frozen)
| Patch line count | 3087 |
| Patch MD5 | `9effe235a95940d806f626ee1dc841c8` |

Tag kept immutable so older wheels referencing it remain reproducible.

## Driver-side companion changes (NOT in this patch — they live in the repo)

Both behaviors above require the driver to participate; see commit history
on `feat/calibration-v2`:
- `max_quality/scripts/build_self_traces_calib_vllm.py`
  - New flag `--imatrix-checkpoint-every-chunks` (default 1).
  - Calls `load_imatrix_checkpoint` after `setup(llm)` when `--resume` and the
    checkpoint file exists.
  - Calls `dump_imatrix_checkpoint` inside the chunk loop after each JSONL
    flush boundary.
  - Final `dump_imatrix` uses `get_n_prompts_accumulated()` (cumulative
    across instance lifetimes).
  - `.npz` logit sidecars now written via tmp+rename for atomicity.
  - JSONL resume validates each line as JSON and truncates at the first
    parse failure (drops trailing partial lines from preempted writes).

## Schema bumps

Sidecar schema versions for the cached-calibration-signals provider-pair
infrastructure. Bump these when changing the dataclass layout in
`max_quality/src/moe_compress/utils/cached_calibration_signals.py`.

| Signal | schema_version | Notes |
|---|---|---|
| `phase_b` | 1 | initial |
| `stage2_profile` | 1 | initial |
| `reap_scores` | 1 | initial — V1+V2 writers campaign, REAP Eq. 9 (arXiv:2510.13999); `[n_layers, n_experts] float32` + matching `int64` counts |
| `covariance` | 2 | Item 1 writers campaign — dict-valued payload, keys `(layer_idx, expert_idx, matrix_name)` -> fp16 `Tensor[d_in, d_in]`, byte-shape-compatible with the Stage 2 writer's `_stage2_input_covariance.pt`. v1 was never persisted by a production writer; forward-only bump. |
| `router_kd_logits` | 1 | initial — matches the existing .npz writer format in `build_self_traces_calib_vllm.py` |
| `block_hidden` | 1 | initial |
| `teacher_eval` | 1 | initial |

## When to bump the immutable tag

Whenever the patch's functional contents change (line count or MD5 differ),
create a new `calib-v2-<short-mnemonic>` tag pointing at the new patch
commit, update this file's "Current" section, update the HF Jobs build
script's expected MD5 comment, and re-run the HF Jobs build to publish a
fresh wheel. Older wheels keep working with their original tag.
