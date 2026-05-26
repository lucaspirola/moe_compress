# `vllm_calibration_hooks.patch` — manifest

Single source of truth for the patch's identity. Other places that reference
the patch (the HF Jobs build script, the README uploaded to
`pirola/vllm-patched-calib`) should match this file.

## Current

| Field | Value |
|---|---|
| Immutable tag | `calib-v2-reap-scores-writer` |
| Branch (active) | `feat/calibration-v2` |
| vLLM upstream SHA | `ad7125a43e176d4161099480a66f0169609a690` (v0.21.0) |
| Patch line count | **4408** |
| Patch MD5 | **`654c2ea84aa47b8b63d1c27f12849323`** |
| HF model repo | `pirola/vllm-patched-calib` |
| Wheel filename pattern | `vllm-0.21.1.dev0+gad7125a43.d<YYYYMMDD>-cp312-cp312-linux_x86_64.whl` |
| Torch / CUDA pinned in build | `torch==2.11.0+cu130` |
| `TORCH_CUDA_ARCH_LIST` | `8.0;9.0a;10.0;12.0` (A100 / H100·H200 / B200 / RTX 6000 Pro) |

## Verifying locally

```bash
md5sum max_quality/patches/vllm_calibration_hooks.patch
# expect: 654c2ea84aa47b8b63d1c27f12849323
wc -l max_quality/patches/vllm_calibration_hooks.patch
# expect: 4408

# Re-apply against a fresh v0.21.0 checkout (idempotency check):
git clone --depth 1 --branch v0.21.0 https://github.com/vllm-project/vllm /tmp/vllm-fresh
cd /tmp/vllm-fresh
git apply --check /path/to/vllm_calibration_hooks.patch && echo OK
```

## Change log

### `calib-v2-reap-scores-writer` (current)
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
| `covariance` | 1 | initial |
| `router_kd_logits` | 1 | initial — matches the existing .npz writer format in `build_self_traces_calib_vllm.py` |
| `block_hidden` | 1 | initial |
| `teacher_eval` | 1 | initial |

## When to bump the immutable tag

Whenever the patch's functional contents change (line count or MD5 differ),
create a new `calib-v2-<short-mnemonic>` tag pointing at the new patch
commit, update this file's "Current" section, update the HF Jobs build
script's expected MD5 comment, and re-run the HF Jobs build to publish a
fresh wheel. Older wheels keep working with their original tag.
