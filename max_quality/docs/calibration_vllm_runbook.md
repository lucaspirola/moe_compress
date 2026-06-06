# Calibration self-traces (vLLM path) — operations runbook

How to run `scripts/build_self_traces_calib_vllm.py` on a fresh cloud GPU box
without re-discovering the failure modes. The hardened launcher
`scripts/run_calib_vllm.sh` bakes in every fix below; this doc explains the
*why* so the fixes survive refactors and reviews.

## TL;DR

```bash
# On a GPU box with the repo, a venv, and the teacher cached on durable storage:
DATA_ROOT=/mnt/data bash max_quality/scripts/run_calib_vllm.sh
```

The launcher installs the host build dep, removes the conflicting `kernels`
package, caps the JIT compiler fan-out, points compile caches at durable
storage, and runs the driver with the full capture suite + `--resume`.

## The model is single-GPU by design

The driver pins `tensor_parallel_size=1` (see `_load_teacher_vllm`) for
deterministic teacher logits — there is **no** tensor/data-parallel path. A
multi-GPU instance wastes every GPU but one. Size the box to fit the teacher on
**one** GPU:

- Qwen3.6-35B-A3B BF16 ≈ 67 GB weights → a single 1×H200 (141 GB) or
  1×B200 (180 GB) fits weights + KV cache comfortably.
- Per spot/on-demand availability, prefer the cheapest single-GPU SKU that fits.

## Failure modes & fixes

| # | Symptom | Root cause | Fix |
|---|---------|-----------|-----|
| 1 | `from vllm import LLM` → `ValueError: Either a revision or a version must be specified` (in `kernels`/`transformers` hub-kernels import) | The `kernels` package (only needed for an **FP8** teacher) is incompatible with the installed `transformers` and runs at import time. | `pip uninstall -y kernels` for non-FP8 teachers. Launcher does this unless `CALIB_KEEP_KERNELS=1` or `DTYPE=fp8`. |
| 2 | Model-arch inspect fails; buried `cuda_utils.c: fatal error: Python.h: No such file or directory` | Image ships a python venv but not the dev headers; vLLM JIT-compiles a small CUDA util needing `Python.h`. | `apt-get install python3-dev build-essential`. Launcher does this; driver warns if `Python.h` is missing. |
| 3 | Run process `Killed` (SIGKILL) during/after `torch.compile`, host-RAM OOM (`dmesg` shows `oom-kill ... task=cicc`) | First forward pass JIT-compiles the FlashInfer GDN prefill kernel, forking ~one `cicc` per vCPU, each ~6 GB RSS → exhausts host RAM. | Cap with `MAX_JOBS` (16 is a good default; ~96 GB worst case) + `NVCC_THREADS=1`. Driver `setdefault`s these; launcher exports them. VRAM is *not* the constraint here — it's host RAM. |

## Compile caches — persist them

Two separate JIT caches; persist both onto durable storage so a restart after a
spot preemption skips recompilation (saves ~6-20 min/restart):

- **torch.compile / vLLM**: `VLLM_CACHE_ROOT` (driver `setdefault`s it next to
  `--output`; launcher sets `$DATA_ROOT/vllm_cache`).
- **FlashInfer GDN kernel**: FlashInfer ignores `FLASHINFER_WORKSPACE_DIR` and
  hardcodes `~/.cache/flashinfer`. The launcher **symlinks** that onto durable
  storage. Without this the GDN compile (the slow part) repeats every restart.

## Spot-preemption recovery

The run is resumable and preemption-tolerant:

- Output (`self_traces_<cachekey>.jsonl.tmp`), all capture sidecars (`.ckpt`),
  and per-prompt logits (`.npz`) are checkpointed **per chunk** to the volume.
- Keep prep on a **persistent volume** that detaches/re-attaches across boxes
  (the teacher cache, venv, repo, and caches all live there).
- On a fresh box: re-attach the volume, re-run the launcher. `--resume`
  continues from the last completed chunk; **no generation is lost**, only
  ~6-8 min of restart overhead.
- Volumes are location-locked on most clouds — provision the GPU in the same
  region as the volume.

## Monitoring

- The live output file is **cache-key-suffixed**: `self_traces_<hash>.jsonl.tmp`
  (renamed to the final `.jsonl` at the end) — **not** the bare `--output`
  name. Count progress with a glob: `cat <dir>/self_traces_*.jsonl* | wc -l`.
- Total rows target = `--num-prompts` minus completeness-filtered rows, written
  in chunks of `--chunk-size` (default 200).
- Logits `.npz` count < trace count is expected: teacher-forced rows reuse the
  dataset's canonical answer and have no generated logits.
- `nvidia-smi` at ~100% util + the `Processed prompts` tqdm bar advancing =
  steady-state generation. (tqdm redraws with `\r`; read true progress via
  `tr '\r' '\n' < run.log | grep 'Processed prompts' | tail -1`.)
- sshd on a loaded box can be slow to accept connections during the compile —
  retry with `-o ConnectTimeout=90`; don't assume the box died.

## Off-box backup

Per-chunk checkpoints cover **spot preemption** (volume survives), but the
volume itself is a single point of failure during a multi-hour run. For
volume-loss safety, incrementally `hf upload` the artifacts dir to a dataset
repo on a timer (Xet dedup → only new files transfer, so the final upload at
run end is near-instant). Run the upload **synchronously** from whatever drives
it — a backgrounded uploader started from a transient SSH session gets reaped.

## `--num-prompts` vs downstream `num_sequences`

The driver over-generates: `--num-prompts 8000` raw prompts yield ~5600-6400
after the completeness filter, covering the `num_sequences: 4000` the
downstream config expects (see `tasks/CALIBRATION_MIX_V2_PLAN.md`). The driver
defaults to the v1 value (6500) — always pass `--num-prompts` explicitly.

## Optional: skip the GDN JIT entirely

vLLM exposes a Triton GDN prefill backend that avoids the heavy nvcc compile
(`--gdn-prefill-backend triton` on the vLLM CLI). The driver uses the offline
`LLM()` API and does not currently pass this through; if the capped compile is
still too slow for your cadence, wiring that kwarg/env through `_load_teacher_vllm`
is the cleanest next improvement (verify the exact vLLM API for the pinned
version first).
