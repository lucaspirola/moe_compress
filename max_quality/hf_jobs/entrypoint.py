"""HF Jobs entrypoint for the Strategy A compression pipeline.

This is a UV script (PEP 723). Run it on Hugging Face Jobs via

    hf jobs uv run hf_jobs/entrypoint.py \\
        --flavor a100-large \\
        --volume pirola/moe-cache:/mnt/cache \\
        --secrets HF_TOKEN \\
        --timeout 5h

The script expects:
- A single bucket mounted at ``/mnt/cache`` — used for both the HF model
  snapshot cache (persisted across runs) and pipeline artifacts.
- ``HF_TOKEN`` secret (set in the job env) with read+write scope on the
  ``pirola`` namespace so we can download the code repo and upload the
  final compressed model.
- Environment overrides: ``CODE_REPO``, ``MODEL_REPO``, ``RESULT_REPO``,
  ``TARGET_RATIO``, ``FLAVOR_HINT`` — all optional.

On success the script uploads ``stage5_final/`` plus per-stage JSON artifacts
to ``RESULT_REPO`` (created if missing, private by default). On failure it
still uploads whatever artifacts exist so partial progress is not lost.

Either way the script returns and the HF Jobs runtime releases the GPU.
"""

# /// script
# requires-python = ">=3.10"
# dependencies = [
#     # HF Jobs a100-large (and most GPU flavors as of 2026-04) run NVIDIA
#     # drivers at CUDA 12.9. torch 2.11+ is linked against CUDA 13 and will
#     # fall back to CPU silently on those hosts — we cap below 2.11 so UV
#     # resolves a cu124/cu126 wheel. Revisit after HF upgrades drivers.
#     "torch>=2.5.0,<2.11.0",
#     "transformers>=4.57.0",
#     "accelerate>=1.0.0",
#     "datasets>=3.0.0",
#     "safetensors>=0.4.5",
#     "tokenizers>=0.20.0",
#     "sentencepiece>=0.2.0",
#     "huggingface_hub>=0.26.0",
#     "einops>=0.8.0",
#     "numpy>=1.26.0",
#     "scipy>=1.11.0",
#     "peft>=0.13.0",
#     "pyyaml>=6.0",
#     "lm-eval>=0.4.5",
#     # Observability — Trackio (HF-native experiment tracking, syncs to
#     # pirola/trackio Space) + pynvml/psutil for system metrics. See
#     # src/moe_compress/utils/system_metrics.py.
#     "trackio>=0.5.0",
#     "nvidia-ml-py>=12.0",
#     "psutil>=5.9",
#     # 4-bit teacher quantization for Stage 5 router KD — avoids holding
#     # ~120 GB (teacher 70 GB BF16 + student 50 GB BF16) on a single 80 GB
#     # A100, which would force CPU offload and 5–10× slowdown.
#     "bitsandbytes>=0.43.0",
# ]
# ///

from __future__ import annotations

import logging
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

LOG = logging.getLogger("hf_jobs.entrypoint")


# ---------------------------------------------------------------------------
# Config from env
# ---------------------------------------------------------------------------

CODE_REPO        = os.environ.get("CODE_REPO",       "pirola/moe-compress-code")
MODEL_REPO       = os.environ.get("MODEL_REPO",      "Qwen/Qwen3.6-35B-A3B")
RESULT_REPO      = os.environ.get("RESULT_REPO",     "")           # auto-generated if empty
TARGET_RATIO     = float(os.environ.get("TARGET_RATIO", "0.30"))
CACHE_MOUNT      = Path(os.environ.get("CACHE_MOUNT", "/mnt/cache"))
CONFIG_PATH      = os.environ.get("CONFIG_PATH",     "configs/qwen36_35b_a3b_30pct.yaml")
RESUME_FROM      = int(os.environ.get("RESUME_FROM_STAGE", "0"))
STOP_AFTER       = int(os.environ.get("STOP_AFTER_STAGE",  "6"))
UPLOAD_ON_STOP   = os.environ.get("UPLOAD_ON_STOP", "1") not in ("0", "false", "False")
# When resuming from stage 3+, the bucket may have a partial/missing prior
# checkpoint. Set PRIOR_STAGE_REPO to the HF model repo (e.g.
# "pirola/qwen3-6-35b-a3b-strategy-a-30pct-stop2-...") to download it into
# artifacts/stage2_pruned/ before the pipeline starts.
PRIOR_STAGE_REPO = os.environ.get("PRIOR_STAGE_REPO", "")


def _main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    LOG.info("========== Strategy A pipeline on HF Jobs ==========")
    LOG.info("CODE_REPO=%s MODEL_REPO=%s TARGET_RATIO=%s",
             CODE_REPO, MODEL_REPO, TARGET_RATIO)
    LOG.info("CACHE_MOUNT=%s", CACHE_MOUNT)

    _sanity_check()

    # Persist HF cache to the mounted bucket so downloads don't repeat each run.
    code_dir = CACHE_MOUNT / "code"
    hf_home  = CACHE_MOUNT / "hf_cache"
    artifacts_dir = CACHE_MOUNT / "artifacts"
    for p in (code_dir, hf_home, artifacts_dir):
        p.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["TRANSFORMERS_CACHE"] = str(hf_home / "hub")
    LOG.info("HF_HOME=%s", hf_home)

    # 1. Download our code repo into the mounted bucket.
    # The pirola/moe-compress dataset stores the project under max_quality/.
    # We support both layouts (top-level src/ and max_quality/src/) so older
    # code repos with src/ at root continue to work.
    _download_code(CODE_REPO, code_dir)
    if (code_dir / "max_quality" / "src").is_dir():
        code_root = code_dir / "max_quality"
    else:
        code_root = code_dir
    sys.path.insert(0, str(code_root / "src"))

    # 2. Prime the model snapshot (idempotent — ``hf_hub_download``/``snapshot``
    #    short-circuits on cache hit, and the cache lives in the bucket).
    from huggingface_hub import snapshot_download
    LOG.info("Ensuring model snapshot is resident at %s", hf_home / "hub")
    snapshot_download(MODEL_REPO, cache_dir=hf_home / "hub", allow_patterns=["*"])

    # 2b. If resuming from stage 3+ and the bucket checkpoint is stale/partial,
    #     download the full prior-stage checkpoint from PRIOR_STAGE_REPO into the
    #     correct artifacts/<prior_stage>/ subdir so run_pipeline can load it.
    if PRIOR_STAGE_REPO and RESUME_FROM >= 3:
        _restore_prior_checkpoint(PRIOR_STAGE_REPO, artifacts_dir, RESUME_FROM)

    # 3. Run the pipeline.
    result_repo = RESULT_REPO or _default_result_repo()
    LOG.info("Result repo will be: %s", result_repo)

    # Per-stage Hub upload base — set so run_pipeline.py uploads each stage's
    # checkpoint to its own per-stage repo immediately after the stage finishes.
    # Bucket FUSE writes are NOT durable on cancel/SIGKILL; the Hub commit is
    # the only durability boundary. See docs/huggingface_jobs_and_buckets.md.
    per_stage_base = _per_stage_base_repo(result_repo)
    os.environ["PIPELINE_HUB_RESULT_REPO_BASE"] = per_stage_base
    LOG.info("Per-stage Hub upload base: %s (each stage → <base>-stage{N})", per_stage_base)

    exit_code = 0
    pipeline_error: BaseException | None = None
    metrics = None
    trackio_run = None
    try:
        # Trackio: HF-native experiment tracking. Auto-creates the Space on
        # first run. Inside try/finally so a slow Space-create or network
        # hang still hits the cleanup path on cancel/timeout. Returns the
        # Run object (not the module) so .log() works from background threads.
        trackio_run = _init_trackio(result_repo)

        # Background system-metrics sampler (GPU/CPU/VRAM/RAM every 30s).
        # Logs to stdout AND the Trackio Run object so the curve is durable
        # even if Trackio is off. Started inside the try so any failure
        # flows through the finally (and so a startup error never skips
        # the Trackio finish).
        try:
            from moe_compress.utils.system_metrics import SystemMetrics
            metrics = SystemMetrics(interval_sec=30.0, trackio_run=trackio_run)
            metrics.start()
        except Exception as exc:                 # noqa: BLE001
            LOG.warning("SystemMetrics startup failed (%s) — continuing without sampler",
                        exc)
            metrics = None

        # Build llama.cpp in the background while stages 1-5 run — it will be
        # ready long before Stage 6 needs it, without blocking the pipeline start.
        import threading
        threading.Thread(
            target=_install_llama_cpp,
            args=(CACHE_MOUNT / "llama.cpp",),
            daemon=True,
            name="llama-cpp-build",
        ).start()

        # Import after sys.path manipulation.
        from moe_compress.run_pipeline import main as run_pipeline_main
        argv = [
            "--config", str(code_root / CONFIG_PATH),
            "--model", MODEL_REPO,
            "--artifacts-dir", str(artifacts_dir),
            "--target-ratio", str(TARGET_RATIO),
            "--resume-from-stage", str(RESUME_FROM),
            "--stop-after-stage",  str(STOP_AFTER),
        ]
        LOG.info("Invoking run_pipeline.main(%s)", argv)
        exit_code = run_pipeline_main(argv)
    except SystemExit as exc:
        exit_code = int(exc.code) if isinstance(exc.code, int) else 1
    except BaseException as exc:                 # noqa: BLE001
        pipeline_error = exc
        LOG.error("Pipeline raised: %s\n%s", exc, traceback.format_exc())
        exit_code = 2
    finally:
        # Drain metrics and finalize Trackio BEFORE the artifact upload so the
        # dashboard captures the full timeline regardless of pipeline outcome.
        if metrics is not None:
            try:
                metrics.stop()
            except Exception as exc:             # noqa: BLE001
                LOG.warning("metrics.stop failed: %s", exc)
        if trackio_run is not None:
            try:
                trackio_run.finish()
            except Exception as exc:             # noqa: BLE001
                LOG.warning("trackio_run.finish failed: %s", exc)

    # 4. Upload artifacts regardless of success/failure — partial progress
    #    is worth keeping (per-stage artifacts can restart the next run).
    _upload_results(artifacts_dir, result_repo, ok=(pipeline_error is None))

    if pipeline_error is not None:
        LOG.error("Exiting with error.")
    else:
        LOG.info("Pipeline finished cleanly; GPU will release on exit.")
    return exit_code


# ---------------------------------------------------------------------------


def _sanity_check() -> None:
    """Fail fast if the job environment is misconfigured."""
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError(
            "HF_TOKEN is not set. Pass --secrets HF_TOKEN to `hf jobs run`."
        )
    if not CACHE_MOUNT.exists():
        raise RuntimeError(
            f"Expected bucket mount at {CACHE_MOUNT}. Pass --volume "
            "pirola/moe-cache:/mnt/cache to `hf jobs run`."
        )
    # Surface GPU availability early — HARD FAIL if CUDA isn't usable.
    # Running the 35 B pipeline on CPU is a multi-hour waste (silent OOM in
    # practice), so we'd rather pay $0.05 for a fast crash than $3 for a
    # mid-run cancellation.
    import torch
    avail = torch.cuda.is_available()
    LOG.info(
        "torch=%s torch.cuda=%s avail=%s device_count=%d%s",
        torch.__version__,
        getattr(torch.version, "cuda", "?"),
        avail,
        torch.cuda.device_count(),
        f" [{torch.cuda.get_device_name(0)}]" if avail else "",
    )
    if not avail:
        raise RuntimeError(
            "torch.cuda.is_available() is False on this job — refusing to "
            "run the compression pipeline on CPU. Most likely the PEP 723 "
            "torch pin resolved to a CUDA-toolkit version newer than the "
            "host driver. Check the 'CUDA initialization' warning above. "
            "Tighten the torch pin in hf_jobs/entrypoint.py and re-submit."
        )


# Maps RESUME_FROM_STAGE → name of the prior stage's checkpoint subdir.
# Mirrors run_pipeline.STAGE_REGISTRY[stage][1] but local to entrypoint so we
# don't have to import the pipeline module before sys.path is set up.
_PRIOR_STAGE_DIRNAME = {
    3: "stage2_pruned",
    4: "stage3_svd",
    5: "stage4_eora",
    6: "stage5_final",
}


def _restore_prior_checkpoint(repo_id: str, artifacts_dir: Path, resume_from: int) -> None:
    """Download a prior-stage HF model repo into ``artifacts/<prior_stage>/``.

    Used when ``RESUME_FROM_STAGE >= 3`` and the bucket artifact is stale or
    incomplete (e.g. the prior stage uploaded the full model to Hub but only
    a partial copy made it into the bucket).

    Sidecar files saved by earlier stages at ``artifacts_dir/_stage*_*.pt`` are
    uploaded by ``_upload_results`` under ``artifacts/<file>`` in the Hub repo.
    On download they land under ``<dest>/artifacts/<file>``; we move them up
    to ``artifacts_dir/<file>`` so Stage 3/4 find them at the expected path.
    """
    from huggingface_hub import snapshot_download
    dirname = _PRIOR_STAGE_DIRNAME.get(resume_from)
    if dirname is None:
        LOG.warning("No prior-stage dirname for RESUME_FROM_STAGE=%d — skipping restore",
                    resume_from)
        return
    dest = artifacts_dir / dirname
    dest.mkdir(parents=True, exist_ok=True)
    # `save_pretrained(safe_serialization=True)` writes a sharded model with
    # `model.safetensors.index.json` only when the state_dict exceeds ~5 GB;
    # smaller compressed checkpoints (e.g. Stage 3+ after rank reduction) emit
    # a single `model.safetensors`. Accept either as proof of a complete dir.
    index   = dest / "model.safetensors.index.json"
    single  = dest / "model.safetensors"
    meta    = dest / "compressed_metadata.json"
    if (index.exists() or single.exists()) and meta.exists():
        LOG.info("%s already complete at %s — skipping download", dirname, dest)
    else:
        LOG.info("Downloading prior-stage checkpoint from %s → %s", repo_id, dest)
        snapshot_download(
            repo_id,
            repo_type="model",
            local_dir=str(dest),
            ignore_patterns=["*.metadata", "job_status.txt"],
        )
        LOG.info("Prior-stage checkpoint ready at %s", dest)

    # Hoist any sidecar files (covariance, originals) from <dest>/artifacts/*
    # up one level to <artifacts_dir>/* so Stage 3 / Stage 4 find them at the
    # paths their loaders expect.
    sidecar_src = dest / "artifacts"
    if sidecar_src.is_dir():
        for p in sidecar_src.iterdir():
            if not p.is_file():
                continue
            target = artifacts_dir / p.name
            if target.exists():
                continue
            shutil.move(str(p), str(target))
            LOG.info("Hoisted sidecar %s → %s", p.name, target)


def _install_llama_cpp(dest: Path) -> None:
    """Shallow-clone and CUDA-build llama.cpp, then export LLAMA_CPP_DIR.

    Idempotent: skips the build if the llama-imatrix binary already exists
    (e.g. the bucket cache still has the build from a prior job run).
    """
    imatrix_bin = dest / "build" / "bin" / "llama-imatrix"
    if imatrix_bin.exists():
        LOG.info("llama.cpp already built at %s — skipping.", dest)
        os.environ["LLAMA_CPP_DIR"] = str(dest)
        return

    LOG.info("Building llama.cpp with CUDA at %s", dest)
    dest.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/ggerganov/llama.cpp", str(dest)],
            check=True, timeout=300,
        )
        build_dir = dest / "build"
        subprocess.run(
            ["cmake", "-B", str(build_dir), str(dest),
             "-DGGML_CUDA=ON", "-DCMAKE_BUILD_TYPE=Release"],
            check=True, timeout=120,
        )
        subprocess.run(
            ["cmake", "--build", str(build_dir), "--config", "Release",
             f"-j{os.cpu_count() or 4}",
             "--target", "llama-imatrix", "llama-quantize"],
            check=True, timeout=1800,
        )
        os.environ["LLAMA_CPP_DIR"] = str(dest)
        LOG.info("llama.cpp build complete → %s", imatrix_bin)
    except Exception as exc:  # noqa: BLE001
        LOG.warning("llama.cpp build failed (%s) — Stage 6 imatrix will be skipped.", exc)


def _download_code(repo_id: str, dest: Path) -> None:
    """Clone-equivalent: fresh snapshot every job start so we don't run stale code."""
    from huggingface_hub import snapshot_download
    LOG.info("snapshot_download %s → %s", repo_id, dest)
    # Reset the dir to make sure stale files are cleared.
    if dest.exists():
        for p in dest.iterdir():
            if p.is_dir() and p.name not in ("__pycache__",):
                shutil.rmtree(p, ignore_errors=True)
            else:
                p.unlink(missing_ok=True)
    snapshot_download(
        repo_id,
        repo_type="dataset",
        local_dir=dest,
        allow_patterns=[
            # Top-level layout (legacy pirola/moe-compress-code)
            "*.py", "*.yaml", "*.yml", "*.txt", "*.md",
            "configs/*", "src/**/*", "hf_jobs/*",
            # Nested layout (current pirola/moe-compress dataset)
            "max_quality/**/*.py", "max_quality/**/*.yaml",
            "max_quality/**/*.yml", "max_quality/**/*.txt", "max_quality/**/*.md",
            "max_quality/configs/*", "max_quality/src/**/*", "max_quality/hf_jobs/*",
            "max_quality/scripts/*",
        ],
    )


def _init_trackio(result_repo: str):
    """Initialize Trackio for this run. Returns the **Run** object on success,
    or None on any failure (graceful fallback so a missing dep / Space-creation
    error never blocks the pipeline). Uses a single shared Space at
    ``$TRACKIO_SPACE_ID`` (default ``pirola/trackio``); each run lands as a
    new name on the same dashboard.

    NB: we return the Run object (not the trackio module) because Trackio's
    module-level ``trackio.log`` reads a thread-local current-run pointer,
    which means the SystemMetrics daemon thread (a different thread) can't
    log via ``trackio.log``. ``run.log`` is object-bound and works from any
    thread.
    """
    space_id = os.environ.get("TRACKIO_SPACE_ID", "pirola/trackio")
    run_name = result_repo.split("/", 1)[-1]
    try:
        import trackio
    except ImportError as exc:
        LOG.warning("trackio not installed (%s) — observability metrics disabled", exc)
        return None
    try:
        run = trackio.init(
            project="moe-compress-strategy-a",
            name=run_name,
            space_id=space_id,
            # Trackio's built-in GPU sampler — extra belt next to our own
            # SystemMetrics CPU/RAM thread, in case our thread breaks.
            auto_log_gpu=True,
            gpu_log_interval=30.0,
            config={
                "model": MODEL_REPO,
                "target_ratio": TARGET_RATIO,
                "resume_from": RESUME_FROM,
                "stop_after": STOP_AFTER,
                "prior_stage_repo": PRIOR_STAGE_REPO or None,
                "result_repo": result_repo,
            },
        )
        LOG.info("Trackio initialized: project=moe-compress-strategy-a, name=%s, space=%s",
                 run_name, space_id)
        return run
    except Exception as exc:                         # noqa: BLE001
        LOG.warning("trackio.init failed (%s) — continuing without remote tracking", exc)
        return None


def _default_result_repo() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    # Strip a "Qwen/" prefix for brevity in the result repo name.
    stem = MODEL_REPO.split("/", 1)[-1].lower().replace(".", "-")
    pct = int(round(TARGET_RATIO * 100))
    # Per-stage runs get their own repo so supervision artifacts don't collide.
    stage_tag = f"-stop{STOP_AFTER}" if STOP_AFTER < 6 else ""
    return f"pirola/{stem}-strategy-a-{pct}pct{stage_tag}-{ts}"


def _per_stage_base_repo(result_repo: str) -> str:
    """Strip the ``-stopN`` segment from a result repo name so per-stage uploads
    land in a stable family of repos regardless of where the operator chose to
    stop. ``run_pipeline.py`` appends ``-stage{N}`` per stage."""
    import re
    return re.sub(r"-stop\d+", "", result_repo)


def _upload_results(artifacts_dir: Path, repo_id: str, *, ok: bool) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    # The final model lives at artifacts_dir/stage5_final — upload that as a
    # model repo when the pipeline succeeded; the rest (budgets, merge_map,
    # scores, eval JSON) goes in as auxiliary files.
    try:
        api.create_repo(
            repo_id, repo_type="model", private=True, exist_ok=True,
        )
    except Exception as err:                     # noqa: BLE001
        LOG.warning("create_repo(%s): %s — continuing with existing.", repo_id, err)

    LOG.info("Uploading artifacts to %s (ok=%s)", repo_id, ok)

    # Prefer the final stage as the main model weights; fall back to the
    # latest available stage when earlier stages crashed.
    final_dir = artifacts_dir / "stage5_final"
    if not final_dir.exists():
        for candidate in ("stage4_eora", "stage3_svd", "stage2_pruned"):
            alt = artifacts_dir / candidate
            if alt.exists():
                LOG.warning("stage5_final missing — uploading %s as the main model dir.", alt)
                final_dir = alt
                break

    # Upload the main model dir (if any), flattened to repo root.
    if final_dir.exists():
        api.upload_large_folder(
            folder_path=str(final_dir),
            repo_id=repo_id,
            repo_type="model",
        )

    # Auxiliary artifacts (small JSONs and multi-GB sidecars like
    # ``_stage3_original_weights.pt``) all go under ``artifacts/`` in the repo.
    # We stage them into a single folder mirroring the Hub layout, then use
    # ``upload_large_folder`` for resumable, chunked, retried uploads — the
    # large sidecars (5–20 GB) make per-file ``upload_file`` calls fragile.
    aux_files = [
        "stage1_blacklist.json",
        "stage1_budgets.json",
        "stage2_layer_mse.json",
        "budget_decomposition.json",
        "stage6_eval.json",
        "_stage2_input_covariance.pt",   # needed for Stage 3 AA-SVD on resume
        "_stage3_original_weights.pt",   # needed for Stage 4 EoRA residuals on resume
        "calibration_imatrix.txt",       # multi-domain imatrix calibration text (Stage 6)
        "imatrix.gguf",                  # importance matrix for GGUF quantization (Stage 6)
    ]
    aux_stage = artifacts_dir / "_aux_stage"
    if aux_stage.exists():
        shutil.rmtree(aux_stage, ignore_errors=True)
    (aux_stage / "artifacts").mkdir(parents=True, exist_ok=True)

    def _stage(src: Path, name: str) -> None:
        target = aux_stage / "artifacts" / name
        if target.exists() or target.is_symlink():
            return
        # Try symlink first (zero copy, works on FUSE mounts), then hardlink
        # (zero copy, same-fs only), then full copy as last resort. The 5–20 GB
        # sidecars (covariance, originals) make a real copy expensive on FUSE.
        src_abs = src.resolve()
        try:
            os.symlink(src_abs, target)
            return
        except OSError:
            pass
        try:
            os.link(src_abs, target)
            return
        except OSError:
            pass
        shutil.copy2(src_abs, target)

    staged_count = 0
    for name in aux_files:
        p = artifacts_dir / name
        if not p.exists():
            continue
        _stage(p, name)
        staged_count += 1
    # merge_map sits inside stage2_pruned. When stage2_pruned IS the final_dir
    # being uploaded to repo root, it already lands there — skip the duplicate
    # staging under artifacts/. Otherwise (success path, final_dir = stage5_final
    # etc.) we do need it under artifacts/.
    mm = artifacts_dir / "stage2_pruned" / "merge_map.json"
    if mm.exists() and final_dir.name != "stage2_pruned":
        _stage(mm, "merge_map.json")
        staged_count += 1

    if staged_count:
        LOG.info("Uploading %d aux artifact(s) via upload_large_folder", staged_count)
        api.upload_large_folder(
            folder_path=str(aux_stage),
            repo_id=repo_id,
            repo_type="model",
        )

    # P2-2: clean up the staging dir so the bucket doesn't retain stale links.
    shutil.rmtree(aux_stage, ignore_errors=True)

    # A small status file makes it trivial to grep across runs.
    status_path = artifacts_dir / "_job_status.txt"
    status_path.write_text(
        f"{'SUCCESS' if ok else 'FAILURE'} at "
        f"{datetime.now(timezone.utc).isoformat()}\n"
        f"CODE_REPO={CODE_REPO}\nMODEL_REPO={MODEL_REPO}\n"
        f"TARGET_RATIO={TARGET_RATIO}\n"
    )
    api.upload_file(
        path_or_fileobj=str(status_path),
        path_in_repo="job_status.txt",
        repo_id=repo_id,
        repo_type="model",
    )
    LOG.info("Upload complete → https://huggingface.co/%s", repo_id)


# ---------------------------------------------------------------------------


if __name__ == "__main__":
    # Make CTRL-C / SIGTERM reach the Python layer so cleanup runs.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    sys.exit(_main())
