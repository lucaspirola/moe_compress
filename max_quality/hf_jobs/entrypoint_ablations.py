"""HF Jobs entrypoint for the Stage 2 v2 ablation harness.

UV script (PEP 723). Mirrors the production entrypoint.py boot sequence
(download code repo, prime model snapshot, set up HF cache) but invokes
``moe_compress.run_ablations.main`` instead of ``run_pipeline.main``.

Run via:

    hf jobs uv run hf_jobs/entrypoint_ablations.py \\
        --flavor a100-large \\
        --volume pirola/moe-ablations:/mnt/cache \\
        --secrets HF_TOKEN \\
        --timeout 48h

Environment:
- ``HF_TOKEN`` (secret) — read+write on the ``pirola`` namespace.
- ``CODE_REPO`` (default ``pirola/moe-compress``, dataset)
- ``MODEL_REPO`` (default ``Qwen/Qwen3.6-35B-A3B``)
- ``NUM_SEQUENCES`` (default 1000) — calibration size for ablations.
- ``ONLY`` (default empty) — comma-separated ablation IDs to run.
- ``CACHE_MOUNT`` (default ``/mnt/cache``) — bucket mount point.
"""
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch>=2.5.0,<2.11.0",
#     "transformers>=4.57.0",
#     "accelerate>=1.0.0",
#     "datasets>=3.0.0",
#     "safetensors>=0.4.5",
#     # Required by transformers.integrations.finegrained_fp8 (FP8 KD teacher).
#     "kernels>=0.14.0",
#     "tokenizers>=0.20.0",
#     "sentencepiece>=0.2.0",
#     "huggingface_hub>=0.26.0",
#     "einops>=0.8.0",
#     "numpy>=1.26.0",
#     "scipy>=1.12.0",
#     "ortools>=9.10",
#     "peft>=0.13.0",
#     "pyyaml>=6.0",
#     "lm-eval>=0.4.5",
#     "trackio>=0.5.0",
#     "nvidia-ml-py>=12.0",
#     "psutil>=5.9",
#     "bitsandbytes>=0.43.0",
#     # Fast-path linear attention + Mamba conv1d kernels for Qwen3.5/3.6.
#     # Without these, transformers falls back to a pure-PyTorch implementation
#     # (~10-30% slower on the affected attention layers).
#     #
#     # flash-linear-attention is pure Python (py3-none-any wheel on PyPI), no
#     # build needed. Pinned to a known-good version.
#     "flash-linear-attention==0.5.0",
#     # causal-conv1d has NO PyPI wheels (sdist-only, all versions). HF Jobs
#     # UV env has no nvcc, so building from source fails ("NameError: name
#     # 'bare_metal_version' is not defined" → nvcc-not-found path in setup.py).
#     # Workaround: pin directly to the prebuilt wheel from Dao-AILab's GitHub
#     # release that matches the (cu128, torch 2.10, cp312, x86_64) combo UV
#     # resolves under HF Jobs.
#     # If UV ever resolves a different python/torch/cuda combo this URL will
#     # 404 — pick the right wheel from
#     # https://github.com/Dao-AILab/causal-conv1d/releases and update.
#     "causal-conv1d @ https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.2.post1/causal_conv1d-1.6.2.post1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl",
# ]
# ///
from __future__ import annotations

import logging
import os
import subprocess
import sys
import traceback
from pathlib import Path

LOG = logging.getLogger("hf_jobs.entrypoint_ablations")

CODE_REPO = os.environ.get("CODE_REPO", "pirola/moe-compress")
MODEL_REPO = os.environ.get("MODEL_REPO", "Qwen/Qwen3.6-35B-A3B")
NUM_SEQUENCES = int(os.environ.get("NUM_SEQUENCES", "1000"))
ONLY = os.environ.get("ONLY", "")
PREFLIGHT_ONLY = os.environ.get("PREFLIGHT_ONLY", "0") == "1"
CACHE_MOUNT = Path(os.environ.get("CACHE_MOUNT", "/mnt/cache"))
CONFIG_PATH = os.environ.get(
    "CONFIG_PATH", "max_quality/configs/qwen36_35b_a3b_30pct.yaml"
)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    LOG.info("========== Stage 2 v2 Ablation Harness on HF Jobs ==========")
    LOG.info("CODE_REPO=%s MODEL_REPO=%s NUM_SEQUENCES=%s ONLY=%s",
             CODE_REPO, MODEL_REPO, NUM_SEQUENCES, ONLY or "(all 12)")
    LOG.info("CACHE_MOUNT=%s", CACHE_MOUNT)

    _sanity_check()

    code_dir = CACHE_MOUNT / "code"
    hf_home = CACHE_MOUNT / "hf_cache"
    ablations_root = CACHE_MOUNT / "ablations"
    for p in (code_dir, hf_home, ablations_root):
        p.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["TRANSFORMERS_CACHE"] = str(hf_home / "hub")
    LOG.info("HF_HOME=%s, ablations_root=%s", hf_home, ablations_root)

    # 1. Download code repo (HF dataset).
    _download_code(CODE_REPO, code_dir)
    sys.path.insert(0, str(code_dir / "max_quality" / "src"))

    # 2. Prime model snapshot (idempotent — bucket cache short-circuits).
    from huggingface_hub import snapshot_download
    LOG.info("Ensuring model snapshot resident at %s", hf_home / "hub")
    snapshot_download(MODEL_REPO, cache_dir=hf_home / "hub", allow_patterns=["*"])

    # 3. Run the ablation harness.
    config_path = code_dir / CONFIG_PATH
    if not config_path.exists():
        raise RuntimeError(f"Config not found at {config_path}")

    # Initialize Trackio with a job-level run name BEFORE the pre-flight
    # Stage 1 fires any _trackio_log() calls. The per-ablation run_ablations
    # driver later finishes this run and starts per-ablation runs. Without
    # this init the pre-flight Stage 1 emits would surface a "Call trackio.init
    # before trackio.log" warning + drop the metrics.
    try:
        import trackio
        trackio.init(
            project="moe-compress-strategy-a",
            name="ablation-preflight",
            space_id=os.environ.get("TRACKIO_SPACE_ID", "pirola/trackio"),
            config={"role": "preflight_stage1"},
        )
        LOG.info("Trackio initialized (project=moe-compress-strategy-a, name=ablation-preflight)")
    except Exception as exc:  # noqa: BLE001
        LOG.warning("trackio.init failed (%s) — continuing without observability", exc)

    from moe_compress.run_ablations import main as run_ablations_main
    argv = [
        "--config", str(config_path),
        "--model", MODEL_REPO,
        "--ablations-root", str(ablations_root),
        "--num-sequences", str(NUM_SEQUENCES),
    ]
    if ONLY:
        argv += ["--only", ONLY]
    if PREFLIGHT_ONLY:
        argv += ["--preflight-only"]
    LOG.info("Invoking run_ablations.main(%s)", argv)
    try:
        return run_ablations_main(argv)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 1
    except BaseException as exc:  # noqa: BLE001
        LOG.error("Harness raised: %s\n%s", exc, traceback.format_exc())
        return 2


def _sanity_check() -> None:
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN not set. Pass --secrets HF_TOKEN to `hf jobs run`.")
    if not CACHE_MOUNT.exists():
        raise RuntimeError(
            f"Expected bucket mount at {CACHE_MOUNT}. Pass "
            f"--volume pirola/moe-ablations:{CACHE_MOUNT} to `hf jobs run`."
        )
    import torch
    avail = torch.cuda.is_available()
    LOG.info(
        "torch=%s torch.cuda=%s avail=%s device_count=%d%s",
        torch.__version__, getattr(torch.version, "cuda", "?"), avail,
        torch.cuda.device_count(),
        f" [{torch.cuda.get_device_name(0)}]" if avail else "",
    )
    if not avail:
        raise RuntimeError(
            "torch.cuda.is_available() is False — refusing to run ablations on CPU."
        )


def _download_code(code_repo: str, dst: Path) -> None:
    """Download the dataset code repo. Dataset (not model) because the
    canonical home of the moe_compress code is the pirola/moe-compress
    HF dataset, mirroring GitHub."""
    from huggingface_hub import snapshot_download
    LOG.info("Downloading code repo %s into %s", code_repo, dst)
    snapshot_download(
        code_repo, repo_type="dataset", local_dir=str(dst),
        allow_patterns=["max_quality/**"],
    )


if __name__ == "__main__":
    sys.exit(main())
