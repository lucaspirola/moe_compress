"""HF Jobs entrypoint for the Phase A MA-formation detector diagnostic.

Single-file PEP 723 script. Mirrors the boot sequence of
``entrypoint_ablations.py`` (download code from pirola/moe-compress dataset,
prime the model snapshot, set up HF cache) but invokes the standalone
``scripts/phase_a_diagnostic.py`` instead of the full ablation harness.

Wall time on H200 with 256 samples: ~5 min model load + ~3 min Phase A = ~8 min.
Significantly cheaper than re-running full Stage 1 (which is 5 h).

Run via:

    hf jobs uv run hf_jobs/entrypoint_phase_a_diagnostic.py \\
        --flavor h200 \\
        --volume pirola/moe-ablations:/mnt/cache \\
        --secrets HF_TOKEN \\
        --timeout 1h

The diagnostic JSON is uploaded to the bucket at
``/mnt/cache/diagnostics/phase_a_<timestamp>.json``.
"""
# /// script
# requires-python = ">=3.10"
# dependencies = [
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
#     "pyyaml>=6.0",
#     "nvidia-ml-py>=12.0",
#     "psutil>=5.9",
#     # Fast-path linear attention + Mamba conv1d kernels for Qwen3.5/3.6.
#     "flash-linear-attention>=0.1.2",
#     "causal-conv1d>=1.4.0",
# ]
# ///
from __future__ import annotations

import logging
import os
import sys
import time
import traceback
from pathlib import Path

LOG = logging.getLogger("hf_jobs.entrypoint_phase_a_diagnostic")

CODE_REPO = os.environ.get("CODE_REPO", "pirola/moe-compress")
MODEL_REPO = os.environ.get("MODEL_REPO", "Qwen/Qwen3.6-35B-A3B")
NUM_SAMPLES = int(os.environ.get("NUM_SAMPLES", "256"))
CACHE_MOUNT = Path(os.environ.get("CACHE_MOUNT", "/mnt/cache"))


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    LOG.info("========== Phase A diagnostic on HF Jobs ==========")
    LOG.info("CODE_REPO=%s MODEL_REPO=%s NUM_SAMPLES=%d", CODE_REPO, MODEL_REPO, NUM_SAMPLES)
    LOG.info("CACHE_MOUNT=%s", CACHE_MOUNT)

    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN not set. Pass --secrets HF_TOKEN to `hf jobs run`.")
    if not CACHE_MOUNT.exists():
        raise RuntimeError(
            f"Expected bucket mount at {CACHE_MOUNT}. Pass "
            f"--volume pirola/moe-ablations:{CACHE_MOUNT} to `hf jobs run`."
        )

    import torch
    LOG.info(
        "torch=%s cuda=%s avail=%s device_count=%d%s",
        torch.__version__, getattr(torch.version, "cuda", "?"),
        torch.cuda.is_available(), torch.cuda.device_count(),
        f" [{torch.cuda.get_device_name(0)}]" if torch.cuda.is_available() else "",
    )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available — refusing to run on CPU.")

    code_dir = CACHE_MOUNT / "code"
    hf_home = CACHE_MOUNT / "hf_cache"
    diag_dir = CACHE_MOUNT / "diagnostics"
    cache_dir = CACHE_MOUNT / "phase_a_cache"
    for p in (code_dir, hf_home, diag_dir, cache_dir):
        p.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["TRANSFORMERS_CACHE"] = str(hf_home / "hub")

    # 1. Download code repo (HF dataset).
    from huggingface_hub import snapshot_download
    LOG.info("Downloading code repo %s into %s", CODE_REPO, code_dir)
    snapshot_download(
        CODE_REPO, repo_type="dataset", local_dir=str(code_dir),
        allow_patterns=["max_quality/**"],
    )
    sys.path.insert(0, str(code_dir / "max_quality" / "src"))

    # 2. Prime model snapshot (idempotent — bucket cache short-circuits).
    LOG.info("Ensuring model snapshot resident at %s", hf_home / "hub")
    snapshot_download(MODEL_REPO, cache_dir=hf_home / "hub", allow_patterns=["*"])

    # 3. Invoke the diagnostic.
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = diag_dir / f"phase_a_{timestamp}.json"
    diag_script = code_dir / "max_quality" / "scripts" / "phase_a_diagnostic.py"

    LOG.info("Running diagnostic → %s", output_path)

    # Import as a module so we can call main() with argv directly (avoids spawning
    # a subprocess and re-loading torch/transformers).
    import importlib.util
    spec = importlib.util.spec_from_file_location("phase_a_diagnostic", diag_script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Use the production run's exact ablation_config.yaml if present, so the
    # calibration distribution matches what Stage 1 saw (nvidia-cascade with
    # the specific subset_weights). Falls back to the diagnostic default
    # (c4-math-code) only if the production config isn't in the bucket.
    prod_config = CACHE_MOUNT / "ablations" / "_shared" / "ablation_config.yaml"
    argv = [
        "--model", MODEL_REPO,
        "--num-samples", str(NUM_SAMPLES),
        "--output", str(output_path),
        "--cache-dir", str(cache_dir),
    ]
    if prod_config.exists():
        LOG.info("Using production calibration config: %s", prod_config)
        argv += ["--config", str(prod_config)]
    else:
        LOG.warning(
            "Production config %s not found — falling back to diagnostic default "
            "(c4-math-code); diagnostic may NOT match Stage 1's residual signal exactly.",
            prod_config,
        )
    LOG.info("argv = %s", argv)
    try:
        rc = mod.main(argv)
    except SystemExit as exc:
        rc = int(exc.code) if isinstance(exc.code, int) else 1
    except BaseException as exc:  # noqa: BLE001
        LOG.error("Diagnostic raised: %s\n%s", exc, traceback.format_exc())
        return 2

    if rc == 0:
        LOG.info("=" * 60)
        LOG.info(" >>> SUCCESS — diagnostic JSON at %s", output_path)
        LOG.info(" >>> Download with:")
        LOG.info("     hf buckets cp hf://buckets/pirola/moe-ablations/diagnostics/%s .",
                 output_path.name)
        LOG.info("=" * 60)
    return rc


if __name__ == "__main__":
    sys.exit(main())
