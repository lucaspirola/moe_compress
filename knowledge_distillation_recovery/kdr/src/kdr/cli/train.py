"""`kdr-train` CLI entrypoint.

Usage (single-GPU smoke):

    python -m kdr.cli.train \
        --config configs/zaya1_8b_bf16.yaml \
        --student Zyphra/ZAYA1-reasoning-base \
        --artifacts-dir ./artifacts

Usage (multi-GPU + DeepSpeed):

    accelerate launch --use_deepspeed \
        --deepspeed_config_file ds_configs/zero3_offload_optim.json \
        --mixed_precision bf16 \
        -m kdr.cli.train \
        --config configs/zaya1_8b_da_qad_nvfp4_int4kv.yaml \
        --student Zyphra/ZAYA1-reasoning-base \
        --artifacts-dir ./artifacts \
        --mode da_qad

Resume:

    --resume-from ./artifacts/kdr_bf16_partial_step100/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, get_args

import torch
import yaml
from accelerate import Accelerator

from ..adapters.zaya1_8b import Zaya1Adapter
from ..config import Config
from ..io.resume import find_latest_partial
from ..io.save import COMPRESSED_METADATA_FILENAME
from ..modes import Mode
from ..training.loop import run_recovery

log = logging.getLogger(__name__)


# REQ: LLR-0008
def _parse(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="kdr — Knowledge Distillation Recovery (Ch1 BF16 + Ch3 DA-QAD)."
    )
    p.add_argument("--config", required=True, help="Path to a kdr YAML.")
    p.add_argument(
        "--student",
        required=False,
        default=None,
        help="Override student.source (path or HF repo id).",
    )
    p.add_argument(
        "--mode",
        choices=list(get_args(Mode)),
        default=None,
        help="Override the YAML's `mode`. Defaults to the YAML value.",
    )
    p.add_argument(
        "--artifacts-dir",
        required=True,
        help="Directory for partial / final checkpoints.",
    )
    # REQ: LLR-0034
    p.add_argument(
        "--resume-from",
        default=None,
        help=(
            "Resume from a specific partial dir. If omitted, kdr searches "
            "`artifacts_dir/` for the highest-step `kdr_{mode}_partial_step*/` "
            "with `_SAVE_COMPLETE`; the search miss starts a fresh run from "
            "step 0."
        ),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entrypoint for `kdr-train`."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    args = _parse(argv)

    config = _load_config(Path(args.config))

    # CLI overrides honour LLR-0008's contract: `--mode` defaults to YAML and
    # overrides if given; `--student` overrides student.source.
    if args.mode is not None:
        config = config.model_copy(update={"mode": args.mode})
    if args.student is not None:
        config = config.model_copy(
            update={"student": config.student.model_copy(update={"source": args.student})}
        )

    artifacts_dir = Path(args.artifacts_dir).resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Persist the validated config to disk for postmortems. This runs before
    # the (potentially-expensive) accelerator + adapter setup so even a
    # failure in the resume search leaves a recoverable config snapshot.
    _dump_resolved_config(artifacts_dir, config)

    # Resume discovery — explicit --resume-from takes precedence.
    resume_step = 0
    if args.resume_from is not None:
        # REQ: LLR-0034
        resume_dir = Path(args.resume_from)
        if not resume_dir.exists():
            raise FileNotFoundError(
                f"--resume-from path does not exist: {resume_dir}"
            )
        try:
            resume_step = int(resume_dir.name.split("step")[-1])
        except ValueError as err:
            raise ValueError(
                f"Could not parse step from --resume-from path "
                f"{resume_dir} — expected a `kdr_{{mode}}_partial_step{{N}}` "
                "naming."
            ) from err
        log.info("Manual resume: step=%d from %s", resume_step, resume_dir)
    else:
        latest = find_latest_partial(artifacts_dir, config.mode)
        if latest is not None:
            partial_dir, resume_step = latest
            log.info("Auto-resume: step=%d from %s", resume_step, partial_dir)

    accelerator = Accelerator()
    if accelerator.is_main_process:
        log.info(
            "World size: %d  device: %s  distributed_type: %s",
            accelerator.num_processes,
            accelerator.device,
            accelerator.distributed_type,
        )

    adapter = Zaya1Adapter()
    source_metadata_path = _resolve_source_metadata_path(config.student.source)

    # Calibration batches: production builds them via moe_compress.utils;
    # extracted here so tests can monkey-patch a synthetic builder. The lazy
    # import keeps kdr importable in environments without moe_compress (e.g.
    # this WSL dev box).
    batches = _build_calibration_batches(config, accelerator)

    out_dir = run_recovery(
        config=config,
        adapter=adapter,
        accelerator=accelerator,
        artifacts_dir=artifacts_dir,
        batches=batches,
        resume_step=resume_step,
        source_metadata_path=source_metadata_path,
    )
    if accelerator.is_main_process:
        log.info("kdr complete -> %s", out_dir)
    return 0


def _load_config(path: Path) -> Config:
    """Load a YAML file and validate it via `Config.model_validate`."""
    with path.open() as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(
            f"Config file {path} did not parse to a mapping (got "
            f"{type(raw).__name__})."
        )
    return Config.model_validate(raw)


def _resolve_source_metadata_path(student_source: str) -> Path | None:
    """Return the path to the input student's `compressed_metadata.json` if
    it exists on disk (HLR-0005). For HF-hub repo ids the file is omitted —
    the adapter's `from_pretrained` brings it in transparently if present.

    Defensive: only resolves when `student_source` is a real directory.
    A repo id like `org/model` could spuriously match a CWD-relative
    `org/model/compressed_metadata.json` if such a directory happens to
    exist; the `is_dir()` guard rejects that case without a false positive
    on hub-only inputs.
    """
    src = Path(student_source)
    if not src.is_dir():
        return None
    p = src / COMPRESSED_METADATA_FILENAME
    return p if p.exists() else None


def _build_calibration_batches(
    config: Config, accelerator: Accelerator
) -> list[torch.Tensor]:
    """Build the calibration tensor and yield it as a list of micro-batches.

    Lazy-imports `moe_compress.utils.calibration` (a sibling project, NOT a
    kdr install dep). The function is split out so unit tests can patch
    `_build_calibration_batches` directly without touching the rest of the
    CLI.
    """
    try:
        from moe_compress.utils.calibration import (  # type: ignore[import-not-found]
            build_calibration_tensor,
            iter_batches,
            spec_from_config,
        )
    except ImportError as err:
        raise RuntimeError(
            "moe_compress is not installed. The CLI's calibration path "
            "depends on `moe_compress.utils.calibration` from the sibling "
            "max_quality project; install it (or invoke kdr's library API "
            "with pre-built `batches` directly)."
        ) from err

    # Calibration cache is decided by moe_compress (default
    # ./artifacts/_calibration_cache, CWD-relative). Rely on the API's
    # default — its signature types `cache_dir: str | Path` and would
    # TypeError on None.
    spec = spec_from_config(
        config.calibration.model_dump(), seed_offset=7
    )
    tokenizer = _load_tokenizer_for_calibration(config)
    calib = build_calibration_tensor(tokenizer, spec)
    batches: list[torch.Tensor] = list(
        iter_batches(calib, batch_size=config.distillation.per_device_batch_size)
    )
    if accelerator.is_main_process:
        log.info("Calibration: %d global micro-batches built.", len(batches))
    return batches


def _load_tokenizer_for_calibration(config: Config) -> Any:
    """Load the student's tokenizer for the calibration-tensor build.

    Calibration tokenisation must use the STUDENT's tokenizer, not the
    teacher's — the saved checkpoint ships the student's tokenizer, so the
    Chapter-2 handoff contract requires that.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        config.student.source, trust_remote_code=True
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def _dump_resolved_config(artifacts_dir: Path, config: Config) -> None:
    """Persist the validated config to artifacts_dir for postmortems."""
    payload = config.model_dump(mode="json")
    (artifacts_dir / "resolved_config.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True)
    )


if __name__ == "__main__":
    sys.exit(main())
