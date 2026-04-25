"""CLI orchestrator for Chapter 1 — Structural Recovery at BF16.

Single-purpose: KD only. Teacher correction is a SEPARATE script
(``structural_recovery.teacher_correction``) that runs in its own subprocess
to keep DeepSpeed engine state clean (only one engine per process).

Usage:

    accelerate launch --use_deepspeed \
        --deepspeed_config_file ds_configs/zero3_offload_optim.json \
        --mixed_precision bf16 \
        -m structural_recovery.run_recovery \
        --config configs/qwen36_35b_a3b_chapter1_light.yaml \
        --student /path/to/stage5_final \
        --artifacts-dir ./recovery_artifacts \
        [--teacher-source /path/to/teacher_corrected_bf16] \
        [--smoke]

For smoke (single-GPU, no DS):

    python -m structural_recovery.run_recovery \
        --config configs/qwen36_35b_a3b_chapter1_smoke.yaml \
        --student ./stage5_final \
        --artifacts-dir ./recovery_artifacts --smoke
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

log = logging.getLogger(__name__)


# Stays at module scope so deepspeed.zero.Init keeps its context alive after
# the loader function returns. (HfDeepSpeedConfig holds a thread-local global
# but we keep this reference too as defense-in-depth.)
_DSCHF_HOLDER: list = []


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    config = _load_config(args.config)
    if args.student:
        config["student"]["source"] = args.student
    if args.teacher_source:
        config["teacher"]["name_or_path"] = args.teacher_source
        log.info("Override: teacher.name_or_path = %s (from --teacher-source)",
                 args.teacher_source)
    if args.smoke:
        config["distillation"]["total_tokens"] = min(
            int(config["distillation"]["total_tokens"]),
            50_000_000,
        )

    artifacts_dir = Path(args.artifacts_dir).absolute()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    log.info("Artifacts directory: %s", artifacts_dir)

    accelerator = _build_accelerator(config)
    if accelerator.is_main_process:
        log.info("World size: %d  device: %s  distributed_type: %s",
                 accelerator.num_processes, accelerator.device,
                 accelerator.distributed_type)
        (artifacts_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config))

    # 1. Load FP8 teacher (sharded under ZeRO-3 if DS, else replicated on device)
    teacher, _teacher_tok = _load_teacher(config, accelerator)

    # 2. Load compressed student (sharded by DS in accelerator.prepare). We
    #    use the STUDENT's tokenizer for everything downstream — calibration
    #    and the saved chapter1_recovered/ checkpoint must use the tokenizer
    #    that ships with the student (Strategy A doesn't change vocabulary,
    #    but using the student's tokenizer is the spec contract for the
    #    Chapter 2 handoff). Sanity-check vocab sizes match the teacher's.
    student, tokenizer = _load_student(config, accelerator)
    if len(tokenizer) != len(_teacher_tok):
        raise RuntimeError(
            f"Tokenizer vocab mismatch: student={len(tokenizer)} "
            f"teacher={len(_teacher_tok)}. KD requires aligned vocabularies."
        )

    # 3. Set trainable params per scope
    from .distillation import enable_student_training, run_distillation
    enable_student_training(student, scope=config["distillation"]["trainable_scope"])

    # 4. Train
    out_dir = run_distillation(
        teacher, student, tokenizer, config, artifacts_dir, accelerator,
    )

    # 5. Final eval — collective (every rank participates)
    from . import eval_quick
    metrics = eval_quick.final_report(student, tokenizer, config, accelerator)
    if accelerator.is_main_process and metrics:
        (artifacts_dir / "chapter1_final_metrics.json").write_text(
            json.dumps(metrics, indent=2)
        )
    if accelerator.is_main_process:
        log.info("Chapter 1 complete -> %s", out_dir)
    return 0


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------


def _parse(argv) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chapter 1 — Structural Recovery at BF16.")
    p.add_argument("--config", required=True)
    p.add_argument("--student", default=None,
                   help="Override student.source (path to stage5_final or HF repo id).")
    p.add_argument("--teacher-source", default=None,
                   help="Override teacher.name_or_path (e.g. point at the local "
                        "artifacts/teacher_corrected_bf16/ produced by "
                        "structural_recovery.teacher_correction).")
    p.add_argument("--artifacts-dir", default="./recovery_artifacts")
    p.add_argument("--smoke", action="store_true",
                   help="Cap total_tokens at 50M regardless of config.")
    return p.parse_args(argv)


def _load_config(path: str) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def _build_accelerator(config: dict[str, Any]):
    """Construct an Accelerator. DeepSpeed plumbing comes from the launcher
    (``accelerate launch --use_deepspeed --deepspeed_config_file ...``). We
    never construct a DS plugin inline — that would conflict with the
    launcher-provided one."""
    from accelerate import Accelerator
    return Accelerator()


# ---------------------------------------------------------------------------
# DeepSpeed helpers
# ---------------------------------------------------------------------------


def _is_deepspeed(accelerator) -> bool:
    from accelerate.utils import DistributedType
    return accelerator.distributed_type == DistributedType.DEEPSPEED


def _is_zero3(accelerator) -> bool:
    if not _is_deepspeed(accelerator):
        return False
    plugin = getattr(accelerator.state, "deepspeed_plugin", None)
    return plugin is not None and int(plugin.zero_stage) >= 3


def _activate_zero3_init(accelerator) -> None:
    """Pin a global ``HfDeepSpeedConfig`` so subsequent ``from_pretrained``
    calls construct ZeRO-3-sharded params via ``deepspeed.zero.Init``.

    HfDeepSpeedConfig stores a process-wide reference internally; we ALSO
    keep one in ``_DSCHF_HOLDER`` so a future GC doesn't drop the only ref.
    No-op if not running under DeepSpeed ZeRO-3.
    """
    if not _is_zero3(accelerator):
        return
    if _DSCHF_HOLDER:
        return  # already activated this process
    from transformers.integrations import HfDeepSpeedConfig
    plugin = accelerator.state.deepspeed_plugin
    ds_config = plugin.deepspeed_config        # dict, fully resolved
    _DSCHF_HOLDER.append(HfDeepSpeedConfig(ds_config))
    log.info("HfDeepSpeedConfig activated for ZeRO-3 sharded from_pretrained.")


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------


def _load_teacher(config: dict[str, Any], accelerator):
    """Load the (FP8 by default) teacher.

    Under DeepSpeed ZeRO-3 we activate ``HfDeepSpeedConfig`` BEFORE
    ``from_pretrained`` so the teacher params are immediately sharded across
    ranks (~37 GB FP8 / 4 ≈ 9 GB per GPU). Forward passes use deepspeed.zero
    hooks to all-gather the params on demand.

    Without DeepSpeed: load normally, replicate on the configured device.
    """
    from transformers import AutoConfig, AutoTokenizer
    from moe_compress.utils.model_io import _pick_auto_class  # canonical impl

    name = config["teacher"]["name_or_path"]
    revision = config["teacher"].get("revision", "main")
    dtype_str = config["teacher"].get("torch_dtype", "bfloat16")
    dtype = getattr(torch, dtype_str)
    attn_impl = config["teacher"].get("attn_implementation", "sdpa")

    cfg = AutoConfig.from_pretrained(name, revision=revision)
    auto_cls = _pick_auto_class(list(getattr(cfg, "architectures", []) or []))

    # Activate ZeRO-3 sharded init if applicable.
    _activate_zero3_init(accelerator)

    if accelerator.is_main_process:
        log.info("Loading TEACHER %s with %s (dtype=%s, attn=%s, sharded=%s)",
                 name, auto_cls.__name__, dtype, attn_impl, _is_zero3(accelerator))

    teacher = auto_cls.from_pretrained(
        name, revision=revision, dtype=dtype,
        attn_implementation=attn_impl, low_cpu_mem_usage=True,
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    if not _is_deepspeed(accelerator):
        # Single-GPU smoke: explicit placement.
        teacher = teacher.to(accelerator.device)

    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if accelerator.is_main_process:
        n = sum(p.numel() for p in teacher.parameters())
        log.info("Teacher loaded: %.2fB params (frozen, sharded=%s)",
                 n / 1e9, _is_zero3(accelerator))
    return teacher, tokenizer


def _load_student(config: dict[str, Any], accelerator):
    """Reconstruct the compressed student via max_quality's loader.

    The custom loader does ``from_config + _resize_moe_stack_to_metadata +
    load_state_dict(assign=True)``. We pass ``device_map=None`` so it lands
    on CPU; ``accelerator.prepare`` (called inside ``run_distillation``)
    then hands it to ``deepspeed.initialize`` which shards from CPU to GPU.

    NOTE on memory: each rank loads the FULL ~70 GB student state dict on
    CPU. With 4 ranks that's 280 GB host RAM during the load — fits within
    the 568 GB on HF Jobs ``a100x4``. After ``accelerator.prepare`` shards
    the model, per-rank CPU footprint drops back to ~17.5 GB.

    Returns ``(student, tokenizer)`` — the student's tokenizer is used for
    calibration and the saved checkpoint (Chapter 2 handoff contract).
    """
    from moe_compress.utils.model_io import load_compressed_model

    src = config["student"]["source"]
    dtype_str = config["student"].get("torch_dtype", "bfloat16")
    attn_impl = config["student"].get("attn_implementation", "sdpa")

    if accelerator.is_main_process:
        log.info("Loading STUDENT from %s (dtype=%s, attn=%s)", src, dtype_str, attn_impl)

    # device_map="auto" would fight ZeRO-3 placement. None lands on CPU.
    student, tokenizer, meta = load_compressed_model(
        src,
        device_map=None,
        torch_dtype=dtype_str,
        attn_implementation=attn_impl,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if accelerator.is_main_process:
        log.info("Student loaded on CPU: %.2fB params, pipeline_stage=%s",
                 sum(p.numel() for p in student.parameters()) / 1e9,
                 meta.get("pipeline_stage", "?"))
    return student, tokenizer


if __name__ == "__main__":
    sys.exit(main())
