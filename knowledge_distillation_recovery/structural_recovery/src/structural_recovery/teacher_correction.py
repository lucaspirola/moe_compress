"""Teacher correction (Minitron BP#9, +~3 MMLU when distill data ≠ pretrain data).

**Standalone CLI** — runs as its own subprocess so its DeepSpeed engine state
doesn't conflict with the KD engine in ``run_recovery``. DeepSpeed only allows
one engine per process, so chaining the two phases in one process is fragile.

Usage:

    accelerate launch --use_deepspeed \
        --deepspeed_config_file ds_configs/zero3_offload_optim.json \
        --mixed_precision bf16 \
        -m structural_recovery.teacher_correction \
        --config configs/qwen36_35b_a3b_chapter1_light.yaml \
        --artifacts-dir ./recovery_artifacts \
        [--bf16-teacher Qwen/Qwen3.6-35B-A3B]    # default: derived from config

Or single-GPU (smoke):

    python -m structural_recovery.teacher_correction \
        --config configs/qwen36_35b_a3b_chapter1_smoke.yaml \
        --artifacts-dir ./recovery_artifacts

Output: ``artifacts_dir/teacher_corrected_bf16/`` — a standard
``save_pretrained`` BF16 checkpoint. The KD entrypoint then passes
``--teacher-source <out_dir>`` to ``run_recovery`` to use the corrected
teacher instead of the original FP8.

Note: this saves BF16 (not FP8). The KD step accepts a BF16 teacher fine —
it costs ~33 GB extra VRAM vs the FP8 path but still fits ``a100x4`` with
margin. A future revision could re-quantize via llm-compressor.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

log = logging.getLogger(__name__)


# Module-scope holder so HfDeepSpeedConfig instances aren't GC'd between the
# loader function returning and the subsequent ``from_pretrained`` calls.
# (HfDeepSpeedConfig keeps a thread-local global internally, but defence in
# depth — this also makes the lifetime obvious to readers.)
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
    if not config["teacher_correction"]["enabled"]:
        log.info("teacher_correction.enabled=false in config; nothing to do.")
        return 0

    target = config["teacher_correction"].get("target", "bf16_then_requantize")
    if target == "lora_on_fp8":
        raise NotImplementedError(
            "teacher_correction.target='lora_on_fp8' is reserved for a future "
            "revision. Use 'bf16_then_requantize' or set "
            "teacher_correction.enabled=false."
        )
    if target != "bf16_then_requantize":
        raise ValueError(
            f"Unknown teacher_correction.target: {target!r}. "
            "Expected 'bf16_then_requantize' or 'lora_on_fp8'."
        )

    artifacts_dir = Path(args.artifacts_dir).absolute()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    out_dir = artifacts_dir / "teacher_corrected_bf16"

    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        log.info("Output already exists at %s; pass --overwrite to redo.", out_dir)
        return 0

    # Build accelerator (picks up DeepSpeed from launcher if applicable).
    from accelerate import Accelerator
    accelerator = Accelerator()

    # Resolve which BF16 teacher to fine-tune.
    bf16_name = (
        args.bf16_teacher
        or config["teacher_correction"].get("bf16_teacher_name_or_path")
        or _strip_fp8_suffix(config["teacher"]["name_or_path"])
    )
    if accelerator.is_main_process:
        log.info("======== Teacher Correction (BP#9) ========")
        log.info("BF16 teacher: %s", bf16_name)
        log.info("Target output: %s", out_dir)

    # Load teacher (sharded under ZeRO-3 if applicable).
    teacher, tokenizer = _load_bf16_teacher(bf16_name, config, accelerator)

    # Fine-tune.
    _finetune_ce(teacher, tokenizer, config, artifacts_dir, accelerator)

    # Save (gathered under ZeRO-3).
    _save_bf16(teacher, tokenizer, out_dir, accelerator)

    if accelerator.is_main_process:
        log.info("Teacher correction complete -> %s", out_dir)
    return 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse(argv) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Teacher correction (Minitron BP#9).")
    p.add_argument("--config", required=True)
    p.add_argument("--artifacts-dir", default="./recovery_artifacts")
    p.add_argument("--bf16-teacher", default=None,
                   help="HF repo or local path to the BF16 teacher to fine-tune. "
                        "Defaults to stripping the -FP8 suffix from "
                        "config.teacher.name_or_path.")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-run even if teacher_corrected_bf16/ already exists.")
    return p.parse_args(argv)


def _load_config(path: str) -> dict[str, Any]:
    """Load YAML config; stamp source path for actionable warnings."""
    with open(path) as f:
        config = yaml.safe_load(f)
    config["_source_path"] = str(Path(path).resolve())
    return config


def _strip_fp8_suffix(name: str) -> str:
    """``Qwen/Qwen3.6-35B-A3B-FP8`` → ``Qwen/Qwen3.6-35B-A3B``."""
    for suffix in ("-FP8", "-fp8"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _is_zero3(accelerator) -> bool:
    from accelerate.utils import DistributedType
    if accelerator.distributed_type != DistributedType.DEEPSPEED:
        return False
    plugin = getattr(accelerator.state, "deepspeed_plugin", None)
    return plugin is not None and int(plugin.zero_stage) >= 3


def _activate_zero3_init(accelerator) -> None:
    """Pin a module-scope HfDeepSpeedConfig so subsequent from_pretrained
    constructs ZeRO-3-sharded params via ``deepspeed.zero.Init``.

    Same pattern as ``run_recovery._activate_zero3_init`` — uses the
    module-scope ``_DSCHF_HOLDER`` so the pin survives function exits.
    No-op if not under DeepSpeed ZeRO-3 or if already activated.
    """
    if not _is_zero3(accelerator):
        return
    if _DSCHF_HOLDER:
        return  # already activated this process
    from transformers.integrations import (
        HfDeepSpeedConfig, is_deepspeed_zero3_enabled,
    )
    plugin = accelerator.state.deepspeed_plugin
    ds_config = plugin.deepspeed_config
    _DSCHF_HOLDER.append(HfDeepSpeedConfig(ds_config))

    # Item 7 mirror: hard-fail if the activation didn't take effect.
    if not is_deepspeed_zero3_enabled():
        try:
            import deepspeed                              # noqa: F401
            ds_avail = True
        except ImportError:
            ds_avail = False
        raise RuntimeError(
            "HfDeepSpeedConfig was instantiated but "
            "is_deepspeed_zero3_enabled() returned False — the BF16 teacher "
            "would load full-rank on each rank and OOM. Inspect: "
            f"plugin.zero_stage={getattr(plugin, 'zero_stage', '?')}, "
            f"deepspeed_importable={ds_avail}, "
            f"ds_config['zero_optimization']['stage']="
            f"{ds_config.get('zero_optimization', {}).get('stage', '?')}."
        )
    log.info("HfDeepSpeedConfig activated for ZeRO-3 sharded from_pretrained.")


def _load_bf16_teacher(name: str, config, accelerator):
    from transformers import AutoConfig, AutoTokenizer
    from moe_compress.utils.model_io import _pick_auto_class

    revision = config["teacher"].get("revision", "main")
    attn_impl = config["teacher"].get("attn_implementation", "sdpa")

    cfg = AutoConfig.from_pretrained(name, revision=revision)
    auto_cls = _pick_auto_class(list(getattr(cfg, "architectures", []) or []))

    # Pin HfDeepSpeedConfig if ZeRO-3 (lifetime managed via _DSCHF_HOLDER).
    _activate_zero3_init(accelerator)

    if accelerator.is_main_process:
        log.info("Loading BF16 teacher %s with %s (sharded=%s)",
                 name, auto_cls.__name__, _is_zero3(accelerator))
    teacher = auto_cls.from_pretrained(
        name, revision=revision, dtype=torch.bfloat16,
        attn_implementation=attn_impl, low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return teacher, tokenizer


def _finetune_ce(teacher, tokenizer, config, artifacts_dir, accelerator):
    """Brief cross-entropy fine-tune on Cascade calibration tokens."""
    from moe_compress.utils.calibration import (
        build_calibration_tensor, iter_batches, spec_from_config,
    )
    from .distillation import (
        _all_finite, _is_deepspeed, _shard_batches_per_rank, build_optimizer,
    )

    tcc = config["teacher_correction"]

    # Make all params trainable.
    for p in teacher.parameters():
        p.requires_grad_(True)

    # Gradient checkpointing must be enabled BEFORE accelerator.prepare.
    if config["distillation"]["use_gradient_checkpointing"]:
        teacher.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    # Calibration: small slice (just enough for the configured steps).
    cal = config["calibration"]
    seq_len = int(cal["sequence_length"])
    micro = int(tcc["per_device_batch_size"])
    grad_accum = int(tcc["gradient_accumulation"])
    steps = int(tcc["steps"])
    world = max(1, accelerator.num_processes)
    needed_seqs = micro * grad_accum * steps * world  # global, then strided per-rank
    spec = spec_from_config(
        cal,
        num_sequences_override=needed_seqs,
        sequence_length_override=seq_len,
        seed_offset=8,                                 # disjoint from KD's 7
    )
    calib = build_calibration_tensor(
        tokenizer, spec, cache_dir=artifacts_dir / "_calibration_cache",
    )
    all_batches = iter_batches(calib, batch_size=micro)
    # Truncate to a (world × grad_accum) multiple before per-rank shard so
    # all ranks finish lockstep (avoid NCCL hang on tail mismatch).
    truncate_to = (len(all_batches) // (world * grad_accum)) * (world * grad_accum)
    if truncate_to < len(all_batches):
        all_batches = all_batches[:truncate_to]
    batches = _shard_batches_per_rank(all_batches, accelerator)
    if accelerator.is_main_process:
        log.info("teacher_correction :: %d global → %d local batches per rank",
                 len(all_batches), len(batches))

    # Optimizer: same machinery as KD.
    # (For teacher correction we use the same optimizer choice as KD —
    # adamw_bnb_8bit on smoke, deepspeed_cpu_adam on light.)
    fake_dconf = {
        "optimizer": config["distillation"]["optimizer"],
        "learning_rate": tcc["learning_rate"],
        "betas": [0.9, 0.999],
        "weight_decay": 0.0,
    }
    optim = build_optimizer(teacher, fake_dconf)
    teacher, optim = accelerator.prepare(teacher, optim)
    teacher.train()

    is_ds = _is_deepspeed(accelerator)
    grad_clip = float(config["distillation"]["grad_clip_norm"])
    warmup = int(tcc["warmup_steps"])
    optim.zero_grad(set_to_none=True)

    step = 0
    micro_idx = 0
    for batch in batches:
        ids = batch.to(accelerator.device, non_blocking=True)
        out = teacher(input_ids=ids, labels=ids)
        loss = out.loss

        if not _all_finite(loss, accelerator):
            if accelerator.is_main_process:
                log.warning("teacher_correction :: step=%d non-finite loss; substituting zero.", step)
            if is_ds:
                # Build zero loss graph-connected via teacher logits so DS's
                # micro-batch counter advances. NaN * 0 = NaN, so use a
                # nan-cleaned tensor.
                cleaned_logits = torch.nan_to_num(
                    out.logits, nan=0.0, posinf=0.0, neginf=0.0,
                )
                loss = cleaned_logits.sum() * 0.0
            else:
                optim.zero_grad(set_to_none=True)
                micro_idx = 0
                continue

        accelerator.backward(loss / grad_accum)
        micro_idx += 1
        if micro_idx % grad_accum == 0:
            if not is_ds and grad_clip > 0:
                accelerator.clip_grad_norm_(teacher.parameters(), grad_clip)
            # Linear warmup, constant LR after (no decay over ~400 steps).
            lr = float(tcc["learning_rate"]) * min(1.0, (step + 1) / max(1, warmup))
            for g in optim.param_groups:
                g["lr"] = lr
            optim.step()
            optim.zero_grad(set_to_none=True)
            step += 1
            micro_idx = 0
            if accelerator.is_main_process and step % 25 == 0:
                log.info("teacher_correction :: step=%d/%d ce_loss=%.4f lr=%.2e",
                         step, steps, float(loss.item()), lr)
            if step >= steps:
                break

    teacher.eval()
    # Re-disable grad ckpt so subsequent (out-of-process) consumers don't
    # incur its overhead and don't trip "no input requires grad" warnings.
    try:
        accelerator.unwrap_model(teacher).gradient_checkpointing_disable()
    except Exception:                                            # noqa: BLE001
        pass


def _save_bf16(teacher, tokenizer, out_dir: Path, accelerator) -> None:
    """Save the corrected teacher as a standard transformers BF16 checkpoint.

    Under DS3 we use ``accelerator.get_state_dict`` (CPU-streamed gather on
    rank 0), NOT ``GatheredParameters`` (which would re-materialise the full
    70 GB BF16 model on every GPU and OOM).
    """
    accelerator.wait_for_everyone()

    state_dict = accelerator.get_state_dict(teacher)

    if accelerator.is_main_process:
        out_dir.mkdir(parents=True, exist_ok=True)
        unwrapped = accelerator.unwrap_model(teacher)
        unwrapped.save_pretrained(
            out_dir, state_dict=state_dict, safe_serialization=True,
        )
        tokenizer.save_pretrained(out_dir)

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    sys.exit(main())
