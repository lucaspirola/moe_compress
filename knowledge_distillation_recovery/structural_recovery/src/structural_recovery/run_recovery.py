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


# _DSCHF_HOLDER and _activate_zero3_init live in distillation so all phases
# share one canonical implementation (imported below where needed).


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
    _validate_config(config)
    if args.student:
        config["student"]["source"] = args.student
    if args.teacher_source:
        config["teacher"]["name_or_path"] = args.teacher_source
        log.info("Override: teacher.name_or_path = %s (from --teacher-source)",
                 args.teacher_source)
    if args.smoke:
        # --smoke is an UPPER BOUND on total_tokens, not an override of the
        # YAML's tier choice. The smoke vs light split (FP8 vs BF16 teacher,
        # bnb vs DSCPUAdam, single-GPU vs ZeRO-3) is driven entirely by which
        # YAML you pass; this flag only caps the run length so a hand-passed
        # light YAML doesn't silently burn $60.
        config["distillation"]["total_tokens"] = min(
            int(config["distillation"]["total_tokens"]),
            50_000_000,
        )

    artifacts_dir = Path(args.artifacts_dir).absolute()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    log.info("Artifacts directory: %s", artifacts_dir)

    accelerator = _build_accelerator()
    if accelerator.is_main_process:
        log.info("World size: %d  device: %s  distributed_type: %s",
                 accelerator.num_processes, accelerator.device,
                 accelerator.distributed_type)
        (artifacts_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config))

    # SystemMetrics daemon: rank 0 only — GPU is shared so one sampler suffices.
    # moe_compress.utils is available via PYTHONPATH (entrypoint sets code_dir/src).
    metrics = None
    if accelerator.is_main_process:
        try:
            from moe_compress.utils.system_metrics import SystemMetrics
            metrics = SystemMetrics(interval_sec=30.0)
            metrics.start()
        except Exception as exc:
            log.warning("SystemMetrics startup failed (%s) — continuing without sampler.", exc)

    try:
        return _run(config, args, artifacts_dir, accelerator, metrics)
    finally:
        if metrics is not None:
            try:
                metrics.stop()
            except Exception as exc:
                log.warning("metrics.stop failed: %s", exc)


def _run(config, args, artifacts_dir, accelerator, _metrics) -> int:  # noqa: ARG001
    """Inner body of main() — separated so SystemMetrics stop is always called."""

    # 1. Load teacher (sharded under ZeRO-3 if DS, else replicated on device).
    #    Light tier on a100x4: BF16 (FP8 needs Hopper). Smoke on H200: FP8.
    teacher, teacher_tok = _load_teacher(config, accelerator)

    # 2. Load compressed student (sharded by DS in accelerator.prepare). We
    #    use the STUDENT's tokenizer for everything downstream — calibration
    #    and the saved chapter1_recovered/ checkpoint must use the tokenizer
    #    that ships with the student (Strategy A doesn't change vocabulary,
    #    but using the student's tokenizer is the spec contract for the
    #    Chapter 2 handoff). Sanity-check vocab sizes match the teacher's.
    #
    #    Auto-resume: if a valid partial exists on the bucket from a prior run,
    #    load its weights instead of the original student. The partial dir is a
    #    fully valid compressed checkpoint (contains compressed_metadata.json),
    #    so _load_student() handles it transparently via load_compressed_model().
    from .distillation import _load_latest_partial, enable_student_training, run_distillation
    partial_dir, resume_step = _load_latest_partial(artifacts_dir)
    if resume_step > 0:
        log.info("Auto-resume: step=%d from %s — overriding student source.", resume_step, partial_dir)
        config["student"]["source"] = str(partial_dir)

    student, tokenizer = _load_student(config, accelerator)
    _assert_tokenizers_compatible(tokenizer, teacher_tok)

    # 3. Set trainable params per scope
    enable_student_training(student, scope=config["distillation"]["trainable_scope"])

    # 4. Train
    out_dir = run_distillation(
        teacher, student, tokenizer, config, artifacts_dir, accelerator,
        resume_step=resume_step,
    )

    # 5. Final eval — collective (every rank participates)
    from . import eval_quick
    final_metrics = eval_quick.final_report(student, tokenizer, config, accelerator)
    if accelerator.is_main_process and final_metrics:
        (artifacts_dir / "chapter1_final_metrics.json").write_text(
            json.dumps(final_metrics, indent=2)
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
                   help="Cap total_tokens at 50M (upper bound, not override). "
                        "The smoke vs light tier choice (FP8 vs BF16 teacher, "
                        "ZeRO-3 vs single-GPU) comes entirely from the YAML "
                        "you pass via --config; this flag only caps run length.")
    return p.parse_args(argv)


def _load_config(path: str) -> dict[str, Any]:
    """Load YAML config and stamp the source path under ``_source_path`` so
    downstream code can produce actionable warnings ("bump foo in <this file>")."""
    with open(path) as f:
        config = yaml.safe_load(f)
    config["_source_path"] = str(Path(path).resolve())
    return config


# Keys consumed somewhere in the orchestrator/distillation/eval path. Keeping
# this list explicit catches typos at startup with one clear error rather
# than a bare KeyError deep inside training.
_REQUIRED_CONFIG_KEYS: tuple[tuple[str, ...], ...] = (
    ("student", "source"),
    ("teacher", "name_or_path"),
    ("calibration", "num_sequences"),
    ("calibration", "sequence_length"),
    ("distillation", "total_tokens"),
    ("distillation", "trainable_scope"),
    ("distillation", "optimizer"),
    ("distillation", "learning_rate"),
    ("distillation", "min_learning_rate"),
    ("distillation", "warmup_steps"),
    ("distillation", "per_device_batch_size"),
    ("distillation", "gradient_accumulation"),
    ("distillation", "sequence_length"),
    ("distillation", "temperature"),
    ("distillation", "grad_clip_norm"),
    ("distillation", "weight_decay"),
    ("distillation", "betas"),
    ("distillation", "use_gradient_checkpointing"),
    ("distillation", "log_every_n_steps"),
    ("distillation", "eval_every_n_steps"),
    ("distillation", "save_every_n_steps"),
)


def _validate_config(config: dict[str, Any]) -> None:
    """Fail-fast schema check: every required nested key is present.

    Reports ALL missing keys at once (not one-at-a-time on KeyError) plus the
    config file path so the operator can fix them in a single pass.
    """
    missing: list[str] = []
    for path in _REQUIRED_CONFIG_KEYS:
        node: Any = config
        ok = True
        for key in path:
            if not isinstance(node, dict) or key not in node:
                ok = False
                break
            node = node[key]
        if not ok:
            missing.append(".".join(path))
    if missing:
        src = config.get("_source_path", "<unknown>")
        raise ValueError(
            "Config is missing required keys (in {}): {}".format(
                src, ", ".join(missing),
            )
        )
    # Type-coerce the load-bearing scalar so a YAML string like "1.8e9"
    # doesn't reach min(...) as a non-int.
    try:
        tt = int(config["distillation"]["total_tokens"])
    except (TypeError, ValueError) as err:
        raise ValueError(
            f"distillation.total_tokens must be an int-coercible number; "
            f"got {config['distillation']['total_tokens']!r} ({err})"
        ) from None
    if tt <= 0:
        raise ValueError(
            f"distillation.total_tokens must be > 0; got {tt}."
        )


def _build_accelerator():
    """Construct an Accelerator. DeepSpeed plumbing comes from the launcher
    (``accelerate launch --use_deepspeed --deepspeed_config_file ...``). We
    never construct a DS plugin inline — that would conflict with the
    launcher-provided one."""
    from accelerate import Accelerator
    return Accelerator()


# Item 5: vocab/special-token compatibility check. ``len(tokenizer)`` alone
# misses subtle mismatches like a different ``eos_token_id`` that would
# silently break generation and KD.
_VOCAB_FIELDS = (
    "pad_token_id",
    "eos_token_id",
    "bos_token_id",
    "unk_token_id",
)


def _assert_tokenizers_compatible(student_tok, teacher_tok) -> None:
    """Raise if student/teacher tokenisations would diverge in any way that
    matters for KD. Emits a per-field diff in the error message."""
    diffs: list[tuple[str, Any, Any]] = []
    s_vocab, t_vocab = len(student_tok), len(teacher_tok)
    if s_vocab != t_vocab:
        diffs.append(("vocab_size", s_vocab, t_vocab))
    for field in _VOCAB_FIELDS:
        sv = getattr(student_tok, field, None)
        tv = getattr(teacher_tok, field, None)
        if sv != tv:
            diffs.append((field, sv, tv))
    s_specials = dict(getattr(student_tok, "special_tokens_map", {}) or {})
    t_specials = dict(getattr(teacher_tok, "special_tokens_map", {}) or {})
    if s_specials != t_specials:
        diffs.append(("special_tokens_map", s_specials, t_specials))

    if not diffs:
        log.info("Tokenizer compatibility: OK (vocab=%d, all fields match).", s_vocab)
        return

    lines = ["Tokenizer mismatch between student and teacher:"]
    # All fields, even matching, for context.
    all_fields = ["vocab_size"] + list(_VOCAB_FIELDS) + ["special_tokens_map"]
    diff_keys = {d[0] for d in diffs}
    for field in all_fields:
        if field == "vocab_size":
            sv, tv = s_vocab, t_vocab
        elif field == "special_tokens_map":
            sv, tv = s_specials, t_specials
        else:
            sv = getattr(student_tok, field, None)
            tv = getattr(teacher_tok, field, None)
        marker = "✗" if field in diff_keys else "✓"
        lines.append(f"  {field:<20s} student={sv!r}  teacher={tv!r}  {marker}")
    lines.append(
        "KD requires aligned tokenization. Verify both come from the same "
        "base model — Strategy A does not change vocabulary, so a divergence "
        "here means the student or teacher source is wrong."
    )
    raise RuntimeError("\n".join(lines))


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------


def _load_teacher(config: dict[str, Any], accelerator):
    """Load the teacher whose dtype is set by the YAML.

    Under DeepSpeed ZeRO-3 we activate ``HfDeepSpeedConfig`` BEFORE
    ``from_pretrained`` so the teacher params are immediately sharded across
    ranks. For the Light tier on a100x4 (BF16 teacher, ~70 GB), each rank
    sees ~17.5 GB. For the Smoke tier on 1×H200 the FP8 teacher (~37 GB)
    fits replicated. (FP8 inference needs Hopper — A100 has no FP8 tensor
    cores, so the BF16 teacher is the only option on a100x4.) Forward passes
    use deepspeed.zero hooks to all-gather sharded params on demand.

    Without DeepSpeed: load normally, replicate on the configured device.
    """
    from transformers import AutoConfig, AutoTokenizer
    from moe_compress.utils.model_io import _pick_auto_class  # canonical impl
    from .distillation import _activate_zero3_init, _is_deepspeed, _is_zero3

    name = config["teacher"]["name_or_path"]
    revision = config["teacher"].get("revision", "main")
    dtype_str = config["teacher"].get("torch_dtype", "bfloat16")
    dtype = getattr(torch, dtype_str) if isinstance(dtype_str, str) and dtype_str != "auto" else dtype_str
    attn_impl = config["teacher"].get("attn_implementation", "sdpa")

    cfg = AutoConfig.from_pretrained(name, revision=revision)
    auto_cls = _pick_auto_class(list(getattr(cfg, "architectures", []) or []))

    # Activate ZeRO-3 sharded init if applicable.
    _activate_zero3_init(accelerator)

    is_z3 = _is_zero3(accelerator)
    if accelerator.is_main_process:
        log.info("Loading TEACHER %s with %s (dtype=%s, attn=%s, sharded=%s)",
                 name, auto_cls.__name__, dtype, attn_impl, is_z3)

    teacher = auto_cls.from_pretrained(
        name, revision=revision, dtype=dtype,
        attn_implementation=attn_impl, low_cpu_mem_usage=True,
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    if not _is_deepspeed(accelerator):
        # Non-DS path (smoke / DDP): each rank places its replicated weights
        # on its own device.
        teacher = teacher.to(accelerator.device)

    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if accelerator.is_main_process:
        if is_z3:
            log.info("Teacher loaded (frozen, sharded across %d ranks)",
                     accelerator.num_processes)
        else:
            n = sum(p.numel() for p in teacher.parameters())
            log.info("Teacher loaded: %.2fB params (frozen)", n / 1e9)
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
