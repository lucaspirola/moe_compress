"""Stage 5 — Router KD (router-only KL distillation against the uncompressed
teacher).

Spec (VALIDATED_STRATEGIES §Stage 5, corrected hyperparameters):

- Optimizer       : AdamW
- Learning rate   : 5e-5 (CORRECTED from 1e-5)
- Epochs          : 1
- Batch size      : 2
- Grad accumulate : 4  (effective batch = 8)
- Max seq length  : 512
- KD temperature  : 1.0
- Max samples     : 3000
- Calibration     : C4 (this stage uses the same mix as Stage 2's calibration)

Only ``mlp.gate.weight`` (and ``.bias`` if present) is trainable. All other
parameters are frozen. The teacher and student must have matching routers
*before* this stage (Stage 2 re-sliced student routers so they now have the
surviving-experts set). For the teacher we reload the original pretrained
checkpoint.

Loss per layer:
    p_t(x) = softmax(logits_t / T)
    p_s(x) = softmax(logits_s / T)
    L      = mean_layer KL(p_t || p_s) · T²

The KL is computed over the *student's surviving experts* only — we read the
merge_map and reduce the teacher logits via max-pooling within each group
(which corresponds to "where the student sends a token, the teacher would
have preferred one of its children").

Artifact: ``stage5_final/`` — final safetensors + updated ``config.json``.
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils.calibration import CalibrationSpec, build_calibration_tensor, iter_batches
from .utils.model_io import (
    iter_moe_layers,
    load_json_artifact,
    load_model,
    save_checkpoint,
)

log = logging.getLogger(__name__)


def run(
    student,
    tokenizer,
    config: dict,
    artifacts_dir: Path,
    *,
    device=None,
) -> Path:
    s5 = config["stage5_router_kd"]
    cal = config["calibration"]

    merge_map = load_json_artifact(artifacts_dir / "stage2_pruned" / "merge_map.json")
    merge_map = {int(li): {int(k): list(v) for k, v in grp.items()} for li, grp in merge_map.items()}

    # Load teacher fresh from the original pretrained weights.
    log.info("Loading teacher for router KD: %s", config["model"]["name_or_path"])
    teacher, _ = load_model(
        config["model"]["name_or_path"],
        revision=config["model"]["revision"],
        torch_dtype=config["model"]["torch_dtype"],
        device_map=config["model"]["device_map"],
        attn_implementation=config["model"]["attn_implementation"],
        load_in_4bit=config["model"].get("load_in_4bit", False),
        trust_remote_code=config["model"].get("trust_remote_code", False),
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # Freeze everything on the student except routers.
    _freeze_non_routers(student, s5["trainable_name_patterns"])

    # Enable router logits on both for this stage.
    _set_output_router_logits(student, True)
    _set_output_router_logits(teacher, True)

    # Build calibration
    spec = CalibrationSpec(
        num_sequences=s5["max_calibration_samples"],
        sequence_length=s5["max_sequence_length"],
        seed=cal["seed"] + 5,
        domain_mix=cal["domain_mix"],
        c4_dataset=cal["dataset"],
        c4_subset=cal["subset"],
    )
    calib = build_calibration_tensor(
        tokenizer, spec, cache_dir=artifacts_dir / "_calibration_cache"
    )
    batches = iter_batches(calib, batch_size=s5["batch_size"])

    optim = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=s5["learning_rate"],
    )
    grad_accum = s5["gradient_accumulation"]
    T = s5["kd_temperature"]

    # FIX (review bug #4): capture the real MoE layer indices in order so we
    # can key into ``merge_map`` by global layer_idx rather than enumeration
    # position (which would alias on models with dense layers).
    student_moe_indices = [ref.layer_idx for ref in iter_moe_layers(student)]

    student.train()
    total_steps = (len(batches) // grad_accum) * s5["epochs"]
    log.info("Stage 5: %d routers trainable; %d steps (grad-accum=%d)",
             sum(1 for _ in student.parameters() if _.requires_grad),
             total_steps, grad_accum)

    step = 0
    optim.zero_grad()
    for epoch in range(s5["epochs"]):
        for i, batch in enumerate(batches):
            if device is not None:
                batch = batch.to(device)
            with torch.no_grad():
                teacher_out = teacher(input_ids=batch, output_router_logits=True)
            student_out = student(input_ids=batch, output_router_logits=True)

            t_router_logits = _router_logits_per_layer(teacher_out)
            s_router_logits = _router_logits_per_layer(student_out)
            if not t_router_logits or not s_router_logits:
                raise RuntimeError(
                    "Router logits not exposed in model outputs. "
                    "Check that transformers supports `output_router_logits=True` "
                    "for this model and that config.output_router_logits is True."
                )
            if len(t_router_logits) != len(s_router_logits):
                raise RuntimeError(
                    f"Teacher/student router-logit count mismatch: "
                    f"{len(t_router_logits)} vs {len(s_router_logits)}"
                )
            if len(s_router_logits) != len(student_moe_indices):
                raise RuntimeError(
                    f"Student router-logit count ({len(s_router_logits)}) "
                    f"does not match student MoE layer count "
                    f"({len(student_moe_indices)}). The model is emitting "
                    "router logits for a different set of layers than "
                    "iter_moe_layers() reports; update _router_logits_per_layer "
                    "to preserve positional alignment with None entries."
                )

            loss = torch.zeros((), device=s_router_logits[0].device, dtype=torch.float32)
            for pos, (tl, sl) in enumerate(zip(t_router_logits, s_router_logits)):
                li = student_moe_indices[pos] if pos < len(student_moe_indices) else -1
                if li not in merge_map:
                    # Teacher and student routers match → ordinary KL.
                    t_p = F.softmax(tl.to(torch.float32) / T, dim=-1)
                    s_lp = F.log_softmax(sl.to(torch.float32) / T, dim=-1)
                    loss = loss + F.kl_div(s_lp, t_p, reduction="batchmean") * (T ** 2)
                    continue
                # Pool teacher logits through merge_map: for each student slot,
                # the teacher provides logsumexp over its merged children.
                t_pool = _pool_teacher_logits(tl, merge_map[li])
                t_p = F.softmax(t_pool.to(torch.float32) / T, dim=-1)
                s_lp = F.log_softmax(sl.to(torch.float32) / T, dim=-1)
                loss = loss + F.kl_div(s_lp, t_p, reduction="batchmean") * (T ** 2)

            loss = loss / max(len(t_router_logits), 1)
            (loss / grad_accum).backward()

            if (i + 1) % grad_accum == 0:
                optim.step()
                optim.zero_grad()
                step += 1
                if step % config["logging"]["log_every_n_steps"] == 0:
                    log.info("  epoch=%d step=%d loss=%.6f", epoch, step, float(loss.item()))
        optim.zero_grad()

    _set_output_router_logits(student, False)

    out_dir = artifacts_dir / "stage5_final"
    save_checkpoint(student, tokenizer, out_dir)
    log.info("Stage 5 complete → %s", out_dir)
    return out_dir


def _freeze_non_routers(model: nn.Module, trainable_patterns: list[str]) -> None:
    for name, p in model.named_parameters():
        trainable = any(pat in name for pat in trainable_patterns)
        p.requires_grad_(trainable)


def _set_output_router_logits(model: nn.Module, flag: bool) -> None:
    cfg = getattr(model, "config", None)
    if cfg is None:
        return
    text_cfg = getattr(cfg, "text_config", cfg)
    if hasattr(text_cfg, "output_router_logits"):
        text_cfg.output_router_logits = flag
    else:
        setattr(text_cfg, "output_router_logits", flag)


def _router_logits_per_layer(model_output) -> list[torch.Tensor]:
    """Extract per-layer router logits from a transformers MoE output.

    FIX (Round 2 bug N-3): we preserve positional alignment even when the
    model emits ``None`` for dense/non-MoE layers. The caller filters the
    list using ``student_moe_indices`` as the ground truth, so non-tensor
    entries must be dropped *after* that alignment is computed — here we
    simply drop them (valid for this model because every layer is MoE), but
    any caller wanting per-layer positional correspondence should walk the
    raw output themselves.
    """
    rl = getattr(model_output, "router_logits", None)
    if rl is None:
        rl = getattr(model_output, "all_router_logits", None)
    if rl is None:
        return []
    if isinstance(rl, (list, tuple)):
        return [x for x in rl if isinstance(x, torch.Tensor)]
    return [rl]


def _pool_teacher_logits(
    teacher_logits: torch.Tensor, merge_map_layer: dict[int, list[int]],
) -> torch.Tensor:
    """Reduce teacher logits [B, T, num_teacher_experts] to
    [B, T, num_student_experts] by logsumexp within each merge group."""
    num_student = len(merge_map_layer)
    leading = teacher_logits.shape[:-1]
    out = torch.empty((*leading, num_student), dtype=teacher_logits.dtype, device=teacher_logits.device)
    for student_idx in range(num_student):
        children = merge_map_layer[student_idx]
        sub = teacher_logits.index_select(-1, torch.as_tensor(children, device=teacher_logits.device))
        out[..., student_idx] = torch.logsumexp(sub, dim=-1)
    return out
