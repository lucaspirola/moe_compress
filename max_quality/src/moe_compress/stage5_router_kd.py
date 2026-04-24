"""Stage 5 — Router KD, fused-experts-aware.

Structural change from the pre-refactor version: ``Qwen3_5MoeSparseMoeBlock``
does NOT expose router logits through the model output tuple. We capture
them via a forward hook on each layer's ``gate`` (Qwen3_5MoeTopKRouter)
using :func:`capture_router_outputs`.

Everything else (merge-map pooling, KL loss, AdamW hyperparameters) is
unchanged from the earlier design.
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils.activation_hooks import capture_router_outputs
from .utils.calibration import build_calibration_tensor, iter_batches, spec_from_config
from .utils.model_io import (
    iter_moe_layers,
    load_json_artifact,
    load_model,
    save_compressed_checkpoint,
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

    _freeze_non_routers(student, s5["trainable_name_patterns"])

    spec = spec_from_config(
        cal,
        num_sequences_override=s5["max_calibration_samples"],
        sequence_length_override=s5["max_sequence_length"],
        seed_offset=5,
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

    teacher_refs = list(iter_moe_layers(teacher))
    student_refs = list(iter_moe_layers(student))
    assert len(teacher_refs) == len(student_refs), \
        f"Teacher/student layer count mismatch: {len(teacher_refs)} vs {len(student_refs)}"

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

            with torch.no_grad(), capture_router_outputs(teacher_refs) as t_out:
                teacher(input_ids=batch)

            with capture_router_outputs(student_refs) as s_out:
                student(input_ids=batch)

            loss = torch.zeros((), device=batch.device, dtype=torch.float32)
            n_layers = 0
            for t_ref, s_ref in zip(teacher_refs, student_refs):
                li = t_ref.layer_idx
                t_logits = t_out.get(li, [])
                s_logits = s_out.get(li, [])
                if not t_logits or not s_logits:
                    continue
                tl = t_logits[-1].to(torch.float32)
                sl = s_logits[-1].to(torch.float32)
                if li in merge_map:
                    tl = _pool_teacher_logits(tl, merge_map[li])
                t_p = F.softmax(tl / T, dim=-1)
                s_lp = F.log_softmax(sl / T, dim=-1)
                loss = loss + F.kl_div(s_lp, t_p, reduction="batchmean") * (T ** 2)
                n_layers += 1
            if n_layers == 0:
                raise RuntimeError(
                    "No router logits captured from either teacher or student. "
                    "Hooks may not be firing — check iter_moe_layers + router classes."
                )
            loss = loss / n_layers
            (loss / grad_accum).backward()

            if (i + 1) % grad_accum == 0:
                optim.step()
                optim.zero_grad()
                step += 1
                if step % config["logging"]["log_every_n_steps"] == 0:
                    log.info("  epoch=%d step=%d loss=%.6f", epoch, step, float(loss.item()))
        optim.zero_grad()

    out_dir = artifacts_dir / "stage5_final"
    save_compressed_checkpoint(
        student, tokenizer, out_dir, pipeline_stage="stage5_final",
    )
    log.info("Stage 5 complete → %s", out_dir)
    return out_dir


def _freeze_non_routers(model: nn.Module, trainable_patterns: list[str]) -> None:
    for name, p in model.named_parameters():
        p.requires_grad_(any(pat in name for pat in trainable_patterns))


def _pool_teacher_logits(
    teacher_logits: torch.Tensor, merge_map_layer: dict[int, list[int]],
) -> torch.Tensor:
    num_student = len(merge_map_layer)
    leading = teacher_logits.shape[:-1]
    out = torch.empty((*leading, num_student),
                      dtype=teacher_logits.dtype,
                      device=teacher_logits.device)
    for student_idx in range(num_student):
        children = merge_map_layer[student_idx]
        sub = teacher_logits.index_select(-1, torch.as_tensor(children, device=teacher_logits.device))
        out[..., student_idx] = torch.logsumexp(sub, dim=-1)
    return out
