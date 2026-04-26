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
from .utils.trackio_log import trackio_log as _trackio_log

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

    # Stage 5 holds teacher (~70 GB BF16) AND student (~50 GB BF16) on cuda
    # at once — exceeds 80 GB A100 and forces CPU offload (5–10× slowdown).
    # Two mitigations:
    #   (A) teacher_load_in_4bit: true  — bitsandbytes NF4, ~17 GB live.
    #   (B) teacher_logits_cache: <path> — sidecar produced by
    #       hf_jobs/precompute_teacher_logits.py; skip live teacher entirely.
    # If both are set, (B) wins.
    teacher = None
    teacher_refs = None
    teacher_logits_cache = None
    cache_path_cfg = s5.get("teacher_logits_cache")
    if cache_path_cfg:
        cache_path = Path(cache_path_cfg)
        if not cache_path.is_absolute():
            cache_path = artifacts_dir / cache_path
        if cache_path.exists():
            log.info("Stage 5: loading precomputed teacher logits from %s", cache_path)
            # mmap=True keeps the ~30 GB sidecar memory-mapped instead of
            # materializing the whole thing in CPU RAM. Each per-batch
            # slice pages in only what the loop touches.
            cache_payload = torch.load(cache_path, map_location="cpu", mmap=True)
            teacher_logits_cache = cache_payload
            # Schema check: any future format change must bump this.
            fmt = int(cache_payload.get("format_version", 0))
            if fmt != 1:
                raise RuntimeError(
                    f"Teacher-logits cache format_version={fmt} unsupported "
                    "(this Stage 5 only knows version 1). Regenerate the cache "
                    "or upgrade Stage 5."
                )
            cached_bs = int(cache_payload.get("batch_size", -1))
            if cached_bs != int(s5["batch_size"]):
                raise RuntimeError(
                    f"Teacher-logits cache batch_size={cached_bs} disagrees with "
                    f"stage5_router_kd.batch_size={s5['batch_size']}. Re-run the "
                    "precompute or align the configs — the cache is keyed token-by-token."
                )
            if int(cache_payload.get("sequence_length", -1)) != int(s5["max_sequence_length"]):
                raise RuntimeError(
                    "Teacher-logits cache sequence_length disagrees with config — "
                    "re-run precompute or align configs."
                )
            # F3 fix: also verify num_samples matches. A cache built with
            # fewer samples than Stage 5 expects would silently return
            # zero-length slices for late batches → degenerate KD signal.
            cache_n = int(cache_payload.get("num_samples", -1))
            cfg_n = int(s5["max_calibration_samples"])
            if cache_n != cfg_n:
                raise RuntimeError(
                    f"Teacher-logits cache num_samples={cache_n} disagrees with "
                    f"stage5_router_kd.max_calibration_samples={cfg_n}. Stage 5 "
                    "would read past the end of the cache — regenerate or align."
                )
            log.info("Stage 5: cache covers %d samples, %d sequence_length, %d layers",
                     cache_payload.get("num_samples"), cache_payload.get("sequence_length"),
                     len(cache_payload.get("logits", {})))
        else:
            log.warning("Stage 5: teacher_logits_cache=%s not found at %s — falling back to live teacher",
                        cache_path_cfg, cache_path)

    if teacher_logits_cache is None:
        load_in_4bit = bool(s5.get("teacher_load_in_4bit", False))
        # If the global model load_in_4bit is on (Stages 0-2 used 4-bit on
        # smaller hardware) but Stage 5 explicitly didn't opt into 4-bit
        # for the teacher, the user is about to load a BF16 35 B teacher
        # alongside the student → almost certainly OOM. Warn loudly.
        if config["model"].get("load_in_4bit", False) and not load_in_4bit:
            log.warning(
                "Stage 5: config['model']['load_in_4bit']=true but "
                "stage5_router_kd.teacher_load_in_4bit=false. The teacher "
                "will load in BF16 (~70 GB) and likely OOM the A100. "
                "Set teacher_load_in_4bit: true to match."
            )
        # When 4-bit, force device_map={"": 0} so the entire quantized teacher
        # lands on the same GPU as the student. bnb's NF4 packs Qwen3.6-35B
        # into ~17 GB → fits comfortably alongside the ~50 GB student.
        device_map = {"": 0} if load_in_4bit else config["model"]["device_map"]
        log.info("Loading teacher for router KD: %s (load_in_4bit=%s, device_map=%s)",
                 config["model"]["name_or_path"], load_in_4bit, device_map)
        # Decouple Stage 5's teacher 4-bit decision from the global
        # config["model"]["load_in_4bit"]. The global flag is meant for
        # initial-model loads (Stages 0-2); Stage 5 should make its own
        # explicit decision.
        teacher, _ = load_model(
            config["model"]["name_or_path"],
            revision=config["model"]["revision"],
            torch_dtype=config["model"]["torch_dtype"],
            device_map=device_map,
            attn_implementation=config["model"]["attn_implementation"],
            load_in_4bit=load_in_4bit,
            trust_remote_code=config["model"].get("trust_remote_code", False),
        )
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        teacher_refs = list(iter_moe_layers(teacher))

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

    student_refs = list(iter_moe_layers(student))
    student_layer_idxs = {r.layer_idx for r in student_refs}
    if teacher_refs is not None:
        assert len(teacher_refs) == len(student_refs), \
            f"Teacher/student layer count mismatch: {len(teacher_refs)} vs {len(student_refs)}"
        assert {r.layer_idx for r in teacher_refs} == student_layer_idxs, \
            "Teacher/student MoE layer indices disagree"
    elif teacher_logits_cache is not None:
        cache_layer_idxs = set(teacher_logits_cache["logits"].keys())
        if cache_layer_idxs != student_layer_idxs:
            raise RuntimeError(
                f"Teacher-logits cache covers layers {sorted(cache_layer_idxs)} "
                f"but student has MoE layers {sorted(student_layer_idxs)}. "
                "Regenerate the cache against the same model architecture."
            )

    student.train()
    total_steps = (len(batches) // grad_accum) * s5["epochs"]
    log.info("Stage 5: %d routers trainable; %d steps (grad-accum=%d)",
             sum(1 for _ in student.parameters() if _.requires_grad),
             total_steps, grad_accum)

    step = 0
    optim.zero_grad()
    cache_seq_len = int(s5["max_sequence_length"])
    cache_batch_size = int(s5["batch_size"])
    cache_tokens_per_batch = cache_batch_size * cache_seq_len
    for epoch in range(s5["epochs"]):
        for i, batch in enumerate(batches):
            if device is not None:
                batch = batch.to(device)

            if teacher_logits_cache is not None:
                # Path B: pull this batch's per-layer teacher logits from
                # the precomputed cache. Cache is keyed token-by-token in
                # the same order iter_batches produces, so:
                #   batch i covers tokens [i*B*T, (i+1)*B*T)
                t_out = {}
                token_start = i * cache_tokens_per_batch
                token_end = token_start + (batch.shape[0] * batch.shape[1])
                for li, full_tensor in teacher_logits_cache["logits"].items():
                    t_out[li] = [
                        full_tensor[token_start:token_end].to(device=batch.device, dtype=torch.float32)
                    ]
                ref_layer_indices = list(t_out.keys())
            else:
                with torch.no_grad(), capture_router_outputs(teacher_refs) as t_out_ctx:
                    teacher(input_ids=batch)
                t_out = {li: list(v) for li, v in t_out_ctx.items()}
                ref_layer_indices = [r.layer_idx for r in teacher_refs]

            with capture_router_outputs(student_refs) as s_out:
                student(input_ids=batch)

            loss = torch.zeros((), device=batch.device, dtype=torch.float32)
            per_layer_loss: dict[int, float] = {}
            n_layers = 0
            for li in ref_layer_indices:
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
                layer_loss = F.kl_div(s_lp, t_p, reduction="batchmean") * (T ** 2)
                loss = loss + layer_loss
                per_layer_loss[li] = float(layer_loss.detach().item())
                n_layers += 1
            if n_layers == 0:
                raise RuntimeError(
                    "No router logits captured from either teacher or student. "
                    "Hooks may not be firing — check iter_moe_layers + router classes."
                )
            loss = loss / n_layers
            (loss / grad_accum).backward()

            if (i + 1) % grad_accum == 0:
                # Pre-step: compute gradient norm over trainable params.
                grad_sq = 0.0
                for p in student.parameters():
                    if p.requires_grad and p.grad is not None:
                        grad_sq += float(p.grad.detach().norm().item() ** 2)
                grad_norm = grad_sq ** 0.5
                optim.step()
                optim.zero_grad()
                step += 1
                if step % config["logging"]["log_every_n_steps"] == 0:
                    loss_val = float(loss.item())
                    # Highest- and lowest-loss layers tell us where the router
                    # is having the hardest time matching the teacher.
                    if per_layer_loss:
                        worst_li, worst_v = max(per_layer_loss.items(), key=lambda x: x[1])
                        best_li, best_v = min(per_layer_loss.items(), key=lambda x: x[1])
                    else:
                        worst_li = worst_v = best_li = best_v = 0
                    log.info(
                        "  epoch=%d step=%d loss=%.6f grad_norm=%.4f "
                        "(worst L%d=%.4f, best L%d=%.4f)",
                        epoch, step, loss_val, grad_norm,
                        worst_li, worst_v, best_li, best_v,
                    )
                    # Distribution stats over per-layer losses (worst/best
                    # already captured above). p50/p95 give shape; mean/std
                    # give scale. We do NOT log one Trackio series per layer
                    # — at ~40 layers × thousands of steps that floods the
                    # dashboard with low-signal series. Worst/best+stats is
                    # enough to spot a single layer drifting.
                    losses_sorted = sorted(per_layer_loss.values()) if per_layer_loss else [0.0]
                    n = len(losses_sorted)
                    p50 = losses_sorted[n // 2]
                    p95 = losses_sorted[min(n - 1, int(0.95 * n))]
                    mean_v = sum(losses_sorted) / n
                    payload = {
                        "stage5/epoch": epoch,
                        "stage5/step": step,
                        "stage5/loss": loss_val,
                        "stage5/grad_norm": grad_norm,
                        "stage5/worst_layer_idx": float(worst_li),
                        "stage5/worst_layer_loss": worst_v,
                        "stage5/best_layer_idx": float(best_li),
                        "stage5/best_layer_loss": best_v,
                        "stage5/layer_loss_p50": p50,
                        "stage5/layer_loss_p95": p95,
                        "stage5/layer_loss_mean": mean_v,
                    }
                    _trackio_log(payload)
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
