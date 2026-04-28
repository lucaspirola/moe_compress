"""Stage 5 — Router KD via vocabulary-level output distillation.

Reference: 2603.02217 (Router Knowledge Distillation), Eq. 3.

The paper distills at the **vocabulary output logit** level, NOT at the
intermediate router gate level. From §4:

  "By distilling output logits rather than matching router gate values
   explicitly, Router KD avoids requiring the teacher and student to share
   identical expert sets or gate dimensionalities."

This means the loss is:

  L_RKD = (τ²/N_x) Σ_t  KL(softmax(z_T^t / τ) ‖ softmax(z_S^t / τ))

where z_T, z_S ∈ ℝ^|V| are the teacher/student vocabulary logits for
next-token prediction, and the sum is over unmasked token positions.

Only router weights are trainable; all expert weights are frozen. The
vocabulary-level signal propagates gradients through the full forward pass
including the routing decisions, which naturally adapts the router to the
compressed expert set.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

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

    # NOTE: merge_map is no longer used by vocabulary-level KD (it was needed
    # only for the old router-level logsumexp pooling). Kept for backward
    # compatibility with checkpoint metadata that references it.
    # merge_map = load_json_artifact(artifacts_dir / "stage2_pruned" / "merge_map.json")

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
    ckpt_every = int(s5.get("checkpoint_every_n_steps", 0))

    # Teacher/student MoE layer count sanity check (router structure must match
    # even though we're distilling at vocab level — the student's routers are
    # what we're training).
    student_refs = list(iter_moe_layers(student))

    if teacher is not None:
        teacher_refs = list(iter_moe_layers(teacher))
        assert len(teacher_refs) == len(student_refs), \
            f"Teacher/student MoE layer count mismatch: {len(teacher_refs)} vs {len(student_refs)}"

    # -----------------------------------------------------------------------
    # Crash-resume: find latest checkpoint and restore router + optim state.
    # -----------------------------------------------------------------------
    partial_dir = artifacts_dir / "_stage5_partial"
    partial_dir.mkdir(parents=True, exist_ok=True)
    resume_step = 0
    resume_epoch = 0
    resume_batch_i = -1

    ckpts = sorted(
        partial_dir.glob("step_*.pt"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    if ckpts:
        latest = ckpts[-1]
        try:
            payload = torch.load(latest, map_location="cpu")
        except Exception as exc:
            raise RuntimeError(
                f"Stage 5 resume: failed to load checkpoint {latest}: {exc}"
            ) from exc
        fv = int(payload.get("format_version", 0))
        if fv != 1:
            raise RuntimeError(
                f"Stage 5 checkpoint {latest} has format_version={fv} "
                "(expected 1) — delete _stage5_partial/ and re-run Stage 5"
            )
        # Restore router parameters into the student model.
        for pname, t in payload["router_state"].items():
            parts = pname.split(".")
            obj = student
            for part in parts[:-1]:
                obj = getattr(obj, part)
            getattr(obj, parts[-1]).data.copy_(t)
        optim.load_state_dict(payload["optim_state"])
        if device is not None:
            _move_optimizer_state_to_device(optim, device)
        resume_step = int(payload["step"])
        resume_epoch = int(payload["epoch"])
        resume_batch_i = int(payload["batch_idx"])
        log.info("Stage 5: resumed from step %d (epoch %d, batch %d)",
                 resume_step, resume_epoch, resume_batch_i)

    student.train()
    total_steps = (len(batches) // grad_accum) * s5["epochs"]
    log.info("Stage 5: %d routers trainable; %d steps (grad-accum=%d)",
             sum(1 for _ in student.parameters() if _.requires_grad),
             total_steps, grad_accum)

    step = resume_step
    optim.zero_grad()
    for epoch in range(s5["epochs"]):
        if epoch < resume_epoch:
            continue
        for i, batch in enumerate(batches):
            # Fast-forward: skip batches already processed in the resumed run.
            if epoch == resume_epoch and i <= resume_batch_i:
                continue

            if device is not None:
                batch = batch.to(device)

            # --- Vocabulary-level KD (paper 2603.02217, Eq. 3) ---
            # Teacher: full forward pass, get vocabulary logits z_T ∈ ℝ^{B×L×|V|}
            if teacher_logits_cache is not None:
                # Path B: precomputed teacher vocab logits.
                cache_seq_len = int(s5["max_sequence_length"])
                cache_batch_size = int(s5["batch_size"])
                cache_tokens_per_batch = cache_batch_size * cache_seq_len
                token_start = i * cache_tokens_per_batch
                token_end = token_start + (batch.shape[0] * batch.shape[1])
                teacher_vocab_logits = teacher_logits_cache["vocab_logits"][token_start:token_end]
                teacher_vocab_logits = teacher_vocab_logits.to(device=batch.device, dtype=torch.float32)
                teacher_vocab_logits = teacher_vocab_logits.view(batch.shape[0], batch.shape[1], -1)
            else:
                with torch.no_grad():
                    teacher_out = teacher(input_ids=batch)
                teacher_vocab_logits = teacher_out.logits.to(torch.float32)  # [B, L, |V|]

            # Student: full forward pass with gradients (routers are trainable).
            student_out = student(input_ids=batch)
            student_vocab_logits = student_out.logits.to(torch.float32)  # [B, L, |V|]

            # KL(teacher ‖ student) over vocabulary, per-token, scaled by τ².
            # Paper Eq. 3: L_RKD = (τ²/N_x) Σ_t KL(p_T^t ‖ p_S^t)
            # Shift logits: predict token t+1 from position t (standard causal LM).
            # Use positions [0, L-2] to predict [1, L-1].
            t_logits_shift = teacher_vocab_logits[:, :-1, :]   # [B, L-1, |V|]
            s_logits_shift = student_vocab_logits[:, :-1, :]   # [B, L-1, |V|]

            t_p = F.softmax(t_logits_shift / T, dim=-1)
            s_lp = F.log_softmax(s_logits_shift / T, dim=-1)

            # Per-token KL, then mean over all (non-pad) tokens.
            per_token_kl = F.kl_div(s_lp, t_p, reduction="none").sum(dim=-1)  # [B, L-1]
            n_tokens = per_token_kl.numel()
            loss = (per_token_kl.sum() / max(n_tokens, 1)) * (T ** 2)

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
                    log.info(
                        "  epoch=%d step=%d loss=%.6f grad_norm=%.4f",
                        epoch, step, loss_val, grad_norm,
                    )
                    payload = {
                        "stage5/epoch": epoch,
                        "stage5/step": step,
                        "stage5/loss": loss_val,
                        "stage5/grad_norm": grad_norm,
                    }
                    _trackio_log(payload)

                # Periodic checkpoint for crash-resume.
                if ckpt_every > 0 and step % ckpt_every == 0:
                    _save_stage5_checkpoint(partial_dir, step, epoch, i, student, optim)
                    # Keep only the two most recent checkpoints to bound disk use.
                    old_step = step - 2 * ckpt_every
                    if old_step > 0:
                        old_ckpt = partial_dir / f"step_{old_step}.pt"
                        if old_ckpt.exists():
                            old_ckpt.unlink()
        optim.zero_grad()

    out_dir = artifacts_dir / "stage5_final"
    save_compressed_checkpoint(
        student, tokenizer, out_dir, pipeline_stage="stage5_final",
    )
    # Keep _stage5_partial/ on success: checkpoints are useful for debugging
    # convergence and post-mortem analysis. The directory is small (≤ 2 × few MB).
    log.info("Stage 5 complete → %s", out_dir)
    return out_dir


def _save_stage5_checkpoint(
    partial_dir: Path,
    step: int,
    epoch: int,
    batch_idx: int,
    student: nn.Module,
    optim: torch.optim.Optimizer,
) -> None:
    router_state = {
        name: p.data.cpu().clone()
        for name, p in student.named_parameters()
        if p.requires_grad
    }
    payload = {
        "format_version": 1,
        "step": step,
        "epoch": epoch,
        "batch_idx": batch_idx,
        "router_state": router_state,
        "optim_state": optim.state_dict(),
    }
    tmp = partial_dir / f"step_{step}.pt.tmp"
    final = partial_dir / f"step_{step}.pt"
    torch.save(payload, tmp)
    os.replace(tmp, final)
    log.info("Stage 5: checkpoint saved at step %d (epoch %d, batch %d)", step, epoch, batch_idx)


def _move_optimizer_state_to_device(optim: torch.optim.Optimizer, device) -> None:
    """Move all optimizer state tensors to the target device.

    Required after load_state_dict() when the checkpoint was saved on CPU
    but the training params live on a CUDA device — otherwise the first
    optimizer step silently mixes CPU and CUDA tensors.
    """
    for state in optim.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)


def _freeze_non_routers(model: nn.Module, trainable_patterns: list[str]) -> None:
    for name, p in model.named_parameters():
        p.requires_grad_(any(pat in name for pat in trainable_patterns))



