"""Logit-only forward KLD distillation — Minitron Chapter 1.

Loss: ``KLD(p_teacher || p_student)`` on logits at temperature τ = 1.0,
no cross-entropy, no intermediate-state, no embedding loss
(arxiv:2407.14679 §3 Eq. 2; Table 15 ablation; BP#7).

Optimizer:
  * **smoke tier (single-GPU, no DeepSpeed)**: ``bitsandbytes.optim.AdamW8bit``
  * **light tier (a100x4 + DeepSpeed ZeRO-3)**: ``deepspeed.ops.adam.DeepSpeedCPUAdam``

The bnb-8bit + ZeRO-3 combination is broken — bnb owns its optim.step which
fights DS's fp32 reduce. DeepSpeedCPUAdam offloads the FP32 master weights
+ Adam moments to host RAM (HF Jobs ``a100x4`` has 568 GB host RAM),
keeping per-GPU footprint to teacher (sharded) + student (sharded) +
student grads (sharded) + activations (grad ckpt'd).
"""
from __future__ import annotations

import logging
import math
import shutil
from itertools import islice
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loss (P1: per-token mean, not per-sample)
# ---------------------------------------------------------------------------


def forward_kld_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Forward KL: ``KLD(p_teacher || p_student)`` averaged per token.

    PyTorch's ``F.kl_div(reduction='batchmean')`` divides the sum by the FIRST
    dim of the input only. With logits shaped ``[B, T, V]`` that gives
    per-sample-summed-over-tokens — i.e. the loss is ~T× larger than the
    per-token mean Minitron's hyperparameters were tuned against. We reshape
    to ``[B*T, V]`` so ``batchmean`` divides by ``B*T``, the actual token count.

    Both inputs are upcast to fp32 before the softmax so the loss is
    numerically stable on bf16 logits — the underlying model can still run in bf16.
    """
    V = student_logits.shape[-1]
    s = student_logits.reshape(-1, V).float()
    t = teacher_logits.reshape(-1, V).float()
    s_log_p = F.log_softmax(s / temperature, dim=-1)
    t_p = F.softmax(t / temperature, dim=-1)
    return F.kl_div(s_log_p, t_p, reduction="batchmean") * (temperature ** 2)


# ---------------------------------------------------------------------------
# Trainable-scope toggling (P0: structural, not substring matching)
# ---------------------------------------------------------------------------


def enable_student_training(student: nn.Module, scope: str = "full") -> int:
    """Set ``requires_grad`` on the student per the configured scope.

    Uses ``moe_compress.utils.model_io.iter_moe_layers`` for structural
    traversal — substring matching on parameter names is fragile (e.g.
    ``shared_expert`` vs ``shared_experts``, ``mlp.gate`` vs ``mlp.gate_proj_*``).

    ``FactoredExperts`` (max_quality) initialises every U/V param with
    ``requires_grad=False`` because Stage 5 only trained the router. For
    Chapter 1 we usually want the whole model (Minitron baseline). The
    ``experts_only`` and ``factored_only`` scopes exist for the smoke tier
    on a single GPU and for future ablations.

    Returns the number of params that ended up trainable.
    """
    from moe_compress.utils.model_io import FactoredExperts, iter_moe_layers

    # First, freeze everything. We then selectively unfreeze.
    for p in student.parameters():
        p.requires_grad_(False)

    if scope == "full":
        for p in student.parameters():
            p.requires_grad_(True)

    elif scope == "experts_only":
        for ref in iter_moe_layers(student):
            # Routed experts module (FactoredExperts U/V banks OR fused stack).
            for p in ref.experts_module.parameters():
                p.requires_grad_(True)
            # Router (mlp.gate). Note: shared_expert is intentionally left frozen
            # — Strategy A protects shared experts from compression so they
            # already match the teacher.
            for p in ref.router.parameters():
                p.requires_grad_(True)

    elif scope == "factored_only":
        for ref in iter_moe_layers(student):
            em = ref.experts_module
            if not isinstance(em, FactoredExperts):
                # Layer wasn't factored (Stage 3 may have skipped some) — nothing
                # to unfreeze in factored_only mode.
                continue
            for attr in ("gate_proj_U", "gate_proj_V", "up_proj_U", "up_proj_V",
                         "down_proj_U", "down_proj_V"):
                p = getattr(em, attr, None)
                if isinstance(p, nn.Parameter):
                    p.requires_grad_(True)

    else:
        raise ValueError(
            f"Unknown trainable_scope: {scope!r}. "
            "Expected one of: full | experts_only | factored_only"
        )

    n = sum(p.numel() for p in student.parameters() if p.requires_grad)
    log.info("trainable_scope=%s -> %.3fB params trainable", scope, n / 1e9)
    return n


# ---------------------------------------------------------------------------
# Optimizer (P1: bnb-8bit OR DeepSpeedCPUAdam, switched by config)
# ---------------------------------------------------------------------------


def build_optimizer(student: nn.Module, dconf: dict[str, Any]) -> torch.optim.Optimizer:
    """Build the configured optimizer.

    Two options, chosen by ``dconf['optimizer']``:

      * ``adamw_bnb_8bit`` — bitsandbytes 8-bit AdamW. Use on single-GPU
        smoke runs (no DeepSpeed). Drop-in for fp32 AdamW with ~75%
        optimizer-state reduction.

      * ``deepspeed_cpu_adam`` — DeepSpeed's CPU-offloaded Adam. Use under
        DeepSpeed ZeRO-3 (Light tier on a100x4). Optimizer state lives in
        host RAM, GPU only sees the param + grad shards.
    """
    trainable = [p for p in student.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError(
            "build_optimizer: no trainable parameters. Did you forget "
            "enable_student_training()?"
        )
    name = dconf.get("optimizer", "adamw_bnb_8bit")
    lr = float(dconf["learning_rate"])
    betas = tuple(float(b) for b in dconf["betas"])
    wd = float(dconf["weight_decay"])

    if name == "adamw_bnb_8bit":
        import bitsandbytes as bnb
        return bnb.optim.AdamW8bit(trainable, lr=lr, betas=betas, weight_decay=wd)

    if name == "deepspeed_cpu_adam":
        # adamw_mode=True selects AdamW (decoupled weight decay), matching bnb.
        from deepspeed.ops.adam import DeepSpeedCPUAdam
        return DeepSpeedCPUAdam(trainable, lr=lr, betas=betas, weight_decay=wd,
                                adamw_mode=True)

    raise ValueError(
        f"Unknown optimizer: {name!r}. "
        "Expected 'adamw_bnb_8bit' or 'deepspeed_cpu_adam'."
    )


def cosine_with_warmup(step: int, *, warmup_steps: int, total_steps: int,
                       lr_max: float, lr_min: float) -> float:
    """Linear warmup [step 0..warmup-1] → cosine decay [warmup..total_steps-1].

    Convention: ``step`` is the 0-based index of the optimizer step about to
    be taken (called BEFORE optim.step). At ``step = total_steps - 1`` the
    cosine returns ~lr_min (numerically the cosine never hits exactly -1
    unless total_steps == warmup + 1, but the floor is 1e-5 below lr_min).
    """
    if step < warmup_steps:
        return lr_max * (step + 1) / max(1, warmup_steps)
    if step >= total_steps:
        return lr_min
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * progress))


def set_lr(optim: torch.optim.Optimizer, lr: float) -> None:
    for g in optim.param_groups:
        g["lr"] = lr


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def _is_deepspeed(accelerator) -> bool:
    """True if the active Accelerator was initialised with DeepSpeed."""
    from accelerate.utils import DistributedType
    return accelerator.distributed_type == DistributedType.DEEPSPEED


def _is_zero3(accelerator) -> bool:
    if not _is_deepspeed(accelerator):
        return False
    plugin = getattr(accelerator.state, "deepspeed_plugin", None)
    if plugin is None:
        return False
    return int(plugin.zero_stage) >= 3


def _all_finite(loss: torch.Tensor, accelerator) -> bool:
    """Collective NaN/Inf check. Returns True iff loss is finite on EVERY rank.

    Critical under DeepSpeed: a rank-local skip would mismatch reductions on
    the next backward and hang NCCL. We collectively detect → collectively skip.
    """
    flag = torch.tensor(
        1.0 if torch.isfinite(loss).all() else 0.0,
        device=accelerator.device,
    )
    if accelerator.num_processes > 1:
        # MIN reduces 1.0/0.0 to 0.0 if ANY rank had a non-finite loss.
        flag = accelerator.reduce(flag, reduction="min")
    return bool(flag.item() >= 0.5)


def _shard_batches_per_rank(batches: list, accelerator) -> list:
    """Strided slice so each rank consumes a disjoint subset of batches.

    Under DeepSpeed ZeRO-3 the params are model-sharded across ranks, but the
    DATA must still be data-parallel for any compute speedup. Without this
    slice, every rank would compute the same loss on the same tokens —
    correctness-OK but ~world× wasteful in wall-clock.
    """
    pi = accelerator.process_index
    np = accelerator.num_processes
    if np <= 1:
        return list(batches)
    return list(batches[pi::np])


def run_distillation(
    teacher: nn.Module,
    student: nn.Module,
    tokenizer,
    config: dict[str, Any],
    artifacts_dir: Path,
    accelerator,
) -> Path:
    """One Chapter 1 distillation pass. Saves to ``artifacts_dir/chapter1_recovered``.

    Pre-conditions:
      * ``teacher`` is frozen (requires_grad=False, eval mode).
        - Under DeepSpeed: already ZeRO-3-sharded via HfDeepSpeedConfig
          context applied during load (see ``run_recovery._load_teacher``).
          Do NOT pass to ``accelerator.prepare`` — DeepSpeed only allows one
          engine per Accelerator.
        - Without DeepSpeed: already ``.to(accelerator.device)``.
      * ``student.requires_grad`` is set per ``trainable_scope``.
    """
    from moe_compress.utils.calibration import (
        build_calibration_tensor, iter_batches, spec_from_config,
    )

    from . import eval_quick

    dconf = config["distillation"]
    is_ds = _is_deepspeed(accelerator)

    # Gradient checkpointing must be enabled BEFORE accelerator.prepare so
    # the engine wraps a checkpointed forward. use_reentrant=False is the
    # supported path for accelerate + deepspeed.
    if dconf["use_gradient_checkpointing"]:
        student.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    # Calibration: reuse the SAME on-disk cache produced by max_quality (same
    # subset weights → same hash). seed_offset=7 differs from any max_quality
    # stage so the rows we pull are disjoint from those used by Stages 0/2/3/5.
    # Build the FULL shared tensor once (cached on disk by content hash),
    # then strided-slice per rank for data parallelism.
    spec = spec_from_config(config["calibration"], seed_offset=7)
    calib = build_calibration_tensor(
        tokenizer, spec, cache_dir=artifacts_dir / "_calibration_cache",
    )
    all_batches = iter_batches(calib, batch_size=int(dconf["per_device_batch_size"]))
    # Truncate to a multiple of (world × grad_accum) BEFORE per-rank shard so
    # every rank gets identical local batch counts and the loop terminates
    # in lock-step. Without this, ranks with floor(N/world) batches finish
    # one micro-batch ahead of ranks with ceil(N/world) → NCCL hang.
    world_pre = max(1, accelerator.num_processes)
    grad_accum_pre = int(dconf["gradient_accumulation"])
    truncate_to = (len(all_batches) // (world_pre * grad_accum_pre)) * (world_pre * grad_accum_pre)
    if truncate_to < len(all_batches):
        log.info("Truncating calibration: %d -> %d batches (multiple of %d×%d)",
                 len(all_batches), truncate_to, world_pre, grad_accum_pre)
        all_batches = all_batches[:truncate_to]
    batches = _shard_batches_per_rank(all_batches, accelerator)
    log.info("rank %d/%d sees %d local batches (of %d global)",
             accelerator.process_index, accelerator.num_processes,
             len(batches), len(all_batches))

    # Schedule
    grad_accum = int(dconf["gradient_accumulation"])
    seq_len = int(dconf["sequence_length"])
    micro_bsz = int(dconf["per_device_batch_size"])
    world = max(1, accelerator.num_processes)
    tokens_per_step = micro_bsz * world * grad_accum * seq_len
    total_steps = max(1, int(dconf["total_tokens"]) // tokens_per_step)
    warmup = int(dconf["warmup_steps"])
    lr_max = float(dconf["learning_rate"])
    lr_min = float(dconf["min_learning_rate"])
    if warmup >= total_steps:
        raise ValueError(
            f"warmup_steps={warmup} must be < total_steps={total_steps}. "
            "Increase total_tokens or reduce warmup_steps."
        )
    if total_steps < 50:
        log.warning(
            "total_steps=%d is small (<50); cosine decay has limited headroom. "
            "OK for smoke runs, undersized for real recovery.",
            total_steps,
        )
    log.info(
        "schedule: %d total steps (tokens_per_step=%d, world=%d, grad_accum=%d, seq=%d)",
        total_steps, tokens_per_step, world, grad_accum, seq_len,
    )

    # Calibration shortage check (P1: warn explicitly if too short).
    # Each rank consumes `total_steps * grad_accum` of its local batches.
    needed_micro_local = total_steps * grad_accum
    if len(batches) < needed_micro_local:
        log.warning(
            "rank %d: calibration is short — need %d local micro-batches but "
            "have %d. Training will exit early — only %d optim steps will run "
            "(out of %d planned). Bump calibration.num_sequences in YAML.",
            accelerator.process_index, needed_micro_local, len(batches),
            len(batches) // grad_accum, total_steps,
        )

    # Optimizer: built BEFORE accelerator.prepare. DS prepare swaps it
    # into a wrapped engine optimizer. bnb/CPUAdam don't compose with each
    # other so we never mix them.
    optim = build_optimizer(student, dconf)

    # Prepare ONLY the student (and its optimizer). The teacher is already
    # placed correctly by run_recovery._load_teacher. DeepSpeed only allows
    # one engine per Accelerator.
    student, optim = accelerator.prepare(student, optim)

    student.train()
    nan_guard = bool(dconf.get("nan_loss_circuit_breaker", True))
    log_every = int(dconf["log_every_n_steps"])
    eval_every = int(dconf["eval_every_n_steps"])
    save_every = int(dconf["save_every_n_steps"])
    temperature = float(dconf["temperature"])
    grad_clip = float(dconf["grad_clip_norm"])

    optim.zero_grad(set_to_none=True)
    step = 0
    micro_in_window = 0
    last_loss: float | None = None

    # tokens_consumed counts tokens passed through forward (incl. NaN-skipped).
    # tokens_with_grad counts tokens that contributed to a successful optim.step.
    tokens_consumed = 0
    tokens_with_grad = 0

    for batch in islice(batches, needed_micro_local):
        ids = batch.to(accelerator.device, non_blocking=True)

        with torch.no_grad():
            t_logits = teacher(input_ids=ids).logits

        s_logits = student(input_ids=ids).logits
        loss = forward_kld_loss(s_logits, t_logits, temperature=temperature)

        # Each rank's batch is disjoint (data parallel), so total tokens this
        # micro-batch across the world = ids.numel() * world.
        tokens_consumed += int(ids.numel()) * world

        # Collective NaN check. Under DeepSpeed we MUST keep the engine's
        # micro-batch counter aligned (it tracks reduce-scatter timing).
        # Instead of skipping (which corrupts the comm pattern), substitute a
        # clean zero loss that's graph-connected via the student forward, so
        # backward propagates zero gradients and the engine counter advances.
        # NOTE: ``loss * 0.0`` would be NaN * 0 = NaN — won't work. Build a
        # fresh zero loss from a NaN-cleaned copy of s_logits.
        finite = _all_finite(loss, accelerator)
        if nan_guard and not finite:
            log.warning("step=%d micro=%d non-finite loss; substituting zero.",
                        step, micro_in_window)
            if is_ds:
                cleaned = torch.nan_to_num(s_logits, nan=0.0, posinf=0.0, neginf=0.0)
                loss = cleaned.sum() * 0.0
            else:
                optim.zero_grad(set_to_none=True)
                micro_in_window = 0
                del t_logits, s_logits
                continue

        accelerator.backward(loss / grad_accum)
        last_loss = float(loss.detach().item())
        # Free per-batch tensors aggressively now that backward is queued.
        del t_logits, s_logits
        micro_in_window += 1

        if micro_in_window % grad_accum == 0:
            # Under DeepSpeed, the engine owns gradient clipping (config:
            # gradient_clipping: "auto"). Don't double-apply.
            if not is_ds and grad_clip and grad_clip > 0:
                accelerator.clip_grad_norm_(
                    [p for p in student.parameters() if p.requires_grad],
                    grad_clip,
                )

            set_lr(optim, cosine_with_warmup(
                step, warmup_steps=warmup, total_steps=total_steps,
                lr_max=lr_max, lr_min=lr_min,
            ))
            optim.step()
            optim.zero_grad(set_to_none=True)
            tokens_with_grad += grad_accum * micro_bsz * world * seq_len
            step += 1
            micro_in_window = 0

            if accelerator.is_main_process and step % log_every == 0:
                log.info("step=%d/%d lr=%.3e loss=%.6f tok=%.2fB/%.2fB",
                         step, total_steps, optim.param_groups[0]["lr"],
                         last_loss, tokens_with_grad / 1e9,
                         int(dconf["total_tokens"]) / 1e9)

            if eval_every > 0 and step % eval_every == 0:
                # COLLECTIVE: every rank participates in the forward.
                eval_quick.run(student, tokenizer, config, accelerator)

            if save_every > 0 and step % save_every == 0:
                _save(student, tokenizer, config, artifacts_dir, accelerator,
                      step=step, tokens_with_grad=tokens_with_grad,
                      tokens_consumed=tokens_consumed, partial=True)

            if step >= total_steps:
                break

    return _save(student, tokenizer, config, artifacts_dir, accelerator,
                 step=step, tokens_with_grad=tokens_with_grad,
                 tokens_consumed=tokens_consumed, partial=False)


# ---------------------------------------------------------------------------
# Save (P1: gather under ZeRO-3; rolling partial dirs)
# ---------------------------------------------------------------------------


def _save(student, tokenizer, config, artifacts_dir, accelerator,
          *, step: int, tokens_with_grad: int, tokens_consumed: int,
          partial: bool) -> Path:
    """Save the recovered checkpoint, mirroring max_quality's
    ``save_compressed_checkpoint`` layout (sharded safetensors + tokenizer +
    ``compressed_metadata.json``).

    Under DeepSpeed ZeRO-3 we cannot use ``GatheredParameters`` — gathering
    a 70 GB student on every rank exceeds the 80 GB GPU budget. Instead we
    use ``accelerator.get_state_dict(student)`` which uses DS's streamed
    consolidation hook (``_zero3_consolidated_16bit_state_dict``): the gather
    runs on rank 0 INTO CPU memory only, leaving GPU footprint unchanged.
    Other ranks return an empty dict.

    Partial saves go to ``chapter1_recovered_partial_step{N}/`` and old ones
    are pruned (keep last 2) — cheap insurance against a kill mid-write.
    """
    import json

    from moe_compress.utils.model_io import (
        COMPRESSED_METADATA_FILENAME, FactoredExperts, iter_moe_layers,
    )

    accelerator.wait_for_everyone()

    if partial:
        out_dir = artifacts_dir / f"chapter1_recovered_partial_step{step}"
    else:
        out_dir = artifacts_dir / "chapter1_recovered"

    # 1. Build metadata from architecture (cheap; works on sharded model
    #    because num_experts/ranks are stored as Python ints, not tensor
    #    shapes).
    unwrapped = accelerator.unwrap_model(student)
    per_layer_num_experts: dict[str, int] = {}
    factored_layers: list[int] = []
    factored_ranks: dict[str, dict[str, int]] = {}
    for ref in iter_moe_layers(unwrapped):
        per_layer_num_experts[str(ref.layer_idx)] = ref.num_routed_experts
        if isinstance(ref.experts_module, FactoredExperts):
            factored_layers.append(ref.layer_idx)
            factored_ranks[str(ref.layer_idx)] = dict(ref.experts_module.ranks)
    metadata = {
        "version": 1,
        "pipeline_stage": "chapter1_recovered",
        "per_layer_num_experts": per_layer_num_experts,
        "factored_layers": sorted(factored_layers),
        "factored_ranks": factored_ranks,
        "extra": {
            "recovery_method": "minitron_logit_kld",
            "optimizer": config["distillation"]["optimizer"],
            "teacher_model": config["teacher"]["name_or_path"],
            "student_source": config["student"]["source"],
            "trainable_scope": config["distillation"]["trainable_scope"],
            "tokens_with_grad": tokens_with_grad,
            "tokens_consumed": tokens_consumed,
            "step": step,
            "partial": partial,
        },
    }

    # 2. Get the consolidated state dict. COLLECTIVE under DS3.
    #    Returns CPU-resident dict on rank 0, empty dict on others.
    state_dict = accelerator.get_state_dict(student)

    # 3. Rank 0 writes everything.
    if accelerator.is_main_process:
        out_dir.mkdir(parents=True, exist_ok=True)
        unwrapped.save_pretrained(
            out_dir, state_dict=state_dict, safe_serialization=True,
        )
        if tokenizer is not None:
            tokenizer.save_pretrained(out_dir)
        (out_dir / COMPRESSED_METADATA_FILENAME).write_text(
            json.dumps(metadata, indent=2)
        )
        log.info("Saved %s checkpoint to %s (step=%d, tokens_with_grad=%.2fB)",
                 "PARTIAL" if partial else "FINAL", out_dir, step,
                 tokens_with_grad / 1e9)
        if partial:
            _prune_old_partials(artifacts_dir, keep=2)

    accelerator.wait_for_everyone()
    return out_dir


def _prune_old_partials(artifacts_dir: Path, *, keep: int = 2) -> None:
    """Delete all but the K most-recent ``chapter1_recovered_partial_stepN/`` dirs."""
    pattern = "chapter1_recovered_partial_step*"
    dirs: list[tuple[int, Path]] = []
    for p in artifacts_dir.glob(pattern):
        if not p.is_dir():
            continue
        try:
            n = int(p.name.split("step")[-1])
        except ValueError:
            continue
        dirs.append((n, p))
    dirs.sort()  # ascending step
    for _, p in dirs[:-keep]:
        try:
            shutil.rmtree(p)
            log.info("Pruned old partial: %s", p)
        except OSError as err:
            log.warning("Failed to prune %s: %s", p, err)
