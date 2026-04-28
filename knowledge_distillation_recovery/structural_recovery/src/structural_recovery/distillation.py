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

import json
import logging
import math
import os
import shutil
from itertools import islice
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loss (P1: per-token mean, not per-sample)
# ---------------------------------------------------------------------------


# Module-level cache so we don't construct LogitsDistillationLoss per micro-batch.
_KLD_LOSS_CACHE: dict[float, Any] = {}


def _get_kld_loss_fn(temperature: float):
    fn = _KLD_LOSS_CACHE.get(temperature)
    if fn is None:
        from modelopt.torch.distill.losses import LogitsDistillationLoss
        fn = LogitsDistillationLoss(temperature=temperature, reduction="batchmean")
        _KLD_LOSS_CACHE[temperature] = fn
    return fn


def forward_kld_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Forward KL: ``KLD(p_teacher || p_student)`` averaged per token.

    Delegates to ``modelopt.torch.distill.LogitsDistillationLoss`` — the loss
    class QUALITY_RECOVERY_GUIDE.md §1.8.2 names as canonical.

    Reduction is ``"batchmean"`` after reshaping to ``[B*T, V]`` so PyTorch
    divides by the token count (``B*T``), giving per-token mean. ModelOpt's
    default ``"mean"`` would divide by ``B*T*V`` (an extra factor of vocab
    size, ~150k) and silently collapse the gradient signal.

    Both inputs are upcast to fp32 before the softmax so the loss is
    numerically stable on bf16 logits — the underlying model can still run in bf16.

    Pad/mask contract: the calibration tensor produced by
    ``moe_compress.utils.calibration._tokenize_to_fixed_length`` is fully
    packed (concatenated streams separated by EOS, hard 5%-shortage cap).
    Every position is a real token, so per-position averaging is correct
    without an attention_mask. If a future call site feeds pad-bearing
    sequences this contract is violated — assert at the boundary there.
    """
    if student_logits.shape[-1] != teacher_logits.shape[-1]:
        raise ValueError(
            f"forward_kld_loss: vocab mismatch — student V={student_logits.shape[-1]} "
            f"vs teacher V={teacher_logits.shape[-1]}. Same-tokenizer distillation "
            "is required (Strategy A does not change vocabulary)."
        )
    V = student_logits.shape[-1]
    s = student_logits.reshape(-1, V).float()
    t = teacher_logits.reshape(-1, V).float()
    return _get_kld_loss_fn(temperature)(s, t)


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


# Module-scope holder so HfDeepSpeedConfig instances survive function exits.
# Keyed by ``id(deepspeed_config)`` so a second call with a *different*
# config (e.g. teacher correction → KD in-process, or unit tests) re-runs
# activation. With the same config the activation is a no-op.
_DSCHF_HOLDER: dict[int, Any] = {}


def _activate_zero3_init(accelerator) -> None:
    """Pin a module-scope HfDeepSpeedConfig so from_pretrained shards params
    via deepspeed.zero.Init. No-op if not under ZeRO-3 or already activated
    for this exact ds_config object."""
    if not _is_zero3(accelerator):
        return
    from transformers.integrations import HfDeepSpeedConfig, is_deepspeed_zero3_enabled
    plugin = accelerator.state.deepspeed_plugin
    ds_config = plugin.deepspeed_config
    cfg_key = id(ds_config)
    if cfg_key in _DSCHF_HOLDER:
        return
    _DSCHF_HOLDER[cfg_key] = HfDeepSpeedConfig(ds_config)
    if not is_deepspeed_zero3_enabled():
        try:
            import deepspeed  # noqa: F401
            ds_avail = True
        except ImportError:
            ds_avail = False
        raise RuntimeError(
            "HfDeepSpeedConfig was instantiated but is_deepspeed_zero3_enabled() "
            "returned False — the model would load full-rank on each rank and OOM. "
            f"plugin.zero_stage={getattr(plugin, 'zero_stage', '?')}, "
            f"deepspeed_importable={ds_avail}, "
            f"ds_config['zero_optimization']['stage']="
            f"{ds_config.get('zero_optimization', {}).get('stage', '?')}."
        )
    log.info("HfDeepSpeedConfig activated for ZeRO-3 sharded from_pretrained.")


def _all_finite(loss: torch.Tensor, accelerator) -> bool:
    """Collective NaN/Inf check. Returns True iff loss is finite on EVERY rank.

    Critical under DeepSpeed: a rank-local skip would mismatch reductions on
    the next backward and hang NCCL. We collectively detect → collectively skip.

    Item 8: when the answer is False, also surface WHICH rank(s) reported
    non-finite. Useful for diagnosing single-GPU ECC errors that look like
    "training-only" NaN — if the same rank ID keeps appearing, suspect that
    GPU's hardware.
    """
    is_finite_local = 1.0 if torch.isfinite(loss).all() else 0.0
    flag = torch.tensor(is_finite_local, device=accelerator.device)
    if accelerator.num_processes > 1:
        # gather first → log per-rank; then min → collective bool
        all_flags = accelerator.gather(flag.unsqueeze(0))      # shape [world]
        if accelerator.is_main_process and (all_flags < 0.5).any():
            bad = (all_flags < 0.5).nonzero(as_tuple=True)[0].tolist()
            log.warning(
                "non-finite loss on ranks %s — investigate hardware if recurrent "
                "on the same rank.", bad,
            )
        flag = all_flags.min()
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


def _read_resume_token_counters(artifacts_dir: Path) -> dict[str, int]:
    """Read cumulative token counters from the latest valid partial's metadata.

    Returns dict with keys ``tokens_consumed``, ``tokens_with_grad``,
    ``tokens_skipped_nan``, ``tokens_dropped_window`` (defaults to 0). Used
    to seed counters on resume so the saved metadata reflects cumulative
    totals across resume segments, not just the current run.
    """
    out = {
        "tokens_consumed": 0,
        "tokens_with_grad": 0,
        "tokens_skipped_nan": 0,
        "tokens_dropped_window": 0,
    }
    partial, _ = _load_latest_partial(artifacts_dir)
    if partial is None:
        return out
    try:
        meta = json.loads((partial / "compressed_metadata.json").read_text())
        extra = meta.get("extra", {})
        for k in out:
            v = extra.get(k)
            if isinstance(v, int) and v >= 0:
                out[k] = v
    except (OSError, json.JSONDecodeError, KeyError):
        # Corrupt metadata: counters stay at 0; resume_step still advances
        # because that came from the dir name. Saved totals will under-
        # report — log so the operator notices.
        log.warning("Could not read resume token counters from %s; "
                    "starting at 0 — saved totals will under-report.",
                    partial)
    return out


def run_distillation(
    teacher: nn.Module,
    student: nn.Module,
    tokenizer,
    config: dict[str, Any],
    artifacts_dir: Path,
    accelerator,
    resume_step: int = 0,
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
    config_path = config.get("_source_path", "<config>")
    if total_steps < 50:
        # Item 4: actionable warning.
        tokens_for_50_steps = 50 * tokens_per_step
        log.warning(
            "total_steps=%d is small (<50) — cosine decay has limited headroom. "
            "For meaningful recovery, increase `distillation.total_tokens` to "
            "at least %d (currently %d) in %s.",
            total_steps, tokens_for_50_steps, int(dconf["total_tokens"]),
            config_path,
        )
    log.info(
        "schedule: %d total steps (tokens_per_step=%d, world=%d, grad_accum=%d, seq=%d)",
        total_steps, tokens_per_step, world, grad_accum, seq_len,
    )

    # Calibration shortage check — each rank consumes (total_steps - resume_step)
    # * grad_accum of its local batches in the *remaining* run. We compute the
    # full requirement first so the warning quotes the absolute number a fresh
    # run would need; we slice ``batches`` for resume right below.
    needed_micro_local = total_steps * grad_accum
    if len(batches) < needed_micro_local:
        # Item 3: actionable warning. Compute exactly how many sequences the
        # YAML must request to cover the budget on the current world size.
        current_n = int(config["calibration"]["num_sequences"])
        required_n = total_steps * grad_accum * world * micro_bsz
        log.warning(
            "rank %d: calibration is short — need %d local micro-batches but "
            "have %d. Training will exit early — only %d optim steps will run "
            "(out of %d planned). To run all planned steps, bump "
            "`calibration.num_sequences` from %d to >= %d in %s.",
            accelerator.process_index, needed_micro_local, len(batches),
            len(batches) // grad_accum, total_steps,
            current_n, required_n, config_path,
        )

    # Resume: skip microbatches already consumed in a prior run. Calibration
    # batches are deterministic from the fixed cache, so slicing the list is
    # equivalent to replaying and discarding the consumed prefix.
    if resume_step > 0:
        skip_micros = resume_step * grad_accum
        if skip_micros < len(batches):
            batches = batches[skip_micros:]
            log.info("Resuming from step=%d: skipped %d local micro-batches.",
                     resume_step, skip_micros)
        else:
            log.warning(
                "resume_step=%d would skip all %d local batches; starting from scratch.",
                resume_step, len(batches),
            )
            resume_step = 0

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

    # M3: assert grad-clip matches the DS engine config (they're specified
    # separately and can silently diverge if one is edited without the other).
    if is_ds:
        try:
            ds_cfg = accelerator.state.deepspeed_plugin.deepspeed_config
            ds_clip = ds_cfg.get("gradient_clipping")
            if ds_clip is not None and abs(float(ds_clip) - grad_clip) > 1e-6:
                raise RuntimeError(
                    f"grad_clip_norm={grad_clip} in YAML ≠ "
                    f"gradient_clipping={ds_clip} in DS config — "
                    "update ds_configs/zero3_offload_optim.json to match."
                )
        except (AttributeError, KeyError):
            pass

    optim.zero_grad(set_to_none=True)
    step = resume_step
    micro_in_window = 0
    last_real_loss: float | None = None

    # Token accounting — explicit four-way split so the bookkeeping invariant
    # holds at any exit point:
    #   tokens_consumed       — every micro-batch's forward (incl. substituted)
    #   tokens_with_grad      — micros in windows that committed an optim.step
    #   tokens_skipped_nan    — micros whose loss was substituted to zero (DS path)
    #   tokens_dropped_window — micros in windows abandoned by the non-DS NaN
    #                           branch OR left over in a partial trailing window
    # Invariant: consumed = with_grad + skipped_nan + dropped_window
    #
    # Seed from the resume partial's metadata so saved totals are CUMULATIVE
    # across resume segments, not per-segment.
    _resume_counters = _read_resume_token_counters(artifacts_dir) if resume_step > 0 else {
        "tokens_consumed": 0, "tokens_with_grad": 0,
        "tokens_skipped_nan": 0, "tokens_dropped_window": 0,
    }
    tokens_consumed = _resume_counters["tokens_consumed"]
    tokens_with_grad = _resume_counters["tokens_with_grad"]
    tokens_skipped_nan = _resume_counters["tokens_skipped_nan"]
    tokens_dropped_window = _resume_counters["tokens_dropped_window"]
    pending_window_tokens = 0  # credited to with_grad on commit, dropped on abandon

    # NaN escalation state (item 2). consecutive_nan_windows increments at the
    # END of each window (every grad_accum micros) where ANY micro substituted;
    # resets on a clean window.
    nan_threshold = int(dconf.get("consecutive_nan_threshold", 5))
    consecutive_nan_windows = 0
    nan_in_current_window = False
    nan_diagnostic_emitted = False  # only dump per-layer JSON once per run

    # After resume, ``batches`` was already trimmed by ``skip_micros``. We
    # iterate the *remaining* micros via a manual iterator so a non-DS NaN
    # skip (which doesn't advance ``step``) doesn't deplete the budget — and
    # ``islice`` won't mis-cap the count after resume.
    remaining_micros = max(0, (total_steps - step) * grad_accum)
    if remaining_micros > len(batches):
        log.warning(
            "rank %d: only %d local micros available but %d are needed to "
            "reach step=%d. Run will exit early.",
            accelerator.process_index, len(batches), remaining_micros, total_steps,
        )
    batch_iter = iter(batches)
    consumed_micros = 0

    for batch in batch_iter:
        if consumed_micros >= remaining_micros:
            break
        consumed_micros += 1
        ids = batch.to(accelerator.device, non_blocking=True)

        with torch.no_grad():
            t_logits = teacher(input_ids=ids).logits

        s_logits = student(input_ids=ids).logits
        loss = forward_kld_loss(s_logits, t_logits, temperature=temperature)

        # Each rank's batch is disjoint (data parallel), so total tokens this
        # micro-batch across the world = ids.numel() * world.
        this_micro_tokens = int(ids.numel()) * world
        tokens_consumed += this_micro_tokens
        pending_window_tokens += this_micro_tokens

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
            # Item 2: dump diagnostic on FIRST occurrence only. We snapshot
            # the (already-computed) student/teacher logits — no extra forward
            # so we don't deadlock under ZeRO-3 (where forward is collective).
            # Running rank 0 only is therefore safe.
            if not nan_diagnostic_emitted and accelerator.is_main_process:
                try:
                    _dump_nan_diagnostic(
                        ids=ids, s_logits=s_logits, t_logits=t_logits,
                        loss=loss, artifacts_dir=artifacts_dir,
                        step=step, micro=micro_in_window,
                    )
                except Exception as err:                          # noqa: BLE001
                    log.warning("nan diagnostic failed: %s", err)
            nan_diagnostic_emitted = True

            # Item 6: account for skipped tokens.
            tokens_skipped_nan += this_micro_tokens
            # The micro is now booked under skipped_nan, not pending_window.
            pending_window_tokens -= this_micro_tokens
            nan_in_current_window = True

            if is_ds:
                # Build a graph-connected zero loss so DS's micro-batch
                # counter advances and backward touches the engine
                # bookkeeping. ``s_logits * 0.0`` is NaN-on-NaN, so first
                # nan_to_num the logits, then multiply by 0.0 — zero
                # gradient, but the graph stays attached to student params.
                cleaned = torch.nan_to_num(s_logits, nan=0.0,
                                           posinf=0.0, neginf=0.0)
                loss = (cleaned * 0.0).sum()
            else:
                optim.zero_grad(set_to_none=True)
                # Abandoned window: any prior committed-but-not-stepped micros
                # in this window are dropped; credit them to dropped_window so
                # the invariant holds.
                tokens_dropped_window += pending_window_tokens
                pending_window_tokens = 0
                micro_in_window = 0
                nan_in_current_window = False  # window abandoned — don't count as bad
                del t_logits, s_logits
                continue

        accelerator.backward(loss / grad_accum)
        if finite:  # don't record the zero-substitute value from NaN steps
            last_real_loss = float(loss.detach().item())
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
            # Credit the committed window to with_grad (sum of real per-micro
            # token counts, identical to grad_accum * micro_bsz * world *
            # seq_len for a packed window — but using the running sum is
            # robust to varying ids.numel() across micros).
            tokens_with_grad += pending_window_tokens
            pending_window_tokens = 0
            step += 1
            micro_in_window = 0

            # Item 2: end-of-window NaN escalation. A window is "bad" if any
            # of its grad_accum micros substituted; we count consecutive bad
            # windows and hard-raise after the threshold.
            if nan_in_current_window:
                consecutive_nan_windows += 1
                if consecutive_nan_windows >= nan_threshold:
                    raise RuntimeError(
                        f"NaN circuit breaker tripped: {consecutive_nan_windows} "
                        f"consecutive windows with non-finite loss "
                        f"(threshold={nan_threshold}). Inspect "
                        f"artifacts/nan_diagnostic_step*.json for per-layer "
                        f"activation stats. Likely root cause: corrupted "
                        f"student weights, incompatible teacher/student "
                        f"tokenisation, or sharded-init mismatch."
                    )
            else:
                consecutive_nan_windows = 0
            nan_in_current_window = False

            if accelerator.is_main_process and step % log_every == 0:
                log.info(
                    "step=%d/%d lr=%.3e loss=%.6f tok=%.2fB/%.2fB nan_skip=%.2fM",
                    step, total_steps, optim.param_groups[0]["lr"],
                    last_real_loss if last_real_loss is not None else float("nan"),
                    tokens_with_grad / 1e9,
                    int(dconf["total_tokens"]) / 1e9,
                    tokens_skipped_nan / 1e6,
                )

            if eval_every > 0 and step % eval_every == 0:
                # COLLECTIVE: every rank participates in the forward.
                eval_quick.run(student, tokenizer, config, accelerator)

            if save_every > 0 and step % save_every == 0:
                _save(student, tokenizer, config, artifacts_dir, accelerator,
                      step=step, tokens_with_grad=tokens_with_grad,
                      tokens_consumed=tokens_consumed,
                      tokens_skipped_nan=tokens_skipped_nan,
                      tokens_dropped_window=tokens_dropped_window,
                      partial=True)

            if step >= total_steps:
                break

    # On exit, any uncommitted micros in pending_window_tokens are dropped.
    tokens_dropped_window += pending_window_tokens
    pending_window_tokens = 0

    return _save(student, tokenizer, config, artifacts_dir, accelerator,
                 step=step, tokens_with_grad=tokens_with_grad,
                 tokens_consumed=tokens_consumed,
                 tokens_skipped_nan=tokens_skipped_nan,
                 tokens_dropped_window=tokens_dropped_window,
                 partial=False)


# ---------------------------------------------------------------------------
# Save (P1: gather under ZeRO-3; rolling partial dirs)
# ---------------------------------------------------------------------------


def _save(student, tokenizer, config, artifacts_dir, accelerator,
          *, step: int, tokens_with_grad: int, tokens_consumed: int,
          tokens_skipped_nan: int = 0, tokens_dropped_window: int = 0,
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
    are pruned (keep last 2). Writes are atomic via ``.tmp`` + rename, with a
    ``_SAVE_COMPLETE`` sentinel inside each finished dir.
    """
    from moe_compress.utils.model_io import (
        COMPRESSED_METADATA_FILENAME, FactoredExperts, iter_moe_layers,
    )

    accelerator.wait_for_everyone()

    if partial:
        out_dir = artifacts_dir / f"chapter1_recovered_partial_step{step}"
    else:
        out_dir = artifacts_dir / "chapter1_recovered"
    # Item 1: write to a .tmp directory and atomically rename. Readers either
    # see the previous (intact) directory or the new (complete) one — never a
    # half-written checkpoint.
    tmp_dir = out_dir.parent / f"{out_dir.name}.tmp"

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
            "tokens_skipped_nan": tokens_skipped_nan,
            "tokens_dropped_window": tokens_dropped_window,
            "step": step,
            "partial": partial,
        },
    }

    # 2. Get the consolidated state dict. COLLECTIVE under DS3.
    #    Returns CPU-resident dict on rank 0, empty dict on others.
    state_dict = accelerator.get_state_dict(student)

    # 3. Rank 0 writes everything to TMP, then atomic-replaces the final dir.
    if accelerator.is_main_process:
        # Clean any stale .tmp from a previous failed save.
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True)
        unwrapped.save_pretrained(
            tmp_dir, state_dict=state_dict, safe_serialization=True,
        )
        if tokenizer is not None:
            tokenizer.save_pretrained(tmp_dir)
        (tmp_dir / COMPRESSED_METADATA_FILENAME).write_text(
            json.dumps(metadata, indent=2)
        )
        # Sentinel file written LAST — its presence inside out_dir is a
        # post-rename guarantee that the dir was fully populated before the
        # rename.
        (tmp_dir / "_SAVE_COMPLETE").write_text(
            json.dumps({"step": step, "partial": partial})
        )
        # Compute sha256 of first shard for the manifest BEFORE the rename
        # so we don't double-walk after.
        first_shard_sha = _sha256_of_first_shard(tmp_dir)

        # Atomic rename. Replaces existing out_dir if any.
        _atomic_replace_dir(tmp_dir, out_dir)

        log.info("Saved %s checkpoint to %s (step=%d, tokens_with_grad=%.2fB, "
                 "tokens_skipped_nan=%.2fM)",
                 "PARTIAL" if partial else "FINAL", out_dir, step,
                 tokens_with_grad / 1e9, tokens_skipped_nan / 1e6)

        if partial:
            _append_to_partials_manifest(
                artifacts_dir, step=step, path=out_dir.name,
                sha256_first_shard=first_shard_sha,
            )
            _prune_old_partials(artifacts_dir, keep=2)
        else:
            # Final save: clean up all partial dirs so they don't linger on
            # the bucket or get picked up by _upload_results as a fallback.
            _prune_old_partials(artifacts_dir, keep=0)

    accelerator.wait_for_everyone()
    return out_dir


def _prune_old_partials(artifacts_dir: Path, *, keep: int = 2) -> None:
    """Delete all but the K most-recent partial dirs AND drop them from
    ``partials.json``. ``.tmp`` siblings are not counted."""
    pattern = "chapter1_recovered_partial_step*"
    dirs: list[tuple[int, Path]] = []
    for p in artifacts_dir.glob(pattern):
        if not p.is_dir() or p.name.endswith(".tmp"):
            continue
        try:
            n = int(p.name.split("step")[-1])
        except ValueError:
            continue
        dirs.append((n, p))
    dirs.sort()  # ascending step
    to_delete = dirs if keep == 0 else dirs[:-keep]
    for _, p in to_delete:
        try:
            shutil.rmtree(p)
            log.info("Pruned old partial: %s", p)
        except OSError as err:
            log.warning("Failed to prune %s: %s", p, err)

    # Update manifest: drop entries whose dirs no longer exist.
    manifest_path = artifacts_dir / "partials.json"
    if manifest_path.exists():
        try:
            entries = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            entries = []
        kept = [e for e in entries if (artifacts_dir / e["path"]).is_dir()]
        if len(kept) != len(entries):
            _atomic_write_json(manifest_path, kept)


# ---------------------------------------------------------------------------
# Partial checkpoint discovery (resume)
# ---------------------------------------------------------------------------


def _load_latest_partial(artifacts_dir: Path) -> "tuple[Path | None, int]":
    """Find the latest completed partial checkpoint for resume.

    Returns (partial_dir, resume_step) or (None, 0) if none found.
    A partial is valid iff _SAVE_COMPLETE exists inside it — written last,
    after atomic rename, so its presence guarantees a fully-written dir.
    """
    dirs: list[tuple[int, Path]] = []
    for p in artifacts_dir.glob("chapter1_recovered_partial_step*"):
        if not p.is_dir() or p.name.endswith(".tmp"):
            continue
        if not (p / "_SAVE_COMPLETE").exists():
            log.warning("Partial %s missing _SAVE_COMPLETE — skipping (incomplete write).", p.name)
            continue
        try:
            step = int(p.name.split("step")[-1])
        except ValueError:
            log.warning("Could not parse step from %s — skipping.", p.name)
            continue
        dirs.append((step, p))
    if not dirs:
        return None, 0
    dirs.sort(reverse=True)
    best_step, best_path = dirs[0]
    log.info("Found partial checkpoint at step=%d: %s", best_step, best_path)
    return best_path, best_step


# ---------------------------------------------------------------------------
# Atomic save helpers (item 1)
# ---------------------------------------------------------------------------


def _atomic_replace_dir(src: Path, dst: Path) -> None:
    """Atomically replace ``dst`` with ``src``. Both must be on the same FS.

    POSIX ``rename(2)`` (and Python's ``os.rename``) refuses to replace a
    non-empty directory. We work around by moving the existing ``dst`` aside
    first; on rename failure, restore it.
    """
    if dst.exists():
        backup = dst.with_name(dst.name + ".bak")
        if backup.exists():
            shutil.rmtree(backup)
        os.rename(dst, backup)
        try:
            os.rename(src, dst)
        except Exception:
            # Restore the backup on any failure.
            os.rename(backup, dst)
            raise
        shutil.rmtree(backup, ignore_errors=True)
    else:
        os.rename(src, dst)


def _sha256_of_first_shard(dir_: Path) -> str:
    """Hash the first ``model-*.safetensors`` shard for the manifest. A few
    seconds of I/O — cheap insurance for the eyeball test "is this checkpoint
    the one I think it is?"."""
    import hashlib
    shards = sorted(dir_.glob("model-*.safetensors"))
    if not shards:
        # Fallback for non-sharded saves.
        shards = sorted(dir_.glob("*.safetensors"))
    if not shards:
        return ""
    h = hashlib.sha256()
    with shards[0].open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_write_json(path: Path, payload) -> None:
    """Write JSON to ``path`` via ``.tmp`` + ``os.replace``."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)


def _append_to_partials_manifest(artifacts_dir: Path, *, step: int,
                                 path: str, sha256_first_shard: str) -> None:
    """Append/update an entry for ``step`` in ``artifacts/partials.json``."""
    from datetime import datetime, timezone
    manifest_path = artifacts_dir / "partials.json"
    entries: list[dict] = []
    if manifest_path.exists():
        try:
            entries = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as err:
            log.warning("partials.json unreadable, recreating: %s", err)
            entries = []
    # Replace any existing entry for this step (idempotent on retry).
    entries = [e for e in entries if e.get("step") != step]
    entries.append({
        "step": step,
        "path": path,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "sha256_first_shard": sha256_first_shard,
    })
    entries.sort(key=lambda e: e["step"])
    _atomic_write_json(manifest_path, entries)


# ---------------------------------------------------------------------------
# NaN diagnostic (item 2)
# ---------------------------------------------------------------------------


def _dump_nan_diagnostic(*, ids, s_logits, t_logits, loss, artifacts_dir: Path,
                         step: int, micro: int) -> None:
    """Snapshot the NaN-triggering inputs + logits stats to a JSON file.

    No extra forward — uses tensors already in the training-step scope, so
    safe to call from rank 0 only without deadlocking ZeRO-3 collectives.
    Per-layer activation hooks are intentionally NOT done here (they would
    need a collective re-forward); the saved input_ids let the operator run
    a postmortem with whatever tooling they prefer.
    """
    out = artifacts_dir / f"nan_diagnostic_step{step}_micro{micro}.json"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    def _stats(t):
        try:
            x = t.detach().float()
            return {
                "shape": list(t.shape),
                "dtype": str(t.dtype),
                "has_nan": bool(torch.isnan(x).any().item()),
                "has_inf": bool(torch.isinf(x).any().item()),
                "min": float(torch.nan_to_num(x, nan=float("inf")).min().item()),
                "max": float(torch.nan_to_num(x, nan=float("-inf")).max().item()),
                "mean_finite": float(
                    x[torch.isfinite(x)].mean().item()
                    if torch.isfinite(x).any() else float("nan")
                ),
            }
        except Exception as err:                                 # noqa: BLE001
            return {"error": str(err)}

    payload = {
        "step": step,
        "micro": micro,
        "input_ids_first_64": ids.detach().cpu().reshape(-1).tolist()[:64],
        "input_ids_shape": list(ids.shape),
        "loss_value": float(loss.detach().item()) if loss.numel() == 1 else None,
        "student_logits": _stats(s_logits),
        "teacher_logits": _stats(t_logits),
        "hint": (
            "If has_nan=True on student_logits but not teacher_logits, the "
            "student's forward is the culprit — likely sharded-init mismatch "
            "or corrupted weights. If both are NaN, suspect input_ids out of "
            "vocab range or a tokenizer mismatch. Per-layer hooks were NOT "
            "captured (would require a collective re-forward under ZeRO-3); "
            "rerun a single-batch forward locally with these input_ids to dig "
            "deeper."
        ),
    }
    _atomic_write_json(out, payload)
    log.warning("NaN diagnostic written to %s", out)
