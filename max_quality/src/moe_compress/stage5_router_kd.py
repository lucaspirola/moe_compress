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
    no_resume: bool = False,
    stage_key: str = "stage5",
) -> Path:
    """Run Router KD.

    ``stage_key`` controls the partial-dir and output-dir names, allowing the
    same code to serve both Stage 2.5 (``stage_key="stage2p5"``) and Stage 5
    (``stage_key="stage5"``).  The config section read is always
    ``stage5_router_kd`` regardless of ``stage_key``.
    """
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

    # --- torch.compile acceleration (spec §8) ---
    # Assigned BEFORE _get_teacher closure so the closure's reference resolves correctly.
    use_compile = bool(s5.get("torch_compile", False))

    if teacher_logits_cache is None:
        # Deferred teacher load: only load the teacher on the first live training
        # batch. On resume, fast-forward iterates without ever touching the teacher —
        # saves ~60s + 70 GB VRAM when resuming deep into training.
        _teacher_state: dict = {"model": None, "refs": None}

        def _get_teacher(student_refs_count: int):
            if _teacher_state["model"] is None:
                load_in_4bit = bool(s5.get("teacher_load_in_4bit", False))
                if config["model"].get("load_in_4bit", False) and not load_in_4bit:
                    log.warning(
                        "Stage 5: config['model']['load_in_4bit']=true but "
                        "stage5_router_kd.teacher_load_in_4bit=false. The teacher "
                        "will load in BF16 (~70 GB) and likely OOM the A100. "
                        "Set teacher_load_in_4bit: true to match."
                    )
                _device_map = {"": 0} if load_in_4bit else config["model"]["device_map"]
                log.info("Loading teacher for KD (first live batch): %s "
                         "(teacher_load_in_4bit=%s, device_map=%s)",
                         config["model"]["name_or_path"], load_in_4bit, _device_map)
                _t, _ = load_model(
                    config["model"]["name_or_path"],
                    revision=config["model"]["revision"],
                    torch_dtype=config["model"]["torch_dtype"],
                    device_map=_device_map,
                    attn_implementation=config["model"]["attn_implementation"],
                    load_in_4bit=load_in_4bit,
                    trust_remote_code=config["model"].get("trust_remote_code", False),
                )
                _t.eval()
                if use_compile:
                    try:
                        log.info("Stage 5: torch.compile(teacher, mode='reduce-overhead')")
                        _t = torch.compile(_t, mode="reduce-overhead")
                    except Exception as exc:
                        log.warning("Stage 5: torch.compile(teacher) failed (%s) — eager", exc)
                _teacher_state["model"] = _t
                _teacher_state["refs"] = list(iter_moe_layers(_t))
                assert len(_teacher_state["refs"]) == student_refs_count, (
                    f"Teacher/student MoE layer count mismatch: "
                    f"{len(_teacher_state['refs'])} vs {student_refs_count}"
                )
            return _teacher_state["model"]
    # Freeze non-router parameters BEFORE compiling the student so that the
    # compiled graph is traced with the final requires_grad flags. Compiling
    # before freeze risks the compiler baking in the wrong gradient-enabled
    # state for parameters that are about to be frozen.
    _freeze_non_routers(student, s5["trainable_name_patterns"])

    # Optimizer constructed AFTER freezing so it only receives parameters that
    # have requires_grad=True at construction time.
    optim = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=s5["learning_rate"],
        weight_decay=0.0,  # paper Table 1 does not specify weight decay; default 0.01 would
                           # regularize router weights toward zero, counteracting KD gradient.
    )

    # torch.compile applied AFTER freeze+optimizer construction so the compiled
    # graph reflects the final frozen parameter layout. named_parameters() on
    # the compiled wrapper delegates to the underlying module — the optimizer
    # already holds the correct parameter references before compilation.
    if use_compile:
        try:
            log.info("Stage 5: torch.compile(student, mode='reduce-overhead')")
            student = torch.compile(student, mode="reduce-overhead")
        except Exception as exc:
            log.warning("Stage 5: torch.compile failed (%s) — falling back to eager mode", exc)
            use_compile = False

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
    grad_accum = s5["gradient_accumulation"]
    T = s5["kd_temperature"]
    ckpt_every = int(s5.get("checkpoint_every_n_steps", 100))

    # Teacher/student MoE layer count sanity check (router structure must match
    # even though we're distilling at vocab level — the student's routers are
    # what we're training).
    student_refs = list(iter_moe_layers(student))

    # -----------------------------------------------------------------------
    # Crash-resume: find latest checkpoint and restore router + optim state.
    # -----------------------------------------------------------------------
    resume_step = 0
    resume_epoch = 0
    resume_batch_i = -1

    if no_resume:
        partial_dir = None
        # Delete any stale partial dir so a future non-no-resume run does not
        # accidentally resume from a prior run's checkpoints.
        stale = artifacts_dir / f"_{stage_key}_partial"
        if stale.exists():
            import shutil as _shutil
            _shutil.rmtree(stale, ignore_errors=True)
    else:
        partial_dir = artifacts_dir / f"_{stage_key}_partial"
        partial_dir.mkdir(parents=True, exist_ok=True)
        for _stale in partial_dir.glob("*.tmp"):
            _stale.unlink(missing_ok=True)

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
                    f"(expected 1) — delete _{stage_key}_partial/ and re-run"
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
            if "gradient_accumulation" in payload:
                saved_ga = int(payload["gradient_accumulation"])
                if saved_ga != grad_accum:
                    raise RuntimeError(
                        f"Stage 5 resume: gradient_accumulation mismatch — "
                        f"checkpoint has {saved_ga}, config has {grad_accum}. "
                        f"Delete _{stage_key}_partial/ and re-run or align the config."
                    )
            log.info("Stage 5: resumed from step %d (epoch %d, batch %d)",
                     resume_step, resume_epoch, resume_batch_i)

    student.train()
    total_steps = (len(batches) // grad_accum) * s5["epochs"]
    log.info("Stage 5: %d routers trainable; %d steps (grad-accum=%d)",
             sum(1 for _ in student.parameters() if _.requires_grad),
             total_steps, grad_accum)
    trailing = len(batches) % grad_accum
    if trailing != 0:
        log.warning(
            "Stage 5: %d trailing batches will not form a complete grad-accum "
            "window (grad_accum=%d) — their gradients are dropped at epoch end.",
            trailing, grad_accum,
        )

    step = resume_step
    optim.zero_grad()
    for epoch in range(s5["epochs"]):
        if epoch < resume_epoch:
            continue
        for i, batch in enumerate(batches):
            # Fast-forward: skip batches already processed in the resumed run.
            # resume_batch_i is the last batch of the grad-accum window that
            # triggered the checkpoint — the optimizer step has already occurred
            # for that entire window (including batch resume_batch_i itself).
            # Skip 0..resume_batch_i (inclusive) to avoid re-running the already-
            # completed step and triggering a spurious duplicate optimizer step.
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
                    teacher_out = _get_teacher(len(student_refs))(input_ids=batch)
                teacher_vocab_logits = teacher_out.logits.to(torch.float32)  # [B, L, |V|]
                del teacher_out  # free the full output object before student backward pass

            # Student: full forward pass with gradients (routers are trainable).
            student_out = student(input_ids=batch)
            student_vocab_logits = student_out.logits.to(torch.float32)  # [B, L, |V|]
            del student_out  # free model output object; student_vocab_logits retains grad_fn

            # KL(teacher ‖ student) over vocabulary, per-token, scaled by τ².
            # Paper Eq. 3: L_RKD = (τ²/N_x) Σ_t KL(p_T^t ‖ p_S^t)
            # Shift logits: predict token t+1 from position t (standard causal LM).
            t_logits_shift = teacher_vocab_logits[:, :-1, :]   # [B, L-1, |V|]
            s_logits_shift = student_vocab_logits[:, :-1, :]   # [B, L-1, |V|]

            # Chunked KL to bound peak memory: at |V|≈150K with B=4, chunk=128,
            # peak intermediate is ~300 MB vs ~1.2 GB for the full sequence.
            seq_chunk = int(s5.get("kd_seq_chunk_size", 128))
            loss = _chunked_vocab_kl(s_logits_shift, t_logits_shift, T, chunk_size=seq_chunk)

            (loss / grad_accum).backward()

            if (i + 1) % grad_accum == 0:
                # Pre-step: compute gradient norm over trainable params.
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in student.parameters() if p.requires_grad and p.grad is not None],
                        float('inf'),
                    )
                )
                optim.step()
                optim.zero_grad()
                step += 1
                if step % config["logging"]["log_every_n_steps"] == 0:
                    loss_val = float(loss.item())  # per-batch loss; effective gradient = loss / grad_accum
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
                if partial_dir is not None and ckpt_every > 0 and step % ckpt_every == 0:
                    _save_stage5_checkpoint(partial_dir, step, epoch, i, student, optim,
                                            grad_accum=grad_accum)
                    # Keep only the two most recent checkpoints to bound disk use.
                    # Sort by step number (ascending) and delete all but the newest two.
                    all_ckpts = sorted(
                        partial_dir.glob("step_*.pt"),
                        key=lambda p: int(p.stem.split("_")[1]),
                    )
                    for old_ckpt in all_ckpts[:-2]:
                        old_ckpt.unlink(missing_ok=True)
        trailing = len(batches) % grad_accum if grad_accum > 1 else 0
        if trailing:
            log.debug(
                "Epoch %d: %d trailing batch(es) not stepped (grad_accum=%d)",
                epoch, trailing, grad_accum,
            )
        optim.zero_grad()

    out_dir = artifacts_dir / f"{stage_key}_final"
    save_compressed_checkpoint(
        student, tokenizer, out_dir, pipeline_stage=f"{stage_key}_final",
    )
    log.info("Stage %s complete → %s", stage_key, out_dir)
    return out_dir


def _chunked_vocab_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
    chunk_size: int = 128,
) -> torch.Tensor:
    """Compute vocab-level KL(teacher ‖ student) in sequence chunks.

    Processes ``chunk_size`` sequence positions at a time to bound peak
    intermediate memory. At chunk_size=128 with |V|=150K and B=4:
      Peak intermediate per chunk ≈ 4 × 128 × 150K × 4 bytes ≈ 300 MB
      vs ≈1.2 GB for the full sequence at L=512.

    Returns scalar loss = (τ²/N_tokens) × Σ_t KL(teacher_t ‖ student_t).
    """
    B, L, V = student_logits.shape
    total_kl = torch.zeros((), device=student_logits.device, dtype=torch.float32)
    n_tokens = 0
    for start in range(0, L, chunk_size):
        end = min(start + chunk_size, L)
        s_chunk = student_logits[:, start:end, :]
        t_chunk = teacher_logits[:, start:end, :]
        t_p = F.softmax(t_chunk / temperature, dim=-1)
        s_lp = F.log_softmax(s_chunk / temperature, dim=-1)
        chunk_kl = F.kl_div(s_lp, t_p, reduction="none").sum(dim=-1)  # [B, chunk_len]
        total_kl = total_kl + chunk_kl.sum()
        n_tokens += chunk_kl.numel()
        del t_p, s_lp, chunk_kl  # free intermediates eagerly
    return (total_kl / max(n_tokens, 1)) * (temperature ** 2)


def _save_stage5_checkpoint(
    partial_dir: Path,
    step: int,
    epoch: int,
    batch_idx: int,
    student: nn.Module,
    optim: torch.optim.Optimizer,
    grad_accum: int = 1,
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
        "gradient_accumulation": grad_accum,
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



