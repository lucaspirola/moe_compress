"""Phase C.5 block-level joint refinement (S3-6 of the Stage 3 plugin refactor).

Home of the Phase C.5 block-refine core relocated VERBATIM from the legacy
``stage3_svd.py`` monolith:

* ``_phase_c5_block_refine`` — block-level joint refinement of the factored
  U/V slots and the block-local RMSNorm scales via AdamW on the anchored MSE
  objective ‖ℒ_i(X) − ℒ'_i(X')‖² (paper 2604.02119, Algorithm 2 §3.3);
* ``_advance_streams`` — forwards both the student and teacher decoder layers
  (no grad) on each batch's current stream and returns the next-block-input
  tensors as bf16 CPU lists.

Both symbols are byte-identical copies of the monolith bodies; the monolith
re-imports them (``# noqa: F401`` block in ``stage3_svd.py``) so ``run()`` and
external callers/tests keep their existing import paths. This is the LAST
stage-3 plugin extraction.

``_phase_c5_block_refine`` carries three NESTED closures —
``_capture_first_pass`` / ``_capture_block_input`` / ``_lr_at`` — that ride
along inside its body verbatim; they are nested ``def``s, not module-level
symbols, so the relocation set is exactly the 2 module-level functions. The
function-scope ``import os as _os`` / ``import shutil as _shutil`` inside
``_phase_c5_block_refine`` likewise stay inline verbatim.

Circular-import note (mirror of ``stage3/plugins/aa_svd_factor.py``): the
block-refine core is SELF-CONTAINED — it imports only stdlib / ``torch`` /
``...utils.*`` / ``...pipeline.context``, and NEVER ``stage3_svd`` or
``stage3.orchestrator``. ``_phase_c5_block_refine`` has zero monolith-resident
dependency, so this module needs NO lazy / function-scope imports for the
relocation. ``stage3_svd`` imports *this* module at load time, so a module-top
``from ...stage3_svd import ...`` here would deadlock the import cycle — but no
such import is needed.

``BlockRefinePlugin`` is the FIRST genuinely config-GATED stage-3 plugin: its
``is_enabled`` returns the ``stage3_svd.block_refine.enabled`` flag (default
False) rather than an unconditional ``True``. It is registered-but-INERT at
S3-6 — no walk or test invokes its ``refine_blocks`` hook. S3-7 wires it into
the live Stage 3 plugin sequencer.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ...utils.model_io import (
    MATRIX_NAMES,
    FactoredExperts,
    MoELayerRef,
    iter_decoder_layers,
)
from ...utils.trackio_log import trackio_log as _trackio_log
from ...pipeline.context import PipelineContext

log = logging.getLogger(__name__)


def _phase_c5_block_refine(
    student,
    teacher,
    moe_layers: list[MoELayerRef],
    teacher_moe_layers: list[MoELayerRef],
    calib_tensor: torch.Tensor,
    *,
    batch_size: int,
    learning_rate: float,
    epochs: int,
    warmup_ratio: float,
    weight_decay: float,
    artifacts_dir: Path,
    no_resume: bool,
    device,
) -> None:
    """Phase C.5 — block-level joint refinement (paper 2604.02119, Algorithm 2 §3.3).

    For each decoder block i sequentially (0 → N−1):
      1. Compute the teacher-block target ℒ_i(X_i^teacher) once per batch on
         the still-resident teacher (no grad).
      2. Train the block's factored U/V slots and the two RMSNorm scales
         (input_layernorm, post_attention_layernorm) jointly by AdamW
         (fp32 moments + fp32 master) for `epochs` over the calibration
         data with cosine schedule + linear warmup, batch_size batches at
         a time, MSE loss against the teacher target.
      3. Advance both upstream streams (X_(i+1)^teacher = teacher.layers[i](...)
         and X'_(i+1) = refined_student.layers[i](...)) for the next block.
      4. Save a per-block atomic checkpoint with the refined U/V + RMSNorm
         state for crash-resume.
    """
    # All decoder layers (MoE + any dense interlayers) participate in the
    # forward stream advance so X' produced for block i+1 reflects every
    # intervening transform. Only MoE blocks (subset) get the AdamW
    # refinement — dense layers have nothing factored to refine.
    s_layers_all = {idx: layer for idx, layer in iter_decoder_layers(student)}
    t_layers_all = {idx: layer for idx, layer in iter_decoder_layers(teacher)}
    s_layers_by_idx = {ref.layer_idx: ref.layer_module for ref in moe_layers}
    t_layers_by_idx = {ref.layer_idx: ref.layer_module for ref in teacher_moe_layers}
    moe_idx_to_pos = {ref.layer_idx: i for i, ref in enumerate(moe_layers)}
    all_indices = sorted(s_layers_all.keys())
    n_blocks = len(all_indices)
    n_moe_blocks = len(moe_layers)
    log.info("Stage 3 Phase C.5: %d decoder layers (%d MoE refined) × %d epochs (lr=%.1e, batch=%d)",
             n_blocks, n_moe_blocks, epochs, learning_rate, batch_size)

    partial_dir = None if no_resume else artifacts_dir / "_stage3_phase_c5_partial"
    if partial_dir is not None:
        partial_dir.mkdir(parents=True, exist_ok=True)
        for stale in partial_dir.glob("*.tmp"):
            stale.unlink(missing_ok=True)

    # Build input batches once. calib_tensor is already token-id integer; we
    # forward through the model's embedding + decoder stack manually so the
    # captured kwargs (position_embeddings, attention_mask, position_ids,
    # cache_position) come from the model's own prep code.
    # drop_last: kwargs (attention_mask, position_ids, position_embeddings)
    # are captured once from batch 0 and replayed; a trailing partial batch
    # would shape-mismatch the cached masks.
    n_seq, seq_len = calib_tensor.shape
    n_batches = n_seq // batch_size
    if n_batches == 0:
        raise RuntimeError(
            f"Stage 3 Phase C.5: calibration tensor has {n_seq} sequences "
            f"but batch_size={batch_size}; need at least one full batch."
        )
    if n_batches * batch_size < n_seq:
        log.info("Stage 3 Phase C.5: dropping trailing partial batch "
                 "(%d sequences) to keep cached kwargs shape-stable",
                 n_seq - n_batches * batch_size)
    batches = [calib_tensor[b * batch_size:(b + 1) * batch_size] for b in range(n_batches)]
    log.info("Stage 3 Phase C.5: %d calibration sequences in %d batches of %d",
             n_batches * batch_size, n_batches, batch_size)

    # Capture per-layer kwargs once via a forward pre-hook on each layer.
    # kwargs are stable across batches with the same shape (attention masks,
    # position_ids, position_embeddings); we capture from batch 0 and reuse.
    def _capture_first_pass(model_, layers_by_idx, sample_batch):
        captured_kwargs: dict[int, dict] = {}
        captured_inputs: dict[int, torch.Tensor] = {}
        handles = []
        for li, layer in layers_by_idx.items():
            def _make_hook(idx):
                def _hook(_mod, args, kwargs):
                    captured_inputs[idx] = args[0].detach() if args else kwargs.get("hidden_states").detach()
                    captured_kwargs[idx] = {k: v for k, v in kwargs.items() if k != "hidden_states"}
                return _hook
            handles.append(layer.register_forward_pre_hook(_make_hook(li), with_kwargs=True))
        try:
            with torch.no_grad():
                model_(input_ids=sample_batch.to(device))
        finally:
            for h in handles:
                h.remove()
        return captured_kwargs, captured_inputs

    # Use batch 0 to capture stable kwargs (these don't depend on weight values,
    # only on input_ids shape/positions).
    sample = batches[0]
    # Initialize both upstream streams: per-batch hidden state at the input to
    # decoder layer 0. Captured via a one-shot forward through the embed prefix
    # with a pre-hook + EarlyExit on the first decoder layer.
    first_idx = all_indices[0]
    log.info("Stage 3 Phase C.5: capturing initial upstream streams at layer %d", first_idx)

    def _capture_block_input(model_, layer_module, all_batches):
        """Run the model forward and capture the hidden_state input to
        ``layer_module`` once per batch via a one-shot pre-hook + EarlyExit.
        Returns a list of CPU bf16 tensors, one per batch."""
        captured: list[torch.Tensor | None] = [None] * len(all_batches)
        cur_idx = [0]

        class _EarlyExit(Exception):
            pass

        def _hook(_mod, args, kwargs):
            t = args[0] if args else kwargs.get("hidden_states")
            captured[cur_idx[0]] = t.detach().to(dtype=torch.bfloat16, device="cpu")
            raise _EarlyExit

        handle = layer_module.register_forward_pre_hook(_hook, with_kwargs=True)
        try:
            for bi, batch in enumerate(all_batches):
                cur_idx[0] = bi
                try:
                    with torch.no_grad():
                        model_(input_ids=batch.to(device))
                except _EarlyExit:
                    pass
        finally:
            handle.remove()
        if any(c is None for c in captured):
            raise RuntimeError("Phase C.5: failed to capture block input for some batches")
        return captured  # type: ignore

    s_first_layer = s_layers_all[first_idx]
    t_first_layer = t_layers_all[first_idx]
    X_student = _capture_block_input(student, s_first_layer, batches)
    X_teacher = _capture_block_input(teacher, t_first_layer, batches)

    # Capture full per-decoder-layer kwargs for ALL layers (including dense
    # interlayers) so the stream advance is faithful for mixed architectures.
    student_kwargs_all, _ = _capture_first_pass(student, s_layers_all, sample)
    teacher_kwargs_all, _ = _capture_first_pass(teacher, t_layers_all, sample)

    student_dtype = next(student.parameters()).dtype

    for layer_idx in all_indices:
        s_layer = s_layers_all[layer_idx]
        t_layer = t_layers_all[layer_idx]
        is_moe = layer_idx in moe_idx_to_pos
        block_pos = moe_idx_to_pos.get(layer_idx)
        if not is_moe:
            # Dense decoder layer between MoE blocks: just advance both streams.
            X_student, X_teacher = _advance_streams(
                s_layer, t_layer, X_student, X_teacher,
                student_kwargs_all.get(layer_idx, {}),
                teacher_kwargs_all.get(layer_idx, {}), device,
            )
            continue

        ckpt_path = partial_dir / f"block_{layer_idx}.pt" if partial_dir is not None else None
        if ckpt_path is not None and ckpt_path.exists():
            payload = torch.load(ckpt_path, map_location="cpu")
            if int(payload.get("format_version", 0)) != 1:
                raise RuntimeError(
                    f"Stage 3 Phase C.5 resume: {ckpt_path} format_version != 1; "
                    "delete _stage3_phase_c5_partial/ and re-run."
                )
            fe = moe_layers[block_pos].experts_module
            ref_dev = getattr(fe, "gate_proj_U").device
            for name in MATRIX_NAMES:
                getattr(fe, f"{name}_U").data.copy_(
                    payload[f"{name}_U"].to(device=ref_dev, dtype=student_dtype))
                getattr(fe, f"{name}_V").data.copy_(
                    payload[f"{name}_V"].to(device=ref_dev, dtype=student_dtype))
            for path in ("input_layernorm", "post_attention_layernorm",
                         "self_attn.q_norm", "self_attn.k_norm"):
                mod = s_layer
                for part in path.split("."):
                    mod = getattr(mod, part, None)
                    if mod is None:
                        break
                if mod is not None and hasattr(mod, "weight") and path in payload:
                    mod.weight.data.copy_(
                        payload[path].to(device=mod.weight.device, dtype=student_dtype))
            log.info("Stage 3 Phase C.5 block %d/%d (idx=%d) — resumed from checkpoint",
                     block_pos + 1, n_blocks, layer_idx)
            # Still need to advance the streams using the (resumed) refined block.
            X_student, X_teacher = _advance_streams(
                s_layer, t_layer, X_student, X_teacher,
                student_kwargs_all.get(layer_idx, {}),
                teacher_kwargs_all.get(layer_idx, {}), device,
            )
            continue

        # Collect trainables for this block. FactoredExperts U/V slots + the
        # two RMSNorm scales. All other params remain frozen (we set
        # requires_grad on the trainable subset only).
        fe = moe_layers[block_pos].experts_module
        if not isinstance(fe, FactoredExperts):
            log.info("Stage 3 Phase C.5 block %d skipped (not factored); "
                     "advancing streams without refinement", layer_idx)
            X_student, X_teacher = _advance_streams(
                s_layer, t_layer, X_student, X_teacher,
                student_kwargs_all.get(layer_idx, {}),
                teacher_kwargs_all.get(layer_idx, {}), device,
            )
            continue
        trainables: list[nn.Parameter] = []
        for name in MATRIX_NAMES:
            for slot in (f"{name}_U", f"{name}_V"):
                p = getattr(fe, slot)
                p.requires_grad_(True)
                trainables.append(p)
        # RMSNorm scope (paper 2604.02119 Algorithm 2 line 9 / Appendix B.2):
        # all block-local norms participate in θ_i. For Qwen3 this includes
        # input_layernorm + post_attention_layernorm (block-level), and the
        # per-head q_norm + k_norm inside self-attention.
        norm_params: list[nn.Parameter] = []
        norm_module_paths = ["input_layernorm", "post_attention_layernorm",
                             "self_attn.q_norm", "self_attn.k_norm"]
        for path in norm_module_paths:
            mod = s_layer
            ok = True
            for part in path.split("."):
                mod = getattr(mod, part, None)
                if mod is None:
                    ok = False
                    break
            if ok and hasattr(mod, "weight") and isinstance(mod.weight, nn.Parameter):
                mod.weight.requires_grad_(True)
                norm_params.append(mod.weight)
        trainables.extend(norm_params)

        # Spec §6 Phase C.5: AdamW must run with fp32 moments + fp32 master
        # weights. Vanilla `torch.optim.AdamW` initializes `exp_avg`/`exp_avg_sq`
        # with the same dtype as the parameter — so for bf16 params, moments
        # are bf16, losing the precision rationale. Promote trainables to
        # fp32 in-place before the optimizer is constructed; restore the
        # original dtype after refinement. Frozen params in the same layer
        # stay bf16; PyTorch dtype-promotes through `nn.Linear` and RMSNorm
        # so the layer forward runs cleanly in mixed precision.
        original_dtypes: dict[int, torch.dtype] = {}
        for p in trainables:
            original_dtypes[id(p)] = p.dtype
            if p.dtype != torch.float32:
                p.data = p.data.to(torch.float32)
        opt = torch.optim.AdamW(trainables, lr=learning_rate, weight_decay=weight_decay)
        total_steps = max(1, epochs * len(batches))
        warmup_steps = max(1, int(warmup_ratio * total_steps))

        def _lr_at(step: int) -> float:
            # Step is 0-indexed; offset by 1 so the first step uses a non-zero
            # warmup fraction rather than lr=0 (paper-typical schedules ramp
            # from a small fraction up to 1.0, not literally 0). Likewise the
            # cosine never reaches exactly 0 at total_steps − 1.
            s = step + 1
            if s <= warmup_steps:
                return s / max(1, warmup_steps)
            progress = (s - warmup_steps) / max(1, total_steps - warmup_steps + 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        # Pre-compute teacher targets once per batch (no grad).
        teacher_targets: list[torch.Tensor] = []
        with torch.no_grad():
            for bi, _ in enumerate(batches):
                x_t = X_teacher[bi].to(device=device, dtype=student_dtype)
                out = t_layer(x_t, **teacher_kwargs_all.get(layer_idx, {}))
                if isinstance(out, tuple):
                    out = out[0]
                teacher_targets.append(out.detach().to(dtype=torch.bfloat16, device="cpu"))

        # AdamW loop.
        loss_first: float | None = None
        loss_last: float | None = None
        step = 0
        for epoch in range(epochs):
            for bi, _ in enumerate(batches):
                x_s = X_student[bi].to(device=device, dtype=student_dtype)
                target = teacher_targets[bi].to(device=device, dtype=student_dtype)
                out = s_layer(x_s, **student_kwargs_all.get(layer_idx, {}))
                if isinstance(out, tuple):
                    out = out[0]
                loss = nn.functional.mse_loss(out.to(torch.float32),
                                               target.to(torch.float32))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                # Apply LR schedule by overwriting param_group lr each step.
                lr_now = learning_rate * _lr_at(step)
                for g in opt.param_groups:
                    g["lr"] = lr_now
                opt.step()
                step += 1
                if loss_first is None:
                    loss_first = float(loss.item())
                loss_last = float(loss.item())

        # Restore frozen state and original dtypes.
        for p in trainables:
            p.requires_grad_(False)
            target_dtype = original_dtypes.get(id(p))
            if target_dtype is not None and p.dtype != target_dtype:
                p.data = p.data.to(target_dtype)

        rel_drop = (loss_first - loss_last) / max(loss_first or 1e-12, 1e-12) if loss_first else 0.0
        log.info("  Phase C.5 block %d/%d (idx=%d) loss %.4e → %.4e (%.1f%%↓)",
                 block_pos + 1, n_blocks, layer_idx,
                 loss_first or 0.0, loss_last or 0.0, 100 * rel_drop)
        _trackio_log({
            "stage3/c5_layer_idx": float(layer_idx),
            "stage3/c5_loss_init": loss_first or 0.0,
            "stage3/c5_loss_final": loss_last or 0.0,
            "stage3/c5_loss_rel_drop": rel_drop,
            # Additive: training-loop shape and warmup configuration. All in scope.
            "stage3/c5_total_steps": int(total_steps),
            "stage3/c5_warmup_steps": int(warmup_steps),
            "stage3/c5_trainable_param_count": int(len(trainables)),
        })

        # Save per-block checkpoint atomically.
        if ckpt_path is not None:
            payload = {"format_version": 1, "layer_idx": layer_idx}
            for path in ("input_layernorm", "post_attention_layernorm",
                         "self_attn.q_norm", "self_attn.k_norm"):
                mod = s_layer
                for part in path.split("."):
                    mod = getattr(mod, part, None)
                    if mod is None:
                        break
                if mod is not None and hasattr(mod, "weight"):
                    payload[path] = mod.weight.detach().cpu()
            for name in MATRIX_NAMES:
                payload[f"{name}_U"] = getattr(fe, f"{name}_U").detach().cpu()
                payload[f"{name}_V"] = getattr(fe, f"{name}_V").detach().cpu()
            tmp = ckpt_path.with_suffix(".pt.tmp")
            torch.save(payload, tmp)
            import os as _os
            _os.replace(tmp, ckpt_path)

        # Advance streams for the next block (no grad).
        X_student, X_teacher = _advance_streams(
            s_layer, t_layer, X_student, X_teacher,
            student_kwargs_all.get(layer_idx, {}),
            teacher_kwargs_all.get(layer_idx, {}), device,
        )

    # Cleanup: remove checkpoint dir on success.
    if partial_dir is not None and partial_dir.exists():
        import shutil as _shutil
        _shutil.rmtree(partial_dir, ignore_errors=True)
        log.info("Stage 3 Phase C.5: removed checkpoint dir (run completed cleanly)")


def _advance_streams(s_layer, t_layer, X_student, X_teacher,
                     s_kwargs, t_kwargs, device):
    """Forward both layers (no grad) on each batch's current stream and return
    the next-block-input tensors as bf16 CPU lists."""
    new_s: list[torch.Tensor] = []
    new_t: list[torch.Tensor] = []
    student_dtype = next(s_layer.parameters()).dtype
    teacher_dtype = next(t_layer.parameters()).dtype
    with torch.no_grad():
        for x_s_cpu, x_t_cpu in zip(X_student, X_teacher):
            x_s = x_s_cpu.to(device=device, dtype=student_dtype)
            out_s = s_layer(x_s, **s_kwargs)
            if isinstance(out_s, tuple):
                out_s = out_s[0]
            new_s.append(out_s.detach().to(dtype=torch.bfloat16, device="cpu"))
            x_t = x_t_cpu.to(device=device, dtype=teacher_dtype)
            out_t = t_layer(x_t, **t_kwargs)
            if isinstance(out_t, tuple):
                out_t = out_t[0]
            new_t.append(out_t.detach().to(dtype=torch.bfloat16, device="cpu"))
    return new_s, new_t


class BlockRefinePlugin:
    """Stage 3 Phase C.5 block-refine plugin (S3-6 — registered-but-INERT).

    Owns the block-level joint refinement core: ``_phase_c5_block_refine``
    (the per-block AdamW refinement of the factored U/V slots and the
    block-local RMSNorm scales, anchored against the teacher block output)
    and ``_advance_streams`` (the no-grad dual-stream forward that produces
    the next block's inputs). The core lives in the module-level symbols
    relocated verbatim from the monolith (block-refine paper 2604.02119,
    Algorithm 2 §3.3).

    Unlike the S3-2..S3-5 plugins, ``BlockRefinePlugin`` is the FIRST
    genuinely config-GATED stage-3 plugin: ``is_enabled`` returns the
    ``stage3_svd.block_refine.enabled`` flag (default False) rather than an
    unconditional ``True``.

    S3-6 wires this class into the plugin registry as metadata only — no walk
    or test invokes ``refine_blocks``. S3-7 plugs the hook into the live
    Stage 3 plugin sequencer.
    """

    name = "block_refine"
    paper = (
        "Phase C.5 block-level joint refinement — per-block AdamW refinement "
        "of the factored U/V slots + block-local RMSNorm scales on the "
        "anchored MSE objective ‖ℒ_i(X) − ℒ'_i(X')‖² "
        "(paper 2604.02119, Algorithm 2 §3.3)."
    )
    config_key = "stage3_svd.block_refine.enabled"
    reads: tuple[str, ...] = (
        "model", "teacher_model", "moe_layers", "teacher_moe_layers",
        "calib", "config", "artifacts_dir", "no_resume", "device",
    )
    # Phase C.5 refines the installed FactoredExperts U/V slots (and the
    # block-local RMSNorm scales) IN PLACE; it produces no new ctx slot.
    writes: tuple[str, ...] = ()
    provides: tuple[str, ...] = ()

    def is_enabled(self, config: dict) -> bool:
        """Return whether Phase C.5 block-refine is enabled.

        THE distinguishing trait of this plugin: it is the FIRST genuinely
        config-gated stage-3 plugin. The monolith ``run()`` gates Phase C.5
        on ``config["stage3_svd"]["block_refine"]["enabled"]`` (computed as
        ``_block_refine_enabled``); this replicates that navigation exactly,
        defaulting to False when any key in the chain is absent.
        """
        return bool(
            config.get("stage3_svd", {})
            .get("block_refine", {})
            .get("enabled", False)
        )

    def contribute_artifact(self, ctx: Any) -> dict:
        return {}

    def refine_blocks(self, ctx: PipelineContext) -> None:
        """Phase hook — Phase C.5 block-level joint refinement (S3-7a live).

        Filled at S3-7a with the VERBATIM Phase C.5 invocation from the
        monolith ``run()`` — the gated ``_phase_c5_block_refine`` call,
        including the ``teacher_model is None`` guard. This plugin is
        config-gated (``is_enabled`` on ``stage3_svd.block_refine.enabled``),
        so the orchestrator drops it from the enabled set when block-refine is
        off and ``walk_phases(("refine_blocks",), ...)`` becomes a no-op —
        byte-identical to the monolith's ``if _block_refine_enabled:`` skip.
        """
        model = ctx.get("model")
        teacher_model = ctx.get("teacher_model")
        moe_layers = ctx.get("moe_layers")
        teacher_moe_layers = ctx.get("teacher_moe_layers")
        calib = ctx.get("calib")
        config = ctx.get("config")
        artifacts_dir = ctx.get("artifacts_dir")
        no_resume = ctx.get("no_resume")
        device = ctx.get("device")

        s3 = config["stage3_svd"]
        # ---- VERBATIM Phase C.5 invocation from the monolith run() ----------
        if teacher_model is None or teacher_moe_layers is None:
            raise RuntimeError(
                "Stage 3 Phase C.5 requires the teacher model to be resident. "
                "Either disable stage3_svd.block_refine.enabled or ensure "
                "Phase A loaded the teacher (check aa_svd.cross_covariance "
                "and the resume path)."
            )
        br = s3["block_refine"]
        _phase_c5_block_refine(
            model, teacher_model, moe_layers, teacher_moe_layers, calib,
            batch_size=int(br.get("batch_size", 32)),
            learning_rate=float(br.get("learning_rate", 1.0e-4)),
            epochs=int(br.get("epochs", 25)),
            warmup_ratio=float(br.get("warmup_ratio", 0.1)),
            weight_decay=float(br.get("weight_decay", 0.0)),
            artifacts_dir=artifacts_dir, no_resume=no_resume, device=device,
        )
