"""AA-SVD block-level joint refinement (Algorithm 2 §3.3, gated).

Paper
-----
"AA-SVD: Activation-Aware SVD" — arXiv:2604.02119 Algorithm 2 §3.3.
audit/spec_compliance/01_papers/2604.02119/source.md.

After each block's per-(layer, expert) factorization
(:mod:`stage3.plugins.aa_svd_factor`), all factorized weight factors
``{U_j, V_j}`` and block-local RMSNorm scale parameters ``θ_i`` are
jointly optimized via AdamW (lr=1e-4, 25 epochs, cosine schedule,
batch 32) to minimize block output MSE against the original model:

    min ‖ℒ_i(X) − ℒ'_i(X')‖²

where ``ℒ_i`` is the teacher's block-``i`` forward, ``ℒ'_i`` is the
student's block-``i`` forward with factorized weights, and ``X / X'``
are the teacher's / student's block-``i`` inputs (cross-block cascade
restoration — see deviation D-no-intra-block-cascade at
:mod:`stage3.plugins.aa_svd_factor`).

This plugin is **gated** by ``block_refine.enabled`` and only fires
on MoE decoder blocks (see deviation D-c5-moe-only below).

Official code
-------------
``atulkumarin/AA-SVD`` @ commit
``1fa1b686cd9b13a77607a676564e37d438a176c8`` (2026-04-22) —
github.com/atulkumarin/AA-SVD. Cross-checked against the project's
Phase C.5 implementation for the per-block AdamW loop + stream-advance
mechanics.

Deviation: D-c5-moe-only
------------------------
Paper Algorithm 2 (lines 2-11) iterates the block refinement over
**every** block ``L_i ∈ M``; the paper applies the joint AdamW
objective to each block's full parameter set (factorized factors +
RMSNorm scales).

This plugin's Stage 3 only factorizes MoE expert matrices
(``gate_proj`` / ``up_proj`` / ``down_proj``) per project §10
Protected Components; attention projections, embeddings, lm_head,
shared experts, and dense (non-MoE) decoder layers are untouched.
Phase C.5 therefore only fires on MoE decoder layers, where there are
factorized ``{U_j, V_j}`` to update. Non-MoE decoder layers (if any
in the architecture) participate in the stream-advance forward but
skip the AdamW refinement; their RMSNorm scales remain frozen.

Rationale: the skipped quality lift is bounded — dense interlayers
contribute only RMSNorm scale corrections to the paper's objective.
For Qwen3-30B-A3B (the project's target model) every non-shared
decoder layer is MoE, so the deviation is **vacuous in practice** —
every layer with refinable factors is refined. The deviation is named
explicitly so a future port to a mixed dense/MoE architecture cannot
silently drop the dense-block refinement step.

Deviation: D-c5-no-bias
-----------------------
Paper §3.3 / Appendix B.2 names θ_i as "normalization scales **and
biases**" for each block. This plugin currently collects only the
RMSNorm scale parameters (``input_layernorm`` / ``post_attention_layernorm`` /
``self_attn.q_norm`` / ``self_attn.k_norm`` — see ``norm_module_paths``)
and not block-local ``*.bias`` parameters.

**Vacuous for Qwen3-30B-A3B**: the target architecture has
``attention_bias=False`` (no bias on q/k/v/o projections), RMSNorm has
no bias by construction, and the MoE expert factorization replaces
gate/up/down ``nn.Linear`` modules with biasless ``FactoredExperts``
slots. There are no block-local bias parameters to train.

The deviation is named so a future port to an architecture that DOES
carry biases (e.g. Llama-2's ``mlp.gate_proj.bias`` if a variant ships
biases, or any model with ``attention_bias=True``) cannot silently
drop bias updates from θ_i.

Deviation: D-c5-lr-offset
-------------------------
Upstream ``finetune_layer`` (commit 1fa1b686, llama_adapter.py L226-L238)
defines its LR schedule with the raw step index ``step`` (0-indexed):
``step / warmup_steps`` during warmup and
``(step - warmup_steps) / (total_steps - warmup_steps)`` after.

This plugin uses ``s = step + 1`` and ``total_steps - warmup_steps + 1``
in the denominator so the very first step uses a non-zero warmup
fraction (``1 / warmup_steps``) rather than literally ``0``, and the
cosine tail never lands exactly on the floor at the last step. Paper
§B.2 prose ("cosine schedule with linear warmup") does not pin a
specific edge convention, but the pinned upstream code is the
deviation reference and uses no offset.

Numerical impact is bounded (one step of warmup ramp shifted by 1/N).
Named explicitly per project §6 "deviations from pinned upstream are
named".

Naming-history note
-------------------
"Phase C.5" (legacy Stage 3 monolith terminology) is naming-historical.
The current plugin architecture has no phase taxonomy; new prose
drops the labels. Existing log lines / Trackio keys preserved for
dashboard back-compat.

Tool inventory (relocated verbatim):

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
False) rather than an unconditional ``True``. S3-7a wired it into the live
Stage 3 plugin sequencer (``stage3.orchestrator`` —
``walk_phases(("refine_blocks",), ...)``); when the gate is off the plugin
is dropped from the enabled set and the walk is a no-op.
"""
from __future__ import annotations

import logging
import math
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from ...utils.model_io import (
    MATRIX_NAMES,
    FactoredExperts,
    MoELayerRef,
    iter_decoder_layers,
)
from ...utils.trackio_log import trackio_log as _trackio_log

if TYPE_CHECKING:
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
    teacher_targets_cache: "dict[int, torch.Tensor] | None" = None,
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

    # Levers 1+2 are a CUDA-only optimization: ``.pin_memory()`` allocates
    # non-pageable host memory via the CUDA driver, so it raises
    # ``RuntimeError: No CUDA GPUs are available`` on a CPU-only host (or
    # ``CUDA_VISIBLE_DEVICES=""``). Pinning is also pointless when the
    # consuming ``.to(device, non_blocking=True)`` targets CPU (a CPU->CPU
    # copy is never an async H2D). Gate the pin on the resolved device type:
    # on CUDA we pin (byte-identical, async H2D); on CPU it is the identity,
    # which loses nothing.
    _pin = (
        (lambda t: t.pin_memory())
        if (getattr(device, "type", device) == "cuda")
        else (lambda t: t)
    )

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
    #
    # Precondition (enforced by the upstream calibration assembly): every
    # batch is the SAME shape ``(batch_size, seq_len)`` carved from one
    # ``calib_tensor`` of shape ``(n_seq, seq_len)`` (see ``batches``
    # construction above with ``drop_last`` semantics). Causal masks,
    # ``position_ids``, and rotary ``position_embeddings`` are pure functions
    # of that shape — not of token-id contents — so the batch-0 capture is
    # bit-identical to the per-batch recapture for batches 1..N-1. A divergent
    # mask/position tensor would imply variable seq_len across batches, which
    # would have already failed the ``batches`` slice. The recapture is
    # therefore omitted as an optimization, not a correctness hazard.
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
            # Levers 1+2: pin the bf16 CPU source so the per-step H2D in the
            # AdamW loop (non_blocking=True) is async. .pin_memory() returns
            # a byte-identical copy in non-pageable host memory; values are
            # unchanged.
            captured[cur_idx[0]] = _pin(
                t.detach().to(dtype=torch.bfloat16, device="cpu")
            )
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

        # Project §6 (NOT paper §B.2): AdamW must run with fp32 moments +
        # fp32 master weights. The paper (arXiv:2604.02119 Appendix B.2)
        # only names "AdamW, lr=1e-4, 25 epochs, batch=32" — it does not
        # specify optimizer-state precision. The fp32-moments / fp32-master
        # convention is a PROJECT decision (project §6 numerical-stability
        # rules) and is paper-faithful by construction (a strictly more
        # precise realization of the same optimizer).
        #
        # Vanilla `torch.optim.AdamW` initializes `exp_avg`/`exp_avg_sq`
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
            # LR schedule — linear warmup then cosine decay with a 10% floor,
            # matching upstream AA-SVD finetune_layer (commit 1fa1b686,
            # llama_adapter.py L226-L238):
            #     0.1 + 0.9 * 0.5 * (1 + cos(pi * progress))
            # i.e. the cosine never decays below 10% of the peak learning rate.
            # See deviation D-c5-lr-offset for the (s = step + 1) edge handling.
            s = step + 1
            if s <= warmup_steps:
                return s / max(1, warmup_steps)
            progress = (s - warmup_steps) / max(1, total_steps - warmup_steps + 1)
            return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

        # Pre-compute teacher targets once per batch (no grad). The V2
        # block-hidden cache (calibration-v2 Item 7) can substitute these
        # tensors directly, skipping the live teacher block forward
        # entirely on a cache hit. The cache holds ONE un-chunked
        # ``[n_prompts, seq_len, hidden]`` bf16 CPU tensor per layer
        # (decoupled from the consumer's batch_size to dodge the C1
        # hazard where the writer would have to know Stage 3's
        # block_refine.batch_size). The consumer slices
        # ``cached[bi*bs:(bi+1)*bs]`` per batch index here; each slice
        # carries the SAME shape contract as the live path's
        # ``out.detach().to(dtype=torch.bfloat16, device="cpu")``.
        # Shape-mismatch on a cached entry falls through to the live
        # path with a warning (a malformed cache must not corrupt the
        # refinement objective).
        teacher_targets: list[torch.Tensor] | None = None
        if teacher_targets_cache is not None:
            cached = teacher_targets_cache.get(int(layer_idx))
            n_seq_cal, seq_len_cal = int(calib_tensor.shape[0]), int(calib_tensor.shape[1])
            if (
                cached is not None
                and cached.dim() == 3
                and cached.shape[0] >= n_batches * batch_size
                and cached.shape[1] == seq_len_cal
            ):
                # Levers 1+2: pin the per-batch slice copy (NOT the whole
                # cached tensor) so the per-step H2D below is async. The
                # contiguous() slice is already a fresh bf16 CPU tensor;
                # .pin_memory() makes it non-pageable with identical bytes.
                teacher_targets = [
                    _pin(
                        cached[bi * batch_size:(bi + 1) * batch_size]
                        .contiguous()
                    )
                    for bi in range(n_batches)
                ]
                log.info(
                    "Stage 3 Phase C.5 layer %d: using cached teacher "
                    "targets (%d batches sliced from un-chunked "
                    "[%d, %d, %d]; skipping live teacher block forward)",
                    layer_idx, n_batches,
                    int(cached.shape[0]), int(cached.shape[1]),
                    int(cached.shape[2]),
                )
            elif cached is not None:
                log.warning(
                    "Stage 3 Phase C.5 layer %d: cached teacher targets "
                    "shape mismatch (got %s, expected un-chunked "
                    "[>=%d, %d, hidden]) -- falling through to live "
                    "teacher forward.",
                    layer_idx, tuple(cached.shape),
                    n_batches * batch_size, seq_len_cal,
                )

        if teacher_targets is None:
            teacher_targets = []
            with torch.no_grad():
                for bi, _ in enumerate(batches):
                    x_t = X_teacher[bi].to(device=device, dtype=student_dtype)
                    out = t_layer(x_t, **teacher_kwargs_all.get(layer_idx, {}))
                    if isinstance(out, tuple):
                        out = out[0]
                    # Levers 1+2: pinned bf16 CPU source for the async
                    # per-step H2D below. Byte-identical to the pageable copy.
                    teacher_targets.append(
                        _pin(out.detach().to(dtype=torch.bfloat16, device="cpu"))
                    )

        # AdamW loop.
        loss_first: float | None = None
        loss_last: float | None = None
        step = 0
        for epoch in range(epochs):
            for bi, _ in enumerate(batches):
                # Levers 1+2: the source tensors are pinned (produced at
                # capture/teacher-target sites), so non_blocking=True makes
                # this per-step H2D async + overlappable on the default
                # stream. Same stream + no RNG => the consuming kernel sees
                # bit-identical values; op order and AdamW step order are
                # unchanged. (pin + non_blocking are inseparable: pageable
                # + non_blocking is silently synchronous.)
                x_s = X_student[bi].to(device=device, dtype=student_dtype, non_blocking=True)
                target = teacher_targets[bi].to(device=device, dtype=student_dtype, non_blocking=True)
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
                # Gradient clipping — matches upstream AA-SVD finetune_layer
                # (commit 1fa1b686, llama_adapter.py L291): clip every step
                # to ``max_norm=1.0``. Stabilizes AdamW updates against the
                # occasional large MSE gradient spike on a freshly factored
                # block. Scope: the trainable subset only (factored U/V slots
                # + RMSNorm scales); frozen params have no grad to clip.
                torch.nn.utils.clip_grad_norm_(trainables, max_norm=1.0)
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

        # ``loss_first`` / ``loss_last`` are populated on EVERY refined block
        # (the outer ``epochs * len(batches)`` loop runs at least once — see
        # ``total_steps = max(1, ...)`` and the ``n_batches == 0`` guard
        # above). Defensive ``is None`` fallbacks below protect against a
        # future refactor where the loop becomes skippable; an exact-zero
        # loss is a valid numeric outcome (self-distillation degenerate
        # case) and MUST NOT collapse to the ``0.0`` placeholder branch.
        _lf = loss_first if loss_first is not None else 0.0
        _ll = loss_last if loss_last is not None else 0.0
        rel_drop = (_lf - _ll) / max(_lf, 1e-12) if loss_first is not None else 0.0
        log.info("  Phase C.5 block %d/%d (idx=%d) loss %.4e → %.4e (%.1f%%↓)",
                 block_pos + 1, n_blocks, layer_idx,
                 _lf, _ll, 100 * rel_drop)
        _trackio_log({
            "stage3/c5_layer_idx": float(layer_idx),
            "stage3/c5_loss_init": _lf,
            "stage3/c5_loss_final": _ll,
            "stage3/c5_loss_rel_drop": rel_drop,
            # Additive: training-loop shape and warmup configuration. All in scope.
            "stage3/c5_total_steps": int(total_steps),
            "stage3/c5_warmup_steps": int(warmup_steps),
            "stage3/c5_trainable_param_count": int(len(trainables)),
        })

        # Save per-block checkpoint atomically.
        #
        # Save/resume symmetry invariant: the set of norm-module paths
        # iterated here MUST exactly match the set iterated by the resume
        # branch above (search ``input_layernorm`` in this file). Resume
        # restores only keys present in the payload — if a key is added
        # here without being added there, resumed runs would silently drop
        # the refined norm scale; if a key is added there without being
        # added here, the resume guard ``path in payload`` skips it
        # cleanly. The MATRIX_NAMES iteration shares the source-of-truth
        # constant on both sides, so U/V symmetry is automatic.
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
            os.replace(tmp, ckpt_path)

        # Advance streams for the next block (no grad).
        X_student, X_teacher = _advance_streams(
            s_layer, t_layer, X_student, X_teacher,
            student_kwargs_all.get(layer_idx, {}),
            teacher_kwargs_all.get(layer_idx, {}), device,
        )

    # Cleanup: remove checkpoint dir on success.
    if partial_dir is not None and partial_dir.exists():
        shutil.rmtree(partial_dir, ignore_errors=True)
        log.info("Stage 3 Phase C.5: removed checkpoint dir (run completed cleanly)")


def _advance_streams(s_layer, t_layer, X_student, X_teacher,
                     s_kwargs, t_kwargs, device):
    """Forward both layers (no grad) on each batch's current stream and return
    the next-block-input tensors as bf16 CPU lists."""
    new_s: list[torch.Tensor] = []
    new_t: list[torch.Tensor] = []
    student_dtype = next(s_layer.parameters()).dtype
    teacher_dtype = next(t_layer.parameters()).dtype
    # Levers 1+2 are CUDA-only (``.pin_memory()`` needs the CUDA driver and is
    # pointless for a CPU->CPU copy); gate the pin on the resolved device type
    # so the CPU device path does not raise ``No CUDA GPUs are available``.
    _pin = (
        (lambda t: t.pin_memory())
        if (getattr(device, "type", device) == "cuda")
        else (lambda t: t)
    )
    with torch.no_grad():
        for x_s_cpu, x_t_cpu in zip(X_student, X_teacher):
            x_s = x_s_cpu.to(device=device, dtype=student_dtype)
            out_s = s_layer(x_s, **s_kwargs)
            if isinstance(out_s, tuple):
                out_s = out_s[0]
            # Levers 1+2: pin the next-block bf16 CPU inputs so the next
            # block's per-step H2D (non_blocking=True) is async. Byte-identical.
            new_s.append(_pin(out_s.detach().to(dtype=torch.bfloat16, device="cpu")))
            x_t = x_t_cpu.to(device=device, dtype=teacher_dtype)
            out_t = t_layer(x_t, **t_kwargs)
            if isinstance(out_t, tuple):
                out_t = out_t[0]
            new_t.append(_pin(out_t.detach().to(dtype=torch.bfloat16, device="cpu")))
    return new_s, new_t


class BlockRefinePlugin:
    """Stage 3 Phase C.5 block-refine plugin (S3-7a — live in orchestrator).

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

    S3-7a wired this plugin into the live Stage 3 sequencer:
    ``stage3.orchestrator`` invokes ``walk_phases(("refine_blocks",), ...)``
    after the factorization loop. When the config gate is off, the plugin
    is dropped from the enabled set and the walk is a no-op — byte-identical
    to the monolith's ``if _block_refine_enabled:`` skip.
    """

    name = "block_refine"
    paper = (
        "AA-SVD Algorithm 2 §3.3 block-level joint refinement — "
        "arXiv:2604.02119 (atulkumarin/AA-SVD @ "
        "1fa1b686cd9b13a77607a676564e37d438a176c8). "
        "Per-block AdamW MSE refinement of factored {U_j, V_j} + "
        "block-local RMSNorm scales. Deviation D-c5-moe-only "
        "(MoE-only — vacuous on Qwen3-30B-A3B). See module docstring."
    )
    config_key = "stage3_svd.block_refine.enabled"
    reads: tuple[str, ...] = (
        "model", "teacher_model", "moe_layers", "teacher_moe_layers",
        "calib", "config", "artifacts_dir", "no_resume", "device",
        # Optional V2 block-hidden cache populated by
        # Stage3BlockHiddenCacheProvider via dispatch_first("on_load",
        # ...) in the orchestrator. Declared in ``reads`` so the
        # plugin-registry contract reflects the actual dataflow; the
        # ``refine_blocks`` hook reads it with a ``ctx.has`` guard so
        # absence is a benign cache miss (live teacher forward), not a
        # contract violation. (I3.)
        "teacher_targets_cache",
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
        # Optional V2 block-hidden cache populated by
        # Stage3BlockHiddenCacheProvider via dispatch_first("on_load", ...)
        # in the orchestrator. None on cache miss; the live teacher
        # forward runs unchanged in that case.
        teacher_targets_cache = (
            ctx.get("teacher_targets_cache")
            if ctx.has("teacher_targets_cache") else None
        )

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
        # ``weight_decay`` default 0.01: matches pinned upstream AA-SVD
        # ``finetune_layer`` (commit 1fa1b686, llama_adapter.py L226:
        # ``torch.optim.AdamW(..., weight_decay=0.01)``). Paper §B.2 names
        # AdamW but not the WD value, so upstream is authoritative. Configs
        # that set ``stage3_svd.block_refine.weight_decay`` explicitly are
        # honored verbatim.
        _phase_c5_block_refine(
            model, teacher_model, moe_layers, teacher_moe_layers, calib,
            batch_size=int(br.get("batch_size", 32)),
            learning_rate=float(br.get("learning_rate", 1.0e-4)),
            epochs=int(br.get("epochs", 25)),
            warmup_ratio=float(br.get("warmup_ratio", 0.1)),
            weight_decay=float(br.get("weight_decay", 0.01)),
            artifacts_dir=artifacts_dir, no_resume=no_resume, device=device,
            teacher_targets_cache=teacher_targets_cache,
        )
