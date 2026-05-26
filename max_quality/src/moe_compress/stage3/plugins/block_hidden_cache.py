"""Stage 3 cache provider for per-MoE-block teacher hidden-state targets.

Reads the per-layer ``BlockHiddenPayload`` sidecars produced by the
``--capture-block-outputs`` calibration flag (Item 7 writer in
``vllm.calibration_block_outputs``). On cache hit, populates the
``teacher_targets_cache`` slot with a ``dict[layer_idx -> list[Tensor]]``
of per-batch ``[batch_size, seq_len, hidden]`` bf16 tensors. Stage 3
Phase C.5 (:mod:`stage3.plugins.block_refine`) checks for this slot
before running the live teacher block forward; on hit, the teacher
block forward is skipped entirely.

Architecture
------------
Provider-pair pattern per
``max_quality/docs/calibration_v2_data_capture_plan.md`` Section 0. The
per-layer sidecars store flat ``[n_tokens, hidden]`` bf16 tensors --
collected during the vLLM calibration run on a fixed N-prompt subset
(typical 128 prompts). The cache reader reshapes
``[n_tokens, hidden] -> [n_prompts, seq_len, hidden]`` assuming uniform
``seq_len`` (matches ``build_calibration_tensor`` semantics), then
chunks along dim 0 into per-batch ``[batch_size, seq_len, hidden]``
tensors. The block_refine consumer reads
``teacher_targets_cache[layer_idx]`` and shape-checks against its
``batches`` list.

Best-effort hit / fall-through on miss
--------------------------------------
The reader is LAZY -- no I/O at construction. ``on_load`` walks the
sidecars dir; if the directory or any per-layer sidecar is missing, the
cache returns ``None`` and block_refine falls through to the live
teacher forward. This is intentional: the cache is an OPTIMIZATION, not
a contract change. Stage 3 must run correctly even with no sidecars.

Token-count alignment
---------------------
The cache hit requires the writer's flat ``[n_tokens]`` count to match
the consumer's ``n_prompts × seq_len``. Stage 3's
:func:`build_calibration_tensor` derives ``(n_seq, seq_len)`` from
``calibration.num_sequences`` × ``calibration.sequence_length`` which is
independent of the vLLM-side calibration prompt source. Operators wiring
the block-hidden cache MUST configure ``calibration.num_sequences`` to
match ``--block-outputs-subset-size`` AND use a tokenization that
produces uniform ``seq_len`` (the build_calibration_tensor default).

If the math fails -- e.g. the sidecar holds ``n_tokens=300K`` and the
Stage 3 calibration spec asks for 128 × 2048 = 262K tokens -- the
provider logs an actionable warning and returns ``None`` (cache miss).
A future schema bump could store ``seq_len`` directly to short-circuit
the inference; v1 keeps the dataclass minimal and infers from the slot.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from ...pipeline.context import PipelineContext
from ...utils.cached_calibration_signals import (
    BaseCacheProvider,
    BlockHiddenPayload,
    load_block_hidden,
    sidecar_path,
)

log = logging.getLogger(__name__)


class Stage3BlockHiddenCacheProvider(BaseCacheProvider):
    """Cache-side provider for the Item-7 block-hidden sidecars.

    On hit, populates ``ctx.teacher_targets_cache`` with a
    ``dict[layer_idx -> list[Tensor]]`` of per-batch
    ``[batch_size, seq_len, hidden]`` bf16 tensors and returns a
    non-None marker so the Stage 3 orchestrator's ``dispatch_first``
    call sees a winner. On miss (sidecars dir missing, any per-layer
    file missing, or token-count alignment failure) returns ``None`` and
    leaves ctx untouched so block_refine can fall through to the live
    teacher block forward.

    Slot contract: ``teacher_targets_cache: dict[int, list[Tensor]]``
    where the outer key is the MoE layer_idx and the inner list contains
    one ``[batch_size, seq_len, hidden]`` bf16 CPU tensor per batch (the
    consumer moves each entry to device just-in-time inside the per-
    batch loop, identical to the live ``teacher_targets`` allocation
    pattern in :func:`stage3.plugins.block_refine._phase_c5_block_refine`).
    """

    name: str = "stage3_block_hidden_cache"
    paper: str = (
        "Cache provider for the V2 block-hidden writer "
        "(calibration-v2 Item 7). Reads "
        "sidecars/block_hidden/layer_{idx:04d}.pt and populates "
        "ctx.teacher_targets_cache so Stage 3 Phase C.5 skips the live "
        "teacher block forward."
    )
    config_key: str = "calibration.block_hidden_cache"
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ("teacher_targets_cache",)
    provides: tuple[str, ...] = ()

    def is_enabled(self, config: dict) -> bool:
        return True

    def contribute_artifact(self, ctx: PipelineContext) -> dict:
        return {}

    def _load_layers(self, jsonl_path: Path) -> dict[int, BlockHiddenPayload]:
        """Walk ``<jsonl_path>/sidecars/block_hidden/`` and load every
        per-layer payload found.

        Returns an empty dict if the directory doesn't exist; this is the
        ordinary cache-miss path. Missing individual sidecars within an
        existing directory is also a miss (the consumer needs every MoE
        layer, not a partial set).
        """
        # sidecar_path with a slashed signal name returns the per-layer
        # file path; the parent of that path is the sidecars/block_hidden/
        # dir we want to walk.
        any_path = sidecar_path(jsonl_path, "block_hidden/layer_0000")
        bh_dir = any_path.parent
        if not bh_dir.exists():
            return {}
        loaded: dict[int, BlockHiddenPayload] = {}
        for child in sorted(bh_dir.iterdir()):
            if not child.is_file() or not child.name.startswith("layer_"):
                continue
            if not child.name.endswith(".pt"):
                continue
            try:
                idx_str = child.stem.removeprefix("layer_")
                layer_idx = int(idx_str)
            except ValueError:
                continue
            payload = load_block_hidden(jsonl_path, layer_idx)
            if payload is None:
                # Should not happen given iterdir() saw the file, but
                # treat as a partial miss.
                continue
            loaded[layer_idx] = payload
        return loaded

    def on_load(
        self, ctx: PipelineContext, jsonl_path: Path
    ) -> BlockHiddenPayload | None:
        """Try to load the per-layer block-hidden sidecars; on hit
        populate ``ctx.teacher_targets_cache`` and return the first
        payload (non-None marker); on miss return ``None``.

        Schema mismatch surfaces as ``ValueError`` from
        ``load_block_hidden`` -- the caller MUST NOT mask that exception
        (the message is actionable: "Delete the sidecar to regenerate").

        Token-count alignment + seq_len inference
        -----------------------------------------
        ``ctx`` SHOULD have ``calib`` (an ``[n_prompts, seq_len]`` int64
        token tensor) bound; the provider reads the shape to determine
        ``n_prompts`` and ``seq_len``, then validates against the flat
        ``[n_tokens]`` row count on each sidecar tensor. The
        ``batches`` and ``batch_size`` slots are read from ctx to chunk
        into per-batch tensors. If any of these slots is missing or the
        token-count math doesn't divide cleanly, the provider returns
        ``None`` (cache miss).
        """
        payloads = self._load_layers(jsonl_path)
        if not payloads:
            return None

        # Read alignment-relevant slots from ctx; bail to miss on any
        # missing slot so block_refine falls through to the live forward
        # rather than surfacing a confusing KeyError.
        calib = ctx.get("calib") if ctx.has("calib") else None
        if calib is None or calib.dim() != 2:
            log.warning(
                "stage3-block-hidden-cache: ctx.calib unavailable or "
                "wrong rank -- cannot infer (n_prompts, seq_len); "
                "treating as cache miss.",
            )
            return None
        n_prompts, seq_len = int(calib.shape[0]), int(calib.shape[1])

        # The block_refine consumer iterates over ``batches`` (a list of
        # token slices); the cache must mirror that chunking. Read batch
        # size from ctx; default 1 if absent.
        batch_size = 1
        if ctx.has("batches"):
            batches = ctx.get("batches")
            if batches:
                try:
                    batch_size = int(batches[0].shape[0])
                except (AttributeError, IndexError):
                    batch_size = 1

        # n_batches = floor(n_prompts / batch_size) -- matches
        # block_refine's drop_last semantics.
        n_batches = n_prompts // batch_size
        if n_batches == 0:
            log.warning(
                "stage3-block-hidden-cache: n_prompts=%d < batch_size=%d "
                "-- no full batches; treating as cache miss.",
                n_prompts, batch_size,
            )
            return None

        # Per-layer reshape + chunk. Bail to miss on any token-count
        # mismatch; the live forward is still correct.
        teacher_targets_cache: dict[int, list[torch.Tensor]] = {}
        first_payload: BlockHiddenPayload | None = None
        for layer_idx, payload in payloads.items():
            hs = payload.hidden_states           # [n_tokens, hidden]
            n_tokens = int(hs.shape[0])
            expected = n_prompts * seq_len
            if n_tokens != expected:
                log.warning(
                    "stage3-block-hidden-cache: layer_idx=%d sidecar has "
                    "n_tokens=%d but ctx.calib implies n_prompts=%d × "
                    "seq_len=%d = %d. Token count mismatch -- cache "
                    "miss. (Operator: align calibration.num_sequences "
                    "with --block-outputs-subset-size and "
                    "calibration.sequence_length with the writer's "
                    "actual per-prompt length.)",
                    layer_idx, n_tokens, n_prompts, seq_len, expected,
                )
                return None
            # Reshape to [n_prompts, seq_len, hidden] then carve into
            # per-batch [batch_size, seq_len, hidden] tensors -- byte-
            # identical to the live ``teacher_targets.append(out.detach()
            # .to(dtype=torch.bfloat16, device="cpu"))`` shape contract
            # in _phase_c5_block_refine.
            reshaped = hs.reshape(n_prompts, seq_len, -1)
            batches_list: list[torch.Tensor] = []
            for b in range(n_batches):
                start = b * batch_size
                end = start + batch_size
                batches_list.append(reshaped[start:end].contiguous())
            teacher_targets_cache[int(layer_idx)] = batches_list
            if first_payload is None:
                first_payload = payload

        ctx.set("teacher_targets_cache", teacher_targets_cache, overwrite=True)
        log.info(
            "stage3-block-hidden-cache: hydrated %d-layer × %d-batch "
            "teacher_targets_cache (n_prompts=%d, seq_len=%d, "
            "batch_size=%d) from %s",
            len(teacher_targets_cache), n_batches, n_prompts, seq_len,
            batch_size,
            sidecar_path(jsonl_path, "block_hidden/layer_0000").parent,
        )
        return first_payload
