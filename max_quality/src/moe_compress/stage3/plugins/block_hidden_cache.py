"""Stage 3 cache provider for per-MoE-block teacher hidden-state targets.

Reads the per-layer ``BlockHiddenPayload`` sidecars produced by the
``--capture-block-outputs`` calibration flag (Item 7 writer in
``vllm.calibration_block_outputs``). On cache hit, populates the
``teacher_targets_cache`` slot with a ``dict[layer_idx -> Tensor]``
of un-chunked ``[n_prompts, seq_len, hidden]`` bf16 tensors. Stage 3
Phase C.5 (:mod:`stage3.plugins.block_refine`) checks for this slot
before running the live teacher block forward; on hit, the per-layer
loop slices ``cached[bi*bs:(bi+1)*bs]`` per batch index and skips the
live teacher block forward entirely.

Architecture
------------
Provider-pair pattern per
``max_quality/docs/calibration_v2_data_capture_plan.md`` Section 0. The
per-layer sidecars store flat ``[n_tokens, hidden]`` bf16 tensors --
collected during the vLLM calibration run on a fixed N-prompt subset
(typical 128 prompts). The cache reader reshapes
``[n_tokens, hidden] -> [n_prompts, seq_len, hidden]`` assuming uniform
``seq_len`` (matches ``build_calibration_tensor`` semantics) and stores
the un-chunked per-layer tensor. The block_refine consumer owns the
chunking (slicing per-batch from the un-chunked tensor using its own
``batch_size``) so the cache is decoupled from the consumer's batch
size. This avoids the C1 hazard where the reader would chunk with
``ctx.batches[0].shape[0]`` (driven by ``stage3_svd.batch_size`` /
``bcov_batch_size``, default 1) while the consumer chunks with
``stage3_svd.block_refine.batch_size`` (default 32) -- a silent shape
mismatch that would have caused every layer's cache entry to be
rejected at consumption time.

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

Prompt-identity matching is OPERATOR responsibility (I1+I2)
-----------------------------------------------------------
The cache is keyed on ``(jsonl_path, n_prompts, seq_len)``, NOT on the
actual prompt contents. The writer captures the FIRST N prompts in the
calibration-driver's queue order (chunk-boundary effects may cause it
to capture marginally more than ``--block-outputs-subset-size`` -- see
the writer's ``dump_block_outputs`` docstring); Stage 3's
:func:`build_calibration_tensor` builds its own calibration tensor via
a seeded sampler whose ordering may differ.

A cache hit that satisfies the token-count check could therefore
produce WRONG targets for unrelated prompts if the two pipelines are
fed from different calibration sources. The reader defends against the
specific case where the prompt COUNTS diverge (``payload.n_prompts_in_subset
!= n_prompts``) by falling through to the live forward with a warning;
content-level divergence cannot be detected from the sidecar metadata
alone.

Operator contract: run the calibration writer (``--capture-block-outputs``)
AND Stage 3 from the SAME calibration JSONL source, with matching
``calibration.num_sequences`` / ``--block-outputs-subset-size`` /
``calibration.sequence_length`` so the byte content of the writer's
captured-prompt-set is bit-identical to what
``build_calibration_tensor`` returns.
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
    ``dict[layer_idx -> Tensor]`` of un-chunked
    ``[n_prompts, seq_len, hidden]`` bf16 tensors and returns a
    non-None marker so the Stage 3 orchestrator's ``dispatch_first``
    call sees a winner. On miss (sidecars dir missing, any per-layer
    file missing, token-count alignment failure, or prompt-count
    divergence) returns ``None`` and leaves ctx untouched so
    block_refine can fall through to the live teacher block forward.

    Slot contract: ``teacher_targets_cache: dict[int, Tensor]``
    where the outer key is the MoE layer_idx and the value is a single
    un-chunked ``[n_prompts, seq_len, hidden]`` bf16 CPU tensor (the
    consumer slices ``cached[bi*bs:(bi+1)*bs]`` per batch index and
    moves each slice to device just-in-time inside the per-batch loop,
    identical to the live ``teacher_targets`` allocation pattern in
    :func:`stage3.plugins.block_refine._phase_c5_block_refine`). The
    un-chunked layout DECOUPLES the cache from the consumer's
    ``batch_size`` so the writer (which has no knowledge of Stage 3's
    block-refine batch size) cannot produce a shape-mismatched entry.
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
        ``[n_tokens]`` row count on each sidecar tensor. If the calib
        slot is missing or the token-count math doesn't divide cleanly
        the provider returns ``None`` (cache miss). The reader is
        explicitly INDEPENDENT of ``ctx.batches`` / any consumer batch
        size -- the cache stores an un-chunked
        ``[n_prompts, seq_len, hidden]`` tensor per layer and the
        block_refine consumer slices per-batch at consumption time.

        Prompt-count divergence check (I2)
        ----------------------------------
        The writer's ``payload.n_prompts_in_subset`` records the actual
        number of prompts the writer captured (which may exceed the
        ``--block-outputs-subset-size`` cap by a chunk-boundary
        remainder -- see the writer's docstring). If this differs from
        ``ctx.calib.shape[0]`` the two pipelines are likely fed from
        different calibration sources and the tensor contents would
        bear the WRONG targets even if the per-layer token counts
        happen to match. The reader falls through to the live forward
        (returns ``None``, logs a warning) in that case.
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

        # Per-layer reshape only -- no per-batch chunking. The
        # block_refine consumer slices its own batches at consumption
        # time using its own ``batch_size``, so the cache is decoupled
        # from any specific batch-size configuration (C1 fix). Bail to
        # miss on any token-count or prompt-count mismatch; the live
        # forward is still correct.
        teacher_targets_cache: dict[int, torch.Tensor] = {}
        first_payload: BlockHiddenPayload | None = None
        for layer_idx, payload in payloads.items():
            # I2: prompt-count divergence guard. ``n_prompts_in_subset``
            # is the actual number of prompts the writer captured; if
            # it differs from the Stage 3 calibration tensor's
            # ``n_prompts`` the two pipelines are reading different
            # subsets and the per-prompt content would not align even
            # when the per-layer token counts happen to coincide.
            if int(payload.n_prompts_in_subset) != n_prompts:
                log.warning(
                    "stage3-block-hidden-cache: layer_idx=%d sidecar has "
                    "n_prompts_in_subset=%d but ctx.calib.shape[0]=%d. "
                    "Prompt-count divergence -- cache miss. (Operator: "
                    "run --capture-block-outputs and Stage 3 from the "
                    "SAME calibration source with matching "
                    "calibration.num_sequences / "
                    "--block-outputs-subset-size.)",
                    layer_idx, int(payload.n_prompts_in_subset), n_prompts,
                )
                return None

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
            # Reshape to [n_prompts, seq_len, hidden] and store
            # un-chunked. The block_refine consumer slices
            # ``cached[bi*bs:(bi+1)*bs]`` per batch index.
            reshaped = hs.reshape(n_prompts, seq_len, -1).contiguous()
            teacher_targets_cache[int(layer_idx)] = reshaped
            if first_payload is None:
                first_payload = payload

        ctx.set("teacher_targets_cache", teacher_targets_cache, overwrite=True)
        log.info(
            "stage3-block-hidden-cache: hydrated %d-layer "
            "teacher_targets_cache (n_prompts=%d, seq_len=%d, "
            "un-chunked [n_prompts, seq_len, hidden]) from %s",
            len(teacher_targets_cache), n_prompts, seq_len,
            sidecar_path(jsonl_path, "block_hidden/layer_0000").parent,
        )
        return first_payload
