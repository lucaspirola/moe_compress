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
from ...utils import cached_calibration_signals as _ccs
from ...utils.cached_calibration_signals import (
    BaseCacheProvider,
    BlockHiddenPayload,
    _resolve_sidecar_for_load,
    _validate_manifest_or_warn,
    load_block_hidden,
    sidecar_path,
)

log = logging.getLogger(__name__)


class _LazyTeacherTargetsCache:
    """Lazy ``dict[layer_idx -> Tensor]``-compatible mapping for the
    block-hidden teacher targets (Lever 3 — design (b)).

    Presence (which layers exist + manifest-validate) is established
    EAGERLY and all-or-nothing in :meth:`Stage3BlockHiddenCacheProvider.on_load`
    (stat + manifest read, NO payload load). This object holds only the
    validated ``layer_idx`` set + the reshape dims; the per-layer
    ``[n_prompts, seq_len, hidden]`` tensor is materialized on demand in
    :meth:`get` and NOT retained, so at most one layer's tensor is
    resident at a time (block_refine reads each layer exactly once).

    Contract (matches the ``dict`` API block_refine / the orchestrator
    rely on):

    * ``len(mapping)`` → the validated layer count, with NO materialization
      (orchestrator HIT-log at ``stage3/orchestrator.py:710``).
    * ``mapping.get(layer_idx)`` → the lazily reshaped tensor for a present
      key, or ``None`` for an absent key (``block_refine.py:465`` relies on
      ``dict.get`` returning ``None`` on miss).

    Byte-identity: ``get`` performs the SAME
    ``hs.reshape(n_prompts, seq_len, -1).contiguous()`` the old eager path
    did, so the consumed tensor is bit-identical — only allocation timing
    changes. ``get`` re-applies origin/main's I2 prompt-count divergence
    guard (``payload.n_prompts_in_subset != n_prompts``) using the payload
    it has already deserialized — at zero extra I/O — and for a divergent
    OR present-but-malformed sidecar (flat token count not factoring as
    ``n_prompts × seq_len``) returns the raw 2-D ``[n_tokens, hidden]``
    tensor; block_refine's per-layer guard (``cached.dim() == 3`` etc.)
    then falls through to the live teacher forward for that layer — the
    same numerically-safe fall-through origin/main forced via a full miss.
    """

    def __init__(
        self,
        jsonl_path: Path,
        layer_indices: set[int],
        n_prompts: int,
        seq_len: int,
    ) -> None:
        self._jsonl_path = jsonl_path
        self._layers = frozenset(layer_indices)
        self._n_prompts = n_prompts
        self._seq_len = seq_len

    def __len__(self) -> int:
        # Validated layer count from the eager presence scan — no
        # tensor materialization (orchestrator:710 HIT-log).
        return len(self._layers)

    def __contains__(self, layer_idx: object) -> bool:
        return layer_idx in self._layers

    def __iter__(self):
        return iter(sorted(self._layers))

    def keys(self):
        # Validated layer ids, sorted — no materialization. Lets callers
        # treat the lazy cache like the old eager dict for key inspection.
        return sorted(self._layers)

    def __getitem__(self, layer_idx: int) -> Any:
        if layer_idx not in self._layers:
            raise KeyError(layer_idx)
        tensor = self.get(layer_idx)
        if tensor is None:
            raise KeyError(layer_idx)
        return tensor

    def get(self, layer_idx: int, default: Any = None) -> Any:
        if layer_idx not in self._layers:
            return default
        payload = load_block_hidden(self._jsonl_path, int(layer_idx))
        if payload is None:
            # Presence was validated up-front; a None here means the
            # sidecar vanished between scan and read. Mirror dict.get
            # absent semantics so block_refine falls through to live.
            return default
        hs = payload.hidden_states  # [n_tokens, hidden]
        # I2 (Lever 3): prompt-count divergence guard. origin/main's
        # eager ``on_load`` checked ``n_prompts_in_subset != n_prompts``
        # up front and forced a full cache miss. The lazy presence scan
        # cannot do this without a payload load, BUT we have already
        # deserialized ``payload`` here, so ``n_prompts_in_subset`` is in
        # hand at ZERO extra I/O. The dangerous case is a writer payload
        # whose flat token count coincidentally equals n_prompts×seq_len
        # but whose prompt/seq geometry diverges: a bare reshape would
        # produce wrong-content [n_prompts, seq_len, hidden] targets that
        # block_refine's shape-only guard (dim()==3, shape[1]==seq_len)
        # would ACCEPT. Returning the raw 2-D tensor forces block_refine's
        # numerically-safe per-layer fall-through (dim()==3 rejection),
        # exactly as origin/main forced a miss.
        if int(payload.n_prompts_in_subset) != self._n_prompts:
            return hs
        n_tokens = int(hs.shape[0])
        expected = self._n_prompts * self._seq_len
        if n_tokens != expected:
            # Malformed-but-present sidecar: token count does not factor
            # as n_prompts × seq_len. Return the raw 2-D tensor so
            # block_refine's per-layer shape guard (dim()==3 check) falls
            # through to the live teacher forward for this layer. Drop
            # the payload reference; only the returned tensor survives.
            return hs
        # Same reshape the old eager path produced (former line 278).
        return hs.reshape(self._n_prompts, self._seq_len, -1).contiguous()


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

    def _presence_scan(self, jsonl_path: Path) -> set[int] | None:
        """Walk ``<jsonl_path>/sidecars/block_hidden/`` and presence-validate
        every per-layer sidecar WITHOUT loading any payload tensor (Lever 3).

        For each ``layer_NNNN.pt`` discovered, resolve its read path
        (``_resolve_sidecar_for_load``) and run the manifest schema-version
        check (``_validate_manifest_or_warn`` — reads only the sibling
        ``.MANIFEST.json``, never the multi-GB payload). Returns the set of
        validated ``layer_idx`` on success; returns ``None`` when the
        directory is absent (the ordinary cache-miss path). The all-or-
        nothing miss semantic for a *malformed manifest* is preserved by
        ``_validate_manifest_or_warn`` raising — it is NOT swallowed here.

        A torn sidecar whose manifest is absent (pre-S1 layout) passes the
        one-shot-WARN fallback in ``_validate_manifest_or_warn`` exactly as
        the old eager loader did.
        """
        # sidecar_path with a slashed signal name returns the per-layer
        # file path; the parent of that path is the sidecars/block_hidden/
        # dir we want to walk.
        any_path = sidecar_path(jsonl_path, "block_hidden/layer_0000")
        bh_dir = any_path.parent
        if not bh_dir.exists():
            return None
        present: set[int] = set()
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
            # Presence + manifest-schema validation only. No torch.load
            # of the payload tensor (the ~52 GB up-front stall this lever
            # removes). _resolve_sidecar_for_load returning None means the
            # file vanished between iterdir() and resolve -- skip it.
            resolved = _resolve_sidecar_for_load(
                jsonl_path, f"block_hidden/layer_{layer_idx:04d}"
            )
            if resolved is None:
                continue
            # Read SCHEMA_VERSIONS via the module attribute (not the import-
            # time binding) so a test/code bump of the version is honored,
            # matching load_block_hidden's behavior.
            _validate_manifest_or_warn(
                resolved,
                expected_schema_version=_ccs.SCHEMA_VERSIONS["block_hidden"],
                signal_name="block_hidden",
            )
            present.add(layer_idx)
        return present

    def on_load(
        self, ctx: PipelineContext, jsonl_path: Path
    ) -> "_LazyTeacherTargetsCache | None":
        """Presence-validate the per-layer block-hidden sidecars (Lever 3,
        design (b)); on hit populate ``ctx.teacher_targets_cache`` with a
        LAZY mapping and return it (non-None marker); on miss return ``None``.

        Schema mismatch surfaces as ``RuntimeError`` from
        ``_validate_manifest_or_warn`` -- the caller MUST NOT mask that
        exception (the message is actionable: "re-run calibration").

        All-or-nothing PRESENCE miss (preserved)
        ----------------------------------------
        Presence validation is EAGER and all-or-nothing, but via file +
        manifest EXISTENCE (``_presence_scan``), NOT by deserializing any
        payload tensor. If the sidecars dir is absent the provider returns
        ``None`` (full miss). This preserves the contract that a partial
        cache never silently mixes cached + live targets at the directory
        level. Shape / ``n_tokens`` / ``n_prompts_in_subset`` content
        validation is OUT OF SCOPE up-front -- ``load_block_hidden`` has no
        shape-only path and the manifest carries no shape field -- so a
        present-but-malformed sidecar is caught at consumption time by
        ``block_refine``'s per-layer shape guard (``block_refine.py:466-492``)
        which falls through to the live teacher forward for that layer.
        Each layer's target is independent, so a per-layer fall-through is
        numerically safe.

        Lazy materialization
        --------------------
        The ``teacher_targets_cache`` slot is a :class:`_LazyTeacherTargetsCache`
        that materializes (``load_block_hidden`` + reshape) one layer's
        ``[n_prompts, seq_len, hidden]`` tensor on ``.get(layer_idx)`` and
        retains no reference, so at most one layer is resident at a time
        (block_refine reads each layer exactly once). ``ctx.calib`` supplies
        ``(n_prompts, seq_len)`` for the reshape; if it is missing the
        provider returns ``None`` (cache miss) so block_refine falls through.
        The reader is INDEPENDENT of ``ctx.batches`` / any consumer batch
        size; block_refine slices per-batch at consumption time.
        """
        present = self._presence_scan(jsonl_path)
        if not present:
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

        lazy_cache = _LazyTeacherTargetsCache(
            jsonl_path, present, n_prompts, seq_len
        )
        ctx.set("teacher_targets_cache", lazy_cache, overwrite=True)
        log.info(
            "stage3-block-hidden-cache: presence-validated %d-layer "
            "teacher_targets_cache (n_prompts=%d, seq_len=%d, "
            "lazy un-chunked [n_prompts, seq_len, hidden] per layer) "
            "from %s",
            len(lazy_cache), n_prompts, seq_len,
            sidecar_path(jsonl_path, "block_hidden/layer_0000").parent,
        )
        return lazy_cache
