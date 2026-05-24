"""Stage-agnostic one-forward-pass calibration-pass engine.

This module exposes :class:`CalibrationEngine` — a declarative wrapper around
the existing primitives in :mod:`moe_compress.utils.activation_hooks`. Detector
plugins (Stage 1 detectors, Stage 2 detectors, and beyond) register their
accumulators with the engine declaring *what* per-expert / per-batch hook
channels they need; the engine handles *how* (which ``contextlib.ExitStack``
context managers to enter, in what order, how to multiplex callbacks, how to
drain per-batch state).

The engine is the single source of truth for the profiling forward pass: it is
a one-forward-pass accumulator multiplexer. Multiple accumulators that each need
their own hook channels share a single model forward — the engine installs every
declared channel exactly once and fans events out to all subscribers. It is
stage-agnostic by design; it knows nothing about the *meaning* of any
accumulator, only how to feed it.

Public surface (import directly from
``moe_compress.tools.calibration_pass``)::

    HookKind                # enum: per-expert + per-batch channels
    HookSpec                # frozen dataclass: declarative hook bundle
    PerBatchContext         # dataclass passed to per-batch handlers
    AccumulatorRegistration # one row of the registration table
    CalibrationEngine       # the driver
    run_calibration_pass    # thin convenience wrapper around the driver
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Iterable

import torch

from ..utils.activation_hooks import (
    capture_router_outputs,
    instrument_experts,
    run_calibration,
)


# ---------------------------------------------------------------------------
# Channel enum
# ---------------------------------------------------------------------------


class HookKind(str, Enum):
    """Declarative hook intent — what the engine must wire on behalf of the accumulator.

    Membership is intentionally small and additive: every supported plumbing
    pattern is one enum value. Detector plugins (Stage 1 sub-tasks 6-9 and
    future Stage 2 detectors) declare which value(s) they need; the engine
    knows nothing about the *meaning* of an accumulator, only how to feed it.

    Values
    ------
    DOWN_PROJ
        The accumulator wants the ``down`` callback from
        :func:`~moe_compress.utils.activation_hooks.instrument_experts` for
        every MoE layer. The engine multiplexes a single ``down`` callback
        across all DOWN_PROJ accumulators (matches the legacy ``down_cb``
        which feeds BOTH :class:`DownProjMaxAccumulator` and
        :class:`ExpertOutputAccumulator`).
    EXPERT_INPUT, EXPERT_INTERMEDIATE, EXPERT_GATE_UP_OUT
        The four other callback channels accepted by ``instrument_experts``
        (see its docstring at ``utils/activation_hooks.py:1233-1240``). Not
        used by Stage 1 today but added now so future detectors (Stage 2's
        REAM/REAP/InputCov pipelines already consume these) can declare them
        without an engine edit.
    ROUTER_LOGITS_PER_BATCH
        The accumulator wants the per-batch router-logits storage from
        :func:`~moe_compress.utils.activation_hooks.capture_router_outputs`
        AND a hook that runs AFTER each model forward to drain + reset the
        storage. Used by sink-token routing today.
    INPUT_IDS_PER_BATCH
        The accumulator wants a model-level pre-forward hook that captures
        the batch's ``input_ids`` into a closure cell; every accumulator
        using INPUT_IDS_PER_BATCH can read it via the per-batch handler's
        :class:`PerBatchContext`.
    """

    DOWN_PROJ = "down_proj"
    EXPERT_INPUT = "expert_input"
    EXPERT_INTERMEDIATE = "expert_intermediate"
    EXPERT_GATE_UP_OUT = "expert_gate_up_out"
    ROUTER_LOGITS_PER_BATCH = "router_logits_per_batch"
    INPUT_IDS_PER_BATCH = "input_ids_per_batch"


# Mapping from declarative channel kind → the string channel name expected by
# ``instrument_experts``'s ``callbacks`` dict (see
# ``utils/activation_hooks.py:1233-1240``). Kept module-private; the engine
# inspects it during ``run()``.
_KIND_TO_CHANNEL: dict[HookKind, str] = {
    HookKind.DOWN_PROJ: "down",
    HookKind.EXPERT_INPUT: "input",
    HookKind.EXPERT_INTERMEDIATE: "intermediate",
    HookKind.EXPERT_GATE_UP_OUT: "gate_up_out",
}


# ---------------------------------------------------------------------------
# Spec dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HookSpec:
    """Declarative bundle of hooks an accumulator needs.

    Bundles multiple :class:`HookKind` values because a single accumulator
    (like :class:`~moe_compress.utils.sink_token_routing.SinkTokenRoutingAccumulator`)
    requires *all* of router-logits + input-ids + a per-batch drain in one
    coordinated unit. The engine inspects ``kinds`` to decide which context
    managers / hooks to enter and calls ``per_batch`` (if set) after every
    forward.

    Fields
    ------
    kinds
        Which hook channels this accumulator subscribes to. The engine
        installs every channel mentioned by any registered accumulator
        exactly once and routes events to all subscribers.
    expert_callback
        Per-expert-event handler for the EXPERT_* / DOWN_PROJ channels.
        Signature: ``(layer_idx, expert_idx, tensor, ctx) -> None`` — same
        as ``instrument_experts``'s ``CallbackFn``. The engine multiplexes:
        if N accumulators register DOWN_PROJ each with their own callback,
        the engine builds one ``down`` callback that fans out to all N.
    per_batch
        Per-batch drain handler invoked after each model forward. The engine
        passes a :class:`PerBatchContext` and clears the router-logits
        storage AFTER all per-batch handlers have run (so the handler can
        read the storage).

    Notes
    -----
    Frozen dataclass so an accumulator's hook intent is immutable after
    registration. ``kinds`` is a frozenset so duplicate enum mentions are
    harmless.
    """

    kinds: frozenset[HookKind]
    expert_callback: Callable[[int, int, torch.Tensor, dict], None] | None = None
    per_batch: Callable[["PerBatchContext"], None] | None = None


@dataclass
class PerBatchContext:
    """State the engine passes to every per-batch handler after each forward.

    Fields
    ------
    batch_idx
        Zero-based index of the just-completed batch.
    router_logits_storage
        The storage dict returned by ``capture_router_outputs``. ``None``
        when no registered accumulator declared
        :attr:`HookKind.ROUTER_LOGITS_PER_BATCH` (skips the cost of
        installing the router pre-hook). The engine clears all per-layer
        lists AFTER all per-batch handlers have run.
    input_ids
        The current batch's ``input_ids`` as captured by the pre-forward
        hook. ``None`` when no registered accumulator declared
        :attr:`HookKind.INPUT_IDS_PER_BATCH` OR when the forward call didn't
        pass input_ids through. The engine matches the legacy inline
        behaviour: the tensor is captured via ``.detach()`` and is NOT
        unsqueezed (consumers that need a 2-D ``[B, T]`` view do the
        unsqueeze themselves — mirrors the legacy ``_phase_b_per_batch_cb``).

        WARNING — shape may be 1-D ``[T]`` or 2-D ``[B, T]``.
            The engine intentionally does NOT normalise the shape: callers
            that need ``[B, T]`` MUST handle the 1-D case themselves, e.g.::

                ids = pbc.input_ids
                if ids.dim() == 1:
                    ids = ids.unsqueeze(0)

            This mirrors the legacy ``_phase_b_per_batch_cb`` byte-for-byte.
            Plugin authors (sink-token and beyond) MUST follow this
            contract — forgetting the unsqueeze on a 1-D batch silently
            produces wrong-shape downstream tensors.
    """

    batch_idx: int
    router_logits_storage: dict[int, list[torch.Tensor]] | None
    input_ids: torch.Tensor | None


@dataclass(frozen=True)
class AccumulatorRegistration:
    """One row in the engine's registration table.

    Fields
    ------
    name
        Human-readable id for the accumulator. Unique across one engine
        instance (duplicate :meth:`CalibrationEngine.register_accumulator`
        calls raise :class:`KeyError`).
    accumulator
        The actual accumulator object — its ``finalize()`` is NOT called by
        the engine; the engine is forward-pass plumbing only. Callers own
        the lifecycle (matches the legacy inline pattern).
    spec
        Declarative hook intent.
    """

    name: str
    accumulator: Any
    spec: HookSpec


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class CalibrationEngine:
    """Cross-stage calibration-pass driver.

    Plugins register accumulators with declarative hook intent (a
    :class:`HookSpec` bundling :class:`HookKind` values + handler
    callables). :meth:`run` then builds a single
    :class:`contextlib.ExitStack` that:

    1. Wires ``instrument_experts`` on every MoE layer iff any accumulator
       declared a per-expert channel (DOWN_PROJ / EXPERT_*). All registered
       per-expert callbacks are multiplexed onto one dict of fan-out
       callbacks passed to ``instrument_experts``.
    2. Enters ``capture_router_outputs(moe_layers)`` iff any accumulator
       declared ROUTER_LOGITS_PER_BATCH.
    3. Installs a model-level pre-forward hook that captures the batch's
       ``input_ids`` into an internal closure cell iff any accumulator
       declared INPUT_IDS_PER_BATCH.
    4. Builds a fan-out per-batch callback that calls every accumulator's
       ``per_batch`` handler (in registration order) with a fresh
       :class:`PerBatchContext`, then drains the router-logits storage
       AFTER all handlers have run.

    The engine then invokes
    :func:`~moe_compress.utils.activation_hooks.run_calibration` and tears
    down on exit.

    Determinism
    -----------
    Per-expert callbacks fan out in registration order. Per-batch handlers
    fire in registration order. Hook installation order:
    ``instrument_experts`` → ``capture_router_outputs`` → ``input_ids``
    pre-hook — matches the legacy inline order byte-for-byte.

    Reuse
    -----
    A single :class:`CalibrationEngine` instance is one-shot: :meth:`run`
    may be called at most once. Multi-pass workflows construct a fresh
    engine per pass (matches Stage 1's Phase A / Phase B separation).
    """

    def __init__(self) -> None:
        self._registrations: list[AccumulatorRegistration] = []
        self._names: set[str] = set()
        self._has_run: bool = False

    # ----- registration ----------------------------------------------------
    def register_accumulator(
        self,
        name: str,
        accumulator: Any,
        hook_spec: HookSpec,
    ) -> None:
        """Record one accumulator's hook intent.

        Hooks are NOT installed here — installation happens in :meth:`run`.
        Storing the intent in a list lets the engine batch-install all
        hooks under one :class:`contextlib.ExitStack`, which is the single
        source of teardown.

        Raises
        ------
        KeyError
            If ``name`` was already registered on this engine instance.
        TypeError
            If ``hook_spec`` is not a :class:`HookSpec`.
        ValueError
            If ``hook_spec.kinds`` is empty, or if it contains a non-:class:`HookKind`
            value. The latter is a defensive guard against a future enum
            extension that lands an unhandled value here.
        RuntimeError
            If :meth:`run` has already been called on this engine.
        """
        if self._has_run:
            raise RuntimeError(
                "CalibrationEngine.register_accumulator: cannot register after run()"
            )
        if not isinstance(hook_spec, HookSpec):
            raise TypeError(
                f"hook_spec must be a HookSpec, got {type(hook_spec).__name__}"
            )
        if not hook_spec.kinds:
            raise ValueError(f"hook_spec.kinds is empty for accumulator {name!r}")
        unknown = [k for k in hook_spec.kinds if not isinstance(k, HookKind)]
        if unknown:
            raise ValueError(
                f"hook_spec.kinds contains non-HookKind values for accumulator "
                f"{name!r}: {sorted(repr(k) for k in unknown)}"
            )
        if name in self._names:
            raise KeyError(f"accumulator {name!r} already registered")
        self._names.add(name)
        self._registrations.append(
            AccumulatorRegistration(name=name, accumulator=accumulator, spec=hook_spec)
        )

    # ----- introspection ---------------------------------------------------
    def __len__(self) -> int:
        return len(self._registrations)

    def names(self) -> tuple[str, ...]:
        """Names of registered accumulators in registration order."""
        return tuple(r.name for r in self._registrations)

    def required_hook_kinds(self) -> frozenset[HookKind]:
        """Union of all :class:`HookKind` values declared by registered accumulators.

        Used by :meth:`run` to decide which context managers / hooks to
        enter. Exposed publicly so callers (orchestrator, tests) can inspect
        the effective wiring before invoking :meth:`run`.
        """
        out: set[HookKind] = set()
        for r in self._registrations:
            out.update(r.spec.kinds)
        return frozenset(out)

    # ----- the calibration pass -------------------------------------------
    def run(
        self,
        model: torch.nn.Module,
        batches: Iterable[torch.Tensor],
        *,
        moe_layers: Iterable[Any],
        device: torch.device | None = None,
        progress_label: str = "calibration",
        per_batch_hooks: Iterable[Callable[[int], None]] = (),
    ) -> None:
        """Run one calibration pass with all registered accumulators wired.

        Parameters
        ----------
        model
            The MoE model to forward. The engine calls
            ``run_calibration(model, batches, device=device, per_batch_callback=...)``.
        batches
            The calibration batches (already a materialised list from
            ``utils.calibration.iter_batches``). Reused — not consumed.
        moe_layers
            The layers to instrument. Provided by the caller (the engine is
            stage-agnostic and does not call ``iter_moe_layers`` itself).
        device
            Forwarded to :func:`~moe_compress.utils.activation_hooks.run_calibration`.
        progress_label
            Reserved for future telemetry; today the engine does not consume
            it. Per-batch progress callbacks are passed via
            ``per_batch_hooks``.
        per_batch_hooks
            Extra per-batch callbacks chained AFTER the accumulator
            per-batch handlers (matches today's
            legacy ``_phase_b_per_batch_cb → progress_cb`` chain). Each
            callable is invoked with the batch index.

        Raises
        ------
        RuntimeError
            If :meth:`run` has already been called on this engine instance.
        """
        # ``progress_label`` is reserved for future telemetry. Bind it to ``_``
        # rather than ``del``-ing it so the name stays accessible for a future
        # log/metric call here without tripping a NameError mid-run.
        _ = progress_label
        if self._has_run:
            raise RuntimeError(
                "CalibrationEngine.run: already run; construct a fresh engine"
            )
        self._has_run = True

        moe_layers = list(moe_layers)
        kinds = self.required_hook_kinds()
        wants_expert_cb = bool(
            kinds
            & {
                HookKind.DOWN_PROJ,
                HookKind.EXPERT_INPUT,
                HookKind.EXPERT_INTERMEDIATE,
                HookKind.EXPERT_GATE_UP_OUT,
            }
        )
        wants_router = HookKind.ROUTER_LOGITS_PER_BATCH in kinds
        wants_input_ids = HookKind.INPUT_IDS_PER_BATCH in kinds

        # Closure cell for input_ids — single source of truth across all
        # per-batch handlers (mirrors the legacy ``_current_input_ids``).
        current_input_ids: list[torch.Tensor | None] = [None]

        def _capture_input_ids(_module, args, kwargs):
            ids: torch.Tensor | None = None
            if kwargs and "input_ids" in kwargs and kwargs["input_ids"] is not None:
                ids = kwargs["input_ids"]
            elif args and len(args) > 0 and isinstance(args[0], torch.Tensor):
                ids = args[0]
            current_input_ids[0] = ids.detach() if ids is not None else None

        # Build channel -> list[callable] in registration order. Each entry in
        # ``channel_callbacks[channel]`` is one accumulator's expert_callback;
        # the engine fans out per channel with one fused closure.
        channel_callbacks: dict[str, list[Callable]] = {}
        for r in self._registrations:
            if r.spec.expert_callback is None:
                continue
            for kind in r.spec.kinds:
                channel = _KIND_TO_CHANNEL.get(kind)
                if channel is None:
                    continue
                channel_callbacks.setdefault(channel, []).append(r.spec.expert_callback)

        def _make_fanout(channel: str) -> Callable:
            cbs = channel_callbacks[channel]
            if len(cbs) == 1:
                # Pass-through: only one subscriber, skip the fan-out closure.
                # Single subscriber: pass it through verbatim so the closure
                # adds no measurable overhead on the hot path.
                return cbs[0]

            def _fan(li, e, tensor, ctx):
                for cb in cbs:
                    cb(li, e, tensor, ctx)

            return _fan

        fanout_callbacks: dict[str, Callable] = {
            channel: _make_fanout(channel) for channel in channel_callbacks
        }

        per_batch_handlers = [
            r.spec.per_batch
            for r in self._registrations
            if r.spec.per_batch is not None
        ]
        extra_hooks: tuple[Callable[[int], None], ...] = tuple(per_batch_hooks)

        with contextlib.ExitStack() as stack:
            # 1. instrument_experts per layer (matches the legacy inline order).
            if wants_expert_cb and fanout_callbacks:
                for ref in moe_layers:
                    stack.enter_context(instrument_experts(ref, fanout_callbacks))

            # 2. capture_router_outputs (matches the legacy inline order).
            router_storage: dict[int, list[torch.Tensor]] | None = None
            if wants_router:
                router_storage = stack.enter_context(capture_router_outputs(moe_layers))

            # 3. input_ids pre-hook (matches the legacy inline order).
            if wants_input_ids:
                handle = model.register_forward_pre_hook(
                    _capture_input_ids, with_kwargs=True
                )
                stack.callback(handle.remove)

            # 4. Per-batch fan-out: accumulator handlers (in registration
            #    order) → extra per_batch_hooks → router-storage drain.
            need_per_batch = bool(
                per_batch_handlers or extra_hooks or router_storage is not None
            )

            def _per_batch_cb(batch_idx: int) -> None:
                pbc = PerBatchContext(
                    batch_idx=batch_idx,
                    router_logits_storage=router_storage,
                    input_ids=current_input_ids[0],
                )
                for handler in per_batch_handlers:
                    handler(pbc)
                for hook in extra_hooks:
                    hook(batch_idx)
                # Drain router storage AFTER all handlers have read it
                # (matches the legacy Phase-B per-batch drain).
                if router_storage is not None:
                    for li in router_storage:
                        router_storage[li].clear()

            run_calibration(
                model,
                batches,
                device=device,
                per_batch_callback=_per_batch_cb if need_per_batch else None,
            )


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------


def run_calibration_pass(
    model,
    batches,
    *,
    registrations,
    moe_layers,
    device=None,
    progress_label="calibration",
    per_batch_hooks=(),
) -> None:
    """Construct a :class:`CalibrationEngine`, register every accumulator, run.

    A thin convenience wrapper for the common "register N accumulators then
    run one pass" case. It builds a fresh one-shot engine, registers each
    ``(name, accumulator, spec)`` triple from ``registrations`` (an ordered
    sequence) **in the given sequence order** — registration order is
    byte-identical-critical because it fixes both the per-expert fan-out order
    and the per-batch handler order — then invokes :meth:`CalibrationEngine.run`.

    Parameters
    ----------
    model
        The MoE model to forward. Passed straight through to
        :meth:`CalibrationEngine.run`.
    batches
        The calibration batches. Passed straight through.
    registrations
        An ordered sequence of ``(name, accumulator, spec)`` triples. Each is
        registered via :meth:`CalibrationEngine.register_accumulator` in
        iteration order. An empty sequence yields a bare engine — :meth:`run`
        is still called, producing a plain no-op forward pass.
    moe_layers
        The layers to instrument. Forwarded as the ``moe_layers`` keyword.
    device
        Forwarded as the ``device`` keyword.
    progress_label
        Forwarded as the ``progress_label`` keyword.
    per_batch_hooks
        Forwarded as the ``per_batch_hooks`` keyword.
    """
    engine = CalibrationEngine()
    for name, accumulator, spec in registrations:
        engine.register_accumulator(name, accumulator, spec)
    engine.run(
        model,
        batches,
        moe_layers=moe_layers,
        device=device,
        progress_label=progress_label,
        per_batch_hooks=per_batch_hooks,
    )


__all__ = [
    "HookKind",
    "HookSpec",
    "PerBatchContext",
    "AccumulatorRegistration",
    "CalibrationEngine",
    "run_calibration_pass",
]
