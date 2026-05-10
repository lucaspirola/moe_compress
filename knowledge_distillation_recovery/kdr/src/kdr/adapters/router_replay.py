"""Router-replay context managers for MoE QAD stability (LLR-0025).

arxiv:2605.05365 §IV-C identifies router replay as the #1 MoE stability
technique under precision drift; QAD has even more drift than RL, so the
mechanism applies *more strongly*. Replay pins the student's expert-routing
decisions to the teacher's so the recovery loss is computed across a stable
expert assignment rather than a precision-drifted one.

Two implementations:

  * :class:`RouterReplayContextManager` — for MoE models. Walks teacher and
    student for router submodules (matching the configured name pattern),
    captures teacher router outputs per microbatch, replays them on the
    student. The capture/replay handshake is sequential per microbatch:
    call ``start_microbatch()`` BETWEEN each ``(teacher_forward,
    student_forward)`` pair.

  * :class:`NoOpReplayContextManager` — for non-MoE adapters (LLR-0025 AC #2).
    Same shape as the real hook so the training loop's call sites stay
    polymorphic without any ``isinstance`` branching.

Both classes expose:

  * ``__enter__`` / ``__exit__``  — context-manager protocol
  * ``start_microbatch()``        — reset per-microbatch state

The training loop creates the hook once at run start, enters it around the
whole training loop, and calls ``start_microbatch()`` at the top of every
``_step_one_micro``.
"""

# REQ: LLR-0025

from __future__ import annotations

import logging
from collections.abc import Iterable
from types import TracebackType
from typing import Protocol, runtime_checkable

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


@runtime_checkable
class RouterReplayHookProtocol(Protocol):
    """Shape every router-replay implementation must satisfy.

    The training loop interacts with the hook only through this Protocol;
    adapters return a concrete implementation from ``router_replay_hook``.
    """

    def __enter__(self) -> RouterReplayHookProtocol: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None: ...

    def start_microbatch(self) -> None: ...


class NoOpReplayContextManager:
    """No-op replay hook for non-MoE adapters (LLR-0025 AC #2)."""

    def __enter__(self) -> NoOpReplayContextManager:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None

    def start_microbatch(self) -> None:
        return None


class RouterReplayContextManager:
    """MoE router-replay hook: pins student expert assignments to teacher's.

    Walks both ``teacher`` and ``student`` looking for submodules whose dotted
    name contains ``router_path_pattern``. Each match becomes a hookable
    router. Teacher routers get a *capture* hook that records the output
    tensor (or first tensor of a tuple); student routers get a *replay* hook
    that overrides their output with the captured tensor at the same index.

    Per-microbatch state must be reset by calling ``start_microbatch()``
    BEFORE each ``(teacher_forward, student_forward)`` pair. The capture and
    replay indices are kept in lockstep: as long as the teacher and student
    have the same router topology and produce router calls in the same order,
    pin-on-rank-i works.

    Args:
        teacher: the teacher model (frozen during distillation).
        student: the student model (the one being recovered).
        router_path_pattern: last-segment of the router's dotted module
            path. ``"router"`` matches ``model.layers.0.mlp.router`` but
            NOT ``model.layers.0.mlp.router.linear`` (the router's own
            internal linear). HF MoE variants often use ``"gate"`` instead
            (Mixtral) — adapters override the pattern accordingly.

    Raises:
        ValueError: if teacher and student have different numbers of router
            submodules — that breaks the index-aligned pinning contract.
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        *,
        router_path_pattern: str = "router",
    ) -> None:
        self.teacher = teacher
        self.student = student
        self.router_path_pattern = router_path_pattern
        # `None` slots are kept in `_captured` for routers whose output shape
        # we couldn't intercept (non-tensor / unrecognised tuple); the replay
        # side falls through to the student's natural output for those.
        self._captured: list[torch.Tensor | None] = []
        self._replay_idx = 0
        self._teacher_handles: list[torch.utils.hooks.RemovableHandle] = []
        self._student_handles: list[torch.utils.hooks.RemovableHandle] = []

    def __enter__(self) -> RouterReplayContextManager:
        teacher_routers = list(self._find_routers(self.teacher))
        student_routers = list(self._find_routers(self.student))
        if len(teacher_routers) != len(student_routers):
            raise ValueError(
                "RouterReplayContextManager: teacher and student have "
                f"{len(teacher_routers)} vs {len(student_routers)} router "
                f"submodules matching pattern {self.router_path_pattern!r} — "
                "the index-aligned pinning contract requires equal counts."
            )
        for module in teacher_routers:
            self._teacher_handles.append(module.register_forward_hook(self._capture_hook))
        for module in student_routers:
            self._student_handles.append(module.register_forward_hook(self._replay_hook))
        log.info(
            "RouterReplayContextManager: hooked %d router pairs (pattern=%r)",
            len(teacher_routers),
            self.router_path_pattern,
        )
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        for h in self._teacher_handles:
            h.remove()
        for h in self._student_handles:
            h.remove()
        self._teacher_handles.clear()
        self._student_handles.clear()
        self._captured.clear()

    def start_microbatch(self) -> None:
        """Reset capture buffer + replay index — call before each microbatch.

        Capture and replay are sequential within a microbatch; resetting here
        keeps the next teacher forward starting from index 0 and the next
        student forward replaying from index 0. NOT calling this between
        microbatches would cause the second microbatch's student to replay
        the first microbatch's teacher decisions (drift).
        """
        self._captured.clear()
        self._replay_idx = 0

    # ── Helpers ────────────────────────────────────────────────────────────

    def _find_routers(self, model: nn.Module) -> Iterable[nn.Module]:
        # Match the LAST dotted segment exactly — substring matching would
        # double-count the router and its internal `Linear` (router.linear),
        # breaking index-aligned pinning.
        suffix = "." + self.router_path_pattern
        return (
            m
            for name, m in model.named_modules()
            if name == self.router_path_pattern or name.endswith(suffix)
        )

    def _capture_hook(
        self, _module: nn.Module, _inputs: object, output: object
    ) -> object:
        """Record the router's output. Returns ``output`` unchanged."""
        if isinstance(output, torch.Tensor):
            self._captured.append(output.detach().clone())
        elif isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
            # Many HF routers return ``(logits, top_k_indices, ...)`` — capture
            # element 0 (router_logits) so the replay drives the student's
            # downstream gating math from the same logits.
            self._captured.append(output[0].detach().clone())
        else:
            # Unsupported shape: keep index alignment with a `None` placeholder.
            self._captured.append(None)
        return output

    def _replay_hook(
        self, _module: nn.Module, _inputs: object, output: object
    ) -> object:
        """Override student's router output with the captured teacher tensor."""
        if self._replay_idx >= len(self._captured):
            # Capture/replay drift — should not happen if start_microbatch
            # is called correctly; fall through gracefully rather than crash.
            return output

        captured = self._captured[self._replay_idx]
        self._replay_idx += 1

        if captured is None:
            return output  # capture-side couldn't intercept; pass through

        if isinstance(output, torch.Tensor):
            return captured
        if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
            # Replace logits at position 0 ONLY. Trailing tuple elements
            # (top-k indices, gates) are passed through UNCHANGED — they
            # are the STUDENT's pre-computed values, not the teacher's.
            # For the pin to actually take effect, downstream MoE gating
            # must consume only ``output[0]`` (logits) and re-derive
            # routing from there. If a model family's router returns
            # `(logits, indices)` and downstream uses `indices` directly,
            # the pin silently no-ops on this code path — that family
            # needs a tuple-aware override here that recomputes the trailing
            # elements from `captured`. ZAYA1's CCGQA-MoE router is
            # verified at Phase 7 instantiation; if its router returns a
            # tuple-with-indices, this branch must be specialized.
            return (captured, *output[1:])
        return output
