"""``PipelineContext`` — scoped shared-state holder for a pipeline run.

Design choice: ``dict[str, Any]`` with explicit ``get`` / ``set`` accessors,
**not** a dataclass.

Rationale
---------
1. **Dataclass-of-typed-slots was considered and rejected.** Each stage will
   have ~15-25 typed context fields (Stage 1 alone has on the order of:
   ``model, tokenizer, config, artifacts_dir, calibration_spec, moe_layers, L,
   residual_growth, moe_output_growth, moe_output_max, max_acc, output_acc,
   sink_acc, aimer_scores, bottom_pct_by_layer, candidates, blacklist,
   candidate_deltas, baseline_nll, D_matrices, per_layer_target_experts,
   per_layer_redundancy, achieved_budget, requested_budget``). A dataclass
   would force every new plugin to edit a shared typed-context file —
   exactly the cross-plugin coupling the refactor wants to eliminate. Plugins
   should be able to add a new field to the context with zero churn outside
   their own module.
2. **``dict[str, Any]`` with strict accessors** gives the same "meaningful
   error on read-before-write" guarantee as a dataclass (via the custom
   ``get`` that raises ``KeyError``), without coupling. A plugin declares the
   keys it writes in ``writes`` and reads in ``reads``; static-consistency
   checking moves to the registry, not to a Python class shape.
3. **Set-once default** — a second ``set(name, ...)`` raises unless
   ``overwrite=True``. If a plugin genuinely needs to update a slot, the
   pattern is to pass a mutable container (e.g. a ``CandidateBag`` instance)
   once and mutate it in place. The set-once guard is on the *binding*, not
   the *value*.

Parent/child scopes
-------------------
This generalizes stage-1's single flat dict with parent/child scoping, and
supersedes stage-2's typed ``RunContext`` + ``LayerContext`` split. That split
made every field ``Optional`` (because a slot is only populated partway through
the run) and still needed an ``extras`` escape hatch for anything unforeseen —
proof that typed contexts do not scale: every new per-layer slot edits a shared
file. Instead there is one ``PipelineContext``. Run-scope is the root context;
each loop iteration (e.g. per-layer) gets a fresh :meth:`child`. ``get`` falls
through to the parent chain, while ``set`` is always local to the scope it is
called on.

The resolution asymmetry is intentional: :meth:`get` and :meth:`has` resolve
through the parent chain, but :meth:`keys`, :meth:`__contains__`, :meth:`__iter__`
and :meth:`drop` are LOCAL-scope only. A child can *read* parent state but its
own write surface (and what it may delete or enumerate) is exactly its own
scope.

There is no ``UNSET`` sentinel: presence of the binding is the source of truth.
``None`` is a legal stored value, distinguished from "not written" by
:meth:`has` (or by :meth:`get` raising :class:`KeyError`).
"""

from __future__ import annotations

from typing import Any, Iterator


class PipelineContext:
    """Scoped cross-stage shared-state holder for a single pipeline run.

    Plugins read/write named slots via :meth:`get` / :meth:`set`. Reads of
    unwritten slots raise :class:`KeyError` with a message listing the local
    written slots. Writes default to set-once semantics: a second
    ``set(name, ...)`` raises unless ``overwrite=True``.

    A context may have a ``parent``: :meth:`get` / :meth:`has` resolve through
    the parent chain, while :meth:`set` / :meth:`drop` / :meth:`keys` /
    :meth:`__contains__` / :meth:`__iter__` act on the local scope only. Call
    :meth:`child` to open a nested scope (e.g. one per loop iteration).
    """

    def __init__(self, parent: "PipelineContext | None" = None) -> None:
        self._state: dict[str, Any] = {}
        self._parent = parent

    def set(self, name: str, value: Any, *, overwrite: bool = False) -> None:
        # The set-once check inspects ONLY this scope's local state (never the
        # parent chain / get): a child MAY intentionally shadow a parent slot.
        if not overwrite and name in self._state:
            raise KeyError(
                f"PipelineContext slot {name!r} already written; "
                f"pass overwrite=True to replace it."
            )
        self._state[name] = value

    def get(self, name: str) -> Any:
        if name in self._state:
            return self._state[name]
        if self._parent is not None and self._parent.has(name):
            return self._parent.get(name)
        raise KeyError(
            f"PipelineContext slot {name!r} not set "
            f"(written slots: {sorted(self._state)!r})."
        )

    def has(self, name: str) -> bool:
        return name in self._state or (
            self._parent is not None and self._parent.has(name)
        )

    def drop(self, name: str) -> None:
        """Remove a written slot from this scope, releasing its reference.

        Mirrors :meth:`get`'s strictness: raises :class:`KeyError` if the
        slot was never written *in this scope*. Used by the orchestrator to
        free large intermediates (e.g. the CKA expert-output reservoir) once
        their last consumer has run — matching the legacy ``del output_acc``.

        LOCAL-scope only: a child cannot drop a parent slot — :meth:`drop`
        raises here even though :meth:`has` would return ``True`` for that
        slot. This ``drop`` (local) vs ``has`` (chain) asymmetry is intentional.
        """
        if name not in self._state:
            raise KeyError(
                f"PipelineContext slot {name!r} not set "
                f"(written slots: {sorted(self._state)!r})."
            )
        del self._state[name]

    def child(self) -> "PipelineContext":
        """Open a nested scope whose parent is this context."""
        return PipelineContext(parent=self)

    def keys(self) -> tuple[str, ...]:
        """Slot names written in this scope only (excludes parent slots)."""
        return tuple(self._state.keys())

    def __contains__(self, name: object) -> bool:
        """Local-scope membership test.

        Note ``name in ctx`` (local) differs from :meth:`has` (chains through
        parents) for a child context — only :meth:`has` sees parent slots.
        """
        return isinstance(name, str) and name in self._state

    def __iter__(self) -> Iterator[str]:
        """Iterate slot names written in this scope only."""
        return iter(self._state)


__all__ = ["PipelineContext"]
