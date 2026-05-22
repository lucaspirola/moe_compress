"""``StagePlugin`` Protocol + ``PluginRegistry`` — cross-stage plugin framework.

The Protocol declares the *attributes* a stage plugin must expose so the
registry / orchestrator can inspect a plugin without invoking it. Concrete
plugins (sub-tasks 3-9) are plain classes that just declare these attributes;
the Protocol is :class:`typing.Protocol` + ``@runtime_checkable`` so tests can
do ``isinstance(plugin, StagePlugin)`` without subclassing.

The registry is an ordered, **immutable-after-construction** collection: a
stage's plugin sequence is part of its contract, so no ``register()`` method
is provided — the orchestrator (sub-task 10) builds a fresh registry at
startup.
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable


@runtime_checkable
class StagePlugin(Protocol):
    """Contract every stage plugin must satisfy.

    The attributes are class-level (not method-level) so the registry can
    inspect a plugin instance without calling anything. ``reads`` / ``writes``
    enable a future static check that every plugin reads keys some prior
    plugin produced; ``accumulators`` is consumed by the shared calibration
    engine (sub-task 2) to know which hooks to register.

    Design choices
    --------------
    1. ``runtime_checkable`` — tests can do ``isinstance(plugin, StagePlugin)``
       without subclassing. The Protocol is structural, not nominal, because
       later detector/merge plugins will be plain classes that just have the
       right attributes; we don't want them inheriting a base class.
    2. ``ctx: "Any"`` in ``run`` / ``contribute_artifact`` — string-quoted
       because the concrete type is ``Stage1Context`` (or future
       ``Stage2Context``), defined under ``stage1/`` / ``stage2/``. The
       framework-level Protocol cannot import a stage-specific type without
       a cycle. Plugins themselves can annotate with the concrete type.
    3. Attribute declaration order matches the overarching plan's Protocol
       snippet exactly — do not reorder.
    """

    name: str                                # Unique plugin id (e.g. "ma_detection")
    paper: str                               # Citation / one-liner
    config_key: str                          # Dotted path into YAML (e.g. "stage1_grape.super_expert_detection.aimer_enabled")
    reads: tuple[str, ...]                   # Context fields this plugin consumes
    writes: tuple[str, ...]                  # Context fields this plugin produces
    accumulators: tuple[str, ...]            # Named accumulators it needs Phase B to run

    def is_enabled(self, config: dict) -> bool: ...
    def run(self, ctx: "Any") -> None: ...   # ctx is a PipelineContext subclass; Any to avoid stage1 import cycle
    def contribute_artifact(self, ctx: "Any") -> dict: ...


class PluginRegistry:
    """Ordered, immutable-after-build registry of plugins.

    Construction order = execution order. The orchestrator (sub-task 10)
    builds the registry once at startup; later mutations are not supported
    by design — a stage's plugin sequence is part of its contract.

    Design choices
    --------------
    * Tuple, not list, for ``_plugins`` → immutability after construction.
    * Duplicate-name check at construction — fail loud rather than have two
      plugins silently fight over the same context slot.
    * ``required_accumulators`` returns a tuple in first-occurrence order,
      not a set, because sub-task 2's calibration engine will register hooks
      in a deterministic order to keep the golden snapshot stable.
    * No ``register()`` method — registry is read-only after ``__init__``.
      Mutation belongs to the orchestrator's setup phase, which builds a
      fresh registry.
    """

    def __init__(self, plugins: Sequence[StagePlugin]) -> None:
        names = [p.name for p in plugins]
        dups = {n for n in names if names.count(n) > 1}
        if dups:
            raise ValueError(f"Duplicate plugin names in registry: {sorted(dups)}")
        self._plugins: tuple[StagePlugin, ...] = tuple(plugins)

    def __iter__(self):
        return iter(self._plugins)

    def __len__(self) -> int:
        return len(self._plugins)

    def names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self._plugins)

    def enabled(self, config: dict) -> tuple[StagePlugin, ...]:
        """Return the subset of plugins whose ``is_enabled(config)`` is true,
        preserving insertion order."""
        return tuple(p for p in self._plugins if p.is_enabled(config))

    def required_accumulators(self, config: dict) -> tuple[str, ...]:
        """Union of all ``accumulators`` declared by *enabled* plugins, in
        first-occurrence order. Sub-task 2's CalibrationEngine consumes this
        to decide which hooks to register."""
        seen: dict[str, None] = {}  # dict preserves insertion order
        for p in self.enabled(config):
            for acc in p.accumulators:
                seen.setdefault(acc, None)
        return tuple(seen.keys())
