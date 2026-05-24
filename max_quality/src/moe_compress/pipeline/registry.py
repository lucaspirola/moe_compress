"""``PluginRegistry`` — ordered, immutable collection of pipeline plugins.

The registry is an ordered, **immutable-after-construction** collection: a
stage's plugin sequence is part of its contract, so the orchestrator builds a
fresh registry once at startup and the registry is never mutated thereafter.
Construction order is the execution order — there is no separate priority knob.

Design choices
--------------
1. **Ordered + immutable-after-construction.** ``_plugins`` is a ``tuple``,
   not a ``list``, so it cannot be appended to or reordered once built.
   Construction order = execution order.
2. **No ``register()`` method.** Unlike stage-2's mutable registry, this one
   exposes no ``register()`` / ``active()`` / ``classes()`` — a stage's plugin
   sequence is its contract, fixed at construction. Mutation would let two
   call sites disagree about the pipeline shape; instead the orchestrator
   builds one registry with the full, ordered sequence up front.
3. **Tuple, not list, for ``_plugins``** → immutability after construction;
   the input ``Sequence`` is materialized exactly once (``tuple(plugins)``
   first thing in ``__init__``) so a one-shot iterable / generator is safe to
   pass and is not exhausted by the duplicate-name scan.
4. **Duplicate-name check at construction** — fail loud with a ``ValueError``
   rather than have two plugins silently fight over the same context slot or
   artifact key. An empty registry is legal.
5. **``provides`` is stage-1's ``required_accumulators`` renamed.** It unions
   the plugins' ``provides`` tuples (stage-1's attribute was ``accumulators``)
   over the *enabled* subset only, in first-occurrence order via an
   insertion-ordered ``dict``, not a ``set`` — task F-6's calibration pass
   consumes this and registers calibration hooks in a deterministic order to
   keep golden snapshots stable.
6. **``dispatch_first`` is the slot-style helper.** A ``@staticmethod`` taking
   an explicit plugins sequence (typically ``registry.enabled(cfg)``), for
   hooks where exactly one plugin should win per slot.
"""

from __future__ import annotations

from typing import Any, Sequence

from .plugin import PipelinePlugin


class PluginRegistry:
    """Ordered, immutable-after-build registry of pipeline plugins.

    Construction order = execution order. The orchestrator builds the registry
    once at startup; later mutation is unsupported by design — a stage's plugin
    sequence is part of its contract, so there is no ``register()`` method.

    Design choices
    --------------
    * Tuple, not list, for ``_plugins`` → immutability after construction; the
      input ``Sequence`` is materialized once so generators are safe to pass.
    * Duplicate-name check at construction — fail loud rather than have two
      plugins silently fight over the same context slot. An empty registry is
      legal.
    * :meth:`provides` returns a tuple in first-occurrence order, not a set,
      because task F-6's calibration pass registers hooks in a deterministic
      order to keep the golden snapshot stable.
    * :meth:`dispatch_first` is a ``@staticmethod`` slot helper: it takes an
      explicit plugins sequence (the Protocol has no fixed phase hooks, so the
      registry cannot own a per-hook dispatch path).
    """

    def __init__(self, plugins: Sequence[PipelinePlugin]) -> None:
        # Materialize the input once: a one-shot iterable (e.g. a generator)
        # would otherwise be exhausted by the duplicate-name scan below.
        self._plugins: tuple[PipelinePlugin, ...] = tuple(plugins)
        names = [p.name for p in self._plugins]
        dups = {n for n in names if names.count(n) > 1}
        if dups:
            raise ValueError(f"Duplicate plugin names in registry: {sorted(dups)}")

    def __iter__(self):
        return iter(self._plugins)

    def __len__(self) -> int:
        return len(self._plugins)

    def names(self) -> tuple[str, ...]:
        """Plugin names in construction (= execution) order."""
        return tuple(p.name for p in self._plugins)

    def enabled(self, config: dict) -> tuple[PipelinePlugin, ...]:
        """Return the subset of plugins whose ``is_enabled(config)`` is true,
        preserving construction order."""
        return tuple(p for p in self._plugins if p.is_enabled(config))

    def provides(self, config: dict) -> tuple[str, ...]:
        """Union of every ``provides`` tuple declared by the *enabled* plugins,
        in first-occurrence order.

        Task F-6's calibration pass consumes this to decide which named
        accumulators a calibration sweep must run. Disabled plugins contribute
        nothing. Order is deterministic (first occurrence wins) so golden
        snapshots stay stable.
        """
        seen: dict[str, None] = {}  # dict preserves insertion order
        for p in self.enabled(config):
            for acc in p.provides:
                seen.setdefault(acc, None)
        return tuple(seen.keys())

    @staticmethod
    def dispatch_first(
        plugins: Sequence[PipelinePlugin],
        hook_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any | None:
        """Call ``hook_name`` on each plugin in order; return the first
        non-None result.

        The slot-style helper for hooks where exactly one plugin should win.
        Unlike stage-2's ``dispatch_first``, this both *skips a plugin lacking
        the hook* and *skips a non-callable attribute colliding with the hook
        name* — the universal Protocol declares no fixed phase hooks, so a
        given plugin may simply not implement a given slot. The first
        **non-None** result wins (``is not None``, never truthiness — a hook
        legitimately returning ``0`` / ``False`` / ``""`` / ``[]`` must count
        as a winner).
        """
        for p in plugins:
            hook = getattr(p, hook_name, None)
            if not callable(hook):
                continue
            result = hook(*args, **kwargs)
            if result is not None:
                return result
        return None


__all__ = ["PluginRegistry"]
