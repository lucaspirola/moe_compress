"""Reflective phase scheduler — drives plugins through an ordered phase list.

The universal ``PipelinePlugin`` Protocol declares *no* fixed phase hooks
(phase-hook names are an open vocabulary — see ``pipeline/plugin.py`` design
note 3). The walker discovers hooks reflectively: for each phase name it looks
up an attribute of that name on each plugin and calls it if present and
callable.

Design choices
--------------
* **Phase-major / plugin-minor order.** :func:`walk_phases` iterates phases on
  the outer loop and plugins on the inner loop, so *every* plugin runs phase
  ``A`` before *any* plugin runs phase ``B``. The argument is a *phase
  schedule*, not a plugin sequence: plugins are peers within a phase, and
  cross-plugin ordering is expressed by where a hook lives in the phase list,
  never by plugin registration order across phases.
* **``getattr`` + ``callable`` guard.** A plugin that simply lacks a hook for
  some phase is silently skipped — universal plugins are not required to
  implement every phase. The ``callable`` check additionally tolerates a
  non-callable attribute that happens to collide with a phase name (e.g. a
  plugin with a data attribute ``foo`` when ``foo`` is also a phase): such an
  attribute is skipped rather than raising ``TypeError``.
* **Plain ``Sequence[PipelinePlugin]``, not a registry.** The walker takes an
  already-resolved plugin sequence; the caller passes
  ``registry.enabled(config)``. This keeps the walker decoupled from the
  registry, consistent with ``dispatch_first``.
* **Returns ``None``.** :func:`walk_phases` mutates ``ctx`` in place; there is
  no return value to thread.
"""

from __future__ import annotations

from typing import Any, Sequence

from ..pipeline.context import PipelineContext
from ..pipeline.plugin import PipelinePlugin


def walk_phases(
    phases: Sequence[str],
    plugins: Sequence[PipelinePlugin],
    ctx: PipelineContext,
) -> None:
    """Drive ``plugins`` through ``phases`` in phase-major / plugin-minor order.

    For each phase name, every plugin is inspected via ``getattr``; if the
    plugin exposes a *callable* attribute of that name, it is called with
    ``ctx``. Plugins lacking a hook (or carrying a non-callable colliding
    attribute) for a phase are skipped.

    All plugins run a given phase before any plugin runs the next phase: the
    ``phases`` argument is a *phase schedule*, not a plugin sequence. ``plugins``
    is a plain ``Sequence`` (the caller typically passes
    ``registry.enabled(config)``) — the walker is decoupled from the registry.

    Mutates ``ctx`` in place and returns ``None``.
    """
    for phase in phases:
        for plugin in plugins:
            hook = getattr(plugin, phase, None)
            if callable(hook):
                hook(ctx)


def loop_over(
    items: Sequence[Any],
    plugins: Sequence[PipelinePlugin],
    phases: Sequence[str],
    parent_ctx: PipelineContext,
    *,
    item_key: str,
) -> list[PipelineContext]:
    """Run ``phases`` over ``plugins`` once per item, each in a fresh child scope.

    For every item in ``items`` a new :meth:`PipelineContext.child` scope is
    opened off ``parent_ctx``, the item is bound under ``item_key`` in that
    child, and :func:`walk_phases` is run on the child. ``item_key`` is
    keyword-only.

    Returns the list of child contexts, in item order, so callers can harvest
    per-item results (each child's writes stayed local to its own scope). This
    return type extends the master-plan §1.7 sketch, which omitted it — the
    extension is intentional: callers need the children to read back results.
    """
    children: list[PipelineContext] = []
    for item in items:
        child = parent_ctx.child()
        child.set(item_key, item)
        walk_phases(phases, plugins, child)
        children.append(child)
    return children


__all__ = ["walk_phases", "loop_over"]
