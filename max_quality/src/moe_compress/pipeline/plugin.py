"""``PipelinePlugin`` Protocol + ``BasePlugin`` — universal plugin framework.

The Protocol declares the *attributes* a pipeline plugin must expose so the
registry / orchestrator can inspect a plugin without invoking it. Concrete
plugins are plain classes that just declare these attributes; the Protocol is
:class:`typing.Protocol` + ``@runtime_checkable`` so tests can do
``isinstance(plugin, PipelinePlugin)`` without subclassing.

``BasePlugin`` is an optional convenience base for plugins that prefer
nominal subclassing over structural conformance — both are first-class.
``PluginRegistry`` lives in ``registry.py`` (task F-3), not here.

.. note::
   Attribute-level ``isinstance`` conformance against ``PipelinePlugin``
   requires Python ≥3.12; on older interpreters ``runtime_checkable`` only
   verifies methods, not the presence of non-callable attributes.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class PipelinePlugin(Protocol):
    """Contract every pipeline plugin must satisfy.

    The attributes are class-level (not method-level) so the registry can
    inspect a plugin instance without calling anything. ``reads`` / ``writes``
    enable a future static check that every plugin reads keys some prior
    plugin produced; ``provides`` is consumed by the F-6 calibration-pass
    multiplexer to know which named accumulators a calibration sweep must run.

    Design choices
    --------------
    1. ``runtime_checkable`` — tests can do ``isinstance(plugin,
       PipelinePlugin)`` without subclassing. The Protocol is structural, not
       nominal, because most plugins will be plain classes that simply have the
       right attributes; we do not want to force them into an inheritance
       hierarchy.
    2. ``ctx: Any`` in ``contribute_artifact`` — the concrete type is
       ``PipelineContext``, which lives in ``pipeline/context.py`` (task F-2).
       Annotating the framework-level Protocol with that type directly would
       couple this module to one that does not exist yet (and risk an import
       cycle), so ``ctx`` is typed ``Any``; plugins themselves are free to
       annotate with the concrete type.
    3. No phase-hook methods on the Protocol. Phase-hook names are an open
       vocabulary: stages add their own hooks freely and they are discovered
       reflectively by the ``walk_phases`` walker (``tools/phase_walker.py``,
       task F-5) via ``getattr``. The Protocol therefore carries only the
       *universal core* — ``is_enabled`` and ``contribute_artifact`` — plus the
       inspectable metadata. ``provides`` replaces stage-1's ``accumulators``;
       ``PluginRegistry`` lives in ``registry.py`` (task F-3).
    """

    # Attributes are class-level so the registry can inspect a plugin
    # without invoking it.
    name: str                                # Unique plugin id (e.g. "ma_detection")
    paper: str                               # Citation / one-liner
    config_key: str                          # Dotted path into YAML (e.g. "stage1_grape.super_expert_detection.aimer_enabled")
    reads: tuple[str, ...]                   # Context fields this plugin consumes
    writes: tuple[str, ...]                  # Context fields this plugin produces
    provides: tuple[str, ...]                # Named accumulators it needs a calibration pass to run

    def is_enabled(self, config: dict) -> bool: ...
    def contribute_artifact(self, ctx: Any) -> dict: ...


class BasePlugin:
    """Optional convenience base for pipeline plugins.

    Provides metadata defaults plus no-op implementations of the universal
    core hooks, so a subclass only needs to override what it cares about. You
    may subclass ``BasePlugin`` *or* satisfy :class:`PipelinePlugin`
    structurally — both styles are first-class.

    This base carries *only* the universal core (``is_enabled``,
    ``contribute_artifact``); it intentionally declares no phase-specific
    hooks, since those are an open vocabulary discovered reflectively by
    the phase walker.
    """

    name: str = ""
    paper: str = ""
    config_key: str = ""
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()
    provides: tuple[str, ...] = ()

    def is_enabled(self, config: dict) -> bool:
        return True

    def contribute_artifact(self, ctx: Any) -> dict:
        # Fresh dict literal every call — never a shared module-level object.
        return {}


__all__ = ["PipelinePlugin", "BasePlugin"]
