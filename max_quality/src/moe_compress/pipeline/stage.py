"""``Stage`` Protocol — the orchestrator-facing compression-stage contract.

This Protocol is the *orchestrator-facing* contract, distinct from
``PipelinePlugin`` (``pipeline/plugin.py``, task F-1), which is the *intra-stage*
contract. A :class:`Stage` is a whole compression stage — one of stages
``1, 2, 2.5, 3, 4, 5, 6, 6alt`` — whereas a ``PipelinePlugin`` is one paper
*inside* a stage.

Defining the ``Stage`` contract now makes every stage orchestrator-ready: the
future universal orchestrator simply iterates over :class:`Stage` objects,
calling :meth:`Stage.is_enabled` and then :meth:`Stage.run`. A stage satisfies
this Protocol either via a module-level singleton (``STAGE1`` … ``STAGE6ALT``)
or via a factory (``make_router_kd_stage(stage_id)``) — both styles are
first-class, since the Protocol is structural.

.. note::
   Attribute-level ``isinstance`` conformance against ``Stage`` requires
   Python ≥3.12; on older interpreters ``runtime_checkable`` only verifies
   methods, not the presence of non-callable attributes (here, ``stage_id``).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .context import PipelineContext


@runtime_checkable
class Stage(Protocol):
    """Contract every compression stage must satisfy for the orchestrator.

    A stage exposes a ``stage_id`` on the object, plus the two universal-core
    methods :meth:`is_enabled` (does the run config request this stage?) and
    :meth:`run` (execute the whole stage).

    Design choices
    --------------
    1. ``runtime_checkable`` — conformance is structural, not nominal: tests
       and the orchestrator can do ``isinstance(stage, Stage)`` without forcing
       stages into an inheritance hierarchy. Stages may be module-level
       singletons or factory-built instances; both pass.
    2. ``stage_id`` is part of the Protocol's required surface: every
       conforming stage exposes it as a ``str`` on the *object*. The
       orchestrator works with :class:`Stage` objects, so it does not matter
       *where* the attribute is set — a module-level singleton stage typically
       declares it as a class attribute, while a factory-built stage (e.g.
       Router-KD's ``make_router_kd_stage(stage_id)``) sets it as an instance
       attribute in ``__init__``. Both satisfy the structural Protocol equally.
    3. ``run`` returns ``None`` — a stage communicates by writing to ``ctx``
       (the :class:`PipelineContext`) and to disk, never via a return value.
       The orchestrator's loop is purely side-effecting.
    4. ``run``'s parameter is annotated with the concrete
       :class:`PipelineContext` type, not ``Any``. This intentionally diverges
       from ``PipelinePlugin.contribute_artifact`` in ``plugin.py``, whose
       ``ctx`` is ``Any`` *only* because ``context.py`` did not yet exist when
       task F-1 landed. ``context.py`` exists now and imports only ``typing``,
       so importing :class:`PipelineContext` here introduces no cycle. Do not
       "fix" this back to ``Any``.
    """

    stage_id: str   # Stage identifier: "1","2","2.5","3","4","5","6","6alt"

    def is_enabled(self, config: dict) -> bool: ...
    def run(self, ctx: PipelineContext) -> None: ...


__all__ = ["Stage"]
