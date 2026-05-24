"""Stage 4 as a ``Stage``-conforming object — the orchestrator-facing adapter.

This module exposes the existing Stage 4 plugin pipeline
(:func:`moe_compress.stage4.orchestrator.run`) through the universal
:class:`~moe_compress.pipeline.stage.Stage` Protocol, so the future universal
orchestrator can drive Stage 4 the same way it drives every other stage:
iterate over :class:`Stage` objects, call :meth:`Stage.is_enabled`, then
:meth:`Stage.run`.

This is a *purely additive* adapter — it holds no Stage 4 logic. :meth:`run`
only unwraps the :class:`~moe_compress.pipeline.context.PipelineContext` into
the orchestrator's positional/keyword arguments and writes the output path
back onto the context. The Stage 4 work itself is unchanged and still lives
in ``orchestrator.py`` and the ``stage4/plugins/`` plugins.

Context slots
-------------
Reads (required): ``model``, ``tokenizer``, ``config``, ``artifacts_dir``.

Reads (optional): ``no_resume`` — passed through when present, else ``False``.

Writes: ``stage4_eora_path`` — the ``Path`` of the EoRA-compensated checkpoint
directory returned by the orchestrator.
"""

from __future__ import annotations

from ..pipeline.context import PipelineContext
from .orchestrator import run as _orchestrator_run


class _Stage4:
    """``Stage``-conforming adapter for Stage 4 (EoRA residual compensation).

    A thin shim over :func:`moe_compress.stage4.orchestrator.run`. The
    module-level singleton :data:`STAGE4` is the object the orchestrator uses.

    Context slots
    -------------
    Reads (required): ``model``, ``tokenizer``, ``config``, ``artifacts_dir``.

    Reads (optional): ``no_resume`` — passed through when present, else
    ``False``.

    Writes: ``stage4_eora_path`` — the ``Path`` of the EoRA-compensated
    checkpoint directory returned by the orchestrator. The ``stage4_`` prefix
    namespaces it to avoid future cross-stage slot collisions.
    """

    stage_id: str = "4"

    def is_enabled(self, config: dict) -> bool:
        """Always ``True`` — Stage 4 is mandatory; stage selection belongs to
        the future universal orchestrator, not to the stage itself. There is
        no ``stage4_eora.enabled`` config knob."""
        return True

    def run(self, ctx: PipelineContext) -> None:
        """Run Stage 4 by unwrapping ``ctx`` into the orchestrator call.

        Reads ``model``, ``tokenizer``, ``config`` and ``artifacts_dir`` from
        ``ctx`` (plus optional ``no_resume``), calls the Stage 4 orchestrator,
        and writes ``stage4_eora_path`` back onto ``ctx``. Returns ``None``.

        Note: Stage 4's orchestrator ``run`` takes neither ``decomposition``
        nor ``device`` (unlike Stage 3), so this adapter reads neither.
        """
        model = ctx.get("model")
        tokenizer = ctx.get("tokenizer")
        config = ctx.get("config")
        artifacts_dir = ctx.get("artifacts_dir")
        no_resume = ctx.get("no_resume") if ctx.has("no_resume") else False

        out_dir = _orchestrator_run(
            model, tokenizer, config, artifacts_dir,
            no_resume=no_resume,
        )

        ctx.set("stage4_eora_path", out_dir)
        return None


STAGE4 = _Stage4()

__all__ = ["STAGE4"]
