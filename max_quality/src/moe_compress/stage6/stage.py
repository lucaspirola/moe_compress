"""Stage 6 as a ``Stage``-conforming object — the orchestrator-facing adapter.

This module exposes the existing Stage 6 plugin pipeline
(:func:`moe_compress.stage6.orchestrator.run`) through the universal
:class:`~moe_compress.pipeline.stage.Stage` Protocol, so the future universal
orchestrator can drive Stage 6 the same way it drives every other stage:
iterate over :class:`Stage` objects, call :meth:`Stage.is_enabled`, then
:meth:`Stage.run`.

This is a *purely additive* adapter — it holds no Stage 6 logic. :meth:`run`
only unwraps the :class:`~moe_compress.pipeline.context.PipelineContext` into
the orchestrator's positional/keyword arguments and writes the output path
back onto the context. The Stage 6 work itself is unchanged and still lives
in ``orchestrator.py`` and the ``stage6/plugins/`` plugins.

Context slots
-------------
Reads (required): ``model``, ``tokenizer``, ``config``, ``artifacts_dir``.

Reads (optional): ``device`` — passed through when present, else ``None``.

Writes: ``stage6_eval_path`` — the ``Path`` of the ``stage6_eval.json``
artifact returned by the orchestrator.
"""

from __future__ import annotations

from ..pipeline.context import PipelineContext
from .orchestrator import run as _orchestrator_run


class _Stage6:
    """``Stage``-conforming adapter for Stage 6 (validation gate).

    A thin shim over :func:`moe_compress.stage6.orchestrator.run`. The
    module-level singleton :data:`STAGE6` is the object the orchestrator uses.

    Context slots
    -------------
    Reads (required): ``model``, ``tokenizer``, ``config``, ``artifacts_dir``.

    Reads (optional): ``device`` — passed through when present, else ``None``.

    Writes: ``stage6_eval_path`` — the ``Path`` of the ``stage6_eval.json``
    artifact returned by the orchestrator. The ``stage6_`` prefix namespaces
    it to avoid future cross-stage slot collisions.
    """

    stage_id: str = "6"

    def is_enabled(self, config: dict) -> bool:
        """Always ``True`` — Stage 6 is mandatory; stage selection belongs to
        the future universal orchestrator, not to the stage itself. There is
        no ``stage6_validate.enabled`` config knob."""
        return True

    def run(self, ctx: PipelineContext) -> None:
        """Run Stage 6 by unwrapping ``ctx`` into the orchestrator call.

        Reads ``model``, ``tokenizer``, ``config`` and ``artifacts_dir`` from
        ``ctx`` (plus optional ``device``), calls the Stage 6 orchestrator,
        and writes ``stage6_eval_path`` back onto ``ctx``. Returns ``None``.
        """
        model = ctx.get("model")
        tokenizer = ctx.get("tokenizer")
        config = ctx.get("config")
        artifacts_dir = ctx.get("artifacts_dir")
        device = ctx.get("device") if ctx.has("device") else None

        out_path = _orchestrator_run(
            model, tokenizer, config, artifacts_dir, device=device,
        )

        ctx.set("stage6_eval_path", out_path)
        return None


STAGE6 = _Stage6()

__all__ = ["STAGE6"]
