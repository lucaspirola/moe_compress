"""Stage 6alt as a ``Stage``-conforming object — the orchestrator-facing adapter.

This module exposes the existing Stage 6alt plugin pipeline
(:func:`moe_compress.stage6alt.orchestrator.run`) through the universal
:class:`~moe_compress.pipeline.stage.Stage` Protocol, so the future
universal orchestrator can drive Stage 6alt the same way it drives every
other stage: iterate over :class:`Stage` objects, call
:meth:`Stage.is_enabled`, then :meth:`Stage.run`.

This is a *purely additive* adapter — it holds no Stage 6alt logic.
:meth:`run` only unwraps the
:class:`~moe_compress.pipeline.context.PipelineContext` into the
orchestrator's positional/keyword arguments and writes the output path
back onto the context. The Stage 6alt work itself is unchanged and still
lives in ``orchestrator.py`` and the ``stage6alt/plugins/`` plugins.

Context slots
-------------
Reads (required): ``model``, ``tokenizer``, ``config``, ``artifacts_dir``.

Reads (optional): ``device`` — passed through when present, else ``None``.

Writes: ``stage6alt_eval_path`` — the ``Path`` of the
``stage6alt_eval.json`` artifact returned by the orchestrator.
"""
from __future__ import annotations

from ..pipeline.context import PipelineContext
from .orchestrator import run as _orchestrator_run


class _Stage6Alt:
    """``Stage``-conforming adapter for Stage 6alt (thermometer eval).

    A thin shim over :func:`moe_compress.stage6alt.orchestrator.run`. The
    module-level singleton :data:`STAGE6ALT` is the object the universal
    orchestrator uses.

    Context slots
    -------------
    Reads (required): ``model``, ``tokenizer``, ``config``, ``artifacts_dir``.

    Reads (optional): ``device`` — passed through when present, else ``None``.

    Writes: ``stage6alt_eval_path`` — the ``Path`` of the
    ``stage6alt_eval.json`` artifact returned by the orchestrator. The
    ``stage6alt_`` prefix namespaces it to avoid cross-stage slot
    collisions (mirrors STAGE6's ``stage6_eval_path``).
    """

    stage_id: str = "6alt"

    def is_enabled(self, config: dict) -> bool:
        """Always ``True`` — Stage 6alt selection (full vs thermometer) is a
        run-pipeline-level dispatch on ``stage6_validate.mode``, not a knob
        on the stage object itself. Mirrors :class:`_Stage6`.
        """
        return True

    def run(self, ctx: PipelineContext) -> None:
        """Run Stage 6alt by unwrapping ``ctx`` into the orchestrator call.

        Reads ``model``, ``tokenizer``, ``config`` and ``artifacts_dir``
        from ``ctx`` (plus optional ``device``), calls the Stage 6alt
        orchestrator, and writes ``stage6alt_eval_path`` back onto ``ctx``.
        Returns ``None``.
        """
        model = ctx.get("model")
        tokenizer = ctx.get("tokenizer")
        config = ctx.get("config")
        artifacts_dir = ctx.get("artifacts_dir")
        device = ctx.get("device") if ctx.has("device") else None

        out_path = _orchestrator_run(
            model, tokenizer, config, artifacts_dir, device=device,
        )

        ctx.set("stage6alt_eval_path", out_path)
        return None


STAGE6ALT = _Stage6Alt()

__all__ = ["STAGE6ALT"]
