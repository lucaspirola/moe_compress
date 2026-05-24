"""Stage 3 as a ``Stage``-conforming object — the orchestrator-facing adapter.

This module exposes the existing Stage 3 plugin pipeline
(:func:`moe_compress.stage3.orchestrator.run`) through the universal
:class:`~moe_compress.pipeline.stage.Stage` Protocol, so the future universal
orchestrator can drive Stage 3 the same way it drives every other stage:
iterate over :class:`Stage` objects, call :meth:`Stage.is_enabled`, then
:meth:`Stage.run`.

This is a *purely additive* adapter — it holds no Stage 3 logic. :meth:`run`
only unwraps the :class:`~moe_compress.pipeline.context.PipelineContext` into
the orchestrator's positional/keyword arguments and writes the output path
back onto the context. The Stage 3 work itself is unchanged and still lives
in ``orchestrator.py`` and the ``stage3/plugins/`` plugins.

Context slots
-------------
Reads (required): ``model``, ``tokenizer``, ``config``, ``artifacts_dir``,
``decomposition``.

Reads (optional): ``device`` — passed through when present, else ``None``;
``no_resume`` — passed through when present, else ``False``.

Writes: ``stage3_svd_path`` — the ``Path`` of the SVD-factorized checkpoint
directory returned by the orchestrator.
"""

from __future__ import annotations

from ..pipeline.context import PipelineContext
from .orchestrator import run as _orchestrator_run


class _Stage3:
    """``Stage``-conforming adapter for Stage 3 (non-uniform SVD factorization).

    A thin shim over :func:`moe_compress.stage3.orchestrator.run`. The
    module-level singleton :data:`STAGE3` is the object the orchestrator uses.

    Context slots
    -------------
    Reads (required): ``model``, ``tokenizer``, ``config``, ``artifacts_dir``,
    ``decomposition``.

    Reads (optional): ``device`` — passed through when present, else ``None``;
    ``no_resume`` — passed through when present, else ``False``.

    Writes: ``stage3_svd_path`` — the ``Path`` of the SVD-factorized
    checkpoint directory returned by the orchestrator. The ``stage3_`` prefix
    namespaces it to avoid future cross-stage slot collisions.
    """

    stage_id: str = "3"

    def is_enabled(self, config: dict) -> bool:
        """Always ``True`` — Stage 3 is mandatory; stage selection belongs to
        the future universal orchestrator, not to the stage itself. There is
        no ``stage3_svd.enabled`` config knob."""
        return True

    def run(self, ctx: PipelineContext) -> None:
        """Run Stage 3 by unwrapping ``ctx`` into the orchestrator call.

        Reads ``model``, ``tokenizer``, ``config``, ``artifacts_dir`` and
        ``decomposition`` from ``ctx`` (plus optional ``device``,
        ``no_resume``), calls the Stage 3 orchestrator, and writes
        ``stage3_svd_path`` back onto ``ctx``. Returns ``None``.
        """
        model = ctx.get("model")
        tokenizer = ctx.get("tokenizer")
        config = ctx.get("config")
        artifacts_dir = ctx.get("artifacts_dir")
        decomposition = ctx.get("decomposition")
        device = ctx.get("device") if ctx.has("device") else None
        no_resume = ctx.get("no_resume") if ctx.has("no_resume") else False

        out_dir = _orchestrator_run(
            model, tokenizer, config, artifacts_dir, decomposition,
            device=device,
            no_resume=no_resume,
        )

        ctx.set("stage3_svd_path", out_dir)
        return None


STAGE3 = _Stage3()

__all__ = ["STAGE3"]
