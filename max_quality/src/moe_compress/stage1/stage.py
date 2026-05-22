"""Stage 1 as a ``Stage``-conforming object — the orchestrator-facing adapter.

This module exposes the existing Stage 1 plugin pipeline
(:func:`moe_compress.stage1.orchestrator.run`) through the universal
:class:`~moe_compress.pipeline.stage.Stage` Protocol, so the future universal
orchestrator can drive Stage 1 the same way it drives every other stage:
iterate over :class:`Stage` objects, call :meth:`Stage.is_enabled`, then
:meth:`Stage.run`.

This is a *purely additive* adapter — it holds no Stage 1 logic. :meth:`run`
only unwraps the :class:`~moe_compress.pipeline.context.PipelineContext` into
the orchestrator's positional/keyword arguments and writes the two output
paths back onto the context. The Stage 1 work itself is unchanged and still
lives in ``orchestrator.py`` and the eight ``stage1/plugins/`` plugins.
"""

from __future__ import annotations

from ..pipeline.context import PipelineContext
from .orchestrator import run as _orchestrator_run


class _Stage1:
    """``Stage``-conforming adapter for Stage 1 (SE detection + GRAPE budgets).

    A thin shim over :func:`moe_compress.stage1.orchestrator.run`. The
    module-level singleton :data:`STAGE1` is the object the orchestrator uses.

    Context slots
    -------------
    Reads (required): ``model``, ``tokenizer``, ``config``, ``artifacts_dir``,
    ``decomposition``.

    Reads (optional): ``device`` — passed through when present, else ``None``.

    Writes: ``stage1_blacklist_path`` and ``stage1_budgets_path`` — the two
    ``Path`` objects returned by the orchestrator. The ``stage1_`` prefix
    namespaces them to avoid future cross-stage slot collisions.
    """

    stage_id: str = "1"

    def is_enabled(self, config: dict) -> bool:
        """Always ``True`` — Stage 1 is mandatory; stage selection belongs to
        the future universal orchestrator, not to the stage itself."""
        return True

    def run(self, ctx: PipelineContext) -> None:
        """Run Stage 1 by unwrapping ``ctx`` into the orchestrator call.

        Reads ``model``, ``tokenizer``, ``config``, ``artifacts_dir`` and
        ``decomposition`` from ``ctx`` (plus optional ``device``), calls the
        Stage 1 orchestrator, and writes ``stage1_blacklist_path`` and
        ``stage1_budgets_path`` back onto ``ctx``. Returns ``None``.
        """
        model = ctx.get("model")
        tokenizer = ctx.get("tokenizer")
        config = ctx.get("config")
        artifacts_dir = ctx.get("artifacts_dir")
        decomposition = ctx.get("decomposition")
        device = ctx.get("device") if ctx.has("device") else None

        blacklist_path, budgets_path = _orchestrator_run(
            model, tokenizer, config, artifacts_dir, decomposition,
            device=device,
        )

        ctx.set("stage1_blacklist_path", blacklist_path)
        ctx.set("stage1_budgets_path", budgets_path)
        return None


STAGE1 = _Stage1()

__all__ = ["STAGE1"]
