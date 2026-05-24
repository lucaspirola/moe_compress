"""Stage 2 as a ``Stage``-conforming object — the orchestrator-facing adapter.

This module exposes the existing Stage 2 plugin pipeline
(:func:`moe_compress.stage2.orchestrator.run`) through the universal
:class:`~moe_compress.pipeline.stage.Stage` Protocol, so the future universal
orchestrator can drive Stage 2 the same way it drives every other stage:
iterate over :class:`Stage` objects, call :meth:`Stage.is_enabled`, then
:meth:`Stage.run`.

This is a *purely additive* adapter — it holds no Stage 2 logic. :meth:`run`
only unwraps the :class:`~moe_compress.pipeline.context.PipelineContext` into
the orchestrator's positional/keyword arguments and writes the output path
back onto the context. The Stage 2 work itself is unchanged and still lives
in ``orchestrator.py`` and the ``stage2/plugins/`` plugins.
"""

from __future__ import annotations

from ..pipeline.context import PipelineContext
from .orchestrator import run as _orchestrator_run


class _Stage2:
    """``Stage``-conforming adapter for Stage 2 (REAP scoring + REAM merging).

    A thin shim over :func:`moe_compress.stage2.orchestrator.run`. The
    module-level singleton :data:`STAGE2` is the object the orchestrator uses.

    Context slots
    -------------
    Reads (required): ``model``, ``tokenizer``, ``config``, ``artifacts_dir``.

    Reads (optional): ``device`` — passed through when present, else ``None``;
    ``no_resume`` — passed through when present, else ``False``;
    ``stage1_budgets_path`` — the budgets ``Path`` written by ``STAGE1.run``;
    when present it threads to the orchestrator's ``stage1_budget_path`` kwarg
    so STAGE1→STAGE2 compose via ctx, else ``None`` (the orchestrator's
    accepted default — it falls back to ``artifacts_dir/stage1_budgets.json``).

    Writes: ``stage2_pruned_path`` — the ``Path`` of the pruned checkpoint
    directory returned by the orchestrator. The ``stage2_`` prefix namespaces
    it to avoid future cross-stage slot collisions.
    """

    stage_id: str = "2"

    def is_enabled(self, config: dict) -> bool:
        """Always ``True`` — Stage 2 is mandatory; stage selection belongs to
        the future universal orchestrator, not to the stage itself. There is
        no ``stage2_reap_ream.enabled`` config knob."""
        return True

    def run(self, ctx: PipelineContext) -> None:
        """Run Stage 2 by unwrapping ``ctx`` into the orchestrator call.

        Reads ``model``, ``tokenizer``, ``config``, ``artifacts_dir`` from
        ``ctx`` (plus optional ``device``, ``no_resume``,
        ``stage1_budgets_path``), calls the Stage 2 orchestrator, and writes
        ``stage2_pruned_path`` back onto ``ctx``. Returns ``None``.
        """
        model = ctx.get("model")
        tokenizer = ctx.get("tokenizer")
        config = ctx.get("config")
        artifacts_dir = ctx.get("artifacts_dir")
        device = ctx.get("device") if ctx.has("device") else None
        no_resume = ctx.get("no_resume") if ctx.has("no_resume") else False
        stage1_budget_path = (
            ctx.get("stage1_budgets_path")
            if ctx.has("stage1_budgets_path")
            else None
        )

        out_dir = _orchestrator_run(
            model, tokenizer, config, artifacts_dir,
            device=device,
            stage1_budget_path=stage1_budget_path,
            no_resume=no_resume,
        )

        ctx.set("stage2_pruned_path", out_dir)
        return None


STAGE2 = _Stage2()

__all__ = ["STAGE2"]
