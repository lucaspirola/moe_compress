"""Router-KD as ``Stage``-conforming objects — the orchestrator-facing adapter.

Router-KD is a UNIFIED module: the same code serves Stage 2.5 and Stage 5,
selected by the monolith's keyword-only ``stage_key`` parameter
(``"stage2p5"`` / ``"stage5"``). Unlike Stages 1..4 — each a single
module-level singleton (``STAGE1`` … ``STAGE4``) — Router-KD needs *two*
``Stage`` objects, one per invocation. :func:`make_router_kd_stage` is the
factory that builds them.

The factory takes the canonical :class:`~moe_compress.pipeline.stage.Stage`
``stage_id`` values ``"2.5"`` / ``"5"`` (consistent with
``STAGE1.stage_id == "1"`` … ``STAGE4.stage_id == "4"``) and maps each to the
monolith's internal ``stage_key`` form. The ``Stage`` Protocol is structural
and explicitly sanctions factory-built stages that set ``stage_id`` as an
instance attribute (see ``pipeline/stage.py``).

This is a *purely additive* adapter — it holds no Router-KD logic. :meth:`run`
only unwraps the :class:`~moe_compress.pipeline.context.PipelineContext` into
the orchestrator's positional/keyword arguments and writes the output path
back onto the context.

Context slots
-------------
Reads (required): ``tokenizer``, ``config``, ``artifacts_dir``; plus the
student model — read from ``student`` when present, else from ``model``.

Reads (optional): ``device`` — passed through when present, else ``None``;
``no_resume`` — passed through when present, else ``False``.

Writes: ``router_kd_<stage_key>_path`` — the ``Path`` of the KD-trained
checkpoint directory returned by the orchestrator. The slot is namespaced per
``stage_key`` (``router_kd_stage2p5_path`` / ``router_kd_stage5_path``) so the
two Router-KD invocations in a single run do not collide.
"""
from __future__ import annotations

from ..pipeline.context import PipelineContext
from .orchestrator import run as _orchestrator_run

# Maps the canonical Protocol stage_id ("2.5"/"5") to the monolith stage_key.
_STAGE_KEY = {"2.5": "stage2p5", "5": "stage5"}


class _RouterKdStage:
    """``Stage``-conforming adapter for one Router-KD invocation.

    Built by :func:`make_router_kd_stage`. Each instance is bound to one
    ``stage_id`` (``"2.5"`` or ``"5"``) and threads the matching ``stage_key``
    into :func:`moe_compress.router_kd.orchestrator.run`.

    ``stage_id`` is set as an INSTANCE attribute in :meth:`__init__` — the
    ``Stage`` Protocol is structural and sanctions factory-built stages doing
    exactly this (unlike the Stage 1..4 singletons, which declare it as a
    class attribute).
    """

    def __init__(self, stage_id: str) -> None:
        if stage_id not in _STAGE_KEY:
            raise ValueError(
                f"make_router_kd_stage: unsupported stage_id={stage_id!r}; "
                f"expected one of {sorted(_STAGE_KEY)}"
            )
        self.stage_id: str = stage_id
        self._stage_key: str = _STAGE_KEY[stage_id]

    def is_enabled(self, config: dict) -> bool:
        """Always ``True`` — Router-KD is mandatory at both Stage 2.5 and
        Stage 5; stage selection belongs to the future universal orchestrator,
        not to the stage itself."""
        return True

    def run(self, ctx: PipelineContext) -> None:
        """Run this Router-KD invocation by unwrapping ``ctx`` into the call.

        Reads the student model (``student`` slot, falling back to ``model``),
        ``tokenizer``, ``config`` and ``artifacts_dir`` from ``ctx`` (plus
        optional ``device``, ``no_resume``), calls the Router-KD orchestrator
        with this instance's ``stage_key``, and writes the namespaced
        ``router_kd_<stage_key>_path`` slot back onto ``ctx``. Returns ``None``.
        """
        student = ctx.get("student") if ctx.has("student") else ctx.get("model")
        tokenizer = ctx.get("tokenizer")
        config = ctx.get("config")
        artifacts_dir = ctx.get("artifacts_dir")
        device = ctx.get("device") if ctx.has("device") else None
        no_resume = ctx.get("no_resume") if ctx.has("no_resume") else False

        out_dir = _orchestrator_run(
            student, tokenizer, config, artifacts_dir,
            device=device, no_resume=no_resume, stage_key=self._stage_key,
        )

        ctx.set(f"router_kd_{self._stage_key}_path", out_dir)
        return None


def make_router_kd_stage(stage_id: str) -> _RouterKdStage:
    """Build a ``Stage``-conforming Router-KD adapter for ``stage_id``.

    ``stage_id`` must be the canonical Protocol value ``"2.5"`` or ``"5"``;
    any other value (including the monolith's ``stage_key`` form ``"stage5"``)
    raises :class:`ValueError`.
    """
    return _RouterKdStage(stage_id)


__all__ = ["make_router_kd_stage"]
