"""Unit tests for the Stage 2 pipeline scaffolding.

Validates ``PipelineContext`` set-once / child-scope semantics and the
universal ``walk_phases`` phase walk's setup / teardown fan-out plus the
canonical ``_STAGE2_LAYER_PHASES`` schedule. Plugins are plain classes that
satisfy the universal
:class:`~moe_compress.pipeline.plugin.PipelinePlugin` Protocol structurally;
there is no stage-2-specific plugin base class or mutable registry. No torch /
numpy needed at this layer.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from moe_compress.pipeline.context import PipelineContext
from moe_compress.stage2.orchestrator import _STAGE2_LAYER_PHASES
from moe_compress.tools.phase_walker import walk_phases


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _RecordingPlugin:
    """Plain plugin that records lifecycle calls so tests assert dispatch order."""

    name = "recording"

    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def on_run_setup(self, run_ctx):
        self.calls.append(("on_run_setup", run_ctx))

    def on_run_teardown(self, run_ctx):
        self.calls.append(("on_run_teardown", run_ctx))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_run_ctx(tmp_path: Path, cfg: dict | None = None) -> PipelineContext:
    rc = PipelineContext()
    rc.set("model", object())
    rc.set("tokenizer", object())
    rc.set("config", cfg or {})
    rc.set("artifacts_dir", tmp_path)
    rc.set("partial_dir", tmp_path / "_partial")
    rc.set("device", "cpu")
    return rc


def _make_layer_ctx() -> PipelineContext:
    ctx = PipelineContext().child()
    ctx.set("layer_idx", 0)
    ctx.set("layer_ref", object())
    ctx.set("n_experts", 4)
    ctx.set("target", 2)
    ctx.set("blacklist", ())
    return ctx


# ---------------------------------------------------------------------------
# Pipeline shell
# ---------------------------------------------------------------------------


def test_pipeline_phases_are_declared_in_canonical_order():
    """_STAGE2_LAYER_PHASES documents the per-layer execution order.

    The tuple is 10 entries. The four fine-grained sub-hooks
    (``compute_cost``, ``apply_cost_mask``, ``solve_assignment``,
    ``refine_assignment``) are an open vocabulary discovered reflectively by
    the tolerant phase walk and are NOT iterated by ``walk_phases`` over this
    schedule — ``orchestrator._run_assignment`` drives them inside the
    compound ``compute_assignment`` slot to preserve the bump-loop control
    flow.
    """
    assert _STAGE2_LAYER_PHASES == (
        "on_layer_setup",
        "on_profile",
        "on_score",
        "compute_assignment",
        "pre_merge_snapshot",
        "merge",
        "post_merge",
        "write_artifacts",
        "on_post_merge",
        "on_layer_teardown",
    )


def test_run_setup_and_teardown_fan_out_in_order(tmp_path):
    """walk_phases invokes every plugin's run-scope hook in registration order."""
    a, b = _RecordingPlugin(), _RecordingPlugin()
    plugins = [a, b]
    run_ctx = _make_run_ctx(tmp_path)

    walk_phases(("on_run_setup",), plugins, run_ctx)
    walk_phases(("on_run_teardown",), plugins, run_ctx)

    assert [name for name, _ in a.calls] == ["on_run_setup", "on_run_teardown"]
    assert [name for name, _ in b.calls] == ["on_run_setup", "on_run_teardown"]
    # Same run-scope PipelineContext instance passed to every plugin.
    assert all(arg is run_ctx for _, arg in a.calls)
    assert all(arg is run_ctx for _, arg in b.calls)


def test_run_context_slots_are_set_once():
    """PipelineContext slots are set-once: a second write without overwrite raises."""
    rc = _make_run_ctx(Path("/tmp"))
    with pytest.raises(KeyError):
        rc.set("device", "cuda")  # already written by _make_run_ctx
    # overwrite=True is the explicit escape hatch.
    rc.set("device", "cuda", overwrite=True)
    assert rc.get("device") == "cuda"


def test_child_context_accepts_arbitrary_slots():
    """A child PipelineContext is an open namespace — any plugin can add a slot."""
    lc = _make_layer_ctx()
    lc.set("scores", "fake-scores")
    lc.set("my_plugin", {"foo": 1})
    assert lc.get("scores") == "fake-scores"
    assert lc.get("my_plugin") == {"foo": 1}
    # Run-scope slots resolve through the parent chain via get/has.
    assert lc.has("layer_idx")
