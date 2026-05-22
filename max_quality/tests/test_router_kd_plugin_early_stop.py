"""RK-7 — Router-KD best-tracker + early-stop plugin extraction tests.

Verifies the RK-7 ``EarlyStopPlugin`` scaffolding in
``router_kd/plugins/early_stop.py``:

* the relocated ``_save_best_router_state`` and ``EarlyStopPlugin`` import
  from the plugin module;
* the ``stage5_router_kd`` monolith re-exports the SAME
  ``_save_best_router_state`` object (the ``# noqa: F401`` re-import block is
  load-bearing — ``run()`` calls it on its original import path);
* ``EarlyStopPlugin`` satisfies the universal ``PipelinePlugin`` Protocol,
  carries correct metadata, and exposes the four (RK-8) early-stop hooks;
* the UNCONDITIONAL ``is_enabled`` — best-tracking always runs; the patience
  block is gated internally, not by ``is_enabled`` (mirrors ``VocabKdPlugin``,
  NOT the stage-gated ``MergeRepairPlugin``);
* the module never imports the ``stage5_router_kd`` monolith or
  ``router_kd.orchestrator`` at any scope (the circular-import contract);
* a light check of the relocated best-tracker / early-stop logic itself.

RK-7 is a MIXED pattern: a Pattern A relocation (``_save_best_router_state``
relocated verbatim, the monolith re-imports it) plus a Pattern B
``EarlyStopPlugin`` reproducing the inline ``run()`` best-tracker /
early-stop glue in four INERT hooks (the monolith ``run()`` is untouched).
The byte-identical behavioral gate is the RK-0 golden snapshot
(``test_router_kd_golden_snapshot.py``); the live early-stop coverage lives
in ``test_stage5_early_stop.py``. This file checks the relocation plumbing
and a light slice of the hook logic.
"""
from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from moe_compress.pipeline.context import PipelineContext


def test_early_stop_module_imports():
    """Both RK-7 symbols import from the plugin module."""
    from moe_compress.router_kd.plugins.early_stop import (
        EarlyStopPlugin,
        _save_best_router_state,
    )

    assert isinstance(EarlyStopPlugin, type)
    assert callable(_save_best_router_state)


def test_monolith_reexports_save_best_router_state():
    """The monolith re-exports the SAME ``_save_best_router_state`` object.

    ``is``-identity proves the ``# noqa: F401`` re-import block in
    ``stage5_router_kd.py`` keeps ``run()`` on its original import path — the
    inline early-stop glue calls ``_save_best_router_state`` unqualified.
    """
    from moe_compress import stage5_router_kd
    from moe_compress.router_kd.plugins import early_stop

    assert (
        stage5_router_kd._save_best_router_state
        is early_stop._save_best_router_state
    )


def test_plugin_satisfies_protocol():
    """``EarlyStopPlugin`` structurally satisfies ``PipelinePlugin``."""
    from moe_compress.pipeline.plugin import PipelinePlugin
    from moe_compress.router_kd.plugins.early_stop import EarlyStopPlugin

    assert isinstance(EarlyStopPlugin(), PipelinePlugin)


def test_plugin_metadata():
    """Plugin metadata — name / paper id / config_key / tuple-typed fields."""
    from moe_compress.router_kd.plugins.early_stop import EarlyStopPlugin

    plugin = EarlyStopPlugin()
    assert plugin.name == "early_stop"
    assert "2603.02217" in plugin.paper
    assert plugin.config_key == "stage5_router_kd.early_stop_patience"
    assert isinstance(plugin.reads, tuple)
    assert isinstance(plugin.writes, tuple)
    assert isinstance(plugin.provides, tuple)
    assert plugin.provides == ()
    assert "early_stop_should_stop" in plugin.writes
    assert "best_raw_kl_ema" in plugin.writes


def test_is_enabled_unconditional():
    """``is_enabled`` is unconditionally True — best-tracking always runs.

    Mirrors ``VocabKdPlugin``: ``config_key`` only names the patience knob;
    the patience block is gated INTERNALLY on ``early_stop_patience > 0``,
    not by ``is_enabled``. So even an empty config / patience=0 → True.
    """
    from moe_compress.router_kd.plugins.early_stop import EarlyStopPlugin

    plugin = EarlyStopPlugin()
    assert plugin.is_enabled({}) is True
    assert plugin.is_enabled(
        {"stage5_router_kd": {"early_stop_patience": 0}}
    ) is True
    assert plugin.is_enabled(
        {"stage5_router_kd": {"early_stop_patience": 5}}
    ) is True


def test_plugin_has_hooks():
    """The four RK-8 early-stop phase hooks are present and callable."""
    from moe_compress.router_kd.plugins.early_stop import EarlyStopPlugin

    plugin = EarlyStopPlugin()
    assert callable(getattr(plugin, "setup_early_stop", None))
    assert callable(getattr(plugin, "update_best_tracker", None))
    assert callable(getattr(plugin, "check_early_stop", None))
    assert callable(getattr(plugin, "reload_best_checkpoint", None))


def test_contribute_artifact_empty():
    """``contribute_artifact`` returns a fresh empty dict."""
    from moe_compress.router_kd.plugins.early_stop import EarlyStopPlugin

    plugin = EarlyStopPlugin()
    assert plugin.contribute_artifact(PipelineContext()) == {}


def test_no_monolith_import_at_any_scope():
    """The module never imports ``stage5_router_kd`` / ``router_kd.orchestrator``.

    The module docstring's contract says NEVER import the monolith (or the
    orchestrator) at any scope — module-top OR function-local — since either
    would risk an import cycle (the monolith re-imports *this* module at load
    time). Parse the source with ``ast`` and walk the FULL tree so a
    function-local ``import stage5_router_kd`` cannot slip past.

    For ``ImportFrom`` both halves are checked: ``node.module`` (the
    ``from X import ...`` package) AND ``node.names`` (the imported symbols) —
    so the cycle-causing ``from moe_compress import stage5_router_kd`` form
    (``module="moe_compress"``, name ``stage5_router_kd``) is also caught.
    """
    from moe_compress.router_kd.plugins import early_stop as mod

    src = Path(mod.__file__).read_text()
    tree = ast.parse(src)
    forbidden = ("stage5_router_kd", "router_kd.orchestrator", "orchestrator")
    for node in ast.walk(tree):  # any nesting level, not just module-top
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not any(f in alias.name for f in forbidden), (
                    f"forbidden import at any scope: {alias.name}"
                )
        elif isinstance(node, ast.ImportFrom):
            mod_name = node.module or ""
            assert not any(f in mod_name for f in forbidden), (
                f"forbidden import-from at any scope: {mod_name}"
            )
            # Also inspect the imported NAMES: ``from moe_compress import
            # stage5_router_kd`` carries the monolith as an ``alias.name``, not
            # in ``node.module`` — without this it would slip past undetected.
            for alias in node.names:
                assert not any(f in alias.name for f in forbidden), (
                    f"forbidden import-from name at any scope: "
                    f"from {mod_name} import {alias.name}"
                )


# ---------------------------------------------------------------------------
# Light logic checks — the heavy behavioral coverage of the live early-stop
# path lives in test_stage5_early_stop.py / test_smoke_stage5_resume.py.
# ---------------------------------------------------------------------------


class _OneParamModule(nn.Module):
    """Tiny model with one trainable param — enough for best.pt round-trips."""

    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.zeros(3))


def _setup_ctx(tmp_path, *, patience: int, save_best: bool = True,
               alpha: float = 0.2) -> PipelineContext:
    """Run ``setup_early_stop`` on a fresh ctx and return it."""
    from moe_compress.router_kd.plugins.early_stop import EarlyStopPlugin

    ctx = PipelineContext()
    ctx.set("config", {
        "stage5_router_kd": {
            "best_metric_ema_alpha": alpha,
            "save_best": save_best,
            "early_stop_patience": patience,
        }
    })
    ctx.set("partial_dir", tmp_path)
    ctx.set("student", _OneParamModule())
    ctx.set("stage_key", "stage5")
    EarlyStopPlugin().setup_early_stop(ctx)
    return ctx


def _window(ctx, plugin, *, step: int, raw_kl: float, epoch: int = 0) -> float:
    """Push one log window through ``update_best_tracker``; return the EMA."""
    ctx.set("step", step, overwrite=ctx.has("step"))
    ctx.set("epoch", epoch, overwrite=ctx.has("epoch"))
    ctx.set("raw_kl_val", raw_kl, overwrite=ctx.has("raw_kl_val"))
    plugin.update_best_tracker(ctx)
    return float(ctx.get("raw_kl_ema"))


def test_setup_seeds_state(tmp_path):
    """``setup_early_stop`` seeds the best-tracker + early-stop state."""
    ctx = _setup_ctx(tmp_path, patience=3, alpha=0.3)
    assert ctx.get("best_ema_alpha") == 0.3
    assert ctx.get("save_best") is True
    assert math.isinf(ctx.get("best_raw_kl_ema"))
    assert ctx.get("best_step") == -1
    assert math.isinf(ctx.get("prev_ema"))
    assert ctx.get("early_stop_patience") == 3
    assert ctx.get("no_improve_windows") == 0
    assert math.isinf(ctx.get("es_ref_ema"))


def test_setup_rejects_negative_patience(tmp_path):
    """``early_stop_patience < 0`` is fail-loud in setup."""
    from moe_compress.router_kd.plugins.early_stop import EarlyStopPlugin

    ctx = PipelineContext()
    ctx.set("config", {"stage5_router_kd": {"early_stop_patience": -1}})
    ctx.set("partial_dir", tmp_path)
    ctx.set("student", _OneParamModule())
    ctx.set("stage_key", "stage5")
    with pytest.raises(ValueError, match="must be >= 0"):
        EarlyStopPlugin().setup_early_stop(ctx)


def test_setup_resume_restore(tmp_path):
    """The resume-restore slots overwrite the freshly-seeded state."""
    from moe_compress.router_kd.plugins.early_stop import EarlyStopPlugin

    ctx = PipelineContext()
    ctx.set("config", {"stage5_router_kd": {"early_stop_patience": 5}})
    ctx.set("partial_dir", tmp_path)
    ctx.set("student", _OneParamModule())
    ctx.set("stage_key", "stage5")
    ctx.set("resume_best_raw_kl_ema", 0.42)
    ctx.set("resume_best_step", 17)
    ctx.set("resume_prev_ema", 0.5)
    ctx.set("resume_no_improve_windows", 2)
    ctx.set("resume_es_ref_ema", 0.41)
    EarlyStopPlugin().setup_early_stop(ctx)
    assert ctx.get("best_raw_kl_ema") == 0.42
    assert ctx.get("best_step") == 17
    assert ctx.get("prev_ema") == 0.5
    assert ctx.get("no_improve_windows") == 2
    assert ctx.get("es_ref_ema") == 0.41


def test_ema_bootstrap_then_decay(tmp_path):
    """First window bootstraps ema=raw_kl; later windows EMA-decay it."""
    from moe_compress.router_kd.plugins.early_stop import EarlyStopPlugin

    plugin = EarlyStopPlugin()
    ctx = _setup_ctx(tmp_path, patience=0, alpha=0.2)
    # prev_ema=+inf -> bootstrap: ema == raw_kl_val.
    ema0 = _window(ctx, plugin, step=1, raw_kl=2.0)
    assert ema0 == pytest.approx(2.0)
    # Second window: ema = 0.2*raw_kl + 0.8*prev_ema.
    ema1 = _window(ctx, plugin, step=2, raw_kl=1.0)
    assert ema1 == pytest.approx(0.2 * 1.0 + 0.8 * 2.0)


def test_save_best_writes_on_improvement_not_on_flat(tmp_path):
    """``best.pt`` is written on an improving window, not on a flat one."""
    from moe_compress.router_kd.plugins.early_stop import EarlyStopPlugin

    plugin = EarlyStopPlugin()
    ctx = _setup_ctx(tmp_path, patience=0, alpha=1.0)  # alpha=1 -> ema==raw_kl
    best_path = tmp_path / "best.pt"

    # First window: +inf seed -> always improves -> best.pt written.
    _window(ctx, plugin, step=1, raw_kl=1.0)
    assert best_path.exists()
    blob1 = torch.load(best_path, map_location="cpu")
    assert blob1["step"] == 1
    assert ctx.get("best_raw_kl_ema") == pytest.approx(1.0)
    assert ctx.get("best_step") == 1

    # Flat window (same ema) -> NOT < best -> best.pt unchanged.
    _window(ctx, plugin, step=2, raw_kl=1.0)
    blob2 = torch.load(best_path, map_location="cpu")
    assert blob2["step"] == 1  # still the step-1 snapshot
    assert ctx.get("best_step") == 1

    # Improving window -> best.pt rewritten at step 3.
    _window(ctx, plugin, step=3, raw_kl=0.5)
    blob3 = torch.load(best_path, map_location="cpu")
    assert blob3["step"] == 3
    assert ctx.get("best_raw_kl_ema") == pytest.approx(0.5)
    assert ctx.get("best_step") == 3


def test_patience_counter_climbs_and_resets(tmp_path):
    """The no-improve counter climbs on flat windows, resets on improvement."""
    from moe_compress.router_kd.plugins.early_stop import EarlyStopPlugin

    plugin = EarlyStopPlugin()
    # alpha=1.0 -> ema == raw_kl_val, so the metric trajectory is direct.
    ctx = _setup_ctx(tmp_path, patience=3, alpha=1.0)

    # Window 1: bootstrap, ema=1.0 < +inf es_ref -> counter stays 0.
    _window(ctx, plugin, step=1, raw_kl=1.0)
    assert ctx.get("no_improve_windows") == 0
    assert ctx.get("es_ref_ema") == pytest.approx(1.0)

    # Flat windows: ema not < es_ref_ema -> counter increments each window.
    _window(ctx, plugin, step=2, raw_kl=1.0)
    assert ctx.get("no_improve_windows") == 1
    _window(ctx, plugin, step=3, raw_kl=1.0)
    assert ctx.get("no_improve_windows") == 2

    # An improving window resets the counter to 0 and lowers es_ref_ema.
    _window(ctx, plugin, step=4, raw_kl=0.5)
    assert ctx.get("no_improve_windows") == 0
    assert ctx.get("es_ref_ema") == pytest.approx(0.5)


def test_check_early_stop_flips_flag_at_patience(tmp_path):
    """``check_early_stop`` flips ``early_stop_should_stop`` at the threshold."""
    from moe_compress.router_kd.plugins.early_stop import EarlyStopPlugin

    plugin = EarlyStopPlugin()
    ctx = _setup_ctx(tmp_path, patience=3, alpha=1.0)

    # Window 1: bootstrap -> counter 0; not stopping yet.
    _window(ctx, plugin, step=1, raw_kl=1.0)
    plugin.check_early_stop(ctx)
    assert ctx.get("early_stop_should_stop") is False

    # Flat windows climb the counter to patience=3.
    for s in (2, 3, 4):
        _window(ctx, plugin, step=s, raw_kl=1.0)
        plugin.check_early_stop(ctx)
    assert ctx.get("no_improve_windows") == 3
    assert ctx.get("early_stop_should_stop") is True


def test_check_early_stop_disabled_when_patience_zero(tmp_path):
    """patience=0 -> ``early_stop_should_stop`` never flips True."""
    from moe_compress.router_kd.plugins.early_stop import EarlyStopPlugin

    plugin = EarlyStopPlugin()
    ctx = _setup_ctx(tmp_path, patience=0, alpha=1.0)
    for s in range(1, 8):
        _window(ctx, plugin, step=s, raw_kl=1.0)
        plugin.check_early_stop(ctx)
        assert ctx.get("early_stop_should_stop") is False
    # The patience block is skipped entirely -> counter never advances.
    assert ctx.get("no_improve_windows") == 0


def test_save_best_false_still_advances_patience(tmp_path):
    """With ``save_best=false`` the patience ``es_ref_ema`` still advances.

    ``es_ref_ema`` is independent of the best.pt WRITE gate — early stopping
    must work even when save-best is off. No ``best.pt`` is written, but the
    counter climbs to patience and the stop flag flips.
    """
    from moe_compress.router_kd.plugins.early_stop import EarlyStopPlugin

    plugin = EarlyStopPlugin()
    ctx = _setup_ctx(tmp_path, patience=2, save_best=False, alpha=1.0)

    _window(ctx, plugin, step=1, raw_kl=1.0)
    assert ctx.get("es_ref_ema") == pytest.approx(1.0)
    # save_best off -> no best.pt, but best_raw_kl_ema stays +inf.
    assert not (tmp_path / "best.pt").exists()
    assert math.isinf(ctx.get("best_raw_kl_ema"))

    # Flat windows still advance the patience counter.
    _window(ctx, plugin, step=2, raw_kl=1.0)
    _window(ctx, plugin, step=3, raw_kl=1.0)
    plugin.check_early_stop(ctx)
    assert ctx.get("no_improve_windows") == 2
    assert ctx.get("early_stop_should_stop") is True


def test_reload_best_checkpoint_swaps_params(tmp_path):
    """``reload_best_checkpoint`` swaps the trainable params for the best.pt."""
    from moe_compress.router_kd.plugins.early_stop import (
        EarlyStopPlugin,
        _save_best_router_state,
    )

    plugin = EarlyStopPlugin()
    ctx = _setup_ctx(tmp_path, patience=0)
    student = ctx.get("student")

    # Snapshot a known "best" param state, then mutate the live model.
    with torch.no_grad():
        student.w.copy_(torch.tensor([1.0, 2.0, 3.0]))
    _save_best_router_state(tmp_path, student, step=5, epoch=0, raw_kl_ema=0.1)
    with torch.no_grad():
        student.w.copy_(torch.tensor([9.0, 9.0, 9.0]))

    plugin.reload_best_checkpoint(ctx)
    assert torch.equal(student.w.data, torch.tensor([1.0, 2.0, 3.0]))


def test_reload_best_checkpoint_noop_without_best_pt(tmp_path):
    """No ``best.pt`` (best-tracker never fired) -> reload is a no-op."""
    from moe_compress.router_kd.plugins.early_stop import EarlyStopPlugin

    plugin = EarlyStopPlugin()
    ctx = _setup_ctx(tmp_path, patience=0)
    student = ctx.get("student")
    with torch.no_grad():
        student.w.copy_(torch.tensor([4.0, 5.0, 6.0]))
    # No best.pt on disk -> reload leaves the live params untouched.
    plugin.reload_best_checkpoint(ctx)
    assert torch.equal(student.w.data, torch.tensor([4.0, 5.0, 6.0]))


def test_reload_best_checkpoint_noop_when_save_best_disabled(tmp_path):
    """``save_best=false`` -> reload returns early, never touches the model
    (even if a stale best.pt happens to exist on disk)."""
    from moe_compress.router_kd.plugins.early_stop import (
        EarlyStopPlugin,
        _save_best_router_state,
    )

    plugin = EarlyStopPlugin()
    ctx = _setup_ctx(tmp_path, patience=0, save_best=False)
    student = ctx.get("student")

    # A stale best.pt on disk must be ignored when save_best is off.
    with torch.no_grad():
        student.w.copy_(torch.tensor([1.0, 2.0, 3.0]))
    _save_best_router_state(tmp_path, student, step=5, epoch=0, raw_kl_ema=0.1)
    with torch.no_grad():
        student.w.copy_(torch.tensor([7.0, 8.0, 9.0]))

    plugin.reload_best_checkpoint(ctx)
    # save_best=false -> early return; live params untouched.
    assert torch.equal(student.w.data, torch.tensor([7.0, 8.0, 9.0]))
