"""RK-2 — Router-KD trainable-scope plugin extraction tests.

Verifies the RK-2 ``TrainableScopePlugin`` scaffolding in
``router_kd/plugins/trainable_scope.py``:

* ``_freeze_non_routers`` and ``TrainableScopePlugin`` import from the plugin
  module;
* the ``stage5_router_kd`` monolith re-exports the SAME ``_freeze_non_routers``
  object (the ``# noqa: F401`` re-import block is load-bearing);
* ``TrainableScopePlugin`` satisfies the universal ``PipelinePlugin``
  Protocol, carries correct metadata, is unconditionally enabled, and exposes
  the (RK-8) ``setup_trainable_scope`` phase hook;
* the module never imports the ``stage5_router_kd`` monolith or
  ``router_kd.orchestrator`` at any scope (the circular-import contract);
* ``_freeze_non_routers`` freezes the correct parameters;
* the ``setup_trainable_scope`` hook reproduces the monolith's
  trainable/frozen conflict check (raises ``RuntimeError`` on overlap) and the
  freeze.

RK-2 covers a MIXED pattern: ``_freeze_non_routers`` is relocated verbatim
(the monolith re-imports it); the conflict check — inline ``run()`` code in
the monolith — is reproduced in the inert hook (the monolith ``run()`` is NOT
modified for it). The byte-identical behavioral gate is the RK-0 golden
snapshot (``test_router_kd_golden_snapshot.py``); this file only checks the
relocation plumbing and the relocated logic.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest


def test_trainable_scope_module_imports():
    """``_freeze_non_routers`` / ``TrainableScopePlugin`` import from the module."""
    from moe_compress.router_kd.plugins.trainable_scope import (
        TrainableScopePlugin,
        _freeze_non_routers,
    )

    assert isinstance(TrainableScopePlugin, type)
    assert callable(_freeze_non_routers)


def test_monolith_reexports_freeze_non_routers():
    """The monolith re-exports the SAME ``_freeze_non_routers`` object.

    Proves the ``# noqa: F401`` re-import block in ``stage5_router_kd.py``
    keeps ``run()`` and external callers/tests on their original import path.
    """
    from moe_compress import stage5_router_kd
    from moe_compress.router_kd.plugins import trainable_scope

    assert (
        stage5_router_kd._freeze_non_routers
        is trainable_scope._freeze_non_routers
    )


def test_plugin_satisfies_protocol():
    """``TrainableScopePlugin`` structurally satisfies ``PipelinePlugin``."""
    from moe_compress.pipeline.plugin import PipelinePlugin
    from moe_compress.router_kd.plugins.trainable_scope import TrainableScopePlugin

    assert isinstance(TrainableScopePlugin(), PipelinePlugin)


def test_plugin_metadata():
    """Plugin metadata — name / paper id / config_key / tuple-typed fields."""
    from moe_compress.router_kd.plugins.trainable_scope import TrainableScopePlugin

    plugin = TrainableScopePlugin()
    assert plugin.name == "trainable_scope"
    assert "2603.02217" in plugin.paper
    assert plugin.config_key == "stage5_router_kd.trainable_name_patterns"
    assert isinstance(plugin.reads, tuple)
    assert isinstance(plugin.writes, tuple)
    assert isinstance(plugin.provides, tuple)


def test_plugin_is_enabled_unconditional():
    """Freezing non-router parameters is UNCONDITIONAL — ``is_enabled`` True.

    Every Router-KD run must freeze the non-router parameters before
    training; ``config_key`` only names which parameters stay trainable, it
    never gates the plugin as a whole.
    """
    from moe_compress.router_kd.plugins.trainable_scope import TrainableScopePlugin

    plugin = TrainableScopePlugin()
    assert plugin.is_enabled({}) is True
    assert plugin.is_enabled(
        {"stage5_router_kd": {"trainable_name_patterns": ["mlp.gate.weight"]}}
    ) is True


def test_plugin_has_setup_trainable_scope_hook():
    """The RK-8 phase hook ``setup_trainable_scope`` is present and callable."""
    from moe_compress.router_kd.plugins.trainable_scope import TrainableScopePlugin

    plugin = TrainableScopePlugin()
    assert callable(getattr(plugin, "setup_trainable_scope", None))


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
    from moe_compress.router_kd.plugins import trainable_scope as mod

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


def test_freeze_non_routers_freezes_correctly(tiny_model):
    """``_freeze_non_routers`` makes exactly the matching params trainable.

    With pattern ``["mlp.gate.weight"]`` only the router (gate) weights are
    trainable; every other parameter must have ``requires_grad=False``.
    """
    from moe_compress.router_kd.plugins.trainable_scope import _freeze_non_routers

    _freeze_non_routers(tiny_model, ["mlp.gate.weight"])

    trainable, frozen = [], []
    for name, p in tiny_model.named_parameters():
        (trainable if p.requires_grad else frozen).append(name)

    assert trainable, "expected at least one trainable param"
    for name in trainable:
        assert "mlp.gate.weight" in name, f"unexpectedly trainable: {name}"
    for name in frozen:
        assert "mlp.gate.weight" not in name, f"unexpectedly frozen: {name}"


def test_setup_trainable_scope_hook_conflict_raises(tiny_model):
    """The inert hook reproduces the monolith conflict check + freeze.

    Overlapping ``trainable_name_patterns`` / ``frozen_name_patterns`` (both
    matching a real parameter name) raises ``RuntimeError`` with the verbatim
    monolith message. A non-overlapping config raises nothing and freezes the
    model correctly.
    """
    from moe_compress.pipeline.context import PipelineContext
    from moe_compress.router_kd.plugins.trainable_scope import TrainableScopePlugin

    plugin = TrainableScopePlugin()

    # Overlapping patterns: "mlp.gate.weight" matches BOTH lists.
    ctx_conflict = PipelineContext()
    ctx_conflict.set("student", tiny_model)
    ctx_conflict.set("config", {
        "stage5_router_kd": {
            "trainable_name_patterns": ["mlp.gate.weight"],
            "frozen_name_patterns": ["gate.weight"],
        }
    })
    with pytest.raises(RuntimeError, match="match BOTH"):
        plugin.setup_trainable_scope(ctx_conflict)

    # Non-overlapping patterns: no raise, model frozen correctly.
    ctx_ok = PipelineContext()
    ctx_ok.set("student", tiny_model)
    ctx_ok.set("config", {
        "stage5_router_kd": {
            "trainable_name_patterns": ["mlp.gate.weight"],
            "frozen_name_patterns": ["experts", "shared_expert", "embed", "lm_head"],
        }
    })
    plugin.setup_trainable_scope(ctx_ok)

    for name, p in tiny_model.named_parameters():
        if "mlp.gate.weight" in name:
            assert p.requires_grad, f"expected trainable: {name}"
        else:
            assert not p.requires_grad, f"expected frozen: {name}"
