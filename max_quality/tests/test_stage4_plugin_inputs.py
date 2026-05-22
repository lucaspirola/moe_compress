"""S4-2 — EoRA input-load plugin extraction tests.

Verifies the S4-2 ``EoraInputsPlugin`` scaffolding in
``stage4/plugins/eora_inputs.py``:

* the plugin class imports from the plugin module;
* ``EoraInputsPlugin`` satisfies the universal ``PipelinePlugin`` Protocol,
  carries correct metadata, is unconditionally enabled, and exposes the
  (S4-4) ``load_eora_inputs`` phase hook;
* the module never imports the ``stage4_eora`` monolith or
  ``stage4.orchestrator`` at module scope (the circular-import contract).

Unlike S3-2/S3-3, S4-2 relocates no standalone function — the covered logic
is inline in the ``stage4_eora.py`` monolith ``run()``. The plugin hook
reproduces it; the monolith is NOT modified. The byte-identical behavioral
gate is the S4-0 golden snapshot (``test_stage4_golden_snapshot.py``), which
is trivially safe since the monolith stays byte-identical; this file only
checks the relocation plumbing.
"""
from __future__ import annotations

import ast
from pathlib import Path


def test_eora_inputs_module_imports():
    """``EoraInputsPlugin`` imports from the plugin module and is a class."""
    from moe_compress.stage4.plugins.eora_inputs import EoraInputsPlugin

    assert isinstance(EoraInputsPlugin, type)


def test_plugin_satisfies_protocol():
    """``EoraInputsPlugin`` structurally satisfies ``PipelinePlugin``."""
    from moe_compress.pipeline.plugin import PipelinePlugin
    from moe_compress.stage4.plugins.eora_inputs import EoraInputsPlugin

    assert isinstance(EoraInputsPlugin(), PipelinePlugin)


def test_plugin_metadata():
    """Plugin metadata — name / paper id / config_key / tuple-typed fields."""
    from moe_compress.stage4.plugins.eora_inputs import EoraInputsPlugin

    plugin = EoraInputsPlugin()
    assert plugin.name == "eora_inputs"
    assert "2410.21271" in plugin.paper
    assert plugin.config_key == "stage4_eora.compensation_budget_pct"
    assert isinstance(plugin.reads, tuple)
    assert isinstance(plugin.writes, tuple)
    assert isinstance(plugin.provides, tuple)


def test_plugin_is_enabled_unconditional():
    """The EoRA input load is UNCONDITIONAL — ``is_enabled`` always True.

    Every Stage 4 run must load the A-cov and Stage-3 originals;
    ``config_key`` only parametrises the downstream compensation budget,
    never the plugin as a whole.
    """
    from moe_compress.stage4.plugins.eora_inputs import EoraInputsPlugin

    plugin = EoraInputsPlugin()
    assert plugin.is_enabled({}) is True
    assert plugin.is_enabled(
        {"stage4_eora": {"compensation_budget_pct": 0.0}}
    ) is True


def test_plugin_has_load_eora_inputs_hook():
    """The S4-4 phase hook ``load_eora_inputs`` is present and callable."""
    from moe_compress.stage4.plugins.eora_inputs import EoraInputsPlugin

    plugin = EoraInputsPlugin()
    assert callable(getattr(plugin, "load_eora_inputs", None))


def test_no_monolith_import_at_any_scope():
    """The module never imports ``stage4_eora`` / ``stage4.orchestrator``.

    The module docstring's contract says NEVER import the monolith (or the
    orchestrator that S4-4 will make import this module) at any scope —
    module-top OR function-local — since either would risk an import cycle.
    Parse the source with ``ast`` and walk the FULL tree so a function-local
    ``import stage4_eora`` cannot slip past. Assert no ``Import`` /
    ``ImportFrom`` names the forbidden modules at any nesting level.
    """
    from moe_compress.stage4.plugins import eora_inputs as mod

    src = Path(mod.__file__).read_text()
    tree = ast.parse(src)
    forbidden = ("stage4_eora", "stage4.orchestrator")
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
