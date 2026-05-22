"""S3-6 — Phase C.5 block-refine plugin extraction tests.

Verifies the pure-relocation of the two Phase C.5 block-refine symbols out of
the ``stage3_svd.py`` monolith into ``stage3/plugins/block_refine.py``:

* ``_phase_c5_block_refine`` (carrying its three nested closures
  ``_capture_first_pass`` / ``_capture_block_input`` / ``_lr_at``) and
  ``_advance_streams``;
* the plugin module exposes the relocated symbols;
* the monolith RE-IMPORTS them (identity, not copy) so ``run()`` and external
  callers keep their import paths;
* ``BlockRefinePlugin`` satisfies the universal ``PipelinePlugin`` Protocol,
  carries correct metadata, and exposes the (S3-7) ``refine_blocks`` phase hook;
* ``BlockRefinePlugin`` is the FIRST genuinely config-gated stage-3 plugin —
  ``is_enabled`` returns the ``stage3_svd.block_refine.enabled`` flag (default
  False) rather than an unconditional ``True``.

The byte-identical behavioral gate is the S3-0 golden snapshot
(``test_stage3_golden_snapshot.py``); this file only checks the relocation
plumbing plus the ``is_enabled`` gating logic.
"""
from __future__ import annotations


_BLOCK_REFINE_SYMBOLS = (
    "_phase_c5_block_refine",
    "_advance_streams",
)


def test_block_refine_module_imports():
    """The 2 relocated symbols + ``BlockRefinePlugin`` import from the plugin
    module with the correct kinds (callables / type)."""
    from moe_compress.stage3.plugins import block_refine
    from moe_compress.stage3.plugins.block_refine import BlockRefinePlugin

    for name in _BLOCK_REFINE_SYMBOLS:
        assert hasattr(block_refine, name), name
        assert callable(getattr(block_refine, name)), name
    assert isinstance(BlockRefinePlugin, type)


def test_monolith_reexports_block_refine_symbols():
    """The monolith re-imports the relocated symbols — identity, not copy.

    ``IS`` identity proves ``stage3_svd`` holds the *same* objects as the
    plugin module (a re-import), not independent copies that could drift.
    """
    import moe_compress.stage3_svd as monolith
    import moe_compress.stage3.plugins.block_refine as plugin

    for name in _BLOCK_REFINE_SYMBOLS:
        assert getattr(monolith, name) is getattr(plugin, name), name


def test_plugin_satisfies_protocol():
    """``BlockRefinePlugin`` structurally satisfies ``PipelinePlugin``."""
    from moe_compress.pipeline.plugin import PipelinePlugin
    from moe_compress.stage3.plugins.block_refine import BlockRefinePlugin

    assert isinstance(BlockRefinePlugin(), PipelinePlugin)


def test_plugin_metadata():
    """Plugin metadata — name / paper id / config_key / tuple-typed fields.

    ``writes == ()``: Phase C.5 refines the installed ``FactoredExperts`` U/V
    slots in place and produces no new ctx slot.
    """
    from moe_compress.stage3.plugins.block_refine import BlockRefinePlugin

    plugin = BlockRefinePlugin()
    assert plugin.name == "block_refine"
    assert "2604.02119" in plugin.paper
    assert plugin.config_key == "stage3_svd.block_refine.enabled"
    assert isinstance(plugin.reads, tuple)
    assert isinstance(plugin.writes, tuple)
    assert isinstance(plugin.provides, tuple)
    assert plugin.writes == ()


def test_plugin_is_enabled_gates():
    """``BlockRefinePlugin`` is the FIRST genuinely config-gated stage-3 plugin.

    ``is_enabled`` returns the ``stage3_svd.block_refine.enabled`` flag,
    replicating the monolith ``run()``'s ``_block_refine_enabled`` navigation
    (``config["stage3_svd"]["block_refine"]["enabled"]``) and defaulting to
    False when any key in the chain is absent. Results are real ``bool``.
    """
    from moe_compress.stage3.plugins.block_refine import BlockRefinePlugin

    plugin = BlockRefinePlugin()
    assert plugin.is_enabled(
        {"stage3_svd": {"block_refine": {"enabled": True}}}
    ) is True
    assert plugin.is_enabled(
        {"stage3_svd": {"block_refine": {"enabled": False}}}
    ) is False
    assert plugin.is_enabled({}) is False
    assert plugin.is_enabled({"stage3_svd": {}}) is False
    assert plugin.is_enabled({"stage3_svd": {"block_refine": {}}}) is False


def test_plugin_has_refine_blocks_hook():
    """The S3-7 phase hook ``refine_blocks`` is present and callable."""
    from moe_compress.stage3.plugins.block_refine import BlockRefinePlugin

    plugin = BlockRefinePlugin()
    assert callable(getattr(plugin, "refine_blocks", None))
