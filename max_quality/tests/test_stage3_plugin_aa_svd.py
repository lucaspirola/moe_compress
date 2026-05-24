"""S3-5 — AA-SVD factorization-core plugin extraction tests.

Verifies the pure-relocation of the six AA-SVD-core symbols out of the
``stage3_svd.py`` monolith into ``stage3/plugins/aa_svd_factor.py``:

* ``_NOISE_FLOOR_BY_DTYPE`` / ``_EighDecomp`` / ``_precompute_eigh`` /
  ``_aa_svd_precomputed`` / ``_aa_svd`` / ``_cov_lookup``;
* the plugin module exposes the relocated symbols;
* the monolith RE-IMPORTS them (identity, not copy) so ``run()`` and external
  callers keep their import paths;
* S3-4's ``swift_svd_alpha`` lazy import of the AA-SVD core (via the
  ``stage3_svd`` re-export) still resolves to the same objects;
* ``AaSvdFactorPlugin`` satisfies the universal ``PipelinePlugin`` Protocol,
  carries correct metadata, is unconditionally enabled, and exposes the (S3-7)
  ``factor_layer`` phase hook;
* ``_cov_lookup``'s ``up_proj`` → ``gate_proj`` fallback still behaves as in
  the monolith.

The byte-identical behavioral gate is the S3-0 golden snapshot
(``test_stage3_golden_snapshot.py``); this file only checks the relocation
plumbing plus the ``_cov_lookup`` pure-unit logic. The model-driven AA-SVD
numerics are covered by ``test_aa_svd_correctness.py`` /
``test_aa_svd_bf16_quantized.py`` / ``test_aa_svd_fp16_quantized.py``.
"""
from __future__ import annotations

import torch


_AA_SVD_SYMBOLS = (
    "_NOISE_FLOOR_BY_DTYPE",
    "_EighDecomp",
    "_precompute_eigh",
    "_aa_svd_precomputed",
    "_aa_svd",
    "_cov_lookup",
)


def test_aa_svd_module_imports():
    """The 6 relocated symbols + ``AaSvdFactorPlugin`` import from the plugin
    module with the correct kinds (dict / type / callables)."""
    from moe_compress.stage3.plugins import aa_svd_factor
    from moe_compress.stage3.plugins.aa_svd_factor import AaSvdFactorPlugin

    for name in _AA_SVD_SYMBOLS:
        assert hasattr(aa_svd_factor, name), name
    assert isinstance(aa_svd_factor._NOISE_FLOOR_BY_DTYPE, dict)
    assert isinstance(aa_svd_factor._EighDecomp, type)
    for name in ("_precompute_eigh", "_aa_svd_precomputed", "_aa_svd", "_cov_lookup"):
        assert callable(getattr(aa_svd_factor, name)), name
    assert isinstance(AaSvdFactorPlugin, type)


def test_monolith_reexports_aa_svd_symbols():
    """The monolith re-imports the relocated symbols — identity, not copy.

    ``IS`` identity proves ``stage3_svd`` holds the *same* objects as the
    plugin module (a re-import), not independent copies that could drift.
    """
    import moe_compress.stage3_svd as monolith
    import moe_compress.stage3.plugins.aa_svd_factor as plugin

    for name in _AA_SVD_SYMBOLS:
        assert getattr(monolith, name) is getattr(plugin, name), name


def test_swift_svd_lazy_import_resolves():
    """S3-4-survival regression guard: ``swift_svd_alpha``'s function-scope
    lazy import of the AA-SVD core (``from ...stage3_svd import ...``) still
    resolves — and resolves to the *same* ``aa_svd_factor`` objects.
    """
    from moe_compress.stage3_svd import (
        _cov_lookup,
        _precompute_eigh,
        _aa_svd,
        _aa_svd_precomputed,
    )
    import moe_compress.stage3.plugins.aa_svd_factor as plugin

    assert _cov_lookup is plugin._cov_lookup
    assert _precompute_eigh is plugin._precompute_eigh
    assert _aa_svd is plugin._aa_svd
    assert _aa_svd_precomputed is plugin._aa_svd_precomputed


def test_plugin_satisfies_protocol():
    """``AaSvdFactorPlugin`` structurally satisfies ``PipelinePlugin``."""
    from moe_compress.pipeline.plugin import PipelinePlugin
    from moe_compress.stage3.plugins.aa_svd_factor import AaSvdFactorPlugin

    assert isinstance(AaSvdFactorPlugin(), PipelinePlugin)


def test_plugin_metadata():
    """Plugin metadata — name / paper id / config_key / tuple-typed fields."""
    from moe_compress.stage3.plugins.aa_svd_factor import AaSvdFactorPlugin

    plugin = AaSvdFactorPlugin()
    assert plugin.name == "aa_svd_factor"
    assert "2604.02119" in plugin.paper
    assert plugin.config_key == "stage3_svd.aa_svd.cross_covariance"
    assert isinstance(plugin.reads, tuple)
    assert isinstance(plugin.writes, tuple)
    assert isinstance(plugin.provides, tuple)
    assert "rank_map" in plugin.writes


def test_plugin_is_enabled_unconditional():
    """AA-SVD rank-k factorization is UNCONDITIONAL — ``is_enabled`` always True.

    ``config_key`` only selects Path 1 (cross-covariance) vs. Path 3 (B-only);
    it never gates the plugin as a whole.
    """
    from moe_compress.stage3.plugins.aa_svd_factor import AaSvdFactorPlugin

    plugin = AaSvdFactorPlugin()
    assert plugin.is_enabled({}) is True
    assert plugin.is_enabled(
        {"stage3_svd": {"aa_svd": {"cross_covariance": True}}}
    ) is True


def test_plugin_has_factor_layer_hook():
    """The S3-7 phase hook ``factor_layer`` is present and callable."""
    from moe_compress.stage3.plugins.aa_svd_factor import AaSvdFactorPlugin

    plugin = AaSvdFactorPlugin()
    assert callable(getattr(plugin, "factor_layer", None))


def test_cov_lookup_fallback():
    """``_cov_lookup`` does a per-bank covariance dict lookup with an
    ``up_proj`` → ``gate_proj`` fallback. Pure unit test, no model.

    * a direct (layer, expert, name) hit returns that entry;
    * a missing ``up_proj`` key falls back to the same expert's ``gate_proj``;
    * a missing key with no fallback (e.g. ``gate_proj`` / ``down_proj``)
      returns ``None``.
    """
    from moe_compress.stage3.plugins.aa_svd_factor import _cov_lookup

    gate0 = torch.eye(3)
    down0 = torch.ones(3, 3)
    up1 = torch.zeros(3, 3)
    cov = {
        (0, 0, "gate_proj"): gate0,
        (0, 0, "down_proj"): down0,
        (1, 0, "up_proj"): up1,
    }

    # Direct hits.
    assert _cov_lookup(cov, 0, 0, "gate_proj") is gate0
    assert _cov_lookup(cov, 0, 0, "down_proj") is down0
    assert _cov_lookup(cov, 1, 0, "up_proj") is up1

    # up_proj missing → falls back to the same expert's gate_proj.
    assert _cov_lookup(cov, 0, 0, "up_proj") is gate0

    # up_proj missing AND no gate_proj fallback present → None.
    assert _cov_lookup(cov, 5, 0, "up_proj") is None

    # Missing non-up_proj keys take no fallback → None.
    assert _cov_lookup(cov, 9, 0, "gate_proj") is None
    assert _cov_lookup(cov, 9, 0, "down_proj") is None
