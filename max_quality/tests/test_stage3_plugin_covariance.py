"""S3-2 — covariance-collection plugin extraction tests.

Verifies the pure-relocation of ``_collect_covariances`` /
``_load_stage2_covariance`` (+ RSS helpers) out of the ``stage3_svd.py``
monolith into ``stage3/plugins/covariance_collection.py``:

* the plugin module exposes the relocated symbols;
* the monolith RE-IMPORTS them (identity, not copy) so external callers and
  tests keep their import paths;
* ``CovarianceCollectionPlugin`` satisfies the universal ``PipelinePlugin``
  Protocol, carries correct metadata, is unconditionally enabled, and exposes
  the (S3-7) ``collect_covariances`` phase hook.

The byte-identical behavioral gate is the S3-0 golden snapshot
(``test_stage3_golden_snapshot.py``); this file only checks the relocation
plumbing.
"""
from __future__ import annotations

from pathlib import Path

import torch


def test_covariance_collection_module_imports():
    """The relocated symbols import from the plugin module."""
    from moe_compress.stage3.plugins.covariance_collection import (
        _collect_covariances,
        _collect_pruned_input_covariance,
        _load_stage2_covariance,
        CovarianceCollectionPlugin,
    )

    assert callable(_collect_covariances)
    assert callable(_collect_pruned_input_covariance)
    assert callable(_load_stage2_covariance)
    assert isinstance(CovarianceCollectionPlugin, type)


def test_monolith_reexports_covariance_functions():
    """The monolith re-imports the relocated functions — identity, not copy.

    ``IS`` identity proves ``stage3_svd`` holds the *same* function objects as
    the plugin module (a re-import), not independent copies that could drift.
    """
    import moe_compress.stage3_svd as monolith
    import moe_compress.stage3.plugins.covariance_collection as plugin

    assert monolith._collect_covariances is plugin._collect_covariances
    assert monolith._load_stage2_covariance is plugin._load_stage2_covariance
    assert (
        monolith._collect_pruned_input_covariance
        is plugin._collect_pruned_input_covariance
    )


def test_collect_pruned_alias_identity():
    """``_collect_pruned_input_covariance`` is an alias of ``_collect_covariances``."""
    from moe_compress.stage3.plugins.covariance_collection import (
        _collect_covariances,
        _collect_pruned_input_covariance,
    )

    assert _collect_pruned_input_covariance is _collect_covariances


def test_plugin_satisfies_protocol():
    """``CovarianceCollectionPlugin`` structurally satisfies ``PipelinePlugin``."""
    from moe_compress.pipeline.plugin import PipelinePlugin
    from moe_compress.stage3.plugins.covariance_collection import (
        CovarianceCollectionPlugin,
    )

    assert isinstance(CovarianceCollectionPlugin(), PipelinePlugin)


def test_plugin_metadata():
    """Plugin metadata — name / paper id / config_key / tuple-typed fields."""
    from moe_compress.stage3.plugins.covariance_collection import (
        CovarianceCollectionPlugin,
    )

    plugin = CovarianceCollectionPlugin()
    assert plugin.name == "covariance_collection"
    assert "2604.02119" in plugin.paper
    assert plugin.config_key == "stage3_svd.aa_svd.cross_covariance"
    assert isinstance(plugin.reads, tuple)
    assert isinstance(plugin.writes, tuple)
    assert isinstance(plugin.provides, tuple)


def test_plugin_is_enabled_unconditional():
    """Covariance collection is UNCONDITIONAL — ``is_enabled`` always True.

    B-covariance is mandatory; ``config_key`` only gates the internal
    cross-covariance branch, never the plugin as a whole.
    """
    from moe_compress.stage3.plugins.covariance_collection import (
        CovarianceCollectionPlugin,
    )

    plugin = CovarianceCollectionPlugin()
    assert plugin.is_enabled({}) is True
    assert plugin.is_enabled(
        {"stage3_svd": {"aa_svd": {"cross_covariance": False}}}
    ) is True


def test_plugin_has_collect_covariances_hook():
    """The S3-7 phase hook ``collect_covariances`` is present and callable."""
    from moe_compress.stage3.plugins.covariance_collection import (
        CovarianceCollectionPlugin,
    )

    plugin = CovarianceCollectionPlugin()
    assert callable(getattr(plugin, "collect_covariances", None))


def test_load_stage2_covariance_missing_path():
    """A missing covariance path returns an empty dict (AA-SVD fallback)."""
    from moe_compress.stage3.plugins.covariance_collection import (
        _load_stage2_covariance,
    )

    assert _load_stage2_covariance(Path("/nonexistent/x.pt")) == {}


def test_load_stage2_covariance_roundtrip(tmp_path):
    """A payload with a ``covariance`` key roundtrips; one without returns {}."""
    from moe_compress.stage3.plugins.covariance_collection import (
        _load_stage2_covariance,
    )

    p = tmp_path / "cov.pt"
    payload = {"covariance": {"k": torch.tensor([1.0])}}
    torch.save(payload, p)
    loaded = _load_stage2_covariance(p)
    assert set(loaded.keys()) == {"k"}
    assert torch.equal(loaded["k"], torch.tensor([1.0]))

    # Payload WITHOUT a "covariance" key → empty dict.
    p2 = tmp_path / "no_cov.pt"
    torch.save({"something_else": 1}, p2)
    assert _load_stage2_covariance(p2) == {}
