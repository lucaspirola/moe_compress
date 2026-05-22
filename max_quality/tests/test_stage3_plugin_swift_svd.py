"""S3-4 — Swift-SVD+ α-search plugin extraction tests.

Verifies the pure-relocation of the eight Swift-SVD+ α-selection symbols out of
the ``stage3_svd.py`` monolith into ``stage3/plugins/swift_svd_alpha.py``:

* ``_snapshot_originals`` / ``_build_wikitext2_validation`` /
  ``_evaluate_wikitext2_ppl`` / ``_factor_model_at_ranks`` /
  ``_restore_fused_experts`` / ``_swift_svd_plus_alpha_search_validation`` /
  ``_swift_svd_plus_alpha_search`` / ``_redistribute_ranks_swift_svd_plus``;
* the plugin module exposes the relocated symbols;
* the monolith RE-IMPORTS them (identity, not copy) so ``run()`` and external
  callers keep their import paths;
* ``SwiftSvdAlphaPlugin`` satisfies the universal ``PipelinePlugin`` Protocol,
  carries correct metadata, is unconditionally enabled, and exposes the (S3-7)
  ``select_alpha`` phase hook;
* the empty-tensor PPL guard and the rank-redistribution budget conservation
  still behave as in the monolith.

The byte-identical behavioral gate is the S3-0 golden snapshot
(``test_stage3_golden_snapshot.py``); this file only checks the relocation
plumbing plus a couple of pure-unit logic assertions. The model-driven paths —
``_snapshot_originals`` / ``_factor_model_at_ranks`` / ``_restore_fused_experts``
/ ``_swift_svd_plus_alpha_search_validation`` and the AA-SVD-core lazy-import
escape — need a real fused-experts model and are covered by the smoke / golden
suites rather than re-exercised here.
"""
from __future__ import annotations

import torch


_SWIFT_SVD_SYMBOLS = (
    "_snapshot_originals",
    "_build_wikitext2_validation",
    "_evaluate_wikitext2_ppl",
    "_factor_model_at_ranks",
    "_restore_fused_experts",
    "_swift_svd_plus_alpha_search_validation",
    "_swift_svd_plus_alpha_search",
    "_redistribute_ranks_swift_svd_plus",
)


def test_swift_svd_module_imports():
    """The 8 relocated functions + ``SwiftSvdAlphaPlugin`` import from the
    plugin module."""
    from moe_compress.stage3.plugins import swift_svd_alpha
    from moe_compress.stage3.plugins.swift_svd_alpha import SwiftSvdAlphaPlugin

    for name in _SWIFT_SVD_SYMBOLS:
        assert callable(getattr(swift_svd_alpha, name)), name
    assert isinstance(SwiftSvdAlphaPlugin, type)


def test_monolith_reexports_swift_svd_symbols():
    """The monolith re-imports the relocated symbols — identity, not copy.

    ``IS`` identity proves ``stage3_svd`` holds the *same* objects as the
    plugin module (a re-import), not independent copies that could drift.
    """
    import moe_compress.stage3_svd as monolith
    import moe_compress.stage3.plugins.swift_svd_alpha as plugin

    for name in _SWIFT_SVD_SYMBOLS:
        assert getattr(monolith, name) is getattr(plugin, name), name


def test_plugin_satisfies_protocol():
    """``SwiftSvdAlphaPlugin`` structurally satisfies ``PipelinePlugin``."""
    from moe_compress.pipeline.plugin import PipelinePlugin
    from moe_compress.stage3.plugins.swift_svd_alpha import SwiftSvdAlphaPlugin

    assert isinstance(SwiftSvdAlphaPlugin(), PipelinePlugin)


def test_plugin_metadata():
    """Plugin metadata — name / paper id / config_key / tuple-typed fields."""
    from moe_compress.stage3.plugins.swift_svd_alpha import SwiftSvdAlphaPlugin

    plugin = SwiftSvdAlphaPlugin()
    assert plugin.name == "swift_svd_alpha"
    assert "2604.01609" in plugin.paper
    assert plugin.config_key == "stage3_svd.swift_svd_plus.alpha_grid"
    assert isinstance(plugin.reads, tuple)
    assert isinstance(plugin.writes, tuple)
    assert isinstance(plugin.provides, tuple)


def test_plugin_is_enabled_unconditional():
    """Swift-SVD+ α selection is UNCONDITIONAL — ``is_enabled`` always True.

    An ``alpha_grid`` of length ≤ 1 yields the uniform path; it does not
    disable the phase. ``config_key`` only parametrises the grid.
    """
    from moe_compress.stage3.plugins.swift_svd_alpha import SwiftSvdAlphaPlugin

    plugin = SwiftSvdAlphaPlugin()
    assert plugin.is_enabled({}) is True
    assert plugin.is_enabled(
        {"stage3_svd": {"swift_svd_plus": {"alpha_grid": [0.0, 0.5, 1.0]}}}
    ) is True


def test_plugin_has_select_alpha_hook():
    """The S3-7 phase hook ``select_alpha`` is present and callable."""
    from moe_compress.stage3.plugins.swift_svd_alpha import SwiftSvdAlphaPlugin

    plugin = SwiftSvdAlphaPlugin()
    assert callable(getattr(plugin, "select_alpha", None))


def test_evaluate_wikitext2_ppl_empty_tensor():
    """``_evaluate_wikitext2_ppl`` returns ``inf`` for an empty validation
    tensor — the ``numel() == 0`` guard. Pure, no model is touched.
    """
    from moe_compress.stage3.plugins.swift_svd_alpha import _evaluate_wikitext2_ppl

    empty = torch.empty((0, 0), dtype=torch.long)
    # model is never reached: the numel()==0 guard returns before any forward.
    ppl = _evaluate_wikitext2_ppl(None, empty, device=None)
    assert ppl == float("inf")


class _FakeBank:
    """Minimal duck-typed expert bank for ``_redistribute_ranks_swift_svd_plus``
    — only ``.get(e)`` is touched (``A_cov=None`` skips ``.shape()``)."""

    def __init__(self, weights):
        self._w = weights  # list of [d_out, d_in] tensors

    def get(self, e):
        return self._w[e]


class _FakeRef:
    """Minimal MoE layer ref — ``_redistribute_ranks_swift_svd_plus`` only
    filters ``moe_layers`` by ``.layer_idx``."""

    def __init__(self, layer_idx):
        self.layer_idx = layer_idx


def test_redistribute_ranks_budget_conservation(monkeypatch):
    """``_redistribute_ranks_swift_svd_plus`` conserves the per-group rank
    budget: sum of per-expert ranks == base_rank × n_experts.

    ``A_cov=None`` takes the raw-``svdvals`` branch — the ``_cov_lookup``
    whitening path (and thus the AA-SVD-core lazy import) is not exercised, so
    no model is needed. ``build_banks`` is monkeypatched to a stub returning
    synthetic banks.
    """
    from moe_compress.stage3.plugins import swift_svd_alpha
    from moe_compress.stage3.plugins.swift_svd_alpha import (
        _GroupStats,
        _redistribute_ranks_swift_svd_plus,
    )

    torch.manual_seed(0)
    n_experts, d_out, d_in = 5, 32, 24
    # One (layer, matrix) group. Square-ish so the cap min(d_out,d_in)-1 = 23
    # comfortably exceeds the per-expert ranks at base_rank=8.
    layer_idx, matrix = 0, "gate_proj"

    fake_weights = [torch.randn(d_out, d_in) for _ in range(n_experts)]
    fake_banks = {matrix: _FakeBank(fake_weights)}

    def _stub_build_banks(ref):
        # ref is the single-element list's [0]; ignore it, return the stub.
        return fake_banks

    monkeypatch.setattr(swift_svd_alpha, "build_banks", _stub_build_banks)

    group_stats = {
        (layer_idx, matrix): _GroupStats(
            d_out=d_out,
            d_in=d_in,
            n_experts=n_experts,
            singular_values_mean=torch.ones(min(d_out, d_in)),
            effective_rank=float(min(d_out, d_in)) / 2.0,
            omega=n_experts * (d_out + d_in),
        ),
    }
    base_rank = 8
    base_ranks = {(layer_idx, matrix): base_rank}
    moe_layers = [_FakeRef(layer_idx)]
    alpha_by_type = {"all": 0.5}

    out = _redistribute_ranks_swift_svd_plus(
        moe_layers, group_stats, base_ranks, alpha_by_type, A_cov=None,
    )

    # One entry per (layer, matrix, expert).
    assert set(out.keys()) == {
        (layer_idx, matrix, e) for e in range(n_experts)
    }
    # Budget conservation: total per-expert rank == base_rank × n_experts.
    assert sum(out.values()) == base_rank * n_experts
    # Every expert keeps at least the δ=0.5 rank floor and stays under the cap.
    cap = min(d_out, d_in) - 1
    assert all(1 <= r <= cap for r in out.values())
