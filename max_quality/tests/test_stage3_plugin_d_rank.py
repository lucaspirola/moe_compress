"""S3-3 — D-Rank rank-allocation plugin extraction tests.

Verifies the pure-relocation of ``_GroupStats`` / ``_group_stat`` / ``_pad`` /
``_compute_T_budget`` / ``_d_rank_allocate`` out of the ``stage3_svd.py``
monolith into ``stage3/plugins/d_rank_allocate.py``:

* the plugin module exposes the relocated symbols;
* the monolith RE-IMPORTS them (identity, not copy) so ``run()`` and external
  callers keep their import paths;
* ``DRankAllocatePlugin`` satisfies the universal ``PipelinePlugin`` Protocol,
  carries correct metadata, is unconditionally enabled, and exposes the (S3-7)
  ``allocate_ranks`` phase hook;
* the relocated allocation arithmetic still produces budget-conserving ranks.

The byte-identical behavioral gate is the S3-0 golden snapshot
(``test_stage3_golden_snapshot.py``); this file only checks the relocation
plumbing plus a couple of pure-unit logic assertions.
"""
from __future__ import annotations

import dataclasses

import torch


def test_d_rank_module_imports():
    """The 5 relocated symbols import from the plugin module."""
    from moe_compress.stage3.plugins.d_rank_allocate import (
        _GroupStats,
        _group_stat,
        _pad,
        _compute_T_budget,
        _d_rank_allocate,
        DRankAllocatePlugin,
    )

    assert isinstance(_GroupStats, type)
    assert callable(_group_stat)
    assert callable(_pad)
    assert callable(_compute_T_budget)
    assert callable(_d_rank_allocate)
    assert isinstance(DRankAllocatePlugin, type)


def test_monolith_reexports_d_rank_symbols():
    """The monolith re-imports the relocated symbols — identity, not copy.

    ``IS`` identity proves ``stage3_svd`` holds the *same* objects as the
    plugin module (a re-import), not independent copies that could drift.
    """
    import moe_compress.stage3_svd as monolith
    import moe_compress.stage3.plugins.d_rank_allocate as plugin

    assert monolith._GroupStats is plugin._GroupStats
    assert monolith._group_stat is plugin._group_stat
    assert monolith._pad is plugin._pad
    assert monolith._compute_T_budget is plugin._compute_T_budget
    assert monolith._d_rank_allocate is plugin._d_rank_allocate


def test_group_stats_is_dataclass():
    """``_GroupStats`` is a dataclass with the expected field set."""
    from moe_compress.stage3.plugins.d_rank_allocate import _GroupStats

    assert dataclasses.is_dataclass(_GroupStats)
    fields = {f.name for f in dataclasses.fields(_GroupStats)}
    assert fields == {
        "d_out",
        "d_in",
        "n_experts",
        "singular_values_mean",
        "effective_rank",
        "omega",
    }


def test_plugin_satisfies_protocol():
    """``DRankAllocatePlugin`` structurally satisfies ``PipelinePlugin``."""
    from moe_compress.pipeline.plugin import PipelinePlugin
    from moe_compress.stage3.plugins.d_rank_allocate import DRankAllocatePlugin

    assert isinstance(DRankAllocatePlugin(), PipelinePlugin)


def test_plugin_metadata():
    """Plugin metadata — name / paper id / config_key / tuple-typed fields."""
    from moe_compress.stage3.plugins.d_rank_allocate import DRankAllocatePlugin

    plugin = DRankAllocatePlugin()
    assert plugin.name == "d_rank_allocate"
    assert "2509.25622" in plugin.paper
    assert plugin.config_key == "stage3_svd.d_rank.per_projection_weight"
    assert isinstance(plugin.reads, tuple)
    assert isinstance(plugin.writes, tuple)
    assert isinstance(plugin.provides, tuple)


def test_plugin_is_enabled_unconditional():
    """D-Rank allocation is UNCONDITIONAL — ``is_enabled`` always True.

    Every Stage 3 run needs a rank budget and allocation; ``config_key`` only
    biases the allocation, never gates the plugin as a whole.
    """
    from moe_compress.stage3.plugins.d_rank_allocate import DRankAllocatePlugin

    plugin = DRankAllocatePlugin()
    assert plugin.is_enabled({}) is True
    assert plugin.is_enabled(
        {"stage3_svd": {"d_rank": {"per_projection_weight": {"gate_proj": 1.75}}}}
    ) is True


def test_plugin_has_allocate_ranks_hook():
    """The S3-7 phase hook ``allocate_ranks`` is present and callable."""
    from moe_compress.stage3.plugins.d_rank_allocate import DRankAllocatePlugin

    plugin = DRankAllocatePlugin()
    assert callable(getattr(plugin, "allocate_ranks", None))


def _synthetic_group_stats():
    """A small synthetic ``group_stats`` dict of ``_GroupStats``.

    Two (layer, matrix) groups with square d_out == d_in so the per-group cap
    ``min(d_out, d_in) - 1`` is comfortably above the allocated ranks — no cap
    binds, so the budget-correction loop fully conserves the budget.
    """
    from moe_compress.stage3.plugins.d_rank_allocate import _GroupStats

    return {
        (0, "gate_proj"): _GroupStats(
            d_out=64,
            d_in=64,
            n_experts=4,
            singular_values_mean=torch.ones(64),
            effective_rank=32.0,
            omega=4 * (64 + 64),
        ),
        (0, "down_proj"): _GroupStats(
            d_out=64,
            d_in=64,
            n_experts=4,
            singular_values_mean=torch.ones(64),
            effective_rank=16.0,
            omega=4 * (64 + 64),
        ),
    }


def test_compute_T_budget_arithmetic():
    """``_compute_T_budget`` returns the expected ``int`` for a known ratio.

    Both groups: total_full = 2 · (4·64·64) = 32768; costs = [512, 512] →
    avg_cost = 512. For svd_rank_ratio = 0.5: target_params = 16384 →
    T_budget = int(16384 / 512) = 32.
    """
    from moe_compress.stage3.plugins.d_rank_allocate import _compute_T_budget

    gs = _synthetic_group_stats()
    t = _compute_T_budget(gs, 0.5)
    assert isinstance(t, int)
    assert t == 32
    assert t >= 1


def test_d_rank_allocate_budget_conservation():
    """``_d_rank_allocate`` conserves the budget when no per-group cap binds.

    With square 64×64 groups the cap is 63 per group; a budget of 32 across
    two groups never approaches the cap, so the correction loop drives the
    total to exactly ``T_budget``. Every per-group rank is >= 1.
    """
    from moe_compress.stage3.plugins.d_rank_allocate import (
        _compute_T_budget,
        _d_rank_allocate,
    )

    gs = _synthetic_group_stats()
    t_budget = _compute_T_budget(gs, 0.5)
    ranks = _d_rank_allocate(gs, t_budget)

    assert set(ranks.keys()) == set(gs.keys())
    assert sum(ranks.values()) == t_budget
    assert all(r >= 1 for r in ranks.values())


class _FakeBank:
    """Minimal duck-typed expert bank for ``_group_stat`` — ``.shape()`` +
    ``.get(e)``, the only two members ``_group_stat`` touches."""

    def __init__(self, weights):
        self._w = weights  # list of [d_out, d_in] tensors

    def shape(self):
        return tuple(self._w[0].shape)

    def get(self, e):
        return self._w[e]


def test_group_stat_raw_fallback():
    """``_group_stat`` with ``A_g=None`` takes the raw-SVD fallback path and
    returns a well-formed ``_GroupStats`` — also exercises ``_pad``."""
    from moe_compress.stage3.plugins.d_rank_allocate import _GroupStats, _group_stat

    torch.manual_seed(0)
    d_out, d_in, n_experts = 8, 6, 3
    bank = _FakeBank([torch.randn(d_out, d_in) for _ in range(n_experts)])

    gs = _group_stat(n_experts, bank, A_g=None)

    assert isinstance(gs, _GroupStats)
    assert gs.d_out == d_out and gs.d_in == d_in and gs.n_experts == n_experts
    # _pad clips/pads the per-expert singular spectra to min(d_out, d_in).
    assert gs.singular_values_mean.shape == (min(d_out, d_in),)
    assert gs.effective_rank > 0.0
    assert gs.omega == n_experts * (d_out + d_in)


def test_group_stat_whitened_path():
    """``_group_stat`` with a PSD ``A_g`` takes the Cholesky-whitened SVD path
    and returns a well-formed ``_GroupStats``."""
    from moe_compress.stage3.plugins.d_rank_allocate import _GroupStats, _group_stat

    torch.manual_seed(1)
    d_out, d_in, n_experts = 8, 6, 3
    bank = _FakeBank([torch.randn(d_out, d_in) for _ in range(n_experts)])
    # A_g must be a (d_in, d_in) symmetric positive-definite covariance.
    m = torch.randn(d_in, d_in)
    a_g = m @ m.T + d_in * torch.eye(d_in)

    gs = _group_stat(n_experts, bank, A_g=a_g)

    assert isinstance(gs, _GroupStats)
    assert gs.singular_values_mean.shape == (min(d_out, d_in),)
    assert gs.effective_rank > 0.0
    assert gs.omega == n_experts * (d_out + d_in)
