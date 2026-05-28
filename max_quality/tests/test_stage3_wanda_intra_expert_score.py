"""Tests for the Stage 3 routing-weighted Wanda intra-expert score plugin.

Covers:

* Module / Protocol conformance (the plugin satisfies ``PipelinePlugin``,
  exposes the expected ``collect_wanda_scores`` hook, and is OFF by default).
* Pattern C config-validation: unknown keys raise ``ValueError``; dtype knobs
  are range-checked.
* Math correctness on a tiny fixture: ``scalar_row`` matches the upstream
  fusion_bench formula ``E[(x · g)^2]`` per channel, and the final score
  equals ``|W| · sqrt(scalar_row)`` (broadcast on the input-channel axis).
* Manifest / orchestrator ordering: the plugin lives between
  ``CovarianceCollectionPlugin`` and ``DRankAllocatePlugin`` in the
  orchestrator's registry; ``collect_wanda_scores`` is a no-op when disabled.

The fixtures imported from ``conftest.py`` (``tiny_model``, ``tiny_config``)
mirror the conventions of the other ``test_stage3_plugin_*.py`` tests in
this directory.
"""
from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from moe_compress.pipeline.context import PipelineContext
from moe_compress.pipeline.plugin import PipelinePlugin
from moe_compress.pipeline.registry import PluginRegistry
from moe_compress.stage3.plugins.wanda_intra_expert_score import (
    _ARTIFACT_FORMAT_VERSION,
    _WandaScalarRowAccumulator,
    _compute_scores,
    WandaIntraExpertScorePlugin,
)
from moe_compress.utils.model_io import MATRIX_NAMES, build_banks, iter_moe_layers


# ==========================================================================
# Module / Protocol conformance (fast, no model required)
# ==========================================================================


def test_plugin_satisfies_protocol():
    """``WandaIntraExpertScorePlugin`` structurally satisfies ``PipelinePlugin``."""
    plugin = WandaIntraExpertScorePlugin()
    assert isinstance(plugin, PipelinePlugin)


def test_plugin_metadata():
    """Metadata: name, paper cite, config_key, tuple-typed reads/writes/provides."""
    plugin = WandaIntraExpertScorePlugin()
    assert plugin.name == "wanda_intra_expert_score"
    # Citation must include both the MoE-Pruner arXiv id and fusion_bench upstream.
    assert "2410.12013" in plugin.paper
    assert "fusion_bench" in plugin.paper
    # Clean-room attribution stamp present.
    assert "clean-room" in plugin.paper.lower()
    assert plugin.config_key == "stage3.wanda_intra_expert.enabled"
    assert isinstance(plugin.reads, tuple)
    assert isinstance(plugin.writes, tuple)
    assert isinstance(plugin.provides, tuple)
    # Public output slot is in writes.
    assert "stage3.wanda_intra_expert_score" in plugin.writes
    assert "wanda_scalar_row" in plugin.provides


def test_plugin_has_collect_wanda_scores_hook():
    """The phase hook ``collect_wanda_scores`` is present and callable."""
    plugin = WandaIntraExpertScorePlugin()
    assert callable(getattr(plugin, "collect_wanda_scores", None))


def test_plugin_disabled_by_default():
    """``is_enabled`` returns False when no config requests the plugin."""
    plugin = WandaIntraExpertScorePlugin()
    # Empty config → False.
    assert plugin.is_enabled({}) is False
    # Defensive: missing intermediates → False, no KeyError.
    assert plugin.is_enabled({"stage3": {}}) is False
    assert plugin.is_enabled({"stage3": {"wanda_intra_expert": {}}}) is False
    # Explicit False → False.
    assert plugin.is_enabled(
        {"stage3": {"wanda_intra_expert": {"enabled": False}}}
    ) is False


def test_plugin_enabled_when_opt_in():
    """``is_enabled`` flips True when the operator opts in."""
    plugin = WandaIntraExpertScorePlugin()
    assert plugin.is_enabled(
        {"stage3": {"wanda_intra_expert": {"enabled": True}}}
    ) is True


# ==========================================================================
# Pattern C — config-validation-at-top-of-run
# ==========================================================================


def test_config_validation_rejects_unknown_keys():
    """Unknown config keys raise ValueError with the typo + allowed-set hint."""
    plugin = WandaIntraExpertScorePlugin()
    bad_cfg = {"enabled": True, "enabld": True}  # typo: enabld vs enabled
    with pytest.raises(ValueError, match="unknown config keys"):
        plugin._validate_config(bad_cfg)


def test_config_validation_rejects_invalid_score_dtype():
    """``score_dtype`` outside the allowed set raises ValueError."""
    plugin = WandaIntraExpertScorePlugin()
    with pytest.raises(ValueError, match="score_dtype"):
        plugin._validate_config({"enabled": True, "score_dtype": "float64"})


def test_config_validation_rejects_invalid_scalar_row_dtype():
    """``scalar_row_dtype`` outside the allowed set raises ValueError."""
    plugin = WandaIntraExpertScorePlugin()
    with pytest.raises(ValueError, match="scalar_row_dtype"):
        plugin._validate_config(
            {"enabled": True, "scalar_row_dtype": "int8"}
        )


def test_config_validation_rejects_empty_sidecar_filename():
    """An empty sidecar filename raises ValueError."""
    plugin = WandaIntraExpertScorePlugin()
    with pytest.raises(ValueError, match="sidecar_filename"):
        plugin._validate_config({"enabled": True, "sidecar_filename": ""})


def test_config_validation_defaults_applied():
    """All knobs have safe defaults when omitted; enabled defaults False."""
    plugin = WandaIntraExpertScorePlugin()
    cfg = plugin._validate_config({})
    assert cfg["enabled"] is False
    assert cfg["write_sidecar"] is True
    assert cfg["sidecar_filename"].endswith(".pt")
    assert cfg["score_dtype"] == "float32"
    assert cfg["scalar_row_dtype"] == "float32"


# ==========================================================================
# Math correctness — _WandaScalarRowAccumulator running mean
# ==========================================================================


def test_scalar_row_accumulator_single_batch_matches_formula():
    """For a single batch, scalar_row should equal mean over tokens of (x · g)²."""
    torch.manual_seed(0)
    T, d_in = 5, 4
    x = torch.randn(T, d_in)
    g = torch.randn(T).abs() + 0.1  # positive routing weights

    acc = _WandaScalarRowAccumulator()
    acc.update(layer_idx=0, expert_idx=2, matrix_name="gate_proj",
               x_rows=x, g_weights=g)
    acc.finalize_layer(0)
    sr = acc.get_scalar_row(0, 2, "gate_proj")

    # Hand-derived expectation: sum over rows of (x_c · g)^2 / T
    expected = ((x.to(torch.float32) * g.to(torch.float32).reshape(-1, 1))
                .pow(2).sum(dim=0) / T)
    assert sr is not None
    assert sr.shape == (d_in,)
    torch.testing.assert_close(sr, expected, rtol=1e-5, atol=1e-6)


def test_scalar_row_accumulator_multi_batch_running_mean():
    """Two-batch update should equal the single-batch mean over the concat."""
    torch.manual_seed(1)
    d_in = 6
    x1 = torch.randn(3, d_in)
    g1 = torch.tensor([0.4, 0.7, 0.2])
    x2 = torch.randn(5, d_in)
    g2 = torch.tensor([0.1, 0.9, 0.3, 0.6, 0.5])

    acc = _WandaScalarRowAccumulator()
    acc.update(0, 0, "gate_proj", x1, g1)
    acc.update(0, 0, "gate_proj", x2, g2)
    acc.finalize_layer(0)
    sr = acc.get_scalar_row(0, 0, "gate_proj")

    x_cat = torch.cat([x1, x2], dim=0).to(torch.float32)
    g_cat = torch.cat([g1, g2], dim=0).to(torch.float32).reshape(-1, 1)
    expected = (x_cat * g_cat).pow(2).sum(dim=0) / float(x_cat.shape[0])

    assert sr is not None
    torch.testing.assert_close(sr, expected, rtol=1e-5, atol=1e-6)


def test_scalar_row_accumulator_matches_upstream_fusion_bench_formula():
    """Cross-check: our scalar_row update must mirror fusion_bench's update.

    Upstream (``MoEPrunerHookFnForMixtralLinear``):

        scalar_row *= nsamples / (nsamples + batch_size)
        nsamples += batch_size
        scalar_row += torch.norm(inp * routing_weights, p=2, dim=1)**2 / nsamples

    where ``inp`` is the *transposed* input (``[d_in, T]``) — so dim=1 sums
    over the token axis, equivalent to our ``(x * g).pow(2).sum(dim=0)`` on
    a ``[T, d_in]`` orientation.
    """
    torch.manual_seed(2)
    T1, T2, d_in = 4, 6, 5
    x_batches = [torch.randn(T1, d_in), torch.randn(T2, d_in)]
    g_batches = [torch.rand(T1) + 0.01, torch.rand(T2) + 0.01]

    # ---- Upstream-shaped reference (transposed layout) -------------------
    scalar_row_ref = torch.zeros(d_in, dtype=torch.float32)
    nsamples = 0
    for x, g in zip(x_batches, g_batches):
        bs = int(x.shape[0])
        # inp = x.T → shape [d_in, T]; routing_weights = g.reshape(1, -1)
        inp = x.to(torch.float32).t()
        rw = g.to(torch.float32).reshape(1, -1)
        scalar_row_ref *= nsamples / (nsamples + bs)
        nsamples += bs
        scalar_row_ref += torch.norm(inp * rw, p=2, dim=1).pow(2) / nsamples

    # ---- Our accumulator -------------------------------------------------
    acc = _WandaScalarRowAccumulator()
    for x, g in zip(x_batches, g_batches):
        acc.update(0, 0, "gate_proj", x, g)
    acc.finalize_layer(0)
    sr = acc.get_scalar_row(0, 0, "gate_proj")

    torch.testing.assert_close(sr, scalar_row_ref, rtol=1e-5, atol=1e-6)


def test_scalar_row_accumulator_validates_shapes():
    """Mismatched g_weights shape or non-2D x_rows raises ValueError."""
    acc = _WandaScalarRowAccumulator()
    with pytest.raises(ValueError, match="g_weights"):
        acc.update(0, 0, "gate_proj",
                   x_rows=torch.randn(4, 3),
                   g_weights=torch.randn(5))
    with pytest.raises(ValueError, match="2D"):
        acc.update(0, 0, "gate_proj",
                   x_rows=torch.randn(2, 3, 4),
                   g_weights=torch.randn(2))


def test_scalar_row_accumulator_empty_batch_is_noop():
    """Empty x_rows leaves the accumulator untouched."""
    acc = _WandaScalarRowAccumulator()
    acc.update(0, 0, "gate_proj",
               x_rows=torch.empty(0, 5), g_weights=torch.empty(0))
    acc.finalize_layer(0)
    assert acc.get_scalar_row(0, 0, "gate_proj") is None


# ==========================================================================
# End-to-end: collect_wanda_scores hook on the tiny fixture
# ==========================================================================


def _make_score_ctx(model, batches, tmp_path: Path) -> PipelineContext:
    """Build a minimal ctx for the collect_wanda_scores hook."""
    config = {
        "stage3": {
            "wanda_intra_expert": {
                "enabled": True,
                "write_sidecar": False,
                "score_dtype": "float32",
                "scalar_row_dtype": "float32",
            }
        }
    }
    ctx = PipelineContext()
    ctx.set("model", model)
    ctx.set("moe_layers", list(iter_moe_layers(model)))
    ctx.set("batches", batches)
    ctx.set("device", None)
    ctx.set("config", config)
    ctx.set("artifacts_dir", tmp_path)
    return ctx


def test_collect_wanda_scores_publishes_score_map(tiny_model, tmp_path):
    """End-to-end: the hook populates ctx with a (layer→expert→matrix) score map."""
    torch.manual_seed(7)
    batches = [torch.randint(0, 32, (1, 4), dtype=torch.long)]
    ctx = _make_score_ctx(tiny_model, batches, tmp_path)

    plugin = WandaIntraExpertScorePlugin()
    plugin.collect_wanda_scores(ctx)

    assert ctx.has("stage3.wanda_intra_expert_score")
    score_map = ctx.get("stage3.wanda_intra_expert_score")
    assert isinstance(score_map, dict)
    # Tiny fixture has 2 MoE layers; some experts may have no routing,
    # but at least one (layer, expert, matrix) entry must be populated.
    n_entries = sum(
        len(per_expert)
        for per_layer in score_map.values()
        for per_expert in per_layer.values()
    )
    assert n_entries > 0

    # Metadata sidecar is present with format_version (Pattern B).
    md = ctx.get("stage3.wanda_intra_expert_metadata")
    assert md["format_version"] == _ARTIFACT_FORMAT_VERSION
    assert md["n_score_tensors"] == n_entries


def test_collect_wanda_scores_disabled_is_noop(tiny_model, tmp_path):
    """When ``enabled`` is False the hook leaves ctx untouched."""
    batches = [torch.randint(0, 32, (1, 4), dtype=torch.long)]
    ctx = _make_score_ctx(tiny_model, batches, tmp_path)
    cfg = ctx.get("config")
    cfg["stage3"]["wanda_intra_expert"]["enabled"] = False

    plugin = WandaIntraExpertScorePlugin()
    plugin.collect_wanda_scores(ctx)

    assert not ctx.has("stage3.wanda_intra_expert_score")
    assert not ctx.has("stage3.wanda_intra_expert_metadata")


def test_score_map_matches_W_times_sqrt_scalar_row(tiny_model, tmp_path):
    """The score tensor for any (layer, expert, matrix) equals
    ``|W[expert]| · sqrt(scalar_row)`` with scalar_row broadcast over the
    output axis. This proves the fusion_bench formula is honored end-to-end.
    """
    torch.manual_seed(11)
    batches = [torch.randint(0, 32, (1, 4), dtype=torch.long)]
    ctx = _make_score_ctx(tiny_model, batches, tmp_path)

    plugin = WandaIntraExpertScorePlugin()
    plugin.collect_wanda_scores(ctx)
    score_map = ctx.get("stage3.wanda_intra_expert_score")

    moe_layers = list(iter_moe_layers(tiny_model))
    # Pick the first populated (layer, expert, matrix) triple and cross-check.
    checked = 0
    for ref in moe_layers:
        if ref.layer_idx not in score_map:
            continue
        banks = build_banks(ref)
        for e, per_matrix in score_map[ref.layer_idx].items():
            for matrix_name, score in per_matrix.items():
                W = banks[matrix_name].get(e).detach().to(torch.float32).cpu()
                assert score.shape == W.shape, (
                    f"score shape {tuple(score.shape)} != W shape "
                    f"{tuple(W.shape)} for (l={ref.layer_idx}, e={e}, "
                    f"matrix={matrix_name})"
                )
                # Score must factor as |W| · row_scalar (per-input-channel
                # multiplier). Reconstruct row_scalar from the score where
                # |W| > 0 and verify it is constant across the output axis
                # for each input channel (within float tolerance).
                Wabs = W.abs()
                # Use a column with at least one strictly positive |W| value.
                mask = Wabs > 1e-12
                # For each input channel c, the ratio score[:, c] / |W[:, c]|
                # must be identical across all out rows where |W[r, c]| > 0.
                # Use channel 0 if available, else first column with values.
                for c in range(W.shape[1]):
                    col_mask = mask[:, c]
                    if col_mask.sum() < 2:
                        continue
                    ratios = score[col_mask, c] / Wabs[col_mask, c]
                    # All ratios in this column must be identical → constant
                    # row-scalar = sqrt(scalar_row[c]).
                    torch.testing.assert_close(
                        ratios, ratios[0].expand_as(ratios),
                        rtol=1e-4, atol=1e-5,
                    )
                checked += 1
    assert checked > 0, "Expected at least one score tensor to validate"


def test_collect_wanda_scores_writes_sidecar(tiny_model, tmp_path):
    """When ``write_sidecar=True`` an atomic .pt + .MANIFEST.json appear."""
    batches = [torch.randint(0, 32, (1, 4), dtype=torch.long)]
    ctx = _make_score_ctx(tiny_model, batches, tmp_path)
    cfg = ctx.get("config")
    cfg["stage3"]["wanda_intra_expert"]["write_sidecar"] = True
    cfg["stage3"]["wanda_intra_expert"]["sidecar_filename"] = "_wanda_test.pt"

    plugin = WandaIntraExpertScorePlugin()
    plugin.collect_wanda_scores(ctx)

    sidecar = tmp_path / "_wanda_test.pt"
    manifest = tmp_path / "_wanda_test.pt.MANIFEST.json"
    assert sidecar.exists()
    assert manifest.exists()
    payload = torch.load(sidecar, map_location="cpu", weights_only=False)
    # Pattern B: format_version at the top level.
    assert payload["format_version"] == _ARTIFACT_FORMAT_VERSION
    assert "wanda_intra_expert_score" in payload
    assert "metadata" in payload


def test_collect_wanda_scores_rejects_invalid_config(tiny_model, tmp_path):
    """The Pattern C validator fires from collect_wanda_scores too."""
    batches = [torch.randint(0, 32, (1, 4), dtype=torch.long)]
    ctx = _make_score_ctx(tiny_model, batches, tmp_path)
    cfg = ctx.get("config")
    cfg["stage3"]["wanda_intra_expert"]["bogus_key"] = 1

    plugin = WandaIntraExpertScorePlugin()
    with pytest.raises(ValueError, match="unknown config keys"):
        plugin.collect_wanda_scores(ctx)


# ==========================================================================
# Registry / orchestrator wiring
# ==========================================================================


def test_plugin_in_orchestrator_registry_in_canonical_position():
    """The Wanda plugin sits between covariance_collection and d_rank_allocate.

    The orchestrator's registry order IS the execution order (Stage 3 plugin
    registry contract). The Wanda plugin must come AFTER covariance_collection
    (it can reuse calibration signals) and BEFORE d_rank_allocate (a future
    rank allocator could consult the score).
    """
    from moe_compress.stage3.plugins.aa_svd_factor import AaSvdFactorPlugin
    from moe_compress.stage3.plugins.block_hidden_cache import Stage3BlockHiddenCacheProvider
    from moe_compress.stage3.plugins.block_refine import BlockRefinePlugin
    from moe_compress.stage3.plugins.covariance_collection import CovarianceCollectionPlugin
    from moe_compress.stage3.plugins.d_rank_allocate import DRankAllocatePlugin
    from moe_compress.stage3.plugins.input_cov_cache import Stage3InputCovCacheProvider
    from moe_compress.stage3.plugins.swift_svd_alpha import SwiftSvdAlphaPlugin

    # Mirror the orchestrator's registry construction.
    registry = PluginRegistry([
        Stage3InputCovCacheProvider(),
        Stage3BlockHiddenCacheProvider(),
        CovarianceCollectionPlugin(),
        WandaIntraExpertScorePlugin(),
        DRankAllocatePlugin(),
        SwiftSvdAlphaPlugin(),
        AaSvdFactorPlugin(),
        BlockRefinePlugin(),
    ])
    names = registry.names()
    assert "wanda_intra_expert_score" in names
    idx = names.index("wanda_intra_expert_score")
    assert names[idx - 1] == "covariance_collection"
    assert names[idx + 1] == "d_rank_allocate"


def test_plugin_dropped_from_enabled_when_disabled(tiny_config):
    """``registry.enabled(config)`` drops Wanda when not opted in (default)."""
    from moe_compress.stage3.plugins.covariance_collection import CovarianceCollectionPlugin
    from moe_compress.stage3.plugins.d_rank_allocate import DRankAllocatePlugin

    registry = PluginRegistry([
        CovarianceCollectionPlugin(),
        WandaIntraExpertScorePlugin(),
        DRankAllocatePlugin(),
    ])
    # tiny_config has no stage3.wanda_intra_expert section → default OFF.
    enabled = registry.enabled(tiny_config)
    names = [p.name for p in enabled]
    assert "wanda_intra_expert_score" not in names

    # Flip on → present.
    cfg_on = copy.deepcopy(tiny_config)
    cfg_on.setdefault("stage3", {}).setdefault(
        "wanda_intra_expert", {}
    )["enabled"] = True
    enabled_on = registry.enabled(cfg_on)
    names_on = [p.name for p in enabled_on]
    assert "wanda_intra_expert_score" in names_on


def test_orchestrator_imports_plugin():
    """The orchestrator module imports the Wanda plugin class."""
    import moe_compress.stage3.orchestrator as orch
    assert hasattr(orch, "WandaIntraExpertScorePlugin")
    assert orch.WandaIntraExpertScorePlugin is WandaIntraExpertScorePlugin


# ==========================================================================
# _compute_scores helper — direct unit test
# ==========================================================================


def test_compute_scores_uses_gate_scalar_row_for_up_proj(tiny_model):
    """``up_proj`` reuses ``gate_proj``'s scalar_row (D-gate-up-share)."""
    moe_layers = list(iter_moe_layers(tiny_model))
    # Stub: write scalar_row for gate_proj + down_proj on the first layer's
    # first expert only.
    acc = _WandaScalarRowAccumulator()
    ref = moe_layers[0]
    banks = build_banks(ref)
    d_in_gate = banks["gate_proj"].get(0).shape[1]
    d_in_down = banks["down_proj"].get(0).shape[1]
    # Inject a known scalar_row (cpu fp32) directly.
    acc._cpu[(ref.layer_idx, 0, "gate_proj")] = torch.full(
        (d_in_gate,), 4.0, dtype=torch.float32,
    )
    acc._cpu[(ref.layer_idx, 0, "down_proj")] = torch.full(
        (d_in_down,), 9.0, dtype=torch.float32,
    )

    score_map = _compute_scores(
        moe_layers, acc, score_dtype=torch.float32,
    )
    per_expert = score_map[ref.layer_idx][0]
    # All three matrix entries should be present (up reuses gate's scalar_row).
    assert set(per_expert.keys()) == {"gate_proj", "up_proj", "down_proj"}
    # Gate uses scalar_row=4 → factor sqrt(4)=2.
    W_gate = banks["gate_proj"].get(0).abs().to(torch.float32)
    torch.testing.assert_close(
        per_expert["gate_proj"], (W_gate * 2.0).cpu(),
        rtol=1e-5, atol=1e-6,
    )
    # Up should ALSO use scalar_row=4 (alias to gate) → factor 2.
    W_up = banks["up_proj"].get(0).abs().to(torch.float32)
    torch.testing.assert_close(
        per_expert["up_proj"], (W_up * 2.0).cpu(),
        rtol=1e-5, atol=1e-6,
    )
    # Down uses scalar_row=9 → factor sqrt(9)=3.
    W_down = banks["down_proj"].get(0).abs().to(torch.float32)
    torch.testing.assert_close(
        per_expert["down_proj"], (W_down * 3.0).cpu(),
        rtol=1e-5, atol=1e-6,
    )
