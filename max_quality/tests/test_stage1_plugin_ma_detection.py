"""Unit tests for ``moe_compress.stage1.plugins.ma_detection`` (sub-task 9).

Verifies:

1. The plugin's class-level Protocol attributes match the registry
   contract (``PipelinePlugin``).
2. ``is_enabled`` is always ``True`` (mandatory phase — no flag).
3. ``_flag_layer_dual_signal`` is byte-equivalent to the legacy helper
   (the OR truth table).
4. ``run`` populates the four Phase-A write slots from a forward pass
   over ``tiny_model``, and is a thin wrapper around ``_detect_ma_layers``.
5. The 0.75-depth fallback fires when the dynamic detector returns ∅.
6. ``contribute_artifact`` returns the canonical three-key ``dual_signal``
   block, with NaN/±Inf serialised to JSON ``null``.
7. Missing required slots cause ``KeyError`` so the orchestrator
   (sub-task 10) gets a clear contract violation.
"""

from __future__ import annotations

import json
import math

import pytest

from moe_compress.pipeline.plugin import PipelinePlugin
from moe_compress.pipeline.context import PipelineContext
from moe_compress.stage1.plugins.ma_detection import (
    MADetectionPlugin,
    _detect_ma_layers,
    _flag_layer_dual_signal,
)
from moe_compress.utils.calibration import (
    build_calibration_tensor,
    iter_batches,
    spec_from_config,
)
from moe_compress.utils.model_io import iter_moe_layers


class _TinyTokenizer:
    name_or_path = "tiny-tokenizer"
    eos_token_id = 0

    def __call__(self, text, *_, **__):
        return {"input_ids": [min(ord(c) % 32, 31) for c in (text or " ")]}

    def save_pretrained(self, *_args, **_kwargs):
        return None


def _populated_ctx(tiny_model, tiny_config, tmp_path) -> PipelineContext:
    ctx = PipelineContext()
    ctx.set("model", tiny_model)
    ctx.set("tokenizer", _TinyTokenizer())
    ctx.set("config", tiny_config)
    ctx.set("artifacts_dir", tmp_path)
    ctx.set("device", None)
    return ctx


# ---------------------------------------------------------------------------
# Protocol-attribute contract
# ---------------------------------------------------------------------------


def test_plugin_protocol_attributes():
    p = MADetectionPlugin()
    assert p.name == "ma_detection"
    assert p.paper.startswith("MA-formation")
    assert p.config_key == "stage1_grape.super_expert_detection"
    assert p.reads == ("model", "tokenizer", "config", "artifacts_dir", "device")
    assert p.writes == (
        "L",
        "residual_growth",
        "moe_output_growth",
        "moe_output_max",
    )
    assert p.provides == ()


def test_plugin_is_runtime_checkable_pipelineplugin():
    assert isinstance(MADetectionPlugin(), PipelinePlugin)


# ---------------------------------------------------------------------------
# ``is_enabled`` — always True (mandatory, no flag)
# ---------------------------------------------------------------------------


def test_plugin_is_enabled_empty_config():
    assert MADetectionPlugin().is_enabled({}) is True


def test_plugin_is_enabled_ignores_unrecognised_flag():
    cfg = {"stage1_grape": {"super_expert_detection": {"ma_enabled": False}}}
    assert MADetectionPlugin().is_enabled(cfg) is True


# ---------------------------------------------------------------------------
# ``_flag_layer_dual_signal`` — byte-equivalent to legacy (OR truth table)
# ---------------------------------------------------------------------------


def test_flag_layer_dual_signal_or_truth_table():
    # Residual-only above threshold → flag.
    assert _flag_layer_dual_signal(
        residual_ratio=4.0, moe_ratio=1.5,
        residual_threshold=3.0, moe_threshold=2.0,
    ) is True
    # MoE-only above threshold → flag.
    assert _flag_layer_dual_signal(
        residual_ratio=2.5, moe_ratio=2.5,
        residual_threshold=3.0, moe_threshold=2.0,
    ) is True
    # Neither above threshold → don't flag.
    assert _flag_layer_dual_signal(
        residual_ratio=2.5, moe_ratio=1.5,
        residual_threshold=3.0, moe_threshold=2.0,
    ) is False
    # Both above threshold → flag.
    assert _flag_layer_dual_signal(
        residual_ratio=4.0, moe_ratio=2.5,
        residual_threshold=3.0, moe_threshold=2.0,
    ) is True


# ---------------------------------------------------------------------------
# ``run`` — populates the four slots + byte-equivalence to _detect_ma_layers
# ---------------------------------------------------------------------------


def test_run_populates_four_slots(tiny_model, tiny_config, tmp_path):
    ctx = _populated_ctx(tiny_model, tiny_config, tmp_path)
    MADetectionPlugin().run(ctx)

    L = ctx.get("L")
    residual_growth = ctx.get("residual_growth")
    moe_output_growth = ctx.get("moe_output_growth")
    moe_output_max = ctx.get("moe_output_max")

    assert isinstance(L, set)
    assert all(isinstance(li, int) for li in L)
    for d in (residual_growth, moe_output_growth, moe_output_max):
        assert isinstance(d, dict)
        assert all(isinstance(k, int) for k in d)
        assert all(isinstance(v, float) for v in d.values())

    first_moe = min(ref.layer_idx for ref in iter_moe_layers(tiny_model))
    assert math.isnan(residual_growth[first_moe])
    assert moe_output_growth[first_moe] == 0.0


def test_run_byte_equivalent_to_legacy_detect(tiny_model, tiny_config, tmp_path):
    """The plugin is a thin wrapper: ``run`` must produce the same four
    outputs as a direct ``_detect_ma_layers`` call on the same batches."""
    ctx = _populated_ctx(tiny_model, tiny_config, tmp_path)
    MADetectionPlugin().run(ctx)

    s1 = tiny_config["stage1_grape"]
    spec = spec_from_config(
        tiny_config["calibration"],
        num_sequences_override=s1.get("num_calibration_samples"),
        seed_offset=1,
    )
    calib = build_calibration_tensor(
        _TinyTokenizer(), spec, cache_dir=tmp_path / "_calibration_cache",
    )
    batches = iter_batches(calib, batch_size=int(s1.get("phase_a_batch_size", 32)))
    moe_layers = list(iter_moe_layers(tiny_model))
    L, rg, mog, mom = _detect_ma_layers(tiny_model, batches, moe_layers, None)

    assert ctx.get("L") == L
    # NaN-aware comparison for residual_growth.
    assert ctx.get("residual_growth").keys() == rg.keys()
    for k in rg:
        a, b = ctx.get("residual_growth")[k], rg[k]
        assert (math.isnan(a) and math.isnan(b)) or a == b
    assert ctx.get("moe_output_growth") == mog
    assert ctx.get("moe_output_max") == mom


def test_ma_formation_fallback_when_L_empty(tiny_model, tiny_config, tmp_path):
    """Impossible thresholds force the dynamic detector to ∅; the returned
    L must equal the 0.75-depth fallback set."""
    s1 = tiny_config["stage1_grape"]
    spec = spec_from_config(
        tiny_config["calibration"],
        num_sequences_override=s1.get("num_calibration_samples"),
        seed_offset=1,
    )
    calib = build_calibration_tensor(
        _TinyTokenizer(), spec, cache_dir=tmp_path / "_calibration_cache",
    )
    batches = iter_batches(calib, batch_size=int(s1.get("phase_a_batch_size", 32)))
    moe_layers = list(iter_moe_layers(tiny_model))

    L, _, _, _ = _detect_ma_layers(
        tiny_model, batches, moe_layers, None,
        ma_ratio=1.0e30, ma_growth_ratio=1.0e30, moe_output_growth_ratio=1.0e30,
    )

    moe_indices = sorted(ref.layer_idx for ref in moe_layers)
    total_layers = tiny_model.config.num_hidden_layers
    cutoff = round(0.75 * total_layers)
    expected = {li for li in moe_indices if li < cutoff}
    assert L == expected


# ---------------------------------------------------------------------------
# ``contribute_artifact`` — the three-key dual_signal block
# ---------------------------------------------------------------------------


def test_contribute_artifact_three_keys():
    ctx = PipelineContext()
    ctx.set("residual_growth", {0: float("nan"), 1: 2.5})
    ctx.set("moe_output_growth", {0: 0.0, 1: float("inf")})
    ctx.set("moe_output_max", {0: 1.0, 1: 3.0})

    payload = MADetectionPlugin().contribute_artifact(ctx)
    assert set(payload.keys()) == {
        "residual_growth_per_layer",
        "moe_output_growth_per_layer",
        "moe_output_max_per_layer",
    }
    # Keys stringified layer indices.
    assert set(payload["residual_growth_per_layer"].keys()) == {"0", "1"}
    # NaN / Inf serialised to None.
    assert payload["residual_growth_per_layer"]["0"] is None
    assert payload["moe_output_growth_per_layer"]["1"] is None
    assert payload["residual_growth_per_layer"]["1"] == 2.5
    # JSON-serialisable.
    json.dumps(payload)


def test_contribute_artifact_first_layer_nan_becomes_null():
    """The golden-snapshot-critical case: a NaN first-layer residual_growth
    entry must serialise to JSON null."""
    ctx = PipelineContext()
    ctx.set("residual_growth", {3: float("nan"), 4: 1.2})
    ctx.set("moe_output_growth", {3: 0.0, 4: 1.1})
    ctx.set("moe_output_max", {3: 5.0, 4: 6.0})

    payload = MADetectionPlugin().contribute_artifact(ctx)
    assert payload["residual_growth_per_layer"]["3"] is None
    assert json.loads(json.dumps(payload))["residual_growth_per_layer"]["3"] is None


# ---------------------------------------------------------------------------
# Missing-slot errors — KeyError per slot
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "missing",
    ["model", "tokenizer", "config", "artifacts_dir", "device"],
)
def test_run_rejects_missing_slot(tiny_model, tiny_config, tmp_path, missing):
    ctx = _populated_ctx(tiny_model, tiny_config, tmp_path)
    new = PipelineContext()
    for k in ctx.keys():
        if k == missing:
            continue
        new.set(k, ctx.get(k))
    with pytest.raises(KeyError, match=missing):
        MADetectionPlugin().run(new)


def test_contribute_artifact_rejects_missing_residual_growth():
    ctx = PipelineContext()
    ctx.set("moe_output_growth", {0: 0.0})
    ctx.set("moe_output_max", {0: 1.0})
    with pytest.raises(KeyError, match="residual_growth"):
        MADetectionPlugin().contribute_artifact(ctx)
