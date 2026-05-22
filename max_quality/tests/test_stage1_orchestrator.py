"""Unit + integration tests for ``moe_compress.stage1.orchestrator`` (sub-task 10).

The golden snapshot test (``test_stage1_golden_snapshot.py``) is the
*primary* gate — it pins byte-identical artifacts through the new
orchestrator path. This file adds focused orchestrator unit tests:

1. ``run`` accepts the 6-arg signature with ``device=None`` and returns a
   2-tuple of ``Path``.
2. ``run`` produces all three Stage 1 JSON artifacts.
3. ``stage1_blacklist.json`` has exactly the 7-key schema.
4. ``STAGE1_PLUGIN_MANIFEST`` is in the canonical execution order.
5. ``required_accumulators`` returns the byte-identity-critical order
   ``(downproj_max, sink_routing, output_reservoir)``.
6-9. The ``_build_accumulator`` factory for each accumulator name + the
   unknown-name guard.
10. A disabled ``sink_token`` omits ``sink_routing`` and still yields a
   valid 7-key blacklist.
11. ``setup()`` runs before the calibration engine pass.
12. A full ``orchestrator.run()`` drives the ``_detect_ma_layers``
   monkeypatch path through ``MADetectionPlugin``.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from moe_compress.budget.solver import BudgetDecomposition
from moe_compress.stage1._framework.artifact_assembly import REQUIRED_BLACKLIST_TOP_LEVEL_KEYS
from moe_compress.stage1._framework.calibration_engine import HookKind, HookSpec
from moe_compress.stage1._framework.plugin import PluginRegistry
from moe_compress.stage1 import orchestrator
from moe_compress.stage1.context import Stage1Context
from moe_compress.stage1.plugins import STAGE1_PLUGIN_MANIFEST
from moe_compress.utils.activation_hooks import (
    DownProjMaxAccumulator,
    ExpertOutputAccumulator,
)
from moe_compress.utils.model_io import iter_moe_layers


class _TinyTokenizer:
    """The tiny-model fixture tokenizer."""

    name_or_path = "tiny-tokenizer"
    eos_token_id = 0

    def __call__(self, text, *_, **__):
        return {"input_ids": [min(ord(c) % 32, 31) for c in (text or " ")]}

    def save_pretrained(self, *_args, **_kwargs):
        return None


def _decomp() -> BudgetDecomposition:
    return BudgetDecomposition(
        total_reduction_ratio=0.20,
        expert_prune_ratio=0.25,
        svd_rank_ratio=0.0,
        global_expert_budget=5,
        min_experts_per_layer=2,
        blacklisted_experts={},
    )


# ---------------------------------------------------------------------------
# 1. Signature + return contract
# ---------------------------------------------------------------------------


def test_orchestrator_run_signature(tiny_model, tiny_config, tmp_path):
    """``orchestrator.run`` accepts the 6-arg signature with ``device=None``
    and returns a 2-tuple of ``Path``."""
    result = orchestrator.run(
        tiny_model, _TinyTokenizer(), tiny_config, tmp_path, _decomp(),
        device=None,
    )
    assert isinstance(result, tuple) and len(result) == 2
    blacklist_path, budgets_path = result
    assert isinstance(blacklist_path, Path)
    assert isinstance(budgets_path, Path)


# ---------------------------------------------------------------------------
# 2. Produces three artifacts
# ---------------------------------------------------------------------------


def test_orchestrator_produces_three_artifacts(tiny_model, tiny_config, tmp_path):
    orchestrator.run(
        tiny_model, _TinyTokenizer(), tiny_config, tmp_path, _decomp(),
    )
    for name in ("stage1_blacklist.json", "stage1_budgets.json",
                 "stage1_ablation_filter.json"):
        assert (tmp_path / name).exists(), f"orchestrator did not produce {name}"


# ---------------------------------------------------------------------------
# 3. Seven-top-level-keys schema
# ---------------------------------------------------------------------------


def test_blacklist_seven_top_level_keys(tiny_model, tiny_config, tmp_path):
    blacklist_path, _ = orchestrator.run(
        tiny_model, _TinyTokenizer(), tiny_config, tmp_path, _decomp(),
    )
    payload = json.loads(blacklist_path.read_text())
    assert set(payload.keys()) == set(REQUIRED_BLACKLIST_TOP_LEVEL_KEYS)


# ---------------------------------------------------------------------------
# 4. Manifest order
# ---------------------------------------------------------------------------


def test_plugin_manifest_order():
    names = tuple(p.name for p in STAGE1_PLUGIN_MANIFEST)
    assert names == (
        "ma_detection", "three_way_and", "aimer", "sink_token",
        "magnitude_topk", "ablation_filter", "cka_distance", "grape_merge",
    )


# ---------------------------------------------------------------------------
# 5. required_accumulators order (byte-identity hazard)
# ---------------------------------------------------------------------------


def test_required_accumulators_order(tiny_config):
    registry = PluginRegistry(STAGE1_PLUGIN_MANIFEST)
    assert registry.required_accumulators(tiny_config) == (
        "downproj_max", "sink_routing", "output_reservoir",
    )


# ---------------------------------------------------------------------------
# 6-9. Accumulator factory
# ---------------------------------------------------------------------------


def test_accumulator_factory_downproj_max():
    ctx = Stage1Context()
    acc, spec = orchestrator._build_accumulator(
        "downproj_max", n_per_layer=4, moe_layers=[], tokenizer=None, ctx=ctx,
    )
    assert isinstance(acc, DownProjMaxAccumulator)
    assert isinstance(spec, HookSpec)
    assert spec.kinds == frozenset({HookKind.DOWN_PROJ})
    assert callable(spec.expert_callback)


def test_accumulator_factory_output_reservoir():
    ctx = Stage1Context()
    acc, spec = orchestrator._build_accumulator(
        "output_reservoir", n_per_layer=4, moe_layers=[], tokenizer=None, ctx=ctx,
    )
    assert isinstance(acc, ExpertOutputAccumulator)
    assert acc.max_tokens_per_expert == 256
    assert spec.kinds == frozenset({HookKind.DOWN_PROJ})
    assert callable(spec.expert_callback)


def test_accumulator_factory_sink_routing():
    """The factory READS ``ctx['sink_acc']`` (built by ``setup()``) and
    returns that exact instance + a router/input-ids HookSpec."""
    sentinel = object()
    ctx = Stage1Context()
    ctx.set("sink_acc", sentinel)
    acc, spec = orchestrator._build_accumulator(
        "sink_routing", n_per_layer=4, moe_layers=[], tokenizer=None, ctx=ctx,
    )
    assert acc is sentinel
    assert spec.kinds == frozenset({
        HookKind.ROUTER_LOGITS_PER_BATCH, HookKind.INPUT_IDS_PER_BATCH,
    })
    assert callable(spec.per_batch)


def test_accumulator_factory_unknown_name_raises():
    ctx = Stage1Context()
    with pytest.raises(ValueError):
        orchestrator._build_accumulator(
            "not_a_real_accumulator", n_per_layer=4, moe_layers=[],
            tokenizer=None, ctx=ctx,
        )


# ---------------------------------------------------------------------------
# 10. Disabled sink_token omits sink_routing, still yields a valid artifact
# ---------------------------------------------------------------------------


def test_sink_disabled_omits_sink_routing(tiny_model, tiny_config, tmp_path):
    cfg = copy.deepcopy(tiny_config)
    cfg["stage1_grape"]["super_expert_detection"]["sink_token_enabled"] = False

    registry = PluginRegistry(STAGE1_PLUGIN_MANIFEST)
    assert "sink_routing" not in registry.required_accumulators(cfg)

    blacklist_path, _ = orchestrator.run(
        tiny_model, _TinyTokenizer(), cfg, tmp_path, _decomp(),
    )
    payload = json.loads(blacklist_path.read_text())
    assert set(payload.keys()) == set(REQUIRED_BLACKLIST_TOP_LEVEL_KEYS)
    # The sink_token fragment is present but its score dicts are empty.
    assert payload["sink_token"]["mean_router_score_sink"] == {}
    assert payload["sink_token"]["candidates"] == {}


# ---------------------------------------------------------------------------
# 11. setup() runs before the calibration engine pass
# ---------------------------------------------------------------------------


def test_setup_called_before_calibration(tiny_model, tiny_config, tmp_path, monkeypatch):
    """``SinkTokenDetectorPlugin.setup`` must run before
    ``CalibrationEngine.run`` — the factory reads the accumulator setup()
    builds, so a swapped order would break instance identity."""
    from moe_compress.stage1._framework import calibration_engine
    from moe_compress.stage1.plugins.sink_token import SinkTokenDetectorPlugin

    order: list[str] = []
    real_setup = SinkTokenDetectorPlugin.setup
    real_run = calibration_engine.CalibrationEngine.run

    def _spy_setup(self, ctx):
        order.append("setup")
        return real_setup(self, ctx)

    def _spy_run(self, *args, **kwargs):
        order.append("engine_run")
        return real_run(self, *args, **kwargs)

    monkeypatch.setattr(SinkTokenDetectorPlugin, "setup", _spy_setup)
    monkeypatch.setattr(
        calibration_engine.CalibrationEngine, "run", _spy_run)

    orchestrator.run(
        tiny_model, _TinyTokenizer(), tiny_config, tmp_path, _decomp(),
    )
    assert order == ["setup", "engine_run"], (
        f"setup() must precede the calibration pass; got {order}"
    )


# ---------------------------------------------------------------------------
# 12. The orchestrator drives the _detect_ma_layers monkeypatch path
# ---------------------------------------------------------------------------


def test_orchestrator_ma_detection_monkeypatch(
    tiny_model, tiny_config, tmp_path, monkeypatch,
):
    """A full ``orchestrator.run()`` routes Phase A through
    ``MADetectionPlugin``, which calls
    ``moe_compress.stage1.plugins.ma_detection._detect_ma_layers``.

    Patch that symbol with a spy and assert it fired — this proves the
    orchestrator wires Phase A to the real detector. The spy delegates to
    the real detector so it returns a valid
    ``(L, residual_growth, moe_output_growth, moe_output_max)`` tuple
    (mirrors ``test_stage1_plugin_ma_detection.py::test_ma_formation_fallback_when_L_empty``).
    """
    from moe_compress.stage1.plugins import ma_detection

    captured: dict[str, bool] = {}
    real_detect = ma_detection._detect_ma_layers

    def _spy(model, batches, moe_layers, device, **kwargs):
        captured["called"] = True
        return real_detect(model, batches, moe_layers, device, **kwargs)

    monkeypatch.setattr(
        "moe_compress.stage1.plugins.ma_detection._detect_ma_layers", _spy)

    orchestrator.run(
        tiny_model, _TinyTokenizer(), tiny_config, tmp_path, _decomp(),
    )

    assert captured.get("called") is True, (
        "orchestrator.run did not invoke _detect_ma_layers via MADetectionPlugin"
    )
