"""Unit tests for ``moe_compress.stage1.plugins.magnitude_topk`` (sub-task 8).

Verifies:

1. The plugin's class-level Protocol attributes match the registry
   contract (``PipelinePlugin``).
2. ``is_enabled`` reads
   ``config["stage1_grape"]["super_expert_detection"]["magnitude_topk_per_l_layer"]``
   and returns ``True`` iff the value > 0 (default 16).
3. ``_magnitude_topk_candidates`` is byte-equivalent to the legacy
   helper (top-K experts per l ∈ L by per_expert_max) — including the
   top_k=0 / empty-L / empty-input edge cases.
4. ``run`` adds top-K candidates with tag ``"magnitude_topk"``.
5. ``run`` with ``top_k=0`` is a no-op.
6. Missing required slots cause ``KeyError``.
7. ``contribute_artifact`` returns ``{}``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from moe_compress.stage1._framework.candidates import CandidateBag
from moe_compress.pipeline.plugin import PipelinePlugin
from moe_compress.stage1.context import Stage1Context
from moe_compress.stage1.plugins.magnitude_topk import (
    MagnitudeTopkPlugin,
    _magnitude_topk_candidates,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _layer5_per_expert_max() -> dict[tuple[int, int], float]:
    """Six experts in layer 5 plus one rejected layer-9 entry."""
    return {
        (5, 0): 1.0, (5, 1): 9.0, (5, 2): 5.0,
        (5, 3): 8.0, (5, 4): 7.0, (5, 5): 2.0,
        (9, 0): 100.0,  # layer 9 ∉ L → ignored
    }


def _config(magnitude_topk_per_l_layer: int = 16) -> dict:
    return {
        "stage1_grape": {
            "super_expert_detection": {
                "magnitude_topk_per_l_layer": magnitude_topk_per_l_layer,
            }
        }
    }


def _populated_ctx(
    per_expert_max=None,
    L=None,
    config=None,
    candidate_bag=None,
) -> Stage1Context:
    ctx = Stage1Context()
    ctx.set(
        "max_acc",
        SimpleNamespace(
            per_expert_max=per_expert_max
            if per_expert_max is not None
            else _layer5_per_expert_max()
        ),
    )
    ctx.set("L", L if L is not None else {5})
    ctx.set("config", config if config is not None else _config())
    ctx.set(
        "candidate_bag",
        candidate_bag if candidate_bag is not None else CandidateBag(),
    )
    return ctx


# ---------------------------------------------------------------------------
# Protocol-attribute contract
# ---------------------------------------------------------------------------


def test_plugin_protocol_attributes():
    p = MagnitudeTopkPlugin()
    assert p.name == "magnitude_topk"
    assert p.paper.startswith("Magnitude top-K")
    assert (
        p.config_key
        == "stage1_grape.super_expert_detection.magnitude_topk_per_l_layer"
    )
    assert p.reads == ("max_acc", "L", "candidate_bag", "config")
    assert p.writes == ("candidate_bag",)
    assert p.provides == ("downproj_max",)


def test_plugin_is_runtime_checkable_pipelineplugin():
    assert isinstance(MagnitudeTopkPlugin(), PipelinePlugin)


# ---------------------------------------------------------------------------
# ``is_enabled`` — gated by magnitude_topk_per_l_layer > 0
# ---------------------------------------------------------------------------


def test_plugin_is_enabled_default_true():
    # Default value 16 → enabled.
    assert MagnitudeTopkPlugin().is_enabled({}) is True


def test_plugin_is_enabled_zero_disables():
    cfg = {"stage1_grape": {"super_expert_detection": {"magnitude_topk_per_l_layer": 0}}}
    assert MagnitudeTopkPlugin().is_enabled(cfg) is False


def test_plugin_is_enabled_positive_enables():
    cfg = {"stage1_grape": {"super_expert_detection": {"magnitude_topk_per_l_layer": 5}}}
    assert MagnitudeTopkPlugin().is_enabled(cfg) is True


def test_plugin_is_enabled_negative_disables():
    cfg = {"stage1_grape": {"super_expert_detection": {"magnitude_topk_per_l_layer": -1}}}
    assert MagnitudeTopkPlugin().is_enabled(cfg) is False


# ---------------------------------------------------------------------------
# ``_magnitude_topk_candidates`` — byte-equivalent to legacy
# ---------------------------------------------------------------------------


def test_magnitude_topk_candidates_byte_equivalent():
    out = _magnitude_topk_candidates(_layer5_per_expert_max(), {5}, top_k=3)
    # Top-3 in layer 5 by magnitude: experts 1 (9.0), 3 (8.0), 4 (7.0).
    assert out == {(5, 1), (5, 3), (5, 4)}


def test_magnitude_topk_candidates_top_k_zero():
    assert _magnitude_topk_candidates({(5, 0): 1.0, (5, 1): 2.0}, {5}, top_k=0) == set()


def test_magnitude_topk_candidates_empty_L():
    assert _magnitude_topk_candidates({(5, 0): 1.0, (5, 1): 2.0}, set(), top_k=5) == set()


def test_magnitude_topk_candidates_empty_per_expert_max():
    assert _magnitude_topk_candidates({}, {5}, top_k=5) == set()


# ---------------------------------------------------------------------------
# ``run`` — adds top-K candidates with the "magnitude_topk" tag
# ---------------------------------------------------------------------------


def test_run_adds_topk_candidates():
    plugin = MagnitudeTopkPlugin()
    ctx = _populated_ctx(L={5}, config=_config(magnitude_topk_per_l_layer=3))

    plugin.run(ctx)

    bag: CandidateBag = ctx.get("candidate_bag")
    # by_tag returns int-keyed dict with sorted expert lists.
    assert bag.by_tag("magnitude_topk") == {5: [1, 3, 4]}
    # Layer 9 (∉ L) never enters.
    assert bag.tags_for(9, 0) == ()


def test_run_top_k_zero_is_noop():
    plugin = MagnitudeTopkPlugin()
    ctx = _populated_ctx(L={5}, config=_config(magnitude_topk_per_l_layer=0))

    plugin.run(ctx)

    assert len(ctx.get("candidate_bag")) == 0


def test_run_empty_L_is_noop():
    plugin = MagnitudeTopkPlugin()
    ctx = _populated_ctx(L=set())

    plugin.run(ctx)

    assert len(ctx.get("candidate_bag")) == 0


# ---------------------------------------------------------------------------
# Missing-slot errors — KeyError per slot
# ---------------------------------------------------------------------------


def test_run_rejects_missing_max_acc():
    plugin = MagnitudeTopkPlugin()
    ctx = Stage1Context()
    ctx.set("L", {5})
    ctx.set("candidate_bag", CandidateBag())
    ctx.set("config", _config())

    with pytest.raises(KeyError, match="max_acc"):
        plugin.run(ctx)


def test_run_rejects_missing_L():
    plugin = MagnitudeTopkPlugin()
    ctx = Stage1Context()
    ctx.set("max_acc", SimpleNamespace(per_expert_max={}))
    ctx.set("candidate_bag", CandidateBag())
    ctx.set("config", _config())

    with pytest.raises(KeyError, match="'L'"):
        plugin.run(ctx)


def test_run_rejects_missing_candidate_bag():
    plugin = MagnitudeTopkPlugin()
    ctx = Stage1Context()
    ctx.set("max_acc", SimpleNamespace(per_expert_max={}))
    ctx.set("L", {5})
    ctx.set("config", _config())

    with pytest.raises(KeyError, match="candidate_bag"):
        plugin.run(ctx)


def test_run_rejects_missing_config():
    plugin = MagnitudeTopkPlugin()
    ctx = Stage1Context()
    ctx.set("max_acc", SimpleNamespace(per_expert_max={}))
    ctx.set("L", {5})
    ctx.set("candidate_bag", CandidateBag())

    with pytest.raises(KeyError, match="config"):
        plugin.run(ctx)


# ---------------------------------------------------------------------------
# ``contribute_artifact`` — returns {}
# ---------------------------------------------------------------------------


def test_contribute_artifact_returns_empty_dict():
    plugin = MagnitudeTopkPlugin()
    ctx = _populated_ctx()
    plugin.run(ctx)
    assert plugin.contribute_artifact(ctx) == {}
