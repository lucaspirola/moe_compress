"""Unit tests for ``moe_compress.stage1.plugins.three_way_and`` (sub-task 8).

Verifies:

1. The plugin's class-level Protocol attributes match the registry
   contract (``PipelinePlugin``).
2. ``is_enabled`` is always ``True`` (mandatory paper criterion — no flag).
3. ``_compute_se_thresholds`` is byte-equivalent to the legacy helper
   (P99.5 + a_max over l ∈ L) — including empty-L / empty-input cases.
4. ``_apply_paper_criterion`` is byte-equivalent to the legacy helper
   (three-way AND: a > P99.5 ∧ a > 0.1·a_max ∧ l ∈ L).
5. ``run`` writes ``p995`` / ``a_max`` / ``a_max_threshold`` to ctx and
   adds tagged candidates to the shared ``CandidateBag``.
6. ``run`` with empty ``L`` writes (0.0, 0.0, 0.0) and adds zero
   candidates.
7. Missing required slots cause ``KeyError`` so the orchestrator
   (sub-task 10) gets a clear contract violation.
8. ``contribute_artifact`` returns ``{}``.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from moe_compress.pipeline.candidates import CandidateBag
from moe_compress.pipeline.plugin import PipelinePlugin
from moe_compress.pipeline.context import PipelineContext
from moe_compress.stage1.plugins.three_way_and import (
    ThreeWayAndPlugin,
    _apply_paper_criterion,
    _compute_se_thresholds,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _outlier_per_expert_max() -> dict[tuple[int, int], float]:
    """200 small (1.0) fillers in layer 5 plus one outlier; one rejected layer 9."""
    pem: dict[tuple[int, int], float] = {(5, e): 1.0 for e in range(200)}
    pem[(5, 7)] = 100.0  # outlier — passes P99.5 + 0.1*a_max
    pem[(9, 3)] = 10000.0  # layer 9 ∉ L → must be rejected
    return pem


def _default_config(a_max_fraction: float = 0.1) -> dict:
    return {
        "stage1_grape": {
            "super_expert_detection": {
                "a_max_fraction": a_max_fraction,
            }
        }
    }


def _populated_ctx(
    per_expert_max=None,
    L=None,
    config=None,
    candidate_bag=None,
) -> PipelineContext:
    ctx = PipelineContext()
    ctx.set(
        "max_acc",
        SimpleNamespace(
            per_expert_max=per_expert_max
            if per_expert_max is not None
            else _outlier_per_expert_max()
        ),
    )
    ctx.set("L", L if L is not None else {5})
    ctx.set("config", config if config is not None else _default_config())
    ctx.set(
        "candidate_bag",
        candidate_bag if candidate_bag is not None else CandidateBag(),
    )
    return ctx


# ---------------------------------------------------------------------------
# Protocol-attribute contract
# ---------------------------------------------------------------------------


def test_plugin_protocol_attributes():
    p = ThreeWayAndPlugin()
    assert p.name == "three_way_and"
    # `paper` must cite the source paper (arXiv:2507.23279 Eq. 6 — the
    # three-way AND SE criterion) AND the golden official-code commit
    # (ZunhaiSu/Super-Experts-Profilling @
    # 573aead3127ae593ba267758b832944f8fed1485).
    assert "arXiv:2507.23279" in p.paper
    assert "Equation 6" in p.paper
    assert "573aead3127ae593ba267758b832944f8fed1485" in p.paper
    assert p.config_key == "stage1_grape.super_expert_detection"
    assert p.reads == ("max_acc", "L", "candidate_bag", "config")
    assert p.writes == ("p995", "a_max", "a_max_threshold", "candidate_bag")
    assert p.provides == ("downproj_max",)


def test_plugin_is_runtime_checkable_pipelineplugin():
    assert isinstance(ThreeWayAndPlugin(), PipelinePlugin)


# ---------------------------------------------------------------------------
# ``is_enabled`` — always True (mandatory paper criterion)
# ---------------------------------------------------------------------------


def test_plugin_is_enabled_empty_config():
    assert ThreeWayAndPlugin().is_enabled({}) is True


def test_plugin_is_enabled_ignores_unrecognised_flag():
    cfg = {"stage1_grape": {"super_expert_detection": {"three_way_enabled": False}}}
    assert ThreeWayAndPlugin().is_enabled(cfg) is True


# ---------------------------------------------------------------------------
# ``_compute_se_thresholds`` — byte-equivalent to legacy
# ---------------------------------------------------------------------------


def test_compute_se_thresholds_byte_equivalent():
    pem = _outlier_per_expert_max()
    L = {5}
    p995, a_max = _compute_se_thresholds(pem, L)

    # Independent re-build over l ∈ L only.
    A = np.array([v for (li, _e), v in pem.items() if li in L], dtype=np.float64)
    assert p995 == float(np.percentile(A, 99.5))
    assert a_max == float(A.max())
    assert a_max == 100.0


def test_compute_se_thresholds_empty_L():
    p995, a_max = _compute_se_thresholds(_outlier_per_expert_max(), set())
    assert (p995, a_max) == (0.0, 0.0)


def test_compute_se_thresholds_empty_per_expert_max():
    p995, a_max = _compute_se_thresholds({}, {5})
    assert (p995, a_max) == (0.0, 0.0)


def test_compute_se_thresholds_no_expert_in_L():
    # L references a layer with no per_expert_max entries.
    p995, a_max = _compute_se_thresholds({(5, 0): 1.0}, {9})
    assert (p995, a_max) == (0.0, 0.0)


# ---------------------------------------------------------------------------
# ``_apply_paper_criterion`` — byte-equivalent to legacy
# ---------------------------------------------------------------------------


def test_apply_paper_criterion_byte_equivalent():
    pem = _outlier_per_expert_max()
    L = {5}
    p995, a_max = _compute_se_thresholds(pem, L)
    a_max_threshold = 0.1 * a_max
    bl = _apply_paper_criterion(pem, L, p995, a_max_threshold)
    # Only the outlier in layer 5 qualifies; layer 9 (∉ L) is excluded.
    assert bl == {5: [7]}


def test_apply_paper_criterion_empty_L_returns_empty():
    assert _apply_paper_criterion(_outlier_per_expert_max(), set(), 0.0, 0.0) == {}


# ---------------------------------------------------------------------------
# ``run`` — writes three slots + adds candidates
# ---------------------------------------------------------------------------


def test_run_populates_slots_and_adds_candidates():
    plugin = ThreeWayAndPlugin()
    pem = _outlier_per_expert_max()
    ctx = _populated_ctx(per_expert_max=pem, L={5})

    plugin.run(ctx)

    expected_p995, expected_a_max = _compute_se_thresholds(pem, {5})
    assert ctx.get("p995") == expected_p995
    assert ctx.get("a_max") == expected_a_max
    assert ctx.get("a_max_threshold") == 0.1 * expected_a_max

    bag: CandidateBag = ctx.get("candidate_bag")
    assert bag.tags_for(5, 7) == ("phase_c",)
    # Layer 9 expert 3 (∉ L) must not be flagged.
    assert bag.tags_for(9, 3) == ()
    assert len(bag) == 1


def test_run_respects_a_max_fraction_config():
    plugin = ThreeWayAndPlugin()
    pem = _outlier_per_expert_max()
    ctx = _populated_ctx(
        per_expert_max=pem, L={5}, config=_default_config(a_max_fraction=0.25)
    )

    plugin.run(ctx)

    _, expected_a_max = _compute_se_thresholds(pem, {5})
    assert ctx.get("a_max_threshold") == 0.25 * expected_a_max


def test_run_empty_L_writes_zero_stats_and_no_candidates():
    plugin = ThreeWayAndPlugin()
    ctx = _populated_ctx(L=set())

    plugin.run(ctx)

    assert ctx.get("p995") == 0.0
    assert ctx.get("a_max") == 0.0
    assert ctx.get("a_max_threshold") == 0.0
    assert len(ctx.get("candidate_bag")) == 0


# ---------------------------------------------------------------------------
# Missing-slot errors — KeyError per slot
# ---------------------------------------------------------------------------


def test_run_rejects_missing_max_acc():
    plugin = ThreeWayAndPlugin()
    ctx = PipelineContext()
    ctx.set("L", {5})
    ctx.set("candidate_bag", CandidateBag())
    ctx.set("config", _default_config())

    with pytest.raises(KeyError, match="max_acc"):
        plugin.run(ctx)


def test_run_rejects_missing_L():
    plugin = ThreeWayAndPlugin()
    ctx = PipelineContext()
    ctx.set("max_acc", SimpleNamespace(per_expert_max={}))
    ctx.set("candidate_bag", CandidateBag())
    ctx.set("config", _default_config())

    with pytest.raises(KeyError, match="'L'"):
        plugin.run(ctx)


def test_run_rejects_missing_candidate_bag():
    plugin = ThreeWayAndPlugin()
    ctx = PipelineContext()
    ctx.set("max_acc", SimpleNamespace(per_expert_max={}))
    ctx.set("L", {5})
    ctx.set("config", _default_config())

    with pytest.raises(KeyError, match="candidate_bag"):
        plugin.run(ctx)


def test_run_rejects_missing_config():
    plugin = ThreeWayAndPlugin()
    ctx = PipelineContext()
    ctx.set("max_acc", SimpleNamespace(per_expert_max={}))
    ctx.set("L", {5})
    ctx.set("candidate_bag", CandidateBag())

    with pytest.raises(KeyError, match="config"):
        plugin.run(ctx)


# ---------------------------------------------------------------------------
# ``contribute_artifact`` — returns {}
# ---------------------------------------------------------------------------


def test_contribute_artifact_returns_empty_dict():
    plugin = ThreeWayAndPlugin()
    ctx = _populated_ctx()
    plugin.run(ctx)
    assert plugin.contribute_artifact(ctx) == {}
