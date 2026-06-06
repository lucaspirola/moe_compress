"""Unit — Stage 2 post-merge survivor==target guard (probe §6).

``_assert_survivor_count`` mirrors upstream REAM ``merger.py:463``
(``assert moe_layer.num_experts == self.merge_size``). It is OPT-IN at the
call site (``stage2_reap_ream.assert_survivors_match_target``, default False)
so default runs are byte-identical; the probe enables it to abort on a REAM
bump-loop overshoot or a faithful keep-count divergence.
"""
from __future__ import annotations

import pytest

from moe_compress.stage2.orchestrator import _assert_survivor_count


def test_passes_when_survivors_equal_target():
    # Exact match → no raise (the by-the-book probe posture: kept==166).
    _assert_survivor_count(166, 166, layer_idx=0, faithful=False)
    _assert_survivor_count(166, 166, layer_idx=12, faithful=True)


def test_raises_on_overshoot_merge_path():
    # REAM bump-loop overshoot: realised > target.
    with pytest.raises(RuntimeError, match="survivor count 168 != target 166"):
        _assert_survivor_count(168, 166, layer_idx=3, faithful=False)


def test_raises_on_undershoot():
    with pytest.raises(RuntimeError, match="survivor count 165 != target 166"):
        _assert_survivor_count(165, 166, layer_idx=7, faithful=False)


def test_error_names_prune_mode():
    with pytest.raises(RuntimeError, match="prune_mode=faithful_prune"):
        _assert_survivor_count(167, 166, layer_idx=1, faithful=True)
    with pytest.raises(RuntimeError, match="prune_mode=merge"):
        _assert_survivor_count(167, 166, layer_idx=1, faithful=False)


def test_flag_default_false_keeps_guard_off():
    """The orchestrator reads the gate as ``s2.get(..., False)`` — absent key
    means the guard never runs (default runs byte-identical)."""
    s2 = {}
    assert bool(s2.get("assert_survivors_match_target", False)) is False
    s2 = {"assert_survivors_match_target": True}
    assert bool(s2.get("assert_survivors_match_target", False)) is True
