"""Unit tests for the ReapScoringPlugin and its pure helper.

Covers the pure ``select_centroids_by_reap`` function (5 cases) plus a
plugin-lifecycle smoke test that exercises ``on_layer_setup`` + ``on_score``
end-to-end against a tiny ``ReapAccumulator``.
"""
from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from moe_compress.pipeline.context import PipelineContext
from moe_compress.stage2.plugins.reap_scoring import (
    ReapScoringPlugin,
    select_centroids_by_reap,
)
from moe_compress.utils.activation_hooks import ReapAccumulator


# ---------------------------------------------------------------------------
# Pure-function tests
# ---------------------------------------------------------------------------


def test_top_k_by_score_descending():
    """select_centroids_by_reap picks the top-k experts in descending-score order."""
    scores = np.array([0.1, 0.9, 0.3, 0.7])
    freq = {0: 100, 1: 100, 2: 100, 3: 100}
    selected = select_centroids_by_reap(
        scores, freq,
        ream_target=2,
        min_active_tokens=1,
        protected=(),
        layer_idx=0,
        log=logging.getLogger(__name__),
    )
    assert selected == [1, 3]


def test_protected_experts_excluded():
    """Protected experts are never returned even if they have the top score."""
    scores = np.array([0.1, 0.9, 0.3, 0.7])
    freq = {0: 100, 1: 100, 2: 100, 3: 100}
    selected = select_centroids_by_reap(
        scores, freq,
        ream_target=2,
        min_active_tokens=1,
        protected={1},
        layer_idx=0,
        log=logging.getLogger(__name__),
    )
    assert selected == [3, 2]


def test_min_active_tokens_filters_low_freq():
    """Low-frequency experts are filtered from centroid candidacy."""
    scores = np.array([0.1, 0.9, 0.3, 0.7])
    freq = {0: 100, 1: 5, 2: 100, 3: 100}
    selected = select_centroids_by_reap(
        scores, freq,
        ream_target=2,
        min_active_tokens=10,
        protected=(),
        layer_idx=0,
        log=logging.getLogger(__name__),
    )
    assert selected == [3, 2]


def test_ream_target_zero_returns_empty(caplog):
    """ream_target=0 short-circuits to []; no warning is emitted."""
    scores = np.array([0.1, 0.9, 0.3, 0.7])
    freq = {0: 100, 1: 100, 2: 100, 3: 100}
    test_log = logging.getLogger("test.reap_scoring.ream_target_zero")
    test_log.propagate = True
    with caplog.at_level(logging.WARNING):
        selected = select_centroids_by_reap(
            scores, freq,
            ream_target=0,
            min_active_tokens=1,
            protected=(),
            layer_idx=0,
            log=test_log,
        )
    assert selected == []
    assert not [
        rec for rec in caplog.records
        if "REAM centroid selection" in rec.getMessage()
    ]


def test_all_protected_returns_empty_and_warns(caplog):
    """When everything is protected the result is empty and one warning is emitted."""
    scores = np.array([0.1, 0.9, 0.3, 0.7])
    freq = {0: 100, 1: 100, 2: 100, 3: 100}
    test_log = logging.getLogger("test.reap_scoring.all_protected")
    # Some pytest plugins in this env default new loggers to propagate=False;
    # the real stage2 logger has propagate=True, so mirror that here.
    test_log.propagate = True
    with caplog.at_level(logging.WARNING):
        selected = select_centroids_by_reap(
            scores, freq,
            ream_target=2,
            min_active_tokens=1,
            protected={0, 1, 2, 3},
            layer_idx=7,
            log=test_log,
        )
    assert selected == []
    warnings = [
        rec.getMessage() for rec in caplog.records
        if "REAM centroid selection yielded 0 < 2" in rec.getMessage()
    ]
    assert len(warnings) == 1


# ---------------------------------------------------------------------------
# Plugin-lifecycle test
# ---------------------------------------------------------------------------


def test_reap_scoring_plugin_lifecycle():
    """on_layer_setup creates a fresh accumulator; on_score finalizes + publishes."""
    plugin = ReapScoringPlugin()
    layer_ref = SimpleNamespace(layer_idx=3, num_routed_experts=4)
    ctx = PipelineContext().child()
    ctx.set("layer_idx", 3)
    ctx.set("layer_ref", layer_ref)
    ctx.set("n_experts", 4)
    ctx.set("target", 2)

    plugin.on_layer_setup(ctx)
    assert isinstance(ctx.get("reap_acc"), ReapAccumulator)

    # Inject contributions for experts 1 and 2; experts 0 and 3 stay absent.
    # Using a CPU tensor + 0-dim scalar mirrors what ReapAccumulator.add_gpu()
    # receives in production (per-expert REAP saliency sum).
    n_tokens = 17
    ctx.get("reap_acc").add_gpu((3, 1), torch.tensor(2.0), n_tokens)
    ctx.get("reap_acc").add_gpu((3, 2), torch.tensor(4.0), n_tokens)

    plugin.on_score(ctx)

    scores = ctx.get("scores")
    assert isinstance(scores, np.ndarray)
    assert scores.shape == (4,)
    assert scores[0] == 0.0
    assert scores[1] > 0.0
    assert scores[2] > 0.0
    assert scores[3] == 0.0
    assert ctx.get("freq") == {0: 0, 1: n_tokens, 2: n_tokens, 3: 0}
