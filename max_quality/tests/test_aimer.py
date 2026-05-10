"""Tests for AIMER weight-only expert scoring."""
import numpy as np
import pytest
import torch

from moe_compress.utils.aimer import aimer_score_tensor, aimer_bottom_pct_per_layer


def test_aimer_score_uniform_weight_is_one():
    # All-equal weights → ||w||_1 = N·c, ||w||_2 = sqrt(N)·c, score = 1.0
    w = torch.ones(100)
    assert aimer_score_tensor(w) == pytest.approx(1.0)


def test_aimer_score_one_hot_is_minimum():
    # One-hot → ||w||_1 = c, ||w||_2 = c, score = 1/sqrt(N)
    w = torch.zeros(100)
    w[0] = 1.0
    assert aimer_score_tensor(w) == pytest.approx(1.0 / np.sqrt(100))


def test_aimer_bottom_pct_picks_lowest():
    scores = {(0, 0): 0.9, (0, 1): 0.1, (0, 2): 0.5, (0, 3): 0.05}
    bottom = aimer_bottom_pct_per_layer(scores, pct=0.5)
    assert bottom == {0: [3, 1]}  # 50% of 4 = 2, sorted ascending by score
