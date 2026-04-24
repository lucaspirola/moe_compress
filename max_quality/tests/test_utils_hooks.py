"""Hook and activation-accumulator sanity checks."""
from __future__ import annotations

import pytest
import torch

from moe_compress.utils.activation_hooks import (
    DownProjMaxAccumulator,
    ReapAccumulator,
    record_reap,
)


def test_down_proj_max_accumulator():
    acc = DownProjMaxAccumulator()
    acc.update(0, 3, torch.tensor([-7.0, 1.0, 2.0]))
    acc.update(0, 3, torch.tensor([0.5, 0.1]))          # smaller — should not override
    acc.update(0, 4, torch.tensor([0.2, 0.3]))
    assert acc.per_expert_max[(0, 3)] == pytest.approx(7.0)
    assert acc.per_expert_max[(0, 4)] == pytest.approx(0.3)


def test_reap_scoring_shape_mismatch_raises():
    acc = ReapAccumulator()
    gate = torch.ones(4)
    # 3 outputs but 4 gate values — assert should trip.
    outs = torch.randn(3, 16)
    with pytest.raises(RuntimeError, match="gate_vals.numel"):
        record_reap(acc, 0, 0, gate, outs)


def test_reap_scoring_accumulates_contribution():
    acc = ReapAccumulator()
    gate = torch.tensor([0.5, 0.25])
    outs = torch.tensor([[1.0, 0.0], [0.0, 2.0]])      # norms = [1.0, 2.0]
    record_reap(acc, 1, 7, gate, outs)
    # contrib = 0.5*1.0 + 0.25*2.0 = 1.0
    assert acc.sums[(1, 7)] == pytest.approx(1.0)
    assert acc.counts[(1, 7)] == 2
    assert acc.score(1, 7) == pytest.approx(0.5)
