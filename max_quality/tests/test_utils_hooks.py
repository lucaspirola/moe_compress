"""Hook and activation-accumulator sanity checks."""
from __future__ import annotations

import pytest
import torch

from moe_compress.utils.activation_hooks import (
    DownProjMaxAccumulator,
    ExpertOutputAccumulator,
    ReapAccumulator,
    record_reap,
)


def test_down_proj_max_accumulator():
    acc = DownProjMaxAccumulator()
    acc.update(0, 3, torch.tensor([-7.0, 1.0, 2.0]))
    acc.update(0, 3, torch.tensor([0.5, 0.1]))          # smaller — should not override
    acc.update(0, 4, torch.tensor([0.2, 0.3]))
    acc.finalize()                                      # drain GPU-resident tensors to CPU
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
    acc.finalize_layer(1)                               # drain GPU sums to CPU
    # contrib = 0.5*1.0 + 0.25*2.0 = 1.0
    assert acc.sums[(1, 7)] == pytest.approx(1.0)
    assert acc.counts[(1, 7)] == 2
    assert acc.score(1, 7) == pytest.approx(0.5)


def test_expert_output_accumulator_reservoir_cap():
    """Push >256 tokens through ExpertOutputAccumulator; per-expert buffer must be capped at 256."""
    cap = 256
    acc = ExpertOutputAccumulator(max_tokens_per_expert=cap)
    d_out = 8
    layer_idx, expert_idx = 0, 0
    # Push 1024 tokens total (> 256) in batches of 64.
    n_total = 1024
    batch_size = 64
    for start in range(0, n_total, batch_size):
        x = torch.randn(batch_size, d_out)
        acc.update(layer_idx, expert_idx, x)
    acc.finalize()
    R = acc.get_representations(layer_idx, expert_idx)
    assert R is not None
    # Reservoir capped at the configured cap.
    assert R.shape[0] == cap
    assert R.shape[1] == d_out
