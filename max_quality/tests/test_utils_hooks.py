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


def test_expert_output_accumulator_fill_to_sample_boundary():
    """Reservoir must split correctly within a single batch that crosses the
    fill→sample boundary. cap=10, push 7 then 7 → first 3 of the second batch
    fill remaining slots, last 4 enter sampling regime."""
    cap = 10
    acc = ExpertOutputAccumulator(max_tokens_per_expert=cap)
    d_out = 4

    # Batch 1: 7 tokens (all fill). Tag each token with its global index in dim 0.
    x1 = torch.arange(7, dtype=torch.float32).reshape(7, 1).repeat(1, d_out)
    acc.update(0, 0, x1)

    # Batch 2: 7 tokens with global indices [7..13]. Capacity has 3 free slots
    # (cap - 7 = 3) so first 3 fill, last 4 enter reservoir-sampling regime.
    x2 = torch.arange(7, 14, dtype=torch.float32).reshape(7, 1).repeat(1, d_out)
    acc.update(0, 0, x2)

    acc.finalize()
    R = acc.get_representations(0, 0)
    assert R.shape == (cap, d_out)

    # Tokens 0..9 (all 10) MUST be in the reservoir before sampling; specifically:
    # slots [0..6] from batch 1 + slots [7..9] from x2[:3]. The sampling tail
    # (x2[3:]) is allowed to OVERWRITE any of the 10 slots probabilistically.
    # Conservative check: every retained token's tag is from the union [0..13].
    tags = R[:, 0].tolist()
    assert all(0.0 <= t <= 13.0 for t in tags), f"unexpected tag: {tags}"
    # Slots 0..6 must hold tags 0..6 (head fill is deterministic, sampling
    # only overwrites slot indices ∈ randint(0, cap), but Phase 1 head-fill
    # writes to slots [n_filled : n_filled + n_to_fill] = [0:7] and [7:10]
    # in deterministic order before the sampling tail runs).
    # The randint sampling could overwrite slots [0..9] uniformly, so we cannot
    # assert any specific slot contents — but the reservoir cap is enforced.


def test_expert_output_accumulator_aliasing_safety():
    """Mutating the source tensor after update() must not corrupt the
    reservoir. Tests that indexed assignment copies values, not views."""
    cap = 4
    acc = ExpertOutputAccumulator(max_tokens_per_expert=cap)
    d_out = 3
    x = torch.tensor([[1.0, 1.0, 1.0],
                      [2.0, 2.0, 2.0]], dtype=torch.float32)
    acc.update(0, 0, x)
    # Mutate x in place AFTER update returns.
    x[0] = -999.0
    x[1] = -999.0
    acc.finalize()
    R = acc.get_representations(0, 0)
    assert R.shape == (2, d_out)
    # Reservoir must hold the original values, not the mutated ones.
    assert torch.allclose(R[0], torch.tensor([1.0, 1.0, 1.0]))
    assert torch.allclose(R[1], torch.tensor([2.0, 2.0, 2.0]))


def test_expert_output_accumulator_multi_key_isolation():
    """Different (layer, expert) reservoirs must not bleed into each other.
    With per-key lazy allocation, this protects against an aliasing or shared
    storage bug."""
    cap = 3
    acc = ExpertOutputAccumulator(max_tokens_per_expert=cap)
    d_out = 2

    x_a = torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float32)  # 2 tokens for (0, 0)
    x_b = torch.tensor([[2.0, 2.0], [2.0, 2.0], [2.0, 2.0]], dtype=torch.float32)  # 3 tokens for (1, 5)
    x_c = torch.tensor([[3.0, 3.0]], dtype=torch.float32)  # 1 token for (0, 1) — same layer, different expert

    acc.update(0, 0, x_a)
    acc.update(1, 5, x_b)
    acc.update(0, 1, x_c)
    acc.finalize()

    R_a = acc.get_representations(0, 0)
    R_b = acc.get_representations(1, 5)
    R_c = acc.get_representations(0, 1)

    assert R_a.shape == (2, d_out) and torch.allclose(R_a, torch.full((2, d_out), 1.0))
    assert R_b.shape == (3, d_out) and torch.allclose(R_b, torch.full((3, d_out), 2.0))
    assert R_c.shape == (1, d_out) and torch.allclose(R_c, torch.full((1, d_out), 3.0))
