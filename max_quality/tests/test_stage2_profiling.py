"""Tests for moe_compress.stage2.profiling (Task 3 of the plugin refactor).

Scope: verify ``_LayerInputAccumulator`` is importable from the new module
location and its reservoir-sampling contract is intact. ``_profile_layer``'s
behaviour is exercised end-to-end by the existing stage-2 + smoke tests
(test_stage2_merge.py, test_smoke_stage2_resume.py, ...); duplicating that
coverage here would be redundant.
"""
from __future__ import annotations

import torch

from moe_compress.stage2.profiling import _LayerInputAccumulator, _profile_layer


# ---------------------------------------------------------------------------
# _LayerInputAccumulator - reservoir-sampling contract
# ---------------------------------------------------------------------------


def test_layer_input_acc_caps_at_max_samples():
    acc = _LayerInputAccumulator(max_samples=16)
    # Feed three batches totalling 48 tokens (> cap).
    for _ in range(3):
        acc.add(torch.randn(4, 4, 8))  # (batch, seq, hidden) -> 16 tokens each
    out = acc.get()
    assert out is not None
    assert out.shape == (16, 8)


def test_layer_input_acc_get_before_any_add_is_none():
    acc = _LayerInputAccumulator(max_samples=4)
    assert acc.get() is None


def test_layer_input_acc_is_deterministic_under_seed():
    """Two independent accumulators with the same seed must produce identical buffers
    after the same input sequence - pins the seeded-generator invariant documented
    in the class docstring (F2 fix)."""
    torch.manual_seed(0)
    inputs = [torch.randn(8, 1, 4) for _ in range(5)]  # 40 tokens total, cap = 20

    a = _LayerInputAccumulator(max_samples=20, seed=123)
    b = _LayerInputAccumulator(max_samples=20, seed=123)
    for x in inputs:
        a.add(x)
        b.add(x)

    out_a, out_b = a.get(), b.get()
    assert out_a is not None and out_b is not None
    assert out_a.shape == (20, 4)
    assert torch.equal(out_a, out_b)


def test_layer_input_acc_different_seeds_diverge():
    """Different seeds must produce different reservoirs once past the cap -
    sanity check that the seed argument is actually wired into the generator."""
    torch.manual_seed(0)
    inputs = [torch.randn(8, 1, 4) for _ in range(5)]  # 40 tokens, cap = 20

    a = _LayerInputAccumulator(max_samples=20, seed=1)
    b = _LayerInputAccumulator(max_samples=20, seed=2)
    for x in inputs:
        a.add(x)
        b.add(x)

    out_a, out_b = a.get(), b.get()
    assert out_a is not None and out_b is not None
    # First 20 inputs filled the buffer identically; only reservoir replacement
    # under seed control should diverge. With 20 extra candidates and seed-driven
    # randint, expect at least one differing row in practice.
    assert not torch.equal(out_a, out_b)
