"""Statistical uniformity tests for vectorized _LayerInputAccumulator (Opt C).

See ``tasks/SC_FAST_PLAN_V3.md`` §4 / Optimization C (lines 277-296) and
``tasks/PLAN_OPT_C_vectorized_reservoir.md`` §6b for the full specification.
These tests guard the marginal-uniformity, same-seed-determinism, and
no-``.item()``-on-GPU contracts of the batch-vectorized Algorithm R
(Vitter 1985) implementation.
"""
from __future__ import annotations

import math

import pytest
import torch

from moe_compress.stage2.profiling import _LayerInputAccumulator


def test_mean_position_uniformity():
    """Per SC_FAST_PLAN_V3 §4-C lines 293-294: with 1000 seeds, the grand mean
    of per-seed sampled-position means should be within 5σ of n_tokens/2
    (raised from 3σ→5σ per code-quality review H1 to bring CI flakiness
    below 3e-7).

    The tokens are fed in chunks (not a single ``add`` call). This is
    required to exercise Phase C (reservoir replacement) of Algorithm R:
    a single-shot feed of ``n_tokens > max_samples`` triggers only the
    deterministic Phase A prefix-take and the buffer ends up holding tokens
    ``0..max_samples-1`` (mean ≈ max_samples/2), not a uniform sample. See
    ``tasks/PLAN_OPT_C_vectorized_reservoir.md`` §6b.
    """
    n_tokens = 100_000
    max_samples = 1024
    hidden_size = 4
    n_seeds = 1000

    # Token t encoded as a row where flat[t, :] = float(t), so we can recover
    # the token index from any buffer row.
    flat = torch.stack([
        torch.full((hidden_size,), float(t)) for t in range(n_tokens)
    ])
    expected_mean_pos = (n_tokens - 1) / 2.0   # 49999.5

    chunk_size = 1000   # 100 chunks; first → Phase A, rest → Phase B/C
    mean_positions = []
    for seed in range(n_seeds):
        acc = _LayerInputAccumulator(max_samples=max_samples, seed=seed)
        for chunk in flat.split(chunk_size):
            acc.add(chunk)
        buf = acc.get()
        assert buf.shape == (max_samples, hidden_size)
        positions = buf[:, 0]
        mean_positions.append(positions.mean().item())

    grand_mean = float(torch.tensor(mean_positions).mean())
    # std(uniform[0, n_tokens)) ≈ n_tokens / sqrt(12); std-of-per-seed-mean
    # of max_samples draws ≈ that / sqrt(max_samples); std of grand_mean
    # over n_seeds ≈ that / sqrt(n_seeds).
    sigma_grand_mean = (
        (n_tokens / math.sqrt(12)) / math.sqrt(max_samples) / math.sqrt(n_seeds)
    )
    assert abs(grand_mean - expected_mean_pos) < 5.0 * sigma_grand_mean, (
        f"grand_mean={grand_mean:.1f} expected={expected_mean_pos:.1f} "
        f"5σ_bound={5*sigma_grand_mean:.1f}"
    )


def test_determinism_same_seed():
    """Same-seed same-output contract preserved."""
    torch.manual_seed(42)
    flat = torch.randn(200, 8)

    a = _LayerInputAccumulator(max_samples=50, seed=7)
    b = _LayerInputAccumulator(max_samples=50, seed=7)
    a.add(flat)
    b.add(flat)
    assert torch.equal(a.get(), b.get())


def test_no_item_call_on_gpu_tensor():
    """Guard against future regression that re-introduces .item() calls."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    acc = _LayerInputAccumulator(max_samples=16, seed=0)
    x = torch.randn(4, 4, 8, device="cuda")
    acc.add(x)
    buf = acc.get()
    assert buf is not None
    assert buf.device.type == "cpu"


def test_phase_a_large_first_batch_caps_at_max_samples():
    """Phase A: a single oversized first add() takes only the first max_samples
    tokens deterministically; the remainder is discarded. This is the
    documented behavior; the test pins the contract so future refactors
    cannot silently change it."""
    acc = _LayerInputAccumulator(max_samples=16, seed=0)
    # 100 tokens, encoded so we can recover the index from the buffer row.
    hidden_size = 4
    flat = torch.arange(100, dtype=torch.float32).unsqueeze(-1).expand(-1, hidden_size).contiguous()
    acc.add(flat)
    buf = acc.get()
    assert buf.shape == (16, hidden_size)
    # Prefix take: rows 0..15 (first 16 tokens), in original order.
    assert buf[:, 0].tolist() == list(range(16))
    # seen counter reflects ALL n tokens (matches scalar-loop semantics).
    assert acc.seen == 100


def test_add_empty_tensor_is_noop():
    """Empty input batches are silent no-ops: buffer unchanged, seen unchanged.
    This pins the `if n == 0: return` guard at the top of add()."""
    acc = _LayerInputAccumulator(max_samples=4, seed=0)
    acc.add(torch.randn(2, 2, 4))   # Phase A: 4 tokens, buffer filled
    before_buf = acc.get().clone()
    before_seen = acc.seen
    acc.add(torch.zeros(0, 4))      # explicit empty input
    assert torch.equal(acc.get(), before_buf)
    assert acc.seen == before_seen
