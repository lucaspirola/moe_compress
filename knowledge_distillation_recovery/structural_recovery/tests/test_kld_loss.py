"""Unit tests for forward_kld_loss + LR schedule + per-rank batch slicing.

CPU-only, no model required.
"""
from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest
import torch

from structural_recovery.distillation import (
    _shard_batches_per_rank, cosine_with_warmup, forward_kld_loss,
)


# ---------------------------------------------------------------------------
# forward_kld_loss
# ---------------------------------------------------------------------------


def test_kld_zero_at_identity():
    """KLD(p || p) ≈ 0 for identical logit tensors, regardless of T."""
    torch.manual_seed(0)
    logits = torch.randn(2, 16, 1024)
    for T in (0.5, 1.0, 2.0):
        loss = forward_kld_loss(logits, logits, temperature=T)
        assert torch.isfinite(loss)
        assert loss.item() == pytest.approx(0.0, abs=1e-4)


def test_kld_positive_for_different_distributions():
    torch.manual_seed(1)
    s = torch.randn(2, 16, 1024)
    t = torch.randn(2, 16, 1024)
    loss = forward_kld_loss(s, t)
    assert loss.item() > 0


def test_kld_temperature_scaling_keeps_finite():
    """Big and small T should both produce finite, non-negative losses."""
    torch.manual_seed(2)
    s = torch.randn(1, 4, 256) * 5.0
    t = torch.randn(1, 4, 256) * 5.0
    for T in (0.1, 0.5, 1.0, 3.0, 10.0):
        loss = forward_kld_loss(s, t, temperature=T)
        assert torch.isfinite(loss), f"non-finite at T={T}: {loss}"
        assert loss.item() >= 0.0


def test_kld_handles_bfloat16_inputs():
    """Loss must work on bf16 logits (the runtime model dtype)."""
    s = torch.randn(2, 8, 512, dtype=torch.bfloat16)
    t = torch.randn(2, 8, 512, dtype=torch.bfloat16)
    loss = forward_kld_loss(s, t)
    assert torch.isfinite(loss)
    assert loss.dtype == torch.float32


def test_kld_gradient_flows_to_student_only():
    """Backward pass should populate student grads but not teacher grads."""
    torch.manual_seed(3)
    student = torch.randn(2, 4, 64, requires_grad=True)
    teacher = torch.randn(2, 4, 64, requires_grad=False)
    loss = forward_kld_loss(student, teacher)
    loss.backward()
    assert student.grad is not None
    assert teacher.grad is None


def test_kld_per_token_normalisation():
    """Loss should be per-token mean, not per-sample sum.

    Equivalent reshape vs flat: passing [B, T, V] should give the same loss
    as passing [B*T, 1, V] (or [1, B*T, V]) because we reshape to [B*T, V]
    internally before kl_div.
    """
    torch.manual_seed(4)
    B, T, V = 3, 8, 64
    s = torch.randn(B, T, V)
    t = torch.randn(B, T, V)

    loss_3d = forward_kld_loss(s, t)
    loss_flat = forward_kld_loss(s.reshape(1, B * T, V), t.reshape(1, B * T, V))

    # Same loss regardless of how we group the tokens.
    assert loss_3d.item() == pytest.approx(loss_flat.item(), rel=1e-5)


def test_kld_per_token_is_T_times_smaller_than_batch_sum():
    """Sanity-check the bug fix: per-token loss is much smaller than the
    old (broken) per-sample-summed-over-tokens loss."""
    import torch.nn.functional as F
    torch.manual_seed(5)
    B, T, V = 2, 32, 256
    s = torch.randn(B, T, V)
    t = torch.randn(B, T, V)

    # Old (broken) behaviour: kl_div on raw [B, T, V] divides by B only.
    old_log_p = F.log_softmax(s.float(), dim=-1)
    old_p = F.softmax(t.float(), dim=-1)
    old_loss = F.kl_div(old_log_p, old_p, reduction="batchmean")

    new_loss = forward_kld_loss(s, t, temperature=1.0)
    # New loss should be ~T× smaller (token vs sample normalisation).
    ratio = old_loss.item() / new_loss.item()
    assert ratio == pytest.approx(T, rel=0.05), \
        f"expected new_loss ≈ old_loss / T={T}, got ratio={ratio}"


# ---------------------------------------------------------------------------
# cosine_with_warmup
# ---------------------------------------------------------------------------


def test_cosine_with_warmup_endpoints():
    lr_max, lr_min = 2e-4, 4.5e-7
    warmup, total = 100, 1000

    # Step 0: 1/warmup of peak.
    assert cosine_with_warmup(0, warmup_steps=warmup, total_steps=total,
                              lr_max=lr_max, lr_min=lr_min) == lr_max / warmup

    # End of warmup: at peak.
    assert cosine_with_warmup(warmup - 1, warmup_steps=warmup, total_steps=total,
                              lr_max=lr_max, lr_min=lr_min) == lr_max

    # End of run (clamped): lr_min.
    assert cosine_with_warmup(total, warmup_steps=warmup, total_steps=total,
                              lr_max=lr_max, lr_min=lr_min) == lr_min

    # Past the end: clamped to lr_min.
    assert cosine_with_warmup(total + 50, warmup_steps=warmup, total_steps=total,
                              lr_max=lr_max, lr_min=lr_min) == lr_min


def test_cosine_with_warmup_monotone_after_warmup():
    """Strictly decreasing across the cosine portion."""
    lr_max, lr_min = 1e-3, 1e-6
    warmup, total = 10, 200
    prev = math.inf
    for s in range(warmup, total + 1, 5):
        cur = cosine_with_warmup(s, warmup_steps=warmup, total_steps=total,
                                 lr_max=lr_max, lr_min=lr_min)
        assert cur < prev, f"non-monotone at step {s}: prev={prev}, cur={cur}"
        prev = cur


def test_cosine_with_warmup_last_step_near_min():
    """At step = total_steps - 1 we should be ~lr_min (within 1% of lr_max - lr_min)."""
    lr_max, lr_min = 2e-4, 4.5e-7
    warmup, total = 100, 1700
    last = cosine_with_warmup(total - 1, warmup_steps=warmup, total_steps=total,
                              lr_max=lr_max, lr_min=lr_min)
    # Cosine at progress=(total-1-warmup)/(total-warmup) ≈ 1 - tiny → cos(~pi) ≈ -1
    # → lr ≈ lr_min + tiny.
    assert last == pytest.approx(lr_min, abs=(lr_max - lr_min) * 0.001)


# ---------------------------------------------------------------------------
# _shard_batches_per_rank
# ---------------------------------------------------------------------------


def _mock_acc(process_index: int, num_processes: int):
    acc = MagicMock()
    acc.process_index = process_index
    acc.num_processes = num_processes
    return acc


def test_shard_batches_single_process_passthrough():
    batches = list(range(10))
    out = _shard_batches_per_rank(batches, _mock_acc(0, 1))
    assert out == batches


def test_shard_batches_strided_disjoint():
    """Each rank gets a disjoint stride; union covers the original list."""
    batches = list(range(20))
    world = 4
    parts = [
        _shard_batches_per_rank(batches, _mock_acc(r, world))
        for r in range(world)
    ]
    # Disjoint
    seen = set()
    for p in parts:
        for x in p:
            assert x not in seen, f"duplicate {x}"
            seen.add(x)
    # Cover all
    assert seen == set(batches)
    # Each rank has roughly 20/4 = 5 entries
    for p in parts:
        assert len(p) in (4, 5)


def test_shard_batches_with_remainder():
    """When num_batches is not divisible by world, ranks get ceil/floor counts."""
    batches = list(range(7))
    parts = [
        _shard_batches_per_rank(batches, _mock_acc(r, 3))
        for r in range(3)
    ]
    assert parts[0] == [0, 3, 6]
    assert parts[1] == [1, 4]
    assert parts[2] == [2, 5]
