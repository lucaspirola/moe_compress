"""Forward KLD loss correctness tests (LLR-0001, LLR-0003).

These tests verify the loss is correct against a torch-native reference,
without requiring modelopt to be installed (the loss delegates to modelopt
at call time via lazy import; the tests therefore need modelopt installed
to run, gated via `pytest.importorskip`).

A separate `test_kld_parity.py` runs the bit-equality regression test against
`structural_recovery`'s implementation.
"""

from __future__ import annotations

import math

import pytest
import torch

from kdr.kd_loss import _get_kld_loss_fn, forward_kld_loss

# All loss execution requires modelopt (lazy-imported in `_get_kld_loss_fn`).
modelopt = pytest.importorskip("modelopt.torch.distill.losses")


def _torch_native_forward_kld(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Reference forward-KL implementation using only torch primitives.

    `KLD(p_teacher || p_student) = sum(p_teacher * (log p_teacher - log p_student))`,
    averaged per token after reshape to `[B*T, V]`, then multiplied by
    `temperature**2` to match modelopt's `LogitsDistillationLoss` which
    applies the same gradient-scaling correction (Hinton et al. 2015).
    """
    vocab = student_logits.shape[-1]
    s = student_logits.reshape(-1, vocab).float()
    t = teacher_logits.reshape(-1, vocab).float()
    log_q = torch.log_softmax(s / temperature, dim=-1)
    log_p = torch.log_softmax(t / temperature, dim=-1)
    p = torch.exp(log_p)
    per_row_kl = (p * (log_p - log_q)).sum(dim=-1)
    # T**2 scaling matches modelopt's LogitsDistillationLoss; without it the
    # reference diverges from kdr's implementation at any T != 1.0.
    return per_row_kl.mean() * (temperature**2)


# ─── LLR-0003: vocab mismatch raises ─────────────────────────────────────────


# VERIFIES: LLR-0003
def test_vocab_mismatch_raises_value_error() -> None:
    student = torch.zeros(2, 3, 100)
    teacher = torch.zeros(2, 3, 200)  # different V
    with pytest.raises(ValueError, match="vocab mismatch"):
        forward_kld_loss(student, teacher)


# VERIFIES: LLR-0003
def test_vocab_mismatch_error_names_both_sides() -> None:
    student = torch.zeros(1, 1, 100)
    teacher = torch.zeros(1, 1, 200)
    with pytest.raises(ValueError, match=r"V=100.*V=200"):
        forward_kld_loss(student, teacher)


# ─── LLR-0001: function shape and types ──────────────────────────────────────


# VERIFIES: LLR-0001
def test_returns_zero_dim_tensor() -> None:
    student = torch.randn(2, 4, 16)
    teacher = torch.randn(2, 4, 16)
    loss = forward_kld_loss(student, teacher)
    assert loss.ndim == 0
    assert loss.dtype == torch.float32


# VERIFIES: LLR-0001
def test_fp32_upcast_from_low_precision() -> None:
    """Low-precision (bf16 / fp16) inputs should produce an fp32 loss — the
    upcast happens before softmax. Works on whichever low-precision dtype the
    available device supports.
    """

    def _can_make_dtype_on_device(dtype: torch.dtype, device: str) -> bool:
        try:
            torch.zeros(1, dtype=dtype, device=device)
        except Exception:
            return False
        return True

    if torch.cuda.is_available() and _can_make_dtype_on_device(torch.bfloat16, "cuda"):
        student = torch.randn(2, 4, 16, dtype=torch.bfloat16, device="cuda")
        teacher = torch.randn(2, 4, 16, dtype=torch.bfloat16, device="cuda")
    elif torch.cuda.is_available() and _can_make_dtype_on_device(torch.float16, "cuda"):
        student = torch.randn(2, 4, 16, dtype=torch.float16, device="cuda")
        teacher = torch.randn(2, 4, 16, dtype=torch.float16, device="cuda")
    else:
        # CPU bf16 fallback. Modern PyTorch (>=2.0) supports CPU bf16 ops.
        student = torch.randn(2, 4, 16, dtype=torch.bfloat16)
        teacher = torch.randn(2, 4, 16, dtype=torch.bfloat16)
    loss = forward_kld_loss(student, teacher)
    assert loss.dtype == torch.float32, f"expected fp32 result, got {loss.dtype}"


# ─── LLR-0001 AC: forward-KL argument order (NOT reverse) ────────────────────


# VERIFIES: LLR-0001
def test_argument_order_is_forward_kl_not_reverse() -> None:
    """KLD(p_teacher || p_student) is asymmetric: KL(P||Q) != KL(Q||P) generally.

    Construct asymmetric distributions where forward and reverse KL differ
    measurably. The kdr loss MUST match forward-KL (teacher as target),
    NOT reverse-KL.
    """
    torch.manual_seed(42)
    # Asymmetric distributions: teacher concentrated, student spread.
    teacher = torch.tensor([[10.0, 0.0, 0.0, 0.0]])  # one-hot-ish
    student = torch.tensor([[1.0, 1.0, 1.0, 1.0]])  # uniform-ish

    forward_kl = forward_kld_loss(student, teacher)
    reference_forward = _torch_native_forward_kld(student, teacher)

    # forward_kld_loss should match the forward-KL reference, NOT the reverse.
    assert torch.allclose(forward_kl, reference_forward, atol=1e-4), (
        f"forward_kld_loss disagreed with reference forward KL: "
        f"got {forward_kl.item()}, expected {reference_forward.item()}"
    )

    # Sanity: also confirm that swapping arguments produces a measurably
    # different value (so we know the asymmetry is real for this fixture).
    swapped = forward_kld_loss(teacher, student)  # reverse direction
    assert not torch.allclose(swapped, forward_kl), (
        "swapping student and teacher produced the same loss — fixture isn't asymmetric "
        "enough to verify argument order"
    )


# ─── Reference correctness at T=1.0 ──────────────────────────────────────────


# VERIFIES: LLR-0001
def test_matches_torch_native_reference_at_t1() -> None:
    """At T=1.0, modelopt's LogitsDistillationLoss should match the formula
    `KLD(p_teacher || p_student)` computed via torch primitives within float
    tolerance.
    """
    torch.manual_seed(1337)
    student = torch.randn(4, 8, 32)
    teacher = torch.randn(4, 8, 32)
    actual = forward_kld_loss(student, teacher, temperature=1.0)
    expected = _torch_native_forward_kld(student, teacher, temperature=1.0)
    assert torch.allclose(actual, expected, atol=1e-5), (
        f"forward_kld_loss diverges from torch-native reference: "
        f"actual={actual.item()}, expected={expected.item()}, "
        f"diff={abs(actual.item() - expected.item())}"
    )


# ─── Cache helper ────────────────────────────────────────────────────────────


# VERIFIES: LLR-0002
def test_cache_returns_same_instance() -> None:
    """LLR-0002: identical temperature returns the same cached instance."""
    a = _get_kld_loss_fn(1.0)
    b = _get_kld_loss_fn(1.0)
    assert a is b


# VERIFIES: LLR-0002
def test_cache_distinguishes_temperatures() -> None:
    """LLR-0002: different temperatures produce different cached instances."""
    a = _get_kld_loss_fn(1.0)
    b = _get_kld_loss_fn(2.0)
    assert a is not b


# VERIFIES: LLR-0002
def test_cache_finite_loss_on_simple_input() -> None:
    """Smoke: cache returns a callable that produces a finite loss."""
    student = torch.randn(2, 8)
    teacher = torch.randn(2, 8)
    fn = _get_kld_loss_fn(1.0)
    out = fn(student.float(), teacher.float())
    assert torch.isfinite(out)
    assert not math.isnan(out.item())


# VERIFIES: LLR-0002
def test_cache_thread_race_distinct_keys_no_exception() -> None:
    """Two threads racing to populate distinct keys must not deadlock or raise.

    `LLR-0002` AC: `_KLD_LOSS_CACHE` access is thread-safe under DataLoader
    worker concurrency. Each thread inserts a different temperature; both
    should complete and both keys should be present afterward.
    """
    import threading

    from kdr.kd_loss import _KLD_LOSS_CACHE

    # Use temperatures that aren't used by other tests so the cache state is
    # isolated. Pick small, non-overlapping floats.
    t_a, t_b = 0.123, 0.456
    _KLD_LOSS_CACHE.pop(t_a, None)
    _KLD_LOSS_CACHE.pop(t_b, None)

    errors: list[BaseException] = []
    barrier = threading.Barrier(2)

    def race(temp: float) -> None:
        try:
            barrier.wait(timeout=5)
            for _ in range(50):
                _ = _get_kld_loss_fn(temp)
        except BaseException as e:
            errors.append(e)

    ta = threading.Thread(target=race, args=(t_a,))
    tb = threading.Thread(target=race, args=(t_b,))
    ta.start()
    tb.start()
    ta.join(timeout=10)
    tb.join(timeout=10)

    assert not ta.is_alive() and not tb.is_alive(), "thread did not finish — possible deadlock"
    assert errors == [], f"thread race raised: {errors}"
    assert t_a in _KLD_LOSS_CACHE
    assert t_b in _KLD_LOSS_CACHE


# VERIFIES: LLR-0002
def test_cache_thread_race_same_key_returns_identical_instance() -> None:
    """Two threads racing on the SAME key must end up with both seeing the
    same cached instance — no duplicate construction. This is the observable-
    behaviour gate from LLR-0002 ("equal-by-id results for matching keys").

    To make the race window non-trivial we monkey-patch
    `LogitsDistillationLoss.__init__` to sleep ~50 ms on construction. Without
    the lock, both threads' outer cache-miss → construct path would run, each
    creating its own instance, and one would overwrite the other in the dict.
    With the lock, the second thread waits at lock-acquire, then the inner
    check inside the lock hits the populated entry and returns the existing
    instance.
    """
    import threading
    import time

    from modelopt.torch.distill.losses import LogitsDistillationLoss

    from kdr.kd_loss import _KLD_LOSS_CACHE

    t = 0.789
    _KLD_LOSS_CACHE.pop(t, None)

    # Slow down construction so both threads can reach the inner-check
    # window concurrently. We restore the original __init__ in `finally`
    # to avoid polluting other tests.
    original_init = LogitsDistillationLoss.__init__

    def slow_init(self: object, *args: object, **kwargs: object) -> None:
        time.sleep(0.05)
        original_init(self, *args, **kwargs)  # type: ignore[arg-type]

    seen: list[object] = []
    barrier = threading.Barrier(2)
    record_lock = threading.Lock()

    def race() -> None:
        barrier.wait(timeout=5)
        fn = _get_kld_loss_fn(t)
        with record_lock:
            seen.append(fn)

    LogitsDistillationLoss.__init__ = slow_init  # type: ignore[method-assign]
    try:
        ta = threading.Thread(target=race)
        tb = threading.Thread(target=race)
        ta.start()
        tb.start()
        ta.join(timeout=10)
        tb.join(timeout=10)
    finally:
        LogitsDistillationLoss.__init__ = original_init  # type: ignore[method-assign]
        _KLD_LOSS_CACHE.pop(t, None)

    assert len(seen) == 2, f"expected both threads to record, got {seen}"
    assert seen[0] is seen[1], (
        "two threads got different cached instances — double-checked locking failed "
        "(removing _CACHE_LOCK in kd_loss.py would reproduce this)"
    )
