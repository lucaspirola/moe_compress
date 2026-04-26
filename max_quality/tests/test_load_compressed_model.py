"""Regression tests for ``load_compressed_model`` and ``_assign_storage``.

Each named test below corresponds to a bug we burned real GPU hours on
this session. The goal is "if any one of these starts failing, we've
regressed a bug we already paid to fix".

CPU-only tests cover ``_assign_storage`` correctness (shape, dtype, grad,
buffer-kind handling). The CUDA-only tests reproduce the actual memory
behavior of the streaming load, catching the second-pass ``state_dict``
pinning bug deterministically.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from moe_compress.utils.model_io import _assign_storage


# ---------------------------------------------------------------------------
# _assign_storage unit tests (CPU; fast)
# ---------------------------------------------------------------------------


def test_assign_storage_param_swap_rebinds_data_ptr():
    """``param.data = tensor`` must actually swap the underlying storage.
    Catches a regression where the rebind silently no-ops or copies."""
    m = nn.Linear(4, 4, bias=False)
    old_ptr = m.weight.data.data_ptr()
    new_t = torch.full_like(m.weight.data, 7.0)
    _assign_storage(m, "weight", new_t)
    assert m.weight.data.data_ptr() == new_t.data_ptr(), \
        "weight.data was not rebound to the new tensor's storage"
    assert m.weight.data.data_ptr() != old_ptr, \
        "weight.data still points at the original skeleton storage"
    assert torch.allclose(m.weight.data, new_t)


def test_assign_storage_persistent_buffer_round_trip():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("b", torch.zeros(3))
    m = M()
    new_t = torch.ones(3) * 5.0
    _assign_storage(m, "b", new_t)
    assert torch.allclose(m.b, new_t)
    # Persistence flag preserved → still in state_dict.
    assert "b" in m.state_dict()


def test_assign_storage_non_persistent_buffer_round_trip():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("b", torch.zeros(3), persistent=False)
    m = M()
    new_t = torch.ones(3) * 5.0
    _assign_storage(m, "b", new_t)
    assert torch.allclose(m.b, new_t)
    # Non-persistence preserved → NOT in state_dict.
    assert "b" not in m.state_dict()


def test_assign_storage_shape_mismatch_raises():
    """Shape mismatch must raise with a clear message — pre-fix this would
    raise with a cryptic torch internal error 50 layers later."""
    m = nn.Linear(4, 4, bias=False)
    bad = torch.zeros(4, 5)
    with pytest.raises(RuntimeError, match="shape mismatch"):
        _assign_storage(m, "weight", bad)


def test_assign_storage_dtype_mismatch_raises():
    """Dtype mismatch must raise — pre-fix this would silently change the
    param's dtype and corrupt forward many layers downstream."""
    m = nn.Linear(4, 4, bias=False).to(torch.bfloat16)
    bad = torch.zeros(4, 4, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="dtype mismatch"):
        _assign_storage(m, "weight", bad)


def test_assign_storage_clears_stale_grad():
    """``.data =`` rebind must clear ``._grad`` — a stale grad at the old
    shape/state would otherwise corrupt any subsequent backward pass."""
    m = nn.Linear(4, 4, bias=False)
    m.weight.grad = torch.zeros_like(m.weight)
    new_t = torch.ones_like(m.weight) * 2.0
    _assign_storage(m, "weight", new_t)
    assert m.weight.grad is None, \
        "stale ._grad not cleared after .data rebind"


def test_assign_storage_unknown_kind_raises():
    """Reaching the else branch (raw attribute, neither Parameter nor
    persistent buffer) must fail loud — silently writing a raw Tensor
    would produce a model that silently misbehaves at forward."""
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            # Plain attribute, NOT register_buffer / nn.Parameter.
            self.__dict__["foo"] = torch.zeros(3)
    m = M()
    with pytest.raises(RuntimeError, match="resolves to"):
        _assign_storage(m, "foo", torch.ones(3))


# ---------------------------------------------------------------------------
# Memory-bound regression tests (CUDA required)
# ---------------------------------------------------------------------------


cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for memory-bound tests"
)


def _fresh_cuda(n_params: int = 4, dim: int = 2048) -> tuple[nn.Module, int, int]:
    """Allocate a tiny skeleton on CUDA and return baseline allocated bytes
    plus the per-tensor size for assertions."""
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    m = nn.Sequential(*[
        nn.Linear(dim, dim, bias=False) for _ in range(n_params)
    ]).cuda()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    per_tensor = dim * dim * 4   # float32
    return m, baseline, per_tensor


@cuda_required
def test_streaming_swap_keeps_memory_bounded():
    """Per-tensor swap should hold steady at ~skeleton size: each rebind
    drops the old storage's refcount synchronously, the cuda allocator
    reclaims, the next allocation reuses. Caught by review pass #2."""
    m, baseline, _ = _fresh_cuda(n_params=4, dim=2048)

    for i in range(4):
        new_t = torch.randn(2048, 2048, device="cuda")
        _assign_storage(m, f"{i}.weight", new_t)
        del new_t

    torch.cuda.synchronize()
    final = torch.cuda.memory_allocated()
    # Allow 20% slop for caching-allocator metadata; if we leaked an old
    # storage somewhere this would be ~2× baseline.
    assert final <= baseline * 1.2, (
        f"swap leaked memory: baseline={baseline / 1e9:.2f} GB, "
        f"final={final / 1e9:.2f} GB"
    )


@cuda_required
def test_state_dict_binding_pins_storages_regression():
    """The exact bug review pass #2 caught: ``state = model.state_dict()``
    keeps detached aliases that pin every original storage's refcount,
    so per-tensor swaps don't actually free anything. This test
    *positively asserts* the bug exists when state is bound, then
    confirms releasing the binding drops memory back to baseline."""
    m, baseline, per_tensor = _fresh_cuda(n_params=4, dim=2048)

    # Bind the state_dict the way the buggy code did. Each value is a
    # detached alias that bumps the original storage's refcount.
    state = m.state_dict()

    # Swap each weight with a fresh tensor on cuda.
    for i in range(4):
        new_t = torch.randn(2048, 2048, device="cuda")
        _assign_storage(m, f"{i}.weight", new_t)
        del new_t

    torch.cuda.synchronize()
    pinned = torch.cuda.memory_allocated()

    # While `state` is bound, both the original AND the new storages live
    # on cuda → roughly 2× baseline. If this assertion fails, either the
    # test setup is wrong or torch changed state_dict() semantics.
    assert pinned >= baseline * 1.7, (
        f"expected memory ~2× baseline while state_dict is bound, "
        f"got pinned={pinned / 1e9:.2f} GB, baseline={baseline / 1e9:.2f} GB. "
        "Did torch.state_dict() change to clone instead of detach?"
    )

    # Now drop the binding and force the allocator to recycle.
    del state
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    released = torch.cuda.memory_allocated()

    # After release, memory should be ~baseline (just the new tensors,
    # which are the same shape as the originals).
    assert released <= baseline * 1.2, (
        f"after dropping state_dict binding, memory should return to "
        f"baseline; got {released / 1e9:.2f} GB vs baseline "
        f"{baseline / 1e9:.2f} GB. The fix may have regressed."
    )


@cuda_required
def test_streaming_swap_handles_dtype_mismatch_loudly_on_cuda():
    """Run on cuda to make sure the dtype guardrail works after the
    tensor lands on the device, not just on CPU."""
    m = nn.Linear(64, 64, bias=False).to("cuda", dtype=torch.bfloat16)
    bad = torch.zeros(64, 64, device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError, match="dtype mismatch"):
        _assign_storage(m, "weight", bad)
