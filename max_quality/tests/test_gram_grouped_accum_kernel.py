"""Numerical-parity test for the grouped-SYRK Gram accumulation CUDA kernel.

Proves the standalone kernel (max_quality/csrc/calibration/gram_grouped_accum.cu)
computes, per expert e, the RAW Gram  cov[e] += X_e^T X_e  and exact token
counts, matching a torch fp32 reference, across multiple dim configs (including
d != hidden, top_k variants, and empty experts) to demonstrate it is
dimension-agnostic. Also checks in-place accumulation (call twice == 2x).

Runnable directly:  /usr/bin/python3 .../test_gram_grouped_accum_kernel.py
or via pytest. Requires a CUDA GPU (dev: shared RTX5080 -- ONE gpu job at a time).
"""
import os
from pathlib import Path

# Toolchain: torch 2.11.0+cu130 needs a CUDA-13 nvcc (cu130 major == 13).
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-13.2")
os.environ["PATH"] = "/usr/local/cuda-13.2/bin:" + os.environ.get("PATH", "")
# Build only for the dev card (sm_120 / Blackwell). The shipped wheel uses the
# full 8.0;9.0a;10.0;12.0 set.
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")

import torch  # noqa: E402
from torch.utils.cpp_extension import load  # noqa: E402

_CU = Path(__file__).resolve().parents[1] / "csrc" / "calibration" / "gram_grouped_accum.cu"


def _build():
    # TORCH_LIBRARY-only extension (no PyInit): load the .so for its op-registration
    # side effect and reach the op via torch.ops.
    load(
        name="calib_gram_ext",
        sources=[str(_CU)],
        extra_cuda_cflags=["-O2"],
        is_python_module=False,
        verbose=True,
    )
    return torch.ops.calib_gram.gram_grouped_accum


def _counting_sort(x: torch.Tensor, topk_ids: torch.Tensor, E: int):
    """Build (x_sorted [R,d], offsets [E+1]) the way the graph-safe prologue
    will: each routed (token, slot) contributes the token's input row x[token].
    Sorted so each expert's rows are contiguous."""
    top_k = topk_ids.shape[1]
    flat = topk_ids.reshape(-1)                       # [R] expert per routed slot
    order = torch.argsort(flat, stable=True)          # group by expert
    token_of = (order // top_k)                       # which token each slot came from
    x_sorted = x[token_of].contiguous()               # [R, d]
    counts = torch.bincount(flat, minlength=E)        # [E]
    offsets = torch.zeros(E + 1, dtype=torch.int64, device=x.device)
    offsets[1:] = torch.cumsum(counts, 0)
    return x_sorted, offsets


def _reference(x_sorted, offsets, E, d, device):
    cov = torch.zeros(E, d, d, dtype=torch.float32, device=device)
    cnt = torch.zeros(E, dtype=torch.int64, device=device)
    for e in range(E):
        lo, hi = int(offsets[e]), int(offsets[e + 1])
        cnt[e] = hi - lo
        if hi > lo:
            seg = x_sorted[lo:hi]                      # [n_e, d]
            cov[e] = seg.transpose(0, 1) @ seg
    return cov, cnt


def _one_case(op, E, d, top_k, T, seed, force_empty_expert=None):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(T, d, device="cuda", dtype=torch.float32, generator=g)
    topk_ids = torch.randint(0, E, (T, top_k), device="cuda", generator=g)
    if force_empty_expert is not None:
        # Guarantee at least one empty expert by routing away from it.
        topk_ids[topk_ids == force_empty_expert] = (force_empty_expert + 1) % E

    x_sorted, offsets = _counting_sort(x, topk_ids, E)

    cov = torch.zeros(E, d, d, dtype=torch.float32, device="cuda")
    counts = torch.zeros(E, dtype=torch.int64, device="cuda")
    op(cov, counts, x_sorted, offsets)
    torch.cuda.synchronize()

    ref_cov, ref_cnt = _reference(x_sorted, offsets, E, d, "cuda")

    assert torch.equal(counts, ref_cnt), (
        f"counts mismatch E={E} d={d} top_k={top_k}: {counts} vs {ref_cnt}")
    torch.testing.assert_close(cov, ref_cov, rtol=1e-4, atol=1e-3)

    if force_empty_expert is not None:
        assert int(counts[force_empty_expert]) == 0
        assert torch.count_nonzero(cov[force_empty_expert]) == 0

    # In-place accumulation: a second call must double the Gram and counts.
    op(cov, counts, x_sorted, offsets)
    torch.cuda.synchronize()
    torch.testing.assert_close(cov, 2 * ref_cov, rtol=1e-4, atol=2e-3)
    assert torch.equal(counts, 2 * ref_cnt)


def test_gram_grouped_accum_parity():
    assert torch.cuda.is_available(), "needs CUDA"
    op = _build()
    cases = [
        # (E, d, top_k, T, seed, force_empty_expert)
        dict(E=3, d=8,  top_k=2, T=10, seed=0,  force_empty_expert=None),
        dict(E=5, d=16, top_k=1, T=7,  seed=1,  force_empty_expert=2),
        dict(E=4, d=33, top_k=8, T=50, seed=2,  force_empty_expert=None),  # d not tile-multiple
        dict(E=6, d=64, top_k=4, T=80, seed=3,  force_empty_expert=5),
        dict(E=2, d=128,top_k=2, T=64, seed=4,  force_empty_expert=None),  # down-like larger d
    ]
    for c in cases:
        _one_case(op, **c)
        print(f"  PASS  E={c['E']} d={c['d']} top_k={c['top_k']} T={c['T']}"
              f"{' (empty expert)' if c['force_empty_expert'] is not None else ''}")


if __name__ == "__main__":
    test_gram_grouped_accum_parity()
    print("ALL PASS")
