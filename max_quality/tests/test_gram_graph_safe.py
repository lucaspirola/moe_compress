"""CUDA-graph-safety test for the input_cov Gram prologue + accumulation op.

The whole point of the rewrite is to drop ``enforce_eager``: the per-step Gram
accumulation must be capturable in a CUDA graph. This test captures the FULL
graph-safe prologue (compact counting-sort: scatter_add counts -> cumsum
offsets -> argsort order -> gather x_sorted) plus the grouped-SYRK op into a
``torch.cuda.CUDAGraph``, replays it on fresh input values, and asserts the
replayed Gram matches an eager recomputation. No ``.item()``/host sync /
dynamic shape inside the captured region.

Runnable directly: /usr/bin/python3 .../test_gram_graph_safe.py  (needs CUDA).
"""
import os
from pathlib import Path

os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-13.2")
os.environ["PATH"] = "/usr/local/cuda-13.2/bin:" + os.environ.get("PATH", "")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")

import torch  # noqa: E402
from torch.utils.cpp_extension import load  # noqa: E402

_CU = Path(__file__).resolve().parents[1] / "csrc" / "calibration" / "gram_grouped_accum.cu"


def _build():
    load(name="calib_gram_ext", sources=[str(_CU)],
         extra_cuda_cflags=["-O2"], is_python_module=False, verbose=False)
    return torch.ops.calib_gram.gram_grouped_accum


def _prologue(x, topk_ids, top_k, *, out):
    """Graph-safe compact counting-sort. Writes into preallocated `out` tensors
    (fixed shapes) so it is capturable. No bincount (avoids host-size sync), no
    .item(), no data-dependent shapes."""
    x_sorted, offsets, counts_scratch = out
    flat = topk_ids.reshape(-1)                                  # [R]
    counts_scratch.zero_()
    counts_scratch.scatter_add_(0, flat, torch.ones_like(flat))  # [E] per-expert token count
    offsets.zero_()
    torch.cumsum(counts_scratch, 0, out=offsets[1:])             # [E+1] prefix sum
    order = torch.argsort(flat, stable=True)                     # group rows by expert
    token_of = order // top_k                                    # [R] source token per routed slot
    torch.index_select(x, 0, token_of, out=x_sorted)             # [R, d] expert-contiguous rows


def _eager_reference(x, topk_ids, E, d, top_k):
    flat = topk_ids.reshape(-1)
    counts = torch.zeros(E, dtype=torch.int64, device="cuda")
    counts.scatter_add_(0, flat, torch.ones_like(flat))
    offsets = torch.zeros(E + 1, dtype=torch.int64, device="cuda")
    torch.cumsum(counts, 0, out=offsets[1:])
    order = torch.argsort(flat, stable=True)
    x_sorted = x[order // top_k].contiguous()
    cov = torch.zeros(E, d, d, dtype=torch.float32, device="cuda")
    cnt = torch.zeros(E, dtype=torch.int64, device="cuda")
    for e in range(E):
        lo, hi = int(offsets[e]), int(offsets[e + 1])
        cnt[e] = hi - lo
        if hi > lo:
            seg = x_sorted[lo:hi]
            cov[e] = seg.transpose(0, 1) @ seg
    return cov, cnt


def test_graph_capture_replay():
    assert torch.cuda.is_available(), "needs CUDA"
    op = _build()
    E, d, top_k, T = 6, 64, 4, 96
    R = T * top_k

    # Static (persistent-address) buffers — required for CUDA graph capture.
    x = torch.randn(T, d, device="cuda", dtype=torch.float32)
    topk_ids = torch.randint(0, E, (T, top_k), device="cuda", dtype=torch.int64)
    cov = torch.zeros(E, d, d, dtype=torch.float32, device="cuda")
    counts = torch.zeros(E, dtype=torch.int64, device="cuda")
    out = (
        torch.zeros(R, d, dtype=torch.float32, device="cuda"),   # x_sorted
        torch.zeros(E + 1, dtype=torch.int64, device="cuda"),    # offsets
        torch.zeros(E, dtype=torch.int64, device="cuda"),        # counts_scratch
    )

    def step():
        _prologue(x, topk_ids, top_k, out=out)
        op(cov, counts, out[0], out[1])

    # Warmup on a side stream (required before capture).
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            cov.zero_(); counts.zero_()
            step()
    torch.cuda.current_stream().wait_stream(s)

    # Capture.
    cov.zero_(); counts.zero_()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        step()

    # Replay on FRESH input values (copied into the static buffers in place).
    torch.manual_seed(123)
    new_x = torch.randn(T, d, device="cuda", dtype=torch.float32)
    new_ids = torch.randint(0, E, (T, top_k), device="cuda", dtype=torch.int64)
    x.copy_(new_x); topk_ids.copy_(new_ids)
    cov.zero_(); counts.zero_()
    g.replay()
    torch.cuda.synchronize()

    ref_cov, ref_cnt = _eager_reference(new_x, new_ids, E, d, top_k)
    assert torch.equal(counts, ref_cnt), f"counts: {counts} vs {ref_cnt}"
    torch.testing.assert_close(cov, ref_cov, rtol=1e-4, atol=2e-3)
    print(f"  PASS  graph capture+replay matches eager (E={E} d={d} top_k={top_k} T={T})")

    # Second replay accumulates again (proves the in-place += survives replay).
    g.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(cov, 2 * ref_cov, rtol=1e-4, atol=4e-3)
    assert torch.equal(counts, 2 * ref_cnt)
    print("  PASS  second replay doubles (in-place accumulation graph-safe)")


if __name__ == "__main__":
    test_graph_capture_replay()
    print("ALL PASS")
