"""TDD tests for Stage-3 covariance efficiency knobs.

Task A — CPU hot-accumulator: a default-OFF knob that migrates the
running-sum (`_pending`) tensor to CPU after the GPU GEMM, freeing
per-layer GPU Gram VRAM. Default path (flag None) is byte-identical.
"""


def test_cpu_accum_flag_exists():
    from moe_compress.utils.activation_hooks import InputCovarianceAccumulator
    acc = InputCovarianceAccumulator()
    assert hasattr(acc, "_hot_accum_device") and acc._hot_accum_device is None
    assert callable(getattr(acc, "set_hot_accumulator_device", None))


def test_update_gpu_hot_vs_cpu_hot_bitwise():
    import torch
    from moe_compress.utils.activation_hooks import InputCovarianceAccumulator

    torch.manual_seed(0)
    d = 16
    acc_gpu = InputCovarianceAccumulator()
    acc_cpu = InputCovarianceAccumulator()
    acc_cpu.set_hot_accumulator_device("cpu")

    for _ in range(3):
        x = torch.randn(8, d)   # CPU tensor; no CUDA required
        acc_gpu.update(0, 0, "gate_proj", x)
        acc_cpu.update(0, 0, "gate_proj", x)

    acc_gpu.finalize_layer(0)
    acc_cpu.finalize_layer(0)

    k = (0, 0, "gate_proj")
    assert torch.equal(acc_gpu.covariance[k], acc_cpu.covariance[k])


def test_update_gemm_on_gpu_pending_on_cpu():
    import pytest
    import torch
    from moe_compress.utils.activation_hooks import InputCovarianceAccumulator

    if not torch.cuda.is_available():
        pytest.skip("no CUDA")

    acc = InputCovarianceAccumulator()
    acc.set_hot_accumulator_device("cpu")

    x = torch.randn(4, 8, device="cuda")
    acc.update(0, 0, "gate_proj", x)

    k = (0, 0, "gate_proj")
    assert acc._pending[k].device.type == "cpu", \
        "_pending must be on CPU when hot_accum_device='cpu'"
    assert acc._pending[k].abs().max() > 0, \
        "result must be non-zero (GEMM ran correctly on GPU)"


def test_update_cross_gpu_hot_vs_cpu_hot_bitwise():
    import torch
    from moe_compress.utils.activation_hooks import InputCovarianceAccumulator

    torch.manual_seed(7)
    d = 12
    acc_gpu = InputCovarianceAccumulator()
    acc_cpu = InputCovarianceAccumulator()
    acc_cpu.set_hot_accumulator_device("cpu")

    for _ in range(4):
        cross = torch.randn(d, d)   # CPU tensor
        acc_gpu.update_cross(0, 0, "gate_proj", cross, n_tokens=8)
        acc_cpu.update_cross(0, 0, "gate_proj", cross, n_tokens=8)

    acc_gpu.finalize_layer(0)
    acc_cpu.finalize_layer(0)

    k = (0, 0, "gate_proj")
    assert torch.equal(acc_gpu.covariance[k], acc_cpu.covariance[k])


def test_update_cross_pending_on_cpu_after_gpu_compute():
    import pytest
    import torch
    from moe_compress.utils.activation_hooks import InputCovarianceAccumulator

    if not torch.cuda.is_available():
        pytest.skip("no CUDA")

    acc = InputCovarianceAccumulator()
    acc.set_hot_accumulator_device("cpu")

    cross = torch.randn(8, 8, device="cuda")
    acc.update_cross(0, 0, "gate_proj", cross, n_tokens=4)

    k = (0, 0, "gate_proj")
    assert acc._pending[k].device.type == "cpu", \
        "_pending must be on CPU when hot_accum_device='cpu'"
    expected = cross.to(torch.float32).cpu()
    assert torch.equal(acc._pending[k], expected)


def test_finalize_layer_already_cpu_pending():
    import torch
    from moe_compress.utils.activation_hooks import InputCovarianceAccumulator

    acc = InputCovarianceAccumulator()
    acc.set_hot_accumulator_device("cpu")
    x = torch.randn(6, 10)
    acc.update(0, 2, "gate_proj", x)

    k = (0, 2, "gate_proj")
    assert acc._pending[k].device.type == "cpu"

    acc.finalize_layer(0)

    assert k in acc.covariance
    assert acc.covariance[k].device.type == "cpu"
    expected = (x.T @ x).to(acc.storage_dtype)
    assert torch.allclose(acc.covariance[k], expected, atol=1e-6)


def test_default_none_pending_stays_on_input_device():
    import pytest
    import torch
    from moe_compress.utils.activation_hooks import InputCovarianceAccumulator

    if not torch.cuda.is_available():
        pytest.skip("no CUDA")

    acc = InputCovarianceAccumulator()   # _hot_accum_device=None (default)
    x = torch.randn(4, 8, device="cuda")
    acc.update(0, 0, "gate_proj", x)

    k = (0, 0, "gate_proj")
    assert acc._pending[k].device.type == "cuda", \
        "default path must leave _pending on the input device"


# ---------------------------------------------------------------------------
# Task B — single-pass G=N
# ---------------------------------------------------------------------------


def test_resolve_cov_window_single_pass_spellings():
    from moe_compress.stage3.plugins.covariance_collection import _resolve_cov_window

    n = 40
    assert _resolve_cov_window({"stage3_svd": {"cov_single_pass": True}}, n) == n
    assert _resolve_cov_window({"multi_gpu": {"cov_window_size": "all"}}, n) == n
    assert _resolve_cov_window({"multi_gpu": {"cov_window_size": "ALL"}}, n) == n

    # Default path: valid int in [1, n]
    result = _resolve_cov_window({}, n)
    assert 1 <= result <= n

    # n_layers=0 guard unchanged (fires before new code)
    assert _resolve_cov_window({"stage3_svd": {"cov_single_pass": True}}, 0) == 1


def test_single_pass_vs_windowed_bitwise_tiny():
    import torch
    from moe_compress.utils.activation_hooks import InputCovarianceAccumulator

    torch.manual_seed(42)
    n_layers, n_experts, d = 4, 2, 8

    data = {
        (li, ei): torch.randn(6, d)
        for li in range(n_layers) for ei in range(n_experts)
    }

    def _run(window_size, use_cpu_accum):
        acc = InputCovarianceAccumulator()
        if use_cpu_accum:
            acc.set_hot_accumulator_device("cpu")
        for w_start in range(0, n_layers, window_size):
            w_end = min(w_start + window_size, n_layers)
            for li in range(w_start, w_end):
                for ei in range(n_experts):
                    acc.update(li, ei, "gate_proj", data[(li, ei)])
            for li in range(w_start, w_end):
                acc.finalize_layer(li)
        return acc.covariance

    cov_windowed = _run(window_size=2, use_cpu_accum=False)
    cov_single   = _run(window_size=4, use_cpu_accum=True)

    assert set(cov_windowed.keys()) == set(cov_single.keys())
    for k in cov_windowed:
        assert torch.equal(cov_windowed[k], cov_single[k]), f"mismatch at {k}"


def test_maybe_cpu_hot_accum_helper():
    import pytest
    from moe_compress.utils.activation_hooks import InputCovarianceAccumulator

    try:
        from moe_compress.stage3.orchestrator import _maybe_cpu_hot_accum
    except ImportError:
        pytest.fail("_maybe_cpu_hot_accum not yet in orchestrator")

    # (a) single_pass=True sets _hot_accum_device="cpu"
    acc_on = InputCovarianceAccumulator()
    result = _maybe_cpu_hot_accum(acc_on, single_pass=True)
    assert acc_on._hot_accum_device == "cpu"

    # (b) single_pass=False leaves _hot_accum_device None
    acc_off = InputCovarianceAccumulator()
    result_off = _maybe_cpu_hot_accum(acc_off, single_pass=False)
    assert acc_off._hot_accum_device is None

    # (c) returns the same accumulator object (for chaining / call-site clarity)
    assert result is acc_on
    assert result_off is acc_off
