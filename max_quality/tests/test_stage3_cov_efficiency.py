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
