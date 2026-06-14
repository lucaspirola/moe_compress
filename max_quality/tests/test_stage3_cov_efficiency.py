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
