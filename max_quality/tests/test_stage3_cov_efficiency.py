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


def test_single_pass_detection_case_and_whitespace_insensitive():
    """The orchestrator's `_single_pass` "all" detection must normalize
    case/whitespace IDENTICALLY to `_resolve_cov_window`. Otherwise spellings
    like "ALL" / " all " make the cov window collapse to G=N (all layers hooked)
    while `_single_pass` stays False, the CPU hot-accumulator is never engaged,
    and the all-layer Grams stay GPU-resident → the OOM Task B exists to prevent.
    """
    from moe_compress.utils.activation_hooks import InputCovarianceAccumulator
    from moe_compress.stage3.orchestrator import _maybe_cpu_hot_accum

    # Mirror the orchestrator's `_single_pass` expression exactly.
    def _single_pass(s3: dict, mg: dict) -> bool:
        raw = mg.get("cov_window_size", "auto")
        return bool(
            s3.get("cov_single_pass", False)
            or (isinstance(raw, str) and raw.strip().lower() == "all")
        )

    # Non-lowercase / whitespace "all" spellings must engage single-pass (and
    # thus the CPU hot-accumulator at every call site).
    for spelling in ("all", "ALL", "All", " all ", "\tAll\n"):
        sp = _single_pass({}, {"cov_window_size": spelling})
        assert sp is True, f"{spelling!r} must be detected as single-pass"
        acc = InputCovarianceAccumulator()
        _maybe_cpu_hot_accum(acc, sp)
        assert acc._hot_accum_device == "cpu", \
            f"helper must migrate to CPU for cov_window_size={spelling!r}"

    # cov_single_pass: true also engages regardless of window spelling.
    assert _single_pass({"cov_single_pass": True}, {}) is True

    # A real int window and "auto" must NOT trigger single-pass.
    for non_sp in (8, "auto", "AUTO", " auto "):
        sp = _single_pass({}, {"cov_window_size": non_sp})
        assert sp is False, f"{non_sp!r} must NOT be single-pass"
        acc = InputCovarianceAccumulator()
        _maybe_cpu_hot_accum(acc, sp)
        assert acc._hot_accum_device is None, \
            f"helper must be a no-op for cov_window_size={non_sp!r}"

    # Absent multi_gpu / cov_window_size (default path) → not single-pass.
    assert _single_pass({}, {}) is False


def test_resolve_single_pass_shared_helper():
    """The shared `_resolve_single_pass(config)` — single source of truth used by
    BOTH the orchestrator and the DP replica worker — must detect the same
    spellings/negatives as the inline expression it replaced.
    """
    from moe_compress.stage3.plugins.covariance_collection import _resolve_single_pass

    # "all" spellings (case/whitespace-insensitive) → single-pass.
    for spelling in ("all", "ALL", "All", " all ", "\tAll\n"):
        assert _resolve_single_pass({"multi_gpu": {"cov_window_size": spelling}}) is True, \
            f"{spelling!r} must be single-pass"

    # cov_single_pass: true → single-pass regardless of window.
    assert _resolve_single_pass({"stage3_svd": {"cov_single_pass": True}}) is True
    assert _resolve_single_pass(
        {"stage3_svd": {"cov_single_pass": True}, "multi_gpu": {"cov_window_size": 8}}
    ) is True

    # Int window / "auto" / absent → NOT single-pass (default byte-identical path).
    for non_sp in (8, "auto", "AUTO", " auto ", None):
        assert _resolve_single_pass({"multi_gpu": {"cov_window_size": non_sp}}) is False, \
            f"{non_sp!r} must NOT be single-pass"
    assert _resolve_single_pass({}) is False
    assert _resolve_single_pass({"stage3_svd": {}, "multi_gpu": {}}) is False


def test_dp_replica_worker_engages_cpu_hot_under_single_pass():
    """CRITICAL seam guard: the DP replica worker creates its OWN B_acc/C_acc and
    runs the actual _collect_covariances. Under single-pass (G=N) it MUST engage
    the CPU hot-accumulator on BOTH, or all N layers' Grams stay GPU-resident →
    OOM. The worker is a model-loading subprocess, so we guard the seam at the
    source level: the worker body must resolve single_pass via the shared helper
    and call set_hot_accumulator_device("cpu") on both accumulators.
    """
    import inspect

    from moe_compress.stage3.plugins import covariance_collection as cc

    src = inspect.getsource(cc._cov_replica_worker)
    assert "_resolve_single_pass(config)" in src, \
        "worker must resolve single-pass via the shared helper"
    assert "single_pass" in src
    # Both accumulators engage CPU-hot under single_pass.
    assert 'B_acc.set_hot_accumulator_device("cpu")' in src, \
        "worker must engage CPU-hot on B_acc under single-pass"
    assert 'C_acc.set_hot_accumulator_device("cpu")' in src, \
        "worker must engage CPU-hot on C_acc under single-pass"


# ---------------------------------------------------------------------------
# Task C — cov_num_sequences knob
# ---------------------------------------------------------------------------


def test_spec_from_config_num_sequences_override_contract():
    from moe_compress.utils.calibration import spec_from_config

    cal = {
        "num_sequences": 2048, "sequence_length": 512, "seed": 0,
        "source": "nvidia-cascade", "dataset": "nvidia/Nemotron-Cascade-2-SFT-Data",
        "subset_weights": {"math": 1.0},
    }

    spec_with = spec_from_config(cal, seed_offset=2, num_sequences_override=512)
    assert spec_with.num_sequences == 512

    spec_without = spec_from_config(cal, seed_offset=2, num_sequences_override=None)
    assert spec_without.num_sequences == 2048

    # Seed is unchanged by the override
    assert spec_with.seed == spec_without.seed

    # cal dict is not mutated
    assert cal["num_sequences"] == 2048


def test_resolve_bcov_spec_helper():
    import pytest

    try:
        from moe_compress.stage3.orchestrator import _resolve_bcov_spec
    except ImportError:
        pytest.fail("_resolve_bcov_spec not yet in orchestrator")

    cal = {
        "num_sequences": 2048, "sequence_length": 512, "seed": 0,
        "source": "nvidia-cascade", "dataset": "nvidia/Nemotron-Cascade-2-SFT-Data",
        "subset_weights": {"math": 1.0},
    }

    spec_with = _resolve_bcov_spec({"cov_num_sequences": 512}, cal)
    assert spec_with.num_sequences == 512

    spec_without = _resolve_bcov_spec({}, cal)
    assert spec_without.num_sequences == 2048

    # Seed is unchanged by the override (self-contained guard)
    assert spec_with.seed == spec_without.seed

    # cal dict not mutated
    assert cal["num_sequences"] == 2048


def test_dp_replica_worker_threads_cov_num_sequences_override():
    """The DP multi-GPU replica worker MUST honor cov_num_sequences so it builds
    the SAME calib length as the in-process _resolve_bcov_spec path; otherwise
    the parent's shard bounds (computed against the override-length calib) slice
    a differently-sized worker calib → corrupted reduced covariance.

    Spawning real processes isn't feasible here, so this asserts (a) the worker
    exposes the override param, and (b) the worker's spec-build (same kwargs)
    produces a spec IDENTICAL to _resolve_bcov_spec for the same override.
    """
    import inspect

    from moe_compress.stage3.orchestrator import _resolve_bcov_spec
    from moe_compress.stage3.plugins.covariance_collection import _cov_replica_worker
    from moe_compress.utils.calibration import spec_from_config

    # (a) param exists with a None default (default path = no override).
    sig = inspect.signature(_cov_replica_worker)
    assert "cov_num_sequences_override" in sig.parameters
    assert sig.parameters["cov_num_sequences_override"].default is None

    cal = {
        "num_sequences": 2048, "sequence_length": 512, "seed": 0,
        "source": "nvidia-cascade", "dataset": "nvidia/Nemotron-Cascade-2-SFT-Data",
        "subset_weights": {"math": 1.0},
    }

    # (b) For any override (incl. None), the worker's spec build (same kwargs it
    # uses internally) matches the in-process _resolve_bcov_spec spec.
    for override, s3 in ((512, {"cov_num_sequences": 512}), (None, {})):
        worker_spec = spec_from_config(
            cal, seed_offset=2, num_sequences_override=override
        )
        inproc_spec = _resolve_bcov_spec(s3, cal)
        assert worker_spec.num_sequences == inproc_spec.num_sequences
        assert worker_spec.seed == inproc_spec.seed


def test_run_with_oom_backoff_gentler_sequence():
    import pytest, torch
    from moe_compress.utils.auto_batch import run_with_oom_backoff

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA for OutOfMemoryError class")

    threshold = 10
    attempts = []

    def _fake_run(batch):
        attempts.append(batch)
        if batch > threshold:
            raise torch.cuda.OutOfMemoryError("simulated")
        return batch

    result = run_with_oom_backoff(_fake_run, start_batch=18, floor=4)

    # 18 -> min(int(18*0.75), 17) = min(13, 17) = 13  (OOM)
    # 13 -> min(int(13*0.75), 12) = min(9,  12) = 9   (success)
    assert attempts == [18, 13, 9], f"expected [18, 13, 9], got {attempts}"
    assert result == 9
    assert len(attempts) > 2, "must take more steps than halving ([18, 9])"


def test_run_with_oom_backoff_floor_reraise():
    import pytest, torch
    from moe_compress.utils.auto_batch import run_with_oom_backoff

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA for OutOfMemoryError class")

    def _always_oom(batch):
        raise torch.cuda.OutOfMemoryError("always")

    with pytest.raises(torch.cuda.OutOfMemoryError):
        run_with_oom_backoff(_always_oom, start_batch=4, floor=4)


def test_run_with_oom_backoff_strict_decrease_to_floor():
    import pytest, torch
    from moe_compress.utils.auto_batch import run_with_oom_backoff

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA for OutOfMemoryError class")

    attempts = []

    def _run(batch):
        attempts.append(batch)
        if batch > 2:
            raise torch.cuda.OutOfMemoryError("oom")
        return batch

    result = run_with_oom_backoff(_run, start_batch=100, floor=2)
    assert result == 2

    for i in range(1, len(attempts)):
        assert attempts[i] < attempts[i - 1], \
            f"not strictly decreasing at index {i}: {attempts}"
    assert attempts[-1] == 2
