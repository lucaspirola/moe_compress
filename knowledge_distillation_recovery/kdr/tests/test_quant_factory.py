"""Tests for `kdr.quant.factory.partition_and_dispatch` (LLR-0017, LLR-0014, LLR-0042).

# VERIFIES: LLR-0014
# VERIFIES: LLR-0017
# VERIFIES: LLR-0042
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from kdr.config import KVQuantBlock, QuantBlock
from kdr.quant.factory import _take_first_n_sequences, partition_and_dispatch
from kdr.quant.modelopt_backend.backend import ModelOptBackend
from kdr.quant.native_backend.backend import NativeBackend
from kdr.quant.specs import (
    KVQuantSpec,
    MixedWeightSpec,
    UniformWeightSpec,
    WeightPatternSpec,
    WeightQuantSpec,
)


def _kv(bits: int, fmt: str, granularity: str = "channel") -> KVQuantSpec:
    return KVQuantSpec(bits=bits, format=fmt, granularity=granularity, transform="none")  # type: ignore[arg-type]


def _w(bits: int, fmt: str, granularity: str = "channel") -> WeightQuantSpec:
    """Uniform-shape weight spec for QuantBlock construction (parent YAML shape)."""
    return WeightQuantSpec(bits=bits, format=fmt, granularity=granularity, transform="none")  # type: ignore[arg-type]


def _wp(bits: int, fmt: str, granularity: str = "channel", pattern: str = "") -> WeightPatternSpec:
    """List-of-one pattern helper for `QuantBlockSubset.weight` (post-Phase-7.2 shape)."""
    return WeightPatternSpec(  # type: ignore[arg-type]
        pattern=pattern,
        bits=bits,
        format=fmt,
        granularity=granularity,
        transform="none",
    )


def _qb(weight: UniformWeightSpec, key: KVQuantSpec, value: KVQuantSpec) -> QuantBlock:
    return QuantBlock(weight=weight, kv_quant=KVQuantBlock(key=key, value=value))


def _qb_mixed(
    spec_map: list[WeightPatternSpec], key: KVQuantSpec, value: KVQuantSpec
) -> QuantBlock:
    return QuantBlock(
        weight=MixedWeightSpec(spec_map=spec_map),
        kv_quant=KVQuantBlock(key=key, value=value),
    )


def _mock_mtq_quantize() -> Callable[[Any, Any, Any], None]:
    """Patch target for `mtq.quantize` — counts calls; doesn't actually quantize."""
    return MagicMock()


@pytest.fixture
def tiny_model() -> nn.Module:
    """A `Sequential` with two Linears so weight-quant has something to wrap."""
    return nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 4))


@pytest.fixture
def calib_batches() -> list[torch.Tensor]:
    """Four `[B=2, T=4]` long tensors → 8 sequences total."""
    return [torch.zeros(2, 4, dtype=torch.long) for _ in range(4)]


def test_route_all_to_modelopt_emits_single_dispatch(
    tiny_model: nn.Module, calib_batches: list[torch.Tensor]
) -> None:
    """LLR-0014 AC #2: when ModelOpt covers all three quantizers, factory
    merges into ONE QuantBlockSubset and ModelOpt's apply_quant runs once."""
    qb = _qb(_w(4, "nvfp4"), _kv(8, "fp8"), _kv(8, "fp8"))
    with patch("modelopt.torch.quantization.quantize", new=_mock_mtq_quantize()) as m_q:
        backends = partition_and_dispatch(
            tiny_model,
            qb,
            calibration_batches=calib_batches,
            ptq_subset_size=4,
        )
    assert len(backends) == 1
    assert backends[0].name == "modelopt"
    assert m_q.call_count == 1


def test_route_kv_native_weight_modelopt_emits_two_dispatches(
    tiny_model: nn.Module, calib_batches: list[torch.Tensor]
) -> None:
    """NVFP4 weight (modelopt) + INT3 KV (native) → 2 backends, weight on
    modelopt called once, KV on native called once."""
    qb = _qb(_w(4, "nvfp4"), _kv(3, "int"), _kv(3, "int", granularity="token"))
    with patch("modelopt.torch.quantization.quantize", new=_mock_mtq_quantize()) as m_q:
        backends = partition_and_dispatch(
            tiny_model,
            qb,
            calibration_batches=calib_batches,
            ptq_subset_size=4,
            attention_module_paths=[],  # No KV hook targets in this tiny test model.
        )
    names = [b.name for b in backends]
    assert sorted(names) == ["modelopt", "native"]
    # ModelOpt called exactly once (weight only).
    assert m_q.call_count == 1


def test_route_all_to_native_skips_modelopt(
    tiny_model: nn.Module, calib_batches: list[torch.Tensor]
) -> None:
    """All-native recipe must NOT call mtq.quantize at all."""
    qb = _qb(_w(3, "int"), _kv(3, "int"), _kv(3, "int", granularity="token"))
    with patch("modelopt.torch.quantization.quantize", new=_mock_mtq_quantize()) as m_q:
        backends = partition_and_dispatch(
            tiny_model,
            qb,
            calibration_batches=calib_batches,
            ptq_subset_size=4,
            attention_module_paths=[],
        )
    assert len(backends) == 1
    assert backends[0].name == "native"
    assert m_q.call_count == 0


def test_modelopt_routing_requires_calibration_batches(tiny_model: nn.Module) -> None:
    """Routing to ModelOpt without calibration_batches is a programming error."""
    qb = _qb(_w(4, "nvfp4"), _kv(8, "fp8"), _kv(8, "fp8"))
    with pytest.raises(ValueError, match="calibration_batches is missing or empty"):
        partition_and_dispatch(
            tiny_model,
            qb,
            calibration_batches=None,
            ptq_subset_size=4,
        )


def test_modelopt_routing_rejects_empty_calibration_batches(tiny_model: nn.Module) -> None:
    """An empty list silently produces a no-op calibrate_loop, leaving modelopt
    with default / zero scales — guard explicitly so this fails loudly."""
    qb = _qb(_w(4, "nvfp4"), _kv(8, "fp8"), _kv(8, "fp8"))
    with pytest.raises(ValueError, match="calibration_batches is missing or empty"):
        partition_and_dispatch(
            tiny_model,
            qb,
            calibration_batches=[],
            ptq_subset_size=4,
        )


def test_modelopt_routing_requires_positive_ptq_subset(
    tiny_model: nn.Module, calib_batches: list[torch.Tensor]
) -> None:
    qb = _qb(_w(4, "nvfp4"), _kv(8, "fp8"), _kv(8, "fp8"))
    with pytest.raises(ValueError, match="ptq_subset_size must be > 0"):
        partition_and_dispatch(
            tiny_model,
            qb,
            calibration_batches=calib_batches,
            ptq_subset_size=0,
        )


def test_native_only_path_does_not_require_calibration(tiny_model: nn.Module) -> None:
    """All-native recipe doesn't need calibration batches (modelopt isn't called)."""
    qb = _qb(_w(3, "int"), _kv(3, "int"), _kv(3, "int", granularity="token"))
    backends = partition_and_dispatch(
        tiny_model,
        qb,
        calibration_batches=None,
        ptq_subset_size=0,
        attention_module_paths=[],
    )
    assert len(backends) == 1
    assert backends[0].name == "native"


# ─────────────────────────────────────────────────────────────────────────────
# Calibration subset selector (LLR-0042)
# ─────────────────────────────────────────────────────────────────────────────


def test_take_first_n_sequences_full_batches() -> None:
    """N divides evenly: returns the leading whole batches."""
    batches = [torch.zeros(2, 4) for _ in range(5)]
    out = _take_first_n_sequences(batches, 6)
    assert len(out) == 3
    assert all(b.shape[0] == 2 for b in out)


def test_take_first_n_sequences_truncates_last_batch() -> None:
    """N falls mid-batch: last batch sliced to fit exactly."""
    batches = [torch.arange(8).reshape(4, 2) for _ in range(3)]
    out = _take_first_n_sequences(batches, 5)
    # 1st batch (4 seq) + 1 from 2nd → 5 total, 2 batches returned.
    assert len(out) == 2
    assert out[0].shape[0] == 4
    assert out[1].shape[0] == 1
    # Contiguity: the truncated batch's first row equals the source batch's first row.
    assert torch.equal(out[1], batches[1][:1])


def test_take_first_n_sequences_n_zero_returns_empty() -> None:
    batches = [torch.zeros(2, 4) for _ in range(3)]
    assert _take_first_n_sequences(batches, 0) == []


def test_take_first_n_sequences_n_exceeds_returns_all() -> None:
    """N > total available: returns everything (≤ N invariant of LLR-0042 AC #2)."""
    batches = [torch.zeros(2, 4) for _ in range(3)]
    out = _take_first_n_sequences(batches, 100)
    assert len(out) == 3


# ─────────────────────────────────────────────────────────────────────────────
# Backends are constructible with adapter wiring (smoke)
# ─────────────────────────────────────────────────────────────────────────────


def test_modelopt_backend_construct() -> None:
    b = ModelOptBackend(
        calibrate_loop=lambda m: None,
        fp32_carve_outs=["lm_head"],
        weight_target_pattern="*weight_quantizer",
    )
    assert b.name == "modelopt"
    assert b.fp32_carve_outs == ["lm_head"]


def test_native_backend_construct() -> None:
    b = NativeBackend(
        attention_module_paths=["model.layers.0.self_attn"],
        kv_quant_exempt_indices=[3, 7],
        fp32_carve_outs=["lm_head"],
    )
    assert b.name == "native"
    assert b.kv_quant_exempt_indices == [3, 7]


def test_backend_apply_quant_rejects_empty_block() -> None:
    """LLR-0013 AC #3: empty subset is a contract violation."""
    from kdr.quant.interface import QuantBlockSubset

    empty = QuantBlockSubset()
    with pytest.raises(ValueError, match="empty quant block"):
        ModelOptBackend().apply_quant(nn.Linear(4, 4), empty)
    with pytest.raises(ValueError, match="empty quant block"):
        NativeBackend().apply_quant(nn.Linear(4, 4), empty)


def test_native_weight_quant_installs_parametrization(tiny_model: nn.Module) -> None:
    """NativeBackend.apply_quant on a weight-only subset registers parametrize."""
    import torch.nn.utils.parametrize as parametrize

    from kdr.quant.interface import QuantBlockSubset

    backend = NativeBackend()
    subset = QuantBlockSubset(weight=[_wp(3, "int")])
    backend.apply_quant(tiny_model, subset)
    # Both Linears in `tiny_model` are non-carve-out → both parametrized.
    n_param = sum(
        1 for m in tiny_model.modules()
        if isinstance(m, nn.Linear) and parametrize.is_parametrized(m, "weight")
    )
    assert n_param == 2
    # Forward still works after parametrization.
    out = tiny_model(torch.randn(1, 8))
    assert out.shape == (1, 4)
    backend.remove_all_hooks()


def test_native_weight_quant_skips_carve_outs() -> None:
    """fp32_carve_outs entries must skip parametrization."""
    import torch.nn.utils.parametrize as parametrize

    from kdr.quant.interface import QuantBlockSubset

    model = nn.Sequential()
    model.add_module("body", nn.Linear(8, 8))
    model.add_module("lm_head", nn.Linear(8, 16))

    backend = NativeBackend(fp32_carve_outs=["lm_head"])
    backend.apply_quant(model, QuantBlockSubset(weight=[_wp(3, "int")]))
    body_param = parametrize.is_parametrized(model.body, "weight")
    head_param = parametrize.is_parametrized(model.lm_head, "weight")
    assert body_param, "body Linear should be parametrized"
    assert not head_param, "lm_head must be skipped (FP32 carve-out)"
    backend.remove_all_hooks()


# ─────────────────────────────────────────────────────────────────────────────
# Phase 7.2 mixed-precision dispatch (Task 3)
# ─────────────────────────────────────────────────────────────────────────────


def test_uniform_weight_routes_to_modelopt_for_supported_formats(
    tiny_model: nn.Module, calib_batches: list[torch.Tensor]
) -> None:
    """Uniform NVFP4 weight + INT4 KV routes weight to ModelOpt; same
    observable behaviour as today after the uniform-to-mixed shim."""
    qb = _qb(_w(4, "nvfp4"), _kv(4, "int"), _kv(4, "int", granularity="token"))
    with patch("modelopt.torch.quantization.quantize", new=_mock_mtq_quantize()):
        backends = partition_and_dispatch(
            tiny_model,
            qb,
            calibration_batches=calib_batches,
            ptq_subset_size=4,
            attention_module_paths=[],
        )
    names = sorted(b.name for b in backends)
    # Weight + KV (int4) are all modelopt-supported → single modelopt dispatch.
    assert names == ["modelopt"]
    # The single-entry uniform shim landed on modelopt.
    modelopt = backends[0]
    sub = getattr(modelopt, "_quant_block", None)
    assert sub is not None
    assert sub.weight is not None
    assert len(sub.weight) == 1
    assert sub.weight[0].pattern == ""
    assert sub.weight[0].format == "nvfp4"


def test_uniform_weight_routes_to_native_for_unsupported_formats(
    tiny_model: nn.Module,
) -> None:
    """A Uniform config with `format=q3_k` routes weight to Native (single-
    pattern GGUF fall-through)."""
    qb = _qb(_w(3, "q3_k"), _kv(3, "int"), _kv(3, "int", granularity="token"))
    # NativeBackend's apply_quant for q3_k will hit its in-backend
    # NotImplementedError (format != "int") — that's a Task 5 concern.
    # We catch and inspect routing only.
    with (
        patch("modelopt.torch.quantization.quantize", new=_mock_mtq_quantize()),
        pytest.raises(NotImplementedError, match="format='q3_k'"),
    ):
        partition_and_dispatch(
            tiny_model,
            qb,
            calibration_batches=None,
            ptq_subset_size=0,
            attention_module_paths=[],
        )


def test_mixed_weight_all_gguf_routes_to_native(
    tiny_model: nn.Module,
) -> None:
    """All four GGUF entries land on Native; ModelOpt receives no weight."""
    spec_map = [
        _wp(2, "iq2_xs", granularity="block", pattern="experts"),
        _wp(3, "q3_k", granularity="block", pattern="up_proj"),
        _wp(4, "iq4_xs", granularity="block", pattern="down_proj"),
        _wp(5, "q5_k", granularity="block", pattern=""),  # catch-all last
    ]
    qb = _qb_mixed(
        spec_map,
        _kv(3, "int"),
        _kv(3, "int", granularity="token"),
    )
    # The NativeBackend.apply_quant will raise NotImplementedError on the
    # multi-pattern list (Task 5 territory). We assert that error to confirm
    # all four landed there.
    with pytest.raises(NotImplementedError, match="mixed-spec weight install"):
        partition_and_dispatch(
            tiny_model,
            qb,
            calibration_batches=None,
            ptq_subset_size=0,
            attention_module_paths=[],
        )


def test_mixed_weight_hybrid_splits_across_backends(
    tiny_model: nn.Module, calib_batches: list[torch.Tensor]
) -> None:
    """One NVFP4 entry (modelopt) + one IQ2_XS entry (Native) → NVFP4 →
    ModelOpt as a single-entry list (passes the A4/A5 guard), IQ2_XS →
    Native as a single-entry list with a non-empty pattern (raises
    Task-5 NotImplementedError before the format/granularity gates fire).
    """
    spec_map = [
        _wp(4, "nvfp4", granularity="block", pattern="up_proj"),
        _wp(2, "iq2_xs", granularity="block", pattern="down_proj"),
    ]
    qb = _qb_mixed(spec_map, _kv(4, "int"), _kv(4, "int", granularity="token"))
    # NativeBackend.apply_quant trips on the IQ2_XS single-pattern entry
    # (non-empty pattern → Task-5 NotImplementedError fires before the
    # format/granularity gates inside _install_weight_quant).
    with (
        patch("modelopt.torch.quantization.quantize", new=_mock_mtq_quantize()),
        pytest.raises(NotImplementedError, match="mixed-spec weight install"),
    ):
        partition_and_dispatch(
            tiny_model,
            qb,
            calibration_batches=calib_batches,
            ptq_subset_size=4,
            attention_module_paths=[],
        )


def test_describe_subset_handles_list_shape() -> None:
    """`_describe_subset` enumerates per-pattern entries in the rendered log line."""
    from kdr.quant.factory import _describe_subset
    from kdr.quant.interface import QuantBlockSubset

    sub = QuantBlockSubset(
        weight=[
            _wp(4, "nvfp4", granularity="block", pattern="up_proj"),
            _wp(2, "iq2_xs", granularity="block", pattern=""),
        ],
        key=_kv(4, "int"),
        value=_kv(4, "int", granularity="token"),
    )
    rendered = _describe_subset(sub)
    # Per-pattern weight rendering: `weight[<label>]=<bits>b/<format>`.
    assert "weight[up_proj]=4b/nvfp4" in rendered
    assert "weight[<default>]=2b/iq2_xs" in rendered
    assert "key=4b/int" in rendered
    assert "value=4b/int" in rendered


def test_quant_block_subset_rejects_empty_weight_list() -> None:
    """QuantBlockSubset(weight=[]) must raise ValidationError (review H3)."""
    import pydantic

    from kdr.quant.interface import QuantBlockSubset

    with pytest.raises(pydantic.ValidationError, match="non-empty list"):
        QuantBlockSubset(weight=[])
