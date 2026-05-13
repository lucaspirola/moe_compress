"""Task 6 / LLR-0056: per-pattern config_groups emission + NativeBackend.save."""

# REQ: LLR-0056

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

from kdr.config import KVQuantBlock, QuantBlock
from kdr.io.save import (
    _build_quantization_config,
    _weight_spec_to_ct,
    save_kdr_artifact,
)
from kdr.quant.interface import QuantBlockSubset
from kdr.quant.native_backend.backend import NativeBackend, _IntQuantWeight
from kdr.quant.specs import (
    KVQuantSpec,
    MixedWeightSpec,
    UniformWeightSpec,
    WeightPatternSpec,
)


def _kv() -> tuple[KVQuantSpec, KVQuantSpec]:
    k = KVQuantSpec(bits=4, format="int", granularity="channel", transform="none")  # type: ignore[arg-type]
    v = KVQuantSpec(bits=4, format="int", granularity="token", transform="none")  # type: ignore[arg-type]
    return k, v


def _wp(bits: int, fmt: str, granularity: str = "block", pattern: str = "") -> WeightPatternSpec:
    return WeightPatternSpec(  # type: ignore[arg-type]
        pattern=pattern,
        bits=bits,
        format=fmt,
        granularity=granularity,
        transform="none",
    )


# ─────────────────────────────────────────────────────────────────────────────
# _build_quantization_config — Mixed path
# ─────────────────────────────────────────────────────────────────────────────


def test_mixed_four_distinct_triples_produce_four_groups() -> None:
    """LLR-0056 AC: four distinct (format, bits, granularity) triples → 4 groups."""
    k, v = _kv()
    qb = QuantBlock(
        weight=MixedWeightSpec(
            spec_map=[
                _wp(2, "iq2_xs", pattern="gate_proj"),
                _wp(3, "q3_k", pattern="down_proj"),
                _wp(4, "iq4_xs", pattern="q_proj"),
                _wp(5, "q5_k", pattern="embed_tokens"),
            ]
        ),
        kv_quant=KVQuantBlock(key=k, value=v),
    )
    cfg = _build_quantization_config(qb, ["lm_head"])
    groups = cfg["config_groups"]
    assert set(groups.keys()) == {"group_0", "group_1", "group_2", "group_3"}
    assert groups["group_0"]["targets"] == ["gate_proj"]
    assert groups["group_0"]["weights"]["num_bits"] == 2
    assert groups["group_1"]["targets"] == ["down_proj"]
    assert groups["group_1"]["weights"]["num_bits"] == 3
    assert groups["group_2"]["targets"] == ["q_proj"]
    assert groups["group_2"]["weights"]["num_bits"] == 4
    assert groups["group_3"]["targets"] == ["embed_tokens"]
    assert groups["group_3"]["weights"]["num_bits"] == 5


def test_mixed_same_triple_patterns_collapse_to_one_group() -> None:
    """LLR-0056 AC: same (format, bits, granularity) triple → 1 group, multi-target."""
    k, v = _kv()
    qb = QuantBlock(
        weight=MixedWeightSpec(
            spec_map=[
                _wp(2, "iq2_xs", pattern="gate_proj"),
                _wp(2, "iq2_xs", pattern="up_proj"),
                _wp(3, "q3_k", pattern="down_proj"),
            ]
        ),
        kv_quant=KVQuantBlock(key=k, value=v),
    )
    cfg = _build_quantization_config(qb, [])
    groups = cfg["config_groups"]
    assert set(groups.keys()) == {"group_0", "group_1"}
    assert groups["group_0"]["targets"] == ["gate_proj", "up_proj"]
    assert groups["group_0"]["weights"]["num_bits"] == 2
    assert groups["group_1"]["targets"] == ["down_proj"]
    assert groups["group_1"]["weights"]["num_bits"] == 3


def test_uniform_path_unchanged() -> None:
    """LLR-0056 AC: Uniform-path emission preserved (single group_0 / Linear)."""
    k, v = _kv()
    qb = QuantBlock(
        weight=UniformWeightSpec(bits=4, format="nvfp4", granularity="channel", transform="none"),  # type: ignore[arg-type]
        kv_quant=KVQuantBlock(key=k, value=v),
    )
    cfg = _build_quantization_config(qb, ["lm_head", "rmsnorm"])
    assert list(cfg["config_groups"].keys()) == ["group_0"]
    assert cfg["config_groups"]["group_0"]["targets"] == ["Linear"]
    assert cfg["config_groups"]["group_0"]["weights"]["num_bits"] == 4
    assert cfg["config_groups"]["group_0"]["weights"]["type"] == "float"


def test_weight_spec_to_ct_accepts_uniform_and_pattern() -> None:
    """LLR-0056 AC: _weight_spec_to_ct accepts both UniformWeightSpec and WeightPatternSpec."""
    u = UniformWeightSpec(bits=4, format="int", granularity="channel", transform="none")  # type: ignore[arg-type]
    p = WeightPatternSpec(pattern="proj", bits=4, format="int", granularity="channel", transform="none")  # type: ignore[arg-type]
    assert _weight_spec_to_ct(u) == _weight_spec_to_ct(p)


def test_ignore_and_kv_cache_scheme_shape_preserved() -> None:
    """LLR-0056 AC: ignore = list(fp32_carve_outs); kv_cache_scheme has key+value."""
    k, v = _kv()
    qb = QuantBlock(
        weight=MixedWeightSpec(spec_map=[_wp(2, "iq2_xs", pattern="x")]),
        kv_quant=KVQuantBlock(key=k, value=v),
    )
    cfg = _build_quantization_config(qb, ["foo", "bar", "baz"])
    assert cfg["ignore"] == ["foo", "bar", "baz"]
    assert set(cfg["kv_cache_scheme"].keys()) == {"key", "value"}
    assert cfg["kv_cache_scheme"]["key"]["num_bits"] == 4
    assert cfg["kv_cache_scheme"]["value"]["strategy"] == "token"


def test_unknown_weight_type_raises_value_error() -> None:
    """LLR-0056 AC: defense-in-depth ValueError on non-union weight type."""
    k, v = _kv()
    qb = QuantBlock(
        weight=UniformWeightSpec(bits=4, format="int", granularity="channel", transform="none"),  # type: ignore[arg-type]
        kv_quant=KVQuantBlock(key=k, value=v),
    )
    # Forcibly clobber the type-checker-proven branch with an object that
    # Pydantic would never produce, to exercise the defense-in-depth check.
    qb.__dict__["weight"] = object()
    with pytest.raises(ValueError, match="unexpected weight type"):
        _build_quantization_config(qb, [])


# ─────────────────────────────────────────────────────────────────────────────
# NativeBackend.save — no longer raises NotImplementedError
# ─────────────────────────────────────────────────────────────────────────────


class _SaveablePretrained(nn.Module):
    """Stand-in module that mimics HF's save_pretrained surface.

    Records the state_dict it was handed and writes a stub safetensors
    file plus a config.json so the post-save inspection has something to
    look at.
    """

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.saved_state_dict: dict[str, Any] | None = None

    def save_pretrained(
        self,
        output_dir: Path,
        *,
        state_dict: dict[str, Any] | None = None,
        safe_serialization: bool = True,
    ) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.saved_state_dict = state_dict
        (out / "config.json").write_text(json.dumps({"hidden_size": 4}))
        (out / "model.safetensors").write_bytes(b"\x00" * 16)


def test_native_backend_save_emits_bf16_safetensors(tmp_path: Path) -> None:
    """LLR-0056 AC: NativeBackend.save calls save_pretrained with BF16 state_dict."""
    model = _SaveablePretrained()
    parametrize.register_parametrization(
        model.linear, "weight", _IntQuantWeight(bits=3, axis=0)
    )
    assert parametrize.is_parametrized(model.linear, "weight")
    backend = NativeBackend()
    backend._quant_block = QuantBlockSubset(
        weight=[_wp(3, "int", granularity="channel", pattern="linear")]
    )
    backend.save(model, tmp_path / "out")
    assert (tmp_path / "out" / "model.safetensors").exists()
    assert model.saved_state_dict is not None
    for name, tensor in model.saved_state_dict.items():
        assert tensor.dtype == torch.bfloat16, (
            f"state_dict[{name!r}] dtype is {tensor.dtype}; expected bfloat16"
        )


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end: save_kdr_artifact with a 2-entry MixedWeightSpec
# ─────────────────────────────────────────────────────────────────────────────


class _SaveableBackend:
    """Stand-in backend that owns a model and writes a stub safetensors dir."""

    name = "fake"

    def __init__(self, weight: list[WeightPatternSpec]) -> None:
        self._quant_block = QuantBlockSubset(weight=weight)

    def apply_quant(self, model: nn.Module, qb: QuantBlockSubset) -> None:
        pass

    def save(self, model: nn.Module, output_dir: Path) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text(json.dumps({"hidden_size": 4}))
        (out / "model.safetensors").write_bytes(b"\x00" * 16)


def test_save_kdr_artifact_end_to_end_two_entry_mixed_spec(tmp_path: Path) -> None:
    """LLR-0056 AC: end-to-end 2-entry MixedWeightSpec → safetensors + 2-group config_groups."""
    k, v = _kv()
    spec_map = [
        _wp(2, "iq2_xs", pattern="gate_proj"),
        _wp(3, "q3_k", pattern="down_proj"),
    ]
    qb = QuantBlock(
        weight=MixedWeightSpec(spec_map=spec_map),
        kv_quant=KVQuantBlock(key=k, value=v),
    )
    backend = _SaveableBackend(spec_map)
    save_kdr_artifact(
        nn.Linear(4, 4),
        tmp_path / "out",
        backends=[backend],  # type: ignore[list-item]
        quant_block=qb,
        fp32_carve_outs=["lm_head"],
    )
    # (a) safetensors file present
    assert (tmp_path / "out" / "model.safetensors").exists()
    # (b) config.json has two config_groups with expected targets
    cfg = json.loads((tmp_path / "out" / "config.json").read_text())
    groups = cfg["quantization_config"]["config_groups"]
    assert set(groups.keys()) == {"group_0", "group_1"}
    assert groups["group_0"]["targets"] == ["gate_proj"]
    assert groups["group_1"]["targets"] == ["down_proj"]
