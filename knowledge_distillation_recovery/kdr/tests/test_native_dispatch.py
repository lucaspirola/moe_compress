"""Task 5 / LLR-0055: NativeBackend per-pattern weight install."""

# REQ: LLR-0055

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

from kdr.quant.interface import QuantBlockSubset
from kdr.quant.native_backend.backend import (
    _GGUF_PARAMETRIZATIONS,
    NativeBackend,
    _IntQuantWeight,
    _IQ2XSQuantWeight,
    _IQ4XSQuantWeight,
    _Q3KQuantWeight,
    _Q5KQuantWeight,
)
from kdr.quant.specs import WeightPatternSpec


def _wp(
    bits: int,
    fmt: str,
    *,
    granularity: str = "block",
    pattern: str = "",
    transform: str = "none",
) -> WeightPatternSpec:
    return WeightPatternSpec(  # type: ignore[arg-type]
        bits=bits,
        format=fmt,
        granularity=granularity,
        transform=transform,
        pattern=pattern,
    )


class _ThreeProjModel(nn.Module):
    """Stand-in MoE-expert-like model with named gate/up/down projections."""

    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Module()
        # in_features chosen as 256 (smallest GGUF super-block multiple) so
        # the parametrized forward could in principle run; the install-only
        # tests here never call forward.
        self.mlp.gate_proj = nn.Linear(256, 256)
        self.mlp.up_proj = nn.Linear(256, 256)
        self.mlp.down_proj = nn.Linear(256, 256)


def _param_class(module: nn.Linear) -> type[nn.Module]:
    """Return the type of the single parametrization installed on weight."""
    plist = module.parametrizations["weight"]
    assert len(plist) == 1, f"expected exactly one parametrization, got {len(plist)}"
    return type(plist[0])


# ─────────────────────────────────────────────────────────────────────────────
# Profile-J three-Linear dispatch
# ─────────────────────────────────────────────────────────────────────────────


def test_profile_j_three_linears_get_format_specific_parametrizations() -> None:
    """LLR-0055 AC: gate_proj/up_proj/down_proj receive their per-format STE."""
    model = _ThreeProjModel()
    spec_map = [
        _wp(2, "iq2_xs", pattern="gate_proj"),
        _wp(2, "iq2_xs", pattern="up_proj"),
        _wp(3, "q3_k", pattern="down_proj"),
    ]
    backend = NativeBackend()
    backend.apply_quant(model, QuantBlockSubset(weight=spec_map))
    assert _param_class(model.mlp.gate_proj) is _IQ2XSQuantWeight
    assert _param_class(model.mlp.up_proj) is _IQ2XSQuantWeight
    assert _param_class(model.mlp.down_proj) is _Q3KQuantWeight
    backend.remove_all_hooks()


# ─────────────────────────────────────────────────────────────────────────────
# First-match-wins precedence
# ─────────────────────────────────────────────────────────────────────────────


def test_first_match_wins_precedence() -> None:
    """LLR-0055 AC: pattern='proj' wins over a later pattern='q_proj'."""
    model = nn.Module()
    model.attn = nn.Module()
    model.attn.q_proj = nn.Linear(256, 256)
    spec_map = [
        _wp(4, "iq4_xs", pattern="proj"),
        _wp(5, "q5_k", pattern="q_proj"),  # would have won if order reversed
    ]
    backend = NativeBackend()
    backend.apply_quant(model, QuantBlockSubset(weight=spec_map))
    assert _param_class(model.attn.q_proj) is _IQ4XSQuantWeight
    backend.remove_all_hooks()


# ─────────────────────────────────────────────────────────────────────────────
# Carve-out precedence (carve-out wins after spec_map miss)
# ─────────────────────────────────────────────────────────────────────────────


def test_explicit_pattern_wins_over_carve_out() -> None:
    """LLR-0055 AC: explicit (non-empty) spec_map pattern wins over carve-out."""
    model = nn.Module()
    model.lm_head = nn.Linear(256, 256)
    backend = NativeBackend(fp32_carve_outs=["lm_head"])
    backend.apply_quant(
        model,
        QuantBlockSubset(weight=[_wp(5, "q5_k", pattern="lm_head")]),
    )
    # Explicit pattern overrides the carve-out — the locked design choice.
    assert _param_class(model.lm_head) is _Q5KQuantWeight
    backend.remove_all_hooks()


def test_carve_out_wins_after_spec_map_miss() -> None:
    """LLR-0055 AC: a Linear matched only by a carve-out installs nothing."""
    model = nn.Sequential()
    model.add_module("body", nn.Linear(256, 256))
    model.add_module("lm_head", nn.Linear(256, 256))
    spec_map = [
        _wp(2, "iq2_xs", pattern="body"),  # only matches body
    ]
    backend = NativeBackend(fp32_carve_outs=["lm_head"])
    backend.apply_quant(model, QuantBlockSubset(weight=spec_map))
    assert parametrize.is_parametrized(model.body, "weight")
    assert not parametrize.is_parametrized(model.lm_head, "weight")
    backend.remove_all_hooks()


# ─────────────────────────────────────────────────────────────────────────────
# Same pattern matches multiple Linears
# ─────────────────────────────────────────────────────────────────────────────


def test_same_pattern_matches_multiple_linears() -> None:
    """LLR-0055 AC: a single pattern that matches N Linears installs N times."""
    model = nn.Module()
    model.layer_a = nn.Linear(256, 256)
    model.layer_b = nn.Linear(256, 256)
    model.layer_c = nn.Linear(256, 256)
    backend = NativeBackend()
    backend.apply_quant(
        model,
        QuantBlockSubset(weight=[_wp(2, "iq2_xs", pattern="layer_")]),
    )
    for sub in (model.layer_a, model.layer_b, model.layer_c):
        assert _param_class(sub) is _IQ2XSQuantWeight
    backend.remove_all_hooks()


# ─────────────────────────────────────────────────────────────────────────────
# Unmatched non-carve-out → ValueError with the module name
# ─────────────────────────────────────────────────────────────────────────────


def test_unmatched_non_carve_out_raises_with_name() -> None:
    """LLR-0055 AC: ValueError names the unmatched module."""
    model = nn.Module()
    model.weird_layer = nn.Linear(256, 256)
    backend = NativeBackend()
    with pytest.raises(ValueError, match="weird_layer"):
        backend.apply_quant(
            model,
            QuantBlockSubset(weight=[_wp(2, "iq2_xs", pattern="nope")]),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Unwired GGUF format → NotImplementedError at install (eager fail)
# ─────────────────────────────────────────────────────────────────────────────


def test_unwired_gguf_format_raises_at_install() -> None:
    """LLR-0055 AC: format='iq3_xs' raises NotImplementedError eagerly."""
    model = nn.Module()
    model.a = nn.Linear(256, 256)
    backend = NativeBackend()
    with pytest.raises(NotImplementedError, match="iq3_xs"):
        backend.apply_quant(
            model,
            QuantBlockSubset(weight=[_wp(3, "iq3_xs", pattern="a")]),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Granularity-coherence: incompatible (format, granularity) → NotImplementedError
# ─────────────────────────────────────────────────────────────────────────────


def test_granularity_coherence_int_with_block_raises() -> None:
    """LLR-0055 AC: int with granularity='block' is incoherent."""
    model = nn.Module()
    model.a = nn.Linear(256, 256)
    backend = NativeBackend()
    with pytest.raises(NotImplementedError, match="granularity"):
        backend.apply_quant(
            model,
            QuantBlockSubset(
                weight=[_wp(3, "int", granularity="block", pattern="a")]
            ),
        )


def test_granularity_coherence_gguf_with_channel_raises() -> None:
    """LLR-0055 AC: GGUF format with granularity='channel' is incoherent."""
    model = nn.Module()
    model.a = nn.Linear(256, 256)
    backend = NativeBackend()
    with pytest.raises(NotImplementedError, match="granularity"):
        backend.apply_quant(
            model,
            QuantBlockSubset(
                weight=[_wp(2, "iq2_xs", granularity="channel", pattern="a")]
            ),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Uniform shim regression: single-entry list with pattern="" installs INT
# ─────────────────────────────────────────────────────────────────────────────


def test_uniform_shim_regression_installs_int_globally() -> None:
    """LLR-0055 AC: a v0-style uniform YAML installs INT on every non-carve-out Linear.

    Loads a uniform-weight YAML (no spec_map) into a `QuantBlock`, normalizes
    it via the factory's uniform→mixed shim, and hands the result to
    `NativeBackend._install_weight_quant`. Verifies the shim produces a
    single-entry list with `pattern=""` and that `_IntQuantWeight` lands on
    every non-carve-out Linear.
    """
    import yaml

    from kdr.config import QuantBlock
    from kdr.quant.factory import _normalize_weight_to_patterns

    uniform_yaml = """
    weight:
      bits: 3
      format: int
      granularity: channel
      transform: none
    kv_quant:
      key:
        bits: 3
        format: int
        granularity: channel
        transform: none
      value:
        bits: 3
        format: int
        granularity: token
        transform: none
    """
    qb = QuantBlock.model_validate(yaml.safe_load(uniform_yaml))
    patterns = _normalize_weight_to_patterns(qb.weight)
    assert len(patterns) == 1
    assert patterns[0].pattern == ""

    model = nn.Sequential()
    model.add_module("body", nn.Linear(256, 256))
    model.add_module("lm_head", nn.Linear(256, 256))
    backend = NativeBackend(fp32_carve_outs=["lm_head"])
    backend._install_weight_quant(model, patterns)
    assert _param_class(model.body) is _IntQuantWeight
    assert not parametrize.is_parametrized(model.lm_head, "weight")
    backend.remove_all_hooks()


# ─────────────────────────────────────────────────────────────────────────────
# Any-length-≥1 list contract
# ─────────────────────────────────────────────────────────────────────────────


def test_multi_entry_list_succeeds_end_to_end() -> None:
    """LLR-0055 AC: a multi-entry list installs without raising on the supported set."""
    model = _ThreeProjModel()
    spec_map = [
        _wp(2, "iq2_xs", pattern="gate_proj"),
        _wp(3, "q3_k", pattern="up_proj"),
        _wp(4, "iq4_xs", pattern="down_proj"),
    ]
    backend = NativeBackend()
    backend.apply_quant(model, QuantBlockSubset(weight=spec_map))
    assert _param_class(model.mlp.gate_proj) is _IQ2XSQuantWeight
    assert _param_class(model.mlp.up_proj) is _Q3KQuantWeight
    assert _param_class(model.mlp.down_proj) is _IQ4XSQuantWeight
    backend.remove_all_hooks()


# ─────────────────────────────────────────────────────────────────────────────
# Parametrization classes — interface sanity
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "fmt,cls",
    [
        ("iq2_xs", _IQ2XSQuantWeight),
        ("q3_k", _Q3KQuantWeight),
        ("iq4_xs", _IQ4XSQuantWeight),
        ("q5_k", _Q5KQuantWeight),
    ],
)
def test_parametrization_classes_have_expected_interface(
    fmt: str, cls: type[nn.Module]
) -> None:
    """LLR-0055 AC: each class subclasses nn.Module, takes no args, forward(w) is callable."""
    assert _GGUF_PARAMETRIZATIONS[fmt] is cls
    instance = cls()
    assert isinstance(instance, nn.Module)
    w = torch.randn(2, 256)
    out = instance.forward(w)
    assert out.shape == w.shape
