"""Task 8 / LLR-0058: Profile-J YAML loads and matches the locked recipe."""

# REQ: LLR-0058

from __future__ import annotations

from pathlib import Path

import yaml

from kdr.config import Config
from kdr.quant.specs import MixedWeightSpec

_PROFILE_J_YAML = (
    Path(__file__).resolve().parent.parent
    / "configs"
    / "zaya1_8b_da_qad_profileJ_gguf.yaml"
)

# (pattern -> (bits, format, granularity)) from the locked Profile-J table
# against ZAYA1's actual module names (Megatron-style fused).
_EXPECTED_SPEC_MAP: dict[str, tuple[int, str, str]] = {
    "linear_fc1": (2, "iq2_xs", "block"),
    "linear_fc2": (3, "q3_k", "block"),
    "linear_q": (4, "iq4_xs", "block"),
    "linear_k": (4, "iq4_xs", "block"),
    "o_proj": (4, "iq4_xs", "block"),
    "embed_tokens": (5, "q5_k", "block"),
    "lm_head": (5, "q5_k", "block"),
}


def _load() -> Config:
    with _PROFILE_J_YAML.open() as fh:
        return Config.model_validate(yaml.safe_load(fh))


def test_profile_j_yaml_exists() -> None:
    """LLR-0058 AC: the file exists at the expected path."""
    assert _PROFILE_J_YAML.is_file(), f"missing config: {_PROFILE_J_YAML}"


def test_profile_j_yaml_loads_without_validation_error() -> None:
    """LLR-0058 AC: yaml.safe_load + Config.model_validate succeeds."""
    cfg = _load()
    assert isinstance(cfg, Config)


def test_profile_j_mode_is_da_qad() -> None:
    """LLR-0058 AC: top-level mode == 'da_qad'."""
    cfg = _load()
    assert cfg.mode == "da_qad"


def test_profile_j_weight_is_mixed_weight_spec() -> None:
    """LLR-0058 AC: quant.weight is a MixedWeightSpec (not Uniform)."""
    cfg = _load()
    assert isinstance(cfg.quant.weight, MixedWeightSpec)


def test_profile_j_spec_map_has_seven_entries() -> None:
    """LLR-0058 AC: exactly 7 spec_map entries."""
    cfg = _load()
    assert isinstance(cfg.quant.weight, MixedWeightSpec)
    assert len(cfg.quant.weight.spec_map) == 7


def test_profile_j_spec_map_patterns_match_set() -> None:
    """LLR-0058 AC: spec_map patterns equal the expected 7-element set."""
    cfg = _load()
    assert isinstance(cfg.quant.weight, MixedWeightSpec)
    patterns = {s.pattern for s in cfg.quant.weight.spec_map}
    assert patterns == set(_EXPECTED_SPEC_MAP.keys())


def test_profile_j_each_pattern_has_locked_triple() -> None:
    """LLR-0058 AC: each pattern's (bits, format, granularity) matches."""
    cfg = _load()
    assert isinstance(cfg.quant.weight, MixedWeightSpec)
    for spec in cfg.quant.weight.spec_map:
        expected = _EXPECTED_SPEC_MAP[spec.pattern]
        assert (spec.bits, spec.format, spec.granularity) == expected, (
            f"{spec.pattern}: expected {expected}, got "
            f"{(spec.bits, spec.format, spec.granularity)}"
        )


def test_profile_j_kv_quant_kivi_int4() -> None:
    """LLR-0058 AC: KIVI INT4 K (per-channel) + INT4 V (per-token)."""
    cfg = _load()
    k, v = cfg.quant.kv_quant.key, cfg.quant.kv_quant.value
    assert (k.bits, k.format, k.granularity, k.transform) == (
        4,
        "int",
        "channel",
        "none",
    )
    assert (v.bits, v.format, v.granularity, v.transform) == (
        4,
        "int",
        "token",
        "none",
    )


def test_profile_j_student_source_is_reasoning_base() -> None:
    """LLR-0058 AC: self-distillation seed."""
    cfg = _load()
    assert cfg.student.source == "Zyphra/ZAYA1-reasoning-base"


def test_profile_j_spec_map_excludes_carve_out_substrings() -> None:
    """LLR-0058 AC: no pattern contains val_proj/router/norm."""
    cfg = _load()
    assert isinstance(cfg.quant.weight, MixedWeightSpec)
    forbidden = {"val_proj", "router", "norm"}
    for spec in cfg.quant.weight.spec_map:
        for sub in forbidden:
            assert sub not in spec.pattern, (
                f"spec_map pattern {spec.pattern!r} contains carve-out "
                f"substring {sub!r}"
            )


def test_profile_j_spec_map_patterns_substring_disjoint() -> None:
    """LLR-0058 AC: no spec_map pattern is a substring of another.

    Confirms first-match-wins is inconsequential here — each pattern matches
    a disjoint module set on ZAYA1's named_modules() output.
    """
    cfg = _load()
    assert isinstance(cfg.quant.weight, MixedWeightSpec)
    patterns = [s.pattern for s in cfg.quant.weight.spec_map]
    for i, p1 in enumerate(patterns):
        for j, p2 in enumerate(patterns):
            if i == j:
                continue
            assert p1 not in p2, (
                f"pattern {p1!r} is a substring of {p2!r} — first-match-wins "
                f"would affect dispatch order"
            )
