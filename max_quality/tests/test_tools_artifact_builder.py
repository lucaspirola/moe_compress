"""Tests for the schema-parametric ``ArtifactBuilder``.

Covers fragment/top-level assembly and merge order, the mandatory keyword-only
``required_keys`` schema argument (missing/extra/both rejected, naming the
offending keys), duplicate-add rejection across both stores, value-type
validation, and that the same builder class works against arbitrary schemas.
"""

from __future__ import annotations

import pytest

from moe_compress.tools.artifact_builder import ArtifactBuilder


def test_assembles_top_level_and_fragments():
    b = ArtifactBuilder()
    b.set_top_level("blacklist", [1, 2, 3])
    b.add_fragment("aimer", {"score": 0.5})
    out = b.assemble(required_keys=frozenset({"blacklist", "aimer"}))
    assert out == {"blacklist": [1, 2, 3], "aimer": {"score": 0.5}}


def test_assemble_without_required_keys_raises_type_error():
    b = ArtifactBuilder()
    with pytest.raises(TypeError):
        b.assemble()  # type: ignore[call-arg]


def test_required_keys_is_keyword_only():
    b = ArtifactBuilder()
    b.set_top_level("k", 1)
    with pytest.raises(TypeError):
        # required_keys passed positionally must raise.
        b.assemble(frozenset({"k"}))  # type: ignore[misc]


def test_rejects_missing_key():
    b = ArtifactBuilder()
    b.set_top_level("present", 1)
    with pytest.raises(ValueError) as exc:
        b.assemble(required_keys=frozenset({"present", "absent"}))
    assert "absent" in str(exc.value)
    assert "missing" in str(exc.value)


def test_rejects_extra_key():
    b = ArtifactBuilder()
    b.set_top_level("expected", 1)
    b.set_top_level("surprise", 2)
    with pytest.raises(ValueError) as exc:
        b.assemble(required_keys=frozenset({"expected"}))
    assert "surprise" in str(exc.value)
    assert "extra" in str(exc.value)


def test_rejects_missing_and_extra_together():
    b = ArtifactBuilder()
    b.set_top_level("wrong", 1)
    with pytest.raises(ValueError) as exc:
        b.assemble(required_keys=frozenset({"right"}))
    msg = str(exc.value)
    assert "missing" in msg and "right" in msg
    assert "extra" in msg and "wrong" in msg


def test_duplicate_add_fragment_raises_key_error():
    b = ArtifactBuilder()
    b.add_fragment("dup", {"a": 1})
    with pytest.raises(KeyError):
        b.add_fragment("dup", {"b": 2})


def test_duplicate_set_top_level_raises_key_error():
    b = ArtifactBuilder()
    b.set_top_level("dup", 1)
    with pytest.raises(KeyError):
        b.set_top_level("dup", 2)


def test_cross_method_duplicate_raises_key_error_fragment_then_top_level():
    b = ArtifactBuilder()
    b.add_fragment("k", {"a": 1})
    with pytest.raises(KeyError):
        b.set_top_level("k", 2)


def test_cross_method_duplicate_raises_key_error_top_level_then_fragment():
    b = ArtifactBuilder()
    b.set_top_level("k", 1)
    with pytest.raises(KeyError):
        b.add_fragment("k", {"a": 2})


def test_non_dict_fragment_raises_type_error():
    b = ArtifactBuilder()
    with pytest.raises(TypeError):
        b.add_fragment("bad", [1, 2, 3])  # type: ignore[arg-type]


def test_empty_or_non_string_name_raises_value_error():
    b = ArtifactBuilder()
    with pytest.raises(ValueError):
        b.add_fragment("", {"a": 1})
    with pytest.raises(ValueError):
        b.add_fragment(42, {"a": 1})  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        b.set_top_level("", 1)
    with pytest.raises(ValueError):
        b.set_top_level(42, 1)  # type: ignore[arg-type]


def test_merge_order_top_level_then_fragments():
    b = ArtifactBuilder()
    b.set_top_level("first", "tl")
    b.add_fragment("second", {"k": "frag"})
    out = b.assemble(required_keys=frozenset({"first", "second"}))
    # top_level keys come first in iteration order, then fragments.
    assert list(out.keys()) == ["first", "second"]


def test_top_level_accepts_non_dict_value():
    b = ArtifactBuilder()
    b.set_top_level("scalar", 7)
    b.set_top_level("listed", [1, 2])
    b.set_top_level("nulled", None)
    out = b.assemble(required_keys=frozenset({"scalar", "listed", "nulled"}))
    assert out == {"scalar": 7, "listed": [1, 2], "nulled": None}


def test_assemble_returns_plain_dict():
    b = ArtifactBuilder()
    b.set_top_level("k", 1)
    out = b.assemble(required_keys=frozenset({"k"}))
    assert type(out) is dict


def test_schema_parametric_two_builders_two_schemas():
    b1 = ArtifactBuilder()
    b1.set_top_level("alpha", 1)
    b1.add_fragment("beta", {"x": 1})
    out1 = b1.assemble(required_keys=frozenset({"alpha", "beta"}))
    assert out1 == {"alpha": 1, "beta": {"x": 1}}

    b2 = ArtifactBuilder()
    b2.set_top_level("gamma", 2)
    b2.add_fragment("delta", {"y": 2})
    b2.add_fragment("epsilon", {"z": 3})
    out2 = b2.assemble(required_keys=frozenset({"gamma", "delta", "epsilon"}))
    assert out2 == {"gamma": 2, "delta": {"y": 2}, "epsilon": {"z": 3}}


def test_empty_builder_empty_schema_returns_empty_dict():
    b = ArtifactBuilder()
    out = b.assemble(required_keys=frozenset())
    assert out == {}
