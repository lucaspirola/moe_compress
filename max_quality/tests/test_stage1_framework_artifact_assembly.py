"""Unit tests for ``moe_compress.stage1._framework.artifact_assembly``.

Verifies ``ArtifactBuilder``'s 7-required-key schema validator (rejects BOTH
missing and extra keys), duplicate-add protection, and schema-equivalence with
the legacy ``stage1_blacklist.json`` payload.
"""

from __future__ import annotations

import pytest

from moe_compress.stage1._framework.artifact_assembly import (
    REQUIRED_BLACKLIST_TOP_LEVEL_KEYS,
    ArtifactBuilder,
)


def _builder_with_full_blacklist_payload() -> ArtifactBuilder:
    """Hand-crafted builder mirroring the legacy ``stage1_blacklist.json`` payload.

    Top-level keys are orchestrator-owned; ``dual_signal`` / ``aimer`` /
    ``sink_token`` are plugin fragments.
    """
    b = ArtifactBuilder()
    b.set_top_level("blacklist", {})
    b.set_top_level("per_expert_max", {})
    b.set_top_level("config", {})
    b.set_top_level("blacklist_provenance", {})
    b.add_fragment("dual_signal", {})
    b.add_fragment("aimer", {})
    b.add_fragment("sink_token", {})
    return b


def test_artifact_builder_assembles_all_seven_keys():
    b = _builder_with_full_blacklist_payload()
    out = b.assemble()
    assert set(out.keys()) == REQUIRED_BLACKLIST_TOP_LEVEL_KEYS


def test_artifact_builder_rejects_missing_key():
    b = ArtifactBuilder()
    b.set_top_level("blacklist", {})
    b.set_top_level("per_expert_max", {})
    b.set_top_level("config", {})
    b.set_top_level("blacklist_provenance", {})
    b.add_fragment("dual_signal", {})
    b.add_fragment("aimer", {})
    # ``sink_token`` deliberately omitted.
    with pytest.raises(ValueError) as exc:
        b.assemble()
    msg = str(exc.value)
    assert "missing" in msg
    assert "sink_token" in msg


def test_artifact_builder_rejects_extra_key():
    b = _builder_with_full_blacklist_payload()
    b.add_fragment("foo", {"bar": 1})
    with pytest.raises(ValueError) as exc:
        b.assemble()
    msg = str(exc.value)
    assert "extra" in msg
    assert "foo" in msg


def test_artifact_builder_rejects_duplicate_add():
    b = ArtifactBuilder()
    b.add_fragment("aimer", {})
    with pytest.raises(KeyError):
        b.add_fragment("aimer", {})

    b2 = ArtifactBuilder()
    b2.set_top_level("blacklist", {})
    with pytest.raises(KeyError):
        b2.set_top_level("blacklist", {})

    # Cross-method duplicate: fragment then top-level using the same name.
    b3 = ArtifactBuilder()
    b3.add_fragment("aimer", {})
    with pytest.raises(KeyError):
        b3.set_top_level("aimer", {})


def test_artifact_builder_rejects_non_dict_fragment():
    b = ArtifactBuilder()
    with pytest.raises(TypeError):
        b.add_fragment("aimer", [1, 2])


def test_artifact_builder_rejects_empty_key_name():
    b = ArtifactBuilder()
    with pytest.raises(ValueError):
        b.set_top_level("", "v")
    with pytest.raises(ValueError):
        b.add_fragment("", {})


def test_artifact_builder_custom_required_keys():
    b = ArtifactBuilder()
    b.set_top_level("a", 1)
    b.add_fragment("b", {"x": 1})
    out = b.assemble(required_keys=frozenset({"a", "b"}))
    assert set(out.keys()) == {"a", "b"}


def test_artifact_builder_assembled_dict_matches_legacy_stage1_blacklist_shape():
    """Schema-level equivalence with the test-locked set in
    ``test_stage1_e2e.py``."""
    b = _builder_with_full_blacklist_payload()
    out = b.assemble()
    expected = {
        "blacklist",
        "per_expert_max",
        "config",
        "blacklist_provenance",
        "dual_signal",
        "aimer",
        "sink_token",
    }
    assert set(out.keys()) == expected
    # And it equals the public constant exposed by the module.
    assert set(out.keys()) == set(REQUIRED_BLACKLIST_TOP_LEVEL_KEYS)
