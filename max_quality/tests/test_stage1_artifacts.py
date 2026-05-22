"""Unit tests for ``moe_compress.stage1.artifacts``.

Pins the Stage 1 → Stage 2 contract: ``REQUIRED_BLACKLIST_TOP_LEVEL_KEYS``
must stay a ``frozenset`` with exactly its seven members. Any drift here
breaks the schema the orchestrator enforces via ``ArtifactBuilder.assemble``.
"""

from __future__ import annotations

from moe_compress.stage1.artifacts import REQUIRED_BLACKLIST_TOP_LEVEL_KEYS


def test_required_keys_is_frozenset():
    assert isinstance(REQUIRED_BLACKLIST_TOP_LEVEL_KEYS, frozenset)


def test_required_keys_exact_seven_members():
    assert REQUIRED_BLACKLIST_TOP_LEVEL_KEYS == frozenset({
        "blacklist",
        "per_expert_max",
        "config",
        "blacklist_provenance",
        "dual_signal",
        "aimer",
        "sink_token",
    })
    assert len(REQUIRED_BLACKLIST_TOP_LEVEL_KEYS) == 7
