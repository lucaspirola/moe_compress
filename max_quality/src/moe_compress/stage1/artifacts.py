"""Stage-1 artifact schema constants.

Declares the top-level-key schema the orchestrator enforces when assembling
``stage1_blacklist.json`` via the universal
:class:`~moe_compress.tools.artifact_builder.ArtifactBuilder`. Keeping the
schema in a stage-owned module (rather than the generic tool) lets the
orchestrator pass it explicitly as ``required_keys`` while the tool stays
stage-agnostic.
"""

from __future__ import annotations


REQUIRED_BLACKLIST_TOP_LEVEL_KEYS: frozenset[str] = frozenset({
    "blacklist",
    "per_expert_max",
    "config",
    "blacklist_provenance",
    "dual_signal",
    "aimer",
    "sink_token",
})
"""The 7-top-level-keys schema for ``stage1_blacklist.json``.

Test-locked at ``max_quality/tests/test_stage1_e2e.py``
(``test_blacklist_schema_seven_top_level_keys``). Any drift here breaks the
Stage 1 → Stage 2 contract.
"""


__all__ = ["REQUIRED_BLACKLIST_TOP_LEVEL_KEYS"]
