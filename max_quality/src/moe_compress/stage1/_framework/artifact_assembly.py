"""``ArtifactBuilder`` — assembles ``stage1_blacklist.json`` from per-plugin fragments.

Plugins contribute named *fragments* (one per plugin: ``aimer``,
``sink_token``, ``dual_signal``); the orchestrator sets *top-level* keys it
owns directly (``blacklist``, ``per_expert_max``, ``config``,
``blacklist_provenance``). :meth:`ArtifactBuilder.assemble` validates that
the union covers :data:`REQUIRED_BLACKLIST_TOP_LEVEL_KEYS` exactly — extra
keys are rejected too, to keep the Stage 2 reader contract from silently
drifting.

The legacy inline ``save_json_artifact({...})`` call defines the schema
this assembler enforces.
"""

from __future__ import annotations

from typing import Any


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


class ArtifactBuilder:
    """Builds a JSON-ready dict from per-plugin fragments + orchestrator-owned keys.

    Plugins contribute named *fragments* (one per plugin: ``aimer``,
    ``sink_token``, ``dual_signal``); the orchestrator sets *top-level* keys
    it owns directly (``blacklist``, ``per_expert_max``, ``config``,
    ``blacklist_provenance``).

    :meth:`assemble` validates that the union covers
    :data:`REQUIRED_BLACKLIST_TOP_LEVEL_KEYS` exactly — extra keys are
    rejected too, to keep the Stage 2 reader contract from silently drifting.

    Design choices
    --------------
    * ``required_keys`` defaulting to :data:`REQUIRED_BLACKLIST_TOP_LEVEL_KEYS`
      — Stage 1's blacklist is the primary use, but the class is reusable.
      Stage 2's future artifacts can pass a different ``frozenset``.
    * **Reject extras**, not just missing — silently accepting unknown keys
      would mask a contract drift (e.g. a future plugin adding a ``foo``
      field that downstream loaders don't know about).
    * Duplicate-add raises — protects against two plugins both claiming the
      same slot.
    * Plain dict output, not a custom wrapper — the caller passes the result
      straight to ``save_json_artifact``, which expects a plain dict.
    * No JSON sanitisation here — :func:`safe_float` happens at fragment-
      construction time (inside each plugin's ``contribute_artifact``). The
      assembler is a pure dict-merger; it must not silently convert values,
      or it would mask bugs where a plugin forgets to sanitise.
    """

    def __init__(self) -> None:
        self._fragments: dict[str, dict[str, Any]] = {}
        self._top_level: dict[str, Any] = {}

    def add_fragment(self, name: str, fragment: dict[str, Any]) -> None:
        """Register a plugin fragment under a top-level key ``name``.

        ``name`` becomes a top-level key in the assembled dict; ``fragment``
        is its value (typically a sub-dict like the legacy ``aimer_payload``).
        """
        if not isinstance(name, str) or not name:
            raise ValueError(f"fragment name must be a non-empty string; got {name!r}")
        if not isinstance(fragment, dict):
            raise TypeError(
                f"fragment {name!r} must be a dict, got {type(fragment).__name__}"
            )
        if name in self._fragments or name in self._top_level:
            raise KeyError(f"Artifact key {name!r} already added")
        self._fragments[name] = fragment

    def set_top_level(self, key: str, value: Any) -> None:
        """Register an orchestrator-owned top-level key (e.g. ``blacklist``)."""
        if not isinstance(key, str) or not key:
            raise ValueError(f"top-level key must be a non-empty string; got {key!r}")
        if key in self._fragments or key in self._top_level:
            raise KeyError(f"Artifact key {key!r} already added")
        self._top_level[key] = value

    def assemble(
        self,
        *,
        required_keys: frozenset[str] = REQUIRED_BLACKLIST_TOP_LEVEL_KEYS,
    ) -> dict[str, Any]:
        """Build the final artifact dict, validating top-level keys against ``required_keys``.

        Raises
        ------
        ValueError
            If the union of fragment names and top-level keys is not exactly
            ``required_keys`` (missing OR extra). The error message lists the
            symmetric difference.
        """
        produced = set(self._fragments) | set(self._top_level)
        missing = required_keys - produced
        extra = produced - required_keys
        if missing or extra:
            parts = []
            if missing:
                parts.append(f"missing: {sorted(missing)}")
            if extra:
                parts.append(f"extra: {sorted(extra)}")
            raise ValueError(
                "ArtifactBuilder schema mismatch (" + "; ".join(parts) + ")"
            )
        merged: dict[str, Any] = {}
        merged.update(self._top_level)
        merged.update(self._fragments)
        return merged
