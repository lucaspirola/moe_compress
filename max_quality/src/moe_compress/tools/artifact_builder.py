"""``ArtifactBuilder`` — schema-parametric assembler for JSON-ready artifacts.

Plugins contribute named *fragments* (one sub-dict per plugin); the
orchestrator sets *top-level* keys it owns directly. :meth:`ArtifactBuilder.assemble`
merges them into a plain dict, validating that the union of keys exactly
matches a caller-supplied ``required_keys`` schema — rejecting both missing and
extra keys so a downstream reader contract cannot silently drift.

This is the stage-agnostic port of stage-1's ``pipeline/artifact_assembly.py``.
The one substantive change is that ``assemble`` takes ``required_keys`` as a
*mandatory* keyword-only argument — there is no built-in default schema. The
stage-1 ``REQUIRED_BLACKLIST_TOP_LEVEL_KEYS`` constant is deliberately *not*
carried over: that is a stage-1-specific schema and belongs with the stage-1
package, not in this shared tool.

Design choices
--------------
* **Mandatory ``required_keys``** — the schema-parametric change. A reusable
  assembler must not privilege one stage's schema; every caller declares its
  own ``frozenset`` of required keys explicitly. Stage 1's blacklist schema,
  Stage 2's future artifacts, etc., each pass their own.
* **Reject extras, not just missing** — silently accepting unknown keys would
  mask a contract drift (e.g. a future plugin adding a ``foo`` field that
  downstream loaders do not know about).
* **Duplicate-add raises** — protects against two plugins (or a plugin and the
  orchestrator) both claiming the same slot.
* **Plain dict output, not a custom wrapper** — the caller passes the result
  straight to ``save_json_artifact``, which expects a plain dict.
* **No JSON sanitisation here** — value sanitisation (e.g. ``safe_float``)
  happens at fragment-construction time inside each plugin. The assembler is a
  pure dict-merger; silently converting values would mask bugs where a plugin
  forgets to sanitise.
"""

from __future__ import annotations

from typing import Any


class ArtifactBuilder:
    """Builds a JSON-ready dict from per-plugin fragments + orchestrator-owned keys.

    Plugins contribute named *fragments* via :meth:`add_fragment` (each value a
    sub-dict); the orchestrator sets *top-level* keys it owns directly via
    :meth:`set_top_level`. :meth:`assemble` validates that the union of keys
    exactly matches a caller-supplied ``required_keys`` schema — extra keys are
    rejected too, to keep a downstream reader contract from silently drifting.

    Design choices
    --------------
    * ``required_keys`` is a *mandatory* keyword-only argument to
      :meth:`assemble` — the schema-parametric change versus stage-1's
      assembler. There is no built-in default schema; every caller declares
      its own ``frozenset``.
    * **Reject extras**, not just missing — silently accepting unknown keys
      would mask a contract drift.
    * Duplicate-add raises — protects against two contributors both claiming
      the same slot.
    * Plain dict output, not a custom wrapper — the caller passes the result
      straight to ``save_json_artifact``, which expects a plain dict.
    * No JSON sanitisation here — value sanitisation happens at
      fragment-construction time inside each plugin. The assembler is a pure
      dict-merger; it must not silently convert values.
    """

    def __init__(self) -> None:
        self._fragments: dict[str, dict[str, Any]] = {}
        self._top_level: dict[str, Any] = {}

    def add_fragment(self, name: str, fragment: dict[str, Any]) -> None:
        """Register a plugin fragment under a top-level key ``name``.

        ``name`` becomes a top-level key in the assembled dict; ``fragment``
        is its value (typically a sub-dict). ``name`` must be a non-empty
        string and ``fragment`` must be a dict; a name already used by a
        fragment OR a top-level key raises :class:`KeyError`.
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
        """Register an orchestrator-owned top-level key.

        ``key`` must be a non-empty string; ``value`` may be any type. A key
        already used by a fragment OR a top-level key raises :class:`KeyError`.
        """
        if not isinstance(key, str) or not key:
            raise ValueError(f"top-level key must be a non-empty string; got {key!r}")
        if key in self._fragments or key in self._top_level:
            raise KeyError(f"Artifact key {key!r} already added")
        self._top_level[key] = value

    def assemble(self, *, required_keys: frozenset[str]) -> dict[str, Any]:
        """Build the final artifact dict, validating keys against ``required_keys``.

        ``required_keys`` is a *mandatory* keyword-only argument: there is no
        default schema. The union of fragment names and top-level keys must
        equal ``required_keys`` exactly.

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


__all__ = ["ArtifactBuilder"]
