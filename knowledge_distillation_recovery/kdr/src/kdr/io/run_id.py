"""Deterministic run_id derivation (LLR-0031).

The bootstrap script computes a stable hash over the validated config + the
student repo's git SHA + the mode flag. Same inputs → same hash on different
machines, so vast.ai instances re-spun against the same job find their
prior partial checkpoints on HF Hub (``pirola/kdr-partials-{run_id}``).

Hash construction (LLR-0031 AC #4):

  ``sha256(canonical_dump(config) + b'\\x00' + sha + b'\\x00' + mode)[:16]``

The literal ``\\x00`` separator prevents the (improbable but real) class of
collisions where variable-length first-fields could absorb the prefix of the
second when concatenated. The hash is reduced to the first 16 hex chars
(64 bits of namespace) — collision-free at any conceivable kdr-job scale,
short enough to fit cleanly in a HF Hub repo ID.

Whitespace and comment changes in the source YAML do NOT change the hash
(LLR-0031 AC #1) because the canonical dump goes through Pydantic's
``model_dump_json(sort_keys=True)``, which ignores YAML formatting entirely.
Semantic changes (``total_tokens``, ``bits``, etc.) DO change the hash
(LLR-0031 AC #2).
"""

# REQ: LLR-0031

from __future__ import annotations

import hashlib

from ..config import Config

_HASH_PREFIX_HEX_CHARS = 16
_FIELD_SEPARATOR = b"\x00"


def derive_run_id(config: Config, student_repo_sha: str, mode: str) -> str:
    """Return the first 16 hex chars of the canonical-input sha256.

    Args:
        config: validated kdr ``Config`` instance (NOT raw YAML text).
        student_repo_sha: git SHA of the student source. Must be the
            full 40-char SHA or its truncated form (any string is accepted,
            but the contract is reproducibility-from-the-same-string).
        mode: ``"bf16"`` or ``"da_qad"``. Embedded in the hash so a job
            re-run under a different mode cannot accidentally inherit a
            prior mode's partials.

    Returns:
        16-char lowercase hex string. Suitable for direct use in HF Hub
        repo names (``pirola/kdr-partials-{run_id}``).
    """
    payload = (
        canonical_yaml_dump(config).encode("utf-8")
        + _FIELD_SEPARATOR
        + student_repo_sha.encode("utf-8")
        + _FIELD_SEPARATOR
        + mode.encode("utf-8")
    )
    digest = hashlib.sha256(payload).hexdigest()
    return digest[:_HASH_PREFIX_HEX_CHARS]


def canonical_yaml_dump(config: Config) -> str:
    """Deterministic JSON dump of ``config``.

    Pydantic v2's ``model_dump_json`` emits fields in class-declaration order,
    which is fixed at class definition time and stable across machines as
    long as the ``Config`` source code itself doesn't change. ``indent=None``
    produces a compact single-line dump (no whitespace dependence).

    Crucially: ``model_dump_json`` operates on the PARSED model, not the raw
    YAML text — so whitespace, comment, and key-order changes in the source
    YAML do NOT affect this output (LLR-0031 AC #1). Semantic value changes
    (``total_tokens``, ``bits``, etc.) DO change it (LLR-0031 AC #2).
    """
    return config.model_dump_json(indent=None)


__all__ = ["canonical_yaml_dump", "derive_run_id"]
