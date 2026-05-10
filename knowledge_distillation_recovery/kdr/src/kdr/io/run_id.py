"""Deterministic run_id derivation (LLR-0031).

Stubbed in Phase 2; real implementation lands in Phase 6.
"""

from __future__ import annotations

from pathlib import Path

from ..config import Config


def derive_run_id(config: Config, student_repo_sha: str, mode: str) -> str:
    """Return the first 16 hex chars of
    `sha256(canonical_yaml_dump(config) + b'\\x00' + student_sha + b'\\x00' + mode)`.

    The `\\x00` separator prevents length-collision aliasing across variable-
    length YAML dumps. Same inputs → same hash on different machines.

    Phase 2: stub.
    """
    raise NotImplementedError("Phase 6: derive_run_id")


def canonical_yaml_dump(config: Config) -> str:
    """Pydantic `model_dump_json` of `config` — stable hash input.

    Phase 2: stub.
    """
    raise NotImplementedError("Phase 6: canonical_yaml_dump")


__all__ = ["Path", "canonical_yaml_dump", "derive_run_id"]
