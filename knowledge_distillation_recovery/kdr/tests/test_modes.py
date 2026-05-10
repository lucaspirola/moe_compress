"""Mode literal tests (LLR-0006)."""

from __future__ import annotations

from typing import get_args

from kdr.modes import Mode


def test_mode_has_exactly_two_values() -> None:
    """Plan locks the mode flag to binary `bf16` | `da_qad`."""
    assert set(get_args(Mode)) == {"bf16", "da_qad"}
