"""Shared test setup.

Adds the sibling ``max_quality/src`` to ``sys.path`` so tests can import
``moe_compress.utils.model_io.FactoredExperts`` etc. without requiring
``pip install -e ../../max_quality``.
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PKG_ROOT = _HERE.parent                                  # structural_recovery/
_REPO_ROOT = _PKG_ROOT.parent.parent                      # moe_compress/
_MAX_QUALITY_SRC = _REPO_ROOT / "max_quality" / "src"
_PKG_SRC = _PKG_ROOT / "src"

for p in (_MAX_QUALITY_SRC, _PKG_SRC):
    s = str(p)
    if p.is_dir() and s not in sys.path:
        sys.path.insert(0, s)
