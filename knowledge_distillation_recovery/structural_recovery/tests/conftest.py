"""Shared test setup.

Adds the sibling ``max_quality/src`` to ``sys.path`` so tests can import
``moe_compress.utils.model_io.FactoredExperts`` etc. without requiring
``pip install -e ../../max_quality``.

If max_quality is not discoverable, tests that depend on it (currently
``test_param_groups.py``) are SKIPPED with an actionable message — pure-math
tests in ``test_kld_loss.py`` and ``test_defensive.py`` continue to run.

Override the auto-discovered path with the env var
``MOE_COMPRESS_MAX_QUALITY=<path-to-max_quality/src>``.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


_HERE = Path(__file__).resolve().parent
_PKG_ROOT = _HERE.parent                                  # structural_recovery/
_REPO_ROOT = _PKG_ROOT.parent.parent                      # moe_compress/
_PKG_SRC = _PKG_ROOT / "src"

# Always make this package importable.
if str(_PKG_SRC) not in sys.path and _PKG_SRC.is_dir():
    sys.path.insert(0, str(_PKG_SRC))

# Resolve max_quality src: env var override > sibling auto-discovery.
_MQ_ENV = os.environ.get("MOE_COMPRESS_MAX_QUALITY")
_MAX_QUALITY_SRC = Path(_MQ_ENV) if _MQ_ENV else (_REPO_ROOT / "max_quality" / "src")

# Try to make max_quality importable; record success/failure for the skip hook.
_MAX_QUALITY_AVAILABLE: bool = False
_MAX_QUALITY_REASON: str = ""

if _MAX_QUALITY_SRC.is_dir():
    s = str(_MAX_QUALITY_SRC)
    if s not in sys.path:
        sys.path.insert(0, s)
    try:
        import moe_compress.utils.model_io  # noqa: F401
        _MAX_QUALITY_AVAILABLE = True
    except ImportError as err:
        _MAX_QUALITY_REASON = (
            f"max_quality src found at {_MAX_QUALITY_SRC} but import failed: {err}"
        )
else:
    _MAX_QUALITY_REASON = (
        f"max_quality src not found at {_MAX_QUALITY_SRC}. "
        "Either clone moe_compress next to this repo, or set "
        "MOE_COMPRESS_MAX_QUALITY=<path-to-max_quality/src> before running pytest."
    )


# Test files whose tests need max_quality. Anything else runs unconditionally.
_DEPS_ON_MAX_QUALITY = {"test_param_groups.py"}


def pytest_collection_modifyitems(config, items):
    """Skip max_quality-dependent tests with an actionable reason."""
    if _MAX_QUALITY_AVAILABLE:
        return
    skip_marker = pytest.mark.skip(reason=_MAX_QUALITY_REASON)
    for item in items:
        if Path(item.fspath).name in _DEPS_ON_MAX_QUALITY:
            item.add_marker(skip_marker)
