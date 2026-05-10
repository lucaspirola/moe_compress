"""pytest config for kdr.

Ensures `src/kdr` is importable without an installed package (since Phase 2 is
skeleton-only and the user may not have `pip install -e .`'d yet).
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
