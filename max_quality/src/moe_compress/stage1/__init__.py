"""Stage 1 — Super-Expert detection + GRAPE budgets (plugin architecture)."""
from .orchestrator import run
from .stage import STAGE1

__all__ = ["run", "STAGE1"]
