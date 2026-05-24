"""Stage 2 — REAP scoring + REAM expert merging (plugin architecture)."""
from .orchestrator import run
from .stage import STAGE2

__all__ = ["run", "STAGE2"]
