"""Stage 2 plugin substrate.

Public surface for the refactor: contexts, plugin base class, registry, and
the pipeline shell. Concrete plugins live under ``pipeline/plugins/`` and
are added in later tasks; this module ships only the scaffolding.
"""
from __future__ import annotations

from .base import Stage2Plugin
from .context import LayerContext, RunContext
from .pipeline import Stage2Pipeline
from .registry import PluginRegistry

__all__ = [
    "LayerContext",
    "PluginRegistry",
    "RunContext",
    "Stage2Pipeline",
    "Stage2Plugin",
]
