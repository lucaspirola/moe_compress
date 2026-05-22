"""Stage 2 pipeline substrate.

Public surface for the refactor: the per-layer context carrier and the
pipeline shell. Concrete plugins live under ``plugins/`` and conform to the
universal :class:`~moe_compress.pipeline.plugin.PipelinePlugin` Protocol
structurally — there is no stage-2-specific plugin base class or registry.
"""
from __future__ import annotations

from ...pipeline.context import PipelineContext
from .pipeline import Stage2Pipeline

__all__ = [
    "PipelineContext",
    "Stage2Pipeline",
]
