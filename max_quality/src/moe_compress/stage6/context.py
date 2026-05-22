"""Stage 6 context — re-export of the universal :class:`PipelineContext`.

Stages 1–5 have no stage-local context class; they use the universal
:class:`moe_compress.pipeline.context.PipelineContext`. Stage 6 follows the
same convention. This module re-exports ``PipelineContext`` so that
intra-stage-6 modules get a stable ``moe_compress.stage6.context`` import
path. If tasks S6-2..S6-7 need a stage-6-specific context helper, it lands
here.
"""
from __future__ import annotations

from ..pipeline.context import PipelineContext

__all__ = ["PipelineContext"]
