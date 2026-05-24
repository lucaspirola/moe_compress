"""Stage 4 context — re-export of the universal :class:`PipelineContext`.

Stages 1, 2, and 3 have no stage-local context class; they use the universal
:class:`moe_compress.pipeline.context.PipelineContext`. Stage 4 follows the
same convention. This module re-exports ``PipelineContext`` so that
intra-stage-4 modules get a stable ``moe_compress.stage4.context`` import
path. If tasks S4-2..S4-3 need a stage-4-specific context helper, it lands
here.
"""
from __future__ import annotations

from ..pipeline.context import PipelineContext

__all__ = ["PipelineContext"]
