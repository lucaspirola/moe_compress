"""Stage 3 context — re-export of the universal :class:`PipelineContext`.

Stages 1 and 2 have no stage-local context class; they use the universal
:class:`moe_compress.pipeline.context.PipelineContext`. Stage 3 follows the
same convention. This module re-exports ``PipelineContext`` so that
intra-stage-3 modules get a stable ``moe_compress.stage3.context`` import
path. If tasks S3-2..S3-6 need a stage-3-specific context helper, it lands
here.
"""
from __future__ import annotations

from ..pipeline.context import PipelineContext

__all__ = ["PipelineContext"]
