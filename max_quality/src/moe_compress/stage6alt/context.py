"""Stage 6alt context — re-export of the universal :class:`PipelineContext`.

Stages 1–5 have no stage-local context class; they use the universal
:class:`moe_compress.pipeline.context.PipelineContext`. Stage 6alt follows
the same convention. This module re-exports ``PipelineContext`` so that
intra-stage-6alt modules get a stable ``moe_compress.stage6alt.context``
import path. If tasks S6A-2..S6A-5 need a stage-6alt-specific context
helper, it lands here.
"""
from __future__ import annotations

from ..pipeline.context import PipelineContext

__all__ = ["PipelineContext"]
