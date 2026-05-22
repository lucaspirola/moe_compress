"""Router-KD context — re-export of the universal :class:`PipelineContext`.

Stages 1, 2, and 3 have no stage-local context class; they use the universal
:class:`moe_compress.pipeline.context.PipelineContext`, and Stage 4 follows
the same convention. Router-KD (the unified Stage 2.5 + Stage 5 module) does
too. This module re-exports ``PipelineContext`` so that intra-router-kd
modules get a stable ``moe_compress.router_kd.context`` import path. If tasks
RK-2..RK-7 need a router-kd-specific context helper, it lands here.
"""
from __future__ import annotations

from ..pipeline.context import PipelineContext

__all__ = ["PipelineContext"]
