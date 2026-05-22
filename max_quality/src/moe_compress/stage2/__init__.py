"""Stage-2 plugin package.

This package holds the stage-2 (REAP/REAM expert-merging) pipeline as it is
migrated onto the universal plugin interface.

Migration status:
  - S2-1: ported the stage-2 algorithm-helper modules into this package.
  - S2-2 .. S2-13: progressively migrate plugins and the orchestrator.
  - S2-4: retired the stage-2-specific pipeline shell; the orchestrator now
    drives plugins through :func:`moe_compress.tools.phase_walker.walk_phases`
    over a :class:`moe_compress.pipeline.registry.PluginRegistry`, the same
    universal wiring every other stage uses.

This module is intentionally INERT: it performs no imports and re-exports
nothing until the migration wires the orchestrator entrypoint in S2-13.
"""
