"""Stage-2 plugin package.

This package holds the stage-2 (REAP/REAM expert-merging) pipeline as it is
migrated onto the universal plugin interface.

Migration status:
  - S2-1: ported the stage-2 framework + algorithm-helper modules into this
    package (framework lives in the temporary private ``_framework/`` subpackage;
    helpers live directly here).
  - S2-2 .. S2-13: progressively migrate plugins and the orchestrator.

This module is intentionally INERT: it performs no imports and re-exports
nothing until the migration wires the orchestrator entrypoint in S2-13.
"""
