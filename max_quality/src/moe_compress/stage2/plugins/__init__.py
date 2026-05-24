"""Stage 2 plugin implementations.

Each plugin owns one algorithm or one orchestration step. The plugin-architecture
refactor (tasks S2-1..S2-12) split the legacy per-layer loop body into focused
plugins — REAP scoring, cost matrix, solver, refinement, distillation, heal, and
the ``LayerMergePlugin`` merge spine; S2-12 deleted the transitional
``LegacyAdapter``.
"""
