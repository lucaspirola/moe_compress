"""Stage 2 plugin implementations.

Each plugin owns one algorithm or one orchestration step. T6 ships a single
``LegacyAdapter`` that holds the legacy per-layer loop body verbatim; tasks
T7–T17 split it into focused, real plugins (REAP scoring, cost matrix, solver,
distillation, heal, etc.); T18 deletes the adapter and the env-var hatch.
"""
