"""Stage 1 typed-context subclass.

Inherits :class:`PipelineContext` verbatim. Exists so plugins under
``stage1/plugins/`` can annotate ``ctx: Stage1Context`` for type-checker
narrowing, and so the orchestrator (sub-task 10) has a single import target
for the stage's shared-state holder.

Slots populated across sub-tasks 3-10 (forward-looking; not all live yet):

- ``model`` (Phase A, sub-task 9 onward)
- ``tokenizer`` (Phase A, sub-task 9 onward)
- ``config`` (Phase F today, sub-task 3)
- ``artifacts_dir`` (orchestrator, sub-task 10)
- ``moe_layers`` (Phase F today, sub-task 3)
- ``decomposition`` (Phase F today, sub-task 3)
- ``L``, ``residual_growth``, ``moe_output_growth``, ``moe_output_max``
  (Phase A, sub-task 9)
- ``max_acc``, ``output_acc``, ``sink_acc`` (Phase B accumulators,
  sub-tasks 6-8)
- ``candidates`` (Phase C, sub-tasks 6-8)
- ``blacklist`` (Phase D today + Phase F today, sub-task 3 + sub-task 5)
- ``candidate_deltas``, ``baseline_nll`` (Phase D, sub-task 5)
- ``D_matrices`` (Phase E today, sub-task 3 + sub-task 4)
- ``per_layer_targets`` (sub-task 3) — layer→total-expert-count map
- ``per_layer_target_experts``, ``per_layer_redundancy``,
  ``achieved_budget``, ``requested_budget``, ``grape_config`` (Phase F
  writes, sub-task 3)

No typed properties are added in this sub-task. All access goes through
:meth:`PipelineContext.get` / :meth:`PipelineContext.set`. A future
sub-task may promote frequently-read slots to ``@property`` helpers.
"""

from __future__ import annotations

from ..pipeline.context import PipelineContext


class Stage1Context(PipelineContext):
    """Stage-1-specific :class:`PipelineContext`.

    Today this is an empty subclass: every read/write goes through the
    base class. The class exists for two purposes:

    1. **Stable import target.** Every plugin under ``stage1/plugins/``
       and the future orchestrator imports ``Stage1Context`` — not
       ``PipelineContext`` directly — so a future stage-specific
       invariant (e.g. slot-completeness check at end of ``run()``)
       has a single seam.
    2. **Type-checker narrowing.** Plugin signatures
       ``def run(self, ctx: Stage1Context) -> None`` give IDEs / mypy
       Stage-1-specific narrowing without leaking ``Any`` from the
       framework Protocol's ``ctx: "Any"`` shape.
    """
