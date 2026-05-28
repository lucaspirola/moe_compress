"""Stage 2 plugin implementations.

Each plugin owns one algorithm or one orchestration step. The plugin-architecture
refactor (tasks S2-1..S2-12) split the legacy per-layer loop body into focused
plugins — REAP scoring, cost matrix, solver, refinement, distillation, heal, and
the ``LayerMergePlugin`` merge spine; S2-12 deleted the transitional
``LegacyAdapter``.

Merge-step alternatives (selected via ``stage2_reap_ream.merge_step``):

* ``"freq_weighted"`` (default) — REAM Eq. 6 / Cerebras REAP. Implemented
  inline in :mod:`stage2.merging`.
* ``"mergemoe"`` — MergeMoE closed-form T₁=Q·P† for down_proj (Miao et al.
  arXiv:2510.14436). Math in :mod:`stage2.mergemoe`; selected via the same
  knob; no separate plugin class.
* ``"regmean"`` — RegMean closed-form W_M=(Σ G_i)⁻¹Σ G_i W_i per Linear
  (Jin et al. arXiv:2212.09849). Math in :mod:`stage2.regmean`; metadata
  / Pattern C shim in :class:`regmean_merge.RegMeanMergeStepPlugin`.
"""
