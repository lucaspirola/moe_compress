"""Router-KD plugin implementations.

Holds ``trainable_scope.py`` (added by RK-2 — the trainable/frozen-parameter
scope concern), ``kd_optimizer.py`` (added by RK-3 — the optimizer +
LR-scheduler concern), ``vocab_kd.py`` (added by RK-4 — the KD-loss
concern: the chunked vocab-KL kernel, the loss combiner and the NaN sanity
probes), ``teacher.py`` (added by RK-5 — the teacher-logits concern:
``TeacherCachePlugin`` and ``TeacherLivePlugin``, two plugins sharing the
``provide_teacher_logits`` slot hook — cache wins on a hit, live teacher is
the universal fallback) ``merge_repair.py`` (added by RK-6 — the
Direction-E Stage-2.5 merge-repair concern: the merge-map loader, merged-
centroid identification, the centroid unfreeze + grad-mask, the MoE-block
output capture and the per-layer MSE term, plus ``MergeRepairPlugin`` with a
stage-2.5-gated ``is_enabled``) and ``early_stop.py`` (added by RK-7 — the
best-tracker + early-stop concern: the atomic ``best.pt`` writer
``_save_best_router_state``, plus ``EarlyStopPlugin`` reproducing the inline
``run()`` best-tracker / early-stop glue — the EMA-smoothed raw-KL tracking,
the save-on-improvement, the patience-based early-stop decision and the
end-of-training best-checkpoint reload — in four inert hooks, with an
unconditional ``is_enabled``). The unified Router-KD algorithm — the KD
training loop serving both Stage 2.5 and Stage 5 — is extracted from the
legacy ``stage5_router_kd.py`` monolith into focused plugins here; RK-7 is
the last Router-KD plugin extraction.
"""
