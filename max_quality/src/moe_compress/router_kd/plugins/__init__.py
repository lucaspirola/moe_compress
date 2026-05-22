"""Router-KD plugin implementations.

Holds ``trainable_scope.py`` (added by RK-2 — the trainable/frozen-parameter
scope concern), ``kd_optimizer.py`` (added by RK-3 — the optimizer +
LR-scheduler concern), ``vocab_kd.py`` (added by RK-4 — the KD-loss
concern: the chunked vocab-KL kernel, the loss combiner and the NaN sanity
probes) and ``teacher.py`` (added by RK-5 — the teacher-logits concern:
``TeacherCachePlugin`` and ``TeacherLivePlugin``, two plugins sharing the
``provide_teacher_logits`` slot hook — cache wins on a hit, live teacher is
the universal fallback). The unified Router-KD algorithm — the KD training
loop serving both Stage 2.5 and Stage 5 — is extracted from the legacy
``stage5_router_kd.py`` monolith into focused plugins here; the remaining
Router-KD plugins land in this package by tasks RK-6..RK-7.
"""
