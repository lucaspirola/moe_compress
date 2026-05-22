"""Router-KD plugin implementations.

Holds ``trainable_scope.py`` (added by RK-2 — the trainable/frozen-parameter
scope concern) and ``kd_optimizer.py`` (added by RK-3 — the optimizer +
LR-scheduler concern). The unified Router-KD algorithm — the KD training loop
serving both Stage 2.5 and Stage 5 — is extracted from the legacy
``stage5_router_kd.py`` monolith into focused plugins here; the remaining
Router-KD plugins land in this package by tasks RK-4..RK-7.
"""
