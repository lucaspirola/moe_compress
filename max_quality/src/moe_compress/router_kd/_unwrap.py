"""Single source of truth for peeling training-student wrappers.

DDP wraps the student in ``.module``; ``torch.compile`` wraps it in
``_orig_mod``. The codebase already peels ``_orig_mod`` at every save /
``named_parameters()`` / ``iter_moe_layers`` site for the STUDENT; DDP adds a
second wrapper. When DDP wraps a compiled student the ordering is
``DDP(compile(model))`` — so the unwrap must peel BOTH (``module`` then
``_orig_mod``), in any order. This one helper replaces the ad-hoc
``getattr(student, "_orig_mod", student)`` at every TRAINING-STUDENT site so no
site is missed (the teacher / pre-wrap ``_set_experts_implementation`` are NOT
DDP-wrapped and keep their plain getattr).
"""
from __future__ import annotations


def unwrap_student(model):
    """Peel DDP (``.module``) and torch.compile (``._orig_mod``) wrappers, in
    any nesting order, to the underlying ``nn.Module``. Idempotent on a bare
    module (returns it unchanged)."""
    seen: set[int] = set()
    while True:
        nxt = getattr(model, "module", None)
        if nxt is None:
            nxt = getattr(model, "_orig_mod", None)
        if nxt is None or id(nxt) in seen:
            return model
        seen.add(id(model))
        model = nxt


__all__ = ["unwrap_student"]
