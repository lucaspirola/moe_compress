"""Stage 3 task-parallel levers — equivalence + resolution + determinism.

Two RESULT-PRESERVING, default-OFF multi-GPU levers in Stage 3:

* **Lever 2 (per-expert SVD factor)** — ``aa_svd_factor.factor_layer`` fans the
  per-expert AA-SVD solve across worker devices via the SAME EoRA concurrency
  engine (``_run_expert_bands`` / ``_resolve_worker_devices``), assembling rows
  in ascending-e on the main thread. Default ``factor_workers=1`` ⇒ inline
  serial, byte-identical.

* **Lever 1 (α-grid)** — ``swift_svd_alpha`` distributes the 11-candidate
  WikiText-2 PPL grid across process-spawn replicas; the parent argmins on a
  completion-order-independent ``(grid_idx, alpha, ppl)`` fold. Default
  ``alpha_workers`` absent ⇒ serial.

All run WITHOUT a real multi-GPU box: worker devices are CPU via the
``factor_worker_devices`` / ``eora_worker_devices`` ctx seam (no monkeypatch —
project rule). Live ≥2-GPU validation is deferred to a real box.
"""
from __future__ import annotations

import torch


# --------------------------------------------------------------------------- #
# Lever 2 — T2.0: the EoRA concurrency engine is importable for reuse.
# --------------------------------------------------------------------------- #
def test_factor_engine_imports():
    """Lever 2 reuses the landed EoRA band engine in place (Q1 option (a)).

    ``_run_expert_bands`` + ``_resolve_worker_devices`` import from
    ``stage4.plugins.eora_compensation`` — a deliberate stage3→stage4 reuse of
    the stage-agnostic concurrency engine (the EoRA file + its golden are left
    untouched; Lever 2 only IMPORTS them).
    """
    from moe_compress.stage4.plugins.eora_compensation import (
        _resolve_worker_devices,
        _run_expert_bands,
    )

    assert callable(_run_expert_bands)
    assert callable(_resolve_worker_devices)
