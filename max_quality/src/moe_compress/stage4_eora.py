"""Stage 4 — EoRA residual compensation (legacy entry-point shim).

S4-4a retired the monolith ``run()`` that used to live here. The REAL Stage 4
orchestration now lives in :func:`moe_compress.stage4.orchestrator.run` — a
``PipelineContext`` + ``PluginRegistry`` driving the two stage-4 plugins
through the schedule ``load_eora_inputs → LOOP layers[compensate_layer] →
finalize``.

EoRA itself: for each (layer, expert, matrix) factored in Stage 3, compute
the residual ΔW_e = W_orig_e − U_e @ V_e, project it through the **√Λ-scaled
eigenspace** of the input activation covariance (paper 2410.21271 Algorithm 1,
step 3: Q' = Q·√Λ), take a rank-r SVD of the *full* projected error
ΔW' = ΔW·Q', back-project via V_corr = V'^T · (√Λ)^{-1} · Q^T and **widen**
the corresponding ``FactoredExperts`` U / V along the rank dim.

This module now serves two purposes only:

1. ``stage4_eora.run`` is a thin shim delegating to the orchestrator — the
   stable legacy entry point (``run_pipeline.py``, the golden / smoke tests
   call it).
2. The S4-3 ``# noqa: F401`` re-import block keeps the relocated algorithm
   symbols (``_compute_eora_factors`` / ``_spill_layer``) resolvable from this
   module — external callers and tests still import them here.
"""
from __future__ import annotations

import logging
from pathlib import Path

# S4-3: the EoRA residual kernel (_compute_eora_factors) and the per-layer
# crash-resume spill (_spill_layer) relocated to stage4/plugins/eora_compensation.
# Re-imported so external callers/tests keep their stage4_eora import paths.
from .stage4.plugins.eora_compensation import (  # noqa: F401
    _compute_eora_factors,
    _spill_layer,
)

log = logging.getLogger(__name__)


def run(
    model,
    tokenizer,
    config: dict,
    artifacts_dir: Path,
    *,
    no_resume: bool = False,
) -> Path:
    """Run Stage 4 — thin shim delegating to the plugin-driven orchestrator.

    S4-4a flipped the relationship: the REAL Stage 4 orchestration now lives
    in :func:`moe_compress.stage4.orchestrator.run` (a ``PipelineContext`` +
    ``PluginRegistry`` driving the two stage-4 plugins through the schedule
    ``load_eora_inputs → LOOP layers[compensate_layer] → finalize``).
    This module retains ``stage4_eora.run`` only as the stable legacy entry
    point — ``run_pipeline.py`` and the golden / smoke tests still call
    ``stage4_eora.run``.

    The import of the orchestrator is function-local: the ``stage4/plugins``
    modules re-imported above already pull in the plugin layer, and a
    module-top ``from .stage4.orchestrator import run`` is unnecessary churn
    for a shim that is only ever called at runtime.
    """
    from .stage4.orchestrator import run as _orchestrator_run
    return _orchestrator_run(
        model, tokenizer, config, artifacts_dir, no_resume=no_resume,
    )
