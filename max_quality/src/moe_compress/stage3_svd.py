"""Stage 3 — Non-uniform SVD, fused-experts-aware (legacy entry-point shim).

S3-7a retired the ~570-line monolith ``run()`` that used to live here. The
REAL Stage 3 orchestration now lives in
:func:`moe_compress.stage3.orchestrator.run` — a ``PipelineContext`` +
``PluginRegistry`` driving the five stage-3 plugins through the schedule
``collect_covariances → allocate_ranks → select_alpha →
LOOP layers[factor_layer] → refine_blocks → finalize``.

This module now serves two purposes only:

1. ``stage3_svd.run`` is a thin shim delegating to the orchestrator — the
   stable legacy entry point (``run_pipeline.py``, the golden / smoke tests
   call it) and the ``monkeypatch`` surface for ``build_calibration_tensor``
   / ``save_compressed_checkpoint`` / ``load_model``.
2. The S3-2..S3-6 ``# noqa: F401`` re-import blocks keep the relocated
   algorithm symbols (``_aa_svd``, ``_cov_lookup``, ``_NOISE_FLOOR_BY_DTYPE``,
   ``_collect_pruned_input_covariance``, …) resolvable from this module —
   external callers (``stage4_eora``) and tests still import them here.

The pipeline itself: each MoE layer arrives with a fused experts module
post-prune; the orchestrator computes D-Rank group stats, allocates ranks
to the global ``T_budget``, runs Swift-SVD+ α selection, factors every
expert via AA-SVD into a :class:`FactoredExperts`, and optionally runs the
Phase C.5 block-level joint refinement.
"""
from __future__ import annotations

import logging
from pathlib import Path

from .budget.solver import BudgetDecomposition

# S3-7a: the monolith ``run()`` body was deleted — ``stage3_svd.run`` is now a
# thin shim over ``stage3.orchestrator.run``. These three names are kept
# imported NOT because the shim ``run()`` uses them directly, but because the
# golden / smoke tests ``monkeypatch.setattr`` them on THIS module object
# (``stage3_svd.build_calibration_tensor`` / ``.save_compressed_checkpoint`` /
# ``.load_model``). The orchestrator calls ``load_model`` module-qualified
# through this module so the patch is honoured (orchestrator HAZARD H3).
from .utils.calibration import build_calibration_tensor  # noqa: F401
from .utils.model_io import (  # noqa: F401
    load_model,
    save_compressed_checkpoint,
)

log = logging.getLogger(__name__)

# S3-2: covariance collection relocated to stage3/plugins/covariance_collection.
# Re-imported so run() + external callers/tests keep their import paths.
from .stage3.plugins.covariance_collection import (  # noqa: F401
    _collect_covariances,
    _collect_pruned_input_covariance,
    _load_stage2_covariance,
)

# S3-3: D-Rank group-stats + rank allocation relocated to
# stage3/plugins/d_rank_allocate. Re-imported so run() keeps its import paths.
from .stage3.plugins.d_rank_allocate import (  # noqa: F401
    _GroupStats,
    _group_stat,
    _pad,
    _compute_T_budget,
    _d_rank_allocate,
)

# S3-4: Swift-SVD+ alpha-search (both searches), rank redistribution,
# snapshot/restore + WikiText-2 PPL validation relocated to
# stage3/plugins/swift_svd_alpha. Re-imported so run() keeps its paths.
from .stage3.plugins.swift_svd_alpha import (  # noqa: F401
    _snapshot_originals,
    _build_wikitext2_validation,
    _evaluate_wikitext2_ppl,
    _factor_model_at_ranks,
    _restore_fused_experts,
    _swift_svd_plus_alpha_search_validation,
    _swift_svd_plus_alpha_search,
    _redistribute_ranks_swift_svd_plus,
)

# S3-5: AA-SVD core (eigendecomposition cache, rank-k factorization, the
# per-bank covariance lookup, the storage-dtype noise-floor table) relocated
# to stage3/plugins/aa_svd_factor. Re-imported so run() + external callers/
# tests + S3-4's swift_svd_alpha lazy import keep their stage3_svd paths.
from .stage3.plugins.aa_svd_factor import (  # noqa: F401
    _NOISE_FLOOR_BY_DTYPE,
    _EighDecomp,
    _precompute_eigh,
    _aa_svd_precomputed,
    _aa_svd,
    _cov_lookup,
)

# S3-6: Phase C.5 block-level joint refinement (_phase_c5_block_refine +
# _advance_streams) relocated to stage3/plugins/block_refine. Re-imported so
# run() keeps its import path. This is the LAST stage-3 plugin extraction.
from .stage3.plugins.block_refine import (  # noqa: F401
    _phase_c5_block_refine,
    _advance_streams,
)


def run(
    model,
    tokenizer,
    config: dict,
    artifacts_dir: Path,
    decomposition: BudgetDecomposition,
    *,
    device=None,
    no_resume: bool = False,
) -> Path:
    """Run Stage 3 — thin shim delegating to the plugin-driven orchestrator.

    S3-7a flipped the relationship: the REAL Stage 3 orchestration now lives
    in :func:`moe_compress.stage3.orchestrator.run` (a ``PipelineContext`` +
    ``PluginRegistry`` driving the five stage-3 plugins through the phase
    schedule). This module retains ``stage3_svd.run`` only as the stable
    legacy entry point — ``run_pipeline.py`` and the golden / smoke tests
    still call ``stage3_svd.run`` — and as the patch surface for
    ``build_calibration_tensor`` / ``save_compressed_checkpoint`` /
    ``load_model`` (the orchestrator calls those module-qualified through the
    modules the tests patch; see the orchestrator's HAZARD H3 note).

    The import of the orchestrator is function-local: the orchestrator does a
    function-local ``from .. import stage3_svd`` of its own (it needs this
    module's object so a monkeypatched ``load_model`` is honoured), so a
    module-top import here would close the cycle.
    """
    from .stage3.orchestrator import run as _orchestrator_run
    return _orchestrator_run(
        model, tokenizer, config, artifacts_dir, decomposition,
        device=device, no_resume=no_resume,
    )


# ---------------------------------------------------------------------------
# Group stats + allocations
# ---------------------------------------------------------------------------
# S3-3: ``_GroupStats`` / ``_group_stat`` / ``_pad`` / ``_compute_T_budget`` /
# ``_d_rank_allocate`` relocated to ``stage3/plugins/d_rank_allocate`` and
# re-imported above (see the S3-3 ``# noqa: F401`` block).


# S3-4: Swift-SVD+ alpha-search (both searches), rank redistribution,
# snapshot/restore + WikiText-2 PPL validation relocated to
# ``stage3/plugins/swift_svd_alpha``. They are re-imported in the
# top-of-module import block above (see the S3-4 ``# noqa: F401`` block).


# ---------------------------------------------------------------------------
# AA-SVD per matrix
# ---------------------------------------------------------------------------
# S3-5: ``_NOISE_FLOOR_BY_DTYPE`` / ``_EighDecomp`` / ``_precompute_eigh`` /
# ``_aa_svd_precomputed`` / ``_aa_svd`` / ``_cov_lookup`` relocated to
# ``stage3/plugins/aa_svd_factor`` and re-imported in the top-of-module import
# block above (see the S3-5 ``# noqa: F401`` block).


# ---------------------------------------------------------------------------
# Post-prune input covariance (for AA-SVD B matrix)
# ---------------------------------------------------------------------------
# S3-2: ``_collect_covariances`` / ``_collect_pruned_input_covariance`` /
# ``_load_stage2_covariance`` relocated to stage3/plugins/covariance_collection.
# They are re-imported in the top-of-module import block above.


# ---------------------------------------------------------------------------
# Per-matrix L-BFGS reconstruction refine
# ---------------------------------------------------------------------------


# S3-6: ``_phase_c5_block_refine`` / ``_advance_streams`` relocated to
# ``stage3/plugins/block_refine`` and re-imported in the top-of-module import
# block above (see the S3-6 ``# noqa: F401`` block). This is the LAST stage-3
# plugin extraction.


