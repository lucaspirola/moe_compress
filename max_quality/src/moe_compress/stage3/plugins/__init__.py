"""Stage 3 plugin implementations.

Plugins exposed by this package, in canonical Stage 3 schedule order
(``stage3/orchestrator.py`` builds the :class:`PluginRegistry` from this
sequence):

  1. :class:`covariance_collection.CovarianceCollectionPlugin` —
     ``collect_covariances`` phase. AA-SVD B + cross-covariance C.
  2. :class:`wanda_intra_expert_score.WandaIntraExpertScorePlugin` —
     ``collect_wanda_scores`` phase. Routing-weighted Wanda intra-expert
     importance score; opt-in via ``stage3.wanda_intra_expert.enabled``
     (default OFF). Runs AFTER covariance_collection (reuses the same
     calibration-pass-shape but with its own sweep — see D-zero-extra-
     forward in that module's docstring) and BEFORE allocate_ranks so
     downstream allocators can consult the score.
  3. :class:`d_rank_allocate.DRankAllocatePlugin` — ``allocate_ranks``.
  4. :class:`swift_svd_alpha.SwiftSvdAlphaPlugin` — ``select_alpha``.
  5. :class:`aa_svd_factor.AaSvdFactorPlugin` — ``factor_layer``.
  6. :class:`block_refine.BlockRefinePlugin` — ``refine_blocks``.

Cache providers (registered alongside the algorithmic plugins, hit via
``PluginRegistry.dispatch_first``):

  * :class:`input_cov_cache.Stage3InputCovCacheProvider` — V2 input-cov
    sidecar.
  * :class:`block_hidden_cache.Stage3BlockHiddenCacheProvider` —
    block-hidden teacher targets cache for Phase C.5.

The original S3-1 scaffold docstring noted "no plugin manifest exists
yet". The manifest now lives in :mod:`stage3.orchestrator` (the
:class:`PluginRegistry` construction in :func:`stage3.orchestrator.run`).
This module-level docstring mirrors that order so a reader of just the
plugins package sees the canonical schedule without having to read the
orchestrator.
"""
