"""EoRA inputs loader — A-cov + Stage 3 originals + ranks snapshot.

Paper
-----
"EoRA: Eigenspace Low-Rank Approximation for Post-Training Compression
of LLMs" — arXiv:2410.21271. Algorithm 1 reads the input
auto-covariance ``A = X̃^T X̃`` and the factorization residual
``ΔW = W_orig − Ŵ`` (where ``Ŵ`` is the post-Stage-3 factored
expert), then computes the √Λ-scaled eigenspace projection of ΔW
and back-projects a rank-r correction.

This plugin owns the **inputs** half of the Stage 4 pipeline: load
the post-Stage-3 A-covariance (reusing the Stage 2 covariance sidecar
``_stage2_input_covariance.pt`` per D-drank-premerge-A — consumed at
:mod:`stage3.plugins.d_rank_allocate`), load the Stage 3 ranks
sidecar (``rank_map.json``) for snapshot, and set up the crash-resume
partial directory.

The compensation half (the per-(layer, expert) residual SVD and
factor widening) lives at
:mod:`stage4.plugins.eora_compensation`.

Official code
-------------
``NVlabs/EoRA`` @ commit
``6a42e2edcc7559422d14ccf79b0105b2d8a78c76`` (2026-04-21) —
github.com/NVlabs/EoRA. Reference implementation for the
√Λ-eigenspace projection + correction.

Naming-history note
-------------------
"S4-2" / "load_eora_inputs phase" are stage-naming-historical. The
current plugin architecture has no phase taxonomy; new prose drops
the labels. Existing log lines / Trackio keys preserved.

Original module header retained:

S4-2 deviates from the verbatim-relocation pattern of S3-2/S3-3. The three
pieces this plugin covers — the A-cov / originals load, the file-deleted
double-widen guard, and the ``stage3_ranks`` snapshot — are NOT standalone
functions in the ``stage4_eora.py`` monolith; they are INLINE statements in
the monolith's ``run()`` body (lines ~47-109). There is therefore nothing to
relocate and re-import.

Consequences (the accepted S3-4/S3-5 pattern):

(i)  The monolith ``stage4_eora.py`` is NOT modified by S4-2. The
     ``load_eora_inputs`` hook below REPRODUCES the monolith's inline
     ``run()`` input-load logic rather than relocating it. This is an
     intentional, temporary logic duplication — the same pattern S3-4's
     ``select_alpha`` hook used for the monolith's α-dispatch. S4-4 deletes
     the monolith ``run()`` and wires this hook live; the duplication
     resolves at that point. Until then the monolith stays byte-identical,
     so the S4-0 golden snapshot is trivially safe.

(ii) Circular-import note (mirror of ``stage3/plugins/covariance_collection``):
     this module imports only from ``...utils.*``, ``...pipeline.*`` and
     stdlib/torch — NEVER from ``stage4_eora`` or ``stage4.orchestrator`` at
     any scope (module-top or function-local). A module-top
     ``from ..stage4_eora import ...`` would risk an import cycle once S4-4
     wires the orchestrator to import this module; nothing here does that.

(iii) ``EoraInputsPlugin`` is registered-but-INERT at S4-2 — no orchestrator
      walk or test invokes its ``load_eora_inputs`` hook. S4-4 plugs the hook
      into the live Stage 4 plugin sequencer.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from ...utils.model_io import MATRIX_NAMES, FactoredExperts, iter_moe_layers
from ...utils.trackio_log import trackio_log as _trackio_log
from ...pipeline.context import PipelineContext

log = logging.getLogger(__name__)


class EoraInputsPlugin:
    """Stage 4 EoRA input-load plugin (S4-2 — registered-but-INERT).

    Owns the EoRA ``load_eora_inputs`` phase: loading the Stage-2 input
    activation covariance (``_stage2_input_covariance.pt``, isotropic
    fallback when absent), loading the Stage-3 original expert weights
    (``_stage3_original_weights.pt``) behind the file-deleted double-widen
    guard, building the MoE-layer list, setting up the crash-resume partial
    directory, and snapshotting the Stage-3 per-matrix ranks before any
    widening occurs.

    Unlike S3-2/S3-3, S4-2 relocates no standalone function — the covered
    logic is inline in the ``stage4_eora.py`` monolith ``run()``. The
    ``load_eora_inputs`` hook below reproduces that inline logic faithfully;
    the monolith is NOT modified (see module docstring). S4-2 wires this
    class into the plugin registry as metadata only — no walk or test
    invokes ``load_eora_inputs``. S4-4 plugs the hook into the live Stage 4
    plugin sequencer and deletes the monolith.
    """

    name = "eora_inputs"
    paper = (
        "EoRA inputs loader (Algorithm 1 prereqs) — arXiv:2410.21271 "
        "(NVlabs/EoRA @ 6a42e2edcc7559422d14ccf79b0105b2d8a78c76). "
        "Loads post-Stage-3 A-cov + Stage 3 originals + ranks snapshot. "
        "See module docstring."
    )
    config_key = "stage4_eora.compensation_budget_pct"
    reads: tuple[str, ...] = (
        "model", "config", "artifacts_dir",
    )
    writes: tuple[str, ...] = (
        "A_cov", "a_storage_dtype", "originals", "layers",
        "partial_dir", "stage3_ranks",
    )
    # Empty: EoraInputsPlugin needs no calibration pass — every Stage 4 input
    # (A-covariance, Stage-3 originals) is a precomputed on-disk artifact.
    provides: tuple[str, ...] = ()

    def is_enabled(self, config: dict) -> bool:
        """Always True — the EoRA input load is UNCONDITIONAL.

        Every Stage 4 run must load the A-covariance and the Stage-3
        originals before any residual compensation can happen, so this phase
        always runs. ``config_key`` only parametrises the per-matrix
        compensation budget downstream; it never gates this plugin.
        """
        return True

    def contribute_artifact(self, ctx: Any) -> dict:
        return {}

    def load_eora_inputs(self, ctx: PipelineContext) -> None:
        """Phase hook — EoRA input load (S4-4 wiring surface).

        INERT at S4-2: no orchestrator walk or test invokes this hook. S4-4
        replaces the Stage 4 orchestrator body with the plugin sequencer and
        dispatches this hook in place of the monolith's inline ``run()``
        input-load block (``stage4_eora.py`` lines ~47-109). The body below
        reproduces that block faithfully — it is dead code at S4-2 but S4-4
        relies on it once the monolith is deleted.

        Reproduces (in monolith order): resolve ``a_storage_dtype`` from the
        Stage-2 config, load the A-covariance with the verbatim isotropic
        fallback warning, load the Stage-3 originals behind the verbatim
        file-deleted double-widen guard, build the MoE-layer list, set up the
        crash-resume partial directory, and snapshot the Stage-3 per-matrix
        ranks. Does NOT include the in-process per-matrix double-widen
        ``assert`` — that belongs to the ``compensate_layer`` phase (S4-3).
        """
        # Required slots — direct get(): a missing one is a wiring bug and
        # SHOULD raise. Optional slots are has()-guarded (get() raises
        # KeyError on an unset slot).
        artifacts_dir = ctx.get("artifacts_dir")
        config = ctx.get("config")
        model = ctx.get("model")
        no_resume = ctx.get("no_resume") if ctx.has("no_resume") else False

        # A-cov was persisted by Stage 2 in this storage dtype; the eigh
        # threshold in `_compute_eora_factors` must be tuned to that dtype's
        # quantization noise floor or it will keep noise-inflated directions.
        s2 = config.get("stage2_reap_ream", {})
        a_storage_dtype = getattr(
            torch, s2.get("covariance_storage_dtype", "float16")
        )
        A_cov_path = artifacts_dir / "_stage2_input_covariance.pt"
        A_cov: dict = {}
        if A_cov_path.exists():
            A_cov = torch.load(A_cov_path, map_location="cpu").get("covariance", {})
        else:
            log.warning("Stage 4: no Stage 2 covariance at %s — isotropic fallback",
                        A_cov_path)

        originals_path = artifacts_dir / "_stage3_original_weights.pt"
        if not originals_path.exists():
            # If Stage 4 already completed, the originals are intentionally
            # deleted. Re-entering Stage 4 on an already-widened model is a
            # double-widen.
            if (artifacts_dir / "stage4_eora" / "eora_ranks.json").exists():
                raise AssertionError(
                    "Stage 4 double-widen detected: _stage3_original_weights.pt was deleted "
                    "after a prior successful Stage 4 run. "
                    "widen_rank() has already been applied to this model."
                )
            raise FileNotFoundError(
                f"Stage 4 requires Stage 3 original weights at {originals_path}. "
                "Re-run Stage 3 first."
            )
        originals: dict = torch.load(originals_path, map_location="cpu")

        layers = list(iter_moe_layers(model))
        log.info("Stage 4: EoRA residual compensation over %d MoE layers", len(layers))

        # One-shot Trackio emit: run-level shape constants. Both values are
        # already computed from the model and used in the log.info above.
        _n_experts_first = (
            layers[0].experts_module.num_experts
            if layers and hasattr(layers[0].experts_module, "num_experts")
            else 0
        )
        _trackio_log({
            "stage4/config/n_moe_layers": len(layers),
            "stage4/config/n_experts_per_layer": int(_n_experts_first),
            "stage4/config/no_resume": bool(no_resume),
        })

        if no_resume:
            partial_dir = None
        else:
            partial_dir = artifacts_dir / "_stage4_partial"
            partial_dir.mkdir(parents=True, exist_ok=True)
            for _stale in partial_dir.glob("*.tmp"):
                _stale.unlink(missing_ok=True)

        # Snapshot Stage 3 ranks before any widening occurs.
        # Used by the per-matrix double-widen guard (S4-3) to detect
        # in-process re-runs.
        stage3_ranks: dict[int, dict[str, int]] = {}
        for ref in layers:
            fe = ref.experts_module
            if isinstance(fe, FactoredExperts):
                stage3_ranks[ref.layer_idx] = {
                    name: fe.ranks[name] for name in MATRIX_NAMES
                }

        ctx.set("A_cov", A_cov)
        ctx.set("a_storage_dtype", a_storage_dtype)
        ctx.set("originals", originals)
        ctx.set("layers", layers)
        ctx.set("partial_dir", partial_dir)
        ctx.set("stage3_ranks", stage3_ranks)
