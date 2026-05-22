"""Stage 4 orchestrator — the real plugin-driven phase sequencer (S4-4a).

S4-1 shipped this module as a thin delegation to the legacy
``stage4_eora.run`` monolith. S4-2..S4-3 extracted the EoRA algorithm
(``eora_inputs`` A-cov / Stage-3 originals load, ``eora_compensation``
√Λ-weighted per-layer residual compensation) into ``stage4/plugins/``.
S4-4a flips the relationship: :func:`run` here is now the REAL orchestrator
and ``stage4_eora.run`` is a thin shim that delegates to it.

The schedule
------------
``load_eora_inputs → LOOP layers[compensate_layer] → finalize``.

Division of labour
------------------
The two plugin hooks own the ALGORITHM. ``EoraInputsPlugin.load_eora_inputs``
reproduces the monolith's inline input-load block (A-cov / Stage-3 originals
load, the file-deleted double-widen guard, the layer list, the partial-dir
setup, the ``stage3_ranks`` snapshot). ``EoraCompensationPlugin.compensate_layer``
reproduces the monolith's per-layer EoRA COMPUTE branch (per-matrix budget
calc, per-expert ``_compute_eora_factors`` loop, the in-process double-widen
``assert``, ``widen_rank``, trackio, spill). This orchestrator owns the GLUE:
the per-layer loop itself, the RESUME-FROM-SPILL branch (UNOWNED by either
plugin), and the finalize block (checkpoint save, ``eora_ranks.json`` write,
spill cleanup, sidecar deletion). Every glue line is a verbatim copy from the
monolith ``run()``, just reorganized around the ``walk_phases`` calls.

Why a plain ``for`` loop, not ``loop_over``
-------------------------------------------
The per-layer loop runs on the ROOT ctx via a plain ``for ref in layers:``
loop, NOT ``loop_over``. ``loop_over`` opens a fresh child scope per layer;
``compensated_params`` is a scalar int the ``compensate_layer`` hook rebinds
each iteration via ``ctx.set(..., overwrite=True)`` — in a child scope that
rebind would shadow the parent and the total would NOT accumulate across
layers. Dispatching against the ROOT ctx makes the ``overwrite=True`` rebind
land on the root, so the scalar accumulates. The plain loop also lets the
RESUME-FROM-SPILL branch ``continue`` past a layer — something ``loop_over``
cannot express.

Monkeypatch survival (HAZARD H3)
--------------------------------
The golden / smoke tests ``monkeypatch.setattr`` ``save_compressed_checkpoint``
on its SOURCE module ``utils.model_io``. So the finalize block calls it
module-qualified (``mio.save_compressed_checkpoint``) — the patch reaches it
without the test fixture needing to also patch ``stage4.orchestrator``.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

import torch
import torch.nn as nn

from ..pipeline.context import PipelineContext
from ..pipeline.registry import PluginRegistry
from ..tools.phase_walker import walk_phases
from ..utils import model_io as mio
from ..utils.model_io import MATRIX_NAMES, FactoredExperts, save_json_artifact

from .plugins.eora_compensation import EoraCompensationPlugin
from .plugins.eora_inputs import EoraInputsPlugin

log = logging.getLogger(__name__)


def run(
    model,
    tokenizer,
    config: dict,
    artifacts_dir: Path,
    *,
    no_resume: bool = False,
) -> Path:
    """Run Stage 4 — EoRA residual compensation — via the plugin pipeline.

    Threads one :class:`PipelineContext` through the three-phase schedule
    ``load_eora_inputs → LOOP layers[compensate_layer] → finalize``. Returns
    the Stage 4 output directory (``artifacts_dir / "stage4_eora"``), same as
    the legacy monolith ``run()``.
    """
    # ---- one PipelineContext: input slots + run-glue intermediates --------
    # rank_map: ONE shared mutable dict set on the ROOT ctx. Both the
    # compensate_layer hook and the resume-from-spill glue mutate this same
    # dict in place across iterations (never a per-layer rank_map).
    # compensated_params: a scalar running total — see the module docstring
    # on why the per-layer loop dispatches against the ROOT ctx.
    run_ctx = PipelineContext()
    run_ctx.set("model", model)
    run_ctx.set("tokenizer", tokenizer)
    run_ctx.set("config", config)
    run_ctx.set("artifacts_dir", artifacts_dir)
    run_ctx.set("no_resume", no_resume)
    run_ctx.set("rank_map", {})
    run_ctx.set("compensated_params", 0)

    registry = PluginRegistry([EoraInputsPlugin(), EoraCompensationPlugin()])
    plugins = registry.enabled(config)

    # ---- load_eora_inputs ------------------------------------------------
    # EoraInputsPlugin.load_eora_inputs writes A_cov / a_storage_dtype /
    # originals / layers / partial_dir / stage3_ranks onto run_ctx.
    walk_phases(("load_eora_inputs",), plugins, run_ctx)
    layers = run_ctx.get("layers")
    partial_dir = run_ctx.get("partial_dir")

    # ---- compensate_layer (per-layer loop) -------------------------------
    # Plain for-loop on the ROOT ctx (NOT loop_over): the scalar
    # compensated_params accumulation needs the rebind to land on the root,
    # and the resume branch needs to `continue` past a layer.
    for k, ref in enumerate(layers):
        fe = ref.experts_module
        if not isinstance(fe, FactoredExperts):
            continue
        dev = fe.gate_proj_U.device
        dtype = fe.gate_proj_U.dtype

        # Crash-resume: load saved layer state if present. This branch is
        # UNOWNED by either plugin — it lives here as orchestrator glue,
        # copied byte-for-byte from the monolith run()'s per-layer loop.
        spill_path = partial_dir / f"layer_{ref.layer_idx}.pt" if partial_dir is not None else None
        if spill_path is not None and spill_path.exists():
            try:
                payload = torch.load(spill_path, map_location="cpu")
            except Exception as exc:
                raise RuntimeError(
                    f"Stage 4 resume: failed to load {spill_path}: {exc}"
                ) from exc
            fv = int(payload.get("format_version", 0))
            if fv != 1:
                raise RuntimeError(
                    f"Stage 4 resume: {spill_path} has format_version={fv} "
                    "(expected 1) — delete _stage4_partial/ and re-run Stage 4"
                )
            for name in MATRIX_NAMES:
                u = payload[f"{name}_U"].to(device=dev, dtype=dtype)
                v = payload[f"{name}_V"].to(device=dev, dtype=dtype)
                setattr(fe, f"{name}_U", nn.Parameter(u, requires_grad=False))
                setattr(fe, f"{name}_V", nn.Parameter(v, requires_grad=False))
                fe.ranks[name] = int(payload["ranks"][name])
                if "effective_ranks" in payload:
                    fe.effective_ranks[name] = [int(r) for r in payload["effective_ranks"][name]]
            run_ctx.get("rank_map").update(payload["rank_map_layer"])
            run_ctx.set(
                "compensated_params",
                run_ctx.get("compensated_params") + int(payload["compensated_params_layer"]),
                overwrite=True,
            )
            log.info("Stage 4 layer %d/%d (idx=%d) — resumed from partial",
                     k + 1, len(layers), ref.layer_idx)
            continue

        # Compute branch — dispatch the compensate_layer hook against the
        # ROOT ctx. The hook reads layer_ref + the run-scope slots off the
        # root and mutates rank_map + compensated_params on the root.
        run_ctx.set("layer_ref", ref, overwrite=("layer_ref" in run_ctx))
        walk_phases(("compensate_layer",), plugins, run_ctx)

    # ---- finalize --------------------------------------------------------
    rank_map = run_ctx.get("rank_map")
    compensated_params = run_ctx.get("compensated_params")
    out_dir = artifacts_dir / "stage4_eora"
    mio.save_compressed_checkpoint(
        model, tokenizer, out_dir,
        pipeline_stage="stage4_eora",
        extra_metadata={"compensated_params": compensated_params},
    )
    save_json_artifact({
        "rank_map": rank_map,
        "compensated_params": compensated_params,
        "config": config["stage4_eora"],
    }, out_dir / "eora_ranks.json")

    if partial_dir is not None:
        shutil.rmtree(partial_dir, ignore_errors=True)

    # Stage 4 is the last consumer of both `_stage3_original_weights.pt` and
    # `_stage2_input_covariance.pt`. Both are already durable on the per-stage
    # Hub repos (`<base>-stage2`, `<base>-stage3`) — leaving them on the
    # bucket only causes the entrypoint's job-exit aux upload to push ~140 GB
    # of already-uploaded data to the aggregate result repo. Delete on Stage 4
    # success only; on failure they stay so a re-run can pick up cleanly.
    for sidecar in ("_stage3_original_weights.pt", "_stage2_input_covariance.pt"):
        p = artifacts_dir / sidecar
        if p.exists():
            try:
                p.unlink()
                log.info("Deleted %s (no longer needed past Stage 4; durable on Hub)", p)
            except OSError as exc:
                log.warning("Could not delete %s: %s", p, exc)
    log.info("Stage 4 complete — EoRA added %d params → %s", compensated_params, out_dir)
    return out_dir


__all__ = ["run"]
