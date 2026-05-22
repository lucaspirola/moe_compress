"""EoRA residual compensation (S4-3 of the Stage 4 plugin-architecture refactor).

Home of the ``EoraCompensationPlugin``, which owns the EoRA per-layer
``compensate_layer`` phase: the per-matrix compensation-budget calculation,
the per-expert ``_compute_eora_factors`` loop, the in-process double-widen
``assert``, the ``FactoredExperts.widen_rank`` call, the trackio emit, and the
per-layer crash-resume spill.

S4-3 is a MIXED relocation — it has three parts:

(1) Two STANDALONE functions are relocated VERBATIM from the
    ``stage4_eora.py`` monolith (the S3-2/S3-3 pattern):

    * ``_spill_layer``        — atomic per-layer crash-resume spill;
    * ``_compute_eora_factors`` — the paper-correct EoRA residual kernel
      (2410.21271, Algorithm 1).

    The monolith re-imports both via a ``# noqa: F401`` block so ``run()`` and
    external callers/tests keep their ``stage4_eora`` import paths.

(2) The per-matrix budget calc + per-expert widen loop in the monolith
    ``run()`` body (lines ~152-250) is NOT a standalone function — it is
    inline ``run()`` code. It is therefore REPRODUCED in the inert
    ``compensate_layer`` hook below rather than relocated; the monolith
    ``run()`` is left BYTE-IDENTICAL for those statements (the S4-2 pattern).
    S4-4 deletes the monolith ``run()`` and wires this hook live; the
    duplication resolves at that point.

(3) The dtype noise-floor table ``_NOISE_FLOOR_BY_DTYPE`` was relocated by
    S4-3 to ``tools/dtype_noise_floor`` (a pure literal shared by stage 3 and
    stage 4); see the deviation note below.

THE ONE DEVIATION FROM VERBATIM. The monolith's ``_compute_eora_factors``
resolves the dtype noise floor through a function-scope
``from moe_compress.stage3_svd import _NOISE_FLOOR_BY_DTYPE`` (monolith
~line 369) — a stage4→stage3 cross-import. In this relocated copy that
function-scope import is DELETED; the name is supplied instead by the
module-top ``from ...tools.dtype_noise_floor import _NOISE_FLOOR_BY_DTYPE``.
This is behavior-identical — the same dict object, the same lookup — and is
gated by the S4-0 golden snapshot. No other change to the relocated bodies.

Circular-import note (mirror of ``stage4/plugins/eora_inputs``): this module
imports only from ``...utils.*``, ``...pipeline.*``, ``...tools.*`` and
stdlib/torch — NEVER from ``stage4_eora`` or ``stage4.orchestrator`` at any
scope (module-top OR function-local). The monolith ``stage4_eora`` imports
*this* module at load time (the S4-3 ``# noqa: F401`` re-import block), so a
``from ..stage4_eora import ...`` here — at any scope — would deadlock the
import cycle; nothing here does that.

``EoraCompensationPlugin`` is registered-but-INERT at S4-3 — no orchestrator
walk or test invokes its ``compensate_layer`` hook. S4-4 plugs the hook into
the live Stage 4 plugin sequencer and deletes the monolith ``run()``.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import torch

from ...pipeline.context import PipelineContext
from ...tools.dtype_noise_floor import _NOISE_FLOOR_BY_DTYPE
from ...utils.model_io import MATRIX_NAMES, FactoredExperts
from ...utils.trackio_log import trackio_log as _trackio_log

log = logging.getLogger(__name__)


def _spill_layer(
    partial_dir: Path,
    layer_idx: int,
    fe: FactoredExperts,
    rank_map_layer: dict[str, int],
    compensated_params_layer: int,
) -> None:
    payload = {
        "format_version": 1,
        "layer_idx": layer_idx,
        "ranks": dict(fe.ranks),
        "effective_ranks": {n: list(v) for n, v in fe.effective_ranks.items()},
        "rank_map_layer": rank_map_layer,
        "compensated_params_layer": compensated_params_layer,
    }
    for name in MATRIX_NAMES:
        payload[f"{name}_U"] = getattr(fe, f"{name}_U").data.cpu()
        payload[f"{name}_V"] = getattr(fe, f"{name}_V").data.cpu()
    tmp = partial_dir / f"layer_{layer_idx}.pt.tmp"
    final = partial_dir / f"layer_{layer_idx}.pt"
    torch.save(payload, tmp)
    os.replace(tmp, final)


def _compute_eora_factors(
    delta: torch.Tensor,
    A: torch.Tensor | None,
    r: int,
    device,
    *,
    storage_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Paper-correct EoRA (2410.21271) Algorithm 1.

    Steps (matching paper notation, restricted to signal eigenspace):
      1. ΔW = W_orig − Ŵ                           [d_out × d_in]  (caller)
      2. Eigendecompose X̃X̃ᵀ = QΛQᵀ                [d_in × d_in]
         Keep n_keep eigenvectors above noise floor.
      3. Q' = Q_keep · √Λ_keep                      [d_in × n_keep]
      4. ΔW' = ΔW · Q'                              [d_out × n_keep]  (full projection)
      5. SVD(ΔW') full_matrices=False → top take_eff=min(r, min(d_out, n_keep))
      6. B' = U'[:,:take_eff] · Σ'[:take_eff]       [d_out × take_eff]
         A  = Vh'[:take_eff] · diag(1/√Λ_keep) @ Q_keep^T   [take_eff × d_in]

    The √Λ scaling is the core innovation of EoRA: it importance-weights the
    eigenspace so SVD concentrates rank budget on high-variance input directions.
    Without it, this degenerates to the Act-S baseline.

    Returns (U, V, take_eff) where `take_eff <= r` is the effective rank.
    When `take_eff < r`, U/V are zero-padded to width r.
    """
    if r <= 0:
        return (torch.zeros(delta.shape[0], 0, device=device),
                torch.zeros(0, delta.shape[1], device=device), 0)
    delta = delta.to(device=device, dtype=torch.float32)
    d_out, d_in = delta.shape

    def _plain_svd_padded() -> tuple[torch.Tensor, torch.Tensor, int]:
        # Fallback: plain SVD without activation weighting.
        U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
        rk = min(r, U.shape[1])
        if rk == r:
            return U[:, :r] * S[:r], Vh[:r, :], r
        U_out = torch.zeros(d_out, r, device=device, dtype=delta.dtype)
        V_out = torch.zeros(r, d_in, device=device, dtype=delta.dtype)
        U_out[:, :rk] = U[:, :rk] * S[:rk]
        V_out[:rk, :] = Vh[:rk, :]
        return U_out, V_out, rk

    if A is None:
        return _plain_svd_padded()

    A = A.to(device=device, dtype=torch.float32)
    A = 0.5 * (A + A.T)
    if A.shape != (d_in, d_in):
        log.warning("EoRA: covariance shape %s != (%d,%d), falling back to plain SVD",
                    A.shape, d_in, d_in)
        return _plain_svd_padded()

    # Step 2: Eigendecompose activation covariance X̃X̃ᵀ = QΛQᵀ  (A shape [d_in × d_in])
    eigvals, eigvecs = torch.linalg.eigh(A)  # ascending order

    sigma_max = float(eigvals[-1].clamp_min(0).item())
    # Dtype-aware noise floor — see _NOISE_FLOOR_BY_DTYPE in tools.dtype_noise_floor.
    rel_floor = _NOISE_FLOOR_BY_DTYPE.get(storage_dtype or torch.float32, 1e-6)
    thresh = max(sigma_max * rel_floor, 1e-12)

    keep_mask = eigvals > thresh
    if not keep_mask.any():
        return _plain_svd_padded()

    # Keep only directions above the noise floor for the projection.
    eigvals_keep = eigvals[keep_mask].clamp_min(0)
    eigvecs_keep = eigvecs[:, keep_mask]         # [d_in, n_keep]
    n_keep = int(eigvals_keep.numel())

    # Step 3: Q' = Q · √Λ  (paper Algorithm 1 step 3).
    # FULL projection matrix — NOT truncated to r. The SVD in step 5
    # will optimally select the best r directions from the FULL d_in-
    # dimensional projected error. Pre-truncating to r would eliminate
    # the joint optimisation that distinguishes EoRA from Act-S.
    sqrt_lambda = eigvals_keep.sqrt()                         # [n_keep]
    Q_prime = eigvecs_keep * sqrt_lambda.unsqueeze(0)         # [d_in, n_keep]

    # Step 4: ΔW' = ΔW · Q'  (FULL projection, [d_out × n_keep])
    delta_prime = delta @ Q_prime                             # [d_out, n_keep]

    # Step 5: rank-r SVD of the full projected error
    U_p, S_p, Vh_p = torch.linalg.svd(delta_prime, full_matrices=False)
    take_eff = min(r, int(U_p.shape[1]))

    # Step 6a: B' = U' Σ'
    U_corr = U_p[:, :take_eff] * S_p[:take_eff]              # [d_out, take_eff]

    # Step 6b: Back-project A = V'^T · Q'^{-1}
    # Q'^{-1} = diag(1/√Λ) · Q^T  (since Q has orthonormal columns)
    # So: A = V'^T · diag(1/√Λ) · Q^T = (Vh_p[:take_eff] · diag(1/√Λ)) @ Q^T
    inv_sqrt_lambda = eigvals_keep.clamp_min(1e-30).rsqrt()   # [n_keep]
    V_corr = (Vh_p[:take_eff, :] * inv_sqrt_lambda.unsqueeze(0)) @ eigvecs_keep.T
    # V_corr shape: [take_eff, d_in]

    # Zero-pad to fixed r so caller's pre-allocated tensors stay shape-stable.
    if take_eff >= r:
        return U_corr[:, :r], V_corr[:r, :], r
    U_out = torch.zeros(d_out, r, device=device, dtype=U_corr.dtype)
    V_out = torch.zeros(r, d_in, device=device, dtype=V_corr.dtype)
    U_out[:, :take_eff] = U_corr
    V_out[:take_eff, :] = V_corr
    return U_out, V_out, take_eff


class EoraCompensationPlugin:
    """Stage 4 EoRA residual-compensation plugin (S4-3 — registered-but-INERT).

    Owns the EoRA per-layer ``compensate_layer`` phase: for each matrix type,
    the per-matrix compensation-budget calculation (capped at
    ``compensation_budget_pct`` of the Stage-3 saved params), the per-expert
    ``_compute_eora_factors`` residual kernel loop, the in-process
    double-widen ``assert``, the ``FactoredExperts.widen_rank`` call, the
    trackio emit, and the per-layer crash-resume spill (``_spill_layer``).

    The residual kernel ``_compute_eora_factors`` and the spill helper
    ``_spill_layer`` are relocated VERBATIM from the ``stage4_eora.py``
    monolith (one deviation — see the module docstring); the monolith
    re-imports them. The per-matrix budget + widen loop is NOT a standalone
    function in the monolith — it is inline ``run()`` code — so the
    ``compensate_layer`` hook below REPRODUCES it; the monolith ``run()`` is
    left byte-identical for those statements.

    S4-3 wires this class into the plugin registry as metadata only — no walk
    or test invokes ``compensate_layer``. S4-4 plugs the hook into the live
    Stage 4 plugin sequencer and deletes the monolith ``run()``.
    """

    name = "eora_compensation"
    paper = (
        "EoRA residual compensation — √Λ-scaled eigenspace projection of the "
        "factorization residual ΔW, rank-r SVD of the full projected error, "
        "back-projected widen of FactoredExperts U/V (paper 2410.21271, "
        "Algorithm 1)."
    )
    config_key = "stage4_eora.compensation_budget_pct"
    # ``compensate_layer`` runs inside a per-layer scope: it reads the layer
    # ref under ``layer_ref`` (the loop item key) and the remaining run-scope
    # slots through the parent ctx chain.
    reads: tuple[str, ...] = (
        "layer_ref", "originals", "A_cov", "a_storage_dtype", "config",
        "partial_dir", "stage3_ranks", "rank_map", "compensated_params",
    )
    # ``rank_map`` is a shared mutable dict the hook MUTATES in place across
    # loop iterations (mirror of ``aa_svd_factor.factor_layer``'s HAZARD H1)
    # rather than rebinding via ``ctx.set``; ``compensated_params`` is the
    # running total the hook advances. Both remain this plugin's declared
    # write surface.
    writes: tuple[str, ...] = ("rank_map", "compensated_params")
    # Empty: EoraCompensationPlugin needs no calibration pass — the residual
    # compensation consumes only precomputed inputs (Stage-3 originals, the
    # A-covariance) already loaded by EoraInputsPlugin.
    provides: tuple[str, ...] = ()

    def is_enabled(self, config: dict) -> bool:
        """Always True — EoRA compensation is UNCONDITIONAL.

        Every Stage 4 run applies residual compensation. The per-matrix
        budget calc may compute ``r_per_expert <= 0`` and ``continue`` past an
        individual matrix internally, but the plugin as a whole always runs;
        ``config_key`` only parametrises the per-matrix compensation budget,
        it never gates the plugin.
        """
        return True

    def contribute_artifact(self, ctx: Any) -> dict:
        return {}

    def compensate_layer(self, ctx: PipelineContext) -> None:
        """Phase hook — EoRA per-layer residual compensation (S4-4 wiring surface).

        INERT at S4-3: no orchestrator walk or test invokes this hook. S4-4
        replaces the Stage 4 orchestrator body with the plugin sequencer and
        dispatches this hook in place of the monolith ``run()``'s inline
        per-layer ``for name in MATRIX_NAMES:`` block (``stage4_eora.py``
        lines ~152-250). The body below reproduces that block faithfully — it
        is dead code at S4-3 but S4-4 relies on it once the monolith is
        deleted.

        Reproduces (in monolith order): the per-matrix compensation-budget
        calc, the per-expert ``_compute_eora_factors`` loop, the in-process
        double-widen ``assert`` (INCLUDED — S4-4 relies on it), the
        ``fe.widen_rank`` call, the trackio emit, and the per-layer tail
        (``rank_map.update`` / ``compensated_params +=`` / ``_spill_layer``).

        The layer ref arrives under ``ctx["layer_ref"]`` (the loop item key);
        ``originals`` / ``A_cov`` / ``a_storage_dtype`` / ``config`` /
        ``stage3_ranks`` / ``rank_map`` / ``compensated_params`` resolve
        through the parent ctx chain. ``partial_dir`` is ``has()``-guarded —
        it is ``None`` when ``no_resume=True``.
        """
        # Required slots — direct get(): a missing one is a wiring bug and
        # SHOULD raise. ``partial_dir`` is optional (has()-guarded).
        ref = ctx.get("layer_ref")
        originals = ctx.get("originals")
        A_cov = ctx.get("A_cov")
        a_storage_dtype = ctx.get("a_storage_dtype")
        config = ctx.get("config")
        stage3_ranks = ctx.get("stage3_ranks")
        rank_map = ctx.get("rank_map")
        compensated_params = ctx.get("compensated_params")
        partial_dir = ctx.get("partial_dir") if ctx.has("partial_dir") else None

        s4 = config["stage4_eora"]
        fe = ref.experts_module
        if not isinstance(fe, FactoredExperts):
            return
        dev = fe.gate_proj_U.device
        dtype = fe.gate_proj_U.dtype
        N = fe.num_experts

        layer_compensated_params = 0
        rank_map_layer: dict[str, int] = {}

        for name in MATRIX_NAMES:
            # Per-matrix-type, per-layer: pool per-expert residuals independently.
            # Budget: ≤ compensation_budget_pct of saved params for this matrix.
            d_out, d_in = fe.matrix_shape(name)
            cur_rank = fe.ranks[name]
            saved_per_expert = d_out * d_in - cur_rank * (d_out + d_in)
            saved_for_matrix = max(0, saved_per_expert) * N
            param_budget = int(s4["compensation_budget_pct"] * saved_for_matrix)
            r_per_expert = param_budget // max(N * (d_out + d_in), 1)
            r_per_expert = min(r_per_expert, s4["eigenspace_rank_cap"], min(d_out, d_in))
            r_per_expert = max(0, r_per_expert)
            if r_per_expert <= 0:
                continue

            U_corr = torch.zeros(N, d_out, r_per_expert, dtype=dtype, device=dev)
            V_corr = torch.zeros(N, r_per_expert, d_in, dtype=dtype, device=dev)
            # Per-expert effective rank of the EoRA correction. Defaults to
            # 0 for experts without an `originals` entry (no correction
            # applied → no parameters added in effective terms).
            eff_per_expert: list[int] = [0] * N
            # Track residual norm before/after to verify EoRA actually helps.
            res_before_sum = 0.0
            res_after_sum = 0.0
            n_eligible = 0
            for e in range(N):
                key = (ref.layer_idx, e, name)
                W_orig = originals.get(key)
                if W_orig is None:
                    continue
                W_orig_f = W_orig.to(device=dev, dtype=torch.float32)
                U_e = fe.gate_proj_U.data[e] if name == "gate_proj" else \
                       fe.up_proj_U.data[e]   if name == "up_proj"   else \
                       fe.down_proj_U.data[e]
                V_e = fe.gate_proj_V.data[e] if name == "gate_proj" else \
                       fe.up_proj_V.data[e]   if name == "up_proj"   else \
                       fe.down_proj_V.data[e]
                delta = W_orig_f - (U_e.to(torch.float32) @ V_e.to(torch.float32))
                res_before_sum += float(delta.norm().item() ** 2)
                # up_proj shares the gate_proj input covariance (same fused tensor).
                cov_key = (ref.layer_idx, e, "gate_proj") if name == "up_proj" else key
                A = A_cov.get(cov_key)
                Uc, Vc, take_eff = _compute_eora_factors(
                    delta, A, r_per_expert, dev, storage_dtype=a_storage_dtype,
                )
                U_corr[e] = Uc.to(dtype)
                V_corr[e] = Vc.to(dtype)
                eff_per_expert[e] = int(take_eff)
                # Residual after applying the planned correction (Uc @ Vc).
                res_after = delta - (Uc.to(torch.float32) @ Vc.to(torch.float32))
                res_after_sum += float(res_after.norm().item() ** 2)
                n_eligible += 1
                if (e + 1) % 32 == 0:
                    log.info("  L%d/%s expert %d/%d", ref.layer_idx, name, e + 1, N)

            # Double-widen guard: assert ranks haven't been modified yet.
            # Protects against in-process re-runs (notebooks, test harnesses)
            # where widen_rank() would double-apply EoRA correction.
            assert fe.ranks[name] == stage3_ranks.get(ref.layer_idx, {}).get(name, fe.ranks[name]), (
                f"Stage 4 double-widen detected: layer={ref.layer_idx}, matrix={name}, "
                f"current_rank={fe.ranks[name]}, "
                f"stage3_rank={stage3_ranks.get(ref.layer_idx, {}).get(name)}. "
                "widen_rank() has already been applied in this process."
            )
            fe.widen_rank(name, U_corr, V_corr, added_effective_per_expert=eff_per_expert)
            rank_map_layer[f"L{ref.layer_idx}_{name}"] = fe.ranks[name]
            layer_compensated_params += int(U_corr.numel() + V_corr.numel())
            res_before = (res_before_sum / max(n_eligible, 1)) ** 0.5
            res_after = (res_after_sum / max(n_eligible, 1)) ** 0.5
            rel_drop = (res_before - res_after) / max(res_before, 1e-12)
            log.info("  L%d/%s widened by r=%d → new rank=%d; residual %.4e→%.4e (-%.1f%%)",
                     ref.layer_idx, name, r_per_expert, fe.ranks[name],
                     res_before, res_after, 100 * rel_drop)
            _eff_list = [v for v in eff_per_expert if v is not None]
            _trackio_log({
                "stage4/layer_idx": ref.layer_idx,
                f"stage4/{name}_added_rank": r_per_expert,
                f"stage4/{name}_new_rank": fe.ranks[name],
                f"stage4/{name}_residual_before": res_before,
                f"stage4/{name}_residual_after": res_after,
                f"stage4/{name}_residual_rel_drop": rel_drop,
                "stage4/compensated_params": compensated_params + layer_compensated_params,
                # Additive v2 keys: per-layer aggregates of in-scope variables.
                f"stage4/{name}_n_eligible_experts": int(n_eligible),
                f"stage4/{name}_eff_rank_mean": (
                    float(sum(_eff_list) / len(_eff_list)) if _eff_list else 0.0
                ),
                f"stage4/{name}_eff_rank_max": int(max(_eff_list)) if _eff_list else 0,
                f"stage4/{name}_eff_rank_min": int(min(_eff_list)) if _eff_list else 0,
                # Per-matrix contribution (not the per-layer running total —
                # which is already in `stage4/compensated_params`).
                f"stage4/{name}_matrix_compensated_params": int(U_corr.numel() + V_corr.numel()),
            })

        rank_map.update(rank_map_layer)
        compensated_params += layer_compensated_params
        # S4-4: dispatched against the ROOT ctx by a plain for-loop (not
        # loop_over); the overwrite=True rebind of the root scalar accumulates
        # across layers.
        ctx.set("compensated_params", compensated_params, overwrite=True)

        # Atomically persist this layer's FactoredExperts state for crash-resume.
        if partial_dir is not None:
            _spill_layer(partial_dir, ref.layer_idx, fe, rank_map_layer, layer_compensated_params)
