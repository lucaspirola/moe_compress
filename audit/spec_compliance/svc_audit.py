"""SVC audit — Singular Value Calibration projection coefficients for REAM merges.

Diagnostic-only spec-compliance audit. Surfaces per-(layer, merge-group, rank,
pre-merge-expert) projection coefficients quantifying how much each pre-merge
expert's principal output-activation directions survive REAM's linear-combination
merge.

Algorithm reference
-------------------

The "Singular Value Calibration" (SVC) paper

    "When Shared Knowledge Hurts: Spectral Over-Accumulation in Model Merging"
    arXiv:2602.05536 (referenced by fusion_bench's
    ``method/singular_value_calibration/svc.py:18``)

defines a per-subspace projection coefficient (Eq. 8) that diagnoses whether
the merged matrix over-counts a direction relative to a donor's contribution::

    s_r^i = <a_r^merge, a_r^i> / ||a_r^i||^2

In the upstream weight-space formulation the per-task response is
``a_r^i = u_r^T · ΔW_i`` projected onto the merged matrix's left singular
vectors. This audit adapts the same projection-coefficient primitive to the
**output-activation space** so we can characterise REAM merges in the same
spectral basis the downstream layers actually consume:

* ``a_r^merge`` — the r-th left singular vector of the merged centroid expert's
  output activation matrix ``Y^merge = W^merge · X^T`` (shape ``[d_out, N]``).
* ``a_r^i`` — the r-th left singular vector of the i-th pre-merge donor
  expert's output activation matrix ``Y^i = W^i · X^T``.
* ``X`` — token activations into the expert. For each donor ``i`` we evaluate
  both ``W^merge`` and ``W^i`` on the same input distribution ``X_i``
  (the donor's own pre-merge input stream) so that the comparison is
  apples-to-apples: "did the merged centroid preserve donor ``i``'s output
  principal directions on donor ``i``'s typical inputs?".

Interpretation:

* ``s_r^i ≈ 1`` — REAM preserved that direction faithfully.
* ``s_r^i ≈ 0`` — REAM over-accumulated other donors' contributions there,
  effectively dropping donor ``i``'s contribution in subspace ``r``.
* ``|s_r^i| > 1`` — the merged direction amplifies donor ``i``'s direction
  (constructive accumulation; the upstream paper treats this as
  "over-counting" of a shared direction).
* ``s_r^i < 0`` — anti-aligned (the merged direction is the opposite sense).

Because left singular vectors are unit norm, ``||a_r^i||^2 == 1`` and the
formula reduces to a plain inner product (a cosine similarity in ``[-1, 1]``);
we keep the divisor in the code path so the script transparently extends to
non-unit response vectors if a future variant wants to operate on the upstream
weight-delta interpretation (where ``||a_r^i||^2 != 1``).

Pattern H (clean-room) — license + attribution
----------------------------------------------

This file is a **clean-room re-implementation from the paper's prose**
(arXiv:2602.05536 Eq. 8) and from the public docstrings in
``fusion_bench/method/singular_value_calibration/utils.py``. No upstream
source code is vendored. The cross-check reference implementation is
``github.com/tanganke/fusion_bench`` (MIT, Copyright (c) 2024 Anke Tang) at
``fusion_bench/method/singular_value_calibration/utils.py:52``
(``compute_projection_coefficient``) and ``:14``
(``project_onto_singular_vectors``). License verified 2026-05-28 via the
upstream LICENSE file (standard MIT, fully permissive). Because no code
text is copied, fusion_bench's MIT attribution requirement (preserve the
copyright + permission notice in copies of substantial portions) does not
strictly apply to this file, but we cite the reference for paper-fidelity
audit-trail purposes (per the project's ``paper_fidelity_review_loop``
discipline).

Operational scope
-----------------

This is a **diagnostic** script. It does not mutate any pipeline artifact
or runtime weight. It reads:

1. The Stage 3 originals snapshot ``_stage3_original_weights.pt`` (a
   ``dict[(layer, expert, matrix), Tensor]`` produced by
   ``stage3.plugins.swift_svd_alpha._snapshot_originals``).
2. The post-Stage-2 merged checkpoint (a HF-format directory; we pull
   the fused expert tensors back into the same key space as the originals).
3. The per-layer merge map ``merge_map.json`` (centroid → donor list,
   produced by ``stage2.shared_io._write_merge_json`` and aggregated by
   the Stage 2 orchestrator's final ``save_json_artifact``).
4. The per-(layer, expert, matrix) input covariance ``Σ_in`` cached at
   ``_stage2_input_covariance.pt`` (the same artifact the Stage 3
   AA-SVD path consumes).

Outputs:

* ``svc_audit_results.json`` — per-(layer, centroid, donor, matrix, rank)
  projection coefficient ``s_r^i``.
* ``svc_audit_summary.md`` — per-layer human-readable summary table
  (mean / |s| > 1.3 outlier count / "dropped donor" count per matrix).

CLI
---

    python audit/spec_compliance/svc_audit.py \
        --stage2-artifacts <dir-containing-merge_map.json-+-covariance.pt> \
        --originals-pt <path-to-_stage3_original_weights.pt> \
        --merged-checkpoint <hf-format-dir-or-state-dict.pt> \
        --output-dir audit/spec_compliance/

For tests / development, the math primitives
(``compute_projection_coefficient``, ``svc_scores_for_group``) are exposed
directly so the heavy I/O can be bypassed with synthetic tensors.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

log = logging.getLogger("svc_audit")


# --------------------------------------------------------------------------- #
# Math primitives                                                             #
# --------------------------------------------------------------------------- #


def compute_projection_coefficient(
    merged_direction: torch.Tensor,
    donor_direction: torch.Tensor,
    eps: float = 1e-12,
) -> float:
    """Eq. 8 of arXiv:2602.05536, adapted to the audit's vector form.

    ``s_r^i = <a_r^merge, a_r^i> / ||a_r^i||^2``

    Args:
        merged_direction: ``a_r^merge`` — r-th left singular vector of the
            merged centroid's output activation matrix (shape ``[d_out]``).
        donor_direction: ``a_r^i`` — r-th left singular vector of donor
            ``i``'s output activation matrix (shape ``[d_out]``).
        eps: Numerical floor on ``||a_r^i||^2`` to avoid division by zero
            when the donor's r-th singular value is degenerate.

    Returns:
        Scalar projection coefficient. For unit-norm singular vectors this
        reduces to ``cos(angle) ∈ [-1, 1]``; the divisor is kept in code
        so the function transparently handles non-unit-norm inputs (e.g.
        if the script is extended to operate on the upstream weight-delta
        subspace responses where ``a_r^i = u_r^T · ΔW_i`` is not unit norm).
    """
    if merged_direction.shape != donor_direction.shape:
        raise ValueError(
            f"compute_projection_coefficient: shape mismatch "
            f"merged={tuple(merged_direction.shape)} donor={tuple(donor_direction.shape)}"
        )
    if merged_direction.dim() != 1:
        raise ValueError(
            "compute_projection_coefficient expects 1-D inputs "
            f"(got merged.dim()={merged_direction.dim()})"
        )
    # Promote to float64 for the inner product so a single bf16 / fp16 input
    # does not silently round the small <a_r^merge, a_r^i> dot to zero on the
    # near-orthogonal directions (where the diagnostic actually carries signal).
    merged_f64 = merged_direction.to(torch.float64)
    donor_f64 = donor_direction.to(torch.float64)
    denom = torch.dot(donor_f64, donor_f64).clamp_min(eps)
    numer = torch.dot(merged_f64, donor_f64)
    return float((numer / denom).item())


def _top_left_singular_vectors(
    activation_matrix: torch.Tensor,
    rank: int,
) -> torch.Tensor:
    """Top-``rank`` left singular vectors of an activation matrix.

    The activation matrix is taken with shape ``[d_out, N]`` so that left
    singular vectors live in the output-feature space (``R^{d_out}``) —
    consistent with the upstream convention where ``U`` of an SVD of a
    ``[d_out, d_in]`` weight matrix spans the output space.

    Args:
        activation_matrix: ``Y`` of shape ``[d_out, N]``.
        rank: Number of leading left singular vectors to return.

    Returns:
        Tensor of shape ``[rank, d_out]`` whose row ``r`` is ``a_r``,
        the r-th left singular vector (unit norm).
    """
    if activation_matrix.dim() != 2:
        raise ValueError(
            f"_top_left_singular_vectors expects a 2-D matrix "
            f"(got shape {tuple(activation_matrix.shape)})"
        )
    d_out = activation_matrix.shape[0]
    max_rank = min(d_out, activation_matrix.shape[1])
    if rank > max_rank:
        raise ValueError(
            f"_top_left_singular_vectors: requested rank={rank} exceeds "
            f"min(d_out, N) = min({d_out}, {activation_matrix.shape[1]}) = {max_rank}"
        )
    # full_matrices=False keeps the thin SVD; float64 promotion stabilises
    # the eigenvalue ordering on near-degenerate spectra (the diagnostic's
    # most interesting regime).
    # N4: explicit ``no_grad`` documents intent — this primitive is
    # diagnostic-only and never participates in autograd. All input
    # tensors arrive detached (loaded via ``torch.load`` from disk), so
    # this is belt-and-suspenders rather than a behavioural fix.
    with torch.no_grad():
        U, _S, _Vh = torch.linalg.svd(
            activation_matrix.to(torch.float64), full_matrices=False
        )
        # Columns of U are left singular vectors; return rows for caller ergonomics.
        return U[:, :rank].transpose(0, 1).contiguous()


def _activation_matrix_from_cov(
    weight: torch.Tensor,
    sigma_in: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct an N-equivalent activation matrix from Σ_in.

    Given a weight ``W ∈ R^{d_out × d_in}`` and a cached input covariance
    ``Σ_in = X^T X / N`` (shape ``[d_in, d_in]``), we want a synthetic
    activation matrix ``Ỹ`` whose left singular structure matches the true
    activation matrix ``Y = W X^T ∈ R^{d_out × N}``. Because

        Y Y^T = W (X^T X) W^T = N · W Σ_in W^T,

    the left singular vectors of ``Y`` are the eigenvectors of ``W Σ_in W^T``.
    Equivalently, factoring ``Σ_in = L L^T`` (Cholesky, with a tiny jitter
    for PSD safety) yields ``Ỹ = W L`` with ``Ỹ Ỹ^T = W Σ_in W^T`` and
    therefore the same left singular vectors and singular values
    (up to the ``sqrt(N)`` scale factor that does not affect singular
    directions).

    Returns:
        ``W L`` of shape ``[d_out, d_in]``.
    """
    if weight.dim() != 2:
        raise ValueError(
            f"_activation_matrix_from_cov: weight must be 2-D "
            f"(got shape {tuple(weight.shape)})"
        )
    if sigma_in.dim() != 2 or sigma_in.shape[0] != sigma_in.shape[1]:
        raise ValueError(
            f"_activation_matrix_from_cov: sigma_in must be a square 2-D "
            f"matrix (got shape {tuple(sigma_in.shape)})"
        )
    if weight.shape[1] != sigma_in.shape[0]:
        raise ValueError(
            f"_activation_matrix_from_cov: weight.d_in={weight.shape[1]} "
            f"does not match sigma_in.d_in={sigma_in.shape[0]}"
        )
    sigma_f64 = sigma_in.to(torch.float64)
    # Symmetrise to absorb tiny floating-point asymmetries from cached
    # accumulators (the cached Σ_in is conceptually symmetric).
    sigma_f64 = 0.5 * (sigma_f64 + sigma_f64.transpose(0, 1))
    d_in = sigma_f64.shape[0]
    # Jitter ladder identical in spirit to ``stage3.cov_sqrt`` — try a few
    # increasing damping levels rather than failing on first non-PD attempt.
    # N2: ``.item()`` here forces a CPU↔GPU sync per call. This is
    # diagnostic-only code (one call per merge group, off the training
    # critical path), so the sync cost is negligible; future refactors
    # that lift this primitive into a hot path MUST keep the jitter on
    # the device side (e.g. compute the floor without ``.item()``).
    jitter_floor = max(
        float(sigma_f64.diagonal().abs().mean().item()) * 1e-8, 1e-10
    )
    last_exc: torch.linalg.LinAlgError | None = None
    L: torch.Tensor | None = None
    for j_mul in (1.0, 10.0, 100.0, 1_000.0, 10_000.0):
        try:
            jitter = jitter_floor * j_mul
            L = torch.linalg.cholesky(
                sigma_f64 + jitter * torch.eye(d_in, dtype=torch.float64)
            )
            break
        except torch.linalg.LinAlgError as exc:
            # N3: ``torch.linalg.cholesky`` raises this specific type
            # on non-PD inputs. Narrowing the except keeps real bugs
            # (shape mismatch, OOM, etc.) loud rather than silently
            # retrying at a higher jitter level.
            last_exc = exc
    if L is None:
        raise RuntimeError(
            f"_activation_matrix_from_cov: Cholesky failed at all jitter "
            f"levels (last error: {last_exc!r})"
        )
    return weight.to(torch.float64) @ L


# --------------------------------------------------------------------------- #
# Per-group SVC scoring                                                       #
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class SVCDonorScore:
    """Per-(donor, rank) projection coefficient for one merge group."""

    donor_expert_idx: int
    rank: int
    s_r: float


@dataclasses.dataclass
class SVCGroupResult:
    """SVC audit result for one (layer, centroid, matrix) merge group."""

    layer_idx: int
    centroid_expert_idx: int
    matrix_name: str
    donor_expert_ids: list[int]
    rank: int
    scores: list[SVCDonorScore]

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "layer_idx": self.layer_idx,
            "centroid_expert_idx": self.centroid_expert_idx,
            "matrix_name": self.matrix_name,
            "donor_expert_ids": list(self.donor_expert_ids),
            "rank": self.rank,
            "scores": [
                {
                    "donor_expert_idx": s.donor_expert_idx,
                    "rank": s.rank,
                    "s_r": s.s_r,
                }
                for s in self.scores
            ],
        }


def svc_scores_for_group(
    *,
    merged_weight: torch.Tensor,
    donor_weights: Mapping[int, torch.Tensor],
    donor_input_covariances: Mapping[int, torch.Tensor],
    rank: int,
    layer_idx: int = -1,
    centroid_expert_idx: int = -1,
    matrix_name: str = "",
) -> SVCGroupResult:
    """Compute SVC projection coefficients for one merge group.

    For each donor ``i`` in the group, reconstruct an activation matrix for
    both the merged centroid (under donor ``i``'s input distribution) and
    the pre-merge donor, take their top-``rank`` left singular vectors, and
    score each rank with Eq. 8.

    Args:
        merged_weight: ``W^merge`` of shape ``[d_out, d_in]`` — the post-Stage-2
            merged centroid weight for one matrix slot (e.g. ``gate_proj``).
        donor_weights: ``{donor_id: W^i}`` — pre-Stage-2 donor weights, all
            with the same shape as ``merged_weight``.
        donor_input_covariances: ``{donor_id: Σ_in^i}`` — per-donor input
            covariance, shape ``[d_in, d_in]``, indexed by the same keys as
            ``donor_weights``.
        rank: Number of leading singular directions to score.
        layer_idx / centroid_expert_idx / matrix_name: provenance metadata
            propagated into the result.

    Returns:
        :class:`SVCGroupResult` with one :class:`SVCDonorScore` per
        ``(donor, rank)`` pair, ordered (donor1 r=0, donor1 r=1, ..., donor2 r=0, ...).
    """
    if rank <= 0:
        raise ValueError(f"svc_scores_for_group: rank must be > 0, got {rank}")
    donor_ids = sorted(donor_weights.keys())
    if not donor_ids:
        raise ValueError("svc_scores_for_group: donor_weights is empty")
    if set(donor_ids) != set(donor_input_covariances.keys()):
        missing = set(donor_ids) - set(donor_input_covariances.keys())
        extra = set(donor_input_covariances.keys()) - set(donor_ids)
        raise ValueError(
            f"svc_scores_for_group: donor_weights / donor_input_covariances "
            f"key mismatch (missing covariances for {sorted(missing)}, "
            f"extra covariances for {sorted(extra)})"
        )

    scores: list[SVCDonorScore] = []
    for donor_id in donor_ids:
        W_donor = donor_weights[donor_id]
        sigma_donor = donor_input_covariances[donor_id]
        if W_donor.shape != merged_weight.shape:
            raise ValueError(
                f"svc_scores_for_group: donor {donor_id} weight shape "
                f"{tuple(W_donor.shape)} != merged shape "
                f"{tuple(merged_weight.shape)}"
            )
        # Both activation matrices live in the SAME input distribution
        # (donor i's Σ_in) so the comparison is apples-to-apples — we are
        # asking "on donor i's typical inputs, did the merged centroid
        # preserve donor i's top output directions?".
        Y_donor = _activation_matrix_from_cov(W_donor, sigma_donor)
        Y_merged = _activation_matrix_from_cov(merged_weight, sigma_donor)
        u_donor = _top_left_singular_vectors(Y_donor, rank=rank)  # [rank, d_out]
        u_merged = _top_left_singular_vectors(Y_merged, rank=rank)
        for r in range(rank):
            s_r = compute_projection_coefficient(
                merged_direction=u_merged[r], donor_direction=u_donor[r]
            )
            scores.append(SVCDonorScore(donor_expert_idx=donor_id, rank=r, s_r=s_r))

    return SVCGroupResult(
        layer_idx=layer_idx,
        centroid_expert_idx=centroid_expert_idx,
        matrix_name=matrix_name,
        donor_expert_ids=donor_ids,
        rank=rank,
        scores=scores,
    )


# --------------------------------------------------------------------------- #
# Artifact loaders                                                            #
# --------------------------------------------------------------------------- #


class RunIdMismatchError(RuntimeError):
    """S-2: raised when ``_stage2_partial/merge_*.json`` files disagree on
    ``stage2_run_id`` (the partial dir contains files from two different
    Stage 2 runs, or the codebase was upgraded mid-run mixing pre-S-2 and
    post-S-2 files in the same dir).

    Operator action: delete ``_stage2_partial/`` and let Stage 2 re-write a
    uniform set (cross-check is impossible on a contaminated dir).
    """


def load_merge_map(
    stage2_artifacts_dir: Path,
) -> tuple[dict[int, dict[int, list[int]]], str | None]:
    """Return ``({layer_idx: {centroid_expert_id: [donor_id, ...]}}, stage2_run_id)``.

    Tries the top-level ``merge_map.json`` (the orchestrator's final
    aggregate) first, then falls back to the per-layer
    ``_stage2_partial/merge_{layer}.json`` files (a partial run snapshot,
    same shape).

    S-2 (PLAN_S2_SVC_LOAD_MERGE_MAP.md §2.5): returns a second element —
    the ``stage2_run_id`` recovered from the on-disk artifact (or ``None``
    for pre-S-2 writers). ``svc_audit.main`` cross-checks this against the
    merged checkpoint's ``stage2_run_id`` so operator-mixed
    ``--stage2-artifacts`` and ``--merged-checkpoint`` from two different
    runs cannot silently produce garbage SVC projection coefficients.

    Aggregate path:
      * Wrapper shape (post-S-2):
        ``{"format_version": 1, "stage2_run_id": "<hex>", "merge_map": {...}}``
        → returns ``(_parse_aggregate_merge_map(inner), raw["stage2_run_id"])``.
      * Bare-dict legacy shape (pre-S-2): returns
        ``(_parse_aggregate_merge_map(raw), None)`` plus a WARN.

    Partial-dir path:
      * Aggregates per-layer merge JSONs AND collects the optional
        ``stage2_run_id`` field from each.
      * All files agree on ``stage2_run_id`` → returns ``(parsed, id)``.
      * All files are pre-S-2 (no field) → returns ``(parsed, None)`` + WARN.
      * Any two files disagree (incl. mixed pre-S-2 + post-S-2) →
        raises :class:`RunIdMismatchError`.
    """
    aggregate = stage2_artifacts_dir / "merge_map.json"
    if aggregate.exists():
        raw = json.loads(aggregate.read_text(encoding="utf-8"))
        # Wrapper-shape detection (post-S-2 writer).
        if (isinstance(raw, dict)
                and "merge_map" in raw
                and isinstance(raw["merge_map"], dict)):
            inner = raw["merge_map"]
            run_id = raw.get("stage2_run_id")
            if not isinstance(run_id, str):
                # Wrapper present but no run-id — treat as missing.
                run_id = None
            return (_parse_aggregate_merge_map(inner), run_id)
        # Legacy bare-dict shape (pre-S-2 writer).
        log.warning(
            "load_merge_map: %s is a pre-S-2 bare-dict payload "
            "(no stage2_run_id); cross-check vs --merged-checkpoint is "
            "DISABLED for this artifact set.",
            aggregate,
        )
        return (_parse_aggregate_merge_map(raw), None)

    partial_dir = stage2_artifacts_dir / "_stage2_partial"
    if partial_dir.is_dir():
        out: dict[int, dict[int, list[int]]] = {}
        # Track per-layer run-id alongside the merge groups so we can
        # detect cross-run contamination (or mixed pre-S-2 + post-S-2)
        # before returning.
        seen_run_ids: list[tuple[str, str | None]] = []
        for path in sorted(partial_dir.glob("merge_*.json")):
            layer_idx = int(path.stem.split("_")[-1])
            payload = json.loads(path.read_text(encoding="utf-8"))
            grouped = payload.get("grouped", {})
            out[layer_idx] = {
                int(centroid): [int(d) for d in donors]
                for centroid, donors in grouped.items()
            }
            rid_raw = payload.get("stage2_run_id")
            rid = rid_raw if isinstance(rid_raw, str) else None
            seen_run_ids.append((path.name, rid))
        if out:
            # Validate run-id consistency. ANY drift (incl. mixed
            # pre-S-2 + post-S-2 within the same partial dir) is a HARD
            # FAIL — that combination almost certainly means two runs
            # writing to the same dir, and we cannot prove which Stage 2
            # state produced the merged checkpoint we are auditing.
            distinct = {rid for _, rid in seen_run_ids}
            if len(distinct) > 1:
                pairs = ", ".join(
                    f"{name}={rid!r}" for name, rid in seen_run_ids
                )
                raise RunIdMismatchError(
                    f"_stage2_partial/ under {stage2_artifacts_dir} contains "
                    f"merge_*.json files with conflicting stage2_run_id "
                    f"values: {pairs}. The partial dir is contaminated with "
                    f"files from different Stage 2 runs (or mixes pre-S-2 + "
                    f"current writers — operator NOTE: if you upgraded the "
                    f"codebase mid-run, delete _stage2_partial/ before "
                    f"resuming). Cross-check is impossible on this artifact "
                    f"set."
                )
            # Single-value set: either {actual_id} or {None}.
            unified = next(iter(distinct))
            if unified is None:
                log.warning(
                    "load_merge_map: _stage2_partial/ under %s is a pre-S-2 "
                    "payload set (no stage2_run_id on any file); cross-check "
                    "vs --merged-checkpoint is DISABLED for this artifact "
                    "set.",
                    stage2_artifacts_dir,
                )
            return (out, unified)

    raise FileNotFoundError(
        f"load_merge_map: no merge_map.json under {stage2_artifacts_dir} and "
        f"no _stage2_partial/merge_*.json fallbacks either"
    )


def _load_merged_checkpoint_run_id(checkpoint_path: Path) -> str | None:
    """Return ``stage2_run_id`` from the merged checkpoint, or None for
    legacy / non-directory layouts.

    S-2 (PLAN_S2_SVC_LOAD_MERGE_MAP.md §2.6): reads
    ``<checkpoint_path>/compressed_metadata.json -> extra.stage2_run_id``.

    NOTE: ``.pt`` (state-dict) checkpoints SHORT-CIRCUIT to ``None`` here
    because they don't carry a sidecar ``compressed_metadata.json``. The
    cross-check then degrades to the legacy-WARN path (no hard fail).
    This is an ACCEPTED limitation under the current threat model: the
    ``--merged-checkpoint <foo.pt>`` operator path is rare (HF-dir is the
    canonical layout), and a determined operator who concatenates two
    runs' .pt files past the audit is outside scope. If a future audit
    upgrade needs to close this hole, embed the run-id in the .pt payload
    alongside the state_dict and add a branch here.
    """
    if not checkpoint_path.is_dir():
        return None
    meta_path = checkpoint_path / "compressed_metadata.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    extra = meta.get("extra", {})
    if isinstance(extra, dict):
        rid = extra.get("stage2_run_id")
        if isinstance(rid, str):
            return rid
    return None


def _cross_check_run_ids(
    merge_map_run_id: str | None,
    merged_run_id: str | None,
) -> int:
    """Compare run identities from --stage2-artifacts and --merged-checkpoint.

    Returns:
        0 on cross-check OK (or skipped due to legacy artifacts).
        2 on MISMATCH (operator pointed at two different runs).

    Side-effect: logs at INFO on OK, ERROR on mismatch, WARN on skip.
    Tests target this helper directly without spinning up the full audit
    pipeline.
    """
    if merge_map_run_id is not None and merged_run_id is not None:
        if merge_map_run_id != merged_run_id:
            log.error(
                "RUN IDENTITY MISMATCH: --stage2-artifacts run_id=%s but "
                "--merged-checkpoint run_id=%s. The merge map and merged "
                "weights come from DIFFERENT Stage 2 runs. SVC projection "
                "coefficients would be meaningless. Re-run with consistent "
                "paths, or delete the stale artifact set.",
                merge_map_run_id, merged_run_id,
            )
            return 2
        log.info(
            "Run identity cross-check OK: stage2_run_id=%s",
            merged_run_id,
        )
        return 0
    # Exactly one side has the field, or neither. Either case:
    #   - operator pointed at a pre-S-2 merge map + a current merged
    #     checkpoint (or vice versa), which strongly suggests they ARE
    #     from different runs but we can't prove it. WARN, don't fail.
    #   - both sides are pre-S-2 (legacy) — no cross-check possible. WARN.
    log.warning(
        "Run-identity cross-check skipped: merge_map run_id=%r, "
        "merged-checkpoint run_id=%r. One or both sides predate the "
        "S-2 run-id field (pre-S-2 Stage 2 writer, or .pt checkpoint "
        "without sidecar). Cross-run contamination is undetectable on "
        "this artifact set.",
        merge_map_run_id, merged_run_id,
    )
    return 0


def _parse_aggregate_merge_map(raw: Any) -> dict[int, dict[int, list[int]]]:
    """The Stage 2 orchestrator's aggregate format may differ slightly per
    pipeline revision; this function normalises whatever it gets into the
    ``{layer: {centroid: [donors]}}`` shape this script uses.

    Accepted shapes:
        * ``{layer_idx_str: {centroid_str: [donor_ids]}}``  (standard)
        * ``{layer_idx_str: {"grouped": {centroid_str: [donor_ids]}, ...}}``
          (verbatim per-layer JSON aggregated under the layer key).
    """
    out: dict[int, dict[int, list[int]]] = {}
    for layer_key, layer_val in raw.items():
        try:
            layer_idx = int(layer_key)
        except (TypeError, ValueError):
            # N5: surface skipped non-int keys at DEBUG so future
            # forensics can confirm whether a header / metadata key
            # (e.g. ``"schema_version"``) was silently swallowed vs a
            # real layer key dropped due to a typo.
            log.debug(
                "_parse_aggregate_merge_map: skipping non-int layer key %r",
                layer_key,
            )
            continue  # skip header / metadata keys
        if isinstance(layer_val, dict) and "grouped" in layer_val:
            grouped = layer_val["grouped"]
        else:
            grouped = layer_val
        if not isinstance(grouped, dict):
            continue
        out[layer_idx] = {
            int(centroid): [int(d) for d in donors]
            for centroid, donors in grouped.items()
        }
    if not out:
        raise ValueError(
            "_parse_aggregate_merge_map: parsed zero layers — input "
            "JSON shape is not recognised"
        )
    return out


def load_originals_snapshot(
    path: Path,
) -> dict[tuple[int, int, str], torch.Tensor]:
    """Load Stage 3's ``_stage3_original_weights.pt`` snapshot.

    Pattern O (atomic-write manifest validation) — mirrors the canonical
    Stage 4 reader at
    ``max_quality/src/moe_compress/stage4/plugins/eora_inputs.py:199-243``.
    Stage 3 writes the manifest LAST, after the ``.pt``'s fsync, so a
    torn ``.pt`` (mid-write SIGKILL on the ~50 GB payload) leaves NO
    sibling manifest. We validate manifest-first to fail loudly on a
    torn write instead of silently consuming a partial file.

    Backward-compat fallback: a ``.pt`` produced by a pre-manifest Stage 3
    writer has no manifest sibling — we accept it with a single WARNING
    (skipping validation), matching the eora_inputs.py legacy shim. Once
    all in-flight runs upgrade to a manifest-emitting writer, the
    fallback branch becomes loud-fail territory.
    """
    # LOW-5: manifest naming consistency with F-RK-1 — the canonical
    # writer appends ``.MANIFEST.json`` AFTER the payload suffix
    # (``..._weights.pt.MANIFEST.json``). The legacy
    # ``..._weights.MANIFEST.json`` (suffix-replaced) is also consulted
    # for back-compat with pre-LOW-5 Stage 3 runs. See
    # eora_inputs.py:171-184.
    manifest_path = path.with_suffix(path.suffix + ".MANIFEST.json")
    legacy_manifest_path = path.with_suffix(".MANIFEST.json")
    if not manifest_path.exists() and legacy_manifest_path.exists():
        manifest_path = legacy_manifest_path
    if manifest_path.exists():
        from moe_compress.utils.atomic_io import (
            ManifestMismatchError,
            read_and_validate_manifest,
        )
        try:
            read_and_validate_manifest(
                path,
                manifest_path,
                expected_schema_version=1,
            )
        except ManifestMismatchError as exc:
            log.error(
                "load_originals_snapshot: Stage 3 originals manifest "
                "validation FAILED for %s — %s. This is the classic "
                "torn-write signature on a ~50 GB artifact. Delete "
                "both %s and %s and re-run Stage 3.",
                path,
                exc,
                path.name,
                manifest_path.name,
            )
            raise
    else:
        # Legacy back-compat shim — pre-manifest Stage 3 writers produced
        # .pt files without sibling manifests. Same WARN-and-continue
        # contract as the canonical reader at eora_inputs.py:237-243.
        log.warning(
            "load_originals_snapshot: %s has no MANIFEST.json sibling "
            "(pre-manifest Stage 3 writer?). Proceeding without manifest "
            "validation; if torch.load errors below, the .pt may be torn — "
            "delete it and re-run Stage 3.",
            path,
        )

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(
            f"load_originals_snapshot: expected dict-payload at {path}, "
            f"got {type(payload).__name__}"
        )
    return payload  # type: ignore[return-value]


def load_input_covariance(
    path: Path,
) -> dict[tuple[int, int, str], torch.Tensor]:
    """Load ``_stage2_input_covariance.pt`` and return its sigma_in dict.

    S-2: validate the MANIFEST.json sidecar before loading the multi-GB
    .pt. Stage 2 writes the manifest LAST, after the .pt's fsync, so a
    torn .pt (mid-write SIGKILL) leaves NO sibling manifest. We validate
    manifest-first to fail loudly on a torn write instead of silently
    consuming a partial file. Mirrors load_originals_snapshot above.

    Backward-compat fallback: a ``.pt`` produced by a pre-S-2 Stage 2
    writer has no manifest sibling — we accept it with a single WARNING
    (skipping validation). Once all in-flight runs upgrade to a
    manifest-emitting writer, the fallback branch becomes loud-fail
    territory.
    """
    # S-2: Stage 2 cov has no pre-rename legacy artifacts, so unlike
    # load_originals_snapshot we do NOT consult a legacy-suffix
    # manifest path. There is exactly one manifest path:
    # ``...covariance.pt.MANIFEST.json``.
    manifest_path = path.with_suffix(path.suffix + ".MANIFEST.json")
    if manifest_path.exists():
        from moe_compress.utils.atomic_io import (
            ManifestMismatchError,
            read_and_validate_manifest,
        )
        try:
            read_and_validate_manifest(
                path,
                manifest_path,
                expected_schema_version=1,
            )
        except ManifestMismatchError as exc:
            log.error(
                "load_input_covariance: Stage 2 covariance manifest "
                "validation FAILED for %s — %s. This is the classic "
                "torn-write signature on a multi-GB artifact. Delete "
                "both %s and %s and re-run Stage 2.",
                path,
                exc,
                path.name,
                manifest_path.name,
            )
            raise
    else:
        # MEDIUM-S2 TODO(post-2026-Q3): remove this back-compat shim
        # once all sidecars under /opt/output/* are regenerated with
        # manifests. Same WARN-and-continue contract as
        # load_originals_snapshot above.
        log.warning(
            "load_input_covariance: %s has no MANIFEST.json sibling "
            "(pre-S-2 Stage 2 writer?). Proceeding without manifest "
            "validation; if torch.load errors below, the .pt may be "
            "torn — delete it and re-run Stage 2.",
            path,
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(
            f"load_input_covariance: expected dict-payload at {path}, "
            f"got {type(payload).__name__}"
        )
    if "covariance" in payload:
        return payload["covariance"]
    if "sigma_in" in payload:
        return payload["sigma_in"]
    # Bare dict-of-tensors keyed by (layer, expert, matrix) — already in the
    # right shape.
    return payload  # type: ignore[return-value]


def load_merged_expert_weights(
    checkpoint_path: Path,
) -> dict[tuple[int, int, str], torch.Tensor]:
    """Pull the merged centroid expert weights out of a post-Stage-2 checkpoint.

    Accepts either:

      * A ``state_dict.pt`` (or ``.bin``) torch file mapping HF-style key names
        like ``model.layers.{L}.mlp.experts.{E}.gate_proj.weight``.
      * A directory containing a ``state_dict.pt`` (legacy layout) or HF
        safetensors shards (``model.safetensors.index.json`` + shards).

    Returns a dict in the same key-space as the originals snapshot so the
    two can be cross-indexed directly.
    """
    if checkpoint_path.is_dir():
        index_path = checkpoint_path / "model.safetensors.index.json"
        if index_path.exists():
            return _load_from_safetensors_dir(checkpoint_path, index_path)
        legacy = checkpoint_path / "state_dict.pt"
        if legacy.exists():
            raw = torch.load(legacy, map_location="cpu", weights_only=False)
            return _normalise_hf_state_dict(raw)
        raise FileNotFoundError(
            f"load_merged_expert_weights: {checkpoint_path} is a directory "
            f"but has neither model.safetensors.index.json nor state_dict.pt"
        )

    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    # Accept either the HF flat state-dict layout (string keys) or a
    # pre-normalised tuple-keyed dict (e.g., the originals-snapshot layout)
    # for operator convenience when running on hand-prepared snapshots.
    if raw and all(
        isinstance(k, tuple) and len(k) == 3 and isinstance(v, torch.Tensor)
        for k, v in raw.items()
    ):
        return raw
    return _normalise_hf_state_dict(raw)


def _load_from_safetensors_dir(
    ckpt_dir: Path, index_path: Path
) -> dict[tuple[int, int, str], torch.Tensor]:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise ImportError(
            "load_merged_expert_weights: safetensors is required to read "
            "HF-format directories; install safetensors or point "
            "--merged-checkpoint at a state_dict.pt"
        ) from exc
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map", {})
    shard_to_keys: dict[str, list[str]] = {}
    for k, shard in weight_map.items():
        shard_to_keys.setdefault(shard, []).append(k)
    out: dict[str, torch.Tensor] = {}
    for shard, keys in shard_to_keys.items():
        with safe_open(str(ckpt_dir / shard), framework="pt", device="cpu") as f:
            for k in keys:
                out[k] = f.get_tensor(k)
    return _normalise_hf_state_dict(out)


def _normalise_hf_state_dict(
    raw: Mapping[str, torch.Tensor],
) -> dict[tuple[int, int, str], torch.Tensor]:
    """Reduce an HF-style flat state-dict to ``{(layer, expert, matrix): W}``.

    Handles both the non-fused layout (one tensor per expert per matrix,
    e.g. ``...experts.{E}.gate_proj.weight``) and the fused layout (a single
    stacked ``...experts.gate_up_proj`` of shape ``[E, 2·d_int, d_hid]`` and
    ``...experts.down_proj`` of shape ``[E, d_hid, d_int]``).

    For the fused layout we split ``gate_up_proj`` along dim 1 into
    ``gate_proj`` (first ``d_int``) and ``up_proj`` (remaining ``d_int``)
    to align with ``MATRIX_NAMES = (gate_proj, up_proj, down_proj)`` and
    the originals-snapshot convention.

    Each per-expert weight ``W`` returned has shape ``[d_out, d_in]``
    (PyTorch nn.Linear convention).
    """
    out: dict[tuple[int, int, str], torch.Tensor] = {}
    for key, tensor in raw.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        if ".mlp.experts." not in key and ".experts." not in key:
            continue

        # Non-fused: ...layers.{L}.mlp.experts.{E}.{matrix}.weight
        parts = key.split(".")
        if "experts" in parts:
            ei = parts.index("experts")
        else:
            continue
        # The token after "experts" is either an expert index (non-fused)
        # or a matrix name like "gate_up_proj" / "down_proj" (fused).
        if ei + 1 >= len(parts):
            continue
        nxt = parts[ei + 1]
        if "layers" not in parts:
            continue
        layer_idx = int(parts[parts.index("layers") + 1])

        if nxt.isdigit():
            # Non-fused.
            expert_idx = int(nxt)
            if ei + 2 >= len(parts):
                continue
            matrix_name = parts[ei + 2]
            if matrix_name.endswith("_proj"):
                # weight tensor for nn.Linear has shape [out, in] already.
                out[(layer_idx, expert_idx, matrix_name)] = tensor.detach()
        else:
            # Fused. ``nxt`` came from ``key.split(".")``, so it cannot
            # itself end in ``.weight`` (a trailing ``.weight`` would be
            # a separate part). The earlier ``_weight`` / ``.weight``
            # suffix-strip branches were dead code (L3) and have been
            # removed. The trailing ``.weight`` part, when present, is
            # already filtered by the ``endswith("_proj")`` test below.
            fused_name = nxt
            if fused_name == "gate_up_proj":
                num_experts = tensor.shape[0]
                d_int_x2 = tensor.shape[1]
                d_int = d_int_x2 // 2
                for e in range(num_experts):
                    out[(layer_idx, e, "gate_proj")] = tensor[e, :d_int, :].detach()
                    out[(layer_idx, e, "up_proj")] = tensor[e, d_int:, :].detach()
            elif fused_name == "down_proj":
                num_experts = tensor.shape[0]
                for e in range(num_experts):
                    out[(layer_idx, e, "down_proj")] = tensor[e].detach()
    if not out:
        raise ValueError(
            "_normalise_hf_state_dict: parsed zero expert weights — the "
            "input state-dict shape is not recognised"
        )
    return out


# --------------------------------------------------------------------------- #
# Driver                                                                      #
# --------------------------------------------------------------------------- #


def run_audit(
    *,
    originals: Mapping[tuple[int, int, str], torch.Tensor],
    merged: Mapping[tuple[int, int, str], torch.Tensor],
    input_cov: Mapping[tuple[int, int, str], torch.Tensor],
    merge_map: Mapping[int, Mapping[int, Sequence[int]]],
    rank: int,
    matrix_names: Sequence[str] = ("gate_proj", "up_proj", "down_proj"),
) -> list[SVCGroupResult]:
    """Compute SVC scores for every non-singleton merge group.

    Singleton groups (centroid is its own only donor) are skipped — the
    projection coefficient is trivially 1 by construction.
    """
    results: list[SVCGroupResult] = []
    skipped_groups = 0
    for layer_idx in sorted(merge_map.keys()):
        for centroid_id, donor_ids in merge_map[layer_idx].items():
            donor_list = list(donor_ids)
            if len(donor_list) <= 1:
                skipped_groups += 1
                continue
            for matrix in matrix_names:
                merged_key = (layer_idx, centroid_id, matrix)
                if merged_key not in merged:
                    log.warning(
                        "run_audit: missing merged weight for %s — skipping",
                        merged_key,
                    )
                    continue
                # Gather donor weights + covariances; tolerate per-donor
                # cache misses by skipping the donor (rather than the whole
                # group) and noting it.
                donor_weights: dict[int, torch.Tensor] = {}
                donor_covs: dict[int, torch.Tensor] = {}
                for donor_id in donor_list:
                    wk = (layer_idx, donor_id, matrix)
                    if wk not in originals:
                        log.warning(
                            "run_audit: missing originals[%s]", wk
                        )
                        continue
                    if wk not in input_cov:
                        log.warning(
                            "run_audit: missing input_cov[%s]", wk
                        )
                        continue
                    donor_weights[donor_id] = originals[wk]
                    donor_covs[donor_id] = input_cov[wk]
                if not donor_weights:
                    continue
                merged_W = merged[merged_key]
                # Clip the requested rank to what the smaller of the
                # donor / merged matrices can support — diagnostic should
                # never crash on a thin matrix; we simply report fewer
                # ranks for that slot.
                effective_rank = min(
                    rank,
                    merged_W.shape[0],
                    min(W.shape[0] for W in donor_weights.values()),
                    min(C.shape[0] for C in donor_covs.values()),
                )
                if effective_rank < 1:
                    continue
                group_result = svc_scores_for_group(
                    merged_weight=merged_W,
                    donor_weights=donor_weights,
                    donor_input_covariances=donor_covs,
                    rank=effective_rank,
                    layer_idx=layer_idx,
                    centroid_expert_idx=centroid_id,
                    matrix_name=matrix,
                )
                results.append(group_result)
    log.info(
        "run_audit: %d scored groups, %d singleton groups skipped",
        len(results),
        skipped_groups,
    )
    return results


# --------------------------------------------------------------------------- #
# Reporting                                                                   #
# --------------------------------------------------------------------------- #


def write_results_json(results: Sequence[SVCGroupResult], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # L1: filter non-finite ``s_r`` values BEFORE serialisation. Standard
    # JSON has no NaN/Infinity literals; ``json.dumps`` defaults to
    # emitting the JavaScript-only ``NaN`` / ``Infinity`` tokens that
    # downstream strict parsers (e.g. browsers, jq --strict, many BI
    # tools) will refuse. Replacing with ``None`` keeps the document
    # spec-valid and makes the missing-sample visible to consumers
    # without bloating the schema with a per-sample status flag.
    sanitised: list[dict[str, Any]] = []
    dropped = 0
    for r in results:
        rec = r.to_jsonable()
        clean_scores: list[dict[str, Any]] = []
        for s in rec["scores"]:
            s_val = s["s_r"]
            if isinstance(s_val, float) and not math.isfinite(s_val):
                dropped += 1
                s = {**s, "s_r": None}
            clean_scores.append(s)
        rec["scores"] = clean_scores
        sanitised.append(rec)
    if dropped:
        log.warning(
            "write_results_json: %d non-finite s_r samples replaced with "
            "null (likely degenerate-spectrum SVD or empty-donor edge case)",
            dropped,
        )
    payload = {
        "format_version": 1,
        "audit": "svc_audit",
        "paper_reference": "arXiv:2602.05536 Eq. 8 (clean-room re-impl)",
        "results": sanitised,
    }
    # allow_nan=False: belt-and-suspenders — if a non-float NaN sneaks
    # past the filter above, the encoder raises rather than silently
    # emitting non-spec JSON.
    from moe_compress.utils.atomic_io import atomic_write_text

    atomic_write_text(
        out_path,
        json.dumps(payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def write_summary_markdown(
    results: Sequence[SVCGroupResult],
    out_path: Path,
    *,
    over_count_threshold: float = 1.3,
    dropped_threshold: float = 0.1,
) -> None:
    """Per-layer summary table.

    Columns per (layer, matrix):
      * #groups   — number of non-singleton merge groups scored.
      * #ranks    — total (donor, rank) projection-coefficient samples.
      * mean s    — mean projection coefficient.
      * #over     — count of samples with ``|s_r^i| > over_count_threshold``
                    (spectral over-counting; the upstream paper flag).
      * #dropped  — count of samples with ``|s_r^i| < dropped_threshold``
                    (REAM effectively dropped that donor in that subspace).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Aggregate by (layer, matrix).
    buckets: dict[tuple[int, str], list[float]] = {}
    group_counts: dict[tuple[int, str], int] = {}
    for r in results:
        key = (r.layer_idx, r.matrix_name)
        group_counts[key] = group_counts.get(key, 0) + 1
        buckets.setdefault(key, []).extend(s.s_r for s in r.scores)

    lines: list[str] = []
    lines.append("# SVC Audit — REAM Merge Spectral Diagnostics")
    lines.append("")
    lines.append(
        "Projection coefficient `s_r^i = <a_r^merge, a_r^i> / ||a_r^i||^2` "
        f"per arXiv:2602.05536 Eq. 8. Over-counting threshold "
        f"|s| > {over_count_threshold:.2f}; dropped-donor threshold "
        f"|s| < {dropped_threshold:.2f}."
    )
    lines.append("")
    lines.append("| Layer | Matrix | #Groups | #(donor, rank) | mean s | #over | #dropped |")
    lines.append("|------:|:-------|--------:|---------------:|-------:|------:|---------:|")
    for key in sorted(buckets.keys()):
        layer_idx, matrix = key
        scores = buckets[key]
        n = len(scores)
        mean_s = sum(scores) / n if n else float("nan")
        over = sum(1 for s in scores if abs(s) > over_count_threshold)
        dropped = sum(1 for s in scores if abs(s) < dropped_threshold)
        lines.append(
            f"| {layer_idx} | {matrix} | {group_counts[key]} | {n} | "
            f"{mean_s:.4f} | {over} | {dropped} |"
        )
    if not buckets:
        lines.append("| — | — | 0 | 0 | n/a | 0 | 0 |")
    lines.append("")
    from moe_compress.utils.atomic_io import atomic_write_text

    atomic_write_text(out_path, "\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="svc_audit",
        description=(
            "Diagnostic SVC projection-coefficient audit for REAM merges "
            "(arXiv:2602.05536 Eq. 8, output-activation-space variant)."
        ),
    )
    parser.add_argument(
        "--stage2-artifacts",
        type=Path,
        required=True,
        help="Directory containing merge_map.json (and optionally "
        "_stage2_input_covariance.pt and _stage2_partial/).",
    )
    parser.add_argument(
        "--originals-pt",
        type=Path,
        default=None,
        help="Path to _stage3_original_weights.pt (default: "
        "<stage2-artifacts>/../stage3/_stage3_original_weights.pt). "
        "If absent, the script aborts with an actionable error.",
    )
    parser.add_argument(
        "--merged-checkpoint",
        type=Path,
        required=True,
        help="Post-Stage-2 merged checkpoint (HF dir or state_dict.pt).",
    )
    parser.add_argument(
        "--input-cov-pt",
        type=Path,
        default=None,
        help="Path to _stage2_input_covariance.pt (default: "
        "<stage2-artifacts>/_stage2_input_covariance.pt).",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=8,
        help="Number of leading singular directions to score per group "
        "(default: 8).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("audit/spec_compliance"),
        help="Directory for svc_audit_results.json + svc_audit_summary.md.",
    )
    parser.add_argument(
        "--matrix-names",
        nargs="+",
        default=["gate_proj", "up_proj", "down_proj"],
        help="Matrix slots to score (default: all three Qwen3 MoE expert "
        "matrices).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=args.log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )

    # L2: mirror the actionable-error pattern used for missing
    # --originals-pt / --input-cov-pt below — a missing
    # ``--stage2-artifacts`` directory must produce a clear error +
    # exit code 2, not an uncaught FileNotFoundError stack trace.
    try:
        merge_map, merge_map_run_id = load_merge_map(args.stage2_artifacts)
    except FileNotFoundError as exc:
        log.error(
            "Stage 2 merge map not found under %s — pass "
            "--stage2-artifacts pointing at a directory that contains "
            "merge_map.json (or _stage2_partial/merge_*.json). %s",
            args.stage2_artifacts,
            exc,
        )
        return 2
    except RunIdMismatchError as exc:
        # S-2 (PLAN_S2_SVC_LOAD_MERGE_MAP.md §2.5): hard-fail when
        # _stage2_partial/ contains files from two different Stage 2 runs.
        log.error("%s", exc)
        return 2
    log.info("Loaded merge_map for %d layers", len(merge_map))

    originals_path = args.originals_pt or (
        args.stage2_artifacts.parent / "stage3" / "_stage3_original_weights.pt"
    )
    if not originals_path.exists():
        log.error(
            "Originals snapshot not found at %s — pass --originals-pt "
            "with the right path (Stage 3 must have run, or partial-run "
            "stage3 just for the snapshot).",
            originals_path,
        )
        return 2
    originals = load_originals_snapshot(originals_path)
    log.info("Loaded %d originals entries from %s", len(originals), originals_path)

    cov_path = args.input_cov_pt or (
        args.stage2_artifacts / "_stage2_input_covariance.pt"
    )
    if not cov_path.exists():
        log.error(
            "Input covariance not found at %s — pass --input-cov-pt or "
            "re-run Stage 2 with input-covariance capture enabled.",
            cov_path,
        )
        return 2
    input_cov = load_input_covariance(cov_path)
    log.info("Loaded %d input_cov entries from %s", len(input_cov), cov_path)

    merged = load_merged_expert_weights(args.merged_checkpoint)
    log.info("Loaded %d merged expert weights from %s", len(merged), args.merged_checkpoint)

    # S-2 (PLAN_S2_SVC_LOAD_MERGE_MAP.md §2.6): cross-check that
    # ``--stage2-artifacts`` and ``--merged-checkpoint`` come from the
    # SAME Stage 2 run. The threat model is "operator pointed at the wrong
    # dir by accident"; UUID-equality (constant-time, no large-file
    # hashing) is the right tool. Exit code 2 on MISMATCH matches the
    # other "can't proceed" branches above.
    merged_run_id = _load_merged_checkpoint_run_id(args.merged_checkpoint)
    rc = _cross_check_run_ids(merge_map_run_id, merged_run_id)
    if rc != 0:
        return rc

    results = run_audit(
        originals=originals,
        merged=merged,
        input_cov=input_cov,
        merge_map=merge_map,
        rank=args.rank,
        matrix_names=tuple(args.matrix_names),
    )

    json_path = args.output_dir / "svc_audit_results.json"
    md_path = args.output_dir / "svc_audit_summary.md"
    write_results_json(results, json_path)
    write_summary_markdown(results, md_path)
    log.info("Wrote %s and %s", json_path, md_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
