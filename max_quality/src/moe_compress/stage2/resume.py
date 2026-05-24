"""Stage 2 partial-dir discovery for crash-resume.

Paper / spec source
--------------------
No paper. Project-original durability + crash-resume contract,
covering Stage 2's per-layer atomic checkpointing.

Durability invariants (project-wide; all stages):

  - **Atomic writes via ``.tmp + os.replace``** — every artifact is
    written to ``<name>.tmp`` then renamed in one syscall to ``<name>``.
    A crash mid-write leaves at most one ``.tmp`` file (cleaned up on
    the next run start) and never a half-written ``<name>``.
  - **`.pt` before `.json` ordering invariant** — for any layer, the
    binary checkpoint (``layer_{idx}.pt``) must be persisted to disk
    BEFORE the manifest JSON (``merge_{idx}.json``). A crash between
    the two leaves an orphan ``.pt`` file with no JSON — orphans are
    safe to delete (no manifest = no completed-layer claim). The
    inverse ordering would risk a JSON claim with a missing/corrupt
    ``.pt``.
  - **Strict format-version match on resume** — the resume code
    refuses to mix v1 and v2 partial directories within a single run.
    Operators upgrading mid-pipeline must finish a stage on one
    version or restart cleanly.

What this module owns: the *file IO half* of resume — scan
``partial_dir``, delete orphan ``layer_*.pt`` files whose
``merge_*.json`` is absent, parse each completed layer's JSON,
optionally load the per-expert neuron-mean artefact, and return one
``ResumedLayerRecord`` per completed layer.

What stays in :mod:`stage2.plugins.layer_merge`: the *model-mutation
half* — re-applying the merge in-place, resizing the router, reloading
covariance + heal weights into the live model.

Original module-header note: extracted from
``stage2_reap_ream.run()`` in Task 2 of the plugin-architecture
refactor.

The *model-mutation half* (re-applying the merge in-place, resizing the router,
reloading covariance + heal weights into the live model) stays in
``stage2_reap_ream.run()`` for now — Task 4 will own the merge engine and may
absorb it then.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ..utils.activation_hooks import ReamCostAccumulator
from ..utils.model_io import MoELayerRef

log = logging.getLogger(__name__)


@dataclass
class ResumedLayerRecord:
    """One completed layer discovered in ``partial_dir``.

    Carries every field the in-``run()`` re-application loop needs to mutate
    the live model — but does NOT itself mutate anything. ``resume_ream_acc``
    is None when the legacy partial-dir predates the neuron-means artefact
    (caller logs a loud ERROR and falls back to weight-only alignment).
    """

    layer_ref: MoELayerRef
    final_kept_ids: list[int]
    grouped: dict[int, list[int]]
    freq: dict[int, int]
    merge_map_layer: dict[int, list[int]]
    mean_cost_per_pair: float | None
    has_heal_weights_file: bool
    resume_ream_acc: ReamCostAccumulator | None
    # Forensic fields parsed for completeness; not consumed by the re-application
    # loop today but exposed so future tasks / debug tooling can read them.
    assignment_solver_used: str = "greedy"
    cost_alignment_used: str = "pre"
    em_rounds_completed: int = 0
    distill_state: Any = None
    heal_state: Any = None


def discover_completed_layers(
    partial_dir: Path,
    moe_layers: list[MoELayerRef],
    *,
    heal_enabled: bool,
) -> list[ResumedLayerRecord]:
    """Discover layers already completed in a prior interrupted Stage 2 run.

    Side effects (intentional, mirror the legacy in-``run()`` behaviour):
      * ``*.tmp`` files in ``partial_dir`` are deleted (stale tmp from a crash).
      * Any ``layer_*.pt`` without a matching ``merge_*.json`` is deleted —
        a ``.pt`` without ``.json`` means the process died between
        ``_snapshot_cov_layer`` and ``_write_merge_json``; reprocessing the
        ``.pt`` would double-remap (silent numerical corruption).

    Validates each ``merge_*.json`` against ``format_version=2`` and the
    contiguous-key invariants for ``freq`` / ``merge_map_layer``. Loads the
    optional ``_neuron_means_layer*.pt`` into a fresh ``ReamCostAccumulator``
    when present; logs and falls back to weight-only when absent.

    Does NOT mutate the live model — the caller is responsible for replaying
    each record's merge through ``_merge_experts_inplace`` /
    ``_resize_router_for_kept_experts`` / ``cov_acc.load_layer_from_disk`` /
    ``_load_heal_weights``.
    """
    for _stale in partial_dir.glob("*.tmp"):
        _stale.unlink(missing_ok=True)

    # Crash safety: delete any .pt whose matching .json is absent.
    # A .pt without .json means the process died between _snapshot_cov_layer
    # and _write_merge_json. The covariance has been remapped but not recorded.
    # Reprocessing the .pt would double-remap — silent numerical corruption.
    for ref in moe_layers:
        pt_path = partial_dir / f"layer_{ref.layer_idx}.pt"
        json_path = partial_dir / f"merge_{ref.layer_idx}.json"
        if pt_path.exists() and not json_path.exists():
            log.warning(
                "Stage 2 resume: orphaned %s (no matching JSON) — "
                "deleting and reprocessing layer %d",
                pt_path.name, ref.layer_idx,
            )
            pt_path.unlink()

    records: list[ResumedLayerRecord] = []
    for ref in moe_layers:
        merge_path = partial_dir / f"merge_{ref.layer_idx}.json"
        cov_path = partial_dir / f"layer_{ref.layer_idx}.pt"
        if not (merge_path.exists() and cov_path.exists()):
            if merge_path.exists() and not cov_path.exists():
                log.warning("layer %d: found merge JSON but missing covariance .pt; re-running layer", ref.layer_idx)
            continue
        data = json.loads(merge_path.read_text())
        fv = int(data.get("format_version", 0))
        if fv != 2:
            # Stage 2 v2 (spec § 12.1): format_version bumped 1 → 2 to
            # accommodate the new assignment_solver_used / em_rounds_completed
            # / distill_state fields. NO backward-compat shim — see
            # ALGORITHM_REFERENCE.md § 11. Operators upgrading mid-pipeline
            # must finish a stage on one version or restart cleanly.
            raise RuntimeError(
                f"_stage2_partial/merge_{ref.layer_idx}.json has format_version={fv} "
                "(expected 2) — delete _stage2_partial/ and re-run Stage 2"
            )
        # Migration guard: old partial dirs wrote "centroid_ids"; new ones
        # write "final_kept_ids". Accept both for backward compatibility.
        if "final_kept_ids" in data:
            final_kept_ids = [int(x) for x in data["final_kept_ids"]]
        elif "centroid_ids" in data:
            log.warning(
                "Stage 2 resume layer %d: found deprecated 'centroid_ids' field "
                "(expected 'final_kept_ids') — using it for backward compatibility. "
                "Delete _stage2_partial/ to regenerate with the new format.",
                ref.layer_idx,
            )
            final_kept_ids = [int(x) for x in data["centroid_ids"]]
        else:
            raise RuntimeError(
                f"_stage2_partial/merge_{ref.layer_idx}.json missing both "
                "'final_kept_ids' and 'centroid_ids' keys — file is corrupt. "
                "Delete _stage2_partial/ and re-run Stage 2."
            )
        grouped = {int(k): list(v) for k, v in data["grouped"].items()}
        freq = {int(k): int(v) for k, v in data["freq"].items()}
        _freq_keys = set(freq.keys())
        if _freq_keys != set(range(len(freq))):
            raise RuntimeError(
                f"Layer {ref.layer_idx}: loaded freq has non-contiguous or unexpected keys — "
                "delete partial checkpoint and re-run from scratch"
            )
        merge_map_layer = {int(k): list(v) for k, v in data["merge_map_layer"].items()}

        _keys = set(merge_map_layer.keys())
        if _keys != set(range(len(merge_map_layer))):
            raise RuntimeError(
                f"Layer {ref.layer_idx}: loaded merge_map has non-contiguous keys {_keys} — "
                "delete partial checkpoint and re-run from scratch"
            )
        if any(not v for v in merge_map_layer.values()):
            raise RuntimeError(
                f"Layer {ref.layer_idx}: loaded merge_map has empty member lists — "
                "delete partial checkpoint and re-run from scratch"
            )

        # n_pre_merge is derived from len(freq) rather than a dedicated persisted
        # field. This is safe because freq is written with exactly one key per
        # expert (range(n_experts)) at calibration time, so len(freq) always equals
        # the original expert count for this layer before any merging.
        n_pre_merge = len(freq)
        if ref.num_routed_experts != n_pre_merge:
            raise RuntimeError(
                f"Stage 2 resume layer {ref.layer_idx}: expected {n_pre_merge} "
                f"experts (pre-merge) but model has {ref.num_routed_experts}. "
                "The model passed to stage2.run() must be the Stage 1 output, "
                "not a partially-merged model."
            )

        # B-iter5-M-2: try to load persisted per-expert neuron-mean tensors so
        # resume reproduces the spec invariant C = C_wt + C_act (D5b). If the
        # file is missing (legacy partial dirs predating this fix), fall back
        # to weight-only alignment with a louder warning.
        resume_ream_acc: ReamCostAccumulator | None = None
        neuron_means_path = partial_dir / f"_neuron_means_layer{ref.layer_idx}.pt"
        if neuron_means_path.exists():
            try:
                nm_payload = torch.load(neuron_means_path, map_location="cpu", weights_only=False)
                nm_fv = int(nm_payload.get("format_version", 0))
                if nm_fv != 1:
                    raise RuntimeError(
                        f"_stage2_partial/_neuron_means_layer{ref.layer_idx}.pt "
                        f"has format_version={nm_fv} (expected 1)"
                    )
                resume_ream_acc = ReamCostAccumulator()
                # Restore per-expert neuron means by re-creating sum/count pairs
                # such that get_neuron_mean returns the saved mean (sum/count == mean).
                # Setting count=1 and sum=mean is the simplest restoration that
                # preserves equality and avoids serializing both tensors.
                for eid_str, mean_tensor in nm_payload["neuron_means"].items():
                    eid = int(eid_str)
                    resume_ream_acc._neuron_act_sum[(ref.layer_idx, eid)] = mean_tensor.clone()
                    resume_ream_acc._neuron_act_count[(ref.layer_idx, eid)] = 1
            except Exception as _exc:
                log.warning(
                    "layer %d (resume): failed to load neuron-mean artifact (%s); "
                    "falling back to weight-only permutation alignment",
                    ref.layer_idx, _exc,
                )
                resume_ream_acc = None
        if resume_ream_acc is None:
            log.error(
                "layer %d (resume): neuron-mean activation data not available on resume — "
                "permutation alignment uses weight-only cost (C_gate + C_up, no C_act). "
                "Merged weights WILL differ from a fresh run. Delete _stage2_partial/ "
                "to regenerate from scratch with the new artifact format.",
                ref.layer_idx,
            )

        has_heal_weights_file = (
            heal_enabled
            and (partial_dir / f"_heal_weights_layer_{ref.layer_idx}.pt").exists()
        )

        records.append(ResumedLayerRecord(
            layer_ref=ref,
            final_kept_ids=final_kept_ids,
            grouped=grouped,
            freq=freq,
            merge_map_layer=merge_map_layer,
            mean_cost_per_pair=data.get("mean_cost_per_pair"),
            has_heal_weights_file=has_heal_weights_file,
            resume_ream_acc=resume_ream_acc,
            assignment_solver_used=data.get("assignment_solver_used", "greedy"),
            cost_alignment_used=data.get("cost_alignment_used", "pre"),
            em_rounds_completed=int(data.get("em_rounds_completed", 0)),
            distill_state=data.get("distill_state"),
            heal_state=data.get("heal_state"),
        ))

    return records
