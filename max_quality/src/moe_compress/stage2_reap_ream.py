"""Stage 2 — REAP scoring + REAM pseudo-pruning, fused-experts-aware.

Key differences from the pre-refactor version:
  - Weights live in stacked tensors on ``Qwen3_5MoeExperts``; pruning means
    slicing those tensors and the router's ``gate.weight`` rows.
  - Scoring hooks go through :func:`instrument_experts` which monkey-patches
    the fused forward with per-expert callbacks.
  - Input covariance for Stage 3 is collected on two tap points:
      ``"gate_proj"``   → covariance used by gate_proj + up_proj SVD
      ``"down_proj"``   → covariance used by down_proj SVD
    Keys match those used by ``InputCovarianceAccumulator`` (``'gate_proj'``
    covers gate+up projections; ``'down_proj'`` covers the down projection).
    We save these under the (layer, expert, matrix_name) key space that
    Stage 3 consumes.

REAM cost matrix (paper 2604.04356, reference ream/ream.py):
  - δ_gate (Eq. 5): similarity ∈ [0,1] between L2-row-normalized pre-softmax
    gate logit profile vectors — Euclidean distance converted via dist2sim.
  - δ̃_expert (Eq. 8): mean cosine similarity of expert outputs (sparse top-k
    approximation; see `compute_delta_expert` in `activation_hooks.py`),
    rescaled to [0,1] via (cosine+1)/2.
  - δ_REAM = (δ_gate + δ̃_expert) / 2 ∈ [0,1]; cost = 1 − δ_REAM.
  - Grouping: single-pass greedy procedure matching paper §4 exactly (descending
    centroid saliency, absorb up to C nearest unassigned non-centroids per centroid).
    The paper prescribes greedy, not optimal matching; this is spec-compliant.
    Full assignment guaranteed by upfront feasibility check.

Frequency-weighted merge with neuron permutation alignment is preserved.
"""
from __future__ import annotations

import gc
import json
import logging
import math
import os
import shutil
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp

# Stage 2 v2 — solver dispatch literal. Adding new solvers requires
# updating both this Literal AND the if/else chain in
# _assign_children_to_centroids. Keep them in sync.
SolverName = Literal["greedy", "hungarian", "mcf", "auto", "sinkhorn"]

from .utils.activation_hooks import (
    InputCovarianceAccumulator,
    ReamCostAccumulator,
    ReapAccumulator,
    _EarlyExitException,
    capture_router_outputs,
    early_exit_after_layer,
    instrument_experts,
    record_reap,
)
from .utils.calibration import build_calibration_tensor, iter_batches, spec_from_config
from .utils.model_io import (
    MATRIX_NAMES,
    MoELayerRef,
    build_banks,
    iter_moe_layers,
    load_json_artifact,
    save_compressed_checkpoint,
    save_json_artifact,
)
from .utils.runtime_monitor import snapshot_telemetry as _rt_snap, update as _rt_update
from .utils.trackio_log import trackio_log as _trackio_log

log = logging.getLogger(__name__)


# ===========================================================================
# Stage-2 per-layer merge-heal (opt-in; inert when merge_heal_enabled=False)
# ---------------------------------------------------------------------------
# After Stage 2 merges a layer, optionally fine-tune that layer's kept experts
# (+ optionally its resized router) by SELF-DISTILLATION: the layer is trained
# to reproduce its OWN pre-merge MoE-block output on calibration tokens. The
# (input, target) pairs are captured in-process from the layer itself just
# before the merge — no teacher object, no disk sidecar, no cascade buffer.
#
# Every code path below is reached ONLY when `merge_heal_enabled` is True. With
# the flag off, none of this runs and Stage 2 behaviour is byte-identical to
# the pre-feature code.
# ===========================================================================

# Per-layer healed-weight checkpoint schema version.
_HEAL_WEIGHTS_FORMAT_VERSION = 2


class _HealConfig:
    """Validated, frozen view of the ``merge_heal_*`` config knobs.

    Built once at the top of :func:`run` from the ``stage2_reap_ream`` config
    block. All knobs default to OFF/inert values; ``enabled`` gates every
    merge-heal code path so a config that omits the block (or sets
    ``merge_heal_enabled: false``) reproduces the pre-feature behaviour.
    """

    __slots__ = (
        "enabled", "train_router", "lr", "adamw_betas", "grad_clip",
        "holdout_fraction", "patience", "eval_interval", "max_steps",
        "token_cap", "minibatch_size",
    )

    def __init__(self, s2: dict) -> None:
        self.enabled: bool = bool(s2.get("merge_heal_enabled", False))
        self.train_router: bool = bool(s2.get("merge_heal_train_router", True))
        self.lr: float = float(s2.get("merge_heal_lr", 1.0e-4))
        _betas = s2.get("merge_heal_adamw_betas", [0.9, 0.95])
        # This guard runs unconditionally (not gated on self.enabled) because
        # the (float(_betas[0]), float(_betas[1])) unpack below is itself
        # unconditional — a scalar or a string would otherwise crash there with
        # an opaque TypeError / produce float("0")-style nonsense.
        if not isinstance(_betas, (list, tuple)) or len(_betas) != 2:
            raise ValueError(
                f"stage2_reap_ream.merge_heal_adamw_betas={_betas}; "
                "must be a length-2 list [beta1, beta2] of exactly two "
                "numeric values."
            )
        self.adamw_betas: tuple[float, float] = (float(_betas[0]), float(_betas[1]))
        self.grad_clip: float = float(s2.get("merge_heal_grad_clip", 1.0))
        self.holdout_fraction: float = float(s2.get("merge_heal_holdout_fraction", 0.10))
        self.patience: int = int(s2.get("merge_heal_patience", 5))
        self.eval_interval: int = int(s2.get("merge_heal_eval_interval", 25))
        self.max_steps: int = int(s2.get("merge_heal_max_steps", 2000))
        self.token_cap: int = int(s2.get("merge_heal_token_cap", 262144))
        self.minibatch_size: int = int(
            s2.get("merge_heal_minibatch_size", 8192)
        )

        if not self.enabled:
            return

        # Validate only when the feature is active so a disabled run with a
        # partially-specified block never fails.
        if self.grad_clip <= 0.0:
            raise ValueError(
                f"stage2_reap_ream.merge_heal_grad_clip={self.grad_clip}; "
                "must be > 0."
            )
        if self.lr <= 0.0:
            raise ValueError(
                f"stage2_reap_ream.merge_heal_lr={self.lr}; must be > 0."
            )
        for _idx, _beta in enumerate(self.adamw_betas):
            if not (0.0 <= _beta < 1.0):
                raise ValueError(
                    f"stage2_reap_ream.merge_heal_adamw_betas[{_idx}]={_beta}; "
                    "must be in [0.0, 1.0)."
                )
        if not (0.0 < self.holdout_fraction < 1.0):
            raise ValueError(
                f"stage2_reap_ream.merge_heal_holdout_fraction="
                f"{self.holdout_fraction}; must be in (0.0, 1.0)."
            )
        for _name, _val in (
            ("merge_heal_patience", self.patience),
            ("merge_heal_eval_interval", self.eval_interval),
            ("merge_heal_max_steps", self.max_steps),
            ("merge_heal_token_cap", self.token_cap),
            ("merge_heal_minibatch_size", self.minibatch_size),
        ):
            if _val < 1:
                raise ValueError(
                    f"stage2_reap_ream.{_name}={_val}; must be >= 1."
                )
        # The held-out eval (and hence accept/reject + save-best) only fires
        # at multiples of eval_interval. If eval_interval > max_steps it never
        # runs, so every layer would be silently REJECTED — fail fast instead.
        if self.eval_interval > self.max_steps:
            raise ValueError(
                f"stage2_reap_ream.merge_heal_eval_interval={self.eval_interval} "
                f"exceeds merge_heal_max_steps={self.max_steps}; the held-out "
                "eval would never run and every layer would be rejected."
            )


def run(
    model,
    tokenizer,
    config: dict,
    artifacts_dir: Path,
    *,
    device=None,
    stage1_budget_path: Path | None = None,
    no_resume: bool = False,
) -> Path:
    s2 = config["stage2_reap_ream"]
    cal = config["calibration"]

    # Blackwell sm_100 workaround: transformers' default MoE forward uses
    # `torch.nn.functional.grouped_mm`, which deadlocks on B200 partway
    # through Stage 2 (reproduced as a 2-min main-thread hang then SIGSEGV
    # at layer 13 batch ~60 on Qwen3.6-35B-A3B, 2026-05-13). Stages 5 and 6
    # already force `batched_mm` via `_set_experts_implementation` — Stage 2
    # was the only path still going through the broken kernel. Mirror the
    # same override here: env var `EXPERTS_IMPLEMENTATION` wins, then the
    # YAML knob `stage2_reap_ream.experts_implementation`, default
    # `batched_mm`. See memory/project_grouped_mm_blackwell.md.
    from .stage5_router_kd import _set_experts_implementation
    _experts_impl = os.environ.get(
        "EXPERTS_IMPLEMENTATION", s2.get("experts_implementation", "batched_mm")
    )
    _set_experts_implementation(model, _experts_impl)

    # Stage 2 v2 (spec § 6 / D-asymmetric-freq): cost_asymmetric is valid
    # only under freq-weighted merge — the asymmetric factor freq_m/(freq_c+freq_m)
    # is the per-pair version of the merge weight. Reject the combination
    # at the very top of `run` so misconfigured pipelines fail fast before
    # spending compute on calibration / Stage-1 artifact loading.
    if bool(s2.get("cost_asymmetric", False)) and not s2["ream"]["frequency_weighted_merge"]:
        raise ValueError(
            "stage2_reap_ream.cost_asymmetric=True requires "
            "ream.frequency_weighted_merge=True (spec § 5 step 4T(c)(iii) "
            "/ D-asymmetric-freq)."
        )

    if stage1_budget_path is None:
        stage1_budget_path = artifacts_dir / "stage1_budgets.json"
    budgets_payload = load_json_artifact(stage1_budget_path)
    per_layer_target = {
        int(k): int(v) for k, v in budgets_payload["per_layer_target_experts"].items()
    }
    blacklist_payload = load_json_artifact(artifacts_dir / "stage1_blacklist.json")
    blacklist = {int(k): list(v) for k, v in blacklist_payload.get("blacklist", {}).items()}

    spec = spec_from_config(cal, num_sequences_override=s2["num_calibration_samples"])
    calib = build_calibration_tensor(
        tokenizer, spec, cache_dir=(os.environ.get("MOE_CALIB_CACHE_DIR") or (artifacts_dir / "_calibration_cache"))
    )
    batches = iter_batches(calib, batch_size=s2["batch_size"])
    assert isinstance(batches, list), "iter_batches must return a list for multi-pass re-iteration"

    moe_layers = list(iter_moe_layers(model))
    cov_acc = InputCovarianceAccumulator()
    # Spec §5 "Covariance Side-Collection": FP32 storage is recommended by
    # Swift-SVD paper 2604.01609 (avoids numerical degradation in eigendecomposition);
    # the dtype is configurable via covariance_storage_dtype.
    # Default fp16 per §12 D-cov-storage-fp16 (10 mantissa bits, half the
    # disk vs fp32, no measurable downstream PPL drift on Qwen3-30B-A3B).
    # Production config also pins this to fp16; the default is here for
    # config-omitted invocations.
    cov_dtype = getattr(torch, s2.get("covariance_storage_dtype", "float16"))
    cov_acc.set_storage_dtype(cov_dtype)
    merge_map: dict[int, dict[int, list[int]]] = {}

    # -----------------------------------------------------------------------
    # Stage-2 per-layer merge-heal setup (opt-in; inert when disabled).
    # When `merge_heal_enabled` is False, heal_cfg.enabled is False and every
    # heal code path below is skipped — Stage 2 behaviour is byte-identical to
    # the pre-feature code (no capture, no heal). When enabled, each layer's
    # (input, target) pairs are captured in-process just before the merge —
    # no teacher object, no sidecar, no cascade buffer.
    # -----------------------------------------------------------------------
    heal_cfg = _HealConfig(s2)
    _heal_device: torch.device | None = None
    if heal_cfg.enabled:
        _heal_device = device
        if _heal_device is None:
            try:
                _heal_device = next(model.parameters()).device
            except StopIteration:
                _heal_device = torch.device("cpu")
        log.info(
            "Stage-2 merge-heal enabled: self-distillation, "
            "pool=%d tokens/layer, train_router=%s, lr=%g",
            heal_cfg.token_cap, heal_cfg.train_router, heal_cfg.lr,
        )

    # -----------------------------------------------------------------------
    # Crash-resume: scan partial_dir for layers already completed in a prior
    # interrupted run. Re-apply merges in layer order (fast, no forward pass).
    # -----------------------------------------------------------------------
    completed_layers: set[int] = set()
    _layer_mean_costs: list[float] = []  # running history for cost-threshold gate (Strategy C)

    if no_resume:
        partial_dir = None
        # Delete stale partial dir so a future non-no-resume run cannot resume
        # from this run's incomplete (or absent) checkpoints.
        stale = artifacts_dir / "_stage2_partial"
        if stale.exists():
            shutil.rmtree(stale, ignore_errors=True)
    else:
        partial_dir = artifacts_dir / "_stage2_partial"
        partial_dir.mkdir(parents=True, exist_ok=True)
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
            _merge_experts_inplace(ref, grouped, freq,
                                   freq_weighted=s2["ream"]["frequency_weighted_merge"],
                                   ream_acc=resume_ream_acc)
            # build_banks again: _merge_experts_inplace already called it internally, but
            # bank.select() was never called on any of those banks, so the _last_kept_ids_*
            # sentinel is still unset and this select() call is safe.
            banks = build_banks(ref)
            for bank in banks.values():
                bank.select(final_kept_ids)
            _resize_router_for_kept_experts(ref, final_kept_ids)

            # Merge-heal resume: a completed layer's post-heal weights are not
            # reconstructible from merge_*.json — reload them so the in-memory
            # model matches the state the heal left. A 0-merge layer has no
            # heal-weights file (the heal is skipped, heal_state stays None);
            # its banks are already fully reconstructed above, so gate the load
            # on the file existing — mirror the _write_heal_weights condition.
            if heal_cfg.enabled and (
                partial_dir / f"_heal_weights_layer_{ref.layer_idx}.pt"
            ).exists():
                _load_heal_weights(partial_dir, ref, final_kept_ids)

            try:
                cov_acc.load_layer_from_disk(ref.layer_idx, partial_dir)
            except Exception as _exc:
                raise RuntimeError(
                    f"Stage 2 resume: failed to load covariance for layer {ref.layer_idx} "
                    f"from _stage2_partial/ ({_exc}). "
                    "The in-memory model has already been partially mutated — "
                    "restart with a fresh Stage 1 model and delete _stage2_partial/."
                ) from _exc
            merge_map[ref.layer_idx] = merge_map_layer
            completed_layers.add(ref.layer_idx)
            log.info("Stage 2: layer %d resumed from partial (skipping profile + merge)",
                     ref.layer_idx)
            val = data.get("mean_cost_per_pair")
            if val is not None and val > 0.0:
                _layer_mean_costs.append(float(val))

        if completed_layers:
            log.info("Stage 2: resumed %d / %d layers from %s",
                     len(completed_layers), len(moe_layers), partial_dir)

    # B-C-H-1: default to 8 (D5a value) so REAM merging always has a per-centroid
    # cap, preventing degenerate one-centroid-absorbs-all groupings. Setting to 0
    # explicitly disables the cap (uncapped path; rare, ablation-only); users must
    # opt in to that. The `or 8` collapse from earlier was dropped so an explicit
    # `0` in config is honored as "uncapped" rather than silently overridden to 8.
    max_group_cap: int = int(s2.get("max_merge_group_size", 8))
    # B-iter5-L-1 (code): default to 1.5 (D-ream-budget-bump value) so the quality
    # gate is active out-of-the-box; setting to a very large value (e.g. inf) in
    # config disables the gate.
    cost_sigma: float = s2.get("ream_cost_sigma_threshold", 1.5)
    cost_bump_ratio: float = s2.get("ream_cost_bump_ratio", 0.10)
    min_active_tokens: int = s2.get("reap_min_active_tokens", 0)
    # Stage 2 v2 — assignment solver dispatch. Default "greedy" reproduces v1
    # behavior exactly. See max_quality/docs/stage2_assignment_revision.md § 6.
    # Validate the YAML value at the boundary so typos are caught at the
    # config-load site rather than at the per-layer call.
    _solver_value = str(s2.get("assignment_solver", "greedy")).lower()
    _valid_solvers = ("greedy", "hungarian", "mcf", "auto", "sinkhorn")
    if _solver_value not in _valid_solvers:
        raise ValueError(
            f"stage2_reap_ream.assignment_solver={_solver_value!r} is not a "
            f"valid solver name; expected one of {_valid_solvers}."
        )
    assignment_solver: SolverName = _solver_value  # type: ignore[assignment]

    # Stage 2 v2 cost matrix variants (spec § 5 step 4 / § 6).
    # Direction C adds "output": an output-space merge cost that measures the
    # actual change in the layer's gated routed-expert output on calibration
    # tokens when a child is tentatively merged into a centroid. A strictly
    # better merge-damage proxy than the weight-space "pre"/"post" costs.
    cost_alignment_cfg: str = str(s2.get("cost_alignment", "pre")).lower()
    if cost_alignment_cfg not in ("pre", "post", "output"):
        raise ValueError(
            f"stage2_reap_ream.cost_alignment={cost_alignment_cfg!r}; "
            "expected 'pre', 'post', or 'output'."
        )
    # Direction C — output-space cost calibration-token cap. Only consumed when
    # cost_alignment == "output"; bounds the per-pair SwiGLU residual compute.
    cost_output_token_cap: int = int(s2.get("cost_output_token_cap", 1024))
    if cost_output_token_cap < 1:
        raise ValueError(
            f"stage2_reap_ream.cost_output_token_cap={cost_output_token_cap}; "
            "must be >= 1 (number of calibration tokens for the output cost)."
        )
    cost_whitening: str = str(s2.get("cost_whitening", "none")).lower()
    if cost_whitening not in ("none", "diag", "full"):
        raise ValueError(
            f"stage2_reap_ream.cost_whitening={cost_whitening!r}; "
            "expected 'none', 'diag', or 'full'."
        )
    cost_asymmetric: bool = bool(s2.get("cost_asymmetric", False))
    cost_topk_filter: int = int(s2.get("cost_topk_filter", 48))
    capacity_util_threshold: float = float(s2.get("capacity_util_threshold", 0.25))
    em_refinement_rounds: int = int(s2.get("em_refinement_rounds", 0))
    em_convergence_break: bool = bool(s2.get("em_convergence_break", True))
    # Direction D — greedy + 2-opt local refinement. Only active when
    # assignment_solver == "greedy"; strictly-improving so it cannot regress.
    two_opt_refine: bool = bool(s2.get("two_opt_refine", False))
    sinkhorn_epsilon_init: float = float(s2.get("sinkhorn_epsilon_init", 1.0))
    sinkhorn_epsilon_final: float = float(s2.get("sinkhorn_epsilon_final", 0.01))
    sinkhorn_iters: int = int(s2.get("sinkhorn_iters", 200))
    # Direction B — skip-merge floor. Per-layer percentile P over the *finite*
    # entries of the cost matrix; every entry strictly above P is masked to
    # +inf so those pairs fall through to orphan promotion (singleton kept
    # experts). OFF sentinel: 100.0 (the 100th percentile is the max finite
    # cost, so nothing is strictly above it -> no entry masked -> byte-identical
    # to the unmasked run). Valid range [0.0, 100.0].
    skip_merge_percentile: float = float(s2.get("skip_merge_percentile", 100.0))
    if not (0.0 <= skip_merge_percentile <= 100.0):
        raise ValueError(
            f"stage2_reap_ream.skip_merge_percentile={skip_merge_percentile}; "
            "must be in [0.0, 100.0] (100.0 = off, mask nothing)."
        )
    if em_refinement_rounds < 0:
        raise ValueError(
            f"stage2_reap_ream.em_refinement_rounds={em_refinement_rounds}; "
            "must be >= 0 (set 0 to disable)."
        )
    # Phase 3 (M8): per-merge-group expert distillation flags.
    expert_distill_steps: int = int(s2.get("expert_distill_steps", 0))
    expert_distill_lr: float = float(s2.get("expert_distill_lr", 1e-4))
    _betas_raw = s2.get("expert_distill_betas", [0.9, 0.95])
    expert_distill_betas: tuple[float, float] = (float(_betas_raw[0]), float(_betas_raw[1]))
    expert_distill_token_cap: int = int(s2.get("expert_distill_token_cap", 8192))
    expert_distill_skip_singletons: bool = bool(s2.get("expert_distill_skip_singletons", True))
    expert_distill_plateau_steps: int = int(s2.get("expert_distill_loss_plateau_steps", 50))
    expert_distill_plateau_eps: float = float(s2.get("expert_distill_loss_plateau_eps", 1e-4))
    if expert_distill_steps < 0:
        raise ValueError(
            f"stage2_reap_ream.expert_distill_steps={expert_distill_steps}; "
            "must be >= 0 (set 0 to disable)."
        )
    # cost_asymmetric × freq_weighted_merge invariant is checked at the very
    # top of run() (fail-fast); we rely on that here.

    # Stage 2 v2 (spec § 6) — one-shot Trackio emit of the static config so
    # the dashboard run-summary reflects which features are active without
    # parsing per-layer logs. All v2 config flags + the partial-JSON
    # format_version are surfaced under the "stage2/config/*" namespace.
    _trackio_log({
        "stage2/config/assignment_solver": assignment_solver,
        "stage2/config/cost_alignment": cost_alignment_cfg,
        "stage2/config/cost_whitening": cost_whitening,
        "stage2/config/cost_asymmetric": cost_asymmetric,
        "stage2/config/cost_topk_filter": cost_topk_filter,
        "stage2/config/capacity_util_threshold": capacity_util_threshold,
        "stage2/config/em_refinement_rounds": em_refinement_rounds,
        "stage2/config/em_convergence_break": em_convergence_break,
        "stage2/config/two_opt_refine": two_opt_refine,
        "stage2/config/expert_distill_steps": expert_distill_steps,
        "stage2/config/expert_distill_token_cap": expert_distill_token_cap,
        "stage2/config/expert_distill_lr": expert_distill_lr,
        "stage2/config/sinkhorn_iters": sinkhorn_iters,
        "stage2/config/format_version": 2,
    })

    for k, layer_ref in enumerate(moe_layers):
        if layer_ref.layer_idx in completed_layers:
            log.info(
                "Stage 2 layer %d/%d (idx=%d) — skipped (resumed from partial)",
                k + 1, len(moe_layers), layer_ref.layer_idx,
            )
            continue

        target = per_layer_target[layer_ref.layer_idx]
        log.info(
            "Stage 2 layer %d/%d (idx=%d) — profiling then merging to %d experts",
            k + 1, len(moe_layers), layer_ref.layer_idx, target,
        )
        reap_acc = ReapAccumulator()
        ream_acc = ReamCostAccumulator()  # fresh accumulator per layer; discarded after this layer's pass
        # Stage 2 v2 (M1): cache (perm, residual) per (layer, centroid, noncentroid)
        # so the cost-matrix builder and merge step share Hungarian alignments.
        # Cleared at the start of every layer.
        perm_cache = _PermAlignCache()
        # Phase 3 (M8): capture layer-input hidden states only when
        # per-expert distillation is enabled, to keep host-RAM cost zero
        # for runs that don't use the feature.
        # Direction C: the output-space cost (cost_alignment == "output") also
        # needs the layer-input calibration tokens, so the accumulator is
        # likewise enabled in that mode. When BOTH are active the buffer must
        # be large enough for the larger consumer. The accumulator stays None
        # (no capture, no host-RAM cost) for every "pre"/"post" run — keeping
        # those paths byte-identical to main.
        _need_layer_inputs = expert_distill_steps > 0 or cost_alignment_cfg == "output"
        _layer_input_cap = (
            max(
                expert_distill_token_cap if expert_distill_steps > 0 else 0,
                cost_output_token_cap if cost_alignment_cfg == "output" else 0,
            )
            if _need_layer_inputs
            else 0
        )
        layer_input_acc = (
            _LayerInputAccumulator(
                max_samples=_layer_input_cap,
                seed=layer_ref.layer_idx,  # per-layer seed for bit-reproducibility
            )
            if _need_layer_inputs
            else None
        )
        torch.cuda.empty_cache()
        _profile_layer(
            model, layer_ref, batches, reap_acc, cov_acc, ream_acc,
            device=device,
            layer_input_acc=layer_input_acc,
        )
        # These two finalize calls are independent of each other and could be
        # parallelised (e.g., via concurrent.futures) if profiling shows this
        # is a bottleneck in future.
        reap_acc.finalize_layer(layer_ref.layer_idx)
        cov_acc.finalize_layer(layer_ref.layer_idx)

        n_experts = layer_ref.num_routed_experts
        protected = set(blacklist.get(layer_ref.layer_idx, []))
        scores = np.array([reap_acc.score(layer_ref.layer_idx, e) for e in range(n_experts)])
        freq = {e: reap_acc.freq.get((layer_ref.layer_idx, e), 0) for e in range(n_experts)}

        # Protected experts (super experts + shared experts from stage1_blacklist.json)
        # are completely excluded from REAM — not centroids, not non-centroids.
        # Their weights pass through Stage 2 unchanged (spec §5 "Blacklisted Expert Exclusion").
        n_protected = len(protected)

        if target > n_experts:
            raise RuntimeError(
                f"Layer {layer_ref.layer_idx}: budget target {target} > n_experts {n_experts}; "
                "budget allocation is inconsistent with layer expert count"
            )
        if target == n_experts:
            log.warning(
                "layer %d: budget target (%d) equals total expert count (%d) — "
                "no merging will occur; check budget configuration.",
                layer_ref.layer_idx, target, n_experts,
            )

        effective_target = target
        ream_centroid_ids: list[int] = []
        ream_noncentroid_ids: list[int] = []
        grouped: dict[int, list[int]] = {}
        delta = np.empty((0, 0))
        assignment: list[int] = []
        running_mean: float = float("nan")
        em_rounds_done: int = 0  # populated by _em_refine_assignment in the bump loop
        # Stage 2 v2: hoist effective_cost_alignment / effective_cost_asymmetric
        # from the bump-loop's "if not b_fail" branch to layer scope so the
        # per-layer Trackio emit at the bottom of the loop sees them whether
        # or not the bump loop's success branch ran (b_fail / zero-merge
        # fallback leaves the defaults as-is, which is the right thing to
        # log: "no cost matrix was actually built for this layer"). Same for
        # capacity_util_value — defaults to 0.0 (uncapped / fully-slack).
        effective_cost_alignment: str = cost_alignment_cfg
        effective_cost_asymmetric: bool = cost_asymmetric
        capacity_util_value: float = 0.0
        mean_assigned_cost: float = 0.0
        assigned_cost: float = 0.0
        # Invariant: after the bump loop, assignment is either:
        #   (a) a list of length len(ream_noncentroid_ids) with centroid indices (normal path), or
        #   (b) [] with ream_noncentroid_ids also [] (zero-merge fallback path).
        # (c) c_fail last-resort: assignment holds the last above-threshold assignment
        #     (len == len(ream_noncentroid_ids)); applied as best-available merge below.
        # b_fail / c_fail are initialized here so the post-loop fallback check never raises
        # NameError if the range were somehow empty.
        b_fail: bool = False
        c_fail: bool = False
        _warned_ream_target_zero: bool = False

        _original_ream_target = max(effective_target - n_protected, 0)  # target on first attempt

        # Loop runs (1 + n_experts - target) times: 1 initial attempt plus up to
        # (n_experts - target) bumps, one per additional kept expert.
        for _bump_attempt in range(n_experts - target + 1):
            # F1 fix: reset em_rounds_done per bump iteration so the value
            # persisted in the partial JSON reflects the iteration whose
            # assignment is actually committed (not a stale value from a
            # prior bump iteration).
            em_rounds_done = 0
            # REAM centroid count = total target minus the protected slots.
            ream_target = max(effective_target - n_protected, 0)

            if ream_target == 0:
                if not _warned_ream_target_zero:
                    log.warning(
                        "layer %d: ream_target=0 — all %d non-protected experts will be dropped "
                        "(budget fully consumed by %d protected experts); "
                        "check budget configuration.",
                        layer_ref.layer_idx, n_experts - len(protected), len(protected),
                    )
                    _warned_ream_target_zero = True
                break

            # Select top-ream_target non-protected experts by REAP score (descending).
            # This is the greedy centroid selection order: highest-saliency centroid
            # gets priority in the assignment pass (spec §5 Step 3).
            ream_centroid_ids = []
            for _e in np.argsort(-scores):
                if len(ream_centroid_ids) >= ream_target:
                    break
                e = int(_e)
                if e in protected:
                    continue
                if freq[e] < min_active_tokens:
                    # Spec D-reap-min-active-tokens (§12): low-frequency experts
                    # are filtered from centroid candidacy; they become
                    # non-centroids and get merged via Hungarian alignment.
                    continue
                ream_centroid_ids.append(e)

            if len(ream_centroid_ids) < ream_target:
                log.warning(
                    "  layer %d: REAM centroid selection yielded %d < %d — "
                    "%d candidate(s) filtered by reap_min_active_tokens=%d "
                    "(per spec D-reap-min-active-tokens)",
                    layer_ref.layer_idx, len(ream_centroid_ids), ream_target,
                    ream_target - len(ream_centroid_ids), min_active_tokens,
                )

            ream_centroid_set = set(ream_centroid_ids)
            ream_noncentroid_ids = [
                e for e in range(n_experts)
                if e not in protected and e not in ream_centroid_set
            ]

            n_ream_c  = len(ream_centroid_ids)
            n_ream_nc = len(ream_noncentroid_ids)

            # Feasibility check (spec §5 Step 3, reference ream/ream.py L60-62):
            # every non-centroid must be absorbable within the per-centroid cap.
            b_fail = (max_group_cap > 0) and (n_ream_nc > n_ream_c * max_group_cap)

            delta = np.empty((0, 0))
            assignment = []
            mean_cost = 0.0
            c_fail = False

            if not b_fail:
                # Stage 2 v2 capacity-utilization gate (M3, spec § 5 step 3):
                #   u = n_NC / (N'_l × C_max). When u < threshold, the layer
                #   has so much slack capacity that the heavyweight
                #   post-alignment cost matrix is unlikely to change the
                #   assignment meaningfully — fall back to the cheap symmetric
                #   path. This is what skips ~half the layers' compute.
                # Capture the actual u value into the layer-scope variable
                # so the per-layer Trackio emit can surface it; mirrors the
                # division done inside _pick_effective_alignment.
                if max_group_cap <= 0:
                    capacity_util_value = 0.0
                else:
                    capacity_util_value = n_ream_nc / max(n_ream_c * max_group_cap, 1)
                effective_cost_alignment = _pick_effective_alignment(
                    n_nc=n_ream_nc,
                    n_c=n_ream_c,
                    max_group_cap=max_group_cap,
                    threshold=capacity_util_threshold,
                    configured=cost_alignment_cfg,
                )
                effective_cost_asymmetric = (
                    cost_asymmetric and effective_cost_alignment == "post"
                )
                delta = _ream_cost_matrix(
                    layer_ref, ream_noncentroid_ids, ream_centroid_ids,
                    ream_acc=ream_acc,
                    blacklisted_ids=protected,
                    cost_alignment=effective_cost_alignment,
                    cost_whitening=cost_whitening,
                    cost_asymmetric=effective_cost_asymmetric,
                    cost_topk_filter=cost_topk_filter,
                    freq=(
                        freq
                        if (effective_cost_asymmetric
                            or effective_cost_alignment == "output")
                        else None
                    ),
                    cov_acc=cov_acc if effective_cost_alignment == "post" else None,
                    perm_cache=perm_cache,
                    # Direction C: calibration tokens for the output-space cost.
                    # None for "pre"/"post" — those paths never read it.
                    layer_inputs=(
                        layer_input_acc.get()
                        if (effective_cost_alignment == "output"
                            and layer_input_acc is not None)
                        else None
                    ),
                    output_token_cap=cost_output_token_cap,
                )
                # Direction B — skip-merge floor. Mask high-cost pairs to +inf
                # so they fall through to orphan promotion. When the flag is at
                # its OFF sentinel (100.0) this is a no-op on delta's values.
                if skip_merge_percentile < 100.0:
                    delta, _n_skip_masked = _apply_skip_merge_floor(
                        delta, skip_merge_percentile,
                    )
                    if _n_skip_masked > 0:
                        log.info(
                            "layer %d: skip-merge floor (P%.1f) masked %d/%d "
                            "cost entries to +inf — affected children fall "
                            "through to orphan promotion",
                            layer_ref.layer_idx, skip_merge_percentile,
                            _n_skip_masked, delta.size,
                        )
                assignment = _assign_children_to_centroids(
                    delta, n_ream_nc, n_ream_c, max_group_cap,
                    solver=assignment_solver,
                    sinkhorn_epsilon_init=sinkhorn_epsilon_init,
                    sinkhorn_epsilon_final=sinkhorn_epsilon_final,
                    sinkhorn_iters=sinkhorn_iters,
                )
                # Direction D — greedy + 2-opt local refinement (spec §5 step 3.5).
                # Strictly-improving local search; runs only for the greedy solver
                # and only when the flag is set. It cannot regress vs. the greedy
                # assignment, so the EM step below still sees a feasible input.
                if two_opt_refine and assignment_solver == "greedy":
                    assignment = _two_opt_refine(
                        assignment, delta, max_group_cap,
                    )
                elif two_opt_refine:
                    log.warning(
                        "two_opt_refine=true is ignored: it only applies to the "
                        "greedy assignment solver, but assignment_solver=%r.",
                        assignment_solver,
                    )
                # Stage 2 v2 EM refinement (spec § 5 step 4T(e) / M4).
                # Runs only when cost_alignment == "post": "pre" is a no-op
                # (cost is centroid-independent) and "output" is deferred (EM
                # would help but needs layer_inputs threaded — see
                # _em_refine_assignment). It guards on this internally.
                assignment, delta, em_rounds_done = _em_refine_assignment(
                    layer_ref,
                    initial_assignment=assignment,
                    initial_delta=delta,
                    skip_merge_percentile=skip_merge_percentile,
                    ream_centroid_ids=ream_centroid_ids,
                    ream_noncentroid_ids=ream_noncentroid_ids,
                    perm_cache=perm_cache,
                    ream_acc=ream_acc,
                    cov_acc=cov_acc if effective_cost_alignment == "post" else None,
                    freq=freq,
                    max_group_cap=max_group_cap,
                    cost_alignment=effective_cost_alignment,
                    cost_whitening=cost_whitening,
                    cost_asymmetric=effective_cost_asymmetric,
                    cost_topk_filter=cost_topk_filter,
                    assignment_solver=assignment_solver,
                    em_rounds=em_refinement_rounds,
                    em_break=em_convergence_break,
                    blacklisted_ids=protected,
                    sinkhorn_epsilon_init=sinkhorn_epsilon_init,
                    sinkhorn_epsilon_final=sinkhorn_epsilon_final,
                    sinkhorn_iters=sinkhorn_iters,
                )
                _iter_n_assigned = sum(1 for a in assignment if a >= 0)
                _iter_assigned_cost = (
                    sum(float(delta[ch, assignment[ch]])
                        for ch in range(n_ream_nc) if assignment[ch] >= 0)
                    if delta.size > 0 else 0.0
                )
                if _iter_n_assigned == 0 and n_ream_nc == 0:
                    # No non-centroid experts exist — nothing to merge, cost is
                    # genuinely zero.  Skip the c_fail gate entirely: there is no
                    # merge to gate on, and inf would cause a spurious bump.
                    mean_cost = 0.0
                    # c_fail remains False (already set above); do not evaluate gate.
                else:
                    # When nothing was assigned despite having non-centroids, use inf
                    # rather than 0.0: a zero mean_cost would be a false negative,
                    # making an unassigned layer look cheaper than any real merge and
                    # preventing the cost-threshold bump from triggering.
                    mean_cost = (
                        _iter_assigned_cost / _iter_n_assigned
                        if _iter_n_assigned > 0 else float("inf")
                    )
                    # Require at least 4 prior-layer samples before applying the cost-sigma
                    # gate: fewer samples make the running mean too noisy to be meaningful.
                    # Invariant: running_mean is always computed in the same branch as
                    # c_fail = True, so running_mean is guaranteed to be set before
                    # c_fail can become True. Future refactors must preserve this ordering
                    # to avoid referencing running_mean when it is still 0.0 (its default).
                    if len(_layer_mean_costs) >= 4:
                        running_mean = float(np.mean(_layer_mean_costs))
                        c_fail = mean_cost > running_mean * (1.0 + cost_sigma)

            if not b_fail and not c_fail:
                break

            # Spec D-ream-budget-bump: BOTH gates use the same bump formula
            # max(1, ceil(effective_target * cost_bump_ratio)) — applies to
            # feasibility (b_fail) AND quality (c_fail) gates uniformly.
            # Previously the ratio was only applied on c_fail, making
            # b_fail-only iterations bump by exactly 1 (slow convergence).
            bump = max(1, math.ceil(effective_target * cost_bump_ratio))
            new_effective = min(effective_target + bump, n_experts)
            if b_fail:
                log.warning(
                    "  layer %d: infeasible (ream_c=%d × cap=%d < nc=%d) — "
                    "bumping target %d→%d",
                    layer_ref.layer_idx, n_ream_c, max_group_cap, n_ream_nc,
                    effective_target, new_effective,
                )
            # running_mean is always current here: c_fail=True can only be set inside the
            # cost block (not b_fail path), which assigns running_mean before setting c_fail.
            if c_fail:
                assert not math.isnan(running_mean), (
                    "running_mean must be set before c_fail can be True; "
                    "check that the c_fail assignment is co-located with the running_mean assignment"
                )
                log.warning(
                    "  layer %d: mean_cost=%.4f > threshold=%.4f — bumping target %d→%d",
                    layer_ref.layer_idx, mean_cost,
                    running_mean * (1.0 + cost_sigma),
                    effective_target, new_effective,
                )
            effective_target = new_effective
            # We break BEFORE computing a new assignment at effective_target==n_experts;
            # the last assignment from the previous iteration is used as the fallback.
            if effective_target >= n_experts:
                break

        # Post-loop: if the loop exited because effective_target >= n_experts but c_fail
        # was still True (cost gate never cleared), the last above-threshold assignment
        # is used as last resort. Warn so this silent state is observable.
        if c_fail and effective_target >= n_experts:
            log.warning(
                "REAM layer %d: bump loop exhausted (c_fail=True, b_fail=%s, effective_target=%d >= n_experts=%d); "
                "applying above-threshold assignment as last resort",
                layer_ref.layer_idx, b_fail, effective_target, n_experts,
            )
        # Fallback: if the bump loop exhausted without achieving feasibility
        # (b_fail still True and no assignment was built), log a WARNING and fall back
        # to keeping all non-protected experts as centroids (zero merges). This is the
        # safest fallback — it produces the least compression but loses no expert weights.

        # Zero-target case: budget fully consumed by protected experts — no REAM
        # centroids or non-centroids should exist and no merges should be produced.
        # The bump loop broke out early, so ream_centroid_ids/ream_noncentroid_ids/
        # assignment may still hold stale values from a previous attempt (or their
        # initial [] defaults). Reset them explicitly so the grouping code below
        # produces an empty grouped dict and all protected experts flow to final_kept_ids.
        if _original_ream_target == 0:
            ream_centroid_ids = []
            ream_noncentroid_ids = []
            assignment = []
            delta = np.empty((0, 0))
            b_fail = False
            c_fail = False

        # When b_fail: assignment is [] (reset at iteration top; b_fail skips _assign_children_to_centroids).
        # When c_fail last-resort (effective_target >= n_experts break): assignment holds the last
        # computed above-threshold result and is intentionally applied in the grouping step below.
        if b_fail and ream_noncentroid_ids:
            log.warning(
                "  layer %d: bump loop exhausted (effective_target=%d == n_experts=%d) "
                "without achieving feasibility — falling back to zero-merge "
                "(all non-protected experts kept as centroids). "
                "No expert weights are lost, but compression target is not met.",
                layer_ref.layer_idx, effective_target, n_experts,
            )
            # Explicitly set ream_centroid_ids to all non-protected experts (zero-merge
            # fallback). We cannot rely on the last bump iteration's ream_centroid_ids
            # because the loop broke before recomputing it with the final effective_target.
            ream_centroid_ids = [
                e for e in range(n_experts) if e not in protected
            ]
            ream_noncentroid_ids = []
            assignment = []
            delta = np.empty((0, 0))

        if not ream_centroid_ids and ream_noncentroid_ids and _original_ream_target > 0:
            log.warning(
                "REAM layer %d: no centroids selected (all non-protected experts may have failed "
                "min_active_tokens or cost gate); promoting all non-protected experts to singleton "
                "centroids (zero-merge fallback).",
                layer_ref.layer_idx,
            )
            ream_centroid_ids = list(ream_noncentroid_ids)
            ream_noncentroid_ids = []
            assignment = []
            delta = np.empty((0, 0))

        # Build REAM merge groups (keyed by REAM centroid only — protected experts
        # are not in grouped and their weights are not touched by _merge_experts_inplace).
        grouped = {c: [c] for c in ream_centroid_ids}
        # Protected experts should never appear as REAM centroids; verify the invariant.
        _protected_centroids = [eid for eid in protected if eid in grouped]
        if _protected_centroids:
            raise RuntimeError(
                f"Layer {layer_ref.layer_idx}: protected expert(s) {_protected_centroids} "
                "appeared as REAM centroids — invariant violated"
            )
        for child_pos, centroid_pos in enumerate(assignment):
            if centroid_pos >= 0:
                grouped[ream_centroid_ids[centroid_pos]].append(
                    ream_noncentroid_ids[child_pos]
                )

        for child_pos, centroid_pos in enumerate(assignment):
            if centroid_pos < 0:
                # Unassigned non-centroid: promote to singleton centroid to avoid weight loss.
                orphan_eid = ream_noncentroid_ids[child_pos]
                log.warning(
                    "layer %d: non-centroid expert %d unassigned in capped grouping — "
                    "promoted to singleton centroid to avoid weight loss",
                    layer_ref.layer_idx, orphan_eid,
                )
                grouped[orphan_eid] = [orphan_eid]
                ream_centroid_ids.append(orphan_eid)
        ream_centroid_ids = sorted(set(ream_centroid_ids))

        assigned_cost = (
            sum(float(delta[ch, assignment[ch]])
                for ch in range(len(ream_noncentroid_ids)) if assignment[ch] >= 0)
            if delta.size > 0 else 0.0
        )
        n_assigned = sum(1 for a in assignment if a >= 0)
        mean_assigned_cost = assigned_cost / max(n_assigned, 1)

        # Guard mirrors the resume-path condition (val > 0.0): exclude zero costs
        # so that layers with all-zero pair costs don't bias the running mean low
        # and suppress the cost-sigma bump gate for subsequent layers.
        # Also exclude last-resort c_fail assignments (bump loop exhausted with
        # effective_target >= n_experts) — those costs would inflate the running mean
        # and progressively suppress the c_fail gate for subsequent layers.
        if n_assigned > 0 and mean_assigned_cost > 0.0 and not (c_fail and effective_target >= n_experts):
            _layer_mean_costs.append(mean_assigned_cost)

        # Stage-2 merge-heal: capture this layer's pre-merge (input, target)
        # pairs for self-distillation. Done BEFORE _merge_experts_inplace,
        # while layer_ref.mlp is still its original self — so the captured
        # output is the self-distillation target. Skipped for 0-merge layers:
        # a layer that merged nothing is unchanged, so the heal would be a
        # guaranteed no-op the accept/reject guard rejects anyway.
        cap_in: torch.Tensor | None = None
        cap_out: torch.Tensor | None = None
        layer_merged = any(len(m) > 1 for m in grouped.values())
        if heal_cfg.enabled and layer_merged:
            cap_in, cap_out = _capture_mlp_io(
                model, layer_ref, batches,
                device=_heal_device, pool_size=heal_cfg.token_cap,
            )

        # Phase 3 (M8): snapshot pre-merge expert weights BEFORE the merge
        # mutates the bank. The snapshot is consumed only by the per-group
        # distillation step below; released as soon as that finishes for
        # this layer (Python GC since no module-level reference is held).
        pre_merge_weights: dict[int, dict[str, torch.Tensor]] | None = None
        if expert_distill_steps > 0:
            pre_merge_weights = _snapshot_pre_merge_layer_experts(layer_ref)

        _merge_experts_inplace(
            layer_ref, grouped, freq,
            freq_weighted=s2["ream"]["frequency_weighted_merge"],
            ream_acc=ream_acc,
            perm_cache=perm_cache,
        )

        # Phase 3 (M8): per-merge-group expert distillation (spec § 5 step 7b).
        distill_state: dict[int, dict] | None = None
        if expert_distill_steps > 0 and pre_merge_weights is not None:
            layer_inputs_buf = (
                layer_input_acc.get() if layer_input_acc is not None else None
            )
            if layer_inputs_buf is None or layer_inputs_buf.shape[0] == 0:
                log.warning(
                    "layer %d: expert distillation enabled but no layer-input "
                    "samples were captured during profile — skipping.",
                    layer_ref.layer_idx,
                )
            else:
                distill_state = {}
                target_device = layer_ref.layer_module.parameters().__next__().device
                for centroid, members in grouped.items():
                    if expert_distill_skip_singletons and len(members) <= 1:
                        continue
                    state = _distill_merged_group(
                        layer_ref=layer_ref,
                        centroid_id=centroid,
                        members=members,
                        freq=freq,
                        pre_merge_weights=pre_merge_weights,
                        layer_inputs=layer_inputs_buf,
                        steps=expert_distill_steps,
                        lr=expert_distill_lr,
                        betas=expert_distill_betas,
                        plateau_steps=expert_distill_plateau_steps,
                        plateau_eps=expert_distill_plateau_eps,
                        token_cap=expert_distill_token_cap,
                        device=target_device,
                    )
                    distill_state[centroid] = state
                log.info(
                    "  layer %d distillation: %d non-singleton groups distilled",
                    layer_ref.layer_idx, len(distill_state),
                )

        # Final kept set = protected experts (untouched) + REAM centroids (post-merge).
        # Protected experts' rows are preserved in gate.weight and expert tensors.
        final_kept_ids = sorted(list(protected) + ream_centroid_ids)

        if not final_kept_ids:
            raise RuntimeError(
                f"Layer {layer_ref.layer_idx}: final_kept_ids is empty after merge — "
                "target may be inconsistent with protected/blacklisted expert counts"
            )

        banks = build_banks(layer_ref)
        for bank in banks.values():
            bank.select(final_kept_ids)
        _resize_router_for_kept_experts(layer_ref, final_kept_ids)

        # Stage-2 per-layer merge-heal (opt-in). Heal this layer's kept
        # experts (+ optionally the router) by self-distillation toward its
        # OWN pre-merge MoE-block output — right after the router resize,
        # BEFORE the checkpoint block so the persisted weights and the
        # heal_state field reflect the healed layer.
        heal_state: dict | None = None
        if heal_cfg.enabled:
            if cap_in is not None and cap_out is not None:
                heal_state = _heal_layer(
                    layer_ref=layer_ref,
                    final_kept_ids=final_kept_ids,
                    captured_input=cap_in,
                    captured_target=cap_out,
                    heal_cfg=heal_cfg,
                    device=(
                        device if device is not None
                        else layer_ref.router.weight.device
                    ),
                )
                del cap_in, cap_out
            else:
                log.info(
                    "  merge-heal layer %d: skipped (0 merges — layer "
                    "unchanged, nothing to heal)",
                    layer_ref.layer_idx,
                )

        # Correctness depends on the RuntimeError guard above ensuring no protected expert
        # appears in grouped. Without that guard, the else-branch would silently emit [eid]
        # instead of the full merge group for a protected expert that was also a centroid.
        merge_map[layer_ref.layer_idx] = {
            new_idx: (sorted(grouped[eid]) if eid in grouped else [eid])
            for new_idx, eid in enumerate(final_kept_ids)
        }
        # Ordering critical: remap to post-merge indices BEFORE snapshotting.
        # Writing pre-remap covariance would silently corrupt the resume path.
        _remap_covariance_for_layer(cov_acc, layer_ref.layer_idx, final_kept_ids)

        if partial_dir is not None:
            _snapshot_cov_layer(cov_acc, layer_ref.layer_idx, partial_dir)
            # B-iter5-M-2: persist per-expert neuron means BEFORE the merge JSON
            # so that .pt-before-.json ordering invariant (spec §11) holds for
            # the new artifact too. Resume detects missing means by file absence.
            _snapshot_neuron_means_layer(ream_acc, layer_ref.layer_idx, partial_dir)
            # Merge-heal: healed weights are not reconstructible from
            # merge_*.json, so persist them in their own .pt — written BEFORE
            # _write_merge_json so the .pt-before-.json resume invariant holds.
            if heal_cfg.enabled and heal_state is not None:
                _write_heal_weights(
                    partial_dir, layer_ref, final_kept_ids,
                    accepted=bool(heal_state["accepted"]),
                )
            _write_merge_json(
                partial_dir, layer_ref.layer_idx, final_kept_ids, grouped, freq,
                merge_map[layer_ref.layer_idx],
                mean_cost_per_pair=(
                    mean_assigned_cost
                    if n_assigned > 0 and mean_assigned_cost > 0.0 and not (c_fail and effective_target >= n_experts)
                    else None
                ),
                assignment_solver_used=assignment_solver,
                cost_alignment_used=cost_alignment_cfg,
                em_rounds_completed=em_rounds_done,
                distill_state=(
                    {str(k): v for k, v in distill_state.items()}
                    if distill_state is not None
                    else None
                ),
                heal_state=heal_state,
            )

        max_group = max((len(g) for g in grouped.values()), default=0)
        n_noncentroid_members = sum(len(g) - 1 for g in grouped.values())
        mean_group = n_noncentroid_members / len(grouped) if grouped else 0.0
        log.info(
            "  kept %d / %d experts (protected=%d, ream_centroids=%d) — "
            "Σ cost=%.4f, max_group=%d, mean_group=%.2f",
            len(final_kept_ids), n_experts, n_protected, len(ream_centroid_ids),
            assigned_cost, max_group, mean_group,
        )
        _trackio_log({
            # v1 keys — kept verbatim for backward-compatibility with
            # existing Trackio dashboards. Do not rename or remove.
            "stage2/layer_idx": layer_ref.layer_idx,
            "stage2/protected_experts": n_protected,
            "stage2/ream_centroids": len(ream_centroid_ids),
            "stage2/total_experts": n_experts,
            "stage2/sum_assignment_cost": assigned_cost,
            "stage2/mean_cost_per_pair": mean_assigned_cost if n_assigned > 0 else float("nan"),
            "stage2/max_merge_group_size": max_group,
            "stage2/mean_merge_group_size": mean_group,
            "stage2/effective_target": effective_target,
            "stage2/actual_kept_experts": len(final_kept_ids),
            "stage2/stage1_target": target,
            # v2 keys (spec § 5 / § 6) — per-layer runtime state from the
            # new dispatcher / capacity gate / EM / distillation paths.
            "stage2/assignment_solver_used": assignment_solver,
            "stage2/cost_alignment_effective": effective_cost_alignment,
            "stage2/cost_asymmetric_effective": effective_cost_asymmetric,
            "stage2/capacity_util": capacity_util_value,
            "stage2/capacity_regime": (
                "tight" if effective_cost_alignment == "post" else "slack"
            ),
            "stage2/em_rounds_done": em_rounds_done,
            # Distillation aggregates: keys appear only on layers where
            # distillation actually ran (non-empty distill_state). The
            # **{} no-op keeps the emit slim on disabled / singleton-only
            # layers, avoiding dashboard noise.
            **_summarize_distill_state(distill_state),
        })

        # End-of-layer cleanup: drop Python refs to the per-layer accumulators
        # and force the CUDA caching allocator to release unreferenced blocks
        # back to the driver. Two prior segfaults inside CUDA kernels (silu at
        # layer ~34, layer 7 in an earlier run) were traced to allocator
        # fragmentation that accumulated over the long Stage 2 pass: even with
        # PYTORCH_CUDA_ALLOC_CONF=expandable_segments, freed-but-cached blocks
        # are not returned to the driver, so a future large allocation can
        # still fail mid-kernel. Forcing gc.collect() + empty_cache() at every
        # layer boundary keeps the working set bounded.
        del reap_acc, ream_acc, perm_cache
        if layer_input_acc is not None:
            del layer_input_acc
        if pre_merge_weights is not None:
            del pre_merge_weights
        if distill_state is not None:
            del distill_state
        gc.collect()
        torch.cuda.empty_cache()

    out_dir = artifacts_dir / "stage2_pruned"
    if os.environ.get("MOE_SKIP_STAGE2_COV_SAVE") == "1":
        log.info("Skipping _stage2_input_covariance.pt save "
                 "(MOE_SKIP_STAGE2_COV_SAVE=1; Stages 3/4 disabled, file unused)")
    else:
        _save_covariance(cov_acc, artifacts_dir / "_stage2_input_covariance.pt")
    save_compressed_checkpoint(
        model, tokenizer, out_dir,
        pipeline_stage="stage2_pruned",
        extra_metadata={"merge_map_file": "merge_map.json"},
    )
    save_json_artifact(merge_map, out_dir / "merge_map.json")
    if partial_dir is not None:
        if os.environ.get("MOE_KEEP_STAGE2_PARTIAL") == "1":
            # Direction A — budget retune reads per-layer measured damage from
            # _stage2_partial/merge_*.json. Keep the dir so a baseline run's
            # damage signal survives for the retune tool.
            log.info("Keeping %s (MOE_KEEP_STAGE2_PARTIAL=1) for budget retune",
                     partial_dir)
        else:
            shutil.rmtree(partial_dir, ignore_errors=True)
    log.info("Stage 2 complete — pruned checkpoint at %s", out_dir)
    return out_dir


# ---------------------------------------------------------------------------
# Partial-resume helpers
# ---------------------------------------------------------------------------


def _durable_rename(tmp: Path, final: Path) -> None:
    """Fsync *tmp*, atomically rename it to *final*, then fsync the parent dir.

    Spec §11: durable write — fsync file bytes, then fsync parent dir entry,
    then atomic rename so a crash never leaves a truncated final file.
    O_WRONLY|O_APPEND is used for the .tmp file so fsync flushes write data
    (O_RDONLY on a regular file does not guarantee flushing write buffers on POSIX).
    The parent dir must use O_RDONLY (directories cannot be opened for write).

    Note: the tmp file must already be closed (all Python I/O buffers flushed to
    the kernel) before calling _durable_rename; the fsync it performs flushes
    kernel buffers, not Python-level buffers.
    """
    fd = os.open(str(tmp), os.O_WRONLY | os.O_APPEND)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, final)
    # tmp and final share the same parent directory (both are created in the same
    # directory by all callers), so final.parent == tmp.parent and the fsync below
    # correctly flushes the directory entry for the rename regardless of which path
    # is used.
    parent_fd = os.open(str(final.parent), os.O_RDONLY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def _snapshot_cov_layer(
    cov_acc: InputCovarianceAccumulator,
    layer_idx: int,
    partial_dir: Path,
) -> None:
    with cov_acc._lock:
        keys = [k for k in cov_acc.covariance if k[0] == layer_idx]
        if not keys:
            log.debug("_snapshot_cov_layer: no covariance entries for layer %d; skipping snapshot", layer_idx)
            return
        payload = {
            "format_version": 1,
            "covariance": {k: cov_acc.covariance[k].clone() for k in keys},
            "tokens": {k: cov_acc.token_count.get(k, 0) for k in keys},
        }
    tmp = partial_dir / f"layer_{layer_idx}.pt.tmp"
    final = partial_dir / f"layer_{layer_idx}.pt"
    torch.save(payload, tmp)
    _durable_rename(tmp, final)


def _snapshot_neuron_means_layer(
    ream_acc: ReamCostAccumulator,
    layer_idx: int,
    partial_dir: Path,
) -> None:
    """Persist per-expert mean activation vectors for resume-time C_act.

    B-iter5-M-2: spec D5b mandates `C = C_wt + C_act` for permutation alignment.
    Without this artifact, resume falls back to weight-only alignment and merged
    weights diverge from a fresh run. This helper snapshots only the small
    per-expert mean vectors (`[d_intermediate]` per expert), not the full
    intermediate-activation history (which is large and not needed downstream).

    Format version 1: `{"format_version": 1, "neuron_means": {expert_idx: tensor}}`.
    Missing-on-resume → loud ERROR + weight-only fallback (preserves run completion).
    """
    with ream_acc._lock:
        keys = [k for k in ream_acc._neuron_act_sum if k[0] == layer_idx]
        if not keys:
            log.debug("_snapshot_neuron_means_layer: no neuron-mean entries for layer %d; "
                      "skipping snapshot (no merges in this layer)", layer_idx)
            return
        means: dict[int, torch.Tensor] = {}
        for k in keys:
            s = ream_acc._neuron_act_sum[k]
            c = ream_acc._neuron_act_count.get(k, 0)
            if c == 0:
                continue
            means[k[1]] = (s.clone() / c).contiguous()
    if not means:
        return
    payload = {"format_version": 1, "neuron_means": means}
    tmp = partial_dir / f"_neuron_means_layer{layer_idx}.pt.tmp"
    final = partial_dir / f"_neuron_means_layer{layer_idx}.pt"
    torch.save(payload, tmp)
    _durable_rename(tmp, final)


def _write_merge_json(
    partial_dir: Path,
    layer_idx: int,
    final_kept_ids: list[int],
    grouped: dict[int, list[int]],
    freq: dict[int, int],
    merge_map_layer: dict[int, list[int]],
    *,
    mean_cost_per_pair: float | None = None,
    assignment_solver_used: str = "greedy",
    cost_alignment_used: str = "pre",
    em_rounds_completed: int = 0,
    distill_state: dict | None = None,
    heal_state: dict | None = None,
) -> None:
    """Write the per-layer merge record to a durable JSON file.

    Args:
        partial_dir:      Directory for partial/crash-resume checkpoints.
        layer_idx:        MoE layer index.
        final_kept_ids:   Sorted list of all kept expert IDs after merging
                          (protected experts + REAM centroids). Stored under
                          ``"final_kept_ids"`` (renamed from the old
                          ``"centroid_ids"`` field in format_version 1; the
                          resume path accepts both names for backward compat).
        grouped:          Merge groups keyed by centroid expert ID.
        freq:             Per-expert token frequency counts.
        merge_map_layer:  New-index → original-expert-ids mapping for this layer.
        mean_cost_per_pair: Mean REAM assignment cost, for the budget-bump history.
        heal_state:       Per-layer merge-heal outcome dict (None when the
                          opt-in merge-heal feature is disabled).
    """
    payload = {
        "format_version": 2,
        "final_kept_ids": final_kept_ids,
        # list(v) ensures JSON gets a plain list, not a subclass that might not serialize
        "grouped": {str(k): list(v) for k, v in grouped.items()},
        "freq": {str(k): int(v) for k, v in freq.items()},
        # list(v) ensures JSON gets a plain list, not a subclass that might not serialize
        "merge_map_layer": {str(k): list(v) for k, v in merge_map_layer.items()},
        "mean_cost_per_pair": mean_cost_per_pair,
        # Stage 2 v2 (spec § 12.1): forensic / resume fields. ``em_rounds_completed``
        # and ``distill_state`` are reserved for Phases 2 and 3; included here
        # so Phase-1-completed partials are forward-compatible with later phases.
        "assignment_solver_used": assignment_solver_used,
        "cost_alignment_used": cost_alignment_used,
        "em_rounds_completed": em_rounds_completed,
        "distill_state": distill_state,
        # Stage-2 per-layer merge-heal (opt-in). None when healing is off;
        # otherwise the small JSON-able state dict returned by _heal_layer
        # (steps / train_mse / train_mse_at_best / holdout_mse / heal_gap /
        # stop_reason). The healed weights themselves live in
        # _heal_weights_layer_{N}.pt.
        "heal_state": heal_state,
    }
    tmp = partial_dir / f"merge_{layer_idx}.json.tmp"
    final = partial_dir / f"merge_{layer_idx}.json"
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    _durable_rename(tmp, final)


# ---------------------------------------------------------------------------
# Per-layer profiling
# ---------------------------------------------------------------------------


def _profile_layer(
    model,
    layer_ref: MoELayerRef,
    batches,
    reap_acc: ReapAccumulator,
    cov_acc: InputCovarianceAccumulator,
    ream_acc: ReamCostAccumulator,
    *,
    device=None,
    layer_input_acc: "_LayerInputAccumulator | None" = None,
) -> None:
    """Profile a single MoE layer with early-exit forward.

    REAM sequential merging (paper 2604.04356, §4, Fig 1(b)) requires
    that each layer is profiled on hidden states reflecting all prior
    merges.  All metrics (REAP scores, REAM δ_gate/δ̃_expert, input
    covariance) depend only on hidden states arriving *at* this layer,
    not on downstream layers.  We therefore abort the forward pass
    immediately after this layer completes via :func:`early_exit_after_layer`,
    avoiding O(40−L) unnecessary layer-forwards per batch.

    Total layer-forwards across 40 sequential profiling passes:
    1+2+…+40 = 820 (vs 40×40 = 1600 without early exit).
    """
    layer_idx = layer_ref.layer_idx
    n_experts = layer_ref.num_routed_experts
    was_training = model.training
    model.eval()

    # Resolve `device` from model parameters if the caller left it None,
    # so finalize_batch's compute_device always lands on the model's GPU
    # rather than torch.cuda.current_device() (which may diverge in
    # multi-GPU or thread-context scenarios).
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    # Cumulative token offset: tracks the global start index of each batch.
    # Using cumulative addition (not batch_idx * fixed_size) handles the last
    # partial batch when num_calibration_samples % batch_size != 0.
    _batch_offset = 0  # cumulative token start of current batch
    _next_offset = 0   # cumulative token count after current batch

    # DIAG: per-hook time accumulators (input_cb, intermediate_cb, down_cb, call_count)
    # Reset at start of each batch by the batch loop below.
    import time as _diag_time_mod
    _diag_cb = [0.0, 0.0, 0.0, 0]
    # B-C-C-1: full-softmax cache for the current batch's router logits.
    # Spec §5 line 339 + D-ream-sparse-routing require σ(x)_e (the
    # un-renormalized full softmax over ALL experts), not the top-k
    # renormalized weights returned by Qwen3_5MoeTopKRouter.forward.
    # Populated by an experts-module pre-forward hook that runs AFTER the
    # router pre-forward hook (which captures the raw logits) but BEFORE any
    # expert forward (which fires down_cb). down_cb reads _full_softmax[0]
    # to obtain σ(x)_e at active token positions.
    _full_softmax: list[torch.Tensor | None] = [None]

    def input_cb(li, e, tensor, ctx):
        _t = _diag_time_mod.monotonic()
        cov_acc.update(li, e, "gate_proj", tensor)
        _diag_cb[0] += _diag_time_mod.monotonic() - _t
        _diag_cb[3] += 1

    def intermediate_cb(li, e, tensor, ctx):
        _t = _diag_time_mod.monotonic()
        cov_acc.update(li, e, "down_proj", tensor)
        ream_acc.record_neuron_activations(li, e, tensor)
        _diag_cb[1] += _diag_time_mod.monotonic() - _t

    def down_cb(li, e, tensor, ctx):
        _t = _diag_time_mod.monotonic()
        # _batch_offset is only read here, never assigned; no nonlocal declaration needed.
        record_reap(reap_acc, li, e, ctx["top_k_weights"], tensor)
        # B-C-C-1: pass σ(x)_e (full softmax over all experts) at active token
        # positions for this expert, NOT ctx["top_k_weights"] (renormalized to
        # sum=1 over top-k). The pre-forward hook installed below populates
        # _full_softmax[0] before any expert forward fires.
        token_idx = ctx["token_idx"]
        fs = _full_softmax[0]
        if fs is not None:
            # Index the cached [T, n_experts] full-softmax tensor at the
            # active token positions for this expert. fs lives on the same
            # device as the router logits (typically GPU); index with
            # token_idx on its native device to avoid a CPU↔GPU round-trip.
            # Result shape: [|active|]; ensure it lands on the expert-output
            # device for the downstream (gate * expert_output) multiply.
            sigma_e = fs[token_idx.to(fs.device), e].to(tensor.device)
        else:
            # B-iter5-L-4 (code): hook ordering is spec-required to populate
            # _full_softmax[0] before any expert forward fires; reaching this
            # branch indicates a real ordering bug. Log at ERROR so it is not
            # missed; keep the fallback to top_k_weights so the run completes.
            log.error(
                "down_cb: full-softmax cache empty for layer %d expert %d — "
                "falling back to top_k_weights (renormalized; spec-degraded). "
                "This indicates a hook-ordering bug — the experts pre-forward "
                "hook (_populate_full_softmax) should always run before any "
                "expert forward fires.",
                li, e,
            )
            sigma_e = ctx["top_k_weights"]
        ream_acc.record_gated_output(
            li, e, sigma_e, tensor,
            token_idx, _batch_offset,
        )
        _diag_cb[2] += _diag_time_mod.monotonic() - _t

    # B-C-C-1: pre-forward hook on the experts module that computes the full
    # softmax from the latest captured router logits. Runs after the router
    # pre-forward hook (which appends to router_logits_storage[layer_idx])
    # but before the experts forward (which fires down_cb). Because
    # capture_router_outputs's hook is a *router* pre-forward hook and this
    # one is an *experts* pre-forward hook, ordering is guaranteed by the
    # decoder layer's call sequence (router runs first, dispatches to experts).
    def _populate_full_softmax(_module, _inputs):
        if router_logits_storage[layer_idx]:
            batch_logits = router_logits_storage[layer_idx][-1]
            # F.softmax over the last (expert) dim → [T, n_experts] σ(x)_e values.
            # .float() avoids dtype mismatch when the router runs in bf16.
            # Keep on-device (router-logits device) — down_cb indexes with
            # on-device token_idx, avoiding a CPU↔GPU round-trip per expert.
            _full_softmax[0] = F.softmax(batch_logits.float(), dim=-1)
        else:
            _full_softmax[0] = None

    try:
        with instrument_experts(
            layer_ref,
            {"input": input_cb, "intermediate": intermediate_cb, "down": down_cb},
        ), capture_router_outputs([layer_ref]) as router_logits_storage, \
             early_exit_after_layer(model, layer_idx):
            # Install the experts pre-forward hook AFTER capture_router_outputs
            # so the router hook fires first per batch.
            _experts_handle = layer_ref.experts_module.register_forward_pre_hook(
                _populate_full_softmax
            )
            # Phase 3: optionally capture the layer-input hidden states for
            # per-merge-group expert distillation (spec § 5 step 7b / M8).
            # Hook on the decoder layer module — its first input is the
            # hidden_states tensor that the layer's forward operates on.
            _layer_in_handle = None
            if layer_input_acc is not None:
                def _capture_layer_input(_module, inputs):
                    if inputs and inputs[0] is not None:
                        layer_input_acc.add(inputs[0])
                _layer_in_handle = layer_ref.layer_module.register_forward_pre_hook(
                    _capture_layer_input
                )
            try:
                # DIAG: layer-1 hang investigation — log every batch so we can see
                # if the forward pass is making progress and how long each batch takes.
                import time as _diag_time
                _diag_t0 = _diag_time.monotonic()
                _diag_count = 0
                log.info("DIAG layer %d: entering batch loop (calibration tensor + early-exit forwards) | %s", layer_idx, _rt_snap())
                _rt_update(stage="stage2", layer=int(layer_idx), batch=0, phase="profile_layer_start")
                for batch in batches:
                    _diag_t_batch = _diag_time.monotonic()
                    # Reset per-batch hook timers
                    _diag_cb[0] = 0.0; _diag_cb[1] = 0.0; _diag_cb[2] = 0.0; _diag_cb[3] = 0
                    # `device` is guaranteed non-None after the resolution block
                    # at the top of _profile_layer.
                    batch = batch.to(device)
                    _batch_offset = _next_offset
                    router_logits_storage[layer_idx].clear()
                    _full_softmax[0] = None
                    _diag_t_fwd = _diag_time.monotonic()
                    with torch.no_grad():
                        try:
                            model(input_ids=batch)
                        except _EarlyExitException:
                            pass  # expected — target layer completed
                    _diag_fwd_dt = _diag_time.monotonic() - _diag_t_fwd
                    if router_logits_storage[layer_idx]:
                        batch_logits = router_logits_storage[layer_idx][-1]
                        ream_acc.record_router_logits(layer_idx, batch_logits, _batch_offset)
                    ream_acc.finalize_batch(layer_idx, n_experts, compute_device=device)
                    ream_acc.record_batch_token_count(layer_idx, batch.shape[0] * batch.shape[1])
                    _next_offset += batch.shape[0] * batch.shape[1]
                    _diag_count += 1
                    _rt_update(stage="stage2", layer=int(layer_idx), batch=int(_diag_count),
                               phase="profile_layer_batch")
                    _diag_dt = _diag_time.monotonic() - _diag_t_batch
                    if _diag_count <= 3 or _diag_count % 10 == 0:
                        log.info(
                            "DIAG layer %d batch %d: total=%.2fs fwd=%.2fs hooks: input=%.2fs intermed=%.2fs down=%.2fs (n_cb=%d) | cum=%.1fs | %s",
                            layer_idx, _diag_count, _diag_dt, _diag_fwd_dt,
                            _diag_cb[0], _diag_cb[1], _diag_cb[2], _diag_cb[3],
                            _diag_time.monotonic() - _diag_t0, _rt_snap(),
                        )
                log.info("DIAG layer %d: batch loop complete — %d batches in %.1fs, now post-profile work | %s",
                         layer_idx, _diag_count, _diag_time.monotonic() - _diag_t0, _rt_snap())
            finally:
                _experts_handle.remove()
                if _layer_in_handle is not None:
                    _layer_in_handle.remove()
    finally:
        if was_training:
            model.train()


# ---------------------------------------------------------------------------
# REAM cost + assignment
# ---------------------------------------------------------------------------


def _extract_sim_expert_matrix_from_tensor(
    sim_tensor: "torch.Tensor | None",
    noncentroid_ids: list[int],
    centroid_ids: list[int],
    total_tokens: int,
) -> np.ndarray:
    """Vectorized δ̃_expert(child, centroid) submatrix from the dense sim tensor.

    Replaces the former ``n_nc × n_c`` double loop over
    ``ReamCostAccumulator.compute_delta_expert`` (one lock-protected call per
    pair, ~14.6K calls/layer at Qwen scale). Reads the dense ``[E, E]``
    float64 ``_sim_tensor`` once and broadcasts the REAM Eq. 8 rescale over
    the whole submatrix.

    Args:
        sim_tensor: the layer's ``[E, E]`` float64 gated-output cosine-sum
            accumulator (``ReamCostAccumulator._sim_tensor[layer_idx]``), or
            ``None`` if no batch was finalized for the layer.
        noncentroid_ids: child (row) expert IDs.
        centroid_ids: centroid (column) expert IDs.
        total_tokens: |X|, the Eq. 8 denominator (total calibration tokens).

    Returns:
        ``(n_nc, n_c)`` float64 ndarray of δ̃_expert similarities ∈ [0, 1].

    Equivalence with the old per-pair path: ``compute_delta_expert`` returns
        * ``NaN`` when ``total_tokens == 0`` (or no sim data) — the caller
          substituted ``0.5`` (neutral after the (cos+1)/2 rescale);
        * else ``clip((sim_val / total + 1) / 2, 0, 1)``.
      A ``None`` sim_tensor means ``sim_val == 0`` for every pair, so the old
      path yielded ``clip((0/total + 1)/2, 0, 1) == 0.5`` — identical to the
      ``total_tokens == 0`` neutral fill. We therefore collapse both
      degenerate cases to a full-0.5 matrix.
    """
    n_nc = len(noncentroid_ids)
    n_c = len(centroid_ids)
    if total_tokens == 0 or sim_tensor is None:
        # Degenerate: no joint-activation data → neutral 0.5 everywhere
        # (matches compute_delta_expert's NaN→0.5 substitution and the
        # sim_val==0 → 0.5 algebra; see docstring).
        return np.full((n_nc, n_c), 0.5, dtype=np.float64)
    # Index the [E, E] accumulator down to the (child, centroid) submatrix.
    # advanced indexing on the first axis then the second produces the
    # n_nc × n_c block in C-row order matching the old nested loop.
    nc_idx = torch.as_tensor(noncentroid_ids, dtype=torch.long)
    c_idx = torch.as_tensor(centroid_ids, dtype=torch.long)
    sub = sim_tensor.to(torch.float64)[nc_idx][:, c_idx]  # (n_nc, n_c)
    sim = ((sub / total_tokens + 1.0) / 2.0).clamp_(0.0, 1.0)
    return sim.numpy().copy()


def _apply_skip_merge_floor(
    delta: np.ndarray, skip_merge_percentile: float,
) -> tuple[np.ndarray, int]:
    """Direction B — skip-merge floor.

    Compute the ``skip_merge_percentile`` percentile ``P`` over the *finite*
    entries of the cost matrix ``delta`` and mask every entry strictly greater
    than ``P`` to ``+inf``. Masked pairs are forbidden to the assignment solver,
    so the affected children fall through the greedy ``-1`` path and become
    singleton ("orphan-promoted") kept experts downstream.

    ``skip_merge_percentile == 100.0`` is the OFF sentinel: the 100th percentile
    equals the maximum finite cost, nothing is strictly above it, so this helper
    returns a fresh copy with no entries masked. (The Stage-2 ``run()`` call site
    skips this helper entirely at the sentinel, leaving the original array as-is.)

    Returns ``(masked_delta, n_masked)`` where ``masked_delta`` is a new array
    (the input is never mutated) and ``n_masked`` is the count of entries newly
    set to ``+inf`` (entries that were already non-finite are not counted).
    """
    out = delta.astype(np.float64, copy=True)
    if out.size == 0:
        return out, 0
    finite_mask = np.isfinite(out)
    if not finite_mask.any():
        # No finite costs at all — percentile is undefined; mask nothing.
        return out, 0
    finite_vals = out[finite_mask]
    p = float(np.percentile(finite_vals, skip_merge_percentile))
    # Strictly above P, and only entries that are currently finite (so we do
    # not "re-mask" already-+inf entries and inflate the reported count).
    above = finite_mask & (out > p)
    n_masked = int(above.sum())
    out[above] = np.inf
    return out, n_masked


def _ream_cost_matrix(
    layer_ref: MoELayerRef,
    noncentroid_ids: list[int],
    centroid_ids: list[int],
    *,
    ream_acc: ReamCostAccumulator,
    blacklisted_ids: set[int] | None = None,
    cost_alignment: str = "pre",
    cost_whitening: str = "none",
    cost_asymmetric: bool = False,
    cost_topk_filter: int = 48,
    freq: dict[int, int] | None = None,
    cov_acc: "InputCovarianceAccumulator | None" = None,
    perm_cache: "_PermAlignCache | None" = None,
    tentative_centroid_weights: dict[int, dict[str, torch.Tensor]] | None = None,
    layer_inputs: torch.Tensor | None = None,
    output_token_cap: int = 1024,
) -> np.ndarray:
    """Compute the (n_nc × n_c) REAM cost matrix.

    Three modes (Stage 2 v2 spec § 5 step 4 + Direction C):

    - ``cost_alignment="pre"`` (default, v1 behavior): symmetric δ_REAM cost
      ``1 - (δ_gate + δ̃_expert)/2`` over all pairs.
    - ``cost_alignment="post"`` (Tier 2 / v2 path): for each non-centroid m,
      compute the cheap symmetric cost first; take the top-K candidates by
      cheap cost; for those candidates only, compute the per-pair Hungarian
      alignment cost and the whitened Frobenius residual
      ``R_cm = ‖(W_c − P_cm·W_m) · A^{1/2}‖_F`` (sum over gate/up/down per
      § 5 step 4T(c)(ii)). All other entries get +∞ so the assignment solver
      treats them as forbidden. Permutations and residuals are stashed in
      ``perm_cache`` for the merge step to reuse (M1).
    - ``cost_alignment="output"`` (Direction C): for each non-centroid m,
      take the top-K candidates by cheap cost; for those candidates compute
      the *output-space* cost — the routing-weighted change in expert m's
      gated routed output on the captured calibration tokens when m is
      tentatively merged into the centroid. See ``_output_space_cost``.
      Requires ``layer_inputs`` (the layer-input calibration buffer) and
      ``freq`` (for the freq-weighted tentative merge).

    When ``cost_asymmetric=True`` and ``freq`` is provided, the post-alignment
    residual is multiplied by ``freq_m / (freq_c + freq_m)`` (spec § 5 step
    4T(c)(iii) / D-asymmetric-freq). This is valid only under the
    freq-weighted merge path; the caller is responsible for that invariant.
    """
    if not noncentroid_ids or not centroid_ids:
        # Early return produces shape (0, n_c) or (n_nc, 0) rather than (0, 0),
        # which is intentional. Callers guard with `delta.size > 0`, which correctly
        # handles all three degenerate shapes without special-casing each.
        return np.zeros((len(noncentroid_ids), len(centroid_ids)))

    li = layer_ref.layer_idx
    n_nc = len(noncentroid_ids)
    n_c = len(centroid_ids)
    n_experts_total = layer_ref.num_routed_experts

    # Compute δ_gate over the non-protected expert population so that dist2sim
    # normalizes by the global maximum distance among non-protected experts
    # (spec §5 Step 2, REAM ref ream/ream.py lines 37-41). Including protected
    # (super-expert) IDs would let their extreme gate-logit distances dominate
    # d.max(), compressing all noncentroid–centroid similarities toward 1.0
    # — DIST2SIM-PROTECTED-BIAS.
    protected_set = set(blacklisted_ids) if blacklisted_ids else set()
    all_n_ids = [e for e in range(n_experts_total) if e not in protected_set]
    _nc_protected = set(noncentroid_ids) & protected_set
    _c_protected  = set(centroid_ids)    & protected_set
    if _nc_protected or _c_protected:
        raise ValueError(
            f"_ream_cost_matrix: noncentroid_ids or centroid_ids overlap with blacklisted_ids "
            f"(nc={_nc_protected}, c={_c_protected})"
        )
    sim_gate_full = ream_acc.compute_gate_similarity_matrix(li, all_n_ids)
    # id_to_full_row maps expert ID → row index in all_n_ids.
    # Invariant: Stage 2 profiles each layer before merging it, so expert IDs are
    # always pre-merge [0, n_experts_total) when _ream_cost_matrix is called.
    id_to_full_row = {e: i for i, e in enumerate(all_n_ids)}
    # Extract the (n_nc × n_c) submatrix from the full N×N matrix.
    nc_rows = [id_to_full_row[e] for e in noncentroid_ids]
    c_cols  = [id_to_full_row[e] for e in centroid_ids]
    sim_gate_sub = sim_gate_full[np.ix_(nc_rows, c_cols)].numpy().astype(np.float64)  # (n_nc, n_c)

    # δ̃_expert submatrix (REAM Eq. 8): vectorized read of the dense [E, E]
    # gated-output cosine-sum accumulator, replacing a former n_nc × n_c
    # double loop over compute_delta_expert (~14.6K lock-protected calls per
    # layer at Qwen3.6 scale). The NaN→0.5 substitution that the old loop
    # applied per pair is folded into _extract_sim_expert_matrix_from_tensor
    # (degenerate total_tokens==0 / no-data → full-0.5 matrix); see its
    # docstring for the equivalence argument.
    #
    # Lock discipline (R4): snapshot the total-token count and the sim
    # tensor reference under ream_acc._lock, then compute outside the lock.
    # This is safe because Stage 2 finalizes ALL batches for a layer before
    # calling _ream_cost_matrix — there is no concurrent finalize_batch
    # mutating _sim_tensor[li] during this read. Snapshotting the dict
    # reference under the lock still guards against a concurrent clear_layer.
    with ream_acc._lock:
        total_tokens = ream_acc._total_tokens_by_layer.get(li, 0)
        sim_t = ream_acc._sim_tensor.get(li)
    sim_expert_matrix = _extract_sim_expert_matrix_from_tensor(
        sim_t, noncentroid_ids, centroid_ids, total_tokens,
    )  # (n_nc, n_c) float64

    # δ_REAM = (δ_gate + δ̃_expert) / 2 ∈ [0,1]; cost = 1 − δ_REAM ∈ [0,1].
    # Lower cost = more similar (spec §5 Step 2, reference ream/ream.py L46-53).
    cost = 1.0 - (sim_gate_sub + sim_expert_matrix) / 2.0
    np.clip(cost, 0.0, 1.0, out=cost)

    if cost_alignment == "pre":
        return cost

    if cost_alignment == "output":
        # Direction C — output-space merge cost. ``cost`` (the cheap symmetric
        # δ_REAM) is reused only as the top-K candidate filter, exactly like
        # the "post" path uses it.
        return _output_space_cost(
            layer_ref,
            noncentroid_ids,
            centroid_ids,
            cheap_cost=cost,
            ream_acc=ream_acc,
            perm_cache=perm_cache,
            topk=cost_topk_filter,
            freq=freq,
            layer_inputs=layer_inputs,
            token_cap=output_token_cap,
        )

    if cost_alignment != "post":
        raise ValueError(
            f"_ream_cost_matrix: unknown cost_alignment={cost_alignment!r}; "
            "expected 'pre', 'post', or 'output'."
        )

    # Stage 2 v2: post-alignment whitened residual path (spec § 5 step 4T).
    return _post_alignment_cost(
        layer_ref,
        noncentroid_ids,
        centroid_ids,
        cheap_cost=cost,
        ream_acc=ream_acc,
        cov_acc=cov_acc,
        perm_cache=perm_cache,
        whitening_mode=cost_whitening,
        asymmetric=cost_asymmetric,
        topk=cost_topk_filter,
        freq=freq,
        tentative_centroid_weights=tentative_centroid_weights,
    )


def _post_alignment_cost(
    layer_ref: MoELayerRef,
    noncentroid_ids: list[int],
    centroid_ids: list[int],
    *,
    cheap_cost: np.ndarray,
    ream_acc: ReamCostAccumulator,
    cov_acc: "InputCovarianceAccumulator | None",
    perm_cache: "_PermAlignCache | None",
    whitening_mode: str,
    asymmetric: bool,
    topk: int,
    freq: dict[int, int] | None,
    tentative_centroid_weights: dict[int, dict[str, torch.Tensor]] | None = None,
) -> np.ndarray:
    """Build the post-alignment whitened cost matrix per spec § 5 step 4T.

    Steps per non-centroid m:
      1. Pick the top-K candidate centroids by ``cheap_cost`` (lowest values).
      2. For each (c, m) candidate: compute Hungarian alignment via
         ``_permutation_align_to_centroid`` (cached if available), then the
         three-term whitened Frobenius residual.
      3. Optionally multiply by ``freq_m / (freq_c + freq_m)`` (asymmetric).
      4. Stash (perm, residual) into ``perm_cache`` for the merge step.

    All non-candidate entries get ``+inf`` so the assignment solver treats
    them as forbidden arcs.
    """
    from .utils.cov_sqrt import compute_a_sqrt, CovSqrtCache

    li = layer_ref.layer_idx
    n_nc = len(noncentroid_ids)
    n_c = len(centroid_ids)

    if topk < 1:
        raise ValueError(
            f"_post_alignment_cost: cost_topk_filter={topk} < 1 — must be at "
            "least the per-centroid capacity to leave a feasible assignment."
        )

    if cov_acc is None and whitening_mode != "none":
        raise ValueError(
            "_post_alignment_cost: cov_acc is required when "
            f"cost_whitening={whitening_mode!r} (need input covariance for "
            "the whitening factor). Set cost_whitening='none' to disable."
        )

    if asymmetric and freq is None:
        raise ValueError(
            "_post_alignment_cost: cost_asymmetric=True requires freq dict "
            "(per-expert calibration token counts)."
        )

    banks = build_banks(layer_ref)

    # Per-layer eigen-sqrt cache. Bounded by N centroids × 1 matrix per axis.
    a_sqrt_cache = CovSqrtCache(max_entries=2 * n_c + 8)

    def _get_a_sqrt(eid: int, name: str) -> torch.Tensor:
        if whitening_mode == "none":
            return torch.tensor(1.0)
        key = (li, eid, name, whitening_mode)
        cached = a_sqrt_cache.get(key)
        if cached is not None:
            return cached
        # InputCovarianceAccumulator stores covariance under the (layer, expert,
        # matrix_name) key. ``gate_proj`` and ``up_proj`` share the same input
        # covariance (the experts' shared input), so look up under "gate_proj"
        # for both gate and up; "down_proj" has its own covariance.
        cov_key = (li, eid, name)
        if cov_acc is None or cov_key not in cov_acc.covariance:
            raise RuntimeError(
                f"_post_alignment_cost: missing covariance for layer {li} "
                f"expert {eid} matrix {name!r}; check that profiling completed "
                "before cost-matrix construction."
            )
        A = cov_acc.covariance[cov_key].to(torch.float32)
        a_sqrt = compute_a_sqrt(A, mode=whitening_mode)
        a_sqrt_cache.put(key, a_sqrt)
        return a_sqrt

    out = np.full((n_nc, n_c), np.inf, dtype=np.float64)

    # Per-non-centroid: pick the top-K cheapest centroids and compute the
    # expensive cost only for those. All cost-matrix tensor work is
    # read-only on model params; wrap in torch.no_grad() so the leaf
    # nn.Parameters' requires_grad=True does not poison the .numpy() calls
    # in _permutation_align_to_centroid.
    with torch.no_grad():
     for ci in range(n_nc):
        m_id = noncentroid_ids[ci]
        # Top-K centroid indices by cheap cost (smallest first).
        # If n_c <= K, we score all centroids.
        k = min(topk, n_c)
        top_cj = np.argpartition(cheap_cost[ci], k - 1)[:k]
        for cj in top_cj:
            cj = int(cj)
            c_id = centroid_ids[cj]
            cache_key = (li, c_id, m_id)
            cached = perm_cache.get(cache_key) if perm_cache is not None else None
            # When EM provides a tentative merged centroid weight, the cache
            # entry for the original centroid is stale — recompute against
            # the tentative weights instead. F3 fix: single boolean gates
            # both the residual-reuse and perm-reuse branches so they cannot
            # diverge under future refactors.
            tentative_active = (
                tentative_centroid_weights is not None
                and c_id in tentative_centroid_weights
            )
            cache_usable = (cached is not None) and not tentative_active
            if cache_usable and cached[1] is not None:
                # Already computed — reuse both perm and residual.
                residual = cached[1]
            else:
                if tentative_active:
                    tw = tentative_centroid_weights[c_id]  # type: ignore[index]
                    ref_gate = tw["gate_proj"].to(torch.float32)
                    ref_up   = tw["up_proj"].to(torch.float32)
                    ref_down = tw["down_proj"].to(torch.float32)
                else:
                    ref_gate = banks["gate_proj"].get(c_id).to(torch.float32)
                    ref_up   = banks["up_proj"].get(c_id).to(torch.float32)
                    ref_down = banks["down_proj"].get(c_id).to(torch.float32)
                child_gate = banks["gate_proj"].get(m_id).to(torch.float32)
                child_up   = banks["up_proj"].get(m_id).to(torch.float32)
                child_down = banks["down_proj"].get(m_id).to(torch.float32)

                ref_act   = ream_acc.get_neuron_mean(li, c_id) if ream_acc else None
                child_act = ream_acc.get_neuron_mean(li, m_id) if ream_acc else None

                # When the tentative-centroid override is active, the cached
                # perm is stale (it was computed against the original centroid
                # weights) — recompute against the tentative weights.
                if cache_usable:
                    perm = cached[0]
                else:
                    perm = _permutation_align_to_centroid(
                        ref_gate, ref_up, child_gate, child_up,
                        ref_act_mean=ref_act, child_act_mean=child_act,
                    )

                # Whitening still uses the *centroid's own* covariance even
                # when the tentative-centroid weights replace the centroid's
                # row in the residual computation. The covariance is a property
                # of which input distribution the centroid sees post-merge,
                # which is approximated by A_c (the original centroid's input
                # statistics). Using A_c here keeps the whitening consistent
                # across EM rounds; otherwise we'd need to recompute A from
                # scratch each round.
                a_sqrt_gate_up = _get_a_sqrt(c_id, "gate_proj")
                a_sqrt_down    = _get_a_sqrt(c_id, "down_proj")
                residual = _aligned_whitened_residual(
                    ref_gate=ref_gate, ref_up=ref_up, ref_down=ref_down,
                    child_gate=child_gate, child_up=child_up, child_down=child_down,
                    perm=perm,
                    a_sqrt_gate_up=a_sqrt_gate_up,
                    a_sqrt_down=a_sqrt_down,
                    whitening_mode=whitening_mode,
                )

                # Only persist to the cache when the residual reflects the
                # *original* centroid weights (no tentative override). The
                # tentative residual is per-EM-round and would be stale by
                # the time the merge step consumes it.
                if perm_cache is not None and not tentative_active:
                    perm_cache.put(cache_key, perm, residual)

            if asymmetric:
                # freq is guaranteed non-None here by the precondition check
                # at the top of _post_alignment_cost.
                assert freq is not None
                f_c = max(int(freq.get(c_id, 0)), 0)
                f_m = max(int(freq.get(m_id, 0)), 0)
                denom = f_c + f_m
                if denom > 0:
                    factor = f_m / denom
                else:
                    factor = 0.5  # both zero — neutral
                residual = residual * factor

            out[ci, cj] = float(residual)

    return out


# ---------------------------------------------------------------------------
# Direction C — output-space merge cost (cost_alignment == "output")
# ---------------------------------------------------------------------------


def _router_routing_weights(
    layer_ref: MoELayerRef,
    x: torch.Tensor,
) -> torch.Tensor:
    """Full-softmax routing weights ``σ(x)_e`` for every routed expert.

    Recomputes the layer router's pre-softmax routing scores from the layer
    input ``x`` and applies a softmax over the expert axis — the same
    bias-adjusted pre-softmax → softmax path that ``capture_router_outputs``
    uses during profiling. Model-agnostic: it reads ``layer_ref.router``'s
    ``weight`` / ``bias`` / ``e_score_correction_bias`` attributes, none of
    which are Qwen-specific (any linear MoE router exposes ``weight``; the
    bias terms are simply skipped when absent).

    ``x`` : ``(n_tokens, hidden)``. Returns ``(n_tokens, n_experts)`` float32.
    """
    router = layer_ref.router
    w = router.weight
    logits = F.linear(x.to(w.dtype), w)
    bias = getattr(router, "bias", None)
    if bias is not None:
        logits = logits + bias
    esc = getattr(router, "e_score_correction_bias", None)
    if esc is not None:
        logits = logits + esc
    return F.softmax(logits.float(), dim=-1)


def _tentative_merged_weights(
    layer_ref: MoELayerRef,
    centroid_id: int,
    child_id: int,
    freq: dict[int, int],
    ream_acc: "ReamCostAccumulator | None",
    perm_cache: "_PermAlignCache | None",
) -> dict[str, torch.Tensor]:
    """Freq-weighted, permutation-aligned merge of ``child_id`` into
    ``centroid_id`` — the two-member case of ``_em_compute_tentative_weights``.

    ``W_merged = w_c · W_c + w_m · perm_m(W_m)`` where the weights are
    ``freq_e / (freq_c + freq_m)`` and ``perm_m`` aligns the child's
    intermediate neurons to the centroid (identity for the centroid itself).
    Returns float32 weights keyed by ``MATRIX_NAMES``. Model-agnostic: expert
    weights come through ``build_banks`` / ``MATRIX_NAMES``.

    Read-only on model params — wrapped in ``torch.no_grad()`` so the leaf
    ``nn.Parameter``s' ``requires_grad=True`` does not build an autograd graph
    (mirrors ``_post_alignment_cost``).
    """
    li = layer_ref.layer_idx
    banks = build_banks(layer_ref)

    f_c = max(int(freq.get(centroid_id, 0)), 0)
    f_m = max(int(freq.get(child_id, 0)), 0)
    denom = f_c + f_m
    if denom > 0:
        w_c, w_m = f_c / denom, f_m / denom
    else:
        w_c, w_m = 0.5, 0.5  # both zero — neutral average

    with torch.no_grad():
        ref_gate = banks["gate_proj"].get(centroid_id).to(torch.float32)
        ref_up   = banks["up_proj"].get(centroid_id).to(torch.float32)
        child_gate = banks["gate_proj"].get(child_id).to(torch.float32)
        child_up   = banks["up_proj"].get(child_id).to(torch.float32)

        cached = perm_cache.get((li, centroid_id, child_id)) if perm_cache is not None else None
        if cached is not None:
            perm = cached[0]
        else:
            ref_act   = ream_acc.get_neuron_mean(li, centroid_id) if ream_acc else None
            child_act = ream_acc.get_neuron_mean(li, child_id) if ream_acc else None
            perm = _permutation_align_to_centroid(
                ref_gate, ref_up, child_gate, child_up,
                ref_act_mean=ref_act, child_act_mean=child_act,
            )
        perm_t = torch.as_tensor(perm, dtype=torch.long, device=ref_gate.device)

        merged: dict[str, torch.Tensor] = {}
        for name in MATRIX_NAMES:
            W_c = banks[name].get(centroid_id).to(torch.float32)
            W_m = banks[name].get(child_id).to(torch.float32)
            # gate/up permute the intermediate (row) axis; down permutes the
            # intermediate (column) axis. Mirrors _aligned_whitened_residual.
            if name == "down_proj":
                W_m = W_m[:, perm_t]
            else:
                W_m = W_m[perm_t, :]
            merged[name] = w_c * W_c + w_m * W_m
    return merged


def _output_space_cost(
    layer_ref: MoELayerRef,
    noncentroid_ids: list[int],
    centroid_ids: list[int],
    *,
    cheap_cost: np.ndarray,
    ream_acc: "ReamCostAccumulator | None",
    perm_cache: "_PermAlignCache | None",
    topk: int,
    freq: dict[int, int] | None,
    layer_inputs: torch.Tensor | None,
    token_cap: int,
) -> np.ndarray:
    """Direction C — output-space merge cost matrix (spec STRATEGY_NEXT §C).

    For each non-centroid ``m`` and each of its top-K cheapest candidate
    centroids ``c``, the cost is the routing-weighted change in expert ``m``'s
    *gated routed output* on the calibration tokens when ``m`` is tentatively
    merged into ``c``::

        cost(m→c) = mean_t [ σ_m(x_t) · ‖ E_m(x_t) − E_merged(x_t) ‖² ]

    where ``E_m`` is expert ``m``'s SwiGLU forward, ``E_merged`` is the
    forward of the freq-weighted permutation-aligned tentative merge of
    ``m`` into ``c``, and ``σ_m(x_t)`` is token ``t``'s full-softmax routing
    weight for expert ``m``, masked to zero on tokens where ``m`` is not in
    the token's top-``top_k`` routed set (so the cost only counts tokens that
    actually reach expert ``m`` — the topk-routing bound from the spec).

    This is a strictly better merge-damage proxy than the weight-space
    ``pre`` / ``post`` costs: it measures the realised change in what the
    layer emits, not a norm on the weights.

    All non-candidate entries get ``+inf`` so the assignment solver treats
    them as forbidden, exactly like the ``post`` path.

    Model-agnostic: expert count / dims come from ``build_banks`` /
    ``MATRIX_NAMES``; ``top_k`` from ``layer_ref.top_k``; routing weights from
    ``layer_ref.router``; the FFN forward is ``_swiglu_forward`` — the same
    activation abstraction the ``post`` and distillation paths already use.
    """
    n_nc = len(noncentroid_ids)
    n_c = len(centroid_ids)

    if topk < 1:
        raise ValueError(
            f"_output_space_cost: cost_topk_filter={topk} < 1 — must be at "
            "least the per-centroid capacity to leave a feasible assignment."
        )
    if layer_inputs is None or layer_inputs.shape[0] == 0:
        raise RuntimeError(
            "_output_space_cost: no layer-input calibration tokens were "
            "captured — the _LayerInputAccumulator must be enabled when "
            "cost_alignment == 'output' (check the Stage 2 driver)."
        )
    if freq is None:
        raise ValueError(
            "_output_space_cost: freq dict is required (the tentative merge "
            "is freq-weighted)."
        )

    out = np.full((n_nc, n_c), np.inf, dtype=np.float64)

    banks = build_banks(layer_ref)
    # Resolve the compute device + dtype from the model's own expert weights
    # so the cost computation runs wherever the model lives (CPU or GPU),
    # with no hardcoded device.
    _probe = banks["gate_proj"].get(centroid_ids[0] if n_c else noncentroid_ids[0])
    device = _probe.device

    with torch.no_grad():
        # Calibration tokens: deterministic per-layer subsample, capped so the
        # per-pair SwiGLU forward stays bounded (~1k tokens by default).
        x_all = layer_inputs.reshape(-1, layer_inputs.shape[-1])
        n_tokens = x_all.shape[0]
        if n_tokens > token_cap:
            rng = torch.Generator(device="cpu").manual_seed(layer_ref.layer_idx)
            idx = torch.randperm(n_tokens, generator=rng)[:token_cap]
            x_all = x_all[idx]
        x_all = x_all.to(device, dtype=torch.float32)

        # Full-softmax routing weights + the token's top-k routed expert set.
        sigma = _router_routing_weights(layer_ref, x_all)  # (T, n_experts)
        top_k = layer_ref.top_k
        k = min(top_k, sigma.shape[-1])
        topk_idx = torch.topk(sigma, k=k, dim=-1).indices  # (T, k)

        # Per non-centroid m: σ_m masked to tokens that route to m, and m's
        # own expert output E_m(x). Computed once per m (reused across its
        # K candidate centroids).
        for ci in range(n_nc):
            m_id = noncentroid_ids[ci]
            # routing-weighted mask: σ_m(x) on tokens that route to m, else 0.
            routed_m = (topk_idx == m_id).any(dim=-1)  # (T,) bool
            gate_m = sigma[:, m_id] * routed_m.to(sigma.dtype)  # (T,)
            if float(gate_m.sum()) <= 0.0:
                # No calibration token routes to m — the output cost is
                # undefined (every merge is "free" on unseen inputs). Leave
                # the whole row at +inf so the bump loop / orphan-promotion
                # path handles it, exactly as the post path does for a child
                # with no finite candidate. The cheap-cost top-K still picks
                # candidates, but a finite fallback is needed so the solver
                # has a feasible arc — fall back to the cheap symmetric cost.
                k_cand = min(topk, n_c)
                for cj in np.argpartition(cheap_cost[ci], k_cand - 1)[:k_cand]:
                    out[ci, int(cj)] = float(cheap_cost[ci, int(cj)])
                continue

            W_m = {name: banks[name].get(m_id).to(device, torch.float32)
                   for name in MATRIX_NAMES}
            E_m = _swiglu_forward(
                W_m["gate_proj"], W_m["up_proj"], W_m["down_proj"], x_all,
            )  # (T, hidden)

            k_cand = min(topk, n_c)
            top_cj = np.argpartition(cheap_cost[ci], k_cand - 1)[:k_cand]
            for cj in top_cj:
                cj = int(cj)
                c_id = centroid_ids[cj]
                merged = _tentative_merged_weights(
                    layer_ref, c_id, m_id, freq, ream_acc, perm_cache,
                )
                merged = {name: merged[name].to(device, torch.float32)
                          for name in MATRIX_NAMES}
                E_merged = _swiglu_forward(
                    merged["gate_proj"], merged["up_proj"], merged["down_proj"],
                    x_all,
                )  # (T, hidden)
                # Routing-weighted mean squared output change for expert m.
                per_token = (E_m - E_merged).pow(2).sum(dim=-1)  # (T,)
                cost = (gate_m * per_token).sum() / gate_m.sum()
                out[ci, cj] = float(cost)

    return out


# ---------------------------------------------------------------------------
# Phase 3 — per-merge-group expert distillation (spec § 5 step 7b / M8)
# ---------------------------------------------------------------------------


class _LayerInputAccumulator:
    """Reservoir-sample hidden states arriving at a single MoE layer.

    Captured during the profile pass via a forward-pre hook on the decoder
    layer. Used by step 7b expert distillation to provide the calibration
    inputs ``x`` that feed both the merged-centroid student forward and the
    pre-merge group-member target forward.

    Sample size is capped at ``max_samples`` (default 8192 tokens) so the
    host RAM cost is bounded even on long calibration runs. With
    ``hidden_size=2048`` and bf16, a full buffer is ~32 MB.

    A seeded ``torch.Generator`` (default seed = 0) is used for the reservoir
    coin flips so the captured calibration set is bit-reproducible across
    runs; callers can override ``seed`` with the layer index for per-layer
    independence (the Stage 2 driver does this).
    """

    def __init__(self, max_samples: int = 8192, *, seed: int = 0) -> None:
        self.max_samples = max_samples
        self.buffer: torch.Tensor | None = None
        self.seen = 0
        self._generator = torch.Generator(device="cpu").manual_seed(int(seed))

    def add(self, hidden: torch.Tensor) -> None:
        # hidden: (batch, seq, hidden) or (batch*seq, hidden)
        flat = hidden.reshape(-1, hidden.shape[-1]).detach().to("cpu")
        n = flat.shape[0]
        if self.buffer is None:
            take = min(n, self.max_samples)
            self.buffer = flat[:take].contiguous().clone()
            self.seen = n
            return
        # Reservoir-style: replace random rows in the buffer with new samples
        # so the captured set remains a uniform sample across batches.
        for i in range(n):
            self.seen += 1
            if self.buffer.shape[0] < self.max_samples:
                self.buffer = torch.cat([self.buffer, flat[i:i + 1]], dim=0)
            else:
                # Replace a random index with probability max_samples / seen.
                # Seeded generator → bit-reproducible across runs (F2 fix).
                j = int(torch.randint(
                    0, self.seen, (1,), generator=self._generator,
                ).item())
                if j < self.max_samples:
                    self.buffer[j] = flat[i]

    def get(self) -> torch.Tensor | None:
        return self.buffer


def _swiglu_forward(
    W_gate: torch.Tensor,
    W_up: torch.Tensor,
    W_down: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """Standard SwiGLU FFN forward used by Qwen3-MoE experts.

    PyTorch nn.Linear weight shapes (used by the bank get/set):
        W_gate, W_up : (d_int, hidden)      — applied as ``F.linear(x, W)``
        W_down       : (hidden, d_int)
    Input ``x`` has shape ``(*, hidden)``; output has shape ``(*, hidden)``.
    """
    gate = F.linear(x, W_gate)
    up = F.linear(x, W_up)
    intermediate = F.silu(gate) * up
    return F.linear(intermediate, W_down)


def _snapshot_pre_merge_layer_experts(
    layer_ref: MoELayerRef,
) -> dict[int, dict[str, torch.Tensor]]:
    """CPU snapshot of every expert's gate/up/down weights for a single
    layer, taken BEFORE the merge step mutates the bank.

    Used by step 7b (distillation) to compute the pre-merge group-member
    forward as the distillation target. Released by the per-layer driver
    once distillation finishes for the layer.
    """
    banks = build_banks(layer_ref)
    out: dict[int, dict[str, torch.Tensor]] = {}
    n = layer_ref.num_routed_experts
    for eid in range(n):
        out[eid] = {
            name: banks[name].get(eid).detach().cpu().clone()
            for name in MATRIX_NAMES
        }
    return out


def _distill_merged_group(
    *,
    layer_ref: MoELayerRef,
    centroid_id: int,
    members: list[int],
    freq: dict[int, int],
    pre_merge_weights: dict[int, dict[str, torch.Tensor]],
    layer_inputs: torch.Tensor,
    steps: int,
    lr: float,
    betas: tuple[float, float],
    plateau_steps: int,
    plateau_eps: float,
    token_cap: int,
    device: torch.device,
) -> dict:
    """500-step MSE distillation of the merged centroid against the
    freq-weighted pre-merge group-member forward (spec § 5 step 7b / M8).

    **v1 simplification — see D-expert-distill-mse-v1 in spec § 10**: this
    implementation differs from the pinned spec target in two ways:
    (i) freq-weighted-only target ``Σ (freq_e / Σ freq) · E_e^orig(x)``
        (no per-token routing weight ``g_e^orig(x)``);
    (ii) input tokens are the reservoir-sampled layer-input ``layer_inputs``
         for every group, not the routing-restricted ``X_g`` set.
    Phase 3 v2 will lift both. The v1 form provides a correctly-signed
    merge-error gradient on a uniform-token sample.

    Returns a small state dict with the final loss, step count, and break
    reason. The optimizer state is NOT persisted — resume re-runs the
    distillation from scratch for any layer whose partial JSON is missing.
    """
    if steps <= 0 or len(members) <= 1:
        return {"steps": 0, "skip": "trivial"}

    banks = build_banks(layer_ref)
    # Trainable: only the merged centroid's three projections. We pull the
    # current (post-merge) weights, wrap them as nn.Parameter, optimize, then
    # write back. Using nn.Parameter (not the bank tensors directly) lets us
    # build an optimizer cleanly without monkey-patching requires_grad on the
    # shared bank tensor.
    init_gate = banks["gate_proj"].get(centroid_id).to(device, dtype=torch.float32).clone()
    init_up   = banks["up_proj"].get(centroid_id).to(device, dtype=torch.float32).clone()
    init_down = banks["down_proj"].get(centroid_id).to(device, dtype=torch.float32).clone()
    p_gate = nn.Parameter(init_gate)
    p_up   = nn.Parameter(init_up)
    p_down = nn.Parameter(init_down)

    optim = torch.optim.AdamW(
        [p_gate, p_up, p_down], lr=lr, betas=betas, weight_decay=0.0,
    )

    # Token cap: subsample deterministically per layer for reproducibility.
    rng = torch.Generator(device="cpu").manual_seed(layer_ref.layer_idx)
    n_tokens = layer_inputs.shape[0]
    if n_tokens > token_cap:
        idx = torch.randperm(n_tokens, generator=rng)[:token_cap]
        x_all = layer_inputs[idx]
    else:
        x_all = layer_inputs
    x_all = x_all.to(device, dtype=torch.float32)

    # Build the freq-weighted target once (it doesn't change during training).
    weights = np.array([max(freq.get(m, 0), 0) for m in members], dtype=np.float64)
    if weights.sum() <= 0.0:
        weights[:] = 1.0
    weights = weights / weights.sum()

    with torch.no_grad():
        target = torch.zeros_like(x_all)
        for w, m in zip(weights, members):
            W_g = pre_merge_weights[m]["gate_proj"].to(device, dtype=torch.float32)
            W_u = pre_merge_weights[m]["up_proj"  ].to(device, dtype=torch.float32)
            W_d = pre_merge_weights[m]["down_proj"].to(device, dtype=torch.float32)
            target = target + float(w) * _swiglu_forward(W_g, W_u, W_d, x_all)

    initial_loss = None
    plateau_counter = 0
    last_step = 0
    final_loss = float("inf")
    break_reason = "max_steps"

    for step in range(steps):
        optim.zero_grad(set_to_none=True)
        student = _swiglu_forward(p_gate, p_up, p_down, x_all)
        loss = F.mse_loss(student, target)
        loss.backward()
        optim.step()
        last_step = step + 1
        final_loss = float(loss.detach().item())

        if initial_loss is None:
            initial_loss = max(final_loss, 1e-12)

        # Plateau early-break: ``relative_loss = final / initial`` falling
        # below ``plateau_eps`` for ``plateau_steps`` consecutive steps stops
        # training. Uses < (strict) so the very first step at exact threshold
        # is NOT counted, matching spec wording "below 1e-4 of the initial".
        if final_loss / initial_loss < plateau_eps:
            plateau_counter += 1
            if plateau_counter >= plateau_steps:
                break_reason = "plateau"
                break
        else:
            plateau_counter = 0

    # Write the trained weights back to the bank in the original dtype.
    bank_dtype = banks["gate_proj"].get(centroid_id).dtype
    with torch.no_grad():
        banks["gate_proj"].set(centroid_id, p_gate.detach().to(bank_dtype))
        banks["up_proj"  ].set(centroid_id, p_up.detach().to(bank_dtype))
        banks["down_proj"].set(centroid_id, p_down.detach().to(bank_dtype))

    return {
        "steps": last_step,
        "final_loss": final_loss,
        "initial_loss": float(initial_loss) if initial_loss is not None else None,
        "break_reason": break_reason,
    }


# ===========================================================================
# Stage-2 per-layer merge-heal — _heal_layer + checkpoint helpers
# ===========================================================================


def _capture_mlp_io(
    model,
    layer_ref: MoELayerRef,
    calib_batches,
    *,
    device: torch.device,
    pool_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Capture row-aligned ``(mlp_input, mlp_output)`` pools for one MoE layer.

    Runs a forward over ``calib_batches`` with :func:`early_exit_after_layer`
    so each pass stops right after ``layer_ref.layer_idx``. A forward hook on
    ``layer_ref.mlp`` records the MoE-block INPUT (``inputs[0]`` — the
    post-attention / post-layernorm hidden state fed into the block) and the
    MoE-block OUTPUT. This MUST be called BEFORE the layer is merged, while
    ``layer_ref.mlp`` is still its original pre-merge self — so the captured
    output is the self-distillation target.

    Returns two CPU bf16 ``[n, hidden]`` tensors with ``n == min(pool_size,
    total calibration tokens)``; row ``i`` of both is the same token. The
    pools are a contiguous prefix of the calibration tokens, and the forward
    loop stops early once ``pool_size`` rows are in hand.

    The prefix is an UNBIASED uniform sample — NOT because of any sampling
    here, but because :func:`build_calibration_tensor`
    (``utils/calibration.py``) globally shuffles every calibration sequence
    with ``torch.randperm`` before they are batched. The first N sequences
    are therefore already a uniform random draw from the full corpus, so a
    contiguous prefix of their tokens is itself a uniform random sample. No
    reservoir sampling is needed (and adding it would be redundant).
    """
    layer_idx = layer_ref.layer_idx
    in_chunks: list[torch.Tensor] = []
    out_chunks: list[torch.Tensor] = []
    captured = 0

    def _hook(_module, inputs, output):
        nonlocal captured
        x_in = inputs[0]
        x_out = output[0] if isinstance(output, tuple) else output
        in_flat = x_in.reshape(-1, x_in.shape[-1]).detach().to("cpu", torch.bfloat16)
        out_flat = x_out.reshape(-1, x_out.shape[-1]).detach().to("cpu", torch.bfloat16)
        in_chunks.append(in_flat)
        out_chunks.append(out_flat)
        captured += in_flat.shape[0]

    handle = layer_ref.mlp.register_forward_hook(_hook)
    was_training = model.training
    model.train(False)
    try:
        with early_exit_after_layer(model, layer_idx):
            for batch in calib_batches:
                if captured >= pool_size:
                    # Checked once per batch, so capture may overshoot
                    # pool_size by up to one full batch — deliberate; the
                    # final [:pool_size] slice below truncates the excess.
                    break
                with torch.no_grad():
                    try:
                        model(input_ids=batch.to(device))
                    except _EarlyExitException:
                        pass  # expected — target layer completed
    finally:
        handle.remove()
        if was_training:
            model.train()

    if not in_chunks:
        raise RuntimeError(
            f"merge-heal capture: layer {layer_idx} mlp forward hook never "
            "fired — calib_batches empty or layer never reached."
        )
    x_in = torch.cat(in_chunks, dim=0)[:pool_size].contiguous()
    x_out = torch.cat(out_chunks, dim=0)[:pool_size].contiguous()
    if x_in.shape[0] < pool_size:
        log.warning(
            "  merge-heal capture: layer %d — calibration data starved: "
            "%d token rows captured vs requested pool_size=%d; healing will "
            "proceed with the smaller pool.",
            layer_idx, x_in.shape[0], pool_size,
        )
    if x_in.shape != x_out.shape:
        raise RuntimeError(
            f"merge-heal capture: layer {layer_idx} input/target pools "
            f"misaligned ({tuple(x_in.shape)} vs {tuple(x_out.shape)})."
        )
    # Sanity-check the captured feature dimension against the model's hidden
    # size: a wrong hook point (or a tuple-unpack bug) would surface here as a
    # mismatched last dim rather than as a confusing downstream matmul error.
    expected_hidden = layer_ref.router.weight.shape[-1]
    if x_in.shape[-1] != expected_hidden:
        raise RuntimeError(
            f"merge-heal capture: layer {layer_idx} captured feature dim "
            f"{x_in.shape[-1]} != model hidden size {expected_hidden} — the "
            "mlp forward hook captured the wrong tensor."
        )
    log.info(
        "  merge-heal capture: layer %d — %d (input,target) token rows "
        "captured (pool_size=%d)",
        layer_idx, x_in.shape[0], pool_size,
    )
    return x_in, x_out


def _heal_student_moe_output(
    *,
    x: torch.Tensor,
    router_weight: torch.Tensor,
    router_bias: torch.Tensor | None,
    esc_bias: torch.Tensor | None,
    expert_params: dict[int, dict[str, torch.Tensor]],
    centroid_order: list[int],
    top_k: int,
    shared_out: torch.Tensor,
) -> torch.Tensor:
    """Faithful student replica of ``Qwen3_5MoeSparseMoeBlock.forward``.

    Reproduces the model's routed-expert path exactly (see
    ``Qwen3_5MoeTopKRouter.forward`` / ``Qwen3_5MoeSparseMoeBlock.forward``):
      1. ``router_logits = softmax(F.linear(x, router_weight) [+ bias], dim=-1)``
         — full softmax over all KEPT experts.
      2. ``topk_vals, topk_idx = topk(router_logits, top_k)``; then
         ``topk_vals /= topk_vals.sum(-1, keepdim=True)`` — the top-k
         renormalization the Qwen router applies.
      3. routed output ``= Σ_k topk_vals[:,k] · SwiGLU_{expert(k)}(x)``.
      4. block output ``= routed_output + shared_expert_output``.

    This returns that single block-output hidden-state tensor — the same
    quantity ``_capture_mlp_io`` records as the self-distillation target.

    The ``e_score_correction_bias`` term is added to the pre-softmax logits
    when the router exposes it (mirrors ``_router_routing_weights``); the base
    Qwen3.5 router has neither bias, so for that model both are ``None`` and
    this matches ``Qwen3_5MoeTopKRouter.forward`` bit-for-bit.

    Routed-expert dispatch is SPARSE: each expert's SwiGLU runs only on the
    token rows where that expert is in the top-k (``index_select`` to gather
    the subset, ``index_add`` to scatter the weighted result back). This
    matches the dense ``Σ_pos w·SwiGLU(x)`` formulation exactly — every token
    not selecting expert ``pos`` had weight 0 there — while avoiding the
    ``n_kept/top_k``× wasted SwiGLU compute of evaluating every expert on every
    token. Both ``index_select`` and ``index_add`` are differentiable, so
    gradients still flow to the trainable expert params.

    All inputs are fp32 and on the same device; returns fp32 ``[T, hidden]``.
    The shared-expert output is precomputed by the caller (it does not depend
    on the trainable params) and simply added back.
    """
    logits = F.linear(x, router_weight)
    if router_bias is not None:
        logits = logits + router_bias
    if esc_bias is not None:
        logits = logits + esc_bias
    router_probs = F.softmax(logits, dim=-1)  # (T, n_kept)
    k = min(top_k, router_probs.shape[-1])
    topk_vals, topk_idx = torch.topk(router_probs, k=k, dim=-1)  # (T, k)
    topk_vals = topk_vals / topk_vals.sum(dim=-1, keepdim=True)

    routed = torch.zeros_like(shared_out)
    n_kept = router_weight.shape[0]
    # Per-token per-position weight for expert `pos`: the renormalized top-k
    # value where this expert was selected, else 0. Summing over the k slots
    # collapses the (rare) case of the same expert appearing twice.
    for pos in range(n_kept):
        cid = centroid_order[pos]
        ep = expert_params[cid]
        sel = (topk_idx == pos)  # (T, k) bool
        w = (topk_vals * sel.to(topk_vals.dtype)).sum(dim=-1)  # (T,)
        rows = torch.nonzero(w, as_tuple=False).squeeze(-1)  # tokens routed here
        if rows.numel() == 0:
            continue
        x_sub = x.index_select(0, rows)
        e_out = _swiglu_forward(
            ep["gate_proj"], ep["up_proj"], ep["down_proj"], x_sub
        )
        contrib = w.index_select(0, rows).unsqueeze(-1) * e_out
        routed = routed.index_add(0, rows, contrib)
    return routed + shared_out


def _heal_layer(
    *,
    layer_ref: MoELayerRef,
    final_kept_ids: list[int],
    captured_input: torch.Tensor,
    captured_target: torch.Tensor,
    heal_cfg: "_HealConfig",
    device: torch.device,
) -> dict:
    """Per-layer merge-heal by SELF-DISTILLATION.

    Fine-tunes one already-merged layer so it reproduces its OWN pre-merge
    MoE-block output:
      * EVERY kept expert trains — merged centroids AND singletons. The router
        was resized, so even a singleton's contribution to the block output is
        perturbed and benefits from healing.
      * the resized ``router.weight`` (+ ``router.bias`` if present) trains
        only when ``heal_cfg.train_router`` is True; otherwise the router is
        frozen at its mechanical resize.

    Loss = ``F.mse_loss(student_moe_out, captured_target)`` in fp32, where
    ``student_moe_out`` is :func:`_heal_student_moe_output` over
    ``captured_input`` (the layer's own pre-merge MoE-block input, row-aligned
    with ``captured_target`` — both from :func:`_capture_mlp_io`).

    90/10 train/held-out split (``layer_idx``-seeded), patience early-stop on
    held-out MSE. **Monotone-safe accept/reject:** the un-healed plain-merged
    weights are snapshotted before training; if the best healed held-out MSE
    does not beat the plain-merged held-out MSE the layer is REVERTED to the
    plain merge (worst case per layer == the no-heal baseline).

    Returns a JSON-able state dict:
        ``{"steps", "train_mse", "train_mse_at_best", "holdout_mse",
        "plain_merged_holdout_mse", "heal_gap", "accepted", "stop_reason"}``.
    """
    layer_idx = layer_ref.layer_idx
    router = layer_ref.router

    # --- Row-alignment hard-assert (input pool must match target pool) -----
    if captured_input.shape != captured_target.shape:
        raise RuntimeError(
            f"merge-heal layer {layer_idx}: captured input pool "
            f"{tuple(captured_input.shape)} != target pool "
            f"{tuple(captured_target.shape)} — token alignment is broken."
        )
    x_buf = captured_input

    banks = build_banks(layer_ref)
    # CRITICAL indexing contract: `_heal_layer` runs AFTER `bank.select(
    # final_kept_ids)` has rewritten the stacked expert tensors to hold only
    # the kept rows, re-indexed 0..n_kept-1. So banks here MUST be indexed by
    # post-select POSITION, not by the original expert id. Position `pos`
    # corresponds to original id `final_kept_ids[pos]`, and the resized router
    # row `pos` likewise — so `pos` is the consistent index across banks +
    # router. `expert_params` / `healed_experts` stay keyed by original
    # `cid` (that is what `centroid_order=final_kept_ids` expects downstream).
    pos_of: dict[int, int] = {cid: i for i, cid in enumerate(final_kept_ids)}
    bank_dtype = banks["gate_proj"].get(0).dtype

    # --- Trainable expert params: EVERY kept expert (merged + singleton) ----
    # expert_params holds a trainable fp32 leaf for every kept expert.
    expert_params: dict[int, dict[str, torch.Tensor]] = {}
    trainable_2d: list[nn.Parameter] = []
    healed_experts: list[int] = list(final_kept_ids)
    for pos, cid in enumerate(final_kept_ids):
        ep: dict[str, torch.Tensor] = {}
        for name in MATRIX_NAMES:
            # .detach() is essential: banks[name].get(pos) is a VIEW of the
            # model's stacked-experts nn.Parameter (requires_grad=True during
            # Stage 2). Without detach, loss.backward() would accumulate
            # gradients onto the full model's expert tensors — an unbounded
            # GPU memory leak across the heal run.
            w = banks[name].get(pos).detach().to(device, torch.float32).clone()
            p = nn.Parameter(w, requires_grad=True)
            trainable_2d.append(p)
            ep[name] = p
        expert_params[cid] = ep

    # --- Router: whole resized router.weight (+ bias) — trained iff flagged -
    trainable_1d: list[nn.Parameter] = []
    router_bias_p: torch.Tensor | None = None
    if heal_cfg.train_router:
        router_weight: torch.Tensor = nn.Parameter(
            router.weight.detach().to(device, torch.float32).clone(),
            requires_grad=True,
        )
        trainable_2d.append(router_weight)
        if getattr(router, "bias", None) is not None:
            router_bias_p = nn.Parameter(
                router.bias.detach().to(device, torch.float32).clone(),
                requires_grad=True,
            )
            trainable_1d.append(router_bias_p)
    else:
        # Frozen: detached tensors (no grad / not optimized) — still fed to
        # the student forward so routing is correct.
        router_weight = router.weight.detach().to(device, torch.float32)
        if getattr(router, "bias", None) is not None:
            router_bias_p = router.bias.detach().to(device, torch.float32)
    # e_score_correction_bias (if any) is part of the router's pre-softmax
    # path but is NOT trained here — kept frozen, mirrors _router_routing_weights.
    esc = getattr(router, "e_score_correction_bias", None)
    esc_bias = esc.detach().to(device, torch.float32) if esc is not None else None

    # Every kept expert is trainable, so trainable_2d is always non-empty —
    # the optimizer always has at least one parameter.
    top_k = layer_ref.top_k

    # --- Calibration token rows: 90/10 split (layer-idx-seeded) ------------
    n_rows = x_buf.shape[0]
    gen = torch.Generator(device="cpu").manual_seed(layer_idx)
    perm = torch.randperm(n_rows, generator=gen)
    n_use = perm.shape[0]
    n_holdout = max(1, int(round(n_use * heal_cfg.holdout_fraction)))
    n_holdout = min(n_holdout, n_use - 1)  # leave at least 1 training row
    holdout_idx = perm[:n_holdout]
    train_idx = perm[n_holdout:]

    # Move the token rows + self-distillation targets to device, fp32.
    x_train = x_buf[train_idx].to(device, torch.float32)
    x_holdout = x_buf[holdout_idx].to(device, torch.float32)
    t_train = captured_target[train_idx].to(device, torch.float32)
    t_holdout = captured_target[holdout_idx].to(device, torch.float32)

    # --- Frozen shared-expert output (precomputed; not trained) ------------
    # The captured target is the full MoE-block output, which includes the
    # sigmoid-gated shared expert. The shared expert is protected by Stage 2
    # (untouched), so we run the model's own shared_expert / shared_expert_gate
    # and add the result back — exactly like Qwen3_5MoeSparseMoeBlock.forward.
    mlp = layer_ref.mlp
    shared_expert = layer_ref.shared_expert  # canonical MoELayerRef accessor
    shared_gate = getattr(mlp, "shared_expert_gate", None)  # no MoELayerRef field

    def _shared_out(x_in: torch.Tensor) -> torch.Tensor:
        if shared_expert is None:
            return torch.zeros_like(x_in)
        sp = next(shared_expert.parameters())
        sdev, sdtype = sp.device, sp.dtype
        # The heal runs in fp32, but the frozen shared expert keeps the model's
        # native dtype (bf16) — feed it in its own dtype to avoid a matmul
        # dtype mismatch, then cast the (frozen) result back to fp32.
        with torch.no_grad():
            so = shared_expert(x_in.to(sdev, sdtype))
            if shared_gate is not None:
                so = torch.sigmoid(shared_gate(x_in.to(sdev, sdtype))) * so
        return so.to(x_in.device, torch.float32)

    shared_train = _shared_out(x_train)
    shared_holdout = _shared_out(x_holdout)

    # Single fp32 AdamW over every trainable param (all kept experts, plus the
    # router when heal_cfg.train_router). weight_decay=0 — the self-distillation
    # target already anchors the weights to the pre-merge function.
    opt = torch.optim.AdamW(
        trainable_2d + trainable_1d, lr=heal_cfg.lr,
        betas=heal_cfg.adamw_betas, weight_decay=0.0,
    )
    clip_params = trainable_2d + trainable_1d

    def _forward(x_in, shared_in):
        return _heal_student_moe_output(
            x=x_in,
            router_weight=router_weight,
            router_bias=router_bias_p,
            esc_bias=esc_bias,
            expert_params=expert_params,
            centroid_order=final_kept_ids,
            top_k=top_k,
            shared_out=shared_in,
        )

    # --- Minibatched held-out MSE ------------------------------------------
    # Evaluated in chunks of merge_heal_minibatch_size rows so the held-out
    # forward never materializes activations for the whole held-out set at
    # once — bounds peak memory on wide layers exactly like the train step.
    mb = max(1, heal_cfg.minibatch_size)

    @torch.no_grad()
    def _holdout_mse() -> float:
        sq_sum = 0.0
        n_elem = 0
        for c in range(0, x_holdout.shape[0], mb):
            pred = _forward(x_holdout[c:c + mb], shared_holdout[c:c + mb])
            tgt = t_holdout[c:c + mb]
            sq_sum += float(((pred - tgt) ** 2).sum().item())
            n_elem += tgt.numel()
        return sq_sum / max(1, n_elem)

    # --- Best-snapshot bookkeeping -----------------------------------------
    # Snapshots are kept on CPU so the best-state copy does not double the
    # GPU footprint of the trainable params on the widest layer.
    def _snapshot() -> dict:
        return {
            "experts": {
                cid: {n: expert_params[cid][n].detach().to("cpu").clone()
                      for n in MATRIX_NAMES}
                for cid in healed_experts
            },
            # Router params are cloned only when trained — the restore block
            # below is guarded by `if heal_cfg.train_router`, so a frozen
            # router's snapshot data would be dead weight.
            "router_weight": (
                router_weight.detach().to("cpu").clone()
                if heal_cfg.train_router else None
            ),
            "router_bias": (
                router_bias_p.detach().to("cpu").clone()
                if (heal_cfg.train_router and router_bias_p is not None)
                else None
            ),
        }

    # plain_merged_* is the un-healed baseline for the monotone-safe accept/
    # reject guard: the initial holdout MSE + an independent weight snapshot.
    plain_merged_holdout_mse = _holdout_mse()
    plain_merged_state = _snapshot()
    best_holdout = plain_merged_holdout_mse
    best_state = _snapshot()
    # train_mse_at_best pairs with best_holdout — the train MSE recorded at the
    # SAME step the best held-out was seen, so heal_gap is a like-for-like
    # (held-out − train) comparison rather than best-vs-final.
    train_mse_at_best = float("nan")
    evals_since_improve = 0
    last_train_mse = float("nan")
    stop_reason = "max_steps"
    steps_done = 0

    # --- Minibatch sampler over the fixed training pool --------------------
    # Each step draws a fresh random minibatch of merge_heal_minibatch_size
    # rows from x_train; seeded for reproducibility across resumes.
    n_train = x_train.shape[0]
    mb_gen = torch.Generator(device="cpu").manual_seed(layer_idx + 1)

    for step in range(heal_cfg.max_steps):
        if n_train > mb:
            sel = torch.randperm(n_train, generator=mb_gen)[:mb]
            xb = x_train.index_select(0, sel.to(x_train.device))
            sb = shared_train.index_select(0, sel.to(shared_train.device))
            tb = t_train.index_select(0, sel.to(t_train.device))
        else:
            xb, sb, tb = x_train, shared_train, t_train
        opt.zero_grad(set_to_none=True)
        student = _forward(xb, sb)
        loss = F.mse_loss(student, tb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(clip_params, heal_cfg.grad_clip)
        opt.step()
        steps_done = step + 1
        last_train_mse = float(loss.detach().item())

        if steps_done % heal_cfg.eval_interval == 0:
            h = _holdout_mse()
            if h < best_holdout:
                best_holdout = h
                best_state = _snapshot()
                train_mse_at_best = last_train_mse
                evals_since_improve = 0
            else:
                evals_since_improve += 1
                if evals_since_improve >= heal_cfg.patience:
                    stop_reason = "patience"
                    break

    # --- Accept/reject: monotone-safe guard --------------------------------
    # Accept the healed weights only if the best healed held-out MSE strictly
    # beats the plain-merged baseline; otherwise REVERT the layer to the plain
    # merge (worst case per layer == the no-heal baseline).
    accepted = bool(best_holdout < plain_merged_holdout_mse)
    restore_state = best_state if accepted else plain_merged_state

    # bank.set() casts dtype + moves device internally; the router params are
    # re-installed with an explicit dtype+device cast so they land back on the
    # router's original device (the snapshot lives on CPU).
    with torch.no_grad():
        for cid in healed_experts:
            pos = pos_of[cid]  # banks are post-select position-indexed
            for name in MATRIX_NAMES:
                banks[name].set(pos, restore_state["experts"][cid][name].to(bank_dtype))
        # Only reinstall the router when it was actually trained. With
        # train_router=False the router was never an optimizer parameter, so
        # router.weight/bias still hold exactly what the mechanical resize
        # left — reinstalling would be a wasteful no-op that also obscures the
        # frozen-router contract.
        if heal_cfg.train_router:
            _rw = router.weight
            router.weight = nn.Parameter(
                restore_state["router_weight"].to(device=_rw.device, dtype=_rw.dtype),
                requires_grad=_rw.requires_grad,
            )
            if getattr(router, "bias", None) is not None and restore_state["router_bias"] is not None:
                _rb = router.bias
                router.bias = nn.Parameter(
                    restore_state["router_bias"].to(device=_rb.device, dtype=_rb.dtype),
                    requires_grad=_rb.requires_grad,
                )

    # heal_gap pairs the best held-out MSE with the train MSE recorded at the
    # SAME step (train_mse_at_best). If no eval ever improved on the initial
    # snapshot (e.g. max_steps < eval_interval), there is no matching train
    # step — fall back to the final-step train MSE so the gap stays defined.
    gap_train_mse = (
        train_mse_at_best
        if not math.isnan(train_mse_at_best)
        else last_train_mse
    )
    heal_gap = best_holdout - gap_train_mse
    log.info(
        "  merge-heal layer %d: %d steps (%s) — train_mse=%.6e (at_best=%.6e) "
        "holdout_mse=%.6e plain_merged_holdout_mse=%.6e heal_gap=%.6e "
        "accepted=%s (%d kept experts, train_router=%s)",
        layer_idx, steps_done, stop_reason, last_train_mse, gap_train_mse,
        best_holdout, plain_merged_holdout_mse, heal_gap, accepted,
        len(final_kept_ids), heal_cfg.train_router,
    )
    return {
        "steps": steps_done,
        "train_mse": last_train_mse,
        "train_mse_at_best": gap_train_mse,
        "holdout_mse": best_holdout,
        "plain_merged_holdout_mse": plain_merged_holdout_mse,
        "heal_gap": heal_gap,
        "accepted": accepted,
        "stop_reason": stop_reason,
    }


def _write_heal_weights(
    partial_dir: Path,
    layer_ref: MoELayerRef,
    final_kept_ids: list[int],
    *,
    accepted: bool,
) -> None:
    """Persist the post-heal expert tensors + router weight/bias for one layer.

    Post-heal weights are NOT reconstructible from ``merge_N.json`` (the heal
    fine-tunes them), so a per-layer ``_heal_weights_layer_{N}.pt`` is written
    into ``partial_dir`` — atomically, and BEFORE ``_write_merge_json`` so the
    ``.pt``-before-``.json`` resume invariant holds.

    EVERY kept expert is stored (all kept experts are trained by the heal);
    when the layer's heal was rejected the banks already hold the plain-merged
    weights, so the file faithfully captures whatever final state the layer is
    in. ``accepted`` is recorded as telemetry.
    """
    layer_idx = layer_ref.layer_idx
    banks = build_banks(layer_ref)
    router = layer_ref.router

    # Keyed by post-select POSITION (banks were re-indexed to 0..n_kept-1 by
    # bank.select()); _load_heal_weights replays by the same position. All
    # kept experts are stored.
    healed: dict[str, dict[str, torch.Tensor]] = {}
    for pos, _cid in enumerate(final_kept_ids):
        healed[str(pos)] = {
            name: banks[name].get(pos).detach().cpu().clone()
            for name in MATRIX_NAMES
        }
    payload = {
        "format_version": _HEAL_WEIGHTS_FORMAT_VERSION,
        "layer_idx": layer_idx,
        "accepted": bool(accepted),
        "healed_experts": healed,
        "router_weight": router.weight.detach().cpu().clone(),
        "router_bias": (
            router.bias.detach().cpu().clone()
            if getattr(router, "bias", None) is not None else None
        ),
    }
    tmp = partial_dir / f"_heal_weights_layer_{layer_idx}.pt.tmp"
    final = partial_dir / f"_heal_weights_layer_{layer_idx}.pt"
    torch.save(payload, tmp)
    _durable_rename(tmp, final)


def _load_heal_weights(
    partial_dir: Path,
    layer_ref: MoELayerRef,
    final_kept_ids: list[int],
) -> None:
    """Apply a persisted ``_heal_weights_layer_{N}.pt`` to the banks + router.

    Used by the resume path: a layer that completed its heal in a prior run
    has a healed-weights file; reload it so the in-memory model matches the
    state the heal left. A missing file is fatal — the operator must delete
    ``_stage2_partial/`` and re-run.

    For an ACCEPTED layer the persisted weights are genuinely post-heal and
    are NOT reconstructible from ``merge_*.json``. For a REJECTED layer the
    banks were reverted to the plain merge, so the file simply re-persists
    that plain-merged state. The file is written for every *merged* layer
    regardless of the accept/reject outcome, and reloaded here on resume
    whenever it is present; a 0-merge layer has no heal-weights file (the
    heal is skipped), so the caller gates this load on the file existing.
    """
    layer_idx = layer_ref.layer_idx
    path = partial_dir / f"_heal_weights_layer_{layer_idx}.pt"
    if not path.exists():
        raise RuntimeError(
            f"Stage-2 merge-heal resume: layer {layer_idx} completed its merge "
            f"but {path.name} is missing. Healed weights are not "
            "reconstructible from merge_*.json — delete _stage2_partial/ and "
            "re-run Stage 2."
        )
    # weights_only=True is safe: the payload is only tensors + ints + str + None.
    payload = torch.load(path, map_location="cpu", weights_only=True)
    fv = int(payload.get("format_version", 0))
    if fv != _HEAL_WEIGHTS_FORMAT_VERSION:
        raise RuntimeError(
            f"{path} has format_version={fv} "
            f"(expected {_HEAL_WEIGHTS_FORMAT_VERSION}) — delete "
            "_stage2_partial/ and re-run Stage 2."
        )
    banks = build_banks(layer_ref)
    bank_dtype = banks["gate_proj"].get(0).dtype
    n_kept = len(final_kept_ids)
    with torch.no_grad():
        # healed_experts is keyed by post-select position (see _write_heal_weights):
        # banks were re-indexed to 0..n_kept-1 by bank.select().
        for pos_str, mats in payload["healed_experts"].items():
            pos = int(pos_str)
            if not (0 <= pos < n_kept):
                raise RuntimeError(
                    f"{path}: healed-expert position {pos} out of range "
                    f"[0, {n_kept}) — heal-weights file inconsistent with "
                    "merge_*.json."
                )
            for name in MATRIX_NAMES:
                banks[name].set(pos, mats[name].to(bank_dtype))
        router = layer_ref.router
        router.weight = nn.Parameter(
            payload["router_weight"].to(
                device=router.weight.device, dtype=router.weight.dtype,
            ),
            requires_grad=router.weight.requires_grad,
        )
        if payload.get("router_bias") is not None and getattr(router, "bias", None) is not None:
            router.bias = nn.Parameter(
                payload["router_bias"].to(
                    device=router.bias.device, dtype=router.bias.dtype,
                ),
                requires_grad=router.bias.requires_grad,
            )


def _summarize_distill_state(
    distill_state: dict[int, dict] | None,
) -> dict[str, int | float]:
    """Aggregate per-merged-group distillation outcomes into per-layer scalars
    for Trackio emission (spec § 5 step 7b / M8).

    Returns a dict with four keys:
        ``stage2/distill_groups``       — int, number of non-singleton groups
                                          actually distilled this layer.
        ``stage2/distill_mean_final_loss`` — float, mean of per-group ``final_loss``
                                          (NaN when no groups distilled).
        ``stage2/distill_mean_steps``   — float, mean step count across groups
                                          (reflects plateau-break behavior — ratio
                                          to ``expert_distill_steps`` shows how
                                          aggressively groups converged).
        ``stage2/distill_plateau_breaks`` — int, count of groups whose
                                          ``break_reason == "plateau"``.

    Returns an empty dict when ``distill_state is None`` (distillation
    disabled or no non-singleton groups). Caller's `_trackio_log({**existing, **summary})`
    pattern then naturally omits the keys for that layer.
    """
    if not distill_state:
        return {}
    groups = list(distill_state.values())
    # Skip "trivial" skips (singletons, zero-steps) so the means reflect actual
    # distillation work, not no-op placeholders.
    real = [g for g in groups if g.get("skip") != "trivial" and g.get("final_loss") is not None]
    if not real:
        return {
            "stage2/distill_groups": 0,
            "stage2/distill_mean_final_loss": float("nan"),
            "stage2/distill_mean_steps": 0.0,
            "stage2/distill_plateau_breaks": 0,
        }
    n = len(real)
    final_losses = [float(g["final_loss"]) for g in real]
    steps = [int(g.get("steps", 0)) for g in real]
    plateaus = sum(1 for g in real if g.get("break_reason") == "plateau")
    return {
        "stage2/distill_groups": n,
        "stage2/distill_mean_final_loss": sum(final_losses) / n,
        "stage2/distill_mean_steps": sum(steps) / n,
        "stage2/distill_plateau_breaks": plateaus,
    }


def _build_grouped_from_assignment(
    assignment: list[int],
    centroid_ids: list[int],
    noncentroid_ids: list[int],
) -> dict[int, list[int]]:
    """Reconstruct ``{centroid_id: [centroid_id, *absorbed_member_ids]}``
    from a flat assignment list (centroid index per non-centroid)."""
    grouped: dict[int, list[int]] = {c: [c] for c in centroid_ids}
    for child_pos, c_idx in enumerate(assignment):
        if c_idx >= 0:
            grouped[centroid_ids[c_idx]].append(noncentroid_ids[child_pos])
    return grouped


def _em_compute_tentative_weights(
    layer_ref: MoELayerRef,
    grouped: dict[int, list[int]],
    freq: dict[int, int],
    ream_acc: ReamCostAccumulator | None,
    perm_cache: "_PermAlignCache | None",
) -> dict[int, dict[str, torch.Tensor]]:
    """Compute the tentative freq-weighted merged centroid weights for every
    non-singleton group, WITHOUT mutating the bank.

    For each centroid c with members [c, m1, m2, ...]:
        W_c_tentative = Σ (freq_e / Σ freq) · perm_e(W_e)

    Permutations come from ``perm_cache`` if available; otherwise computed
    fresh via ``_permutation_align_to_centroid`` (the centroid contributes
    with identity permutation).

    Used by EM refinement (spec § 5 step 4T(e)) to recompute the cost matrix
    against the tentative merged centroid before reassigning.
    """
    li = layer_ref.layer_idx
    banks = build_banks(layer_ref)
    out: dict[int, dict[str, torch.Tensor]] = {}

    for centroid, members in grouped.items():
        if len(members) <= 1:
            continue  # singleton — nothing to merge

        weights = np.array([max(freq.get(m, 0), 0) for m in members], dtype=np.float64)
        if weights.sum() <= 0.0:
            weights[:] = 1.0
        weights /= weights.sum()

        ref_gate = banks["gate_proj"].get(centroid).to(torch.float32)
        ref_up   = banks["up_proj"].get(centroid).to(torch.float32)
        ref_act  = ream_acc.get_neuron_mean(li, centroid) if ream_acc else None

        accs: dict[str, torch.Tensor | None] = {name: None for name in banks}
        for w, m in zip(weights, members):
            gate_m = banks["gate_proj"].get(m).to(torch.float32)
            up_m   = banks["up_proj"].get(m).to(torch.float32)
            child_act = ream_acc.get_neuron_mean(li, m) if ream_acc else None

            if m == centroid:
                perm = None
            else:
                cached = (
                    perm_cache.get((li, centroid, m))
                    if perm_cache is not None
                    else None
                )
                if cached is not None:
                    perm = cached[0]
                else:
                    perm = _permutation_align_to_centroid(
                        ref_gate, ref_up, gate_m, up_m,
                        ref_act_mean=ref_act, child_act_mean=child_act,
                    )

            for name, bank in banks.items():
                if name == "gate_proj":
                    Wm = gate_m
                elif name == "up_proj":
                    Wm = up_m
                else:
                    Wm = bank.get(m).to(torch.float32)
                if perm is not None:
                    Wm = Wm[perm, :] if name in ("gate_proj", "up_proj") else Wm[:, perm]
                accs[name] = Wm * w if accs[name] is None else accs[name] + Wm * w

        out[centroid] = {name: accs[name] for name in banks}

    return out


def _two_opt_refine(
    assignment: list[int],
    cost: np.ndarray,
    max_group_cap: int,
) -> list[int]:
    """Direction D — greedy + one 2-opt local-refinement loop (spec §5 step 3.5).

    Strictly-improving local search over an already-feasible child→centroid
    assignment. Operates purely on the assignment list, the cost matrix and the
    per-centroid capacity cap, so it is model-agnostic (no expert-count, dim,
    top-k or activation assumptions).

    ``assignment`` is a list of length ``n_children``; ``assignment[ch]`` is the
    centroid *index* in ``[0, n_centroids)`` that child ``ch`` is assigned to, or
    ``-1`` if unassigned. ``cost`` has shape ``(n_children, n_centroids)``.

    Two move types, both applied only when *strictly* lowering total cost:

      * **swap** — for a pair of children ``(i, j)`` in *different* groups,
        exchange their centroids if
        ``cost[i, g_j] + cost[j, g_i] < cost[i, g_i] + cost[j, g_j]``.
        A swap leaves every group size unchanged, so capacity is preserved by
        construction — but we still verify post-swap group sizes defensively.

      * **move** — relocate a single child ``i`` to a different centroid ``g``
        when ``g`` has spare capacity and ``cost[i, g] < cost[i, g_i]``.

    Capacity is re-checked on every accepted move; with ``max_group_cap <= 0``
    (uncapped, ablation-only path) groups are treated as unbounded. Passes
    repeat until a full pass makes no improving move; each pass is O(n²) and
    ``n`` (non-centroid expert count) is small. The function NEVER accepts a
    non-improving move, so the returned assignment's total cost is provably
    ``<=`` the input's.

    Unassigned children (``-1``) and any child whose current cost is non-finite
    are skipped — 2-opt only reshuffles already-feasible finite-cost merges and
    never assigns a previously-unassigned child.

    Returns a new assignment list (the input is not mutated).
    """
    n_children = len(assignment)
    if n_children == 0 or cost.size == 0:
        return list(assignment)

    n_centroids = cost.shape[1]
    result = list(assignment)

    # Per-centroid occupancy (number of non-centroid children currently in each
    # group). Unassigned children (-1) contribute to no group.
    group_size = [0] * n_centroids
    for g in result:
        if g >= 0:
            group_size[g] += 1

    # max_group_cap <= 0 → uncapped; use an effectively-infinite cap so the
    # capacity guards become no-ops without special-casing every check.
    cap = max_group_cap if max_group_cap > 0 else n_children

    def _cost(ch: int, g: int) -> float:
        return float(cost[ch, g])

    improved = True
    while improved:
        improved = False

        # --- single moves ---------------------------------------------------
        for i in range(n_children):
            g_i = result[i]
            if g_i < 0:
                continue
            cur = _cost(i, g_i)
            if not np.isfinite(cur):
                continue
            best_g = g_i
            best_cost = cur
            for g in range(n_centroids):
                if g == g_i:
                    continue
                if group_size[g] >= cap:
                    continue
                c = _cost(i, g)
                if np.isfinite(c) and c < best_cost:
                    best_cost = c
                    best_g = g
            if best_g != g_i:
                # Strict improvement, target has spare capacity.
                group_size[g_i] -= 1
                group_size[best_g] += 1
                result[i] = best_g
                improved = True

        # --- pairwise swaps -------------------------------------------------
        for i in range(n_children):
            g_i = result[i]
            if g_i < 0:
                continue
            for j in range(i + 1, n_children):
                g_j = result[j]
                if g_j < 0 or g_j == g_i:
                    continue
                cur = _cost(i, g_i) + _cost(j, g_j)
                new = _cost(i, g_j) + _cost(j, g_i)
                if not (np.isfinite(cur) and np.isfinite(new)):
                    continue
                if new < cur:
                    # A swap is size-neutral for both groups, so caps are
                    # preserved by construction. Assert the invariant loudly
                    # rather than silently skipping an improving swap.
                    assert group_size[g_i] <= cap and group_size[g_j] <= cap, (
                        "2-opt: group size exceeded cap before a size-neutral swap"
                    )
                    result[i] = g_j
                    result[j] = g_i
                    g_i = g_j
                    improved = True

    return result


def _em_refine_assignment(
    layer_ref: MoELayerRef,
    *,
    initial_assignment: list[int],
    initial_delta: np.ndarray,
    ream_centroid_ids: list[int],
    ream_noncentroid_ids: list[int],
    perm_cache: "_PermAlignCache",
    ream_acc: ReamCostAccumulator,
    cov_acc: "InputCovarianceAccumulator | None",
    freq: dict[int, int],
    max_group_cap: int,
    cost_alignment: str,
    cost_whitening: str,
    cost_asymmetric: bool,
    cost_topk_filter: int,
    assignment_solver: SolverName,
    em_rounds: int,
    em_break: bool,
    blacklisted_ids: set[int] | None,
    sinkhorn_epsilon_init: float = 1.0,
    sinkhorn_epsilon_final: float = 0.01,
    sinkhorn_iters: int = 200,
    skip_merge_percentile: float = 100.0,
) -> tuple[list[int], np.ndarray, int]:
    """EM refinement loop (spec § 5 step 4T(e) / M4).

    For each round r in 1..em_rounds:
      1. Build current groups from ``assignment``.
      2. Compute tentative merged centroid weights (freq-weighted average of
         current group members, using cached perms where available).
      3. Recompute the cost matrix with the tentative centroids substituted.
      4. Re-solve the assignment.
      5. If ``em_break`` and the new assignment equals the old, stop early.

    Returns ``(final_assignment, final_delta, rounds_completed)``. ``rounds_completed``
    is the number of rounds where step 4 actually ran (≥ 1 if em_rounds ≥ 1).

    EM is a no-op when:
      - ``em_rounds <= 0``
      - ``cost_alignment == "pre"`` (the cheap symmetric cost does not depend
        on centroid weights, so a tentative merge does not change the cost
        matrix and the assignment cannot improve).
      - ``cost_alignment == "output"`` — the output-space cost *does* depend on
        the (tentative) centroid weights, so EM would be meaningful here; it is
        deferred only because ``_em_refine_assignment`` does not thread the
        per-layer ``layer_inputs`` calibration tensors that ``_output_space_cost``
        needs. See the TODO at the cost-matrix recompute below.
    """
    if em_rounds <= 0 or cost_alignment != "post":
        return initial_assignment, initial_delta, 0

    n_nc = len(ream_noncentroid_ids)
    n_c = len(ream_centroid_ids)
    assignment = list(initial_assignment)
    delta = initial_delta
    rounds_done = 0

    for r in range(em_rounds):
        grouped = _build_grouped_from_assignment(
            assignment, ream_centroid_ids, ream_noncentroid_ids,
        )
        tentative = _em_compute_tentative_weights(
            layer_ref, grouped, freq, ream_acc, perm_cache,
        )
        if not tentative:
            # No non-singleton groups → tentative is identical to original →
            # cost matrix would be unchanged. Stop early.
            break

        new_delta = _ream_cost_matrix(
            layer_ref, ream_noncentroid_ids, ream_centroid_ids,
            ream_acc=ream_acc,
            blacklisted_ids=blacklisted_ids,
            cost_alignment=cost_alignment,
            cost_whitening=cost_whitening,
            cost_asymmetric=cost_asymmetric,
            cost_topk_filter=cost_topk_filter,
            # freq is also needed by the "output" cost (freq-weighted tentative
            # merge), not just the asymmetric "post" cost — keep consistent with
            # the main _ream_cost_matrix call site. TODO: admitting "output" to
            # the EM guard above additionally requires threading layer_inputs
            # here so _output_space_cost has its calibration tokens.
            freq=freq if (cost_asymmetric or cost_alignment == "output") else None,
            cov_acc=cov_acc,
            perm_cache=perm_cache,
            tentative_centroid_weights=tentative,
        )
        # Direction B — re-apply the skip-merge floor each EM round; the freshly
        # recomputed cost matrix would otherwise un-mask the high-cost pairs.
        if skip_merge_percentile < 100.0:
            new_delta, _ = _apply_skip_merge_floor(new_delta, skip_merge_percentile)
        new_assignment = _assign_children_to_centroids(
            new_delta, n_nc, n_c, max_group_cap,
            solver=assignment_solver,
            sinkhorn_epsilon_init=sinkhorn_epsilon_init,
            sinkhorn_epsilon_final=sinkhorn_epsilon_final,
            sinkhorn_iters=sinkhorn_iters,
        )
        rounds_done = r + 1
        # F2 fix: commit ``delta = new_delta`` BEFORE the break check so
        # downstream assigned_cost reporting uses the EM-refined cost matrix
        # even when the assignment converged this round.
        delta = new_delta
        if em_break and new_assignment == assignment:
            break
        assignment = new_assignment

    return assignment, delta, rounds_done


def _pick_effective_alignment(
    *,
    n_nc: int,
    n_c: int,
    max_group_cap: int,
    threshold: float,
    configured: str,
) -> str:
    """Decide SLACK vs TIGHT for the cost-matrix path (spec § 5 step 3 / M3).

    Capacity-utilization gate:
        u = n_NC / (N'_l × C_max).
    When ``u < threshold`` the layer has so much slack capacity that the
    heavyweight cost matrix is unlikely to change the assignment meaningfully
    — return ``"pre"`` regardless of the configured value.  Otherwise return
    the configured value (``"pre"``, ``"post"``, or Direction C's
    ``"output"``). The output-space cost is heavyweight too, so it is gated
    identically to ``"post"`` (downgraded to ``"pre"`` on slack layers).

    With ``max_group_cap == 0`` (uncapped, ablation-only path) we treat the
    layer as fully slack (u = 0).
    """
    if max_group_cap <= 0:
        util = 0.0
    else:
        capacity = max(n_c * max_group_cap, 1)
        util = n_nc / capacity
    if util < threshold:
        return "pre"
    return configured


def _assign_children_to_centroids(
    cost: np.ndarray,
    n_children: int,
    n_centroids: int,
    max_group_cap: int = 0,
    *,
    solver: SolverName = "greedy",
    sinkhorn_epsilon_init: float = 1.0,
    sinkhorn_epsilon_final: float = 0.01,
    sinkhorn_iters: int = 200,
) -> list[int]:
    """Assign non-centroid children to centroids under a per-centroid cap.

    Solver dispatch (``solver`` argument; spec § 5 Step 3 of
    ``max_quality/docs/stage2_assignment_revision.md``):

    * ``"greedy"`` — single-pass descending-saliency greedy (legacy, paper
      §4); preserves bit-identical behavior with prior Stage 2 runs. **This is
      the default and is required for the Stage 2 v1→v2 compatibility
      invariant.**
    * ``"hungarian"`` — rectangular Hungarian (``scipy.optimize.linear_sum_assignment``)
      on the cost matrix, padded to a square problem when capacity allows
      multiple absorption per centroid. Optimal under capacity-1 problems
      (``n_children ≤ n_centroids``); falls back to MCF when capacitated.
    * ``"mcf"`` — capacitated min-cost flow via OR-Tools' ``SimpleMinCostFlow``.
      Optimal under capacity ``max_group_cap`` per centroid. Drop-in replacement
      for greedy that does not bias toward the highest-saliency centroid.
    * ``"auto"`` — picks ``hungarian`` when ``n_children ≤ n_centroids``,
      else ``mcf``.
    * ``"sinkhorn"`` — capacitated entropy-regularized OT (Tier 3 / M9).
      Solved via log-domain Sinkhorn-Knopp with linear ε-annealing and a
      slack-child dummy-row construction; see :func:`_assign_sinkhorn`.

    NOTE: The greedy branch is unchanged from the v1 Stage 2; the dispatcher
    is structured so flipping ``solver`` to a non-greedy value is the only
    semantic change. With ``solver="greedy"`` the output is bit-identical to
    the prior implementation.

    The legacy greedy path:
      When ``max_group_cap == 0`` (uncapped), each child is independently
      assigned to its nearest centroid by cost (argmin over centroid columns).

      When ``max_group_cap > 0``, iterates centroids once in order
      ``0..n_centroids-1`` (caller builds centroid_ids in descending saliency
      — column 0 = highest-saliency centroid).  For each centroid, greedily
      absorbs up to ``max_group_cap`` unassigned children (lowest cost = most
      similar first).

    The caller is responsible for ensuring feasibility before calling:
    ``n_centroids * max_group_cap >= n_children`` (spec § 5 Step 3). When the
    feasibility check passes and the cost matrix is finite, every child is
    guaranteed to receive ``assignment >= 0``. This guarantee assumes
    ``n_centroids >= 1``; when ``n_centroids == 0`` all children are assigned
    ``-1`` (no centroid).

    Returns:
        List of length ``n_children`` where entry ``ch`` is:
          ``>= 0``  → centroid column index this child is merged into
          ``-1``    → child was not absorbed (should not occur under
                      feasibility + finite costs)
    """
    if n_children == 0 or n_centroids == 0:
        return [-1] * n_children

    solver_lower = solver.lower()
    if solver_lower == "greedy":
        return _assign_greedy(cost, n_children, n_centroids, max_group_cap)
    if solver_lower == "hungarian":
        return _assign_hungarian(cost, n_children, n_centroids, max_group_cap)
    if solver_lower == "mcf":
        return _assign_mcf(cost, n_children, n_centroids, max_group_cap)
    if solver_lower == "auto":
        if n_children <= n_centroids:
            return _assign_hungarian(cost, n_children, n_centroids, max_group_cap)
        return _assign_mcf(cost, n_children, n_centroids, max_group_cap)
    if solver_lower == "sinkhorn":
        return _assign_sinkhorn(
            cost, n_children, n_centroids, max_group_cap,
            epsilon_init=sinkhorn_epsilon_init,
            epsilon_final=sinkhorn_epsilon_final,
            iters=sinkhorn_iters,
        )

    raise ValueError(
        f"_assign_children_to_centroids: unknown solver {solver!r}; expected "
        "one of 'greedy', 'hungarian', 'mcf', 'auto', 'sinkhorn'."
    )


def _assign_greedy(
    cost: np.ndarray, n_children: int, n_centroids: int, max_group_cap: int,
) -> list[int]:
    """Legacy greedy path — extracted from the v1 implementation verbatim.

    Preserves the bit-identical assignment under the v1 default (greedy +
    descending-saliency centroid order).

    Defensive: returns ``[-1] * n_children`` for empty inputs so this helper
    can be called from fallback paths in :func:`_assign_hungarian` /
    :func:`_assign_mcf` without re-doing the dispatcher's early-exit.
    """
    if n_children == 0 or n_centroids == 0:
        return [-1] * n_children
    if max_group_cap == 0:
        # Uncapped: assign each child to its nearest centroid by cost.
        # Iterating children (not centroids) avoids the centroid-order bias that
        # causes centroid 0 to absorb all children in the capped greedy path.
        assignment = [-1] * n_children
        for ch in range(n_children):
            best_c = int(np.argmin(cost[ch, :]))
            if not np.isfinite(cost[ch, best_c]):
                assignment[ch] = -1
            else:
                assignment[ch] = best_c
        n_unassigned = sum(1 for a in assignment if a < 0)
        if n_unassigned > 0 and n_centroids > 0:
            log.warning(
                "_assign_children_to_centroids: %d/%d children unassigned after uncapped pass "
                "(all-inf cost row(s) in cost matrix) — "
                "these children will be dropped from the merge group unless the caller "
                "promotes them as orphan centroids.",
                n_unassigned, n_children,
            )
        return assignment

    # Capped path (max_group_cap > 0): single-pass greedy, centroid order.
    # Note on group-cap semantics (spec §5 Step 3):
    #   max_group_cap counts non-centroids only (not the centroid itself), matching
    #   our spec §5 Step 3 ("absorb up to max_merge_group_size unassigned non-centroids").
    #   The REAM reference's group_size counts total members including the centroid,
    #   so our max_group_cap=8 is equivalent to reference group_size=9.
    # The feasibility check (b_fail) in the bump loop uses the same semantics:
    #   n_ream_nc > n_ream_c * max_group_cap  (non-centroids exceed total centroid capacity).
    assignment = [-1] * n_children
    assigned: set[int] = set()

    for c_idx in range(n_centroids):
        absorbed = 0
        # O(n_children) scan per fill slot — pathological for large expert counts;
        # consider pre-sorting by cost if this becomes a bottleneck.
        while absorbed < max_group_cap:
            best_child = -1
            best_cost = float("inf")
            for ch in range(n_children):
                if ch in assigned:
                    continue
                if cost[ch, c_idx] < best_cost:
                    best_cost = cost[ch, c_idx]
                    best_child = ch
            if best_child < 0:
                # No unassigned children with finite cost remain for this centroid.
                # Break to next centroid; any remaining unassigned children (all-inf
                # cost rows) will be reported and promoted as orphan centroids by the
                # caller. The caller must ensure costs are finite (via feasibility check)
                # to guarantee all children are assigned.
                break
            assignment[best_child] = c_idx
            assigned.add(best_child)
            absorbed += 1

    n_unassigned = sum(1 for a in assignment if a < 0)
    if n_unassigned > 0 and n_centroids > 0:
        log.warning(
            "_assign_children_to_centroids: %d/%d children unassigned after capped greedy pass "
            "(likely cause: inf cost entries in cost matrix preventing assignment) — "
            "these children will be dropped from the merge group unless the caller "
            "promotes them as orphan centroids.",
            n_unassigned, n_children,
        )

    return assignment


def _assign_hungarian(
    cost: np.ndarray, n_children: int, n_centroids: int, max_group_cap: int,
) -> list[int]:
    """Rectangular Hungarian assignment via ``scipy.linear_sum_assignment``.

    Optimal under the 1-1 capacity case (``n_children ≤ n_centroids`` with
    ``max_group_cap >= 1``). When ``n_children > n_centroids``, the problem
    becomes capacitated and Hungarian alone cannot solve it; we fall back to
    MCF. This matches the spec § 5 step 4d "auto" rule (hungarian in slack,
    mcf in tight).

    The cost matrix is shaped ``(n_children, n_centroids)``. ``+inf`` entries
    are replaced with a large finite sentinel before passing to scipy, since
    ``linear_sum_assignment`` raises on inf inputs.

    Defensive: returns ``[-1] * n_children`` for empty inputs so this helper
    can be called directly without re-doing the dispatcher's early-exit.
    """
    if n_children == 0 or n_centroids == 0:
        return [-1] * n_children
    # Capacitated → defer to MCF. ``max_group_cap == 0`` carries the v1
    # "uncapped" semantics (each child to its argmin centroid); MCF with
    # ``max_group_cap = n_children`` reproduces that, so route there too
    # rather than letting scipy's rectangular Hungarian leave excess
    # children unassigned.
    if n_children > n_centroids:
        return _assign_mcf(cost, n_children, n_centroids, max_group_cap)

    # Replace inf with a large finite sentinel above any finite cost so that
    # scipy treats the +∞ entries as effectively forbidden but does not raise.
    finite_max = float(np.nanmax(cost[np.isfinite(cost)])) if np.isfinite(cost).any() else 1.0
    big = max(finite_max, 1.0) * 1e9
    safe_cost = np.where(np.isfinite(cost), cost, big)

    row_ind, col_ind = linear_sum_assignment(safe_cost)
    assignment = [-1] * n_children
    for r, c in zip(row_ind, col_ind):
        # Skip pairs that were forbidden by the +∞ → big sentinel — leave as
        # unassigned; the caller's orphan-promotion path handles them.
        if safe_cost[r, c] >= big * 0.5:
            continue
        assignment[int(r)] = int(c)
    return assignment


def _assign_mcf(
    cost: np.ndarray, n_children: int, n_centroids: int, max_group_cap: int,
) -> list[int]:
    """Capacitated min-cost flow via OR-Tools' ``SimpleMinCostFlow``.

    Models the standard transportation polytope:
        source → each child (supply 1)
        child  → each centroid (cost ``cost[ch, c]``, capacity 1)
        centroid → sink (capacity ``max_group_cap``)
        sink supply = ``n_children`` so all children must be matched.

    Total unimodularity guarantees integer optimality under the LP relaxation
    (Ahuja–Magnanti–Orlin §9 — capacity is a transportation problem). OR-Tools
    runs cost-scaling push-relabel; ~10 ms per layer for our sizes.

    ``+∞`` entries are excluded by simply not adding the corresponding arc.

    Cost normalization: OR-Tools uses int costs. We normalize the finite cost
    range to ``[0, MCF_INT_SCALE]`` before rounding, so this routine is safe
    regardless of cost magnitude (relevant when the post-alignment whitened
    residual is unbounded). The optimal solution is invariant under positive
    affine transformations of the cost matrix.

    Defensive: returns ``[-1] * n_children`` for empty inputs.
    """
    if n_children == 0 or n_centroids == 0:
        return [-1] * n_children

    if max_group_cap < 1:
        # Reduce to assignment when no capacity bound is enforced — still
        # correct for the v1 ``max_group_cap == 0`` "uncapped" semantics by
        # treating uncapped as ``n_children`` per centroid (effectively
        # unlimited within the problem).
        max_group_cap = n_children

    try:
        from ortools.graph.python.min_cost_flow import SimpleMinCostFlow
    except ImportError as exc:
        raise RuntimeError(
            "_assign_mcf requires the 'ortools' package. Add 'ortools>=9.10' "
            "to requirements.txt and reinstall, or set "
            "stage2_reap_ream.assignment_solver back to 'greedy'."
        ) from exc

    # Normalize finite costs to [0, MCF_INT_SCALE] so int-rounding is always
    # safe (no overflow for unbounded post-alignment residuals). Min-cost
    # solutions are invariant under positive affine transformations of cost.
    finite_mask = np.isfinite(cost)
    if not finite_mask.any():
        log.warning(
            "_assign_mcf: cost matrix has no finite entries — falling back "
            "to greedy (which will leave all children unassigned)."
        )
        return _assign_greedy(cost, n_children, n_centroids, max_group_cap)

    finite_min = float(cost[finite_mask].min())
    finite_max = float(cost[finite_mask].max())
    finite_range = finite_max - finite_min
    MCF_INT_SCALE = 1_000_000

    def _to_int_cost(c: float) -> int:
        if finite_range <= 0.0:
            return 0
        normalized = (c - finite_min) / finite_range
        return int(round(normalized * MCF_INT_SCALE))

    smcf = SimpleMinCostFlow()

    # Node ids: 0 = source, 1..n_children = child nodes,
    # n_children+1..n_children+n_centroids = centroid nodes,
    # n_children+n_centroids+1 = sink.
    SRC = 0
    SINK = n_children + n_centroids + 1
    # Inline arithmetic instead of lambdas for clarity.
    # child_node(i) = 1 + i
    # cent_node(j)  = 1 + n_children + j

    # Source → child arcs
    for i in range(n_children):
        smcf.add_arc_with_capacity_and_unit_cost(SRC, 1 + i, 1, 0)

    # Child → centroid arcs (skip +∞)
    for i in range(n_children):
        for j in range(n_centroids):
            c_ij = cost[i, j]
            if not np.isfinite(c_ij):
                continue
            smcf.add_arc_with_capacity_and_unit_cost(
                1 + i, 1 + n_children + j, 1, _to_int_cost(float(c_ij)),
            )

    # Centroid → sink arcs
    for j in range(n_centroids):
        smcf.add_arc_with_capacity_and_unit_cost(
            1 + n_children + j, SINK, max_group_cap, 0,
        )

    # Supply: source = +n_children, sink = -n_children, all others = 0.
    smcf.set_node_supply(SRC, n_children)
    smcf.set_node_supply(SINK, -n_children)

    status = smcf.solve()
    if status != smcf.OPTIMAL:
        log.warning(
            "_assign_mcf: SimpleMinCostFlow returned non-optimal status %s "
            "(infeasible? check cost matrix has finite entries and capacity "
            "satisfies n_centroids * max_group_cap >= n_children). Falling "
            "back to greedy.",
            status,
        )
        return _assign_greedy(cost, n_children, n_centroids, max_group_cap)

    assignment = [-1] * n_children
    for arc in range(smcf.num_arcs()):
        if smcf.flow(arc) <= 0:
            continue
        tail = smcf.tail(arc)
        head = smcf.head(arc)
        # We only care about child→centroid arcs.
        if 1 <= tail <= n_children and (n_children + 1) <= head <= (n_children + n_centroids):
            i = tail - 1
            j = head - n_children - 1
            assignment[i] = j
    return assignment


def _assign_sinkhorn(
    cost: np.ndarray,
    n_children: int,
    n_centroids: int,
    max_group_cap: int,
    *,
    epsilon_init: float = 1.0,
    epsilon_final: float = 0.01,
    iters: int = 200,
) -> list[int]:
    """Capacitated entropy-regularized OT via Sinkhorn-Knopp with a
    dummy-slack-child construction (spec § 5 step 4d / M9 /
    D-sinkhorn-soft-assign).

    The standard Sinkhorn-Knopp algorithm requires equality marginals on
    both sides. Our problem has demand ``n_children`` (each child needs 1)
    and supply ``n_centroids · max_group_cap`` (each centroid absorbs ≤ cap),
    so we balance by inserting one **dummy slack child** with marginal
    ``n_centroids · max_group_cap − n_children`` and uniform high cost to
    every centroid. After convergence, the dummy's mass flows to whichever
    real centroids have leftover capacity, and a simple argmax over the
    real-children rows recovers the hard assignment.

    Note: spec line 152–155 frames the construction as a *virtual centroid*
    rather than a virtual child; the two constructions are dual and produce
    the same hard assignment under argmax. The slack-child form is used
    here because it is simpler to implement: real children's argmax never
    needs to filter out a dummy column.

    Costs are normalized to ``[0, 1]`` before the Sinkhorn iterations so
    that ``epsilon`` values are independent of cost magnitude (relevant
    when post-alignment whitened residuals carry an unbounded scale —
    optimal-transport solutions are invariant under positive affine cost
    transforms).

    Defensive: returns ``[-1] * n_children`` for empty inputs.
    """
    if n_children == 0 or n_centroids == 0:
        return [-1] * n_children

    if max_group_cap < 1:
        # v1 "uncapped" semantics — treat as max_group_cap = n_children so
        # the supply side has effectively unlimited capacity.
        max_group_cap = n_children

    slack = n_centroids * max_group_cap - n_children
    if slack < 0:
        log.warning(
            "_assign_sinkhorn: infeasible — n_C × C_max = %d < n_NC = %d. "
            "Falling back to greedy.",
            n_centroids * max_group_cap, n_children,
        )
        return _assign_greedy(cost, n_children, n_centroids, max_group_cap)

    finite_mask = np.isfinite(cost)
    if not finite_mask.any():
        log.warning(
            "_assign_sinkhorn: cost matrix has no finite entries — "
            "falling back to greedy."
        )
        return _assign_greedy(cost, n_children, n_centroids, max_group_cap)

    # Normalize to [0, 1] so epsilon scaling is cost-magnitude-invariant.
    finite_min = float(cost[finite_mask].min())
    finite_max = float(cost[finite_mask].max())
    finite_range = max(finite_max - finite_min, 1e-12)
    norm_cost = np.where(
        finite_mask,
        (cost - finite_min) / finite_range,
        # +∞ sentinel → very large finite value so the entry is effectively
        # forbidden but Sinkhorn-Knopp doesn't underflow exp(-inf/eps).
        100.0,
    )

    big_dummy = 100.0  # cost of dummy slack child to every centroid

    # Expanded cost: rows 0..n_children-1 are real children, last row is dummy.
    expanded = np.zeros((n_children + 1, n_centroids), dtype=np.float64)
    expanded[:n_children, :] = norm_cost
    expanded[n_children, :] = big_dummy

    a = np.concatenate([np.ones(n_children), [float(slack)]])  # row marginals
    b = np.full(n_centroids, float(max_group_cap), dtype=np.float64)  # col marginals
    # Sanity check: balanced marginals (transportation polytope).
    assert abs(a.sum() - b.sum()) < 1e-9, (
        f"_assign_sinkhorn marginals mismatch: sum(a)={a.sum()} vs "
        f"sum(b)={b.sum()}"
    )

    log_a = np.log(np.maximum(a, 1e-30))
    log_b = np.log(np.maximum(b, 1e-30))

    # Log-domain Sinkhorn-Knopp with linear epsilon annealing.
    f = np.zeros_like(log_a)
    g = np.zeros_like(log_b)
    eps = epsilon_init
    for it in range(max(iters, 1)):
        eps = epsilon_init + (epsilon_final - epsilon_init) * (it / max(iters - 1, 1))
        log_K = -expanded / max(eps, 1e-12)
        # f_i = log_a_i - logsumexp_j(log_K_ij + g_j)
        f = log_a - logsumexp(log_K + g[np.newaxis, :], axis=1)
        # g_j = log_b_j - logsumexp_i(log_K_ij + f_i)
        g = log_b - logsumexp(log_K + f[:, np.newaxis], axis=0)

    log_K = -expanded / max(eps, 1e-12)
    log_T = f[:, np.newaxis] + log_K + g[np.newaxis, :]

    # Argmax over real centroids per real child (drop the dummy row).
    real_log_T = log_T[:n_children, :]
    assignment = [int(np.argmax(row)) for row in real_log_T]
    # Direction B / skip-merge floor: a child whose entire cost row is +inf
    # (all candidate merges forbidden) must orphan — not be force-merged by the
    # argmax over the normalized sentinel. Match the greedy/hungarian/mcf
    # "-1 -> orphan promotion" contract so the floor holds for every solver.
    for ch in range(n_children):
        if not finite_mask[ch].any():
            assignment[ch] = -1
    return assignment


# ---------------------------------------------------------------------------
# Merge + router resize + covariance I/O
# ---------------------------------------------------------------------------


class _PermAlignCache:
    """Per-layer cache of Hungarian permutations and whitened residuals.

    Stage 2 v2 spec § 5 step 4T(c)(i)–(ii) (M1, "reuse merge-time Hungarian
    for the assignment cost"): the cost matrix and the merge step share the
    same per-pair Hungarian alignment. This cache lets both consumers see
    the result of one computation.

    Keys: ``(layer_idx, centroid_id, noncentroid_id)``.
    Values: ``(perm: np.ndarray, residual: float | None)``. ``residual`` is
    ``None`` when the cache entry came from the legacy v1 merge path (which
    only knows the permutation, not the whitened residual).

    Cleared at the start of every layer; bounded by ``N × K`` per layer
    (default 256 × 48 = 12,288 entries × ~512 bytes/perm ≈ 6 MB).
    """

    def __init__(self) -> None:
        self._store: dict[tuple[int, int, int], tuple[np.ndarray, float | None]] = {}

    def get(self, key: tuple[int, int, int]) -> tuple[np.ndarray, float | None] | None:
        return self._store.get(key)

    def put(self, key: tuple[int, int, int], perm: np.ndarray, residual: float | None) -> None:
        self._store[key] = (perm, residual)

    def has(self, key: tuple[int, int, int]) -> bool:
        return key in self._store

    def clear(self) -> None:
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)


def _aligned_whitened_residual(
    *,
    ref_gate: torch.Tensor,
    ref_up: torch.Tensor,
    ref_down: torch.Tensor,
    child_gate: torch.Tensor,
    child_up: torch.Tensor,
    child_down: torch.Tensor,
    perm: np.ndarray,
    a_sqrt_gate_up: torch.Tensor,
    a_sqrt_down: torch.Tensor,
    whitening_mode: str,
) -> float:
    """Three-term whitened Frobenius residual under a fixed permutation.

    Per spec § 5 step 4T(c)(ii):
        R_cm = ‖(W_c_gate − W_m_gate[perm, :]) · A_gate_up^{1/2}‖_F
             + ‖(W_c_up   − W_m_up[perm, :])   · A_gate_up^{1/2}‖_F
             + ‖(W_c_down − W_m_down[:, perm]) · A_down^{1/2}    ‖_F

    The whitening factor multiplies ΔW on the **right** (input axis), per the
    AA-SVD lineage and the Round-1 spec-review dimensional fix.

    Convention (PyTorch nn.Linear weight shapes):
        W_gate, W_up : (d_int, hidden)
        W_down       : (hidden, d_int)
        A_gate_up    : (hidden, hidden)
        A_down       : (d_int, d_int)
        perm         : length d_int — child neurons reordered to align with the centroid.
    """
    # Import here so the module load order doesn't depend on cov_sqrt being
    # available (cov_sqrt itself depends only on torch, no circular risk).
    from .utils.cov_sqrt import whitened_residual

    perm_t = torch.as_tensor(perm, dtype=torch.long, device=ref_gate.device)

    # Aligned child weights (gate / up / down). All three projections need
    # the same per-pair permutation applied on the d_int axis.
    aligned_gate = child_gate[perm_t, :]      # (d_int, hidden)
    aligned_up   = child_up[perm_t, :]        # (d_int, hidden)
    aligned_down = child_down[:, perm_t]      # (hidden, d_int)

    delta_gate = ref_gate - aligned_gate      # (d_int, hidden)
    delta_up   = ref_up   - aligned_up        # (d_int, hidden)
    delta_down = ref_down - aligned_down      # (hidden, d_int)

    r_gate = whitened_residual(delta_gate, a_sqrt_gate_up, mode=whitening_mode)
    r_up   = whitened_residual(delta_up,   a_sqrt_gate_up, mode=whitening_mode)
    r_down = whitened_residual(delta_down, a_sqrt_down,    mode=whitening_mode)

    return float(r_gate + r_up + r_down)


def _permutation_align_to_centroid(
    ref_gate: torch.Tensor,
    ref_up: torch.Tensor,
    child_gate: torch.Tensor,
    child_up: torch.Tensor,
    ref_act_mean: torch.Tensor | None = None,
    child_act_mean: torch.Tensor | None = None,
) -> np.ndarray:
    def _safe_norm(M):
        # B-C-L-2: when M is all-zero (or constant), m_max == m_min and we fall
        # through to torch.zeros_like(M). This means a zero-distance pair stays
        # zero (no cost contribution from that component) — the desired behavior
        # for Hungarian assignment where ties resolve arbitrarily.
        m_min = float(M.min())
        m_max = float(M.max())
        if m_max > m_min:
            return (M - m_min) / (m_max - m_min)
        return torch.zeros_like(M)

    # Keep cost-matrix construction on the device of the input weights — the
    # explicit .cpu() calls present here previously forced ~50-100 ms of CPU
    # cdist per pair-alignment, vs ~1 ms on GPU; with up to ~5K calls/layer ×
    # 40 layers the regression compounded to >10 min/run. The single CPU sync
    # is deferred to the Hungarian step below, which is unavoidably CPU
    # (scipy.optimize.linear_sum_assignment).
    # All inputs must share the same device (callers stage tensors via
    # build_banks(layer_ref), which keys off the live model device); cdist
    # would error on mixed-device inputs.
    C_gate = torch.cdist(ref_gate, child_gate)
    C_up   = torch.cdist(ref_up,   child_up)
    if ref_act_mean is not None and child_act_mean is not None:
        # L2-normalize both activation-mean vectors along the neuron dimension
        # before computing L2 distance (spec §5, F2-PERM-ALIGN-NORM).
        # eps=1e-8 guards against zero-norm vectors (all-zero activations);
        # F.normalize returns a zero vector for those, which is the safest
        # fallback (zero-norm input → zero output, no NaN).
        # Move act_mean tensors to the same device as the weight tensors;
        # they originate from CPU storage in ReamCostAccumulator._neuron_act_sum
        # but ref_gate/child_gate live on the model's device. Without this
        # explicit move, C_act lands on CPU while C_wt is on GPU, and the
        # subsequent `C = C_act + C_wt` raises a device-mismatch RuntimeError.
        _act_device = ref_gate.device
        ref_act_n   = torch.nn.functional.normalize(ref_act_mean.float().to(_act_device),   p=2, dim=0, eps=1e-8)
        child_act_n = torch.nn.functional.normalize(child_act_mean.float().to(_act_device), p=2, dim=0, eps=1e-8)
        C_act = torch.cdist(
            ref_act_n.unsqueeze(-1),
            child_act_n.unsqueeze(-1),
        )
        # Scale each cost component to [0, 1] before summing so that
        # L2-normalized activation distances (O(1/√d_ffn)) are not
        # negligible relative to gate/up weight distances (O(√d_hidden))
        # — spec §5, PERM-ACT-SCALE.
        # B-C-M-1: spec §5 / D5b defines C = C_act + C_wt where C_wt is the
        # gate+up Frobenius distance treated as a SINGLE component (sum first,
        # then normalize once), not two separately-normalized components.
        C_act = _safe_norm(C_act)
        C_wt = _safe_norm(C_gate + C_up)
        C = C_act + C_wt
    else:
        # B-C-M-1: same single-component treatment for the no-activation path.
        C = _safe_norm(C_gate + C_up)
    # Hungarian solver requires CPU numpy — single sync at the end.
    _, col_ind = linear_sum_assignment(C.detach().cpu().numpy())
    return col_ind


def _merge_experts_inplace(
    layer_ref: MoELayerRef,
    grouped: dict[int, list[int]],
    freq: dict[int, int],
    *,
    freq_weighted: bool,
    ream_acc: ReamCostAccumulator | None = None,
    perm_cache: "_PermAlignCache | None" = None,
) -> None:
    banks = build_banks(layer_ref)
    li = layer_ref.layer_idx
    with torch.no_grad():
        for centroid, members in grouped.items():
            if len(members) <= 1:
                continue
            if freq_weighted:
                weights = np.array([max(freq.get(m, 0), 0) for m in members], dtype=np.float64)
                # Guard: if all members have zero calibration frequency (pathological
                # edge case), fall back to equal weights rather than dividing by zero
                # (spec freq_i / Σ freq_j formula requires Σ > 0 — F2-FREQ-WEIGHT-FLOOR).
                if weights.sum() <= 0.0:
                    log.warning(
                        "layer %d centroid %d: all %d merge members have zero calibration "
                        "frequency — falling back to equal weights",
                        li, centroid, len(members),
                    )
                    weights[:] = 1.0
                weights /= weights.sum()
            else:
                # B-C-M-2: spec §5 Step 4 mandates frequency-weighted merge per
                # REAM Eq. 6. The equal-weights branch was an ablation-only
                # fallback the spec never authorized; refuse to proceed with
                # spec-non-compliant merges instead of silently warning.
                raise ValueError(
                    f"Stage 2: ream.frequency_weighted_merge=False produces "
                    f"spec-non-compliant merges (REAM Eq. 6 mandates "
                    f"frequency-weighted averaging). Set ream.frequency_weighted_merge: true "
                    f"in the config — the equal-weights branch was an ablation "
                    f"option that has no §12 D-row and must not be used in "
                    f"production. (layer={li} centroid={centroid} members={len(members)})"
                )

            # The centroid serves a dual role: it is the permutation-alignment reference
            # (via ref_gate/ref_up) AND a member of the weighted average (members[0]).
            # This is intentional — all reads from the weight bank precede the single
            # write-back (bank.set at the end), so the read-then-write-once ordering
            # guarantees correctness: the centroid's original weights are consumed before
            # being overwritten with the merged result.
            ref_gate = banks["gate_proj"].get(centroid).to(torch.float32)
            ref_up   = banks["up_proj"].get(centroid).to(torch.float32)
            ref_act  = ream_acc.get_neuron_mean(li, centroid) if ream_acc else None

            accs: dict[str, torch.Tensor | None] = {name: None for name in banks}
            for w, m in zip(weights, members):
                gate_m = banks["gate_proj"].get(m).to(torch.float32)
                up_m   = banks["up_proj"].get(m).to(torch.float32)
                child_act = ream_acc.get_neuron_mean(li, m) if ream_acc else None
                if m == centroid:
                    perm = None
                else:
                    # Stage 2 v2 (M1): reuse the perm computed during cost-matrix
                    # construction if the cache hit. This avoids a second
                    # Hungarian solve per merge member.
                    cached = (
                        perm_cache.get((li, centroid, m))
                        if perm_cache is not None
                        else None
                    )
                    if cached is not None:
                        perm = cached[0]
                    else:
                        perm = _permutation_align_to_centroid(
                            ref_gate, ref_up, gate_m, up_m,
                            ref_act_mean=ref_act, child_act_mean=child_act,
                        )
                for name, bank in banks.items():
                    if name == "gate_proj":
                        Wm = gate_m
                    elif name == "up_proj":
                        Wm = up_m
                    else:
                        Wm = bank.get(m).to(torch.float32)
                    if perm is not None:
                        Wm = Wm[perm, :] if name in ("gate_proj", "up_proj") else Wm[:, perm]
                    accs[name] = Wm * w if accs[name] is None else accs[name] + Wm * w

            for name, bank in banks.items():
                bank.set(centroid, accs[name])


def _resize_router_for_kept_experts(layer_ref: MoELayerRef, kept_ids: list[int]) -> None:
    router = layer_ref.router
    idx = torch.as_tensor(kept_ids, device=router.weight.device, dtype=torch.long)
    with torch.no_grad():
        new_w = router.weight.data.index_select(0, idx).contiguous().clone()
        router.weight = nn.Parameter(new_w, requires_grad=router.weight.requires_grad)
        if getattr(router, "bias", None) is not None:
            new_b = router.bias.data.index_select(0, idx).contiguous().clone()
            router.bias = nn.Parameter(new_b, requires_grad=router.bias.requires_grad)
    router.num_experts = len(kept_ids)
    # Guard: not all router implementations expose top_k (e.g., custom routers).
    if hasattr(router, "top_k") and router.top_k > len(kept_ids):
        router.top_k = len(kept_ids)

    mlp = layer_ref.mlp
    if hasattr(mlp, "num_experts"):
        mlp.num_experts = len(kept_ids)


def _save_covariance(cov: InputCovarianceAccumulator, path: Path) -> None:
    """Save the full covariance accumulator state to *path*.

    Caller must ensure no active profiling threads are writing to `cov` during
    this call, or hold `cov._lock` externally.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with cov._lock:
        # Clone tensors inside the lock so the snapshot is a deep copy, not a
        # shallow dict of shared tensor references that could be mutated concurrently.
        cov_snapshot = {k: v.clone() for k, v in cov.covariance.items()}
        tok_snapshot = dict(cov.token_count)
    torch.save({"format_version": 1, "covariance": cov_snapshot, "tokens": tok_snapshot}, tmp)
    _durable_rename(tmp, path)
    log.info("Saved Stage 2 input covariance to %s", path)


def _remap_covariance_for_layer(
    cov: InputCovarianceAccumulator,
    layer_idx: int,
    kept_ids: list[int],
) -> None:
    # kept_ids contains both REAM centroids and protected experts (the full post-merge
    # kept set), not just REAM centroids.
    id_to_new = {old: new for new, old in enumerate(kept_ids)}
    new_cov: dict = {}
    new_tokens: dict = {}
    n_dropped = 0
    dropped_expert_ids: set[int] = set()
    with cov._lock:
        for key, val in list(cov.covariance.items()):
            li, eidx, name = key
            if li != layer_idx:
                new_cov[key] = val
                new_tokens[key] = cov.token_count.get(key, 0)
                continue
            if eidx not in id_to_new:
                n_dropped += 1
                dropped_expert_ids.add(eidx)
                continue
            new_key = (li, id_to_new[eidx], name)
            new_cov[new_key] = val
            new_tokens[new_key] = cov.token_count.get(key, 0)
        orphan_token_keys = set(cov.token_count.keys()) - set(cov.covariance.keys())
        if orphan_token_keys:
            log.warning(
                "_remap_covariance_for_layer layer %d: %d orphaned token_count keys "
                "not in covariance will be dropped: %s",
                layer_idx, len(orphan_token_keys), orphan_token_keys,
            )
        cov.covariance, cov.token_count = new_cov, new_tokens
    if n_dropped > 0:
        n_dropped_experts = len(dropped_expert_ids)
        log.warning(
            "  layer %d: _remap_covariance_for_layer dropped %d covariance "
            "entries (= %d unique experts × ~2 matrices/expert); "
            "dropping %d experts from covariance; keeping %d experts; unexpected if "
            "n_dropped_experts > (n_keys_before - n_kept).",
            layer_idx, n_dropped, n_dropped_experts,
            n_dropped_experts, len(kept_ids),
        )
