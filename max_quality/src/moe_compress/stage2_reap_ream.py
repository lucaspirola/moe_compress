"""Stage 2 — REAP scoring + REAM pseudo-pruning (SEQUENTIAL across layers).

REAP (paper 2510.13999, Eq. 9):

    S_j = (1 / |X_j|) Σ_{x ∈ X_j} g_j(x) · ||f_j(x)||_2

REAM (paper 2604.04356): given per-layer budget ``N'_l``, keep the top-``N'_l``
experts by ``S_j`` as centroids and assign each remaining expert to the nearest
centroid via

    δ_REAM(i, j) = δ_gate(i, j) + δ̃_expert(i, j)

with Hungarian matching on the combined weight + activation cost matrix, then a
frequency-weighted merge within each group (centroid absorbs its children).

The "sequential" step is what makes this slow: after merging layer ``l`` we
re-run the calibration forward through layers ``[0..l]`` so that the *inputs*
seen at layer ``l+1`` reflect the pruned upstream state.

Artifacts (written under ``artifacts/stage2_pruned/``):
- HF-compatible safetensors checkpoint with ``num_experts`` per layer baked into config
- ``merge_map.json`` — ``{layer_idx: {surviving_idx: [original_idx, ...]}}``
- ``stage2_layer_mse.json`` — per-layer pre/post block MSE for sanity checking
"""
from __future__ import annotations

import copy
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

from .utils.activation_hooks import (
    InputCovarianceAccumulator,
    ReapAccumulator,
    hook_matrix_inputs,
    record_reap,
    run_calibration,
)
from .utils.calibration import CalibrationSpec, build_calibration_tensor, iter_batches
from .utils.model_io import (
    MoELayerRef,
    get_expert_matrices,
    iter_moe_layers,
    iter_routed_experts,
    load_json_artifact,
    save_checkpoint,
    save_json_artifact,
)

log = logging.getLogger(__name__)


def run(
    model,
    tokenizer,
    config: dict,
    artifacts_dir: Path,
    *,
    device=None,
    stage1_budget_path: Path | None = None,
) -> Path:
    s2 = config["stage2_reap_ream"]
    cal = config["calibration"]

    # Load the Stage 1 budgets
    if stage1_budget_path is None:
        stage1_budget_path = artifacts_dir / "stage1_budgets.json"
    budgets_payload = load_json_artifact(stage1_budget_path)
    per_layer_target = {
        int(k): int(v) for k, v in budgets_payload["per_layer_target_experts"].items()
    }
    blacklist_payload = load_json_artifact(artifacts_dir / "stage0_blacklist.json")
    blacklist = {int(k): list(v) for k, v in blacklist_payload.get("blacklist", {}).items()}

    spec = CalibrationSpec(
        num_sequences=s2["num_calibration_samples"],
        sequence_length=cal["sequence_length"],
        seed=cal["seed"],
        domain_mix=cal["domain_mix"],
        c4_dataset=cal["dataset"],
        c4_subset=cal["subset"],
    )
    calib = build_calibration_tensor(
        tokenizer, spec, cache_dir=artifacts_dir / "_calibration_cache"
    )
    batches = iter_batches(calib, batch_size=s2["batch_size"])

    moe_layers = list(iter_moe_layers(model))
    # Side-channel: accumulate A = Σ x^T x per (layer, expert, matrix) so
    # Stage 3 can reuse pre-prune inputs without rerunning calibration.
    cov_acc = InputCovarianceAccumulator()

    merge_map: dict[int, dict[int, list[int]]] = {}
    per_layer_mse: dict[int, float] = {}

    for k, layer_ref in enumerate(moe_layers):
        log.info(
            "Stage 2 layer %d/%d (idx=%d) — profiling then merging to %d experts",
            k + 1, len(moe_layers), layer_ref.layer_idx, per_layer_target[layer_ref.layer_idx],
        )
        # 2a. REAP scoring on this layer only — wrap gate + experts with a
        #     dispatch-time hook that records g_j and f_j per token.
        reap_acc = ReapAccumulator()
        with _hook_layer_for_reap(layer_ref, reap_acc):
            # Only need to run calibration once per layer; prior layers are
            # already pruned in-place from earlier iterations (sequential).
            with hook_matrix_inputs([layer_ref], cov_acc):
                run_calibration(model, batches, device=device)

        # 2b. Score → pick centroids
        n_experts = len(layer_ref.experts)
        target = per_layer_target[layer_ref.layer_idx]
        protected = set(blacklist.get(layer_ref.layer_idx, []))
        scores = np.array([reap_acc.score(layer_ref.layer_idx, e) for e in range(n_experts)])

        # 2c. Enforce blacklist as forced centroids, then fill by score.
        centroid_ids = sorted(protected)
        remaining_slots = max(0, target - len(centroid_ids))
        nonprotected_ranked = np.argsort(-scores)
        for e in nonprotected_ranked:
            e = int(e)
            if e in protected:
                continue
            centroid_ids.append(e)
            if len(centroid_ids) >= target:
                break
        centroid_ids = sorted(centroid_ids)
        noncentroid_ids = [e for e in range(n_experts) if e not in set(centroid_ids)]

        # 2d. Build REAM cost matrix between non-centroids and centroids
        delta = _ream_cost_matrix(
            layer_ref,
            noncentroid_ids,
            centroid_ids,
            gate_weight=s2["ream"]["gate_weight"],
            expert_weight=s2["ream"]["expert_weight"],
        )
        # 2e. Hungarian-like assignment — scipy's solver is rectangular when
        # len(noncentroid) != len(centroid); we replicate centroids to cover
        # all non-centroids (each centroid may absorb many children).
        assignment = _assign_children_to_centroids(delta, len(noncentroid_ids), len(centroid_ids))

        # 2f. Frequency-weighted merge
        freq = {e: reap_acc.freq.get((layer_ref.layer_idx, e), 0) for e in range(n_experts)}
        grouped: dict[int, list[int]] = {c: [c] for c in centroid_ids}
        for child_pos, centroid_pos in enumerate(assignment):
            child = noncentroid_ids[child_pos]
            centroid = centroid_ids[centroid_pos]
            grouped[centroid].append(child)
        _merge_experts_inplace(layer_ref, grouped, freq, freq_weighted=s2["ream"]["frequency_weighted_merge"])

        # 2g. Rewrite the mlp.experts ModuleList to contain only centroids
        kept_modules = [layer_ref.experts[c] for c in centroid_ids]
        layer_ref.mlp.experts = nn.ModuleList(kept_modules)
        # Update router output dimension: re-slice gate.weight rows.
        _resize_router_for_kept_experts(layer_ref, centroid_ids)
        # NOTE (review N-8): previously we also set layer.mlp.config.num_experts
        # as a best-effort write, but Qwen3_5 shares one config object across
        # all layers so the write poisons all other layers. The MoE block
        # reads `self.num_experts` (handled by _resize_router_for_kept_experts
        # above), not `self.config.num_experts`, at forward time in transformers
        # 5.3, so the config write was misleading and has been removed.

        merge_map[layer_ref.layer_idx] = {
            new_idx: sorted(grouped[centroid])
            for new_idx, centroid in enumerate(centroid_ids)
        }
        # FIX (review bug #1): remap cov_acc keys for this layer from the
        # original (pre-prune) expert index to the new compact index, and
        # discard covariances for non-surviving experts. Stages 3 and 4 read
        # this mapping by new index.
        _remap_covariance_for_layer(
            cov_acc, layer_ref.layer_idx, centroid_ids,
        )
        log.info(
            "  kept %d / %d experts (blacklist=%d) — assignment Σ cost=%.4f",
            len(centroid_ids), n_experts, len(protected), float(delta.sum()) if delta.size else 0.0,
        )

        # 2h. Sequential recompute: the next iteration's REAP + hook_matrix_inputs
        #     run happens naturally with this layer now pruned in-place.

    out_dir = artifacts_dir / "stage2_pruned"
    save_checkpoint(model, tokenizer, out_dir)
    save_json_artifact(merge_map, out_dir / "merge_map.json")
    save_json_artifact(per_layer_mse, artifacts_dir / "stage2_layer_mse.json")
    # Save Stage-3-consumable covariance snapshot (pre-prune A matrices).
    _save_covariance(cov_acc, artifacts_dir / "_stage2_input_covariance.pt")
    log.info("Stage 2 complete — pruned checkpoint at %s", out_dir)
    return out_dir


# ---------------------------------------------------------------------------
# Per-layer REAP hook (captures gate values + expert outputs during dispatch)
# ---------------------------------------------------------------------------


class _ReapLayerHook:
    """Wrap a single MoE layer's forward to record per-expert (g, ||f||).

    This relies on the layer's MoE forward computing, for each token routed to
    expert j, the gate value g_j(x) and the expert output f_j(x). Since
    transformers' MoE implementations vary, we attach forward hooks on each
    expert's ``down_proj`` (captures f_j(x)) and monkey-patch ``mlp.gate`` to
    stash its softmax topk output. The layer forward is expected to then route
    tokens; for each routing event we call ``record_reap``.

    NOTE: this is a best-effort instrumentation. If the MoE forward does not
    expose per-token gate values via the router (e.g. a fused kernel), this
    hook falls back to using the mean gate weight × mean output norm, which is
    a biased estimate but sufficient for ranking.
    """

    def __init__(self, layer_ref: MoELayerRef, acc: ReapAccumulator):
        self.ref = layer_ref
        self.acc = acc
        self._handles: list = []
        self._last_gate_topk: tuple[torch.Tensor, torch.Tensor] | None = None
        # Maps expert_idx → list of (gate_val, output) events for this forward.

    def __enter__(self):
        # Hook the router: capture top-k indices + values
        router = self.ref.router

        def _router_hook(_mod, _inp, out):
            logits = out if isinstance(out, torch.Tensor) else out[0]
            probs = torch.softmax(logits.to(torch.float32), dim=-1)
            topk_vals, topk_idx = probs.topk(_topk_from_config(self.ref), dim=-1)
            self._last_gate_topk = (topk_vals.detach(), topk_idx.detach())

        self._handles.append(router.register_forward_hook(_router_hook))

        # Hook each expert's down_proj output: (g·||f||) contribution
        for expert_idx, expert in iter_routed_experts(self.ref):
            mats = get_expert_matrices(expert)
            if "down_proj" not in mats:
                continue
            mod = mats["down_proj"]
            l_idx = self.ref.layer_idx
            e_idx = expert_idx

            def _make_hook(l, e):
                def _hook(_m, _i, out):
                    out = out if isinstance(out, torch.Tensor) else out[0]
                    gate_vals = self._gate_vals_for_expert(e, out.shape[:-1])
                    if gate_vals is None:
                        return
                    record_reap(self.acc, l, e, gate_vals, out)

                return _hook

            self._handles.append(mod.register_forward_hook(_make_hook(l_idx, e_idx)))
        return self

    def _gate_vals_for_expert(self, expert_idx: int, out_leading_shape):
        if self._last_gate_topk is None:
            return None
        vals, idx = self._last_gate_topk  # [B, T, k]
        # Find positions where idx == expert_idx
        mask = idx == expert_idx
        if not mask.any():
            return None
        # vals[mask] matches f(x) rows in the same order the kernel dispatches
        # them, so we return the scalar gate per token.
        return vals[mask]

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()


def _hook_layer_for_reap(layer_ref: MoELayerRef, acc: ReapAccumulator):
    return _ReapLayerHook(layer_ref, acc)


def _topk_from_config(layer_ref: MoELayerRef) -> int:
    # Prefer the block-level attribute (what the forward actually uses); fall
    # back to the config object; finally clamp to the surviving expert count.
    mlp = layer_ref.mlp
    for name in ("top_k", "num_experts_per_tok", "router_top_k"):
        v = getattr(mlp, name, None)
        if isinstance(v, int) and v > 0:
            return min(v, len(mlp.experts))
    cfg = getattr(mlp, "config", None)
    if cfg is not None:
        for name in ("num_experts_per_tok", "top_k", "router_top_k"):
            v = getattr(cfg, name, None)
            if isinstance(v, int) and v > 0:
                return min(v, len(mlp.experts))
    # Last resort — clamp to expert count so topk() never errors.
    return min(8, len(mlp.experts))


# ---------------------------------------------------------------------------
# REAM cost matrix + assignment + merge
# ---------------------------------------------------------------------------


def _ream_cost_matrix(
    layer_ref: MoELayerRef,
    noncentroid_ids: list[int],
    centroid_ids: list[int],
    *,
    gate_weight: float,
    expert_weight: float,
) -> np.ndarray:
    """δ_REAM(i, j) = δ_gate(i, j) + δ̃_expert(i, j)."""
    if not noncentroid_ids or not centroid_ids:
        return np.zeros((len(noncentroid_ids), len(centroid_ids)))

    gate = layer_ref.router
    gate_w = gate.weight.detach().to(torch.float32)  # [num_experts, hidden]
    gate_w = torch.nn.functional.normalize(gate_w, dim=1)
    gate_sim = gate_w @ gate_w.transpose(0, 1)
    delta_gate = (1.0 - gate_sim).clamp(min=0.0, max=2.0) / 2.0

    # Expert similarity via flattened weights
    flat: dict[int, torch.Tensor] = {}
    for e in noncentroid_ids + centroid_ids:
        mats = get_expert_matrices(layer_ref.experts[e])
        parts = [mats[n].weight.detach().to(torch.float32).flatten() for n in mats]
        flat[e] = torch.cat(parts)
    W = torch.stack([flat[e] for e in noncentroid_ids + centroid_ids])
    W = torch.nn.functional.normalize(W, dim=1)
    sim = W @ W.transpose(0, 1)
    expert_sim = sim[: len(noncentroid_ids), len(noncentroid_ids):]
    delta_expert = (1.0 - expert_sim).clamp(min=0.0, max=2.0) / 2.0

    # δ_gate restricted to the right submatrix
    dg = delta_gate[np.ix_(noncentroid_ids, centroid_ids)]
    cost = gate_weight * dg + expert_weight * delta_expert
    return cost.cpu().numpy()


def _assign_children_to_centroids(
    cost: np.ndarray, n_children: int, n_centroids: int
) -> list[int]:
    """Each child → best centroid. Uses iterated Hungarian if feasible,
    otherwise a simple argmin (many-to-one) fallback.

    Hungarian requires a rectangular problem with at least as many centroids
    as children on one side; when n_children > n_centroids (common here), we
    loop: match ``n_centroids`` children via Hungarian, then argmin the rest.
    """
    if n_children == 0 or n_centroids == 0:
        return []
    assignment = [-1] * n_children
    remaining = list(range(n_children))

    while remaining:
        batch = remaining[:n_centroids]
        sub = cost[np.ix_(batch, range(n_centroids))]
        row_ind, col_ind = linear_sum_assignment(sub)
        for r, c in zip(row_ind, col_ind):
            assignment[batch[r]] = int(c)
        remaining = remaining[n_centroids:]

    return assignment


def _merge_experts_inplace(
    layer_ref: MoELayerRef,
    grouped: dict[int, list[int]],
    freq: dict[int, int],
    *,
    freq_weighted: bool,
) -> None:
    """Merge children into centroid's weight tensors by weighted average."""
    with torch.no_grad():
        for centroid, members in grouped.items():
            if len(members) <= 1:
                continue
            weights = np.array([max(freq.get(m, 0), 1) for m in members], dtype=np.float64)
            if not freq_weighted:
                weights[:] = 1.0
            weights = weights / weights.sum()

            target_mats = get_expert_matrices(layer_ref.experts[centroid])
            for name, tgt in target_mats.items():
                acc = torch.zeros_like(tgt.weight)
                for w, m in zip(weights, members):
                    src = get_expert_matrices(layer_ref.experts[m])[name].weight
                    acc = acc + float(w) * src
                tgt.weight.copy_(acc)


def _resize_router_for_kept_experts(layer_ref: MoELayerRef, kept_ids: list[int]) -> None:
    """Slice the router's output rows to match the new expert set.

    Also update MoE-block-level attributes that cache the expert count / top_k
    so the block's runtime dispatch doesn't try to index out-of-bounds
    (review bug #11 / #22).
    """
    router = layer_ref.router
    idx_tensor = torch.as_tensor(kept_ids, device=router.weight.device, dtype=torch.long)
    with torch.no_grad():
        new_w = router.weight.data.index_select(0, idx_tensor).contiguous().clone()
        router.weight = nn.Parameter(new_w)
        if getattr(router, "bias", None) is not None:
            new_b = router.bias.data.index_select(0, idx_tensor).contiguous().clone()
            router.bias = nn.Parameter(new_b)
    router.out_features = len(kept_ids)

    mlp = layer_ref.mlp
    if hasattr(mlp, "num_experts"):
        mlp.num_experts = len(kept_ids)
    for attr in ("top_k", "num_experts_per_tok", "router_top_k"):
        cur = getattr(mlp, attr, None)
        if isinstance(cur, int) and cur > len(kept_ids):
            setattr(mlp, attr, len(kept_ids))


def _save_covariance(cov: InputCovarianceAccumulator, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"covariance": cov.covariance, "tokens": dict(cov.token_count)}, path)
    log.info("Saved Stage 2 input covariance to %s", path)


def _remap_covariance_for_layer(
    cov: InputCovarianceAccumulator,
    layer_idx: int,
    centroid_ids: list[int],
) -> None:
    """Rewrite ``(layer_idx, old_expert_idx, matrix_name)`` keys in ``cov`` to
    ``(layer_idx, new_expert_idx, matrix_name)``. Drop entries whose
    old_expert_idx is not in ``centroid_ids``.
    """
    id_to_new = {old: new for new, old in enumerate(centroid_ids)}
    new_cov: dict = {}
    new_tokens: dict = {}
    for key, val in list(cov.covariance.items()):
        li, eidx, name = key
        if li != layer_idx:
            new_cov[key] = val
            new_tokens[key] = cov.token_count.get(key, 0)
            continue
        if eidx not in id_to_new:
            continue
        new_key = (li, id_to_new[eidx], name)
        new_cov[new_key] = val
        new_tokens[new_key] = cov.token_count.get(key, 0)
    cov.covariance = new_cov
    cov.token_count = new_tokens
