"""Post-alignment REAM cost plugin (Task 9 of the plugin-architecture refactor).

Home of ``_post_alignment_cost`` — the ``cost_alignment="post"`` branch of the
REAM cost matrix (Stage 2 v2 spec § 5 step 4T). Moved verbatim out of
``stage2_reap_ream.py``; that module re-imports it so external callers and
tests keep their existing import paths.

Circular-import note: this module imports only ``pipeline.permutation_align``,
``pipeline.base``, ``pipeline.context`` and ``moe_compress.utils.*`` — none of
which import ``stage2_reap_ream`` or ``ream_cost``. There is therefore no cycle
at module load. ``ream_cost._ream_cost_matrix`` still imports
``_post_alignment_cost`` at *function scope* inside its ``post`` branch: that
keeps the call symmetric with the still-monolith ``output`` branch and costs
nothing once the module is cached.

``ReamCostPostPlugin`` is the future plugin home for the ``post`` cost path.
For T9 it is an inert shell: its ``compute_cost`` hook is a documented no-op
because the legacy bump loop still calls ``_ream_cost_matrix`` directly.
Wiring ``compute_cost`` into the phase walk is deferred until the assignment
phase is decomposed (T13+).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ...utils.activation_hooks import (
    InputCovarianceAccumulator,  # noqa: F401 — resolves the string type hint
    ReamCostAccumulator,
)
from ...utils.model_io import MoELayerRef, build_banks
from ...pipeline.context import PipelineContext
from ..permutation_align import (
    _PermAlignCache,  # noqa: F401 — resolves the string type hint
    _aligned_whitened_residual,
    _permutation_align_to_centroid,
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
    from ...utils.cov_sqrt import compute_a_sqrt, CovSqrtCache

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


class ReamCostPostPlugin:
    """Plugin home for the REAM post-alignment whitened-residual cost path.

    T9 status: inert shell. The legacy bump loop still calls
    ``_ream_cost_matrix`` directly, so this plugin's ``compute_cost`` hook is a
    deliberate no-op. The plugin exists now so the ``post`` path has a stable
    home; wiring ``compute_cost`` into the phase walk is deferred until the
    assignment phase is decomposed (T13+).
    """

    name = "ream_cost_post"
    paper = "REAM post-alignment whitened-residual cost matrix builder."
    config_key = "stage2_reap_ream.cost_alignment"
    # () until a later task wires the live hook
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()
    provides: tuple[str, ...] = ()

    def is_enabled(self, config: dict) -> bool:
        """True iff ``stage2_reap_ream.cost_alignment`` resolves to ``"post"``.

        Unlike ``"pre"``, ``"post"`` is not the default, so a missing key or a
        missing ``stage2_reap_ream`` block leaves this plugin disabled.
        Case-insensitive to match the ``str(...).lower()`` normalization done
        in ``stage2_reap_ream.run()`` (config validation).
        """
        s2 = config.get("stage2_reap_ream", {}) if isinstance(config, dict) else {}
        return str(s2.get("cost_alignment", "pre")).lower() == "post"

    def contribute_artifact(self, ctx) -> dict:
        return {}

    def compute_cost(self, ctx: PipelineContext) -> Any | None:
        """No-op for T9. See class docstring.

        Returning ``None`` makes ``PluginRegistry.dispatch_first`` skip this
        plugin so the legacy bump loop remains the sole cost-matrix producer.
        """
        return None
