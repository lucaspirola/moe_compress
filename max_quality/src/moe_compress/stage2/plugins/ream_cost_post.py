"""Post-alignment whitened-residual REAM cost matrix builder.

Paper
-----
**No paper for this cost form.** It is project-original — Stage 2 v2's
``cost_alignment="post"`` branch, layered on top of the baseline REAM
greedy (arXiv:2604.04356 — see :mod:`stage2.plugins.ream_cost` for the
REAM Eq. 5/7/8 baseline this replaces). Activation-aware whitening
inherits the AA-SVD lineage from arXiv:2604.02119 (Stage 3) — see
:mod:`stage2.plugins.ream_cost_post` activation-aware framing in
deviation D-whitened-cost below.

Official code
-------------
None — the post-alignment cost is project-original. SamsungSAILMontreal/
ream's baseline (commit pinned in :mod:`stage2.plugins.ream_cost`)
implements only the symmetric ``δ_REAM`` cost.

Deviation: D-whitened-cost
--------------------------
The pre-alignment δ_REAM cost (REAM Eq. 7) is alignment-invariant —
output cosine and gate-logit cosine don't depend on neuron permutations
— but it lacks a weight-space residual term. This branch computes a
per-pair **Hungarian-aligned whitened residual**:

    cost(c, m) = ‖(W_c − P_cm·W_m) · A^{1/2}‖_F   summed over gate/up/down

where ``A^{1/2}`` multiplies ``ΔW`` on the **right** (input axis),
matching the AA-SVD derivation
``E_x ‖ΔW · x‖² = tr(ΔW · A · ΔW^T) = ‖ΔW · A^{1/2}‖_F²``. The
Hungarian permutation ``P_cm`` is computed once per ``(c, m)`` pair via
``_permutation_align_to_centroid`` and **cached for the merge step**
(single Hungarian, two consumers).

``cost_whitening`` selects the whitening form:

- ``"diag"`` — ``sqrt(diag(A))`` (cheap fallback).
- ``"full"`` — ``V · diag(sqrt(λ_clamped)) · V^T`` from
  ``torch.linalg.eigh``, mirroring ``stage3_svd._precompute_eigh``.
- Default ``"none"`` reproduces the v1 (pre-alignment) baseline.

The whitened residual measures merge error in the directions that
actually carry calibration signal (AA-SVD lineage, arXiv:2604.02119,
already used by Stage 3). Explicitly **not** AIM — a different
activation scheme; AIM's formulation is documented in
arXiv:2502.02421.

The K-prefilter (``cost_topk_filter``, default 48) bounds the per-pair
Hungarian compute: per non-centroid ``m``, only the top-K candidate
centroids by cheap symmetric ``δ_REAM`` get the expensive whitened
residual computed; the rest get ``+∞`` and the assignment solver
treats them as forbidden arcs.

Deviation: D-asymmetric-freq (opt-in)
-------------------------------------
``cost_asymmetric=true`` (default ``false``) multiplies the
post-alignment whitened residual by ``freq_m / (freq_c + freq_m)``.
Valid only with ``ream.frequency_weighted_merge=true`` (rejected at
run-time otherwise — fail-fast).

Rationale: the merge formula ``W_merged = Σ (freq_e / Σ freq) · P_e(W_e)``
weights each member by its freq share. A high-freq non-centroid merged
into a low-freq centroid dominates the merged weight (freq washout);
the symmetric cost matrix cannot distinguish merge direction. The
asymmetric factor is the per-pair version of the merge weight:
``freq_m / (freq_c + freq_m)`` is exactly the share of ``freq_m`` in a
2-element merge group ``{c, m}``, so the cost penalizes pairs where
``m`` would dominate ``c``. Under saliency-weighted merge the
analogous factor would be ``sal_m / (sal_c + sal_m)``.

Both-zero edge case → 0.5 neutral.

Naming-history note
-------------------
The post-alignment branch is referred to as "M2" in the Stage 2 v2
revision spec (``docs/stage2_assignment_revision.md``). The plugin
architecture has no module-naming taxonomy. New prose drops the M2
label; existing log lines and Trackio keys keep the historical
identifiers.

Circular-import note: this module imports only
``pipeline.permutation_align``, ``pipeline.base``, ``pipeline.context``
and ``moe_compress.utils.*`` — none of which import
``stage2_reap_ream`` or ``ream_cost``. There is therefore no cycle at
module load. ``ream_cost._ream_cost_matrix`` still imports
``_post_alignment_cost`` at *function scope* inside its ``post`` branch:
that keeps the call symmetric with the still-monolith ``output`` branch
and costs nothing once the module is cached.

``ReamCostPostPlugin`` is the live plugin home for the ``post`` cost
path. S2-6 wired its ``compute_cost`` hook into the ``compute_cost``
assignment slot via the shared ``ream_cost._compute_cost_for_plugin``
helper: when ``cost_alignment`` resolves to ``"post"`` this plugin is
registered ahead of the ``LegacyAdapter`` and wins
``PluginRegistry.dispatch_first`` for the slot.
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
# Shared cost-plugin metadata (reads/writes slot tuples). ream_cost imports
# ream_cost_post only at function scope, so this module-top import is cycle-free.
from .ream_cost import _COST_PLUGIN_READS, _COST_PLUGIN_WRITES


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
    lsa_max_workers: int | None = None,
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
    from ...utils.lsa_pool import parallel_map

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

    def _get_a_sqrt(eid: int, name: str, *, relink: bool = True) -> torch.Tensor:
        if whitening_mode == "none":
            return torch.tensor(1.0)
        key = (li, eid, name, whitening_mode)
        # ``relink=False`` reads the backing dict directly (no LRU ``move_to_end``
        # mutation). The threaded row loop passes it so concurrent workers issue
        # a plain, lock-free read instead of a read-modify-write on the cache's
        # OrderedDict — see the prewarm note above (D3). Byte-identical: the same
        # tensor is returned; only the LRU bookkeeping is skipped, and no eviction
        # can occur because the prewarm set (2*n_c keys) is below max_entries
        # (2*n_c+8). The serial/prewarm callers keep the relinking ``.get()``.
        cached = a_sqrt_cache._store.get(key) if not relink else a_sqrt_cache.get(key)
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

    # Empty matrix → no work; return the all-+∞ out array as-is. The
    # argpartition below would raise on shape[1]==0 otherwise.
    if n_nc == 0 or n_c == 0:
        return out

    # Pattern H hoist: compute the K-smallest centroid columns per non-centroid
    # row ONCE before the loop. Loop-invariant: ``k_cand`` depends only on
    # (topk, n_c); cheap_cost is read-only after construction. Mirrors the same
    # hoist applied to ``_output_space_cost`` in Plugin #3
    # (SC_FAST_PLAN_V3.md §4-B3). Saving ~1 min/row on the SC + post-cost paths.
    # See Plugin #14 audit follow-up item 4 / Plugin #3 audit finding L-B3-3.
    k_cand = min(topk, n_c)
    topk_per_ci = np.argpartition(cheap_cost, k_cand - 1, axis=1)[:, :k_cand]  # (n_nc, k_cand)

    # Item-4 eigh pre-warm (whitening_mode == "full" only). The per-centroid
    # a_sqrt (V·diag(√λ)·Vᵀ via torch.linalg.eigh, CPU LAPACK) is a pure
    # function of the centroid covariance ⇒ pre-computing every centroid's
    # gate_proj/down_proj a_sqrt in ONE pool populates ``a_sqrt_cache`` so the
    # threaded row loop below only ever READS it (no concurrent read-modify-
    # write on the cache; D3). Pre-warm by ``centroid_ids`` is a superset of
    # what rows request (rows call _get_a_sqrt(c_id, ...) with c_id ∈
    # centroid_ids — incl. the tentative-EM branch, which still keys on the
    # ORIGINAL c_id). Byte-identical: same eigh call, same inputs, just issued
    # from a pool. Skipped for "none"/"diag" — neither runs eigh (D3 N3).
    if whitening_mode == "full":
        _warm_keys = [(c_id, name) for c_id in centroid_ids
                      for name in ("gate_proj", "down_proj")]

        def _warm(key):
            c_id, name = key
            # torch.no_grad() is thread-local, so each pool worker must enter it
            # itself (same rationale as _solve_row below).
            with torch.no_grad():
                _get_a_sqrt(c_id, name)  # populates a_sqrt_cache as a side effect

        parallel_map(_warm, _warm_keys, max_workers=lsa_max_workers)

    # Per-non-centroid ROW worker. Returns the row index, the list of
    # (cj, residual_float) cost-cell writes, and the list of
    # (cache_key, perm, residual) cache puts. Collect-then-merge (D2): workers
    # touch only thread-local state; the main thread scatters ``out`` and
    # replays ``perm_cache.put`` in ROW-MAJOR order after the pool joins, so the
    # cache CONTENTS (disjoint keys) AND insertion ORDER are byte-identical to
    # the serial path. Each worker enters torch.no_grad() itself (it is
    # thread-local — otherwise .numpy() on requires-grad leaves raises).
    def _solve_row(ci: int):
        m_id = noncentroid_ids[ci]
        top_cj = topk_per_ci[ci].tolist()
        cells: list[tuple[int, float]] = []
        puts: list[tuple[tuple[int, int, int], np.ndarray, float | None]] = []
        with torch.no_grad():
            for cj in top_cj:
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
                    # scratch each round. For whitening_mode == "full" the a_sqrt
                    # was pre-warmed above ⇒ this is a lock-free cache read.
                    a_sqrt_gate_up = _get_a_sqrt(c_id, "gate_proj", relink=False)
                    a_sqrt_down    = _get_a_sqrt(c_id, "down_proj", relink=False)
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
                    # the time the merge step consumes it. Defer the actual
                    # perm_cache.put to the main-thread replay (collect-then-
                    # merge): record it here, apply it in row-major order below.
                    if perm_cache is not None and not tentative_active:
                        puts.append((cache_key, perm, residual))

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

                cells.append((cj, float(residual)))
        return ci, cells, puts

    # Thread the per-ci rows (item 2). Disjoint out cells + disjoint cache keys
    # (each row owns a unique m_id) ⇒ order-independent CONTENTS; the row-major
    # replay below makes insertion ORDER deterministic too.
    row_results = parallel_map(
        _solve_row, list(range(n_nc)), max_workers=lsa_max_workers,
    )

    # Main-thread merge: scatter out cells and replay perm_cache puts in
    # ROW-MAJOR order (ci ascending, cj/candidate order within the row) —
    # exactly the order the serial loop would have inserted them.
    for ci, cells, puts in row_results:
        for cj, value in cells:
            out[ci, cj] = value
        if perm_cache is not None:
            for cache_key, perm, residual in puts:
                perm_cache.put(cache_key, perm, residual)

    return out


class ReamCostPostPlugin:
    """Live plugin home for the REAM post-alignment whitened-residual cost path.

    S2-6 wired ``compute_cost`` into the ``compute_cost`` assignment slot. When
    ``cost_alignment`` resolves to ``"post"`` the orchestrator registers this
    plugin ahead of the ``LegacyAdapter`` so it wins
    ``PluginRegistry.dispatch_first`` for the slot. The slot body — the
    ``_ream_cost_matrix`` call — lives in the shared module-level helper
    ``ream_cost._compute_cost_for_plugin`` (one source of truth for all three
    cost plugins). S2-10 moved the capacity-util gate out into
    ``CapacityGatePlugin.select_alignment`` (which may still downgrade
    ``post``→``pre`` per layer on slack-capacity layers, run earlier in the bump
    iteration); this slot reads the gate's decision back off ``ctx``.
    """

    name = "ream_cost_post"
    paper = (
        "Post-alignment whitened-residual cost (project-original; no paper). "
        "AA-SVD lineage from arXiv:2604.02119 (Stage 3). Replaces baseline "
        "REAM cost arXiv:2604.04356 (see :mod:`stage2.plugins.ream_cost`). "
        "Deviations: D-whitened-cost (Hungarian-aligned ‖ΔW·A^{1/2}‖_F per "
        "pair, with cached P_cm), D-asymmetric-freq (opt-in "
        "freq_m/(freq_c+freq_m) factor). See module docstring."
    )
    config_key = "stage2_reap_ream.cost_alignment"
    reads: tuple[str, ...] = _COST_PLUGIN_READS
    writes: tuple[str, ...] = _COST_PLUGIN_WRITES  # () — S2-10 moved the gate out
    provides: tuple[str, ...] = ()

    def __init__(
        self,
        *,
        cov_acc,
        cost_alignment_cfg: str,
        cost_whitening: str,
        cost_topk_filter: int,
        cost_output_token_cap: int,
        lsa_threads: int = 8,
    ) -> None:
        # Store every knob the shared compute_cost body reads. NO logic — a
        # faithful mirror of the matching subset of LegacyAdapter.__init__.
        # ``cost_alignment_cfg`` is retained for ``is_enabled``; the capacity
        # gate's knobs moved to CapacityGatePlugin (S2-10).
        self.cov_acc = cov_acc
        self.cost_alignment_cfg = cost_alignment_cfg
        self.cost_whitening = cost_whitening
        self.cost_topk_filter = cost_topk_filter
        self.cost_output_token_cap = cost_output_token_cap
        # Stage-2 LSA threading perf knob (read by _compute_cost_for_plugin).
        self.lsa_threads = lsa_threads

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
        """Slot ``compute_cost`` — REAM cost matrix.

        Delegates to the shared ``ream_cost._compute_cost_for_plugin`` helper
        (same body as the other two cost plugins). Reads the capacity-gate
        decision off ``ctx`` (published by ``CapacityGatePlugin.select_alignment``
        earlier in the bump iteration). Returns the cost matrix ``delta``.
        """
        from .ream_cost import _compute_cost_for_plugin
        return _compute_cost_for_plugin(self, ctx)
