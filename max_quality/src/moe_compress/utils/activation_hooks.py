"""Instrumented forward for fused Qwen3_5MoeExperts.

Replaces the old per-`nn.Linear` forward-hook strategy. Because the fused
``Qwen3_5MoeExperts.forward`` has no per-expert sub-modules to hook, we
monkey-patch the whole forward with an instrumented replica that emits
user-supplied callbacks at each of the three key points:

    input          : (sel_state)                     — input to gate_up_proj
    intermediate   : (act_fn(gate) * up)             — input to down_proj
    down           : (down_proj output)              — expert output

Each callback signature:

    def cb(layer_idx, expert_idx, tensor, context) -> None

where ``context`` is a dict with ``top_k_weights``, ``top_k_pos``, ``token_idx``
so the callee can compute REAP scores (g_j · ||f_j||) without re-reading the
routing metadata.

Usage:

    from moe_compress.utils.activation_hooks import instrument_experts

    callbacks = {
        "down":         down_cb,         # Stage 1 (SE detection + CKA)
        "input":        cov_cb,          # Stage 2/3 gate_up_proj input cov
        "intermediate": int_cov_cb,      # Stage 2/3 down_proj input cov
    }
    with instrument_experts(layer_ref, callbacks):
        for batch in batches:
            model(input_ids=batch)

The instrumentation is per-layer. Install on each MoE layer you want to
observe; caller handles which layers' data to collect.

This module also keeps the accumulator dataclasses
(``DownProjMaxAccumulator``, ``ExpertOutputAccumulator``, ``ReapAccumulator``,
``InputCovarianceAccumulator``, ``ReamCostAccumulator``) because Stages 1/2/3
use their API — only the hook plumbing below them changed.
"""
from __future__ import annotations

import contextlib
import logging
import os
import random
import threading
import types
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_io import MoELayerRef, FactoredExperts

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Accumulators (API preserved from pre-refactor; stages import these)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# REAM cost accumulator (activation-space, paper 2604.04356 Eq. 5 & 8)
# ---------------------------------------------------------------------------


@dataclass
class ReamCostAccumulator:
    """Collects per-token per-expert **pre-softmax routing score** profiles and
    gated expert outputs for computing REAM's activation-space cost matrix
    (paper 2604.04356, Eq. 5 & 8).

    Storage per layer:
      - gate_logit_profiles[layer_idx][expert_idx]: dict mapping global token
        index → pre-softmax routing score (linear projection + optional bias +
        e_score_correction_bias if present). Aligned by global token position
        so δ_gate cosine similarity compares the same tokens across experts
        (paper Eq. 5).  Pre-softmax scores can be negative and unbounded,
        giving the full [-1, 1] cosine similarity range (post-softmax weights
        are non-negative, compressing cosine to [0, 1]).
      - gated_output_sim/count: incremental pairwise cosine similarity of
        gated expert outputs per batch (paper Eq. 8, approximated as
        cosine(mean_gated_i, mean_gated_j) per batch).

    Pre-softmax routing scores are captured via ``capture_router_outputs`` (a
    pre-forward hook on the router module that recomputes
    ``F.linear(hidden, router.weight)`` plus any bias terms).  This runs
    independently of and concurrently with the ``instrument_experts`` hooks
    that capture gated expert outputs.
    """
    # Total number of experts in the MoE layer; 0 means "not set, skip bounds check".
    num_experts: int = 0
    # Per-(layer, expert): dict[global_token_idx → pre-softmax logit].
    # Aligned by global token position for correct δ_gate cosine sim.
    gate_logit_profiles: dict[int, dict[int, dict[int, float]]] = field(  # actually defaultdict(lambda: defaultdict(dict)) — outer and middle levels are auto-created
        default_factory=lambda: defaultdict(lambda: defaultdict(dict))
    )
    # Incremental pairwise cosine similarity of gated expert outputs.
    # gated_output_sim accumulates Σ_{t in shared} cos_sim for each pair.
    # The final δ̃_expert(i,j) divides by |X| (total calibration tokens),
    # NOT by jointly-active count, per REAM Eq. 8 and spec §5 Step 2.
    gated_output_sim: defaultdict[tuple[int, int, int], float] = field(default_factory=lambda: defaultdict(float))  # auto-vivifies with 0.0
    # Total calibration tokens seen per layer (denominator for Eq. 8).
    _total_tokens_by_layer: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    # Temporary per-batch storage: (layer, expert) → {global_token_idx → gated_output [d_hid]}
    _batch_gated_indexed: dict[tuple[int, int], dict[int, torch.Tensor]] = field(
        default_factory=lambda: defaultdict(dict)
    )
    # Per-neuron activation mean for C_act in neuron alignment (REAM §4).
    # Key: (layer_idx, expert_idx) → running sum of intermediate activations [d_intermediate]
    _neuron_act_sum: dict[tuple[int, int], torch.Tensor] = field(default_factory=dict)
    _neuron_act_count: dict[tuple[int, int], int] = field(default_factory=dict)
    # Lock protecting _batch_gated_indexed from concurrent record_gated_output /
    # finalize_batch access (M-3).
    _lock: "threading.Lock" = field(default_factory=threading.Lock)

    def record_router_logits(
        self, layer_idx: int, logits: torch.Tensor, batch_offset: int,
    ) -> None:
        """Record pre-softmax routing scores for ALL experts from one batch.

        Called once per batch per layer (from the ``capture_router_outputs``
        hook), NOT once per expert.  The logits tensor has shape
        ``[n_tokens_in_batch, num_experts]`` — we scatter each expert's
        column into its per-token profile dict.

        "Routing score" here means: linear projection + optional bias +
        e_score_correction_bias if present (pre-softmax routing score).

        Args:
            logits: [T, E] pre-softmax routing scores.
            batch_offset: global offset = batch_idx * batch_size * seq_len.
        """
        logits_cpu = logits.detach().cpu()   # [T, E]
        _, n_experts = logits_cpu.shape
        all_logits = logits_cpu.tolist()  # one .tolist() call
        # Build staging dicts outside the lock
        staging: dict[int, dict[int, float]] = {}
        for e in range(n_experts):
            prof = staging.setdefault(e, {})
            for t_idx, expert_logits in enumerate(all_logits):
                global_t = batch_offset + t_idx
                prof[global_t] = expert_logits[e]
        # Merge under lock — O(T×E) dict.update calls, but lock held only for fast Python merges
        with self._lock:
            for e, new_entries in staging.items():
                existing = self.gate_logit_profiles[layer_idx][e]
                # Check for duplicates
                dups = set(new_entries) & set(existing)
                if dups:
                    log.warning(
                        "record_router_logits: %d duplicate global token indices for expert %d layer %d — overwriting",
                        len(dups), e, layer_idx,
                    )
                existing.update(new_entries)

    def record_gated_output(self, layer_idx: int, expert_idx: int,
                            gate_weights: torch.Tensor, expert_output: torch.Tensor,
                            token_indices: torch.Tensor, batch_offset: int) -> None:
        """Record gated expert output σ(x)_e * E_e(x) keyed by global token index.

        IMPORTANT: ``gate_weights`` MUST be the un-renormalized full softmax
        σ(x)_e at the active token positions for expert ``expert_idx``, NOT the
        top-k renormalized weights (which sum to 1 over the top-k experts).
        Spec §5 line 339 + D-ream-sparse-routing require the full softmax over
        ALL experts so that δ̃_expert(i,j) reflects σ(x)_i · σ(x)_j cosine
        similarity, not (top_k_w_i / Σ_top_k) · (top_k_w_j / Σ_top_k).

        Stage 2 callers compute this via ``F.softmax(router_logits, dim=-1)``
        over the full router-logits tensor and index the resulting [T, E]
        matrix at ``[token_idx, expert_idx]`` to obtain the per-token σ(x)_e
        values for the dispatched expert.
        """
        if self.num_experts > 0 and (expert_idx < 0 or expert_idx >= self.num_experts):
            log.warning(
                "record_gated_output: expert_idx %d out of range [0, %d)",
                expert_idx, self.num_experts,
            )
            return
        # gate_weights: [T], expert_output: [T, d_hid], token_indices: [T]
        gated = (gate_weights.unsqueeze(-1) * expert_output).detach().cpu().to(torch.float32)  # [T, d_hid]
        indices = (token_indices.detach().cpu() + batch_offset).tolist()
        with self._lock:
            prof = self._batch_gated_indexed.setdefault((layer_idx, expert_idx), {})
            for idx, t in enumerate(indices):
                if t in prof:
                    log.warning(
                        "record_gated_output: duplicate token index %d for expert %d layer %d — overwriting",
                        t, expert_idx, layer_idx,
                    )
                prof[t] = gated[idx].clone()

    def record_batch_token_count(self, layer_idx: int, n_tokens: int) -> None:
        """Record the exact number of tokens in a batch for the Eq. 8 denominator.

        Must be called from the _profile_layer loop immediately after
        ream_acc.finalize_batch(...). This gives an exact |X| denominator
        independent of routing activity (fixes the edge case where an entire
        batch has no active expert, which would cause finalize_batch to miss
        those tokens in the union-of-sets count).
        """
        with self._lock:
            self._total_tokens_by_layer[layer_idx] += n_tokens

    def finalize_batch(self, layer_idx: int, num_experts: int) -> None:
        """After a full forward pass through the layer, compute pairwise cosine
        similarities over jointly-active tokens (paper Eq. 8 exact formulation).

        For each pair (i, j), finds the token intersection where BOTH experts
        were active in this batch, computes per-token cosine similarity on the
        gated outputs, and accumulates sums of per-token cosine similarities
        (the actual averaging by |X| happens in compute_delta_expert). This is
        O(E² × |intersection|) per batch, where |intersection| ≈
        (top_k/E)² × T is typically very small.

        Note: token counting for the Eq. 8 denominator is handled by
        record_batch_token_count(), called from the _profile_layer loop with
        the exact batch token count (batch.shape[0] * batch.shape[1]).
        """
        # Collect per-expert {global_token_idx → gated_output} for this batch.
        # Acquire lock to safely read and clear _batch_gated_indexed.
        with self._lock:
            per_expert: dict[int, dict[int, torch.Tensor]] = {}
            for e in range(num_experts):
                key = (layer_idx, e)
                if key in self._batch_gated_indexed and self._batch_gated_indexed[key]:
                    # Pop directly — no copy needed; we own this dict exclusively after the pop
                    per_expert[e] = self._batch_gated_indexed.pop(key)
            # Clear any remaining batch storage for this layer (empty dicts or experts not in range).
            keys_to_clear = [k for k in self._batch_gated_indexed if k[0] == layer_idx]
            for k in keys_to_clear:
                self._batch_gated_indexed.pop(k, None)

        expert_ids = sorted(per_expert.keys())

        # Build per-expert token sets for fast intersection.
        token_sets: dict[int, set[int]] = {e: set(per_expert[e].keys()) for e in expert_ids}

        if len(expert_ids) < 2:
            return

        sim_updates: dict[tuple, float] = {}
        for idx_i, e_i in enumerate(expert_ids):
            for e_j in expert_ids[idx_i + 1:]:
                # Intersection: tokens where BOTH experts were active.
                shared = token_sets[e_i] & token_sets[e_j]
                if not shared:
                    continue
                # Per-token cosine similarity on the intersection.
                # Use sorted() for deterministic per-token ordering (set iteration is non-deterministic).
                vecs_i = torch.stack([per_expert[e_i][t] for t in sorted(shared)])  # [|shared|, d_hid]
                vecs_j = torch.stack([per_expert[e_j][t] for t in sorted(shared)])  # [|shared|, d_hid]
                sims = F.cosine_similarity(vecs_i, vecs_j, dim=-1)           # [|shared|]
                # Guard against NaN from zero-vector gated outputs (e.g. when a
                # gated expert output is the zero vector, cosine_similarity returns
                # NaN). Treat those tokens as contributing zero to the similarity
                # sum (maps to neutral 0.5 after rescaling in compute_delta_expert,
                # not minimum similarity).
                sims = torch.where(torch.isnan(sims), torch.zeros_like(sims), sims)
                # Accumulate sum of per-token cosine similarities (numerator of Eq. 8).
                # Dividing by |X| happens in compute_delta_expert, NOT here.
                sim_sum = float(sims.sum().item())
                k1 = (layer_idx, e_i, e_j)
                k2 = (layer_idx, e_j, e_i)
                sim_updates[k1] = sim_updates.get(k1, 0.0) + sim_sum
                sim_updates[k2] = sim_updates.get(k2, 0.0) + sim_sum
        with self._lock:
            for k, v in sim_updates.items():
                self.gated_output_sim[k] = self.gated_output_sim.get(k, 0.0) + v

    def compute_gate_similarity_matrix(
        self, layer_idx: int, expert_ids: list[int],
    ) -> torch.Tensor:
        """δ_gate similarity matrix per REAM Eq. 5, using the observed-max dist2sim.

        Builds the (len(expert_ids), |X|) gate logit profile matrix, L2-row-
        normalizes each expert's row, computes full pairwise Euclidean distances,
        and converts via `dist2sim = 1 - d / d.max()` — dividing by the
        **observed** maximum across the full N×N pairwise distance matrix, matching
        the reference implementation (ream/ream.py lines 37-41).

        **Spec invariant (§5, δ_gate Eq. 5):** ``expert_ids`` MUST contain ALL
        non-protected (non-SE) experts in the layer — not just a centroid/candidate
        subset.  The dist2sim normalization divides by ``D.max()`` computed from
        this matrix; passing a subset causes ``D.max()`` to be underestimated,
        inflating similarity values for the omitted pairs.  Callers that need the
        cost for a (centroid, non-centroid) subset should still pass all N non-SE
        expert IDs here, then index into the returned matrix.

        Args:
            layer_idx: MoE layer index.
            expert_ids: COMPLETE list of all non-protected expert indices for this
                layer (spec §5: D.max() must be over the full non-protected
                population N, not just a nc+c subset).

        Returns:
            Float32 tensor of shape (len(expert_ids), len(expert_ids)) where
            entry [i, j] = δ_gate similarity ∈ [0, 1] for expert_ids[i] vs
            expert_ids[j]. Returns an all-zeros tensor if NO expert has
            accumulated profile data (i.e. no batches were processed for any
            of the requested experts).
        """
        n = len(expert_ids)
        if n == 0:
            return torch.zeros(0, 0, dtype=torch.float32)

        # Snapshot the profiles under the lock, then release before heavy computation.
        # Deep-copy the innermost token→logit dicts so a concurrent record_router_logits
        # calling existing.update() on those inner dicts cannot cause
        # "dictionary changed size during iteration" or non-deterministic results (M-2).
        with self._lock:
            layer_profiles = {
                k: {t: val for t, val in v.items()}
                for k, v in self.gate_logit_profiles.get(layer_idx, {}).items()
            }

        # Collect all token indices across all requested experts.
        all_tokens = sorted(
            set().union(*[layer_profiles[e].keys() for e in expert_ids if e in layer_profiles])
        )
        if not all_tokens:
            # No data at all — min similarity (0.0) for every pair.
            return torch.zeros(n, n, dtype=torch.float32)

        T = len(all_tokens)
        token_to_col = {t: c for c, t in enumerate(all_tokens)}

        # Build (n, T) matrix; missing token entries default to 0.0.
        mat = torch.zeros(n, T, dtype=torch.float32)
        for row, e in enumerate(expert_ids):
            prof = layer_profiles.get(e, {})
            if not prof:
                # Expert has no profile — row stays zero; zero-norm rows are
                # normalized to zeros after NaN replacement, yielding min similarity.
                continue
            for t, val in prof.items():
                c = token_to_col.get(t)
                # c is guaranteed non-None: all_tokens is built from the union of all profile keys
                if c is not None:
                    mat[row, c] = val

        # L2-row-normalize each expert's profile vector.
        mat = F.normalize(mat, p=2, dim=1)  # (n, T)
        # Guard against NaN from zero-norm rows (experts with an all-zero profile
        # vector). F.normalize produces NaN for zero vectors; replace with zeros
        # so they contribute minimum (0.0) similarity to every pair.
        mat = torch.where(torch.isnan(mat), torch.zeros_like(mat), mat)

        # Early exit for all-zero (or near-zero) profile matrix — cdist would
        # yield all-zero distances, mapping to all-ones similarity, which is
        # meaningless.  Using an absolute threshold (< 1e-9) rather than exact
        # equality avoids the case where near-zero but non-exactly-zero entries
        # pass the check and then cause d.max().clamp(min=1e-12) to produce
        # large negative sim values outside [0, 1].
        if mat.abs().max() < 1e-9:
            return torch.zeros(n, n, dtype=torch.float32)

        # Full pairwise Euclidean distances → (n, n).
        d = torch.cdist(mat, mat, p=2)  # (n, n)

        # dist2sim: 1 - d / d.max() (observed-max normalization, matching reference).
        # Numerical robustness clamp for near-zero profiles (the all-zeros case is handled above).
        sim = 1.0 - d / d.max().clamp(min=1e-12)
        sim.fill_diagonal_(1.0)
        # Safety clamp: floating-point rounding in cdist can push values
        # infinitesimally outside [0, 1]; clamp to guarantee the contract.
        sim.clamp_(0.0, 1.0)

        return sim.to(torch.float32)

    def compute_delta_expert(self, layer_idx: int, expert_i: int, expert_j: int) -> float:
        """δ̃_expert(i,j) per REAM Eq. 8: mean cosine similarity of gated outputs.

        Denominator is |X| (total calibration tokens), NOT jointly-active count.
        Returns similarity ∈ [0, 1] via (avg_cosine + 1) / 2.
        Returns NaN when no data is available (callers must check ``math.isnan``).
        Reference: ream/ream.py lines 99-113.

        Sparse-routing approximation: In sparse top-k routing, experts are only
        dispatched on their top-k tokens. For jointly-active tokens, the top-k
        routing weight equals the full-softmax weight σ(x)_e. Non-jointly-active
        tokens contribute zero to the numerator (expert output not computed); they
        still appear in the denominator |X| (via record_batch_token_count). This
        is a faithful implementation of Eq. 8 under sparse routing, not a deviation.
        """
        key = (layer_idx, expert_i, expert_j)
        with self._lock:
            total = self._total_tokens_by_layer.get(layer_idx, 0)
            sim_val = self.gated_output_sim.get(key, 0.0)
        if total == 0:
            # no profiling data — return NaN sentinel; callers must check math.isnan
            return float("nan")
        return float(min(1.0, max(0.0, (sim_val / total + 1.0) / 2.0)))  # rescale cosine ∈ [-1, 1] → similarity ∈ [0, 1], clamped

    def record_neuron_activations(
        self, layer_idx: int, expert_idx: int, intermediate: torch.Tensor,
    ) -> None:
        """Accumulate per-neuron activation sums for C_act (REAM §4).

        ``intermediate`` is the input to down_proj: shape [T, d_intermediate].
        Each column is a neuron. We accumulate sum(|activation|, dim=0) over
        tokens and batches, then divide by count to get per-neuron mean
        activation magnitude.
        """
        key = (layer_idx, expert_idx)
        # Use abs so the mean captures activation magnitude, not signed average.
        # Compute outside the lock — no shared state touched here.
        batch_sum = intermediate.detach().abs().sum(dim=0).cpu().to(torch.float32)  # [d_intermediate]
        n_tokens = int(intermediate.shape[0])
        with self._lock:
            prev = self._neuron_act_sum.get(key)
            if prev is None:
                self._neuron_act_sum[key] = batch_sum
            else:
                self._neuron_act_sum[key] = prev + batch_sum
            self._neuron_act_count[key] = self._neuron_act_count.get(key, 0) + n_tokens

    def get_neuron_mean(self, layer_idx: int, expert_idx: int) -> torch.Tensor | None:
        """Return per-neuron mean activation magnitude [d_intermediate], or None."""
        key = (layer_idx, expert_idx)
        with self._lock:
            s = self._neuron_act_sum.get(key)
            c = self._neuron_act_count.get(key, 0)
            if s is None or c == 0:
                return None
            return s.clone() / c

    def clear_layer(self, layer_idx: int) -> None:
        """Free memory for a processed layer."""
        with self._lock:
            self.gate_logit_profiles.pop(layer_idx, None)
            keys_to_clear = [k for k in self.gated_output_sim if k[0] == layer_idx]
            for k in keys_to_clear:
                self.gated_output_sim.pop(k, None)
            self._total_tokens_by_layer.pop(layer_idx, None)
            batch_keys = [k for k in self._batch_gated_indexed if k[0] == layer_idx]
            for k in batch_keys:
                self._batch_gated_indexed.pop(k, None)
            # _neuron_act_sum and _neuron_act_count are always updated together, so their
            # key sets are identical — iterate one and clear both.
            neuron_keys = [k for k in self._neuron_act_sum if k[0] == layer_idx]
            for k in neuron_keys:
                self._neuron_act_sum.pop(k, None)
                self._neuron_act_count.pop(k, None)


@dataclass
class DownProjMaxAccumulator:
    """Per-(layer, expert) max(|x|) accumulator, GPU-resident during forward.

    Keeps a 0-dim CUDA tensor per expert and runs ``torch.maximum`` on it
    without syncing. :meth:`finalize` transfers to CPU once at end of
    profiling (not per-expert, per-sample).
    """
    per_expert_max: dict[tuple[int, int], float] = field(default_factory=dict)
    _gpu: dict[tuple[int, int], torch.Tensor] = field(default_factory=dict)
    _lock: "threading.Lock" = field(default_factory=threading.Lock)

    def update(self, layer_idx: int, expert_idx: int, x: torch.Tensor) -> None:
        cur = x.detach().abs().amax()  # 0-dim, stays on device; amax() always returns a fresh tensor
        key = (layer_idx, expert_idx)
        with self._lock:
            prev = self._gpu.get(key)
            if prev is None:
                self._gpu[key] = cur.clone()  # clone: amax() result may alias allocator memory reused by the next forward
            else:
                # torch.maximum on a 0-dim scalar is cheap; do it under the lock to avoid lost updates
                self._gpu[key] = torch.maximum(prev, cur)

    def finalize(self) -> None:
        # Threading invariant: finalize() MUST be called only after all update()
        # calls for the profiling pass are complete.  If update() fires between
        # the two lock sections below (after the first lock clears _gpu but before
        # the second lock writes per_expert_max), the new GPU tensor is left in
        # _gpu and will only be processed on a subsequent finalize() call.  Since
        # finalize() is typically called exactly once (at the end of calibration),
        # any update() that races in will be silently lost.  Callers must ensure
        # all forward passes are done before calling finalize().
        with self._lock:
            gpu_copy = dict(self._gpu)
            self._gpu.clear()
        # Phase 1: GPU→CPU outside lock — may stall CUDA stream, don't block other threads.
        items = [(key, tensor.cpu()) for key, tensor in gpu_copy.items()]
        # Phase 2: merge back under lock — prevents a concurrent finalize() call from
        # racing on per_expert_max between the CPU transfer and the dict write.
        with self._lock:
            for key, cpu_val in items:
                val = float(cpu_val.item())
                if val > self.per_expert_max.get(key, 0.0):
                    self.per_expert_max[key] = val

    def clear_layer(self, layer_idx: int) -> None:
        """Free memory for a processed layer (consistent with ReamCostAccumulator.clear_layer)."""
        with self._lock:
            gpu_keys = [k for k in self._gpu if k[0] == layer_idx]
            for k in gpu_keys:
                self._gpu.pop(k, None)
            cpu_keys = [k for k in self.per_expert_max if k[0] == layer_idx]
            for k in cpu_keys:
                self.per_expert_max.pop(k, None)


@dataclass
class ExpertOutputAccumulator:
    """Per-(layer, expert) expert output representation collector for CKA.

    Collects down_proj output vectors during the calibration forward pass
    using reservoir sampling to bound memory. Used by Stage 1 Phase D to
    compute CKA pairwise similarity matrices.

    GPU-resident reservoir (revised). Each (layer, expert) reservoir is a
    pre-shaped ``[cap, d_out]`` tensor on the device of the expert output.
    Updates are vectorized: instead of a per-token Python loop with a
    CPU transfer + ``.clone()`` per accepted token, we compute acceptance
    probabilities for the entire batch in one GPU op and write all accepted
    rows with a single indexed assignment.

    Why GPU. The CPU implementation transferred ``[T, d_out]`` per call;
    Phase B fires this callback ~10K times per forward (256 experts × 40
    layers). The CPU loop + .clone() per token dominated Phase B (~25 sec
    per forward pass on H200, vs ~0.3 sec for Phase A which has no
    ExpertOutputAccumulator). With this rewrite, Phase B per-forward time
    drops to be model-forward-bound (~0.3-1 sec).

    Memory budget: max_tokens_per_expert=256 × d_out=2048 × 4 bytes = 2 MB
    per active expert. With 256 experts × 40 layers ≈ 20 GB peak — fits
    alongside the ~70 GB Qwen3.6-35B-A3B model in H200's 140 GB. Lazy
    per-(layer, expert) allocation amortizes the cost.

    Statistical equivalence with sequential reservoir sampling. Vectorized
    sampling processes a batch of ``n_batch`` tokens in a single op:
    each token i (1-indexed within the batch) is accepted with probability
    ``cap / (seen + i)`` and, if accepted, overwrites a uniform random slot.
    For a single-batch update this matches sequential reservoir sampling
    in expectation; per-slot occupancy distribution is identical. When two
    accepted tokens collide on the same slot, last-wins applies (PyTorch
    indexed-assign semantics) — the resulting distribution is still uniform
    because the choice of slot is independent of the token content.

    Usage:
        acc = ExpertOutputAccumulator(max_tokens_per_expert=256)
        # ... inside the down_cb callback:
        acc.update(layer_idx, expert_idx, down_proj_output)
        # ... after the forward pass:
        acc.finalize()
        R = acc.get_representations(layer_idx, expert_idx)  # [n, d_out] CPU fp32
    """
    max_tokens_per_expert: int = 256
    # GPU-resident reservoir: (layer, expert) → [cap, d_out] fp32 on device.
    # Lazy-allocated on the first update() for that key.
    _gpu_reservoir: dict[tuple[int, int], torch.Tensor] = field(default_factory=dict)
    # Total tokens seen per (layer, expert) — controls Python-side branches.
    _seen_count: dict[tuple[int, int], int] = field(default_factory=dict)
    # Finalized CPU representations: (layer, expert) → [n_tokens, d_out] fp32.
    _finalized: dict[tuple[int, int], torch.Tensor] = field(default_factory=dict)
    _lock: "threading.Lock" = field(default_factory=threading.Lock)

    def update(self, layer_idx: int, expert_idx: int, x: torch.Tensor) -> None:
        """Collect expert output vectors via vectorized GPU reservoir sampling.

        ``x`` shape: [T, d_out] where T is the number of tokens routed to
        this expert in the current batch. Called from the ``down`` callback.
        Tensor is kept on its native device; no CPU transfer in the hot path.
        """
        if x.dim() < 2 or x.shape[0] == 0:
            return
        key = (layer_idx, expert_idx)
        # Detach + cast on the source device. No .cpu() in the hot path.
        x = x.detach().to(torch.float32)
        n_batch, d_out = x.shape
        cap = self.max_tokens_per_expert
        device = x.device

        with self._lock:
            reservoir = self._gpu_reservoir.get(key)
            if reservoir is None:
                # Lazy allocate on the device of the first observation.
                reservoir = torch.empty(cap, d_out, dtype=torch.float32, device=device)
                self._gpu_reservoir[key] = reservoir

            seen = self._seen_count.get(key, 0)
            n_filled = min(seen, cap)  # currently-occupied slots before this batch

            # Phase 1: fill empty slots with the head of x. Always-accept regime
            # of reservoir sampling — equivalent to seen + i ≤ cap.
            n_to_fill = min(cap - n_filled, n_batch)
            if n_to_fill > 0:
                # Indexed assign copies values; no aliasing with x's storage.
                reservoir[n_filled:n_filled + n_to_fill] = x[:n_to_fill]

            # Phase 2: reservoir sampling for tokens beyond capacity.
            n_remaining = n_batch - n_to_fill
            if n_remaining > 0:
                # Token at remaining-index i is at global position
                # (seen + n_to_fill + i + 1). Acceptance probability cap/position.
                positions = (
                    seen + n_to_fill
                    + torch.arange(1, n_remaining + 1, device=device, dtype=torch.float32)
                )
                probs = float(cap) / positions  # [n_remaining]
                u = torch.rand(n_remaining, device=device)
                accept_mask = u < probs                                # [n_remaining]
                slots = torch.randint(0, cap, (n_remaining,), device=device)
                # Materialize accepted indices once on GPU; one indexed write.
                accepted = accept_mask.nonzero(as_tuple=True)[0]
                if accepted.numel() > 0:
                    # Source rows: accepted positions within the n_remaining tail.
                    src_rows = x[n_to_fill + accepted]
                    dst_slots = slots[accepted]
                    # Last-wins on collisions (PyTorch indexed-assign default);
                    # statistically equivalent for uniform reservoir sampling.
                    reservoir[dst_slots] = src_rows

            self._seen_count[key] = seen + n_batch

    def finalize(self) -> None:
        """Move GPU reservoirs to CPU (for downstream CKA consumption that
        operates on host-resident tensors) and free the GPU storage.

        Caller contract: no concurrent ``update()`` calls during ``finalize()``.
        Phase B closes finalize once after all calibration forwards complete,
        which satisfies this. Violating the contract would not corrupt the
        popped reservoir (the local ``gpu_t`` reference keeps it alive) but
        would race against `_seen_count` clearing, producing inconsistent
        Phase 1/Phase 2 splits in any post-finalize ``update()``.
        """
        with self._lock:
            keys = list(self._gpu_reservoir.keys())
        for key in keys:
            with self._lock:
                seen = self._seen_count.get(key, 0)
                gpu_t = self._gpu_reservoir.pop(key, None)
            if gpu_t is None:
                continue
            n_active = min(seen, self.max_tokens_per_expert)
            if n_active > 0:
                # Slice + cpu transfer happens once per (layer, expert).
                cpu_t = gpu_t[:n_active].detach().to("cpu", copy=True)
            else:
                cpu_t = torch.empty(0, 0, dtype=torch.float32)
            with self._lock:
                self._finalized[key] = cpu_t
        with self._lock:
            self._seen_count.clear()

    def get_representations(self, layer_idx: int, expert_idx: int) -> torch.Tensor | None:
        """Return [n_tokens, d_out] CPU fp32, or None if no data collected.

        Reads the finalized CPU snapshot when ``finalize()`` has been called;
        otherwise materializes from the GPU reservoir on the fly (used by tests
        that introspect mid-stream)."""
        key = (layer_idx, expert_idx)
        with self._lock:
            t = self._finalized.get(key)
            if t is not None:
                return t.clone() if t.numel() > 0 else None
            gpu_t = self._gpu_reservoir.get(key)
            seen = self._seen_count.get(key, 0)
            if gpu_t is None or seen == 0:
                return None
            n_active = min(seen, self.max_tokens_per_expert)
            if n_active == 0:
                return None
            # Snapshot to CPU; the GPU reservoir keeps streaming.
            return gpu_t[:n_active].detach().to("cpu", copy=True)


@dataclass
class ReapAccumulator:
    """REAP score accumulator, GPU-resident during layer profiling.

    Instead of ``sums[k] += float(tensor.cpu().item())`` (which stalls the GPU
    on every expert × sample event), we keep a per-expert 0-dim tensor on the
    same device as the forward and only transfer to CPU via
    :meth:`finalize_layer`.
    """
    # Plain dicts (not defaultdicts) so that external callers reading acc.freq[nonexistent_key]
    # raise KeyError rather than auto-vivifying a spurious zero entry. Use .get() for all
    # internal reads; explicit assignment for writes.
    sums: dict[tuple[int, int], float] = field(default_factory=dict)
    counts: dict[tuple[int, int], int] = field(default_factory=dict)
    # freq: per-(layer, expert) total token count seen across all batches.
    # Read externally by stage2_reap_ream.py to compute expert routing frequency.
    freq: dict[tuple[int, int], int] = field(default_factory=dict)
    _gpu_sums: dict[tuple[int, int], torch.Tensor] = field(default_factory=dict)
    # Lock protecting _gpu_sums, sums, counts, and freq from concurrent
    # add_gpu / finalize_layer / score access.
    _lock: "threading.Lock" = field(default_factory=threading.Lock)

    def add_gpu(self, key: tuple[int, int], contrib: torch.Tensor, n_tokens: int) -> None:
        with self._lock:
            cur = self._gpu_sums.get(key)
            if cur is None:
                self._gpu_sums[key] = contrib.detach().clone()
            else:
                cur.add_(contrib.detach())
            self.counts[key] = self.counts.get(key, 0) + n_tokens
            self.freq[key] = self.freq.get(key, 0) + n_tokens

    def finalize_layer(self, layer_idx: int) -> None:
        # Threading invariant: finalize_layer() MUST be called only after all
        # forward passes for this layer are complete — no add_gpu() calls may
        # be in-flight concurrently.  If add_gpu() fires between the two lock
        # sections below, it can increment counts[k] while sums[k] is still 0;
        # a concurrent score() call in that window would return 0.0 instead of
        # the correct value.  The normal pipeline (all forwards → finalize_layer
        # → score()) is safe; concurrent finalize+score is unsupported.
        with self._lock:
            keys = [k for k in self._gpu_sums if k[0] == layer_idx]
            gpu_items = [(k, self._gpu_sums.pop(k)) for k in keys]
        # GPU→CPU transfer outside the lock — 0-dim tensors are tiny but we
        # keep the pattern consistent. The .cpu().item() is done here so the
        # subsequent lock section can immediately commit the final value.
        cpu_items = [(k, float(gpu.cpu().item())) for k, gpu in gpu_items]
        # Merge all finalized sums in a single lock acquisition so score()
        # never observes a state where _gpu_sums has been popped but sums has
        # not yet been updated (within the expected single-threaded usage — see
        # invariant comment above).
        with self._lock:
            for k, cpu_val in cpu_items:
                self.sums[k] = self.sums.get(k, 0.0) + cpu_val

    def finalize_all(self) -> None:
        # Snapshot of layer_ids is taken at call time; layers added concurrently
        # after the snapshot are silently skipped. This is safe when called after
        # all forward passes complete (the intended usage). (L-2)
        with self._lock:
            layer_ids = {k[0] for k in self._gpu_sums}
        for li in layer_ids:
            self.finalize_layer(li)

    def score(self, layer_idx: int, expert_idx: int) -> float:
        k = (layer_idx, expert_idx)
        with self._lock:
            n = self.counts.get(k, 0)
            s = self.sums.get(k, 0.0)
        if n == 0:
            return 0.0
        return s / n


@dataclass
class InputCovarianceAccumulator:
    """Per-(layer, expert, matrix_name) streaming covariance accumulator.

    Two-tier storage:
      * ``_pending``: CPU-resident covariance accumulations pending the
        atomic lock-commit into ``covariance``. Covariance computation was
        moved to CPU before lock acquisition (to avoid cross-CUDA-stream
        races), so these tensors are CPU-resident despite the name suggesting
        GPU. Updated by :meth:`update` outside the lock; drained by
        :meth:`finalize_layer` inside the lock.
      * ``covariance``: CPU-resident final results, in ``storage_dtype``
        (default float32; callers may set bf16 via set_storage_dtype() for
        disk economy). Populated by :meth:`finalize_layer` once per-layer
        profiling completes.

    Stage 2 drives profiling one layer at a time, so only one layer's worth
    of per-expert pending covariances is live simultaneously (≤ ~256 experts
    × 2048×2048 fp32 ≈ 4.3 GB — fits well under an 80 GB A100).

    Aliasing: ``gate_proj`` and ``up_proj`` share input inside
    ``Qwen3_5MoeExperts`` so writes with ``matrix_name="up_proj"`` are
    ignored (``gate_proj`` already covers it, and :meth:`get` returns the
    gate_proj entry for up_proj lookups).
    """

    covariance: dict[tuple[int, int, str], torch.Tensor] = field(default_factory=dict)
    token_count: defaultdict[tuple[int, int, str], int] = field(default_factory=lambda: defaultdict(int))  # auto-vivifies with 0
    storage_dtype: torch.dtype = torch.float32
    _alias_gate_up: bool = True
    # CPU-resident covariance accumulations pending lock-commit (see class docstring).
    _pending: dict[tuple[int, int, str], torch.Tensor] = field(default_factory=dict)
    _gpu_token_count: dict[tuple[int, int, str], int] = field(default_factory=lambda: defaultdict(int))
    _lock: "threading.Lock" = field(default_factory=threading.Lock)

    def set_storage_dtype(self, dtype: torch.dtype) -> None:
        self.storage_dtype = dtype

    def update(
        self, layer_idx: int, expert_idx: int, matrix_name: str, x: torch.Tensor
    ) -> None:
        """Accumulate xᵀx on the *same device as x*. No CPU sync here.

        Lock discipline: we hold ``_lock`` only around the _pending dict reads and
        writes, NOT during the matmul itself — that would stall the forward
        thread while the background finalize thread waits for the lock.
        """
        if self._alias_gate_up and matrix_name == "up_proj":
            return
        flat = x.detach().reshape(-1, x.shape[-1])
        if flat.numel() == 0:
            return
        # `.clone()` ensures this thread owns the storage: `.to(float32)` returns
        # a view when the input is already float32, and `.clone()` breaks the alias
        # so in-place ops on the caller's source tensor cannot affect our computation.
        flat_f32 = flat.to(torch.float32).clone()
        # Expensive matmul outside the lock — no shared state touched here.
        cov = flat_f32.transpose(0, 1) @ flat_f32
        # Move to CPU before lock acquisition to avoid cross-CUDA-stream race on in-place add.
        cov = cov.cpu()
        n_tok = flat.shape[0]
        key = (layer_idx, expert_idx, matrix_name)
        with self._lock:
            cur = self._pending.get(key)
            if cur is None:
                self._pending[key] = cov
            else:
                cur.add_(cov)
            self._gpu_token_count[key] = self._gpu_token_count.get(key, 0) + n_tok

    def finalize_layer(self, layer_idx: int) -> None:
        # Phase 1: pop pending tensors under lock to prevent concurrent update()
        # calls from racing on _pending while we drain it.
        with self._lock:
            keys = [k for k in self._pending if k[0] == layer_idx]
            gpu_items = [(k, self._pending.pop(k), self._gpu_token_count.pop(k, 0))
                         for k in keys]
            storage_dtype = self.storage_dtype  # capture under lock

        # Phase 2: cast to storage dtype (tensors are already CPU-resident — see update()).
        cpu_items = [(k, gpu_cov.to(storage_dtype), n_tok)
                     for k, gpu_cov, n_tok in gpu_items]

        # Phase 3: merge transferred tensors into CPU-resident covariance dict
        # under lock so get() readers see a consistent state.
        # Use the `storage_dtype` captured in Phase 1 — do not re-read self.storage_dtype here.
        with self._lock:
            for k, cpu_cov, n_tok in cpu_items:
                prev = self.covariance.get(k)
                if prev is None:
                    self.covariance[k] = cpu_cov
                else:
                    self.covariance[k] = (
                        prev.to(torch.float32) + cpu_cov.to(torch.float32)
                    ).to(storage_dtype)
                self.token_count[k] = self.token_count.get(k, 0) + n_tok

    def finalize_all(self) -> None:
        # Snapshot of layer_ids is taken at call time; layers added concurrently
        # after the snapshot are silently skipped. This is safe when called after
        # all forward passes complete (the intended usage). (L-2)
        with self._lock:
            layer_ids = {k[0] for k in self._pending}
        for li in layer_ids:
            self.finalize_layer(li)

    def spill_layer_to_disk(self, layer_idx: int, dir_path) -> None:
        """Spill covariance data for ``layer_idx`` to disk and evict from memory.

        INVARIANT: ``finalize_layer(layer_idx)`` must not be called concurrently
        with ``spill_layer_to_disk(layer_idx)``.  Concurrent calls for *different*
        layer indices are safe.
        """
        # Phase 1: snapshot values under lock — do NOT pop yet so data is not
        # lost if torch.save fails before the write completes.
        with self._lock:
            keys = [k for k in self.covariance if k[0] == layer_idx]
            if not keys:
                return
            # Capture original tensor references for Phase 3 identity check, then
            # clone for thread-safe serialization in Phase 2.  Two separate dicts
            # are required: ``originals`` maps key → the exact Python object that
            # was in self.covariance at snapshot time (for the ``is`` check);
            # ``snapshot`` maps key → a clone safe to hand to torch.save without
            # holding the lock.  Using only the clones for the identity check would
            # always fail because a clone is never ``is`` the original.
            originals = {k: self.covariance[k] for k in keys}
            snapshot = {k: t.clone() for k, t in originals.items()}
            payload = {
                "format_version": 1,
                "covariance": snapshot,
                "tokens": {k: self.token_count.get(k, 0) for k in keys},
            }
        # Phase 2: write outside the lock so the forward thread is not stalled
        # during the (potentially slow) torch.save.
        out = Path(dir_path) / f"layer_{layer_idx}.pt"
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(out.suffix + ".tmp")
        try:
            torch.save(payload, tmp)
            os.replace(tmp, out)
        except BaseException:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            raise
        # Phase 3: evict from memory only after the file is safely on disk.
        # Guard against the TOCTOU race where finalize_layer added new data to
        # an existing key between Phase 1 and Phase 3: only pop if the stored
        # value is the same object we snapshotted (identity check via ``is``
        # against originals, NOT snapshot — snapshot contains clones which are
        # never ``is`` the current stored tensor).
        # If a new tensor was written under the same key, leave it in memory.
        with self._lock:
            for k in keys:
                current = self.covariance.get(k)
                if current is originals[k]:
                    self.covariance.pop(k, None)
                    self.token_count.pop(k, None)
                else:
                    log.warning(
                        "spill_layer_to_disk: key %r was updated between Phase 1 "
                        "and Phase 3 — keeping new value in memory (not evicting).",
                        k,
                    )

    def load_layer_from_disk(self, layer_idx: int, dir_path) -> bool:
        p = Path(dir_path) / f"layer_{layer_idx}.pt"
        if not p.exists():
            return False
        try:
            payload = torch.load(p, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to torch.load spill file {p}: {exc!r}. "
                "The file is likely corrupt — delete it (or the whole "
                "spill dir) and re-run the affected layer."
            ) from exc
        if not isinstance(payload, dict) or "covariance" not in payload:
            raise RuntimeError(
                f"Spill file {p} has unexpected layout: "
                f"got {type(payload).__name__} "
                f"(keys={list(payload.keys()) if isinstance(payload, dict) else 'n/a'}). "
                "Likely a legacy or corrupt format — delete and re-run."
            )
        fmt = int(payload.get("format_version", 0))
        if fmt != 1:
            raise RuntimeError(
                f"Spill file {p} has format_version={fmt} (expected 1). "
                "Regenerate by deleting the per-layer spill dir and re-running."
            )
        # Validate all keys before acquiring the lock
        for k in payload["covariance"]:
            if k[0] != layer_idx:
                raise RuntimeError(
                    f"load_layer_from_disk: key {k!r} does not belong to layer {layer_idx}"
                )
        # Now safe to mutate under the lock
        with self._lock:
            storage_dtype = self.storage_dtype  # capture under lock, consistent with finalize_layer discipline
            for k, disk_cov in payload["covariance"].items():
                prev = self.covariance.get(k)
                if prev is None:
                    self.covariance[k] = disk_cov.to(storage_dtype)
                else:
                    self.covariance[k] = (
                        prev.to(torch.float32) + disk_cov.to(torch.float32)
                    ).to(storage_dtype)
            for k, n in payload.get("tokens", {}).items():
                self.token_count[k] = self.token_count.get(k, 0) + n
        return True

    def unload_layer(self, layer_idx: int) -> None:
        with self._lock:
            for k in [k for k in self.covariance if k[0] == layer_idx]:
                self.covariance.pop(k, None)
                self.token_count.pop(k, None)

    def get(self, key: tuple[int, int, str]) -> torch.Tensor | None:
        """Return the covariance tensor for ``key``, or None if not present.

        Returns a cloned tensor; callers may modify the result without affecting accumulator state.
        """
        with self._lock:
            if key in self.covariance:
                return self.covariance[key].clone()
            if self._alias_gate_up and key[2] == "up_proj":
                alt = (key[0], key[1], "gate_proj")
                t = self.covariance.get(alt)
                if t is not None:
                    return t.clone()
        return None


# ---------------------------------------------------------------------------
# Common REAP recorder (called from an 'intermediate'- or 'down'-point callback)
# ---------------------------------------------------------------------------


def record_reap(
    acc: ReapAccumulator,
    layer_idx: int,
    expert_idx: int,
    gate_vals: torch.Tensor,
    expert_outs: torch.Tensor,
) -> None:
    """``gate_vals`` [T], ``expert_outs`` [T, hidden]. Accumulates on GPU —
    the CPU transfer happens in :meth:`ReapAccumulator.finalize_layer`."""
    if gate_vals.numel() == 0:
        return
    if expert_outs.ndim < 2:
        raise ValueError(
            f"record_reap: expert_outs must be 2-D [T, hidden], got shape {expert_outs.shape}"
        )
    leading = expert_outs.shape[0]
    if gate_vals.numel() != leading:
        raise RuntimeError(
            f"REAP: gate_vals.numel()={gate_vals.numel()} != expert_outs.shape[0]={leading} "
            f"(layer={layer_idx}, expert={expert_idx}). "
            "Instrumented forward is out of sync with the reference dispatch."
        )
    norms = expert_outs.to(torch.float32).norm(dim=-1)
    contrib = (gate_vals.to(torch.float32) * norms).sum()
    acc.add_gpu((layer_idx, expert_idx), contrib, int(gate_vals.numel()))


# ---------------------------------------------------------------------------
# Instrumented forward for one MoE layer
# ---------------------------------------------------------------------------


CallbackFn = Callable[[int, int, torch.Tensor, dict], None]


@contextlib.contextmanager
def instrument_experts(
    layer_ref: MoELayerRef,
    callbacks: dict[str, CallbackFn],
):
    """Install an instrumented forward on ``layer_ref.mlp.experts`` that
    emits callbacks at the three observation points, then restore on exit.

    Works for both ``Qwen3_5MoeExperts`` (fused) and ``FactoredExperts``
    (our rank-k replacement). The two paths share the same dispatch
    structure — only the per-expert matmul sequence differs.

    Accepted callback keys:
      - ``input``         : called with sel_state per (layer, expert, batch)
      - ``intermediate``  : called with act_fn(gate) * up (down_proj input)
      - ``down``          : called with down output
      - ``gate_up_out``   : called with the raw pre-chunk gate_up projection
      - ``gate_up_in``    : alias for ``input`` — BOTH keys fire independently
                            if both are registered (no deduplication)

    Context dict passed to each callback:
      {"top_k_weights": [T], "top_k_pos": [T], "token_idx": [T]}
    """
    experts = layer_ref.experts_module
    is_factored = isinstance(experts, FactoredExperts)
    original_forward = experts.forward
    layer_idx = layer_ref.layer_idx

    # Reentrancy guard: detect if a previous instrument_experts call already replaced
    # this module's forward with one of our instrumented wrappers. We mark instrumented
    # forwards by attaching ``_instrument_experts_patched = True`` to the underlying
    # function so we can detect double-entry without referencing the inner closures.
    _underlying = getattr(original_forward, "__func__", original_forward)
    if getattr(_underlying, "_instrument_experts_patched", False):
        raise RuntimeError(
            f"instrument_experts: layer {layer_idx} is already instrumented — "
            f"double-patching would corrupt the forward chain"
        )

    def _cb(name, e_int: int, tensor, ctx):
        # e_int must already be a Python int — conversion is hoisted to the loop
        # body (once per expert) rather than repeated per callback call (L-4).
        fn = callbacks.get(name)
        if fn is None:
            return
        fn(layer_idx, e_int, tensor, ctx)

    if is_factored:
        def wrapped_factored(self, hidden_states, top_k_index, top_k_weights):
            if top_k_index.dim() != 2:
                raise RuntimeError(
                    f"top_k_index must be 2D [T, top_k], got shape {top_k_index.shape}"
                )
            final = torch.zeros_like(hidden_states)
            with torch.no_grad():
                mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
                hit = (mask.sum(dim=(-1, -2)) > 0).nonzero()
            for expert_idx in hit:
                # hit contains indices from nonzero() on a mask of shape
                # [num_experts, ...], so e is always in [0, num_experts).
                e = expert_idx[0]
                e_int = int(e)  # hoist int() conversion once per expert (L-4)
                top_k_pos, token_idx = torch.where(mask[e])
                sel = hidden_states[token_idx]
                ctx = {"top_k_weights": top_k_weights[token_idx, top_k_pos],
                       "top_k_pos": top_k_pos, "token_idx": token_idx}
                _cb("input", e_int, sel, ctx)
                _cb("gate_up_in", e_int, sel, ctx)
                gate = F.linear(F.linear(sel, self.gate_proj_V[e]), self.gate_proj_U[e])
                up   = F.linear(F.linear(sel, self.up_proj_V[e]),   self.up_proj_U[e])
                # Emit a synthetic gate_up_out consistent with the non-factored path.
                # FactoredExperts computes gate and up via separate low-rank projections
                # rather than a single fused gate_up_proj, so we concatenate them to
                # match the [T, 2*d_ffn] shape the non-factored path emits.
                _cb("gate_up_out", e_int, torch.cat([gate, up], dim=-1), ctx)
                intermediate = self.act_fn(gate) * up
                _cb("intermediate", e_int, intermediate, ctx)
                down = F.linear(F.linear(intermediate, self.down_proj_V[e]),
                                self.down_proj_U[e])
                _cb("down", e_int, down, ctx)
                down = down * top_k_weights[token_idx, top_k_pos, None]
                final.index_add_(0, token_idx, down.to(final.dtype))
            return final
        forward_fn = wrapped_factored
    else:
        def wrapped_fused(self, hidden_states, top_k_index, top_k_weights):
            if top_k_index.dim() != 2:
                raise RuntimeError(
                    f"top_k_index must be 2D [T, top_k], got shape {top_k_index.shape}"
                )
            final = torch.zeros_like(hidden_states)
            with torch.no_grad():
                mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
                hit = (mask.sum(dim=(-1, -2)) > 0).nonzero()
            for expert_idx in hit:
                # hit contains indices from nonzero() on a mask of shape
                # [num_experts, ...], so e is always in [0, num_experts).
                e = expert_idx[0]
                e_int = int(e)  # hoist int() conversion once per expert (L-4)
                top_k_pos, token_idx = torch.where(mask[e])
                sel = hidden_states[token_idx]
                ctx = {"top_k_weights": top_k_weights[token_idx, top_k_pos],
                       "top_k_pos": top_k_pos, "token_idx": token_idx}
                _cb("input", e_int, sel, ctx)
                _cb("gate_up_in", e_int, sel, ctx)
                gate_up = F.linear(sel, self.gate_up_proj[e])
                _cb("gate_up_out", e_int, gate_up, ctx)
                gate, up = gate_up.chunk(2, dim=-1)
                intermediate = self.act_fn(gate) * up
                _cb("intermediate", e_int, intermediate, ctx)
                down = F.linear(intermediate, self.down_proj[e])
                _cb("down", e_int, down, ctx)
                down = down * top_k_weights[token_idx, top_k_pos, None]
                final.index_add_(0, token_idx, down.to(final.dtype))
            return final
        forward_fn = wrapped_fused

    # Mark the wrapper so the reentrancy guard above can detect double-patching.
    forward_fn._instrument_experts_patched = True
    experts.forward = types.MethodType(forward_fn, experts)
    try:
        yield
    finally:
        experts.forward = original_forward


# ---------------------------------------------------------------------------
# Generic calibration runner (no hooks by itself)
# ---------------------------------------------------------------------------


def run_calibration(
    model: nn.Module,
    batches,
    *,
    device=None,
    extra_forward_kwargs: dict | None = None,
    per_batch_callback: Callable[[int], None] | None = None,
    log_every: int = 64,
) -> None:
    model.eval()
    n_total = len(batches) if hasattr(batches, "__len__") else None
    with torch.no_grad():
        for i, batch in enumerate(batches):
            if device is not None:
                batch = batch.to(device)
            model(input_ids=batch, **(extra_forward_kwargs or {}))
            if per_batch_callback is not None:
                per_batch_callback(i)
            if log_every > 0 and (i + 1) % log_every == 0:
                if n_total is not None:
                    log.info("calibration forward %d/%d", i + 1, n_total)
                else:
                    log.info("calibration forward %d", i + 1)


# ---------------------------------------------------------------------------
# Early-exit calibration (REAM sequential merging — skip layers after target)
# ---------------------------------------------------------------------------


class _EarlyExitException(BaseException):
    """Sentinel raised by a forward hook to abort the forward pass early.

    Used by :func:`run_calibration_early_exit` to avoid executing decoder
    layers after the target layer.  The forward runs under ``torch.no_grad()``
    so no autograd graph is corrupted.

    Inherits from ``BaseException`` (not ``Exception``) so that broad
    ``except Exception:`` handlers inside model code cannot accidentally
    swallow the early-exit signal — the correct idiom for control-flow
    exceptions.
    """
    ...


@contextlib.contextmanager
def early_exit_after_layer(model: nn.Module, target_layer_idx: int):
    """Context manager that installs a forward hook on the decoder layer
    *after* ``target_layer_idx`` that raises :class:`_EarlyExitException`,
    aborting the forward pass once the target layer has fully executed.

    For the last MoE layer there is no next layer to hook — we hook the
    text tower's post-layers module (norm / final_layernorm) if it exists,
    otherwise we let the full forward run (no savings, but correct).

    Note: this context manager is designed for single-call use — one model
    forward per context manager entry.  Reusing the same entry across multiple
    forward calls is supported (the hook stays installed for the lifetime of
    the ``with`` block), but the hook fires once per forward and the caller
    must catch :class:`_EarlyExitException` for each call individually.

    Usage::

        with early_exit_after_layer(model, target_layer_idx=5):
            try:
                model(input_ids=batch)
            except _EarlyExitException:
                pass  # expected — layer 5 completed, layers 6+ skipped
    """
    tower = _find_text_tower(model)
    layers = tower.layers
    hook_target = None
    if target_layer_idx + 1 < len(layers):
        hook_target = layers[target_layer_idx + 1]
    else:
        # Last layer — try hooking the post-layers norm to avoid the lm_head.
        for attr in ("norm", "final_layernorm", "ln_f"):
            candidate = getattr(tower, attr, None)
            if isinstance(candidate, nn.Module):
                hook_target = candidate
                break

    if hook_target is None:
        # No layer after target — full forward, no savings.
        yield
        return

    def _exit_hook(_module, _input):
        raise _EarlyExitException()

    handle = hook_target.register_forward_pre_hook(_exit_hook)
    try:
        yield
    finally:
        handle.remove()


# FIXME: duplicated from model_io._find_text_tower — consolidate when both modules are refactored.
def _find_text_tower(model: nn.Module) -> nn.Module:
    """Locate the decoder tower that owns ``.layers``."""
    candidates: list[nn.Module] = [model]
    for attr in ("model", "language_model", "text_model"):
        sub = getattr(model, attr, None)
        if sub is not None:
            candidates.append(sub)
    if hasattr(model, "model"):
        for attr in ("language_model", "text_model", "decoder"):
            sub = getattr(model.model, attr, None)
            if sub is not None:
                candidates.append(sub)
    seen: set[int] = set()
    for c in candidates:
        if id(c) in seen:
            continue
        seen.add(id(c))
        layer_list = getattr(c, "layers", None)
        if isinstance(layer_list, (nn.ModuleList, list)) and len(layer_list) > 0:
            return c
    raise RuntimeError("Could not locate decoder tower with .layers attribute")


def run_calibration_early_exit(
    model: nn.Module,
    batches,
    target_layer_idx: int,
    *,
    device=None,
    extra_forward_kwargs: dict | None = None,
    per_batch_callback: Callable[[int], None] | None = None,
    log_every: int = 64,
) -> None:
    """Like :func:`run_calibration` but aborts the forward pass after
    ``target_layer_idx`` completes.  Layers after the target are never
    executed, giving a ~2× wall-clock speedup for REAM's sequential
    per-layer profiling (paper 2604.04356 §4, Fig 1(b)).

    All metrics collected for the target layer (REAP scores, REAM cost,
    input covariance) depend only on hidden states arriving *at* that
    layer, not on downstream layers.  The early exit is therefore
    mathematically identical to a full forward.
    """
    model.eval()
    n_total = len(batches) if hasattr(batches, "__len__") else None
    with torch.no_grad(), early_exit_after_layer(model, target_layer_idx):
        for i, batch in enumerate(batches):
            if device is not None:
                batch = batch.to(device)
            try:
                model(input_ids=batch, **(extra_forward_kwargs or {}))
            except _EarlyExitException:
                pass  # expected — target layer completed, downstream skipped
            if per_batch_callback is not None:
                per_batch_callback(i)
            if log_every > 0 and (i + 1) % log_every == 0:
                if n_total is not None:
                    log.info("calibration forward (early-exit@L%d) %d/%d",
                             target_layer_idx, i + 1, n_total)
                else:
                    log.info("calibration forward (early-exit@L%d) %d",
                             target_layer_idx, i + 1)


# ---------------------------------------------------------------------------
# Router-output hook (Stage 5 uses this instead of the fused-experts shim)
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def capture_router_outputs(layer_refs: list[MoELayerRef]):
    """Collect **pre-softmax routing scores** for each given layer.

    "Routing score" means: linear projection + optional bias +
    e_score_correction_bias if present (pre-softmax routing score).

    Why a pre-forward hook: ``Qwen3_5MoeTopKRouter.forward`` overwrites its
    first return value with a softmax'd tensor before returning, so a
    ``register_forward_hook`` on the router gives post-softmax probabilities
    rather than the raw scores. For Stage 5 KD we need the raw scores, so we
    recompute ``F.linear(hidden, router.weight)`` plus bias terms ourselves in
    a pre-forward hook. Cheap (one matmul that also runs inside the router)
    and always correct.
    """
    storage: dict[int, list[torch.Tensor]] = {ref.layer_idx: [] for ref in layer_refs}
    handles: list = []

    def _pre_factory(li, router):
        def _h(_m, inputs):
            x = inputs[0]
            if hasattr(router, "hidden_dim"):
                x = x.reshape(-1, router.hidden_dim)
            # Recompute the pre-softmax routing scores from the raw hidden state.
            # Assumption: the Qwen3.5-MoE router applies no pre-linear transforms
            # (e.g. no layer norm) to the hidden state before the weight projection.
            # Verified against the Qwen3.5-MoE router implementation: the forward
            # path is directly F.linear(hidden, weight) + optional bias terms, with
            # no layer norm or other nonlinearity before the linear projection.
            # If a future router variant adds pre-linear transforms, this hook must
            # be updated to replicate them before the F.linear call.
            logits = F.linear(x, router.weight)
            if getattr(router, "bias", None) is not None:
                logits = logits + router.bias
            # Qwen3.5-MoE auxiliary-loss-free load-balancing bias; must be
            # included to match routing decisions.
            # B-iter5-L-3 (code): the bias is included as part of the router's
            # pre-softmax output (which drives top-k selection); REAM δ_gate
            # operates on these same bias-adjusted pre-softmax logits — this is
            # the natural reading (spec is silent on the bias). If a future
            # spec change requires the unbiased logits for δ_gate, capture must
            # be split: top-k selection uses biased logits; δ_gate would consume
            # a separate unbiased capture.
            if hasattr(router, "e_score_correction_bias") and router.e_score_correction_bias is not None:
                logits = logits + router.e_score_correction_bias
            storage[li].append(logits.detach().clone())  # clone: caching allocator may reuse the backing memory on next forward
        return _h

    for ref in layer_refs:
        h = ref.router.register_forward_pre_hook(_pre_factory(ref.layer_idx, ref.router))
        handles.append(h)
    try:
        yield storage
    finally:
        for h in handles:
            h.remove()
