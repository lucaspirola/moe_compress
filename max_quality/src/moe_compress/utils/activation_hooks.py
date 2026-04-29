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
``InputCovarianceAccumulator``) because Stages 1/2/3 use their API — only the
hook plumbing below them changed.
"""
from __future__ import annotations

import contextlib
import logging
import threading
import types
from collections import defaultdict
from dataclasses import dataclass, field
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
    """Collects per-token per-expert **pre-softmax router logit** profiles and
    gated expert outputs for computing REAM's activation-space cost matrix
    (paper 2604.04356, Eq. 5 & 8).

    Storage per layer:
      - gate_logit_profiles[layer_idx][expert_idx]: dict mapping global token
        index → pre-softmax router logit. Aligned by global token position so
        δ_gate cosine similarity compares the same tokens across experts
        (paper Eq. 5).  Pre-softmax logits can be negative and unbounded,
        giving the full [-1, 1] cosine similarity range (post-softmax weights
        are non-negative, compressing cosine to [0, 1]).
      - gated_output_sim/count: incremental pairwise cosine similarity of
        gated expert outputs per batch (paper Eq. 8, approximated as
        cosine(mean_gated_i, mean_gated_j) per batch).

    Pre-softmax logits are captured via ``capture_router_outputs`` (a
    pre-forward hook on the router module that recomputes
    ``F.linear(hidden, router.weight)``).  This runs independently of and
    concurrently with the ``instrument_experts`` hooks that capture gated
    expert outputs.
    """
    # Per-(layer, expert): dict[global_token_idx → pre-softmax logit].
    # Aligned by global token position for correct δ_gate cosine sim.
    gate_logit_profiles: dict[int, dict[int, dict[int, float]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(dict))
    )
    # Incremental pairwise cosine similarity of gated expert outputs.
    gated_output_sim: dict[tuple[int, int, int], float] = field(default_factory=lambda: defaultdict(float))
    gated_output_count: dict[tuple[int, int, int], int] = field(default_factory=lambda: defaultdict(int))
    # Temporary per-batch storage: (layer, expert) → {global_token_idx → gated_output [d_hid]}
    _batch_gated_indexed: dict[tuple[int, int], dict[int, torch.Tensor]] = field(
        default_factory=lambda: defaultdict(dict)
    )    # Per-neuron activation mean for C_act in neuron alignment (REAM §4).
    # Key: (layer_idx, expert_idx) → running sum of intermediate activations [d_intermediate]
    _neuron_act_sum: dict[tuple[int, int], torch.Tensor] = field(default_factory=dict)
    _neuron_act_count: dict[tuple[int, int], int] = field(default_factory=lambda: defaultdict(int))

    def record_router_logits(
        self, layer_idx: int, logits: torch.Tensor, batch_offset: int,
    ) -> None:
        """Record pre-softmax router logits for ALL experts from one batch.

        Called once per batch per layer (from the ``capture_router_outputs``
        hook), NOT once per expert.  The logits tensor has shape
        ``[n_tokens_in_batch, num_experts]`` — we scatter each expert's
        column into its per-token profile dict.

        Args:
            logits: [T, E] pre-softmax router logits.
            batch_offset: global offset = batch_idx * batch_size * seq_len.
        """
        # Detach + CPU once; then scatter per-expert.
        logits_cpu = logits.detach().cpu()   # [T, E]
        n_tokens, n_experts = logits_cpu.shape
        for e in range(n_experts):
            col = logits_cpu[:, e].tolist()
            prof = self.gate_logit_profiles[layer_idx][e]
            for t, val in enumerate(col):
                prof[batch_offset + t] = val

    def record_gated_output(self, layer_idx: int, expert_idx: int,
                            gate_weights: torch.Tensor, expert_output: torch.Tensor,
                            token_indices: torch.Tensor, batch_offset: int) -> None:
        """Record gated expert output σ(x)_e * E_e(x) keyed by global token index."""
        # gate_weights: [T], expert_output: [T, d_hid], token_indices: [T]
        gated = (gate_weights.unsqueeze(-1) * expert_output).detach().cpu().to(torch.float32)  # [T, d_hid]
        indices = (token_indices.detach().cpu() + batch_offset).tolist()
        prof = self._batch_gated_indexed[(layer_idx, expert_idx)]
        for idx, t in enumerate(indices):
            prof[t] = gated[idx]

    def finalize_batch(self, layer_idx: int, num_experts: int) -> None:
        """After a full forward pass through the layer, compute pairwise cosine
        similarities over jointly-active tokens (paper Eq. 8 exact formulation).

        For each pair (i, j), finds the token intersection where BOTH experts
        were active in this batch, computes per-token cosine similarity on the
        gated outputs, and averages. This is O(E² × |intersection|) per batch,
        where |intersection| ≈ (top_k/E)² × T is typically very small.
        """
        # Collect per-expert {global_token_idx → gated_output} for this batch.
        per_expert: dict[int, dict[int, torch.Tensor]] = {}
        for e in range(num_experts):
            key = (layer_idx, e)
            if key in self._batch_gated_indexed and self._batch_gated_indexed[key]:
                per_expert[e] = dict(self._batch_gated_indexed[key])
        # Clear batch storage.
        keys_to_clear = [k for k in self._batch_gated_indexed if k[0] == layer_idx]
        for k in keys_to_clear:
            self._batch_gated_indexed[k].clear()

        expert_ids = sorted(per_expert.keys())
        if len(expert_ids) < 2:
            return

        # Build per-expert token sets for fast intersection.
        token_sets: dict[int, set[int]] = {e: set(per_expert[e].keys()) for e in expert_ids}

        for idx_i, e_i in enumerate(expert_ids):
            for e_j in expert_ids[idx_i + 1:]:
                # Intersection: tokens where BOTH experts were active.
                shared = token_sets[e_i] & token_sets[e_j]
                if not shared:
                    continue
                # Per-token cosine similarity on the intersection.
                vecs_i = torch.stack([per_expert[e_i][t] for t in shared])  # [|shared|, d_hid]
                vecs_j = torch.stack([per_expert[e_j][t] for t in shared])  # [|shared|, d_hid]
                sims = F.cosine_similarity(vecs_i, vecs_j, dim=-1)           # [|shared|]
                avg_sim = float(sims.mean().item())
                k1 = (layer_idx, e_i, e_j)
                k2 = (layer_idx, e_j, e_i)
                n_shared = len(shared)
                self.gated_output_sim[k1] += avg_sim * n_shared
                self.gated_output_sim[k2] += avg_sim * n_shared
                self.gated_output_count[k1] += n_shared
                self.gated_output_count[k2] += n_shared

    def compute_delta_gate(self, layer_idx: int, expert_i: int, expert_j: int) -> float:
        """δ_gate(i,j) per REAM Eq. 5: cosine sim between pre-softmax gate
        logit profile vectors.

        Both vectors are indexed by the same global token positions (union of
        ALL calibration tokens — every token has a logit for every expert).
        """
        prof_i = self.gate_logit_profiles.get(layer_idx, {}).get(expert_i, {})
        prof_j = self.gate_logit_profiles.get(layer_idx, {}).get(expert_j, {})
        if not prof_i and not prof_j:
            return 1.0  # max distance if no data
        all_tokens = sorted(set(prof_i.keys()) | set(prof_j.keys()))
        if not all_tokens:
            return 1.0
        vi = torch.tensor([prof_i.get(t, 0.0) for t in all_tokens], dtype=torch.float32)
        vj = torch.tensor([prof_j.get(t, 0.0) for t in all_tokens], dtype=torch.float32)
        # Guard against zero vectors (expert never activated).
        if vi.norm() < 1e-12 or vj.norm() < 1e-12:
            return 1.0
        sim = float(F.cosine_similarity(vi.unsqueeze(0), vj.unsqueeze(0)).item())
        return (1.0 - sim) / 2.0  # normalize to [0, 1]

    def compute_delta_expert(self, layer_idx: int, expert_i: int, expert_j: int) -> float:
        """δ̃_E(i,j) per REAM Eq. 8: mean cosine sim of gated expert outputs."""
        key = (layer_idx, expert_i, expert_j)
        count = self.gated_output_count.get(key, 0)
        if count == 0:
            return 1.0  # max distance
        avg_sim = self.gated_output_sim[key] / count
        return (1.0 - avg_sim) / 2.0  # normalize to [0, 1]

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
        batch_sum = intermediate.detach().abs().sum(dim=0).cpu().to(torch.float32)  # [d_intermediate]
        n_tokens = int(intermediate.shape[0])
        prev = self._neuron_act_sum.get(key)
        if prev is None:
            self._neuron_act_sum[key] = batch_sum
        else:
            self._neuron_act_sum[key] = prev + batch_sum
        self._neuron_act_count[key] += n_tokens

    def get_neuron_mean(self, layer_idx: int, expert_idx: int) -> torch.Tensor | None:
        """Return per-neuron mean activation magnitude [d_intermediate], or None."""
        key = (layer_idx, expert_idx)
        s = self._neuron_act_sum.get(key)
        c = self._neuron_act_count.get(key, 0)
        if s is None or c == 0:
            return None
        return s / c

    def clear_layer(self, layer_idx: int) -> None:
        """Free memory for a processed layer."""
        self.gate_logit_profiles.pop(layer_idx, None)
        keys_to_clear = [k for k in self.gated_output_sim if k[0] == layer_idx]
        for k in keys_to_clear:
            self.gated_output_sim.pop(k, None)
            self.gated_output_count.pop(k, None)
        neuron_keys = [k for k in self._neuron_act_sum if k[0] == layer_idx]
        for k in neuron_keys:
            self._neuron_act_sum.pop(k, None)
            self._neuron_act_count.pop(k, 0)



@dataclass
class DownProjMaxAccumulator:
    """Per-(layer, expert) max(|x|) accumulator, GPU-resident during forward.

    Keeps a 0-dim CUDA tensor per expert and runs ``torch.maximum`` on it
    without syncing. :meth:`finalize` transfers to CPU once at end of
    profiling (not per-expert, per-sample).
    """
    per_expert_max: dict[tuple[int, int], float] = field(default_factory=dict)
    _gpu: dict[tuple[int, int], torch.Tensor] = field(default_factory=dict)

    def update(self, layer_idx: int, expert_idx: int, x: torch.Tensor) -> None:
        cur = x.detach().abs().amax()                       # 0-dim, stays on device
        key = (layer_idx, expert_idx)
        prev = self._gpu.get(key)
        if prev is None:
            self._gpu[key] = cur
        else:
            prev.copy_(torch.maximum(prev, cur))

    def finalize(self) -> None:
        for key, tensor in self._gpu.items():
            val = float(tensor.cpu().item())
            if val > self.per_expert_max.get(key, 0.0):
                self.per_expert_max[key] = val
        self._gpu.clear()


@dataclass
class ExpertOutputAccumulator:
    """Per-(layer, expert) expert output representation collector for CKA.

    Collects down_proj output vectors during the calibration forward pass
    using reservoir sampling to bound memory. Used by Stage 1 Phase C to
    compute CKA pairwise similarity matrices.

    Memory budget: max_tokens_per_expert=256 × d_out=2048 × 4 bytes = 2 MB
    per expert. With 256 experts × 40 layers ≈ 20 GB total — fits in H200's
    71 GB headroom alongside the model.

    During the forward pass, expert outputs arrive on GPU. We detach and
    transfer to CPU immediately (like DownProjMaxAccumulator.finalize but
    streaming). The reservoir is maintained on CPU to avoid GPU memory pressure.

    Usage:
        acc = ExpertOutputAccumulator(max_tokens_per_expert=256)
        # ... inside the down_cb callback:
        acc.update(layer_idx, expert_idx, down_proj_output)
        # ... after forward pass:
        acc.finalize()
        R = acc.get_representations(layer_idx, expert_idx)  # [n, d_out]
    """
    max_tokens_per_expert: int = 256
    # CPU-resident reservoir: (layer_idx, expert_idx) → list of [d_out] tensors.
    _reservoir: dict[tuple[int, int], list[torch.Tensor]] = field(default_factory=dict)
    # Count of total tokens seen per (layer, expert) — for reservoir sampling.
    _seen_count: dict[tuple[int, int], int] = field(default_factory=lambda: defaultdict(int))
    # Finalized stacked representations: (layer, expert) → [n_tokens, d_out].
    _finalized: dict[tuple[int, int], torch.Tensor] = field(default_factory=dict)

    def update(self, layer_idx: int, expert_idx: int, x: torch.Tensor) -> None:
        """Collect expert output vectors via reservoir sampling.

        ``x`` shape: [T, d_out] where T is the number of tokens routed to
        this expert in the current batch. Called from the ``down`` callback.
        """
        if x.dim() < 2 or x.shape[0] == 0:
            return
        key = (layer_idx, expert_idx)
        batch_cpu = x.detach().cpu().to(torch.float32)  # [T, d_out]
        n_batch = batch_cpu.shape[0]

        reservoir = self._reservoir.get(key)
        if reservoir is None:
            reservoir = []
            self._reservoir[key] = reservoir

        seen = self._seen_count[key]
        cap = self.max_tokens_per_expert

        for i in range(n_batch):
            seen += 1
            if len(reservoir) < cap:
                # Reservoir not full — always accept.
                reservoir.append(batch_cpu[i])
            else:
                # Reservoir sampling: replace a random element with
                # probability cap/seen. Uses Python's random to avoid
                # importing numpy in this hot path.
                import random
                j = random.randint(0, seen - 1)
                if j < cap:
                    reservoir[j] = batch_cpu[i]

        self._seen_count[key] = seen

    def finalize(self) -> None:
        """Stack reservoir lists into contiguous tensors and free the lists."""
        for key, reservoir in self._reservoir.items():
            if reservoir:
                self._finalized[key] = torch.stack(reservoir, dim=0)  # [n, d_out]
            else:
                self._finalized[key] = torch.empty(0, 0, dtype=torch.float32)
        self._reservoir.clear()
        self._seen_count.clear()

    def get_representations(self, layer_idx: int, expert_idx: int) -> torch.Tensor | None:
        """Return [n_tokens, d_out] on CPU, or None if no data collected."""
        key = (layer_idx, expert_idx)
        t = self._finalized.get(key)
        if t is None:
            # Check un-finalized reservoir as fallback.
            reservoir = self._reservoir.get(key)
            if reservoir:
                return torch.stack(reservoir, dim=0)
            return None
        if t.numel() == 0:
            return None
        return t


@dataclass
class ReapAccumulator:
    """REAP score accumulator, GPU-resident during layer profiling.

    Instead of ``sums[k] += float(tensor.cpu().item())`` (which stalls the GPU
    on every expert × sample event), we keep a per-expert 0-dim tensor on the
    same device as the forward and only transfer to CPU via
    :meth:`finalize_layer`.
    """
    sums: dict[tuple[int, int], float] = field(default_factory=lambda: defaultdict(float))
    counts: dict[tuple[int, int], int] = field(default_factory=lambda: defaultdict(int))
    freq: dict[tuple[int, int], int] = field(default_factory=lambda: defaultdict(int))
    _gpu_sums: dict[tuple[int, int], torch.Tensor] = field(default_factory=dict)

    def add_gpu(self, key: tuple[int, int], contrib: torch.Tensor, n_tokens: int) -> None:
        cur = self._gpu_sums.get(key)
        if cur is None:
            self._gpu_sums[key] = contrib.detach().clone()
        else:
            cur.add_(contrib.detach())
        self.counts[key] = self.counts.get(key, 0) + n_tokens
        self.freq[key] = self.freq.get(key, 0) + n_tokens

    def finalize_layer(self, layer_idx: int) -> None:
        keys = [k for k in self._gpu_sums if k[0] == layer_idx]
        for k in keys:
            gpu = self._gpu_sums.pop(k)
            self.sums[k] = self.sums.get(k, 0.0) + float(gpu.cpu().item())

    def finalize_all(self) -> None:
        layer_ids = {k[0] for k in self._gpu_sums}
        for li in layer_ids:
            self.finalize_layer(li)

    def score(self, layer_idx: int, expert_idx: int) -> float:
        k = (layer_idx, expert_idx)
        n = self.counts.get(k, 0)
        if n == 0:
            return 0.0
        return self.sums.get(k, 0.0) / n


@dataclass
class InputCovarianceAccumulator:
    """Per-(layer, expert, matrix_name) streaming covariance accumulator.

    Two-tier storage:
      * ``_gpu``: transient covariances for the *currently-hot layer* kept
        resident on the forward's device. Summed in fp32 on GPU to avoid
        CPU-sync on every expert × sample event.
      * ``covariance``: CPU-resident final results, in ``storage_dtype``
        (default bf16 for disk economy). Populated by :meth:`finalize_layer`
        once per-layer profiling completes.

    Stage 2 drives profiling one layer at a time, so only one layer's worth
    of per-expert GPU covariances is live simultaneously (≤ ~256 experts
    × 2048×2048 fp32 ≈ 4.3 GB — fits well under an 80 GB A100).

    Aliasing: ``gate_proj`` and ``up_proj`` share input inside
    ``Qwen3_5MoeExperts`` so writes with ``matrix_name="up_proj"`` are
    ignored (``gate_proj`` already covers it, and :meth:`get` returns the
    gate_proj entry for up_proj lookups).
    """

    covariance: dict[tuple[int, int, str], torch.Tensor] = field(default_factory=dict)
    token_count: dict[tuple[int, int, str], int] = field(default_factory=lambda: defaultdict(int))
    storage_dtype: torch.dtype = torch.float32
    _alias_gate_up: bool = True
    _gpu: dict[tuple[int, int, str], torch.Tensor] = field(default_factory=dict)
    _gpu_token_count: dict[tuple[int, int, str], int] = field(default_factory=lambda: defaultdict(int))
    _lock: "threading.RLock" = field(default_factory=lambda: threading.RLock())  # noqa: F821

    def set_storage_dtype(self, dtype: torch.dtype) -> None:
        self.storage_dtype = dtype

    def update(
        self, layer_idx: int, expert_idx: int, matrix_name: str, x: torch.Tensor
    ) -> None:
        """Accumulate xᵀx on the *same device as x*. No CPU sync here."""
        if self._alias_gate_up and matrix_name == "up_proj":
            return
        flat = x.detach().reshape(-1, x.shape[-1])
        if flat.numel() == 0:
            return
        flat_f32 = flat.to(torch.float32)
        cov = flat_f32.transpose(0, 1) @ flat_f32
        key = (layer_idx, expert_idx, matrix_name)
        cur = self._gpu.get(key)
        if cur is None:
            self._gpu[key] = cov
        else:
            cur.add_(cov)
        self._gpu_token_count[key] = self._gpu_token_count.get(key, 0) + flat.shape[0]

    def finalize_layer(self, layer_idx: int) -> None:
        keys = [k for k in self._gpu if k[0] == layer_idx]
        for k in keys:
            gpu_cov = self._gpu.pop(k)
            cpu_cov = gpu_cov.to(self.storage_dtype).cpu()
            with self._lock:
                prev = self.covariance.get(k)
                if prev is None:
                    self.covariance[k] = cpu_cov
                else:
                    self.covariance[k] = (
                        prev.to(torch.float32) + cpu_cov.to(torch.float32)
                    ).to(self.storage_dtype)
                self.token_count[k] = (
                    self.token_count.get(k, 0) + self._gpu_token_count.pop(k, 0)
                )

    def finalize_all(self) -> None:
        layer_ids = {k[0] for k in self._gpu}
        for li in layer_ids:
            self.finalize_layer(li)

    def spill_layer_to_disk(self, layer_idx: int, dir_path) -> None:
        import os
        from pathlib import Path
        with self._lock:
            keys = [k for k in self.covariance if k[0] == layer_idx]
            if not keys:
                return
            payload = {
                "format_version": 1,
                "covariance": {k: self.covariance[k] for k in keys},
                "tokens": {k: self.token_count.get(k, 0) for k in keys},
            }
        out = Path(dir_path) / f"layer_{layer_idx}.pt"
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(out.suffix + ".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, out)
        with self._lock:
            for k in keys:
                self.covariance.pop(k, None)

    def load_layer_from_disk(self, layer_idx: int, dir_path) -> bool:
        from pathlib import Path
        p = Path(dir_path) / f"layer_{layer_idx}.pt"
        if not p.exists():
            return False
        try:
            payload = torch.load(p, map_location="cpu")
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
        with self._lock:
            self.covariance.update(payload["covariance"])
            for k, n in payload.get("tokens", {}).items():
                self.token_count[k] = n
        return True

    def unload_layer(self, layer_idx: int) -> None:
        with self._lock:
            for k in [k for k in self.covariance if k[0] == layer_idx]:
                self.covariance.pop(k, None)

    def get(self, key: tuple[int, int, str]) -> torch.Tensor | None:
        if key in self.covariance:
            return self.covariance[key]
        if self._alias_gate_up and key[2] == "up_proj":
            alt = (key[0], key[1], "gate_proj")
            return self.covariance.get(alt)
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
    leading = int(expert_outs.shape[0]) if expert_outs.dim() >= 2 else int(expert_outs.numel())
    if gate_vals.numel() != leading:
        raise RuntimeError(
            f"REAP: gate_vals.numel()={gate_vals.numel()} != expert_outs[0]={leading} "
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
      - ``gate_up_in``    : alias for ``input`` (clarity)

    Context dict passed to each callback:
      {"top_k_weights": [T], "top_k_pos": [T], "token_idx": [T]}
    """
    experts = layer_ref.experts_module
    is_factored = isinstance(experts, FactoredExperts)
    original_forward = experts.forward
    layer_idx = layer_ref.layer_idx

    def _cb(name, eidx, tensor, ctx):
        fn = callbacks.get(name)
        if fn is None:
            return
        fn(layer_idx, int(eidx), tensor, ctx)

    if is_factored:
        def wrapped(self, hidden_states, top_k_index, top_k_weights):
            final = torch.zeros_like(hidden_states)
            with torch.no_grad():
                mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
                hit = (mask.sum(dim=(-1, -2)) > 0).nonzero()
            for expert_idx in hit:
                e = expert_idx[0]
                if e == self.num_experts:
                    continue
                top_k_pos, token_idx = torch.where(mask[e])
                sel = hidden_states[token_idx]
                ctx = {"top_k_weights": top_k_weights[token_idx, top_k_pos],
                       "top_k_pos": top_k_pos, "token_idx": token_idx}
                _cb("input", e, sel, ctx)
                _cb("gate_up_in", e, sel, ctx)
                gate = F.linear(F.linear(sel, self.gate_proj_V[e]), self.gate_proj_U[e])
                up   = F.linear(F.linear(sel, self.up_proj_V[e]),   self.up_proj_U[e])
                intermediate = self.act_fn(gate) * up
                _cb("intermediate", e, intermediate, ctx)
                down = F.linear(F.linear(intermediate, self.down_proj_V[e]),
                                self.down_proj_U[e])
                _cb("down", e, down, ctx)
                down = down * top_k_weights[token_idx, top_k_pos, None]
                final.index_add_(0, token_idx, down.to(final.dtype))
            return final
    else:
        def wrapped(self, hidden_states, top_k_index, top_k_weights):
            final = torch.zeros_like(hidden_states)
            with torch.no_grad():
                mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
                hit = (mask.sum(dim=(-1, -2)) > 0).nonzero()
            for expert_idx in hit:
                e = expert_idx[0]
                if e == self.num_experts:
                    continue
                top_k_pos, token_idx = torch.where(mask[e])
                sel = hidden_states[token_idx]
                ctx = {"top_k_weights": top_k_weights[token_idx, top_k_pos],
                       "top_k_pos": top_k_pos, "token_idx": token_idx}
                _cb("input", e, sel, ctx)
                _cb("gate_up_in", e, sel, ctx)
                gate_up = F.linear(sel, self.gate_up_proj[e])
                _cb("gate_up_out", e, gate_up, ctx)
                gate, up = gate_up.chunk(2, dim=-1)
                intermediate = self.act_fn(gate) * up
                _cb("intermediate", e, intermediate, ctx)
                down = F.linear(intermediate, self.down_proj[e])
                _cb("down", e, down, ctx)
                down = down * top_k_weights[token_idx, top_k_pos, None]
                final.index_add_(0, token_idx, down.to(final.dtype))
            return final

    experts.forward = types.MethodType(wrapped, experts)
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


class _EarlyExitException(Exception):
    """Sentinel raised by a forward hook to abort the forward pass early.

    Used by :func:`run_calibration_early_exit` to avoid executing decoder
    layers after the target layer.  The forward runs under ``torch.no_grad()``
    so no autograd graph is corrupted.
    """
    pass


@contextlib.contextmanager
def early_exit_after_layer(model: nn.Module, target_layer_idx: int):
    """Context manager that installs a forward hook on the decoder layer
    *after* ``target_layer_idx`` that raises :class:`_EarlyExitException`,
    aborting the forward pass once the target layer has fully executed.

    For the last MoE layer there is no next layer to hook — we hook the
    text tower's post-layers module (norm / final_layernorm) if it exists,
    otherwise we let the full forward run (no savings, but correct).

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


def _find_text_tower(model: nn.Module) -> nn.Module:
    """Locate the decoder tower that owns ``.layers``.

    Duplicated from model_io to avoid a circular import; the canonical
    version lives in :mod:`moe_compress.utils.model_io`.
    """
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
    with torch.no_grad(), early_exit_after_layer(model, target_layer_idx) as _:
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
    """Collect **pre-softmax** router logits for each given layer.

    Why a pre-forward hook: ``Qwen3_5MoeTopKRouter.forward`` overwrites its
    first return value with a softmax'd tensor before returning, so a
    ``register_forward_hook`` on the router gives post-softmax probabilities
    rather than logits. For Stage 5 KD we need the raw scores, so we
    recompute ``F.linear(hidden, router.weight)`` ourselves in a pre-forward
    hook. Cheap (one matmul that also runs inside the router) and always
    correct.
    """
    storage: dict[int, list[torch.Tensor]] = {ref.layer_idx: [] for ref in layer_refs}
    handles: list = []

    def _pre_factory(li, router):
        def _h(_m, inputs):
            x = inputs[0]
            if hasattr(router, "hidden_dim"):
                x = x.reshape(-1, router.hidden_dim)
            logits = F.linear(x, router.weight)
            if getattr(router, "bias", None) is not None:
                logits = logits + router.bias
            storage[li].append(logits)
        return _h

    for ref in layer_refs:
        h = ref.router.register_forward_pre_hook(_pre_factory(ref.layer_idx, ref.router))
        handles.append(h)
    try:
        yield storage
    finally:
        for h in handles:
            h.remove()
