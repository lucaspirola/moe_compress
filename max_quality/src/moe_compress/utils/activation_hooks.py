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
        "down":         down_max_cb,     # Stage 0
        "input":        cov_cb,          # Stage 2/3 gate_up_proj input cov
        "intermediate": int_cov_cb,      # Stage 2/3 down_proj input cov
    }
    with instrument_experts(layer_ref, callbacks):
        for batch in batches:
            model(input_ids=batch)

The instrumentation is per-layer. Install on each MoE layer you want to
observe; caller handles which layers' data to collect.

This module also keeps the previously-used accumulator dataclasses
(``DownProjMaxAccumulator``, ``ReapAccumulator``, ``InputCovarianceAccumulator``)
because Stages 0/2/3 still use their API — only the hook plumbing below them
changed.
"""
from __future__ import annotations

import contextlib
import logging
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


@dataclass
class DownProjMaxAccumulator:
    per_expert_max: dict[tuple[int, int], float] = field(default_factory=dict)

    def update(self, layer_idx: int, expert_idx: int, x: torch.Tensor) -> None:
        cur = float(x.detach().abs().max().cpu().item())
        key = (layer_idx, expert_idx)
        if cur > self.per_expert_max.get(key, 0.0):
            self.per_expert_max[key] = cur


@dataclass
class ReapAccumulator:
    sums: dict[tuple[int, int], float] = field(default_factory=lambda: defaultdict(float))
    counts: dict[tuple[int, int], int] = field(default_factory=lambda: defaultdict(int))
    freq: dict[tuple[int, int], int] = field(default_factory=lambda: defaultdict(int))

    def score(self, layer_idx: int, expert_idx: int) -> float:
        k = (layer_idx, expert_idx)
        n = self.counts.get(k, 0)
        if n == 0:
            return 0.0
        return self.sums[k] / n


@dataclass
class InputCovarianceAccumulator:
    """Per-(layer, expert, matrix_name) streaming covariance accumulator.

    Storage optimizations (addresses review P0-3):

    - Covariances are stored on CPU in ``torch.float32`` but **with gate_proj
      and up_proj aliasing the same storage**: both share the same input
      tensor inside ``Qwen3_5MoeExperts`` (x entering ``gate_up_proj``), so
      we intern the tensor once and return it under both keys. Callers that
      write via ``update`` with ``matrix_name="up_proj"`` when the same
      ``(layer, expert)`` already has ``gate_proj`` data become no-ops.

    - Entries are returned as fresh tensors on demand; internal storage uses
      ``float32`` for numerical stability but can be lowered to ``bfloat16``
      via ``set_storage_dtype`` when RAM pressure demands it (matters for
      40-layer × 256-expert full runs where fp32 cov would be ~113 GB).
    """

    covariance: dict[tuple[int, int, str], torch.Tensor] = field(default_factory=dict)
    token_count: dict[tuple[int, int, str], int] = field(default_factory=lambda: defaultdict(int))
    storage_dtype: torch.dtype = torch.float32
    _alias_gate_up: bool = True       # whether up_proj shares gate_proj storage

    def set_storage_dtype(self, dtype: torch.dtype) -> None:
        self.storage_dtype = dtype

    def update(
        self, layer_idx: int, expert_idx: int, matrix_name: str, x: torch.Tensor
    ) -> None:
        # gate_proj and up_proj share input; let only gate_proj actually
        # allocate storage and have get()/items() surface both keys.
        if self._alias_gate_up and matrix_name == "up_proj":
            return
        flat = x.detach().reshape(-1, x.shape[-1]).to(torch.float32)
        if flat.numel() == 0:
            return
        cov_full = flat.transpose(0, 1) @ flat
        cov = cov_full.to(self.storage_dtype).cpu()
        key = (layer_idx, expert_idx, matrix_name)
        if key in self.covariance:
            self.covariance[key] = (self.covariance[key].to(torch.float32) + cov.to(torch.float32)).to(self.storage_dtype)
        else:
            self.covariance[key] = cov
        self.token_count[key] = self.token_count.get(key, 0) + flat.shape[0]

    def get(self, key: tuple[int, int, str]) -> torch.Tensor | None:
        """Bank-aware accessor: returns gate_proj cov when asked for up_proj
        under the alias policy, so callers do not need to know about the
        sharing."""
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
    """``gate_vals`` [T], ``expert_outs`` [T, hidden]."""
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
    k = (layer_idx, expert_idx)
    acc.sums[k] += float(contrib.detach().cpu().item())
    acc.counts[k] += int(gate_vals.numel())
    acc.freq[k] += int(gate_vals.numel())


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
) -> None:
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(batches):
            if device is not None:
                batch = batch.to(device)
            model(input_ids=batch, **(extra_forward_kwargs or {}))
            if per_batch_callback is not None:
                per_batch_callback(i)


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
            storage[li].append(logits.detach())
        return _h

    for ref in layer_refs:
        h = ref.router.register_forward_pre_hook(_pre_factory(ref.layer_idx, ref.router))
        handles.append(h)
    try:
        yield storage
    finally:
        for h in handles:
            h.remove()
