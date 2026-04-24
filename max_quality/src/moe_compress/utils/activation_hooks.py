"""Forward-hook context managers used across stages.

Each stage needs a different pattern of hooks — Stage 0 wants only the
``down_proj`` output max, Stage 2 wants gate values + expert outputs, Stage 3
wants the *input* to each expert matrix to build the A / B covariances used by
AA-SVD. Centralizing the plumbing here keeps stage code short.
"""
from __future__ import annotations

import contextlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn as nn

from .model_io import MoELayerRef, get_expert_matrices, iter_routed_experts


# ---------------------------------------------------------------------------
# Stage 0 : per-expert down_proj max
# ---------------------------------------------------------------------------


@dataclass
class DownProjMaxAccumulator:
    """Track running max(|down_proj output|) per (layer, expert)."""

    per_expert_max: dict[tuple[int, int], float] = field(default_factory=dict)

    def update(self, layer_idx: int, expert_idx: int, x: torch.Tensor) -> None:
        cur = float(x.detach().abs().max().cpu().item())
        key = (layer_idx, expert_idx)
        prev = self.per_expert_max.get(key, 0.0)
        if cur > prev:
            self.per_expert_max[key] = cur


@contextlib.contextmanager
def hook_down_proj_max(
    moe_layers: list[MoELayerRef], acc: DownProjMaxAccumulator
):
    handles: list = []
    for layer_ref in moe_layers:
        for expert_idx, expert in iter_routed_experts(layer_ref):
            mats = get_expert_matrices(expert)
            if "down_proj" not in mats:
                continue
            mod = mats["down_proj"]
            l_idx = layer_ref.layer_idx
            e_idx = expert_idx

            def _make_hook(l: int, e: int):
                def _hook(_mod, _inp, out):
                    acc.update(l, e, out if isinstance(out, torch.Tensor) else out[0])

                return _hook

            handles.append(mod.register_forward_hook(_make_hook(l_idx, e_idx)))
    try:
        yield acc
    finally:
        for h in handles:
            h.remove()


# ---------------------------------------------------------------------------
# Stage 2 : REAP per-expert score = mean_x g(x) · ||f(x)||_2
# ---------------------------------------------------------------------------


@dataclass
class ReapAccumulator:
    """Accumulate ``Σ_x g_j(x) · ||f_j(x)||_2`` and token counts per expert."""

    sums: dict[tuple[int, int], float] = field(default_factory=lambda: defaultdict(float))
    counts: dict[tuple[int, int], int] = field(default_factory=lambda: defaultdict(int))
    # Optional: frequency table for merge-time reweighting
    freq: dict[tuple[int, int], int] = field(default_factory=lambda: defaultdict(int))

    def score(self, layer_idx: int, expert_idx: int) -> float:
        k = (layer_idx, expert_idx)
        n = self.counts.get(k, 0)
        if n == 0:
            return 0.0
        return self.sums[k] / n


def record_reap(
    acc: ReapAccumulator,
    layer_idx: int,
    expert_idx: int,
    gate_vals: torch.Tensor,
    expert_outs: torch.Tensor,
) -> None:
    """Called by the Stage 2 driver for every ``(layer, expert, batch)`` event.

    ``gate_vals`` shape ``[T]`` : router gate g_j(x) for the T tokens routed
    to this expert. ``expert_outs`` shape ``[T, hidden]`` : f_j(x) for those
    tokens. Both on GPU, fp32 recommended.
    """
    if gate_vals.numel() == 0:
        return
    # FIX (review bug #12): fail loudly if the HF MoE forward dispatches
    # expert_outs in an order that doesn't match the (b,t,k) row-major mask
    # we extracted gate_vals from — otherwise REAP scores are silently wrong.
    leading = int(expert_outs.shape[0]) if expert_outs.dim() >= 2 else int(expert_outs.numel())
    if gate_vals.numel() != leading:
        raise RuntimeError(
            f"REAP scoring: gate_vals.numel()={gate_vals.numel()} does not match "
            f"expert_outs leading-dim={leading} (layer={layer_idx}, expert={expert_idx}). "
            "This typically means the HF MoE forward batches expert_outs in a "
            "different order than (b,t,k) — update _ReapLayerHook to match."
        )
    norms = expert_outs.to(torch.float32).norm(dim=-1)         # [T]
    contrib = (gate_vals.to(torch.float32) * norms).sum()      # scalar
    k = (layer_idx, expert_idx)
    acc.sums[k] += float(contrib.detach().cpu().item())
    acc.counts[k] += int(gate_vals.numel())
    acc.freq[k] += int(gate_vals.numel())


# ---------------------------------------------------------------------------
# Stage 3 : per-matrix input covariance  A = Σ x x^T
# ---------------------------------------------------------------------------


@dataclass
class InputCovarianceAccumulator:
    """Collects ``Σ x^T x`` per ``(layer, expert, matrix_name)`` in fp32."""

    covariance: dict[tuple[int, int, str], torch.Tensor] = field(default_factory=dict)
    token_count: dict[tuple[int, int, str], int] = field(default_factory=lambda: defaultdict(int))

    def update(
        self, layer_idx: int, expert_idx: int, matrix_name: str, x: torch.Tensor
    ) -> None:
        # x shape: [..., in_features] → flatten to 2D
        flat = x.detach().reshape(-1, x.shape[-1]).to(torch.float32)
        if flat.numel() == 0:
            return
        cov = flat.transpose(0, 1) @ flat            # [in, in]
        key = (layer_idx, expert_idx, matrix_name)
        if key not in self.covariance:
            self.covariance[key] = cov.cpu()
        else:
            self.covariance[key] = self.covariance[key] + cov.cpu()
        self.token_count[key] += flat.shape[0]


@contextlib.contextmanager
def hook_matrix_inputs(
    moe_layers: list[MoELayerRef],
    acc: InputCovarianceAccumulator,
    matrix_names: tuple[str, ...] = ("gate_proj", "up_proj", "down_proj"),
):
    """Forward pre-hook: ``x`` entering each expert's named Linear."""
    handles: list = []
    for layer_ref in moe_layers:
        for expert_idx, expert in iter_routed_experts(layer_ref):
            mats = get_expert_matrices(expert)
            for name in matrix_names:
                if name not in mats:
                    continue
                mod = mats[name]
                l_idx = layer_ref.layer_idx
                e_idx = expert_idx

                def _make_hook(l: int, e: int, n: str):
                    def _pre(_mod, inp):
                        x = inp[0] if isinstance(inp, tuple) else inp
                        acc.update(l, e, n, x)

                    return _pre

                handles.append(mod.register_forward_pre_hook(_make_hook(l_idx, e_idx, name)))
    try:
        yield acc
    finally:
        for h in handles:
            h.remove()


# ---------------------------------------------------------------------------
# Generic helper : run a set of batches through the model, no grad
# ---------------------------------------------------------------------------


def run_calibration(
    model: nn.Module,
    batches,
    *,
    device=None,
    extra_forward_kwargs: dict | None = None,
    per_batch_callback: Callable[[int], None] | None = None,
) -> None:
    """Forward each batch through the model under ``torch.no_grad()``."""
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(batches):
            if device is not None:
                batch = batch.to(device)
            model(input_ids=batch, **(extra_forward_kwargs or {}))
            if per_batch_callback is not None:
                per_batch_callback(i)
