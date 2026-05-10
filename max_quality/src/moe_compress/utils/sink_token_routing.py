"""Sink-token routing analysis for SE detection.

Identifies experts whose routing pattern is sink-token-dominated (a structural
signature of Super Experts per 2507.23279 Figures 20-21). Sink tokens are
defined as: positions where input_id == BOS_token_id OR position 0 of each
sequence (the leading-position sink that forms regardless of token identity).

Used in Stage 1 Phase C+ to auto-extend the SE blacklist with experts that
exhibit:
- mean_router_score_on_sink >> mean_router_score_on_normal (ratio threshold)
- activation_frequency_on_sink ≈ 1.0 (always fires when a sink token is present)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

EPS = 1e-12


@dataclass
class SinkTokenRoutingAccumulator:
    num_layers: int
    num_experts: int
    bos_token_id: Optional[int] = None
    # internal running stats
    _sum_score_sink: dict[tuple[int, int], float] = field(default_factory=dict)
    _sum_score_normal: dict[tuple[int, int], float] = field(default_factory=dict)
    _count_sink: dict[tuple[int, int], int] = field(default_factory=dict)
    _count_normal: dict[tuple[int, int], int] = field(default_factory=dict)
    _fire_on_sink: dict[tuple[int, int], int] = field(default_factory=dict)
    _total_sink_tokens: int = 0
    # finalized
    mean_router_score_sink: dict[tuple[int, int], float] = field(default_factory=dict)
    mean_router_score_normal: dict[tuple[int, int], float] = field(default_factory=dict)
    freq_on_sink: dict[tuple[int, int], float] = field(default_factory=dict)

    def _build_sink_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        # shape (batch, seq) → bool mask
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        mask[:, 0] = True  # leading-position sink
        if self.bos_token_id is not None:
            mask = mask | (input_ids == self.bos_token_id)
        return mask

    def update(
        self,
        layer_idx: int,
        input_ids: torch.Tensor,           # (batch, seq)
        router_scores: torch.Tensor,        # (batch, seq, num_experts) — post-softmax
        routed_pos: torch.Tensor,           # (batch, seq, top_k) — expert ids actually routed
    ) -> None:
        # Move mask to CPU to match the CPU-resident scores/routed tensors used below.
        # input_ids may be on GPU during real forward passes; without this, indexing
        # the CPU `scores_e` with a GPU `sink_mask` raises:
        #   RuntimeError: indices should be either on cpu or on the same device as
        #   the indexed tensor (cpu)
        sink_mask = self._build_sink_mask(input_ids).cpu()  # (batch, seq) bool, CPU
        sink_idx = sink_mask.nonzero(as_tuple=False)
        self._total_sink_tokens += sink_idx.shape[0]
        scores_cpu = router_scores.detach().to(torch.float32).cpu()
        routed_cpu = routed_pos.detach().cpu()

        for e in range(self.num_experts):
            key = (layer_idx, e)
            scores_e = scores_cpu[..., e]   # (batch, seq)
            sink_scores = scores_e[sink_mask]
            norm_scores = scores_e[~sink_mask]
            self._sum_score_sink[key] = self._sum_score_sink.get(key, 0.0) + float(sink_scores.sum().item())
            self._sum_score_normal[key] = self._sum_score_normal.get(key, 0.0) + float(norm_scores.sum().item())
            self._count_sink[key] = self._count_sink.get(key, 0) + int(sink_scores.numel())
            self._count_normal[key] = self._count_normal.get(key, 0) + int(norm_scores.numel())
            # activation freq on sink: did expert e appear in routed_pos at any sink position?
            fires_at_sink = (routed_cpu == e)             # (batch, seq, top_k) bool
            fires_at_sink = fires_at_sink.any(dim=-1)      # (batch, seq) bool
            sink_fires = (fires_at_sink & sink_mask).sum().item()
            self._fire_on_sink[key] = self._fire_on_sink.get(key, 0) + int(sink_fires)

    def finalize(self) -> None:
        for key, s in self._sum_score_sink.items():
            n = self._count_sink.get(key, 0)
            self.mean_router_score_sink[key] = (s / n) if n > 0 else 0.0
        for key, s in self._sum_score_normal.items():
            n = self._count_normal.get(key, 0)
            self.mean_router_score_normal[key] = (s / n) if n > 0 else 0.0
        for key, fires in self._fire_on_sink.items():
            self.freq_on_sink[key] = (fires / self._total_sink_tokens) if self._total_sink_tokens > 0 else 0.0


def apply_sink_token_extension(
    mean_score_sink: dict[tuple[int, int], float],
    mean_score_normal: dict[tuple[int, int], float],
    freq_on_sink: dict[tuple[int, int], float],
    existing_blacklist: dict[int, list[int]],
    score_ratio: float,
    freq_threshold: float,
) -> dict[int, list[int]]:
    """Return per-layer expert ids that meet the sink-token auto-extension criterion
    AND are not already in `existing_blacklist`."""
    already = {(li, e) for li, lst in existing_blacklist.items() for e in lst}
    out: dict[int, list[int]] = {}
    for key, s_sink in mean_score_sink.items():
        if key in already:
            continue
        s_norm = mean_score_normal.get(key, 0.0)
        f_sink = freq_on_sink.get(key, 0.0)
        ratio = s_sink / max(s_norm, EPS)
        if ratio > score_ratio and f_sink > freq_threshold:
            out.setdefault(key[0], []).append(key[1])
    for li in out:
        out[li] = sorted(out[li])
    return out
