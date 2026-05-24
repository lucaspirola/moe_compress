"""AIMER: Absolute mean over root mean square IMportance for Expert Ranking.

AIMER (arxiv:2603.18492) measures weight-vector "peakedness":

    score = ||w||_1 / (sqrt(N) · ||w||_2)

where N = numel(w). The score is in (0, 1]:
- score = 1.0 → all weights equal magnitude (most distributed/replaceable)
- score = 1/sqrt(N) → one-hot (most concentrated/critical)

Lower score = more peaky/concentrated = more critical to keep unmerged.

Used in Stage 1 Phase C+ to auto-extend the SE blacklist with experts whose
down_proj weights have concentrated energy (the structural signature of an SE
that may have been missed by the residual-stream-based detector).
"""
from __future__ import annotations

import math

import torch


def aimer_score_tensor(w: torch.Tensor) -> float:
    """Compute AIMER score for a single weight tensor (any shape; flattened internally)."""
    flat = w.detach().to(torch.float32).flatten()
    n = flat.numel()
    if n == 0:
        return 0.0
    l1 = float(flat.abs().sum().item())
    l2 = float(flat.norm(p=2).item())
    if l2 == 0.0:
        return 0.0
    return l1 / (math.sqrt(n) * l2)


def aimer_bottom_pct_per_layer(
    scores: dict[tuple[int, int], float],
    pct: float,
) -> dict[int, list[int]]:
    """Return per-layer expert ids in the bottom `pct` fraction of AIMER scores.

    Sorted ascending by score (lowest = most concentrated = most critical first).
    """
    by_layer: dict[int, list[tuple[int, float]]] = {}
    for (li, e), s in scores.items():
        by_layer.setdefault(li, []).append((e, s))
    out: dict[int, list[int]] = {}
    for li, lst in by_layer.items():
        lst.sort(key=lambda t: t[1])
        k = max(1, int(round(len(lst) * pct)))
        out[li] = [e for e, _ in lst[:k]]
    return out
