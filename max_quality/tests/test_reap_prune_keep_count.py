"""Unit — faithful REAP drop-count derivation keeps exactly round((1-f)*N).

The production case is ``n_experts=256, prune_fraction=0.35``: upstream intent is
"keep the top (1 - compression_ratio) experts", i.e. keep
``round((1 - 0.35) * 256) = round(166.4) = 166`` → drop 90. The previous
``int(256 * 0.35) = 89`` derivation kept 167 (off-by-one vs the REAM 166
survivor count). This test pins the corrected derivation on the real config
shape AND verifies it is inert on the integral fractions the rest of the suite
uses (so no other test silently changes behaviour).

``ReapPrunePlugin.compute_assignment`` computes ``_n_prune`` ONCE from the first
layer and publishes ``final_kept_ids``; we drive it directly with a synthetic
256-wide score vector (no model tensors are touched until ``post_merge``, which
we skip).
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from moe_compress.pipeline.context import PipelineContext
from moe_compress.stage2.plugins.reap_prune import ReapPrunePlugin


def _drive(n_experts: int, prune_fraction: float):
    """Run ``compute_assignment`` on one synthetic layer; return the plugin +
    the published ``final_kept_ids``."""
    plugin = ReapPrunePlugin(
        prune_fraction=prune_fraction, blacklist={}, merge_map={},
        partial_dir=None,
    )
    ctx = PipelineContext()
    ctx.set("layer_ref", SimpleNamespace(layer_idx=0))
    # Distinct scores → deterministic top-K, no ties.
    rng = np.random.default_rng(0)
    ctx.set("scores", rng.permutation(n_experts).astype(np.float64))
    ctx.set("n_experts", n_experts)
    # Sentinel: the fail-loud guard gates on ``reap_scores_payload`` presence.
    ctx.set("reap_scores_payload", object())
    plugin.compute_assignment(ctx)
    return plugin, list(ctx.get("final_kept_ids"))


def test_production_case_256_at_035_keeps_166_drops_90():
    plugin, kept = _drive(n_experts=256, prune_fraction=0.35)
    assert plugin._n_prune == 90, (
        f"expected drop=90 (256 - round(0.65*256)=256-166), got {plugin._n_prune}"
    )
    assert len(kept) == 166, f"expected keep=166, got {len(kept)}"


@pytest.mark.parametrize(
    "n_experts, prune_fraction, exp_drop, exp_keep",
    [
        # Integral n_experts*fraction → corrected derivation is INERT (matches
        # the old int() value), so the existing golden/integration suite is
        # unchanged.
        (8, 0.5, 4, 4),
        (256, 0.5, 128, 128),
        (256, 0.25, 64, 192),
        # Non-integral → the corrected keep-rounded derivation differs from the
        # old truncating int() (which would keep 167 at 0.35).
        (256, 0.35, 90, 166),
        (128, 0.35, 45, 83),  # round(0.65*128)=round(83.2)=83 → drop 45
    ],
)
def test_drop_count_is_keep_rounded(n_experts, prune_fraction, exp_drop, exp_keep):
    plugin, kept = _drive(n_experts=n_experts, prune_fraction=prune_fraction)
    assert plugin._n_prune == exp_drop
    assert len(kept) == exp_keep
    # Invariant: drop == N - keep == N - round((1-f)*N).
    assert plugin._n_prune == n_experts - round(n_experts * (1.0 - prune_fraction))
