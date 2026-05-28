"""Tests for ``Stage1PerExpertMaxCacheProvider`` -- the Stage 1 cache-
side provider for the per-(layer, expert) ``down_proj`` output max
L_inf sidecar (Item 2 of the calibration-v2 writers campaign).

Five tests:

1. ``test_save_load_roundtrip`` -- a synthetic ``Stage1PerExpertMaxPayload``
   round-trips through ``save_per_expert_max`` / ``load_per_expert_max``
   with byte-identical tensor values.
2. ``test_schema_version_mismatch_raises`` -- bumping
   ``SCHEMA_VERSIONS["per_expert_max"]`` after a sidecar is written makes
   ``load_per_expert_max`` raise ``ValueError`` with the actionable
   "Delete the sidecar to regenerate" message.
3. ``test_cache_provider_hit`` -- writing a sidecar with known per-
   expert values and calling ``on_load`` with a stub ctx containing
   ``moe_layers`` populates ``ctx.max_acc`` with a
   ``DownProjMaxAccumulator`` whose ``per_expert_max`` dict is keyed by
   ``(layer_idx, expert_id)``.
4. ``test_cache_provider_miss`` -- no sidecar on disk → ``on_load``
   returns ``None`` and leaves the ctx untouched.
5. ``test_zero_expert_excluded_from_dict`` -- experts whose cached
   value is exactly ``0.0`` (zero-traffic) are NOT inserted into the
   accumulator's dict, matching the live
   ``DownProjMaxAccumulator``'s absent-key convention for zero-traffic.

CPU-only by construction (every tensor allocation defaults to CPU).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from moe_compress.pipeline.context import PipelineContext
from moe_compress.stage1.plugins.per_expert_max_cache import (
    Stage1PerExpertMaxCacheProvider,
)
from moe_compress.utils.activation_hooks import DownProjMaxAccumulator
from moe_compress.utils.cached_calibration_signals import (
    SCHEMA_VERSIONS,
    Stage1PerExpertMaxPayload,
    load_per_expert_max,
    save_per_expert_max,
    sidecar_path,
)


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


@dataclass
class _StubMoELayerRef:
    """Tiny stand-in for MoELayerRef -- the cache provider only reads
    ``.layer_idx``, so the test fixture supplies just that."""
    layer_idx: int


def _jsonl(tmp_path: Path) -> Path:
    """Return a non-existent JSONL path inside ``tmp_path`` -- the sidecar
    lives next to it under ``tmp_path/sidecars/``."""
    return tmp_path / "trace.jsonl"


def _make_payload(n_layers: int, n_experts: int,
                  per_expert_max: torch.Tensor | None = None,
                  token_counts: torch.Tensor | None = None,
                  ) -> Stage1PerExpertMaxPayload:
    if per_expert_max is None:
        per_expert_max = torch.arange(
            n_layers * n_experts, dtype=torch.float32,
        ).reshape(n_layers, n_experts) + 1.0   # all > 0 by construction
    if token_counts is None:
        token_counts = torch.full(
            (n_layers, n_experts), 5, dtype=torch.int64,
        )
    return Stage1PerExpertMaxPayload(
        schema_version=SCHEMA_VERSIONS["per_expert_max"],
        n_experts=n_experts,
        n_layers=n_layers,
        per_expert_max=per_expert_max,
        token_counts=token_counts,
    )


# ---------------------------------------------------------------------------
# Test 1 -- save/load round-trip
# ---------------------------------------------------------------------------


def test_save_load_roundtrip(tmp_path):
    """Save the payload, load it back, and verify tensor equality."""
    jsonl = _jsonl(tmp_path)
    original = _make_payload(n_layers=3, n_experts=4)
    save_per_expert_max(original, jsonl)

    expected_path = sidecar_path(jsonl, "per_expert_max")
    assert expected_path.exists()
    assert not Path(str(expected_path) + ".tmp").exists()

    loaded = load_per_expert_max(jsonl)
    assert loaded is not None
    assert loaded.schema_version == SCHEMA_VERSIONS["per_expert_max"]
    assert loaded.n_layers == 3
    assert loaded.n_experts == 4
    assert torch.equal(loaded.per_expert_max, original.per_expert_max.cpu())
    assert torch.equal(loaded.token_counts, original.token_counts.cpu())
    assert loaded.per_expert_max.dtype == torch.float32
    assert loaded.token_counts.dtype == torch.int64


# ---------------------------------------------------------------------------
# Test 2 -- schema-version mismatch
# ---------------------------------------------------------------------------


def test_schema_version_mismatch_raises(tmp_path, monkeypatch):
    """Bumping ``SCHEMA_VERSIONS["per_expert_max"]`` after a sidecar is
    written makes ``load_per_expert_max`` raise ``RuntimeError`` from
    the Pattern O manifest validator (``_validate_manifest_or_warn``)
    with the actionable "Delete both ... re-run calibration" message."""
    jsonl = _jsonl(tmp_path)
    payload = _make_payload(n_layers=2, n_experts=3)
    save_per_expert_max(payload, jsonl)

    # Bump the central schema version dict; the loader must detect the
    # mismatch and raise.
    import moe_compress.utils.cached_calibration_signals as ccs
    monkeypatch.setitem(ccs.SCHEMA_VERSIONS, "per_expert_max", 99)
    with pytest.raises(RuntimeError, match="manifest validation FAILED"):
        load_per_expert_max(jsonl)


# ---------------------------------------------------------------------------
# Test 3 -- cache provider hit
# ---------------------------------------------------------------------------


def test_cache_provider_hit(tmp_path):
    """Write a sidecar with known per-expert values; call ``on_load`` with
    a stub ctx; verify ``ctx.max_acc.per_expert_max`` is a populated dict
    keyed by ``(layer_idx, expert_id)``."""
    jsonl = _jsonl(tmp_path)
    # Two layers with distinct layer_idx values (NOT 0,1) so we exercise
    # the rank -> layer_idx mapping. n_experts = 3.
    moe_layers = [
        _StubMoELayerRef(layer_idx=7),
        _StubMoELayerRef(layer_idx=11),
    ]
    per_expert_max = torch.tensor(
        [[1.5, 2.5, 3.5],
         [4.0, 5.0, 6.0]], dtype=torch.float32,
    )
    payload = _make_payload(
        n_layers=2, n_experts=3, per_expert_max=per_expert_max,
    )
    save_per_expert_max(payload, jsonl)

    ctx = PipelineContext()
    ctx.set("moe_layers", moe_layers)
    provider = Stage1PerExpertMaxCacheProvider()
    returned = provider.on_load(ctx, jsonl)

    assert returned is not None
    assert returned.n_layers == 2
    assert returned.n_experts == 3
    # Provider populated ctx.max_acc.
    assert ctx.has("max_acc")
    acc = ctx.get("max_acc")
    assert isinstance(acc, DownProjMaxAccumulator)
    # Dict is keyed by (layer_idx, expert_id) where layer_idx maps through
    # the live moe_layers list. All six cells are > 0 so all six survive.
    assert acc.per_expert_max == {
        (7, 0): 1.5, (7, 1): 2.5, (7, 2): 3.5,
        (11, 0): 4.0, (11, 1): 5.0, (11, 2): 6.0,
    }


# ---------------------------------------------------------------------------
# Test 4 -- cache provider miss
# ---------------------------------------------------------------------------


def test_cache_provider_miss(tmp_path):
    """No sidecar on disk → ``on_load`` returns None and leaves ctx
    untouched."""
    jsonl = _jsonl(tmp_path)
    # No sidecar written.

    moe_layers = [_StubMoELayerRef(layer_idx=0)]
    ctx = PipelineContext()
    ctx.set("moe_layers", moe_layers)

    provider = Stage1PerExpertMaxCacheProvider()
    returned = provider.on_load(ctx, jsonl)

    assert returned is None
    # ctx is untouched -- max_acc was not added.
    assert not ctx.has("max_acc")


# ---------------------------------------------------------------------------
# Test 5 -- zero-traffic experts excluded from the dict
# ---------------------------------------------------------------------------


def test_zero_expert_excluded_from_dict(tmp_path):
    """Experts whose cached value is exactly 0.0 (zero-traffic) are NOT
    inserted into the accumulator's dict, matching the live
    ``DownProjMaxAccumulator``'s absent-key convention for zero-traffic
    experts."""
    jsonl = _jsonl(tmp_path)
    moe_layers = [
        _StubMoELayerRef(layer_idx=3),
        _StubMoELayerRef(layer_idx=5),
    ]
    # rank 0: e0=2.0, e1=0.0 (zero traffic), e2=4.5, e3=0.0
    # rank 1: all zeros (entire layer saw no traffic)
    per_expert_max = torch.tensor(
        [[2.0, 0.0, 4.5, 0.0],
         [0.0, 0.0, 0.0, 0.0]], dtype=torch.float32,
    )
    payload = _make_payload(
        n_layers=2, n_experts=4, per_expert_max=per_expert_max,
    )
    save_per_expert_max(payload, jsonl)

    ctx = PipelineContext()
    ctx.set("moe_layers", moe_layers)
    provider = Stage1PerExpertMaxCacheProvider()
    returned = provider.on_load(ctx, jsonl)

    assert returned is not None
    acc = ctx.get("max_acc")
    assert isinstance(acc, DownProjMaxAccumulator)
    # Only the two non-zero cells appear in the dict.
    assert acc.per_expert_max == {
        (3, 0): 2.0,
        (3, 2): 4.5,
    }
    # Rank-1 layer (layer_idx=5) is entirely absent from the dict
    # (no keys with layer_idx=5).
    assert not any(k[0] == 5 for k in acc.per_expert_max)
