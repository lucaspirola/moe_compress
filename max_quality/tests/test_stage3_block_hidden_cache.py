"""Tests for ``Stage3BlockHiddenCacheProvider`` (Item 7 reader, Stage 3 side).

Covers:

1. ``test_load_miss_no_dir`` -- no sidecars directory → ``on_load``
   returns None (does not raise) and ctx is untouched.
2. ``test_load_hit_populates_teacher_targets_cache`` -- per-layer
   sidecars present + ctx has calib + batches → ``on_load`` returns a
   non-None payload AND populates
   ``ctx.teacher_targets_cache: dict[int, list[Tensor]]`` with per-batch
   ``[batch_size, seq_len, hidden]`` bf16 tensors per layer.
3. ``test_schema_mismatch_raises`` -- forced schema=99 →
   ``load_block_hidden`` raises ValueError with the actionable
   "Delete the sidecar to regenerate" message.
4. ``test_token_count_mismatch_falls_through_to_miss`` -- sidecar
   ``n_tokens`` does not match ``n_prompts × seq_len`` → ``on_load``
   returns None and does NOT populate ``ctx.teacher_targets_cache``.
5. ``test_dispatch_first_routes_through_registry`` -- a one-element
   registry of [Stage3BlockHiddenCacheProvider] processed via
   ``PluginRegistry.dispatch_first`` returns the first non-None winner
   on cache hit.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from moe_compress.pipeline.context import PipelineContext
from moe_compress.stage3.plugins.block_hidden_cache import (
    Stage3BlockHiddenCacheProvider,
)
from moe_compress.utils.cached_calibration_signals import (
    SCHEMA_VERSIONS,
    BlockHiddenPayload,
    save_block_hidden,
    sidecar_path,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _jsonl(tmp_path: Path) -> Path:
    return tmp_path / "trace_0001.jsonl"


def _write_layer_sidecars(
    tmp_path: Path,
    *,
    layer_indices: list[int],
    n_prompts: int,
    seq_len: int,
    hidden: int,
) -> Path:
    """Write one ``BlockHiddenPayload`` per layer; return the JSONL path."""
    jsonl = _jsonl(tmp_path)
    for li in layer_indices:
        # Deterministic content so tests can assert on it: token i,j,k
        # holds value (li, i*seq_len + j, k) so each cell is identifiable.
        hs = torch.zeros(n_prompts * seq_len, hidden, dtype=torch.bfloat16)
        for t in range(n_prompts * seq_len):
            hs[t, 0] = float(li)
            hs[t, 1] = float(t)
        payload = BlockHiddenPayload(
            schema_version=SCHEMA_VERSIONS["block_hidden"],
            layer_idx=li,
            n_prompts_in_subset=n_prompts,
            hidden_states=hs,
        )
        save_block_hidden(payload, jsonl)
    return jsonl


def _populate_ctx_for_alignment(
    ctx: PipelineContext, *, n_prompts: int, seq_len: int, batch_size: int
) -> None:
    """Populate ctx.calib + ctx.batches the way the orchestrator does."""
    calib = torch.zeros(n_prompts, seq_len, dtype=torch.long)
    ctx.set("calib", calib)
    n_batches = n_prompts // batch_size
    batches = [calib[b * batch_size:(b + 1) * batch_size]
               for b in range(n_batches)]
    ctx.set("batches", batches)


# ---------------------------------------------------------------------------
# Test 1 -- miss (no sidecars dir)
# ---------------------------------------------------------------------------


def test_load_miss_no_dir(tmp_path):
    jsonl = _jsonl(tmp_path)
    # Nothing is written under tmp_path; sidecars dir doesn't exist.
    provider = Stage3BlockHiddenCacheProvider()
    ctx = PipelineContext()
    result = provider.on_load(ctx, jsonl)
    assert result is None
    assert not ctx.has("teacher_targets_cache")


# ---------------------------------------------------------------------------
# Test 2 -- hit populates teacher_targets_cache
# ---------------------------------------------------------------------------


def test_load_hit_populates_teacher_targets_cache(tmp_path):
    n_prompts = 8
    seq_len = 4
    hidden = 3
    batch_size = 2
    layer_indices = [3, 7]

    jsonl = _write_layer_sidecars(
        tmp_path,
        layer_indices=layer_indices,
        n_prompts=n_prompts,
        seq_len=seq_len,
        hidden=hidden,
    )
    # The sidecar files must exist on disk under sidecars/block_hidden/.
    assert sidecar_path(jsonl, "block_hidden/layer_0003").exists()
    assert sidecar_path(jsonl, "block_hidden/layer_0007").exists()

    ctx = PipelineContext()
    _populate_ctx_for_alignment(
        ctx, n_prompts=n_prompts, seq_len=seq_len, batch_size=batch_size,
    )

    provider = Stage3BlockHiddenCacheProvider()
    result = provider.on_load(ctx, jsonl)
    assert result is not None
    assert ctx.has("teacher_targets_cache")

    cache = ctx.get("teacher_targets_cache")
    assert set(cache.keys()) == {3, 7}
    n_batches = n_prompts // batch_size  # = 4
    for li in layer_indices:
        batches_list = cache[li]
        assert len(batches_list) == n_batches
        for b in batches_list:
            assert b.dtype == torch.bfloat16
            assert b.shape == (batch_size, seq_len, hidden)
            assert b.device.type == "cpu"

    # Spot-check deterministic content: layer 3, batch 0 (covers prompts
    # 0..1), token (0, 0, 0) was written as (li=3, t=0).
    b0 = cache[3][0]
    assert b0[0, 0, 0].item() == 3.0
    assert b0[0, 0, 1].item() == 0.0
    # layer 7, batch 1 (covers prompts 2..3), token (prompt=1, pos=0)
    # in batch coords corresponds to global prompt 3, position 0
    # -> t = 3 * seq_len + 0 = 12.
    b1 = cache[7][1]
    assert b1[1, 0, 0].item() == 7.0
    assert b1[1, 0, 1].item() == 12.0


# ---------------------------------------------------------------------------
# Test 3 -- schema mismatch raises
# ---------------------------------------------------------------------------


def test_schema_mismatch_raises(tmp_path, monkeypatch):
    n_prompts, seq_len, hidden = 4, 2, 2
    jsonl = _write_layer_sidecars(
        tmp_path,
        layer_indices=[0],
        n_prompts=n_prompts,
        seq_len=seq_len,
        hidden=hidden,
    )

    # Bump the central version AFTER the write, mimicking a code upgrade.
    bumped = dict(SCHEMA_VERSIONS)
    bumped["block_hidden"] = 99
    monkeypatch.setattr(
        "moe_compress.utils.cached_calibration_signals.SCHEMA_VERSIONS",
        bumped,
    )

    ctx = PipelineContext()
    _populate_ctx_for_alignment(
        ctx, n_prompts=n_prompts, seq_len=seq_len, batch_size=2,
    )

    provider = Stage3BlockHiddenCacheProvider()
    with pytest.raises(ValueError) as exc:
        provider.on_load(ctx, jsonl)
    msg = str(exc.value)
    assert "schema_version=1" in msg
    assert "expected 99" in msg
    assert "Delete the sidecar to regenerate" in msg


# ---------------------------------------------------------------------------
# Test 4 -- token count mismatch -> miss (not raise)
# ---------------------------------------------------------------------------


def test_token_count_mismatch_falls_through_to_miss(tmp_path):
    """The sidecar holds ``n_tokens = n_prompts × seq_len`` but ctx.calib
    advertises a different geometry. The reader must treat this as a
    cache miss (return None, leave ctx untouched), NOT raise -- a
    misaligned operator config is not a contract violation, just a
    missed optimization."""
    # Sidecar shape: 4 prompts × 3 seq_len = 12 tokens.
    n_prompts_write, seq_len_write, hidden = 4, 3, 2
    jsonl = _write_layer_sidecars(
        tmp_path,
        layer_indices=[1, 2],
        n_prompts=n_prompts_write,
        seq_len=seq_len_write,
        hidden=hidden,
    )

    # Ctx advertises 8 × 2 = 16 tokens -- doesn't match the sidecar's 12.
    ctx = PipelineContext()
    _populate_ctx_for_alignment(ctx, n_prompts=8, seq_len=2, batch_size=2)

    provider = Stage3BlockHiddenCacheProvider()
    result = provider.on_load(ctx, jsonl)
    assert result is None
    assert not ctx.has("teacher_targets_cache")


# ---------------------------------------------------------------------------
# Test 5 -- dispatch_first routes through the registry on a hit
# ---------------------------------------------------------------------------


def test_dispatch_first_routes_through_registry(tmp_path):
    """A one-element list [Stage3BlockHiddenCacheProvider] dispatched via
    ``PluginRegistry.dispatch_first("on_load", ...)`` must return a
    non-None winner on cache hit AND populate ctx.teacher_targets_cache
    so the orchestrator's downstream block_refine call can read it."""
    from moe_compress.pipeline.registry import PluginRegistry

    n_prompts, seq_len, hidden = 6, 2, 2
    jsonl = _write_layer_sidecars(
        tmp_path,
        layer_indices=[0],
        n_prompts=n_prompts,
        seq_len=seq_len,
        hidden=hidden,
    )

    ctx = PipelineContext()
    _populate_ctx_for_alignment(
        ctx, n_prompts=n_prompts, seq_len=seq_len, batch_size=3,
    )

    plugins = [Stage3BlockHiddenCacheProvider()]
    winner = PluginRegistry.dispatch_first(
        plugins, "on_load", ctx, jsonl,
    )
    assert winner is not None
    assert ctx.has("teacher_targets_cache")
    cache = ctx.get("teacher_targets_cache")
    assert 0 in cache
    # n_batches = 6 // 3 = 2
    assert len(cache[0]) == 2
    for b in cache[0]:
        assert b.shape == (3, seq_len, hidden)
