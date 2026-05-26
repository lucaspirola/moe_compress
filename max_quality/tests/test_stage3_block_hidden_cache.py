"""Tests for ``Stage3BlockHiddenCacheProvider`` (Item 7 reader, Stage 3 side).

Covers:

1. ``test_load_miss_no_dir`` -- no sidecars directory → ``on_load``
   returns None (does not raise) and ctx is untouched.
2. ``test_load_hit_populates_teacher_targets_cache`` -- per-layer
   sidecars present + ctx has calib → ``on_load`` returns a
   non-None payload AND populates
   ``ctx.teacher_targets_cache: dict[int, Tensor]`` with un-chunked
   ``[n_prompts, seq_len, hidden]`` bf16 tensors per layer (decoupled
   from any consumer batch_size -- C1 fix).
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
6. ``test_prompt_count_divergence_falls_through_to_miss`` -- sidecar's
   ``n_prompts_in_subset`` does not match ``ctx.calib.shape[0]`` →
   ``on_load`` returns None (I2 prompt-count divergence guard).
7. ``test_c1_batch_size_decoupled`` -- a writer with bcov-style batch
   size 1 produces sidecars; the reader populates the un-chunked
   tensor; downstream block_refine consumer with batch_size=32 slices
   correctly into per-batch tensors. Exercises the C1 fix's promise
   that the cache is independent of the consumer's batch size.
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
    """Populate ctx.calib (+ ctx.batches for historical parity) the way
    the orchestrator does.

    The reader only reads ``ctx.calib`` after the C1 fix (it stores an
    un-chunked tensor and the block_refine consumer slices per batch
    itself); ``ctx.batches`` is set anyway so the helper matches the
    real orchestrator's run_ctx population.
    """
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
    # Post-C1: each entry is a SINGLE un-chunked tensor of shape
    # [n_prompts, seq_len, hidden]; the consumer slices per batch.
    for li in layer_indices:
        tensor = cache[li]
        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == torch.bfloat16
        assert tensor.shape == (n_prompts, seq_len, hidden)
        assert tensor.device.type == "cpu"

    # Spot-check deterministic content. The writer helper assigned
    # token-flat index t in [0, n_prompts*seq_len) to value t at column
    # 1, and the layer index at column 0. After the reader's reshape
    # [n_prompts*seq_len, hidden] -> [n_prompts, seq_len, hidden] the
    # cell (prompt=p, pos=q, channel=1) carries (p * seq_len + q).
    # layer 3, prompt 0, pos 0 -> t=0 -> column 1 = 0.0; layer id = 3.
    t3 = cache[3]
    assert t3[0, 0, 0].item() == 3.0
    assert t3[0, 0, 1].item() == 0.0
    # layer 7, prompt 3, pos 0 -> t = 3 * seq_len + 0 = 12.
    t7 = cache[7]
    assert t7[3, 0, 0].item() == 7.0
    assert t7[3, 0, 1].item() == 12.0


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
    # Post-C1: un-chunked [n_prompts, seq_len, hidden] per layer.
    tensor = cache[0]
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (n_prompts, seq_len, hidden)


# ---------------------------------------------------------------------------
# Test 6 -- prompt-count divergence guard (I2)
# ---------------------------------------------------------------------------


def test_prompt_count_divergence_falls_through_to_miss(tmp_path):
    """The sidecar's ``n_prompts_in_subset`` does not match
    ``ctx.calib.shape[0]``. Even if the per-layer ``n_tokens`` happens
    to match (it does here because the writer chose to over-collect by
    a chunk-boundary remainder), the reader must treat this as a cache
    miss because the per-prompt content is no longer guaranteed to
    align with the consumer's calibration tensor."""
    n_prompts_ctx, seq_len, hidden = 6, 4, 2
    # Build a payload whose hidden_states has the right SHAPE (6×4=24
    # rows) but whose n_prompts_in_subset advertises 130 -- e.g. a
    # writer that captured beyond the 128-cap on a chunk boundary AND
    # the Stage 3 calib spec asks for a totally different 6-prompt
    # tensor. The token-count math would coincidentally check out only
    # if the per-prompt seq_len also lined up; here the divergence
    # check fires before the token-count check.
    jsonl = _jsonl(tmp_path)
    hs = torch.zeros(n_prompts_ctx * seq_len, hidden, dtype=torch.bfloat16)
    payload = BlockHiddenPayload(
        schema_version=SCHEMA_VERSIONS["block_hidden"],
        layer_idx=0,
        n_prompts_in_subset=130,  # diverges from ctx n_prompts=6
        hidden_states=hs,
    )
    save_block_hidden(payload, jsonl)

    ctx = PipelineContext()
    _populate_ctx_for_alignment(
        ctx, n_prompts=n_prompts_ctx, seq_len=seq_len, batch_size=2,
    )

    provider = Stage3BlockHiddenCacheProvider()
    result = provider.on_load(ctx, jsonl)
    assert result is None
    assert not ctx.has("teacher_targets_cache")


# ---------------------------------------------------------------------------
# Test 7 -- C1 fix: cache populates correctly regardless of consumer
# batch size; downstream block_refine slicing yields the expected
# per-batch tensors. Exercises the exact mismatch scenario from the
# C1 finding: writer with bcov-style batch_size=1, reader called from
# a block_refine context with batch_size=32 (here scaled to 4 vs 1).
# ---------------------------------------------------------------------------


def test_c1_batch_size_decoupled(tmp_path):
    """Writer pipeline carves ``ctx.batches`` with ``bcov_batch_size=1``
    (Stage 3 default for the cross-covariance pass). The block_refine
    consumer uses its own ``batch_size`` (default 32, scaled to 4 in
    the test for unit-test budget). Pre-C1 the reader would have
    chunked the per-layer cache into per-batch tensors of shape
    ``[1, seq_len, hidden]`` and the block_refine consumer's per-batch
    shape check ``(4, seq_len, hidden)`` would have rejected EVERY
    cached entry, falling through to the live forward and silently
    defeating the cache. Post-C1 the cache stores the un-chunked
    ``[n_prompts, seq_len, hidden]`` tensor and the consumer slices
    per-batch from it -- so the per-batch tensors carry the
    consumer's batch_size regardless of the writer-side chunking."""
    n_prompts, seq_len, hidden = 8, 2, 3
    bcov_batch_size = 1            # writer-side / cross-cov pass
    consumer_batch_size = 4        # block_refine.batch_size (scaled-down 32)

    jsonl = _write_layer_sidecars(
        tmp_path,
        layer_indices=[5],
        n_prompts=n_prompts,
        seq_len=seq_len,
        hidden=hidden,
    )

    ctx = PipelineContext()
    # Populate ctx.batches with bcov_batch_size=1 (mirrors the live
    # orchestrator: the run_ctx carries the bcov_batch_size carve,
    # NOT the block_refine.batch_size). Post-C1 the reader IGNORES
    # ctx.batches; the test still sets it so the regression scenario
    # is faithful.
    _populate_ctx_for_alignment(
        ctx, n_prompts=n_prompts, seq_len=seq_len,
        batch_size=bcov_batch_size,
    )
    # Sanity: ctx.batches[0].shape[0] is 1 (the pre-C1 reader would
    # have chunked the per-layer tensor with this batch_size).
    assert ctx.get("batches")[0].shape[0] == bcov_batch_size

    provider = Stage3BlockHiddenCacheProvider()
    result = provider.on_load(ctx, jsonl)
    assert result is not None
    cache = ctx.get("teacher_targets_cache")
    assert 5 in cache
    cached_tensor = cache[5]
    # The cache holds the un-chunked [n_prompts, seq_len, hidden]
    # tensor REGARDLESS of either bcov or consumer batch size.
    assert cached_tensor.shape == (n_prompts, seq_len, hidden)

    # Simulate the block_refine consumer's per-batch slicing. The
    # consumer uses its OWN batch_size (consumer_batch_size=4 here),
    # NOT ctx.batches[0].shape[0]=1. The slice must produce per-batch
    # tensors of shape (consumer_batch_size, seq_len, hidden).
    n_batches = n_prompts // consumer_batch_size  # = 2
    per_batch_targets = [
        cached_tensor[bi * consumer_batch_size:(bi + 1) * consumer_batch_size]
        for bi in range(n_batches)
    ]
    assert len(per_batch_targets) == n_batches
    for tb in per_batch_targets:
        assert tb.shape == (consumer_batch_size, seq_len, hidden)
        assert tb.dtype == torch.bfloat16
    # Spot-check content alignment: layer 5, batch 0 covers prompts
    # 0..3; batch 0 row 0 == cached_tensor[0, 0, :]. The writer
    # helper sets column 0 = layer_id (5) and column 1 = t-index.
    assert per_batch_targets[0][0, 0, 0].item() == 5.0
    assert per_batch_targets[0][0, 0, 1].item() == 0.0
    # batch 1 row 0 covers prompt 4; t = 4 * seq_len + 0 = 8.
    assert per_batch_targets[1][0, 0, 1].item() == 8.0
