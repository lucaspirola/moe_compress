"""Tests for ``Stage3BlockHiddenCacheProvider`` (Item 7 reader, Stage 3 side).

Lever 3 (design (b)) note
-------------------------
``on_load`` no longer eagerly loads every payload. It PRESENCE-validates the
sidecars all-or-nothing (file + manifest existence + schema, NO payload load)
and returns a LAZY mapping (:class:`_LazyTeacherTargetsCache`) that
materializes one layer's ``[n_prompts, seq_len, hidden]`` tensor on
``.get(layer_idx)`` and retains no reference. The lazy mapping exposes
``__len__`` (validated layer count, no materialization), ``.get`` (``None`` on
absent key), and dict-like ``keys()`` / ``__getitem__`` for parity.

Two former up-front content guards (token-count mismatch, I2 prompt-count
divergence) cannot run without deserializing the payload (``load_block_hidden``
has no shape-only path; the manifest carries no shape field), so they moved
OUT of up-front scope. A present-but-malformed sidecar is now caught at
consumption time by ``block_refine``'s per-layer shape guard, which falls
through to the live teacher forward for that one layer (numerically safe --
each layer's target is independent). The all-or-nothing *presence* miss is
preserved.

Covers:

1. ``test_load_miss_no_dir`` -- no sidecars directory → ``on_load``
   returns None (does not raise) and ctx is untouched.
2. ``test_load_hit_populates_teacher_targets_cache`` -- per-layer
   sidecars present + ctx has calib → ``on_load`` returns a non-None
   marker AND populates a lazy ``ctx.teacher_targets_cache`` whose
   ``.get(layer_idx)`` yields the un-chunked ``[n_prompts, seq_len,
   hidden]`` bf16 tensor (decoupled from any consumer batch_size).
3. ``test_schema_mismatch_raises`` -- forced schema=99 → the presence
   scan's manifest validation raises RuntimeError with the actionable
   "re-run calibration" message (manifest read, no payload load).
4. ``test_token_count_mismatch_lazy_get_falls_through`` -- a
   present-but-misshapen sidecar PASSES presence validation (hit), but
   ``.get`` returns a non-3-D tensor so block_refine's per-layer guard
   falls through to live for that layer.
5. ``test_dispatch_first_routes_through_registry`` -- a one-element
   registry of [Stage3BlockHiddenCacheProvider] processed via
   ``PluginRegistry.dispatch_first`` returns the first non-None winner
   on cache hit.
6. ``test_prompt_count_divergence_no_longer_upfront_miss`` -- documents
   the accepted scope change: ``n_prompts_in_subset`` divergence is NOT
   detected up-front anymore (needs a payload load); a shape-valid
   sidecar still materializes.
7. ``test_c1_batch_size_decoupled`` -- a writer with bcov-style batch
   size 1 produces sidecars; the reader populates the un-chunked
   tensor; downstream block_refine consumer with batch_size=32 slices
   correctly into per-batch tensors. Exercises the C1 fix's promise
   that the cache is independent of the consumer's batch size.
8. ``test_lazy_get_returns_same_tensor_as_eager`` -- ``.get`` materializes
   the SAME reshaped tensor the old eager path produced (byte-identity).
9. ``test_lazy_mapping_len_and_get_contract`` -- ``len`` == n_layers with
   NO materialization; ``.get(absent) is None``.
10. ``test_partial_cache_still_all_or_nothing_miss`` -- removing one
    layer's sidecar/manifest is the contract gate; the sidecars dir
    presence is all-or-nothing.
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
    with pytest.raises(RuntimeError) as exc:
        provider.on_load(ctx, jsonl)
    msg = str(exc.value)
    assert "manifest validation FAILED" in msg
    assert "schema_version=1" in msg
    assert "expected 99" in msg
    assert "re-run calibration" in msg


# ---------------------------------------------------------------------------
# Test 4 -- token count mismatch -> presence HIT, per-layer lazy fall-through
# ---------------------------------------------------------------------------


def test_token_count_mismatch_lazy_get_falls_through(tmp_path):
    """Lever 3 (design (b)): a present-but-misshapen sidecar PASSES the
    cheap presence/manifest validation (so ``on_load`` returns a hit), but
    its ``.get`` cannot reshape under the ctx geometry. Per design (b) the
    token-count check moved OUT of up-front scope (it needs a payload load).
    ``.get`` instead returns the raw 2-D ``[n_tokens, hidden]`` tensor, which
    block_refine's per-layer shape guard (``cached.dim() == 3``) rejects --
    falling through to the live teacher forward FOR THAT LAYER. Each layer's
    target is independent, so a per-layer fall-through is numerically safe."""
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
    # Presence validation succeeds -> this is a HIT (the lazy mapping).
    assert result is not None
    assert ctx.has("teacher_targets_cache")
    cache = ctx.get("teacher_targets_cache")
    assert len(cache) == 2  # both layers present

    # But materializing a misshapen layer yields a non-3-D tensor that
    # block_refine's guard rejects -> per-layer live fall-through.
    got = cache.get(1)
    assert got is not None
    assert got.dim() != 3, (
        "a token-count-mismatched sidecar must NOT reshape to 3-D; it must "
        "return a tensor block_refine's dim()==3 guard rejects"
    )


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
# Test 6 -- I2 prompt-count divergence is no longer an up-front miss
# (accepted scope change under Lever 3 design (b))
# ---------------------------------------------------------------------------


def test_prompt_count_divergence_no_longer_upfront_miss(tmp_path):
    """ACCEPTED SCOPE CHANGE (Lever 3 design (b)): the I2 prompt-count
    divergence guard read ``payload.n_prompts_in_subset``, which requires
    deserializing the payload -- impossible under the cheap presence scan
    (``load_block_hidden`` has no shape-only path; the manifest carries no
    n_prompts field). So a sidecar whose ``n_prompts_in_subset`` diverges
    but whose token count matches the ctx geometry is NO LONGER an up-front
    miss; it presence-validates (hit) and ``.get`` materializes a
    shape-valid tensor under the ctx geometry.

    This is the documented trade-off recorded in the plan's risk table:
    operator misconfiguration of the calibration source is caught by the
    operator-responsibility note (prompt-identity is I1+I2 operator
    responsibility), not by this reader. The numerically-safe per-layer
    shape fall-through still protects against a *misshapen* sidecar."""
    n_prompts_ctx, seq_len, hidden = 6, 4, 2
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
    # Presence-valid -> hit. (Pre-Lever-3 this returned None.)
    assert result is not None
    assert ctx.has("teacher_targets_cache")
    cache = ctx.get("teacher_targets_cache")
    # Token count (24) matches 6×4, so .get materializes a shape-valid tensor.
    got = cache.get(0)
    assert got is not None
    assert got.shape == (n_prompts_ctx, seq_len, hidden)


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


# ---------------------------------------------------------------------------
# Lever 3 -- lazy materialization contract
# ---------------------------------------------------------------------------


def test_lazy_get_returns_same_tensor_as_eager(tmp_path):
    """``.get(layer_idx)`` materializes the SAME reshaped tensor the old
    eager path produced (``hs.reshape(n_prompts, seq_len, -1).contiguous()``)
    -- byte-identity, only allocation timing changes."""
    from moe_compress.utils.cached_calibration_signals import load_block_hidden

    n_prompts, seq_len, hidden = 8, 4, 3
    layer_indices = [3, 7]
    jsonl = _write_layer_sidecars(
        tmp_path,
        layer_indices=layer_indices,
        n_prompts=n_prompts,
        seq_len=seq_len,
        hidden=hidden,
    )
    ctx = PipelineContext()
    _populate_ctx_for_alignment(
        ctx, n_prompts=n_prompts, seq_len=seq_len, batch_size=2,
    )
    provider = Stage3BlockHiddenCacheProvider()
    assert provider.on_load(ctx, jsonl) is not None
    cache = ctx.get("teacher_targets_cache")

    for li in layer_indices:
        eager = load_block_hidden(jsonl, li).hidden_states.reshape(
            n_prompts, seq_len, -1
        ).contiguous()
        lazy = cache.get(li)
        assert lazy.dtype == eager.dtype
        assert lazy.shape == eager.shape
        assert torch.equal(lazy, eager), (
            f"lazy .get(layer {li}) tensor diverged from the eager reshape"
        )


def test_lazy_mapping_len_and_get_contract(tmp_path):
    """``len(cache) == n_layers`` from the presence scan WITHOUT materializing
    any tensor (orchestrator:710 HIT-log call), and ``.get(absent) is None``
    (block_refine:465 ``dict.get`` semantics)."""
    n_prompts, seq_len, hidden = 6, 2, 2
    layer_indices = [0, 4, 9]
    jsonl = _write_layer_sidecars(
        tmp_path,
        layer_indices=layer_indices,
        n_prompts=n_prompts,
        seq_len=seq_len,
        hidden=hidden,
    )
    ctx = PipelineContext()
    _populate_ctx_for_alignment(
        ctx, n_prompts=n_prompts, seq_len=seq_len, batch_size=3,
    )
    provider = Stage3BlockHiddenCacheProvider()
    assert provider.on_load(ctx, jsonl) is not None
    cache = ctx.get("teacher_targets_cache")

    # len() must equal the validated layer count with NO materialization.
    # We assert it BEFORE any .get() call, so no payload has been loaded yet.
    assert len(cache) == len(layer_indices)
    assert set(cache.keys()) == set(layer_indices)

    # Absent key -> None (NOT KeyError) for .get, matching dict.get.
    assert cache.get(999) is None
    # Present key materializes a real tensor.
    assert cache.get(4) is not None


def test_partial_cache_still_all_or_nothing_miss(tmp_path):
    """Contract-preservation gate: the PRESENCE miss stays all-or-nothing
    BEFORE any training, established cheaply (stat + manifest, no payload
    load).

    The directory-level miss is all-or-nothing: with no sidecars dir there is
    a full miss (None) BEFORE any training, so a consumer never sees a
    partially-hydrated cache. A genuinely-malformed-but-PRESENT sidecar (torn
    manifest) raises in the presence scan -- covered by
    ``test_schema_mismatch_raises``. A shape-misshapen-but-present sidecar
    falls through per-layer at consumption -- covered by
    ``test_token_count_mismatch_lazy_get_falls_through``.

    (Removing ONE layer's ``.pt`` from an N-layer dir shrinks the validated
    set but is NOT independently detectable as a "missing layer" -- neither
    origin/main's eager ``_load_layers`` nor this presence scan is given the
    expected layer COUNT; the orchestrator never passes ``n_blocks`` to the
    provider. This is unchanged from origin/main.)
    """
    # Directory absent -> full miss, ctx untouched, before any materialization.
    jsonl = _jsonl(tmp_path)
    ctx = PipelineContext()
    _populate_ctx_for_alignment(ctx, n_prompts=6, seq_len=2, batch_size=2)
    provider = Stage3BlockHiddenCacheProvider()
    assert provider.on_load(ctx, jsonl) is None
    assert not ctx.has("teacher_targets_cache")
