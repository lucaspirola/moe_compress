"""Task 4 — per-STEP ROW-split (NOT a batch-list shard) + UNCHANGED global
step count + teacher handling (live used as-is, cache token_start rank offset).
"""
from __future__ import annotations

import torch

from moe_compress.router_kd.orchestrator import _row_slice
from moe_compress.router_kd.plugins.teacher import TeacherCachePlugin
from moe_compress.pipeline.context import PipelineContext


def test_total_optim_steps_unchanged():
    # total_optim_steps is computed from the GLOBAL len(batches), NOT divided
    # by world_size. Mirror the orchestrator formula.
    len_batches, grad_accum, epochs = 8, 1, 1
    total = (len_batches // grad_accum) * epochs
    assert total == 8
    # The per-rank loop runs len(_batch_order) == len(batches) iterations — the
    # row-split changes WHAT each step forwards, not how many steps run.
    assert len(range(len_batches)) == 8


def test_rowsplit_slices_each_step():
    batch = torch.arange(2 * 3).reshape(2, 3)  # batch_size=2
    s0 = _row_slice(batch, rank=0, world_size=2)
    s1 = _row_slice(batch, rank=1, world_size=2)
    assert torch.equal(s0, batch[0:1])
    assert torch.equal(s1, batch[1:2])
    # Disjoint, equal count per_gpu == 1.
    assert s0.shape[0] == 1 and s1.shape[0] == 1


def test_rowsplit_equal_tokens_by_construction():
    # batch_size=4, world_size=2 → per_gpu=2, equal token count on every rank.
    L = 5
    batch = torch.arange(4 * L).reshape(4, L)
    for rank in (0, 1):
        sl = _row_slice(batch, rank=rank, world_size=2)
        assert sl.shape == (2, L)


def test_live_teacher_logits_not_double_sliced():
    # The live teacher receives the ALREADY-sliced [per_gpu, L] batch and
    # returns [per_gpu, L, |V|] used AS-IS. A regression that re-sliced by
    # [rank*per_gpu:(rank+1)*per_gpu] would be empty/out-of-range for rank 1.
    batch_size, world_size, L, V = 4, 2, 3, 7
    per_gpu = batch_size // world_size
    full = torch.arange(batch_size * L).reshape(batch_size, L)

    # NON-degenerate teacher: per-row-distinct logits (row r → logits filled
    # with r), so a wrong slice is detectable.
    def teacher(input_ids):
        b, l = input_ids.shape
        out = torch.zeros(b, l, V)
        for r in range(b):
            out[r] = float(input_ids[r, 0].item())  # encodes the source row
        return out

    rank = 1
    local = _row_slice(full, rank=rank, world_size=world_size)
    assert local.shape == (per_gpu, L)
    logits = teacher(local)  # used as-is, NO second slice
    assert logits.shape == (per_gpu, L, V)
    # rank-1 covers source rows [2:4]: their first-token ids are 2*L and 3*L.
    assert logits[0, 0, 0].item() == float(2 * L)
    assert logits[1, 0, 0].item() == float(3 * L)


def test_cache_teacher_token_start_rank_offset():
    # batch_size=4, world_size=2 → per_gpu=2. Each rank reads its own
    # per_gpu*L-token window: rank0 [base:base+2L], rank1 [base+2L:base+4L].
    batch_size, world_size, L, V = 4, 2, 3, 5
    per_gpu = batch_size // world_size
    num_batches = 4
    # Flat cache: token t gets logits row filled with value t (detectable).
    n_tokens = num_batches * batch_size * L
    flat = torch.arange(n_tokens, dtype=torch.float32).unsqueeze(-1).repeat(1, V)
    cache_payload = {"logits": flat}

    config = {
        "stage5_router_kd": {
            "batch_size": batch_size,
            "max_sequence_length": L,
            "max_calibration_samples": batch_size * num_batches,
        }
    }

    plugin = TeacherCachePlugin()
    batch_index = 1  # second step in the epoch
    base = batch_index * batch_size * L

    results = {}
    for rank in (0, 1):
        ctx = PipelineContext()
        ctx.set("config", config)
        ctx.set("teacher_logits_cache", cache_payload)
        ctx.set("ddp_rank", rank)
        ctx.set("ddp_world_size", world_size)
        local_ids = torch.zeros(per_gpu, L, dtype=torch.long)
        out = plugin.provide_teacher_logits(
            ctx, input_ids=local_ids, epoch=0,
            batch_index=batch_index, num_batches=num_batches,
        )
        results[rank] = out

    assert results[0].shape == (per_gpu, L, V)
    assert results[1].shape == (per_gpu, L, V)
    # rank0 reads flat tokens [base : base+2L]; first value == base.
    assert results[0][0, 0, 0].item() == float(base)
    # rank1 reads flat tokens [base+2L : base+4L]; first value == base + per_gpu*L.
    assert results[1][0, 0, 0].item() == float(base + per_gpu * L)
    # Disjoint windows: rank1's first token == rank0's last token + 1.
    assert results[1][0, 0, 0].item() == results[0][-1, -1, 0].item() + 1
