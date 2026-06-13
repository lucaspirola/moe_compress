"""Task 8 — GOLDEN GATE: 1-GPU completions == concat of 2-shard completions.

The crux. A gen completion is a pure function of its group-of-8 PADDED WIDTH
(left-pad geometry). The width-derived stub model makes this provable: its
generate() emits a token sequence encoding ``w = input_ids.shape[1]`` (exactly
the real ``input_len`` at eval_harness.py:150), decoded to ``W{w}``. Prompts of
deliberately varied lengths produce DIFFERENT group-of-8 max-widths under a
mod-8 grouping vs a non-mod-8 grouping, so:

  * Positive gate: mod-8 sharded == single-GPU golden (byte-identical).
  * Negative control (MANDATORY, non-vacuous): a non-mod-8 split provably
    changes >=1 completion -> assert naive_sharded != golden.

The stub model + tokenizer are reconstructed inside the spawn worker via the
``_STUB_WIDTH_MODEL`` seam so the REAL ``_generate_batched`` (left-pad +
pad_to_multiple_of=64 grouping) runs on each shard.
"""
from __future__ import annotations

import math

import pytest
import torch

from moe_compress.tools import eval_shard
from moe_compress.tools.eval_harness import _generate_batched, PINNED_GEN_BATCH_SIZE
from moe_compress.tools.eval_shard import (
    _merge_completions,
    _merge_ppl,
    run_dp_generate,
    run_dp_ppl,
)
from moe_compress.tools.eval_shard_stub import (
    build_width_stub_model,
    build_width_stub_tokenizer,
)


def _varied_prompts(n=20):
    # Lengths chosen so mod-8 grouping vs a [0:10),[10:20) split give different
    # per-group max widths (after pad_to_multiple_of=64) for at least one prompt.
    out = []
    for i in range(n):
        # length grows across the list with a jump at index 10 so the two
        # groupings see different maxima.
        base = 10 + i * 5
        if i >= 10:
            base += 300  # force a different 64-bucket beyond the mod-8 boundary
        out.append("x" * base)
    return out


def test_golden_gate_single_vs_2shard_byte_identical(tmp_path):
    prompts = _varied_prompts(20)
    model = build_width_stub_model()
    tok = build_width_stub_tokenizer()

    golden = _generate_batched(
        model, tok, prompts, max_new=4, device=None,
        batch_size=PINNED_GEN_BATCH_SIZE,
    )

    sharded = run_dp_generate(
        prompts,
        tmp_dir=str(tmp_path / "src"),
        replicas=2,
        gpus_per_replica=1,
        max_new=4,
        experts_impl_generative="batched_mm",
        cfg=None,
        out_dir=str(tmp_path / "out"),
        _stub_model=eval_shard._STUB_WIDTH_MODEL,
    )
    # Positive gate (binding).
    assert sharded == golden


def test_golden_gate_negative_control_non_mod8_differs(tmp_path):
    prompts = _varied_prompts(20)
    model = build_width_stub_model()
    tok = build_width_stub_tokenizer()
    golden = _generate_batched(
        model, tok, prompts, max_new=4, device=None,
        batch_size=PINNED_GEN_BATCH_SIZE,
    )

    # Force a NAIVE non-mod-8 split [(0,10),(10,20)] and run the SAME stub.
    naive = run_dp_generate(
        prompts,
        tmp_dir=str(tmp_path / "src"),
        replicas=2,
        gpus_per_replica=1,
        max_new=4,
        experts_impl_generative="batched_mm",
        cfg=None,
        out_dir=str(tmp_path / "out"),
        _stub_model=eval_shard._STUB_WIDTH_MODEL,
        _force_bounds=[(0, 10), (10, 20)],
    )
    # Non-vacuous by construction: the regrouped max-widths differ -> >=1 flip.
    assert naive != golden


def test_golden_gate_ppl_exact(tmp_path):
    torch.manual_seed(3)
    chunks = torch.randint(0, 50, (12, 8), dtype=torch.long)
    # single-pass deterministic stub == the run_dp_ppl stub
    numel, n_rows = chunks.numel(), chunks.shape[0]
    per_tok = (chunks.to(torch.float64) % 7).sum() / (numel - n_rows)
    nll = float(per_tok) * (numel - n_rows)
    single = math.exp(nll / (numel - n_rows))

    merged = run_dp_ppl(
        chunks, tmp_dir=str(tmp_path / "src"), replicas=2, gpus_per_replica=1,
        ppl_bs=1, experts_impl_generative="batched_mm", cfg=None,
        out_dir=str(tmp_path / "out"), _stub_ppl=eval_shard._STUB_PPL_MOD7,
    )
    assert math.isclose(merged, single, rel_tol=0.0, abs_tol=0.0)
