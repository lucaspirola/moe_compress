"""Task 4 — gen replica worker + spawn driver (run_dp_generate).

Tests the spawn + group-aligned-split + index-ordered-merge PLUMBING with a
deterministic stub generate (NOT a real model). The worker is run via real
``torch.multiprocessing`` spawn with ``CUDA_VISIBLE_DEVICES=""`` so
``device_count()==0`` and everything runs on CPU.

The stub generate is injected through a picklable, module-level
``_STUB_GENERATE_KEY`` registry so it survives the spawn boundary.
"""
from __future__ import annotations

import pytest

from moe_compress.tools import eval_shard
from moe_compress.tools.eval_shard import run_dp_generate


def _stub_completion(prompt: str) -> str:
    # deterministic function of the prompt
    return f"<{prompt}|len{len(prompt)}>"


def test_run_dp_generate_merges_in_order(tmp_path):
    prompts = [f"p{i}" for i in range(20)]
    expected = [_stub_completion(p) for p in prompts]

    merged = run_dp_generate(
        prompts,
        tmp_dir=str(tmp_path / "src"),
        replicas=2,
        gpus_per_replica=1,
        max_new=4,
        experts_impl_generative="batched_mm",
        cfg=None,
        out_dir=str(tmp_path / "out"),
        _stub_generate=eval_shard._STUB_GENERATE_PROMPTLEN,
    )
    assert merged == expected


def test_run_dp_generate_single_replica_inprocess(tmp_path):
    # replicas<=1 must NOT spawn — falls through to one shard.
    prompts = [f"p{i}" for i in range(12)]
    merged = run_dp_generate(
        prompts,
        tmp_dir=str(tmp_path / "src"),
        replicas=1,
        gpus_per_replica=1,
        max_new=4,
        experts_impl_generative="batched_mm",
        cfg=None,
        out_dir=str(tmp_path / "out"),
        _stub_generate=eval_shard._STUB_GENERATE_PROMPTLEN,
    )
    assert merged == [_stub_completion(p) for p in prompts]


def test_run_dp_generate_three_replicas_short_tail(tmp_path):
    # 20 prompts = 3 groups (8,8,4); 3 replicas -> one group each, short tail whole.
    prompts = [f"x{i}" for i in range(20)]
    merged = run_dp_generate(
        prompts,
        tmp_dir=str(tmp_path / "src"),
        replicas=3,
        gpus_per_replica=1,
        max_new=4,
        experts_impl_generative="batched_mm",
        cfg=None,
        out_dir=str(tmp_path / "out"),
        _stub_generate=eval_shard._STUB_GENERATE_PROMPTLEN,
    )
    assert merged == [_stub_completion(p) for p in prompts]
