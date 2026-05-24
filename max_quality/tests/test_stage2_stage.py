"""Tests for ``STAGE2`` — the ``Stage``-conforming Stage 2 adapter.

Covers the metadata/Protocol surface (``stage_id``, ``is_enabled``,
structural ``isinstance`` against :class:`Stage`) plus one functional test
that drives :meth:`STAGE2.run` end-to-end on the tiny-model fixture and
asserts the ``stage2_pruned_path`` ctx-slot contract. Stage 2 hard-requires
``stage1_blacklist.json`` + ``stage1_budgets.json`` in ``artifacts_dir``, so
the functional test runs Stage 1 first; the fixture setup mirrors
``test_stage2_pipeline_run_layer.py``.
"""

from __future__ import annotations

import pytest
import torch

from moe_compress import stage1
from moe_compress.budget.solver import BudgetDecomposition
from moe_compress.pipeline.context import PipelineContext
from moe_compress.pipeline.stage import Stage
from moe_compress.stage2 import STAGE2
from moe_compress.stage2 import orchestrator as stage2_reap_ream


class _TinyTokenizer:
    name_or_path = "tiny-tokenizer"
    eos_token_id = 0

    def __call__(self, text, *_, **__):
        return {"input_ids": [min(ord(c) % 32, 31) for c in (text or " ")]}

    def save_pretrained(self, *_args, **_kwargs):
        return None


def _noop_save(model, tokenizer, path, **kwargs):
    from pathlib import Path

    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


@pytest.fixture
def patched_stage2(monkeypatch, tiny_config):
    """Monkeypatch the calibration + checkpoint-save IO on the Stage 2
    orchestrator so the functional test runs fast and writes no real
    checkpoint. Copied from ``test_stage2_pipeline_run_layer.py``."""
    from moe_compress.utils import calibration as cal_mod

    def _fake_build(tokenizer, spec, cache_dir=None):
        torch.manual_seed(spec.seed)
        return torch.randint(0, 32, (spec.num_sequences, spec.sequence_length),
                             dtype=torch.long)

    def _fake_slice(tokenizer, spec, num_samples, cache_dir=None):
        torch.manual_seed(spec.seed + 1)
        return torch.randint(0, 32, (num_samples, spec.sequence_length),
                             dtype=torch.long)

    monkeypatch.setattr(cal_mod, "build_calibration_tensor", _fake_build)
    monkeypatch.setattr(cal_mod, "build_super_expert_slice", _fake_slice)
    monkeypatch.setattr(stage2_reap_ream, "build_calibration_tensor", _fake_build)

    from moe_compress.utils import model_io as mio
    monkeypatch.setattr(mio, "save_compressed_checkpoint", _noop_save)
    monkeypatch.setattr(stage2_reap_ream, "save_compressed_checkpoint", _noop_save)
    return tiny_config


def _run_stage1(model, config, tmp_path):
    tokenizer = _TinyTokenizer()
    decomp = BudgetDecomposition(
        total_reduction_ratio=0.2,
        expert_prune_ratio=0.5,
        svd_rank_ratio=0.14,
        global_expert_budget=4,
        min_experts_per_layer=2,
        blacklisted_experts={},
    )
    stage1.run(model, tokenizer, config, tmp_path, decomp)


def test_stage_id_is_two():
    assert STAGE2.stage_id == "2"


def test_is_enabled_always_true():
    assert STAGE2.is_enabled({}) is True
    assert STAGE2.is_enabled({"anything": 1}) is True


def test_stage2_conforms_to_protocol():
    assert isinstance(STAGE2, Stage)


def test_stage2_run_writes_pruned_path_slot(tiny_model, patched_stage2, tmp_path):
    """``STAGE2.run`` unwraps the ctx, runs Stage 2, and writes the
    ``stage2_pruned_path`` slot pointing at the pruned-checkpoint dir."""
    _run_stage1(tiny_model, patched_stage2, tmp_path)

    ctx = PipelineContext()
    ctx.set("model", tiny_model)
    ctx.set("tokenizer", _TinyTokenizer())
    ctx.set("config", patched_stage2)
    ctx.set("artifacts_dir", tmp_path)
    ctx.set("device", None)
    ctx.set("no_resume", True)

    result = STAGE2.run(ctx)
    assert result is None

    pruned_path = ctx.get("stage2_pruned_path")
    assert pruned_path == tmp_path / "stage2_pruned"
    assert pruned_path.is_dir()
