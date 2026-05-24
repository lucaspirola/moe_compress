"""Tests for ``STAGE3`` — the ``Stage``-conforming Stage 3 adapter.

Covers the metadata/Protocol surface (``stage_id``, ``is_enabled``,
structural ``isinstance`` against :class:`Stage`) plus one functional test
that drives :meth:`STAGE3.run` end-to-end on the tiny-model fixture and
asserts the ``stage3_svd_path`` ctx-slot contract. Stage 3 hard-requires the
Stage 1 + Stage 2 artifacts in ``artifacts_dir``, so the functional test
runs Stages 1→2 first; the fixture setup mirrors
``test_stage3_golden_snapshot.py``.

Helpers (``_TinyTokenizer``, ``_noop_save``, ``_run_stages_1_2``) are
redeclared locally on purpose — tests in this codebase do not import from
each other (mirrors ``test_stage2_stage.py``'s discipline).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from moe_compress import stage1, stage3_svd
from moe_compress.budget.solver import BudgetDecomposition
from moe_compress.pipeline.context import PipelineContext
from moe_compress.pipeline.stage import Stage
from moe_compress.stage2 import orchestrator as stage2_reap_ream
from moe_compress.stage3 import STAGE3


class _TinyTokenizer:
    name_or_path = "tiny-tokenizer"
    eos_token_id = 0

    def __call__(self, text, *_, **__):
        return {"input_ids": [min(ord(c) % 32, 31) for c in (text or " ")]}

    def save_pretrained(self, *_args, **_kwargs):
        return None


def _noop_save(model, tokenizer, path, **kwargs):
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


@pytest.fixture
def patched_stage3(monkeypatch, tiny_config):
    """Patch the stage-2 + stage-3 calibration loaders and the checkpoint
    saver so the functional test runs fast and writes no real checkpoint.
    Mirrors the ``patched_stage3`` fixture in ``test_stage3_golden_snapshot.py``
    (fp32 case).

    ``load_model`` is intentionally NOT patched: ``tiny_config`` sets
    ``cross_covariance: False`` and ``block_refine.enabled: False``, so the
    Stage 3 orchestrator never enters the teacher-load branch. A config that
    enables either would additionally need a ``load_model`` patch here."""
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
    monkeypatch.setattr(stage3_svd, "build_calibration_tensor", _fake_build)

    from moe_compress.utils import model_io as mio
    monkeypatch.setattr(mio, "save_compressed_checkpoint", _noop_save)
    monkeypatch.setattr(stage2_reap_ream, "save_compressed_checkpoint", _noop_save)
    monkeypatch.setattr(stage3_svd, "save_compressed_checkpoint", _noop_save)

    return tiny_config


def _run_stages_1_2(model, config, tmp_path):
    """Run Stages 1→2 to get a post-prune model + Stage 2 covariance artifact.

    Returns the ``BudgetDecomposition`` that Stage 3 consumes.
    """
    decomp = BudgetDecomposition(
        total_reduction_ratio=0.2,
        expert_prune_ratio=0.5,
        svd_rank_ratio=0.14,
        global_expert_budget=4,
        min_experts_per_layer=2,
        blacklisted_experts={},
    )
    stage1.run(model, _TinyTokenizer(), config, tmp_path, decomp)
    stage2_reap_ream.run(
        model, _TinyTokenizer(), config, tmp_path, device=None,
    )
    return decomp


def test_stage_id_is_three():
    assert STAGE3.stage_id == "3"


def test_is_enabled_always_true():
    assert STAGE3.is_enabled({}) is True
    assert STAGE3.is_enabled({"anything": 1}) is True


def test_stage3_conforms_to_protocol():
    assert isinstance(STAGE3, Stage)


def test_stage3_run_writes_svd_path_slot(tiny_model, patched_stage3, tmp_path):
    """``STAGE3.run`` unwraps the ctx, runs Stage 3, and writes the
    ``stage3_svd_path`` slot pointing at the SVD checkpoint dir."""
    decomp = _run_stages_1_2(tiny_model, patched_stage3, tmp_path)

    ctx = PipelineContext()
    ctx.set("model", tiny_model)
    ctx.set("tokenizer", _TinyTokenizer())
    ctx.set("config", patched_stage3)
    ctx.set("artifacts_dir", tmp_path)
    ctx.set("decomposition", decomp)
    ctx.set("device", None)
    ctx.set("no_resume", True)

    result = STAGE3.run(ctx)
    assert result is None

    svd_path = ctx.get("stage3_svd_path")
    assert svd_path == tmp_path / "stage3_svd"
    assert svd_path.is_dir()
