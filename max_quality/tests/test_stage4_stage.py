"""Tests for ``STAGE4`` — the ``Stage``-conforming Stage 4 adapter.

Covers the metadata/Protocol surface (``stage_id``, ``is_enabled``,
structural ``isinstance`` against :class:`Stage`) plus one functional test
that drives :meth:`STAGE4.run` end-to-end on the tiny-model fixture and
asserts the ``stage4_eora_path`` ctx-slot contract. Stage 4 hard-requires the
Stage 1 + Stage 2 + Stage 3 artifacts in ``artifacts_dir``, so the functional
test runs Stages 1→2→3 first; the fixture setup mirrors
``test_stage4_golden_snapshot.py``.

Helpers (``_TinyTokenizer``, ``_noop_save``, ``_run_stages_0123``) are
redeclared locally on purpose — tests in this codebase do not import from
each other (mirrors ``test_stage3_stage.py``'s discipline).
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
from moe_compress.stage4 import STAGE4


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
def patched_stage4(monkeypatch, tiny_config):
    """Patch the stage-2/3 calibration loaders and the checkpoint saver so the
    functional test runs fast and writes no real checkpoint. Mirrors the
    ``patched_stage4`` fixture in ``test_stage4_golden_snapshot.py`` (fp32 case).

    The Stage 4 orchestrator calls ``save_compressed_checkpoint`` module-
    qualified through ``utils.model_io``, so the ``model_io`` patch covers it.
    """
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


def _run_stages_0123(model, config, tmp_path):
    """Run Stages 1→2→3 to get a post-SVD model + the sidecars Stage 4 needs.

    Stage 4 reads ``_stage3_original_weights.pt``, the stage-3-factored model
    and ``_stage2_input_covariance.pt`` — so this must complete before
    ``STAGE4.run`` is invoked. Returns the ``BudgetDecomposition`` (consumed
    internally by Stages 1/3; unused by Stage 4, which takes no decomposition
    argument — returned only for API symmetry).
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
    stage2_reap_ream.run(model, _TinyTokenizer(), config, tmp_path, device=None)
    stage3_svd.run(model, _TinyTokenizer(), config, tmp_path, decomp, device=None)
    return decomp


def test_stage_id_is_four():
    assert STAGE4.stage_id == "4"


def test_is_enabled_always_true():
    assert STAGE4.is_enabled({}) is True
    assert STAGE4.is_enabled({"anything": 1}) is True


def test_stage4_conforms_to_protocol():
    assert isinstance(STAGE4, Stage)


def test_stage4_run_writes_eora_path_slot(tiny_model, patched_stage4, tmp_path):
    """``STAGE4.run`` unwraps the ctx, runs Stage 4, and writes the
    ``stage4_eora_path`` slot pointing at the EoRA checkpoint dir."""
    _run_stages_0123(tiny_model, patched_stage4, tmp_path)

    ctx = PipelineContext()
    ctx.set("model", tiny_model)
    ctx.set("tokenizer", _TinyTokenizer())
    ctx.set("config", patched_stage4)
    ctx.set("artifacts_dir", tmp_path)
    ctx.set("no_resume", True)

    result = STAGE4.run(ctx)
    assert result is None

    eora_path = ctx.get("stage4_eora_path")
    assert eora_path == tmp_path / "stage4_eora"
    assert eora_path.is_dir()
