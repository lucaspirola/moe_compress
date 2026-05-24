"""Tests for ``STAGE1`` — the ``Stage``-conforming Stage 1 adapter.

Covers the metadata/Protocol surface (``stage_id``, ``is_enabled``,
structural ``isinstance`` against :class:`Stage`) plus one functional test
that drives :meth:`STAGE1.run` end-to-end on the tiny-model fixture and
asserts the ``stage1_blacklist_path`` / ``stage1_budgets_path`` ctx-slot
contract. The functional fixture setup mirrors ``test_stage1_e2e.py``.
"""

from __future__ import annotations

from moe_compress.budget.solver import BudgetDecomposition
from moe_compress.pipeline.context import PipelineContext
from moe_compress.pipeline.stage import Stage
from moe_compress.stage1 import STAGE1


class _TinyTokenizer:
    name_or_path = "tiny-tokenizer"
    eos_token_id = 0

    def __call__(self, text, *_, **__):
        return {"input_ids": [min(ord(c) % 32, 31) for c in (text or " ")]}

    def save_pretrained(self, *_args, **_kwargs):
        return None


def test_stage_id_is_one():
    assert STAGE1.stage_id == "1"


def test_is_enabled_always_true():
    assert STAGE1.is_enabled({}) is True
    assert STAGE1.is_enabled({"anything": 1}) is True


def test_stage1_conforms_to_protocol():
    assert isinstance(STAGE1, Stage)


def test_stage1_run_writes_output_path_slots(tiny_model, tiny_config, tmp_path):
    """``STAGE1.run`` unwraps the ctx, runs Stage 1, and writes the two
    ``stage1_*_path`` slots pointing at the on-disk artifacts."""
    decomp = BudgetDecomposition(
        total_reduction_ratio=0.20,
        expert_prune_ratio=0.25,
        svd_rank_ratio=0.0,
        global_expert_budget=5,
        min_experts_per_layer=2,
        blacklisted_experts={},
    )

    ctx = PipelineContext()
    ctx.set("model", tiny_model)
    ctx.set("tokenizer", _TinyTokenizer())
    ctx.set("config", tiny_config)
    ctx.set("artifacts_dir", tmp_path)
    ctx.set("decomposition", decomp)
    ctx.set("device", None)

    result = STAGE1.run(ctx)
    assert result is None

    blacklist_path = ctx.get("stage1_blacklist_path")
    budgets_path = ctx.get("stage1_budgets_path")
    assert blacklist_path == tmp_path / "stage1_blacklist.json"
    assert budgets_path == tmp_path / "stage1_budgets.json"
    assert blacklist_path.exists()
    assert budgets_path.exists()
