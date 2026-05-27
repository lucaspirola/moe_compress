"""Plugin #7 — RKD paper-recipe (Row P) config-override plugin tests.

Verifies the ``RkdPaperRecipePlugin`` scaffolding in
``router_kd/plugins/rkd_paper_recipe.py``:

* the plugin imports, satisfies the universal ``PipelinePlugin`` Protocol,
  and carries correct metadata (``name`` / ``paper`` / ``config_key`` /
  tuple-typed ``reads`` / ``writes`` / ``provides``);
* the ``is_enabled`` gate keys off ``stage5_router_kd.rkd_recipe == "paper"``;
* ``apply_config_overrides`` is a true no-op on the default
  ``"current"`` / missing-key paths (Row C is byte-identical to pre-plugin
  behavior);
* ``apply_config_overrides`` correctly mutates all 5 keys (4 numeric deltas
  + ``teacher_logits_cache`` clearance + calibration-source swap) on the
  ``"paper"`` path;
* downstream Stage 2.5 / Stage 5 plugins (``KdOptimizerPlugin``,
  ``EarlyStopPlugin``) read the OVERRIDDEN values after
  ``apply_config_overrides`` runs — i.e. the contract holds end-to-end;
* the orchestrator-integration smoke (a config goes in unmutated; the
  ``run()``-entry override leaves it as Row P);
* the ``wikitext-103-raw`` corpus adapter is registered in
  ``utils/calibration.py``, ``spec_from_config`` dispatches to it, and
  ``_stream_texts_wikitext_103_raw`` skips empty rows + does NOT invoke
  the tokenizer's chat template (raw-text invariant);
* the module never imports the ``stage5_router_kd`` monolith or
  ``router_kd.orchestrator`` at any scope (circular-import contract).

DO NOT add tests that exercise the orchestrator's full ``run()`` path —
that requires a real student / dataset / GPU. The smoke here covers
``apply_config_overrides`` only, which is the orchestrator's first call.
"""
from __future__ import annotations

import ast
import copy
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Group A — plugin scaffolding (imports / Protocol / metadata / no-fwd-import)
# ---------------------------------------------------------------------------


def test_plugin_imports():
    """``RkdPaperRecipePlugin`` imports from the plugin module."""
    from moe_compress.router_kd.plugins.rkd_paper_recipe import (
        RkdPaperRecipePlugin,
    )

    assert isinstance(RkdPaperRecipePlugin, type)


def test_plugin_satisfies_protocol():
    """``RkdPaperRecipePlugin`` structurally satisfies ``PipelinePlugin``.

    Class-level attributes + ``is_enabled`` + ``contribute_artifact`` —
    no phase hooks are required by the Protocol.
    """
    from moe_compress.pipeline.plugin import PipelinePlugin
    from moe_compress.router_kd.plugins.rkd_paper_recipe import (
        RkdPaperRecipePlugin,
    )

    assert isinstance(RkdPaperRecipePlugin(), PipelinePlugin)


def test_plugin_metadata():
    """Plugin metadata — name / paper id / config_key / tuple-typed fields."""
    from moe_compress.router_kd.plugins.rkd_paper_recipe import (
        RkdPaperRecipePlugin,
    )

    plugin = RkdPaperRecipePlugin()
    assert plugin.name == "rkd_paper_recipe"
    assert "2603.02217" in plugin.paper
    assert plugin.config_key == "stage5_router_kd.rkd_recipe"
    assert isinstance(plugin.reads, tuple)
    assert isinstance(plugin.writes, tuple)
    assert isinstance(plugin.provides, tuple)
    assert plugin.reads == ("config",)
    assert plugin.writes == ()
    assert plugin.provides == ()


def test_contribute_artifact_empty():
    """``contribute_artifact`` returns a fresh empty dict (Protocol contract)."""
    from moe_compress.pipeline.context import PipelineContext
    from moe_compress.router_kd.plugins.rkd_paper_recipe import (
        RkdPaperRecipePlugin,
    )

    plugin = RkdPaperRecipePlugin()
    assert plugin.contribute_artifact(PipelineContext()) == {}


def test_no_forbidden_import():
    """The module never imports ``stage5_router_kd`` / ``router_kd.orchestrator``.

    The orchestrator imports *this* module at the module top, so any
    reverse import here would deadlock the import graph. AST-walk the
    full source tree to catch function-local imports too.

    The orchestrator check specifically targets the router_kd orchestrator
    (not any unrelated module whose name happens to contain "orchestrator"):
    we match the bare ``orchestrator`` module name, any module ending in
    ``.orchestrator``, or the fully-qualified
    ``moe_compress.router_kd.orchestrator`` / ``router_kd.orchestrator``.
    """
    from moe_compress.router_kd.plugins import rkd_paper_recipe as mod

    def _is_orchestrator(name: str) -> bool:
        return (
            name == "orchestrator"
            or name.endswith(".orchestrator")
            or name == "router_kd.orchestrator"
            or name == "moe_compress.router_kd.orchestrator"
        )

    def _is_forbidden(name: str) -> bool:
        # ``stage5_router_kd`` is a substring-unique monolith module name
        # (no other module in the tree contains this token), so substring
        # matching is safe here; the orchestrator check is tightened above.
        return "stage5_router_kd" in name or _is_orchestrator(name)

    src = Path(mod.__file__).read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):  # any nesting level, not just module-top
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not _is_forbidden(alias.name), (
                    f"forbidden import at any scope: {alias.name}"
                )
        elif isinstance(node, ast.ImportFrom):
            mod_name = node.module or ""
            assert not _is_forbidden(mod_name), (
                f"forbidden import-from at any scope: {mod_name}"
            )
            for alias in node.names:
                assert not _is_forbidden(alias.name), (
                    f"forbidden import-from name at any scope: "
                    f"from {mod_name} import {alias.name}"
                )


# ---------------------------------------------------------------------------
# Group B — ``is_enabled`` gate
# ---------------------------------------------------------------------------


def test_is_enabled_paper():
    """``rkd_recipe='paper'`` → enabled."""
    from moe_compress.router_kd.plugins.rkd_paper_recipe import (
        RkdPaperRecipePlugin,
    )

    plugin = RkdPaperRecipePlugin()
    assert plugin.is_enabled(
        {"stage5_router_kd": {"rkd_recipe": "paper"}}
    ) is True


def test_is_enabled_current():
    """``rkd_recipe='current'`` → disabled."""
    from moe_compress.router_kd.plugins.rkd_paper_recipe import (
        RkdPaperRecipePlugin,
    )

    plugin = RkdPaperRecipePlugin()
    assert plugin.is_enabled(
        {"stage5_router_kd": {"rkd_recipe": "current"}}
    ) is False


def test_is_enabled_default():
    """Missing ``rkd_recipe`` key → defaults to ``"current"`` → disabled."""
    from moe_compress.router_kd.plugins.rkd_paper_recipe import (
        RkdPaperRecipePlugin,
    )

    plugin = RkdPaperRecipePlugin()
    assert plugin.is_enabled({"stage5_router_kd": {}}) is False


def test_is_enabled_missing_block():
    """Missing ``stage5_router_kd`` block → graceful False (never raises)."""
    from moe_compress.router_kd.plugins.rkd_paper_recipe import (
        RkdPaperRecipePlugin,
    )

    plugin = RkdPaperRecipePlugin()
    assert plugin.is_enabled({}) is False


def test_is_enabled_null_value():
    """YAML `rkd_recipe: null` (Python None) must NOT enable paper mode."""
    from moe_compress.router_kd.plugins.rkd_paper_recipe import (
        RkdPaperRecipePlugin,
    )

    plugin = RkdPaperRecipePlugin()
    assert plugin.is_enabled({"stage5_router_kd": {"rkd_recipe": None}}) is False


def test_is_enabled_wrong_case():
    """YAML `rkd_recipe: PAPER` (without quotes, treated as string) must NOT
    enable paper mode — string comparison is case-sensitive by design."""
    from moe_compress.router_kd.plugins.rkd_paper_recipe import (
        RkdPaperRecipePlugin,
    )

    plugin = RkdPaperRecipePlugin()
    assert plugin.is_enabled({"stage5_router_kd": {"rkd_recipe": "PAPER"}}) is False


# ---------------------------------------------------------------------------
# Group C — ``apply_config_overrides`` no-op on non-paper recipe
# ---------------------------------------------------------------------------


def _row_c_config() -> dict:
    """Build a config dict carrying the canonical Row C (current) values."""
    return {
        "stage5_router_kd": {
            "rkd_recipe": "current",
            "kd_temperature": 1.0,
            "weight_decay": 0.01,
            "epochs": 1,
            "early_stop_patience": 8,
            "teacher_logits_cache": "/some/path/teacher_cache.pt",
        },
        "calibration": {
            "source": "qwen3-pretrain-mix-v2",
            "num_sequences": 3000,
            "sequence_length": 2048,
            "seed": 1337,
        },
    }


def test_apply_config_overrides_noop_on_current():
    """``rkd_recipe='current'`` → no mutation; all Row C values preserved."""
    from moe_compress.router_kd.plugins.rkd_paper_recipe import (
        RkdPaperRecipePlugin,
    )

    config = _row_c_config()
    # Snapshot a deep-ish copy of the relevant fields to detect mutation.
    s5_before = dict(config["stage5_router_kd"])
    cal_before = dict(config["calibration"])

    RkdPaperRecipePlugin().apply_config_overrides(config)

    assert config["stage5_router_kd"] == s5_before
    assert config["calibration"] == cal_before


def test_apply_config_overrides_noop_on_default():
    """Missing ``rkd_recipe`` key (= default ``current``) → no mutation."""
    from moe_compress.router_kd.plugins.rkd_paper_recipe import (
        RkdPaperRecipePlugin,
    )

    config = _row_c_config()
    del config["stage5_router_kd"]["rkd_recipe"]
    s5_before = dict(config["stage5_router_kd"])
    cal_before = dict(config["calibration"])

    RkdPaperRecipePlugin().apply_config_overrides(config)

    assert config["stage5_router_kd"] == s5_before
    assert config["calibration"] == cal_before
    # Defensive: the no-op path must NOT add ``rkd_recipe`` back in.
    assert "rkd_recipe" not in config["stage5_router_kd"]


def test_apply_config_overrides_noop_on_missing_block():
    """Missing ``stage5_router_kd`` block → graceful return; no mutation."""
    from moe_compress.router_kd.plugins.rkd_paper_recipe import (
        RkdPaperRecipePlugin,
    )

    config = {"calibration": {"source": "qwen3-pretrain-mix-v2"}}
    RkdPaperRecipePlugin().apply_config_overrides(config)
    assert config == {"calibration": {"source": "qwen3-pretrain-mix-v2"}}


def test_apply_config_overrides_null_value_is_noop():
    """rkd_recipe=None is treated as not-paper; no overrides applied."""
    from moe_compress.router_kd.plugins.rkd_paper_recipe import (
        RkdPaperRecipePlugin,
    )

    plugin = RkdPaperRecipePlugin()
    config = {
        "stage5_router_kd": {"rkd_recipe": None, "kd_temperature": 1.0},
        "calibration": {"source": "qwen3-pretrain-mix-v2"},
    }
    before = copy.deepcopy(config)
    plugin.apply_config_overrides(config)
    assert config == before


def test_apply_config_overrides_wrong_case_is_noop():
    """rkd_recipe='PAPER' (wrong case) does NOT enable paper mode; no overrides."""
    from moe_compress.router_kd.plugins.rkd_paper_recipe import (
        RkdPaperRecipePlugin,
    )

    plugin = RkdPaperRecipePlugin()
    config = {
        "stage5_router_kd": {"rkd_recipe": "PAPER", "kd_temperature": 1.0},
        "calibration": {"source": "qwen3-pretrain-mix-v2"},
    }
    before = copy.deepcopy(config)
    plugin.apply_config_overrides(config)
    assert config == before


# ---------------------------------------------------------------------------
# Group D — ``apply_config_overrides`` paper recipe applied
# ---------------------------------------------------------------------------


def _row_p_config() -> dict:
    """Row C config but with ``rkd_recipe='paper'`` so overrides fire."""
    config = _row_c_config()
    config["stage5_router_kd"]["rkd_recipe"] = "paper"
    return config


def test_apply_config_overrides_paper_sets_temperature():
    """``kd_temperature: 1.0 -> 4.0`` under paper recipe."""
    from moe_compress.router_kd.plugins.rkd_paper_recipe import (
        RkdPaperRecipePlugin,
    )

    config = _row_p_config()
    RkdPaperRecipePlugin().apply_config_overrides(config)
    assert config["stage5_router_kd"]["kd_temperature"] == 4.0


def test_apply_config_overrides_paper_sets_weight_decay():
    """``weight_decay: 0.01 -> 0.0`` under paper recipe."""
    from moe_compress.router_kd.plugins.rkd_paper_recipe import (
        RkdPaperRecipePlugin,
    )

    config = _row_p_config()
    RkdPaperRecipePlugin().apply_config_overrides(config)
    assert config["stage5_router_kd"]["weight_decay"] == 0.0


def test_apply_config_overrides_paper_sets_epochs():
    """``epochs: 1 -> 2`` under paper recipe."""
    from moe_compress.router_kd.plugins.rkd_paper_recipe import (
        RkdPaperRecipePlugin,
    )

    config = _row_p_config()
    RkdPaperRecipePlugin().apply_config_overrides(config)
    assert config["stage5_router_kd"]["epochs"] == 2


def test_apply_config_overrides_paper_disables_early_stop():
    """``early_stop_patience: 8 -> 0`` under paper recipe (early-stop off)."""
    from moe_compress.router_kd.plugins.rkd_paper_recipe import (
        RkdPaperRecipePlugin,
    )

    config = _row_p_config()
    RkdPaperRecipePlugin().apply_config_overrides(config)
    assert config["stage5_router_kd"]["early_stop_patience"] == 0


def test_apply_config_overrides_paper_sets_calib_source():
    """Calibration source swapped to ``wikitext-103-raw`` under paper recipe."""
    from moe_compress.router_kd.plugins.rkd_paper_recipe import (
        RkdPaperRecipePlugin,
    )

    config = _row_p_config()
    RkdPaperRecipePlugin().apply_config_overrides(config)
    assert config["calibration"]["source"] == "wikitext-103-raw"


def test_apply_config_overrides_paper_clears_teacher_cache():
    """``teacher_logits_cache`` cleared on paper recipe (multi-epoch guard).

    ``orchestrator.py:585`` raises if ``epochs > 1 and
    teacher_logits_cache is not None``. Row P sets epochs=2, so the
    override must explicitly clear ``teacher_logits_cache`` to None.
    """
    from moe_compress.router_kd.plugins.rkd_paper_recipe import (
        RkdPaperRecipePlugin,
    )

    config = _row_p_config()
    assert config["stage5_router_kd"]["teacher_logits_cache"] == "/some/path/teacher_cache.pt"
    RkdPaperRecipePlugin().apply_config_overrides(config)
    assert config["stage5_router_kd"]["teacher_logits_cache"] is None


def test_apply_config_overrides_paper_all_deltas_together():
    """One-shot check: all 5 mutations applied in a single override call."""
    from moe_compress.router_kd.plugins.rkd_paper_recipe import (
        RkdPaperRecipePlugin,
    )

    config = _row_p_config()
    RkdPaperRecipePlugin().apply_config_overrides(config)

    s5 = config["stage5_router_kd"]
    assert s5["kd_temperature"] == 4.0
    assert s5["weight_decay"] == 0.0
    assert s5["epochs"] == 2
    assert s5["early_stop_patience"] == 0
    assert s5["teacher_logits_cache"] is None
    assert config["calibration"]["source"] == "wikitext-103-raw"


def test_apply_config_overrides_paper_idempotent():
    """Calling ``apply_config_overrides`` twice yields the same final config."""
    from moe_compress.router_kd.plugins.rkd_paper_recipe import (
        RkdPaperRecipePlugin,
    )

    config = _row_p_config()
    plugin = RkdPaperRecipePlugin()
    plugin.apply_config_overrides(config)
    snapshot = {
        "s5": dict(config["stage5_router_kd"]),
        "cal": dict(config["calibration"]),
    }
    plugin.apply_config_overrides(config)
    assert dict(config["stage5_router_kd"]) == snapshot["s5"]
    assert dict(config["calibration"]) == snapshot["cal"]


# ---------------------------------------------------------------------------
# Group E — downstream plugins read OVERRIDDEN values post-mutation
# ---------------------------------------------------------------------------


def test_kd_optimizer_reads_overridden_weight_decay():
    """After ``apply_config_overrides`` fires, ``KdOptimizerPlugin``
    constructs the AdamW with ``weight_decay=0.0`` (not 0.01).

    Proves the end-to-end contract: overrides applied BEFORE plugin
    hooks run → plugins read the correct values via ``s5.get(...)``.
    """
    from moe_compress.pipeline.context import PipelineContext
    from moe_compress.router_kd.plugins.kd_optimizer import KdOptimizerPlugin
    from moe_compress.router_kd.plugins.rkd_paper_recipe import (
        RkdPaperRecipePlugin,
    )

    # Minimal trainable module — kd_optimizer requires at least one
    # ``requires_grad=True`` parameter.
    student = nn.Linear(4, 4)
    config = _row_p_config()
    config["stage5_router_kd"]["learning_rate"] = 5e-5

    # Step 1 — apply overrides (this is what the orchestrator does on entry).
    RkdPaperRecipePlugin().apply_config_overrides(config)
    assert config["stage5_router_kd"]["weight_decay"] == 0.0

    # Step 2 — invoke the downstream plugin and check it sees the override.
    ctx = PipelineContext()
    ctx.set("student", student)
    ctx.set("config", config)
    ctx.set("total_optim_steps", 100)

    KdOptimizerPlugin().build_optimizer(ctx)
    optim = ctx.get("optimizer")
    assert isinstance(optim, torch.optim.AdamW)
    # Single param group (no merge_repair) — weight_decay must be 0.0.
    assert optim.param_groups[0]["weight_decay"] == 0.0


def test_early_stop_reads_overridden_patience(tmp_path):
    """After ``apply_config_overrides`` fires, ``EarlyStopPlugin`` seeds
    ``early_stop_patience=0`` (disabled), not 8.
    """
    from moe_compress.pipeline.context import PipelineContext
    from moe_compress.router_kd.plugins.early_stop import EarlyStopPlugin
    from moe_compress.router_kd.plugins.rkd_paper_recipe import (
        RkdPaperRecipePlugin,
    )

    class _OneParam(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = nn.Parameter(torch.zeros(3))

    config = _row_p_config()
    RkdPaperRecipePlugin().apply_config_overrides(config)
    assert config["stage5_router_kd"]["early_stop_patience"] == 0

    ctx = PipelineContext()
    ctx.set("config", config)
    ctx.set("partial_dir", tmp_path)
    ctx.set("student", _OneParam())
    ctx.set("stage_key", "stage5")

    EarlyStopPlugin().setup_early_stop(ctx)
    assert ctx.get("early_stop_patience") == 0


def test_orchestrator_integration_smoke():
    """Orchestrator imports the plugin and uses it at the right injection point.

    Verifies (1) the plugin is importable from the orchestrator module
    (the wiring statement parses and the module loads), and (2) running
    ``apply_config_overrides`` on a Row P config makes the orchestrator's
    downstream captures (``s5`` / ``cal``) see the overridden values —
    which is the exact contract the orchestrator wiring relies on.

    Does NOT call ``orchestrator.run`` (that requires GPU + real model).
    """
    from moe_compress.router_kd import orchestrator as orch
    from moe_compress.router_kd.plugins.rkd_paper_recipe import (
        RkdPaperRecipePlugin,
    )

    # (1) The orchestrator module imports the plugin class.
    assert hasattr(orch, "RkdPaperRecipePlugin")
    assert orch.RkdPaperRecipePlugin is RkdPaperRecipePlugin

    # (2) Simulate the orchestrator's entry sequence: overrides first,
    # then the s5/cal captures the orchestrator does at lines ~178-179.
    config = _row_p_config()
    RkdPaperRecipePlugin().apply_config_overrides(config)
    s5 = config["stage5_router_kd"]
    cal = config["calibration"]
    # These are the exact values orchestrator.run() will then see.
    assert s5["kd_temperature"] == 4.0
    assert s5["weight_decay"] == 0.0
    assert s5["epochs"] == 2
    assert s5["early_stop_patience"] == 0
    assert s5["teacher_logits_cache"] is None
    assert cal["source"] == "wikitext-103-raw"


# ---------------------------------------------------------------------------
# Group F — wikitext-103-raw corpus adapter (in calibration.py)
# ---------------------------------------------------------------------------


def test_wikitext_103_raw_corpus_registered():
    """``wikitext-103-raw`` is in the registered-corpora list."""
    from moe_compress.utils.calibration import registered_corpora

    assert "wikitext-103-raw" in registered_corpora()


def test_wikitext_103_raw_adapter_lookup():
    """``get_corpus_adapter('wikitext-103-raw')`` returns a sane adapter."""
    from moe_compress.utils.calibration import get_corpus_adapter

    adapter = get_corpus_adapter("wikitext-103-raw")
    assert adapter.name == "wikitext-103-raw"
    assert callable(adapter.parse_yaml)
    assert callable(adapter.stream_texts)


def test_wikitext_103_raw_spec_from_config():
    """``spec_from_config`` dispatches to the wikitext adapter and stamps
    the right ``source`` / ``dataset`` defaults.
    """
    from moe_compress.utils.calibration import spec_from_config

    cal_cfg = {
        "source": "wikitext-103-raw",
        "num_sequences": 4,
        "sequence_length": 16,
        "seed": 0,
    }
    spec = spec_from_config(cal_cfg)
    assert spec.source == "wikitext-103-raw"
    assert spec.dataset == "Salesforce/wikitext"
    assert spec.num_sequences == 4
    assert spec.sequence_length == 16


def test_wikitext_103_raw_spec_honours_dataset_override():
    """Operator-supplied ``dataset:`` overrides the default repo path."""
    from moe_compress.utils.calibration import spec_from_config

    cal_cfg = {
        "source": "wikitext-103-raw",
        "dataset": "some-mirror/wikitext",
        "num_sequences": 2,
        "sequence_length": 8,
        "seed": 0,
    }
    spec = spec_from_config(cal_cfg)
    assert spec.dataset == "some-mirror/wikitext"


def test_wikitext_103_raw_stream_no_chat_template(monkeypatch):
    """The wikitext stream MUST NOT call the tokenizer's chat template.

    Wikitext rows are raw encyclopedic paragraphs, not OpenAI-style
    messages. Applying a chat template would inject role-header tokens
    that are not present in the source data and would skew the
    calibration distribution. Verify by asserting the tokenizer's
    ``apply_chat_template`` MagicMock is never invoked.
    """
    from moe_compress.utils.calibration import (
        CalibrationSpec,
        _stream_texts_wikitext_103_raw,
    )

    fake_rows = [
        {"text": "The quick brown fox jumps over the lazy dog."},
        {"text": "Wikipedia is a free online encyclopedia."},
        {"text": ""},   # blank-line separator — must be skipped
        {"text": "Calibration distributions matter for MoE compression."},
    ]

    class _FakeDS:
        def __init__(self, rows): self._rows = list(rows)
        def shuffle(self, *_a, **_kw): return self
        def __iter__(self): return iter(self._rows)

    def _fake_load_dataset(name, config, *, split, streaming):
        assert config == "wikitext-103-raw-v1"
        assert split == "train"
        assert streaming is True
        return _FakeDS(fake_rows)

    import datasets as _datasets
    monkeypatch.setattr(_datasets, "load_dataset", _fake_load_dataset)

    tokenizer = MagicMock()
    spec = CalibrationSpec(
        num_sequences=3,
        sequence_length=16,
        seed=0,
        source="wikitext-103-raw",
        dataset="Salesforce/wikitext",
    )
    out = _stream_texts_wikitext_103_raw(spec, tokenizer)

    # The chat template is the load-bearing invariant — must never be hit.
    tokenizer.apply_chat_template.assert_not_called()
    # The 3 non-empty rows came through (the empty row is skipped).
    assert out == [
        "The quick brown fox jumps over the lazy dog.",
        "Wikipedia is a free online encyclopedia.",
        "Calibration distributions matter for MoE compression.",
    ]


def test_wikitext_103_raw_stream_skips_empty(monkeypatch):
    """Stream skips empty / whitespace-only ``text`` rows (paragraph separators)."""
    from moe_compress.utils.calibration import (
        CalibrationSpec,
        _stream_texts_wikitext_103_raw,
    )

    fake_rows = [
        {"text": ""},
        {"text": "   "},
        {"text": "\n"},
        {"text": "Real paragraph."},
        {"text": ""},
        {"text": "Another paragraph."},
    ]

    class _FakeDS:
        def __init__(self, rows): self._rows = list(rows)
        def shuffle(self, *_a, **_kw): return self
        def __iter__(self): return iter(self._rows)

    import datasets as _datasets
    monkeypatch.setattr(
        _datasets, "load_dataset",
        lambda *a, **kw: _FakeDS(fake_rows),
    )

    spec = CalibrationSpec(
        num_sequences=10,
        sequence_length=16,
        seed=0,
        source="wikitext-103-raw",
        dataset="Salesforce/wikitext",
    )
    out = _stream_texts_wikitext_103_raw(spec, MagicMock())
    assert out == ["Real paragraph.", "Another paragraph."]
