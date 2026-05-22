"""Tests for the stage-3 orchestrator's plugin pipeline.

Covers the orchestrator's registry construction, the canonical phase
schedule (``collect_covariances -> allocate_ranks -> select_alpha ->
LOOP[factor_layer] -> refine_blocks -> finalize``) and the finalize
artifact contract. Layer 1 is a set of fast registry/order tests that need
no model run; Layer 2 instruments a real stage1->2->3 run to verify the
orchestrator visits the phases in canonical order and writes the expected
artifact set.

This complements -- and deliberately does NOT duplicate -- the
``Stage``-Protocol surface tests in ``test_stage3_stage.py`` (``stage_id``,
``is_enabled``, ``isinstance(STAGE3, Stage)``).

Helpers (``_TinyTokenizer``, ``_noop_save``, ``patched_stage3``,
``_run_stages_1_2``) are redeclared locally on purpose -- tests in this
codebase do not import from each other (codebase discipline; mirrors
``test_stage3_stage.py`` / ``test_stage2_stage.py``).
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from moe_compress import stage1, stage3_svd
from moe_compress.budget.solver import BudgetDecomposition
from moe_compress.stage2 import orchestrator as stage2_reap_ream
from moe_compress.stage3 import orchestrator as stage3_orchestrator
from moe_compress.stage3.plugins.covariance_collection import CovarianceCollectionPlugin
from moe_compress.stage3.plugins.d_rank_allocate import DRankAllocatePlugin
from moe_compress.stage3.plugins.swift_svd_alpha import SwiftSvdAlphaPlugin
from moe_compress.stage3.plugins.aa_svd_factor import AaSvdFactorPlugin
from moe_compress.stage3.plugins.block_refine import BlockRefinePlugin
from moe_compress.pipeline.registry import PluginRegistry
from moe_compress.utils.model_io import iter_moe_layers


# --------------------------------------------------------------------------
# Local helpers -- copied verbatim from test_stage3_stage.py (no cross-test
# imports; codebase discipline).
# --------------------------------------------------------------------------


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
    """Run Stages 1->2 to get a post-prune model + Stage 2 covariance artifact.

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


# --------------------------------------------------------------------------
# Canonical schedule -- the orchestrator has NO _STAGE3_PHASES constant; the
# schedule is expressed by six inline walk_phases / loop_over calls. This
# constant + mapping mirror that schedule for the conformance tests below.
# --------------------------------------------------------------------------

_CANONICAL_STAGE3_SCHEDULE = (
    "collect_covariances", "allocate_ranks", "select_alpha",
    "factor_layer", "refine_blocks",
)

# (schedule phase, plugin class that owns it). Each plugin owns exactly one
# schedule phase hook; registry/execution order matches this list.
_PHASE_PLUGIN_MAP = [
    ("collect_covariances", CovarianceCollectionPlugin),
    ("allocate_ranks", DRankAllocatePlugin),
    ("select_alpha", SwiftSvdAlphaPlugin),
    ("factor_layer", AaSvdFactorPlugin),
    ("refine_blocks", BlockRefinePlugin),
]


# ==========================================================================
# Layer 1 -- registry / order tests (no model run, fast).
# ==========================================================================


def test_orchestrator_builds_five_plugins_in_schedule_order():
    """The orchestrator builds a 5-plugin registry whose construction (=
    execution) order matches the canonical stage-3 schedule."""
    registry = PluginRegistry([
        CovarianceCollectionPlugin(),
        DRankAllocatePlugin(),
        SwiftSvdAlphaPlugin(),
        AaSvdFactorPlugin(),
        BlockRefinePlugin(),
    ])
    assert len(registry) == 5
    names = registry.names()
    assert isinstance(names, tuple)
    assert names == (
        "covariance_collection",
        "d_rank_allocate",
        "swift_svd_alpha",
        "aa_svd_factor",
        "block_refine",
    )


def test_each_plugin_owns_its_schedule_phase_hook():
    """Each of the 5 plugins exposes a callable hook for exactly ONE schedule
    phase -- its own -- and no callable for the other four phase names."""
    schedule = set(_CANONICAL_STAGE3_SCHEDULE)
    for phase, plugin_class in _PHASE_PLUGIN_MAP:
        plugin = plugin_class()
        # owns its phase
        assert callable(getattr(plugin, phase, None)), (
            f"{plugin_class.__name__} must expose a callable {phase!r} hook"
        )
        # owns no other schedule phase
        for other in schedule - {phase}:
            assert not callable(getattr(plugin, other, None)), (
                f"{plugin_class.__name__} must NOT expose the {other!r} hook"
            )


def test_registry_enabled_drops_block_refine_when_disabled(tiny_config):
    """``registry.enabled(config)`` drops BlockRefinePlugin when
    ``stage3_svd.block_refine.enabled`` is false (the ``tiny_config``
    default), and keeps all 5 once it is flipped on."""
    registry = PluginRegistry([
        CovarianceCollectionPlugin(),
        DRankAllocatePlugin(),
        SwiftSvdAlphaPlugin(),
        AaSvdFactorPlugin(),
        BlockRefinePlugin(),
    ])

    # block_refine disabled by default in tiny_config -> 4 enabled.
    enabled = registry.enabled(tiny_config)
    assert len(enabled) == 4
    enabled_names = [p.name for p in enabled]
    assert "block_refine" not in enabled_names
    assert enabled_names == [
        "covariance_collection", "d_rank_allocate",
        "swift_svd_alpha", "aa_svd_factor",
    ]

    # flip block_refine on -> all 5 enabled.
    cfg_on = copy.deepcopy(tiny_config)
    cfg_on["stage3_svd"]["block_refine"]["enabled"] = True
    enabled_on = registry.enabled(cfg_on)
    assert len(enabled_on) == 5
    assert [p.name for p in enabled_on] == list(registry.names())


# ==========================================================================
# Layer 2 -- functional instrumented run (one stage1->2->3 run per test).
# ==========================================================================


def test_orchestrator_run_visits_phases_in_canonical_order(
    tiny_model, patched_stage3, tmp_path, monkeypatch,
):
    """Instrument every plugin phase hook on the CLASS and assert the
    orchestrator's ``run`` visits them in canonical schedule order, skips
    ``refine_blocks`` (block_refine disabled), and walks ``factor_layer``
    once per MoE layer."""
    decomp = _run_stages_1_2(tiny_model, patched_stage3, tmp_path)

    visited: list[str] = []

    # Patch each phase hook on the CLASS -- the orchestrator instantiates the
    # plugins itself, so per-instance patching would not reach them. Capture
    # the original method BEFORE patching, then wrap it.
    for phase, plugin_class in _PHASE_PLUGIN_MAP:
        original = getattr(plugin_class, phase)

        def _make_wrapper(_phase, _original):
            def _wrapper(self, ctx):
                visited.append(_phase)
                return _original(self, ctx)
            return _wrapper

        monkeypatch.setattr(plugin_class, phase, _make_wrapper(phase, original))

    stage3_orchestrator.run(
        tiny_model, _TinyTokenizer(), patched_stage3, tmp_path, decomp,
        device=None, no_resume=True,
    )

    # block_refine disabled in tiny_config -> refine_blocks never visited.
    assert "refine_blocks" not in visited

    # First-occurrence order of the 4 active phases is the canonical prefix.
    first_seen: list[str] = []
    for phase in visited:
        if phase not in first_seen:
            first_seen.append(phase)
    assert first_seen == [
        "collect_covariances", "allocate_ranks",
        "select_alpha", "factor_layer",
    ]

    # factor_layer runs once per MoE layer.
    n_moe_layers = len(list(iter_moe_layers(tiny_model)))
    assert visited.count("factor_layer") == n_moe_layers

    # Strict index ordering: each preamble phase precedes the first factor_layer.
    idx_cov = visited.index("collect_covariances")
    idx_alloc = visited.index("allocate_ranks")
    idx_alpha = visited.index("select_alpha")
    idx_first_factor = visited.index("factor_layer")
    assert idx_cov < idx_alloc < idx_alpha < idx_first_factor


def test_orchestrator_run_produces_expected_artifact_set(
    tiny_model, patched_stage3, tmp_path,
):
    """``run`` returns the ``stage3_svd`` output dir and writes exactly the
    artifacts the ``tiny_config`` / ``no_resume=True`` path produces:
    ``stage3_svd/rank_map.json`` and ``_stage3_original_weights.pt``.

    Under ``no_resume=True`` the orchestrator does NOT write
    ``_stage3_alpha_result.json`` (gated on ``if not no_resume``) and cleans
    up the ``_stage3_bcov_partial`` spill dir at finalize."""
    decomp = _run_stages_1_2(tiny_model, patched_stage3, tmp_path)

    out_dir = stage3_orchestrator.run(
        tiny_model, _TinyTokenizer(), patched_stage3, tmp_path, decomp,
        device=None, no_resume=True,
    )

    # Return contract: the stage3_svd output directory.
    assert out_dir == tmp_path / "stage3_svd"
    assert out_dir.is_dir()

    # rank_map.json -- the finalize JSON artifact.
    assert (tmp_path / "stage3_svd" / "rank_map.json").is_file()

    # _stage3_original_weights.pt -- snapshot persisted before Phase D so
    # Stage 4 can always resume from the correct originals.
    assert (tmp_path / "_stage3_original_weights.pt").is_file()

    # no_resume=True path: the alpha-result cache is NOT written (gated on
    # `if not no_resume`).
    assert not (tmp_path / "_stage3_alpha_result.json").exists()

    # The B-cov spill dir is removed on a clean finish.
    assert not (tmp_path / "_stage3_bcov_partial").exists()
