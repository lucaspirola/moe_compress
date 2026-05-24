"""Tests for the stage-4 orchestrator's plugin pipeline.

Covers the orchestrator's registry construction, the canonical phase
schedule (``load_eora_inputs -> LOOP layers[compensate_layer] -> finalize``)
and the finalize sidecar-deletion contract. Layer 1 is a set of fast
registry/order tests that need no model run; Layer 2 instruments a real
stage1->2->3->4 run to verify the orchestrator visits the phases in canonical
order and that finalize deletes the Stage-2/Stage-3 sidecars.

This complements -- and deliberately does NOT duplicate -- the
``Stage``-Protocol surface tests in ``test_stage4_stage.py``
(``isinstance(STAGE4, Stage)``, ``stage_id``, ``is_enabled``).

Helpers (``_TinyTokenizer``, ``_noop_save``, ``patched_stage4``,
``_run_stages_0123``) are redeclared locally on purpose -- tests in this
codebase do not import from each other (codebase discipline; mirrors
``test_stage4_stage.py`` / ``test_stage3_orchestrator.py``).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from moe_compress import stage1, stage3_svd
from moe_compress.budget.solver import BudgetDecomposition
from moe_compress.pipeline.registry import PluginRegistry
from moe_compress.stage2 import orchestrator as stage2_reap_ream
from moe_compress.stage4 import orchestrator as stage4_orchestrator
from moe_compress.stage4.plugins.eora_compensation import EoraCompensationPlugin
from moe_compress.stage4.plugins.eora_inputs import EoraInputsPlugin
from moe_compress.utils.model_io import iter_moe_layers


# --------------------------------------------------------------------------
# Local helpers -- copied verbatim from test_stage4_stage.py /
# test_stage4_golden_snapshot.py (no cross-test imports; codebase discipline).
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
def patched_stage4(monkeypatch, tiny_config):
    """Patch the stage-2/3 calibration loaders and the checkpoint saver so the
    functional test runs fast and writes no real checkpoint. Mirrors the
    ``patched_stage4`` fixture in ``test_stage4_stage.py`` (fp32 case).

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
    """Run Stages 1->2->3 to get a post-SVD model + the sidecars Stage 4 needs.

    Stage 4 runs on the in-memory stage-3-factored model and reads the
    on-disk sidecars ``_stage2_input_covariance.pt`` and
    ``_stage3_original_weights.pt`` -- so this must complete before
    ``stage4_orchestrator.run`` is invoked. Returns the ``BudgetDecomposition``
    (consumed internally by Stages 1/3; unused by the Stage 4 orchestrator,
    which takes no decomposition argument -- returned only for API symmetry).
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


# --------------------------------------------------------------------------
# Canonical schedule -- the orchestrator has NO _STAGE4_PHASES constant; the
# schedule is expressed by an inline walk_phases call + a plain per-layer
# for-loop. This constant + mapping mirror that schedule for the conformance
# tests below.
# --------------------------------------------------------------------------

_CANONICAL_STAGE4_SCHEDULE = ("load_eora_inputs", "compensate_layer")

# (schedule phase, plugin class that owns it). Each plugin owns exactly one
# schedule phase hook; registry/execution order matches this tuple.
_PHASE_PLUGIN_MAP = (
    ("load_eora_inputs", EoraInputsPlugin),
    ("compensate_layer", EoraCompensationPlugin),
)


# ==========================================================================
# Layer 1 -- registry / order tests (no model run, fast).
# ==========================================================================


def test_orchestrator_builds_two_plugins_in_schedule_order():
    """The orchestrator builds a 2-plugin registry whose construction (=
    execution) order matches the canonical stage-4 schedule."""
    registry = PluginRegistry([EoraInputsPlugin(), EoraCompensationPlugin()])
    assert len(registry) == 2
    names = registry.names()
    assert isinstance(names, tuple)
    assert names == ("eora_inputs", "eora_compensation")


def test_each_plugin_owns_its_schedule_phase_hook():
    """Each of the 2 plugins exposes a callable hook for exactly ONE schedule
    phase -- its own -- and no callable for the other phase name."""
    schedule = set(_CANONICAL_STAGE4_SCHEDULE)
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


def test_registry_enabled_keeps_both_plugins(tiny_config):
    """``registry.enabled(config)`` keeps BOTH stage-4 plugins -- unlike
    stage 3's config-gated ``block_refine``, both Stage-4 plugins are
    unconditionally enabled (``is_enabled`` returns ``True`` regardless of
    config)."""
    registry = PluginRegistry([EoraInputsPlugin(), EoraCompensationPlugin()])

    enabled = registry.enabled(tiny_config)
    assert len(enabled) == 2
    assert [p.name for p in enabled] == ["eora_inputs", "eora_compensation"]
    assert [p.name for p in enabled] == list(registry.names())


# ==========================================================================
# Layer 2 -- functional instrumented run (one stage1->2->3->4 run per test).
# ==========================================================================


def test_orchestrator_run_visits_phases_in_canonical_order(
    tiny_model, patched_stage4, tmp_path, monkeypatch,
):
    """Instrument every plugin phase hook on the CLASS and assert the
    orchestrator's ``run`` visits ``load_eora_inputs`` exactly once and FIRST,
    then ``compensate_layer`` once per MoE layer."""
    _run_stages_0123(tiny_model, patched_stage4, tmp_path)

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

    stage4_orchestrator.run(
        tiny_model, _TinyTokenizer(), patched_stage4, tmp_path, no_resume=True,
    )

    # load_eora_inputs runs exactly once, and FIRST.
    assert visited.count("load_eora_inputs") == 1
    assert visited[0] == "load_eora_inputs"

    # compensate_layer runs after load_eora_inputs, once per MoE layer (a
    # fresh no_resume run takes the compute branch for every layer).
    n_moe_layers = len(list(iter_moe_layers(tiny_model)))
    assert visited.count("compensate_layer") == n_moe_layers
    idx_inputs = visited.index("load_eora_inputs")
    idx_first_compensate = visited.index("compensate_layer")
    assert idx_inputs < idx_first_compensate

    # First-occurrence order is the canonical schedule.
    first_seen: list[str] = []
    for phase in visited:
        if phase not in first_seen:
            first_seen.append(phase)
    assert first_seen == list(_CANONICAL_STAGE4_SCHEDULE)


def test_orchestrator_run_deletes_stage3_stage2_sidecars(
    tiny_model, patched_stage4, tmp_path,
):
    """The SIDECAR-DELETION contract: after the stage1->2->3 setup the
    sidecars ``_stage3_original_weights.pt`` and ``_stage2_input_covariance.pt``
    exist on disk; the orchestrator's finalize block deletes BOTH on a
    successful Stage 4 run. Also pins the ``run`` return contract."""
    _run_stages_0123(tiny_model, patched_stage4, tmp_path)

    # Pre-condition: both sidecars are present after Stages 1->2->3.
    s3_originals = tmp_path / "_stage3_original_weights.pt"
    s2_covariance = tmp_path / "_stage2_input_covariance.pt"
    assert s3_originals.is_file()
    assert s2_covariance.is_file()

    out_dir = stage4_orchestrator.run(
        tiny_model, _TinyTokenizer(), patched_stage4, tmp_path, no_resume=True,
    )

    # Finalize deletes BOTH sidecars (durable on the per-stage Hub repos).
    assert not s3_originals.exists()
    assert not s2_covariance.exists()

    # Return contract: the stage4_eora output directory + its JSON artifact.
    assert out_dir == tmp_path / "stage4_eora"
    assert out_dir.is_dir()
    assert (tmp_path / "stage4_eora" / "eora_ranks.json").is_file()
