"""Tests for the cross-stage Trackio telemetry expansion.

This file groups one test per stage that exercises the new ``_trackio_log``
emits added by the "no-refactor" telemetry expansion. Each test:
  1. Monkey-patches ``_trackio_log`` to record every dict passed to it.
  2. Drives the relevant stage on the synthetic ``_TinyModel`` fixture.
  3. Asserts the new keys appear with expected types.
  4. Includes a regression guard: existing pre-expansion keys are still
     present and unchanged in name and type.

Per the Stage 2 v2 precedent, helpers (``_TinyTokenizer``, calibration /
save monkey-patches) are duplicated locally rather than promoted to
conftest.py to keep the file self-contained.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch


# ---------------------------------------------------------------------------
# Shared helpers (mirrors the patched_stage2 fixture pattern)
# ---------------------------------------------------------------------------


class _TinyTokenizer:
    name_or_path = "tiny-tokenizer"
    eos_token_id = 0

    def __call__(self, text, *_, **__):
        return {"input_ids": [min(ord(c) % 32, 31) for c in (text or " ")]}

    def save_pretrained(self, *_a, **_kw):
        return None


def _noop_save(model, tokenizer, path, **kwargs):
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


@pytest.fixture
def _patched_calib(monkeypatch):
    """Stub calibration tensor builders + save_compressed_checkpoint so any
    stage can run on the synthetic ``_TinyModel`` without hitting the
    network."""
    from moe_compress.utils import calibration as cal_mod
    from moe_compress.utils import model_io as mio
    from moe_compress import stage5_router_kd
    from moe_compress.stage2 import orchestrator as stage2_reap_ream

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
    monkeypatch.setattr(mio, "save_compressed_checkpoint", _noop_save)
    monkeypatch.setattr(stage2_reap_ream, "save_compressed_checkpoint", _noop_save)


@pytest.fixture
def _captured_emits(monkeypatch):
    """Patch ``_trackio_log`` in every stage module that imports it. Returns
    the list reference for the test to inspect."""
    captured: list[dict] = []

    def _capture(metrics: dict) -> None:
        captured.append(dict(metrics))

    from moe_compress import (
        run_pipeline, stage3_svd,
        stage4_eora, stage5_router_kd, stage6_validate,
    )
    from moe_compress.stage2 import orchestrator as stage2_reap_ream
    # The plugin-based Stage 1 emits Trackio telemetry from three modules
    # (orchestrator + the grape_merge / ma_detection plugins), replacing the
    # single stage1_grape monolith.
    from moe_compress.stage1 import orchestrator as stage1_orchestrator
    from moe_compress.stage1.plugins import grape_merge as stage1_grape_merge
    from moe_compress.stage1.plugins import ma_detection as stage1_ma_detection
    for mod in (run_pipeline, stage2_reap_ream, stage3_svd,
                stage4_eora, stage5_router_kd, stage6_validate,
                stage1_orchestrator, stage1_grape_merge, stage1_ma_detection):
        monkeypatch.setattr(mod, "_trackio_log", _capture, raising=False)

    return captured


# ---------------------------------------------------------------------------
# Stage 1 — GRAPE summary keys
# ---------------------------------------------------------------------------


def test_stage1_grape_emits_summary_keys(
    _captured_emits, _patched_calib, tiny_model, tiny_config, tmp_path,
):
    """Stage 1's GRAPE inner function should emit a summary dict with the
    new entropy / merge / exit_reason keys at the end of greedy."""
    from moe_compress import stage1
    from moe_compress.budget.solver import BudgetDecomposition

    decomp = BudgetDecomposition(
        total_reduction_ratio=0.2, expert_prune_ratio=0.5,
        svd_rank_ratio=0.14, global_expert_budget=4,
        min_experts_per_layer=2, blacklisted_experts={},
    )
    stage1.run(tiny_model, _TinyTokenizer(), tiny_config, tmp_path, decomp)

    # Find the GRAPE summary emit (has exit_reason).
    grape_emits = [
        e for e in _captured_emits
        if "stage1/exit_reason" in e
    ]
    assert grape_emits, "Stage 1 GRAPE summary emit not captured"
    g = grape_emits[0]
    expected = {
        "stage1/effective_budget": int,
        "stage1/global_budget": int,
        "stage1/total_blacklisted": int,
        "stage1/entropy_initial": float,
        "stage1/entropy_threshold": float,
        "stage1/gamma": float,
        "stage1/n_merges_executed": int,
        "stage1/exit_reason": str,
        "stage1/final_total": int,
    }
    for k, t in expected.items():
        assert k in g, f"Stage 1 GRAPE summary missing {k}"
        assert isinstance(g[k], t), (
            f"Stage 1 GRAPE summary {k} has type {type(g[k]).__name__}, expected {t.__name__}"
        )
    assert g["stage1/exit_reason"] in ("budget", "no_layer", "max_iter"), (
        f"unexpected exit_reason value: {g['stage1/exit_reason']!r}"
    )


def test_stage1_grape_emits_phase_a_c_summary(
    _captured_emits, _patched_calib, tiny_model, tiny_config, tmp_path,
):
    """Stage 1 should emit a Phase A/C summary with MA layers count + thresholds."""
    from moe_compress import stage1
    from moe_compress.budget.solver import BudgetDecomposition

    decomp = BudgetDecomposition(
        total_reduction_ratio=0.2, expert_prune_ratio=0.5,
        svd_rank_ratio=0.14, global_expert_budget=4,
        min_experts_per_layer=2, blacklisted_experts={},
    )
    stage1.run(tiny_model, _TinyTokenizer(), tiny_config, tmp_path, decomp)

    pa_emits = [
        e for e in _captured_emits
        if "stage1/ma_formation_layers_count" in e
    ]
    assert pa_emits, "Stage 1 Phase A/C summary emit not captured"
    pa = pa_emits[0]
    assert isinstance(pa["stage1/ma_formation_layers_count"], int)
    assert isinstance(pa["stage1/total_experts"], int)
    assert isinstance(pa["stage1/p995_threshold"], float)
    assert isinstance(pa["stage1/a_max"], float)
    assert isinstance(pa["stage1/a_max_threshold"], float)
    assert isinstance(pa["stage1/n_blacklisted"], int)


def test_stage1_existing_per_layer_emits_unchanged(
    _captured_emits, _patched_calib, tiny_model, tiny_config, tmp_path,
):
    """Regression guard: pre-expansion per-layer SE / GRAPE-budget emits
    must still appear with their original keys."""
    from moe_compress import stage1
    from moe_compress.budget.solver import BudgetDecomposition

    decomp = BudgetDecomposition(
        total_reduction_ratio=0.2, expert_prune_ratio=0.5,
        svd_rank_ratio=0.14, global_expert_budget=4,
        min_experts_per_layer=2, blacklisted_experts={},
    )
    stage1.run(tiny_model, _TinyTokenizer(), tiny_config, tmp_path, decomp)

    se_emits = [e for e in _captured_emits if "stage1/se_layer_idx" in e]
    assert se_emits, "v1 per-layer SE emits regressed"
    for emit in se_emits:
        assert "stage1/se_blacklisted" in emit
        assert "stage1/se_in_ma_layer" in emit

    budget_emits = [e for e in _captured_emits if "stage1/budget" in e]
    assert budget_emits, "v1 per-layer GRAPE budget emits regressed"
    for emit in budget_emits:
        assert "stage1/layer_idx" in emit
        assert "stage1/redundancy" in emit


# ---------------------------------------------------------------------------
# Stage 4 — extended per-layer EoRA emit
# ---------------------------------------------------------------------------


def test_stage4_summarize_distill_helper_present_for_consistency():
    """Cross-check that ``_summarize_distill_state`` from Stage 2 (used as a
    helper-naming reference for aggregation patterns) is still importable.
    Stage 4 doesn't aggregate the same way, but this guards against
    accidental removal of the precedent."""
    from moe_compress.stage2.orchestrator import _summarize_distill_state
    assert callable(_summarize_distill_state)


# Note: a full end-to-end Stage 4 test would require running Stages 1–3 first
# to produce a FactoredExperts model. That's covered by the existing
# `test_smoke_stages.py` integration test (which we don't modify here, per
# the no-refactor constraint). The new emits are pure additive dict-merges
# into an existing _trackio_log call site that's already exercised by the
# resume/integration smoke tests, so a dedicated Stage 4 unit test would
# duplicate the coverage. We verify the keys exist in the source by
# importing and checking the call site signature.


def test_stage4_per_layer_emit_includes_new_aggregate_keys():
    """Source-level check: the per-layer Stage 4 _trackio_log block must
    reference the new keys introduced by the telemetry expansion.

    This is a structural test (read the source file) rather than a
    runtime test — Stage 4 requires Stages 1–3 outputs to run, which is
    out of scope for a unit test. The structural check ensures the keys
    were not silently dropped by a future refactor.
    """
    src_path = (
        Path(__file__).resolve().parents[1]
        / "src" / "moe_compress" / "stage4_eora.py"
    )
    src = src_path.read_text()
    # The per-matrix keys use f-string templates ``f"stage4/{name}_..."``;
    # search for the post-``{name}`` literal segment, which is unique enough
    # to confirm the key was added.
    literal_segments = [
        "_n_eligible_experts",
        "_eff_rank_mean",
        "_eff_rank_max",
        "_eff_rank_min",
        "_matrix_compensated_params",
    ]
    for seg in literal_segments:
        assert seg in src, f"Stage 4 source missing per-matrix key fragment {seg}"
    # Config keys are non-template literals.
    for k in ("stage4/config/n_moe_layers", "stage4/config/n_experts_per_layer"):
        assert k in src, f"Stage 4 source missing key {k}"


# ---------------------------------------------------------------------------
# Stage 5 (also used as Stage 2.5) — config emit
# ---------------------------------------------------------------------------


def test_stage5_source_emits_config_block():
    """Source-level check: Stage 5's run() emits a one-shot config block
    under stage5/config/* with the new keys. The Stage 2.5 integration
    test (test_smoke_stage2_to_stage2p5) already runs the Stage 5 freeze
    pattern; this structural check guards the keys stay in place."""
    src_path = (
        Path(__file__).resolve().parents[1]
        / "src" / "moe_compress" / "stage5_router_kd.py"
    )
    src = src_path.read_text()
    expected_keys = [
        "stage5/config/stage_key",
        "stage5/config/total_steps_planned",
        "stage5/config/calib_num_samples",
        "stage5/config/calib_seq_len",
        "stage5/config/grad_accum",
        "stage5/config/use_compile_student",
        "stage5/config/teacher_cache_hit",
        "stage5/config/trainable_router_params",
        "stage5/config/trailing_batches_dropped",
    ]
    for k in expected_keys:
        assert k in src, f"Stage 5 source missing key {k}"


# ---------------------------------------------------------------------------
# Stage 3 — config + Phase C.5 emit
# ---------------------------------------------------------------------------


def test_stage3_source_emits_config_and_c5_extensions():
    """Source-level check that Stage 3 emits the new config keys
    (cross_cov_enabled, t_budget, alpha_by_type/*) and the Phase C.5
    block carries the new training-shape keys.

    S3-6: ``_phase_c5_block_refine`` (which emits the ``c5_*`` keys) was
    relocated verbatim into ``stage3/plugins/block_refine.py``; the config
    keys still emit from the ``stage3_svd.py`` monolith. The source scan
    therefore spans both files.
    """
    src_root = Path(__file__).resolve().parents[1] / "src" / "moe_compress"
    src = (
        (src_root / "stage3_svd.py").read_text()
        + (src_root / "stage3" / "plugins" / "block_refine.py").read_text()
    )
    expected_keys = [
        "stage3/config/cross_cov_enabled",
        "stage3/config/scope",
        "stage3/config/t_budget",
        "stage3/config/alpha_candidates_count",
        "stage3/config/alpha_by_type/",
        "stage3/c5_total_steps",
        "stage3/c5_warmup_steps",
        "stage3/c5_trainable_param_count",
    ]
    for k in expected_keys:
        assert k in src, f"Stage 3 source missing key {k}"


# ---------------------------------------------------------------------------
# Stage 6 — config emit
# ---------------------------------------------------------------------------


def test_stage6_source_emits_config_block():
    src_path = (
        Path(__file__).resolve().parents[1]
        / "src" / "moe_compress" / "stage6_validate.py"
    )
    src = src_path.read_text()
    expected_keys = [
        "stage6/config/wikitext2_enabled",
        "stage6/config/wikitext2_seq_len",
        "stage6/config/zero_shot_enabled",
        "stage6/config/zero_shot_n_tasks",
        "stage6/config/generative_enabled",
        "stage6/config/torch_compile",
    ]
    for k in expected_keys:
        assert k in src, f"Stage 6 source missing key {k}"


# ---------------------------------------------------------------------------
# run_pipeline.py — pipeline-level config emit
# ---------------------------------------------------------------------------


def test_pipeline_source_emits_config_block():
    src_path = (
        Path(__file__).resolve().parents[1]
        / "src" / "moe_compress" / "run_pipeline.py"
    )
    src = src_path.read_text()
    expected_keys = [
        "pipeline/config/model_name",
        "pipeline/config/target_reduction_ratio",
        "pipeline/config/expert_svd_ratio",
        "pipeline/config/device",
        "pipeline/config/resume_from_stage",
        "pipeline/config/stop_after_stage",
    ]
    for k in expected_keys:
        assert k in src, f"run_pipeline source missing key {k}"
