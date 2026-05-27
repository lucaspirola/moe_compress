"""Regression test: pipeline without `pipeline:` key behaves as before.

Verifies that omitting `pipeline.skip_intermediate_stages` (or setting it to
false) preserves the historical full-pipeline behavior: all six stages run,
including Stage 2.5 (router_kd with stage_key=stage2p5) and Stage 5
(router_kd with stage_key=stage5).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class _MockModel:
    class _Cfg:
        name_or_path = "tiny-mock"

    def __init__(self):
        self.config = self._Cfg()


class _MockTokenizer:
    name_or_path = "tiny-mock-tokenizer"
    eos_token_id = 0


@pytest.fixture
def normal_yaml(tmp_path):
    """Minimal config with NO `pipeline:` key (legacy/default behavior)."""
    cfg = {
        "model": {
            "name_or_path": "tiny-mock",
            "revision": "main",
            "torch_dtype": "float32",
            "device_map": "cpu",
            "attn_implementation": "sdpa",
            "load_in_4bit": False,
            "trust_remote_code": False,
        },
        "target": {
            "total_reduction_ratio": 0.30,
            "expert_svd_ratio": 2.0,
        },
        "calibration": {
            "source": "qwen3-pretrain-mix-v2",
            "seed": 1337,
            "num_sequences": 8,
            "sequence_length": 16,
        },
        "stage1_grape": {
            "num_calibration_samples": 4,
            "min_experts_per_layer": 9,
        },
        "stage2_reap_ream": {},
        "stage6_validate": {"mode": "full"},
        "logging": {"level": "INFO"},
    }
    path = tmp_path / "normal.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


@pytest.fixture
def stage_recorder(monkeypatch):
    """Same recorder pattern as test_run_pipeline_reap_exact."""
    from moe_compress import run_pipeline as rp
    from moe_compress.budget import solver as _solver

    calls: dict[str, list] = {
        "stage1": [],
        "stage2": [],
        "stage3": [],
        "stage4": [],
        "stage5_router_kd": [],
        "stage6_validate": [],
        "stage6alt_thermometer": [],
    }

    def _fake_stage1_run(model, tokenizer, config, artifacts_dir, decomposition,
                        device=None, **_kw):
        ad = Path(artifacts_dir)
        bl = ad / "stage1_blacklist.json"
        bg = ad / "stage1_budgets.json"
        bd = ad / "budget_decomposition.json"
        for p in (bl, bg, bd):
            p.write_text(json.dumps({"blacklist": {}}))
        calls["stage1"].append(True)
        return bl, bg

    def _fake_stage2_run(model, tokenizer, config, artifacts_dir, device=None, **_kw):
        ad = Path(artifacts_dir)
        (ad / "stage2_pruned").mkdir(parents=True, exist_ok=True)
        calls["stage2"].append(True)

    def _fake_stage3(*a, **k):
        calls["stage3"].append(True)

    def _fake_stage4(*a, **k):
        calls["stage4"].append(True)

    def _fake_stage5_router_kd(*a, **k):
        calls["stage5_router_kd"].append(k.get("stage_key", "stage5"))

    def _fake_stage6_validate(*a, **k):
        calls["stage6_validate"].append(True)

    def _fake_stage6alt(*a, **k):
        calls["stage6alt_thermometer"].append(True)

    def _fake_solve(*_a, **_k):
        return _solver.BudgetDecomposition(
            total_reduction_ratio=0.30,
            expert_prune_ratio=0.20,
            svd_rank_ratio=0.10,
            global_expert_budget=10,
            min_experts_per_layer=9,
            blacklisted_experts={},
        )

    def _fake_load_for_stage(stage, config, artifacts_dir, *, stop_after_stage=6,
                             load_from_override=None):
        return _MockModel(), _MockTokenizer()

    def _noop_save_json(payload, path):
        Path(path).write_text(json.dumps(payload, default=str))

    monkeypatch.setattr(rp.stage1, "run", _fake_stage1_run)
    monkeypatch.setattr(rp.stage2, "run", _fake_stage2_run)
    monkeypatch.setattr(rp.stage3_svd, "run", _fake_stage3)
    monkeypatch.setattr(rp.stage4_eora, "run", _fake_stage4)
    monkeypatch.setattr(rp.stage5_router_kd, "run", _fake_stage5_router_kd)
    monkeypatch.setattr(rp.stage6_validate, "run", _fake_stage6_validate)
    monkeypatch.setattr(rp.stage6alt_thermometer, "run", _fake_stage6alt)
    monkeypatch.setattr(rp.budget_solver, "solve", _fake_solve)
    monkeypatch.setattr(rp, "_load_for_stage", _fake_load_for_stage)
    monkeypatch.setattr(rp, "save_json_artifact", _noop_save_json)
    monkeypatch.setattr(rp, "_validate_stage1_artifacts", lambda *_a, **_k: None)
    monkeypatch.setattr(rp, "upload_stage_to_hub", lambda *_a, **_k: None)
    monkeypatch.setattr(rp, "hub_repo_base_from_env", lambda: None)
    monkeypatch.setattr(rp, "wait_for_pending_uploads", lambda: None)
    monkeypatch.setattr(rp, "_finish_stage", lambda *_a, **_k: None)
    return calls


def test_normal_mode_runs_all_stages(
    normal_yaml, stage_recorder, tmp_path
):
    """Without a `pipeline:` section, the pipeline runs all six stages."""
    from moe_compress import run_pipeline as rp

    rc = rp.main([
        "--config", str(normal_yaml),
        "--artifacts-dir", str(tmp_path / "artifacts"),
    ])

    assert rc == 0
    assert len(stage_recorder["stage1"]) == 1
    assert len(stage_recorder["stage2"]) == 1
    # Both stage_key invocations of stage5_router_kd must fire: stage2p5 and stage5.
    assert "stage2p5" in stage_recorder["stage5_router_kd"], (
        f"Stage 2.5 must run in normal mode; got {stage_recorder['stage5_router_kd']}"
    )
    assert "stage5" in stage_recorder["stage5_router_kd"], (
        f"Stage 5 must run in normal mode; got {stage_recorder['stage5_router_kd']}"
    )
    assert len(stage_recorder["stage3"]) == 1, "Stage 3 must run in normal mode"
    assert len(stage_recorder["stage4"]) == 1, "Stage 4 must run in normal mode"
    # Default stage6_validate.mode is "full" in this fixture -> stage6_validate
    # is invoked, not stage6alt.
    assert len(stage_recorder["stage6_validate"]) == 1
    assert len(stage_recorder["stage6alt_thermometer"]) == 0


def test_explicit_skip_intermediate_false_runs_all_stages(
    normal_yaml, stage_recorder, tmp_path
):
    """Explicit `pipeline.skip_intermediate_stages: false` also runs everything."""
    from moe_compress import run_pipeline as rp

    cfg = yaml.safe_load(normal_yaml.read_text())
    cfg["pipeline"] = {"skip_intermediate_stages": False}
    normal_yaml.write_text(yaml.safe_dump(cfg))

    rc = rp.main([
        "--config", str(normal_yaml),
        "--artifacts-dir", str(tmp_path / "artifacts"),
    ])

    assert rc == 0
    assert len(stage_recorder["stage3"]) == 1
    assert len(stage_recorder["stage4"]) == 1
    assert "stage5" in stage_recorder["stage5_router_kd"]
