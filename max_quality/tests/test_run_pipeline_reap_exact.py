"""Orchestration smoke test for pipeline.skip_intermediate_stages (REAP-exact).

These tests verify the run_pipeline.main() control flow when the YAML config
turns on `pipeline.skip_intermediate_stages: true`:

- Stage 1 + Stage 2 run as normal.
- Stage 2.5 (the stage5_router_kd.run() invocation with stage_key="stage2p5")
  is suppressed.
- Stages 3, 4, 5 are all suppressed.
- Stage 6 runs - either stage6alt_thermometer.run (evaluator=stage6alt, the
  default) or stage6_validate.run (evaluator=stage6, opt-in).

CPU-only: every heavy callable is monkeypatched to a recorder.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class _MockModel:
    """Stand-in for a HF causal-LM. Just satisfies isinstance/getattr checks."""

    class _Cfg:
        name_or_path = "tiny-mock"

    def __init__(self):
        self.config = self._Cfg()


class _MockTokenizer:
    name_or_path = "tiny-mock-tokenizer"
    eos_token_id = 0


@pytest.fixture
def reap_exact_yaml(tmp_path):
    """Write a minimal but valid YAML config with pipeline.skip_intermediate_stages on."""
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
            "source": "qwen3-pretrain-mix",
            "seed": 1337,
            "num_sequences": 8,
            "sequence_length": 16,
        },
        "stage1_grape": {
            "num_calibration_samples": 4,
            "min_experts_per_layer": 9,
        },
        "stage2_reap_ream": {
            "skip_merge_percentile": 0.0,
            "expert_distill_steps": 0,
            "merge_heal_enabled": False,
            "cost_asymmetric": False,
        },
        "stage6_validate": {
            "mode": "full",
        },
        "logging": {"level": "INFO"},
        "pipeline": {
            "skip_intermediate_stages": True,
            "evaluator": "stage6alt",
        },
    }
    path = tmp_path / "reap_exact.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


@pytest.fixture
def stage_recorder(monkeypatch, tmp_path):
    """Patch out every heavy callable in run_pipeline and record invocations."""
    from moe_compress import run_pipeline as rp
    from moe_compress.budget import solver as _solver

    calls: dict[str, list] = {
        "stage1": [],
        "stage2": [],
        "stage3": [],
        "stage4": [],
        "stage5_router_kd": [],  # list of stage_key strings
        "stage6_validate": [],
        "stage6alt_thermometer": [],
    }

    # Stage 1 returns (blacklist_path, budgets_path). The pipeline reads
    # the blacklist JSON back via load_json_artifact, so we drop a real
    # JSON on disk before returning the paths.
    def _fake_stage1_run(model, tokenizer, config, artifacts_dir, decomposition,
                        device=None, **_kw):
        ad = Path(artifacts_dir)
        blacklist_path = ad / "stage1_blacklist.json"
        budgets_path = ad / "stage1_budgets.json"
        decomp_path = ad / "budget_decomposition.json"
        for p in (blacklist_path, budgets_path, decomp_path):
            p.write_text(json.dumps({"blacklist": {}}))
        calls["stage1"].append(True)
        return blacklist_path, budgets_path

    def _fake_stage2_run(model, tokenizer, config, artifacts_dir, device=None, **_kw):
        ad = Path(artifacts_dir)
        # Drop a stage2_pruned/ directory so the stage 6 loader finds it.
        (ad / "stage2_pruned").mkdir(parents=True, exist_ok=True)
        calls["stage2"].append(True)
        return ad / "stage2_pruned"

    def _fake_stage3(*a, **k):
        calls["stage3"].append(True)

    def _fake_stage4(*a, **k):
        calls["stage4"].append(True)

    def _fake_stage5_router_kd(*a, **k):
        # stage_key disambiguates the Stage-2.5 vs Stage-5 invocations.
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


def test_reap_exact_skips_intermediate_and_runs_stage6alt(
    reap_exact_yaml, stage_recorder, tmp_path
):
    """pipeline.skip_intermediate_stages=true + evaluator=stage6alt -> only
    stages 1/2 + stage6alt run; 2.5/3/4/5 + stage6_validate are skipped."""
    from moe_compress import run_pipeline as rp

    artifacts = tmp_path / "artifacts"
    rc = rp.main([
        "--config", str(reap_exact_yaml),
        "--artifacts-dir", str(artifacts),
    ])

    assert rc == 0
    assert len(stage_recorder["stage1"]) == 1, "Stage 1 must run"
    assert len(stage_recorder["stage2"]) == 1, "Stage 2 must run"
    # Stage 2.5 is the stage5_router_kd.run() with stage_key="stage2p5"
    assert "stage2p5" not in stage_recorder["stage5_router_kd"], (
        f"Stage 2.5 must NOT run; got router_kd calls={stage_recorder['stage5_router_kd']}"
    )
    assert len(stage_recorder["stage3"]) == 0, "Stage 3 must NOT run"
    assert len(stage_recorder["stage4"]) == 0, "Stage 4 must NOT run"
    # Stage 5 is the stage5_router_kd.run() with stage_key="stage5"
    assert "stage5" not in stage_recorder["stage5_router_kd"], (
        f"Stage 5 must NOT run; got router_kd calls={stage_recorder['stage5_router_kd']}"
    )
    assert len(stage_recorder["stage6alt_thermometer"]) == 1, (
        "Stage 6alt thermometer must run (evaluator=stage6alt)"
    )
    assert len(stage_recorder["stage6_validate"]) == 0, (
        "Stage 6_validate must NOT run when evaluator=stage6alt"
    )


def test_reap_exact_evaluator_stage6_runs_stage6_validate(
    reap_exact_yaml, stage_recorder, tmp_path
):
    """pipeline.skip_intermediate_stages=true + evaluator=stage6 -> stage6_validate
    runs (the orchestrator forces mode=full), thermometer does not."""
    from moe_compress import run_pipeline as rp

    # Mutate the YAML in place: flip evaluator to stage6.
    cfg = yaml.safe_load(reap_exact_yaml.read_text())
    cfg["pipeline"]["evaluator"] = "stage6"
    reap_exact_yaml.write_text(yaml.safe_dump(cfg))

    artifacts = tmp_path / "artifacts"
    rc = rp.main([
        "--config", str(reap_exact_yaml),
        "--artifacts-dir", str(artifacts),
    ])

    assert rc == 0
    assert len(stage_recorder["stage6_validate"]) == 1, (
        "Stage 6_validate must run when evaluator=stage6"
    )
    assert len(stage_recorder["stage6alt_thermometer"]) == 0, (
        "Stage 6alt thermometer must NOT run when evaluator=stage6"
    )
    # Intermediates remain skipped regardless of evaluator.
    assert len(stage_recorder["stage3"]) == 0
    assert len(stage_recorder["stage4"]) == 0


def test_reap_exact_rejects_bad_evaluator(reap_exact_yaml, stage_recorder, tmp_path):
    """pipeline.evaluator must be 'stage6' or 'stage6alt'."""
    from moe_compress import run_pipeline as rp

    cfg = yaml.safe_load(reap_exact_yaml.read_text())
    cfg["pipeline"]["evaluator"] = "stage7-fictional"
    reap_exact_yaml.write_text(yaml.safe_dump(cfg))

    with pytest.raises(ValueError, match="pipeline.evaluator"):
        rp.main([
            "--config", str(reap_exact_yaml),
            "--artifacts-dir", str(tmp_path / "artifacts"),
        ])
