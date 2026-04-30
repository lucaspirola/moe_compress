"""Tests for --no-resume CLI flag and _validate_stage1_artifacts."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_no_resume_arg_parses():
    """--no-resume must be accepted by argument parser."""
    from moe_compress import run_pipeline
    args = run_pipeline._parse([
        "--config", "configs/qwen36_35b_a3b_30pct.yaml",
        "--no-resume",
    ])
    assert args.no_resume is True


def test_no_resume_default_false():
    """--no-resume must default to False."""
    from moe_compress import run_pipeline
    args = run_pipeline._parse([
        "--config", "configs/qwen36_35b_a3b_30pct.yaml",
    ])
    assert args.no_resume is False


def test_validate_stage1_artifacts_missing(tmp_path):
    """_validate_stage1_artifacts raises FileNotFoundError when files are absent."""
    from moe_compress.run_pipeline import _validate_stage1_artifacts
    with pytest.raises(FileNotFoundError, match="stage1"):
        _validate_stage1_artifacts(tmp_path)


def test_validate_stage1_artifacts_corrupt(tmp_path):
    """_validate_stage1_artifacts raises RuntimeError on truncated JSON."""
    from moe_compress.run_pipeline import _validate_stage1_artifacts
    for name in ["stage1_blacklist.json", "stage1_budgets.json", "budget_decomposition.json"]:
        (tmp_path / name).write_text('{"valid": true}')
    (tmp_path / "stage1_budgets.json").write_text("{truncated")
    with pytest.raises(RuntimeError, match="corrupt"):
        _validate_stage1_artifacts(tmp_path)


def test_validate_stage1_artifacts_passes(tmp_path):
    """_validate_stage1_artifacts passes when all files exist and are valid JSON."""
    from moe_compress.run_pipeline import _validate_stage1_artifacts
    for name in ["stage1_blacklist.json", "stage1_budgets.json", "budget_decomposition.json"]:
        (tmp_path / name).write_text(json.dumps({"ok": True}))
    _validate_stage1_artifacts(tmp_path)  # must not raise
