"""Tests for `kdr.cli.train` argparse + dispatch wiring.

# VERIFIES: LLR-0008
# VERIFIES: LLR-0034
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from kdr.cli.train import _load_config, _parse, _resolve_source_metadata_path


def _minimal_yaml(tmp_path: Path) -> Path:
    """Write a minimum valid kdr YAML to tmp_path/config.yaml; return path."""
    yaml_text = """
mode: bf16
teacher:
  name_or_path: hf-internal-testing/tiny-random-gpt2
  revision: main
  torch_dtype: bfloat16
  attn_implementation: sdpa
student:
  source: hf-internal-testing/tiny-random-gpt2
  torch_dtype: bfloat16
  attn_implementation: sdpa
calibration:
  source: pirola/calibration-cascade-2-sft
  dataset: nemotron-cascade-2-sft
  seed: 42
  num_sequences: 256
  sequence_length: 2048
  subset_weights: { en: 0.5, code: 0.5 }
  ptq_subset_size: 256
distillation:
  loss: forward_kld
  temperature: 1.0
  optimizer: adamw_bnb_8bit
  learning_rate: 3.0e-5
  min_learning_rate: 3.0e-7
  weight_decay: 0.0
  betas: [0.9, 0.95]
  grad_clip_norm: 1.0
  warmup_steps: 10
  total_tokens: 1000000
  per_device_batch_size: 1
  gradient_accumulation: 1
  sequence_length: 2048
  log_every_n_steps: 1
  eval_every_n_steps: 10
  save_every_n_steps: 0
  trainable_scope: full
  use_gradient_checkpointing: true
eval:
  wikitext2:
    enabled: true
    sequence_length: 2048
    num_sequences: 64
"""
    p = tmp_path / "config.yaml"
    p.write_text(yaml_text)
    return p


# REQ: VERIFIES: LLR-0008
def test_parse_requires_config_and_artifacts_dir() -> None:
    with pytest.raises(SystemExit):
        _parse(["--student", "X"])
    with pytest.raises(SystemExit):
        _parse(["--config", "x.yaml"])


# REQ: VERIFIES: LLR-0008
def test_parse_accepts_optional_mode_override() -> None:
    args = _parse(
        [
            "--config",
            "x.yaml",
            "--artifacts-dir",
            ".",
            "--mode",
            "da_qad",
        ]
    )
    assert args.mode == "da_qad"


# REQ: VERIFIES: LLR-0008
def test_parse_rejects_invalid_mode() -> None:
    with pytest.raises(SystemExit):
        _parse(
            [
                "--config",
                "x.yaml",
                "--artifacts-dir",
                ".",
                "--mode",
                "fp16",
            ]
        )


# REQ: VERIFIES: LLR-0008
def test_parse_mode_defaults_to_yaml_when_omitted() -> None:
    """Per LLR-0008 AC: --mode defaults to whatever the YAML specifies and
    overrides if given. When omitted on the CLI, the parsed Namespace's
    `.mode` is None, signalling "use YAML's value"."""
    args = _parse(["--config", "x.yaml", "--artifacts-dir", "."])
    assert args.mode is None


# REQ: VERIFIES: LLR-0034
def test_parse_resume_from_optional() -> None:
    args = _parse(["--config", "x.yaml", "--artifacts-dir", "."])
    assert args.resume_from is None
    args = _parse(
        ["--config", "x.yaml", "--artifacts-dir", ".", "--resume-from", "./foo"]
    )
    assert args.resume_from == "./foo"


# REQ: VERIFIES: LLR-0034
def test_resume_from_missing_path_raises_filenotfound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """LLR-0034: a missing --resume-from path raises FileNotFoundError."""
    from kdr.cli import train as cli

    cfg = _minimal_yaml(tmp_path)
    artifacts = tmp_path / "artifacts"

    # Stub out the heavy machinery — we only exercise the resume path here.
    monkeypatch.setattr(cli, "Accelerator", lambda: _stub_accelerator())
    monkeypatch.setattr(cli, "Zaya1Adapter", lambda: object())
    monkeypatch.setattr(cli, "_build_calibration_batches", lambda c, a: [])
    monkeypatch.setattr(cli, "run_recovery", lambda **kw: artifacts)

    with pytest.raises(FileNotFoundError):
        cli.main(
            [
                "--config",
                str(cfg),
                "--artifacts-dir",
                str(artifacts),
                "--resume-from",
                str(tmp_path / "does_not_exist"),
            ]
        )


def _stub_accelerator() -> Any:
    from unittest.mock import MagicMock

    a = MagicMock()
    a.is_main_process = True
    a.num_processes = 1
    return a


def test_load_config_validates_with_pydantic(tmp_path: Path) -> None:
    cfg_path = _minimal_yaml(tmp_path)
    config = _load_config(cfg_path)
    assert config.mode == "bf16"
    assert config.distillation.eval_every_n_steps == 10


def test_load_config_rejects_non_mapping_yaml(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("- just a list\n- not a mapping\n")
    with pytest.raises(ValueError, match="did not parse to a mapping"):
        _load_config(p)


def test_resolve_source_metadata_path_returns_path_when_present(
    tmp_path: Path,
) -> None:
    src = tmp_path / "student"
    src.mkdir()
    (src / "compressed_metadata.json").write_text("{}")
    p = _resolve_source_metadata_path(str(src))
    assert p == src / "compressed_metadata.json"


def test_resolve_source_metadata_path_returns_none_for_hub_repo() -> None:
    # An HF repo id like "Zyphra/ZAYA1-reasoning-base" is not a local
    # filesystem path → no compressed_metadata.json on disk.
    assert _resolve_source_metadata_path("Zyphra/ZAYA1-reasoning-base") is None
