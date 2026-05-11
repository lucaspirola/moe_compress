"""Top-level Config Pydantic tests (LLR-0006, LLR-0011, LLR-0041, LLR-0049)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from kdr.config import Config

_REPO_CONFIGS = (
    Path(__file__).resolve().parent.parent / "configs"
)


def test_bf16_config_loads() -> None:
    raw = yaml.safe_load((_REPO_CONFIGS / "zaya1_8b_bf16.yaml").read_text())
    cfg = Config.model_validate(raw)
    assert cfg.mode == "bf16"
    assert cfg.quant is None
    assert cfg.distillation.optimizer == "adamw_bnb_8bit"
    assert cfg.distillation.eval_every_n_steps == 50
    # auto_batch_size defaults to False on existing configs that pre-date
    # the Phase B schema addition. Confirms the new field is optional and
    # backward-compatible.
    assert cfg.distillation.auto_batch_size is False
    # enable_async_save defaults to False on existing configs that pre-date
    # the Phase C schema addition.
    assert cfg.distillation.enable_async_save is False


def test_recovery_config_loads() -> None:
    """Phase B recovery template — opts into auto_batch_size. Verifies the
    new field round-trips through Pydantic strict validation and that the
    template's other fields are all valid against the schema."""
    raw = yaml.safe_load(
        (_REPO_CONFIGS / "zaya1_8b_bf16_recovery.yaml").read_text()
    )
    cfg = Config.model_validate(raw)
    assert cfg.mode == "bf16"
    assert cfg.quant is None
    assert cfg.distillation.auto_batch_size is True
    assert cfg.distillation.enable_async_save is True  # Phase C opt-in
    # tokens_per_step floor: bs × world × ga × seq. The probe caps bs at
    # bs × ga; preserve the invariant in the template.
    expected_tokens_per_step = (
        cfg.distillation.per_device_batch_size
        * cfg.distillation.gradient_accumulation
        * cfg.distillation.sequence_length
    )
    assert cfg.distillation.total_tokens % expected_tokens_per_step == 0, (
        f"recovery template's total_tokens={cfg.distillation.total_tokens} "
        f"must be a multiple of tokens_per_step={expected_tokens_per_step}."
    )


def test_da_qad_config_loads_with_asymmetric_kv_granularity() -> None:
    raw = yaml.safe_load(
        (_REPO_CONFIGS / "zaya1_8b_da_qad_nvfp4_int4kv.yaml").read_text()
    )
    cfg = Config.model_validate(raw)
    assert cfg.mode == "da_qad"
    assert cfg.quant is not None
    assert cfg.quant.weight.format == "nvfp4"
    # KIVI asymmetric granularity: K per-channel, V per-token
    assert cfg.quant.kv_quant.key.granularity == "channel"
    assert cfg.quant.kv_quant.value.granularity == "token"


def test_config_rejects_invalid_mode() -> None:
    raw = yaml.safe_load(
        (_REPO_CONFIGS / "zaya1_8b_bf16.yaml").read_text()
    )
    raw["mode"] = "BF16"  # wrong case
    with pytest.raises(ValidationError):
        Config.model_validate(raw)


def test_config_rejects_unknown_top_level_field() -> None:
    raw = yaml.safe_load(
        (_REPO_CONFIGS / "zaya1_8b_bf16.yaml").read_text()
    )
    raw["extra_block"] = {"foo": "bar"}
    with pytest.raises(ValidationError):
        Config.model_validate(raw)


def test_config_rejects_zero_eval_every_n_steps() -> None:
    """LLR-0049: `eval_every_n_steps > 0` (Field gt=0)."""
    raw = yaml.safe_load(
        (_REPO_CONFIGS / "zaya1_8b_bf16.yaml").read_text()
    )
    raw["distillation"]["eval_every_n_steps"] = 0
    with pytest.raises(ValidationError):
        Config.model_validate(raw)
