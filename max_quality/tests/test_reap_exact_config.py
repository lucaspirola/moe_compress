"""Pure YAML load smoke test for the REAP-exact preset.

Validates the static contract of `qwen36_35b_a3b_reap_exact.yaml`:
- The new `pipeline:` knobs are present with the expected values.
- Stage 6 mode is self-consistent with `pipeline.evaluator`.
- Stage 2 pure-prune knobs are set as REAP-exact requires.
- The project's calibration source is preserved (not the paper's).
"""
from __future__ import annotations

from pathlib import Path

import yaml


def test_reap_exact_preset_loads():
    config_path = (
        Path(__file__).parent.parent
        / "configs"
        / "qwen36_35b_a3b_reap_exact.yaml"
    )
    cfg = yaml.safe_load(config_path.read_text())

    # New pipeline knobs
    assert cfg["pipeline"]["skip_intermediate_stages"] is True
    assert cfg["pipeline"]["evaluator"] == "stage6alt"

    # Stage 6 mode self-consistency
    assert cfg["stage6_validate"]["mode"] == "thermometer"

    # Stage 2 pure-prune config
    assert cfg["stage2_reap_ream"]["skip_merge_percentile"] == 0.0
    assert cfg["stage2_reap_ream"]["expert_distill_steps"] == 0
    assert cfg["stage2_reap_ream"]["merge_heal_enabled"] is False
    assert cfg["stage2_reap_ream"]["cost_asymmetric"] is False

    # Calibration source preserved (project's, not paper's)
    assert cfg["calibration"]["source"] == "qwen3-pretrain-mix-v2"
    # Sequence length bumped to 4096 in CALIBRATION_MIX_V2_PLAN.md
    # Step 7 (the v2 mix's long reasoning + multi-turn rows benefit
    # from the 2× window).
    assert cfg["calibration"]["sequence_length"] == 4096
