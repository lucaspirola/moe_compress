"""Pure YAML load smoke test for the faithful-REAP-prune preset (§6 test 4).

Validates the static contract of ``qwen36_35b_a3b_reap_faithful.yaml``:
- ``prune_mode == "faithful_prune"`` and ``prune_fraction`` present + in (0,1).
- The merge knobs are at their INERT defaults (faithful mode bypasses them).
- Pipeline skip / evaluator are set as the screening tool requires.
- Calibration source is preserved (the project's, not the paper's).
"""
from __future__ import annotations

from pathlib import Path

import yaml


def _load():
    config_path = (
        Path(__file__).parent.parent
        / "configs"
        / "qwen36_35b_a3b_reap_faithful.yaml"
    )
    return yaml.safe_load(config_path.read_text())


def test_faithful_preset_loads():
    cfg = _load()
    s2 = cfg["stage2_reap_ream"]

    # Faithful-prune flags
    assert s2["prune_mode"] == "faithful_prune"
    assert "prune_fraction" in s2
    assert 0.0 < float(s2["prune_fraction"]) < 1.0

    # New pipeline knobs (stages 1+2 → stage6alt)
    assert cfg["pipeline"]["skip_intermediate_stages"] is True
    assert cfg["pipeline"]["evaluator"] == "stage6alt"
    assert cfg["stage6_validate"]["mode"] == "thermometer"


def test_faithful_preset_merge_knobs_inert():
    """The merge machinery knobs stay at their inert defaults so faithful mode
    does not accidentally combine with the merge path (the orchestrator also
    rejects the contradictory combo at run() entry)."""
    s2 = _load()["stage2_reap_ream"]
    assert s2["expert_distill_steps"] == 0
    assert s2["merge_heal_enabled"] is False
    assert s2["cost_asymmetric"] is False


def test_faithful_preset_calibration_source_preserved():
    cfg = _load()
    assert cfg["calibration"]["source"] == "qwen3-pretrain-mix-v2"
