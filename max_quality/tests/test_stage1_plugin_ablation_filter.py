"""Unit tests for ``moe_compress.stage1.plugins.ablation_filter`` (sub-task 5).

Verifies:

1. The plugin's class-level Protocol attributes match the registry
   contract (``StagePlugin``).
2. ``is_enabled`` correctly reads
   ``config["stage1_grape"]["ablation_filter"]["enabled"]`` (default True).
3. ``run`` populates the five write slots from the legacy
   ``run_ablation_filter`` triple plus the two private bookkeeping slots
   consumed by ``contribute_artifact``.
4. The disabled-path fallback (no ablation work) preserves the legacy
   semantics: candidate set used verbatim, empty deltas, baseline_nll=0.0.
5. ``contribute_artifact`` returns the canonical six-key payload byte-
   equivalent to the legacy ``_write_ablation_filter_artifact`` writer.
6. The orchestrator-overwrite path (Phase-C state mixed into the config
   dict) flows through ``contribute_artifact`` unchanged.
7. Missing required slots cause ``KeyError`` so the orchestrator
   (sub-task 10) gets a clear contract violation rather than a silent
   misbehaviour.
8. The deprecated v5 ``run_phase_f`` entry point is still importable
   from the plugin module — back-compat with any out-of-tree caller.
"""

from __future__ import annotations

import pytest

from moe_compress.stage1._framework.plugin import StagePlugin
from moe_compress.stage1.context import Stage1Context
from moe_compress.stage1.plugins.ablation_filter import (
    AblationFilterPlugin,
    _apply_threshold_filter,
    _write_ablation_filter_artifact,
    run_ablation_filter,
)


# ---------------------------------------------------------------------------
# Shared fixtures (kept private — the disabled-path tests do not need a real
# model/tokenizer/forward-pass; only ``candidates`` + the ``enabled=False``
# config flag are populated).
# ---------------------------------------------------------------------------


def _disabled_ctx(candidates: dict | None = None) -> Stage1Context:
    """Build a populated ``Stage1Context`` for the disabled-path tests.

    The disabled-path short-circuit in ``run_ablation_filter`` does not
    touch model/tokenizer/artifacts_dir/device, so they can all be
    ``None``.
    """
    if candidates is None:
        candidates = {
            (10, 0): ["aimer"],
            (10, 1): ["sink_token"],
            (5, 3): ["magnitude_topk"],
        }
    ctx = Stage1Context()
    ctx.set("candidates", candidates)
    ctx.set("model", None)
    ctx.set("tokenizer", None)
    ctx.set(
        "config",
        {"stage1_grape": {"ablation_filter": {"enabled": False}}, "calibration": {}},
    )
    ctx.set("artifacts_dir", None)
    ctx.set("device", None)
    return ctx


# ---------------------------------------------------------------------------
# Protocol-attribute contract
# ---------------------------------------------------------------------------


def test_plugin_protocol_attributes():
    p = AblationFilterPlugin()
    assert p.name == "ablation_filter"
    assert p.paper.startswith("Stage 1 ALGORITHM_REFERENCE")
    assert p.config_key == "stage1_grape.ablation_filter"
    assert p.reads == ("candidates", "model", "tokenizer", "config", "artifacts_dir", "device")
    assert p.writes == (
        "blacklist",
        "candidate_deltas",
        "baseline_nll",
        "ablation_filter_threshold",
        "ablation_filter_config",
    )
    assert p.accumulators == ()


def test_plugin_is_runtime_checkable_stageplugin():
    assert isinstance(AblationFilterPlugin(), StagePlugin)


# ---------------------------------------------------------------------------
# ``is_enabled`` config branching
# ---------------------------------------------------------------------------


def test_plugin_is_enabled_default_true():
    assert AblationFilterPlugin().is_enabled({}) is True


def test_plugin_is_enabled_explicit_false():
    cfg = {"stage1_grape": {"ablation_filter": {"enabled": False}}}
    assert AblationFilterPlugin().is_enabled(cfg) is False


def test_plugin_is_enabled_explicit_true():
    cfg = {"stage1_grape": {"ablation_filter": {"enabled": True}}}
    assert AblationFilterPlugin().is_enabled(cfg) is True


# ---------------------------------------------------------------------------
# ``run`` — disabled path (no model/forward-pass needed)
# ---------------------------------------------------------------------------


def test_plugin_run_disabled_path_uses_candidate_set_verbatim():
    """``enabled=False`` falls back to using the candidate set as the blacklist."""
    ctx = _disabled_ctx()
    AblationFilterPlugin().run(ctx)

    assert ctx.get("blacklist") == {5: [3], 10: [0, 1]}
    assert ctx.get("candidate_deltas") == {}
    assert ctx.get("baseline_nll") == 0.0


def test_plugin_run_disabled_writes_threshold_and_config_slots():
    """The two private bookkeeping slots get default values when ``enabled=False``."""
    ctx = _disabled_ctx()
    AblationFilterPlugin().run(ctx)

    assert ctx.get("ablation_filter_threshold") == 0.001
    assert ctx.get("ablation_filter_config") == {
        "holdout_samples": 100,
        "ablation_filter_threshold": 0.001,
        "ablation_filter_batch_size": 32,
    }


def test_plugin_run_does_not_mutate_candidates_dict():
    """The plugin must not in-place edit the input candidates dict."""
    original = {
        (10, 0): ["aimer"],
        (10, 1): ["sink_token"],
        (5, 3): ["magnitude_topk"],
    }
    ctx = _disabled_ctx(candidates=original)
    snapshot = {k: list(v) for k, v in original.items()}

    AblationFilterPlugin().run(ctx)

    # Same object, same content.
    assert ctx.get("candidates") is original
    assert original == snapshot


def test_plugin_run_enabled_path_wires_run_ablation_filter_return(monkeypatch):
    """When ``enabled=True`` the plugin invokes :func:`run_ablation_filter` and
    wires the returned triple into the three public write slots verbatim."""
    captured = {}

    def fake_run_ablation_filter(
        model, tokenizer, config, artifacts_dir, *, candidates, device
    ):
        captured["model"] = model
        captured["tokenizer"] = tokenizer
        captured["config"] = config
        captured["artifacts_dir"] = artifacts_dir
        captured["candidates"] = candidates
        captured["device"] = device
        return ({0: [1]}, {(0, 1): 0.5}, 1.0)

    monkeypatch.setattr(
        "moe_compress.stage1.plugins.ablation_filter.run_ablation_filter",
        fake_run_ablation_filter,
    )

    ctx = Stage1Context()
    ctx.set("candidates", {(0, 1): ["aimer"]})
    ctx.set("model", "MODEL")
    ctx.set("tokenizer", "TOK")
    ctx.set(
        "config",
        {"stage1_grape": {"ablation_filter": {"enabled": True}}, "calibration": {}},
    )
    ctx.set("artifacts_dir", "ARTIFACTS")
    ctx.set("device", "DEV")

    AblationFilterPlugin().run(ctx)

    # The plugin forwarded every input.
    assert captured["model"] == "MODEL"
    assert captured["tokenizer"] == "TOK"
    assert captured["artifacts_dir"] == "ARTIFACTS"
    assert captured["device"] == "DEV"
    assert captured["candidates"] == {(0, 1): ["aimer"]}

    # The triple was wired into the public write slots.
    assert ctx.get("blacklist") == {0: [1]}
    assert ctx.get("candidate_deltas") == {(0, 1): 0.5}
    assert ctx.get("baseline_nll") == 1.0
    # Default threshold (no ``blacklist_threshold`` key in the config).
    assert ctx.get("ablation_filter_threshold") == 0.001


# ---------------------------------------------------------------------------
# ``contribute_artifact`` — six-key payload schema + byte-equivalence
# ---------------------------------------------------------------------------


def test_plugin_contribute_artifact_six_keys():
    ctx = _disabled_ctx()
    plugin = AblationFilterPlugin()
    plugin.run(ctx)
    payload = plugin.contribute_artifact(ctx)

    assert set(payload.keys()) == {
        "baseline_mean_nll",
        "ablation_filter_threshold",
        "candidate_count",
        "blacklist_count",
        "candidates",
        "config",
    }
    assert isinstance(payload["baseline_mean_nll"], float)
    assert isinstance(payload["ablation_filter_threshold"], float)
    assert isinstance(payload["candidate_count"], int)
    assert isinstance(payload["blacklist_count"], int)
    assert isinstance(payload["candidates"], dict)
    assert isinstance(payload["config"], dict)


def test_plugin_contribute_artifact_candidates_inner_shape():
    """Every ``candidates[L<li>E<e>]`` entry has exactly the three inner keys."""
    ctx = _disabled_ctx()
    plugin = AblationFilterPlugin()
    plugin.run(ctx)
    payload = plugin.contribute_artifact(ctx)

    assert set(payload["candidates"].keys()) == {"L10E0", "L10E1", "L5E3"}
    for entry in payload["candidates"].values():
        assert set(entry.keys()) == {"delta_nll", "provenance", "passed_filter"}
        # Disabled path: no ablation, so delta_nll is None.
        assert entry["delta_nll"] is None
        # Disabled-path fallback uses every candidate, so passed_filter is True.
        assert entry["passed_filter"] is True
    assert payload["candidates"]["L10E0"]["provenance"] == ["aimer"]
    assert payload["candidates"]["L10E1"]["provenance"] == ["sink_token"]
    assert payload["candidates"]["L5E3"]["provenance"] == ["magnitude_topk"]


def test_plugin_writes_string_keys_in_candidates_payload():
    """Keys must use the canonical ``L{li}E{e}`` format (no zero-padding,
    no namespace prefix) — load-bearing for golden-snapshot byte equality."""
    ctx = _disabled_ctx()
    plugin = AblationFilterPlugin()
    plugin.run(ctx)
    payload = plugin.contribute_artifact(ctx)

    for key in payload["candidates"]:
        # Match e.g. "L10E0", "L5E3" — no padding, no namespace.
        assert key.startswith("L")
        assert "E" in key
        li_str, e_str = key[1:].split("E", 1)
        assert li_str.isdigit()
        assert e_str.isdigit()


def test_plugin_contribute_artifact_byte_equivalent_to_legacy_writer(tmp_path):
    """The plugin's ``contribute_artifact`` payload must equal the dict
    that the legacy :func:`_write_ablation_filter_artifact` would have
    constructed from the same inputs — byte-anchor for the JSON schema."""
    ctx = _disabled_ctx()
    plugin = AblationFilterPlugin()
    plugin.run(ctx)
    payload = plugin.contribute_artifact(ctx)

    # Build the reference payload by calling the module-level helper
    # directly and reading the JSON it would have written — a genuine
    # plugin-class vs module-level-helper byte-anchor test.
    import json

    from moe_compress.stage1.plugins import ablation_filter as _legacy

    out_path = _legacy._write_ablation_filter_artifact(
        tmp_path,
        candidates=ctx.get("candidates"),
        deltas=ctx.get("candidate_deltas"),
        baseline_nll=ctx.get("baseline_nll"),
        threshold=ctx.get("ablation_filter_threshold"),
        blacklist=ctx.get("blacklist"),
        config_dict=ctx.get("ablation_filter_config"),
    )
    legacy_dict = json.loads(out_path.read_text())

    # JSON round-trip on the plugin payload to neutralise tuple/list and key
    # type differences (the legacy helper goes through json.dump which sorts
    # keys and stringifies; the comparison should be on canonical JSON).
    plugin_dict = json.loads(json.dumps(payload, sort_keys=True))
    legacy_dict_norm = json.loads(json.dumps(legacy_dict, sort_keys=True))

    assert plugin_dict == legacy_dict_norm


def test_plugin_contribute_artifact_respects_config_overwrite():
    """The orchestrator-overwrite path (§4.2.3 of the plan): a five-key
    Phase-C-augmented config dict set on the ctx after ``run()`` flows
    through ``contribute_artifact`` unchanged."""
    ctx = _disabled_ctx()
    plugin = AblationFilterPlugin()
    plugin.run(ctx)

    augmented = {
        "holdout_samples": 50,
        "magnitude_topk_per_l_layer": 7,
        "ablation_filter_threshold": 0.001,
        "ablation_filter_batch_size": 16,
        "ma_formation_layers": [5, 9],
    }
    ctx.set("ablation_filter_config", augmented, overwrite=True)

    payload = plugin.contribute_artifact(ctx)
    assert payload["config"] == augmented


# ---------------------------------------------------------------------------
# Missing-slot KeyError per slot
# ---------------------------------------------------------------------------


def _ctx_missing(slot: str) -> Stage1Context:
    """Build a fully-populated disabled-path ctx, then drop the named slot."""
    ctx = _disabled_ctx()
    new = Stage1Context()
    for k in ctx.keys():
        if k == slot:
            continue
        new.set(k, ctx.get(k))
    return new


@pytest.mark.parametrize(
    "missing",
    ["candidates", "model", "tokenizer", "config", "artifacts_dir", "device"],
)
def test_plugin_run_rejects_missing_slot(missing):
    """Every read slot is required: dropping it must raise ``KeyError``."""
    ctx = _ctx_missing(missing)
    with pytest.raises(KeyError) as excinfo:
        AblationFilterPlugin().run(ctx)
    assert missing in str(excinfo.value)


# ---------------------------------------------------------------------------
# Back-compat: deprecated v5 entry point + threshold-filter helper still
# importable from the plugin module
# ---------------------------------------------------------------------------


def test_run_phase_f_remains_importable_via_plugin_module():
    """``run_phase_f`` survives the migration (zero in-tree callers, but
    re-exported via the shim per §2.5 of the plan)."""
    from moe_compress.stage1.plugins.ablation_filter import run_phase_f

    assert callable(run_phase_f)


def test_apply_threshold_filter_reachable_from_plugin_module():
    """The threshold-filter helper is the byte-anchor for the disabled-path
    blacklist semantics; it must reach the plugin module as well as the
    legacy shim."""
    deltas = {(10, 5): 0.005, (10, 9): 0.0005}
    bl = _apply_threshold_filter(deltas, threshold=0.001)
    assert bl == {10: [5]}
