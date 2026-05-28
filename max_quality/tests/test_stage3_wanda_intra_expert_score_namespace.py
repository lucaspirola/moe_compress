"""W-2 namespace contract tests for the Wanda intra-expert score plugin.

Audit W-2 (MEDIUM, 2026-05-29) moved the plugin's score-map sidecar from
the legacy ``artifacts_dir/_stage3_wanda_intra_expert_score.pt`` layout
to the F-H-7 namespaced layout
``<jsonl.parent>/sidecars/<jsonl.stem>/wanda_intra_expert_score.pt`` —
the same path-derivation contract used by every other Stage 3
calibration sidecar.

This file hosts the W-2 *new-path-contract* tests:

* **T2** — the path materialised by the writer matches the byte-equal
  value returned by
  :func:`moe_compress.utils.cached_calibration_signals.sidecar_path`
  with signal name ``"wanda_intra_expert_score"``. This is the
  single-source-of-truth lock — if the helper changes layout, the writer
  follows automatically.
* **T3** — collision-resistance: two Stage 3 runs that share an
  ``artifacts_dir`` but produce DIFFERENT calibration JSONL stems land
  in DIFFERENT namespaced subdirectories. This is the regression
  reproducer for the audit's collision scenario; pre-W-2 both runs
  silently overwrote the same legacy file.
* **T4** — graceful skip when ``calibration_jsonl_path`` is absent
  from ctx. The plugin still publishes its in-memory score map but
  declines to write a sidecar and emits a WARNING so operators see the
  actionable cause (typical case: invoking the plugin outside the
  Stage 3 orchestrator, e.g. from a standalone analysis script).

Tests T1 (inline writer smoke) and T5 (legacy → namespaced update) live
in the sibling ``test_stage3_wanda_intra_expert_score.py`` file —
co-located with the rest of the plugin's protocol / math / orchestrator
tests because they exercise the existing-test surface, not the new
path contract per se.
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch

from moe_compress.pipeline.context import PipelineContext
from moe_compress.stage3.plugins.wanda_intra_expert_score import (
    _ARTIFACT_FORMAT_VERSION,
    WandaIntraExpertScorePlugin,
)
from moe_compress.utils.cached_calibration_signals import sidecar_path
from moe_compress.utils.model_io import iter_moe_layers


# --------------------------------------------------------------------------
# Local helper -- mirrors the one in the sibling test file (codebase
# discipline: tests do not import from each other; the helper is small
# enough that duplication is preferred to cross-file coupling).
# --------------------------------------------------------------------------


def _make_namespace_ctx(
    model, batches, tmp_path: Path, *, jsonl_stem: str,
) -> PipelineContext:
    """Build a minimal ctx with the W-2 ``calibration_jsonl_path`` slot set
    to ``tmp_path / <jsonl_stem>.jsonl``. The .jsonl file does NOT need to
    exist — the plugin only consults its ``.parent`` and ``.stem`` via
    :func:`sidecar_path`.
    """
    config = {
        "stage3": {
            "wanda_intra_expert": {
                "enabled": True,
                "write_sidecar": True,
                "score_dtype": "float32",
                "scalar_row_dtype": "float32",
            }
        }
    }
    ctx = PipelineContext()
    ctx.set("model", model)
    ctx.set("moe_layers", list(iter_moe_layers(model)))
    ctx.set("batches", batches)
    ctx.set("device", None)
    ctx.set("config", config)
    ctx.set("calibration_jsonl_path", tmp_path / f"{jsonl_stem}.jsonl")
    return ctx


# ==========================================================================
# T2 — writer path matches the sidecar_path helper byte-for-byte
# ==========================================================================


def test_wanda_score_sidecar_path_matches_helper(tiny_model, tmp_path):
    """T2 — the path materialised by the writer must equal
    ``sidecar_path(jsonl, \"wanda_intra_expert_score\")`` byte-for-byte.

    This is the single-source-of-truth lock: if a future F-H-7 layout
    bump changes ``sidecar_path``'s layout (e.g. adds a per-config-hash
    subdir), the writer follows automatically. A drift between writer
    and helper would silently re-introduce a collision class.
    """
    torch.manual_seed(0)
    batches = [torch.randint(0, 32, (1, 4), dtype=torch.long)]
    jsonl_stem = "trace_t2"
    ctx = _make_namespace_ctx(
        tiny_model, batches, tmp_path, jsonl_stem=jsonl_stem,
    )

    plugin = WandaIntraExpertScorePlugin()
    plugin.collect_wanda_scores(ctx)

    jsonl_path = ctx.get("calibration_jsonl_path")
    expected = sidecar_path(jsonl_path, "wanda_intra_expert_score")
    # str() comparison locks the absolute-path representation; .exists()
    # locks that the writer actually produced a file at that path.
    assert expected.exists(), (
        f"expected sidecar at {expected} (from sidecar_path helper) "
        f"does not exist on disk"
    )
    # Defensive: the writer's logged path is the helper's path.
    assert str(expected) == str(
        tmp_path / "sidecars" / jsonl_stem / "wanda_intra_expert_score.pt"
    )


# ==========================================================================
# T3 — two distinct JSONL stems => two distinct sidecars, no collision
# ==========================================================================


def test_wanda_score_sidecar_two_jsonl_stems_no_collision(
    tiny_model, tmp_path,
):
    """T3 — regression reproducer for the audit W-2 collision scenario.

    Two Stage 3 runs share ``tmp_path`` (the moral equivalent of a
    shared ``artifacts_dir`` in the audit's A0..A11 ablation example) but
    advertise DIFFERENT calibration JSONL stems
    (``trace_alpha.jsonl``, ``trace_beta.jsonl``). Each run must land
    its sidecar in its own ``<stem>/`` subdirectory; neither overwrites
    the other. Pre-W-2, both runs silently clobbered the same legacy
    file at ``artifacts_dir/_stage3_wanda_intra_expert_score.pt``.
    """
    torch.manual_seed(42)
    # Distinct batches so the two payloads differ — sanity-check that
    # the scope split actually carries through to distinct on-disk
    # tensor maps (rules out "both runs wrote the same file by happening
    # to use identical data").
    batches_alpha = [torch.randint(0, 32, (1, 4), dtype=torch.long)]
    torch.manual_seed(99)
    batches_beta = [torch.randint(0, 32, (1, 4), dtype=torch.long)]

    plugin = WandaIntraExpertScorePlugin()

    ctx_alpha = _make_namespace_ctx(
        tiny_model, batches_alpha, tmp_path, jsonl_stem="trace_alpha",
    )
    plugin.collect_wanda_scores(ctx_alpha)

    ctx_beta = _make_namespace_ctx(
        tiny_model, batches_beta, tmp_path, jsonl_stem="trace_beta",
    )
    plugin.collect_wanda_scores(ctx_beta)

    sidecar_alpha = (
        tmp_path / "sidecars" / "trace_alpha" / "wanda_intra_expert_score.pt"
    )
    sidecar_beta = (
        tmp_path / "sidecars" / "trace_beta" / "wanda_intra_expert_score.pt"
    )
    manifest_alpha = sidecar_alpha.with_suffix(".pt.MANIFEST.json")
    manifest_beta = sidecar_beta.with_suffix(".pt.MANIFEST.json")

    assert sidecar_alpha.exists(), f"missing {sidecar_alpha}"
    assert sidecar_beta.exists(), f"missing {sidecar_beta}"
    assert manifest_alpha.exists(), f"missing {manifest_alpha}"
    assert manifest_beta.exists(), f"missing {manifest_beta}"

    # The two payloads must differ — proves the scope split actually
    # carried distinct calibration data through to distinct on-disk
    # tensor maps (not just two writes to the same file via aliasing).
    payload_alpha = torch.load(
        sidecar_alpha, map_location="cpu", weights_only=False,
    )
    payload_beta = torch.load(
        sidecar_beta, map_location="cpu", weights_only=False,
    )
    assert payload_alpha["format_version"] == _ARTIFACT_FORMAT_VERSION
    assert payload_beta["format_version"] == _ARTIFACT_FORMAT_VERSION

    # Walk both score maps and find at least one (layer, expert, matrix)
    # triple where the tensors are NOT byte-equal. The two runs used
    # different input batches (different torch seed); the score maps
    # must diverge somewhere.
    sm_alpha = payload_alpha["wanda_intra_expert_score"]
    sm_beta = payload_beta["wanda_intra_expert_score"]
    found_divergent = False
    for layer_idx in sm_alpha.keys() & sm_beta.keys():
        for expert_idx in (
            sm_alpha[layer_idx].keys() & sm_beta[layer_idx].keys()
        ):
            for mat in (
                sm_alpha[layer_idx][expert_idx].keys()
                & sm_beta[layer_idx][expert_idx].keys()
            ):
                t_a = sm_alpha[layer_idx][expert_idx][mat]
                t_b = sm_beta[layer_idx][expert_idx][mat]
                if not torch.equal(t_a, t_b):
                    found_divergent = True
                    break
            if found_divergent:
                break
        if found_divergent:
            break
    assert found_divergent, (
        "alpha and beta runs produced byte-identical score maps for every "
        "shared key — the per-stem namespace might be aliasing writes "
        "back to one file (collision regression)"
    )


# ==========================================================================
# T4 — missing calibration_jsonl_path ctx slot: skip with WARN, no raise
# ==========================================================================


def test_wanda_score_sidecar_missing_jsonl_ctx_skips_write(
    tiny_model, tmp_path, caplog,
):
    """T4 — when ``calibration_jsonl_path`` is absent from ctx and
    ``write_sidecar=True``, the plugin must:

    * NOT raise (the in-memory score map is still valuable even without
      a sidecar — operators may be running the plugin standalone).
    * Still publish ``ctx[\"stage3.wanda_intra_expert_score\"]`` —
      callers can consume the in-memory result.
    * NOT write any sidecar file under ``tmp_path``.
    * Emit a WARNING log line so operators see the actionable cause
      (\"orchestrator did not populate calibration_jsonl_path\") rather
      than a silent skip.
    """
    torch.manual_seed(3)
    batches = [torch.randint(0, 32, (1, 4), dtype=torch.long)]

    # Ctx WITHOUT calibration_jsonl_path -- intentional omission.
    config = {
        "stage3": {
            "wanda_intra_expert": {
                "enabled": True,
                "write_sidecar": True,
                "score_dtype": "float32",
                "scalar_row_dtype": "float32",
            }
        }
    }
    ctx = PipelineContext()
    ctx.set("model", tiny_model)
    ctx.set("moe_layers", list(iter_moe_layers(tiny_model)))
    ctx.set("batches", batches)
    ctx.set("device", None)
    ctx.set("config", config)
    # NOTE: calibration_jsonl_path intentionally NOT set.

    plugin = WandaIntraExpertScorePlugin()

    # caplog default propagates at WARNING from the root; explicitly
    # bind the plugin's logger to ensure capture even if the project's
    # logging config detaches handlers from propagation.
    plugin_logger = logging.getLogger(
        "moe_compress.stage3.plugins.wanda_intra_expert_score",
    )
    prev_propagate = plugin_logger.propagate
    plugin_logger.propagate = True
    try:
        with caplog.at_level(logging.WARNING):
            plugin.collect_wanda_scores(ctx)  # must NOT raise
    finally:
        plugin_logger.propagate = prev_propagate

    # In-memory result is still published.
    assert ctx.has("stage3.wanda_intra_expert_score")
    assert ctx.has("stage3.wanda_intra_expert_metadata")

    # No sidecar file anywhere under tmp_path (no jsonl_path => no
    # namespaced subdir created; no legacy fallback either).
    written_files = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert written_files == [], (
        f"expected no sidecar files; found {written_files}"
    )

    # WARNING log emitted with the actionable cause.
    warnings = [
        rec.message for rec in caplog.records
        if rec.levelno == logging.WARNING
    ]
    assert any(
        "calibration_jsonl_path" in msg for msg in warnings
    ), (
        f"expected a WARNING mentioning 'calibration_jsonl_path'; "
        f"got warnings: {warnings}"
    )


