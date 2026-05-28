"""RegMean Stage-2 merge-step plugin (alternative to REAM freq-weighted merge).

This plugin is a metadata / validation shim for the ``merge_step="regmean"``
config-knob branch of :func:`stage2.merging._merge_experts_inplace`. The
actual closed-form math lives in :mod:`stage2.regmean` (function
``_regmean_solve_one_linear``); the merge spine in
:class:`stage2.plugins.layer_merge.LayerMergePlugin` calls into it when
the operator sets ``stage2_reap_ream.merge_step = "regmean"``.

Why a separate plugin file?
---------------------------
Pattern E (algorithm-branch knob): ``merge_step`` is the existing knob
already used by ``"freq_weighted"`` (default) and ``"mergemoe"`` (Plugin #9 /
S2_MM). RegMean is a third value. Per the project plugin discipline
(``max_quality/docs/stage2_plugin_guide.md``), each non-default algorithm
gets its own file so:

  1. The math and its paper citation live in one discoverable place.
  2. Pattern C config-validation surfaces as a class-level method that
     callers can invoke independently of the orchestrator.
  3. ``contribute_artifact`` can declare what the algorithm writes (here:
     nothing extra — the merged weights land in the existing
     ``_stage2_partial/`` artifacts via the merge spine).

Paper
-----
Jin et al., *Dataless Knowledge Fusion by Merging Weights of Language
Models*, ICLR 2023 — arXiv:2212.09849.

Math (paper §3.1 Eq. 2)
-----------------------
For each per-expert ``nn.Linear`` matrix (``gate_proj`` / ``up_proj`` /
``down_proj``) in a merge cluster ``C`` of ``N`` permutation-aligned
source experts::

    W_M^T = (Σ_i G_i)^{-1} · Σ_i (G_i · W_i^T)
    W_M   = (W_M^T).T

with ``G_i = X_i^T · X_i`` the per-source input Gram (Pearson-style, no
centering) collected by :class:`stage2.utils.activation_hooks.InputCovarianceAccumulator`
during the same Stage-2 forward profile pass that REAP / REAM use. No
extra calibration pass is required — RegMean piggybacks on the existing
profile.

Data flow
---------

  Stage-2 profile (`_profile_layer`)
       ├── cov_acc.update(layer, expert, "gate_proj", x)   per-expert G_gate
       └── cov_acc.update(layer, expert, "down_proj", x)   per-expert G_down
                                  │
                                  ▼
  LayerMergePlugin.merge(ctx)
       └── _merge_experts_inplace(..., merge_step="regmean", cov_acc=cov_acc)
                                  │
                                  ▼
       per merge cluster, per (gate_proj / up_proj / down_proj):
           _regmean_solve_one_linear(weights, grams, alphas)

Pattern C validation
--------------------
The orchestrator's top-of-``run()`` config validator accepts
``merge_step in {"freq_weighted", "mergemoe", "regmean"}``. The per-cluster
fallback inside ``_regmean_solve_one_linear`` handles the (rare)
degenerate-conditioning case; the per-cluster fallback inside
``_merge_experts_inplace`` handles the (rare) zero-traffic case where
``cov_acc.get(...)`` returns ``None`` for some member. See the
``D-regmean-*`` deviations in :mod:`stage2.regmean`.

Plugin discipline
-----------------
* Pattern B: no artifact ``format_version`` here — RegMean writes
  weights into the same ``_stage2_partial/layer_*.pt`` files as the
  default merge spine, governed by the existing
  ``_HEAL_WEIGHTS_FORMAT_VERSION`` / ``merge_*.json`` schemas. The
  schema bump that Plugin #9 / S2_MM did for ``merge_step`` already
  records the configured value at ``stage2/config/merge_step`` (Trackio).
* Pattern C: ``ensure_cov_acc_populated_for_cluster`` is a static helper
  the merge spine can call; raising actionable error with the operator
  remediation step (set ``merge_step`` back to ``freq_weighted``).
* Pattern E: ``merge_step`` is the existing algorithm-branch knob; this
  plugin adds a third value, not a new knob.
* Pattern H: clean-room re-implementation of upstream
  ``tanganke/fusion_bench`` ``fusion_bench/method/regmean/regmean.py``
  ``merging_with_regmean_weights`` (lines 43–120, MIT (c) 2024 Anke
  Tang). No code copied verbatim. Attribution path:
  the upstream paper (Jin et al. arXiv:2212.09849) is cited; the
  fusion_bench reference impl is named in :mod:`stage2.regmean`. Paper
  re-verification stamp: 2026-05-28.
"""
from __future__ import annotations

import logging
from typing import Any

from ...pipeline.context import PipelineContext

log = logging.getLogger(__name__)


class RegMeanMergeStepPlugin:
    """Metadata + config-validation shim for the RegMean merge-step branch.

    Instances of this class carry no per-run state; the math is invoked
    inline by :func:`stage2.merging._merge_experts_inplace` when the
    operator sets ``stage2_reap_ream.merge_step = "regmean"``. The
    plugin's ``is_enabled`` gate is True only when that knob is set, so
    the plugin can be registered unconditionally — it self-deselects on
    every other run.

    The plugin is intentionally side-effect-free: it does not register
    any phase hooks (``on_layer_setup`` / ``on_profile`` / ``merge`` / …)
    because the orchestrator's existing ``LayerMergePlugin.merge`` hook
    is the one place that needs to dispatch on ``merge_step``, and that
    dispatch is already in place from Plugin #9 / S2_MM. This shim
    exists to make RegMean **discoverable** (instance metadata, plugin
    registry inspection) and to centralize the Pattern C validation.
    """

    name: str = "regmean_merge_step"
    paper: str = (
        "RegMean (Jin et al., ICLR 2023, arXiv:2212.09849) — closed-form "
        "per-Linear least-squares merge W_M = (Σ G_i)⁻¹ Σ G_i W_i with "
        "G_i = X_i^T·X_i. Clean-room reimpl; upstream reference: "
        "tanganke/fusion_bench fusion_bench/method/regmean/regmean.py "
        "(MIT, Copyright (c) 2024 Anke Tang). Math derived from paper "
        "Eq. 2 only. Deviations: D-regmean-damping, D-regmean-cond-fallback, "
        "D-regmean-zero-cov-fallback, D-regmean-no-non-diagonal-reduction, "
        "D-regmean-no-renormalization-by-tokens, D-regmean-fp32-solve "
        "(see :mod:`stage2.regmean`). Paper re-verification stamp: 2026-05-28."
    )
    config_key: str = "stage2_reap_ream.merge_step"
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()
    provides: tuple[str, ...] = ()

    def is_enabled(self, config: dict) -> bool:
        """True iff ``stage2_reap_ream.merge_step == "regmean"``.

        Reads the resolved value via the same accessor the orchestrator
        uses — defensively lowercased so case-insensitive YAML keys
        ("RegMean" / "REGMEAN") all resolve to the same gate.
        """
        s2 = config.get("stage2_reap_ream") or {}
        return str(s2.get("merge_step", "freq_weighted")).lower() == "regmean"

    def contribute_artifact(self, ctx: PipelineContext) -> dict:
        # Pattern B: no new on-disk artifact; merged weights go through
        # the existing ``_stage2_partial/`` writer in
        # :class:`stage2.plugins.layer_merge.LayerMergePlugin`.
        return {}

    @staticmethod
    def validate_cov_acc_present(cov_acc) -> None:
        """Pattern C — Stage 2 calibration check at run() entry.

        Raises ``ValueError`` if ``cov_acc`` is ``None`` (the calibration
        infrastructure was not constructed for this run). In normal
        ``stage2.orchestrator.run`` flow ``cov_acc`` is always a
        non-None :class:`InputCovarianceAccumulator` — the orchestrator
        constructs it unconditionally near line 746 — but a defensive
        check here catches misconfigured tests / external callers that
        construct ``LayerMergePlugin`` directly without wiring covariance
        capture.

        The per-cluster, per-member ``cov.get(...)`` -> None case (some
        expert received zero calibration traffic) is handled inside
        :func:`stage2.merging._merge_experts_inplace` itself — that
        per-cluster fallback degrades to freq-weighted merge with a
        WARNING log line (see D-regmean-zero-cov-fallback).
        """
        if cov_acc is None:
            raise ValueError(
                "stage2_reap_ream.merge_step='regmean' requires a populated "
                "InputCovarianceAccumulator (Stage 2 profile must run "
                "before merge). Got cov_acc=None — the calibration "
                "infrastructure was not wired. Either (a) ensure the "
                "Stage 2 profile pass runs and populates cov_acc, or "
                "(b) set stage2_reap_ream.merge_step back to "
                "'freq_weighted' (default) or 'mergemoe'."
            )


__all__: tuple[str, ...] = ("RegMeanMergeStepPlugin",)
