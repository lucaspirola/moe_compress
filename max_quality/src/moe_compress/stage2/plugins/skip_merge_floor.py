"""Skip-merge percentile-mask on the assignment cost matrix.

Paper
-----
**No paper.** Project-original "Direction B" mask — a per-layer
percentile cutoff applied to the assignment cost matrix before the
solver runs. Any candidate ``(centroid, non-centroid)`` pair whose
cost lies above the percentile threshold is masked out (set to ``+∞``),
forcing the solver to either find a lower-cost match or leave the
non-centroid unassigned (orphan-singleton promotion catches that, per
D-ream-budget-bump consumed at :mod:`stage2.plugins.layer_merge`).

The baseline REAM cost (arXiv:2604.04356, see
:mod:`stage2.plugins.ream_cost`) has no such mask.

Official code
-------------
None — Direction B is project-original. The mask logic lives in
:func:`stage2.grouping._apply_skip_merge_floor`.

Why a percentile-mask
---------------------
The cost matrix's distribution is heavy-tailed at low merge budgets:
a small fraction of ``(c, m)`` pairs have near-zero cost (genuinely
similar experts) and the long tail represents pairs whose merge would
materially damage the centroid. The percentile mask is a coarse
"don't merge anything in the tail" gate — set at e.g. the 75th
percentile, the solver is allowed to consider only the bottom-quartile
pairs.

Combined with the per-centroid cap (``max_merge_group_size``,
D5a — consumed at :mod:`stage2.plugins.layer_merge`) and the budget
bump loop (D-ream-budget-bump), the mask reduces the rate of
high-cost merges in heterogeneous layers where the GRAPE budget is
slack.

Config gate
-----------
Enabled iff ``stage2_reap_ream.skip_merge_percentile < 100.0``.
``100.0`` is the OFF sentinel (the 100th percentile equals the max
finite cost, so nothing is strictly above it). A missing key defaults
to ``100.0`` → OFF. Values above 100.0 are also OFF.

Naming-history note
-------------------
"Direction B" is the project's STRATEGY_NEXT label. The current plugin
architecture has no direction-letter taxonomy; new prose drops the
label. Existing log lines / Trackio keys are preserved for dashboard
back-compat.

Wiring status
-------------
LIVE. ``stage2.orchestrator.run()`` registers this plugin in the
``PluginRegistry`` after the three cost plugins and before the merge
spine (``LayerMergePlugin``), so when it is enabled
(``skip_merge_percentile < 100.0``) it wins the ``apply_cost_mask``
``dispatch_first`` slot in ``_run_assignment``'s bump loop. At the OFF
sentinel ``registry.enabled(config)`` filters this plugin out entirely;
``dispatch_first`` then finds no servicer for ``apply_cost_mask`` and
returns ``None``, which the orchestrator handles via
``if masked is not None:`` — the delta is left unmasked. The monolith's
``_em_refine_assignment`` still re-applies the floor each EM round.

OFF-branch return contract
--------------------------
The documented plugin protocol for ``apply_cost_mask`` (see
``docs/stage2_plugin_guide.md``) is "return ``(masked, info)`` or
``None`` to leave the matrix unmasked." This plugin deliberately
diverges in its OFF branch: when ``skip_merge_percentile >= 100.0`` and
the method is invoked directly (e.g. unit tests), it returns
``(delta, {"n_masked": 0, "percentile": ...})`` — the same array object
unchanged, plus a diagnostic info dict. The production OFF path never
hits this branch because ``is_enabled`` already filters the plugin out
before ``dispatch_first`` reaches it. The 2-tuple return is preserved
for unit-test ergonomics (caller can always unpack and inspect
``info["n_masked"] == 0``).

Note: the OFF branch returns the input array un-copied, while the
active branch promotes via ``_apply_skip_merge_floor`` (which copies
and works in float64). Dtype is therefore not stable across the
OFF/active boundary — callers that need a guaranteed copy must
copy themselves.

Circular-import note
--------------------
This module imports only ``...pipeline.context`` and ``..grouping``
(i.e. ``stage2.grouping``) — neither of which imports
``stage2_reap_ream``. No cycle at module load.
"""
from __future__ import annotations

import logging
from typing import Any

from ...pipeline.context import PipelineContext
from ..grouping import _apply_skip_merge_floor

log = logging.getLogger(__name__)

# OFF sentinel: percentile >= this value masks nothing (see grouping docstring).
_SKIP_MERGE_OFF = 100.0


class SkipMergeFloorPlugin:
    """Plugin home for the Direction B skip-merge percentile mask.

    LIVE. ``stage2.orchestrator.run()`` registers this plugin after the three
    cost plugins and before the merge spine (``LayerMergePlugin``); when
    enabled (``skip_merge_percentile < 100.0``) it wins the ``apply_cost_mask``
    ``dispatch_first`` slot in ``_run_assignment``'s bump loop. ``apply_cost_mask``
    delegates to ``grouping._apply_skip_merge_floor``. At the OFF sentinel
    ``registry.enabled`` filters this plugin out, ``dispatch_first`` returns
    ``None``, and the orchestrator's ``if masked is not None:`` branch leaves
    the delta unmasked.

    The percentile is stored at construction: ``apply_cost_mask`` only receives
    a ``PipelineContext`` (which carries no cfg), so the value cannot be read at
    call time. Use :func:`make_skip_merge_floor_plugin` to build the plugin
    from a config dict — the factory keeps the constructor's
    ``skip_merge_percentile`` aligned with the config's
    ``stage2_reap_ream.skip_merge_percentile``. ``is_enabled(config)`` reads
    the **config** percentile and ignores ``self.skip_merge_percentile`` —
    callers that construct the plugin directly are responsible for keeping
    the two consistent.
    """

    name = "skip_merge_floor"
    paper = (
        "Skip-merge percentile mask (project-original; no paper). "
        "STRATEGY_NEXT 'Direction B'. Baseline REAM cost arXiv:2604.04356 "
        "has no such mask. Opt-in via skip_merge_percentile < 100.0. "
        "See module docstring."
    )
    config_key = "stage2_reap_ream.skip_merge_percentile"
    # S2-7: the live ``apply_cost_mask`` slot reads ``layer_ref`` for the
    # INFO log line emitted when entries are masked.
    reads: tuple[str, ...] = ("layer_ref",)
    writes: tuple[str, ...] = ()
    provides: tuple[str, ...] = ()

    def __init__(self, skip_merge_percentile: float = _SKIP_MERGE_OFF) -> None:
        """Store the percentile for ``apply_cost_mask``.

        Defaults to the OFF sentinel so a bare ``SkipMergeFloorPlugin()`` is a
        no-op masker (matches the config default).
        """
        self.skip_merge_percentile = float(skip_merge_percentile)

    def is_enabled(self, config: dict) -> bool:
        """True iff ``stage2_reap_ream.skip_merge_percentile`` is < 100.0.

        ``100.0`` (the default) and any missing key → False (OFF). Values
        above 100.0 are also OFF (nothing can sit strictly above a >100th
        percentile clamp). Only a value strictly below 100.0 turns it on.
        """
        s2 = config.get("stage2_reap_ream", {}) if isinstance(config, dict) else {}
        return float(s2.get("skip_merge_percentile", _SKIP_MERGE_OFF)) < _SKIP_MERGE_OFF

    def contribute_artifact(self, ctx) -> dict:
        """Return an empty artifact dict; this plugin contributes none."""
        return {}

    def apply_cost_mask(
        self, ctx: PipelineContext, delta: Any
    ) -> tuple[Any, dict] | None:
        """Mask cost-matrix entries above the skip-merge percentile to +inf.

        Delegates to ``grouping._apply_skip_merge_floor``. At the OFF sentinel
        (percentile >= 100.0) returns the input array unchanged (no copy) and
        an info dict with ``n_masked == 0`` — see the module-level
        "OFF-branch return contract" note for the deliberate divergence from
        the documented ``None``-to-decline protocol. In production the OFF
        branch is unreachable: ``is_enabled`` filters the plugin out before
        ``dispatch_first`` reaches it.

        Returns ``(masked_delta, info)`` with ``info = {"n_masked": int,
        "percentile": float}``, plus an INFO log line emitted when entries
        are masked. ``ctx`` carries ``layer_ref`` for that log; it is ``None``
        in unit tests, so the log is guarded on ``ctx is not None``.

        Dtype note: the active path runs through ``_apply_skip_merge_floor``
        which promotes to float64; the OFF branch returns the input array
        with its original dtype. Dtype is not stable across the boundary.
        """
        if self.skip_merge_percentile >= _SKIP_MERGE_OFF:
            return delta, {"n_masked": 0, "percentile": self.skip_merge_percentile}
        masked, n_masked = _apply_skip_merge_floor(delta, self.skip_merge_percentile)
        if n_masked > 0 and ctx is not None:
            layer_ref = ctx.get("layer_ref")
            log.info(
                "layer %d: skip-merge floor (P%.1f) masked %d/%d "
                "cost entries to +inf — affected children fall "
                "through to orphan promotion",
                layer_ref.layer_idx, self.skip_merge_percentile,
                n_masked, masked.size,
            )
        return masked, {"n_masked": n_masked, "percentile": self.skip_merge_percentile}


def make_skip_merge_floor_plugin(cfg: dict[str, Any]) -> SkipMergeFloorPlugin:
    """Construct a :class:`SkipMergeFloorPlugin` from a Stage 2 config dict.

    Reads ``cfg["stage2_reap_ream"]["skip_merge_percentile"]`` (default
    ``100.0``). Mirrors how ``stage2_reap_ream.run()`` will wire the plugin
    once ``compute_assignment`` is decomposed (T13+).
    """
    s2 = cfg.get("stage2_reap_ream", {}) if isinstance(cfg, dict) else {}
    return SkipMergeFloorPlugin(
        skip_merge_percentile=float(s2.get("skip_merge_percentile", _SKIP_MERGE_OFF))
    )
