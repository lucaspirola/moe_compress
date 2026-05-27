"""REAM sequential merging — per-layer cache invalidation plugin.

Paper
-----
**REAM: Merging Improves Pruning of Experts in LLMs** (Liu et al., Apr 2026,
Samsung SAIL Montreal) — arXiv:2604.04356, §4 *Sequential merging*. Open
source: https://github.com/SamsungSAILMontreal/ream.

The paper's central operational claim (§4, audit transcript
``audit/spec_compliance/01_papers/2604.04356/source.md`` lines 435-448):

    Prior expert pruning and merging methods run a single forward pass
    through the original, unmodified model to collect per-layer statistics.
    The pre-collected statistics are then used to compress all layers
    independently. However, once the experts in layer ℓ are compressed, its
    modified outputs render the statistics for the subsequent layers as
    stale. Instead, we propose updating the model outputs to reflect the
    currently merged layers. After merging layer ℓ, a second forward pass
    is run through this layer to recompute its activations to be used by
    the subsequent layer ℓ + 1.

This is REAM Recommendation 4a in the project's terminology — the
"sequential greedy with downstream propagation" lever called out as the
single highest-leverage Stage 2 gap in
``tasks/SC_STAGE12_COMPREHENSIVE_PLAN.md`` §5.2 / §523-532 (row ``S2_SEQ``).

What this plugin does
---------------------
Implements the **invalidation half** of REAM's sequential loop. Plugin #6
(commit ``ce49a1b``) added the ``on_post_merge`` phase to
``_STAGE2_POST_ASSIGN_PHASES`` between ``post_merge`` and
``write_artifacts``; this plugin is its first consumer. On the
``on_post_merge`` hook it sets the three per-layer cost-accumulator ctx
slots to ``None``:

* ``cov_acc`` — input-covariance accumulator used by ``post`` / ``output``
  cost modes and Stage 3 (see :mod:`stage2.shared_io`).
* ``ream_acc`` — REAM gate-logit + gated-output statistics, used by the
  ``pre`` / ``post`` cost plugins (see :mod:`stage2.plugins.ream_cost`,
  :mod:`stage2.plugins.ream_cost_post`).
* ``layer_input_acc`` — pre-MoE input reservoir used by the ``post`` /
  ``output`` cost plugins (see :mod:`stage2.profiling`).

The next per-layer iteration's ``on_layer_setup`` already rebuilds these
accumulators from scratch (see ``LayerMergePlugin.on_layer_setup``,
lines 425-470 of :mod:`stage2.plugins.layer_merge`). Clearing the slots
here makes the contract explicit and protects against any cross-layer
state leakage: once the next layer's ``on_profile`` runs a fresh forward
pass against the **live, just-merged model**, the accumulators it builds
naturally reflect the compressed upstream context — which is REAM's
sequential merging at the functional level.

Why "invalidate, don't re-profile"
-----------------------------------
Our per-layer driver (``stage2.orchestrator.run``) **already iterates
layers against the live model state**: ``LayerMergePlugin.merge`` calls
``_merge_experts_inplace`` on the actual model object, so the next
iteration's ``on_layer_setup`` → ``on_profile`` cycle naturally sees the
merged upstream context. The paper-time concern — "pre-collected
statistics are stale" — does not apply to us 1:1 because we never
pre-collect across layers; we collect per-layer at iteration time. What
**does** apply is the residual hazard that per-layer accumulators (or
the run-scope ``cov_acc`` on the ``LayerMergePlugin`` instance) carry
forward stale per-layer entries. Setting the three ctx slots to ``None``
forces the rebuild path in the next ``on_layer_setup`` and pins the
contract.

The user explicitly elected the unconditional cache-invalidation form
in commit ``ce49a1b`` (Plugin #6 commit message: "originally
conditional; user elected to apply unconditionally"). This plugin is the
canonical consumer.

Deviations from the REAM reference repo
---------------------------------------
* **D-seq-1** — *No second-pass driver.* The reference repo
  (https://github.com/SamsungSAILMontreal/ream) runs a dedicated second
  forward pass through the just-merged layer to refresh activation
  accumulators. Our orchestrator already runs the per-layer profile
  pass against the live (mutated) model, so the second pass collapses
  into the existing per-layer loop. Functional equivalence: ``cov_acc``
  / ``ream_acc`` / ``layer_input_acc`` are rebuilt by the **next**
  layer's ``on_profile`` from a forward pass against the merged model
  state.
* **D-seq-2** — *Knob naming.* Reference repo CLI flag
  ``--sequential-merging`` is exposed here as the YAML knob
  ``stage2_reap_ream.sequential_reprofile``, default ``false``.
* **D-seq-3** — *Three accumulators, not one.* The reference repo
  invalidates one activation-cost accumulator; this plugin clears three
  ctx slots because our Stage 2 cost framework has three independent
  cost modes (``pre`` / ``post`` / ``output``) backed by three
  accumulators. Strict superset of what REAM clears, so behaviour at
  the cost-matrix level is at least as fresh as REAM's.

Wiring status
-------------
LIVE. ``stage2.orchestrator.run()`` registers this plugin at the END of
the ``PluginRegistry`` list (after ``MergeHealPlugin``). ``LayerMergePlugin``
does not implement ``on_post_merge`` at all (it owns ``on_layer_setup`` /
``on_profile`` / ``merge`` / ``post_merge`` / ``write_artifacts`` /
``on_layer_teardown`` instead), so any per-layer-merge cleanup the spine
might do happens in ``post_merge`` — strictly before this plugin's hook
runs in the phase-major walk.

Config gate
-----------
Enabled iff ``stage2_reap_ream.sequential_reprofile`` is truthy.
Default ``False`` (absent key, ``False``, ``0``, ``""`` all OFF).
``registry.enabled(config)`` drops the plugin at the OFF default, so
**no existing config file needs editing** — the byte-identical
non-sequential path is preserved.

Note on ``cov_acc`` storage
---------------------------
In the current codebase the *primary* ``cov_acc`` lives as run-scope
plugin-instance state on ``LayerMergePlugin`` (constructed once per
``orchestrator.run()`` invocation). It is **not** currently a ctx slot
written by ``LayerMergePlugin``. The spec
(``SC_STAGE12_COMPREHENSIVE_PLAN.md`` §582) nonetheless mandates
``ctx.set("cov_acc", None, overwrite=True)`` — an unconditional upsert
that establishes the ctx slot as ``None``. Future plugins that read
``ctx.get("cov_acc")`` (none currently) will observe the invalidation.
The run-scope instance state is left untouched, per the project
constraint that this plugin must not modify the existing 18 Stage 2
plugins. This is documented as a known property, not a bug: the next
layer's profile pass writes NEW covariance keys ``(L+1, *, *)`` populated
by forward passes through the just-merged upstream — prior layers'
frozen keys cannot collide with new keys, so no stale-data path exists.

Circular-import note
--------------------
This module imports only ``...pipeline.context`` — no cycle at module
load. The orchestrator imports this class lazily inside ``run()``,
matching the pattern of every other Stage 2 plugin.
"""
from __future__ import annotations

import logging
from typing import Any

from ...pipeline.context import PipelineContext

log = logging.getLogger(__name__)


class Stage2ReamSequentialPlugin:
    """REAM sequential-merging cache invalidator (Plugin #10 / row ``S2_SEQ``).

    LIVE. ``stage2.orchestrator.run()`` registers this plugin at the end of
    the ``PluginRegistry`` list. On the ``on_post_merge`` phase it sets the
    three per-layer cost-accumulator ctx slots (``cov_acc``, ``ream_acc``,
    ``layer_input_acc``) to ``None`` so the next layer's
    ``on_layer_setup`` → ``on_profile`` cycle rebuilds them from a forward
    pass against the just-merged model state — REAM §4's "sequential
    merging" contract.

    Inert at the default config: ``registry.enabled(config)`` drops this
    plugin unless ``stage2_reap_ream.sequential_reprofile`` is truthy, so
    the byte-identical non-sequential path is preserved with no config
    edits required.

    See the module docstring for the paper anchor, the reference-repo
    deviations (D-seq-1 / D-seq-2 / D-seq-3), and the note on the
    ``cov_acc`` run-scope vs ctx-slot dichotomy.
    """

    name = "ream_sequential"
    paper = (
        "REAM §4 Sequential Merging — arXiv:2604.04356 (Liu et al., 2026, "
        "Samsung SAIL Montreal). Official code: "
        "github.com/SamsungSAILMontreal/ream. Plugin clears cov_acc / "
        "ream_acc / layer_input_acc on on_post_merge so the next layer's "
        "on_layer_setup → on_profile sees fresh state reflecting the "
        "freshly-merged upstream context. Deviations: D-seq-1 (no "
        "second-pass driver — uses the existing per-layer loop's natural "
        "sequential structure against the live mutated model), D-seq-2 "
        "(YAML knob ``sequential_reprofile`` vs ref-repo CLI "
        "``--sequential-merging``), D-seq-3 (clears 3 accumulators because "
        "our cost framework has 3 modes — strict superset of what REAM "
        "clears). See module docstring."
    )
    config_key = "stage2_reap_ream.sequential_reprofile"
    # No reads — the hook only writes (sets-to-None) three ctx slots.
    reads: tuple[str, ...] = ()
    # ``writes`` declares the ctx slots this plugin touches; all three are
    # set to ``None`` (an upsert via ``overwrite=True``). Matches the
    # ``LayerMergePlugin.on_layer_teardown`` invalidation pattern (lines
    # 717-720 of :mod:`stage2.plugins.layer_merge`).
    writes: tuple[str, ...] = ("cov_acc", "ream_acc", "layer_input_acc")
    provides: tuple[str, ...] = ()

    def is_enabled(self, config: dict) -> bool:
        """True iff ``stage2_reap_ream.sequential_reprofile`` is truthy.

        Default OFF: absent ``stage2_reap_ream`` block, absent
        ``sequential_reprofile`` key, or any falsy value (``False`` /
        ``0`` / ``""`` / ``None``) all disable the plugin. Non-dict
        configs are treated as empty — ``None`` configs return ``False``
        as well, courtesy of the same non-dict guard.

        With the gate OFF, ``registry.enabled(config)`` drops this plugin
        and the byte-identical non-sequential path is preserved — no
        existing YAML needs editing.
        """
        if not isinstance(config, dict):
            return False
        s2 = config.get("stage2_reap_ream", {})
        if not isinstance(s2, dict):
            return False
        return bool(s2.get("sequential_reprofile", False))

    def contribute_artifact(self, ctx: Any) -> dict:
        """Return an empty artifact dict; this plugin contributes none."""
        return {}

    def on_post_merge(self, ctx: PipelineContext) -> None:
        """Invalidate per-layer cost accumulators after a layer's merge.

        Sets ``cov_acc`` / ``ream_acc`` / ``layer_input_acc`` to ``None``
        on the per-layer ctx so the next layer's ``on_layer_setup`` →
        ``on_profile`` cycle rebuilds them from a forward pass against
        the just-merged model state. This is the **invalidation half**
        of REAM's Section 4 sequential-merging contract; the
        re-profile itself happens naturally in the next iteration of the
        per-layer loop, which our orchestrator already drives against
        the live (mutated) model.

        ``overwrite=True`` is an upsert: the call works whether the slot
        was ever set on this scope or not, and whether its current value
        is ``None`` or a live accumulator. Mirrors the same idiom in
        ``LayerMergePlugin.on_layer_teardown`` (lines 717-720 of
        :mod:`stage2.plugins.layer_merge`).

        Per SC_STAGE12 §582 — the user elected to apply this
        unconditionally; the gate ``sequential_reprofile`` controls only
        whether this plugin is **registered** at all, not what it does
        when invoked.
        """
        ctx.set("cov_acc", None, overwrite=True)
        ctx.set("ream_acc", None, overwrite=True)
        ctx.set("layer_input_acc", None, overwrite=True)
        # Per-layer DEBUG breadcrumb. The layer_ref slot is written by
        # the orchestrator before the post-assign phase walk runs (see
        # ``stage2.orchestrator.run``); capture once into a local binding
        # so unit-test ctxs without a layer_ref produce a generic message
        # instead of KeyError-ing through the production hook.
        # ``PipelineContext.get`` is strict (no default arg), so a single
        # ``has`` guard upfront drives both branches off one truthy local —
        # cleaner than re-reading the slot inside the truthy branch.
        if log.isEnabledFor(logging.DEBUG):
            layer_ref = ctx.get("layer_ref") if ctx.has("layer_ref") else None
            if layer_ref is not None:
                layer_idx = getattr(layer_ref, "layer_idx", "?")
                log.debug(
                    "ream_sequential: invalidated cov_acc/ream_acc/"
                    "layer_input_acc after layer %s merge",
                    layer_idx,
                )
            else:
                log.debug(
                    "ream_sequential: invalidated cov_acc/ream_acc/"
                    "layer_input_acc after merge",
                )


__all__ = ["Stage2ReamSequentialPlugin"]
