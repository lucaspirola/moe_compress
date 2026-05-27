"""RKD paper-recipe (Row P) config-override plugin — Plugin #7.

Paper
-----
Hyeon & Do, "Is Retraining-Free Enough? The Necessity of Router Calibration
for Efficient MoE Compression" — arXiv:2603.02217 (Hyeon & Do, Mar 2026).
Eq. 3 (§5) defines

    L_RKD = (τ² / N_x) · Σ_t  m_{t+1} · D_KL(p_T ‖ p_S)

with τ=4 in the canonical recipe (Hinton et al. 2015 — the τ² scaling
convention; softening softens both teacher and student distributions and
scales the gradient by τ², which is exactly what the project's
``vocab_kd._chunked_vocab_kl`` kernel implements).

This plugin is NOT a loss-kernel replacement — pre-flight verification
(see ``tasks/PLAN_PLUGIN_07_rkd_paper_recipe.md`` §2) confirmed that
``vocab_kd.py`` already computes the correct forward-KL with τ² scaling and
the fully-packed padding-mask invariant. The plugin's job is purely to
swap in the 4 paper-recipe hyperparameters + the wikitext-103-raw
calibration source, so an A/B against the current Row C production recipe
isolates the paper-vs-project recipe deltas.

The 4 deltas (Row P vs Row C)
-----------------------------
+------------------------+-----------------+-----------------------+
| Knob                   | Row C (current) | Row P (paper)         |
+------------------------+-----------------+-----------------------+
| ``kd_temperature``     | 1.0             | **4.0**               |
| ``weight_decay``       | 0.01            | **0.0**               |
| ``epochs``             | 1               | **2**                 |
| ``early_stop_patience``| 8               | **0** (disabled)      |
| Calibration source     | qwen3-pretrain  | **wikitext-103-raw**  |
|                        | -mix-v2         |                       |
+------------------------+-----------------+-----------------------+

Multi-epoch + cache guard: ``orchestrator.py`` (line 585) raises if
``epochs > 1 and teacher_logits_cache is not None``. Row P sets
``epochs=2``, so ``apply_config_overrides`` explicitly clears
``s5["teacher_logits_cache"]`` to ``None`` to prevent the guard from firing
when an operator accidentally has a cache configured.

Architecture decision — pre-flight config mutation (NOT a walk_phases hook)
--------------------------------------------------------------------------
The Router-KD orchestrator captures all config locals at the very top of
``run()`` before any plugins are dispatched (``s5 = config["stage5_router_kd"]``
at line 172, ``cal = config["calibration"]`` at line 173). Any
``walk_phases``-dispatched hook runs AFTER those captures, by which time
``s5`` and ``cal`` are already bound to their original values; mutating
``ctx["config"]`` from a phase hook is too late.

Chosen approach: the plugin exposes ``apply_config_overrides(config) -> None``
that mutates ``config`` in-place. The orchestrator calls this method as the
very first statement of ``run()``, BEFORE the ``s5`` / ``cal`` captures.

Contract
--------
1. ``apply_config_overrides`` reads
   ``config["stage5_router_kd"].get("rkd_recipe", "current")``.
2. If the value is ``"current"`` (or any non-``"paper"`` value, or the key
   is missing, or the ``stage5_router_kd`` block is missing entirely), the
   method returns immediately without touching ``config``. Row C runs are
   byte-identical to pre-plugin behavior.
3. If the value is ``"paper"``, the method mutates ``config`` in-place:
     * ``s5["kd_temperature"] = 4.0``
     * ``s5["weight_decay"] = 0.0``
     * ``s5["epochs"] = 2``
     * ``s5["early_stop_patience"] = 0``
     * ``s5["teacher_logits_cache"] = None``  (multi-epoch guard)
     * ``config["calibration"]["source"] = "wikitext-103-raw"``
4. The existing Stage 2.5 / Stage 5 plugins
   (:mod:`~moe_compress.router_kd.plugins.vocab_kd`,
   :mod:`~moe_compress.router_kd.plugins.kd_optimizer`,
   :mod:`~moe_compress.router_kd.plugins.early_stop`) then read their
   effective values from the mutated ``config`` — no changes needed in
   those plugins.

The plugin is NOT registered in the orchestrator's ``PluginRegistry`` list
because it carries no ``walk_phases`` hooks. The orchestrator calls
``apply_config_overrides`` directly. ``is_enabled`` is provided for
registry-style audit/reporting only.

Circular-import contract (mirror of vocab_kd / merge_repair / early_stop):
this module imports only from stdlib at any scope. It NEVER imports
``stage5_router_kd`` or ``router_kd.orchestrator``, since the orchestrator
itself imports *this* module — the reverse direction would deadlock.
"""
from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


class RkdPaperRecipePlugin:
    """Pre-flight config-override plugin for the Row P paper recipe.

    Satisfies the universal :class:`PipelinePlugin` Protocol via class-level
    attributes (no phase hooks — ``apply_config_overrides`` is the sole
    functional entry point, called by the orchestrator before its config
    captures).

    When ``stage5_router_kd.rkd_recipe == "paper"`` is set in the YAML,
    ``apply_config_overrides(config)`` mutates ``config`` in place to apply
    the 4 paper-recipe deltas + the wikitext-103-raw calibration source.
    The default value ``"current"`` makes the method a no-op, so existing
    runs that do not opt in are byte-identical to pre-plugin behavior.
    """

    name = "rkd_paper_recipe"
    paper = (
        "arXiv:2603.02217 (Hyeon & Do — Router-KD vocab-KL distillation, "
        "Eq. 3 / §F.3 Table 1) + Hinton et al. 2015 (τ² scaling convention)."
    )
    config_key = "stage5_router_kd.rkd_recipe"
    reads: tuple[str, ...] = ("config",)
    writes: tuple[str, ...] = ()  # No ctx slot publications. NOTE: this plugin
    # mutates ``config`` in-place via apply_config_overrides
    # — see class docstring "Injection-point contract".
    provides: tuple[str, ...] = ()

    def is_enabled(self, config: dict) -> bool:
        """True iff the operator opted in to the paper recipe.

        Reads ``config["stage5_router_kd"].get("rkd_recipe", "current")``.
        Default ``"current"`` → False (the no-op path). Missing or non-dict
        ``stage5_router_kd`` block → False (graceful: never raise during a
        registry-style audit walk).
        """
        s5 = config.get("stage5_router_kd")
        if not isinstance(s5, dict):
            return False
        return s5.get("rkd_recipe", "current") == "paper"

    def contribute_artifact(self, ctx: Any) -> dict:
        # Fresh empty dict literal each call — never a shared module-level
        # object (mirrors the other Router-KD plugins).
        return {}

    def apply_config_overrides(self, config: dict) -> None:
        """Mutate ``config`` in place to apply the Row P paper recipe.

        Called by ``router_kd.orchestrator.run`` as the FIRST statement of
        the function body, BEFORE ``s5 = config["stage5_router_kd"]`` and
        ``cal = config["calibration"]`` capture the live dicts.

        Behaviour
        ---------
        * No-op when ``stage5_router_kd.rkd_recipe`` is ``"current"``, any
          other non-``"paper"`` value, or absent — and also when the
          ``stage5_router_kd`` block itself is missing (defensive; the real
          orchestrator will raise later on the missing block, but this
          method must never raise on a non-paper path).
        * When the value is ``"paper"``, applies the 4 deltas + the
          calibration-source swap + the teacher_logits_cache clearance.

        Idempotent: applying the override twice yields the same final
        config (each assignment is unconditional). The orchestrator only
        calls this once per run().
        """
        s5 = config.get("stage5_router_kd")
        if not isinstance(s5, dict):
            return
        if s5.get("rkd_recipe", "current") != "paper":
            return

        # The 4 numeric/scalar deltas from the paper recipe.
        s5["kd_temperature"] = 4.0
        s5["weight_decay"] = 0.0
        s5["epochs"] = 2
        s5["early_stop_patience"] = 0

        # Multi-epoch + cache guard (orchestrator.py:585 raises if
        # epochs>1 and teacher_logits_cache is not None). Row P sets
        # epochs=2, so clear the cache slot defensively.
        s5["teacher_logits_cache"] = None

        # Calibration source swap (paper §F.3 Table 1 uses raw text; we
        # mirror with the wikitext-103-raw adapter registered in
        # ``utils/calibration.py``). The ``calibration:`` block must exist
        # for any Router-KD run; the orchestrator validates it downstream,
        # so we assume it is present here and create / mutate the source key.
        cal = config.get("calibration")
        if not isinstance(cal, dict):
            log.warning(
                "RkdPaperRecipePlugin: config has no 'calibration' block; "
                "creating minimal stub with source='wikitext-103-raw'. "
                "Downstream calibration setup may KeyError on missing keys "
                "(num_sequences, sequence_length, seed). Add a complete "
                "calibration block to the config to silence this warning."
            )
            cal = {}
            config["calibration"] = cal
        cal["source"] = "wikitext-103-raw"

        log.info(
            "RkdPaperRecipePlugin: applied Row P overrides — "
            "kd_temperature=4.0, weight_decay=0.0, epochs=2, "
            "early_stop_patience=0, teacher_logits_cache=None, "
            "calibration.source='wikitext-103-raw'."
        )


__all__ = ["RkdPaperRecipePlugin"]
