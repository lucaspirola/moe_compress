"""Thermometer eval-environment setup (S6A-2 of the Stage 6alt plugin-architecture refactor).

Paper / spec source
--------------------
No upstream paper for Stage 6alt's "thermometer" sweep — it is a
project-original cheap ablation harness (BPT + small lm-eval subset
+ per-token argmax) that runs the same compressed model against a
fixed corpus for fast quality-temperature readings across many
ablation rows. See Stage 6 (:mod:`stage6.plugins.eval_environment`)
for the full eval gate.

This plugin reuses the same two Hopper environment helpers as Stage 6
(``_set_experts_implementation_s6`` + ``_apply_stage6_kernel_patches``)
without re-relocating them — they live in
:mod:`stage6.plugins.eval_environment` per the S6-2 refactor.

Home of the Stage 6alt environment-setup concern, extracted from the legacy
``stage6alt_thermometer.py`` monolith. The Stage 6alt thermometer reuses the
SAME two cu130/Hopper environment helpers as Stage 6 — ``_set_experts_implementation_s6``
and ``_apply_stage6_kernel_patches`` — without re-relocating them: they
already live in ``stage6.plugins.eval_environment`` (per S6-2). This module
is therefore Pattern-B only (an inert ``setup_thermo_environment`` hook that
reproduces the monolith's inline environment-setup call sequence) and
does NOT relocate any helper symbol.

Pattern A vs Pattern B
----------------------
S6A-2's environment slice is Pattern B *only*:

* **Pattern A — relocated verbatim**: NONE. The two env helpers
  (``_set_experts_implementation_s6``, ``_apply_stage6_kernel_patches``)
  stay in their S6-2 home (``stage6.plugins.eval_environment``) and are
  imported here for the hook body — re-relocating them would create two
  divergent copies of the same kernel-patch helper.
* **Pattern B — reproduced in an inert hook**: the monolith ``run()``'s
  inline environment-setup block (resolve ``experts_impl`` via the
  ``EXPERTS_IMPLEMENTATION`` env override, call
  ``_set_experts_implementation_s6`` then ``_apply_stage6_kernel_patches``
  with ``role="student"``) is reproduced in the inert
  ``setup_thermo_environment`` hook below. The monolith ``run()`` is NOT
  modified for it. This is an intentional, temporary logic duplication
  that resolves at S6A-6 when the orchestrator flip wires this hook live
  and the monolith ``run()`` becomes a thin shim.

Circular-import contract (mirror of ``stage6/plugins/eval_environment.py``):
this module imports only from ``..context`` / ``...stage6.plugins.eval_environment``
/ stdlib — NEVER from ``stage6alt_thermometer`` or ``stage6alt.orchestrator``
at any scope (module-top OR function-local). The monolith re-imports the
``ThermoEnvironmentPlugin`` at load time, so a ``from ..stage6alt_thermometer
import ...`` here would deadlock the import; nothing in this module does that.

``ThermoEnvironmentPlugin`` is registered-but-INERT at S6A-2 — no
orchestrator walk or test invokes its ``setup_thermo_environment`` hook.
S6A-6 plugs the hook into the live Stage 6alt plugin sequencer and turns
the monolith ``run()`` into a thin shim.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from ..context import PipelineContext
from ...stage6.plugins.eval_environment import (
    _apply_stage6_kernel_patches,
    _set_experts_implementation_s6,
)

log = logging.getLogger(__name__)


class ThermoEnvironmentPlugin:
    """Stage 6alt thermometer eval-environment plugin (S6A-2 — registered-but-INERT).

    Owns the Stage 6alt environment-setup concern: the MoE experts-implementation
    shim and the cu130/Hopper kernel patches applied to the student model
    before any forward pass. Both helpers are reused from
    ``stage6.plugins.eval_environment`` (the S6-2 relocation home) — see this
    module's docstring for why no symbol is relocated here.

    S6A-2 wires this class into the plugin registry as metadata only — no
    orchestrator walk or test invokes ``setup_thermo_environment``. S6A-6
    plugs the hook into the live Stage 6alt plugin sequencer.
    """

    name = "thermo_environment"
    paper = "Stage 6alt thermometer environment setup (project-original; reuses :mod:`stage6.plugins.eval_environment` helpers). See module docstring."
    config_key = "stage6_validate.thermometer"
    reads: tuple[str, ...] = ("model", "config")
    writes: tuple[str, ...] = ("experts_impl",)
    # `provides` is empty: this plugin sets up the runtime environment, it
    # does not need a calibration-pass accumulator. (Mirrors the S6-2
    # ``EvalEnvironmentPlugin``'s rationale.)
    provides: tuple[str, ...] = ()

    def is_enabled(self, config: dict) -> bool:
        """Always True — environment setup is UNCONDITIONAL.

        Every Stage 6alt thermometer run must set the experts-implementation
        shim and apply the cu130/Hopper kernel patches before any forward
        pass; ``config_key`` only names the thermometer config sub-tree, it
        never gates the plugin as a whole.
        """
        return True

    def contribute_artifact(self, ctx: Any) -> dict:
        return {}

    def setup_thermo_environment(self, ctx: PipelineContext) -> None:
        """Phase hook — Stage 6alt thermometer environment setup (S6A-6 wiring surface).

        INERT at S6A-2: no orchestrator walk or test invokes this hook. S6A-6
        replaces the Stage 6alt orchestrator body with the plugin sequencer
        and dispatches this hook in place of the monolith ``run()``'s inline
        environment-setup block. The body below reproduces that inline block
        faithfully — it is dead code at S6A-2 but S6A-6 relies on it once
        the monolith ``run()`` becomes a thin shim.

        Reproduces, in order:

        1. **Experts-implementation shim** — ``_set_experts_implementation_s6``
           (env var ``EXPERTS_IMPLEMENTATION`` overrides the YAML default
           ``batched_mm``; the override mirrors the monolith).
        2. **cu130/Hopper kernel patches** — ``_apply_stage6_kernel_patches``
           on the student.

        The resolved ``experts_impl`` is written to ctx so a teacher-side
        plugin (S6A-3) can apply the matching shim without re-resolving the
        env var.
        """
        # Required slots — direct get(): a missing one is a wiring bug and
        # SHOULD raise.
        model = ctx.get("model")
        config = ctx.get("config")
        s6 = config["stage6_validate"]

        # Resolve the experts-implementation: env override beats YAML default,
        # which itself defaults to "batched_mm" (the grouped_mm Blackwell-
        # deadlock workaround). Mirrors the monolith verbatim.
        experts_impl = os.environ.get(
            "EXPERTS_IMPLEMENTATION", s6.get("experts_implementation", "batched_mm")
        )
        _set_experts_implementation_s6(model, experts_impl)
        _apply_stage6_kernel_patches(model, role="student")

        ctx.set("experts_impl", experts_impl)


__all__ = ["ThermoEnvironmentPlugin"]
