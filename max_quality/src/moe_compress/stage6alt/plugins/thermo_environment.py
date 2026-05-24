"""Thermometer eval-environment setup plugin for the Stage 6alt plugin pipeline.

Paper / spec source
--------------------
No upstream paper for Stage 6alt's "thermometer" sweep — it is a
project-original cheap ablation harness (BPT + small lm-eval subset
+ per-token argmax) that runs the same compressed model against a
fixed corpus for fast quality-temperature readings across many
ablation rows. See Stage 6 (``stage6.plugins.eval_environment``)
for the full eval gate.

This plugin reuses the same two Hopper environment helpers as Stage 6
(``_set_experts_implementation_s6`` + ``_apply_stage6_kernel_patches``)
without re-relocating them — they live in
``stage6.plugins.eval_environment`` and are imported from there. No
helper symbol is relocated to this module: re-relocating would create
two divergent copies of the same kernel-patch helper.

Live wiring
-----------
``ThermoEnvironmentPlugin.setup_thermo_environment`` is dispatched by
``stage6alt.orchestrator.run`` via
``walk_phases(("setup_thermo_environment",), plugins, run_ctx)``. The
legacy ``stage6alt_thermometer.run()`` is a thin shim that delegates to
``stage6alt.orchestrator.run``. Tests
(``tests/test_stage6alt_plugin_corpus.py::test_setup_thermo_environment_hook``)
invoke the hook directly.

Intentional asymmetry vs Stage 6
--------------------------------
Stage 6's ``EvalEnvironmentPlugin`` declares additional generative slots
(``pre_compile_forward`` / ``experts_implementation_generative``). This
plugin deliberately does NOT — the thermometer harness has no
``model.generate()`` consumers, no ``torch.compile``, and both BPT and
the zero-shot lm-eval subset are pure forward / log-likelihood paths.
The single ``writes=("experts_impl",)`` slot matches the single
``ctx.set("experts_impl", ...)`` call below.

Circular-import contract (mirror of ``stage6/plugins/eval_environment.py``):
this module imports only from ``..context`` / ``...stage6.plugins.eval_environment``
/ stdlib — NEVER from ``stage6alt_thermometer`` or ``stage6alt.orchestrator``
at any scope (module-top OR function-local). The legacy monolith
re-imports ``ThermoEnvironmentPlugin`` at load time, so any back-edge
from here would deadlock the import; nothing in this module creates one.
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
    """Stage 6alt thermometer eval-environment plugin.

    Owns the Stage 6alt environment-setup concern: the MoE experts-implementation
    shim and the cu130/Hopper kernel patches applied to the student model
    before any forward pass. Both helpers are reused from
    ``stage6.plugins.eval_environment`` — see this module's docstring for
    why no symbol is relocated here.

    Dispatched live by ``stage6alt.orchestrator.run`` via
    ``walk_phases(("setup_thermo_environment",), plugins, run_ctx)``; the
    legacy ``stage6alt_thermometer.run()`` is a thin delegating shim.
    """

    name = "thermo_environment"
    paper = "Stage 6alt thermometer environment setup (project-original; reuses stage6.plugins.eval_environment helpers). See module docstring."
    config_key = "stage6_validate.thermometer"
    reads: tuple[str, ...] = ("model", "config")
    writes: tuple[str, ...] = ("experts_impl",)
    # `provides` is empty: this plugin sets up the runtime environment, it
    # does not need a calibration-pass accumulator. (Mirrors Stage 6's
    # ``EvalEnvironmentPlugin`` rationale.)
    provides: tuple[str, ...] = ()

    def is_enabled(self, config: dict) -> bool:
        """Always True — environment setup is UNCONDITIONAL.

        Every Stage 6alt thermometer run must set the experts-implementation
        shim and apply the cu130/Hopper kernel patches before any forward
        pass. The declared ``config_key`` (``stage6_validate.thermometer``)
        is purely a registry-level locator for the thermometer sub-tree; the
        hook body actually reads the sibling
        ``stage6_validate.experts_implementation`` key off the parent
        ``stage6_validate`` tree (mirroring the legacy monolith). Neither
        sub-tree gates the plugin as a whole.
        """
        return True

    def contribute_artifact(self, ctx: Any) -> dict:
        """Return per-plugin artifact contributions for ``stage6alt_thermometer.json``.

        Environment setup writes no thermometer artifact rows of its own —
        the resolved ``experts_impl`` it publishes on ctx is consumed by
        downstream plugins (teacher provider, report) which record it in
        their own contributions. Returning ``{}`` keeps the contract
        symmetric with Stage 6's ``EvalEnvironmentPlugin``.
        """
        return {}

    def setup_thermo_environment(self, ctx: PipelineContext) -> None:
        """Phase hook — Stage 6alt thermometer environment setup (live).

        Dispatched by ``stage6alt.orchestrator.run`` via
        ``walk_phases(("setup_thermo_environment",), plugins, run_ctx)``.
        The legacy ``stage6alt_thermometer.run()`` is a thin delegator; tests
        (``test_stage6alt_plugin_corpus.test_setup_thermo_environment_hook``)
        invoke this hook directly.

        Performs, in order:

        1. **Experts-implementation shim** — ``_set_experts_implementation_s6``
           (env var ``EXPERTS_IMPLEMENTATION`` overrides the YAML default
           ``batched_mm``; the override mirrors the monolith).
        2. **cu130/Hopper kernel patches** — ``_apply_stage6_kernel_patches``
           on the student.

        The resolved ``experts_impl`` is written to ctx so the thermo
        teacher-side provider plugin can apply the matching shim without
        re-resolving the env var.
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
