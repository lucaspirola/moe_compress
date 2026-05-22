"""Stage 2 pipeline shell.

Task 6 wires ``run_layer``: it walks every phase in canonical order and dispatches
to each plugin's matching hook. The phase tuple is tightened from the original
11-entry sketch (T1) to the 8-entry execution tuple — the bump loop (cost +
solve + refine + grouping) is collapsed into a single ``compute_assignment``
phase because it is multi-pass and the gates that drive its inner iteration do
not yet have a plugin-substrate-friendly representation. Tasks 8/9/13/14/15
will decompose ``compute_assignment`` back into the four fine-grained phases
when they extract real cost / solver / refine plugins. See
``docs/superpowers/plans/2026-05-22-stage2-task6-pipeline-drives-loop.md``.
"""
from __future__ import annotations

from .base import Stage2Plugin
from .context import LayerContext, RunContext


class Stage2Pipeline:
    """Hold an ordered list of plugins and dispatch lifecycle phases."""

    # Canonical per-layer phase order (T6 execution tuple). The four fine-grained
    # sub-hooks (``compute_cost``, ``apply_cost_mask``, ``solve_assignment``,
    # ``refine_assignment``) remain declared on ``Stage2Plugin`` for future tasks
    # but are NOT iterated here — the LegacyAdapter folds them into
    # ``compute_assignment`` so the bump-loop control flow stays intact.
    phases: tuple[str, ...] = (
        "on_layer_setup",
        "on_profile",
        "on_score",
        "compute_assignment",
        "pre_merge_snapshot",
        "merge",
        "post_merge",
        "write_artifacts",
        "on_layer_teardown",
    )

    def __init__(self, plugins: list[Stage2Plugin]) -> None:
        self.plugins: list[Stage2Plugin] = list(plugins)

    def run_setup(self, run_ctx: RunContext) -> None:
        """Call ``on_run_setup`` on every plugin in registration order."""
        for plugin in self.plugins:
            plugin.on_run_setup(run_ctx)

    def run_teardown(self, run_ctx: RunContext) -> None:
        """Call ``on_run_teardown`` on every plugin in registration order."""
        for plugin in self.plugins:
            plugin.on_run_teardown(run_ctx)

    def run_layer(self, ctx: LayerContext) -> None:
        """Drive one layer through every phase in canonical order.

        For each phase, every plugin's hook is called in registration order.
        ``write_artifacts`` is the only hook with a second positional argument
        (``partial_dir``); the pipeline pulls that value from the LegacyAdapter
        plugin (or any plugin exposing ``partial_dir`` as an instance
        attribute) so the T6 contract stays narrow. T13/T16/T17 will surface
        ``partial_dir`` on ``LayerContext`` directly when those tasks land.
        """
        # Discover ``partial_dir`` from any plugin that exposes it as a
        # run-scope attribute. LegacyAdapter sets ``self.partial_dir`` during
        # ``__init__``; future plugins that need it will follow the same
        # convention. A plugin without the attribute falls through to None.
        partial_dir = None
        for p in self.plugins:
            if hasattr(p, "partial_dir"):
                partial_dir = getattr(p, "partial_dir")
                break

        for phase in self.phases:
            for plugin in self.plugins:
                hook = getattr(plugin, phase)
                if phase == "write_artifacts":
                    hook(ctx, partial_dir)
                else:
                    hook(ctx)
