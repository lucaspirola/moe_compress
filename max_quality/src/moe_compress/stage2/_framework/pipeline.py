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

from typing import Any

from ...pipeline.context import PipelineContext


class Stage2Pipeline:
    """Hold an ordered list of plugins and dispatch lifecycle phases."""

    # Canonical per-layer phase order (T6 execution tuple). The four fine-grained
    # sub-hooks (``compute_cost``, ``apply_cost_mask``, ``solve_assignment``,
    # ``refine_assignment``) are an open vocabulary discovered reflectively by
    # the tolerant ``getattr`` walk below; they are NOT iterated here — the
    # LegacyAdapter folds them into ``compute_assignment`` so the bump-loop
    # control flow stays intact.
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

    def __init__(self, plugins: list[Any]) -> None:
        self.plugins: list[Any] = list(plugins)

    def run_setup(self, run_ctx: PipelineContext) -> None:
        """Call ``on_run_setup`` on every plugin in registration order.

        Hook lookup is tolerant (``getattr`` + ``callable``): a plugin that
        does not implement the hook is skipped, matching
        :func:`tools.phase_walker.walk_phases`. Plugins no longer rely on a
        base class supplying no-op defaults.
        """
        for plugin in self.plugins:
            hook = getattr(plugin, "on_run_setup", None)
            if callable(hook):
                hook(run_ctx)

    def run_teardown(self, run_ctx: PipelineContext) -> None:
        """Call ``on_run_teardown`` on every plugin in registration order.

        Tolerant hook lookup — see :meth:`run_setup`.
        """
        for plugin in self.plugins:
            hook = getattr(plugin, "on_run_teardown", None)
            if callable(hook):
                hook(run_ctx)

    def run_layer(self, ctx: PipelineContext) -> None:
        """Drive one layer through every phase in canonical order.

        For each phase, every plugin's hook is called in registration order.
        Hook lookup is tolerant (``getattr`` + ``callable``): a plugin that
        does not implement a given phase is skipped — plugins no longer rely
        on a base class supplying no-op defaults, matching
        :func:`tools.phase_walker.walk_phases`.

        ``write_artifacts`` is the only hook with a second positional argument
        (``partial_dir``); the pipeline pulls that value from the LegacyAdapter
        plugin (or any plugin exposing ``partial_dir`` as an instance
        attribute) so the T6 contract stays narrow. T13/T16/T17 will surface
        ``partial_dir`` on the per-layer context directly when those tasks land.
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
                hook = getattr(plugin, phase, None)
                if not callable(hook):
                    continue
                if phase == "write_artifacts":
                    hook(ctx, partial_dir)
                else:
                    hook(ctx)
