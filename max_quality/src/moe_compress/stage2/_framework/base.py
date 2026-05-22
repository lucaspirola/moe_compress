"""Stage 2 plugin base class: lifecycle hooks with no-op defaults.

Subclasses opt into the phases they care about by overriding hooks. Slot-style
hooks (``compute_cost``, ``solve_assignment``) return ``None`` by default so
``PluginRegistry.dispatch_first`` skips them; the first plugin that returns a
non-None value wins.
"""
from __future__ import annotations

from typing import Any

from .context import LayerContext, RunContext


class Stage2Plugin:
    """Base class for every Stage 2 plugin. All hooks are optional no-ops."""

    name: str = ""
    enabled_by: tuple[str, ...] = ()

    @classmethod
    def is_enabled(cls, cfg: dict[str, Any]) -> bool:
        """True iff every flag in ``enabled_by`` is truthy in cfg['stage2_reap_ream']."""
        s2 = cfg.get("stage2_reap_ream", {}) if isinstance(cfg, dict) else {}
        for flag in cls.enabled_by:
            if not s2.get(flag):
                return False
        return True

    # ------------------------------------------------------------------
    # Run-scope hooks
    # ------------------------------------------------------------------
    def on_run_setup(self, run_ctx: RunContext) -> None:
        """Called once before any layer is processed."""

    def on_run_teardown(self, run_ctx: RunContext) -> None:
        """Called once after all layers are processed."""

    # ------------------------------------------------------------------
    # Per-layer lifecycle (in execution order)
    # ------------------------------------------------------------------
    def on_layer_setup(self, ctx: LayerContext) -> None:
        """Called at the start of a layer, before any profiling."""

    def on_profile(self, ctx: LayerContext) -> None:
        """Attach forward hooks / collect activations for this layer."""

    def on_score(self, ctx: LayerContext) -> None:
        """Compute and publish per-layer saliency scores (REAP) for downstream phases."""

    def compute_cost(self, ctx: LayerContext) -> Any | None:
        """Return a cost matrix for this layer, or None to defer to another plugin."""
        return None

    def apply_cost_mask(self, ctx: LayerContext, delta: Any) -> tuple[Any, dict] | None:
        """Optionally mutate the cost matrix (e.g. skip-merge floor); return (new_delta, info)."""
        return None

    def solve_assignment(self, ctx: LayerContext, delta: Any) -> Any | None:
        """Return an Assignment for this layer, or None to defer to another plugin."""
        return None

    def refine_assignment(
        self, ctx: LayerContext, asg: Any, delta: Any
    ) -> tuple[Any, Any, dict] | None:
        """Optionally refine an assignment (two-opt, EM); return (new_asg, new_delta, info)."""
        return None

    def compute_assignment(self, ctx: LayerContext) -> None:
        """Compound phase: cost → solve → refine → grouping (the bump loop).

        T6 keeps this collapsed into a single hook on the LegacyAdapter; tasks
        T8/T9/T13/T14/T15 will decompose it back into the four fine-grained
        sub-hooks above (``compute_cost`` / ``apply_cost_mask`` /
        ``solve_assignment`` / ``refine_assignment``) when real algorithm
        plugins replace the legacy slice. Default is a no-op so plugins that
        only care about other phases can stay quiet.
        """

    def pre_merge_snapshot(self, ctx: LayerContext) -> None:
        """Snapshot pre-merge layer state (used by distill / heal plugins)."""

    def merge(self, ctx: LayerContext) -> None:
        """Execute the in-place merge for this layer."""

    def post_merge(self, ctx: LayerContext) -> None:
        """Run post-merge work (per-group distill, layer heal)."""

    def write_artifacts(self, ctx: LayerContext, partial_dir: Any) -> dict[str, Any]:
        """Return extra key/value pairs to merge into merge_*.json."""
        return {}

    def on_layer_teardown(self, ctx: LayerContext) -> None:
        """Release per-layer resources (detach hooks, drop accumulators)."""
