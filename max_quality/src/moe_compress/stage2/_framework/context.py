"""Run-scope and per-layer context objects shared across Stage 2 plugins.

``RunContext`` is frozen: the orchestrator builds it once at the top of
``stage2_reap_ream.run`` and passes the same instance to every plugin. Plugins
must not mutate it; per-layer mutable state lives on ``LayerContext`` instead.

``LayerContext`` is the per-layer scratchpad. Every field is optional and
populated incrementally as the layer flows through the phases (profile →
cost → solve → refine → merge → post-merge → artifacts). ``LayerContext.extras``
is an escape hatch for plugin-private state that does not yet warrant a typed
slot.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunContext:
    """Run-scope inputs shared by every plugin for the entire Stage 2 invocation."""

    model: Any
    tokenizer: Any
    config: dict[str, Any]
    artifacts_dir: Path
    partial_dir: Path
    device: str


@dataclass
class LayerContext:
    """Per-layer mutable scratchpad. Plugins read/write these fields as phases run."""

    layer_idx: int
    layer_ref: Any  # MoELayerRef-like; opaque at this layer of abstraction.
    n_experts: int
    target: int  # number of experts to keep after merging.
    blacklist: tuple[int, ...] = ()       # protected expert ids (Stage 1 super-experts).
    protected: tuple[int, ...] = ()       # alias / extension slot; populated by plugins.
    freq: Any = None                      # routing-frequency tensor.
    scores: Any = None                    # REAP saliency scores.
    reap_acc: Any = None                  # ReapAccumulator instance (Task 7).
    ream_acc: Any = None                  # ReamCostAccumulator instance (Task 8/9).
    cov_acc: Any = None                   # InputCovarianceAccumulator instance.
    perm_cache: Any = None                # _PermAlignCache (Task 4).
    layer_input_acc: Any = None           # captured layer-input tensor (Task 10/16).
    ream_centroid_ids: tuple[int, ...] = ()
    ream_noncentroid_ids: tuple[int, ...] = ()
    assignment: Any = None                # solver output.
    delta: Any = None                     # cost matrix (children × centroids).
    grouped: Any = None                   # _build_grouped_from_assignment output.
    final_kept_ids: tuple[int, ...] = ()
    mean_assigned_cost: float | None = None
    heal_state: Any = None                # merge-heal book-keeping (Task 17).
    distill_state: Any = None             # per-group distill book-keeping (Task 16).
    extras: dict[str, Any] = field(default_factory=dict)
