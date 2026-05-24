"""Stage 5 — Router KD via vocabulary-level output distillation (thin shim).

Reference: 2603.02217 (Router Knowledge Distillation), Eq. 3.

The paper distills at the **vocabulary output logit** level, NOT at the
intermediate router gate level. From §4:

  "By distilling output logits rather than matching router gate values
   explicitly, Router KD avoids requiring the teacher and student to share
   identical expert sets or gate dimensionalities."

This means the loss is:

  L_RKD = (τ²/N_x) Σ_t  KL(softmax(z_T^t / τ) ‖ softmax(z_S^t / τ))

where z_T, z_S ∈ ℝ^|V| are the teacher/student vocabulary logits for
next-token prediction, and the sum is over unmasked token positions.

Only router weights are trainable; all expert weights are frozen. The
vocabulary-level signal propagates gradients through the full forward pass
including the routing decisions, which naturally adapts the router to the
compressed expert set.

RK-8 status — thin shim
-----------------------
The Router-KD algorithm has been fully decomposed into the ``router_kd/``
plugin package: ``router_kd.orchestrator.run`` is now the REAL plugin-driven
phase sequencer and :func:`run` here is a thin shim that delegates to it. The
RK-2..RK-7 ``# noqa: F401`` re-import blocks below are kept so external
callers/tests keep their historical ``stage5_router_kd.<name>`` import paths;
``_save_stage5_checkpoint`` and ``_set_experts_implementation`` were relocated
verbatim into ``router_kd.orchestrator`` (RK-8 made the orchestrator their only
caller) and are re-imported here for the same reason.
"""
from __future__ import annotations

from pathlib import Path

# RK-8: _save_stage5_checkpoint + _set_experts_implementation relocated to
# router_kd/orchestrator (the orchestrator is now their only caller).
# Re-imported so external callers/tests keep their `stage5_router_kd.<name>`
# import paths resolvable.
from .router_kd.orchestrator import (  # noqa: F401
    _save_stage5_checkpoint,
    _set_experts_implementation,
)

# RK-2: _freeze_non_routers relocated to router_kd/plugins/trainable_scope.
# Re-imported so run() + external callers/tests (test_stage5_merge_repair.py)
# keep their import paths.
from .router_kd.plugins.trainable_scope import (  # noqa: F401
    _freeze_non_routers,
)

# RK-3: _move_optimizer_state_to_device relocated to
# router_kd/plugins/kd_optimizer. Re-imported so external callers/tests keep
# their `stage5_router_kd.<name>` import path.
from .router_kd.plugins.kd_optimizer import (  # noqa: F401
    _move_optimizer_state_to_device,
)

# RK-4: the vocab-KL kernel (_chunked_vocab_kl / _combine_kd_loss) and the
# NaN sanity probes relocated to router_kd/plugins/vocab_kd. Re-imported so
# external callers/tests keep their `stage5_router_kd.<name>` import paths.
from .router_kd.plugins.vocab_kd import (  # noqa: F401
    _chunked_vocab_kl,
    _combine_kd_loss,
    _log_first_batch_sanity,
    _dump_nan_diagnostics,
    _check_param_sanity,
)

# RK-6: the Stage-2.5 merge-repair pieces relocated to
# router_kd/plugins/merge_repair. Re-imported so external callers/tests
# (test_stage5_merge_repair.py) keep their `stage5_router_kd.<name>` import
# paths.
from .router_kd.plugins.merge_repair import (  # noqa: F401
    _load_merge_map,
    _merged_centroid_rows,
    _select_merge_repair_layers,
    _experts_param_tensors,
    _unfreeze_merged_experts,
    _LayerOutputCapture,
    _merge_repair_mse,
)

# RK-7: _save_best_router_state relocated to router_kd/plugins/early_stop.
# Re-imported so external callers/tests keep their `stage5_router_kd.<name>`
# import path; the inline best-tracker / early-stop glue is reproduced
# (Pattern B) in EarlyStopPlugin's hooks, which RK-8's orchestrator drives.
from .router_kd.plugins.early_stop import (  # noqa: F401
    _save_best_router_state,
)


def run(
    student,
    tokenizer,
    config: dict,
    artifacts_dir: Path,
    *,
    device=None,
    no_resume: bool = False,
    stage_key: str = "stage5",
) -> Path:
    """Run Router KD — thin shim delegating to the plugin orchestrator.

    RK-8: the real plugin-driven phase sequencer lives in
    :func:`moe_compress.router_kd.orchestrator.run`. This shim forwards every
    argument verbatim — the four positionals (``student``, ``tokenizer``,
    ``config``, ``artifacts_dir``) plus the three keyword-only arguments
    (``device``, ``no_resume``, ``stage_key``) — and returns its ``Path``
    result unchanged. ``stage_key`` selects Stage 2.5 (``"stage2p5"``) vs
    Stage 5 (``"stage5"``); the config section read is always
    ``stage5_router_kd`` regardless.

    The signature is kept byte-identical to the orchestrator ``run`` —
    parameter names, kinds, defaults, and annotations — so the two stay
    swap-compatible (a signature-parity test guards this).
    """
    from .router_kd.orchestrator import run as _orchestrator_run

    return _orchestrator_run(
        student, tokenizer, config, artifacts_dir,
        device=device, no_resume=no_resume, stage_key=stage_key,
    )
