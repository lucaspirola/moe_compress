"""Trainable-scope concern (RK-2 of the Router-KD plugin-architecture refactor).

Paper
-----
Hyeon & Do, "Is Retraining-Free Enough? The Necessity of Router
Calibration for Efficient MoE Compression" — arXiv:2603.02217 (§F.3,
Eq. 3, Table 1). See ``audit/spec_compliance/01_papers/2603.02217/source.md``.

Equation 3 (paper source.md L800-823): the per-sequence vocab-KL
distillation objective

    L_RKD(x; θ_T, θ_R) = (τ² / N_x) · Σ_{t=1..L-1} m_{t+1} · D_KL(p_T^{(t)} || p_S^{(t)})

where ``p_T^{(t)}``, ``p_S^{(t)}`` are the teacher/student temperature-softened
next-token vocab distributions at position ``t``, ``m_{t+1}`` is the
next-token mask (zero on padded positions), ``N_x = Σ m_{t+1}`` is the
unmasked-position count, ``τ`` is the distillation temperature, and the sum
runs over the ``L-1`` next-token positions of sequence ``x``. The per-position
KL term lives in :mod:`router_kd.plugins.vocab_kd`; the mask/normalizer/sum are
applied there. This plugin owns only the trainable-vs-frozen *parameter*
scope (the θ_R selection); the loss itself is the vocab_kd plugin's concern.

§F.3 fixes the calibration data and hyperparameters; Table 1 reports
the resulting recovery on Mixtral/Qwen-MoE post-pruning/post-merging.

Official code
-------------
**None published.** Verified 2026-05: the paper's source.md contains
no code link; first author Sieun Hyeon (Seoul National University) has
no public router-KD repo.

Calibration deviation D11 (SHARED with Stage 2 / Stage 2.5)
-----------------------------------------------------------
Paper §F.3 Table 1 uses ``c4``. The project uses multi-domain
Nemotron-Cascade-2-SFT-Data with weighted subsets — task-aware
calibration better matches target deployment distribution. The D11
row's canonical owner is :mod:`stage2.plugins.reap_scoring`.

Home of the Router-KD trainable/frozen-parameter scope concern, extracted
from the legacy ``stage5_router_kd.py`` monolith. RK-2 covers TWO pieces with
TWO different patterns:

Piece A — relocated verbatim (the S3-2/S3-3/S4-3 pattern):
  ``_freeze_non_routers`` is a STANDALONE function in the monolith. It is
  relocated here character-for-character; ``stage5_router_kd.py`` (now a thin
  shim — see below) re-imports it via a ``# noqa: F401`` block so external
  callers/tests (``test_stage5_merge_repair.py``) keep their import paths.

Piece B — the trainable/frozen pattern-conflict check (the S3-4/S4-2 pattern):
  originally inline ``run()`` code in the monolith, this check is now owned by
  the ``setup_trainable_scope`` hook below. RK-8 has landed: the Router-KD
  orchestrator (:mod:`router_kd.orchestrator`) imports, registers, and
  dispatches this plugin via ``walk_phases(("setup_trainable_scope",), …)``,
  and ``stage5_router_kd.run`` is a thin shim delegating to it.

Circular-import note (mirror of ``stage4/plugins/eora_inputs.py``): this
module imports only from ``...pipeline.*`` / ``..context`` / stdlib / torch —
NEVER from ``stage5_router_kd`` or ``router_kd.orchestrator`` at any scope
(module-top OR function-local). The monolith shim re-imports *this* module at
load time, so a ``from ..stage5_router_kd import ...`` here would deadlock the
import; nothing in this module does that.

``TrainableScopePlugin`` is LIVE: it is enabled-and-dispatched by the
Router-KD orchestrator at every Stage 5 / Stage 2.5 run.
"""
from __future__ import annotations

import logging
from typing import Any

import torch.nn as nn

from ..context import PipelineContext

log = logging.getLogger(__name__)


def _freeze_non_routers(model: nn.Module, trainable_patterns: list[str]) -> None:
    for name, p in model.named_parameters():
        p.requires_grad_(any(pat in name for pat in trainable_patterns))


class TrainableScopePlugin:
    """Router-KD trainable-scope plugin (RK-2 — LIVE since RK-8).

    Owns the Router-KD trainable/frozen-parameter scope concern: freezing
    every non-router parameter before the student is compiled
    (``_freeze_non_routers``, relocated verbatim above) and the
    trainable/frozen pattern-conflict sanity check that fails loud when a
    parameter name matches BOTH ``trainable_name_patterns`` and
    ``frozen_name_patterns``.

    RK-2 covers a mixed pattern: ``_freeze_non_routers`` is relocated verbatim
    (the monolith shim re-imports it), while the conflict check — originally
    inline ``run()`` code in the monolith — is now owned by the
    ``setup_trainable_scope`` hook below. RK-8 has landed: the Router-KD
    orchestrator imports, registers, and dispatches this plugin (see
    :mod:`router_kd.orchestrator`); ``stage5_router_kd.run`` is a thin shim
    delegating to that orchestrator.
    """

    name = "trainable_scope"
    paper = (
        "Router-KD vocab-KL distillation Eq. 3 — arXiv:2603.02217 (Hyeon & Do). "
        "Official code: none published. "
        "Concern: trainable/frozen-parameter scope (router-only freezing). "
        "Calibration deviation D11 (SHARED — canonical owner "
        ":mod:`stage2.plugins.reap_scoring`). "
        "See module docstring for Eq. 3 expansion and full citation."
    )
    # Primary config key (asserted by tests). The hook *also* reads
    # ``stage5_router_kd.frozen_name_patterns`` (optional, informational) for
    # the conflict-overlap sanity check in ``setup_trainable_scope``.
    config_key = "stage5_router_kd.trainable_name_patterns"
    # ``student``/``model`` are the primary/fallback slot for the model the
    # hook reads (RK-8 keeps both for back-compat with callers that publish
    # under either name); both are declared here.
    reads: tuple[str, ...] = ("student", "model", "config")
    # Empty: the freeze mutates ``requires_grad`` on the student parameters
    # in place — there is no new context slot to publish.
    writes: tuple[str, ...] = ()
    # Empty: setting up the trainable scope needs no calibration pass.
    provides: tuple[str, ...] = ()

    def is_enabled(self, config: dict) -> bool:
        """Always True — freezing non-router parameters is UNCONDITIONAL.

        Every Router-KD run must freeze the non-router parameters before
        training; ``config_key`` only names *which* parameters stay trainable,
        it never gates the plugin as a whole.
        """
        return True

    def contribute_artifact(self, ctx: Any) -> dict:
        return {}

    def setup_trainable_scope(self, ctx: PipelineContext) -> None:
        """Phase hook — Router-KD trainable-scope setup (live since RK-8).

        Dispatched by :mod:`router_kd.orchestrator` via
        ``walk_phases(("setup_trainable_scope",), …)`` immediately after
        ``load_teacher_cache`` and BEFORE the student is compiled. Reproduces
        (in monolith order) the original inline ``run()`` block: read
        ``frozen_name_patterns`` / ``trainable_name_patterns`` from
        ``stage5_router_kd`` config, run the trainable/frozen pattern-conflict
        check against the compile-unwrapped student's parameter names —
        raising the verbatim ``RuntimeError`` on any overlap — then freeze
        every non-router parameter via the local ``_freeze_non_routers``.
        """
        # Required slots — direct get(): a missing one is a wiring bug and
        # SHOULD raise. The student may be published under either "student"
        # or "model"; has()-guard the preferred slot.
        student = ctx.get("student") if ctx.has("student") else ctx.get("model")
        config = ctx.get("config")
        s5 = config["stage5_router_kd"]

        # Sanity check: raise if any parameter name matches BOTH trainable and
        # frozen patterns. ``frozen_name_patterns`` is informational only — it
        # is NOT consulted by ``_freeze_non_routers``; the freeze is driven
        # entirely by
        #     ``requires_grad_(any(p in name for p in trainable_name_patterns))``.
        # Names matching only ``frozen_name_patterns`` are still correctly
        # frozen because they fail the trainable-pattern check. The frozen
        # patterns list exists solely for this conflict-overlap sanity check
        # below: trainable wins on overlap, but a name in both is almost
        # certainly a config bug.
        _frozen_patterns = s5.get("frozen_name_patterns", []) or []
        _trainable_patterns = s5["trainable_name_patterns"]
        if _frozen_patterns:
            # Unwrap convention (conflict-check ONLY): the conflict scan
            # walks the compile-unwrapped student so parameter names match
            # the user's pattern strings (``torch.compile`` prefixes names
            # with ``_orig_mod.``). ``_freeze_non_routers`` below is called
            # with the *wrapped* student deliberately — ``requires_grad_``
            # is shared with the underlying parameter tensor, so the freeze
            # is identical either way.
            _base_for_check = getattr(student, "_orig_mod", student)
            _conflicts = [
                name for name, _ in _base_for_check.named_parameters()
                if any(pat in name for pat in _trainable_patterns)
                and any(pat in name for pat in _frozen_patterns)
            ]
            if _conflicts:
                raise RuntimeError(
                    f"Stage 5 config error: {len(_conflicts)} parameter(s) match BOTH "
                    f"trainable_name_patterns and frozen_name_patterns (e.g. {_conflicts[:3]}). "
                    "Resolve the overlap in stage5_router_kd config."
                )
        _freeze_non_routers(student, _trainable_patterns)
