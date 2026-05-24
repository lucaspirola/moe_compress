"""Trainable-scope concern (RK-2 of the Router-KD plugin-architecture refactor).

Home of the Router-KD trainable/frozen-parameter scope concern, extracted
from the legacy ``stage5_router_kd.py`` monolith. RK-2 covers TWO pieces with
TWO different patterns:

Piece A — relocated verbatim (the S3-2/S3-3/S4-3 pattern):
  ``_freeze_non_routers`` is a STANDALONE function in the monolith. It is
  relocated here character-for-character; the ``stage5_router_kd.py`` monolith
  re-imports it (``# noqa: F401`` block) so ``run()`` and external
  callers/tests (``test_stage5_merge_repair.py``) keep their import paths.

Piece B — reproduced in an inert hook (the S3-4/S4-2 pattern):
  the trainable/frozen pattern-conflict check is INLINE ``run()`` code in the
  monolith, not a standalone function — there is nothing to relocate. The
  ``setup_trainable_scope`` hook below REPRODUCES that inline logic faithfully;
  the monolith ``run()`` is NOT modified for it. This is an intentional,
  temporary logic duplication that resolves at RK-8 when the monolith ``run()``
  is deleted and this hook is wired live.

Circular-import note (mirror of ``stage4/plugins/eora_inputs.py``): this
module imports only from ``...pipeline.*`` / ``..context`` / stdlib / torch —
NEVER from ``stage5_router_kd`` or ``router_kd.orchestrator`` at any scope
(module-top OR function-local). The monolith re-imports *this* module at load
time, so a ``from ..stage5_router_kd import ...`` here would deadlock the
import; nothing in this module does that.

``TrainableScopePlugin`` is registered-but-INERT at RK-2 — no orchestrator
walk or test invokes its ``setup_trainable_scope`` hook. RK-8 plugs the hook
into the live Router-KD plugin sequencer and deletes the monolith ``run()``.
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
    """Router-KD trainable-scope plugin (RK-2 — registered-but-INERT).

    Owns the Router-KD trainable/frozen-parameter scope concern: freezing
    every non-router parameter before the student is compiled
    (``_freeze_non_routers``, relocated verbatim above) and the
    trainable/frozen pattern-conflict sanity check that fails loud when a
    parameter name matches BOTH ``trainable_name_patterns`` and
    ``frozen_name_patterns``.

    RK-2 covers a mixed pattern: ``_freeze_non_routers`` is relocated verbatim
    (the monolith re-imports it), while the conflict check — inline ``run()``
    code in the monolith — is reproduced in the ``setup_trainable_scope`` hook
    below; the monolith ``run()`` is NOT modified for it (see module
    docstring). RK-2 wires this class into the plugin registry as metadata
    only — no walk or test invokes ``setup_trainable_scope``. RK-8 plugs the
    hook into the live Router-KD plugin sequencer.
    """

    name = "trainable_scope"
    paper = "Router Knowledge Distillation (paper 2603.02217, Eq. 3)."
    config_key = "stage5_router_kd.trainable_name_patterns"
    # ``student``/``model`` are the primary/fallback slot for the model the
    # hook reads (RK-8 will canonicalize to one); both are declared here.
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
        """Phase hook — Router-KD trainable-scope setup (RK-8 wiring surface).

        INERT at RK-2: no orchestrator walk or test invokes this hook. RK-8
        replaces the Router-KD orchestrator body with the plugin sequencer and
        dispatches this hook in place of the monolith's inline ``run()``
        conflict-check + ``_freeze_non_routers`` call. The body below
        reproduces that inline block faithfully — it is dead code at RK-2 but
        RK-8 relies on it once the monolith ``run()`` is deleted.

        Reproduces (in monolith order): read ``frozen_name_patterns`` /
        ``trainable_name_patterns`` from ``stage5_router_kd`` config, run the
        trainable/frozen pattern-conflict check against the (compile-unwrapped)
        student's parameter names — raising the verbatim ``RuntimeError`` on
        any overlap — then freeze every non-router parameter via the local
        ``_freeze_non_routers``.
        """
        # Required slots — direct get(): a missing one is a wiring bug and
        # SHOULD raise. The student may be published under either "student"
        # or "model"; has()-guard the preferred slot.
        student = ctx.get("student") if ctx.has("student") else ctx.get("model")
        config = ctx.get("config")
        s5 = config["stage5_router_kd"]

        # Sanity check: warn if any parameter name matches BOTH trainable and
        # frozen patterns (frozen_name_patterns is informational only — it is NOT
        # consulted by _freeze_non_routers; freeze is driven entirely by
        # `requires_grad_(any(p in name for p in trainable_name_patterns))`.
        # Names that match only frozen_name_patterns are still correctly frozen
        # because they fail the trainable-pattern check. The patterns list exists
        # solely for the conflict-overlap sanity check below; trainable wins,
        # but a name in both is almost certainly a config bug).
        _frozen_patterns = s5.get("frozen_name_patterns", []) or []
        _trainable_patterns = s5["trainable_name_patterns"]
        if _frozen_patterns:
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
