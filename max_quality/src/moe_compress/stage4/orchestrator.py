"""Stage 4 orchestrator — S4-1 scaffold (thin delegation).

S4-1 ships this module as a thin delegation to the legacy
``moe_compress.stage4_eora.run`` monolith. :func:`run` here forwards every
argument unchanged to it. Tasks S4-2..S4-3 extract the EoRA algorithm
(``eora_inputs`` residual input collection, ``eora_compensation`` √Λ-weighted
residual compensation) into ``stage4/plugins/``. S4-4 then replaces this
delegating body with the real plugin-driven phase sequencer and deletes the
monolith.

The local ``def run`` wrapper below — rather than a raw
``from ..stage4_eora import run`` re-export — is the stable seam S4-4 swaps:
its body changes, its signature does not.
"""
from __future__ import annotations

from pathlib import Path

from .. import stage4_eora as _legacy_stage4


def run(
    model,
    tokenizer,
    config: dict,
    artifacts_dir: Path,
    *,
    no_resume: bool = False,
) -> Path:
    """Delegate to the legacy Stage 4 monolith (``stage4_eora.run``).

    S4-1 scaffold: a thin pass-through that forwards every argument unchanged.
    Matches the legacy signature exactly (4 positionals + kw-only
    ``no_resume``) so S4-4 can swap the body for the real plugin sequencer
    without touching any caller. Returns the Stage 4 output directory.
    """
    return _legacy_stage4.run(
        model, tokenizer, config, artifacts_dir, no_resume=no_resume,
    )


__all__ = ["run"]
