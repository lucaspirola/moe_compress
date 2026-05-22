"""Router-KD orchestrator — RK-1 scaffold (thin delegation).

At the RK-1 scaffold stage this :func:`run` is a thin delegation to the
legacy Router-KD monolith :func:`moe_compress.stage5_router_kd.run`. Tasks
RK-2..RK-7 extract the unified KD algorithm (the Stage 2.5 + Stage 5 router
fine-tuning loop) into ``router_kd/plugins/``; RK-8 swaps this delegating
body for the real plugin-driven phase sequencer.

The local :func:`run` wrapper is the stable seam RK-8 swaps: callers (the
package ``__init__`` re-export, the ``router_kd.stage`` adapter) bind to
*this* function, so flipping its body to the real orchestrator in RK-8 needs
no caller changes.
"""
from __future__ import annotations

from pathlib import Path

from .. import stage5_router_kd as _legacy_router_kd


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
    """Delegate to the legacy Router-KD monolith (``stage5_router_kd.run``).

    RK-1 scaffold: forwards every argument verbatim — the four positionals
    (``student``, ``tokenizer``, ``config``, ``artifacts_dir``) plus the three
    keyword-only arguments (``device``, ``no_resume``, ``stage_key``) — to the
    unified monolith and returns its ``Path`` result unchanged. ``stage_key``
    selects Stage 2.5 (``"stage2p5"``) vs Stage 5 (``"stage5"``).

    This signature is kept byte-identical to the monolith ``run`` — parameter
    names, kinds, defaults, and annotations — so the two stay swap-compatible
    (a signature-parity test guards this). RK-8 replaces this body with the
    real plugin sequencer; the seam itself does not move.
    """
    return _legacy_router_kd.run(
        student, tokenizer, config, artifacts_dir,
        device=device, no_resume=no_resume, stage_key=stage_key,
    )


__all__ = ["run"]
