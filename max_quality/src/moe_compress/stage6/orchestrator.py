"""Stage 6 orchestrator — thin delegation shim (S6-1 scaffold).

S6-1 ships this module as a thin delegation to the legacy
``stage6_validate.run`` monolith. Tasks S6-2..S6-7 will extract the
Stage 6 validation algorithm (WikiText-2 PPL, zero-shot, generative evals,
imatrix pipeline, threshold gating) into ``stage6/plugins/``. S6-8 flips
the relationship: :func:`run` here becomes the REAL orchestrator and
``stage6_validate.run`` becomes the thin shim that delegates to it.

Until S6-8 this is intentionally a pass-through — zero behaviour change.
"""
from __future__ import annotations

from pathlib import Path


def run(model, tokenizer, config: dict, artifacts_dir: Path, *, device=None) -> Path:
    """Run Stage 6 validation.

    Delegates to the legacy ``stage6_validate.run`` monolith until S6-8
    extracts all plugins and flips the delegation direction.
    """
    # Function-local import: at S6-8, ``stage6_validate.run`` will become a
    # thin shim that delegates HERE, making a module-top import circular.
    # Keeping it local here honours the pattern established by stage3 / stage4.
    from .. import stage6_validate as _stage6_validate
    return _stage6_validate.run(
        model, tokenizer, config, artifacts_dir, device=device,
    )


__all__ = ["run"]
