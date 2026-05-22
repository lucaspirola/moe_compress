"""Stage 3 orchestrator — S3-1 scaffold.

This is the S3-1 scaffold: :func:`run` is a thin delegation to
:func:`moe_compress.stage3_svd.run`, the legacy Stage 3 monolith. Tasks
S3-2..S3-6 extract the SVD algorithm (D-Rank allocation, Swift-SVD+ α-search,
AA-SVD, Phase C.5 block-refine) into ``stage3/plugins/``; S3-7 replaces this
function body with the real plugin sequencer and deletes the monolith.

The local ``run`` wrapper is intentional — it is the stable seam S3-7 swaps,
so callers import ``moe_compress.stage3.run`` and never need to change.
"""
from __future__ import annotations

from pathlib import Path

from .. import stage3_svd as _legacy_stage3


def run(
    model,
    tokenizer,
    config: dict,
    artifacts_dir: Path,
    decomposition: "BudgetDecomposition",
    *,
    device=None,
    no_resume: bool = False,
) -> Path:
    """Delegate to the legacy Stage 3 monolith (``stage3_svd.run``).

    S3-1 scaffold: this forwards every argument unchanged to the legacy
    implementation. S3-7 replaces this body with the plugin sequencer.
    """
    return _legacy_stage3.run(
        model, tokenizer, config, artifacts_dir, decomposition,
        device=device, no_resume=no_resume,
    )
