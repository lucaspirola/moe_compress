"""Stage 6alt orchestrator — thin delegation shim (S6A-1 scaffold).

S6A-1 ships this module as a thin delegation to the legacy
``stage6alt_thermometer.run`` monolith. Tasks S6A-2..S6A-5 will extract
the Stage 6alt thermometer algorithm (calibration-corpus build,
teacher-cache provider, BPT measurement, lm-eval subset, validation
report) into ``stage6alt/plugins/``. S6A-6 flips the relationship:
:func:`run` here becomes the REAL orchestrator and
``stage6alt_thermometer.run`` becomes the thin shim that delegates to it.

Until S6A-6 this is intentionally a pass-through — zero behaviour change.

The ``stage6alt_thermometer`` import is performed inside :func:`run` (a
function-local import) rather than at module top. At S6A-6,
``stage6alt_thermometer.run`` will become a thin shim that delegates HERE,
which would make a module-top import circular. Keeping the import local
means S6A-6 can flip the seam without touching any call site that imports
:func:`run` from ``moe_compress.stage6alt``.
"""
from __future__ import annotations

from pathlib import Path


def run(model, tokenizer, config: dict, artifacts_dir: Path, *, device=None) -> Path:
    """Run Stage 6alt thermometer validation.

    Delegates to the legacy ``stage6alt_thermometer.run`` monolith until
    S6A-6 extracts all plugins and flips the delegation direction.
    """
    # Function-local import: at S6A-6, ``stage6alt_thermometer.run`` will
    # become a thin shim that delegates HERE, making a module-top import
    # circular. Keeping it local honours the pattern established by
    # stage3 / stage4 / stage6.
    from moe_compress import stage6alt_thermometer
    return stage6alt_thermometer.run(
        model, tokenizer, config, artifacts_dir, device=device,
    )


__all__ = ["run"]
