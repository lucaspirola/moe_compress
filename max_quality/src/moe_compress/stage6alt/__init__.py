"""Stage 6alt — Thermometer validation (plugin architecture, S6A-1 scaffold).

S6A-1 ships this package skeleton — an ``__init__`` re-exporting ``run``,
an ``orchestrator`` module that thinly delegates to the legacy
``stage6alt_thermometer.run`` monolith, a ``context`` re-export shim, and
an (empty) ``plugins`` package. Tasks S6A-2..S6A-5 will extract the Stage
6alt thermometer algorithm (calibration-corpus build, teacher-cache
provider, BPT measurement, lm-eval subset, and validation report) into
``stage6alt/plugins/``. S6A-6 flips the delegation direction:
:func:`run` here becomes the REAL orchestrator and
``stage6alt_thermometer.run`` becomes the thin shim that delegates to it;
that same task also introduces the ``STAGE6ALT`` plugin manifest object.

Until S6A-6, no ``STAGE6ALT`` symbol is exported — only :func:`run`.
"""
from .orchestrator import run

__all__ = ["run"]
