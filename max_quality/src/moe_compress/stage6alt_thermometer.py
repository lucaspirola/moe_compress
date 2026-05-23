"""Stage 6alt — "thermometer" cheap directional eval.

A lightweight alternative to the full Stage 6 validation suite. Where Stage 6
runs WikiText PPL + lm-eval + 164-problem HumanEval + 500-problem MATH-500 on
both student and teacher (~$50-120 per ablation row at thinking-mode generation
speeds), this stage measures a single forward-pass signal good enough to tell
the operator whether an ablation knob HELPED or HURT vs the prior row.

Primary metric — bits-per-token (BPT): mean next-token NLL (in bits) over a
fixed 64-seq x 2048-token corpus. Pure forward pass, no generation. Reports
`student_bpt`, `teacher_bpt`, and `bpt_gap = student_bpt - teacher_bpt`.

CORPUS CHOICE (config `thermometer.corpus`) — critical for interpreting bpt_gap:
  - "wikitext": WikiText-2 test split. General text the student was NOT
    Stage-2.5-trained on, so `bpt_gap` is a FAIR teacher-vs-student
    compression-damage number (expected sign: positive — student worse).
  - "nemotron" (default): a held-out slice of the Nemotron-Cascade SFT data.
    This is the SAME distribution Stage 2.5 Router KD trains the student on,
    so `bpt_gap` here CONFLATES compression damage with the student's
    distribution adaptation (it can go negative). Trust it only for cross-row
    A0..A11 RANKING — where the adaptation is common-mode and cancels — never
    as an absolute teacher-vs-student claim.

Secondary signals — ARC-Easy/HellaSwag zero-shot `acc_norm` summed, and
`top1_agreement`: the fraction of corpus positions where student and teacher
argmax the same next token. Agreement is a training-distribution-independent
damage measure (it asks "did the student stay faithful to the teacher",
not "did the student get good at this text").

The teacher BPT, lm-eval, and per-token argmax are computed once and cached to
a sweep-shared file so all 12 ablation rows reuse them (teacher is constant).

Selected via `config["stage6_validate"]["mode"] == "thermometer"`; see
run_pipeline.py's Stage 6 dispatch. Default mode is "full" (stage6_validate).
"""

from __future__ import annotations

import logging
from pathlib import Path

# S6A-2: re-export the Stage 6alt thermometer corpus Pattern-A symbols
# (constants + functions) from their plugin home so existing import paths
# (this module's own `run()`, plus external callers like
# `stage2/orchestrator.py`'s xD calibration that imports
# `_thermo_wikitext_tensor`) keep working unchanged. The plugin classes
# `ThermoEnvironmentPlugin` / `ThermoCorpusPlugin` are imported alongside
# so an external registry walker can pick them up via the monolith too.
from .stage6alt.plugins.thermo_corpus import (  # noqa: F401
    THERMO_SEED_OFFSET,
    _DEFAULT_SUBSET_WEIGHTS,
    _thermo_corpus_spec,
    _thermo_wikitext_tensor,
    _build_thermo_corpus,
    ThermoCorpusPlugin,
)
from .stage6alt.plugins.thermo_environment import ThermoEnvironmentPlugin  # noqa: F401

# S6A-3: re-export the Stage 6alt thermometer BPT-metric + zero-shot-subset
# Pattern-A symbols (the two helper functions) from their plugin homes so the
# existing import path keeps working. The S6A-0 golden snapshot patches
# ``stage6alt_thermometer._bpt_from_nll`` / ``stage6alt_thermometer._lm_eval_subset``
# directly via ``monkeypatch.setattr``; the re-import puts the SAME function
# object on the monolith namespace, so that patch-by-attribute keeps biting.
# The plugin classes ``BptMetricPlugin`` / ``ZeroShotSubsetPlugin`` are imported
# alongside so an external registry walker can pick them up via the monolith too.
from .stage6alt.plugins.bpt_metric import (  # noqa: F401
    _bpt_from_nll,
    BptMetricPlugin,
)
from .stage6alt.plugins.zero_shot_subset import (  # noqa: F401
    _lm_eval_subset,
    ZeroShotSubsetPlugin,
)

# S6A-4: re-export the Stage 6alt thermometer teacher-cache Pattern-A symbols
# (the format-version constant + the three cache helpers) from their plugin
# home so the existing import path keeps working. The S6A-0 golden snapshot
# patches ``stage6alt_thermometer._load_thermo_teacher_cache`` (etc.) directly
# via ``monkeypatch.setattr``; the re-import puts the SAME function object on
# the monolith namespace, so that patch-by-attribute keeps biting. The plugin
# class ``ThermoTeacherProviderPlugin`` is imported alongside so an external
# registry walker can pick it up via the monolith too.
from .stage6alt.plugins.thermo_teacher_provider import (  # noqa: F401
    THERMO_TEACHER_CACHE_FORMAT_VERSION,
    _thermo_teacher_cache_key,
    _load_thermo_teacher_cache,
    _save_thermo_teacher_cache,
    ThermoTeacherProviderPlugin,
)

log = logging.getLogger(__name__)

# S6A-2: `THERMO_SEED_OFFSET` and `_DEFAULT_SUBSET_WEIGHTS` are relocated to
# `stage6alt/plugins/thermo_corpus.py` and re-imported above (see the
# `# noqa: F401` block) so existing call sites in this module / external
# callers still resolve them via `stage6alt_thermometer.THERMO_SEED_OFFSET`.

# S6A-3: `_bpt_from_nll` is relocated to `stage6alt/plugins/bpt_metric.py` and
# re-imported above (see the `# noqa: F401` block); call sites below resolve
# it through that re-import. The orphaned `iter_batches` import (only used by
# `_bpt_from_nll`) was removed alongside the relocation.


# S6A-2: `_thermo_corpus_spec`, `_thermo_wikitext_tensor`, `_build_thermo_corpus`
# are relocated to `stage6alt/plugins/thermo_corpus.py` and re-imported above
# (see the `# noqa: F401` block); call sites below resolve them through that
# re-import.


# S6A-4: `THERMO_TEACHER_CACHE_FORMAT_VERSION`, `_thermo_teacher_cache_key`,
# `_load_thermo_teacher_cache`, `_save_thermo_teacher_cache` are relocated to
# `stage6alt/plugins/thermo_teacher_provider.py` and re-imported above (see the
# `# noqa: F401` block); call sites in `run()` resolve them through that
# re-import. The orphaned `hashlib` and `json` imports (only used by the
# relocated helpers) were removed alongside the relocation.


# S6A-3: `_lm_eval_subset` is relocated to `stage6alt/plugins/zero_shot_subset.py`
# and re-imported above (see the `# noqa: F401` block); call sites below resolve
# it through that re-import.


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run(model, tokenizer, config: dict, artifacts_dir: Path, *, device=None) -> Path:
    """Stage 6alt thermometer. Thin shim — delegates to the plugin orchestrator (S6A-6).

    S6A-6 flipped the relationship: the REAL thermometer sequencer now
    lives in :func:`moe_compress.stage6alt.orchestrator.run`; this
    function-local-import shim preserves the existing call site
    (``from moe_compress import stage6alt_thermometer``) used by
    ``run_pipeline.py`` and the S6A-0 golden test. Z-1 will clean up the
    dead Pattern-A re-imports above; until then they remain so any caller
    that still resolves the symbols via ``stage6alt_thermometer`` keeps
    working unchanged.
    """
    from .stage6alt.orchestrator import run as _orchestrator_run
    return _orchestrator_run(model, tokenizer, config, artifacts_dir, device=device)
