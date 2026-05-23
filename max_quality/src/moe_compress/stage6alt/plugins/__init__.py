"""Stage 6alt plugin implementations.

Holds:

* ``thermo_environment.py`` (added by S6A-2 — the thermometer environment-
  setup concern: Pattern-B only, an inert ``setup_thermo_environment``
  hook that reproduces the monolith's inline experts-implementation shim
  + cu130/Hopper kernel-patch call sequence; both helpers are reused from
  ``stage6.plugins.eval_environment`` without re-relocation).
* ``thermo_corpus.py`` (added by S6A-2 — the thermometer calibration-
  corpus concern: the relocated ``THERMO_SEED_OFFSET`` /
  ``_DEFAULT_SUBSET_WEIGHTS`` / ``_thermo_corpus_spec`` /
  ``_thermo_wikitext_tensor`` / ``_build_thermo_corpus`` symbols plus a
  ``ThermoCorpusPlugin`` with an unconditional ``is_enabled`` and an
  inert ``build_corpus`` hook).
* ``bpt_metric.py`` (added by S6A-3 — the thermometer BPT-measurement
  concern: the relocated ``_bpt_from_nll`` helper plus a
  ``BptMetricPlugin`` with an unconditional ``is_enabled`` and an
  inert ``compute_bpt`` hook).
* ``zero_shot_subset.py`` (added by S6A-3 — the thermometer ARC-Easy +
  HellaSwag zero-shot subset concern: the relocated ``_lm_eval_subset``
  helper plus a ``ZeroShotSubsetPlugin`` with an unconditional
  ``is_enabled`` and an inert ``compute_zero_shot_subset`` hook;
  the underlying ``_lm_eval_tasks`` harness wrapper stays in its S6-3
  home, ``stage6.plugins.zero_shot_lm_eval``).
* ``thermo_teacher_provider.py`` (added by S6A-4 — the thermometer
  teacher-cache provider concern: the relocated
  ``THERMO_TEACHER_CACHE_FORMAT_VERSION`` /
  ``_thermo_teacher_cache_key`` / ``_load_thermo_teacher_cache`` /
  ``_save_thermo_teacher_cache`` symbols plus a
  ``ThermoTeacherProviderPlugin`` with an unconditional ``is_enabled``
  and an inert ``provide_thermo_teacher_side`` hook that reproduces
  the monolith's teacher block — cache-hit shortcut + cache-miss
  load/score/save path).

The Stage 6alt thermometer algorithm — environment setup, calibration-
corpus build, BPT measurement, zero-shot subset, teacher-cache provider,
and validation-report assembly — is being extracted from the legacy
``stage6alt_thermometer.py`` monolith into focused plugins here by tasks
S6A-2, S6A-3, and the follow-on extraction sub-tasks. No plugin manifest
exists yet — a later sub-task introduces the ``STAGE6ALT`` object and
flips the orchestrator-vs-monolith delegation direction.
"""
