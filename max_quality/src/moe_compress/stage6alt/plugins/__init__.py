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
* ``teacher_provider.py`` (added by S6A-3 — the thermometer teacher-cache
  provider concern: the relocated ``_thermo_teacher_cache_key`` /
  ``_load_thermo_teacher_cache`` / ``_save_thermo_teacher_cache`` helpers
  plus a ``TeacherProviderPlugin`` with an unconditional ``is_enabled``
  and an inert ``provide_teacher_side`` hook).
* ``bpt_measurement.py`` (added by S6A-4 — the thermometer BPT-measurement
  concern: the relocated ``_bpt_from_nll`` helper plus a
  ``BptMeasurementPlugin`` with an unconditional ``is_enabled`` and an
  inert ``eval_task`` hook).
* ``lm_eval_subset.py`` (added by S6A-5 — the thermometer lm-eval subset
  concern: the relocated ``_lm_eval_subset`` helper plus an
  ``LmEvalSubsetPlugin`` with an ``is_enabled`` gated on
  ``stage6alt_thermometer.lm_eval.enabled`` and an inert ``eval_task`` hook).

The Stage 6alt thermometer algorithm — environment setup, calibration-
corpus build, teacher-cache provider, BPT measurement, lm-eval subset,
and validation-report assembly — is extracted from the legacy
``stage6alt_thermometer.py`` monolith into focused plugins here by tasks
S6A-2..S6A-5. No plugin manifest exists yet — S6A-6 introduces the
``STAGE6ALT`` object and flips the orchestrator-vs-monolith delegation
direction.
"""
