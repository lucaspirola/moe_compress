"""Stage 6 plugin implementations.

Holds:

* ``eval_environment.py`` (added by S6-2 — the eval-environment setup concern:
  dataset revision pinning, the cu130/Hopper kernel patches, the MoE
  experts-implementation shim, the imatrix calibration-corpus build and the
  torch.compile setup, plus ``EvalEnvironmentPlugin`` with an unconditional
  ``is_enabled`` and an inert ``setup_environment`` hook).
* ``wikitext_ppl.py`` (added by S6-3 — the WikiText-2 perplexity sub-metric:
  the relocated ``_wikitext2_ppl`` helper plus ``WikitextPplPlugin`` with an
  ``is_enabled`` gated on ``stage6_validate.wikitext2.enabled`` and an inert
  ``eval_task`` hook).
* ``zero_shot_lm_eval.py`` (added by S6-3 — the zero-shot lm-eval sub-metric:
  the relocated ``_ZERO_SHOT_TASKS`` constant and ``_lm_eval_tasks`` helper
  plus ``ZeroShotLmEvalPlugin`` with an ``is_enabled`` gated on
  ``stage6_validate.zero_shot.enabled`` and an inert ``eval_task`` hook).
* ``humaneval.py`` (added by S6-4 — the HumanEval pass@1 generative sub-metric:
  the relocated ``_humaneval`` driver and ``_check_humaneval`` scorer plus
  ``HumanEvalPlugin`` with an ``is_enabled`` gated on
  ``stage6_validate.generative.enabled`` AND a ``humaneval`` sub-key and an
  inert ``eval_task`` hook). Its shared batched-generation / chat-format
  primitives live in ``tools/eval_harness`` (also added by S6-4).
* ``math500.py`` (added by S6-4 — the MATH-500 accuracy generative sub-metric:
  the relocated ``_math500`` driver, its ``_extract_boxed`` / ``_last_numeric``
  / ``_check_math`` / ``_math_fallback_extract`` grading helpers and the
  optional-SymPy guard, plus ``Math500Plugin`` with an ``is_enabled`` gated on
  ``stage6_validate.generative.enabled`` AND a ``math500`` sub-key and an inert
  ``eval_task`` hook).
* ``teacher_provider.py`` (added by S6-5 — the Stage 6 teacher-provider
  concern: the relocated ``TEACHER_CACHE_FORMAT_VERSION`` constant +
  ``_safe_pkg_version`` / ``_teacher_cache_key`` / ``_load_teacher_cache`` /
  ``_save_teacher_cache`` / ``_preload_teacher_to_cpu`` helpers plus
  ``TeacherProviderPlugin`` with an unconditional ``is_enabled`` and an inert
  ``provide_teacher_side`` hook that reproduces the monolith's teacher-side
  eval block — cache-hit shortcut, preload-thread join, kernel-patch +
  experts-impl shim, the four conditional teacher-side eval calls, cache
  save).

The Stage 6 validation algorithm — WikiText-2 PPL, zero-shot (ARC-C,
HellaSwag), generative (HumanEval, MATH-500), imatrix pipeline, and threshold
gating — is extracted from the legacy ``stage6_validate.py`` monolith into
focused plugins here by tasks S6-2..S6-7. No plugin manifest exists yet.
"""
