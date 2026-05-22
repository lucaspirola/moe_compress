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

The Stage 6 validation algorithm — WikiText-2 PPL, zero-shot (ARC-C,
HellaSwag), generative (HumanEval, MATH-500), imatrix pipeline, and threshold
gating — is extracted from the legacy ``stage6_validate.py`` monolith into
focused plugins here by tasks S6-2..S6-7. No plugin manifest exists yet.
"""
