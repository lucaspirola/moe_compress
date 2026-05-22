"""Stage 6 plugin implementations.

Holds ``eval_environment.py`` (added by S6-2 — the eval-environment setup
concern: dataset revision pinning, the cu130/Hopper kernel patches, the MoE
experts-implementation shim, the imatrix calibration-corpus build and the
torch.compile setup, plus ``EvalEnvironmentPlugin`` with an unconditional
``is_enabled`` and an inert ``setup_environment`` hook). The Stage 6
validation algorithm — WikiText-2 PPL, zero-shot (ARC-C, HellaSwag),
generative (HumanEval, MATH-500), imatrix pipeline, and threshold gating —
is extracted from the legacy ``stage6_validate.py`` monolith into focused
plugins here by tasks S6-2..S6-7. No plugin manifest exists yet.
"""
