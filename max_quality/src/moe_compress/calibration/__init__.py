"""Calibration-side helpers shared between the moe_compress repo and the
vLLM calibration-hooks patch.

The Stage 2 profile writer has TWO implementations that must be kept in sync:
  1. Canonical: ``stage2_profile_writer.py`` (this package) — unit-testable
     without vLLM; exercised by the 4 test files in
     ``tests/test_stage2_profile_sidecar_*.py``.
  2. Patched: ``vllm/calibration_stage2_profile.py`` (in
     ``max_quality/patches/vllm_calibration_stage2_profile.patch``) — runs
     inside the rebuilt patched-wheel; reimplements the state machine but
     imports the shared types (``ReamCostAccumulator``,
     ``InputCovarianceAccumulator``, ``Stage2ProfilePayloadV4``,
     ``save_stage2_profile_v4``) from this package so the bug-fix path is
     unified.

The patch is a FULL REIMPLEMENTATION (not a thin wrapper around the canonical
module): the two files are independent code trees that happen to share the
shared accumulator + payload types. Changes to the canonical writer that
affect serialization, callback semantics, or accumulator state MUST be
mirrored into the patch file — the two code paths diverging silently is a
known maintenance hazard (Pattern L's risk profile). The tested entry points
(``_on_router_callback``, ``_on_expert_out_unweighted_callback``,
``_finalize_batch_for_layer``, ``dump_stage2_profile``) live in the canonical
file; the production-path equivalents live in the patch file.
"""
