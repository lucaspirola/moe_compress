"""Invariant: Stage 6 `_teacher_cache_key` must NOT depend on
`stage5_router_kd.teacher_model_repo`.

This is load-bearing for the A0-fills/A1..A11-hits flow used by the ablation
harness: A0 runs the BF16 teacher and writes `_shared/teacher_eval_cache.json`;
A1..A11 hit that cache without ever loading the BF16 teacher on H200. If a
future refactor folds the Stage 5 KD-teacher identity into the Stage 6 eval
cache key, swapping the KD teacher (e.g., to `Qwen/Qwen3.6-35B-A3B-FP8`) on
A1..A11 would invalidate the cache and force every ablation to reload the
70 GB teacher for eval — defeating the whole point of the shared cache.

Lock the invariant with this test.
"""
from __future__ import annotations

import copy

from moe_compress.stage6_validate import _teacher_cache_key


def test_teacher_cache_key_ignores_stage5_teacher_repo_override(tiny_config):
    """A KD-teacher override at `stage5_router_kd.teacher_model_repo` must
    leave the Stage 6 cache key unchanged.
    """
    base_key = _teacher_cache_key(tiny_config)

    cfg_with_override = copy.deepcopy(tiny_config)
    cfg_with_override["stage5_router_kd"]["teacher_model_repo"] = (
        "Qwen/Qwen3.6-35B-A3B-FP8"
    )
    override_key = _teacher_cache_key(cfg_with_override)

    assert base_key == override_key, (
        "Stage 6 teacher_eval_cache key changed when stage5_router_kd."
        "teacher_model_repo was set. This breaks the A0-fills/A1..A11-hits "
        "ablation flow: any KD-teacher override on H200 would invalidate the "
        "BF16-teacher eval cache that A0 wrote. _teacher_cache_key must "
        "consume only config.model + config.stage6_validate."
    )


def test_teacher_cache_key_ignores_stage5_calibration_overrides(tiny_config):
    """Lever (b) overrides — `max_calibration_samples` and `max_sequence_length`
    on `stage5_router_kd` — must also leave the Stage 6 cache key unchanged.
    These knobs control the Stage 2.5 KD calibration footprint, not the
    Stage 6 evaluation, and so must not invalidate the shared eval cache.
    """
    base_key = _teacher_cache_key(tiny_config)

    cfg_with_overrides = copy.deepcopy(tiny_config)
    cfg_with_overrides["stage5_router_kd"]["max_calibration_samples"] = 1500
    cfg_with_overrides["stage5_router_kd"]["max_sequence_length"] = 256
    override_key = _teacher_cache_key(cfg_with_overrides)

    assert base_key == override_key, (
        "Stage 6 teacher_eval_cache key changed when stage5_router_kd "
        "max_calibration_samples / max_sequence_length were overridden. "
        "These knobs are Stage 5 calibration-shaping, not Stage 6 evaluation "
        "— the eval cache must not invalidate."
    )


def test_teacher_cache_key_still_changes_on_model_change(tiny_config):
    """Sanity counter-check: changing the **eval** teacher identity (i.e.
    config.model.name_or_path) MUST change the cache key. If this fails,
    the cache would silently serve wrong-teacher results.
    """
    base_key = _teacher_cache_key(tiny_config)

    cfg_diff_model = copy.deepcopy(tiny_config)
    cfg_diff_model["model"]["name_or_path"] = "different/teacher"
    diff_key = _teacher_cache_key(cfg_diff_model)

    assert base_key != diff_key, (
        "Stage 6 teacher_eval_cache key did NOT change when "
        "config.model.name_or_path changed. The cache would silently serve "
        "stale results from a different teacher model."
    )
