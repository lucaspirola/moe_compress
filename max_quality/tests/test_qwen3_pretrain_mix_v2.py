"""Unit tests for the ``qwen3-pretrain-mix-v2`` calibration corpus.

Step 1 coverage (constants + registration + cache_key):
  * F.5.1 — weights sum to 1.0; dict keys match across all six sole-truth dicts.
  * F.5.2 — every policy is in {GENERATE, TEACHER_FORCED}.
  * F.5.3 — corpus is registered AND v1 corpus is still registered.
  * F.5.4 — v1 vs v2 CalibrationSpec cache_keys differ.

Steps 3+4 add parametric row-extraction tests (F.5.5) and per-subset
behaviour tests (F.5.6 / F.5.7 / F.5.8 / F.5.9).

All tests are CPU-only with no network access — they exercise constants
and registry behavior only.
"""
from __future__ import annotations

import pytest

from moe_compress.utils.calibration import (
    CalibrationSpec,
    _QWEN3_MIX_V2_AVG_TOKENS,
    _QWEN3_MIX_V2_DATASET,
    _QWEN3_MIX_V2_DATASET_CONFIG,
    _QWEN3_MIX_V2_DATASET_SPLIT,
    _QWEN3_MIX_V2_POLICY,
    _QWEN3_MIX_V2_WEIGHTS,
    get_corpus_adapter,
)


# The 12 subsets the plan locks in.
_EXPECTED_SUBSETS = {
    "tulu3",
    "math",
    "qa",
    "creative",
    "multilingual",
    "fineweb",
    "papers",
    "mot_math",
    "mot_code",
    "mot_science",
    "swe_smith",
    "function_calling",
}


# ---------------------------------------------------------------------------
# F.5.1 — weights / dict-key consistency
# ---------------------------------------------------------------------------


def test_weights_sum_to_one():
    """The 12 weights must sum to exactly 1.0 within float tolerance."""
    total = sum(_QWEN3_MIX_V2_WEIGHTS.values())
    assert total == pytest.approx(1.0, abs=1e-9), (
        f"weights sum to {total!r}, expected 1.0"
    )


def test_all_dicts_share_the_same_subset_keys():
    """The six sole-truth dicts must agree on the subset set — otherwise
    a subset can be silently missing config/policy/dataset/etc. lookups
    at dispatch time."""
    keys = set(_QWEN3_MIX_V2_WEIGHTS)
    assert keys == _EXPECTED_SUBSETS, (
        f"_QWEN3_MIX_V2_WEIGHTS keys {sorted(keys)} != expected "
        f"{sorted(_EXPECTED_SUBSETS)}"
    )
    assert keys == set(_QWEN3_MIX_V2_AVG_TOKENS)
    assert keys == set(_QWEN3_MIX_V2_DATASET)
    assert keys == set(_QWEN3_MIX_V2_DATASET_CONFIG)
    assert keys == set(_QWEN3_MIX_V2_DATASET_SPLIT)
    assert keys == set(_QWEN3_MIX_V2_POLICY)


# ---------------------------------------------------------------------------
# F.5.2 — policy validity
# ---------------------------------------------------------------------------


def test_policy_values_are_valid():
    """Every per-subset policy must be one of the two recognized values.
    The build scripts dispatch on this string; any drift would silently
    skip a subset."""
    valid = {"GENERATE", "TEACHER_FORCED"}
    for subset, policy in _QWEN3_MIX_V2_POLICY.items():
        assert policy in valid, (
            f"subset {subset!r}: invalid policy {policy!r} (expected one of {valid})"
        )


# ---------------------------------------------------------------------------
# F.5.3 — corpus registration (v1 + v2 coexist)
# ---------------------------------------------------------------------------


def test_v2_corpus_registered():
    """The v2 adapter must be registered and look up by name."""
    adapter = get_corpus_adapter("qwen3-pretrain-mix-v2")
    assert adapter.name == "qwen3-pretrain-mix-v2"
    assert callable(adapter.parse_yaml)
    assert callable(adapter.stream_texts)


def test_v1_corpus_still_registered():
    """Backward compat: registering v2 must not displace or override v1."""
    adapter = get_corpus_adapter("qwen3-pretrain-mix")
    assert adapter.name == "qwen3-pretrain-mix"


# ---------------------------------------------------------------------------
# F.5.4 — v1 vs v2 cache-key uniqueness
# ---------------------------------------------------------------------------


def test_cache_key_distinct_for_v1_v2():
    """CalibrationSpec.cache_key folds ``source`` into the payload; v1 and
    v2 specs that differ only in ``source`` must produce different keys.
    A collision would let v2 overwrite v1's cache file (or vice versa)."""
    v1 = CalibrationSpec(
        num_sequences=4000,
        sequence_length=2048,
        seed=1337,
        source="qwen3-pretrain-mix",
    )
    v2 = CalibrationSpec(
        num_sequences=4000,
        sequence_length=2048,
        seed=1337,
        source="qwen3-pretrain-mix-v2",
    )
    k1 = v1.cache_key("Qwen/Qwen3.6-35B-A3B")
    k2 = v2.cache_key("Qwen/Qwen3.6-35B-A3B")
    assert k1 != k2, (
        f"v1 and v2 cache_keys collide ({k1!r} == {k2!r}). Cache files "
        "would clobber each other on disk."
    )
