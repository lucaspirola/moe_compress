"""B0 hook-fix CPU tests (no GPU, no live model, no patched wheel).

Covers the three CPU-testable surfaces of tasks/PLAN_B0_HOOK_FIX.md:

  * M1 fail-fast — ``assert_enabled_captures_nonempty`` in the driver:
    after the first generate chunk, every ENABLED ``--capture-*`` writer
    must report a nonzero ``captured_entry_count()`` or the run aborts
    with ``SystemExit(2)``. Tested with INJECTED fake writer objects via
    the public ``writer_resolver`` seam — no monkeypatch of production
    code, per [[no-monkey-patches]].

  * C2 predicate — discovery must key off ``global_num_experts``
    (FusedMoE sets that, never ``num_experts``). Tested against a stub
    ``nn.Module``-like object via a faithful local reproduction of the
    patch's discovery predicate (the real loop lives inside the patched
    vLLM wheel, not importable on CPU).

  * Patch git-apply check — documented as a command (see
    ``test_patches_apply_check_command_documented``); the live
    ``git apply --check`` against /tmp/vllm_b0 @ ad7125a is run in the
    task's VERIFY step and recorded in the PR, since it needs the pinned
    clone which is not a test fixture.

The C1 env edit (``VLLM_ENABLE_V1_MULTIPROCESSING=0``) is asserted to be
present at module level in the driver (a stale shell export aside, the
``setdefault`` must run before any vllm import).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

# Inject scripts/ onto sys.path so we can import the driver module.
# Mirrors the shim in test_build_self_traces_calib_vllm_c1.py:31.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_self_traces_calib_vllm import (  # noqa: E402
    _CAPTURE_WRITER_MODULES,
    assert_enabled_captures_nonempty,
)


# ---------------------------------------------------------------------------
# Test doubles: a tiny fake writer registry. NO monkeypatch — the production
# helper takes a ``writer_resolver`` callable; we pass a registry-backed one.
# ---------------------------------------------------------------------------


class _FakeWriter:
    """Stub vLLM calibration writer exposing the public count method."""

    def __init__(self, count: int):
        self._count = count

    def captured_entry_count(self) -> int:
        return self._count


class _FakeWriterNoCount:
    """Stub for a pre-B0 wheel: writer module without the count method."""


def _resolver_from(mapping):
    """Build a writer_resolver that returns fakes keyed by module name."""

    def _resolve(module_name: str):
        if module_name not in mapping:
            raise ModuleNotFoundError(module_name)
        return mapping[module_name]

    return _resolve


# ---------------------------------------------------------------------------
# M1 fail-fast tests
# ---------------------------------------------------------------------------


def test_all_captures_nonempty_no_raise():
    """Every enabled capture > 0 -> returns counts, no SystemExit."""
    mapping = {
        "vllm.calibration_imatrix": _FakeWriter(12),
        "vllm.calibration_reap_scores": _FakeWriter(48),
    }
    counts = assert_enabled_captures_nonempty(
        ["capture_imatrix", "capture_reap_scores"],
        model_class="Qwen3_5MoeForCausalLM",
        writer_resolver=_resolver_from(mapping),
    )
    assert counts == {"capture_imatrix": 12, "capture_reap_scores": 48}


def test_one_empty_capture_raises_systemexit_2():
    """A single enabled capture reporting 0 aborts with SystemExit(2),
    and the ERROR names the offending capture."""
    mapping = {
        "vllm.calibration_imatrix": _FakeWriter(0),  # the broken one
        "vllm.calibration_reap_scores": _FakeWriter(48),
    }
    records: list[logging.LogRecord] = []
    handler = _RecordingHandler(records)
    _log = logging.getLogger("build_self_traces_calib_vllm")
    _log.addHandler(handler)
    try:
        with pytest.raises(SystemExit) as exc:
            assert_enabled_captures_nonempty(
                ["capture_imatrix", "capture_reap_scores"],
                model_class="Qwen3_5MoeForCausalLM",
                writer_resolver=_resolver_from(mapping),
            )
    finally:
        _log.removeHandler(handler)
    assert exc.value.code == 2
    errs = [r for r in records if r.levelno == logging.ERROR]
    assert len(errs) == 1
    msg = errs[0].getMessage()
    assert "--capture-imatrix" in msg
    # the healthy capture must NOT be named as empty
    assert "--capture-reap-scores" not in msg
    assert "Qwen3_5MoeForCausalLM" in msg


def test_multiple_empty_captures_all_named():
    mapping = {
        "vllm.calibration_imatrix": _FakeWriter(0),
        "vllm.calibration_reap_scores": _FakeWriter(0),
        "vllm.calibration_routing_stats": _FakeWriter(7),
    }
    records: list[logging.LogRecord] = []
    handler = _RecordingHandler(records)
    _log = logging.getLogger("build_self_traces_calib_vllm")
    _log.addHandler(handler)
    try:
        with pytest.raises(SystemExit):
            assert_enabled_captures_nonempty(
                ["capture_imatrix", "capture_reap_scores",
                 "capture_routing_stats"],
                writer_resolver=_resolver_from(mapping),
            )
    finally:
        _log.removeHandler(handler)
    msg = [r for r in records if r.levelno == logging.ERROR][0].getMessage()
    assert "--capture-imatrix" in msg
    assert "--capture-reap-scores" in msg
    assert "--capture-routing-stats" not in msg


def test_allow_empty_downgrades_to_warning():
    """--allow-empty-captures: an empty capture warns instead of aborting."""
    mapping = {"vllm.calibration_imatrix": _FakeWriter(0)}
    records: list[logging.LogRecord] = []
    handler = _RecordingHandler(records)
    _log = logging.getLogger("build_self_traces_calib_vllm")
    _log.addHandler(handler)
    try:
        # No SystemExit expected.
        assert_enabled_captures_nonempty(
            ["capture_imatrix"],
            allow_empty=True,
            writer_resolver=_resolver_from(mapping),
        )
    finally:
        _log.removeHandler(handler)
    warns = [r for r in records if r.levelno == logging.WARNING]
    assert len(warns) == 1
    assert "--capture-imatrix" in warns[0].getMessage()


def test_pre_b0_wheel_writer_skipped_with_warning():
    """Installed writer lacking captured_entry_count() (pre-B0 wheel) is
    SKIPPED with a WARN, never crashing and never aborting."""
    mapping = {"vllm.calibration_imatrix": _FakeWriterNoCount()}
    records: list[logging.LogRecord] = []
    handler = _RecordingHandler(records)
    _log = logging.getLogger("build_self_traces_calib_vllm")
    _log.addHandler(handler)
    try:
        counts = assert_enabled_captures_nonempty(
            ["capture_imatrix"],
            writer_resolver=_resolver_from(mapping),
        )
    finally:
        _log.removeHandler(handler)
    # Skipped -> not in counts, no raise.
    assert counts == {}
    warns = [r for r in records if r.levelno == logging.WARNING]
    assert any("captured_entry_count" in w.getMessage() for w in warns)


def test_layer_input_reservoir_is_not_count_checkable():
    """layer_input_reservoir rides stage2_profile (no own module) -> it is
    intentionally absent from the count-checkable mapping and is silently
    skipped (covered by stage2_profile's count)."""
    assert "capture_layer_input_reservoir" not in _CAPTURE_WRITER_MODULES
    mapping = {"vllm.calibration_stage2_profile": _FakeWriter(5)}
    counts = assert_enabled_captures_nonempty(
        ["capture_stage2_profile", "capture_layer_input_reservoir"],
        writer_resolver=_resolver_from(mapping),
    )
    assert counts == {"capture_stage2_profile": 5}


# ---------------------------------------------------------------------------
# C2 predicate test
# ---------------------------------------------------------------------------


class _StubFusedMoE:
    """Mirrors vLLM's FusedMoE attribute surface for discovery: it sets
    ``moe_layer_id`` + ``global_num_experts`` (and ``logical_num_experts``)
    but NEVER ``num_experts`` — exactly the shape that broke C2."""

    def __init__(self, moe_layer_id: int, global_num_experts: int):
        self.moe_layer_id = moe_layer_id
        self.global_num_experts = global_num_experts
        self.logical_num_experts = global_num_experts
        # crucially: no self.num_experts


def _discover_count(module, *, attr: str) -> int:
    """Faithful reproduction of the patch discovery predicate
    (vllm_calibration_hooks.patch setup() loops): a module counts as one
    MoE layer iff it has an int moe_layer_id AND an int ``attr`` > 0."""
    moe_layer_id = getattr(module, "moe_layer_id", None)
    n = getattr(module, attr, None)
    if isinstance(moe_layer_id, int) and isinstance(n, int) and n > 0:
        return 1
    return 0


def test_c2_global_num_experts_predicate_matches():
    """The hardened predicate (global_num_experts) counts a FusedMoE-shaped
    module; the OLD predicate (num_experts) misses it -> reproduces the B0
    always-zero discovery bug and its fix."""
    stub = _StubFusedMoE(moe_layer_id=0, global_num_experts=128)
    # New predicate: counts it.
    assert _discover_count(stub, attr="global_num_experts") == 1
    # Old predicate: always misses (num_experts is unset on FusedMoE).
    assert _discover_count(stub, attr="num_experts") == 0


def test_c2_predicate_rejects_non_moe_module():
    """A plain module with neither attr is not counted under either key."""

    class _Plain:
        pass

    plain = _Plain()
    assert _discover_count(plain, attr="global_num_experts") == 0
    assert _discover_count(plain, attr="num_experts") == 0


def test_c2_predicate_rejects_zero_experts():
    stub = _StubFusedMoE(moe_layer_id=3, global_num_experts=0)
    assert _discover_count(stub, attr="global_num_experts") == 0


# ---------------------------------------------------------------------------
# C1 env + patch apply-check documentation
# ---------------------------------------------------------------------------


def test_c1_env_set_at_module_level():
    """Importing the driver must have already set
    VLLM_ENABLE_V1_MULTIPROCESSING (module-level setdefault), so it is in
    os.environ by the time any vllm import would run."""
    import os

    # The import at the top of this file already ran the driver module body.
    assert os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") == "0"


def test_patches_apply_check_command_documented():
    """The live git-apply check needs the pinned clone /tmp/vllm_b0 @
    ad7125a, which is not a test fixture. The command (run in the task's
    VERIFY step) is:

        cd /tmp/vllm_b0 && git checkout . && git clean -fdx \\
          && git apply --check \\
             max_quality/patches/vllm_calibration_hooks.patch \\
             max_quality/patches/vllm_calibration_stage2_profile.patch

    This test just asserts both patch files exist and are non-empty so a
    CI run flags a deleted/truncated patch even without the clone.
    """
    root = Path(__file__).resolve().parents[2]
    hooks = root / "max_quality/patches/vllm_calibration_hooks.patch"
    stage2 = root / "max_quality/patches/vllm_calibration_stage2_profile.patch"
    assert hooks.is_file() and hooks.stat().st_size > 0
    assert stage2.is_file() and stage2.stat().st_size > 0
    # The M3 hunk must be present in the hooks patch.
    text = hooks.read_text()
    assert "b/vllm/model_executor/models/qwen3_next.py" in text
    assert "Qwen3NextSparseMoeBlock" in text
    # C2: discovery now keys off global_num_experts at every site.
    assert 'getattr(module, "num_experts"' not in text
    assert text.count('getattr(module, "global_num_experts"') >= 8


class _RecordingHandler(logging.Handler):
    """Minimal log-record sink (no monkeypatch; attach/detach explicitly)."""

    def __init__(self, sink: list[logging.LogRecord]):
        super().__init__()
        self._sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        self._sink.append(record)
