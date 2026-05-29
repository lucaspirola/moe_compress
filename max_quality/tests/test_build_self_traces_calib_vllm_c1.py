"""C-1 (audit/calib-resume-spot-eviction R-1): fresh-on-resume writer
abort. Driver must refuse to start when --resume is set, the operator
enabled --capture-X, the JSONL has rows from a prior session, but no
checkpoint exists for writer X.

Tests target the pure helper ``_ckpt_existence_check`` exposed at module
scope in ``max_quality/scripts/build_self_traces_calib_vllm.py``. The
helper is independently testable per the NIT-4 / LOW-4 extraction
precedent set by ``_ckpt_counter_check``.

No monkeypatch / mock.patch on production code, per the project rule
[[no-monkey-patches]].
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

# Inject scripts/ onto sys.path so we can import the driver module.
# Mirrors max_quality/tests/conftest.py:24's `parents[1] / "src"` shim,
# but pointed at scripts/ — the driver lives there, not in src/, and
# conftest only injects src/ today (verified). 3-line local shim is
# preferred over editing conftest because conftest's path injection is
# project-wide and adding scripts/ there would leak the driver script's
# top-level names (build_self_traces_calib_vllm, etc.) into every test
# module's import resolution. If a second test file ever needs the
# driver, lift this shim into conftest at that point.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_self_traces_calib_vllm import _ckpt_existence_check  # noqa: E402


def test_fresh_writer_on_resume_aborts(tmp_path):
    """Operator runs --resume --capture-imatrix on a JSONL with 1000
    rows from a prior session that didn't capture imatrix. The
    existence-check must raise ValueError with an actionable message
    that names the writer, the row count, the ckpt path, and the
    escape-hatch flag."""
    ckpt = tmp_path / "trace.imatrix.ckpt"
    assert not ckpt.exists()
    with pytest.raises(ValueError) as exc_info:
        _ckpt_existence_check(
            "imatrix",
            capture_enabled=True,
            already_done=1000,
            ckpt_path=ckpt,
            allow_counter_divergence=False,
        )
    msg = str(exc_info.value)
    assert "imatrix" in msg
    assert "1000" in msg
    assert "--allow-counter-divergence" in msg
    assert str(ckpt) in msg


def test_fresh_writer_with_allow_divergence_warns(tmp_path):
    """Escape hatch: --allow-counter-divergence downgrades the abort to
    a WARN; the run proceeds AND the warning message names the writer,
    the JSONL row count, and the escape-hatch flag so the operator can
    audit the degraded-metadata claim from stdout alone.

    Capture pattern: a ``logging.Handler`` subclass attached to the
    driver's module-level logger via ``addHandler`` / ``removeHandler``.
    Per [[no-monkey-patches]], use the ``_attach_capture_handler``
    precedent at ``max_quality/tests/test_audit_svc.py:578-627`` as the
    env-independent default for capturing records from a named logger.
    The ``caplog`` approach is not used here — even when ``caplog``
    would technically work for this specific logger, the project rule
    is to default to the env-independent handler-attach pattern to
    avoid the class of env-dependency bugs that motivated the rule.
    """
    # ``test_audit_svc`` is the sibling test module in this same dir.
    # max_quality/ is not a top-level package (no ``__init__.py`` at the
    # project root), so neither ``max_quality.tests.test_audit_svc`` nor
    # a bare ``test_audit_svc`` resolves through normal import. Load
    # the helper directly via ``importlib.util.spec_from_file_location``
    # — exactly the same idiom that ``test_audit_svc.py`` itself uses
    # at module body to load ``audit/spec_compliance/svc_audit.py``.
    # If a future second user lands, lift the helper to a shared
    # ``max_quality/tests/_capture_helpers.py``.
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location(
        "test_audit_svc_for_capture_helper",
        str(Path(__file__).resolve().parent / "test_audit_svc.py"),
    )
    assert _spec is not None and _spec.loader is not None
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)  # type: ignore[union-attr]
    _attach_capture_handler = _mod._attach_capture_handler

    ckpt = tmp_path / "trace.imatrix.ckpt"
    # The driver module's logger name. Verified against the script:
    # `log = logging.getLogger("build_self_traces_calib_vllm")` at
    # `max_quality/scripts/build_self_traces_calib_vllm.py:102`
    # (note: the module-level variable is `log`, not `logger`; the
    # explicit string arg is what `_attach_capture_handler` must be
    # passed, not the local variable name).
    records, cleanup = _attach_capture_handler(
        "build_self_traces_calib_vllm",
    )
    try:
        _ckpt_existence_check(
            "imatrix",
            capture_enabled=True,
            already_done=1000,
            ckpt_path=ckpt,
            allow_counter_divergence=True,
        )
    finally:
        cleanup()

    # Exactly one WARN-level record; message names writer, row count,
    # AND the escape-hatch flag. This pins the operator-audit contract.
    warns = [r for r in records if r.levelno == logging.WARNING]
    assert len(warns) == 1, (
        f"expected 1 WARN, got {len(warns)}: {records}"
    )
    msg = warns[0].getMessage()
    assert "imatrix" in msg
    assert "1000" in msg
    assert "--allow-counter-divergence" in msg


def test_capture_disabled_skips(tmp_path):
    """No-op when --capture-X is not set, regardless of resume state."""
    ckpt = tmp_path / "trace.imatrix.ckpt"
    # Should not raise.
    _ckpt_existence_check(
        "imatrix",
        capture_enabled=False,
        already_done=1000,
        ckpt_path=ckpt,
        allow_counter_divergence=False,
    )


def test_already_done_zero_skips(tmp_path):
    """No-op on the first session (already_done == 0 means we're not
    resuming on top of prior data)."""
    ckpt = tmp_path / "trace.imatrix.ckpt"
    _ckpt_existence_check(
        "imatrix",
        capture_enabled=True,
        already_done=0,
        ckpt_path=ckpt,
        allow_counter_divergence=False,
    )


def test_ckpt_exists_skips(tmp_path):
    """No-op when the checkpoint exists — the existing F-H-6 check will
    handle the counter-divergence case separately."""
    ckpt = tmp_path / "trace.imatrix.ckpt"
    ckpt.touch()
    _ckpt_existence_check(
        "imatrix",
        capture_enabled=True,
        already_done=1000,
        ckpt_path=ckpt,
        allow_counter_divergence=False,
    )
