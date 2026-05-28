"""Unit tests for F-H-6 ``_ckpt_counter_check`` (LOW-4 / NIT-4 extract).

The helper enforces ``loaded_prompts == already_done`` on resume — see
``build_self_traces_calib_vllm._ckpt_counter_check`` docstring. By
extracting it from the ``main()`` closure into a module-level free
function we can test it directly without spinning up vLLM, prompts, or
GPU state.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

# Inject the scripts/ dir onto sys.path so the script module imports
# resolve (it grabs sibling helpers from build_self_traces_calib).
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from build_self_traces_calib_vllm import _ckpt_counter_check  # type: ignore  # noqa: E402


def test_counter_check_match_returns_silently(caplog):
    """Happy path: counts match → no log, no raise."""
    _ckpt_counter_check(
        "imatrix",
        loaded_prompts=128,
        already_done=128,
        ckpt_path=Path("/tmp/fake.ckpt"),
        allow_counter_divergence=False,
    )
    assert caplog.records == []


def test_counter_check_divergence_default_raises():
    """F-H-6 default: divergence → ValueError with the ckpt path in the
    message + instructions to delete the file or pass the flag."""
    with pytest.raises(ValueError, match=r"checkpoint has 100 prompts.*JSONL has 128 rows"):
        _ckpt_counter_check(
            "reap_scores",
            loaded_prompts=100,
            already_done=128,
            ckpt_path=Path("/tmp/reap.ckpt"),
            allow_counter_divergence=False,
        )


def test_counter_check_message_includes_ckpt_path():
    """The error message MUST include the specific ckpt path so the
    operator's recovery action (rm <ckpt>) is unambiguous when several
    sidecars diverge in the same run."""
    with pytest.raises(ValueError, match=r"/tmp/specific\.ckpt"):
        _ckpt_counter_check(
            "per_expert_max",
            loaded_prompts=50,
            already_done=64,
            ckpt_path=Path("/tmp/specific.ckpt"),
            allow_counter_divergence=False,
        )


def test_counter_check_with_override_logs_warning(caplog):
    """``--allow-counter-divergence`` downgrades the hard-fail to a
    WARNING. No raise; the caller continues with the (smaller) loaded
    count.

    Uses the [[caplog-propagate-restore]] pattern (Pattern N) since the
    script's logger is non-root ('build_self_traces_calib_vllm').
    """
    caplog.set_level(logging.WARNING)
    logger = logging.getLogger("build_self_traces_calib_vllm")
    prev = logger.propagate
    logger.propagate = True
    try:
        _ckpt_counter_check(
            "routing_stats",
            loaded_prompts=100,
            already_done=128,
            ckpt_path=Path("/tmp/x.ckpt"),
            allow_counter_divergence=True,
        )
    finally:
        logger.propagate = prev
    # At least one WARNING about the under-count.
    warns = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("Proceeding with the smaller counter" in r.getMessage() for r in warns), (
        f"expected WARNING about smaller counter; got {[r.getMessage() for r in caplog.records]}"
    )


def test_counter_check_signal_name_propagates():
    """The ``signal_name`` arg flows into the raised message so the
    operator knows which accumulator is mis-counted (typical run has
    5+ checkpoints; pinpointing the one that diverges matters)."""
    with pytest.raises(ValueError, match=r"output_reservoir:"):
        _ckpt_counter_check(
            "output_reservoir",
            loaded_prompts=10,
            already_done=20,
            ckpt_path=Path("/tmp/or.ckpt"),
            allow_counter_divergence=False,
        )
