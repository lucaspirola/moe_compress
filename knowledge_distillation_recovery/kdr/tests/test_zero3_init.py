"""Tests for `kdr.training.zero3_init` (LLR-0040, LLR-0048).

The real `Accelerator` instances and `HfDeepSpeedConfig` mechanics need a
DeepSpeed runtime; we mock the accelerate / transformers integration
surface and verify the predicate + context behaviours.

# VERIFIES: LLR-0040
# VERIFIES: LLR-0048
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from kdr.training.zero3_init import (
    _DSCHF_HOLDER,
    activate_zero3_init,
    is_deepspeed,
    is_zero3,
)


@pytest.fixture(autouse=True)
def _reset_holder() -> None:
    """Clear the module-level holder between tests so leak-detection works."""
    _DSCHF_HOLDER.clear()


# ── is_deepspeed / is_zero3 ──────────────────────────────────────────────────


def _fake_accelerator(*, distributed_type: Any, zero_stage: int | None) -> MagicMock:
    accel = MagicMock()
    accel.distributed_type = distributed_type
    if zero_stage is None:
        accel.state.deepspeed_plugin = None
    else:
        plugin = MagicMock()
        plugin.zero_stage = zero_stage
        accel.state.deepspeed_plugin = plugin
    return accel


def test_is_deepspeed_returns_false_for_no_distributed() -> None:
    from accelerate.utils import DistributedType

    accel = _fake_accelerator(distributed_type=DistributedType.NO, zero_stage=None)
    assert not is_deepspeed(accel)
    assert not is_zero3(accel)


def test_is_zero3_false_for_stage_2() -> None:
    from accelerate.utils import DistributedType

    accel = _fake_accelerator(distributed_type=DistributedType.DEEPSPEED, zero_stage=2)
    assert is_deepspeed(accel)
    assert not is_zero3(accel)


def test_is_zero3_true_for_stage_3() -> None:
    from accelerate.utils import DistributedType

    accel = _fake_accelerator(distributed_type=DistributedType.DEEPSPEED, zero_stage=3)
    assert is_deepspeed(accel)
    assert is_zero3(accel)


def test_is_zero3_false_when_plugin_missing() -> None:
    """Defensive: a misconfigured DEEPSPEED accelerator with no plugin
    should not crash; returns False."""
    from accelerate.utils import DistributedType

    accel = _fake_accelerator(
        distributed_type=DistributedType.DEEPSPEED, zero_stage=None
    )
    assert not is_zero3(accel)


# ── activate_zero3_init ──────────────────────────────────────────────────────


def test_activate_zero3_init_no_op_when_not_zero3() -> None:
    """Single-GPU bnb-8bit path: the context is a pure no-op."""
    from accelerate.utils import DistributedType

    accel = _fake_accelerator(distributed_type=DistributedType.NO, zero_stage=None)
    with activate_zero3_init(accel):
        # No HfDeepSpeedConfig should be installed.
        assert _DSCHF_HOLDER == {}
    assert _DSCHF_HOLDER == {}


def test_activate_zero3_init_pins_holder_under_zero3() -> None:
    """Under ZeRO-3 the context installs a sentinel into _DSCHF_HOLDER."""
    from accelerate.utils import DistributedType

    accel = _fake_accelerator(distributed_type=DistributedType.DEEPSPEED, zero_stage=3)
    ds_config_obj: dict[str, Any] = {"zero_optimization": {"stage": 3}}
    accel.state.deepspeed_plugin.deepspeed_config = ds_config_obj

    fake_hfds = MagicMock()
    with patch(
        "transformers.integrations.deepspeed.HfDeepSpeedConfig",
        return_value=fake_hfds,
    ), patch(
        "transformers.integrations.deepspeed.is_deepspeed_zero3_enabled",
        return_value=True,
    ):
        with activate_zero3_init(accel):
            assert id(ds_config_obj) in _DSCHF_HOLDER
            assert _DSCHF_HOLDER[id(ds_config_obj)] is fake_hfds
        # On normal exit the strong ref is RETAINED (model stays governed).
        assert id(ds_config_obj) in _DSCHF_HOLDER


def test_activate_zero3_init_releases_on_exception() -> None:
    """LLR-0048 AC: an exception during the wrapped block releases the
    HfDeepSpeedConfig strong ref so a retry doesn't leak."""
    from accelerate.utils import DistributedType

    accel = _fake_accelerator(distributed_type=DistributedType.DEEPSPEED, zero_stage=3)
    ds_config_obj: dict[str, Any] = {"zero_optimization": {"stage": 3}}
    accel.state.deepspeed_plugin.deepspeed_config = ds_config_obj

    with patch(
        "transformers.integrations.deepspeed.HfDeepSpeedConfig",
        return_value=MagicMock(),
    ), patch(
        "transformers.integrations.deepspeed.is_deepspeed_zero3_enabled",
        return_value=True,
    ), pytest.raises(RuntimeError, match="boom"), activate_zero3_init(accel):
        raise RuntimeError("boom")
    # Holder cleaned up on exception path.
    assert id(ds_config_obj) not in _DSCHF_HOLDER


def test_activate_zero3_init_raises_when_sentinel_inactive() -> None:
    """If HfDeepSpeedConfig is constructed but is_deepspeed_zero3_enabled()
    still returns False, the context raises and releases the strong ref."""
    from accelerate.utils import DistributedType

    accel = _fake_accelerator(distributed_type=DistributedType.DEEPSPEED, zero_stage=3)
    ds_config_obj: dict[str, Any] = {"zero_optimization": {"stage": 3}}
    accel.state.deepspeed_plugin.deepspeed_config = ds_config_obj

    with patch(
        "transformers.integrations.deepspeed.HfDeepSpeedConfig",
        return_value=MagicMock(),
    ), patch(
        "transformers.integrations.deepspeed.is_deepspeed_zero3_enabled",
        return_value=False,
    ), pytest.raises(RuntimeError, match="HfDeepSpeedConfig was instantiated"), activate_zero3_init(accel):
        pass  # pragma: no cover — context raises before yield
    assert id(ds_config_obj) not in _DSCHF_HOLDER


def test_activate_zero3_init_reentry_with_same_config_is_noop() -> None:
    """Re-entering with the same DS config object reuses the existing pin —
    teacher correction → KD in-process must not double-pin."""
    from accelerate.utils import DistributedType

    accel = _fake_accelerator(distributed_type=DistributedType.DEEPSPEED, zero_stage=3)
    ds_config_obj: dict[str, Any] = {"zero_optimization": {"stage": 3}}
    accel.state.deepspeed_plugin.deepspeed_config = ds_config_obj

    constructor = MagicMock(return_value=MagicMock())
    with patch(
        "transformers.integrations.deepspeed.HfDeepSpeedConfig",
        new=constructor,
    ), patch(
        "transformers.integrations.deepspeed.is_deepspeed_zero3_enabled",
        return_value=True,
    ), activate_zero3_init(accel), activate_zero3_init(accel):
        pass
    # Constructor was called exactly once (re-entry was a no-op).
    assert constructor.call_count == 1
