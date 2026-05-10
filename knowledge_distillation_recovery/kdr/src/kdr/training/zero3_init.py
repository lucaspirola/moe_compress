"""ZeRO-3 detection + `HfDeepSpeedConfig` activation context.

Direct port from `structural_recovery/distillation.py:226-275`.

The `_DSCHF_HOLDER` pattern keeps a strong reference to the
`HfDeepSpeedConfig` instance so Python's GC cannot drop it before
`from_pretrained` finishes constructing the model. If the config is GC'd
during construction, the model loads as a single full-precision copy on
rank 0 and any subsequent shard call silently no-ops — eating
GPU/CPU memory until the run OOMs hours later (P0 hazard per the plan).

Required call order under ZeRO-3 (verified by tests):
  1. enter `activate_zero3_init(accelerator)` context
  2. inside: `from_pretrained(teacher)`
  3. inside: `from_pretrained(student)`
  4. inside (still): `mtq.quantize(student, ...)` if mode == "da_qad"
  5. exit context
  6. outside: `accelerator.prepare(student, optimizer)` — DeepSpeed installs
     runtime hooks on the now-quantized, sharded student.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager

from accelerate import Accelerator

log = logging.getLogger(__name__)


# REQ: LLR-0048
# Module-level dict pinning HfDeepSpeedConfig instances by id() of the
# associated DS config dict. Strong refs survive function exits so that
# `from_pretrained` (which checks `is_deepspeed_zero3_enabled()` via a
# weakref-aware sentinel) sees the active config. The annotation is
# `dict[int, object]` (not `dict[int, HfDeepSpeedConfig]`) because the
# `transformers.deepspeed` module is in mypy's `ignore_missing_imports`
# override list — naming the class directly would leak `Any` through the
# strict-typing wall. Stored values ARE `HfDeepSpeedConfig` at runtime.
#
# Concurrency: the check-then-set in `activate_zero3_init` is intentionally
# unsynchronised. The caller contract is that `activate_zero3_init` runs on
# the main training thread only — DataLoader workers and CUDA streams do
# not enter this context. A `threading.Lock` is therefore unnecessary; the
# parallel pattern in `kd_loss._KLD_LOSS_CACHE` uses a lock because that
# cache may be touched from DataLoader workers, a constraint that does not
# apply here. If a future call site invokes `activate_zero3_init` from a
# worker thread, add a lock — until then it would be cosmetic.
_DSCHF_HOLDER: dict[int, object] = {}


def is_deepspeed(accelerator: Accelerator) -> bool:
    """True if the active Accelerator was initialised with DeepSpeed."""
    from accelerate.utils import DistributedType

    return bool(accelerator.distributed_type == DistributedType.DEEPSPEED)


# REQ: LLR-0040
def is_zero3(accelerator: Accelerator) -> bool:
    """True iff `accelerator` is configured with DeepSpeed ZeRO stage 3.

    Returns False outside DeepSpeed. Returns True only when
    `plugin.zero_stage >= 3` — stages 1 and 2 do not shard parameters and
    therefore do not need `HfDeepSpeedConfig` activation.
    """
    if not is_deepspeed(accelerator):
        return False
    plugin = getattr(accelerator.state, "deepspeed_plugin", None)
    if plugin is None:
        return False
    stage = getattr(plugin, "zero_stage", None)
    if stage is None:
        return False
    return int(stage) >= 3


# REQ: LLR-0048
@contextmanager
def activate_zero3_init(accelerator: Accelerator) -> Iterator[None]:
    """Pin a `HfDeepSpeedConfig` for the lifetime of model construction.

    No-op when `is_zero3(accelerator) == False` — single-GPU bnb-8bit and
    non-DS multi-GPU paths are unaffected. Idempotent: re-entering with the
    same DS config object reuses the existing pin (so e.g. teacher correction
    → KD in-process doesn't double-pin).

    On exception during the wrapped block, the strong reference is released
    so a retry doesn't leak the previous (now-garbage) `HfDeepSpeedConfig`.
    On normal exit the reference is RETAINED — the constructed model still
    needs `is_deepspeed_zero3_enabled()` to return True for collective ops.
    """
    if not is_zero3(accelerator):
        yield
        return

    # The public re-exports in `transformers.integrations` are not declared
    # in the package's `__all__`; mypy reports `attr-defined` on the parent
    # package even though both names exist at runtime. Importing from the
    # actual submodule (`transformers.integrations.deepspeed`) avoids that.
    from transformers.integrations.deepspeed import (
        HfDeepSpeedConfig,
        is_deepspeed_zero3_enabled,
    )

    plugin = accelerator.state.deepspeed_plugin
    ds_config = plugin.deepspeed_config
    cfg_key = id(ds_config)

    holder_was_set = False
    if cfg_key not in _DSCHF_HOLDER:
        _DSCHF_HOLDER[cfg_key] = HfDeepSpeedConfig(ds_config)
        holder_was_set = True
        if not is_deepspeed_zero3_enabled():
            # The HfDS sentinel didn't activate — release the strong ref before
            # raising so a retry isn't poisoned.
            _DSCHF_HOLDER.pop(cfg_key, None)
            try:
                import deepspeed  # noqa: F401

                ds_avail = True
            except ImportError:
                ds_avail = False
            raise RuntimeError(
                "HfDeepSpeedConfig was instantiated but "
                "is_deepspeed_zero3_enabled() returned False — the model "
                "would load full-rank on each rank and OOM. "
                f"plugin.zero_stage={getattr(plugin, 'zero_stage', '?')}, "
                f"deepspeed_importable={ds_avail}, "
                f"ds_config['zero_optimization']['stage']="
                f"{ds_config.get('zero_optimization', {}).get('stage', '?')}."
            )
        log.info("HfDeepSpeedConfig activated for ZeRO-3 sharded from_pretrained.")

    try:
        yield
    except BaseException:
        # Release the strong ref ONLY if we were the ones who set it on this
        # entry — re-entering a still-active context shouldn't cancel a
        # previous successful activation.
        if holder_was_set:
            _DSCHF_HOLDER.pop(cfg_key, None)
        raise
