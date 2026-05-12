"""ModelOptBackend — wraps ``mtq.quantize`` and the compressed-tensors
converter behind the QuantBackend Protocol (LLR-0014, LLR-0021).

Phase 4 verification scope (per
``/home/lucas/.claude/plans/radiant-pondering-beacon.md``): real
``mtq.quantize`` and the compressed-tensors converter are exercised on
vast.ai in Phase 6. Phase 4 lands the code shape (mypy/ruff/pytest clean),
the call-count invariant, and the constructor-injected calibration loop;
unit tests stub modelopt and compressed_tensors.

Constructor injection
---------------------

The QuantBackend Protocol declares ``apply_quant(model, quant_block)`` —
two args, no calibration channel. The calibration loop callable
``mtq.quantize`` consumes is therefore bound to the backend at construction
time by ``factory.partition_and_dispatch`` (which has access to the YAML's
calibration block and the pre-tokenized batch tensor). This keeps the
Protocol shape symmetric across ModelOpt and Native (which has no
calibration step).
"""

# REQ: LLR-0014
# REQ: LLR-0021

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import torch.nn as nn

from ..interface import QuantBlockSubset
from .config_map import quant_block_to_modelopt_config
from .feature_matrix import resolve_converter_class

if TYPE_CHECKING:
    # Avoid hard-importing modelopt at module load — kdr must be importable
    # without it. The runtime import lives inside ``apply_quant``.
    pass

log = logging.getLogger(__name__)


# Type alias for the calibration loop callable: takes a model, runs it on
# the PTQ subset, returns nothing. ``factory.partition_and_dispatch`` builds
# this closure from the YAML's calibration block + pre-tokenized batches.
CalibrateLoop = Callable[[nn.Module], None]


class ModelOptBackend:
    """``QuantBackend`` Protocol implementation for NVIDIA modelopt."""

    name: str = "modelopt"

    def __init__(
        self,
        *,
        calibrate_loop: CalibrateLoop | None = None,
        fp32_carve_outs: list[str] | None = None,
        weight_target_pattern: str | None = None,
        key_target_pattern: str | None = None,
        value_target_pattern: str | None = None,
    ) -> None:
        """Construct with the calibration loop + adapter-supplied wildcards.

        Args:
            calibrate_loop: callable consumed by ``mtq.quantize``'s
                ``forward_loop`` parameter; runs the model on the PTQ subset
                so per-tensor scales / zero-points get fitted. ``None`` is
                accepted (e.g. for unit tests where calibration is skipped).
            fp32_carve_outs: substring patterns of submodules to exclude
                from quantization (the adapter's FP32 carve-outs).
            weight_target_pattern: optional override for the ``weight_quantizer``
                wildcard. Default is stock-HF ``*weight_quantizer``.
            key_target_pattern: optional override for the K output_quantizer
                wildcard. Default is stock-HF ``*[Kk]_proj*output_quantizer``.
            value_target_pattern: optional override for the V output_quantizer
                wildcard. Default is stock-HF ``*[Vv]_proj*output_quantizer``.
        """
        self.calibrate_loop = calibrate_loop
        self.fp32_carve_outs = list(fp32_carve_outs or [])
        self.weight_target_pattern = weight_target_pattern
        self.key_target_pattern = key_target_pattern
        self.value_target_pattern = value_target_pattern
        self._quant_block: QuantBlockSubset | None = None

    def apply_quant(self, model: nn.Module, quant_block: QuantBlockSubset) -> None:
        """Build a single modelopt config from ``quant_block`` and call
        ``mtq.quantize`` exactly once.

        LLR-0014 AC #1: this method calls ``mtq.quantize`` exactly once per
        invocation. The factory's partition-or-merge logic (LLR-0014 AC #2)
        guarantees that even when ModelOpt is routed to multiple quantizers,
        only one combined ``QuantBlockSubset`` arrives here per run, so the
        invariant holds at the entry point.
        """
        if quant_block.is_empty():
            raise ValueError("backend received an empty quant block")

        # ModelOptBackend's compressed-tensors converter is single-format-
        # per-run. A Mixed config can only route a single ModelOpt-supported
        # pattern here; multi-entry weights are a Task 5 / future-Task-6
        # concern (review H4 — runtime gate at the entry point rather than
        # buried inside config_map.py).
        if quant_block.weight is not None and len(quant_block.weight) != 1:
            raise NotImplementedError(
                "ModelOptBackend.apply_quant: per-pattern weight quant with "
                "multiple patterns is not supported. Mixed configs route "
                "every GGUF pattern to NativeBackend; only one "
                "modelopt-routable pattern can land here. Got "
                f"{len(quant_block.weight)} patterns: "
                f"{[p.pattern for p in quant_block.weight]!r}"
            )

        kwargs: dict[str, str] = {}
        if self.weight_target_pattern is not None:
            kwargs["weight_target_pattern"] = self.weight_target_pattern
        if self.key_target_pattern is not None:
            kwargs["key_target_pattern"] = self.key_target_pattern
        if self.value_target_pattern is not None:
            kwargs["value_target_pattern"] = self.value_target_pattern
        cfg = quant_block_to_modelopt_config(
            quant_block,
            ignore=self.fp32_carve_outs,
            **kwargs,
        )

        import modelopt.torch.quantization as mtq

        # Single call enforced by entry-point shape — LLR-0014 AC #1.
        mtq.quantize(model, cfg, self.calibrate_loop)
        self._quant_block = quant_block
        log.info(
            "ModelOptBackend.apply_quant: mtq.quantize complete "
            "(weight=%s, key=%s, value=%s, ignore=%d patterns)",
            None
            if quant_block.weight is None
            else (
                f"{quant_block.weight[0].pattern!r}->"
                f"{quant_block.weight[0].bits}b/{quant_block.weight[0].format}"
            ),
            None if quant_block.key is None else quant_block.key.format,
            None if quant_block.value is None else quant_block.value.format,
            len(self.fp32_carve_outs),
        )

    def save(self, model: nn.Module, output_dir: Path) -> None:
        """Emit compressed-tensors safetensors via the converter selected by
        ``feature_matrix`` for the weight format actually used.

        LLR-0021: the converter class is resolved at runtime from
        ``compressed_tensors.converters`` so unsupported formats raise a
        clear error rather than silently emitting modelopt-internal layouts.
        """
        if self._quant_block is None:
            raise RuntimeError(
                "ModelOptBackend.save called before apply_quant; the converter "
                "needs the recipe (weight format) to pick its class."
            )
        if self._quant_block.weight is None:
            raise RuntimeError(
                "ModelOptBackend.save: weight quantization is required for the "
                "compressed-tensors save path. KV-only ModelOpt routing is "
                "unusual; the weight-handling backend owns the tensor save."
            )
        # Defense-in-depth (review H4): ``apply_quant`` already enforces
        # len==1, but ``save`` is reachable independently of ``apply_quant``
        # (after a process restart from a checkpoint, for example).
        if len(self._quant_block.weight) != 1:
            raise NotImplementedError(
                "ModelOptBackend.save: per-pattern weight save with multiple "
                "patterns is not supported by compressed-tensors converters. "
                "Mixed-precision configs that include modelopt-routable "
                "formats alongside others will need Task 6's GGUF save path "
                "or a future per-pattern compressed-tensors path. Got "
                f"{len(self._quant_block.weight)} patterns."
            )
        weight_spec = self._quant_block.weight[0]

        output_dir.mkdir(parents=True, exist_ok=True)

        converter_cls = resolve_converter_class(weight_spec.format)
        converter = converter_cls()

        # Compressed-tensors converters expose either ``save_pretrained(model,
        # output_dir)`` or ``convert(model, output_dir)``. Probe both rather
        # than pinning a method name across versions of ``compressed_tensors``.
        save_pretrained = getattr(converter, "save_pretrained", None)
        convert = getattr(converter, "convert", None)
        if callable(save_pretrained):
            save_pretrained(model, output_dir)
        elif callable(convert):
            convert(model, output_dir)
        else:
            raise RuntimeError(
                f"compressed_tensors converter {converter_cls.__name__!r} "
                "exposes neither `save_pretrained` nor `convert`; "
                "extend ModelOptBackend.save when this changes."
            )
        log.info(
            "ModelOptBackend.save: wrote compressed-tensors checkpoint to %s "
            "via %s",
            output_dir,
            converter_cls.__name__,
        )
