"""NativeBackend — hand-rolled STE behind the QuantBackend Protocol (LLR-0015).

NativeBackend installs forward hooks (and `nn.utils.parametrize`
parametrizations for weight quant) that route through the pure simulator
functions in :mod:`.ste_simulators`. It does NOT call ``mtq.quantize`` —
NativeBackend is what makes kdr's recipe space larger than modelopt's
matrix.

Granularity mapping (v0):

  * weight ``channel``  → ``axis = 0`` (per-output-channel of an ``nn.Linear``
    weight ``[out_features, in_features]``)
  * key ``channel``     → ``axis = -1`` (per-channel along the K head_dim)
  * value ``token``     → ``axis = -2`` (per-token along the V seq_len axis,
    assuming a ``[B, H, T, D]`` layout — Phase 5 verifies against ZAYA1's CCA)

Other granularities raise ``NotImplementedError`` with a Phase-5 pointer; the
simulator functions themselves are general (any axis), so adding more is a
backend-local change.
"""

# REQ: LLR-0015

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

from ..interface import QuantBlockSubset
from ..specs import KVQuantSpec, WeightPatternSpec
from .ste_simulators import int_quant_ste, mxfp4_kv_ste

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Weight parametrization (LLR-0015)
# ─────────────────────────────────────────────────────────────────────────────


class _IntQuantWeight(nn.Module):
    """``nn.utils.parametrize`` parametrization that fake-quants a weight.

    On every access of ``module.weight``, PyTorch routes through this
    parametrization's forward, so the matmul sees the fake-quanted tensor
    with STE gradient pass-through.
    """

    def __init__(self, bits: int, axis: int) -> None:
        super().__init__()
        self.bits = bits
        self.axis = axis

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        return int_quant_ste(w, self.bits, axis=self.axis)


# ─────────────────────────────────────────────────────────────────────────────
# Backend
# ─────────────────────────────────────────────────────────────────────────────


class NativeBackend:
    """``QuantBackend`` Protocol implementation with hand-rolled STE."""

    name: str = "native"

    def __init__(
        self,
        *,
        attention_module_paths: list[str] | None = None,
        kv_quant_exempt_indices: list[int] | None = None,
        fp32_carve_outs: list[str] | None = None,
    ) -> None:
        self.attention_module_paths = attention_module_paths or []
        self.kv_quant_exempt_indices = list(kv_quant_exempt_indices or [])
        self.fp32_carve_outs = list(fp32_carve_outs or [])
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._parametrized: list[nn.Linear] = []
        self._quant_block: QuantBlockSubset | None = None

    # ---- Protocol surface -------------------------------------------------

    def apply_quant(self, model: nn.Module, quant_block: QuantBlockSubset) -> None:
        """Install fake-quant on ``model`` for the populated fields of ``quant_block``."""
        if quant_block.is_empty():
            raise ValueError("backend received an empty quant block")
        self._quant_block = quant_block

        if quant_block.weight is not None:
            if (
                len(quant_block.weight) == 1
                and quant_block.weight[0].pattern == ""
            ):
                # Uniform shim path: single empty-pattern entry, install
                # globally over every non-carve-out Linear as before.
                self._install_weight_quant(model, quant_block.weight[0])
            else:
                raise NotImplementedError(
                    "NativeBackend mixed-spec weight install (list with "
                    "multiple entries or non-empty patterns) lands in "
                    "Phase 7.2 Task 5. "
                    f"Got {len(quant_block.weight)} pattern(s): "
                    f"{[p.pattern for p in quant_block.weight]!r}"
                )
        if quant_block.key is not None or quant_block.value is not None:
            self._install_kv_quant(
                model, key=quant_block.key, value=quant_block.value
            )

    def save(self, model: nn.Module, output_dir: Path) -> None:
        """Pure-Native compressed-tensors save is a Phase 6 concern.

        v0 supports Native only for KV-cache fake-quant (KV-cache schemes are
        runtime metadata, not stored tensors); when Native owns weight quant
        too, the saved tensors live in the ``compressed_tensors`` packed
        formats and require a Native-side converter that does not exist in
        Phase 4. Mixed-mode runs (NVFP4 weight via ModelOpt + INT3 KV via
        Native) save via the weight-handling backend (typically ModelOpt).
        """
        raise NotImplementedError(
            "NativeBackend.save: pure-Native weight save requires a "
            "compressed_tensors converter for the chosen INT-N format. v0 "
            "supports Native for KV quant only; pure-Native weight runs "
            "wait on the Phase 6 vast.ai bootstrap to add an INT3/INT2 "
            "compressed-tensors converter wrapper."
        )

    # ---- Hook lifecycle ---------------------------------------------------

    def remove_all_hooks(self) -> None:
        """Tear down every parametrization + forward hook this backend owns.

        Useful in tests. Not strictly required at end-of-training because
        the model object is dropped on save.
        """
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        for module in self._parametrized:
            if parametrize.is_parametrized(module, "weight"):
                parametrize.remove_parametrizations(module, "weight", leave_parametrized=False)
        self._parametrized.clear()

    # ---- Weight install ---------------------------------------------------

    def _install_weight_quant(
        self, model: nn.Module, spec: WeightPatternSpec
    ) -> None:
        """Register a parametrization on every non-carve-out ``nn.Linear``."""
        assert spec.pattern == "", (
            "_install_weight_quant in Task 3 only handles the uniform shim "
            "(empty pattern; matches every Linear). Per-pattern install "
            "lands in Task 5; apply_quant's caller must enforce this "
            "invariant before reaching here."
        )
        if spec.granularity != "channel":
            raise NotImplementedError(
                f"NativeBackend weight granularity={spec.granularity!r} not "
                "supported in v0; only 'channel' is implemented. Phase 5+ "
                "extends this when needed."
            )
        if spec.format != "int":
            raise NotImplementedError(
                f"NativeBackend weight format={spec.format!r} not supported in "
                "v0; INT3/INT2 paths use 'int'. Other formats route to "
                "ModelOpt via feature_matrix."
            )
        if spec.transform != "none":
            raise NotImplementedError(
                f"weight transform={spec.transform!r} deferred to v1+ per the plan."
            )

        n_installed = 0
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if self._is_carved_out(name):
                continue
            parametrize.register_parametrization(
                module, "weight", _IntQuantWeight(spec.bits, axis=0)
            )
            self._parametrized.append(module)
            n_installed += 1
        log.info(
            "NativeBackend: installed weight STE on %d Linear modules "
            "(bits=%d, axis=0)",
            n_installed,
            spec.bits,
        )

    def _is_carved_out(self, dotted_name: str) -> bool:
        """Substring match against the carve-out list.

        v0 keeps this deliberately simple — substrings cover the ZAYA1 §IV-D
        targets (``lm_head``, ``rmsnorm``, etc.) without parsing wildcards.
        Phase 5+ may upgrade to ``fnmatch`` if the carve-out grammar grows.
        """
        return any(p in dotted_name for p in self.fp32_carve_outs)

    # ---- KV install -------------------------------------------------------

    def _install_kv_quant(
        self,
        model: nn.Module,
        *,
        key: KVQuantSpec | None,
        value: KVQuantSpec | None,
    ) -> None:
        """Register forward hooks at adapter-declared attention module paths.

        Phase 4 verification scope: hooks attach at the declared paths and
        invoke the right simulator on tensor outputs. The exact intercept
        point inside ZAYA1's CCA module is Phase 5's responsibility (the
        adapter's ``attention_module_paths`` is what locks the contract).
        """
        if not self.attention_module_paths:
            log.info(
                "NativeBackend: attention_module_paths is empty — no KV hooks "
                "installed. Adapter must declare paths for KV quant to take "
                "effect."
            )
            return

        if key is not None:
            self._validate_kv_spec("key", key)
        if value is not None:
            self._validate_kv_spec("value", value)

        exempt = set(self.kv_quant_exempt_indices)
        n_installed = 0
        for idx, path in enumerate(self.attention_module_paths):
            if idx in exempt:
                continue
            try:
                module = model.get_submodule(path)
            except AttributeError as e:
                raise ValueError(
                    f"attention_module_paths[{idx}]={path!r} not found on model"
                ) from e
            handle = module.register_forward_hook(
                _make_kv_hook(key_spec=key, value_spec=value)
            )
            self._handles.append(handle)
            n_installed += 1
        log.info(
            "NativeBackend: installed KV STE on %d attention modules "
            "(key=%s, value=%s)",
            n_installed,
            None if key is None else f"{key.bits}b/{key.format}",
            None if value is None else f"{value.bits}b/{value.format}",
        )

    @staticmethod
    def _validate_kv_spec(role: str, spec: KVQuantSpec) -> None:
        if role == "key" and spec.granularity != "channel":
            raise NotImplementedError(
                f"NativeBackend key granularity={spec.granularity!r} not "
                "supported; only 'channel' (per-channel along head_dim)."
            )
        if role == "value" and spec.granularity != "token":
            raise NotImplementedError(
                f"NativeBackend value granularity={spec.granularity!r} not "
                "supported; only 'token' (per-token along seq_len)."
            )
        if spec.format not in ("int", "mxfp4"):
            raise NotImplementedError(
                f"NativeBackend KV format={spec.format!r} not supported "
                "(only 'int' and 'mxfp4' are implemented natively)."
            )
        if spec.transform != "none":
            raise NotImplementedError(
                f"KV transform={spec.transform!r} deferred to v1+ per the plan."
            )


# ─────────────────────────────────────────────────────────────────────────────
# KV hook factory (module-level so closures capture nothing pickle-hostile)
# ─────────────────────────────────────────────────────────────────────────────

# Hardcoded axis convention (matches BHTD layout). The actual axis for ZAYA1's
# CCA may differ; Phase 5 confirms via direct shape inspection at adapter
# instantiation. v0 keeps these constants explicit so misalignment is visible
# in code review rather than buried in module reshape gymnastics.
_KEY_AXIS = -1   # head_dim (per-channel along K head_dim)
_VALUE_AXIS = -2  # seq_len  (per-token along V seq_len)


def _make_kv_hook(
    *,
    key_spec: KVQuantSpec | None,
    value_spec: KVQuantSpec | None,
) -> Callable[[nn.Module, object, object], object]:
    """Build a forward-hook closure that fake-quants tensor outputs.

    The closure handles three module-output shapes:

      * single tensor → applied as `value` if value_spec is set, else as `key`
      * (k, v) tuple  → element 0 fake-quanted with key_spec, element 1 with
        value_spec (if set)
      * other tuples / dicts → returned unchanged (Phase 5 specialises)

    The hook returns the new output; PyTorch substitutes it transparently.
    """

    def _quant_one(t: torch.Tensor, spec: KVQuantSpec | None, axis: int) -> torch.Tensor:
        if spec is None or not isinstance(t, torch.Tensor):
            return t
        if spec.format == "int":
            return int_quant_ste(t, spec.bits, axis=axis)
        if spec.format == "mxfp4":
            return mxfp4_kv_ste(t, axis=axis)
        # Validation in `_validate_kv_spec` rejects other formats earlier;
        # this branch is unreachable given current validation but kept as a
        # defensive fallback.
        return t

    def hook(_module: nn.Module, _inputs: object, output: object) -> object:
        if isinstance(output, torch.Tensor):
            # Ambiguous: a single-tensor output can't carry both K and V.
            # Prefer value if both are configured (more common asymmetry).
            if value_spec is not None:
                return _quant_one(output, value_spec, _VALUE_AXIS)
            return _quant_one(output, key_spec, _KEY_AXIS)
        if isinstance(output, tuple) and len(output) >= 2:
            k_in, v_in = output[0], output[1]
            new_k = (
                _quant_one(k_in, key_spec, _KEY_AXIS)
                if isinstance(k_in, torch.Tensor)
                else k_in
            )
            new_v = (
                _quant_one(v_in, value_spec, _VALUE_AXIS)
                if isinstance(v_in, torch.Tensor)
                else v_in
            )
            return (new_k, new_v, *output[2:])
        return output

    return hook
