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

# REQ: LLR-0013
# REQ: LLR-0015
# REQ: LLR-0055
# REQ: LLR-0056

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import get_args

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

from ..interface import QuantBlockSubset
from ..specs import Format, KVQuantSpec, WeightPatternSpec
from .ste_simulators import (
    int_quant_ste,
    iq2_xs_quant_ste,
    iq4_xs_quant_ste,
    mxfp4_kv_ste,
    q3_k_quant_ste,
    q5_k_quant_ste,
)

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


class _IQ2XSQuantWeight(nn.Module):
    """IQ2_XS codebook STE parametrization (axis=-1 per LLR-0015)."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        return iq2_xs_quant_ste(w, axis=-1)


class _Q3KQuantWeight(nn.Module):
    """Q3_K codebook STE parametrization (axis=-1 per LLR-0015)."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        return q3_k_quant_ste(w, axis=-1)


class _IQ4XSQuantWeight(nn.Module):
    """IQ4_XS codebook STE parametrization (axis=-1 per LLR-0015)."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        return iq4_xs_quant_ste(w, axis=-1)


class _Q5KQuantWeight(nn.Module):
    """Q5_K codebook STE parametrization (axis=-1 per LLR-0015)."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        return q5_k_quant_ste(w, axis=-1)


_GGUF_PARAMETRIZATIONS: dict[str, type[nn.Module]] = {
    "iq2_xs": _IQ2XSQuantWeight,
    "q3_k": _Q3KQuantWeight,
    "iq4_xs": _IQ4XSQuantWeight,
    "q5_k": _Q5KQuantWeight,
}


_GGUF_FORMATS_ACCEPTED_BY_SCHEMA: frozenset[str] = frozenset(
    f for f in get_args(Format) if f not in {"int", "fp8", "mxfp4", "nvfp4"}
)


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
            self._install_weight_quant(model, quant_block.weight)
        if quant_block.key is not None or quant_block.value is not None:
            self._install_kv_quant(
                model, key=quant_block.key, value=quant_block.value
            )

    def save(self, model: nn.Module, output_dir: Path) -> None:
        """Save the model with simulated-quanted weights in BF16 safetensors.

        Reading ``module.weight`` while the parametrizations from
        LLR-0055 are active routes through the parametrization's forward,
        so ``state_dict()`` returns the fake-quanted tensors directly. The
        BF16 cast keeps storage at the platform's inference dtype; the
        per-pattern bit-widths are recorded into ``config.json`` by the
        caller's ``_inject_quantization_config`` step (LLR-0056) and
        consumed downstream by the GGUF converter (HLR-0017).

        Pre-condition: this is called after ``apply_quant`` has installed
        the parametrizations. If the dispatched subset carried only KV
        hooks (no weight), the call is still safe — state_dict reads the
        unmodified weights, which is the right behavior for a KV-only
        run.
        """
        unwrapped = model
        state_dict = {
            name: tensor.detach().to(torch.bfloat16)
            for name, tensor in unwrapped.state_dict().items()
        }
        unwrapped.save_pretrained(  # type: ignore[operator]
            output_dir, state_dict=state_dict, safe_serialization=True
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
        self, model: nn.Module, specs: list[WeightPatternSpec]
    ) -> None:
        """Walk every ``nn.Linear`` and install the parametrization for the
        first matching ``WeightPatternSpec`` (first-match-wins).

        Precedence: explicit (non-empty) pattern match → carve-out →
        empty-pattern fallback → ValueError. The empty-string ``pattern``
        is a *fallback default* (the uniform-config shim normalises a
        single uniform spec to ``pattern=""``); carve-outs MUST win over
        it so v0 uniform configs continue to respect ``fp32_carve_outs``.
        Explicit per-pattern specs (Profile-J's `gate_proj`, etc.) still
        win over carve-outs, matching the locked design choice #5.
        """
        # Eager validation: reject incoherent or unwired specs up-front so
        # malformed configs fail before the first forward pass.
        for spec in specs:
            self._validate_weight_spec(spec)

        n_installed: dict[str, int] = {}
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            matched = self._first_explicit_match(name, specs)
            if matched is not None:
                self._install_one(module, matched)
                n_installed[matched.format] = n_installed.get(matched.format, 0) + 1
                continue
            if self._is_carved_out(name):
                continue
            fallback = self._fallback_spec(specs)
            if fallback is not None:
                self._install_one(module, fallback)
                n_installed[fallback.format] = n_installed.get(fallback.format, 0) + 1
                continue
            raise ValueError(
                f"NativeBackend: Linear {name!r} matched no spec_map "
                "pattern and no fp32_carve_outs entry. Either add a "
                "pattern that matches or list it under fp32_carve_outs."
            )
        log.info(
            "NativeBackend: installed weight STE on %d Linear modules (%s)",
            sum(n_installed.values()),
            ", ".join(f"{k}={v}" for k, v in sorted(n_installed.items())),
        )

    @staticmethod
    def _first_explicit_match(
        name: str, specs: list[WeightPatternSpec]
    ) -> WeightPatternSpec | None:
        """First spec whose non-empty ``pattern`` substring matches ``name``.

        Empty-string patterns are NOT considered here — they are the
        fallback default applied after carve-outs (see
        ``_fallback_spec`` / ``_install_weight_quant``).
        """
        for spec in specs:
            if spec.pattern != "" and spec.pattern in name:
                return spec
        return None

    @staticmethod
    def _fallback_spec(
        specs: list[WeightPatternSpec],
    ) -> WeightPatternSpec | None:
        """First spec with empty ``pattern`` (uniform-config fallback), or None."""
        for spec in specs:
            if spec.pattern == "":
                return spec
        return None

    @staticmethod
    def _validate_weight_spec(spec: WeightPatternSpec) -> None:
        """Reject (format, granularity, transform) tuples we can't install.

        Eager validation per HLR-0013 / LLR-0055: raise BEFORE attempting
        installation so a malformed config surfaces at training-start.
        """
        if spec.transform != "none":
            raise NotImplementedError(
                f"weight transform={spec.transform!r} deferred to v1+."
            )
        if spec.format == "int":
            if spec.granularity != "channel":
                raise NotImplementedError(
                    f"NativeBackend weight format='int' granularity="
                    f"{spec.granularity!r} not supported; only 'channel'."
                )
            return
        if spec.format in _GGUF_PARAMETRIZATIONS:
            if spec.granularity != "block":
                raise NotImplementedError(
                    f"NativeBackend weight format={spec.format!r} requires "
                    f"granularity='block' (super-block-of-256 layout); got "
                    f"{spec.granularity!r}."
                )
            return
        if spec.format in _GGUF_FORMATS_ACCEPTED_BY_SCHEMA:
            raise NotImplementedError(
                f"NativeBackend weight format={spec.format!r} accepted by "
                "schema but no STE simulator ships in Phase 7.2 — only "
                "int, iq2_xs, q3_k, iq4_xs, q5_k are wired here."
            )
        # Non-GGUF non-int formats (fp8, mxfp4, nvfp4) shouldn't reach the
        # native backend; the factory routes them to ModelOpt via the
        # feature_matrix.
        raise NotImplementedError(
            f"NativeBackend weight format={spec.format!r} not supported "
            "natively; expected routing to ModelOpt via feature_matrix."
        )

    def _install_one(
        self, module: nn.Linear, spec: WeightPatternSpec
    ) -> None:
        """Register the parametrization for ``spec`` on ``module.weight``."""
        param: nn.Module
        if spec.format == "int":
            param = _IntQuantWeight(spec.bits, axis=0)
        else:
            param = _GGUF_PARAMETRIZATIONS[spec.format]()
        parametrize.register_parametrization(module, "weight", param)
        self._parametrized.append(module)

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
