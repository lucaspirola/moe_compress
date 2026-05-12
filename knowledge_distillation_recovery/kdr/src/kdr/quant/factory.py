"""Per-quantizer backend routing factory (LLR-0017).

``partition_and_dispatch``:

  1. examines the YAML's ``QuantBlock``;
  2. consults ``feature_matrix.SUPPORTED_QUANTS`` to pick ModelOpt vs Native
     for each of the three quantizers (weight, key, value);
  3. *merges* per-backend routes into a single ``QuantBlockSubset`` per
     backend (LLR-0013 AC #4 / LLR-0014 AC #2 — never two dispatches to the
     same backend);
  4. constructs each routed backend with adapter-supplied context
     (``fp32_carve_outs``, attention module paths, KV-exempt indices) and the
     calibration-loop closure;
  5. invokes each backend's ``apply_quant`` exactly once;
  6. returns the list of backends actually used so the caller (training
     loop) can dispatch the matching ``save`` later.

The function signature is widened from the Phase-2 stub to accept the
calibration tensor and adapter info; ``training/loop.py``'s ``da_qad``
branch passes them.
"""

# REQ: LLR-0017
# REQ: LLR-0042

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

from ..config import QuantBlock
from .interface import QuantBackend, QuantBlockSubset
from .modelopt_backend.backend import CalibrateLoop, ModelOptBackend
from .modelopt_backend.feature_matrix import is_supported
from .native_backend.backend import NativeBackend
from .specs import (
    KVQuantSpec,
    MixedWeightSpec,
    UniformWeightSpec,
    WeightPatternSpec,
)

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Routing types
# ─────────────────────────────────────────────────────────────────────────────

# Typo-safety in routing dicts (review H1): a typo like ``"modelpot"`` becomes
# a mypy error rather than a silent "no weight on this backend" via
# ``dict.get()``.
BackendName = Literal["modelopt", "native"]


@dataclass(frozen=True, slots=True)
class WeightRoutes:
    """Per-pattern weight routing. One entry per WeightPatternSpec in spec_map.

    Internal-only to ``factory.py`` — not Pydantic to skip the strict-typing
    auto-discovery in ``tests/test_strict_typing.py`` (see review H2). These
    records are built from already-validated ``WeightPatternSpec`` /
    ``KVQuantSpec`` instances; Pydantic re-validation buys nothing.
    """

    by_backend: dict[BackendName, list[WeightPatternSpec]]


@dataclass(frozen=True, slots=True)
class QuantRoutes:
    """All routing decisions for one QuantBlock."""

    weight: WeightRoutes
    key: BackendName
    value: BackendName


def partition_and_dispatch(
    model: nn.Module,
    quant_block: QuantBlock,
    *,
    calibration_batches: list[torch.Tensor] | None = None,
    ptq_subset_size: int = 0,
    fp32_carve_outs: list[str] | None = None,
    attention_module_paths: list[str] | None = None,
    kv_quant_exempt_indices: list[int] | None = None,
    weight_target_pattern: str | None = None,
    key_target_pattern: str | None = None,
    value_target_pattern: str | None = None,
) -> list[QuantBackend]:
    """Partition ``quant_block`` per backend and dispatch.

    Args:
        model: the student model to install fake-quant on (already loaded
            inside the ``activate_zero3_init`` context per LLR-0048).
        quant_block: the YAML's top-level ``quant`` block (weight +
            ``kv_quant.{key,value}``).
        calibration_batches: pre-tokenized calibration batches as a list of
            ``[B, T]`` long tensors. Required when ModelOpt is routed; used
            to build the ``mtq.quantize`` ``calibrate_forward_loop``.
        ptq_subset_size: PTQ calibration subset size in **sequences**
            (LLR-0042). The first ``ptq_subset_size`` sequences (contiguous
            from index 0) feed modelopt's calibrate_forward_loop.
        fp32_carve_outs: adapter's FP32 carve-out submodule patterns.
        attention_module_paths: adapter's K/V hook target paths (Native).
        kv_quant_exempt_indices: adapter's KV-quant-exempt layer indices.
        weight_target_pattern, key_target_pattern, value_target_pattern:
            optional adapter-driven wildcards for ModelOpt; defaults are
            stock-HF in :mod:`.modelopt_backend.config_map`.

    Returns:
        The list of routed backends (length 0, 1, or 2). Each backend's
        ``apply_quant`` has been called exactly once.

    Raises:
        ValueError: if ``calibration_batches`` is missing while a quantizer
            routes to ModelOpt, or if ``ptq_subset_size`` is non-positive.
    """
    fp32_carve_outs = list(fp32_carve_outs or [])
    attention_module_paths = list(attention_module_paths or [])
    kv_quant_exempt_indices = list(kv_quant_exempt_indices or [])

    # ── 1+2. Route each quantizer to a backend, build per-backend subsets ─
    routes = _route_quantizers(quant_block)
    modelopt_subset = _subset_for(quant_block, routes, "modelopt")
    native_subset = _subset_for(quant_block, routes, "native")

    log.info(
        "partition_and_dispatch: weight_routes=%s, key=%s, value=%s; "
        "modelopt=%s, native=%s",
        {
            k: [f"{p.pattern!r}->{p.bits}b/{p.format}" for p in v]
            for k, v in sorted(routes.weight.by_backend.items())
        },
        routes.key,
        routes.value,
        _describe_subset(modelopt_subset),
        _describe_subset(native_subset),
    )

    backends: list[QuantBackend] = []

    # ── 3. Construct + dispatch ModelOpt (if any quantizers route to it) ──
    if not modelopt_subset.is_empty():
        # Guard both `None` AND empty list: an empty list would silently
        # produce a no-op `calibrate_loop`, leaving modelopt with default /
        # zero scales and a numerically-corrupted quantized model.
        if not calibration_batches:
            raise ValueError(
                "partition_and_dispatch: ModelOpt was routed but "
                "calibration_batches is missing or empty. Pass the "
                "pre-tokenized calibration tensor (≥1 batch) when invoking "
                "with a da_qad config."
            )
        if ptq_subset_size <= 0:
            raise ValueError(
                f"ptq_subset_size must be > 0 when ModelOpt is routed; "
                f"got {ptq_subset_size}"
            )
        calibrate_loop = _make_calibrate_loop(calibration_batches, ptq_subset_size)
        modelopt = ModelOptBackend(
            calibrate_loop=calibrate_loop,
            fp32_carve_outs=fp32_carve_outs,
            weight_target_pattern=weight_target_pattern,
            key_target_pattern=key_target_pattern,
            value_target_pattern=value_target_pattern,
        )
        # REQ: LLR-0014
        modelopt.apply_quant(model, modelopt_subset)
        backends.append(modelopt)

    # ── 4. Construct + dispatch Native (if any quantizers route to it) ────
    if not native_subset.is_empty():
        native = NativeBackend(
            attention_module_paths=attention_module_paths,
            kv_quant_exempt_indices=kv_quant_exempt_indices,
            fp32_carve_outs=fp32_carve_outs,
        )
        native.apply_quant(model, native_subset)
        backends.append(native)

    return backends


# ─────────────────────────────────────────────────────────────────────────────
# Routing
# ─────────────────────────────────────────────────────────────────────────────


def _normalize_weight_to_patterns(
    weight: UniformWeightSpec | MixedWeightSpec,
) -> list[WeightPatternSpec]:
    """Lift either spec shape to a uniform list-of-patterns form for routing.

    Uniform -> single-entry list with ``pattern=""`` (the catch-all default;
    matches every Linear at install time per LLR-0024 v2 precedence).
    Mixed   -> the spec_map as-is.

    The empty-string pattern is a documented catch-all: when authored in a
    user's ``MixedWeightSpec.spec_map``, the Pydantic duplicate-pattern
    validator permits a single entry, and first-match-wins means it should
    be placed LAST to act as the default. The Uniform shim is just the
    degenerate case (one entry = the catch-all).
    """
    if isinstance(weight, MixedWeightSpec):
        return list(weight.spec_map)
    # UniformWeightSpec branch
    return [
        WeightPatternSpec(
            pattern="",
            bits=weight.bits,
            format=weight.format,
            granularity=weight.granularity,
            transform=weight.transform,
        )
    ]


def _route_quantizers(quant_block: QuantBlock) -> QuantRoutes:
    """Route each pattern + K/V to ``modelopt`` or ``native``.

    Weight routing is per-pattern: each ``WeightPatternSpec`` is bucketed
    by backend based on ``is_supported("weight", format, bits)``. ``dict``
    is insertion-ordered (PEP 468 / Python 3.7+); the per-backend
    ``list[WeightPatternSpec]`` therefore preserves spec_map's original
    ordering, which is load-bearing for first-match-wins precedence in
    Task 5's per-pattern install path (review M1).
    """
    patterns = _normalize_weight_to_patterns(quant_block.weight)
    by_backend: dict[BackendName, list[WeightPatternSpec]] = {}
    for p in patterns:
        backend: BackendName = (
            "modelopt" if is_supported("weight", p.format, p.bits) else "native"
        )
        by_backend.setdefault(backend, []).append(p)
    return QuantRoutes(
        weight=WeightRoutes(by_backend=by_backend),
        key=_route_one("kv_key", quant_block.kv_quant.key),
        value=_route_one("kv_value", quant_block.kv_quant.value),
    )


def _route_one(
    target: Literal["kv_key", "kv_value"], spec: KVQuantSpec
) -> BackendName:
    """Return ``"modelopt"`` if the matrix supports this K/V tuple, else
    ``"native"``.

    Narrowed parameter type (review C3) means typos become a mypy error
    rather than a silent fall-through to Native. Weight routing is handled
    per-pattern inline in :func:`_route_quantizers`.
    """
    if is_supported(target, spec.format, spec.bits):
        return "modelopt"
    return "native"


def _subset_for(
    quant_block: QuantBlock,
    routes: QuantRoutes,
    backend_name: BackendName,
) -> QuantBlockSubset:
    """Build a ``QuantBlockSubset`` filtered to ``backend_name``. Empty if none."""
    weight_for_backend = routes.weight.by_backend.get(backend_name)
    return QuantBlockSubset(
        weight=weight_for_backend if weight_for_backend else None,
        key=quant_block.kv_quant.key if routes.key == backend_name else None,
        value=quant_block.kv_quant.value if routes.value == backend_name else None,
    )


def _describe_subset(subset: QuantBlockSubset) -> str:
    """Human-readable one-liner for the log line.

    Per-pattern output: ``weight[<pattern>]=Xb/format``. Review M2 — this
    breaks any external log parser that grepped for ``weight=`` literally;
    no in-tree consumer is known. Acknowledged here so we have a record if
    a Grafana dashboard elsewhere needs updating.
    """
    if subset.is_empty():
        return "<empty>"
    parts: list[str] = []
    if subset.weight is not None:
        for p in subset.weight:
            label = p.pattern if p.pattern else "<default>"
            parts.append(f"weight[{label}]={p.bits}b/{p.format}")
    if subset.key is not None:
        parts.append(f"key={subset.key.bits}b/{subset.key.format}")
    if subset.value is not None:
        parts.append(f"value={subset.value.bits}b/{subset.value.format}")
    return ",".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Calibration loop construction (LLR-0042)
# ─────────────────────────────────────────────────────────────────────────────


def _make_calibrate_loop(
    batches: list[torch.Tensor], ptq_subset_size: int
) -> CalibrateLoop:
    """Build the calibrate_loop closure passed to ``mtq.quantize``.

    LLR-0042 AC #1: subset is contiguous from index 0.
    LLR-0042 AC #2: modelopt receives ≤ ptq_subset_size batches' worth of
    sequences (we slice to exactly ``ptq_subset_size`` sequences when the
    boundary lands mid-batch — the last batch is truncated to fit).
    """
    subset = _take_first_n_sequences(batches, ptq_subset_size)

    def calibrate_loop(model: nn.Module) -> None:
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                for batch in subset:
                    model(input_ids=batch)
        finally:
            if was_training:
                model.train()

    return calibrate_loop


def _take_first_n_sequences(
    batches: list[torch.Tensor], n: int
) -> list[torch.Tensor]:
    """Return the contiguous-from-index-0 sub-list whose total sequence
    count is exactly ``n`` (or fewer, if ``batches`` runs out first).

    Mid-batch truncation: when the boundary lands inside a batch, the last
    included batch is sliced to fit exactly. Per LLR-0042, modelopt
    receives ≤ n batches.
    """
    if n <= 0:
        return []
    out: list[torch.Tensor] = []
    consumed = 0
    for b in batches:
        bsize = b.shape[0]
        if consumed + bsize <= n:
            out.append(b)
            consumed += bsize
            if consumed == n:
                break
        else:
            need = n - consumed
            if need > 0:
                out.append(b[:need])
            break
    return out
