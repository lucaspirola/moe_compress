"""Model loading, MoE discovery, ExpertMatrixBank, FactoredExperts, compressed-checkpoint I/O.

Design assumptions documented in the plan file at
``~/.claude/plans/using-https-huggingface-co-pirola-moe-co-mutable-galaxy.md``:

- Target model is ``qwen3_5_moe`` (Qwen3.6-35B-A3B). Each decoder layer's
  ``mlp`` is a ``Qwen3_5MoeSparseMoeBlock`` whose ``experts`` sub-module stores
  all routed experts as stacked tensors:

      gate_up_proj : [num_experts, 2 * moe_intermediate_size, hidden_size]
      down_proj    : [num_experts, hidden_size, moe_intermediate_size]

  Axis convention: ``[num_experts, d_out, d_in]`` and dispatch uses
  ``F.linear(x, W[e])`` which does ``x @ W[e].T``.

- The shared expert (``mlp.shared_expert``) is unfused (3 × ``nn.Linear``) and
  is protected from compression.

- There is an ``mtp.layers.0.mlp.experts.*`` MoE block in the safetensors
  checkpoint but it is **not loaded at inference** (transformers strips it via
  ``_keys_to_ignore_on_load_unexpected``). We ignore MTP.

``ExpertMatrixBank`` is the central abstraction the compression stages use.
It presents a clean per-expert-per-matrix view into the underlying stacked
tensors, with sub-slicing so ``gate_proj`` and ``up_proj`` are virtual views
into the first / second halves of ``gate_up_proj`` on the output axis.

``FactoredExperts`` is a drop-in replacement for ``Qwen3_5MoeExperts`` that
stores per-expert rank-k factors and matches the original forward signature
so the rest of the model is untouched.
"""
from __future__ import annotations

import gc
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(
    name_or_path: str,
    *,
    revision: str = "main",
    torch_dtype: str | torch.dtype = "bfloat16",
    device_map: str | dict = "auto",
    attn_implementation: str = "sdpa",
    load_in_4bit: bool = False,
    trust_remote_code: bool = False,
):
    from transformers import AutoConfig, AutoTokenizer

    if isinstance(torch_dtype, str):
        if not hasattr(torch, torch_dtype) or not isinstance(getattr(torch, torch_dtype), torch.dtype):
            raise ValueError(
                f"load_model: invalid torch_dtype {torch_dtype!r}; "
                f"must be a valid torch dtype name like 'float16', 'bfloat16', 'float32'"
            )
        dtype = getattr(torch, torch_dtype)
    else:
        if not isinstance(torch_dtype, torch.dtype):
            raise ValueError(
                f"load_model: torch_dtype must be a str or torch.dtype, "
                f"got {type(torch_dtype).__name__!r}"
            )
        dtype = torch_dtype

    kwargs: dict = {
        "revision": revision,
        "torch_dtype": dtype,
        "device_map": device_map,
        "attn_implementation": attn_implementation,
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        if kwargs["device_map"] == "auto":
            kwargs["device_map"] = {"": 0}

    cfg = AutoConfig.from_pretrained(name_or_path, revision=revision, trust_remote_code=trust_remote_code)
    arches = list(getattr(cfg, "architectures", None) or [])
    auto_cls = _pick_auto_class(arches)
    log.info("Loading %s with %s (dtype=%s, device_map=%s)",
             name_or_path, auto_cls.__name__, dtype, device_map)
    try:
        model = auto_cls.from_pretrained(name_or_path, **kwargs)
    except Exception as exc:                         # noqa: BLE001
        if isinstance(exc, (MemoryError, torch.cuda.OutOfMemoryError)):
            raise
        from transformers import AutoModel, AutoModelForCausalLM
        if auto_cls is AutoModel:
            # N-2: no retry — same class would fail identically.
            raise RuntimeError(
                f"load_model: AutoModel.from_pretrained failed for {name_or_path!r}; "
                f"err={exc!r}"
            ) from exc
        # Prefer CausalLM over bare AutoModel — base model lacks lm_head for NLL.
        retry_cls = AutoModelForCausalLM if auto_cls is not AutoModelForCausalLM else AutoModel
        log.warning("%s.from_pretrained failed (%s); retrying with %s",
                    auto_cls.__name__, exc, retry_cls.__name__)
        try:
            model = retry_cls.from_pretrained(name_or_path, **kwargs)
        except Exception as retry_exc:
            if isinstance(retry_exc, (MemoryError, torch.cuda.OutOfMemoryError)):
                raise
            log.warning("load_model: %s retry also failed for %r: %s", retry_cls.__name__, name_or_path, retry_exc)
            raise RuntimeError(
                f"load_model: both {auto_cls.__name__} and {retry_cls.__name__} failed for {name_or_path!r}; "
                f"first_err={exc!r}; retry_err={retry_exc!r}"
            ) from exc

    tokenizer = AutoTokenizer.from_pretrained(
        name_or_path, revision=revision, trust_remote_code=trust_remote_code
    )
    log.info("Model loaded: %s on devices=%s",
             type(model).__name__, _summarize_device_placement(model))
    return model, tokenizer


def _pick_auto_class(architectures: list[str]) -> type:
    from transformers import AutoModel, AutoModelForCausalLM

    if any("causallm" in a.lower() for a in architectures):
        return AutoModelForCausalLM
    if any(
        a.lower().endswith(("imagetexttotextmodel", "conditionalgenerationmodel",
                            "visionencoderdecodermodel"))
        or "imagetexttotext" in a.lower()
        or "conditionalgeneration" in a.lower()
        or "vision" in a.lower()
        for a in architectures
    ):
        try:
            from transformers import AutoModelForImageTextToText
            return AutoModelForImageTextToText
        except ImportError:
            pass
    return AutoModel


def _summarize_device_placement(model) -> str:
    devs: dict = {}
    for p in model.parameters():
        d = str(p.device)
        devs[d] = devs.get(d, 0) + p.numel()
    sorted_devs = sorted(devs.items(), key=lambda x: -x[1])
    top3 = sorted_devs[:3]
    suffix = f" (top 3 of {len(sorted_devs)})" if len(sorted_devs) > 3 else ""
    summary = ", ".join(f"{d}:{n / 1e9:.1f}B" for d, n in top3)
    return summary + suffix


# ---------------------------------------------------------------------------
# MoE layer discovery
# ---------------------------------------------------------------------------


@dataclass
class MoELayerRef:
    """Pointer into the model for one MoE decoder layer, with bank helpers."""
    layer_idx: int
    layer_module: nn.Module              # the decoder layer itself
    mlp: nn.Module                       # Qwen3_5MoeSparseMoeBlock or compatible
    router: nn.Module                    # mlp.gate — Qwen3_5MoeTopKRouter
    experts_module: nn.Module            # mlp.experts — fused or FactoredExperts
    shared_expert: nn.Module | None      # mlp.shared_expert (unfused, protected)
    layer_type: str                      # "linear_attention" | "full_attention" | "unknown"

    @property
    def num_routed_experts(self) -> int:
        ex = self.experts_module
        if hasattr(ex, "num_experts"):
            return int(ex.num_experts)
        # Fallback: read the first stacked weight's leading dim.
        for name in ("gate_up_proj", "down_proj", "down_proj_U"):
            t = getattr(ex, name, None)
            if isinstance(t, (nn.Parameter, torch.Tensor)):
                return int(t.shape[0])
        raise RuntimeError("Could not determine num_experts on experts module")

    @property
    def top_k(self) -> int:
        v = getattr(self.router, "top_k", None)
        if isinstance(v, int):
            return v
        # fall back to config on mlp
        cfg = getattr(self.mlp, "config", None)
        if cfg is not None and hasattr(cfg, "num_experts_per_tok"):
            return int(cfg.num_experts_per_tok)
        raise RuntimeError(
            f"Cannot determine top_k for layer {self.layer_idx}: "
            "no router.top_k or config.num_experts_per_tok found"
        )


def _find_text_tower(model: nn.Module) -> nn.Module:
    """Return the decoder tower module that owns ``.layers`` (the MoE stack)."""
    candidates: list[nn.Module] = [model]
    for attr in ("model", "language_model", "text_model"):
        sub = getattr(model, attr, None)
        if sub is not None:
            candidates.append(sub)
    if hasattr(model, "model"):
        for attr in ("language_model", "text_model", "decoder"):
            sub = getattr(model.model, attr, None)
            if sub is not None:
                candidates.append(sub)

    seen: set[int] = set()
    for c in candidates:
        if id(c) in seen:
            continue
        seen.add(id(c))
        layers = getattr(c, "layers", None)
        if isinstance(layers, (nn.ModuleList, list)) and len(layers) > 0:
            log.debug("text tower = %s (%d layers)", type(c).__name__, len(c.layers))
            return c
    raise RuntimeError(
        "Could not locate decoder tower (looked for `.layers` under model, "
        "model.model, model.language_model, model.text_model, "
        "model.model.language_model, model.model.text_model, model.model.decoder)."
    )


def _is_moe_layer(layer: nn.Module) -> bool:
    """A layer is MoE when its ``mlp`` has a fused-experts module or the
    legacy ModuleList layout."""
    mlp = getattr(layer, "mlp", None)
    if mlp is None:
        return False
    experts = getattr(mlp, "experts", None)
    if experts is None:
        return False
    # Require a gate/router sub-module so we can build a MoELayerRef safely.
    # Without this guard, iter_moe_layers would crash on mlp.gate access for
    # any module that has .experts but is not a true MoE block.
    if not hasattr(mlp, "gate"):
        return False
    # Fused Qwen3_5MoeExperts: identified by having a `gate_up_proj` param.
    if _is_fused_experts(experts):
        return True
    # Factored replacement we install in Stage 3.
    if isinstance(experts, FactoredExperts):
        return True
    # Legacy ModuleList of per-expert Linear triples.
    if isinstance(experts, nn.ModuleList) and len(experts) > 0:
        return True
    return False


def _is_fused_experts(experts: nn.Module) -> bool:
    return (
        hasattr(experts, "gate_up_proj")
        and isinstance(getattr(experts, "gate_up_proj"), (nn.Parameter, torch.Tensor))
        and hasattr(experts, "down_proj")
        and isinstance(getattr(experts, "down_proj"), (nn.Parameter, torch.Tensor))
    )


def _layer_type_from_config(model: nn.Module, layer_idx: int) -> str:
    cfg = getattr(model, "config", None)
    if cfg is None:
        return "unknown"
    text_cfg = getattr(cfg, "text_config", cfg)
    layer_types = getattr(text_cfg, "layer_types", None)
    if layer_types and 0 <= layer_idx < len(layer_types):
        return str(layer_types[layer_idx])
    return "unknown"


def iter_moe_layers(model: nn.Module) -> Iterator[MoELayerRef]:
    """Yield every decoder layer that contains a routed-MoE block."""
    tower = _find_text_tower(model)
    for idx, layer in enumerate(tower.layers):
        if not _is_moe_layer(layer):
            continue
        mlp = layer.mlp
        yield MoELayerRef(
            layer_idx=idx,
            layer_module=layer,
            mlp=mlp,
            router=mlp.gate,
            experts_module=mlp.experts,
            shared_expert=getattr(mlp, "shared_expert", None),
            layer_type=_layer_type_from_config(model, idx),
        )


def iter_decoder_layers(model: nn.Module) -> Iterator[tuple[int, nn.Module]]:
    """Yield every decoder layer in the text tower as ``(layer_idx, layer_module)``.

    Modeled on :func:`iter_moe_layers` but covers the full transformer stack
    (including non-MoE layers).  Used by Stage 1 Phase A's MA-formation
    detection, which must observe the full residual stream — not just the MoE
    layers — so the growth ratio ``max|H_l| / max|H_{l-1}|`` is computed
    against the immediately preceding decoder layer (spec §4 Phase A).
    """
    tower = _find_text_tower(model)
    for idx, layer in enumerate(tower.layers):
        yield idx, layer


# ---------------------------------------------------------------------------
# ExpertMatrixBank: per-(layer, matrix_name) view into stacked tensors
# ---------------------------------------------------------------------------


# Matrix names we care about across all stages.
MATRIX_NAMES = ("gate_proj", "up_proj", "down_proj")


@dataclass
class ExpertMatrixBank:
    """View into one of the three logical expert matrices for a single layer.

    Storage can be either:
      * The fused ``Qwen3_5MoeExperts`` — in which case gate_proj + up_proj
        both slice ``gate_up_proj`` on its output axis, and down_proj reads
        ``down_proj`` directly.
      * The ``FactoredExperts`` — in which case the bank exposes a
        *composed* view: calling ``get(e)`` returns ``U[e] @ V[e]``, and
        ``set(e, W)`` refactors via SVD at the existing rank.
    """
    layer_idx: int
    matrix_name: str
    experts_module: nn.Module
    # For fused storage:
    stacked_attr: str | None = None         # "gate_up_proj", "down_proj", or None (FactoredExperts)
    row_slice: slice | None = None          # sub-slice of dim 1 (out axis), or None

    def __post_init__(self) -> None:
        # When not on the factored path, stacked_attr must always be set.
        # A None stacked_attr is the forbidden "neither fused nor factored"
        # state — catch it early so callers get a clear error rather than an
        # AttributeError deep inside _stacked(). (row_slice is not tested here;
        # None row_slice is valid for down_proj in the fused path.)
        if self.stacked_attr is None and not isinstance(self.experts_module, FactoredExperts):
            raise ValueError(
                f"stacked_attr must be set for non-FactoredExperts modules "
                f"(layer {self.layer_idx}, matrix {self.matrix_name!r})"
            )

    def is_factored(self) -> bool:
        return isinstance(self.experts_module, FactoredExperts)

    def num_experts(self) -> int:
        if self.is_factored():
            return int(self.experts_module.num_experts)
        if self.stacked_attr is None:
            raise RuntimeError(
                "ExpertMatrixBank.num_experts: stacked_attr is None on non-factored bank — "
                "this should have been caught by __post_init__"
            )
        return int(getattr(self.experts_module, self.stacked_attr).shape[0])

    def _stacked(self) -> torch.Tensor:
        # NOTE: uses getattr each time (late-binding). This is intentional:
        # after ExpertMatrixBank.select() replaces the nn.Parameter via setattr,
        # sibling banks that share the same stacked_attr (e.g. gate_proj and
        # up_proj both pointing at gate_up_proj) will see the updated parameter
        # on their next _stacked() call without needing their own reference update.
        return getattr(self.experts_module, self.stacked_attr)

    def get(self, expert_idx: int) -> torch.Tensor:
        """Return a ``[d_out, d_in]`` tensor for this (layer, expert, matrix).

        For the fused path, returns a view (mutations propagate). For the
        factored path, returns a new materialized tensor (mutations do NOT
        propagate; use set_factors/set_factors_from_weight to update parameters).
        """
        if self.is_factored():
            U, V = self.experts_module.factors(expert_idx, self.matrix_name)
            return U @ V
        n = self.num_experts()
        if not (0 <= expert_idx < n):
            raise ValueError(f"ExpertMatrixBank.get: expert_idx={expert_idx} out of range [0, {n})")
        w = self._stacked()[expert_idx]               # [d_out_stacked, d_in]
        if self.row_slice is not None:
            w = w[self.row_slice]
        return w

    def set(self, expert_idx: int, W: torch.Tensor) -> None:
        """Write ``W`` back into the underlying storage (in-place)."""
        if self.is_factored():
            # Refactor at the current rank. Rarely used once factored.
            self.experts_module.set_factors_from_weight(expert_idx, self.matrix_name, W)
            return
        n = self.num_experts()
        if not (0 <= expert_idx < n):
            raise ValueError(f"ExpertMatrixBank.set: expert_idx={expert_idx} out of range [0, {n})")
        target = self._stacked()[expert_idx]
        if self.row_slice is not None:
            target[self.row_slice].copy_(W.to(dtype=target.dtype, device=target.device))
        else:
            target.copy_(W.to(dtype=target.dtype, device=target.device))

    def select(self, kept_ids: list[int]) -> None:
        """Rewrite the underlying stacked tensor with only the chosen expert
        rows.

        On the first call (when the per-attr sentinel
        ``_last_kept_ids_{stacked_attr}`` is not yet set on the experts
        module), always applies the ``index_select`` slice — even if
        ``len(kept_ids) == stacked.shape[0]``, because a reordering like
        ``[3, 1, 0, 2]`` must still be applied. After the slice,
        ``em._last_kept_ids_{stacked_attr}`` is recorded.

        On subsequent calls (the per-attr sentinel is already set): if the new
        ids match exactly, the call is a no-op (idempotent sibling-bank
        handling for gate_proj / up_proj sharing ``gate_up_proj``). Raises
        ValueError if the same length but different IDs are passed.

        Contract: all banks that share the same stacked tensor (i.e. gate_proj
        and up_proj both pointing at ``gate_up_proj``) MUST be called with the
        identical ``kept_ids`` list. Callers should always iterate all three
        banks with the same kept_ids in the same loop body.

        For factored experts, ``FactoredExperts.select_experts`` maintains its
        own ``_last_kept_ids`` sentinel on ``self`` (not on the
        ``experts_module``).
        """
        if not kept_ids:
            raise ValueError(
                "kept_ids cannot be empty; at least one expert must be selected"
            )
        if len(kept_ids) != len(set(kept_ids)):
            raise ValueError(f"kept_ids contains duplicates: {kept_ids}")
        if self.is_factored():
            # FactoredExperts' own select_experts is also idempotent.
            self.experts_module.select_experts(kept_ids)
            return
        stacked = self._stacked()
        em = self.experts_module
        sentinel_attr = f"_last_kept_ids_{self.stacked_attr}"
        last = getattr(em, sentinel_attr, None)
        if last is not None:
            # Subsequent call: sibling bank or duplicate call.
            if list(last) == list(kept_ids):
                # Exact idempotent repeat — skip (sibling bank already sliced).
                return
            raise ValueError(
                f"ExpertMatrixBank.select called with different kept_ids than before "
                f"(prev_ids={list(last)[:8]}, new_ids={list(kept_ids)[:8]}); "
                f"can only be called once per stacked attr '{self.stacked_attr}'"
            )
        # First call: always apply the slice (handles reordering + pruning).
        # kept_ids is guaranteed non-empty by the not-kept_ids guard above.
        if min(kept_ids) < 0:
            raise ValueError(
                f"select: kept_ids contains negative index {min(kept_ids)}"
            )
        if max(kept_ids) >= stacked.shape[0]:
            raise ValueError(
                f"select: kept_ids contains index {max(kept_ids)} >= num_experts {stacked.shape[0]}"
            )
        idx = torch.as_tensor(kept_ids, device=stacked.device, dtype=torch.long)
        new_stacked = stacked.data.index_select(0, idx).clone()
        setattr(em, self.stacked_attr,
                nn.Parameter(new_stacked, requires_grad=stacked.requires_grad))
        # NOTE: num_experts is updated here, on the first select call for this
        # stacked_attr. This update is not transactional: if a second select
        # call (for a different stacked_attr on the same experts_module) raises
        # due to a sentinel mismatch or kept_ids inconsistency, num_experts will
        # already reflect the first attr's len(kept_ids). In practice the
        # sentinel-keyed validation above catches mismatches before the second
        # select modifies anything, making this safe — but it is not atomic.
        if hasattr(em, "num_experts"):
            em.num_experts = len(kept_ids)
        # Record the kept_ids so sibling banks for the same stacked_attr can
        # verify they match. Keyed on stacked_attr so banks for different attrs
        # (e.g. gate_up_proj vs down_proj) do not interfere with each other.
        setattr(em, sentinel_attr, list(kept_ids))

    def shape(self) -> tuple[int, int]:
        """Return ``(d_out, d_in)`` for a single expert's matrix."""
        if self.is_factored():
            return self.experts_module.matrix_shape(self.matrix_name)
        s = self._stacked().shape
        if self.row_slice is not None:
            if self.row_slice.step not in (None, 1):
                raise ValueError(
                    f"ExpertMatrixBank.shape(): non-unit-step row_slice not supported "
                    f"(got step={self.row_slice.step!r})"
                )
            if self.row_slice.start is None or self.row_slice.stop is None:
                raise ValueError(
                    f"ExpertMatrixBank.shape(): row_slice {self.row_slice!r} has None start or stop; "
                    "only explicit integer slices are supported"
                )
            return (self.row_slice.stop - self.row_slice.start, s[2])
        return (s[1], s[2])


def build_banks(layer_ref: MoELayerRef) -> dict[str, ExpertMatrixBank]:
    """Build the three ExpertMatrixBanks for a single MoE layer."""
    em = layer_ref.experts_module
    if isinstance(em, FactoredExperts):
        return {
            name: ExpertMatrixBank(layer_ref.layer_idx, name, em)
            for name in MATRIX_NAMES
        }

    if not _is_fused_experts(em):
        raise RuntimeError(
            f"Layer {layer_ref.layer_idx}: unexpected experts module type "
            f"{type(em).__name__}. Expected fused Qwen3_5MoeExperts or FactoredExperts."
        )

    # Fused layout: gate_up_proj [N, 2·d_int, d_hid] (gate = first half, up = second half)
    #                down_proj    [N, d_hid, d_int]
    gate_up = em.gate_up_proj
    d_int2 = gate_up.shape[1]
    if d_int2 % 2 != 0:
        raise ValueError(f"gate_up_proj dim-1 must be even, got {d_int2}")
    d_int = d_int2 // 2
    return {
        "gate_proj": ExpertMatrixBank(
            layer_ref.layer_idx, "gate_proj", em,
            stacked_attr="gate_up_proj", row_slice=slice(0, d_int),
        ),
        "up_proj": ExpertMatrixBank(
            layer_ref.layer_idx, "up_proj", em,
            stacked_attr="gate_up_proj", row_slice=slice(d_int, d_int2),
        ),
        "down_proj": ExpertMatrixBank(
            layer_ref.layer_idx, "down_proj", em,
            stacked_attr="down_proj", row_slice=None,
        ),
    }


def iter_routed_experts(
    layer_ref: MoELayerRef,
) -> Iterator[tuple[int, dict[str, ExpertMatrixBank]]]:
    """Legacy iterator-style helper; yields ``(expert_idx, banks)`` per expert.

    Most code paths use :func:`build_banks` plus an explicit expert-index loop
    — this helper is provided for parity with the original ModuleList API.
    """
    banks = build_banks(layer_ref)
    for e in range(layer_ref.num_routed_experts):
        yield e, banks


# ---------------------------------------------------------------------------
# FactoredExperts: swap-in replacement for Qwen3_5MoeExperts with rank-k factors
# ---------------------------------------------------------------------------


class FactoredExperts(nn.Module):
    """Drop-in replacement for ``Qwen3_5MoeExperts`` storing per-matrix banks
    of rank-k factors instead of full weights.

    Storage (per MoE layer):
        gate_proj_U: [N, d_int, k_gate]
        gate_proj_V: [N, k_gate, d_hid]
        up_proj_U:   [N, d_int, k_up]
        up_proj_V:   [N, k_up, d_hid]
        down_proj_U: [N, d_hid, k_down]
        down_proj_V: [N, k_down, d_int]

    Forward matches ``Qwen3_5MoeExperts.forward`` byte-for-byte except that
    each single ``F.linear(x, W[e])`` becomes ``F.linear(F.linear(x, V[e]), U[e])``.
    """

    def __init__(
        self,
        num_experts: int,
        hidden_dim: int,
        intermediate_dim: int,
        ranks: dict[str, int],
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str = "cpu",
        act_fn: str = "silu",
    ):
        super().__init__()
        missing = {"gate_proj", "up_proj", "down_proj"} - set(ranks)
        if missing:
            raise ValueError(
                f"FactoredExperts: ranks dict is missing required keys: {sorted(missing)}"
            )
        for key in ("gate_proj", "up_proj", "down_proj"):
            if not isinstance(ranks[key], int) or ranks[key] <= 0:
                raise ValueError(
                    f"FactoredExperts: ranks[{key!r}]={ranks[key]!r} must be a positive integer"
                )
        self.num_experts = int(num_experts)
        self.hidden_dim = int(hidden_dim)
        self.intermediate_dim = int(intermediate_dim)
        self.ranks = dict(ranks)
        # Per-expert effective rank tracking. Stored slot widths in `ranks`
        # may exceed the effective rank when columns are zero-padded (e.g.
        # AA-SVD `k_eff < k`, EoRA `take_eff < r_per_expert`). The honest
        # parameter count uses `effective_ranks`, summed across experts.
        # Initialized to the slot width (assumes full effective rank);
        # callers update via `set_factors` / `widen_rank`.
        self.effective_ranks: dict[str, list[int]] = {
            n: [int(ranks[n])] * int(num_experts)
            for n in ("gate_proj", "up_proj", "down_proj")
        }
        from transformers.activations import ACT2FN
        self.act_fn = ACT2FN[act_fn]

        kg, ku, kd = int(ranks["gate_proj"]), int(ranks["up_proj"]), int(ranks["down_proj"])
        # Use requires_grad=False by default (frozen) — Stage 5 training targets
        # the router only; Stage 3/4 widen U/V manually.
        def make_param(shape):
            return nn.Parameter(torch.zeros(*shape, dtype=dtype, device=device),
                                requires_grad=False)
        self.gate_proj_U = make_param((num_experts, intermediate_dim, kg))
        self.gate_proj_V = make_param((num_experts, kg, hidden_dim))
        self.up_proj_U   = make_param((num_experts, intermediate_dim, ku))
        self.up_proj_V   = make_param((num_experts, ku, hidden_dim))
        self.down_proj_U = make_param((num_experts, hidden_dim, kd))
        self.down_proj_V = make_param((num_experts, kd, intermediate_dim))

    # ---- Bank integration ------------------------------------------------

    def matrix_shape(self, name: str) -> tuple[int, int]:
        if name in {"gate_proj", "up_proj"}:
            return (self.intermediate_dim, self.hidden_dim)
        if name == "down_proj":
            return (self.hidden_dim, self.intermediate_dim)
        raise KeyError(name)

    def factors(self, expert_idx: int, name: str) -> tuple[torch.Tensor, torch.Tensor]:
        if name not in self.ranks:
            raise KeyError(
                f"factors: unknown projection name {name!r}; expected one of {list(self.ranks)}"
            )
        if not (0 <= expert_idx < self.num_experts):
            raise ValueError(
                f"factors: expert_idx={expert_idx} out of range [0, {self.num_experts})"
            )
        U = getattr(self, f"{name}_U")[expert_idx]
        V = getattr(self, f"{name}_V")[expert_idx]
        return U, V

    def set_factors_from_weight(self, expert_idx: int, name: str, W: torch.Tensor) -> None:
        """Refactor ``W`` via SVD at the current rank and overwrite U, V for this expert.

        Note: When called after ``widen_rank``, ``k`` is taken from
        ``self.ranks[name]`` (the post-widen rank). The SVD is truncated to the
        current rank, not the pre-widen rank.
        """
        if name not in self.ranks:
            raise KeyError(
                f"Unknown projection name {name!r}; expected one of {list(self.ranks)}"
            )
        if not (0 <= expert_idx < self.num_experts):
            raise ValueError(
                f"set_factors_from_weight: expert_idx={expert_idx} out of range [0, {self.num_experts})"
            )
        if W.ndim != 2:
            raise ValueError(
                f"set_factors_from_weight: expected 2-D weight matrix for {name!r}, "
                f"got shape {tuple(W.shape)}"
            )
        k = self.ranks[name]
        m, n = W.shape
        d_out, d_in = self.matrix_shape(name)
        if (m, n) != (d_out, d_in):
            raise ValueError(
                f"set_factors_from_weight: W.shape={tuple(W.shape)} expected ({d_out}, {d_in}) "
                f"for {name!r} — check for transposed weight matrix"
            )
        rank_bound = min(m, n)
        if k > rank_bound:
            raise ValueError(
                f"rank {k} exceeds matrix rank bound {rank_bound} for {name!r} "
                f"(matrix shape {(m, n)})"
            )
        U_param = getattr(self, f"{name}_U")
        V_param = getattr(self, f"{name}_V")
        Wf = W.to(device=U_param.device, dtype=torch.float32)
        U, S, Vh = torch.linalg.svd(Wf, full_matrices=False)
        U_k = (U[:, :k] * S[:k]).to(U_param.dtype)
        V_k = Vh[:k, :].to(V_param.dtype)
        U_param.data[expert_idx].copy_(U_k)
        V_param.data[expert_idx].copy_(V_k)
        self.effective_ranks[name][expert_idx] = k

    def set_factors(
        self, expert_idx: int, name: str, U: torch.Tensor, V: torch.Tensor,
        *, effective_rank: int | None = None,
    ) -> None:
        """Write pre-computed factors U, V for one expert/projection pair.

        ``effective_rank`` records how many columns of U (rows of V) carry
        genuine signal — important for honest parameter counting when callers
        zero-pad to a fixed slot width (e.g. AA-SVD k_eff < k, EoRA
        take_eff < r). When omitted, defaults to ``self.ranks[name]`` (the
        full slot width), so ``effective_ranks`` is always kept up to date.
        """
        if name not in self.ranks:
            raise KeyError(
                f"Unknown projection name {name!r}; expected one of {list(self.ranks)}"
            )
        if not (0 <= expert_idx < self.num_experts):
            raise ValueError(
                f"set_factors: expert_idx={expert_idx} out of range [0, {self.num_experts})"
            )
        if effective_rank is None:
            effective_rank = self.ranks[name]
        if not (0 <= effective_rank <= self.ranks[name]):
            raise ValueError(
                f"set_factors: effective_rank={effective_rank} out of range "
                f"[0, {self.ranks[name]}] for {name!r}"
            )
        U_param = getattr(self, f"{name}_U")
        V_param = getattr(self, f"{name}_V")
        # U_param shape: (num_experts, d_out, k); V_param shape: (num_experts, k, d_in)
        exp_U = (U_param.shape[1], U_param.shape[2])
        exp_V = (V_param.shape[1], V_param.shape[2])
        if tuple(U.shape) != exp_U:
            raise ValueError(
                f"set_factors: U.shape={tuple(U.shape)} expected {exp_U} for {name!r}"
            )
        if tuple(V.shape) != exp_V:
            raise ValueError(
                f"set_factors: V.shape={tuple(V.shape)} expected {exp_V} for {name!r}"
            )
        U_param.data[expert_idx].copy_(U.to(device=U_param.device, dtype=U_param.dtype))
        V_param.data[expert_idx].copy_(V.to(device=V_param.device, dtype=V_param.dtype))
        self.effective_ranks[name][expert_idx] = int(effective_rank)

    def select_experts(self, kept_ids: list[int]) -> None:
        if not kept_ids:
            raise ValueError(
                "kept_ids cannot be empty; at least one expert must be selected"
            )
        if len(kept_ids) != len(set(kept_ids)):
            raise ValueError(f"kept_ids contains duplicates: {kept_ids}")
        # Unconditional idempotency/conflict check — must come before any size
        # comparison so that shrinking calls (len(kept_ids) < num_experts) are
        # also guarded against a second pruning pass re-applying index_select on
        # already-pruned tensors.
        last = getattr(self, "_last_kept_ids", None)
        if last is not None:
            if list(kept_ids) == last:
                return  # idempotent repeat
            raise ValueError(
                f"select_experts called twice with different kept_ids "
                f"(prev={last!r}, new={list(kept_ids)!r})"
            )
        if min(kept_ids) < 0:
            raise ValueError(
                f"select_experts: kept_ids contains negative index {min(kept_ids)}"
            )
        if max(kept_ids) >= self.num_experts:
            raise ValueError(
                f"select_experts: kept_ids contains index {max(kept_ids)} >= num_experts {self.num_experts}"
            )
        idx = torch.as_tensor(kept_ids, device=self.gate_proj_U.device, dtype=torch.long)
        for attr in ("gate_proj_U", "gate_proj_V", "up_proj_U", "up_proj_V",
                     "down_proj_U", "down_proj_V"):
            t = getattr(self, attr)
            new_t = t.data.index_select(0, idx).clone()
            setattr(self, attr, nn.Parameter(new_t, requires_grad=t.requires_grad))
        self.num_experts = len(kept_ids)
        self._last_kept_ids = list(kept_ids)
        for nm in self.effective_ranks:
            self.effective_ranks[nm] = [self.effective_ranks[nm][i] for i in kept_ids]

    def widen_rank(
        self, name: str, U_new: torch.Tensor, V_new: torch.Tensor,
        *, added_effective_per_expert: list[int] | None = None,
    ) -> None:
        """Append ``(U_new, V_new)`` per-expert along the rank dim (Stage 4 EoRA).

        ``U_new``: [N, d_out, r], ``V_new``: [N, r, d_in]. Updates ``ranks[name]``.

        ``added_effective_per_expert``: per-expert true rank of the appended
        correction (≤ r). If None, assumes full r per expert. Used to keep
        `effective_ranks` honest when EoRA's eigh path zero-pads columns.
        """
        if name not in self.ranks:
            raise KeyError(
                f"Unknown projection name {name!r}; expected one of {list(self.ranks)}"
            )
        if added_effective_per_expert is not None and len(added_effective_per_expert) != self.num_experts:
            raise ValueError(
                f"added_effective_per_expert length {len(added_effective_per_expert)} "
                f"!= num_experts {self.num_experts}"
            )
        if U_new.ndim != 3:
            raise ValueError(
                f"widen_rank {name!r}: U_new must be 3-D [N, d_out, r], got {U_new.ndim}D"
            )
        if V_new.ndim != 3:
            raise ValueError(
                f"widen_rank {name!r}: V_new must be 3-D [N, r, d_in], got {V_new.ndim}D"
            )
        if U_new.shape[0] != self.num_experts:
            raise ValueError(
                f"widen_rank U_new: expected {self.num_experts} experts, got {U_new.shape[0]}"
            )
        if V_new.shape[0] != self.num_experts:
            raise ValueError(
                f"widen_rank V_new: expected {self.num_experts} experts, got {V_new.shape[0]}"
            )
        if U_new.shape[-1] != V_new.shape[-2]:
            raise ValueError(
                f"widen_rank rank mismatch: U_new rank dim={U_new.shape[-1]}, "
                f"V_new rank dim={V_new.shape[-2]}"
            )
        cur_U = getattr(self, f"{name}_U")
        cur_V = getattr(self, f"{name}_V")
        if U_new.shape[-2] != cur_U.shape[-2]:
            raise ValueError(
                f"widen_rank {name!r}: U_new d_out {U_new.shape[-2]} != existing d_out {cur_U.shape[-2]}"
            )
        if V_new.shape[-1] != cur_V.shape[-1]:
            raise ValueError(
                f"widen_rank {name!r}: V_new d_in {V_new.shape[-1]} != existing d_in {cur_V.shape[-1]}"
            )
        # Validate added_effective_per_expert BEFORE any tensor mutation — partial
        # mutation on bad input would leave tensors widened but effective_ranks wrong.
        added_r = int(U_new.shape[-1])
        if added_effective_per_expert is None:
            added_effective_per_expert = [added_r] * self.num_experts
        eff_adds = [int(x) for x in added_effective_per_expert]
        for e, eff_add in enumerate(eff_adds):
            if not (0 <= eff_add <= added_r):
                raise ValueError(
                    f"widen_rank {name!r}: added_effective_per_expert[{e}]={eff_add} "
                    f"out of range [0, {added_r}]"
                )
        if _canonical_device(U_new.device) != _canonical_device(cur_U.device):
            raise ValueError(
                f"widen_rank {name!r}: U_new is on {U_new.device}, "
                f"existing U is on {cur_U.device}; move U_new to the correct device first"
            )
        if _canonical_device(V_new.device) != _canonical_device(cur_V.device):
            raise ValueError(
                f"widen_rank {name!r}: V_new is on {V_new.device}, "
                f"existing V is on {cur_V.device}"
            )
        # All validation passed — now mutate tensors and metadata.
        new_U = torch.cat([cur_U.data, U_new.to(cur_U.dtype)], dim=-1).contiguous()
        new_V = torch.cat([cur_V.data, V_new.to(cur_V.dtype)], dim=-2).contiguous()
        setattr(self, f"{name}_U", nn.Parameter(new_U, requires_grad=cur_U.requires_grad))
        setattr(self, f"{name}_V", nn.Parameter(new_V, requires_grad=cur_V.requires_grad))
        self.ranks[name] = int(new_U.shape[-1])
        for e, eff_add in enumerate(eff_adds):
            self.effective_ranks[name][e] = int(self.effective_ranks[name][e]) + eff_add

    # ---- Forward: mirrors Qwen3_5MoeExperts.forward --------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        if top_k_weights.ndim != 2:
            raise ValueError(
                f"top_k_weights must be 2-D [n_tokens, top_k], got shape {top_k_weights.shape}"
            )
        if top_k_index.shape != top_k_weights.shape:
            raise ValueError(
                f"top_k_index and top_k_weights shape mismatch: "
                f"{top_k_index.shape} vs {top_k_weights.shape}"
            )
        if self.gate_proj_V.dtype != hidden_states.dtype:
            raise RuntimeError(
                f"FactoredExperts dtype mismatch: hidden_states={hidden_states.dtype}, "
                f"factors={self.gate_proj_V.dtype}"
            )

        # H-1 / N-1: OOB check (including negative indices) before F.one_hot,
        # which would otherwise raise a cryptic PyTorch error on bad indices.
        if top_k_index.numel() > 0 and (
            top_k_index.min().item() < 0
            or top_k_index.max().item() >= self.num_experts
        ):
            raise RuntimeError(
                f"FactoredExperts.forward: top_k_index contains out-of-range expert index "
                f"(min={top_k_index.min().item()}, max={top_k_index.max().item()}, "
                f"num_experts={self.num_experts})"
            )

        final_hidden_states = torch.zeros_like(hidden_states)
        # bool/int mask: autograd never attaches; no_grad not needed
        expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)
        expert_hit = (expert_mask.sum(dim=(-1, -2)) > 0).nonzero(as_tuple=False)

        if expert_hit.numel() == 0:
            return final_hidden_states

        # gathered is a non-grad leaf (requires_grad=False). The detach() on hidden_states indexing
        # makes the graph break explicit — bmm through gathered never reaches hidden_states or U/V parameters.
        # Expert parameters U/V receive zero gradient through this forward.
        # This is intentional: Stage 2–6 never trains expert parameters through FactoredExperts.forward.
        # If training expert factors, use the unfactored path or rewrite this forward.

        # Collect (expert_id, token_indices, top_k_positions) per active expert.
        expert_data: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for row in expert_hit:
            e = row[0]
            top_k_pos, token_idx = torch.where(expert_mask[e])
            expert_data.append((e, token_idx, top_k_pos))

        # Gather all active-expert tokens into a padded batch for bmm.
        # Shape: [n_active, max_tokens, d_hid]. Zero-initialize so padding rows
        # produce zeros through all 6 bmm calls rather than garbage values.
        n_active = len(expert_data)
        max_tokens = max(len(tok) for _, tok, _ in expert_data)
        gathered = hidden_states.new_zeros(n_active, max_tokens, hidden_states.shape[-1])
        for i, (_, token_idx, _) in enumerate(expert_data):
            gathered[i, :len(token_idx)] = hidden_states[token_idx].detach()

        # gate_proj_V is representative: all factor matrices share the same dtype
        # by construction in __init__ and set_factors (both use the same dtype param).

        # Index factor matrices for all active experts at once.
        hit_ids = expert_hit[:, 0]          # [n_active]
        V_g = self.gate_proj_V[hit_ids]     # [n_active, k_g, d_hid]
        U_g = self.gate_proj_U[hit_ids]     # [n_active, d_int, k_g]
        V_u = self.up_proj_V[hit_ids]       # [n_active, k_u, d_hid]
        U_u = self.up_proj_U[hit_ids]       # [n_active, d_int, k_u]
        V_d = self.down_proj_V[hit_ids]     # [n_active, k_d, d_int]
        U_d = self.down_proj_U[hit_ids]     # [n_active, d_hid, k_d]

        # 6 batched matmuls replace ~6*n_active serial F.linear kernel launches.
        gate  = torch.bmm(torch.bmm(gathered, V_g.transpose(-1, -2)), U_g.transpose(-1, -2))
        up    = torch.bmm(torch.bmm(gathered, V_u.transpose(-1, -2)), U_u.transpose(-1, -2))
        inter = self.act_fn(gate) * up      # [n_active, max_tokens, d_int]
        down  = torch.bmm(torch.bmm(inter,   V_d.transpose(-1, -2)), U_d.transpose(-1, -2))
        # down: [n_active, max_tokens, d_hid]

        # Unpad, apply routing weights, and scatter back with a single index_add_.
        flat_tok = torch.cat([tok for _, tok, _   in expert_data])
        flat_pos = torch.cat([pos for _, _,   pos in expert_data])
        flat_out = torch.cat([down[i, :len(tok)] for i, (_, tok, _) in enumerate(expert_data)], dim=0)
        flat_w   = top_k_weights[flat_tok, flat_pos, None]
        final_hidden_states.index_add_(0, flat_tok, (flat_out * flat_w).to(final_hidden_states.dtype))
        return final_hidden_states


# ---------------------------------------------------------------------------
# Parameter counting helpers (bank-aware)
# ---------------------------------------------------------------------------


def count_parameters(model: nn.Module, *, trainable_only: bool = False) -> int:
    total = 0
    for p in model.parameters():
        if trainable_only and not p.requires_grad:
            continue
        total += p.numel()
    return total


def count_parameters_effective(model: nn.Module) -> int:
    """Effective live parameter count, honoring FactoredExperts effective ranks.

    Spec §9 line 785 (live_param_count): FactoredExperts U/V factors must be
    counted at their per-expert *effective* ranks rather than the padded slot
    width allocated by ``ranks``. Padded zero columns (introduced by AA-SVD's
    ``k_eff < k`` and EoRA's ``widen_rank``) are not real parameters.

    Iteration policy:
      * For each FactoredExperts module, sum
        ``effective_ranks[name][i] * (d_out + d_in)`` across experts and the
        three projections (gate / up / down). The factored experts' own
        ``nn.Parameter`` tensors (gate_proj_U/V, up_proj_U/V, down_proj_U/V)
        are NOT visited via ``parameters(recurse=False)`` because they would
        contribute the padded count.
      * For every other module, sum ``numel()`` of its *direct* parameters
        only (``module.parameters(recurse=False)``) so children are counted
        exactly once when the outer ``modules()`` walk reaches them.

    For a model with no FactoredExperts modules this returns the same value
    as ``count_parameters(model)`` (every parameter belongs to exactly one
    module via the recurse=False direct-children policy).
    """
    total = 0
    seen_param_ids: set[int] = set()
    for module in model.modules():
        if isinstance(module, FactoredExperts):
            # Effective count via per-expert ranks; skip the padded U/V tensors.
            for name in ("gate_proj", "up_proj", "down_proj"):
                d_out, d_in = module.matrix_shape(name)
                eff_per_expert = module.effective_ranks.get(
                    name, [module.ranks[name]] * module.num_experts,
                )
                eff_sum = sum(int(r) for r in eff_per_expert)
                total += (d_out + d_in) * eff_sum
            # Mark the FactoredExperts U/V padded tensors as "already accounted
            # for" so a parent module that happens to reach them via parameter
            # sharing (none expected) wouldn't double-count.
            for p in module.parameters(recurse=False):
                seen_param_ids.add(id(p))
            continue
        # Non-FactoredExperts: sum direct (recurse=False) parameters once.
        for p in module.parameters(recurse=False):
            pid = id(p)
            if pid in seen_param_ids:
                continue
            seen_param_ids.add(pid)
            total += p.numel()
    return total


def count_expert_parameters(model: nn.Module, *, routed_only: bool = True) -> int:
    """Parameters inside the routed-experts banks we plan to compress.

    Walks each MoE layer's ``experts_module`` and counts exactly the stacked
    tensors (or the factor banks for `FactoredExperts`). The shared expert is
    included only when ``routed_only=False``.
    """
    total = 0
    for ref in iter_moe_layers(model):
        ex = ref.experts_module
        if isinstance(ex, FactoredExperts):
            # Use effective ranks (per-expert) so zero-padded columns from
            # AA-SVD's k_eff < k or EoRA's take_eff < r aren't counted as
            # real parameters. Each expert contributes (d_out + d_in) × eff
            # for each of {gate, up, down}.
            for name in ("gate_proj", "up_proj", "down_proj"):
                d_out, d_in = ex.matrix_shape(name)
                eff_per_expert = ex.effective_ranks.get(name, [ex.ranks[name]] * ex.num_experts)
                eff_sum = sum(int(r) for r in eff_per_expert)
                total += (d_out + d_in) * eff_sum
        elif _is_fused_experts(ex):
            total += ex.gate_up_proj.numel() + ex.down_proj.numel()
        else:
            for p in ex.parameters():
                total += p.numel()
        if not routed_only and ref.shared_expert is not None:
            for p in ref.shared_expert.parameters():
                total += p.numel()
    return total


# ---------------------------------------------------------------------------
# Standard (uncompressed) checkpoint save (used by Stages that don't change
# the module shapes). Preserves the old API name.
# ---------------------------------------------------------------------------


def save_checkpoint(model: nn.Module, tokenizer, out_dir: str | Path) -> Path:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    log.info("Saving (uncompressed) checkpoint to %s", out_path)
    model.save_pretrained(out_path, safe_serialization=True)
    if tokenizer is not None:
        tokenizer.save_pretrained(out_path)
    return out_path


# ---------------------------------------------------------------------------
# Compressed-checkpoint save / load
# ---------------------------------------------------------------------------


COMPRESSED_METADATA_FILENAME = "compressed_metadata.json"


def save_compressed_checkpoint(
    model: nn.Module,
    tokenizer,
    out_dir: str | Path,
    *,
    pipeline_stage: str,
    extra_metadata: dict | None = None,
) -> Path:
    """Save a compressed model + sidecar metadata.

    The state_dict is whatever the in-memory model has right now. We extract
    per-layer ``num_experts`` and ``FactoredExperts`` info so a custom loader
    can reconstruct the architecture before ``load_state_dict``.
    """
    if not isinstance(pipeline_stage, str) or not pipeline_stage:
        raise ValueError(
            f"pipeline_stage must be a non-empty string, got {pipeline_stage!r}"
        )
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    log.info("Saving compressed checkpoint to %s (stage=%s)", out_path, pipeline_stage)

    per_layer_num_experts: dict[str, int] = {}
    factored_layers: list[int] = []
    factored_ranks: dict[str, dict[str, int]] = {}
    factored_effective_ranks: dict[str, dict[str, list[int]]] = {}
    for ref in iter_moe_layers(model):
        per_layer_num_experts[str(ref.layer_idx)] = ref.num_routed_experts
        if isinstance(ref.experts_module, FactoredExperts):
            factored_layers.append(ref.layer_idx)
            factored_ranks[str(ref.layer_idx)] = dict(ref.experts_module.ranks)
            factored_effective_ranks[str(ref.layer_idx)] = {
                k: [int(v) for v in vs]
                for k, vs in ref.experts_module.effective_ranks.items()
            }

    metadata = {
        "version": 1,
        "pipeline_stage": pipeline_stage,
        "per_layer_num_experts": per_layer_num_experts,
        "factored_layers": sorted(factored_layers),
        "factored_ranks": factored_ranks,
        # Per-expert effective rank — needed for honest parameter counting on
        # reload. Stored slot widths in `factored_ranks` may exceed effective
        # rank when AA-SVD's k_eff < k or EoRA's take_eff < r.
        "factored_effective_ranks": factored_effective_ranks,
    }
    if extra_metadata is not None:
        metadata["extra"] = extra_metadata

    # Write metadata FIRST so that if the shard write fails, the directory has
    # metadata but no shards (obviously incomplete), rather than shards with no
    # metadata (which looks complete but is broken and unloadable).
    # Use save_json_artifact for atomic write (tmp + os.replace) to prevent
    # partial JSON on crash.
    save_json_artifact(metadata, out_path / COMPRESSED_METADATA_FILENAME)
    log.info("  wrote %s (per_layer_num_experts: %d entries, factored_layers: %d)",
             COMPRESSED_METADATA_FILENAME, len(per_layer_num_experts), len(factored_layers))
    model.save_pretrained(out_path, safe_serialization=True)
    if tokenizer is not None:
        tokenizer.save_pretrained(out_path)
    return out_path


def _canonical_device(d: torch.device) -> torch.device:
    """Normalize a torch.device to its canonical form for comparison.

    ``torch.device("cuda") != torch.device("cuda:0")`` in PyTorch even though
    they refer to the same physical device. Normalize both sides before
    comparing to avoid false-positive device-mismatch errors in the streaming
    load path.
    """
    if d.type == "cuda" and d.index is None:
        if not torch.cuda.is_available():
            return d  # can't normalize without CUDA; return as-is
        return torch.device("cuda", torch.cuda.current_device())
    return d


def _assign_storage(model: nn.Module, key: str, tensor: torch.Tensor) -> None:
    """In-place storage swap for ``model.state_dict()[key] = tensor``.

    nn.Parameter: rebind ``.data`` (drops old storage refcount → freed).
    Buffer / non-parameter: replace via ``setattr(parent_module, leaf, tensor)``
    after dropping the existing buffer registration.

    Why one-at-a-time: building a shard-sized partial dict and calling
    ``load_state_dict(partial, assign=True)`` works correctly but holds the
    entire shard on cuda alongside the still-referenced skeleton tensors,
    blowing past the A100 80 GB ceiling. This helper swaps a single tensor
    so the freed skeleton storage is reclaimed before the next tensor lands.
    """
    parent_path, _, leaf = key.rpartition(".")
    parent = model.get_submodule(parent_path) if parent_path else model
    if not hasattr(parent, leaf):
        raise RuntimeError(
            f"_assign_storage: checkpoint key {key!r} resolves to attribute {leaf!r} "
            f"which does not exist on {type(parent).__name__!r}"
        )
    existing = getattr(parent, leaf)
    if not isinstance(existing, torch.Tensor):
        raise RuntimeError(
            f"_assign_storage: expected a Tensor at {key!r}, "
            f"got {type(existing).__name__!r}"
        )
    # Defense-in-depth: a shape mismatch on .data = will raise inside
    # Tensor.set_ with a cryptic message; a dtype mismatch silently
    # changes the param's dtype and corrupts forward dozens of layers
    # later. Either failure mode would only surface hours into a 5-hour
    # run. Catch them at the swap site with a clear error.
    if existing.shape != tensor.shape:
        raise RuntimeError(
            f"_assign_storage: shape mismatch on {key!r} — skeleton "
            f"{tuple(existing.shape)} vs checkpoint {tuple(tensor.shape)}"
        )
    if existing.dtype != tensor.dtype:
        raise RuntimeError(
            f"_assign_storage: dtype mismatch on {key!r} — skeleton "
            f"{existing.dtype} vs checkpoint {tensor.dtype}. "
            "Ensure the torch_dtype passed to load_compressed_model matches the dtype used when the checkpoint was saved."
        )
    existing_dev = _canonical_device(existing.device)
    incoming_dev = _canonical_device(tensor.device)
    # Allow CPU-skeleton → CUDA-tensor: this is the intended streaming scenario where
    # the Transformers skeleton starts on CPU (from_config default) and tensors are
    # loaded one-at-a-time directly onto the target device, freeing CPU storage as we go.
    # Reject anything else that isn't same-device (e.g. CUDA:0 → CUDA:1, CUDA → CPU).
    if existing_dev != incoming_dev and existing_dev != torch.device("cpu"):
        raise RuntimeError(
            f"_assign_storage: unexpected device on {key!r} — "
            f"existing={existing_dev}, incoming={incoming_dev}. "
            "Expected either matching devices or a CPU-skeleton → target-device stream."
        )
    if isinstance(existing, nn.Parameter):
        # Param: rebind .data so the registered Parameter object stays the
        # same (preserves any tying / hooks). Drop refcount on old storage.
        # Also clear .grad so a stale gradient at the old shape doesn't
        # linger (defense for future paths that might call this on a
        # trained model).
        if existing.grad is not None:
            log.warning(
                "_assign_storage: discarding non-None gradient on %s — "
                "this should not happen outside of training", key
            )
        existing.grad = None
        existing.data = tensor
    elif leaf in getattr(parent, "_buffers", {}):
        # Buffer (persistent or non-persistent). Re-register so the module's
        # _buffers dict points at the new tensor; preserve persistent flag.
        # nn.Module.named_buffers() yields BOTH persistent AND non-persistent
        # buffers (PyTorch 1.10+). The authoritative source for the flag is
        # `_non_persistent_buffers_set`: a buffer name in that set is the
        # non-persistent kind (excluded from state_dict). Earlier code here
        # used named_buffers(recurse=False) and assumed it yielded only
        # persistent buffers — that was incorrect, and silently converted
        # non-persistent buffers into persistent on every storage swap.
        non_persistent = leaf in getattr(parent, "_non_persistent_buffers_set", set())
        parent._buffers.pop(leaf, None)
        parent.register_buffer(leaf, tensor, persistent=not non_persistent)
    else:
        # state_dict() only yields parameters + persistent buffers, so this
        # branch is unreachable for any key in expected_keys. Fail loud
        # rather than silently writing a raw Tensor where a Parameter or
        # registered buffer was expected.
        raise RuntimeError(
            f"_assign_storage: key {key!r} resolves to {type(existing).__name__}, "
            "not nn.Parameter or persistent buffer. Streaming load would "
            "leak a raw tensor — investigate the model skeleton."
        )


def load_compressed_model(
    path: str | Path,
    *,
    device_map: str = "auto",
    torch_dtype: str | torch.dtype = "bfloat16",
    attn_implementation: str = "sdpa",
    allow_missing_keys: bool = False,
    trust_remote_code: bool = True,
):
    """Reconstruct a compressed model from a directory produced by
    :func:`save_compressed_checkpoint`.

    Strategy:
      1. Read ``compressed_metadata.json``.
      2. Use transformers' ``AutoConfig.from_pretrained`` to get the base config.
      3. Build the model with ``from_config`` (not ``from_pretrained``) so no
         weights are loaded yet.
      4. Walk the MoE layers and:
         - Resize the fused experts' stacked tensors to match
           ``per_layer_num_experts[i]``.
         - For ``factored_layers[i]``, swap ``mlp.experts`` with a new
           ``FactoredExperts`` built at the stored ranks.
         - Resize ``ref.router.weight`` (``mlp.gate.weight``) to match the new num_experts.
      5. Stream each tensor one-at-a-time from the safetensors shards via
         ``safe_open``, assigning directly into the skeleton's parameter/buffer
         storage with :func:`_assign_storage` (per-tensor swap avoids the dual-
         residence-on-CUDA peak that a full ``load_state_dict`` would cause).
    """
    from transformers import AutoConfig, AutoTokenizer
    from safetensors import safe_open

    if device_map not in ("auto", "cuda", "cpu"):
        raise ValueError(
            f"load_compressed_model() does not support device_map={device_map!r}; "
            "use 'auto', 'cuda', or 'cpu'"
        )
    if device_map == "auto":
        log.debug(
            "device_map='auto': loading to GPU 0 if CUDA is available, else CPU "
            "(no multi-GPU balancing)"
        )

    path = Path(path)
    try:
        meta = json.loads((path / COMPRESSED_METADATA_FILENAME).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"load_compressed_model: {path!r} does not look like a compressed checkpoint "
            f"— missing {COMPRESSED_METADATA_FILENAME!r}"
        ) from exc
    if meta.get("version", 0) != 1:
        raise RuntimeError(
            f"Checkpoint metadata version {meta.get('version')} is not supported; "
            "expected version 1."
        )
    _REQUIRED_META_KEYS = ("per_layer_num_experts", "factored_layers", "factored_ranks")
    missing_keys = [k for k in _REQUIRED_META_KEYS if k not in meta]
    if missing_keys:
        raise RuntimeError(
            f"load_compressed_model: metadata {path!r} is missing required keys "
            f"{missing_keys!r}; the checkpoint may be corrupt or incomplete"
        )
    cfg = AutoConfig.from_pretrained(path, trust_remote_code=trust_remote_code)
    if isinstance(torch_dtype, str):
        if not hasattr(torch, torch_dtype) or not isinstance(getattr(torch, torch_dtype), torch.dtype):
            raise ValueError(
                f"load_compressed_model: invalid torch_dtype {torch_dtype!r}; "
                f"must be a valid torch dtype name like 'float16', 'bfloat16', 'float32'"
            )
        dtype = getattr(torch, torch_dtype)
    else:
        if not isinstance(torch_dtype, torch.dtype):
            raise ValueError(
                f"load_compressed_model: torch_dtype must be a str or torch.dtype, "
                f"got {type(torch_dtype).__name__!r}"
            )
        dtype = torch_dtype
    # Resolve target device. device_map is used only to distinguish cuda vs cpu:
    # "auto" and "cuda" both stream to CUDA when available, "cpu" forces CPU.
    # Arbitrary device_map dicts are not supported by the streaming load path.
    target_device = (
        torch.device("cuda") if torch.cuda.is_available() and device_map != "cpu"
        else torch.device("cpu")
    )
    if device_map == "cuda" and not torch.cuda.is_available():
        log.warning(
            "device_map='cuda' requested but CUDA is not available; falling back to CPU"
        )

    auto_cls = _pick_auto_class(list(getattr(cfg, "architectures", None) or []))
    log.info("Building skeleton %s from config on CPU (no_init_weights — random init skipped)", auto_cls.__name__)
    # from_config builds on CPU by default. _assign_storage permits the CPU→target_device
    # swap so that each tensor can be loaded directly to target_device and the CPU storage
    # freed before the next tensor is opened — one tensor peak, not a full second copy.
    #
    # Wrap in `no_init_weights()` (transformers' canonical init-skip context manager):
    # for a 35B-A3B model with 40 MoE layers × 256 experts, the default `from_config`
    # spends ~12 min recursively calling `_init_weights` on every submodule to populate
    # parameters with kaiming/normal-distribution random values that are then
    # IMMEDIATELY overwritten by the per-tensor state_dict streaming below. The
    # context manager patches `torch.nn.init.*` to no-ops AND flips the module-level
    # `_init_weights` flag to False, which causes the `if _init_weights:` guard
    # inside `PreTrainedModel.init_weights()` to short-circuit (skipping the
    # `initialize_weights()` random-fill + `tie_weights()` call). The `prune_heads`
    # branch inside `init_weights()` is NOT gated by the flag and still runs —
    # intentional and harmless here. Tensors come out allocated (correct shape +
    # dtype) but with undefined memory contents that are overwritten before any
    # forward/backward reads them.
    # Measured saving on the 2026-05-13 Stage 2.5 relaunch: ~12 min → ~30 s.
    #
    # CRITICAL: `init_weights()` is what calls `tie_weights()` internally (under
    # the `if _init_weights:` guard). Because `no_init_weights()` sets that flag
    # to False, the lm_head ↔ embed_tokens tie is skipped during construction.
    # The compressed checkpoint serializes only ONE side of the tie (whichever
    # shard writer picked), so the per-tensor loader would either raise
    # "missing key" (default `allow_missing_keys=False`) or silently load two
    # independent matrices that then diverge during training. Calling
    # `model.tie_weights()` explicitly after the context manager re-establishes
    # the tie using `_tied_weights_keys` (populated during `__init__`/`post_init`
    # arithmetic, NOT during the `_init_weights` guarded section).
    from transformers.initialization import no_init_weights
    with no_init_weights():
        model = auto_cls.from_config(cfg, torch_dtype=dtype, attn_implementation=attn_implementation)
    model.tie_weights()

    _resize_moe_stack_to_metadata(model, meta, dtype=dtype, device=target_device)

    # Stream each tensor one-at-a-time directly into ``target_device``,
    # swapping the existing param/buffer storage in place. Avoids both
    # the ~70 GB CPU state_dict (the Stage 3 OOM commit a98e6cc fixed)
    # AND the dual-residence-on-CUDA peak that a per-shard partial dict
    # would cause: holding a full shard's tensors (~50 GB) on cuda while
    # the resized MoE skeleton (also ~50 GB) is still referenced briefly
    # exceeds the 80 GB A100. By replacing one storage at a time, the
    # allocator can reclaim each freed skeleton block before the next
    # tensor lands. Peak CUDA = skeleton size + 1 tensor (a few hundred MB).
    shards = sorted(path.glob("model-*.safetensors"))
    if not shards:
        fallback = sorted(path.glob("*.safetensors"))
        if not fallback:
            raise FileNotFoundError(f"No safetensors shards found in {path}")
        if len(fallback) > 1:
            log.warning(
                "No model-*.safetensors found; falling back to all *.safetensors — "
                "verify these are model shards"
            )
        shards = fallback
    log.info("Streaming state_dict from %s (%d shards) → %s",
             path, len(shards), target_device)
    # IMPORTANT: harvest the key set without binding the dict. ``state_dict()``
    # internally calls ``param.detach()`` which is ``shallow_copy_and_detach``
    # — the returned dict's values share storage with the originals and bump
    # each storage's refcount by 1. Keeping ``state`` alive across the shard
    # loop pins every skeleton tensor on cuda, defeating the per-tensor swap
    # below. The set comprehension is reaped immediately after this stmt, so
    # detached aliases drop their extra refcount before the load begins.
    expected_keys = set(model.state_dict().keys())
    loaded_keys: set[str] = set()
    unexpected_all: list[str] = []
    # Track the largest single tensor seen during loading. The streaming swap
    # holds one tensor on the new device alongside the to-be-replaced storage,
    # so peak ≈ skeleton + max_tensor. Logging it after loading makes a future
    # config change that introduces a multi-GB single tensor visible in logs.
    max_bytes = 0
    max_key = ""
    for shard_idx, s in enumerate(shards):
        n_loaded_in_shard = 0
        n_unexpected_in_shard = 0
        with safe_open(str(s), framework="pt", device=str(target_device)) as f:
            for key in f.keys():
                t = f.get_tensor(key)            # already on target_device
                if key not in expected_keys:
                    unexpected_all.append(key)
                    n_unexpected_in_shard += 1
                    del t
                    continue
                # Track largest *loaded* tensor (after unexpected-key skip).
                nbytes = t.numel() * t.element_size()
                if nbytes > max_bytes:
                    max_bytes = nbytes
                    max_key = key
                _assign_storage(model, key, t)
                loaded_keys.add(key)
                n_loaded_in_shard += 1
                del t
        log.info("  shard %d/%d: %s (%d loaded, %d unexpected)",
                 shard_idx + 1, len(shards), s.name, n_loaded_in_shard, n_unexpected_in_shard)
        gc.collect()
        if target_device.type == "cuda":
            torch.cuda.empty_cache()
    log.info("Largest single tensor: %s (%.2f GB)", max_key, max_bytes / 1e9)

    missing_final = sorted(expected_keys - loaded_keys)
    log.info("  load done: missing=%d unexpected=%d",
             len(missing_final), len(unexpected_all))
    if missing_final:
        log.debug("  sample missing: %s", missing_final[:5])
    if unexpected_all:
        log.debug("  sample unexpected: %s", unexpected_all[:5])
    # Fail loud on missing keys. Because the skeleton is built under
    # `no_init_weights()`, a missing param/buffer means the parameter's
    # storage holds **undefined `torch.empty()` memory** (possibly NaN/Inf)
    # — even worse than the previous "random init" state, since downstream
    # NaN propagation is harder to debug than just-degraded numerics. The
    # ``allow_missing_keys`` flag is for tests on partial fixtures only.
    if missing_final and not allow_missing_keys:
        raise RuntimeError(
            f"Streaming load completed with {len(missing_final)} missing key(s). "
            f"Sample: {missing_final[:5]}. The skeleton retains undefined "
            "torch.empty() memory for these (no_init_weights skipped the random "
            "init) — refusing to silently corrupt the model with NaN/Inf. "
            "If this is a deliberate partial load, pass allow_missing_keys=True."
        )

    # Catch any leftovers: non-persistent buffers (e.g. RoPE ``inv_freq`` with
    # ``persistent=False``) are NOT in state_dict / safetensors, so streaming
    # never touches them. ``from_config`` initializes them on CPU; without
    # this move, first forward hits a CUDA/CPU device mismatch (the same bug
    # commit a98e6cc fixed). NB: blanket ``model.to_empty`` would obliterate
    # the streamed weights — instead we surface meta leftovers as a hard
    # error so the operator knows to investigate (we don't expect any with
    # the current ``from_config`` path, but safer to fail loud than silent).
    if target_device.type == "cuda":
        meta_leftovers = [
            n for n, p in model.named_parameters() if p.is_meta
        ] + [
            n for n, b in model.named_buffers() if b.is_meta
        ]
        if meta_leftovers:
            raise RuntimeError(
                f"After streaming load, {len(meta_leftovers)} tensor(s) are "
                f"still on meta (sample: {meta_leftovers[:5]}). The skeleton "
                "was not fully populated by the safetensors shards and a "
                "blanket model.to_empty would silently zero out the streamed "
                "weights. Investigate the missing keys reported above."
            )
        # Non-persistent buffers are not in the state_dict and were not moved by
        # _assign_storage; this call only affects them. Parameters are already on
        # target_device from streaming and would be a no-op — BUT only if device
        # identity is canonical (cuda:0 != cuda in PyTorch's __eq__). Canonicalize
        # first to prevent a spurious full-parameter copy that would double peak VRAM.
        canonical_target = _canonical_device(target_device)
        log.debug("Moving non-persistent buffers to %s", canonical_target)
        model.to(canonical_target)
        torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=trust_remote_code)
    return model, tokenizer, meta


def _resize_moe_stack_to_metadata(
    model: nn.Module, meta: dict,
    *, dtype: torch.dtype, device: torch.device,
) -> None:
    """Pre-load hook: install ``FactoredExperts`` for factored layers, shrink
    fused stacks + router for non-factored-but-pruned layers. All new tensors
    are created on the caller-supplied dtype/device (NOT on meta) so the
    subsequent streaming ``safe_open`` per-tensor assignment via
    :func:`_assign_storage` can replace them with real weights from the
    safetensors shards without hitting meta-tensor or shape-mismatch errors.
    """
    per_layer_num = {int(k): int(v) for k, v in meta["per_layer_num_experts"].items()}
    factored_ids  = set(meta["factored_layers"])
    factored_ranks = {int(k): {kk: int(vv) for kk, vv in v.items()}
                      for k, v in meta["factored_ranks"].items()}
    factored_effective = {
        int(k): {kk: [int(x) for x in vs] for kk, vs in v.items()}
        for k, v in meta.get("factored_effective_ranks", {}).items()
    }

    for ref in iter_moe_layers(model):
        li = ref.layer_idx
        target_n = per_layer_num.get(li)
        if target_n is None:
            log.warning(
                "_resize_moe_stack_to_metadata: layer %d not in per_layer_num_experts metadata; "
                "keeping original size %d", li, ref.num_routed_experts
            )
            target_n = ref.num_routed_experts
        if li in factored_ids:
            if li not in factored_ranks:
                raise RuntimeError(
                    f"Layer {li} is listed in factored_layers but has no entry in "
                    "factored_ranks metadata. The checkpoint may be corrupt or was "
                    "saved with an incompatible version."
                )
            cfg = getattr(ref.mlp, "config", None) or getattr(model.config, "text_config", model.config)
            moe_int = getattr(cfg, "moe_intermediate_size", None)
            if moe_int is None:
                raise RuntimeError(
                    f"config is missing 'moe_intermediate_size' — cannot reconstruct "
                    f"FactoredExperts for layer {li}"
                )
            hidden = getattr(cfg, "hidden_size", None)
            if hidden is None:
                raise RuntimeError(
                    f"config is missing 'hidden_size' — cannot reconstruct FactoredExperts for layer {li}"
                )
            new_fact = FactoredExperts(
                num_experts=target_n,
                hidden_dim=hidden,
                intermediate_dim=moe_int,
                ranks=factored_ranks[li],
                dtype=dtype,
                device=device,
                act_fn=getattr(cfg, "hidden_act", "silu"),
            )
            # Restore effective ranks if the metadata captured them; else
            # fall back to the slot-width default that __init__ already set.
            if li in factored_effective:
                for nm, vs in factored_effective[li].items():
                    if nm not in new_fact.effective_ranks:
                        raise RuntimeError(
                            f"Layer {li}: effective_ranks metadata contains unknown "
                            f"projection {nm!r}; expected one of {list(new_fact.effective_ranks)}"
                        )
                    if len(vs) != new_fact.num_experts:
                        raise RuntimeError(
                            f"Layer {li}: effective_ranks[{nm!r}] has {len(vs)} entries "
                            f"but num_experts is {new_fact.num_experts}. The checkpoint "
                            "metadata is inconsistent."
                        )
                    new_fact.effective_ranks[nm] = list(vs)
            ref.mlp.experts = new_fact
            if hasattr(ref.mlp, "num_experts"):
                ref.mlp.num_experts = target_n
        elif target_n != ref.num_routed_experts:
            em = ref.experts_module
            if not _is_fused_experts(em):
                raise RuntimeError(
                    f"Layer {li}: cannot resize non-fused, non-factored experts module "
                    f"of type {type(em).__name__}"
                )
            gup = em.gate_up_proj
            dp = em.down_proj
            new_gup = nn.Parameter(torch.zeros(target_n, gup.shape[1], gup.shape[2],
                                               dtype=dtype, device=device),
                                   requires_grad=gup.requires_grad)
            new_dp = nn.Parameter(torch.zeros(target_n, dp.shape[1], dp.shape[2],
                                              dtype=dtype, device=device),
                                  requires_grad=dp.requires_grad)
            em.gate_up_proj = new_gup
            em.down_proj = new_dp
            em.num_experts = target_n
            if hasattr(ref.mlp, "num_experts"):
                ref.mlp.num_experts = target_n
            # Clear per-stacked_attr sentinels set by ExpertMatrixBank.select
            # so a subsequent select call treats the resized module as a fresh
            # first call.
            for attr_name in ("_last_kept_ids_gate_up_proj", "_last_kept_ids_down_proj"):
                if hasattr(em, attr_name):
                    delattr(em, attr_name)
            # Also clear legacy bare sentinel (from before the stacked_attr keying fix)
            if hasattr(em, "_last_kept_ids"):
                delattr(em, "_last_kept_ids")
        # Always resize the router to match the target num_experts.
        if ref.router.weight.shape[0] != target_n:
            gw = ref.router.weight
            if gw.ndim != 2:
                raise RuntimeError(
                    f"_resize_moe_stack_to_metadata: router weight at layer {ref.layer_idx} "
                    f"expected 2D [num_experts, hidden_size], got shape {tuple(gw.shape)}"
                )
            new_gw = nn.Parameter(torch.zeros(target_n, gw.shape[1],
                                              dtype=dtype, device=device),
                                  requires_grad=gw.requires_grad)
            ref.router.weight = new_gw
            if hasattr(ref.router, "num_experts"):
                ref.router.num_experts = target_n
        if hasattr(ref.router, "top_k") and ref.router.top_k > target_n:
            log.warning(
                "layer %d: clamping router top_k %d → %d to match pruned expert count",
                ref.layer_idx, ref.router.top_k, target_n,
            )
            ref.router.top_k = target_n


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def save_json_artifact(obj, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, sort_keys=True, default=_json_default)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        # Flush the parent-directory entry so the rename survives power loss.
        # This is step 4 of the §11 durable-write protocol.
        _fsync_dir(path.parent)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return path


def _fsync_dir(directory: Path) -> None:
    """fsync the directory so that a rename into it survives a power cut.

    Per §11 of the spec: after ``os.replace(tmp, path)`` the new directory
    entry must be flushed with ``fsync(parent_dir)`` to make it durable on
    POSIX systems that do not journal directory entries synchronously.
    """
    try:
        fd = os.open(str(directory), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        # Best-effort: some filesystems (e.g. tmpfs, FUSE FUSE mounts used by
        # HF Jobs) raise EINVAL or ENOTSUP on fsync(dir). We log and continue
        # rather than crashing — the rename is already atomic at the OS level;
        # the fsync is only needed for kernel-panic / power-loss durability.
        log.debug("_fsync_dir: fsync(%s) raised OSError (non-durable fs?)", directory)


def load_json_artifact(path: str | Path) -> Any:
    path = Path(path)
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Artifact not found: {path}") from e


def _json_default(o: Any) -> Any:
    # nn.Parameter is a subclass of torch.Tensor, so this guard must come first
    # to prevent silent parameter serialization via the torch.Tensor branch below.
    if isinstance(o, (nn.Module, nn.Parameter)):
        raise TypeError(
            f"Object of type {type(o).__name__} is not JSON serializable — "
            "do not embed nn.Module or nn.Parameter instances in metadata"
        )
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    if hasattr(o, "tolist"):
        return o.tolist()
    if hasattr(o, "__dict__"):
        d = dict(o.__dict__)  # copy — do not return the live __dict__ reference
        if len(d) > 50:  # key-count-gated (gates on dict key count, not byte size)
            log.warning(
                "_json_default: large object serialized via __dict__ (%d keys, type=%s)",
                len(d), type(o).__name__,
            )
        else:
            log.debug(
                "_json_default: falling back to __dict__ for type=%s", type(o).__name__
            )
        return d
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")
