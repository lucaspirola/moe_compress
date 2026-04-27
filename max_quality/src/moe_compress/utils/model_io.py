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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

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

    dtype = getattr(torch, torch_dtype) if isinstance(torch_dtype, str) else torch_dtype

    kwargs: dict = {
        "revision": revision,
        "dtype": dtype,
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
    except Exception as err:                         # noqa: BLE001
        log.warning("%s.from_pretrained failed (%s); retrying with AutoModel",
                    auto_cls.__name__, err)
        from transformers import AutoModel
        model = AutoModel.from_pretrained(name_or_path, **kwargs)

    tokenizer = AutoTokenizer.from_pretrained(name_or_path, revision=revision)
    log.info("Model loaded: %s on devices=%s",
             type(model).__name__, _summarize_device_placement(model))
    return model, tokenizer


def _pick_auto_class(architectures: list[str]):
    from transformers import AutoModel, AutoModelForCausalLM

    joined = " ".join(architectures).lower()
    if "imagetexttotext" in joined or "conditionalgeneration" in joined or "vision" in joined:
        try:
            from transformers import AutoModelForImageTextToText
            return AutoModelForImageTextToText
        except ImportError:
            pass
    if "causallm" in joined:
        return AutoModelForCausalLM
    return AutoModel


def _summarize_device_placement(model) -> str:
    devs: dict = {}
    for p in model.parameters():
        d = str(p.device)
        devs[d] = devs.get(d, 0) + p.numel()
    return ", ".join(
        f"{d}:{n / 1e9:.1f}B" for d, n in sorted(devs.items(), key=lambda x: -x[1])[:3]
    )


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
        for name in ("gate_up_proj", "gate_up_U", "down_proj", "down_proj_U"):
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
        if cfg is not None:
            return int(getattr(cfg, "num_experts_per_tok", 8))
        return 8


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
    stacked_attr: str | None = None         # "gate_up_proj" or "down_proj"
    row_slice: slice | None = None          # sub-slice of dim 1 (out axis), or None

    def is_factored(self) -> bool:
        return isinstance(self.experts_module, FactoredExperts)

    def num_experts(self) -> int:
        if self.is_factored():
            return int(self.experts_module.num_experts)
        return int(getattr(self.experts_module, self.stacked_attr).shape[0])

    def _stacked(self) -> torch.Tensor:
        return getattr(self.experts_module, self.stacked_attr)

    def get(self, expert_idx: int) -> torch.Tensor:
        """Return a ``[d_out, d_in]`` tensor for this (layer, expert, matrix)."""
        if self.is_factored():
            U, V = self.experts_module.factors(expert_idx, self.matrix_name)
            return U @ V
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
        target = self._stacked()[expert_idx]
        if self.row_slice is not None:
            target[self.row_slice].copy_(W.to(dtype=target.dtype, device=target.device))
        else:
            target.copy_(W.to(dtype=target.dtype, device=target.device))

    def select(self, kept_ids: list[int]) -> None:
        """Rewrite the underlying stacked tensor with only the chosen expert
        rows. Idempotent: when multiple banks share the same ``stacked_attr``
        (gate_proj and up_proj share ``gate_up_proj``), calling select on all
        three is safe — the second call becomes a no-op because we detect
        that the stacked tensor has already been sliced to ``len(kept_ids)``.
        """
        if self.is_factored():
            # FactoredExperts' own select_experts is also idempotent.
            self.experts_module.select_experts(kept_ids)
            return
        stacked = self._stacked()
        if stacked.shape[0] == len(kept_ids):
            # Sibling bank already sliced; still update num_experts if the
            # experts module tracks it separately.
            if hasattr(self.experts_module, "num_experts"):
                self.experts_module.num_experts = len(kept_ids)
            return
        idx = torch.as_tensor(kept_ids, device=stacked.device, dtype=torch.long)
        new_stacked = stacked.data.index_select(0, idx).contiguous().clone()
        setattr(self.experts_module, self.stacked_attr,
                nn.Parameter(new_stacked, requires_grad=stacked.requires_grad))
        if hasattr(self.experts_module, "num_experts"):
            self.experts_module.num_experts = len(kept_ids)

    def shape(self) -> tuple[int, int]:
        """Return ``(d_out, d_in)`` for a single expert's matrix."""
        if self.is_factored():
            return self.experts_module.matrix_shape(self.matrix_name)
        s = self._stacked().shape
        if self.row_slice is not None:
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
    ):
        super().__init__()
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
        self.act_fn = ACT2FN["silu"]

        kg, ku, kd = int(ranks["gate_proj"]), int(ranks["up_proj"]), int(ranks["down_proj"])
        # Use requires_grad=False by default (frozen) — Stage 5 training targets
        # the router only; Stage 3/4 widen U/V manually.
        def p(shape):
            return nn.Parameter(torch.empty(*shape, dtype=dtype, device=device),
                                requires_grad=False)
        self.gate_proj_U = p((num_experts, intermediate_dim, kg))
        self.gate_proj_V = p((num_experts, kg, hidden_dim))
        self.up_proj_U   = p((num_experts, intermediate_dim, ku))
        self.up_proj_V   = p((num_experts, ku, hidden_dim))
        self.down_proj_U = p((num_experts, hidden_dim, kd))
        self.down_proj_V = p((num_experts, kd, intermediate_dim))

    # ---- Bank integration ------------------------------------------------

    def matrix_shape(self, name: str) -> tuple[int, int]:
        if name == "gate_proj" or name == "up_proj":
            return (self.intermediate_dim, self.hidden_dim)
        if name == "down_proj":
            return (self.hidden_dim, self.intermediate_dim)
        raise KeyError(name)

    def factors(self, expert_idx: int, name: str) -> tuple[torch.Tensor, torch.Tensor]:
        U = getattr(self, f"{name}_U")[expert_idx]
        V = getattr(self, f"{name}_V")[expert_idx]
        return U, V

    def set_factors_from_weight(self, expert_idx: int, name: str, W: torch.Tensor) -> None:
        """Refactor ``W`` via SVD at the current rank and overwrite U, V for this expert."""
        k = self.ranks[name]
        Wf = W.to(torch.float32)
        U, S, Vh = torch.linalg.svd(Wf, full_matrices=False)
        U_k = (U[:, :k] * S[:k]).to(self.gate_proj_U.dtype)
        V_k = Vh[:k, :].to(self.gate_proj_V.dtype)
        getattr(self, f"{name}_U").data[expert_idx].copy_(U_k)
        getattr(self, f"{name}_V").data[expert_idx].copy_(V_k)

    def set_factors(
        self, expert_idx: int, name: str, U: torch.Tensor, V: torch.Tensor,
        *, effective_rank: int | None = None,
    ) -> None:
        getattr(self, f"{name}_U").data[expert_idx].copy_(U.to(self.gate_proj_U.dtype))
        getattr(self, f"{name}_V").data[expert_idx].copy_(V.to(self.gate_proj_V.dtype))
        if effective_rank is not None:
            self.effective_ranks[name][expert_idx] = int(effective_rank)

    def select_experts(self, kept_ids: list[int]) -> None:
        # Idempotent — skip if already matching.
        if self.num_experts == len(kept_ids):
            return
        idx = torch.as_tensor(kept_ids, device=self.gate_proj_U.device, dtype=torch.long)
        for attr in ("gate_proj_U", "gate_proj_V", "up_proj_U", "up_proj_V",
                     "down_proj_U", "down_proj_V"):
            t = getattr(self, attr)
            new_t = t.data.index_select(0, idx).contiguous().clone()
            setattr(self, attr, nn.Parameter(new_t, requires_grad=t.requires_grad))
        self.num_experts = len(kept_ids)

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
        cur_U = getattr(self, f"{name}_U")
        cur_V = getattr(self, f"{name}_V")
        new_U = torch.cat([cur_U.data, U_new.to(cur_U.dtype)], dim=-1).contiguous()
        new_V = torch.cat([cur_V.data, V_new.to(cur_V.dtype)], dim=-2).contiguous()
        setattr(self, f"{name}_U", nn.Parameter(new_U, requires_grad=False))
        setattr(self, f"{name}_V", nn.Parameter(new_V, requires_grad=False))
        self.ranks[name] = int(new_U.shape[-1])
        added_r = int(U_new.shape[-1])
        if added_effective_per_expert is None:
            added_effective_per_expert = [added_r] * self.num_experts
        for e, eff_add in enumerate(added_effective_per_expert):
            self.effective_ranks[name][e] = int(self.effective_ranks[name][e]) + int(eff_add)

    # ---- Forward: mirrors Qwen3_5MoeExperts.forward --------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = (expert_mask.sum(dim=(-1, -2)) > 0).nonzero()

        for expert_idx in expert_hit:
            e = expert_idx[0]
            if e == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[e])
            sel = hidden_states[token_idx]

            gate = F.linear(F.linear(sel, self.gate_proj_V[e]), self.gate_proj_U[e])
            up   = F.linear(F.linear(sel, self.up_proj_V[e]),   self.up_proj_U[e])
            intermediate = self.act_fn(gate) * up
            down = F.linear(F.linear(intermediate, self.down_proj_V[e]),
                            self.down_proj_U[e])
            down = down * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, down.to(final_hidden_states.dtype))
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
            d_int, d_hid = ex.intermediate_dim, ex.hidden_dim
            for name, (d_out, d_in) in (
                ("gate_proj", (d_int, d_hid)),
                ("up_proj",   (d_int, d_hid)),
                ("down_proj", (d_hid, d_int)),
            ):
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
    if extra_metadata:
        metadata["extra"] = extra_metadata

    model.save_pretrained(out_path, safe_serialization=True)
    if tokenizer is not None:
        tokenizer.save_pretrained(out_path)
    (out_path / COMPRESSED_METADATA_FILENAME).write_text(json.dumps(metadata, indent=2))
    log.info("  wrote %s (per_layer_num_experts: %d entries, factored_layers: %d)",
             COMPRESSED_METADATA_FILENAME, len(per_layer_num_experts), len(factored_layers))
    return out_path


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
    existing = getattr(parent, leaf)
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
            f"{existing.dtype} vs checkpoint {tensor.dtype}"
        )
    if isinstance(existing, nn.Parameter):
        # Param: rebind .data so the registered Parameter object stays the
        # same (preserves any tying / hooks). Drop refcount on old storage.
        # Also clear ._grad so a stale gradient at the old shape doesn't
        # linger (defense for future paths that might call this on a
        # trained model).
        existing._grad = None
        existing.data = tensor
    elif leaf in getattr(parent, "_buffers", {}):
        # Persistent buffer. Re-register it so the module's _buffers dict
        # points at the new tensor; preserve persistent / non-persistent flag.
        persistent = leaf not in getattr(parent, "_non_persistent_buffers_set", set())
        parent._buffers.pop(leaf, None)
        parent.register_buffer(leaf, tensor, persistent=persistent)
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
    device_map: str | dict = "auto",
    torch_dtype: str | torch.dtype = "bfloat16",
    attn_implementation: str = "sdpa",
    allow_missing_keys: bool = False,
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
         - Resize ``mlp.gate.weight`` to match the new num_experts.
      5. ``load_state_dict(strict=False)`` from the saved safetensors shards.
    """
    from transformers import AutoConfig, AutoTokenizer
    from safetensors import safe_open

    path = Path(path)
    meta = json.loads((path / COMPRESSED_METADATA_FILENAME).read_text())
    cfg = AutoConfig.from_pretrained(path)
    dtype = getattr(torch, torch_dtype) if isinstance(torch_dtype, str) else torch_dtype
    # Resolve target device. The previous string-compare ``device_map == "auto"``
    # silently fell through to CPU when the caller passed a dict like
    # ``{"": 0}`` (their explicit GPU pin), defeating the streaming intent
    # and risking a host-RAM OOM on the post-load ``model.to``. If CUDA is
    # available, stream to ``cuda``; otherwise CPU regardless of device_map.
    target_device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    auto_cls = _pick_auto_class(list(getattr(cfg, "architectures", None) or []))
    log.info("Building skeleton %s from config (no weights yet)", auto_cls.__name__)
    model = auto_cls.from_config(cfg, dtype=dtype, attn_implementation=attn_implementation)

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
    shards = sorted(path.glob("model-*.safetensors")) or sorted(path.glob("*.safetensors"))
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
    # Pre-scan the largest single tensor we'll load. The streaming swap
    # holds one tensor on the new device alongside the to-be-replaced
    # storage, so peak ≈ skeleton + max_tensor. Logging it makes a future
    # config change that introduces a multi-GB single tensor visible at
    # load start (when we can still abort) instead of mid-load OOM.
    max_bytes = 0
    max_key = ""
    for s in shards:
        with safe_open(str(s), framework="pt", device="cpu") as f:
            for key in f.keys():
                slice_obj = f.get_slice(key)
                shape = slice_obj.get_shape()
                # element size: bf16/fp16=2, fp32=4, int8=1, etc.
                dtype_str = slice_obj.get_dtype()
                elem = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8,
                        "I8": 1, "U8": 1, "I16": 2, "I32": 4, "I64": 8,
                        "BOOL": 1}.get(dtype_str, 4)
                nbytes = elem
                for d in shape:
                    nbytes *= d
                if nbytes > max_bytes:
                    max_bytes = nbytes
                    max_key = key
    log.info("Largest single tensor: %s (%.2f GB)", max_key, max_bytes / 1e9)
    for k, s in enumerate(shards):
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
                _assign_storage(model, key, t)
                loaded_keys.add(key)
                n_loaded_in_shard += 1
                del t
        log.info("  shard %d/%d: %s (%d loaded, %d unexpected)",
                 k + 1, len(shards), s.name, n_loaded_in_shard, n_unexpected_in_shard)
        gc.collect()
        if target_device.type == "cuda":
            torch.cuda.empty_cache()

    missing_final = sorted(expected_keys - loaded_keys)
    log.info("  load done: missing=%d unexpected=%d",
             len(missing_final), len(unexpected_all))
    if missing_final[:5]:
        log.debug("  sample missing: %s", missing_final[:5])
    if unexpected_all[:5]:
        log.debug("  sample unexpected: %s", unexpected_all[:5])
    # Fail loud on missing keys. A missing param/buffer means the skeleton's
    # default _init_weights value (random init) survives — the model will
    # silently train/run on garbage for that layer and we'd only notice
    # hours later via degraded eval. Better to crash here. The
    # ``allow_missing_keys`` flag is for tests on partial fixtures only.
    if missing_final and not allow_missing_keys:
        raise RuntimeError(
            f"Streaming load completed with {len(missing_final)} missing key(s). "
            f"Sample: {missing_final[:5]}. The skeleton retains random-init "
            "weights for these — refusing to silently corrupt the model. "
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
        model.to(target_device)
        torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(path)
    return model, tokenizer, meta


def _resize_moe_stack_to_metadata(
    model: nn.Module, meta: dict,
    *, dtype: torch.dtype, device: torch.device,
) -> None:
    """Pre-load hook: install ``FactoredExperts`` for factored layers, shrink
    fused stacks + router for non-factored-but-pruned layers. All new tensors
    are created on the caller-supplied dtype/device (NOT on meta) so the
    subsequent ``load_state_dict(assign=True)`` can replace them with real
    weights from the safetensors shards without hitting meta-tensor errors.
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
        target_n = per_layer_num.get(li, ref.num_routed_experts)
        if li in factored_ids:
            cfg = getattr(ref.mlp, "config", None) or getattr(model.config, "text_config", model.config)
            new_fact = FactoredExperts(
                num_experts=target_n,
                hidden_dim=cfg.hidden_size,
                intermediate_dim=cfg.moe_intermediate_size,
                ranks=factored_ranks[li],
                dtype=dtype,
                device=device,
            )
            # Restore effective ranks if the metadata captured them; else
            # fall back to the slot-width default that __init__ already set.
            if li in factored_effective:
                for nm, vs in factored_effective[li].items():
                    if len(vs) == new_fact.num_experts:
                        new_fact.effective_ranks[nm] = list(vs)
            ref.mlp.experts = new_fact
            ref.experts_module = new_fact
            ref.mlp.num_experts = target_n
        elif target_n != ref.num_routed_experts:
            em = ref.experts_module
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
            ref.mlp.num_experts = target_n
        # Always resize the router to match the target num_experts.
        if ref.router.weight.shape[0] != target_n:
            gw = ref.router.weight
            new_gw = nn.Parameter(torch.zeros(target_n, gw.shape[1],
                                              dtype=dtype, device=device),
                                  requires_grad=gw.requires_grad)
            ref.router.weight = new_gw
            ref.router.num_experts = target_n
        if hasattr(ref.router, "top_k") and ref.router.top_k > target_n:
            ref.router.top_k = target_n


# ---------------------------------------------------------------------------
# JSON helpers (unchanged)
# ---------------------------------------------------------------------------


def save_json_artifact(obj, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=_json_default)
    return path


def load_json_artifact(path: str | Path):
    with Path(path).open() as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    if hasattr(o, "tolist"):
        return o.tolist()
    if hasattr(o, "__dict__"):
        return o.__dict__
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")
