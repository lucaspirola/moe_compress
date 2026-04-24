"""Model loading, checkpointing, and MoE module discovery helpers.

The qwen3_5_moe architecture in transformers 5.3 exposes the text tower under
``model.language_model`` (or ``model.text_model`` on some snapshots) with each
decoder layer at ``...layers[i]``. Each layer has:

- an attention submodule (``self_attn`` for full_attention layers, a linear
  attention submodule for DeltaNet layers — the actual class name varies)
- an MoE submodule (``mlp``) containing:
    - ``mlp.gate``   : router Linear(hidden_size → num_experts)
    - ``mlp.experts``: ModuleList of routed experts; each expert has
                       ``gate_proj``, ``up_proj``, ``down_proj`` Linear layers
    - ``mlp.shared_expert`` (or ``mlp.shared_experts`` on some configs) :
                       a single shared expert of the same shape — MUST NOT
                       be pruned by Stages 0-2 and is iterated separately.

The helpers below are written defensively — they search module trees by
attribute presence rather than hard-coded paths — so they work across the
language_model / text_model naming drift and across transformers version bumps.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


@dataclass
class MoELayerRef:
    """Pointer into the model for one MoE decoder layer."""

    layer_idx: int
    layer_module: nn.Module          # the decoder layer itself
    mlp: nn.Module                   # the MoE block
    router: nn.Module                # typically mlp.gate (Linear)
    experts: nn.ModuleList           # routed experts only
    shared_expert: nn.Module | None  # None if not present
    layer_type: str                  # "linear_attention" | "full_attention"


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
    """Load a HF model ready for compression.

    Kept thin — just wraps ``AutoModelForCausalLM.from_pretrained``. We do NOT
    set ``torch.compile`` or gradient checkpointing here; stages that need
    those toggle them locally.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, torch_dtype) if isinstance(torch_dtype, str) else torch_dtype

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
        # device_map must be explicit with bnb
        if kwargs["device_map"] == "auto":
            kwargs["device_map"] = {"": 0}

    log.info("Loading %s (dtype=%s, device_map=%s)", name_or_path, dtype, device_map)
    model = AutoModelForCausalLM.from_pretrained(name_or_path, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, revision=revision)
    return model, tokenizer


def _find_text_tower(model: nn.Module) -> nn.Module:
    """Walk the model to find the decoder tower that holds `.layers`."""
    # Candidates observed across transformers versions for multimodal MoE:
    for attr in ("language_model", "text_model", "model"):
        sub = getattr(model, attr, None)
        if sub is None:
            continue
        # Unwrap one more level if the inner module is itself a `...Model`
        inner = getattr(sub, "model", sub)
        if hasattr(inner, "layers") and isinstance(inner.layers, (nn.ModuleList, list)):
            return inner
        if hasattr(sub, "layers") and isinstance(sub.layers, (nn.ModuleList, list)):
            return sub
    if hasattr(model, "layers") and isinstance(model.layers, (nn.ModuleList, list)):
        return model
    raise RuntimeError(
        "Could not locate the decoder tower (no `.layers` ModuleList found "
        "under model, model.model, model.language_model, or model.text_model)."
    )


def _is_moe_layer(layer: nn.Module) -> bool:
    mlp = getattr(layer, "mlp", None)
    if mlp is None:
        return False
    return hasattr(mlp, "experts") and isinstance(mlp.experts, (nn.ModuleList, list))


def _get_shared_expert(mlp: nn.Module) -> nn.Module | None:
    for attr in ("shared_expert", "shared_experts", "shared"):
        sub = getattr(mlp, attr, None)
        if sub is not None:
            return sub
    return None


def _get_router(mlp: nn.Module) -> nn.Module:
    for attr in ("gate", "router", "gating"):
        sub = getattr(mlp, attr, None)
        if sub is not None:
            return sub
    raise RuntimeError(f"No router found on {type(mlp).__name__}; tried gate/router/gating")


def _layer_type(model: nn.Module, layer_idx: int) -> str:
    """Read layer_types from config (qwen3_5_moe style)."""
    cfg = getattr(model, "config", None)
    if cfg is None:
        return "unknown"
    text_cfg = getattr(cfg, "text_config", cfg)
    layer_types = getattr(text_cfg, "layer_types", None)
    if layer_types and 0 <= layer_idx < len(layer_types):
        return str(layer_types[layer_idx])
    return "unknown"


def iter_moe_layers(model: nn.Module) -> Iterator[MoELayerRef]:
    """Yield every decoder layer that contains an MoE block."""
    tower = _find_text_tower(model)
    for idx, layer in enumerate(tower.layers):
        if not _is_moe_layer(layer):
            continue
        mlp = layer.mlp
        yield MoELayerRef(
            layer_idx=idx,
            layer_module=layer,
            mlp=mlp,
            router=_get_router(mlp),
            experts=mlp.experts,
            shared_expert=_get_shared_expert(mlp),
            layer_type=_layer_type(model, idx),
        )


def iter_routed_experts(layer_ref: MoELayerRef) -> Iterator[tuple[int, nn.Module]]:
    """Yield ``(expert_idx, expert_module)`` for routed experts only.

    The shared expert is explicitly excluded and must never be reached via
    this iterator.
    """
    for i, expert in enumerate(layer_ref.experts):
        yield i, expert


def get_expert_matrices(expert: nn.Module) -> dict[str, nn.Linear]:
    """Standard three-matrix layout: gate_proj, up_proj, down_proj.

    Returns empty dict if the expert doesn't follow this layout (defensive).
    """
    out: dict[str, nn.Linear] = {}
    for name in ("gate_proj", "up_proj", "down_proj"):
        mod = getattr(expert, name, None)
        if isinstance(mod, nn.Linear):
            out[name] = mod
    return out


def count_parameters(model: nn.Module, *, trainable_only: bool = False) -> int:
    total = 0
    for p in model.parameters():
        if trainable_only and not p.requires_grad:
            continue
        total += p.numel()
    return total


def count_expert_parameters(model: nn.Module, *, routed_only: bool = True) -> int:
    """Parameters inside routed experts (the pool we compress)."""
    total = 0
    for layer_ref in iter_moe_layers(model):
        for _, expert in iter_routed_experts(layer_ref):
            total += sum(p.numel() for p in expert.parameters())
        if not routed_only and layer_ref.shared_expert is not None:
            total += sum(p.numel() for p in layer_ref.shared_expert.parameters())
    return total


def save_checkpoint(model: nn.Module, tokenizer, out_dir: str | Path) -> Path:
    """Save model + tokenizer to ``out_dir`` as HF safetensors."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    log.info("Saving checkpoint to %s", out_path)
    model.save_pretrained(out_path, safe_serialization=True)
    if tokenizer is not None:
        tokenizer.save_pretrained(out_path)
    return out_path


def save_json_artifact(obj, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=_json_default)
    return path


def _json_default(o):
    if isinstance(o, (torch.Tensor,)):
        return o.detach().cpu().tolist()
    if hasattr(o, "tolist"):
        return o.tolist()
    if hasattr(o, "__dict__"):
        return o.__dict__
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def load_json_artifact(path: str | Path):
    with Path(path).open() as f:
        return json.load(f)
