"""Shared test helpers for Stage-6 eval-shard tests (Tasks 4a, 8).

Builds a TINY but REAL ``Qwen3_5MoeForCausalLM`` so the
``save_compressed_checkpoint`` -> ``load_compressed_model`` round-trip exercises
the actual unpack/repack path NEW-C1 guards (a pure stub cannot, since the
loader goes through ``AutoConfig.from_pretrained`` + ``_pick_auto_class``). The
model is built with a ``linear_attention`` layer so a ``Qwen3_5MoeGatedDeltaNet``
module exists for the kernel-patch marker assertion. A tiny fast tokenizer
(no sentencepiece) is saved so the loader's ``AutoTokenizer.from_pretrained``
succeeds.
"""
from __future__ import annotations

import torch

from moe_compress.utils.model_io import FactoredExperts, iter_moe_layers

_RANKS = {"gate_proj": 3, "up_proj": 2, "down_proj": 4}
_HIDDEN = 16
_INTERMEDIATE = 8
_NUM_EXPERTS = 4


def build_tiny_qwen35_moe(*, seed: int = 0):
    """Return a tiny real Qwen3_5MoeForCausalLM (float32, eager) with a
    GatedDeltaNet (linear_attention) layer and fused MoE experts."""
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
        Qwen3_5MoeTextConfig,
    )
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeForCausalLM,
    )

    torch.manual_seed(seed)
    cfg = Qwen3_5MoeTextConfig(
        vocab_size=64, hidden_size=_HIDDEN, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        moe_intermediate_size=_INTERMEDIATE, shared_expert_intermediate_size=8,
        num_experts=_NUM_EXPERTS, num_experts_per_tok=2,
        linear_key_head_dim=8, linear_value_head_dim=8, linear_num_key_heads=1,
        linear_num_value_heads=2, linear_conv_kernel_dim=4,
        max_position_embeddings=64,
        layer_types=["linear_attention", "full_attention"],
    )
    cfg.architectures = ["Qwen3_5MoeForCausalLM"]
    cfg._attn_implementation = "eager"
    model = Qwen3_5MoeForCausalLM(cfg).to(torch.float32)
    model.eval()
    return model


def install_factored_experts(model, *, seed: int = 0):
    """Swap each MoE layer's fused experts for a FactoredExperts filled with
    deterministic values + mixed effective ranks. Returns a snapshot dict
    {(layer_idx, attr): tensor} of the installed U/V params (pre-save)."""
    torch.manual_seed(seed)
    for ref in iter_moe_layers(model):
        fe = FactoredExperts(
            num_experts=_NUM_EXPERTS, hidden_dim=_HIDDEN,
            intermediate_dim=_INTERMEDIATE, ranks=_RANKS, dtype=torch.float32,
        )
        for n in ("gate_proj", "up_proj", "down_proj"):
            for s in ("_U", "_V"):
                t = getattr(fe, n + s)
                t.data.copy_(torch.randn_like(t))
        fe.effective_ranks["gate_proj"] = [2, 3, 1, 2]
        ref.mlp.experts = fe
    snap = {}
    for ref in iter_moe_layers(model):
        for n in ("gate_proj", "up_proj", "down_proj"):
            for s in ("_U", "_V"):
                snap[(ref.layer_idx, n + s)] = getattr(
                    ref.experts_module, n + s
                ).detach().clone()
    return snap


def make_tiny_tokenizer():
    """Tiny fast tokenizer that AutoTokenizer can reload (no sentencepiece)."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {"<pad>": 0, "<eos>": 1}
    for i, ch in enumerate("abcdefghijklmnopqrstuvwxyz0123456789 "):
        vocab[ch] = i + 2
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<eos>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=tok, pad_token="<pad>", eos_token="<eos>",
        unk_token="<eos>",
    )
