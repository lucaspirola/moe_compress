"""Task 4a — parent materialization + worker reload-fidelity (C1/NEW-C1 blocker).

Stage 6 holds a LIVE in-memory model with NO on-disk student; a spawned worker
has nothing to load. The parent must serialize via ``save_compressed_checkpoint``
(NOT ``model.save_pretrained``, which unpacks FactoredExperts -> "80 missing
keys" on ``load_compressed_model``), and each worker reloads + re-applies the
FULL generative-env contract.

The binding assertion is a STRUCTURAL torch.allclose round-trip over a
stacked/factored-expert REAL model — a ``save_pretrained`` materialization would
fail it, so the test actively guards NEW-C1.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _shard_test_models import (  # noqa: E402
    build_tiny_qwen35_moe,
    install_factored_experts,
    make_tiny_tokenizer,
)

from moe_compress.tools.eval_shard import (  # noqa: E402
    _materialize_student,
    _reload_student_for_worker,
)
from moe_compress.utils.model_io import FactoredExperts, iter_moe_layers  # noqa: E402

_RANKS = {"gate_proj": 3, "up_proj": 2, "down_proj": 4}


def _clear_linear_attention_mapping():
    """Remove the process-global linear_attention mask entry so the reload's
    re-registration is a meaningful (non-vacuous) assertion."""
    from transformers import masking_utils as mu
    mp = getattr(mu, "LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING", None)
    if isinstance(mp, dict):
        mp.pop("linear_attention", None)


def _materialized_dir(tmp_path):
    model = build_tiny_qwen35_moe(seed=3)
    snap = install_factored_experts(model, seed=3)
    tok = make_tiny_tokenizer()
    src = _materialize_student(model, tok)
    return src, snap


def test_materialize_uses_compressed_checkpoint_not_save_pretrained(tmp_path):
    """The temp dir must be a compressed checkpoint (metadata sidecar +
    stacked-form shards), NOT a save_pretrained unpack."""
    src, _ = _materialized_dir(tmp_path)
    src = Path(src)
    assert (src / "compressed_metadata.json").exists(), (
        "no compressed_metadata.json — save_compressed_checkpoint was not used"
    )


def test_reload_structural_round_trip_for_generate(tmp_path):
    _clear_linear_attention_mapping()
    src, snap = _materialized_dir(tmp_path)
    gen_impl = "batched_mm"
    model, tok = _reload_student_for_worker(
        src, experts_impl_generative=gen_impl, for_generate=True,
    )

    # (binding) FactoredExperts survive + params allclose to pre-save.
    refs = list(iter_moe_layers(model))
    assert refs, "no MoE layers after reload"
    for ref in refs:
        fe = ref.experts_module
        assert isinstance(fe, FactoredExperts), (
            f"layer {ref.layer_idx}: experts is {type(fe).__name__}, not "
            "FactoredExperts — save_pretrained unpack would land here"
        )
        assert fe.ranks == _RANKS, fe.ranks
        # effective_ranks reconstructed from compressed_metadata.json
        assert fe.effective_ranks["gate_proj"] == [2, 3, 1, 2], (
            f"effective_ranks lost: {fe.effective_ranks['gate_proj']}"
        )
        for n in ("gate_proj", "up_proj", "down_proj"):
            for s in ("_U", "_V"):
                a = getattr(fe, n + s)
                b = snap[(ref.layer_idx, n + s)]
                assert a.shape == b.shape
                assert not torch.isnan(a).any()
                assert torch.allclose(a, b, atol=1e-6), (
                    f"reload diverged on layer {ref.layer_idx} {n}{s}"
                )

    # generative-env asserts
    assert model.config._attn_implementation == "eager"
    # kernel-patch marker applied (real _apply_stage6_kernel_patches ran)
    marked = [
        m for _, m in model.named_modules()
        if getattr(m, "_stage6_dynamo_disabled", False)
    ]
    assert marked, "no module marked _stage6_dynamo_disabled — kernel patch skipped"
    from transformers import masking_utils as mu
    assert "linear_attention" in mu.LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING
    assert model.config._experts_implementation == gen_impl
    # forward is NOT a torch.compile wrapper (worker never compiles)
    assert "OptimizedModule" not in type(model).__name__
    assert not hasattr(model, "_orig_mod")


def test_reload_ppl_leaves_experts_impl_unswitched(tmp_path):
    _clear_linear_attention_mapping()
    src, snap = _materialized_dir(tmp_path)
    model, tok = _reload_student_for_worker(
        src, experts_impl_generative="batched_mm", for_generate=False,
    )
    # structural round-trip still holds
    for ref in iter_moe_layers(model):
        assert isinstance(ref.experts_module, FactoredExperts)
    # eager + kernel patch + mask passthrough still applied (steps 1-4)
    assert model.config._attn_implementation == "eager"
    assert any(
        getattr(m, "_stage6_dynamo_disabled", False)
        for _, m in model.named_modules()
    )
    from transformers import masking_utils as mu
    assert "linear_attention" in mu.LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING
    # experts impl NOT switched to the generative impl (PPL is forward-only)
    assert getattr(model.config, "_experts_implementation", None) != "batched_mm"
