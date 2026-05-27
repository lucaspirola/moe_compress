"""Partial-hit policy for the Stage 2 profile-pass sidecar (plan §8.2 / §8.3).

A "partial hit" occurs when the sidecar carries materially fewer tokens
for a given layer than the live run expects. The reader MUST classify
this as a partial hit, leaving the live path's fresh accumulators
untouched so :meth:`LayerMergePlugin.on_profile` runs the live forward.

Bug #3 regression coverage (total_tokens_per_layer NOT off by top_k) lives
in ``test_stage2_profile_sidecar_writer_math.py::test_total_tokens_not_off_by_top_k``,
which drives the canonical writer's accumulator end-to-end. Do not add a
top_k tautology here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import torch

from moe_compress.pipeline.context import PipelineContext
from moe_compress.stage2.plugins.stage2_profile_cache import (
    Stage2ProfileCacheProvider,
)
from moe_compress.utils.activation_hooks import (
    InputCovarianceAccumulator,
    ReamCostAccumulator,
)
from moe_compress.utils.cached_calibration_signals import (
    SCHEMA_VERSIONS,
    Stage2ProfilePayloadV3,
)


def _build_payload(
    *, n_layers: int = 2, n_experts: int = 3,
    per_layer_tokens: list[int] | None = None,
    cov_storage_dtype: str = "float16",
) -> Stage2ProfilePayloadV3:
    if per_layer_tokens is None:
        per_layer_tokens = [1000] * n_layers
    gate_logit_profiles: dict[int, list[tuple[int, torch.Tensor]]] = {
        lr: [(0, torch.ones((per_layer_tokens[lr], n_experts), dtype=torch.float32))]
        for lr in range(n_layers)
    }
    cov_dtype = getattr(torch, cov_storage_dtype)
    cov_acc: dict = {
        (lr, e, m): torch.eye(4, dtype=cov_dtype)
        for lr in range(n_layers)
        for e in range(n_experts)
        for m in ("gate_proj", "down_proj")
    }
    cov_token_count: dict = {k: 5 for k in cov_acc}
    neuron_act_sum: dict = {
        (lr, e): torch.zeros((4,), dtype=torch.float32)
        for lr in range(n_layers) for e in range(n_experts)
    }
    neuron_act_count: dict = {k: 7 for k in neuron_act_sum}
    layer_input_reservoir: list = [
        torch.zeros((8, 4), dtype=torch.bfloat16) for _ in range(n_layers)
    ]
    return Stage2ProfilePayloadV3(
        format_version=3,
        schema_version=SCHEMA_VERSIONS["stage2_profile"],
        model_hash="testhash",
        n_layers=n_layers,
        n_experts=n_experts,
        top_k=2,
        cov_storage_dtype=cov_storage_dtype,
        total_tokens_per_layer=torch.tensor(per_layer_tokens, dtype=torch.int64),
        gate_logit_profiles=gate_logit_profiles,
        sim_tensor=torch.zeros(
            (n_layers, n_experts, n_experts), dtype=torch.float64,
        ),
        neuron_act_sum=neuron_act_sum,
        neuron_act_count=neuron_act_count,
        cov_acc=cov_acc,
        cov_token_count=cov_token_count,
        layer_input_reservoir=layer_input_reservoir,
    )


def _build_ctx(*, layer_rank: int, layer_idx: int) -> PipelineContext:
    """Mirror what LayerMergePlugin.on_layer_setup writes to ctx."""
    ctx = PipelineContext()
    ctx.set("_layer_rank", layer_rank)
    ctx.set(
        "layer_ref",
        SimpleNamespace(layer_idx=layer_idx, num_routed_experts=3),
    )
    # Fresh empty accumulators — mirror layer_merge.on_layer_setup.
    ctx.set("ream_acc", ReamCostAccumulator())
    ctx.set("layer_input_acc", None)
    return ctx


def test_partial_hit_does_not_hydrate(tmp_path):
    """When one layer carries far fewer tokens than the others, it's partial."""
    # rank 0 has 100 tokens; rank 1 has 1000 → rank 0 falls below 0.5 × 1000.
    payload = _build_payload(per_layer_tokens=[100, 1000])
    cov_acc = InputCovarianceAccumulator()
    cov_acc.set_storage_dtype(torch.float16)

    provider = Stage2ProfileCacheProvider(
        cov_acc=cov_acc,
        partial_hit_fraction=0.5,
    )
    provider.payload = payload

    ctx = _build_ctx(layer_rank=0, layer_idx=7)
    provider.on_layer_setup(ctx)

    # Partial hit flag set; full hit slot NOT set.
    assert ctx.has("stage2_profile_partial_hit")
    assert ctx.get("stage2_profile_partial_hit") is True
    assert not ctx.has("stage2_profile_full_hit")
    # ream_acc untouched (fresh, empty).
    ream = ctx.get("ream_acc")
    assert ream._total_tokens_by_layer.get(7, 0) == 0
    assert 7 not in ream.gate_logit_profiles
    assert 7 not in ream._sim_tensor
    # cov_acc untouched.
    assert not any(k[0] == 7 for k in cov_acc.covariance)


def test_full_hit_hydrates(tmp_path):
    """When the layer has the maximum token count, it's a full hit and hydrates."""
    payload = _build_payload(per_layer_tokens=[1000, 1000])
    cov_acc = InputCovarianceAccumulator()
    cov_acc.set_storage_dtype(torch.float16)

    provider = Stage2ProfileCacheProvider(
        cov_acc=cov_acc,
        partial_hit_fraction=0.5,
    )
    provider.payload = payload

    ctx = _build_ctx(layer_rank=1, layer_idx=13)
    provider.on_layer_setup(ctx)

    assert ctx.has("stage2_profile_full_hit")
    assert ctx.get("stage2_profile_full_hit") is True
    assert not ctx.has("stage2_profile_partial_hit")
    ream = ctx.get("ream_acc")
    # Hydration happened under layer_idx=13.
    assert ream._total_tokens_by_layer[13] == 1000
    assert 13 in ream.gate_logit_profiles
    assert isinstance(ream.gate_logit_profiles[13], list)
    assert len(ream.gate_logit_profiles[13]) == 1
    off, t = ream.gate_logit_profiles[13][0]
    assert isinstance(off, int)
    assert isinstance(t, torch.Tensor)
    assert t.dtype == torch.float32
    assert 13 in ream._sim_tensor
    # cov_acc hydrated for layer 13's experts.
    assert any(k[0] == 13 for k in cov_acc.covariance)
    for (li, _e, _m), v in cov_acc.covariance.items():
        if li == 13:
            assert v.dtype == torch.float16


def test_zero_tokens_is_partial(tmp_path):
    """layer_tokens == 0 is always a partial hit (degenerate sidecar)."""
    payload = _build_payload(per_layer_tokens=[0, 1000])
    cov_acc = InputCovarianceAccumulator()
    cov_acc.set_storage_dtype(torch.float16)

    provider = Stage2ProfileCacheProvider(
        cov_acc=cov_acc,
        partial_hit_fraction=0.5,
    )
    provider.payload = payload

    ctx = _build_ctx(layer_rank=0, layer_idx=2)
    provider.on_layer_setup(ctx)
    assert ctx.has("stage2_profile_partial_hit")
    assert not ctx.has("stage2_profile_full_hit")


def test_layer_input_acc_hydration_guard(tmp_path):
    """layer_input_acc is None on configs that don't need it (plan §10 N-3 guard).

    Reader MUST NOT crash trying to write .buffer on None.
    """
    payload = _build_payload(per_layer_tokens=[1000, 1000])
    cov_acc = InputCovarianceAccumulator()
    cov_acc.set_storage_dtype(torch.float16)

    provider = Stage2ProfileCacheProvider(
        cov_acc=cov_acc,
        partial_hit_fraction=0.5,
    )
    provider.payload = payload

    ctx = _build_ctx(layer_rank=0, layer_idx=2)
    # ctx.layer_input_acc is None (set by _build_ctx)
    provider.on_layer_setup(ctx)
    # No crash; full hit registered.
    assert ctx.has("stage2_profile_full_hit")


def test_layer_input_acc_hydration_when_present(tmp_path):
    """layer_input_acc hydrated when present (SC strategy path)."""
    payload = _build_payload(per_layer_tokens=[1000, 1000])
    cov_acc = InputCovarianceAccumulator()
    cov_acc.set_storage_dtype(torch.float16)

    provider = Stage2ProfileCacheProvider(
        cov_acc=cov_acc,
        partial_hit_fraction=0.5,
    )
    provider.payload = payload

    ctx = _build_ctx(layer_rank=0, layer_idx=2)
    # Replace None with a real layer_input_acc.
    from moe_compress.stage2.profiling import _LayerInputAccumulator
    layer_input_acc = _LayerInputAccumulator(max_samples=64, seed=0)
    ctx.set("layer_input_acc", layer_input_acc, overwrite=True)

    provider.on_layer_setup(ctx)
    assert ctx.has("stage2_profile_full_hit")
    # buffer hydrated from payload.layer_input_reservoir[0].
    assert layer_input_acc.buffer is not None
    assert layer_input_acc.buffer.dtype == torch.bfloat16
    assert layer_input_acc.buffer.shape[0] == 8
    assert layer_input_acc.seen == 8
