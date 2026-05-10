"""Tests for sink-token routing analysis."""
import inspect

import numpy as np
import torch

from moe_compress.utils.sink_token_routing import (
    SinkTokenRoutingAccumulator,
    apply_sink_token_extension,
)


def test_aggregator_separates_sink_vs_normal_routing():
    acc = SinkTokenRoutingAccumulator(num_layers=1, num_experts=4, bos_token_id=1)
    # 1 batch, 3 tokens — token 0 is BOS (idx 0 always sink), token 1 is BOS via id=1, token 2 is normal.
    input_ids = torch.tensor([[5, 1, 7]])  # position 0 always sink; position 1 has id=1 == bos; position 2 normal
    # router_scores shape: (batch, seq, num_experts)
    router_scores = torch.tensor([
        [
            [0.0, 0.0, 0.0, 0.9],   # position 0 (sink) → expert 3 dominates
            [0.0, 0.0, 0.0, 0.8],   # position 1 (sink) → expert 3 dominates
            [0.1, 0.2, 0.3, 0.0],   # position 2 (normal) → expert 2 dominates, expert 3 zero
        ]
    ])
    routed_pos = torch.tensor([
        [[3], [3], [2]],   # top-1 per token; expert 3 fired on sink positions; expert 2 on normal
    ])  # shape (batch, seq, top_k=1)
    acc.update(layer_idx=0, input_ids=input_ids, router_scores=router_scores, routed_pos=routed_pos)
    acc.finalize()

    # Expert 3: high score on sinks (~0.85 mean), zero on normal, freq_on_sink = 1.0
    assert acc.mean_router_score_sink[(0, 3)] > 0.5
    assert acc.mean_router_score_normal[(0, 3)] == 0.0
    assert acc.freq_on_sink[(0, 3)] == 1.0


def test_freq_on_sink_is_per_layer_not_global():
    """Regression: prior implementation incremented a single global ``_total_sink_tokens``
    counter on every layer's ``update()`` call, which inflated the denominator by
    num_layers and made ``freq_on_sink`` impossible to reach ``1.0``. With 40
    layers and a true per-layer freq of 1.0, the buggy denominator produced
    0.025 — observed live in the 2026-05-10 H200 run, where the 0.95 threshold
    was effectively unreachable.

    Verify that with two layers each updated once per batch and each having an
    expert that fires on every sink token, both experts report freq_on_sink=1.0
    (not 0.5).
    """
    acc = SinkTokenRoutingAccumulator(num_layers=2, num_experts=3, bos_token_id=1)
    # 1 batch, 3 tokens; positions 0 and 1 are sinks (position 0 always; id=1 == bos)
    input_ids = torch.tensor([[5, 1, 7]])
    # Layer 0: expert 1 fires on every sink token; Layer 1: expert 2 fires on every sink.
    router_scores = torch.tensor([[[0.0, 0.9, 0.1], [0.0, 0.8, 0.2], [0.5, 0.3, 0.2]]])
    routed_layer_0 = torch.tensor([[[1], [1], [0]]])  # expert 1 on sinks, expert 0 on normal
    routed_layer_1 = torch.tensor([[[2], [2], [0]]])  # expert 2 on sinks, expert 0 on normal
    acc.update(layer_idx=0, input_ids=input_ids, router_scores=router_scores, routed_pos=routed_layer_0)
    acc.update(layer_idx=1, input_ids=input_ids, router_scores=router_scores, routed_pos=routed_layer_1)
    acc.finalize()

    # Per-layer normalization: each layer has 2 sink tokens; the firing expert
    # appears on both → freq_on_sink == 1.0 (not 0.5, which would be the symptom
    # of the prior global-denominator bug).
    assert acc.freq_on_sink[(0, 1)] == 1.0
    assert acc.freq_on_sink[(1, 2)] == 1.0
    # And experts that DIDN'T fire on sinks have freq 0
    assert acc.freq_on_sink[(0, 0)] == 0.0
    assert acc.freq_on_sink[(0, 2)] == 0.0
    assert acc.freq_on_sink[(1, 0)] == 0.0
    assert acc.freq_on_sink[(1, 1)] == 0.0


def test_vectorized_matches_loop_reference():
    """The vectorized implementation must produce identical aggregates to a
    naive per-expert reference loop on a synthetic batch. Catches any drift
    in the vectorization (one_hot dim, mask broadcast, dtype).
    """
    torch.manual_seed(0)
    B, T, E, K = 2, 17, 8, 3
    input_ids = torch.randint(0, 100, (B, T))
    input_ids[0, 5] = 1  # plant a BOS sink at non-leading position
    router_scores = torch.rand(B, T, E).softmax(dim=-1)
    _, routed_pos = torch.topk(router_scores, k=K, dim=-1)

    acc = SinkTokenRoutingAccumulator(num_layers=1, num_experts=E, bos_token_id=1)
    acc.update(layer_idx=0, input_ids=input_ids, router_scores=router_scores, routed_pos=routed_pos)
    acc.finalize()

    # Reference: explicit per-expert computation
    sink_mask = torch.zeros(B, T, dtype=torch.bool)
    sink_mask[:, 0] = True
    sink_mask = sink_mask | (input_ids == 1)
    n_sink = int(sink_mask.sum().item())
    n_normal = sink_mask.numel() - n_sink

    for e in range(E):
        ref_sink = float((router_scores[..., e] * sink_mask.float()).sum().item()) / max(n_sink, 1)
        ref_norm = float((router_scores[..., e] * (~sink_mask).float()).sum().item()) / max(n_normal, 1)
        # fires: expert e in routed_pos at any sink position
        fires_e = (routed_pos == e).any(dim=-1)
        ref_freq = float((fires_e & sink_mask).sum().item()) / max(n_sink, 1)
        assert acc.mean_router_score_sink[(0, e)] == np.float32(ref_sink) or \
               abs(acc.mean_router_score_sink[(0, e)] - ref_sink) < 1e-5
        assert abs(acc.mean_router_score_normal[(0, e)] - ref_norm) < 1e-5
        assert abs(acc.freq_on_sink[(0, e)] - ref_freq) < 1e-9


def test_update_pins_sink_mask_to_cpu():
    """Regression: the sink mask must be projected to CPU before indexing.

    In real Stage 1 runs, ``input_ids`` arrives on GPU (the calibration tensor
    is moved to the model's device by ``run_calibration``). Since the
    vectorized update broadcasts the mask against CPU-projected scores, we
    must pin the mask to CPU. Production failure observed on HF Jobs H200
    run 2026-05-10. Verified via static inspection so a future refactor that
    drops the projection will fail this test instead of silently breaking
    the live job.
    """
    src = inspect.getsource(SinkTokenRoutingAccumulator.update)
    assert "_build_sink_mask(input_ids).cpu()" in src, (
        "sink_mask must be projected to CPU inside update() — see Stage 1 "
        "production failure dated 2026-05-10."
    )


def test_extension_picks_only_strict_threshold_violators():
    mean_score_sink = {(0, 3): 0.9, (0, 2): 0.05}
    mean_score_normal = {(0, 3): 0.05, (0, 2): 0.5}
    freq_on_sink = {(0, 3): 1.0, (0, 2): 0.1}
    existing_blacklist = {0: []}
    extension = apply_sink_token_extension(
        mean_score_sink, mean_score_normal, freq_on_sink,
        existing_blacklist,
        score_ratio=5.0,
        freq_threshold=0.95,
    )
    assert extension == {0: [3]}  # expert 2 fails ratio (0.05/0.5 = 0.1) and freq (0.1)
