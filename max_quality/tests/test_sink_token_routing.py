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


def test_update_pins_sink_mask_to_cpu():
    """Regression: the sink mask must be projected to CPU before indexing.

    In real Stage 1 runs, ``input_ids`` arrives on GPU (the calibration tensor
    is moved to the model's device by ``run_calibration``). ``scores_cpu`` and
    ``routed_cpu`` are then explicitly projected to CPU inside ``update()``.
    If the mask stayed on the GPU, indexing CPU tensors with it would raise:

        RuntimeError: indices should be either on cpu or on the same device
        as the indexed tensor (cpu)

    Production failure observed on HF Jobs H200 run on 2026-05-10 (Phase B
    crash at sink_token_routing.py:64). The fix is a single ``.cpu()`` call
    on the mask. We pin that invariant via static inspection so a future
    refactor that drops the projection will fail this test instead of silently
    breaking the live job.
    """
    src = inspect.getsource(SinkTokenRoutingAccumulator.update)
    assert "_build_sink_mask(input_ids).cpu()" in src, (
        "sink_mask must be projected to CPU inside update() so indexing "
        "CPU-projected scores/routed_pos doesn't raise on cross-device "
        "input_ids — see Stage 1 production failure dated 2026-05-10."
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
