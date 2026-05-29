"""M-1 (audit/calib-resume-spot-eviction R-5):
``vllm.calibration_router_logits_stats`` two-segment additivity.

Reference run: N tokens through ``_on_router`` in one shot.
Segment run: N/2 tokens → ``set_n_prompts_accumulated(k)`` →
``dump_router_logits_stats_checkpoint`` → module reload →
``load_router_logits_stats_checkpoint`` → remaining N/2 tokens →
assert per-rank accumulators (``_SCORE_SINK_SUM``, ``_SCORE_NORMAL_SUM``,
``_FIRE_ON_SINK``, ``_N_SINK_TOKENS``, ``_N_NORMAL_TOKENS``) are byte-
identical to the reference within the writer's float-tolerance budget.

Pattern mirrors the moe_compress-side wanda T6 test
(``max_quality/tests/test_stage3_wanda_scalar_row_cache.py:539``) +
the in-patch reap-scores additivity template
(``max_quality/patches/vllm_calibration_hooks.patch:3974``).

Per [[no-monkey-patches]]: ``sys.modules.pop`` + ``importlib.import_module``
is the in-repo precedent for fresh module reload (NOT ``importlib.reload``,
NOT ``monkeypatch``). Skips cleanly when the patched vLLM wheel isn't
installed via ``pytest.importorskip``.
"""
from __future__ import annotations

import importlib
import os
import sys

import pytest
import torch


def _reload_rls(env: dict[str, str]):
    """Reload ``vllm.calibration_router_logits_stats`` (and hooks) with
    a fresh env. Module-level ``_CAPTURE_ROUTER_LOGITS_STATS`` is sampled
    at import; per-test isolation requires a fresh import.

    Mirrors the in-patch ``_reload_rls`` helper at
    ``max_quality/patches/vllm_calibration_hooks.patch:4195-4218`` and
    the moe_compress-side ``_reload_wsr`` helper at
    ``max_quality/tests/test_stage3_wanda_scalar_row_cache.py:514-528``.
    """
    sys.modules.pop("vllm.calibration_router_logits_stats", None)
    sys.modules.pop("vllm.calibration_hooks", None)
    for key in (
        "VLLM_CALIB_CAPTURE_ROUTER_LOGITS_STATS",
        "VLLM_CALIB_CAPTURE_ROUTING_STATS",
        "VLLM_CALIB_CAPTURE_ROUTER",
        "VLLM_CALIB_CAPTURE_REAP_SCORES",
        "VLLM_CALIB_CAPTURE_EXPERT_UNWEIGHTED",
        "VLLM_CALIB_CAPTURE_PER_EXPERT_MAX",
        "VLLM_CALIB_CAPTURE_IMATRIX",
        "VLLM_CALIB_CAPTURE_EXPERT",
        "VLLM_CALIB_CAPTURE_BLOCK",
        "VLLM_CALIB_CAPTURE_EXPERT_MID",
        "VLLM_CALIB_CAPTURE_INPUT_COV",
    ):
        os.environ.pop(key, None)
    for k, v in env.items():
        os.environ[k] = v
    importlib.import_module("vllm.calibration_hooks")
    return importlib.import_module("vllm.calibration_router_logits_stats")


def _seed_layer(rls, layer_idx: int, rank: int, n_experts: int) -> None:
    """Manually wire up the per-layer accumulators the way ``setup()``
    would in production. Mirrors the in-patch ``_seed_layer`` helper at
    ``vllm_calibration_hooks.patch:4221-4232``."""
    rls._LAYER_ID_TO_RANK[layer_idx] = rank
    if rank + 1 > rls._N_LAYERS:
        rls._N_LAYERS = rank + 1
    if n_experts > rls._N_EXPERTS:
        rls._N_EXPERTS = n_experts
    rls._SCORE_SINK_SUM[rank] = torch.zeros(n_experts, dtype=torch.float32)
    rls._SCORE_NORMAL_SUM[rank] = torch.zeros(n_experts, dtype=torch.float32)
    rls._FIRE_ON_SINK[rank] = torch.zeros(n_experts, dtype=torch.int64)
    rls._N_SINK_TOKENS[rank] = torch.zeros(1, dtype=torch.int64)
    rls._N_NORMAL_TOKENS[rank] = torch.zeros(1, dtype=torch.int64)


def test_two_segment_additivity_byte_identical(tmp_path):
    """Reference: 8 tokens through ``_on_router`` in one shot.
    Segment: 4 tokens + checkpoint + module-reload + 4 tokens.

    Both paths must yield byte-identical per-rank accumulators
    (within ``atol=1e-5`` for the fp32 sum-of-softmax accumulators,
    matching the wanda T6 convention at
    ``test_stage3_wanda_scalar_row_cache.py:629``). The int64 fire-on-
    sink + sink/normal counters are strict-equal (``torch.equal``).

    Closes M-1 (audit R-5): proves both the per-(layer, expert) sum
    additivity AND that ``set_n_prompts_accumulated`` /
    ``get_n_prompts_accumulated`` round-trip through the checkpoint.
    """
    pytest.importorskip("vllm.calibration_hooks")

    n_experts = 3
    n_tokens_per = 4   # 4 tokens per segment, 8 total
    bos_token_id = 7

    # Deterministic input.
    torch.manual_seed(13)
    # Two segments × 4 tokens each. Mix sink+normal positions:
    # tokens 0, 3, 4, 7 are sink (id == bos); tokens 1, 2, 5, 6 are normal.
    input_ids_all = torch.tensor(
        [7, 3, 5, 7, 7, 2, 4, 7], dtype=torch.int64,
    )
    router_logits_all = torch.randn(
        2 * n_tokens_per, n_experts, dtype=torch.float32,
    )
    topk_ids_all = torch.randint(
        0, n_experts, (2 * n_tokens_per, 1), dtype=torch.int64,
    )
    topk_weights_all = torch.ones(
        (2 * n_tokens_per, 1), dtype=torch.float32,
    )

    # ---- Reference: single uninterrupted run --------------------------
    ref = _reload_rls({
        "VLLM_CALIB_CAPTURE_ROUTER_LOGITS_STATS": "1",
        "VLLM_CALIB_CAPTURE_ROUTER": "1",
    })
    ref._BOS_TOKEN_ID = bos_token_id
    _seed_layer(ref, layer_idx=0, rank=0, n_experts=n_experts)
    ref._on_router(
        layer_idx=0,
        router_logits=router_logits_all,
        topk_weights=topk_weights_all,
        topk_ids=topk_ids_all,
        input_ids=input_ids_all,
    )
    expected_sink = ref._SCORE_SINK_SUM[0].clone()
    expected_normal = ref._SCORE_NORMAL_SUM[0].clone()
    expected_fire = ref._FIRE_ON_SINK[0].clone()
    expected_n_sink = ref._N_SINK_TOKENS[0].clone()
    expected_n_normal = ref._N_NORMAL_TOKENS[0].clone()
    expected_layer_id_to_rank = dict(ref._LAYER_ID_TO_RANK)

    # ---- Segment 1 + checkpoint ---------------------------------------
    seg = _reload_rls({
        "VLLM_CALIB_CAPTURE_ROUTER_LOGITS_STATS": "1",
        "VLLM_CALIB_CAPTURE_ROUTER": "1",
    })
    seg._BOS_TOKEN_ID = bos_token_id
    _seed_layer(seg, layer_idx=0, rank=0, n_experts=n_experts)
    # First 4 tokens.
    seg._on_router(
        layer_idx=0,
        router_logits=router_logits_all[:n_tokens_per],
        topk_weights=topk_weights_all[:n_tokens_per],
        topk_ids=topk_ids_all[:n_tokens_per],
        input_ids=input_ids_all[:n_tokens_per],
    )
    seg.set_n_prompts_accumulated(11)
    ckpt = str(tmp_path / "rls.ckpt")
    seg.dump_router_logits_stats_checkpoint(ckpt)

    # ---- Module reload + load_checkpoint + segment 2 ------------------
    seg2 = _reload_rls({
        "VLLM_CALIB_CAPTURE_ROUTER_LOGITS_STATS": "1",
        "VLLM_CALIB_CAPTURE_ROUTER": "1",
    })
    # Fresh module — no accumulators yet.
    assert seg2._SCORE_SINK_SUM == {}
    assert seg2._BOS_TOKEN_ID is None
    loaded = seg2.load_router_logits_stats_checkpoint(ckpt)
    assert loaded == 11
    assert seg2.get_n_prompts_accumulated() == 11
    assert seg2._BOS_TOKEN_ID == bos_token_id

    # Second 4 tokens.
    seg2._on_router(
        layer_idx=0,
        router_logits=router_logits_all[n_tokens_per:],
        topk_weights=topk_weights_all[n_tokens_per:],
        topk_ids=topk_ids_all[n_tokens_per:],
        input_ids=input_ids_all[n_tokens_per:],
    )

    # ---- Byte-identical assertions ------------------------------------
    # fp32 softmax+sum: atol=1e-5 matches the wanda T6 convention at
    # ``test_stage3_wanda_scalar_row_cache.py:629``. The in-patch tests
    # at patch lines 4336-4466 use ``atol=1e-6``; the laxer 1e-5 is the
    # agreed moe_compress-side default (matches wanda T6).
    assert torch.allclose(
        seg2._SCORE_SINK_SUM[0], expected_sink, rtol=0, atol=1e-5,
    ), (
        f"_SCORE_SINK_SUM mismatch: got {seg2._SCORE_SINK_SUM[0]}, "
        f"expected {expected_sink}"
    )
    assert torch.allclose(
        seg2._SCORE_NORMAL_SUM[0], expected_normal, rtol=0, atol=1e-5,
    ), (
        f"_SCORE_NORMAL_SUM mismatch: got {seg2._SCORE_NORMAL_SUM[0]}, "
        f"expected {expected_normal}"
    )
    # int64 counter tensors — strict equality.
    assert torch.equal(seg2._FIRE_ON_SINK[0], expected_fire), (
        f"_FIRE_ON_SINK mismatch: got {seg2._FIRE_ON_SINK[0]}, "
        f"expected {expected_fire}"
    )
    assert torch.equal(seg2._N_SINK_TOKENS[0], expected_n_sink), (
        f"_N_SINK_TOKENS mismatch: got {seg2._N_SINK_TOKENS[0]}, "
        f"expected {expected_n_sink}"
    )
    assert torch.equal(
        seg2._N_NORMAL_TOKENS[0], expected_n_normal,
    ), (
        f"_N_NORMAL_TOKENS mismatch: got {seg2._N_NORMAL_TOKENS[0]}, "
        f"expected {expected_n_normal}"
    )
    # Bookkeeping dicts.
    assert seg2._LAYER_ID_TO_RANK == expected_layer_id_to_rank
