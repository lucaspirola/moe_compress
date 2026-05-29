"""M-1 (audit/calib-resume-spot-eviction R-5):
``vllm.calibration_block_outputs`` two-segment additivity.

Reference run: N batches through ``_on_block_out`` in one shot.
Segment run: N/2 batches → ``set_n_prompts_accumulated(k)`` →
``dump_block_outputs_checkpoint`` → module reload →
``load_block_outputs_checkpoint`` → remaining N/2 batches →
assert per-rank ``_ACCUM`` list-of-tensors is byte-identical to the
reference (``torch.equal`` — block_outputs is pure cast + concat,
no float math, so strict bitwise equality is required).

Pattern mirrors the moe_compress-side wanda T6 test
(``max_quality/tests/test_stage3_wanda_scalar_row_cache.py:539``) +
the in-patch block_outputs checkpoint round-trip template
(``max_quality/patches/vllm_calibration_hooks.patch:275``).

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


def _reload_bo(env: dict[str, str]):
    """Reload ``vllm.calibration_block_outputs`` (and hooks) with a
    fresh env. Module-level ``_CAPTURE_BLOCK_OUTPUTS`` + ``_SUBSET_SIZE``
    are sampled at import; per-test isolation requires a fresh import.

    Mirrors the in-patch ``_reload_bo`` helper at
    ``max_quality/patches/vllm_calibration_hooks.patch:60-87`` and the
    moe_compress-side ``_reload_wsr`` helper at
    ``max_quality/tests/test_stage3_wanda_scalar_row_cache.py:514-528``.
    """
    sys.modules.pop("vllm.calibration_block_outputs", None)
    sys.modules.pop("vllm.calibration_hooks", None)
    for key in (
        "VLLM_CALIB_CAPTURE_BLOCK_OUTPUTS",
        "VLLM_CALIB_BLOCK_OUTPUTS_SUBSET_SIZE",
        "VLLM_CALIB_CAPTURE_BLOCK",
        "VLLM_CALIB_CAPTURE_OUTPUT_RESERVOIR",
        "VLLM_CALIB_OUTPUT_RESERVOIR_CAP",
        "VLLM_CALIB_CAPTURE_EXPERT_UNWEIGHTED",
        "VLLM_CALIB_CAPTURE_REAP_SCORES",
        "VLLM_CALIB_CAPTURE_ROUTER",
        "VLLM_CALIB_CAPTURE_IMATRIX",
        "VLLM_CALIB_CAPTURE_EXPERT",
        "VLLM_CALIB_CAPTURE_EXPERT_MID",
        "VLLM_CALIB_CAPTURE_INPUT_COV",
        "VLLM_CALIB_CAPTURE_PER_EXPERT_MAX",
    ):
        os.environ.pop(key, None)
    for k, v in env.items():
        os.environ[k] = v
    importlib.import_module("vllm.calibration_hooks")
    return importlib.import_module("vllm.calibration_block_outputs")


def _seed_layer(bo, layer_idx: int, rank: int, n_experts: int = 4) -> None:
    """Manually wire up the per-layer rank table the way ``setup()``
    would in production. Mirrors the in-patch ``_seed_layer`` helper at
    ``vllm_calibration_hooks.patch:90-97``."""
    bo._LAYER_ID_TO_RANK[layer_idx] = rank
    bo._RANK_TO_LAYER_ID[rank] = layer_idx
    if rank + 1 > bo._N_LAYERS:
        bo._N_LAYERS = rank + 1
    if n_experts > bo._N_EXPERTS:
        bo._N_EXPERTS = n_experts


def test_two_segment_additivity_byte_identical(tmp_path):
    """Reference: 4 batches through ``_on_block_out`` in one shot
    (2 layers × 2 dispatches each).
    Segment: 2 batches + checkpoint + module-reload + 2 batches.

    Both paths must yield bitwise-identical per-rank ``_ACCUM`` lists:
    block_outputs is pure ``.to(torch.bfloat16)`` + ``list.append``
    (no float math between input and storage), so ``torch.equal``
    (strict) is the right tolerance — any deviation means the cast or
    concat is non-deterministic, which would itself be a regression.

    Closes M-1 (audit R-5): proves both the per-rank bf16-slab list
    additivity AND that ``set_n_prompts_accumulated`` /
    ``get_n_prompts_accumulated`` round-trip through the checkpoint
    (including ``_SUBSET_CLOSED``, ``_HIDDEN_DIM``, and the rank-table
    bookkeeping).
    """
    pytest.importorskip("vllm.calibration_hooks")

    hidden = 8
    subset_size = 32   # large enough not to trip subset-close mid-test

    # Deterministic input: 4 batches at varying token counts.
    torch.manual_seed(17)
    batches_layer2 = [
        torch.randn(3, hidden, dtype=torch.float32),
        torch.randn(5, hidden, dtype=torch.float32),
    ]
    batches_layer4 = [
        torch.randn(2, hidden, dtype=torch.float32),
        torch.randn(4, hidden, dtype=torch.float32),
    ]

    # ---- Reference: single uninterrupted run --------------------------
    ref = _reload_bo({
        "VLLM_CALIB_CAPTURE_BLOCK_OUTPUTS": "1",
        "VLLM_CALIB_BLOCK_OUTPUTS_SUBSET_SIZE": str(subset_size),
        "VLLM_CALIB_CAPTURE_BLOCK": "1",
    })
    _seed_layer(ref, layer_idx=2, rank=0)
    _seed_layer(ref, layer_idx=4, rank=1)
    # Interleave layer 2 / layer 4 dispatches: 2A → 4A → 2B → 4B.
    ref._on_block_out(layer_idx=2, output=batches_layer2[0])
    ref._on_block_out(layer_idx=4, output=batches_layer4[0])
    ref._on_block_out(layer_idx=2, output=batches_layer2[1])
    ref._on_block_out(layer_idx=4, output=batches_layer4[1])

    expected_rank0 = [t.clone() for t in ref._ACCUM[0]]
    expected_rank1 = [t.clone() for t in ref._ACCUM[1]]
    expected_layer_id_to_rank = dict(ref._LAYER_ID_TO_RANK)
    expected_rank_to_layer_id = dict(ref._RANK_TO_LAYER_ID)
    expected_hidden_dim = ref._HIDDEN_DIM
    expected_subset_closed = ref._SUBSET_CLOSED  # False

    # ---- Segment 1: first 2 dispatches + checkpoint -------------------
    seg = _reload_bo({
        "VLLM_CALIB_CAPTURE_BLOCK_OUTPUTS": "1",
        "VLLM_CALIB_BLOCK_OUTPUTS_SUBSET_SIZE": str(subset_size),
        "VLLM_CALIB_CAPTURE_BLOCK": "1",
    })
    _seed_layer(seg, layer_idx=2, rank=0)
    _seed_layer(seg, layer_idx=4, rank=1)
    seg._on_block_out(layer_idx=2, output=batches_layer2[0])
    seg._on_block_out(layer_idx=4, output=batches_layer4[0])
    seg.set_n_prompts_accumulated(7)
    ckpt = str(tmp_path / "bo.ckpt")
    seg.dump_block_outputs_checkpoint(ckpt)

    # ---- Module reload + load_checkpoint + segment 2 ------------------
    # The reload sets _SUBSET_SIZE from env at import; load_checkpoint
    # cross-checks the ckpt's subset_size and raises if it disagrees.
    seg2 = _reload_bo({
        "VLLM_CALIB_CAPTURE_BLOCK_OUTPUTS": "1",
        "VLLM_CALIB_BLOCK_OUTPUTS_SUBSET_SIZE": str(subset_size),
        "VLLM_CALIB_CAPTURE_BLOCK": "1",
    })
    # Fresh module — no accumulators yet.
    assert seg2._ACCUM == {}
    loaded = seg2.load_block_outputs_checkpoint(ckpt)
    assert loaded == 7
    assert seg2.get_n_prompts_accumulated() == 7

    # Second 2 dispatches.
    seg2._on_block_out(layer_idx=2, output=batches_layer2[1])
    seg2._on_block_out(layer_idx=4, output=batches_layer4[1])

    # ---- Bitwise-identical assertions ---------------------------------
    # Rank 0 (layer 2): 2 entries (batches_layer2[0] from segment 1
    # restored from ckpt, batches_layer2[1] appended in segment 2).
    assert len(seg2._ACCUM[0]) == len(expected_rank0), (
        f"rank-0 length mismatch: got {len(seg2._ACCUM[0])}, "
        f"expected {len(expected_rank0)}"
    )
    for i, (got, want) in enumerate(
        zip(seg2._ACCUM[0], expected_rank0)
    ):
        assert torch.equal(got, want), (
            f"_ACCUM[0][{i}] mismatch (bf16 slab): got shape "
            f"{tuple(got.shape)} dtype {got.dtype}, expected shape "
            f"{tuple(want.shape)} dtype {want.dtype}"
        )

    # Rank 1 (layer 4).
    assert len(seg2._ACCUM[1]) == len(expected_rank1)
    for i, (got, want) in enumerate(
        zip(seg2._ACCUM[1], expected_rank1)
    ):
        assert torch.equal(got, want), (
            f"_ACCUM[1][{i}] mismatch (bf16 slab)"
        )

    # Bookkeeping state.
    assert seg2._LAYER_ID_TO_RANK == expected_layer_id_to_rank
    assert seg2._RANK_TO_LAYER_ID == expected_rank_to_layer_id
    assert seg2._SUBSET_CLOSED == expected_subset_closed
    assert seg2._HIDDEN_DIM == expected_hidden_dim
