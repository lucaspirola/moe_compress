"""Feature A — persistent-pool DP per-layer profile forward (CPU-simulated).

Live ≥2-GPU validation is DEFERRED (no hardware); every test here is a CPU
in-process replica of the persistent-pool / structural-replay / four-accumulator
reduce subsystem. See ``stage2/profile_dp.py`` docstring + the plan A8 note.

Test order mirrors the plan A7 list:
  9. Cov per-seq pin (H1) — batch-invariance + serial byte-identity.
  1. Shard math (A2).
  2-4. Four-accumulator spill+reduce (A3): REAP, REAM gate_gram, REAM sim/total/neuron, cov.
  6. Structural replay (C1).
  5. E2E equivalence (mocked 2 in-process workers).
  7. Byte-identical default gate.
  8. Reservoir guard at resolution.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conftest import _TinyModel  # noqa: E402

from moe_compress.utils.activation_hooks import (  # noqa: E402
    InputCovarianceAccumulator,
    ReamCostAccumulator,
    ReapAccumulator,
)
from moe_compress.utils.model_io import iter_moe_layers  # noqa: E402


def _new_accs(n_experts):
    reap = ReapAccumulator()
    cov = InputCovarianceAccumulator()
    ream = ReamCostAccumulator(num_experts=n_experts)
    return reap, cov, ream


# ---------------------------------------------------------------------------
# A7.9 — Cov per-sequence pin (H1): batch-invariance + serial byte-identity.
# ---------------------------------------------------------------------------
def test_cov_pin_batch_invariance_and_serial_byte_identical():
    """``_profile_layer`` with ``seq_len`` set routes input/intermediate cov
    through ``update_grouped`` so the finalized Gram is batch-size-invariant; with
    ``seq_len=None`` (the serial default) it is byte-identical to a plain
    ``update`` over the same calibration."""
    from moe_compress.stage2 import profiling

    torch.manual_seed(7)
    n_seq, seq_len = 4, 8

    def _calib():
        torch.manual_seed(123)
        return torch.randint(0, 32, (n_seq, seq_len), dtype=torch.long)

    def _run(model, batches, seq_len_arg):
        lr = list(iter_moe_layers(model))[0]
        reap, cov, ream = _new_accs(lr.num_routed_experts)
        profiling._profile_layer(
            model, lr, batches, reap, cov, ream,
            device=torch.device("cpu"), seq_len=seq_len_arg,
        )
        cov.finalize_layer(lr.layer_idx)
        return {k: v.clone() for k, v in cov.covariance.items()}

    # Batch-invariance (DP premise): seq_len pinned, the finalized Gram is the
    # same whether the 4 sequences are fed as one [4,8] batch or two [2,8] batches.
    calib = _calib()
    m1 = _TinyModel(); m1.load_state_dict(_TinyModel().state_dict())
    torch.manual_seed(0); m_a = _TinyModel()
    torch.manual_seed(0); m_b = _TinyModel()
    one_batch = [calib]                          # single [4, 8] batch
    two_batches = [calib[:2], calib[2:]]         # two [2, 8] batches
    cov_one = _run(m_a, one_batch, seq_len)
    cov_two = _run(m_b, two_batches, seq_len)
    assert cov_one.keys() == cov_two.keys() and cov_one
    for k in cov_one:
        assert torch.allclose(cov_one[k], cov_two[k], atol=1e-5), f"pin not batch-invariant @ {k}"

    # Serial byte-identity: seq_len=None must reproduce the plain-update path
    # exactly (the non-DP golden is untouched).
    torch.manual_seed(0); m_none = _TinyModel()
    torch.manual_seed(0); m_plain = _TinyModel()
    cov_none = _run(m_none, one_batch, None)

    # Plain reference: same forward, cov via plain update (seq_len omitted).
    lr = list(iter_moe_layers(m_plain))[0]
    reap, cov_p, ream = _new_accs(lr.num_routed_experts)
    profiling._profile_layer(
        m_plain, lr, one_batch, reap, cov_p, ream, device=torch.device("cpu"),
    )
    cov_p.finalize_layer(lr.layer_idx)
    cov_plain = {k: v.clone() for k, v in cov_p.covariance.items()}

    assert cov_none.keys() == cov_plain.keys() and cov_none
    for k in cov_none:
        assert torch.equal(cov_none[k], cov_plain[k]), f"seq_len=None not byte-identical @ {k}"


# ---------------------------------------------------------------------------
# A7.1 — Sequence-disjoint shard.
# ---------------------------------------------------------------------------
def test_shard_calib_sequences_disjoint_and_complete():
    from moe_compress.stage2 import profile_dp

    calib = torch.arange(10 * 4).reshape(10, 4)  # 10 sequences
    shards = profile_dp.shard_calib_sequences(calib, replicas=3)
    assert len(shards) == 3
    total = sum(s.size(0) for s in shards)
    assert total == 10, "every sequence covered exactly once"
    # Reassembled rows equal the original (contiguous, disjoint, in order).
    cat = torch.cat(shards, dim=0)
    assert torch.equal(cat, calib)
    # replicas<=1 ⇒ single shard (serial / byte-identical premise).
    assert len(profile_dp.shard_calib_sequences(calib, replicas=1)) == 1


# ---------------------------------------------------------------------------
# A7.2 — Reduce REAP: per-(l,e) sums/counts/freq.
# ---------------------------------------------------------------------------
def test_reduce_reap_sums_counts_and_score(tmp_path):
    from moe_compress.stage2 import profile_dp

    a = ReapAccumulator()
    b = ReapAccumulator()
    # Disjoint contributions for layer 0; score() = Σsum / Σcount.
    a.add_gpu((0, 1), torch.tensor(3.0), n_tokens=2)
    b.add_gpu((0, 1), torch.tensor(5.0), n_tokens=3)
    a.add_gpu((0, 2), torch.tensor(7.0), n_tokens=4)
    a.finalize_layer(0)
    b.finalize_layer(0)

    da, db = tmp_path / "ra", tmp_path / "rb"
    profile_dp._spill_reap_layer(a, 0, da)
    profile_dp._spill_reap_layer(b, 0, db)

    merged = ReapAccumulator()
    profile_dp._reduce_reap_dirs([da, db], 0, into=merged)
    assert merged.sums[(0, 1)] == 8.0
    assert merged.counts[(0, 1)] == 5
    assert merged.freq[(0, 1)] == 5
    assert abs(merged.score(0, 1) - 8.0 / 5.0) < 1e-9
    assert merged.sums[(0, 2)] == 7.0 and merged.counts[(0, 2)] == 4


# ---------------------------------------------------------------------------
# A7.3 — Reduce REAM gate_gram (fp64 bit-exact).
# ---------------------------------------------------------------------------
def test_reduce_ream_gate_gram_bit_exact(tmp_path):
    from moe_compress.stage2 import profile_dp

    E = 4
    a = ReamCostAccumulator(num_experts=E)
    b = ReamCostAccumulator(num_experts=E)
    # Feed disjoint router-logit batches; gate_gram = Σ vᵀv (fp64, order-independent).
    la = torch.randn(6, E, dtype=torch.float64)
    lb = torch.randn(5, E, dtype=torch.float64)
    a.record_router_logits(0, la, 0)
    b.record_router_logits(0, lb, 0)

    da, db = tmp_path / "ma", tmp_path / "mb"
    profile_dp._spill_ream_layer(a, 0, da)
    profile_dp._spill_ream_layer(b, 0, db)

    merged = ReamCostAccumulator(num_experts=E)
    profile_dp._reduce_ream_dirs([da, db], 0, into=merged)

    ref = ReamCostAccumulator(num_experts=E)
    ref.record_router_logits(0, la, 0)
    ref.record_router_logits(0, lb, 0)
    assert torch.equal(merged._gate_gram[0], ref._gate_gram[0]), "gate_gram not bit-exact"


# ---------------------------------------------------------------------------
# A7.4 — Reduce REAM sim/total/neuron.
# ---------------------------------------------------------------------------
def test_reduce_ream_sim_total_neuron(tmp_path):
    from moe_compress.stage2 import profile_dp

    E = 3
    a = ReamCostAccumulator(num_experts=E)
    b = ReamCostAccumulator(num_experts=E)
    # sim numerator (dense fp64), total token int, neuron sum/count.
    a._sim_tensor[0] = torch.randn(E, E, dtype=torch.float64)
    b._sim_tensor[0] = torch.randn(E, E, dtype=torch.float64)
    a._total_tokens_by_layer[0] = 10
    b._total_tokens_by_layer[0] = 7
    a._neuron_act_sum[(0, 1)] = torch.arange(4.0)
    a._neuron_act_count[(0, 1)] = 4
    b._neuron_act_sum[(0, 1)] = torch.ones(4)
    b._neuron_act_count[(0, 1)] = 2

    da, db = tmp_path / "sa", tmp_path / "sb"
    profile_dp._spill_ream_layer(a, 0, da)
    profile_dp._spill_ream_layer(b, 0, db)

    merged = ReamCostAccumulator(num_experts=E)
    profile_dp._reduce_ream_dirs([da, db], 0, into=merged)

    assert torch.equal(merged._sim_tensor[0], a._sim_tensor[0] + b._sim_tensor[0])
    assert merged._total_tokens_by_layer[0] == 17
    assert torch.equal(merged._neuron_act_sum[(0, 1)], torch.arange(4.0) + torch.ones(4))
    assert merged._neuron_act_count[(0, 1)] == 6
    # get_neuron_mean == single-pass mean.
    expected_mean = (torch.arange(4.0) + torch.ones(4)) / 6
    assert torch.allclose(merged.get_neuron_mean(0, 1), expected_mean)


# ---------------------------------------------------------------------------
# A7.6 — Structural replay (C1): worker reproduces parent's merged layer.
# ---------------------------------------------------------------------------
def test_structural_replay_matches_parent_select_and_resize():
    from moe_compress.stage2 import profile_dp
    from moe_compress.stage2.merging import _resize_router_for_kept_experts
    from moe_compress.utils.model_io import build_banks

    # Parent + worker start from the SAME model state.
    torch.manual_seed(0); parent = _TinyModel()
    torch.manual_seed(0); worker = _TinyModel()

    final_kept_ids = [0, 2]  # drop experts 1, 3

    p_lr = list(iter_moe_layers(parent))[0]
    w_lr = list(iter_moe_layers(worker))[0]

    # Parent runs the real structural surgery: bank.select + router resize.
    p_banks = build_banks(p_lr)
    for bank in p_banks.values():
        bank.select(final_kept_ids)
    _resize_router_for_kept_experts(p_lr, final_kept_ids)

    # Capture the parent's merged layer + replay it on the worker copy.
    payload = profile_dp.capture_merged_layer(p_lr, final_kept_ids)
    profile_dp.replay_merged_layer(w_lr, payload)

    # Shapes + counts match.
    assert w_lr.experts_module.gate_up_proj.shape == p_lr.experts_module.gate_up_proj.shape
    assert w_lr.experts_module.down_proj.shape == p_lr.experts_module.down_proj.shape
    assert int(w_lr.experts_module.num_experts) == int(p_lr.experts_module.num_experts)
    assert int(w_lr.router.weight.shape[0]) == int(p_lr.router.weight.shape[0])
    assert int(getattr(w_lr.router, "num_experts")) == int(getattr(p_lr.router, "num_experts"))
    assert int(getattr(w_lr.router, "top_k")) == int(getattr(p_lr.router, "top_k"))
    assert int(w_lr.mlp.num_experts) == int(p_lr.mlp.num_experts)
    # Values match exactly.
    assert torch.equal(w_lr.experts_module.gate_up_proj, p_lr.experts_module.gate_up_proj)
    assert torch.equal(w_lr.experts_module.down_proj, p_lr.experts_module.down_proj)
    assert torch.equal(w_lr.router.weight, p_lr.router.weight)
