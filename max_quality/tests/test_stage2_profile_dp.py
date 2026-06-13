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
