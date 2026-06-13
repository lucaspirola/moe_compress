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


# ---------------------------------------------------------------------------
# A7.5 — E2E equivalence: 2 in-process CPU "workers" reduce == serial pass.
# ---------------------------------------------------------------------------
def test_dp_reduce_equivalent_to_serial_profile(tmp_path):
    """Two sequence-disjoint shards, each profiled into its own accumulator set
    with the cov per-seq pin, spilled and reduced, reproduce the serial single-pass
    accumulators: cov/REAP ~1e-5 (fp32), REAM bit-exact (fp64)."""
    from moe_compress.stage2 import profiling, profile_dp
    from moe_compress.utils.activation_hooks import InputCovarianceAccumulator

    n_seq, seq_len = 6, 8
    torch.manual_seed(99)
    calib = torch.randint(0, 32, (n_seq, seq_len), dtype=torch.long)

    def _profile_into(model, batches, seq_len_arg):
        lr = list(iter_moe_layers(model))[0]
        reap, cov, ream = _new_accs(lr.num_routed_experts)
        for b in batches:
            profiling._profile_layer(
                model, lr, [b], reap, cov, ream,
                device=torch.device("cpu"), seq_len=seq_len_arg,
            )
        reap.finalize_layer(lr.layer_idx)
        cov.finalize_layer(lr.layer_idx)
        return reap, cov, ream, lr.layer_idx

    # Serial reference: one model, all sequences as one batch (pin on so the
    # comparison is to the batch-invariant reference, matching the DP target).
    torch.manual_seed(0); m_serial = _TinyModel()
    s_reap, s_cov, s_ream, li = _profile_into(m_serial, [calib], seq_len)

    # DP: two replicas over disjoint shards (each its own fresh model copy).
    shards = profile_dp.shard_calib_sequences(calib, replicas=2)
    rep_dirs = []
    for r, shard in enumerate(shards):
        torch.manual_seed(0); m_r = _TinyModel()
        reap, cov, ream, _ = _profile_into(m_r, [shard], seq_len)
        d = tmp_path / f"rep{r}"
        profile_dp._spill_reap_layer(reap, li, d / "reap")
        cov.spill_layer_to_disk(li, str(d / "cov"))
        profile_dp._spill_ream_layer(ream, li, d / "ream")
        rep_dirs.append(d)

    # Reduce the four accumulators into fresh parent accs.
    from moe_compress.stage3.plugins.covariance_collection import _reduce_spilled_cov_dirs
    p_reap, p_cov, p_ream = _new_accs(s_ream.num_experts)
    profile_dp._reduce_reap_dirs([d / "reap" for d in rep_dirs], li, into=p_reap)
    profile_dp._reduce_ream_dirs([d / "ream" for d in rep_dirs], li, into=p_ream)
    cov_out = tmp_path / "cov_reduced"
    _reduce_spilled_cov_dirs([d / "cov" for d in rep_dirs], cov_out, storage_dtype=torch.float32)
    loaded = InputCovarianceAccumulator()
    loaded.load_layer_from_disk(li, cov_out)

    # cov ~1e-5 (fp32).
    assert set(loaded.covariance) == set(s_cov.covariance) and s_cov.covariance
    for k in s_cov.covariance:
        assert torch.allclose(loaded.covariance[k], s_cov.covariance[k], atol=1e-4), f"cov @ {k}"
    # REAP score ~1e-5.
    for k in s_reap.sums:
        assert abs(p_reap.score(k[0], k[1]) - s_reap.score(k[0], k[1])) < 1e-4, f"reap @ {k}"
    # REAM gate_gram + sim: the REDUCE is bit-exact (A7.3/A7.4), but the
    # serial-vs-DP comparison crosses a matmul-grouping boundary (serial folds
    # one big xᵀx per batch; DP sums per-shard xᵀx), so it matches to fp64
    # rounding (~1e-12), not bit-for-bit. The bit-exactness of the reduce given
    # identical per-batch inputs is pinned by test_reduce_ream_gate_gram_bit_exact.
    assert torch.allclose(p_ream._gate_gram[li], s_ream._gate_gram[li], atol=1e-9)
    if li in s_ream._sim_tensor and li in p_ream._sim_tensor:
        assert torch.allclose(p_ream._sim_tensor[li], s_ream._sim_tensor[li], atol=1e-9)


# ---------------------------------------------------------------------------
# A7.7 — Byte-identical default gate: profile_dp.enabled=false ⇒ serial.
# ---------------------------------------------------------------------------
def test_default_gate_disabled_resolves_serial():
    from moe_compress.stage2 import profile_dp

    s2 = {"profile_dp": {"enabled": False}}
    cfg = profile_dp.resolve_profile_dp_config(
        s2, expert_distill_steps=0, cost_alignment="pre", merge_step="freq_weighted",
        device_count=4,
    )
    assert cfg["enabled"] is False


def test_default_gate_absent_resolves_serial():
    from moe_compress.stage2 import profile_dp

    cfg = profile_dp.resolve_profile_dp_config(
        {}, expert_distill_steps=0, cost_alignment="pre", merge_step="freq_weighted",
        device_count=4,
    )
    assert cfg["enabled"] is False


def test_replicas_auto_resolves_to_device_count():
    from moe_compress.stage2 import profile_dp

    s2 = {"profile_dp": {"enabled": True, "replicas": "auto"}}
    cfg = profile_dp.resolve_profile_dp_config(
        s2, expert_distill_steps=0, cost_alignment="pre", merge_step="freq_weighted",
        device_count=3,
    )
    assert cfg["enabled"] is True
    assert cfg["replicas"] == 3
    # device_count<=1 ⇒ enabled collapses to serial.
    cfg1 = profile_dp.resolve_profile_dp_config(
        s2, expert_distill_steps=0, cost_alignment="pre", merge_step="freq_weighted",
        device_count=1,
    )
    assert cfg1["enabled"] is False


# ---------------------------------------------------------------------------
# A7.8 — Reservoir guard at resolution: distill/output/mergemoe ⇒ DP off + warn.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kwargs,consumer", [
    (dict(expert_distill_steps=5, cost_alignment="pre", merge_step="freq_weighted"), "expert_distill"),
    (dict(expert_distill_steps=0, cost_alignment="output", merge_step="freq_weighted"), "output"),
    (dict(expert_distill_steps=0, cost_alignment="pre", merge_step="mergemoe"), "mergemoe"),
])
def test_reservoir_guard_disables_dp_and_warns(kwargs, consumer):
    import logging as _logging
    from moe_compress.stage2 import profile_dp

    # Attach a capturing handler DIRECTLY to the module logger so the assertion
    # is independent of caplog's propagation assumptions (other tests in the
    # suite may reconfigure root-logger propagation/handlers).
    records: list[_logging.LogRecord] = []

    class _Cap(_logging.Handler):
        def emit(self, record):
            records.append(record)

    logger = _logging.getLogger("moe_compress.stage2.profile_dp")
    h = _Cap(level=_logging.WARNING)
    prev_level, prev_disabled = logger.level, logger.disabled
    logger.addHandler(h)
    logger.setLevel(_logging.WARNING)
    logger.disabled = False
    try:
        s2 = {"profile_dp": {"enabled": True, "replicas": "auto"}}
        cfg = profile_dp.resolve_profile_dp_config(s2, device_count=4, **kwargs)
    finally:
        logger.removeHandler(h)
        logger.setLevel(prev_level)
        logger.disabled = prev_disabled

    assert cfg["enabled"] is False, "reservoir consumer must force serial"
    assert any(consumer in r.getMessage() for r in records), \
        f"warning must name the consumer {consumer!r}"


# ---------------------------------------------------------------------------
# A0/A4 — persistent-pool protocol: RESYNC barrier ordering + run_dp reduce.
# ---------------------------------------------------------------------------
class _RecordingHandler:
    """In-process worker stand-in. Records the order of resync/profile calls and
    runs run_profile_shard against its own model copy so the reduce is real."""

    def __init__(self, model, layer_ref, shard, seq_len):
        self.model = model
        self.layer_ref = layer_ref
        self.shard = shard
        self.seq_len = seq_len
        self.calls: list[tuple] = []

    def resync(self, layer_idx, payload_path):
        from moe_compress.stage2 import profile_dp
        self.calls.append(("resync", layer_idx))
        if payload_path is not None:
            payload = torch.load(payload_path, map_location="cpu", weights_only=False)
            profile_dp.replay_merged_layer(self.layer_ref, payload)

    def profile(self, layer_idx, shard_id, spill_dir, seq_len):
        from moe_compress.stage2 import profile_dp
        self.calls.append(("profile", layer_idx, shard_id))
        profile_dp.run_profile_shard(
            self.model, self.layer_ref, [self.shard], spill_dir,
            seq_len=seq_len, device=torch.device("cpu"),
        )


def test_pool_run_dp_profile_layer_reduces_and_barrier_order(tmp_path):
    from moe_compress.stage2 import profile_dp

    n_seq, seq_len = 4, 8
    torch.manual_seed(5)
    calib = torch.randint(0, 32, (n_seq, seq_len), dtype=torch.long)
    shards = profile_dp.shard_calib_sequences(calib, replicas=2)

    handlers = []
    for shard in shards:
        torch.manual_seed(0); m = _TinyModel()
        lr = list(iter_moe_layers(m))[0]
        handlers.append(_RecordingHandler(m, lr, shard, seq_len))

    pool = profile_dp.Stage2ProfilePool(replicas=2, executor="inprocess")
    pool.start_inprocess(handlers)

    li = handlers[0].layer_ref.layer_idx
    reap, cov, ream = _new_accs(handlers[0].layer_ref.num_routed_experts)
    # prev_layer payload (a no-op replay path here) exercises the RESYNC barrier.
    payload_path = tmp_path / "prev_payload.pt"
    torch.save(
        profile_dp.capture_merged_layer(handlers[0].layer_ref, list(range(4))),
        payload_path,
    )
    profile_dp.run_dp_profile_layer(
        pool, handlers[0].layer_ref,
        reap_acc=reap, cov_acc=cov, ream_acc=ream,
        spill_root=tmp_path / "spill", seq_len=seq_len,
        cov_storage_dtype=torch.float32,
        prev_layer_idx=li - 1 if li > 0 else 99,
        prev_layer_payload_path=payload_path,
    )
    pool.shutdown()

    # Barrier order: every worker did resync BEFORE profile.
    for h in handlers:
        names = [c[0] for c in h.calls]
        assert names == ["resync", "profile"], f"barrier violated: {names}"

    # Reduce populated the parent accumulators.
    assert any(k[0] == li for k in cov.covariance), "cov reduced into parent"
    assert li in ream._gate_gram, "ream gate_gram reduced into parent"
    assert any(k[0] == li for k in reap.counts), "reap reduced into parent"


def test_pool_shutdown_drains_then_joins_and_reads_error_queue():
    """Protocol guard (reviewer Low note): shutdown DRAINS the reduce queue before
    join (so a worker blocked writing it can exit), and a worker ERROR is surfaced
    by READING the queue, not by exitcode alone."""
    import queue
    from moe_compress.stage2 import profile_dp

    pool = profile_dp.Stage2ProfilePool(replicas=2, executor="spawn")
    # Wire fake spawn-mode queues.
    pool._cmd_qs = [queue.Queue(), queue.Queue()]
    pool._reduce_q = queue.Queue()

    class _FakeProc:
        def __init__(self): self.exitcode = 0; self._alive = False
        def is_alive(self): return self._alive
        def join(self, timeout=None): pass
        def terminate(self): self._alive = False

    pool._procs = [_FakeProc(), _FakeProc()]
    pool._started = True
    # Pre-load the reduce queue with stale messages; shutdown must drain them
    # WITHOUT hanging (no DONE/ACK left blocking a join).
    pool._reduce_q.put(("DONE", 3, 0, "/tmp/x"))
    pool._reduce_q.put(("DONE", 3, 1, "/tmp/y"))
    pool.shutdown()  # must drain + join cleanly, no exception (exitcodes 0)
    assert pool._reduce_q.empty(), "reduce queue must be drained before join"

    # ERROR is surfaced by reading the queue during PROFILE (not exitcode).
    pool2 = profile_dp.Stage2ProfilePool(replicas=1, executor="spawn")
    pool2._cmd_qs = [queue.Queue()]
    pool2._reduce_q = queue.Queue()
    pool2._procs = [_FakeProc()]
    pool2._started = True
    pool2._reduce_q.put(("ERROR", 7, "boom-traceback"))
    with pytest.raises(RuntimeError, match="boom-traceback"):
        pool2.profile_layer(7, "/tmp/spill", seq_len=8)
