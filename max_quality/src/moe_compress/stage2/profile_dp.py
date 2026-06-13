"""Stage-2 persistent-pool data-parallel per-layer profile forward (Feature A).

DEFAULT-OFF. Opt-in via ``stage2_reap_ream.profile_dp.enabled``; ``enabled=false``
OR resolved ``replicas<=1`` ⇒ the serial ``_profile_layer`` path, byte-identical.

Why a NEW subsystem (no repo template): Stage-3's DP cov collector spawns *fresh*
workers per call and reduces via disk (``covariance_collection.py:1213-1289``); it
never keeps workers alive or sends them live commands. Stage-2's per-layer merge
mutates the live model in place and layer ``L+1`` profiles through ``0..L`` of that
*mutated* model (``profiling.py:338-342``), so a persistent pool must REPLAY the
structural merge of each upstream layer on every worker before the next profile.
That is the load-bearing constraint that forces a persistent pool + per-layer
RESYNC barrier rather than a stateless per-call spawn.

Per-layer protocol (inside the sequential merge loop):
  1. parent does layer L-1 merge/post_merge on itself (normal serial),
  2. parent sends RESYNC(L-1, merged-layer payload) to all workers and waits for
     ALL ACKs (structural replay done — workers now hold the merged upstream),
  3. parent sends PROFILE(L, shard_r) to worker r,
  4. workers profile their sequence-disjoint shard through 0..L, finalize the four
     additive accumulators per layer, spill to per-replica dirs, send DONE,
  5. parent joins all DONE, then key-wise reduces the four spill sets into its own
     reap_acc / cov_acc / ream_acc. assign/merge run on the parent exactly as serial.

The four Stage-2 accumulators are all additive Σ-over-tokens + a separate additive
count, so for a SEQUENCE-disjoint shard set the reduce is exactly
``Σ_r num_r / Σ_r count_r`` (mean once after the reduce). REAM grams are fp64 ⇒ the
reduce is bit-exact regardless of order; cov + REAP are fp32 ⇒ ~1e-6 drift, the
same class the serial path already tolerates — and ABSENT on the 1-replica default
(no reduce runs).

Open-Q2 (storage-dtype cast ownership): the **reduce** owns the cast. Cov reduce
(``_reduce_spilled_cov_dirs``) casts to ``storage_dtype`` and the result is loaded
straight into ``cov_acc.covariance`` (already-finalized), so the parent-side
``cov_acc.finalize_layer`` on the DP path is a no-op (nothing in ``_pending``). No
double-cast. REAP/REAM reduces produce final CPU values placed directly into the
parent accumulators.

Open-Q5 (DEFERRED): DP × resume is NOT implemented. On a resumed run the orchestrator
`continue`s over already-merged (completed) layers, so the workers — which load the
PRE-merge model at spawn — would forward through un-merged upstream layers (no RESYNC
backlog replay). The orchestrator therefore DISABLES DP on any resume with a loud
warning and falls back to the serial profile. Backlog-replay of all completed layers'
merges at pool spawn is future work.

A8 — DEFERRED: live ≥2-GPU validation. The 1-GPU default-off path is byte-identical
(test ``test_default_gate_*``); the reduce + structural-replay math is proven
in-process (CPU-simulated workers). The FIRST live ≥2-GPU run validates the
persistent-pool IPC + RESYNC barrier + teardown — mirror the Stage-3/4 "untested on
real ≥2-GPU" caveat in any landing note.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# A1 — config resolution (default OFF) + reservoir guard AT RESOLUTION.
#
# The distill-input reservoir (_LayerInputAccumulator) is a RESERVOIR SAMPLE,
# not additive, so it is NOT reducible across shards. When ANY layer-input
# consumer is active — expert_distill_steps>0 OR cost_alignment=="output" OR
# merge_step=="mergemoe" (the exact disjunction at layer_merge.py:468-472) — DP
# profiling is disabled for the WHOLE run with a single log.warning naming the
# consumer, HERE at resolution time (not 40 layers deep). enabled=False OR
# resolved replicas<=1 ⇒ the serial _profile_layer path, byte-identical.
# (Reservoir-merge across shards by global-`seen` weighting is future work.)
# ---------------------------------------------------------------------------
def resolve_profile_dp_config(
    s2: dict,
    *,
    expert_distill_steps: int,
    cost_alignment: str,
    merge_step: str,
    device_count: int,
) -> dict:
    """Resolve the effective ``stage2_reap_ream.profile_dp`` config. Returns a
    dict ``{enabled, replicas, shards_per_model}``. ``enabled`` is forced False
    on any reservoir consumer, on ``device_count<=1``, or on resolved
    ``replicas<=1`` — those paths run the byte-identical serial profile."""
    raw = dict(s2.get("profile_dp") or {})
    enabled = bool(raw.get("enabled", False))
    shards_per_model = int(raw.get("shards_per_model", 1))

    # Reservoir guard — fire at resolution, name the consumer.
    if enabled:
        consumer = None
        if expert_distill_steps and expert_distill_steps > 0:
            consumer = "expert_distill"
        elif str(cost_alignment).lower() == "output":
            consumer = "output"  # cost_alignment="output"
        elif str(merge_step).lower() == "mergemoe":
            consumer = "mergemoe"  # merge_step="mergemoe"
        if consumer is not None:
            log.warning(
                "stage2 profile_dp DISABLED for the whole run: the layer-input "
                "reservoir consumer %r is active (a reservoir sample is not "
                "additive, so it is not reducible across DP shards). Falling back "
                "to the serial profile. (Reservoir-merge across shards is future "
                "work.)",
                consumer,
            )
            enabled = False

    # Resolve replicas: "auto" -> device_count; clamp >=1.
    replicas_raw = raw.get("replicas", "auto")
    if isinstance(replicas_raw, str) and replicas_raw.lower() == "auto":
        replicas = max(int(device_count), 1)
    else:
        replicas = max(int(replicas_raw), 1)

    # device_count<=1 or replicas<=1 ⇒ serial (byte-identical).
    if enabled and (int(device_count) <= 1 or replicas <= 1):
        enabled = False

    return {
        "enabled": enabled,
        "replicas": replicas,
        "shards_per_model": max(shards_per_model, 1),
    }


# ---------------------------------------------------------------------------
# A2 — sequence-disjoint shard (mirror Stage-3 _shard_calib at the
# calibration-SEQUENCE granularity: contiguous dim-0 slices, last absorbs the
# remainder, so every sequence is covered exactly once and the per-key Gram sum
# is exact + replica-independent).
# ---------------------------------------------------------------------------
def shard_calib_sequences(calib: torch.Tensor, replicas: int) -> list[torch.Tensor]:
    """Split the [n_seq, seq_len] calibration tensor into ``replicas`` contiguous,
    disjoint by-sequence shards. ``replicas<=1`` (or empty) ⇒ a single shard
    (the serial / byte-identical premise)."""
    n = calib.size(0)
    if replicas <= 1 or n == 0:
        return [calib]
    replicas = min(replicas, n)
    base = n // replicas
    shards: list[torch.Tensor] = []
    start = 0
    for r in range(replicas):
        end = n if r == replicas - 1 else start + base
        shards.append(calib[start:end])
        start = end
    return shards


# ---------------------------------------------------------------------------
# A3 — per-replica spill + per-layer reduce of the four additive accumulators.
#
# Cov reuses Stage-3's spill_layer_to_disk / _reduce_spilled_cov_dirs verbatim
# (fp32 key-wise sum, storage-dtype cast owned by the reduce). REAP and REAM get
# small spill/reduce helpers here, pattern-identical to _reduce_spilled_cov_dirs
# (sorted dir order for determinism, fp64 REAM bit-exact, fp32 REAP ~1e-6, ints
# exact). Reduce runs PER LAYER inside the sequential loop, reconstructing the
# parent's accumulators so assign/merge consume them exactly as serial.
# ---------------------------------------------------------------------------
def _sorted_dirs(dirs) -> list[Path]:
    return [Path(d) for d in sorted(str(d) for d in dirs)]


def _spill_reap_layer(reap, layer_idx: int, dir_path) -> None:
    """Spill a ReapAccumulator's finalized per-(l,e) {sums, counts, freq} for
    ``layer_idx`` to ``dir_path/layer_{idx}.pt``. The accumulator must be
    finalized (``finalize_layer``) so ``sums`` holds CPU floats."""
    out_dir = Path(dir_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    with reap._lock:
        sums = {k: float(v) for k, v in reap.sums.items() if k[0] == layer_idx}
        counts = {k: int(v) for k, v in reap.counts.items() if k[0] == layer_idx}
        freq = {k: int(v) for k, v in reap.freq.items() if k[0] == layer_idx}
    payload = {"format_version": 1, "sums": sums, "counts": counts, "freq": freq}
    out = out_dir / f"layer_{layer_idx}.pt"
    tmp = out.with_suffix(out.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, out)


def _reduce_reap_dirs(replica_dirs, layer_idx: int, *, into) -> None:
    """Key-wise reduce per-replica REAP spills for ``layer_idx`` into ``into``
    (a fresh ReapAccumulator): sums fp32-add, counts/freq int-add. ``into.score``
    then equals Σsum/Σcount. Sorted dir order for run-to-run determinism."""
    m_sums: dict = {}
    m_counts: dict = {}
    m_freq: dict = {}
    for d in _sorted_dirs(replica_dirs):
        p = d / f"layer_{layer_idx}.pt"
        if not p.exists():
            continue
        payload = torch.load(p, map_location="cpu", weights_only=False)
        for k, v in payload.get("sums", {}).items():
            m_sums[k] = m_sums.get(k, 0.0) + float(v)
        for k, v in payload.get("counts", {}).items():
            m_counts[k] = m_counts.get(k, 0) + int(v)
        for k, v in payload.get("freq", {}).items():
            m_freq[k] = m_freq.get(k, 0) + int(v)
    with into._lock:
        into.sums.update(m_sums)
        into.counts.update(m_counts)
        into.freq.update(m_freq)


def _spill_ream_layer(ream, layer_idx: int, dir_path) -> None:
    """Spill a ReamCostAccumulator's per-layer additive state for ``layer_idx``:
    ``_gate_gram[li]`` ([E,E] fp64), ``_sim_tensor[li]`` ([E,E] fp64),
    ``_total_tokens_by_layer[li]`` (int), and per-expert
    ``_neuron_act_sum/_count`` ([d_int] fp32 + int)."""
    out_dir = Path(dir_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    with ream._lock:
        gate_gram = ream._gate_gram.get(layer_idx)
        sim = ream._sim_tensor.get(layer_idx)
        total = int(ream._total_tokens_by_layer.get(layer_idx, 0))
        neuron_sum = {
            k: v.clone() for k, v in ream._neuron_act_sum.items() if k[0] == layer_idx
        }
        neuron_count = {
            k: int(v) for k, v in ream._neuron_act_count.items() if k[0] == layer_idx
        }
    payload = {
        "format_version": 1,
        "gate_gram": None if gate_gram is None else gate_gram.clone(),
        "sim_tensor": None if sim is None else sim.clone(),
        "total_tokens": total,
        "neuron_sum": neuron_sum,
        "neuron_count": neuron_count,
    }
    out = out_dir / f"layer_{layer_idx}.pt"
    tmp = out.with_suffix(out.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, out)


def _reduce_ream_dirs(replica_dirs, layer_idx: int, *, into) -> None:
    """Key-wise reduce per-replica REAM spills for ``layer_idx`` into ``into``
    (a fresh ReamCostAccumulator): gate_gram + sim_tensor fp64-add (bit-exact,
    order-independent), total_tokens + neuron_count int-add, neuron_sum fp32-add.
    Sorted dir order for determinism."""
    gate_gram = None
    sim = None
    total = 0
    n_sum: dict = {}
    n_count: dict = {}
    for d in _sorted_dirs(replica_dirs):
        p = d / f"layer_{layer_idx}.pt"
        if not p.exists():
            continue
        payload = torch.load(p, map_location="cpu", weights_only=False)
        g = payload.get("gate_gram")
        if g is not None:
            gate_gram = g.clone() if gate_gram is None else gate_gram + g
        s = payload.get("sim_tensor")
        if s is not None:
            sim = s.clone() if sim is None else sim + s
        total += int(payload.get("total_tokens", 0))
        for k, v in payload.get("neuron_sum", {}).items():
            n_sum[k] = v.clone() if k not in n_sum else n_sum[k] + v
        for k, v in payload.get("neuron_count", {}).items():
            n_count[k] = n_count.get(k, 0) + int(v)
    with into._lock:
        if gate_gram is not None:
            into._gate_gram[layer_idx] = gate_gram
        if sim is not None:
            into._sim_tensor[layer_idx] = sim
        if total:
            into._total_tokens_by_layer[layer_idx] = (
                into._total_tokens_by_layer.get(layer_idx, 0) + total
            )
        for k, v in n_sum.items():
            into._neuron_act_sum[k] = v
        for k, v in n_count.items():
            into._neuron_act_count[k] = v


# ---------------------------------------------------------------------------
# C1 — structural replay (recommended Open-Q1 variant: broadcast the parent's
# already-selected kept tensors + the resized router, set them DIRECTLY on the
# worker copy). The re-sync is structural surgery, NOT a value delta:
# ``bank.select`` reshapes the stacked expert tensor; ``_resize_router_for_kept_experts``
# REPLACES the router Parameter + mutates num_experts/top_k/mlp.num_experts.
# ---------------------------------------------------------------------------
_STACKED_ATTRS = ("gate_up_proj", "down_proj", "gate_proj", "up_proj")


def capture_merged_layer(layer_ref, final_kept_ids) -> dict:
    """Snapshot a parent layer AFTER its merge + post_merge (bank.select + router
    resize) into a picklable payload the worker can replay. Captures every stacked
    expert Parameter present on the experts module (post-select shapes) + the
    resized router weight/bias + the post-merge counts. CPU clones — small (one
    merged MoE layer)."""
    em = layer_ref.experts_module
    router = layer_ref.router
    stacked: dict[str, torch.Tensor] = {}
    for attr in _STACKED_ATTRS:
        p = getattr(em, attr, None)
        if isinstance(p, nn.Parameter) or torch.is_tensor(p):
            stacked[attr] = p.detach().to("cpu").clone()
    payload = {
        "format_version": 1,
        "final_kept_ids": list(final_kept_ids),
        "stacked": stacked,
        "router_weight": router.weight.detach().to("cpu").clone(),
        "router_bias": (
            router.bias.detach().to("cpu").clone()
            if getattr(router, "bias", None) is not None else None
        ),
        "experts_num_experts": int(getattr(em, "num_experts", len(final_kept_ids))),
        "router_num_experts": int(getattr(router, "num_experts", len(final_kept_ids))),
        "router_top_k": int(getattr(router, "top_k", 0)) or None,
        "mlp_num_experts": (
            int(getattr(layer_ref.mlp, "num_experts"))
            if hasattr(layer_ref.mlp, "num_experts") else None
        ),
    }
    return payload


def replay_merged_layer(layer_ref, payload: dict) -> None:
    """Reproduce the parent's structural merge on a worker's resident model copy:
    replace each stacked expert Parameter with the broadcast post-select tensor,
    replace the router weight/bias Parameter, and set the experts/router/mlp
    counts. After this the worker's layer is shape- and value-identical to the
    parent's, so the next profile forwards through the SAME merged upstream."""
    em = layer_ref.experts_module
    router = layer_ref.router
    with torch.no_grad():
        for attr, t in payload["stacked"].items():
            cur = getattr(em, attr, None)
            if cur is None:
                continue
            dev = cur.device if torch.is_tensor(cur) else router.weight.device
            req = bool(getattr(cur, "requires_grad", False))
            setattr(em, attr, nn.Parameter(t.to(dev).clone(), requires_grad=req))
        rw = payload["router_weight"]
        router.weight = nn.Parameter(
            rw.to(router.weight.device).clone(), requires_grad=router.weight.requires_grad
        )
        rb = payload.get("router_bias")
        if rb is not None and getattr(router, "bias", None) is not None:
            router.bias = nn.Parameter(
                rb.to(router.bias.device).clone(), requires_grad=router.bias.requires_grad
            )
        if hasattr(em, "num_experts"):
            em.num_experts = int(payload["experts_num_experts"])
        if hasattr(router, "num_experts"):
            router.num_experts = int(payload["router_num_experts"])
        if payload.get("router_top_k") is not None and hasattr(router, "top_k"):
            router.top_k = int(payload["router_top_k"])
        if payload.get("mlp_num_experts") is not None and hasattr(layer_ref.mlp, "num_experts"):
            layer_ref.mlp.num_experts = int(payload["mlp_num_experts"])


# ---------------------------------------------------------------------------
# A3 (parent side) — reduce the four per-replica spill sets for one layer into
# the parent's accumulators, so assign/merge consume them exactly as serial.
# Cov is reduced via Stage-3's _reduce_spilled_cov_dirs into a canonical dir and
# loaded already-finalized (the reduce owns the storage-dtype cast — Open-Q2).
# ---------------------------------------------------------------------------
def reduce_layer_into_parent(
    layer_idx: int,
    replica_dirs,
    *,
    reap_acc,
    cov_acc,
    ream_acc,
    cov_storage_dtype: torch.dtype,
) -> None:
    """Reduce REAP/REAM (this module) + cov (Stage-3 helper) for ``layer_idx``
    from each replica's ``{reap,cov,ream}`` subdir into the parent accumulators."""
    from ..stage3.plugins.covariance_collection import _reduce_spilled_cov_dirs

    rdirs = [Path(d) for d in replica_dirs]
    _reduce_reap_dirs([d / "reap" for d in rdirs], layer_idx, into=reap_acc)
    _reduce_ream_dirs([d / "ream" for d in rdirs], layer_idx, into=ream_acc)
    # Reduce cov into a per-layer canonical dir, then load already-finalized into
    # the parent cov_acc (the reduce already cast to storage_dtype; the parent's
    # cov_acc.finalize_layer is a no-op afterwards — Open-Q2: reduce owns cast).
    cov_canon = rdirs[0].parent / f"_cov_reduced_layer_{layer_idx}"
    _reduce_spilled_cov_dirs(
        [d / "cov" for d in rdirs], cov_canon, storage_dtype=cov_storage_dtype,
    )
    cov_acc.load_layer_from_disk(layer_idx, cov_canon)


# ---------------------------------------------------------------------------
# A3 (worker side) — run the early-exit profile on one shard and spill the four
# accumulators. Shared by the real mp worker and the CPU in-process executor.
# ---------------------------------------------------------------------------
def run_profile_shard(
    model,
    layer_ref,
    shard_batches,
    spill_dir,
    *,
    seq_len: int,
    device=None,
    auto_batch_cfg=None,
) -> None:
    """Profile ``layer_ref`` over this replica's shard (cov per-seq pinned via
    ``seq_len``), finalize the four accumulators for the layer, and spill them to
    ``spill_dir/{reap,cov,ream}``. One call per (layer, replica)."""
    from .profiling import _profile_layer
    from ..utils.activation_hooks import (
        InputCovarianceAccumulator, ReamCostAccumulator, ReapAccumulator,
    )

    li = layer_ref.layer_idx
    n_experts = layer_ref.num_routed_experts
    reap = ReapAccumulator()
    cov = InputCovarianceAccumulator()
    ream = ReamCostAccumulator(num_experts=n_experts)
    _profile_layer(
        model, layer_ref, shard_batches, reap, cov, ream,
        device=device, seq_len=seq_len,
    )
    reap.finalize_layer(li)
    cov.finalize_layer(li)
    sd = Path(spill_dir)
    _spill_reap_layer(reap, li, sd / "reap")
    cov.spill_layer_to_disk(li, str(sd / "cov"))
    _spill_ream_layer(ream, li, sd / "ream")


# ---------------------------------------------------------------------------
# A0 — persistent worker pool + per-layer command/reduce IPC (NEW subsystem).
#
# Message protocol (parent→worker command queue; worker→parent reduce queue):
#   RESYNC(layer_idx, payload_path)  -> worker replays the merged layer, ACKs.
#   PROFILE(layer_idx, shard_id, spill_dir, seq_len, auto_batch_cfg)
#                                    -> worker profiles its shard, spills, DONE.
#   SHUTDOWN                         -> worker exits the loop.
# worker→parent: ("ACK", layer_idx) / ("DONE", layer_idx, shard_id, spill_dir)
#                / ("ERROR", layer_idx, traceback_str).
#
# Large payloads (the merged-layer tensors broadcast in RESYNC, and the four
# accumulators) travel via the FILESYSTEM (torch.save path / spill dir), never
# serialized through the mp.Queue — the queue carries only small control
# messages + paths. This keeps IPC small and reuses the proven disk reduce.
#
# Barrier protocol (deadlock-free): RESYNC -> wait ALL ACK before PROFILE each
# layer; parent joins ALL DONE before its own layer merge. At SHUTDOWN the parent
# DRAINS the reduce queue BEFORE join() and the worker-ERROR handler READS the
# queue (bounded traceback string), so a full pipe buffer + blocked join() cannot
# deadlock (the reviewer's Low note).
# ---------------------------------------------------------------------------
def _profile_worker_main(
    replica_idx: int,
    visible_devices: str,
    config: dict,
    model_path: str,
    shard_start: int,
    shard_end: int,
    cmd_q,
    reduce_q,
) -> None:
    """Spawn target: one persistent DP replica. Pins its GPU subset via
    ``CUDA_VISIBLE_DEVICES``, loads its OWN model copy from disk (the same source
    the parent loaded), rebuilds + slices its sequence-disjoint calibration shard,
    then enters the command loop. Module-level (picklable) so it is a valid
    ``torch.multiprocessing`` spawn target.

    Command loop (blocks on ``cmd_q``):
      RESYNC(layer_idx, payload_path) -> replay_merged_layer on the resident copy,
                                         then reply ("ACK", layer_idx).
      PROFILE(layer_idx, shard_id, spill_dir, seq_len, ...) -> run_profile_shard,
                                         then reply ("DONE", layer_idx, shard_id, spill_dir).
      SHUTDOWN -> break.

    Any exception is caught, formatted to a BOUNDED traceback string, and sent as
    ("ERROR", layer_idx, tb) on the reduce queue so the parent surfaces it by
    READING the queue (not relying on exitcode), then the worker exits non-zero.
    """
    import os as _os
    _os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
    import traceback as _tb
    import torch as _torch
    from ..utils.model_io import (
        load_compressed_model as _load_compressed_model,
        iter_moe_layers as _iter_moe_layers,
    )
    from ..utils.calibration import (
        build_calibration_tensor as _bct,
        spec_from_config as _spec_from_config,
    )

    cur_layer = -1
    try:
        device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
        model, tokenizer, _ = _load_compressed_model(
            model_path,
            device_map=config["model"]["device_map"],
            torch_dtype=config["model"]["torch_dtype"],
            attn_implementation=config["model"].get("attn_implementation", "sdpa"),
        )
        layer_refs = {lr.layer_idx: lr for lr in _iter_moe_layers(model)}
        spec = _spec_from_config(config["calibration"])
        calib = _bct(tokenizer, spec)
        shard = calib[shard_start:shard_end]
        while True:
            msg = cmd_q.get()
            kind = msg[0]
            if kind == "SHUTDOWN":
                break
            if kind == "RESYNC":
                _, cur_layer, payload_path = msg
                payload = _torch.load(payload_path, map_location="cpu", weights_only=False)
                replay_merged_layer(layer_refs[cur_layer], payload)
                reduce_q.put(("ACK", cur_layer))
            elif kind == "PROFILE":
                _, cur_layer, shard_id, spill_dir, sl = msg
                # Each worker re-batches its shard with its own (auto-)batch size;
                # bs=1 here keeps the cov pin trivially exact (A5 default).
                batches = [shard[i:i + 1] for i in range(shard.size(0))]
                run_profile_shard(
                    model, layer_refs[cur_layer], batches, spill_dir,
                    seq_len=sl, device=device,
                )
                reduce_q.put(("DONE", cur_layer, shard_id, spill_dir))
    except BaseException:
        tb = _tb.format_exc()[-4000:]  # bounded — never blow the pipe buffer
        try:
            reduce_q.put(("ERROR", cur_layer, tb))
        except Exception:
            pass
        raise


class _InProcessProc:
    """Test-only synchronous 'process' standing in for a spawned worker. Runs the
    worker command-handler inline so the protocol (barrier, drain, error-read) is
    exercised on CPU without real multiprocessing. exitcode mirrors mp.Process."""

    def __init__(self, handler):
        self._handler = handler
        self.exitcode = None
        self._alive = True

    def is_alive(self):
        return self._alive

    def join(self, timeout=None):
        self._alive = False

    def terminate(self):
        self._alive = False


class Stage2ProfilePool:
    """Persistent DP worker pool for Stage-2 per-layer profiling. Spawn ONCE at
    Stage-2 start; drive per layer via RESYNC + PROFILE; SHUTDOWN at Stage-2 end.

    ``executor='spawn'`` (default) uses ``mp.get_context('spawn')`` with one
    process per replica. ``executor='inprocess'`` runs synchronous in-process
    workers (CPU tests) that exercise the SAME barrier/drain/error protocol.
    """

    def __init__(self, replicas: int, *, executor: str = "spawn"):
        self.replicas = int(replicas)
        self.executor = executor
        self._procs: list = []
        self._cmd_qs: list = []      # parent -> worker
        self._reduce_q = None        # worker -> parent (shared)
        self._started = False

    def start(self, config: dict, model_path: str, calib_n_seq: int,
              *, shards_per_model: int = 1) -> None:
        """Spawn the persistent worker processes ONCE. Each worker pins a GPU
        subset, loads its own model copy, and slices its sequence-disjoint shard.
        Mirrors Stage-3's spawn wiring but keeps the processes alive across layers
        (the per-layer RESYNC barrier is what makes that legal). ``seq_len`` is NOT
        a spawn arg — it travels in each PROFILE message (constant per run, but the
        per-message form keeps the worker stateless about it)."""
        import torch.multiprocessing as _mp

        ctx = _mp.get_context("spawn")
        self._reduce_q = ctx.Queue()
        self._cmd_qs = []
        self._procs = []
        # Contiguous by-sequence shard boundaries (same math as shard_calib_sequences).
        n = int(calib_n_seq)
        replicas = min(self.replicas, max(n, 1))
        base = n // replicas if replicas else 0
        start = 0
        for r in range(replicas):
            end = n if r == replicas - 1 else start + base
            dev_lo = r * shards_per_model
            visible = ",".join(str(d) for d in range(dev_lo, dev_lo + shards_per_model))
            cmd_q = ctx.Queue()
            p = ctx.Process(
                target=_profile_worker_main,
                args=(r, visible, config, model_path, start, end,
                      cmd_q, self._reduce_q),
            )
            p.start()
            self._cmd_qs.append(cmd_q)
            self._procs.append(p)
            start = end
        self.replicas = replicas
        self._started = True

    # -- in-process protocol drivers (CPU tests; mirror the mp message loop) --
    def start_inprocess(self, worker_handlers: list) -> None:
        """Attach a list of per-replica handler objects exposing ``resync(payload)``
        and ``profile(layer_idx, shard_id, spill_dir, seq_len)`` and an optional
        ``fail_on`` hook. Used only by CPU protocol tests."""
        import queue as _queue
        self._reduce_q = _queue.Queue()
        self._procs = [_InProcessProc(h) for h in worker_handlers]
        self._handlers = worker_handlers
        self._started = True

    def resync(self, layer_idx: int, payload_path) -> None:
        """Broadcast RESYNC to every worker and WAIT for all ACKs before
        returning — the barrier that guarantees no worker profiles layer L until
        every worker holds the merged layer L-1."""
        if self.executor == "inprocess":
            for h in self._handlers:
                h.resync(layer_idx, payload_path)   # synchronous => ACK is implicit
            return
        for q in self._cmd_qs:
            q.put(("RESYNC", layer_idx, str(payload_path)))
        # Wait for one ACK per worker (or surface an ERROR by reading the queue).
        acked = 0
        while acked < self.replicas:
            msg = self._reduce_q.get()
            if msg[0] == "ERROR":
                raise RuntimeError(
                    f"profile_dp worker ERROR during RESYNC(layer={msg[1]}):\n{msg[2]}"
                )
            if msg[0] == "ACK":
                acked += 1

    def profile_layer(self, layer_idx: int, spill_root, *, seq_len: int) -> list:
        """Send PROFILE to each worker (one shard each), JOIN all DONE, return the
        per-replica spill dirs in shard order. Surfaces a worker ERROR by reading
        the reduce queue (never relies on exitcode alone)."""
        spill_dirs = [Path(spill_root) / f"_replica_{r}" for r in range(self.replicas)]
        if self.executor == "inprocess":
            for r, h in enumerate(self._handlers):
                h.profile(layer_idx, r, spill_dirs[r], seq_len)
            return spill_dirs
        for r, q in enumerate(self._cmd_qs):
            q.put(("PROFILE", layer_idx, r, str(spill_dirs[r]), seq_len))
        done = 0
        out: dict[int, Path] = {}
        while done < self.replicas:
            msg = self._reduce_q.get()
            if msg[0] == "ERROR":
                raise RuntimeError(
                    f"profile_dp worker ERROR during PROFILE(layer={msg[1]}):\n{msg[2]}"
                )
            if msg[0] == "DONE":
                _, _li, shard_id, spill_dir = msg
                out[shard_id] = Path(spill_dir)
                done += 1
        return [out[r] for r in range(self.replicas)]

    def shutdown(self) -> None:
        """Send SHUTDOWN, DRAIN the reduce queue BEFORE join (Low note: a full
        pipe + blocked join deadlocks), join with timeout, check exitcode per
        worker, terminate + raise on any non-zero/timeout, verify no leaks."""
        if not self._started:
            return
        if self.executor == "inprocess":
            for p in self._procs:
                p.join()
            self._started = False
            return
        for q in self._cmd_qs:
            q.put(("SHUTDOWN",))
        # Drain anything still in the reduce queue so a worker blocked writing it
        # can reach its own exit before we join().
        self._drain_reduce_queue()
        bad = []
        for p in self._procs:
            p.join(timeout=120)
            if p.is_alive():
                p.terminate()
                bad.append((p, "timeout"))
            elif p.exitcode not in (0, None):
                bad.append((p, f"exitcode={p.exitcode}"))
        self._started = False
        if bad:
            raise RuntimeError(
                "profile_dp pool teardown: %d worker(s) failed: %s"
                % (len(bad), ", ".join(str(b[1]) for b in bad))
            )

    def _drain_reduce_queue(self) -> None:
        if self._reduce_q is None:
            return
        try:
            while True:
                self._reduce_q.get_nowait()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# A4 — per-layer DP entrypoint (called from LayerMergePlugin.on_profile on the
# DP path). Issues RESYNC(L-1) (barrier) + PROFILE(L) (join), then reduces the
# four spill sets into the parent's accumulators.
# ---------------------------------------------------------------------------
def run_dp_profile_layer(
    pool: "Stage2ProfilePool",
    layer_ref,
    *,
    reap_acc,
    cov_acc,
    ream_acc,
    spill_root,
    seq_len: int,
    cov_storage_dtype: torch.dtype,
    prev_layer_idx=None,
    prev_layer_payload_path=None,
) -> None:
    """One DP profile step for ``layer_ref``. If ``prev_layer_idx`` is set, RESYNC
    the workers with the just-merged upstream layer (barrier) before profiling.
    Then PROFILE this layer across shards, join, and reduce into the parent."""
    li = layer_ref.layer_idx
    if prev_layer_idx is not None and prev_layer_payload_path is not None:
        pool.resync(prev_layer_idx, prev_layer_payload_path)
    replica_dirs = pool.profile_layer(li, spill_root, seq_len=seq_len)
    reduce_layer_into_parent(
        li, replica_dirs,
        reap_acc=reap_acc, cov_acc=cov_acc, ream_acc=ream_acc,
        cov_storage_dtype=cov_storage_dtype,
    )
