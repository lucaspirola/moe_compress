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
