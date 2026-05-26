"""Library: cached calibration signals (provider-pair infrastructure).

This module ships the schema + atomic-write + load API used by the
**cache-or-live provider-pair pattern** described in
``max_quality/docs/calibration_v2_data_capture_plan.md`` Section 0.

Every cacheable teacher signal has a provider pair:

* **Cache provider** -- tries to load a sidecar via this module's
  ``load_*`` functions. On hit, populates ``ctx.set(slot, payload)`` and
  returns the payload. On miss, returns ``None`` (PluginRegistry's
  ``dispatch_first`` then falls through to the live provider).
* **Live provider** -- runs the existing live calibration code path,
  writes a sidecar via ``save_*``, populates the same ctx slot.

Consumer plugins read the populated slot and never know whether the data
came from cache or from a live forward pass.

Atomic-write contract
---------------------
Every ``save_*`` writes to ``str(final_path) + ".tmp"`` then calls
``os.replace(tmp, final_path)``. A process kill mid-``torch.save`` leaves
the previous version of ``final_path`` intact (only ``.tmp`` is partial).
A kill between ``torch.save`` completing and ``os.replace`` leaves a
stale ``.tmp`` file (orphan); callers may delete orphans on startup but
are not required to.

Schema versioning
-----------------
``SCHEMA_VERSIONS`` is a central dict at the top of this module. Every
``load_*`` function compares the loaded payload's ``schema_version``
field against the central version; a mismatch raises ``ValueError`` with
an actionable message ("Delete the sidecar to regenerate"). Schema bumps
require: (1) modify the dataclass, (2) increment the integer in
``SCHEMA_VERSIONS``, (3) note the bump in ``max_quality/patches/MANIFEST.md``.

Multi-arch portability
----------------------
All ``save_*`` functions move tensor fields to CPU before serializing.
All ``torch.load`` calls pass ``map_location="cpu"``. A sidecar written
on H200 is readable on RTX 6000 Pro (or CPU-only) without a CUDA device.

Single-writer / concurrency
---------------------------
The calibration writer is single-process by contract. No file locking;
``tmp + os.replace`` protects against intra-process SIGTERM. Concurrent
writes from two processes to the same shared-file signal (``phase_b``,
``stage2_profile``, ``covariance``, ``teacher_eval``) are NOT supported.
Sharded signals (``router_kd_logits`` per attempt_idx; ``block_hidden``
per layer_idx) are naturally collision-free if each shard is written by
exactly one process.

Sidecar isolation across calibration runs
-----------------------------------------
Sidecars are collocated with the JSONL by DIRECTORY, not by JSONL stem.
Two distinct calibration runs that write to JSONLs in the same parent
directory will OVERWRITE each other's sidecars. To preserve sidecars
across runs that produce distinct JSONLs (e.g., different cache_keys),
either:

* Use distinct output directories per run, OR
* Rely on the JSONL cache_key suffix being identical across runs that
  should share sidecars (the typical case — same teacher + same prompt
  source + same num_prompts produces the same cache_key, so the
  sidecar from a prior run is the cache hit for the current run).

This is the consequence of the directory-collocation choice (see
``sidecar_path`` docstring); it is intentional and trades isolation
for resilience-to-JSONL-rename. Operators running ablation sweeps
across different cache_keys MUST use distinct output dirs.

Out of scope here
-----------------
This module is library-only. The concrete provider subclasses for the 6
signal pairs (Stage1PhaseBCacheProvider, Stage2ProfileCacheProvider, ...)
and the sidecar-writing calls in ``build_self_traces_calib_vllm.py``
land in items V1+V2 + items 1-10 of the calibration-v2 campaign.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from moe_compress.pipeline.context import PipelineContext
from moe_compress.pipeline.plugin import BasePlugin


# ---------------------------------------------------------------------------
# Schema versions -- central source of truth.
# ---------------------------------------------------------------------------
SCHEMA_VERSIONS: dict[str, int] = {
    "phase_b":             1,
    "stage2_profile":      1,
    "covariance":          2,
    "router_kd_logits":    1,
    "block_hidden":        1,
    "teacher_eval":        1,
    "reap_scores":         1,
    "per_expert_max":      1,
    "routing_stats":       1,
    "router_logits_stats": 1,
    "output_reservoir":    1,
}


# ---------------------------------------------------------------------------
# Path derivation.
# ---------------------------------------------------------------------------
def sidecar_path(jsonl_path: Path, signal_name: str, *, suffix: str = ".pt") -> Path:
    """Derive the sidecar path for a given signal.

    For atomic single-file signals (e.g., signal_name="phase_b"):
        <jsonl_path.parent>/sidecars/phase_b.pt

    For per-shard signals (signal_name contains a slash):
        <jsonl_path.parent>/sidecars/block_hidden/layer_0007.pt

    Sidecars are collocated by directory with the JSONL, NOT by filename
    stem. Renaming the JSONL does not orphan its sidecars. Moving the
    JSONL to a new directory requires moving the sidecars/ subdir.
    """
    return jsonl_path.parent / "sidecars" / (signal_name + suffix)


def router_kd_logits_dir(jsonl_path: Path) -> Path:
    """Returns <jsonl_path.parent>/sidecars/router_kd_logits/ -- the directory
    holding per-attempt-idx .npz shards."""
    return jsonl_path.parent / "sidecars" / "router_kd_logits"


# ---------------------------------------------------------------------------
# Payload dataclasses (one per signal).
# ---------------------------------------------------------------------------
@dataclass
class PhaseBPayload:
    schema_version: int
    n_experts: int
    n_layers: int
    per_expert_max: torch.Tensor          # [n_layers, n_experts] float32
    routing_freq: torch.Tensor            # [n_layers, n_experts] float32
    mean_routing_weight: torch.Tensor     # [n_layers, n_experts] float32
    output_reservoir: torch.Tensor        # [n_layers, n_experts, reservoir_size, hidden_dim]


@dataclass
class Stage2ProfilePayload:
    schema_version: int
    n_experts: int
    n_layers: int
    delta_gate: torch.Tensor              # [n_layers, n_experts, n_experts] float32
    delta_expert: torch.Tensor            # [n_layers, n_experts, n_experts] float64
    a_gate_up: torch.Tensor               # [n_layers, n_experts, intermediate_dim] float32
    a_down: torch.Tensor                  # [n_layers, n_experts, hidden_dim] float32
    token_counts: torch.Tensor            # [n_layers, n_experts] int64


@dataclass
class Stage2ReapPayload:
    """REAP per-(layer, expert) saliency scores from a calibration run.

    S_j = (1/|X_j|) · Σ g_j(x) · ‖f_j(x)‖₂  (REAP Eq. 9, arXiv:2510.13999).

    Written by the calibration writer at run end (via
    ``vllm.calibration_reap_scores.dump_reap_scores``); consumed by
    Stage 2's ``Stage2ReapScoresCacheProvider`` to skip the live
    per-layer REAP-scoring forward pass.

    Indexing convention: ``reap_scores[layer_rank, expert_id]`` where
    ``layer_rank`` is the 0-based index into the ordered MoE layer
    list (same ordering as ``iter_moe_layers``).
    """
    schema_version: int
    n_experts: int
    n_layers: int
    reap_scores: torch.Tensor    # [n_layers, n_experts] float32 — S_j
    token_counts: torch.Tensor   # [n_layers, n_experts] int64 — |X_j|


@dataclass
class Stage1PerExpertMaxPayload:
    """Per-(layer, expert) max output L_inf for Stage 1 cheap-pruning candidate ranking.

    max_j(|f_j(x)|_inf) over all tokens routed to expert j in layer rank_l
    across the calibration run. Consumed by Stage 1's ThreeWayAndPlugin,
    MagnitudeTopkPlugin, and ablation_filter to identify low-magnitude
    pruning candidates.

    Indexing: per_expert_max[layer_rank, expert_id] where layer_rank is the
    0-based ordinal index into the MoE layer list (NOT layer_idx). The
    cache reader maps rank -> layer_idx via the live MoELayerRef list.
    """
    schema_version: int
    n_experts: int
    n_layers: int
    per_expert_max: torch.Tensor   # [n_layers, n_experts] float32
    token_counts: torch.Tensor     # [n_layers, n_experts] int64


@dataclass
class RoutingStatsPayload:
    """Per-(layer, expert) routing-frequency + mean-routing-weight statistics.

    Computed over the entire calibration run from the live router's
    ``topk_ids`` + ``topk_weights``:

    * ``freq[layer_rank, expert_id]`` -- the int64 count of tokens that
      selected expert ``expert_id`` at layer rank ``layer_rank`` (each
      top-k selection counted once; a token with ``top_k=2`` that picks
      experts 3 and 7 contributes +1 to both).
    * ``mean_weight[layer_rank, expert_id]`` -- the float32 mean of the
      router weights ``g_j(x)`` over the same population (i.e.
      ``Σ topk_weight / freq`` with the per-expert weight sum tracked
      internally by the writer; zero where ``freq == 0``).

    Indexing convention: ``layer_rank`` is the 0-based ordinal into the
    MoE layer list (NOT the model's absolute ``layer_idx`` -- the cache
    reader maps rank -> layer_idx via the live ``MoELayerRef`` list
    when needed). The writer uses the same ``named_modules() ->
    moe_layer_id`` ordering as ``vllm.calibration_reap_scores`` and
    ``vllm.calibration_per_expert_max`` so all three are mutually
    consistent.

    Consumer: NONE in the live path at the moment. This payload is laid
    down as infrastructure for future plugins (e.g. routing-aware
    ablation gating, mean-weight-weighted REAP variants). The Stage 1 /
    Stage 2 cache readers shipped alongside this payload only deposit it
    onto ``ctx`` so future read-side plugins can pick it up without
    requiring a fresh schema rev.
    """
    schema_version: int
    n_experts: int
    n_layers: int
    freq: torch.Tensor          # [n_layers, n_experts] int64
    mean_weight: torch.Tensor   # [n_layers, n_experts] float32


@dataclass
class RouterLogitsStatsPayload:
    """Per-(layer, expert) sink-vs-normal router-score aggregates.

    Storage choice: aggregate stats (NOT raw per-token logits). The
    on-disk payload mirrors the finalized output of
    :class:`SinkTokenRoutingAccumulator` -- per-(layer, expert)
    ``mean_router_score_sink``, ``mean_router_score_normal``, and
    ``freq_on_sink`` -- plus the per-layer sink / normal token counts
    needed to invert the means back into sums if a downstream consumer
    wants to reweight or merge across multiple captures.

    All POST-softmax aggregates (despite the "router_logits" name -- the
    hook fires on logits, but the writer softmaxes inline before
    accumulating).

    Indexing convention: per-(layer, expert) tensors are indexed by
    ``[layer_rank, expert_id]`` where ``layer_rank`` is the 0-based
    ordinal into the MoE layer list (NOT the model's absolute
    ``layer_idx``). The Stage 1 cache reader maps rank -> layer_idx via
    the live ``MoELayerRef`` list when hydrating a
    :class:`SinkTokenRoutingAccumulator` from this payload.

    Sink definition (writer-side): a token is "sink" iff
    ``input_id == bos_token_id`` (when the writer has both the
    ``input_ids`` tensor and a non-None ``bos_token_id``). When the
    router-hook payload does not include ``input_ids`` (the current vLLM
    dispatch contract — see ``vllm/calibration_hooks.py``), the writer
    falls back to the leading-position-only convention: token at
    position 0 of the batch is sink. The fallback is documented;
    consumers that care MUST verify the writer's ``bos_token_id`` field
    is set and was used (i.e. that the upstream dispatch grew the
    ``input_ids`` kwarg).

    Consumer: :class:`Stage1RouterLogitsStatsCacheProvider` hydrates a
    pre-finalized :class:`SinkTokenRoutingAccumulator` from this payload
    into ``ctx["sink_acc"]`` -- the SAME slot the live
    ``SinkTokenDetectorPlugin.setup()`` writes -- so the downstream
    sink-token detector consumes the cached aggregates without
    rebuilding them from a router-logits pass.
    """
    schema_version: int
    n_experts: int
    n_layers: int
    score_sink_sum: torch.Tensor      # [n_layers, n_experts] float32
    score_normal_sum: torch.Tensor    # [n_layers, n_experts] float32
    fire_on_sink: torch.Tensor        # [n_layers, n_experts] int64
    n_sink_tokens: torch.Tensor       # [n_layers] int64
    n_normal_tokens: torch.Tensor     # [n_layers] int64
    bos_token_id: int | None          # may be None if not captured


@dataclass
class OutputReservoirPayload:
    """Per-(layer, expert) expert-output reservoir for Stage 1 CKA.

    Reservoir-sampled snapshot of unweighted expert outputs (the slice
    of the Triton MoE persistent buffer corresponding to a routed token,
    BEFORE the topk-weight multiply). Mirrors the live
    :class:`ExpertOutputAccumulator` (``activation_hooks.py``) finalized
    state but stored as a dense 4-D tensor so the sidecar is a single
    ``torch.save`` write.

    Storage shape: ``[n_layers, n_experts, max_tokens, hidden_dim]``
    bfloat16. Unfilled cells (``valid_count[rank, e] < max_tokens``) are
    zero-padded; the cache reader uses ``valid_count`` to slice each
    reservoir down to its truly-populated head before hydrating
    ``ExpertOutputAccumulator._finalized``.

    Indexing convention: ``[layer_rank, expert_id]`` where ``layer_rank``
    is the 0-based ordinal into the MoE layer list (NOT the absolute
    ``layer_idx``); the Stage 1 cache reader maps rank → layer_idx via
    the live ``MoELayerRef`` list when hydrating the accumulator.

    Reservoir-sampling math: identical to
    :meth:`ExpertOutputAccumulator.update` — Phase 1 fills empty slots
    sequentially while ``seen < max_tokens``; Phase 2 accepts each
    further token with probability ``max_tokens / (seen + j)`` for the
    j-th post-fill token and on accept writes to a uniformly-random
    slot (last-wins on collision). The accepted-token distribution is
    uniform across slots; statistically equivalent to sequential
    reservoir sampling.

    Storage budget: ``max_tokens=256`` × ``hidden_dim≈2048`` × 2 bytes
    (bf16) × ``n_layers≈40`` × ``n_experts≈256`` ≈ 10 GB on disk; the
    plan-doc estimate of ~17 GB bf16 is the upper-bound covering
    larger configs (Qwen3.6 has 2880 hidden, 40 layers, 256 experts →
    ~15 GB).

    Consumer: :class:`Stage1OutputReservoirCacheProvider` hydrates a
    pre-finalized :class:`ExpertOutputAccumulator` from this payload
    into ``ctx["output_acc"]`` -- the SAME slot the live Phase B
    calibration pass writes -- so the downstream CKADistancePlugin
    consumes the cached reservoirs without rebuilding them from a
    Phase B forward pass.
    """
    schema_version: int
    n_experts: int
    n_layers: int
    reservoir: torch.Tensor      # [n_layers, n_experts, max_tokens, hidden_dim] bfloat16
    valid_count: torch.Tensor    # [n_layers, n_experts] int64
    total_seen: torch.Tensor     # [n_layers, n_experts] int64
    max_tokens: int              # reservoir capacity


@dataclass
class CovariancePayload:
    """Per-(layer, expert, matrix) teacher input covariance Σ_in.

    Dict-valued storage because the actual on-disk format used by Stage
    3/4 today (``_stage2_input_covariance.pt``) is a dict keyed by
    ``(layer_idx, expert_idx, matrix_name)`` → fp16 ``Tensor[d_in, d_in]``.
    See ``max_quality/src/moe_compress/stage3/plugins/covariance_collection.py``
    for the loader contract (``_load_stage2_covariance`` returns the
    ``"covariance"`` field of the raw payload as a dict of this shape).

    Schema bumped from v1 (which had a single 4-D tensor field) to v2
    (dict of tensors) because the actual consumers (Stage 3 AA-SVD, Stage
    4 EoRA) need per-(layer, expert, matrix) keying that a single 4-D
    tensor cannot represent without a separate index mapping. v1 was
    never written to disk by any production writer; this is a forward-
    only bump.
    """
    schema_version: int
    n_experts: int
    n_layers: int
    # {(layer_idx, expert_idx, matrix_name): Tensor[d_in, d_in] fp16}
    sigma_in: dict
    # {(layer_idx, expert_idx, matrix_name): int}
    token_counts: dict


@dataclass
class RouterKDLogitsPayload:
    schema_version: int
    token_ids: np.ndarray                 # [n_tokens] int32
    top_ids: np.ndarray                   # [n_tokens, top_k] int32
    top_logprobs: np.ndarray              # [n_tokens, top_k] float32
    attempt_idx: int
    top_k: int


@dataclass
class BlockHiddenPayload:
    schema_version: int
    layer_idx: int
    n_prompts_in_subset: int
    hidden_states: torch.Tensor           # [n_tokens, hidden_dim] bfloat16


@dataclass
class TeacherEvalPayload:
    schema_version: int
    cache_key: str                        # SHA-256 from _teacher_cache_key
    teacher_results: dict
    teacher_param_counts: dict | None


# ---------------------------------------------------------------------------
# Atomic-write helpers.
# ---------------------------------------------------------------------------
def _atomic_torch_save(payload: Any, path: Path) -> None:
    """Atomic torch.save: write to tmp + os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _atomic_npz_save(arrays: dict[str, np.ndarray], path: Path) -> None:
    """Atomic np.savez_compressed: write to a tmp PATH WITHOUT the .npz
    extension, then os.replace into the final .npz path.

    CRITICAL: np.savez_compressed auto-appends '.npz' to filenames that
    don't already end in .npz. To avoid '.npz.tmp.npz' double-extension,
    we open the tmp file as a binary handle, pass the HANDLE to
    np.savez_compressed (which does NOT auto-append), then rename the
    finalized file to the final .npz path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **arrays)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Schema-mismatch error helper.
# ---------------------------------------------------------------------------
def _check_schema(signal: str, loaded_version: int, path: Path) -> None:
    expected = SCHEMA_VERSIONS[signal]
    if loaded_version != expected:
        raise ValueError(
            f"{signal} sidecar at {path} has schema_version={loaded_version}, "
            f"expected {expected}. "
            f"Delete the sidecar to regenerate."
        )


# ---------------------------------------------------------------------------
# Signal 1: phase_b (Stage 1 Phase-B accumulators).
# ---------------------------------------------------------------------------
def save_phase_b(payload: PhaseBPayload, jsonl_path: Path) -> None:
    cpu_payload = replace(
        payload,
        per_expert_max=payload.per_expert_max.detach().cpu(),
        routing_freq=payload.routing_freq.detach().cpu(),
        mean_routing_weight=payload.mean_routing_weight.detach().cpu(),
        output_reservoir=payload.output_reservoir.detach().cpu(),
    )
    _atomic_torch_save(cpu_payload, sidecar_path(jsonl_path, "phase_b"))


def load_phase_b(jsonl_path: Path) -> PhaseBPayload | None:
    path = sidecar_path(jsonl_path, "phase_b")
    if not path.exists():
        return None
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    _check_schema("phase_b", loaded.schema_version, path)
    return loaded


# ---------------------------------------------------------------------------
# Signal 2: stage2_profile (Stage 2 Delta/A accumulators).
# ---------------------------------------------------------------------------
def save_stage2_profile(payload: Stage2ProfilePayload, jsonl_path: Path) -> None:
    cpu_payload = replace(
        payload,
        delta_gate=payload.delta_gate.detach().cpu(),
        delta_expert=payload.delta_expert.detach().cpu(),
        a_gate_up=payload.a_gate_up.detach().cpu(),
        a_down=payload.a_down.detach().cpu(),
        token_counts=payload.token_counts.detach().cpu(),
    )
    _atomic_torch_save(cpu_payload, sidecar_path(jsonl_path, "stage2_profile"))


def load_stage2_profile(jsonl_path: Path) -> Stage2ProfilePayload | None:
    path = sidecar_path(jsonl_path, "stage2_profile")
    if not path.exists():
        return None
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    _check_schema("stage2_profile", loaded.schema_version, path)
    return loaded


# ---------------------------------------------------------------------------
# Signal 2b: reap_scores (Stage 2 REAP saliency per (layer, expert)).
# ---------------------------------------------------------------------------
def save_reap_scores(payload: Stage2ReapPayload, jsonl_path: Path) -> None:
    """Atomically write the Stage 2 REAP-scores sidecar.

    Tensors are moved to CPU before serialization so the sidecar is
    device-agnostic (H200 → RTX 6000 Pro round-trip is supported).
    """
    cpu_payload = Stage2ReapPayload(
        schema_version=payload.schema_version,
        n_experts=payload.n_experts,
        n_layers=payload.n_layers,
        reap_scores=payload.reap_scores.detach().to("cpu", dtype=torch.float32),
        token_counts=payload.token_counts.detach().to("cpu", dtype=torch.int64),
    )
    _atomic_torch_save(cpu_payload, sidecar_path(jsonl_path, "reap_scores"))


def load_reap_scores(jsonl_path: Path) -> Stage2ReapPayload | None:
    """Load the Stage 2 REAP-scores sidecar.

    Returns None if the sidecar does not exist (cache miss). Raises
    ValueError on schema_version mismatch with an actionable message.
    """
    path = sidecar_path(jsonl_path, "reap_scores")
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    _check_schema("reap_scores", payload.schema_version, path)
    return payload


# ---------------------------------------------------------------------------
# Signal 2c: per_expert_max (Stage 1 per-(layer, expert) down_proj output max).
# ---------------------------------------------------------------------------
def save_per_expert_max(payload: Stage1PerExpertMaxPayload, jsonl_path: Path) -> None:
    """Atomically write the Stage 1 per-expert-max sidecar.

    Tensors are moved to CPU before serialization so the sidecar is
    device-agnostic (H200 -> RTX 6000 Pro round-trip is supported).
    """
    cpu_payload = Stage1PerExpertMaxPayload(
        schema_version=payload.schema_version,
        n_experts=payload.n_experts,
        n_layers=payload.n_layers,
        per_expert_max=payload.per_expert_max.detach().to("cpu", dtype=torch.float32),
        token_counts=payload.token_counts.detach().to("cpu", dtype=torch.int64),
    )
    _atomic_torch_save(cpu_payload, sidecar_path(jsonl_path, "per_expert_max"))


def load_per_expert_max(jsonl_path: Path) -> Stage1PerExpertMaxPayload | None:
    """Load the Stage 1 per-expert-max sidecar.

    Returns None if the sidecar does not exist (cache miss). Raises
    ValueError on schema_version mismatch with an actionable message.
    """
    path = sidecar_path(jsonl_path, "per_expert_max")
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    _check_schema("per_expert_max", payload.schema_version, path)
    return payload


# ---------------------------------------------------------------------------
# Signal 2d: routing_stats (per-(layer, expert) routing freq + mean weight).
# ---------------------------------------------------------------------------
def save_routing_stats(payload: RoutingStatsPayload, jsonl_path: Path) -> None:
    """Atomically write the routing-stats sidecar.

    Tensors are moved to CPU before serialization so the sidecar is
    device-agnostic (H200 -> RTX 6000 Pro round-trip is supported).
    """
    cpu_payload = RoutingStatsPayload(
        schema_version=payload.schema_version,
        n_experts=payload.n_experts,
        n_layers=payload.n_layers,
        freq=payload.freq.detach().to("cpu", dtype=torch.int64),
        mean_weight=payload.mean_weight.detach().to("cpu", dtype=torch.float32),
    )
    _atomic_torch_save(cpu_payload, sidecar_path(jsonl_path, "routing_stats"))


def load_routing_stats(jsonl_path: Path) -> RoutingStatsPayload | None:
    """Load the routing-stats sidecar.

    Returns None if the sidecar does not exist (cache miss). Raises
    ValueError on schema_version mismatch with an actionable message.
    """
    path = sidecar_path(jsonl_path, "routing_stats")
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    _check_schema("routing_stats", payload.schema_version, path)
    return payload


# ---------------------------------------------------------------------------
# Signal 2e: router_logits_stats (per-(layer, expert) sink-vs-normal aggregates).
# ---------------------------------------------------------------------------
def save_router_logits_stats(payload: RouterLogitsStatsPayload, jsonl_path: Path) -> None:
    """Atomically write the router-logits-stats sidecar.

    Tensors are moved to CPU before serialization so the sidecar is
    device-agnostic (H200 -> RTX 6000 Pro round-trip is supported).
    """
    cpu_payload = RouterLogitsStatsPayload(
        schema_version=payload.schema_version,
        n_experts=payload.n_experts,
        n_layers=payload.n_layers,
        score_sink_sum=payload.score_sink_sum.detach().to(
            "cpu", dtype=torch.float32,
        ),
        score_normal_sum=payload.score_normal_sum.detach().to(
            "cpu", dtype=torch.float32,
        ),
        fire_on_sink=payload.fire_on_sink.detach().to(
            "cpu", dtype=torch.int64,
        ),
        n_sink_tokens=payload.n_sink_tokens.detach().to(
            "cpu", dtype=torch.int64,
        ),
        n_normal_tokens=payload.n_normal_tokens.detach().to(
            "cpu", dtype=torch.int64,
        ),
        bos_token_id=(
            int(payload.bos_token_id)
            if payload.bos_token_id is not None else None
        ),
    )
    _atomic_torch_save(cpu_payload, sidecar_path(jsonl_path, "router_logits_stats"))


def load_router_logits_stats(jsonl_path: Path) -> RouterLogitsStatsPayload | None:
    """Load the router-logits-stats sidecar.

    Returns None if the sidecar does not exist (cache miss). Raises
    ValueError on schema_version mismatch with an actionable message.
    """
    path = sidecar_path(jsonl_path, "router_logits_stats")
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    _check_schema("router_logits_stats", payload.schema_version, path)
    return payload


# ---------------------------------------------------------------------------
# Signal 2f: output_reservoir (per-(layer, expert) reservoir-sampled outputs).
# ---------------------------------------------------------------------------
def save_output_reservoir(payload: OutputReservoirPayload, jsonl_path: Path) -> None:
    """Atomically write the output-reservoir sidecar.

    The reservoir tensor is cast to bfloat16 on CPU before serialization
    (matching the storage-budget contract documented on the dataclass).
    ``valid_count`` and ``total_seen`` are cast to int64 on CPU. The
    resulting sidecar is device-agnostic (H200 -> RTX 6000 Pro round-trip
    is supported).
    """
    cpu_payload = OutputReservoirPayload(
        schema_version=payload.schema_version,
        n_experts=payload.n_experts,
        n_layers=payload.n_layers,
        reservoir=payload.reservoir.detach().to("cpu", dtype=torch.bfloat16),
        valid_count=payload.valid_count.detach().to("cpu", dtype=torch.int64),
        total_seen=payload.total_seen.detach().to("cpu", dtype=torch.int64),
        max_tokens=int(payload.max_tokens),
    )
    _atomic_torch_save(cpu_payload, sidecar_path(jsonl_path, "output_reservoir"))


def load_output_reservoir(jsonl_path: Path) -> OutputReservoirPayload | None:
    """Load the output-reservoir sidecar.

    Returns None if the sidecar does not exist (cache miss). Raises
    ValueError on schema_version mismatch with an actionable message.
    """
    path = sidecar_path(jsonl_path, "output_reservoir")
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    _check_schema("output_reservoir", payload.schema_version, path)
    return payload


# ---------------------------------------------------------------------------
# Signal 3: covariance (teacher-side sigma_in).
# ---------------------------------------------------------------------------
def save_covariance(payload: CovariancePayload, jsonl_path: Path) -> None:
    """Atomically write the per-(layer, expert, matrix) covariance sidecar.

    Every tensor inside ``payload.sigma_in`` is detached + CPU-moved + cast
    to fp16 (the persistent dtype shared with Stage 2's writer, per
    deviation D-cov-storage-fp16 in
    ``stage3/plugins/covariance_collection.py``). ``token_counts`` is a
    plain ``dict[key, int]`` and is copied as-is.
    """
    cpu_sigma = {
        k: v.detach().to("cpu", dtype=torch.float16, copy=True)
        for k, v in payload.sigma_in.items()
    }
    cpu_payload = CovariancePayload(
        schema_version=payload.schema_version,
        n_experts=payload.n_experts,
        n_layers=payload.n_layers,
        sigma_in=cpu_sigma,
        token_counts=dict(payload.token_counts),
    )
    _atomic_torch_save(cpu_payload, sidecar_path(jsonl_path, "covariance"))


def load_covariance(jsonl_path: Path) -> CovariancePayload | None:
    path = sidecar_path(jsonl_path, "covariance")
    if not path.exists():
        return None
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    _check_schema("covariance", loaded.schema_version, path)
    return loaded


# ---------------------------------------------------------------------------
# Signal 4: router_kd_logits (per-attempt-idx .npz shards).
# ---------------------------------------------------------------------------
def save_router_kd_logits(payload: RouterKDLogitsPayload, jsonl_path: Path) -> None:
    arrays = {
        "schema_version": np.int32(payload.schema_version),
        "token_ids": payload.token_ids,
        "top_ids": payload.top_ids,
        "top_logprobs": payload.top_logprobs,
        "attempt_idx": np.int64(payload.attempt_idx),
        "top_k": np.int32(payload.top_k),
    }
    path = router_kd_logits_dir(jsonl_path) / f"{payload.attempt_idx:07d}.npz"
    _atomic_npz_save(arrays, path)


def load_router_kd_logits(jsonl_path: Path, attempt_idx: int) -> RouterKDLogitsPayload | None:
    path = router_kd_logits_dir(jsonl_path) / f"{attempt_idx:07d}.npz"
    if not path.exists():
        return None
    with np.load(path) as f:
        schema = int(f["schema_version"])
        if schema != SCHEMA_VERSIONS["router_kd_logits"]:
            raise ValueError(
                f"router_kd_logits sidecar at {path} has schema_version={schema}, "
                f"expected {SCHEMA_VERSIONS['router_kd_logits']}. "
                f"Delete the sidecar to regenerate."
            )
        return RouterKDLogitsPayload(
            schema_version=schema,
            token_ids=f["token_ids"],
            top_ids=f["top_ids"],
            top_logprobs=f["top_logprobs"],
            attempt_idx=int(f["attempt_idx"]),
            top_k=int(f["top_k"]),
        )


# ---------------------------------------------------------------------------
# Signal 5: block_hidden (per-layer hidden-state cache).
# ---------------------------------------------------------------------------
def save_block_hidden(payload: BlockHiddenPayload, jsonl_path: Path) -> None:
    cpu_payload = replace(
        payload,
        hidden_states=payload.hidden_states.detach().cpu(),
    )
    path = sidecar_path(
        jsonl_path, f"block_hidden/layer_{payload.layer_idx:04d}"
    )
    _atomic_torch_save(cpu_payload, path)


def load_block_hidden(jsonl_path: Path, layer_idx: int) -> BlockHiddenPayload | None:
    path = sidecar_path(jsonl_path, f"block_hidden/layer_{layer_idx:04d}")
    if not path.exists():
        return None
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    _check_schema("block_hidden", loaded.schema_version, path)
    return loaded


# ---------------------------------------------------------------------------
# Signal 6: teacher_eval (eval-harness results keyed by SHA-256 cache_key).
# ---------------------------------------------------------------------------
def save_teacher_eval(payload: TeacherEvalPayload, jsonl_path: Path) -> None:
    # No tensor fields; payload is purely Python-typed.
    _atomic_torch_save(payload, sidecar_path(jsonl_path, "teacher_eval"))


def load_teacher_eval(jsonl_path: Path) -> TeacherEvalPayload | None:
    path = sidecar_path(jsonl_path, "teacher_eval")
    if not path.exists():
        return None
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    _check_schema("teacher_eval", loaded.schema_version, path)
    return loaded


# ---------------------------------------------------------------------------
# Provider-pair ABCs.
# ---------------------------------------------------------------------------
class BaseCacheProvider(BasePlugin, ABC):
    """Base class for cache-side provider plugins.

    Subclasses declare ``name``, ``paper``, ``config_key``, ``reads``,
    ``writes``, ``provides`` (via BasePlugin defaults or class attrs),
    plus the concrete ``on_load`` method.

    Contract:
    - ``on_load(ctx, jsonl_path)`` returns the loaded payload on hit
      (non-None), or ``None`` on miss.
    - On hit, the method also calls ``ctx.set(slot, payload)`` so
      consumer plugins can read via ``ctx.get(slot)``.
    - On miss, no ctx mutation; the registry's ``dispatch_first`` falls
      through to the live provider (which writes the same slot).
    - Load is LAZY: no I/O at construction time. ``Path.exists()`` and
      ``torch.load`` happen only inside ``on_load``.
    """

    @abstractmethod
    def on_load(self, ctx: PipelineContext, jsonl_path: Path) -> Any | None:
        ...


class BaseLiveProvider(BasePlugin, ABC):
    """Base class for live-side provider plugins.

    Subclasses wrap an existing live computation (e.g., Stage 2 profiling
    forward) AND emit a sidecar via the matching ``save_*`` so future runs
    can short-circuit through the cache provider.

    Contract:
    - ``on_load(ctx, jsonl_path)`` runs the live computation, calls the
      matching ``save_*`` to persist the sidecar, calls
      ``ctx.set(slot, payload)``, returns the payload (non-None).
    - Returning ``None`` would be an error (the registry would have no
      next provider); subclasses MUST return non-None.
    """

    @abstractmethod
    def on_load(self, ctx: PipelineContext, jsonl_path: Path) -> Any:
        ...


__all__ = [
    "SCHEMA_VERSIONS",
    "sidecar_path",
    "router_kd_logits_dir",
    "PhaseBPayload",
    "Stage2ProfilePayload",
    "Stage2ReapPayload",
    "Stage1PerExpertMaxPayload",
    "RoutingStatsPayload",
    "RouterLogitsStatsPayload",
    "OutputReservoirPayload",
    "CovariancePayload",
    "RouterKDLogitsPayload",
    "BlockHiddenPayload",
    "TeacherEvalPayload",
    "save_phase_b",
    "load_phase_b",
    "save_stage2_profile",
    "load_stage2_profile",
    "save_reap_scores",
    "load_reap_scores",
    "save_per_expert_max",
    "load_per_expert_max",
    "save_routing_stats",
    "load_routing_stats",
    "save_router_logits_stats",
    "load_router_logits_stats",
    "save_output_reservoir",
    "load_output_reservoir",
    "save_covariance",
    "load_covariance",
    "save_router_kd_logits",
    "load_router_kd_logits",
    "save_block_hidden",
    "load_block_hidden",
    "save_teacher_eval",
    "load_teacher_eval",
    "BaseCacheProvider",
    "BaseLiveProvider",
]
