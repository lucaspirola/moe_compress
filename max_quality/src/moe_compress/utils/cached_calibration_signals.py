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
    "phase_b":          1,
    "stage2_profile":   1,
    "covariance":       2,
    "router_kd_logits": 1,
    "block_hidden":     1,
    "teacher_eval":     1,
    "reap_scores":      1,
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
