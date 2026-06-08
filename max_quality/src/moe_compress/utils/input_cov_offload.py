"""Streaming disk-offload for windowed input_covariance (Gram) capture.

Fixes the RAM bomb in the old offload driver, which accumulated every window's
Gram into an unbounded host dict (``cpu_sigma``) AND re-serialized the whole
growing dict every window (~160 GB host RAM on a big model). Here each layer's
Gram is cast to fp16 and written straight to a per-layer shard on disk the
moment it is captured, then freed -- so host RAM is bounded by ONE layer's Gram
regardless of layer count. The final ``covariance.pt`` is stream-assembled from
the shards (one shard resident at a time), emitting the exact existing contract
(``CovariancePayload`` -> ``sidecars/<stem>/covariance.pt``, fp16 ``sigma_in``,
raw ``token_counts``, manifest-last).

Crash-safety: each shard is written atomically (tmp + fsync + replace +
fsync-parent), so a kill loses at most the in-flight layer; ``scan_done_layers``
drives resume. The final write is itself atomic + manifest-last, so a crash
during assembly never leaves a half-valid sidecar a consumer would accept.

Pure / GPU-free helpers (unit-testable without vLLM); the driver wires them in.
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch

from moe_compress.utils.atomic_io import atomic_torch_save
from moe_compress.utils.cached_calibration_signals import (
    CovariancePayload,
    SCHEMA_VERSIONS,
    _write_payload_and_manifest,
    sidecar_path,
)

log = logging.getLogger(__name__)

_SHARD_SCHEMA = 1


def staging_dir(replay_jsonl: Path, staging_override: str | None = None) -> Path:
    """Per-run staging dir for per-layer Gram shards.

    Defaults to a sibling of the final ``covariance.pt`` (same filesystem, so
    the run's data volume -- not the OS root -- holds the shards). Pass
    ``staging_override`` to place shards on an explicit large volume.
    """
    if staging_override:
        return Path(staging_override)
    return sidecar_path(replay_jsonl, "covariance").parent / "_covariance_staging"


def shard_path(staging: Path, layer_idx: int) -> Path:
    return staging / f"layer_{int(layer_idx):05d}.pt"


def write_layer_shard(
    staging: Path,
    layer_idx: int,
    cov_layer: torch.Tensor,
    counts_layer: torch.Tensor,
    matrix_name: str = "gate_proj",
) -> int:
    """Atomically write ONE layer's Gram as an fp16 shard.

    ``cov_layer`` [E, d, d] (any device/dtype), ``counts_layer`` [E] int. Only
    experts with count > 0 are written (matching the old snapshot guard). The
    caller frees ``cov_layer`` after this returns -- nothing is retained in
    process memory, so peak host RAM is one layer's Gram. Returns the number of
    experts written.
    """
    staging.mkdir(parents=True, exist_ok=True)
    cov_cpu = cov_layer.detach().to("cpu")
    cnt_cpu = counts_layer.detach().to("cpu")
    n_e = int(cov_cpu.shape[0])
    sigma: dict[int, torch.Tensor] = {}
    counts: dict[int, int] = {}
    for e in range(n_e):
        c = int(cnt_cpu[e].item())
        if c <= 0:
            continue
        sigma[e] = cov_cpu[e].to(torch.float16).clone()
        counts[e] = c
    payload = {
        "schema": _SHARD_SCHEMA,
        "layer_idx": int(layer_idx),
        "matrix_name": str(matrix_name),
        "n_experts": n_e,
        "sigma": sigma,     # {expert: Tensor[d, d] fp16}
        "counts": counts,   # {expert: int}
    }
    atomic_torch_save(shard_path(staging, layer_idx), payload)
    return len(sigma)


def scan_done_layers(staging: Path) -> set[int]:
    """Layer ids whose shard is durably present (drives resume). A shard is
    written atomically, so presence == complete."""
    if not staging.is_dir():
        return set()
    done: set[int] = set()
    for p in staging.glob("layer_*.pt"):
        try:
            done.add(int(p.stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return done


def assemble_covariance(
    staging: Path,
    replay_jsonl: Path,
    n_experts: int,
    n_layers: int,
) -> int:
    """Stream-merge per-layer fp16 shards into the single canonical
    ``covariance.pt`` (byte-compatible with ``save_covariance``), bounded
    memory: one shard resident at a time, tensors already fp16 (no re-clone, so
    no transient fp32 doubling). Returns the (layer, expert) entry count.
    """
    shard_files = sorted(staging.glob("layer_*.pt"))
    if not shard_files:
        raise FileNotFoundError(f"no Gram shards under {staging}")
    sigma_in: dict = {}
    token_counts: dict = {}
    for sf in shard_files:
        shard = torch.load(sf, map_location="cpu", weights_only=False)
        li = int(shard["layer_idx"])
        mat = str(shard.get("matrix_name", "gate_proj"))
        for e, t in shard["sigma"].items():
            sigma_in[(li, int(e), mat)] = t          # already fp16 CPU
        for e, c in shard["counts"].items():
            token_counts[(li, int(e), mat)] = int(c)
        del shard
    payload = CovariancePayload(
        schema_version=SCHEMA_VERSIONS["covariance"],
        n_experts=int(n_experts),
        n_layers=int(n_layers),
        sigma_in=sigma_in,
        token_counts=token_counts,
    )
    _write_payload_and_manifest(
        payload,
        sidecar_path(replay_jsonl, "covariance"),
        signal_name="covariance",
    )
    return len(sigma_in)
