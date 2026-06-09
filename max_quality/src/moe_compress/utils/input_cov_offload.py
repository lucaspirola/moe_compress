"""Streaming disk-offload for windowed input_covariance (Gram) capture.

Fixes the RAM bomb in the old offload driver, which accumulated every window's
Gram into an unbounded host dict (``cpu_sigma``) AND re-serialized the whole
growing dict every window (~160 GB host RAM on a big model). Here each layer's
Gram is cast to fp16 and written straight to a per-layer shard on disk the
moment it is captured, then freed -- so DURING CAPTURE host RAM is bounded by
ONE layer's Gram regardless of layer count (the RAM-bomb fix). The final
``covariance.pt`` is assembled from the shards, emitting the exact existing
contract (``CovariancePayload`` -> ``sidecars/<stem>/covariance.pt``, fp16
``sigma_in``, raw ``token_counts``, manifest-last). NOTE: assembly itself is
NOT memory-bounded -- the single-file consumer contract forces the full fp16
dict into RAM before the write; see ``assemble_covariance``.

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
    ``staging_override`` to place shards on an explicit large volume; a fixed
    ``_covariance_staging`` subdirectory is always appended so a fresh-run wipe
    (``rmtree``) never deletes the override path itself (e.g. a shared mount).
    """
    if staging_override:
        return Path(staging_override) / "_covariance_staging"
    return sidecar_path(replay_jsonl, "covariance").parent / "_covariance_staging"


def shard_path(
    staging: Path, layer_idx: int, matrix_name: str = "gate_proj"
) -> Path:
    """Per-(layer, matrix) shard path. The ``matrix_name`` suffix is what lets
    gate_proj and down_proj Grams for the SAME layer coexist (without it the
    second write would clobber the first -- the ride-along collision)."""
    return staging / f"layer_{int(layer_idx):05d}_{matrix_name}.pt"


def write_layer_shard(
    staging: Path,
    layer_idx: int,
    cov_layer: torch.Tensor,
    counts_layer: torch.Tensor,
    matrix_name: str = "gate_proj",
) -> int:
    """Atomically write ONE layer's Gram as a bf16 shard.

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
        # bf16 (NOT fp16): the Gram is a RAW un-normalized sum of x⊗x over
        # ~10^5-10^6 tokens/expert; those sums exceed fp16's 65504 ceiling and
        # saturate to +inf (this corrupted the 2026-06-09 capture -- ~38% of
        # variance diagonals went inf). bf16 shares fp32's exponent range so it
        # cannot overflow here; the consumer eigendecomposes in fp32 and is
        # scale-invariant, so the extra mantissa rounding is harmless. copy=True:
        # one bf16 copy that never aliases cov_cpu (so the slice doesn't keep the
        # whole layer tensor alive after we free it).
        sigma[e] = cov_cpu[e].to(torch.bfloat16, copy=True)
        counts[e] = c
    payload = {
        "schema": _SHARD_SCHEMA,
        "layer_idx": int(layer_idx),
        "matrix_name": str(matrix_name),
        "n_experts": n_e,
        "sigma": sigma,     # {expert: Tensor[d, d] bf16}
        "counts": counts,   # {expert: int}
    }
    atomic_torch_save(shard_path(staging, layer_idx, matrix_name), payload)
    return len(sigma)


def scan_done_layers(staging: Path) -> set[tuple[int, str]]:
    """``(layer_idx, matrix_name)`` pairs whose shard is durably present (drives
    resume). A shard is written atomically, so presence == complete. The shard
    filename is ``layer_<idx>_<matrix>.pt``; a legacy 2-field ``layer_<idx>.pt``
    name (pre matrix-aware shards) is read as ``gate_proj`` for back-compat.

    Returns ``(layer, matrix)`` pairs -- NOT bare layer ids -- so a ride-along
    resume can tell "layer N has gate but not yet down" and re-run that window.
    """
    if not staging.is_dir():
        return set()
    done: set[tuple[int, str]] = set()
    for p in staging.glob("layer_*.pt"):
        parts = p.stem.split("_")
        try:
            idx = int(parts[1])
        except (IndexError, ValueError):
            continue
        mat = "_".join(parts[2:]) or "gate_proj"
        done.add((idx, mat))
    return done


def assemble_covariance(
    staging: Path,
    replay_jsonl: Path,
    n_experts: int,
    n_layers: int,
) -> int:
    """Merge per-layer fp16 shards into the single canonical ``covariance.pt``
    (byte-compatible with ``save_covariance``). Returns the (layer, expert)
    entry count.

    MEMORY: this is NOT bounded -- the consumer contract is a single
    ``torch.load`` of one ``covariance.pt``, so the full assembled ``sigma_in``
    dict (~``n_layers * n_experts * d_in**2 * 2`` bytes, e.g. ~80 GB on a large
    model) is resident in host RAM before the final write. Shards are read one
    at a time and are already fp16 (no fp32 re-clone, halving peak vs reusing
    ``save_covariance``), but the assembled dict is not freed until written.
    Run on a box with enough RAM, or as a separate fresh process after the
    capturing vLLM has exited (see the ``--input-cov-staging-dir`` /
    assemble-only path). True bounded assembly would require a sharded final
    layout that Stage 3/4's ``load_covariance`` does not yet support.
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
            sigma_in[(li, int(e), mat)] = t          # already bf16 CPU
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
