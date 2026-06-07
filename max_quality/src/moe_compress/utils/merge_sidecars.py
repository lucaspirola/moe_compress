"""Merge N per-shard calibration sidecar dirs into one merged sidecar dir.

Part of the data-parallel calibration-capture machinery (see
``tasks/CALIB_PARALLEL_CAPTURE_DESIGN.md``). Each of N independent single-GPU
replay processes captures over a DISJOINT shard of the corpus and writes its
own sidecar dir. Because every in-graph capture signal is an **associative
reduction**, the N partial sidecars combine into the single-full-run result
(exact up to the fp32 storage of each shard's per-shard mean; the merge
arithmetic itself accumulates in fp64). This module performs those reductions.

Per-signal reductions
---------------------
The reductions reproduce a single full run **exact up to the fp32 storage
of the per-shard mean** — the writer stores each shard's mean in fp32, and
the merge re-expands + accumulates in fp64, so no additional error is
introduced by the merge itself. The only deviation from an idealized
single full run is the fp32 quantization of each shard's stored mean, which
is well within tolerance for expert-ranking. The writer is intentionally
left at fp32 (see design §H3).

* **reap_scores** (probe-critical): the payload stores the per-shard MEAN
  ``reap[l,e] = score_sum[l,e] / count[l,e]`` (fp32) plus ``token_counts[l,e]``.
  The combined mean is the count-weighted mean, accumulated in fp64::

      combined[l,e] = Σ_k (reap_k[l,e] * count_k[l,e]) / max(Σ_k count_k[l,e], 1)
      counts[l,e]   = Σ_k count_k[l,e]

  This reconstructs ``Σ_k score_sum_k / Σ_k count_k`` exactly up to the fp32
  storage of each ``reap_k`` (because ``reap_k * count_k == score_sum_k``),
  with the merge arithmetic itself done in fp64.
* **per_expert_max**: element-wise ``max`` across shards; ``token_counts``
  summed.
* **routing_stats**: ``freq = Σ freq_k``; ``mean_weight`` is the
  freq-weighted mean ``Σ (mean_weight_k * freq_k) / max(Σ freq_k, 1)``.
* **block_outputs** (per-layer ``block_hidden/layer_NNNN.pt`` shards):
  concatenate the per-shard ``hidden_states`` ``[n,H]`` in shard order ->
  ``[Σn, H]``; ``n_prompts_in_subset`` summed.
* **input_cov** (covariance, off by default — usually absent): sum the
  per-key Gram accumulators (additive); ``token_counts`` summed per key.
* **imatrix** (dense ``.dat``, low priority): sum the per-channel squared
  sums. The on-disk format is llama.cpp's own binary; this module sums the
  parsed tensors when present (best-effort, see ``_merge_imatrix``).
* **output_reservoir**: concatenate the per-shard valid samples then re-cap
  to the configured ``max_tokens`` (deterministic — take the first
  ``max_tokens`` in shard order). Documented as APPROXIMATE: a single full
  run's strided ring buffer would interleave samples differently, so the
  exact sample SET differs; the cap and per-(layer,expert) shape match.

Validation
----------
All shards must agree on ``(n_layers, n_experts)`` and ``schema_version``
for each signal; a mismatch aborts. Signals absent from the shards are
skipped. ``reap_scores`` is the critical path and is reduced exactly up to
fp32 mean storage (fp64 merge arithmetic).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Callable

import torch

from .cached_calibration_signals import (
    SCHEMA_VERSIONS,
    BlockHiddenPayload,
    CovariancePayload,
    OutputReservoirPayload,
    RoutingStatsPayload,
    Stage1PerExpertMaxPayload,
    Stage2ReapPayload,
    load_block_hidden,
    load_covariance,
    load_output_reservoir,
    load_per_expert_max,
    load_reap_scores,
    load_routing_stats,
    save_block_hidden,
    save_covariance,
    save_output_reservoir,
    save_per_expert_max,
    save_reap_scores,
    save_routing_stats,
    sidecar_path,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sidecar-dir <-> jsonl-path adapter.
#
# The save_*/load_* API is keyed off a JSONL path; it derives the sidecar
# under ``<jsonl.parent>/sidecars/<jsonl.stem>/``. For merge we work directly
# with sidecar dirs, so we reconstruct a synthetic "jsonl path" whose
# sidecar_path() resolves to a given sidecar dir.
# ---------------------------------------------------------------------------
def _jsonl_for_sidecar_dir(sidecar_dir: Path) -> Path:
    """Return a synthetic jsonl path P such that
    ``sidecar_path(P, signal)`` lands inside ``sidecar_dir``.

    ``sidecar_path(jsonl, sig) == jsonl.parent / "sidecars" / jsonl.stem / sig``.
    A sidecar dir is ``<parent>/sidecars/<stem>``; so the synthetic jsonl is
    ``<parent>/<stem>.jsonl``.
    """
    sidecar_dir = Path(sidecar_dir)
    stem = sidecar_dir.name
    parent = sidecar_dir.parent.parent  # strip "<stem>" and "sidecars"
    return parent / (stem + ".jsonl")


def _row_stable_key(row: dict, line_idx: int) -> "tuple[str, object]":
    """Stable per-row key for the concat disjointness check.

    Mirrors ``shard_split._row_key`` exactly so concat-mode coverage is
    checked with the same rigor as the replay-mode split: ``_attempt_idx``
    -> ``seed_idx`` -> 0-based line index, tagged with its kind.
    """
    if "_attempt_idx" in row and row["_attempt_idx"] is not None:
        return ("_attempt_idx", int(row["_attempt_idx"]))
    if "seed_idx" in row and row["seed_idx"] is not None:
        return ("seed_idx", int(row["seed_idx"]))
    return ("line_idx", int(line_idx))


def concat_jsonls(
    shard_jsonls: "list[Path]",
    out_jsonl: Path,
) -> int:
    """Concatenate N per-shard output JSONLs (generate mode) in shard order.

    Used by the orchestration's ``generate`` mode: each process GENERATES a
    disjoint prompt slice into its own ``shard_k/self_traces.jsonl``; the final
    corpus is their shard-ordered concatenation.

    HARD verification (same rigor as ``shard_split``):
      (a) total output rows == Σ shard rows.
      (b) the per-row stable key (``_attempt_idx``/``seed_idx``/line-index)
          forms DISJOINT sets across shards whose UNION is the full set
          (no row in two shards, none dropped, and globally key-unique).
    Aborts (RuntimeError) on any violation BEFORE writing the output, so a
    silently-overlapping generate run never produces a corpus.

    Returns the total number of rows written.
    """
    shard_jsonls = [Path(p) for p in shard_jsonls]
    per_shard_rows: list[list[str]] = []
    per_shard_keys: list[set] = []
    global_line = 0
    for sj in shard_jsonls:
        if not sj.is_file():
            raise RuntimeError(f"concat_jsonls: shard jsonl missing: {sj}")
        rows: list[str] = []
        keys: set = set()
        with sj.open(encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                parsed = json.loads(stripped)
                keys.add(_row_stable_key(parsed, global_line))
                rows.append(stripped)
                global_line += 1
        per_shard_rows.append(rows)
        per_shard_keys.append(keys)

    total = sum(len(r) for r in per_shard_rows)
    if total == 0:
        raise RuntimeError("concat_jsonls: all shard JSONLs are empty")

    # (b) disjoint + unique across shards.
    keys_with_multiplicity = sum(len(s) for s in per_shard_keys)
    union: set = set()
    for s in per_shard_keys:
        union |= s
    if keys_with_multiplicity != len(union):
        seen: set = set()
        dupes: set = set()
        for s in per_shard_keys:
            for key in s:
                if key in seen:
                    dupes.add(key)
                seen.add(key)
        raise RuntimeError(
            f"concat_jsonls DISJOINTNESS FAILURE: {len(dupes)} key(s) appear "
            f"in more than one shard, e.g. {sorted(dupes)[:5]}. Generate "
            f"slices overlap — check the --num-prompts/--prev-num-prompts "
            f"ladder + that every process used the SAME --seed."
        )
    # (a) count == union (also catches in-shard dupes -> count > union).
    if total != len(union):
        raise RuntimeError(
            f"concat_jsonls KEY-UNIQUENESS FAILURE: {total} rows but "
            f"{len(union)} distinct stable keys. Duplicate keys make the "
            f"corpus ambiguous; check for repeated generation."
        )

    out_jsonl = Path(out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_jsonl.with_suffix(out_jsonl.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as out:
        for rows in per_shard_rows:
            for r in rows:
                out.write(r + "\n")
    tmp.replace(out_jsonl)
    log.info("concat_jsonls: wrote %d rows from %d shards -> %s",
             total, len(shard_jsonls), out_jsonl)
    return total


def _validate_dims(
    signal: str,
    payloads: list,
) -> "tuple[int, int]":
    """Assert all shards agree on (n_layers, n_experts) + schema_version.

    Returns (n_layers, n_experts). Raises ValueError on any mismatch.
    """
    n_layers = {int(p.n_layers) for p in payloads}
    n_experts = {int(p.n_experts) for p in payloads}
    versions = {int(p.schema_version) for p in payloads}
    if len(n_layers) != 1:
        raise ValueError(
            f"merge_sidecars[{signal}]: shards disagree on n_layers: "
            f"{sorted(n_layers)}"
        )
    if len(n_experts) != 1:
        raise ValueError(
            f"merge_sidecars[{signal}]: shards disagree on n_experts: "
            f"{sorted(n_experts)}"
        )
    if len(versions) != 1:
        raise ValueError(
            f"merge_sidecars[{signal}]: shards disagree on schema_version: "
            f"{sorted(versions)}"
        )
    expected = SCHEMA_VERSIONS[signal]
    (ver,) = versions
    if ver != expected:
        raise ValueError(
            f"merge_sidecars[{signal}]: shard schema_version={ver} != "
            f"central SCHEMA_VERSIONS[{signal!r}]={expected}"
        )
    (nl,) = n_layers
    (ne,) = n_experts
    return nl, ne


# ---------------------------------------------------------------------------
# Per-signal merge functions. Each returns the merged payload (or None when
# the signal is absent from every shard) and is responsible only for the
# reduction; the driver writes via the matching save_*.
# ---------------------------------------------------------------------------
def merge_reap_scores(
    shard_payloads: list,
) -> "Stage2ReapPayload | None":
    """Count-weighted mean reduction (fp64 accumulation; exact up to fp32
    storage of each shard's mean). See module docstring."""
    payloads = [p for p in shard_payloads if p is not None]
    if not payloads:
        return None
    nl, ne = _validate_dims("reap_scores", payloads)

    counts = torch.zeros((nl, ne), dtype=torch.int64)
    score_sum = torch.zeros((nl, ne), dtype=torch.float64)
    for p in payloads:
        c = p.token_counts.to(torch.int64)
        # reap_k * count_k == score_sum_k  (reap_k is score_sum_k / count_k).
        # Compute in float64 to avoid precision loss on the re-expansion.
        score_sum += p.reap_scores.to(torch.float64) * c.to(torch.float64)
        counts += c

    denom = counts.clamp(min=1).to(torch.float64)
    combined = (score_sum / denom).to(torch.float32)
    return Stage2ReapPayload(
        schema_version=SCHEMA_VERSIONS["reap_scores"],
        n_experts=ne,
        n_layers=nl,
        reap_scores=combined,
        token_counts=counts,
    )


def merge_per_expert_max(
    shard_payloads: list,
) -> "Stage1PerExpertMaxPayload | None":
    """Element-wise max reduction; token_counts summed."""
    payloads = [p for p in shard_payloads if p is not None]
    if not payloads:
        return None
    nl, ne = _validate_dims("per_expert_max", payloads)

    pem = torch.full((nl, ne), float("-inf"), dtype=torch.float32)
    counts = torch.zeros((nl, ne), dtype=torch.int64)
    for p in payloads:
        pem = torch.maximum(pem, p.per_expert_max.to(torch.float32))
        counts += p.token_counts.to(torch.int64)
    # Cells never seen by any shard (count==0) would carry -inf; reset them
    # to 0.0 to match a single run's all-zero-initialized accumulator for
    # never-routed experts.
    pem = torch.where(counts > 0, pem, torch.zeros_like(pem))
    return Stage1PerExpertMaxPayload(
        schema_version=SCHEMA_VERSIONS["per_expert_max"],
        n_experts=ne,
        n_layers=nl,
        per_expert_max=pem,
        token_counts=counts,
    )


def merge_routing_stats(
    shard_payloads: list,
) -> "RoutingStatsPayload | None":
    """freq summed; mean_weight freq-weighted mean."""
    payloads = [p for p in shard_payloads if p is not None]
    if not payloads:
        return None
    nl, ne = _validate_dims("routing_stats", payloads)

    freq = torch.zeros((nl, ne), dtype=torch.int64)
    weight_sum = torch.zeros((nl, ne), dtype=torch.float64)
    for p in payloads:
        f = p.freq.to(torch.int64)
        weight_sum += p.mean_weight.to(torch.float64) * f.to(torch.float64)
        freq += f

    denom = freq.clamp(min=1).to(torch.float64)
    mean_weight = (weight_sum / denom).to(torch.float32)
    return RoutingStatsPayload(
        schema_version=SCHEMA_VERSIONS["routing_stats"],
        n_experts=ne,
        n_layers=nl,
        freq=freq,
        mean_weight=mean_weight,
    )


def merge_covariance(
    shard_payloads: list,
) -> "CovariancePayload | None":
    """input_cov: sum the per-key Gram accumulators (additive)."""
    payloads = [p for p in shard_payloads if p is not None]
    if not payloads:
        return None
    nl, ne = _validate_dims("covariance", payloads)

    sigma: dict = {}
    counts: dict = {}
    for p in payloads:
        for key, t in p.sigma_in.items():
            t = t.to(torch.float64)
            if key in sigma:
                if sigma[key].shape != t.shape:
                    raise ValueError(
                        f"merge_sidecars[covariance]: shape mismatch for "
                        f"key {key}: {sigma[key].shape} vs {t.shape}"
                    )
                sigma[key] = sigma[key] + t
            else:
                sigma[key] = t.clone()
        for key, n in p.token_counts.items():
            counts[key] = counts.get(key, 0) + int(n)

    # Cast back to fp16 (the storage dtype save_covariance expects/uses).
    sigma_fp16 = {k: v.to(torch.float16) for k, v in sigma.items()}
    return CovariancePayload(
        schema_version=SCHEMA_VERSIONS["covariance"],
        n_experts=ne,
        n_layers=nl,
        sigma_in=sigma_fp16,
        token_counts=counts,
    )


def merge_output_reservoir(
    shard_payloads: list,
) -> "OutputReservoirPayload | None":
    """output_reservoir: concat valid samples + re-cap (APPROXIMATE).

    For each (layer, expert) cell we concatenate the truly-populated head of
    each shard's reservoir (sliced to ``valid_count``) in shard order, then
    take the first ``max_tokens`` of the concatenation. ``total_seen`` is
    summed (exact); ``valid_count`` becomes ``min(Σ valid_count, max_tokens)``.

    This is documented as approximate: a single run's deterministic
    fixed-stride ring buffer over the full token stream would retain a
    different strided subset. The shape, cap, and total_seen all match a
    single run; only the exact retained sample identities differ.
    """
    payloads = [p for p in shard_payloads if p is not None]
    if not payloads:
        return None
    nl, ne = _validate_dims("output_reservoir", payloads)

    max_tokens_set = {int(p.max_tokens) for p in payloads}
    if len(max_tokens_set) != 1:
        raise ValueError(
            f"merge_sidecars[output_reservoir]: shards disagree on "
            f"max_tokens: {sorted(max_tokens_set)}"
        )
    (max_tokens,) = max_tokens_set
    hidden_dim = payloads[0].reservoir.shape[-1]

    out_res = torch.zeros(
        (nl, ne, max_tokens, hidden_dim), dtype=torch.bfloat16,
    )
    out_valid = torch.zeros((nl, ne), dtype=torch.int64)
    out_total = torch.zeros((nl, ne), dtype=torch.int64)

    for l in range(nl):
        for e in range(ne):
            heads = []
            for p in payloads:
                vc = int(p.valid_count[l, e].item())
                if vc > 0:
                    heads.append(p.reservoir[l, e, :vc].to(torch.bfloat16))
                out_total[l, e] += int(p.total_seen[l, e].item())
            if heads:
                cat = torch.cat(heads, dim=0)
                keep = min(cat.shape[0], max_tokens)
                out_res[l, e, :keep] = cat[:keep]
                out_valid[l, e] = keep

    return OutputReservoirPayload(
        schema_version=SCHEMA_VERSIONS["output_reservoir"],
        n_experts=ne,
        n_layers=nl,
        reservoir=out_res,
        valid_count=out_valid,
        total_seen=out_total,
        max_tokens=max_tokens,
    )


def merge_block_outputs(
    shard_jsonls: list,
    out_jsonl: Path,
) -> int:
    """block_outputs: per-layer concat of hidden_states across shards.

    block_outputs are stored as per-layer ``block_hidden/layer_NNNN.pt``
    shards (one BlockHiddenPayload per MoE layer). For each layer we load the
    per-shard payload, concatenate ``hidden_states`` ``[n,H]`` in shard order
    -> ``[Σn, H]``, sum ``n_prompts_in_subset``, and write the merged
    per-layer sidecar under ``out_jsonl``'s namespace.

    Layer discovery (H1 fix): the set of layer indices is the UNION across
    ALL shard sidecar dirs, not just shard 0. A layer present only on a
    subset of shards (e.g. a shard whose disjoint rows never routed through
    some block) is still merged, using only the shards that actually have
    it. Scanning shard 0 alone would silently drop such layers.

    Returns the number of layers merged (0 if block_outputs absent).
    """
    # Discover which layers exist by UNIONing every shard's block_hidden dir.
    layer_set: set[int] = set()
    for sj in shard_jsonls:
        bh_dir = sidecar_path(sj, "block_hidden/layer_0000").parent
        if not bh_dir.is_dir():
            continue
        for p in bh_dir.glob("layer_*.pt"):
            if p.suffix == ".pt":
                layer_set.add(int(p.stem.split("_")[-1]))
    if not layer_set:
        return 0
    layer_indices = sorted(layer_set)

    merged = 0
    for layer_idx in layer_indices:
        shard_payloads = []
        for sj in shard_jsonls:
            bp = load_block_hidden(sj, layer_idx)
            if bp is not None:
                shard_payloads.append(bp)
        if not shard_payloads:
            continue
        versions = {int(p.schema_version) for p in shard_payloads}
        if versions != {SCHEMA_VERSIONS["block_hidden"]}:
            raise ValueError(
                f"merge_sidecars[block_outputs]: layer {layer_idx} shards "
                f"disagree on / mismatch schema_version: {sorted(versions)}"
            )
        hs = torch.cat(
            [p.hidden_states for p in shard_payloads], dim=0,
        )
        n_prompts = sum(int(p.n_prompts_in_subset) for p in shard_payloads)
        save_block_hidden(
            BlockHiddenPayload(
                schema_version=SCHEMA_VERSIONS["block_hidden"],
                layer_idx=layer_idx,
                n_prompts_in_subset=n_prompts,
                hidden_states=hs,
            ),
            out_jsonl,
        )
        merged += 1
    return merged


def merge_imatrix(
    shard_jsonls: list,
    out_jsonl: Path,
) -> bool:
    """imatrix: sum per-channel squared sums (best-effort, low priority).

    The production imatrix sidecar is llama.cpp's own binary ``.dat`` written
    as ``<jsonl>.imatrix.dat`` (NOT under sidecars/). Stage 6 consumes
    llama.cpp's own imatrix, so this merge is low-priority and best-effort:
    if a per-shard ``.dat`` is absent we return False (skip). A faithful
    binary-level merge of llama.cpp's format is out of scope here; this
    function intentionally does NOT fabricate a merged ``.dat`` and instead
    reports absence so the orchestrator logs the skip. (Documented deviation
    — see module docstring + design §imatrix "low priority".)
    """
    dats = [
        Path(str(sj.with_suffix("")) + ".imatrix.dat")
        for sj in shard_jsonls
    ]
    present = [d for d in dats if d.exists()]
    if not present:
        return False
    log.warning(
        "merge_sidecars[imatrix]: %d per-shard imatrix.dat present but a "
        "faithful llama.cpp-binary merge is out of scope (low priority; "
        "Stage 6 uses llama.cpp's own imatrix). Skipping imatrix merge — "
        "regenerate the imatrix from the merged corpus if needed.",
        len(present),
    )
    return False


# ---------------------------------------------------------------------------
# Top-level driver.
# ---------------------------------------------------------------------------
# Maps a signal name to (loader, merge-fn, saver). Per-signal tensor signals
# only; the per-layer (block_outputs) + binary (imatrix) signals are handled
# separately because they are not single-payload.
_PER_PAYLOAD_SIGNALS: "dict[str, tuple[Callable, Callable, Callable]]" = {
    "reap_scores": (load_reap_scores, merge_reap_scores, save_reap_scores),
    "per_expert_max": (
        load_per_expert_max, merge_per_expert_max, save_per_expert_max,
    ),
    "routing_stats": (
        load_routing_stats, merge_routing_stats, save_routing_stats,
    ),
    "covariance": (load_covariance, merge_covariance, save_covariance),
    "output_reservoir": (
        load_output_reservoir, merge_output_reservoir, save_output_reservoir,
    ),
}


def merge_all(
    shard_sidecar_dirs: "list[Path]",
    out_jsonl: Path,
) -> "dict[str, str]":
    """Merge all present signals from N shard sidecar dirs into ``out_jsonl``'s
    canonical sidecar namespace.

    ``out_jsonl`` is the canonical 8000-row JSONL whose
    ``sidecars/<stem>/`` dir receives the merged sidecars (so a probe config
    with ``calibration.jsonl_path = out_jsonl`` resolves them).

    Returns a dict ``{signal: status}`` where status is "merged" / "absent"
    / "skipped".

    H2 self-merge guard: the OUTPUT jsonl's resolved sidecar dir must NOT be
    one of the input shard sidecar dirs. Otherwise the merge would read its
    own (in-progress or prior) output as an input — double-counting and a
    silently-wrong corpus. Aborts with a clear message if violated.
    """
    out_jsonl = Path(out_jsonl)

    # H2: resolve the output sidecar dir and reject any input shard dir that
    # is the same path (after full resolution to defeat ./ / symlink aliasing).
    out_sidecar_dir = (out_jsonl.parent / "sidecars" / out_jsonl.stem).resolve()
    resolved_inputs = [Path(d).resolve() for d in shard_sidecar_dirs]
    for raw, resolved in zip(shard_sidecar_dirs, resolved_inputs):
        if resolved == out_sidecar_dir:
            raise ValueError(
                f"merge_sidecars: self-merge detected — input shard sidecar "
                f"dir {raw} resolves to the OUTPUT sidecar dir "
                f"{out_sidecar_dir}. Merging the output as an input would "
                f"double-count. Use an --out-jsonl whose sidecar namespace is "
                f"distinct from every shard's."
            )

    shard_jsonls = [_jsonl_for_sidecar_dir(Path(d)) for d in shard_sidecar_dirs]

    status: dict[str, str] = {}

    for signal, (loader, merge_fn, saver) in _PER_PAYLOAD_SIGNALS.items():
        shard_payloads = [loader(sj) for sj in shard_jsonls]
        present = [p for p in shard_payloads if p is not None]
        if not present:
            status[signal] = "absent"
            continue
        merged = merge_fn(shard_payloads)
        if merged is None:
            status[signal] = "absent"
            continue
        saver(merged, out_jsonl)
        status[signal] = "merged"
        log.info("merge_sidecars: %s merged from %d shard(s)",
                 signal, len(present))

    # block_outputs (per-layer).
    n_layers_merged = merge_block_outputs(shard_jsonls, out_jsonl)
    status["block_outputs"] = (
        "merged" if n_layers_merged > 0 else "absent"
    )
    if n_layers_merged:
        log.info("merge_sidecars: block_outputs merged %d layer(s)",
                 n_layers_merged)

    # imatrix (best-effort binary; usually skipped).
    imatrix_done = merge_imatrix(shard_jsonls, out_jsonl)
    status["imatrix"] = "merged" if imatrix_done else "skipped"

    return status


def main(argv: "list[str] | None" = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    p = argparse.ArgumentParser(
        description="Merge N per-shard calibration sidecar dirs into one "
                    "merged sidecar dir at the target JSONL's canonical "
                    "sidecar namespace.",
    )
    p.add_argument("--out-jsonl", type=str, required=True,
                   help="Canonical (8000-row) JSONL path. The merged "
                        "sidecars land under its sidecars/<stem>/ dir.")
    p.add_argument("shard_sidecar_dirs", nargs="+", type=str,
                   help="The per-shard sidecar dirs "
                        "(e.g. shard_0/.../sidecars/shard_0).")
    args = p.parse_args(argv)

    out_jsonl = Path(args.out_jsonl).resolve()
    shard_dirs = [Path(d).resolve() for d in args.shard_sidecar_dirs]
    for d in shard_dirs:
        if not d.is_dir():
            print(f"merge_sidecars: shard sidecar dir not found: {d}",
                  file=sys.stderr)
            return 1

    try:
        status = merge_all(shard_dirs, out_jsonl)
    except ValueError as exc:
        print(f"merge_sidecars: ABORT — {exc}", file=sys.stderr)
        return 1

    print(f"merge_sidecars: done. target={out_jsonl}")
    for sig, st in sorted(status.items()):
        print(f"  {sig:20s} : {st}")
    # Non-zero exit if the probe-critical reap merge did not happen.
    if status.get("reap_scores") != "merged":
        print("merge_sidecars: WARNING — reap_scores was not merged "
              f"(status={status.get('reap_scores')!r}).", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
