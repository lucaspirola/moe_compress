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
Every ``save_*`` delegates to :mod:`moe_compress.utils.atomic_io`
(Pattern O — atomic-write + manifest-last). The dance is
``tmp + fsync(fd) + os.replace + fsync(parent_dir)`` — not a bare
``tmp + os.replace``. A SIGKILL between the last write() and the next
pdflush cycle on ext4 would otherwise leave the renamed file with
stale/garbage blocks; the explicit fsync covers that gap (MEDIUM-4
docstring refresh; the F-H-3 fix landed in :func:`_atomic_torch_save`
and :func:`_atomic_npz_save` is now thin shims around the shared
helpers in :mod:`utils.atomic_io`). A kill BETWEEN ``torch.save``
completing and ``os.replace`` leaves a stale ``.tmp`` file (orphan);
callers may delete orphans on startup but are not required to.

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

Sidecar isolation across calibration runs (post-F-H-7)
------------------------------------------------------
Sidecars are namespaced by JSONL **stem** under
``<jsonl_path.parent>/sidecars/<jsonl.stem>/`` (see :func:`sidecar_path`).
Two distinct calibration runs that produce different JSONL stems in the
same parent directory (e.g. ablation sweeps under ``artifacts/_shared/``)
NO LONGER overwrite each other's sidecars; each stem owns its own
sidecar subdirectory. MEDIUM-5 docstring refresh: the prior overwrite
behaviour was fixed by F-H-7 (commit 47dbe0d).

Backward compat: legacy non-namespaced sidecars
(``<jsonl.parent>/sidecars/<signal>.pt``) from pre-F-H-7 runs are
consulted as a fallback by :func:`_resolve_sidecar_for_load` AND ONLY
WHEN the parent directory contains exactly one JSONL stem (in-flight
``.jsonl.tmp`` files also count toward the stem disambiguation). A
single-stem legacy hit logs a one-shot WARNING (deduped via
``_already_warned_legacy_paths``); a multi-stem legacy fallback logs
an ERROR and refuses to consume (operator must manually migrate or
delete the legacy file). New writes ALWAYS land at the new namespaced
path; legacy paths are read-only.

Out of scope here
-----------------
This module is library-only. The concrete provider subclasses for the 6
signal pairs (Stage1PhaseBCacheProvider, Stage2ProfileCacheProvider, ...)
and the sidecar-writing calls in ``build_self_traces_calib_vllm.py``
land in items V1+V2 + items 1-10 of the calibration-v2 campaign.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .atomic_io import (
    ManifestMismatchError,
    atomic_npz_save as _shared_atomic_npz_save,
    atomic_torch_save as _shared_atomic_torch_save,
    read_and_validate_manifest,
    write_manifest_last,
)
from moe_compress.pipeline.context import PipelineContext
from moe_compress.pipeline.plugin import BasePlugin

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema versions -- central source of truth.
# ---------------------------------------------------------------------------
SCHEMA_VERSIONS: dict[str, int] = {
    "phase_b":             1,
    # stage2_profile bumped 1 → 3 (skip 2 to signal clean break from the
    # deleted prior Plugin #12 v1 writer; see Stage2ProfilePayloadV3
    # docstring). Pattern K applies forward (v3 → v4 should preserve
    # readers when only optional fields are added), but the v1 → v3 bump
    # is intentionally NOT forward-compatible — the v1 dataclass was
    # never written by a production writer, so no callers exist.
    "stage2_profile":      3,
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

    F-H-7 fix: sidecars are now namespaced by the JSONL filename STEM,
    not just the parent directory, so two distinct calibration runs
    that produce different JSONLs in the same parent directory
    (e.g. ablation sweeps under ``artifacts/_shared/``) do NOT overwrite
    each other's sidecars.

    For atomic single-file signals (e.g., signal_name="phase_b"):
        <jsonl_path.parent>/sidecars/<jsonl.stem>/phase_b.pt

    For per-shard signals (signal_name contains a slash):
        <jsonl_path.parent>/sidecars/<jsonl.stem>/block_hidden/layer_0007.pt

    Backward compat: the legacy non-namespaced path
    (``<jsonl.parent>/sidecars/<signal>.pt``) is consulted as a
    fallback by :func:`_legacy_sidecar_path` so existing sidecars from
    pre-F-H-7 runs continue to load — see ``load_*`` functions which
    check the new path first, then warn-and-fall-back to the legacy
    path. New writes ALWAYS land at the new path.
    """
    return jsonl_path.parent / "sidecars" / jsonl_path.stem / (signal_name + suffix)


def _legacy_sidecar_path(
    jsonl_path: Path, signal_name: str, *, suffix: str = ".pt",
) -> Path:
    """Pre-F-H-7 sidecar path (no JSONL-stem namespace).

    Consulted by load_* functions as a backward-compat fallback when the
    new namespaced path is missing. New writes are NEVER directed here.
    """
    return jsonl_path.parent / "sidecars" / (signal_name + suffix)


def router_kd_logits_dir(jsonl_path: Path) -> Path:
    """Returns <jsonl_path.parent>/sidecars/<jsonl.stem>/router_kd_logits/
    — the directory holding per-attempt-idx .npz shards (F-H-7 namespaced).
    """
    return jsonl_path.parent / "sidecars" / jsonl_path.stem / "router_kd_logits"


def _legacy_router_kd_logits_dir(jsonl_path: Path) -> Path:
    """Pre-F-H-7 router_kd_logits dir (no JSONL-stem namespace)."""
    return jsonl_path.parent / "sidecars" / "router_kd_logits"


# HIGH-4 fix: dedupe set so the one-shot WARNING docstring contract holds.
# Each (legacy_path, signal_name) pair logs at WARNING level exactly once
# per process. The set is module-level (no LRU eviction); legacy-path
# count is bounded by the number of distinct sidecar signals × distinct
# JSONL parents — well under 100 in production, so unbounded growth is
# not a concern. Tests that need a fresh process state can call
# ``_already_warned_legacy_paths.clear()`` between cases.
_already_warned_legacy_paths: set[tuple[Path, str]] = set()


# S-1 Pattern O rollout: dedupe set for the "missing MANIFEST.json"
# back-compat WARNING. Keyed on (payload_path, signal_name) so a
# 40-shard block_hidden artifact emits one WARN per shard (not one
# global). Mirrors ``_already_warned_legacy_paths`` semantics — module-
# level, no LRU eviction. The set is process-local; restarted
# calibration jobs re-emit the WARN once on the first read after
# restart. Tests can call ``.clear()`` for fresh state.
_already_warned_missing_manifest: set[tuple[Path, str]] = set()


def _manifest_path_for(payload_path: Path) -> Path:
    """Sibling manifest path for a calibration sidecar.

    Convention: append ``.MANIFEST.json`` AFTER the payload's full suffix so
    the manifest sorts alphabetically right after the payload. This matches
    the Stage 3 originals (``orchestrator.py:472``), Stage 5 teacher logits
    (``precompute_teacher_logits.py``), and ``wanda_intra_expert_score.py:564``
    sidecar conventions.

    NOTE: these 10 sidecars are local-only caches under the artifacts dir;
    they are NOT in ``utils/hub_upload.py::_STAGE_LAYOUT`` (which only lists
    ``_stage2_input_covariance.pt`` for Stage 2 and the Stage 3 originals .pt
    + its manifest for Stage 3) nor in ``hf_jobs/entrypoint.py::aux_files``
    (which carries only the Stage 2 covariance, Stage 3 originals + manifest,
    and calibration_imatrix.* tooling). The naming convention is followed for
    cross-module consistency with the staged-upload sidecars, NOT because any
    current code path Hub-uploads these calibration sidecars.
    """
    return Path(str(payload_path) + ".MANIFEST.json")


def _validate_manifest_or_warn(
    payload_path: Path,
    *,
    expected_schema_version: int,
    signal_name: str,
) -> None:
    """Pattern O validation with one-shot back-compat WARN fallback.

    Behavior:
    * Manifest exists → ``read_and_validate_manifest``. On failure, raise
      RuntimeError with the canonical "delete + re-run calibration"
      message and chain the ManifestMismatchError as cause.
    * Manifest absent → log a one-shot WARNING (deduped via
      ``_already_warned_missing_manifest`` keyed on
      ``(payload_path, signal_name)``) and return; the caller's existing
      in-payload schema check is the only torn-write guard for these
      legacy sidecars.

    The fallback exists so this fix does NOT invalidate calibration
    artifacts already on disk (~tens of GB per artifacts dir). Operators
    re-running calibration will get a manifest automatically on the next
    save_*; on the run after that, the warning will stop firing.

    NOTE: ``_already_warned_missing_manifest`` is process-local; a
    restarted calibration job (rented GPU killed and respawned) re-emits
    the WARN once per (payload, signal) on the first read after restart.
    Matches ``_already_warned_legacy_paths`` semantics — operators
    should NOT file repeated WARN-on-restart as a regression.

    MEDIUM-8 TODO(post-2026-Q3): drop the fallback once all in-flight
    sidecars under ``/opt/output/*`` have been regenerated. The horizon
    is intentionally identical to the Stage 4 / Stage 5 fallbacks
    (eora_inputs.py:230, teacher.py:301).
    """
    manifest_path = _manifest_path_for(payload_path)
    if manifest_path.exists():
        try:
            read_and_validate_manifest(
                payload_path,
                manifest_path,
                expected_schema_version=expected_schema_version,
            )
        except ManifestMismatchError as exc:
            raise RuntimeError(
                f"Calibration sidecar manifest validation FAILED for "
                f"signal={signal_name!r}: {exc}. Delete both "
                f"{payload_path.name} and {manifest_path.name} from "
                f"{payload_path.parent} and re-run calibration."
            ) from exc
        return
    _key = (payload_path, signal_name)
    if _key not in _already_warned_missing_manifest:
        log.warning(
            "Pattern O back-compat: sidecar %s has no MANIFEST.json "
            "(pre-S1 calibration run?). Proceeding with in-payload schema "
            "check only; next save_* call will emit a manifest. "
            "MEDIUM-8 TODO(post-2026-Q3): remove this fallback once all "
            "in-flight calibration sidecars have been regenerated.",
            payload_path,
        )
        _already_warned_missing_manifest.add(_key)


def _write_payload_and_manifest(
    payload: Any,
    path: Path,
    *,
    signal_name: str,
) -> None:
    """Atomic payload write + manifest-last for a calibration sidecar.

    Pattern O write protocol:
    1. Unlink any stale ``<path>.MANIFEST.json`` so a torn re-write
       leaves NO manifest at all (mirrors ``stage3/orchestrator.py:475``
       and ``wanda_intra_expert_score.py:564`` precedents). Wrapped in
       ``try/except OSError`` per those precedents.
    2. Atomic torch.save the payload (tmp + fsync + os.replace +
       fsync(parent)).
    3. Write the manifest LAST, after the payload's data blocks are
       durable on disk.

    ``compute_sha256=False`` is intentional: the size + schema_version
    cross-check inside ``read_and_validate_manifest`` is the validation
    budget for these multi-GB sidecars; SHA-256 over them would add
    minutes per calibration run. See plan §3.4 SHA-256 policy.
    """
    manifest_path = _manifest_path_for(path)
    try:
        manifest_path.unlink(missing_ok=True)
    except OSError:
        pass
    _atomic_torch_save(payload, path)
    write_manifest_last(
        path,
        manifest_path,
        schema_version=SCHEMA_VERSIONS[signal_name],
        extra_meta={"artifact": signal_name},
        compute_sha256=False,
    )


def _resolve_sidecar_for_load(
    jsonl_path: Path, signal_name: str, *, suffix: str = ".pt",
) -> Path | None:
    """Resolve a sidecar path for READING with F-H-7 backward compat.

    Returns the new-style namespaced path if it exists. If not, checks
    the legacy non-namespaced path and returns it (with a one-shot
    WARNING — see :data:`_already_warned_legacy_paths` for dedupe state)
    when present — but ONLY if there is exactly one JSONL stem living
    in the parent directory (otherwise the legacy file is ambiguous
    between multiple runs and we refuse to consume it; in-flight
    ``.jsonl.tmp`` files also participate in the multi-stem count).
    Returns None if neither path exists.
    """
    new_path = sidecar_path(jsonl_path, signal_name, suffix=suffix)
    if new_path.exists():
        return new_path
    legacy = _legacy_sidecar_path(jsonl_path, signal_name, suffix=suffix)
    if not legacy.exists():
        return None
    # Disambiguate: if the JSONL's parent dir contains exactly one .jsonl
    # file, the legacy sidecar unambiguously belongs to that run. If
    # multiple JSONLs are present, refuse to consume — operator must
    # migrate sidecars manually (mv to the new namespaced layout) or
    # re-run the calibration.
    parent = jsonl_path.parent
    jsonls = [p for p in parent.glob("*.jsonl") if p.is_file()]
    # The .tmp variant during an in-flight resume also counts as the
    # owning JSONL.
    jsonls_tmp = [p for p in parent.glob("*.jsonl.tmp") if p.is_file()]
    n_distinct_runs = len({p.stem for p in jsonls + jsonls_tmp})
    if n_distinct_runs > 1:
        log.error(
            "F-H-7: legacy sidecar %s exists but %s contains %d distinct "
            "JSONL stems (%s) — legacy layout is ambiguous across runs. "
            "Move the legacy sidecar to %s manually. Note: in-flight "
            "``.jsonl.tmp`` files also count toward the multi-stem "
            "disambiguation; either complete or delete them first. "
            "Alternatively delete the legacy sidecar to force live "
            "recomputation.",
            legacy, parent, n_distinct_runs,
            sorted({p.stem for p in jsonls + jsonls_tmp}),
            new_path,
        )
        return None
    # HIGH-4: one-shot dedupe on (legacy_path, signal_name) so the
    # docstring's "one-shot WARNING" contract holds — without this the
    # WARNING was emitted on EVERY load_* call, spamming the log on
    # provider-pair caches that read sidecars per layer.
    _key = (legacy, signal_name)
    if _key not in _already_warned_legacy_paths:
        log.warning(
            "F-H-7 backward-compat: loading sidecar from legacy path %s "
            "(pre-F-H-7 layout). Next save_* call will write to the new "
            "namespaced path %s — consider deleting the legacy file once "
            "the run completes.",
            legacy, new_path,
        )
        _already_warned_legacy_paths.add(_key)
    return legacy


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
class Stage2ProfilePayloadV3:
    """Stage 2 profile-pass sidecar payload (schema v3) — Optimization A REDO.

    Replaces the deleted prior :class:`Stage2ProfilePayload` (v1) with the
    REDO schema described in PLAN_PLUGIN_12_opt_a_redo.md §3. Every field
    is keyed by ``layer_rank`` (0-based ordinal into the MoE layer list),
    NOT absolute ``layer_idx``; the reader translates rank → layer_idx on
    hydration. This makes the sidecar portable across models with
    different dense-prefix layer counts.

    Fields:
        format_version: constant ``3`` — distinguishes from old v1.
        schema_version: ``3`` — checked by ``load_stage2_profile_v3``.
        model_hash: SHA-256 of model name + config (cross-validation).
        n_layers: number of MoE layers.
        n_experts: routed experts per layer.
        top_k: top-k routing (cross-validated).
        cov_storage_dtype: one of {"float16","bfloat16","float32"} —
            cross-validated against the run's
            ``s2.covariance_storage_dtype`` setting at load time.
        total_tokens_per_layer: [n_layers] int64 — Σ_b T_b per layer
            (independent of routing activity; Bug #3 fix).
        gate_logit_profiles: ``dict[layer_rank → list[(offset, Tensor)]]``
            — raw per-batch gate logits, preserved verbatim from the
            live ``ReamCostAccumulator.gate_logit_profiles`` storage
            (Bug #2 fix).
        sim_tensor: [n_layers, E, E] fp64 — Σ_t cos(g_i[t], g_j[t]) over
            jointly-active tokens (Bug #1 fix: per-token pair cosines,
            NOT cos(mean_i, mean_j)).
        neuron_act_sum / neuron_act_count: per-(layer_rank, expert) mean
            intermediate activations for C_act neuron alignment.
        cov_acc: dict[(layer_rank, expert_idx, matrix_name) →
            Tensor[d_in, d_in] in ``cov_storage_dtype``] — FINALIZED
            input covariance (post ``finalize_layer``). ``matrix_name`` ∈
            {"gate_proj", "down_proj"}; up_proj is aliased to gate_proj.
        cov_token_count: dict[(layer_rank, expert_idx, matrix_name) →
            int] — token count per cov entry.
        layer_input_reservoir: ``list[Tensor[N, hidden] bf16]`` of length
            ``n_layers`` — per-rank layer-input samples for SC strategy's
            ``_output_space_cost``. Always captured when the sidecar is
            written (no sub-flag; see plan §6 / OQ-2 resolution).
    """
    format_version: int                     # = 3
    schema_version: int                     # = 3
    model_hash: str
    n_layers: int
    n_experts: int
    top_k: int
    cov_storage_dtype: str                  # one of {"float16","bfloat16","float32"}
    total_tokens_per_layer: torch.Tensor    # [n_layers] int64
    gate_logit_profiles: dict               # dict[int → list[tuple[int, Tensor[T_b, E] fp32]]]
    sim_tensor: torch.Tensor                # [n_layers, E, E] fp64
    neuron_act_sum: dict                    # {(layer_rank, expert_idx): Tensor[d_int] fp32}
    neuron_act_count: dict                  # {(layer_rank, expert_idx): int}
    cov_acc: dict                           # {(layer_rank, expert_idx, matrix_name): Tensor[d, d] in cov_storage_dtype}
    cov_token_count: dict                   # {(layer_rank, expert_idx, matrix_name): int}
    layer_input_reservoir: list             # list[Tensor[N, hidden] bf16] (len == n_layers)


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
    """Canonical in-memory shape for a teacher-harness cache entry.

    OPEN-QUESTION resolution (Sequence 2, MEDIUM-2; resolved 2026-05-29
    via option (b) variant -- *adapter*, not *writer*): Stage 6's
    on-disk JSON format is unchanged. The Stage 6 writer at
    ``stage6/plugins/teacher_provider.py`` (``_save_teacher_cache`` /
    ``_load_teacher_cache``) remains the source of truth for that
    cache file's layout:

    * Storage location: still ``s6.teacher_eval_cache.cache_path``
      (operator-configurable, default
      ``<artifacts_dir>/teacher_eval_cache.json``).
    * Format: still a human-readable JSON file with strong fsync --
      operator inspectability is preserved.
    * Field naming: the JSON's ``format_version`` is adapted onto
      this dataclass's ``schema_version`` at read time.

    What landed: a READ-PATH adapter, :meth:`from_legacy_json`, that
    parses Stage 6's JSON cache into a properly-typed
    ``TeacherEvalPayload`` instance. New consumers that need the
    typed shape (e.g. cross-stage code that wants to read Stage 6's
    teacher numbers without depending on stage6/plugins internals)
    should use ``TeacherEvalPayload.from_legacy_json(path)`` instead
    of parsing the JSON directly. The classmethod validates
    ``format_version`` against the central ``SCHEMA_VERSIONS`` table
    and raises ``ValueError`` with the actionable "Delete the
    sidecar to regenerate" message on mismatch (Pattern C).

    Note that this dataclass remains *canonical only for reads*:
    Stage 6 owns its on-disk JSON format and its writer. No new
    writer from this module lands; the legacy
    :func:`load_teacher_eval` (torch-pt sidecar reader) is
    retained for backward compat with any sidecars an operator may
    have written before the resolution (see NIT-5).
    """

    schema_version: int
    cache_key: str                        # SHA-256 from _teacher_cache_key
    teacher_results: dict
    teacher_param_counts: dict | None

    @classmethod
    def from_legacy_json(cls, path: Path) -> "TeacherEvalPayload":
        """Adapt a Stage 6 ``teacher_eval_cache.json`` into a typed payload.

        Stamp: 2026-05-29 (Pattern H clean-room -- MEDIUM-2 resolution).

        Reads the JSON format written by
        :func:`moe_compress.stage6.plugins.teacher_provider._save_teacher_cache`
        (see ``stage6/plugins/teacher_provider.py`` around L217-L266 for
        the writer). The on-disk schema, copied verbatim from that
        writer, is::

            {
                "cache_key": "<sha256>",
                "teacher_results": { ... },
                "teacher_param_counts": { ... }  # optional (legacy: absent)
                "format_version": 1
            }

        Field mapping JSON -> dataclass:
            ``format_version``     -> ``schema_version``  (validated)
            ``cache_key``          -> ``cache_key``
            ``teacher_results``    -> ``teacher_results``
            ``teacher_param_counts`` (optional) -> ``teacher_param_counts``
                (``None`` when absent -- matches the Stage 6 reader's
                legacy-cache fallback at teacher_provider.py:198-205)

        Pattern C validation:
            * Missing file -> ``FileNotFoundError`` (actionable: "no
              teacher cache at {path}").
            * Malformed JSON -> ``ValueError`` (wraps json.JSONDecodeError).
            * Missing ``cache_key`` or ``teacher_results`` -> ``ValueError``.
            * ``format_version`` mismatch with
              ``SCHEMA_VERSIONS["teacher_eval"]`` -> ``ValueError`` with
              the canonical "Delete the sidecar to regenerate" message.

        Returns a properly-typed :class:`TeacherEvalPayload` instance.

        Cross-ref: keep this aligned with the writer in
        :func:`moe_compress.stage6.plugins.teacher_provider._save_teacher_cache`
        (commit: see ``TEACHER_CACHE_FORMAT_VERSION`` constant there).
        If Stage 6 bumps ``TEACHER_CACHE_FORMAT_VERSION``, also bump
        ``SCHEMA_VERSIONS["teacher_eval"]`` in this module.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"TeacherEvalPayload.from_legacy_json: no teacher cache "
                f"at {path}. Run Stage 6 first to populate "
                f"teacher_eval_cache.json, or pass the correct path."
            )
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"TeacherEvalPayload.from_legacy_json: cache file at "
                f"{path} is not valid JSON ({exc}). "
                f"Delete the sidecar to regenerate."
            ) from exc
        if not isinstance(data, dict):
            raise ValueError(
                f"TeacherEvalPayload.from_legacy_json: cache file at "
                f"{path} did not decode to a JSON object (got "
                f"{type(data).__name__}). "
                f"Delete the sidecar to regenerate."
            )

        on_disk_version = data.get("format_version")
        expected = SCHEMA_VERSIONS["teacher_eval"]
        if on_disk_version is None:
            raise ValueError(
                f"TeacherEvalPayload.from_legacy_json: cache file at "
                f"{path} has no 'format_version' field "
                f"(expected {expected}). "
                f"Delete the sidecar to regenerate."
            )
        if int(on_disk_version) != int(expected):
            raise ValueError(
                f"TeacherEvalPayload.from_legacy_json: cache file at "
                f"{path} has format_version={on_disk_version}, "
                f"expected {expected}. "
                f"Delete the sidecar to regenerate."
            )

        cache_key = data.get("cache_key")
        if not isinstance(cache_key, str) or not cache_key:
            raise ValueError(
                f"TeacherEvalPayload.from_legacy_json: cache file at "
                f"{path} has missing or non-string 'cache_key'. "
                f"Delete the sidecar to regenerate."
            )

        teacher_results = data.get("teacher_results")
        if not isinstance(teacher_results, dict):
            raise ValueError(
                f"TeacherEvalPayload.from_legacy_json: cache file at "
                f"{path} has missing or non-dict 'teacher_results'. "
                f"Delete the sidecar to regenerate."
            )

        # teacher_param_counts is optional (legacy caches written before
        # the field was added omit it; the Stage 6 reader treats absence
        # as a soft warning, not an error -- see teacher_provider.py:198-205).
        teacher_param_counts = data.get("teacher_param_counts")
        if (teacher_param_counts is not None
                and not isinstance(teacher_param_counts, dict)):
            raise ValueError(
                f"TeacherEvalPayload.from_legacy_json: cache file at "
                f"{path} has non-dict 'teacher_param_counts' (got "
                f"{type(teacher_param_counts).__name__}). "
                f"Delete the sidecar to regenerate."
            )

        return cls(
            schema_version=int(on_disk_version),
            cache_key=cache_key,
            teacher_results=teacher_results,
            teacher_param_counts=teacher_param_counts,
        )


# ---------------------------------------------------------------------------
# Atomic-write helpers.
# ---------------------------------------------------------------------------
def _atomic_torch_save(payload: Any, path: Path) -> None:
    """Atomic torch.save: tmp + fsync(fd) + os.replace + fsync(parent_dir).

    F-H-3 fix: previously this helper did only tmp + os.replace — no
    fsync. POSIX atomic-rename guarantees "either old or new", but only
    IF the file's data blocks are flushed before the rename. Without
    fsync(tmp), a kernel-panic / VM eviction between torch.save's last
    write() and the next pdflush cycle (typically 5-30 s on ext4) could
    leave the renamed file with stale/garbage blocks.

    Now delegates to the shared :func:`utils.atomic_io.atomic_torch_save`
    which does the full §11 durable-write dance — fixing all ~15 sidecar
    write sites in this module at once (Pattern O).

    Local arg order is ``(payload, path)`` for backward compat with the
    pre-MEDIUM-1 in-module callers; the shared helper is invoked with
    the new ``(path, payload)`` order it now requires.
    """
    _shared_atomic_torch_save(path, payload)


def _atomic_npz_save(arrays: dict[str, np.ndarray], path: Path) -> None:
    """Atomic np.savez_compressed: write to a tmp PATH WITHOUT the .npz
    extension, then os.replace into the final .npz path.

    CRITICAL: np.savez_compressed auto-appends '.npz' to filenames that
    don't already end in .npz. To avoid '.npz.tmp.npz' double-extension,
    we open the tmp file as a binary handle, pass the HANDLE to
    np.savez_compressed (which does NOT auto-append), then rename the
    finalized file to the final .npz path.

    F-H-3 fix: now delegates to
    :func:`utils.atomic_io.atomic_npz_save` which additionally fsyncs
    the file handle + parent dir for durability under power-loss /
    kernel-panic-class events.
    """
    _shared_atomic_npz_save(path, **arrays)


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
#
# NIT-3 (audit/calibration-completeness): the ``save_phase_b`` writer
# was removed -- the combined Phase-B payload was superseded by the
# per-signal sidecars (``per_expert_max`` + ``routing_stats`` +
# ``output_reservoir``) and had no production caller. ``PhaseBPayload``
# + ``load_phase_b`` are RETAINED for backward-compat reads of any
# legacy phase_b.pt files an operator may still have on disk (also
# exercised by the F-H-7 backward-compat tests in
# ``test_cached_calibration_signals.py``).
# ---------------------------------------------------------------------------
def load_phase_b(jsonl_path: Path) -> PhaseBPayload | None:
    path = _resolve_sidecar_for_load(jsonl_path, "phase_b")
    if path is None:
        return None
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    _check_schema("phase_b", loaded.schema_version, path)
    return loaded


# ---------------------------------------------------------------------------
# Signal 2: stage2_profile (Stage 2 REDO — Optimization A profile-pass sidecar).
#
# Schema v3 (see Stage2ProfilePayloadV3 docstring above). The v1 dataclass
# was deleted; no alias is retained per plan §3 / Low-8 (prior v1 had no
# production writer, so no callers exist).
# ---------------------------------------------------------------------------
_COV_STORAGE_DTYPE_ALLOWED = ("float16", "bfloat16", "float32")


def save_stage2_profile_v3(
    payload: Stage2ProfilePayloadV3, jsonl_path: Path,
) -> None:
    """Atomically write the Stage 2 profile sidecar (schema v3).

    Moves all tensors to CPU before serialization so the sidecar is
    device-agnostic (H200 → RTX 6000 Pro / CPU round-trip is supported).
    The ``gate_logit_profiles`` dict's nested ``list[tuple[int, Tensor]]``
    structure is preserved byte-for-byte: each per-batch tensor is moved
    to CPU but the ``(int, tensor)`` tuple shape and list ordering are
    not altered.
    """
    if payload.cov_storage_dtype not in _COV_STORAGE_DTYPE_ALLOWED:
        raise ValueError(
            f"save_stage2_profile_v3: cov_storage_dtype="
            f"{payload.cov_storage_dtype!r} not in "
            f"{_COV_STORAGE_DTYPE_ALLOWED!r}"
        )
    cov_dtype = getattr(torch, payload.cov_storage_dtype)

    # Move gate_logit_profiles list-of-tuples to CPU, preserving shape.
    cpu_glp: dict[int, list[tuple[int, torch.Tensor]]] = {}
    for layer_rank, batches in payload.gate_logit_profiles.items():
        cpu_glp[int(layer_rank)] = [
            (int(offset), t.detach().to("cpu", dtype=torch.float32).contiguous())
            for offset, t in batches
        ]

    # neuron_act_sum / neuron_act_count are small per-(layer, expert) tensors
    # / ints; CPU-cast tensors, deep-copy counts.
    cpu_nas = {
        (int(lr), int(e)): v.detach().to("cpu", dtype=torch.float32).contiguous()
        for (lr, e), v in payload.neuron_act_sum.items()
    }
    cpu_nac = {(int(lr), int(e)): int(c)
               for (lr, e), c in payload.neuron_act_count.items()}

    # cov_acc dict entries are CPU-cast to the declared storage dtype.
    cpu_cov = {
        (int(lr), int(e), str(m)): v.detach().to(
            "cpu", dtype=cov_dtype, copy=True,
        ).contiguous()
        for (lr, e, m), v in payload.cov_acc.items()
    }
    cpu_ctc = {(int(lr), int(e), str(m)): int(n)
               for (lr, e, m), n in payload.cov_token_count.items()}

    # layer_input_reservoir: list[Tensor[N, hidden] bf16]; CPU + bf16.
    cpu_lir: list = []
    for i, t in enumerate(payload.layer_input_reservoir):
        if t is None:
            # Per plan: the field is always populated when capture is on,
            # but we tolerate Optional entries (e.g. partial captures)
            # rather than crash here. Writer-side §10 guarantees a tensor
            # per rank; this is defense-in-depth.
            cpu_lir.append(None)
        else:
            cpu_lir.append(
                t.detach().to("cpu", dtype=torch.bfloat16).contiguous()
            )

    cpu_payload = Stage2ProfilePayloadV3(
        format_version=int(payload.format_version),
        schema_version=int(payload.schema_version),
        model_hash=str(payload.model_hash),
        n_layers=int(payload.n_layers),
        n_experts=int(payload.n_experts),
        top_k=int(payload.top_k),
        cov_storage_dtype=str(payload.cov_storage_dtype),
        total_tokens_per_layer=payload.total_tokens_per_layer.detach().to(
            "cpu", dtype=torch.int64,
        ).contiguous(),
        gate_logit_profiles=cpu_glp,
        sim_tensor=payload.sim_tensor.detach().to(
            "cpu", dtype=torch.float64,
        ).contiguous(),
        neuron_act_sum=cpu_nas,
        neuron_act_count=cpu_nac,
        cov_acc=cpu_cov,
        cov_token_count=cpu_ctc,
        layer_input_reservoir=cpu_lir,
    )
    _write_payload_and_manifest(
        cpu_payload,
        sidecar_path(jsonl_path, "stage2_profile"),
        signal_name="stage2_profile",
    )


def load_stage2_profile_v3(
    jsonl_path: Path,
    *,
    expected_cov_storage_dtype: str | None = None,
    expected_n_layers: int | None = None,
    expected_n_experts: int | None = None,
    expected_top_k: int | None = None,
    expected_model_hash: str | None = None,
) -> Stage2ProfilePayloadV3 | None:
    """Load the Stage 2 profile sidecar (schema v3).

    Returns ``None`` if the sidecar does not exist (cache miss). Raises
    ``ValueError`` with the "Delete the sidecar to regenerate" message on
    schema_version mismatch or on any cross-validation failure.

    Cross-validation (each is optional; only checked when caller passes
    the corresponding expected_* kwarg):
        * schema_version (always)
        * cov_storage_dtype (driver flag must match Stage 2 YAML)
        * n_layers, n_experts, top_k, model_hash
    """
    path = _resolve_sidecar_for_load(jsonl_path, "stage2_profile")
    if path is None:
        return None
    # Pattern O torn-write guard MUST run BEFORE torch.load (and BEFORE
    # the expected_* cross-validation below) — see plan §11.1: this row
    # has the most cross-validation kwargs and a torn .pt must surface
    # the manifest-specific "delete + re-run calibration" message, not a
    # misleading expected_* cross-validation error.
    _validate_manifest_or_warn(
        path,
        expected_schema_version=SCHEMA_VERSIONS["stage2_profile"],
        signal_name="stage2_profile",
    )
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    _check_schema("stage2_profile", loaded.schema_version, path)
    if not isinstance(loaded, Stage2ProfilePayloadV3):
        raise ValueError(
            f"stage2_profile sidecar at {path} is not Stage2ProfilePayloadV3 "
            f"(got {type(loaded).__name__}). "
            f"Delete the sidecar to regenerate."
        )
    # cov_storage_dtype must be one of the allowed strings even when no
    # expected_* is given — catches a future writer that ships a garbage
    # value.
    if loaded.cov_storage_dtype not in _COV_STORAGE_DTYPE_ALLOWED:
        raise ValueError(
            f"stage2_profile sidecar at {path} has cov_storage_dtype="
            f"{loaded.cov_storage_dtype!r} not in "
            f"{_COV_STORAGE_DTYPE_ALLOWED!r}. "
            f"Delete the sidecar to regenerate."
        )
    if (expected_cov_storage_dtype is not None
            and loaded.cov_storage_dtype != expected_cov_storage_dtype):
        raise ValueError(
            f"stage2_profile sidecar at {path} has cov_storage_dtype="
            f"{loaded.cov_storage_dtype!r} but the run is configured with "
            f"covariance_storage_dtype={expected_cov_storage_dtype!r}. "
            f"Delete the sidecar to regenerate."
        )
    if (expected_n_layers is not None
            and int(loaded.n_layers) != int(expected_n_layers)):
        raise ValueError(
            f"stage2_profile sidecar at {path} has n_layers="
            f"{loaded.n_layers} but the run has {expected_n_layers}. "
            f"Delete the sidecar to regenerate."
        )
    if (expected_n_experts is not None
            and int(loaded.n_experts) != int(expected_n_experts)):
        raise ValueError(
            f"stage2_profile sidecar at {path} has n_experts="
            f"{loaded.n_experts} but the run has {expected_n_experts}. "
            f"Delete the sidecar to regenerate."
        )
    if (expected_top_k is not None
            and int(loaded.top_k) != int(expected_top_k)):
        raise ValueError(
            f"stage2_profile sidecar at {path} has top_k="
            f"{loaded.top_k} but the run has {expected_top_k}. "
            f"Delete the sidecar to regenerate."
        )
    if (expected_model_hash is not None
            and loaded.model_hash != expected_model_hash):
        raise ValueError(
            f"stage2_profile sidecar at {path} has model_hash="
            f"{loaded.model_hash!r} but the run has "
            f"{expected_model_hash!r}. "
            f"Delete the sidecar to regenerate."
        )
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
    _write_payload_and_manifest(
        cpu_payload,
        sidecar_path(jsonl_path, "reap_scores"),
        signal_name="reap_scores",
    )


def load_reap_scores(jsonl_path: Path) -> Stage2ReapPayload | None:
    """Load the Stage 2 REAP-scores sidecar.

    Returns None if the sidecar does not exist (cache miss). Raises
    ValueError on schema_version mismatch with an actionable message.
    """
    path = _resolve_sidecar_for_load(jsonl_path, "reap_scores")
    if path is None:
        return None
    _validate_manifest_or_warn(
        path,
        expected_schema_version=SCHEMA_VERSIONS["reap_scores"],
        signal_name="reap_scores",
    )
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
    _write_payload_and_manifest(
        cpu_payload,
        sidecar_path(jsonl_path, "per_expert_max"),
        signal_name="per_expert_max",
    )


def load_per_expert_max(jsonl_path: Path) -> Stage1PerExpertMaxPayload | None:
    """Load the Stage 1 per-expert-max sidecar.

    Returns None if the sidecar does not exist (cache miss). Raises
    ValueError on schema_version mismatch with an actionable message.
    """
    path = _resolve_sidecar_for_load(jsonl_path, "per_expert_max")
    if path is None:
        return None
    _validate_manifest_or_warn(
        path,
        expected_schema_version=SCHEMA_VERSIONS["per_expert_max"],
        signal_name="per_expert_max",
    )
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
    _write_payload_and_manifest(
        cpu_payload,
        sidecar_path(jsonl_path, "routing_stats"),
        signal_name="routing_stats",
    )


def load_routing_stats(jsonl_path: Path) -> RoutingStatsPayload | None:
    """Load the routing-stats sidecar.

    Returns None if the sidecar does not exist (cache miss). Raises
    ValueError on schema_version mismatch with an actionable message.
    """
    path = _resolve_sidecar_for_load(jsonl_path, "routing_stats")
    if path is None:
        return None
    _validate_manifest_or_warn(
        path,
        expected_schema_version=SCHEMA_VERSIONS["routing_stats"],
        signal_name="routing_stats",
    )
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
    _write_payload_and_manifest(
        cpu_payload,
        sidecar_path(jsonl_path, "router_logits_stats"),
        signal_name="router_logits_stats",
    )


def load_router_logits_stats(jsonl_path: Path) -> RouterLogitsStatsPayload | None:
    """Load the router-logits-stats sidecar.

    Returns None if the sidecar does not exist (cache miss). Raises
    ValueError on schema_version mismatch with an actionable message.
    """
    path = _resolve_sidecar_for_load(jsonl_path, "router_logits_stats")
    if path is None:
        return None
    _validate_manifest_or_warn(
        path,
        expected_schema_version=SCHEMA_VERSIONS["router_logits_stats"],
        signal_name="router_logits_stats",
    )
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
    _write_payload_and_manifest(
        cpu_payload,
        sidecar_path(jsonl_path, "output_reservoir"),
        signal_name="output_reservoir",
    )


def load_output_reservoir(jsonl_path: Path) -> OutputReservoirPayload | None:
    """Load the output-reservoir sidecar.

    Returns None if the sidecar does not exist (cache miss). Raises
    ValueError on schema_version mismatch with an actionable message.
    """
    path = _resolve_sidecar_for_load(jsonl_path, "output_reservoir")
    if path is None:
        return None
    _validate_manifest_or_warn(
        path,
        expected_schema_version=SCHEMA_VERSIONS["output_reservoir"],
        signal_name="output_reservoir",
    )
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
    _write_payload_and_manifest(
        cpu_payload,
        sidecar_path(jsonl_path, "covariance"),
        signal_name="covariance",
    )


def load_covariance(jsonl_path: Path) -> CovariancePayload | None:
    path = _resolve_sidecar_for_load(jsonl_path, "covariance")
    if path is None:
        return None
    _validate_manifest_or_warn(
        path,
        expected_schema_version=SCHEMA_VERSIONS["covariance"],
        signal_name="covariance",
    )
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    _check_schema("covariance", loaded.schema_version, path)
    return loaded


# ---------------------------------------------------------------------------
# Signal 4: router_kd_logits (per-attempt-idx .npz shards).
#
# NIT-4 (audit/calibration-completeness): the ``save_router_kd_logits``
# writer was removed -- it had no production caller (production .npz
# shards are written directly by ``build_self_traces_calib_vllm.py`` in
# a streaming-writer pattern; see audit row C.3). ``RouterKDLogitsPayload``
# + ``load_router_kd_logits`` are RETAINED for backward-compat reads of
# the shards an operator may have on disk (also exercised by the F-H-7
# router_kd backward-compat test in ``test_cached_calibration_signals.py``).
# ---------------------------------------------------------------------------
def load_router_kd_logits(jsonl_path: Path, attempt_idx: int) -> RouterKDLogitsPayload | None:
    path = router_kd_logits_dir(jsonl_path) / f"{attempt_idx:07d}.npz"
    if not path.exists():
        # F-H-7 backward-compat: fall back to the legacy
        # non-namespaced router_kd_logits/ dir if the new namespaced
        # path is empty. Only safe if exactly one JSONL stem lives in
        # the parent dir (same disambiguation rule as
        # _resolve_sidecar_for_load).
        legacy_path = _legacy_router_kd_logits_dir(jsonl_path) / f"{attempt_idx:07d}.npz"
        if not legacy_path.exists():
            return None
        parent = jsonl_path.parent
        jsonls = list(parent.glob("*.jsonl")) + list(parent.glob("*.jsonl.tmp"))
        if len({p.stem for p in jsonls}) > 1:
            log.error(
                "F-H-7: legacy router_kd_logits shard %s exists but %s "
                "contains multiple JSONL stems — refusing to consume "
                "ambiguous legacy shard. Note: in-flight ``.jsonl.tmp`` "
                "files also count; complete or delete them first.",
                legacy_path, parent,
            )
            return None
        # HIGH-4: one-shot dedupe — warn at most once per (legacy_dir,
        # router_kd_logits) tuple across the process lifetime.
        _key = (legacy_path.parent, "router_kd_logits")
        if _key not in _already_warned_legacy_paths:
            log.warning(
                "F-H-7 backward-compat: loading router_kd_logits shard "
                "from legacy path %s; new writes will land at %s.",
                legacy_path, path,
            )
            _already_warned_legacy_paths.add(_key)
        path = legacy_path
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
    # Per-layer Pattern O: each layer shard has its OWN sibling manifest
    # (``block_hidden/layer_NNNN.pt.MANIFEST.json``), so torn-write damage
    # on layer L does NOT invalidate layers L-1 or L+1. The
    # ``_validate_manifest_or_warn`` dedupe key uses the full per-shard
    # ``payload_path``, so legacy 40-layer artifacts emit one WARN per
    # missing-manifest layer (not one global WARN that obscures which
    # layer is legacy).
    _write_payload_and_manifest(cpu_payload, path, signal_name="block_hidden")


def load_block_hidden(jsonl_path: Path, layer_idx: int) -> BlockHiddenPayload | None:
    path = _resolve_sidecar_for_load(jsonl_path, f"block_hidden/layer_{layer_idx:04d}")
    if path is None:
        return None
    # Per-layer Pattern O: each shard's manifest is validated
    # independently, so torn damage on layer L's shard fails LOAD(L)
    # only — layers L-1, L+1 still load.
    _validate_manifest_or_warn(
        path,
        expected_schema_version=SCHEMA_VERSIONS["block_hidden"],
        signal_name="block_hidden",
    )
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    _check_schema("block_hidden", loaded.schema_version, path)
    return loaded


# ---------------------------------------------------------------------------
# Signal 6: teacher_eval (harness results keyed by SHA-256 cache_key).
#
# NIT-5 (audit/calibration-completeness): the ``save_teacher_eval``
# writer was removed -- it had no production caller. Stage 6 has its
# own teacher-eval cache mechanism at ``stage6/plugins/teacher_provider.py``
# (``_save_teacher_cache`` writes a strong-fsync JSON file at an
# operator-configurable path, distinct from this sidecar layout).
#
# MEDIUM-2 resolution (2026-05-29, user picked option (b)-as-adapter):
# Stage 6 keeps its on-disk JSON format unchanged; the READ path now
# also goes through the canonical typed ``TeacherEvalPayload`` via the
# ``from_legacy_json(path)`` classmethod (see the dataclass docstring
# above for the full resolution). New consumers should use that
# classmethod rather than parsing Stage 6's JSON directly. The legacy
# ``load_teacher_eval`` sidecar reader below is RETAINED for
# backward-compat with any torch-pt sidecars an operator may have
# written before the resolution.
# ---------------------------------------------------------------------------
def load_teacher_eval(jsonl_path: Path) -> TeacherEvalPayload | None:
    path = _resolve_sidecar_for_load(jsonl_path, "teacher_eval")
    if path is None:
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
    "Stage2ProfilePayloadV3",
    "Stage2ReapPayload",
    "Stage1PerExpertMaxPayload",
    "RoutingStatsPayload",
    "RouterLogitsStatsPayload",
    "OutputReservoirPayload",
    "CovariancePayload",
    "RouterKDLogitsPayload",
    "BlockHiddenPayload",
    "TeacherEvalPayload",
    "load_phase_b",
    "save_stage2_profile_v3",
    "load_stage2_profile_v3",
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
    "load_router_kd_logits",
    "save_block_hidden",
    "load_block_hidden",
    "load_teacher_eval",
    "BaseCacheProvider",
    "BaseLiveProvider",
]
