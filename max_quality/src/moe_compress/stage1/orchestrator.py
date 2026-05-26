"""Stage 1 plugin orchestrator — the thin phase sequencer.

Replaces the tangled body of the legacy ``stage1.run()`` (sub-task 10 of
the Stage 1 → plugin-architecture refactor). This is the only Stage 1
code path; callers reach it via ``moe_compress.stage1.run``.

What the orchestrator does
--------------------------
- threads **one** :class:`~moe_compress.pipeline.context.PipelineContext`
  through all six phases,
- runs Phase A (``MADetectionPlugin`` — its own dedicated forward pass),
- runs **one** shared calibration pass via
  :func:`~moe_compress.tools.calibration_pass.run_calibration_pass`
  (replacing the legacy inline Phase B),
- runs the four candidate-detector plugins, then ``ablation_filter``,
  ``cka_distance``, ``grape_merge``,
- assembles ``stage1_blacklist.json`` via
  :class:`~moe_compress.tools.artifact_builder.ArtifactBuilder` and
  writes the two output files,
- emits the same Trackio telemetry.

The orchestrator owns only *glue*: the accumulator factory
(``_build_accumulator`` — the plugin-declared ``provides`` name → a
concrete accumulator + :class:`HookSpec` mapping), the artifact assembly
(``_write_artifacts``), and the telemetry block (``_emit_telemetry``).
All phase logic lives inside the eight plugins under ``stage1/plugins/``.
"""
from __future__ import annotations

import logging
import statistics
from pathlib import Path

import torch

from ..budget.solver import BudgetDecomposition
from ..pipeline.candidates import CandidateBag
from ..pipeline.context import PipelineContext
from ..pipeline.registry import PluginRegistry
from ..tools.artifact_builder import ArtifactBuilder
from ..tools.calibration_pass import HookKind, HookSpec, run_calibration_pass
from ..tools.phase_walker import walk_phases
from ..utils.activation_hooks import DownProjMaxAccumulator, ExpertOutputAccumulator
from ..utils.calibration import build_calibration_tensor, iter_batches, spec_from_config
from ..utils.model_io import iter_moe_layers, save_json_artifact
from ..utils.trackio_log import trackio_flush as _trackio_flush
from ..utils.trackio_log import trackio_log as _trackio_log
from .artifacts import REQUIRED_BLACKLIST_TOP_LEVEL_KEYS
from .plugins import STAGE1_PLUGIN_MANIFEST
from .plugins.output_reservoir_cache import (
    Stage1OutputReservoirCacheProvider,
)
from .plugins.per_expert_max_cache import Stage1PerExpertMaxCacheProvider
from .plugins.router_logits_stats_cache import (
    Stage1RouterLogitsStatsCacheProvider,
)
from .plugins.routing_stats_cache import Stage1RoutingStatsCacheProvider

log = logging.getLogger(__name__)

# --- spec-pinned constants (copied verbatim from the legacy Stage 1 module) ---
_PHASE_B_BATCH_SIZE = 16              # Phase B batch size — spec-pinned.
_CKA_RESERVOIR_CAP = 256             # ExpertOutputAccumulator(max_tokens_per_expert=256),
                                     # spec §12 D-ma-detector ("CKA reservoir cap = 256").


# ---------------------------------------------------------------------------
# Calibration JSONL path resolver — shared by STEP 4.5 / 4.6 / 4.7 (and any
# future cache-first attempts at Step 5+). Reads ``config["calibration"]
# ["jsonl_path"]`` (falling back to ``_DEFAULT_SELF_TRACES_PATH``) and
# anchors relative paths against ``Path.cwd()``. Kept at module level so
# each STEP block becomes a single-line lookup instead of re-duplicating
# the same 6-line resolve block.
# ---------------------------------------------------------------------------


def _resolve_calib_jsonl_path(config: dict) -> Path:
    """Return the absolute Path to the calibration JSONL for this run.

    Source: ``config["calibration"]["jsonl_path"]`` if present, else the
    package default ``_DEFAULT_SELF_TRACES_PATH``. Relative paths are
    anchored against ``Path.cwd()`` (matching the legacy STEP 4.5 / 4.6
    / 4.7 inline behavior verbatim).
    """
    from ..utils.calibration import _DEFAULT_SELF_TRACES_PATH
    cal_cfg = config["calibration"]
    calib_source = cal_cfg.get("jsonl_path", _DEFAULT_SELF_TRACES_PATH)
    calib_jsonl_path = Path(calib_source)
    if not calib_jsonl_path.is_absolute():
        calib_jsonl_path = Path.cwd() / calib_jsonl_path
    return calib_jsonl_path


# ---------------------------------------------------------------------------
# Phase-B calibration progress callback — moved verbatim from the legacy
# Stage 1 module (sub-task 10: the orchestrator now owns it).
# ---------------------------------------------------------------------------


def _make_calibration_progress_cb(phase_tag: str, n_total: int, log_every: int = 64):
    """Build a per-batch callback that streams Stage 1 calibration progress to
    Trackio every ``log_every`` batches.

    Tags: ``stage1/{phase_tag}/calibration_progress`` (fraction in [0,1]) and
    ``stage1/{phase_tag}/calibration_step`` (raw batch index).

    Cost: one ``_trackio_log`` (queue put) per ``log_every`` batches.
    Non-blocking — Trackio's internal sender thread uploads to HF on its own
    cadence.

    Why we don't ``_trackio_flush()`` here anymore: the previous version
    flushed synchronously after each ``_trackio_log`` to keep the sender
    thread alive and force a known drain cadence. That call **blocked the
    main thread** waiting for the sender's queue to drain, and on the
    2026-05-10 H200 run we observed periodic ~3-min idle windows during
    Phase B (GPU=1%) coinciding with the sender being unable to upload to
    the trackio bucket (the dashboard stayed empty for the entire run). A
    blocked sender + synchronous flush = a stalled main thread. The fix is
    to never block the main thread on the sender: the sender keeps running
    on its own; if it's healthy we get metrics, if it's dead we lose
    visibility but the calibration loop runs at full speed. A future fix
    can move the flush to a daemon-thread heartbeat, but that's not the
    bottleneck for getting Stage 1 to finish.
    """
    def _cb(i: int) -> None:
        n_done = i + 1
        if log_every > 0 and n_done % log_every == 0:
            _trackio_log({
                f"stage1/{phase_tag}/calibration_progress": n_done / n_total,
                f"stage1/{phase_tag}/calibration_step": n_done,
            })
    return _cb


# ---------------------------------------------------------------------------
# Accumulator factory — name → (instance, HookSpec) glue (plan §4.3).
# ---------------------------------------------------------------------------


def _build_accumulator(
    name: str,
    *,
    n_per_layer: int,
    moe_layers: list,
    tokenizer,
    ctx: PipelineContext,
) -> tuple[object, HookSpec]:
    """Map an accumulator NAME (as declared by a plugin's ``provides``
    tuple) to a concrete ``(accumulator_instance, HookSpec)`` pair the
    calibration pass can register.

    The orchestrator owns this mapping because it is the glue between the
    plugin's declarative ``provides=(...)`` metadata and the calibration
    pass's imperative ``(name, acc, spec)`` registration triples.

    The ``downproj_max`` / ``output_reservoir`` accumulators are
    constructed here. The ``sink_routing`` accumulator is NOT constructed
    here — ``SinkTokenDetectorPlugin.setup()`` builds it onto
    ``ctx["sink_acc"]`` and this factory READS it back (plan §4.5
    Option B). The factory still builds the ``sink_routing`` HookSpec
    (the per-batch closure).
    """
    if name == "downproj_max":
        acc = DownProjMaxAccumulator()
        spec = HookSpec(
            kinds=frozenset({HookKind.DOWN_PROJ}),
            expert_callback=lambda li, e, t, _ctx: acc.update(li, e, t),
        )
        return acc, spec

    if name == "output_reservoir":
        # Hard-pin the cap so a future default change in activation_hooks.py
        # cannot silently drift the spec-pinned reservoir size.
        acc = ExpertOutputAccumulator(max_tokens_per_expert=_CKA_RESERVOIR_CAP)
        spec = HookSpec(
            kinds=frozenset({HookKind.DOWN_PROJ}),
            expert_callback=lambda li, e, t, _ctx: acc.update(li, e, t),
        )
        return acc, spec

    if name == "sink_routing":
        # Plan §4.5 Option B: the authoritative SinkTokenRoutingAccumulator
        # is built by SinkTokenDetectorPlugin.setup() and lives on the ctx.
        # The factory reads it back so the engine feeds the SAME instance the
        # plugin's run()/contribute_artifact() later read — one instance, no
        # overwrite.
        acc = ctx.get("sink_acc")
        top_k_per_layer = {ref.layer_idx: ref.top_k for ref in moe_layers}

        def _sink_per_batch(pbc) -> None:
            # Byte-identical to the legacy ``_phase_b_per_batch_cb`` body
            # — softmax → top-k → sink_acc.update.
            # The engine drains router_logits_storage AFTER all per-batch
            # handlers run, so this closure does NOT clear it (plan §6.3 R2).
            ids = pbc.input_ids
            if ids is None:
                return
            if ids.dim() == 1:
                ids = ids.unsqueeze(0)
            B, T = ids.shape
            storage = pbc.router_logits_storage
            if storage is None:
                return
            for ref in moe_layers:
                logits_list = storage.get(ref.layer_idx)
                if not logits_list:
                    continue
                logits = logits_list[-1]              # [B*T, num_experts]
                if logits.shape[0] != B * T:
                    # Defensive: skip if shape mismatch (should not occur,
                    # but protects against router variants that flatten
                    # differently).
                    continue
                logits_3d = logits.reshape(B, T, -1)
                scores = torch.softmax(logits_3d.float(), dim=-1)  # post-softmax
                k = top_k_per_layer[ref.layer_idx]
                _, routed_pos = torch.topk(scores, k=k, dim=-1)    # [B, T, k]
                acc.update(ref.layer_idx, ids, scores, routed_pos)

        spec = HookSpec(
            kinds=frozenset({
                HookKind.ROUTER_LOGITS_PER_BATCH,
                HookKind.INPUT_IDS_PER_BATCH,
            }),
            per_batch=_sink_per_batch,
        )
        return acc, spec

    raise ValueError(f"orchestrator: unknown accumulator name {name!r}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run(
    model,
    tokenizer,
    config: dict,
    artifacts_dir: Path,
    decomposition: BudgetDecomposition,
    *,
    device=None,
) -> tuple[Path, Path]:
    """Run Stage 1 — SE detection + GRAPE budgets — via the plugin pipeline.

    Threads one :class:`PipelineContext` through all six phases. Returns
    ``(blacklist_path, budgets_path)`` — same as the legacy ``run()``.
    """
    # ---- STEP 1: resolve config + moe layers ------------------------------
    s1 = config["stage1_grape"]
    se_cfg = s1.get("super_expert_detection", {})
    moe_layers = list(iter_moe_layers(model))
    if not moe_layers:
        raise ValueError(
            "Stage 1: model has no MoE layers — check iter_moe_layers() "
            "compatibility with this model architecture."
        )
    n_per_layer = moe_layers[0].num_routed_experts
    if any(ref.num_routed_experts != n_per_layer for ref in moe_layers[1:]):
        log.warning(
            "stage1: layers have heterogeneous expert counts; "
            "GRAPE floor is computed per-layer as num_routed_experts // 2"
        )

    # Deprecated-key warnings — moved verbatim from the legacy Stage 1 module.
    for old_key in ("zscore_threshold", "max_blacklisted_per_layer",
                    "global_blacklist_cap_pct"):
        if old_key in se_cfg:
            log.warning(
                "Stage 1: config key '%s' is deprecated and ignored. "
                "Super expert detection now uses the paper's three-way AND "
                "criterion (P99.5 + 0.1·a_max). Remove this key from your config.",
                old_key,
            )

    # ---- STEP 2: one PipelineContext + the PluginRegistry -----------------
    ctx = PipelineContext()
    ctx.set("model", model)
    ctx.set("tokenizer", tokenizer)
    ctx.set("config", config)
    ctx.set("artifacts_dir", artifacts_dir)
    ctx.set("device", device)
    ctx.set("decomposition", decomposition)
    ctx.set("moe_layers", moe_layers)
    ctx.set("n_per_layer", n_per_layer)
    ctx.set("candidate_bag", CandidateBag())
    ctx.set("per_layer_targets",
            {ref.layer_idx: ref.num_routed_experts for ref in moe_layers})

    registry = PluginRegistry(STAGE1_PLUGIN_MANIFEST)
    # by_name is built from the FULL manifest, not the enabled subset: the
    # detector/cka/grape/ablation run() calls are unconditional (each plugin
    # short-circuits internally on its own flag — matching the legacy run()).
    # is_enabled / enabled() is used ONLY to drive provides.
    # See plan §4.6.
    by_name = {p.name: p for p in registry}

    # ---- STEP 3: Phase A (ma_detection) — its own dedicated pass ----------
    ma_plugin = by_name["ma_detection"]       # mandatory, always enabled
    ma_plugin.run(ctx)                        # writes L, residual_growth, ...
    L = ctx.get("L")
    log.info("Stage 1 Phase A: MA-formation layers L = %s", sorted(L))

    # ---- STEP 4: run setup() on setup-capable plugins ---------------------
    # Must run BEFORE the calibration engine pass — sink_token.setup() builds
    # the authoritative SinkTokenRoutingAccumulator onto ctx["sink_acc"], and
    # the accumulator factory (Step 5) reads it back (plan §4.5 Option B).
    # ``getattr`` is used deliberately: only sink_token has setup() today; a
    # SetupCapablePlugin Protocol is deferred (YAGNI — see plan §4.5).
    for plugin in registry:
        setup = getattr(plugin, "setup", None)
        if callable(setup):
            setup(ctx)

    # ---- STEP 4.5: V2 cache-first attempt for per_expert_max --------------
    # Try the calibration-v2 sidecar at
    # ``<jsonl_dir>/sidecars/per_expert_max.pt`` BEFORE building the live
    # DownProjMaxAccumulator. On hit, the cache provider constructs the
    # accumulator with its ``per_expert_max`` dict pre-populated and sets
    # it on ``ctx.max_acc``; we then skip the live registration of
    # ``downproj_max`` (no Phase B forward needed for max-magnitude
    # collection — the other accumulators still register).
    #
    # Divergence from the canonical provider-pair pattern (see
    # cached_calibration_signals.py module docstring): Stage 1 has no
    # "live provider plugin" for per_expert_max — the live path is the
    # ``DownProjMaxAccumulator`` factory + the ``run_calibration_pass``
    # closure that feeds it. So we cannot use ``PluginRegistry.dispatch_first``
    # to fall through cache → live. Instead, we instantiate the cache
    # provider directly here; on hit it pre-populates ctx.max_acc and we
    # drop "downproj_max" from ``needed`` to skip the live accumulator; on
    # miss we leave ``needed`` unchanged and the live path runs at Phase B.
    _pem_cache_hit = False
    try:
        _calib_jsonl_path = _resolve_calib_jsonl_path(config)
        _pem_provider = Stage1PerExpertMaxCacheProvider()
        _pem_payload = _pem_provider.on_load(ctx, _calib_jsonl_path)
        if _pem_payload is not None:
            _pem_cache_hit = True
            log.info(
                "Stage 1: V2 per_expert_max cache HIT (%d layers × %d "
                "experts) -- skipping live downproj_max registration",
                _pem_payload.n_layers, _pem_payload.n_experts,
            )
    except (FileNotFoundError, OSError) as _exc:
        # Routine filesystem misses should not block the live path.
        # ValueError from _check_schema is NOT caught here -- a schema
        # mismatch is an actionable user error ("Delete the sidecar to
        # regenerate") and silently falling back would mask it.
        log.warning("Stage 1: V2 per_expert_max cache lookup failed (%s) "
                    "-- falling back to live downproj_max accumulator", _exc)

    # ---- STEP 4.6: V2 cache-first attempt for routing_stats ---------------
    # Try the calibration-v2 sidecar at
    # ``<jsonl_dir>/sidecars/routing_stats.pt`` -- a routing-frequency +
    # mean-routing-weight payload produced by the
    # ``--capture-routing-stats`` flag (Item 3 of the writers campaign).
    # On hit, the cache provider deposits the payload on
    # ``ctx.routing_stats_payload`` so future read-side plugins can pick
    # it up.
    #
    # Divergence from the canonical provider-pair pattern (see
    # cached_calibration_signals.py module docstring): Item 3 has NO
    # live counterpart in Stage 1 AND NO immediate downstream consumer.
    # The cache provider is laid down now to keep the on-disk schema
    # stable and the ctx-slot contract testable from day 1; consumer
    # plugins (routing-aware ablation gating, mean-weight-weighted REAP
    # variants, ...) will be added by later items. Unlike STEP 4.5 we
    # do NOT modify ``needed`` (no live accumulator to skip).
    try:
        _calib_jsonl_path = _resolve_calib_jsonl_path(config)
        _rts_provider = Stage1RoutingStatsCacheProvider()
        _rts_payload = _rts_provider.on_load(ctx, _calib_jsonl_path)
        if _rts_payload is not None:
            log.info(
                "Stage 1: V2 routing_stats cache HIT (%d layers x %d "
                "experts) -- ctx.routing_stats_payload populated",
                _rts_payload.n_layers, _rts_payload.n_experts,
            )
    except (FileNotFoundError, OSError) as _exc:
        # Routine filesystem misses should not block the live path.
        # ValueError from _check_schema is NOT caught here -- a schema
        # mismatch is an actionable user error ("Delete the sidecar to
        # regenerate") and silently falling back would mask it.
        log.warning("Stage 1: V2 routing_stats cache lookup failed (%s) "
                    "-- ctx.routing_stats_payload not populated", _exc)

    # ---- STEP 4.7: V2 cache-first attempt for router_logits_stats ---------
    # Try the calibration-v2 sidecar at
    # ``<jsonl_dir>/sidecars/router_logits_stats.pt`` -- per-(layer,
    # expert) sink-vs-normal router-score aggregates produced by the
    # ``--capture-router-logits-stats`` flag (Item 4 of the writers
    # campaign). On hit the provider OVERWRITES ``ctx.sink_acc`` with
    # a pre-finalized SinkTokenRoutingAccumulator hydrated from the
    # cached aggregates -- the same slot SinkTokenDetectorPlugin.setup()
    # already populated earlier in STEP 4 with a fresh-empty live
    # accumulator. We then drop ``"sink_routing"`` from ``needed`` so
    # the live router-logits HookSpec is NOT registered + the live
    # accumulator is NOT built (the orchestrator already has the
    # finalized form on ctx).
    #
    # R3 (sink_token_enabled=False) is honored by the provider: if the
    # user has explicitly disabled the sink-token detector, the
    # provider returns None and STEP 4's ``sink_acc=None`` stays
    # untouched; we ALSO leave ``needed`` unchanged because the live
    # path's HookSpec is gated on sink_acc being non-None at the per-
    # batch closure level anyway.
    #
    # Divergence from the canonical provider-pair pattern: Stage 1 has
    # no "live provider plugin" for sink-routing -- the live path is
    # the SinkTokenDetectorPlugin.setup() + the orchestrator-built
    # HookSpec. So we cannot use PluginRegistry.dispatch_first to fall
    # through cache -> live. Instead, we instantiate the cache provider
    # directly here; on hit it pre-populates ctx.sink_acc + we drop
    # "sink_routing" from ``needed`` to skip the live HookSpec
    # registration; on miss we leave ``needed`` unchanged so the live
    # router-logits + softmax + top-k pass runs at Phase B.
    _rlsx_cache_hit = False
    try:
        _calib_jsonl_path = _resolve_calib_jsonl_path(config)
        _rlsx_provider = Stage1RouterLogitsStatsCacheProvider()
        _rlsx_payload = _rlsx_provider.on_load(ctx, _calib_jsonl_path)
        if _rlsx_payload is not None:
            _rlsx_cache_hit = True
            log.info(
                "Stage 1: V2 router_logits_stats cache HIT (%d layers x "
                "%d experts) -- skipping live sink_routing registration; "
                "ctx.sink_acc replaced with pre-finalized accumulator",
                _rlsx_payload.n_layers, _rlsx_payload.n_experts,
            )
    except (FileNotFoundError, OSError) as _exc:
        # Routine filesystem misses should not block the live path.
        # ValueError from _check_schema is NOT caught here -- a schema
        # mismatch is an actionable user error ("Delete the sidecar to
        # regenerate") and silently falling back would mask it.
        log.warning(
            "Stage 1: V2 router_logits_stats cache lookup failed (%s) "
            "-- falling back to live sink-routing accumulator", _exc,
        )

    # ---- STEP 4.8: V2 cache-first attempt for output_reservoir -----------
    # Try the calibration-v2 sidecar at
    # ``<jsonl_dir>/sidecars/output_reservoir.pt`` -- per-(layer, expert)
    # reservoir-sampled expert outputs produced by the
    # ``--capture-output-reservoir`` flag (Item 6 of the writers
    # campaign). On hit the provider populates a pre-finalized
    # ``ExpertOutputAccumulator`` directly into ``ctx.output_acc`` and
    # we drop ``"output_reservoir"`` from ``needed`` so the live
    # accumulator factory is NOT invoked and the Phase-B forward skips
    # the per-expert reservoir-sample work entirely.
    #
    # Divergence from the canonical provider-pair pattern (see
    # cached_calibration_signals.py module docstring): Stage 1 has no
    # "live provider plugin" for output_reservoir -- the live path is
    # the ``ExpertOutputAccumulator`` factory + the
    # ``run_calibration_pass`` closure. So we cannot use
    # ``PluginRegistry.dispatch_first`` to fall through cache -> live.
    # Instead, we instantiate the cache provider directly here; on hit
    # it pre-populates ctx.output_acc + we drop "output_reservoir" from
    # ``needed`` to skip the live accumulator; on miss we leave
    # ``needed`` unchanged and the live path runs at Phase B.
    _or_cache_hit = False
    try:
        _calib_jsonl_path = _resolve_calib_jsonl_path(config)
        _or_provider = Stage1OutputReservoirCacheProvider()
        _or_payload = _or_provider.on_load(ctx, _calib_jsonl_path)
        if _or_payload is not None:
            _or_cache_hit = True
            log.info(
                "Stage 1: V2 output_reservoir cache HIT (%d layers x "
                "%d experts) -- skipping live output_reservoir "
                "registration; ctx.output_acc replaced with "
                "pre-finalized accumulator",
                _or_payload.n_layers, _or_payload.n_experts,
            )
    except (FileNotFoundError, OSError) as _exc:
        # Routine filesystem misses should not block the live path.
        # ValueError from _check_schema is NOT caught here -- a schema
        # mismatch is an actionable user error ("Delete the sidecar to
        # regenerate") and silently falling back would mask it.
        log.warning(
            "Stage 1: V2 output_reservoir cache lookup failed (%s) "
            "-- falling back to live output_reservoir accumulator",
            _exc,
        )

    # ---- STEP 5: build the ordered accumulator registrations --------------
    # ``registry.provides(config)`` returns the byte-identity-critical
    # accumulator order; iterating it preserves that order in the
    # registration triples handed to ``run_calibration_pass``.
    needed = registry.provides(config)   # ordered tuple
    # On cache hit, exclude ``downproj_max`` from the live registration set
    # -- the cache provider has already populated ``ctx.max_acc`` with the
    # cached values. The other accumulators (output_reservoir, sink_routing,
    # ...) still register and run their live calibration.
    if _pem_cache_hit:
        needed = tuple(n for n in needed if n != "downproj_max")
    # Same surgery for STEP 4.7: the router_logits_stats cache hydrates a
    # pre-finalized SinkTokenRoutingAccumulator into ctx.sink_acc, so
    # the live per-batch sink-routing HookSpec is no longer needed.
    if _rlsx_cache_hit:
        needed = tuple(n for n in needed if n != "sink_routing")
    # Same surgery for STEP 4.8: the output_reservoir cache hydrates a
    # pre-finalized ExpertOutputAccumulator into ctx.output_acc, so the
    # live per-batch reservoir-sample HookSpec is no longer needed.
    if _or_cache_hit:
        needed = tuple(n for n in needed if n != "output_reservoir")
    registrations: list[tuple[str, object, HookSpec]] = []
    built: dict[str, object] = {}
    for acc_name in needed:
        acc, spec = _build_accumulator(
            acc_name, n_per_layer=n_per_layer,
            moe_layers=moe_layers, tokenizer=tokenizer, ctx=ctx,
        )
        registrations.append((acc_name, acc, spec))
        built[acc_name] = acc

    # Publish the built accumulators onto the ctx under the slot names the
    # detector plugins read. ``sink_acc`` is already correct from setup()
    # (the factory reads it, never rebuilds it) — no orchestrator write here.
    # ``max_acc`` is set by the cache provider on hit; otherwise we publish
    # the freshly-built live accumulator now.
    if not _pem_cache_hit:
        ctx.set("max_acc", built["downproj_max"])      # always present
    # ``output_acc`` is set by the cache provider on hit; otherwise we
    # publish the freshly-built live accumulator now.
    if not _or_cache_hit:
        ctx.set("output_acc", built["output_reservoir"])   # always present

    # ---- STEP 6: run the calibration pass ONCE ----------------------------
    cal = config["calibration"]
    spec_cal = spec_from_config(
        cal, num_sequences_override=s1.get("num_calibration_samples"),
        seed_offset=1,
    )
    calib = build_calibration_tensor(
        tokenizer, spec_cal, cache_dir=artifacts_dir / "_calibration_cache",
    )
    phase_b_bs = int(s1.get("phase_b_batch_size", _PHASE_B_BATCH_SIZE))
    batches = iter_batches(calib, batch_size=phase_b_bs)
    progress_cb = _make_calibration_progress_cb("phase_b", n_total=len(batches))
    log.info(
        "Stage 1 Phase B: profiling %d layers on %d batches (magnitude for "
        "L=%s, CKA for all layers)",
        len(moe_layers), len(batches), sorted(L),
    )
    run_calibration_pass(
        model, batches,
        registrations=registrations,
        moe_layers=moe_layers, device=device,
        progress_label="phase_b",
        per_batch_hooks=(progress_cb,),
    )

    # ---- STEP 7: finalize accumulators ------------------------------------
    # ``downproj_max`` may be absent from ``built`` when the V2
    # per_expert_max cache hit short-circuited its construction -- in
    # that case ``ctx.max_acc`` was populated directly by the cache
    # provider and there is no live accumulator to finalize.
    if "downproj_max" in built:
        built["downproj_max"].finalize()
    if "output_reservoir" in built:
        built["output_reservoir"].finalize()
    if "sink_routing" in built:
        built["sink_routing"].finalize()

    # ---- STEP 8: run the four candidate-detector plugins, in order --------
    # Unconditional — each plugin short-circuits internally on its own flag,
    # exactly mirroring the legacy run(). The four detectors are peers within
    # one "run" phase, so the phase walker drives them; the manifest slice
    # fixes their order.
    detector_plugins = (
        by_name["three_way_and"], by_name["aimer"],
        by_name["sink_token"], by_name["magnitude_topk"],
    )
    walk_phases(("run",), detector_plugins, ctx)
    candidates = ctx.get("candidate_bag").to_provenance_dict()
    ctx.set("candidates", candidates)
    log.info(
        "Stage 1 Phase C: collected %d candidates (P99.5=%.3g, a_max_threshold=%.3g)",
        len(candidates), ctx.get("p995"), ctx.get("a_max_threshold"),
    )

    # ---- STEP 9: ablation_filter ------------------------------------------
    log.info("Stage 1 Phase D: ablating %d candidates", len(candidates))
    by_name["ablation_filter"].run(ctx)       # mandatory plugin

    # ---- STEP 10: cka_distance + grape_merge ------------------------------
    by_name["cka_distance"].run(ctx)          # writes D_matrices
    # Free the big expert-output reservoir — matches the legacy
    # ``del output_acc``.
    ctx.drop("output_acc")
    by_name["grape_merge"].run(ctx)           # writes the 5 budget slots

    # ---- STEP 11: assemble + write artifacts ------------------------------
    blacklist_path, budgets_path = _write_artifacts(ctx, by_name)
    _emit_telemetry(ctx, by_name)
    return blacklist_path, budgets_path


# ---------------------------------------------------------------------------
# Artifact assembly (plan §4.6).
# ---------------------------------------------------------------------------


def _write_artifacts(ctx: PipelineContext, by_name: dict) -> tuple[Path, Path]:
    """Assemble + write the three Stage 1 JSON artifacts.

    Writes ``stage1_blacklist.json`` (7-key schema via
    :class:`ArtifactBuilder`), ``stage1_ablation_filter.json``, and
    ``stage1_budgets.json``. Returns ``(blacklist_path, budgets_path)``.
    """
    artifacts_dir = ctx.get("artifacts_dir")
    config = ctx.get("config")
    s1 = config["stage1_grape"]
    se_cfg = s1.get("super_expert_detection", {})
    L = ctx.get("L")
    blacklist = ctx.get("blacklist")
    candidates = ctx.get("candidates")
    max_acc = ctx.get("max_acc")

    blacklist_out = {str(li): sorted(es) for li, es in blacklist.items()}

    # blacklist_provenance — verbatim from the legacy Stage 1 module.
    provenance: dict[str, list[str]] = {}
    for li_str, exps in blacklist_out.items():
        for e in exps:
            tags = candidates.get((int(li_str), int(e)), [])
            provenance[f"L{li_str}E{e}"] = list(tags)

    # blacklist_config — the inner ``config`` block (15 keys). Verbatim from
    # the legacy Stage 1 module. The three-way-AND statistics come from ctx
    # slots written by ThreeWayAndPlugin.run.
    p995 = ctx.get("p995")
    a_max = ctx.get("a_max")
    a_max_threshold = ctx.get("a_max_threshold")
    blacklist_config = {
        "a_max_fraction": float(se_cfg.get("a_max_fraction", 0.1)),
        "ma_ratio": float(se_cfg.get("ma_ratio", 100.0)),
        "ma_growth_ratio": float(se_cfg.get("ma_growth_ratio", 3.0)),
        "moe_output_growth_ratio": float(se_cfg.get("moe_output_growth_ratio", 2.0)),
        "ma_formation_layers": sorted(L),
        "p995_threshold": float(p995),
        "a_max_absolute": float(a_max),
        "a_max_threshold": float(a_max_threshold),
        "aimer_bottom_pct": float(se_cfg.get("aimer_bottom_pct", 0.01)),
        "aimer_layer_max_fraction": float(se_cfg.get("aimer_layer_max_fraction", 0.1)),
        "sink_token_score_ratio": float(se_cfg.get("sink_token_score_ratio", 10.0)),
        "sink_token_freq_threshold": float(se_cfg.get("sink_token_freq_threshold", 0.99)),
        "sink_token_max_per_layer_cap": int(se_cfg.get("sink_token_max_per_layer_cap", 10)),
        "magnitude_topk_per_l_layer": int(se_cfg.get("magnitude_topk_per_l_layer", 16)),
        "ablation_filter_threshold": float(
            s1.get("ablation_filter", {}).get("blacklist_threshold", 0.001)),
    }

    builder = ArtifactBuilder()
    builder.set_top_level("blacklist", blacklist_out)
    builder.set_top_level("per_expert_max", {
        f"L{li}E{e}": v for (li, e), v in max_acc.per_expert_max.items()
    })
    builder.set_top_level("config", blacklist_config)
    builder.set_top_level("blacklist_provenance", provenance)
    # 3 plugin fragments — ma_detection / aimer / sink_token. Both AIMER and
    # sink_token read ctx["candidates"] (set in Step 8); their disabled paths
    # still return well-formed (mostly-empty) dicts.
    builder.add_fragment("dual_signal",
                         by_name["ma_detection"].contribute_artifact(ctx))
    builder.add_fragment("aimer",
                         by_name["aimer"].contribute_artifact(ctx))
    builder.add_fragment("sink_token",
                         by_name["sink_token"].contribute_artifact(ctx))
    blacklist_payload = builder.assemble(
        required_keys=REQUIRED_BLACKLIST_TOP_LEVEL_KEYS,
    )   # validates the 7-key schema

    blacklist_path = artifacts_dir / "stage1_blacklist.json"
    save_json_artifact(blacklist_payload, blacklist_path)

    # stage1_ablation_filter.json — whole-file contributor. Stash the WIDE
    # ablation_filter_config (5 keys) before contribute_artifact — verbatim
    # from the legacy Stage 1 module.
    ctx.set("ablation_filter_config", {
        "holdout_samples": int(s1.get("ablation_filter", {}).get("holdout_samples", 100)),
        "magnitude_topk_per_l_layer": blacklist_config["magnitude_topk_per_l_layer"],
        "ablation_filter_threshold": blacklist_config["ablation_filter_threshold"],
        "ablation_filter_batch_size": int(s1.get("ablation_filter", {}).get("batch_size", 8)),
        "ma_formation_layers": sorted(L),
    }, overwrite=True)
    af_payload = by_name["ablation_filter"].contribute_artifact(ctx)
    save_json_artifact(af_payload, artifacts_dir / "stage1_ablation_filter.json")
    log.info(
        "Stage 1 Phase D: ablation-filter blacklisted %d / %d candidates → %s",
        sum(len(v) for v in blacklist_out.values()), len(candidates),
        blacklist_path,
    )

    # stage1_budgets.json — whole-file contributor.
    budgets_payload = by_name["grape_merge"].contribute_artifact(ctx)
    budgets_path = artifacts_dir / "stage1_budgets.json"
    save_json_artifact(budgets_payload, budgets_path)
    target_vals = [int(v) for v in budgets_payload["per_layer_target_experts"].values()]
    log.info(
        "Stage 1 complete — budgets range=[%d..%d] mean=%.1f → %s",
        min(target_vals), max(target_vals),
        statistics.fmean(target_vals), budgets_path,
    )

    return blacklist_path, budgets_path


# ---------------------------------------------------------------------------
# Telemetry (plan §4.7) — verbatim port of the legacy Stage 1 telemetry.
# ---------------------------------------------------------------------------


def _emit_telemetry(ctx: PipelineContext, by_name: dict) -> None:
    """Emit the Phase A/C/D Trackio block.

    The GRAPE per-layer + summary emits already fire inside
    ``GrapeMergePlugin.run`` — this only emits the SE-detection summary.
    Telemetry ordering does not affect the JSON artifacts.
    """
    L = ctx.get("L")
    moe_layers = ctx.get("moe_layers")
    blacklist = ctx.get("blacklist")
    max_acc = ctx.get("max_acc")
    p995 = ctx.get("p995")
    a_max = ctx.get("a_max")
    se_cfg = ctx.get("config")["stage1_grape"].get("super_expert_detection", {})
    a_max_fraction = float(se_cfg.get("a_max_fraction", 0.1))
    total_experts = sum(ref.num_routed_experts for ref in moe_layers)
    blacklist_out = {str(li): sorted(es) for li, es in blacklist.items()}

    _trackio_log({
        "stage1/ma_formation_layers_count": len(L),
        "stage1/total_experts": int(total_experts),
        "stage1/p995_threshold": float(p995),
        "stage1/a_max": float(a_max),
        "stage1/a_max_threshold": float(a_max_fraction * a_max),
        "stage1/n_blacklisted": int(sum(len(v) for v in blacklist_out.values())),
    })
    for ref in moe_layers:
        in_ma = ref.layer_idx in L
        entry: dict = {
            "stage1/se_layer_idx": ref.layer_idx,
            "stage1/se_blacklisted": len(blacklist.get(ref.layer_idx, [])),
            "stage1/se_in_ma_layer": float(in_ma),
        }
        if in_ma:
            vals = [v for (li, _e), v in max_acc.per_expert_max.items()
                    if li == ref.layer_idx]
            if vals:
                entry["stage1/se_down_max_mean"] = float(statistics.fmean(vals))
                entry["stage1/se_down_max_std"] = (
                    float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0)
                entry["stage1/se_down_max_max"] = float(max(vals))
        _trackio_log(entry)
    _trackio_flush()
