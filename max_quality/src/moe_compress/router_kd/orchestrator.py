"""Router-KD orchestrator — the real plugin-driven phase sequencer (RK-8).

RK-1 shipped this module as a thin delegation to the legacy Router-KD monolith
:func:`moe_compress.stage5_router_kd.run`. RK-2..RK-7 extracted the unified KD
algorithm (the Stage 2.5 + Stage 5 router fine-tuning loop) into
``router_kd/plugins/``. RK-8 flips the relationship: :func:`run` here is now the
REAL orchestrator and ``stage5_router_kd.run`` is a thin shim that delegates to
it.

The schedule
------------
SETUP: ``_set_experts_implementation`` (glue) -> ``load_teacher_cache`` ->
``setup_trainable_scope`` -> ``setup_merge_repair`` -> calibration build +
``total_optim_steps`` (glue) -> ``build_optimizer`` -> ``torch.compile`` (glue)
-> checkpoint-resume restore (glue, publishes ``resume_*`` slots) ->
``setup_early_stop`` -> scheduler resume fast-forward (glue) -> trackio config
emit (glue).

TRAINING LOOP (orchestrator-owned plain ``for epoch / for batch / grad-accum
boundary`` -- NOT ``loop_over``, all hooks dispatched against the ROOT ctx):
per-batch ``dispatch_first("provide_teacher_logits", ...)`` -> live-teacher
glue -> student forward + ``[:, :-1, :]`` shift -> publish ``teacher_logits`` /
``student_logits`` -> ``walk_phases(("compute_merge_repair_mse",
"compute_kd_loss"), ...)`` -> read ``kd_loss`` / ``vocab_kl`` -> backward /
grad-accum / optim.step / scheduler.step (glue) -> per-log-window: publish
``step`` / ``epoch`` / ``raw_kl_val`` -> ``walk_phases(("update_best_tracker",
"check_early_stop"), ...)`` -> log + trackio (glue) -> periodic
``_save_stage5_checkpoint`` (glue) -> early-stop break.

FINALIZE: ``teardown_merge_repair`` -> ``reload_best_checkpoint`` ->
``save_compressed_checkpoint`` (glue) -> ``return out_dir``.

Division of labour
------------------
The plugin hooks own only their algorithm; the orchestrator owns the loop
counters / grad-accum / resume bookkeeping. Every run-glue block below is a
verbatim copy from the monolith ``run()``, just reorganized around the
``walk_phases`` / ``dispatch_first`` calls.

ROOT ctx, not child
-------------------
All hooks are dispatched against the ROOT :class:`PipelineContext` via plain
``for`` loops + ``walk_phases`` / ``dispatch_first`` -- NOT ``loop_over``.
``EarlyStopPlugin`` rebinds ``prev_ema`` / etc. with ``overwrite=True``; a
child scope would shadow them and break the EMA carry. The fast-forward
``continue`` and the early-stop ``break`` require a plain loop anyway.

Set-once collisions
-------------------
``PipelineContext.set`` is set-once. Per microbatch ``VocabKdPlugin`` /
``MergeRepairPlugin`` do plain ``ctx.set`` on ``kd_loss`` / ``vocab_kl`` /
``merge_repair_mse_term`` / ``merge_repair_mse_weight`` /
``teacher_layer_outputs`` -- so the orchestrator ``run_ctx.drop(...)`` those
slots before each microbatch's compute phase (or the 2nd microbatch raises
KeyError). For ``teacher_logits`` / ``student_logits`` / ``step`` / ``epoch`` /
``raw_kl_val`` that the orchestrator sets each iteration, ``overwrite=True``.

Merge-repair teacher-capture timing
-----------------------------------
The monolith registers the teacher ``_LayerOutputCapture`` BEFORE the teacher
forward of the same batch. With ``dispatch_first`` the forward happens inside
``TeacherLivePlugin.provide_teacher_logits``, so when merge-repair is active
the orchestrator pre-loads the live teacher (``_load_teacher``), registers +
clears the capture, THEN calls ``dispatch_first`` -- keeping the capture in
place before the forward, byte-identical to the monolith.

Monkeypatch survival (HAZARD H3)
--------------------------------
The golden / smoke tests ``monkeypatch.setattr`` ``build_calibration_tensor``
and ``_trackio_log`` on whichever module binds them. The orchestrator binds
both by direct import here, so the test fixtures patch
``router_kd.orchestrator.build_calibration_tensor`` /
``router_kd.orchestrator._trackio_log``. ``load_model`` is bound by
``TeacherLivePlugin`` (``router_kd.plugins.teacher``). ``save_compressed_checkpoint``
the orchestrator calls REAL -- the golden pins the metadata it writes.
"""
from __future__ import annotations

import logging
import os
import queue
import threading
from pathlib import Path

import torch
import torch.nn as nn

from ..pipeline.context import PipelineContext
from ..pipeline.registry import PluginRegistry
from ..tools.phase_walker import walk_phases

from ..utils.calibration import (
    build_calibration_tensor,
    iter_batches,
    shared_calibration_cache_dir,
    spec_from_config,
)
from ..utils.model_io import (
    iter_moe_layers,
    save_compressed_checkpoint,
)
from ..utils.runtime_monitor import snapshot_telemetry as _rt_snap, update as _rt_update
from ..utils.trackio_log import trackio_log as _trackio_log

from .plugins.trainable_scope import TrainableScopePlugin
from .plugins.kd_optimizer import KdOptimizerPlugin, _move_optimizer_state_to_device
from .plugins.vocab_kd import (
    VocabKdPlugin,
    _check_param_sanity,
    _dump_nan_diagnostics,
    _log_first_batch_sanity,
)
from .plugins.teacher import TeacherCachePlugin, TeacherLivePlugin
from .plugins.merge_repair import MergeRepairPlugin, _LayerOutputCapture
from .plugins.early_stop import EarlyStopPlugin
from .plugins.rkd_paper_recipe import RkdPaperRecipePlugin

log = logging.getLogger(__name__)


def _set_experts_implementation(model: nn.Module, impl: str) -> None:
    """Override the MoE experts forward dispatch on `model`.

    The `transformers.integrations.moe` decorator dispatches each MoE forward
    by reading `self.config._experts_implementation` at every call (see
    `ExpertsInterface.get_interface`), so this assignment takes effect for
    all subsequent forwards without rebuilding the model. The valid values
    registered in transformers v4.x's `ALL_EXPERTS_FUNCTIONS`:

      * `"grouped_mm"`  - default; uses `torch.nn.functional.grouped_mm`.
                          DEADLOCKS on Blackwell sm_100 (see project memory
                          `project_grouped_mm_blackwell.md`). Do NOT use on
                          B200 / GB200 / B300.
      * `"batched_mm"`  - uses `torch.bmm` per expert group with padding to
                          max active count. ~70-90% of grouped_mm's speed,
                          but bmm is universally supported (Hopper +
                          Blackwell). Recommended default on B200.
      * `"sonicmoe"`    - custom kernel registered by the sonicmoe package.
                          Performance unknown; Blackwell-compatibility
                          unknown. Try as fallback if `batched_mm` is too
                          slow or hits an issue.
      * `"eager"`       - Python loop over active experts, one
                          `nn.functional.linear` per expert. Universally
                          compatible. ~30-50% of grouped_mm's speed.

    Sets the implementation on both the multimodal-level `config` and the
    inner `text_config` if the model is multimodal (Qwen3_5MoeForConditionalGeneration).
    """
    base = getattr(model, "_orig_mod", model)
    cfg = base.config
    if hasattr(cfg, "text_config"):
        cfg.text_config._experts_implementation = impl
    cfg._experts_implementation = impl
    log.info("Stage 5: MoE experts_implementation = %r (forward dispatch via "
             "transformers.integrations.moe.ExpertsInterface)", impl)


def run(
    student,
    tokenizer,
    config: dict,
    artifacts_dir: Path,
    *,
    device=None,
    no_resume: bool = False,
    stage_key: str = "stage5",
) -> Path:
    """Run Router KD via the plugin pipeline.

    ``stage_key`` controls the partial-dir and output-dir names, allowing the
    same code to serve both Stage 2.5 (``stage_key="stage2p5"``) and Stage 5
    (``stage_key="stage5"``).  The config section read is always
    ``stage5_router_kd`` regardless of ``stage_key``.
    """
    # Plugin #7 — RKD paper-recipe (Row P) config overrides. MUST run before
    # any ``config[...]`` capture below (s5 / cal binds the live dicts). The
    # method is a no-op when ``stage5_router_kd.rkd_recipe`` is anything other
    # than ``"paper"``, so Row C runs are byte-identical to pre-plugin behavior.
    # See router_kd/plugins/rkd_paper_recipe.py for the 4 deltas + contract.
    RkdPaperRecipePlugin().apply_config_overrides(config)

    s5 = config["stage5_router_kd"]
    cal = config["calibration"]

    # Validate stage_key early (verbatim from the monolith) - the seed_offset
    # branch below relies on it being one of the two accepted values.
    if stage_key == "stage2p5":
        _seed_offset = 25
    elif stage_key == "stage5":
        _seed_offset = 5
    else:
        raise ValueError(
            f"Stage 5: unsupported stage_key={stage_key!r}; expected "
            "'stage5' or 'stage2p5'."
        )

    # ---- one PipelineContext: input slots + run-glue intermediates --------
    run_ctx = PipelineContext()
    run_ctx.set("student", student)
    run_ctx.set("model", student)
    run_ctx.set("tokenizer", tokenizer)
    run_ctx.set("config", config)
    run_ctx.set("artifacts_dir", artifacts_dir)
    run_ctx.set("device", device)
    run_ctx.set("stage_key", stage_key)
    run_ctx.set("no_resume", no_resume)

    # Cache BEFORE live so dispatch_first prefers the cache on a hit.
    registry = PluginRegistry([
        TrainableScopePlugin(),
        KdOptimizerPlugin(),
        VocabKdPlugin(),
        TeacherCachePlugin(),
        TeacherLivePlugin(),
        MergeRepairPlugin(stage_key=stage_key),
        EarlyStopPlugin(),
    ])
    plugins = registry.enabled(config)
    # Keep a direct reference to the live-teacher plugin so the live-teacher
    # glue below can publish the loaded teacher onto run_ctx ("teacher" slot)
    # for the merge-repair lazy teacher-capture registration, and pre-load it
    # when merge-repair needs the capture in place before the teacher forward.
    # Source it from `plugins` (the enabled subset dispatch_first walks), not
    # the full registry — so the pre-loaded teacher and the plugin that
    # actually answers provide_teacher_logits are always the same object.
    _teacher_live = next(
        (p for p in plugins if isinstance(p, TeacherLivePlugin)), None
    )

    # Set MoE forward dispatch (default 'batched_mm' to work around the
    # grouped_mm Blackwell deadlock - see _set_experts_implementation
    # docstring). Env var `EXPERTS_IMPLEMENTATION` overrides the YAML default
    # for quick A/B without redeploying config.
    _experts_impl = os.environ.get(
        "EXPERTS_IMPLEMENTATION", s5.get("experts_implementation", "batched_mm")
    )
    _set_experts_implementation(student, _experts_impl)

    # --- torch.compile acceleration (spec section 8) ---
    use_compile = bool(s5.get("torch_compile", False))

    # ---- load_teacher_cache ----------------------------------------------
    # TeacherCachePlugin.load_teacher_cache resolves + validates the sidecar
    # and publishes the (validated payload | None) to teacher_logits_cache.
    # The plugin is in `plugins` only when teacher_logits_cache is configured.
    walk_phases(("load_teacher_cache",), plugins, run_ctx)
    teacher_logits_cache = (
        run_ctx.get("teacher_logits_cache")
        if run_ctx.has("teacher_logits_cache")
        else None
    )

    # ---- setup_trainable_scope -------------------------------------------
    # TrainableScopePlugin.setup_trainable_scope runs the trainable/frozen
    # pattern-conflict check and freezes every non-router parameter BEFORE
    # the student is compiled.
    walk_phases(("setup_trainable_scope",), plugins, run_ctx)

    # ---- setup_merge_repair ----------------------------------------------
    # MergeRepairPlugin.setup_merge_repair (enabled only at stage2p5 with the
    # config flag) unfreezes merged centroids + registers the student capture,
    # publishing merge_repair_grad_handles / merge_repair_layers /
    # merge_repair_mse_weight / merge_repair_student_capture. MUST run AFTER
    # setup_trainable_scope and BEFORE build_optimizer (the optimizer reads
    # merge_repair_grad_handles for its split param-group).
    walk_phases(("setup_merge_repair",), plugins, run_ctx)
    _merge_repair_layers = (
        run_ctx.get("merge_repair_layers")
        if run_ctx.has("merge_repair_layers")
        else []
    )
    _merge_repair_layer_indices: list[int] = [
        ref.layer_idx for ref, _ in _merge_repair_layers
    ]
    _merge_repair_active = bool(_merge_repair_layer_indices)

    # seed_offset distinguishes Stage 2.5 (post-merge router KD) from
    # Stage 5 (final router KD) so each pass sees a different calibration
    # draw and the routers are not retrained on the identical 3000 sequences.
    spec = spec_from_config(
        cal,
        num_sequences_override=s5["max_calibration_samples"],
        sequence_length_override=s5["max_sequence_length"],
        seed_offset=_seed_offset,
    )
    calib = build_calibration_tensor(
        tokenizer, spec,
        cache_dir=(os.environ.get("MOE_CALIB_CACHE_DIR") or shared_calibration_cache_dir(artifacts_dir)),
    )
    batches = iter_batches(calib, batch_size=s5["batch_size"])
    grad_accum = s5["gradient_accumulation"]
    # Distillation temperature - a CONSTANT, deliberately not a curriculum.
    # Post-merge router recovery is self-healing, not cross-capacity
    # distillation: the student is the (compressed) teacher being repaired, so
    # Hinton's soft-target "dark knowledge" rationale does not apply. T>1 only
    # optimizes a softened proxy distribution nobody runs at serve time, and a
    # downward *ramp* additionally makes the logged KL non-stationary
    # (loss/T^2 is the softened-distribution KL - it drifts with T regardless
    # of model quality), which corrupts the save-best / early-stop guards. T=1
    # makes the loss the true forward-KL to the teacher == the deploy target.
    T = float(s5.get("kd_temperature", 1.0))
    ckpt_every = int(s5.get("checkpoint_every_n_steps", 100))
    if ckpt_every <= 0:
        raise ValueError(
            f"Stage 5: checkpoint_every_n_steps={ckpt_every} disables "
            "checkpointing; spec section 8 Resume mandates step-boundary "
            "checkpointing every N optimizer steps. Set a positive integer "
            "(default 100) - disabling resume would silently lose progress "
            "on a long-running stage."
        )

    # Teacher/student MoE layer count sanity check (router structure must match
    # even though we're distilling at vocab level - the student's routers are
    # what we're training). The live-teacher plugin runs the teacher-side
    # topology guard; here only the count is needed for that plugin's read.
    _ = sum(1 for _ in iter_moe_layers(getattr(student, "_orig_mod", student)))

    # total_optim_steps MUST be computed before build_optimizer - the plugin
    # builds the optimizer + scheduler together and the scheduler needs it.
    total_steps = (len(batches) // grad_accum) * s5["epochs"]
    total_optim_steps = total_steps  # alias used by scheduler + T-ramp
    run_ctx.set("total_optim_steps", total_optim_steps)

    # ---- build_optimizer -------------------------------------------------
    # KdOptimizerPlugin.build_optimizer constructs the AdamW optimizer (split
    # param-group when merge-repair is active) + the warmup+cosine LambdaLR
    # scheduler, publishing optimizer / lr_scheduler.
    walk_phases(("build_optimizer",), plugins, run_ctx)
    optim = run_ctx.get("optimizer")
    scheduler = run_ctx.get("lr_scheduler")

    # torch.compile applied AFTER freeze+optimizer construction so the compiled
    # graph reflects the final frozen parameter layout. named_parameters() on
    # the compiled wrapper delegates to the underlying module - the optimizer
    # already holds the correct parameter references before compilation.
    # mode='default' (not 'reduce-overhead'): the latter captures CUDA graphs
    # and replays them. On Blackwell B200 with Qwen3.6-A3B's MoE grouped_mm,
    # CUDA graph replay deadlocked after ~1400 steps of sustained training on
    # the 2026-05-13 A0 run - main thread stuck inside
    # `torch.nn.functional.grouped_mm` via `transformers.integrations.moe._grouped_linear`,
    # faulthandler thread dump confirmed kernel-level hang (not a Python exception).
    # The previous 375-step run completed fine - same code path - so the failure
    # mode is sustained-training + CUDA-graph state accumulation specific to
    # `reduce-overhead`. `mode='default'` keeps TorchDynamo + TorchInductor
    # fusion (~ 50-70% of reduce-overhead's speedup) but launches kernels
    # individually, avoiding graph replay entirely.
    if use_compile:
        try:
            log.info("Stage 5: torch.compile(student, mode='default')")
            student = torch.compile(student, mode="default")
        except Exception as exc:
            log.warning("Stage 5: torch.compile failed (%s) - falling back to eager mode", exc)
            use_compile = False
    # Re-publish the (possibly compiled-wrapped) student so downstream hooks
    # read the same object the orchestrator forwards into the training loop.
    run_ctx.set("student", student, overwrite=True)
    run_ctx.set("model", student, overwrite=True)

    # -----------------------------------------------------------------------
    # Crash-resume: find latest checkpoint and restore router + optim state.
    # -----------------------------------------------------------------------
    resume_step = 0
    resume_epoch = 0
    resume_batch_i = -1
    # Captured from a v2 (or later) resume checkpoint and consumed after the
    # LR scheduler / best-tracker locals are constructed below. v1 checkpoints
    # leave these as None - the scheduler is fast-forwarded by replaying
    # scheduler.step() resume_step times; best-tracker re-initializes from +inf.
    _resume_scheduler_state = None
    _resume_best_raw_kl_ema = None
    _resume_best_step = None
    _resume_prev_ema = None
    _resume_no_improve_windows = None
    _resume_es_ref_ema = None

    # `no_resume=True` means "don't pick up existing checkpoints"; it does
    # NOT mean "don't write new ones". Spec section 8 mandates step-boundary
    # checkpointing every N optimizer steps - disabling future writes would
    # silently lose all progress on a crash mid-run.
    partial_dir = artifacts_dir / f"_{stage_key}_partial"
    if no_resume:
        # Delete any stale partial dir from a prior run so the search for a
        # latest checkpoint below finds none, then recreate empty for fresh
        # writes during this run.
        if partial_dir.exists():
            import shutil as _shutil
            _shutil.rmtree(partial_dir, ignore_errors=True)
    partial_dir.mkdir(parents=True, exist_ok=True)
    for _stale in partial_dir.glob("*.tmp"):
        _stale.unlink(missing_ok=True)
    if not no_resume:

        ckpts = sorted(
            partial_dir.glob("step_*.pt"),
            key=lambda p: int(p.stem.split("_")[1]),
        )
        if ckpts:
            latest = ckpts[-1]
            try:
                payload = torch.load(latest, map_location="cpu")
            except Exception as exc:
                raise RuntimeError(
                    f"Stage 5 resume: failed to load checkpoint {latest}: {exc}"
                ) from exc
            fv = int(payload.get("format_version", 0))
            if fv not in (1, 2):
                raise RuntimeError(
                    f"Stage 5 checkpoint {latest} has format_version={fv} "
                    f"(expected 1 or 2) - delete _{stage_key}_partial/ and re-run"
                )
            # Restore router parameters into the student model.
            # F3 fix: walk the attribute tree on the unwrapped module so that
            # parameter names saved by _save_stage5_checkpoint (which also uses
            # the unwrapped module) resolve correctly even when `student` is a
            # torch.compile wrapper.
            _restore_base = getattr(student, "_orig_mod", student)
            for pname, t in payload["router_state"].items():
                parts = pname.split(".")
                obj = _restore_base
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                getattr(obj, parts[-1]).data.copy_(t)
            # Validate that the optimizer state's param groups match the current
            # trainable scope by comparing param-name sets, not just counts.
            # If the trainable_name_patterns config changed since the checkpoint
            # was written, even matching counts could pair stale moments with
            # the wrong parameters.
            _ckpt_names: set[str] = set(payload.get("trainable_param_names", []))
            # Use unwrapped student so that names match what
            # _save_stage5_checkpoint persisted (also unwrapped). With
            # torch_compile=true, student.named_parameters() returns
            # `_orig_mod.*`-prefixed names that wouldn't match the saved
            # unprefixed set, causing every resume to falsely fail the
            # trainable-scope-changed check.
            _unwrapped_for_resume = getattr(student, "_orig_mod", student)
            _current_names = {
                n for n, p in _unwrapped_for_resume.named_parameters() if p.requires_grad
            }
            if _ckpt_names and _ckpt_names != _current_names:
                added = sorted(_current_names - _ckpt_names)
                removed = sorted(_ckpt_names - _current_names)
                raise RuntimeError(
                    f"Stage 5 resume: trainable parameter set changed since "
                    f"checkpoint - added={added[:5]}{'...' if len(added) > 5 else ''}, "
                    f"removed={removed[:5]}{'...' if len(removed) > 5 else ''}. "
                    f"Delete _{stage_key}_partial/ and re-run, or restore the "
                    f"original trainable_name_patterns."
                )
            optim.load_state_dict(payload["optim_state"])
            # Move optimizer state to wherever the trainable params actually
            # live, not just the explicit `device` arg. Under HF
            # `device_map="auto"` the caller may pass `device=None` while
            # params reside on CUDA; loading from a CPU checkpoint without
            # this move would crash on the first optim.step() with a
            # device-mismatch.
            try:
                _trainable_devices = {p.device for p in student.parameters() if p.requires_grad}
                if len(_trainable_devices) == 1:
                    _move_optimizer_state_to_device(optim, next(iter(_trainable_devices)))
                elif device is not None:
                    _move_optimizer_state_to_device(optim, device)
                else:
                    log.warning(
                        "Stage 5 resume: trainable parameters span %d devices (%s) "
                        "and `device` is None - optimizer state left on its loaded "
                        "device; subsequent optim.step() may fail with a CPU/CUDA "
                        "mismatch on multi-device sharded resumes.",
                        len(_trainable_devices), sorted(str(d) for d in _trainable_devices),
                    )
            except Exception as exc:  # noqa: BLE001
                log.warning("Stage 5 resume: optimizer state device migration failed (%s); proceeding", exc)
            resume_step = int(payload["step"])
            resume_epoch = int(payload["epoch"])
            resume_batch_i = int(payload["batch_idx"])
            if "gradient_accumulation" in payload:
                saved_ga = int(payload["gradient_accumulation"])
                if saved_ga != grad_accum:
                    raise RuntimeError(
                        f"Stage 5 resume: gradient_accumulation mismatch - "
                        f"checkpoint has {saved_ga}, config has {grad_accum}. "
                        f"Delete _{stage_key}_partial/ and re-run or align the config."
                    )
            # Capture v2+ fields into outer-scope holders; applied after the
            # LR scheduler is constructed below (which needs total_optim_steps
            # to exist).
            _resume_scheduler_state = payload.get("scheduler_state")
            if payload.get("best_raw_kl_ema") is not None:
                _resume_best_raw_kl_ema = float(payload["best_raw_kl_ema"])
                _resume_best_step = int(payload.get("best_step", -1))
            if payload.get("prev_ema") is not None:
                _resume_prev_ema = float(payload["prev_ema"])
            # `no_improve_windows` is absent from pre-2026-05-17 (and v1)
            # checkpoints - leave it None there so patience starts fresh.
            if payload.get("no_improve_windows") is not None:
                _resume_no_improve_windows = int(payload["no_improve_windows"])
            if payload.get("es_ref_ema") is not None:
                _resume_es_ref_ema = float(payload["es_ref_ema"])
            log.info("Stage 5: resumed from step %d (epoch %d, batch %d)",
                     resume_step, resume_epoch, resume_batch_i)

    # `train()` enables dropout / batchnorm-train semantics on every submodule.
    # For Qwen3-30B-A3B and the production target (no dropout, RMSNorm only),
    # this is a no-op. For architectures with dropout in attention or MLP, the
    # frozen submodules would still emit dropped activations - a silent
    # train-vs-eval mismatch that could distort the KD signal. Spec section 8
    # requires only `mlp.gate.weight` to be trainable; the rest of the module
    # tree is frozen via _freeze_non_routers but stays in train mode here. If a
    # future architecture variant introduces dropout, set frozen submodules to
    # inference mode.
    student.train()
    remaining_steps = max(0, total_steps - resume_step)

    # --- Resume restore for the scheduler ---------------------------------
    # v2 checkpoint: load scheduler state verbatim. v1 checkpoint (or no
    # resume): fast-forward scheduler by replaying scheduler.step() resume_step
    # times so the LR curve picks up at the right point.
    if _resume_scheduler_state is not None:
        scheduler.load_state_dict(_resume_scheduler_state)
    elif resume_step > 0:
        for _ in range(resume_step):
            scheduler.step()

    # ---- setup_early_stop ------------------------------------------------
    # Publish the resume_* slots EarlyStopPlugin.setup_early_stop reads, then
    # dispatch the hook - it seeds the best-tracker + early-stop state and
    # applies the resume-restore, publishing best_ema_alpha / save_best /
    # best_raw_kl_ema / best_step / prev_ema / early_stop_patience /
    # no_improve_windows / es_ref_ema.
    run_ctx.set("partial_dir", partial_dir)
    if _resume_best_raw_kl_ema is not None:
        run_ctx.set("resume_best_raw_kl_ema", _resume_best_raw_kl_ema)
        run_ctx.set("resume_best_step", _resume_best_step)
    if _resume_prev_ema is not None:
        run_ctx.set("resume_prev_ema", _resume_prev_ema)
    if _resume_no_improve_windows is not None:
        run_ctx.set("resume_no_improve_windows", _resume_no_improve_windows)
    if _resume_es_ref_ema is not None:
        run_ctx.set("resume_es_ref_ema", _resume_es_ref_ema)
    walk_phases(("setup_early_stop",), plugins, run_ctx)
    best_raw_kl_ema = float(run_ctx.get("best_raw_kl_ema"))
    best_step = int(run_ctx.get("best_step"))
    _early_stop_patience = int(run_ctx.get("early_stop_patience"))

    _trainable_param_count = sum(1 for _ in student.parameters() if _.requires_grad)
    log.info("Stage 5: %d routers trainable; %d steps total / %d remaining (grad-accum=%d, resume_step=%d)",
             _trainable_param_count,
             total_steps, remaining_steps, grad_accum, resume_step)
    trailing = len(batches) % grad_accum

    # One-shot Trackio emit of Stage 5 / Stage 2.5 run-level config. All
    # values already in scope. Note: namespace is hardcoded "stage5/" even
    # when stage_key=="stage2p5"; per user direction the bug fix is out of
    # scope. Operators can disambiguate via stage5/config/stage_key.
    _trackio_log({
        "stage5/config/stage_key": str(stage_key),
        "stage5/config/total_steps_planned": int(total_steps),
        "stage5/config/remaining_steps": int(remaining_steps),
        "stage5/config/calib_num_batches": int(len(batches)),
        "stage5/config/calib_num_samples": int(spec.num_sequences),
        "stage5/config/calib_seq_len": int(spec.sequence_length),
        "stage5/config/calib_total_tokens": int(spec.num_sequences * spec.sequence_length),
        "stage5/config/grad_accum": int(grad_accum),
        "stage5/config/epochs": int(s5["epochs"]),
        "stage5/config/kd_temperature": T,
        "stage5/config/early_stop_patience": int(_early_stop_patience),
        "stage5/config/shuffle_batches_each_epoch": bool(
            s5.get("shuffle_batches_each_epoch", False)
        ),
        "stage5/config/lr_schedule": str(s5.get("lr_schedule", "none")),
        "stage5/config/warmup_ratio": float(s5.get("warmup_ratio", 0.05)),
        "stage5/config/lr_min_ratio": float(s5.get("lr_min_ratio", 0.10)),
        "stage5/config/trainable_router_params": int(_trainable_param_count),
        "stage5/config/use_compile_student": bool(use_compile),
        "stage5/config/teacher_cache_hit": bool(teacher_logits_cache is not None),
        "stage5/config/trailing_batches_dropped": int(trailing),
    })
    if trailing != 0:
        log.warning(
            "Stage 5: %d trailing batches per epoch will not form a complete "
            "grad-accum window (grad_accum=%d) - their gradients are dropped "
            "at each epoch end (x%d epochs).",
            trailing, grad_accum, int(s5["epochs"]),
        )

    # Spec D-protocol-blend / section 8: when running multi-epoch KD with a
    # multi-epoch teacher-logits cache, the teacher cache index advances
    # as (epoch * len(batches) + i), so the same student input batch IDs
    # would be paired with *different* teacher logits across epochs unless
    # the teacher cache was generated against a fresh-per-epoch input
    # ordering. Refuse multi-epoch + cache combinations that we cannot
    # verify produce a consistent input-vs-logit pairing. (Single-epoch
    # default config is the canonical path; multi-epoch caches require a
    # pre-shuffled cache aligned to a deterministic per-epoch input order
    # the spec does not currently formalize.)
    if int(s5["epochs"]) > 1 and teacher_logits_cache is not None:
        raise RuntimeError(
            f"Stage 5: multi-epoch training (epochs={s5['epochs']}) with "
            "teacher_logits_cache is not supported - calibration batches are "
            "produced once and replayed identically across epochs, but the "
            "cache index advances per (epoch, batch), creating a teacher/"
            "student input mismatch in epochs >= 2. Either set epochs=1 or "
            "regenerate the cache against a deterministic per-epoch shuffle "
            "schedule that this code path also applies."
        )

    step = resume_step
    optim.zero_grad()
    # Tracks whether the first-batch sanity probe has run yet. A boolean is
    # used (instead of a `step == 0 and i == resume_batch_i + 1` predicate)
    # because `step` is the optimizer-step counter and stays > 0 across any
    # non-trivial resume - a step-based guard silently skips the probe on
    # the resumed path, which is when we need it most (router weights may
    # have been NaN-poisoned in the previous run).
    _first_batch_probed = False
    # --- Per-epoch batch reshuffle (2026-05-17 overfit fix, config-gated) ---
    # When shuffle_batches_each_epoch is true, each epoch iterates the batch
    # list in a permuted order to reduce identical-replay memorisation. The
    # permutation is derived from a per-epoch seed so it is fully deterministic
    # - crash-resume reconstructs the identical order, keeping the positional
    # `i <= resume_batch_i` fast-forward correct. Default false -> the order is
    # range(len(batches)) every epoch, byte-identical to pre-2026-05-17 `main`.
    _shuffle_epochs = bool(s5.get("shuffle_batches_each_epoch", False))
    if _shuffle_epochs and teacher_logits_cache is not None:
        raise RuntimeError(
            "Stage 5: shuffle_batches_each_epoch=true is incompatible with "
            "teacher_logits_cache - the cache is indexed positionally by "
            "(epoch * len(batches) + i), so a shuffled student-batch order "
            "would pair each batch with the wrong cached teacher logits. "
            "Disable one of the two."
        )
    _shuffle_seed = int(s5.get("seed", config.get("seed", 0)))

    def _epoch_batch_order(epoch_idx: int) -> list[int]:
        """Deterministic batch-index iteration order for one epoch."""
        n = len(batches)
        if not _shuffle_epochs:
            return list(range(n))
        g = torch.Generator()
        g.manual_seed(_shuffle_seed + epoch_idx)
        return torch.randperm(n, generator=g).tolist()

    # Async periodic-checkpoint writer (Tier-1 Lever C). A single background
    # thread serializes/fsyncs/replaces/prunes the step_*.pt files so the
    # ~815 ms write does not stall the training thread. None when the
    # kill-switch (STAGE5_ASYNC_CKPT=0) is set or there is no partial_dir, in
    # which case _save_stage5_checkpoint falls back to a fully-synchronous
    # write. best.pt (early_stop.py) stays synchronous regardless.
    _ckpt_writer = (
        _Stage5CheckpointWriter()
        if (partial_dir is not None and _async_ckpt_enabled())
        else None
    )
    for epoch in range(s5["epochs"]):
        if epoch < resume_epoch:
            continue
        # Accumulate detached loss tensors across the grad-accum + log windows
        # and pay one .item() sync at log-emission time, instead of paying
        # device->host sync per microbatch (~375/epoch at full scale).
        window_loss_acc: list[torch.Tensor] = []
        # raw_kl = loss / T^2 strips Hinton's gradient-scaling factor. With a
        # CONSTANT temperature (no ramp) it is stationary across the run, so it
        # is a sound save-best / early-stop metric; at the default T=1 it
        # equals the loss itself - the true forward-KL to the teacher.
        window_raw_kl_acc: list[torch.Tensor] = []
        # `i` is the iteration POSITION within the epoch (drives the grad-accum
        # window, resume fast-forward and checkpoint batch_idx). `_batch_order`
        # maps it to the actual batch index, which differs from `i` only when
        # shuffle_batches_each_epoch is on. With shuffle off, _batch_order is
        # range(len(batches)) so `batches[_batch_order[i]] is batches[i]`.
        _batch_order = _epoch_batch_order(epoch)
        for i, _batch_idx in enumerate(_batch_order):
            batch = batches[_batch_idx]
            # Fast-forward: skip batches already processed in the resumed run.
            # resume_batch_i is the last batch of the grad-accum window that
            # triggered the checkpoint - the optimizer step has already occurred
            # for that entire window (including batch resume_batch_i itself).
            # Skip 0..resume_batch_i (inclusive) to avoid re-running the already-
            # completed step and triggering a spurious duplicate optimizer step.
            if epoch == resume_epoch and i <= resume_batch_i:
                continue

            if device is not None:
                batch = batch.to(device)

            # --- Vocabulary-level KD (paper 2603.02217, Eq. 3) ---
            # merge-repair teacher-capture timing: the monolith registers the
            # teacher capture BEFORE the teacher forward of the same batch.
            # dispatch_first runs the forward inside the live-teacher plugin,
            # so when merge-repair is active pre-load the live teacher here and
            # register + clear its detached capture before dispatch_first.
            if (
                _merge_repair_active
                and teacher_logits_cache is None
                and _teacher_live is not None
            ):
                _teacher_model = _teacher_live._load_teacher(run_ctx)
                if not run_ctx.has("teacher"):
                    run_ctx.set("teacher", _teacher_model)
                if not run_ctx.has("merge_repair_teacher_capture"):
                    run_ctx.set(
                        "merge_repair_teacher_capture",
                        _LayerOutputCapture(
                            _teacher_model,
                            set(_merge_repair_layer_indices),
                            detach=True,
                        ),
                    )
                run_ctx.get("merge_repair_teacher_capture").clear()

            # Teacher: dispatch_first walks TeacherCachePlugin then
            # TeacherLivePlugin - the cache wins on a hit, the live teacher
            # answers (running a no-grad forward) on a miss.
            teacher_vocab_logits = PluginRegistry.dispatch_first(
                plugins, "provide_teacher_logits", run_ctx,
                input_ids=batch, epoch=epoch, batch_index=i,
                num_batches=len(batches),
            )

            # Live-teacher glue - runs only on a cache MISS (the live teacher
            # plugin holds the loaded model). Publish the `teacher` slot for
            # merge-repair, snapshot the captured teacher MoE outputs before
            # the next forward clears them.
            _teacher_layer_outputs: dict = {}
            if teacher_logits_cache is None:
                _teacher_model = (
                    _teacher_live._teacher if _teacher_live is not None else None
                )
                if _teacher_model is not None and not run_ctx.has("teacher"):
                    run_ctx.set("teacher", _teacher_model)
                if (
                    _merge_repair_active
                    and run_ctx.has("merge_repair_teacher_capture")
                ):
                    _teacher_layer_outputs = dict(
                        run_ctx.get("merge_repair_teacher_capture").outputs
                    )

            # Student: full forward pass with gradients (routers are trainable).
            _student_capture = (
                run_ctx.get("merge_repair_student_capture")
                if run_ctx.has("merge_repair_student_capture")
                else None
            )
            if _student_capture is not None:
                _student_capture.clear()
            # Tier-1 (Lever A): suppress the KV cache allocation on the student
            # forward. Single full-sequence pass, no incremental decode, the
            # cache is never read -> bit-identical to the HF default.
            student_out = student(input_ids=batch, use_cache=False)
            student_vocab_logits = student_out.logits.to(torch.float32)  # [B, L, |V|]
            del student_out  # free model output object; student_vocab_logits retains grad_fn

            # KL(teacher || student) over vocabulary, per-token, scaled by T^2.
            # Shift logits: predict token t+1 from position t (standard causal LM).
            t_logits_shift = teacher_vocab_logits[:, :-1, :]   # [B, L-1, |V|]
            s_logits_shift = student_vocab_logits[:, :-1, :]   # [B, L-1, |V|]

            # Publish the already-shifted teacher/student logits + the
            # snapshotted teacher MoE outputs for the compute phase. These are
            # set every iteration -> overwrite=True. teacher_layer_outputs is
            # set-once-per-microbatch by the orchestrator; drop-then-set keeps
            # the merge-repair hook's plain ctx.get working.
            run_ctx.set("teacher_logits", t_logits_shift, overwrite=True)
            run_ctx.set("student_logits", s_logits_shift, overwrite=True)
            if run_ctx.has("teacher_layer_outputs"):
                run_ctx.drop("teacher_layer_outputs")
            run_ctx.set("teacher_layer_outputs", _teacher_layer_outputs)

            # Set-once collisions: compute_merge_repair_mse + compute_kd_loss
            # do plain ctx.set on these slots each microbatch - drop them so
            # the 2nd microbatch does not raise KeyError. merge_repair_mse_weight
            # is ALSO set by setup_merge_repair, so it collides on the FIRST
            # microbatch too - drop it whenever present.
            for _slot in (
                "merge_repair_mse_term", "merge_repair_mse_weight",
                "kd_loss", "vocab_kl",
            ):
                if run_ctx.has(_slot):
                    run_ctx.drop(_slot)

            # compute_merge_repair_mse BEFORE compute_kd_loss - the KD-loss
            # combiner reads merge_repair_mse_term / merge_repair_mse_weight.
            walk_phases(
                ("compute_merge_repair_mse", "compute_kd_loss"), plugins, run_ctx
            )
            loss = run_ctx.get("kd_loss")
            kl_loss = run_ctx.get("vocab_kl")

            # --- First-batch sanity probe (added 2026-05-13) ---
            # On the FIRST non-skipped iteration of the run (cold start OR
            # resume), dump teacher/student/loss stats so we can verify the
            # forward path BEFORE the optimizer touches anything. Raises if
            # anything is NaN/Inf - much faster signal than waiting until
            # step 50. The flag-based guard fires correctly on resumed runs
            # (where `step` would be > 0 and a `step == 0` test would miss).
            if not _first_batch_probed:
                _log_first_batch_sanity(t_logits_shift, s_logits_shift, loss)
                _first_batch_probed = True

            # --- NaN tripwire (added 2026-05-13) ---
            # If loss went non-finite, dump diagnostics and abort. Earlier
            # crashes trained through 250+ NaN batches; this stops at batch 1.
            if not torch.isfinite(loss):
                _dump_nan_diagnostics(
                    loss=loss,
                    teacher_logits=t_logits_shift,
                    student_logits=s_logits_shift,
                    student=student,
                    epoch=epoch, step=step, batch_i=i,
                )
                raise RuntimeError(
                    f"Stage 5 KD loss is non-finite at epoch={epoch} step={step} "
                    f"batch={i}: loss={float(loss):.6e}. See ERROR-level dump above "
                    "for teacher/student/param state. Aborting before backward()."
                )

            # --- Per-step debug log (env-gated, added 2026-05-13) ---
            # Fine-grained instantaneous loss for the first N steps when
            # STAGE5_DEBUG_PER_STEP=1. Falls back to the 50-step window log
            # after the burn-in window.
            if os.environ.get("STAGE5_DEBUG_PER_STEP", "0") == "1" and step <= int(
                os.environ.get("STAGE5_DEBUG_PER_STEP_LIMIT", "20")
            ):
                log.info(
                    "  DEBUG epoch=%d step=%d i=%d loss=%.6e t_max=%.3e s_max=%.3e",
                    epoch, step, i, float(loss),
                    float(t_logits_shift.detach().abs().max()),
                    float(s_logits_shift.detach().abs().max()),
                )

            window_loss_acc.append(loss.detach())
            # T is a Python float; max() on scalars guards against T==0.
            # raw_kl tracks the *pure* vocab-KL (kl_loss), NOT the combined
            # loss - so the save-best tracker and cross-run comparisons stay
            # invariant to the Direction-E MSE term. When merge_repair is off
            # `kl_loss is loss`, so this is byte-identical to pre-E `main`.
            window_raw_kl_acc.append(kl_loss.detach() / max(T * T, 1e-12))
            (loss / grad_accum).backward()

            if (i + 1) % grad_accum == 0:
                # Pre-step: compute gradient norm over trainable params, but
                # ONLY on log-window steps (Tier-1 Lever B). grad_norm is read
                # solely inside `if step % log_every_n_steps == 0:` below, so on
                # every other optimizer step the device->host sync + the full
                # parameters() comprehension are wasted. The gate uses the
                # PREDICTED post-increment step value `(step + 1)` (the modulo
                # test below runs after `step += 1`), and the call stays HERE —
                # before `optim.step()`/`optim.zero_grad()` — so it reads the
                # SAME populated grads that produced this window's step. The
                # reported value is therefore bit-identical to the unconditional
                # version. `clip_grad_norm_(..., inf)` is a no-op on the grads
                # (clip-coef inf >= 1, so PyTorch skips the in-place scale).
                grad_norm = float("nan")
                _log_every = config["logging"]["log_every_n_steps"]
                if (step + 1) % _log_every == 0:
                    # Unwrap compiled wrapper so parameters() reflects the original module's leaf params.
                    _params_for_norm = getattr(student, "_orig_mod", student)
                    grad_norm = float(
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in _params_for_norm.parameters() if p.requires_grad and p.grad is not None],
                            float('inf'),
                        )
                    )
                optim.step()
                optim.zero_grad()
                scheduler.step()
                step += 1
                _rt_update(stage="stage5", epoch=int(epoch), step=int(step), batch=int(i),
                           phase="kd_train")
                if step % config["logging"]["log_every_n_steps"] == 0:
                    # Single device->host sync per log boundary (vs per-microbatch).
                    # The window covers the period since the previous log line -
                    # not a single optimizer step - so the reported loss is the
                    # window mean, not an instantaneous step loss. The label below
                    # uses "window_loss" to make this explicit.
                    if window_loss_acc:
                        loss_val = sum(t.item() for t in window_loss_acc) / len(window_loss_acc)
                    else:
                        loss_val = 0.0
                    if window_raw_kl_acc:
                        raw_kl_val = sum(t.item() for t in window_raw_kl_acc) / len(window_raw_kl_acc)
                    else:
                        raw_kl_val = 0.0
                    window_loss_acc.clear()
                    window_raw_kl_acc.clear()

                    # Publish the per-window training-loop signals the
                    # best-tracker / early-stop hooks read. step / epoch /
                    # raw_kl_val are re-published every log window, so
                    # overwrite=True (a no-op-safe unconditional write).
                    run_ctx.set("step", step, overwrite=True)
                    run_ctx.set("epoch", epoch, overwrite=True)
                    run_ctx.set("raw_kl_val", raw_kl_val, overwrite=True)

                    # update_best_tracker: EMA update + best.pt save + patience
                    # counter. check_early_stop: the early-stop DECISION (sets
                    # early_stop_should_stop). Both rebind state on run_ctx
                    # with overwrite=True - dispatched against the ROOT ctx so
                    # the EMA carry survives across windows.
                    walk_phases(
                        ("update_best_tracker", "check_early_stop"),
                        plugins, run_ctx,
                    )
                    ema = float(run_ctx.get("raw_kl_ema"))
                    best_raw_kl_ema = float(run_ctx.get("best_raw_kl_ema"))
                    best_step = int(run_ctx.get("best_step"))
                    no_improve_windows = int(run_ctx.get("no_improve_windows"))
                    _early_stopped = bool(run_ctx.get("early_stop_should_stop"))

                    current_lr = scheduler.get_last_lr()[0]

                    log.info(
                        "  epoch=%d step=%d window_loss=%.6f raw_kl=%.6f "
                        "ema=%.6f best_ema=%.6f@%d lr=%.3e T=%.3f grad_norm=%.4f | %s",
                        epoch, step, loss_val, raw_kl_val, ema, best_raw_kl_ema,
                        best_step, current_lr, T, grad_norm, _rt_snap(),
                    )
                    payload = {
                        "stage5/epoch": epoch,
                        "stage5/step": step,
                        "stage5/loss": loss_val,
                        "stage5/raw_kl": raw_kl_val,
                        "stage5/raw_kl_ema": ema,
                        "stage5/best_raw_kl_ema": best_raw_kl_ema,
                        "stage5/best_step": best_step,
                        "stage5/lr": current_lr,
                        "stage5/temperature": T,
                        "stage5/grad_norm": grad_norm,
                        "stage5/no_improve_windows": no_improve_windows,
                    }
                    _trackio_log(payload)
                else:
                    # Outside a log window the early-stop flag is not refreshed
                    # - keep the prior decision (False on the first windows).
                    _early_stopped = (
                        bool(run_ctx.get("early_stop_should_stop"))
                        if run_ctx.has("early_stop_should_stop")
                        else False
                    )

                # --- Periodic param sanity (env-gated, added 2026-05-13) ---
                # When STAGE5_PARAM_CHECK_EVERY=K, run an O(params) NaN/Inf scan
                # every K optimizer steps. Catches silent NaN drift in router
                # weights that wouldn't show up in the next-batch loss (e.g., a
                # param goes NaN but the forward zeros it before loss computes).
                _param_check_every = int(os.environ.get("STAGE5_PARAM_CHECK_EVERY", "0"))
                if _param_check_every > 0 and step % _param_check_every == 0:
                    _bad_params = _check_param_sanity(student, step)
                    if _bad_params:
                        raise RuntimeError(
                            f"Stage 5 param sanity FAILED at step={step}: "
                            f"non-finite trainable params: {_bad_params}. "
                            "Some router weight went NaN/Inf without surfacing in loss - "
                            "halting to preserve diagnostics."
                        )

                # Periodic checkpoint for crash-resume. The payload is built
                # synchronously (incl. a deep CPU copy of the optimizer state)
                # inside _save_stage5_checkpoint; only the torch.save / fsync /
                # os.replace / prune run on the background writer thread (Tier-1
                # Lever C). The prune (keep newest two step_*.pt) was relocated
                # into the writer, AFTER os.replace, so it never races a
                # not-yet-written checkpoint. When _ckpt_writer is None
                # (kill-switch) the save is fully synchronous, prune included.
                if partial_dir is not None and ckpt_every > 0 and step % ckpt_every == 0:
                    _save_stage5_checkpoint(
                        partial_dir, step, epoch, i, student, optim,
                        grad_accum=grad_accum,
                        scheduler=scheduler,
                        best_raw_kl_ema=best_raw_kl_ema,
                        best_step=best_step,
                        prev_ema=float(run_ctx.get("prev_ema")),
                        no_improve_windows=int(run_ctx.get("no_improve_windows")),
                        es_ref_ema=float(run_ctx.get("es_ref_ema")),
                        writer=_ckpt_writer,
                    )

                # --- Early-stopping break (2026-05-17 overfit fix) ---
                # Triggered inside the optimizer-step block (where the counter
                # is updated) so the break lands on an optimizer-step boundary,
                # consistent with the checkpoint cadence. A final checkpoint is
                # written unconditionally so a subsequent resume sees the exact
                # stopping point rather than re-running to the schedule end.
                # No-op when _early_stop_patience == 0 (_early_stopped stays
                # False) - byte-identical to pre-2026-05-17 `main`.
                if _early_stopped:
                    if partial_dir is not None:
                        # Drain any in-flight async step_*.pt write first so the
                        # final early-stop checkpoint is written AFTER it (no
                        # overlap / prune race), then write the early-stop
                        # checkpoint SYNCHRONOUSLY (writer=None) so it is durable
                        # on disk before we break to finalize.
                        if _ckpt_writer is not None:
                            _ckpt_writer.join()
                        _save_stage5_checkpoint(
                            partial_dir, step, epoch, i, student, optim,
                            grad_accum=grad_accum,
                            scheduler=scheduler,
                            best_raw_kl_ema=best_raw_kl_ema,
                            best_step=best_step,
                            prev_ema=float(run_ctx.get("prev_ema")),
                            no_improve_windows=int(run_ctx.get("no_improve_windows")),
                            es_ref_ema=float(run_ctx.get("es_ref_ema")),
                            writer=None,
                        )
                    break
        # Trailing-batch accounting is computed once before the epoch loop
        # (see the run-start log.warning above); no per-epoch repeat here.
        optim.zero_grad()
        # Early-stop also breaks the outer epoch loop. No-op when
        # _early_stop_patience == 0 (early_stop_should_stop stays False).
        if (
            run_ctx.has("early_stop_should_stop")
            and bool(run_ctx.get("early_stop_should_stop"))
        ):
            break

    # ---- async checkpoint writer drain (Tier-1 Lever C) ------------------
    # Drain + stop the background step_*.pt writer BEFORE the final export so
    # (a) every periodic checkpoint is durably on disk if the process exits,
    # (b) no pending write races save_compressed_checkpoint, and (c) any
    # writer-thread error is re-raised on this (training) thread, halting the
    # run rather than silently losing crash-resume state.
    if _ckpt_writer is not None:
        _ckpt_writer.close()

    # ---- teardown_merge_repair -------------------------------------------
    # MergeRepairPlugin.teardown_merge_repair removes the gradient-mask hooks
    # and the forward-capture hooks before the final save so no hook handles
    # leak into the exported checkpoint's module tree. No-op when merge-repair
    # is off (the containers are empty / absent).
    walk_phases(("teardown_merge_repair",), plugins, run_ctx)

    # ---- reload_best_checkpoint ------------------------------------------
    # EarlyStopPlugin.reload_best_checkpoint swaps the trainable (router)
    # params for the best.pt snapshot before export (when save_best was active
    # and a best.pt exists).
    walk_phases(("reload_best_checkpoint",), plugins, run_ctx)

    out_dir = artifacts_dir / f"{stage_key}_final"
    save_compressed_checkpoint(
        # Unwrap torch.compile wrapper before save so iter_moe_layers inside
        # save_compressed_checkpoint can find the text tower via attribute lookup.
        getattr(student, "_orig_mod", student), tokenizer, out_dir,
        pipeline_stage=f"{stage_key}_final",
    )
    log.info("Stage %s complete -> %s", stage_key, out_dir)
    return out_dir


def _async_ckpt_enabled() -> bool:
    """Async-checkpoint kill-switch (Tier-1 Lever C).

    ``STAGE5_ASYNC_CKPT=0`` forces the fully-synchronous path (payload build +
    torch.save + fsync + os.replace + prune all on the training thread), i.e.
    the pre-Lever-C behaviour. Any other value (or unset) enables the async
    writer. This lets a production run disable async without a code revert.
    """
    return os.environ.get("STAGE5_ASYNC_CKPT", "1") != "0"


def _deep_cpu_copy(obj):
    """Recursively detach+CPU-clone every tensor inside a (possibly nested)
    container so the result shares NO storage with live GPU tensors.

    ``optim.state_dict()`` returns references to the LIVE optimizer moment
    tensors (on the training device). Backgrounding ``torch.save`` on those
    references would let the next ``optim.step()`` mutate them mid-serialize,
    producing a torn checkpoint. Snapshotting them on the training thread BEFORE
    handing the payload to the writer makes the background write independent of
    any subsequent ``optim.step()``.
    """
    if torch.is_tensor(obj):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {k: _deep_cpu_copy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        typ = type(obj)
        return typ(_deep_cpu_copy(v) for v in obj)
    return obj


def _write_checkpoint_payload(
    payload: dict, partial_dir: Path, step: int, epoch: int, batch_idx: int
) -> None:
    """Durably write a fully-host-resident ``payload`` to ``step_{step}.pt``.

    Atomic: write to ``*.pt.tmp`` -> fsync(file) -> os.replace -> fsync(parent).
    Then prune (keep newest two ``step_*.pt``). This is the part that may run on
    the background writer thread; ``payload`` MUST already be CPU/host-resident.
    """
    tmp = partial_dir / f"step_{step}.pt.tmp"
    final = partial_dir / f"step_{step}.pt"
    torch.save(payload, tmp)
    fd = os.open(str(tmp), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, final)
    parent_fd = os.open(str(final.parent), os.O_RDONLY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    # Prune AFTER os.replace so the just-written step_N.pt is visible to the
    # glob (relocated from the orchestrator run() body so it never races a
    # not-yet-written checkpoint). Keep only the two most recent step_*.pt.
    all_ckpts = sorted(
        partial_dir.glob("step_*.pt"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    for old_ckpt in all_ckpts[:-2]:
        old_ckpt.unlink(missing_ok=True)
    log.info("Stage 5: checkpoint saved at step %d (epoch %d, batch %d)", step, epoch, batch_idx)


class _Stage5CheckpointWriter:
    """Single background writer for the periodic ``step_*.pt`` checkpoints.

    One persistent worker thread drains a ``Queue(maxsize=1)``; ``put`` blocks
    until the previous job is taken, so two ``step_*.pt`` writes never overlap
    (the single-writer invariant). Scoped to ``step_*.pt`` ONLY — ``best.pt`` is
    written synchronously by ``early_stop.py`` and is left untouched (distinct
    filename, no collision; the prune globs ``step_*.pt`` only).

    A worker-thread exception is captured and RE-RAISED on the training thread
    at the next ``submit`` and at ``join`` so a failed checkpoint halts the run
    rather than silently losing crash-resume state.
    """

    def __init__(self) -> None:
        self._queue: "queue.Queue" = queue.Queue(maxsize=1)
        self._error: BaseException | None = None
        self._error_lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._drain, name="stage5-ckpt-writer", daemon=True
        )
        self._thread.start()

    def _drain(self) -> None:
        while True:
            job = self._queue.get()
            try:
                if job is None:  # shutdown sentinel
                    return
                payload, partial_dir, step, epoch, batch_idx = job
                _write_checkpoint_payload(payload, partial_dir, step, epoch, batch_idx)
            except BaseException as exc:  # noqa: BLE001 - re-raised on training thread
                with self._error_lock:
                    if self._error is None:
                        self._error = exc
                log.error("Stage 5: async checkpoint writer failed: %r", exc)
            finally:
                self._queue.task_done()

    def _raise_if_error(self) -> None:
        with self._error_lock:
            err = self._error
            self._error = None
        if err is not None:
            raise err

    def submit(
        self, payload: dict, partial_dir: Path, step: int, epoch: int, batch_idx: int
    ) -> None:
        # Re-raise a prior worker failure before enqueuing the next job.
        self._raise_if_error()
        # maxsize=1 put() blocks until the previous job is drained -> the
        # previous step_*.pt write is fully done before this one starts.
        self._queue.put((payload, partial_dir, step, epoch, batch_idx))

    def join(self) -> None:
        """Block until all queued writes are durable; re-raise worker errors."""
        self._queue.join()
        self._raise_if_error()

    def close(self) -> None:
        """Drain, stop the worker, and re-raise any pending worker error."""
        self._queue.put(None)
        self._thread.join()
        self._raise_if_error()


def _save_stage5_checkpoint(
    partial_dir: Path,
    step: int,
    epoch: int,
    batch_idx: int,
    student: nn.Module,
    optim: torch.optim.Optimizer,
    grad_accum: int = 1,
    scheduler: "torch.optim.lr_scheduler._LRScheduler | None" = None,
    best_raw_kl_ema: float | None = None,
    best_step: int | None = None,
    prev_ema: float | None = None,
    no_improve_windows: int | None = None,
    es_ref_ema: float | None = None,
    writer: "_Stage5CheckpointWriter | None" = None,
) -> None:
    # F3 fix: when torch.compile is active, `student` is a compiled wrapper
    # whose named_parameters() may enumerate names that differ from the
    # underlying module's attribute tree (e.g. "_orig_mod.*" prefixes are
    # stripped or mangled). Use the unwrapped module so that the names saved
    # here match the attribute path walked during restore.
    unwrapped = getattr(student, "_orig_mod", student)
    router_state = {
        name: p.data.cpu().clone()
        for name, p in unwrapped.named_parameters()
        if p.requires_grad
    }
    # SYNCHRONOUS snapshot (Tier-1 Lever C): build the ENTIRE payload from
    # host-resident values on the training thread BEFORE any async write.
    #   - router_state: already CPU-cloned above.
    #   - optim.state_dict(): returns LIVE GPU moment-tensor references, so it
    #     MUST be deep-CPU-copied; otherwise a backgrounded torch.save would
    #     race the next optim.step() -> torn checkpoint.
    #   - scheduler_state: plain Python scalars, deep-copied for safety.
    payload = {
        "format_version": 2,
        "step": step,
        "epoch": epoch,
        "batch_idx": batch_idx,
        "router_state": router_state,
        "optim_state": _deep_cpu_copy(optim.state_dict()),
        "gradient_accumulation": grad_accum,
        # Trainable parameter name set; resume validates this matches the
        # current trainable scope so a config change to trainable_name_patterns
        # cannot pair stale moments with the wrong parameters.
        "trainable_param_names": sorted(router_state.keys()),
        # v2 additions (Move A): LR scheduler + best-tracker state. None for
        # legacy code paths that don't pass them; resume tolerates None.
        "scheduler_state": (
            _deep_cpu_copy(scheduler.state_dict()) if scheduler is not None else None
        ),
        "best_raw_kl_ema": best_raw_kl_ema,
        "best_step": best_step,
        "prev_ema": prev_ema,
        # 2026-05-17 early-stop additions. None for callers that don't pass
        # them; resume tolerates None (patience restarts fresh).
        "no_improve_windows": no_improve_windows,
        "es_ref_ema": es_ref_ema,
    }
    # The payload is now fully host-resident and independent of subsequent GPU
    # mutation. Either hand it to the background writer (async) or write it
    # synchronously (no writer / kill-switch). Bytes written are identical.
    if writer is not None:
        writer.submit(payload, partial_dir, step, epoch, batch_idx)
    else:
        _write_checkpoint_payload(payload, partial_dir, step, epoch, batch_idx)


__all__ = ["run"]
