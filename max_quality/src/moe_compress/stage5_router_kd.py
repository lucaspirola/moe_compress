"""Stage 5 — Router KD via vocabulary-level output distillation.

Reference: 2603.02217 (Router Knowledge Distillation), Eq. 3.

The paper distills at the **vocabulary output logit** level, NOT at the
intermediate router gate level. From §4:

  "By distilling output logits rather than matching router gate values
   explicitly, Router KD avoids requiring the teacher and student to share
   identical expert sets or gate dimensionalities."

This means the loss is:

  L_RKD = (τ²/N_x) Σ_t  KL(softmax(z_T^t / τ) ‖ softmax(z_S^t / τ))

where z_T, z_S ∈ ℝ^|V| are the teacher/student vocabulary logits for
next-token prediction, and the sum is over unmasked token positions.

Only router weights are trainable; all expert weights are frozen. The
vocabulary-level signal propagates gradients through the full forward pass
including the routing decisions, which naturally adapts the router to the
compressed expert set.
"""
from __future__ import annotations

import logging
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils.calibration import build_calibration_tensor, iter_batches, spec_from_config
from .utils.model_io import (
    iter_moe_layers,
    load_model,
    save_compressed_checkpoint,
)
from .utils.runtime_monitor import snapshot_telemetry as _rt_snap, update as _rt_update
from .utils.trackio_log import trackio_log as _trackio_log

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Debug instrumentation (added 2026-05-13 after the Stage 2.5 NaN crash on
# vast.ai B200 contract 36639423). Stage 2.5 had been running for the first
# time and the previous bare-bones per-50-step logger gave no signal about
# WHERE NaN entered (teacher? student? KL? params?). These helpers provide:
#   1. A first-batch sanity probe (always on) — log teacher/student logit
#      stats and the initial loss BEFORE the optimizer touches anything.
#   2. A NaN tripwire (always on) — abort on the FIRST non-finite loss with
#      a structured dump of teacher/student/param state.
#   3. Per-step debug log (env STAGE5_DEBUG_PER_STEP=1) — fine-grained loss
#      trajectory for the first N steps instead of the 50-step window mean.
#   4. Periodic param sanity (env STAGE5_PARAM_CHECK_EVERY=K) — scan trainable
#      params for NaN/Inf every K steps, halt if any go non-finite silently.
# ---------------------------------------------------------------------------


def _log_first_batch_sanity(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    loss: torch.Tensor,
) -> None:
    """First-batch sanity probe — log forward-pass stats and abort if any
    NaN/Inf is present BEFORE the optimizer runs."""
    try:
        t_finite = bool(torch.isfinite(teacher_logits).all())
        s_finite = bool(torch.isfinite(student_logits).all())
        loss_finite = bool(torch.isfinite(loss))
        log.info(
            "Stage 5 first-batch sanity: "
            "teacher shape=%s dtype=%s finite=%s abs_max=%.3e mean=%.3e std=%.3e ; "
            "student shape=%s dtype=%s finite=%s abs_max=%.3e mean=%.3e std=%.3e ; "
            "initial_loss=%.6e finite=%s",
            tuple(teacher_logits.shape), teacher_logits.dtype, t_finite,
            float(teacher_logits.detach().abs().max()),
            float(teacher_logits.detach().mean()),
            float(teacher_logits.detach().std()),
            tuple(student_logits.shape), student_logits.dtype, s_finite,
            float(student_logits.detach().abs().max()),
            float(student_logits.detach().mean()),
            float(student_logits.detach().std()),
            float(loss.detach()), loss_finite,
        )
        if not (t_finite and s_finite and loss_finite):
            raise RuntimeError(
                "Stage 5 first-batch sanity FAILED: "
                f"teacher_finite={t_finite} student_finite={s_finite} loss_finite={loss_finite}. "
                "Halting before any optimizer step to surface the actual failure mode "
                "(teacher vs student vs KL) instead of training 50 batches of NaN."
            )
    except RuntimeError:
        raise
    except Exception as exc:
        log.warning("Stage 5 first-batch sanity probe raised %s (non-fatal — continuing)", exc)


def _dump_nan_diagnostics(
    *,
    loss: torch.Tensor,
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    student: nn.Module,
    epoch: int,
    step: int,
    batch_i: int,
) -> None:
    """Structured dump on non-finite loss — teacher/student stats + first 5
    non-finite trainable params (routers only)."""
    try:
        def _stats(t: torch.Tensor) -> str:
            t_det = t.detach()
            n_total = max(1, t_det.numel())
            n_nan = int(torch.isnan(t_det).sum())
            n_inf = int(torch.isinf(t_det).sum())
            return (
                f"shape={tuple(t.shape)} dtype={t.dtype} "
                f"abs_max={float(t_det.abs().max()):.3e} mean={float(t_det.mean()):.3e} "
                f"pct_nan={100.0 * n_nan / n_total:.2f} pct_inf={100.0 * n_inf / n_total:.2f}"
            )
        log.error("Stage 5 NaN-tripwire at epoch=%d step=%d batch=%d: loss=%s",
                  epoch, step, batch_i, float(loss.detach()))
        log.error("  teacher logits: %s", _stats(teacher_logits))
        log.error("  student logits: %s", _stats(student_logits))
        bad_params = _check_param_sanity(student, step)
        if bad_params:
            log.error("  non-finite trainable params (first %d): %s", len(bad_params), bad_params)
        else:
            log.error("  all trainable params still finite — NaN originates in forward, not weights.")
    except Exception as exc:
        log.error("Stage 5 NaN diagnostics raised: %s", exc)


def _check_param_sanity(student: nn.Module, step: int) -> list[str]:
    """Cheap O(params) scan: names of trainable params containing NaN/Inf,
    capped at 5 for log brevity."""
    bad: list[str] = []
    base = getattr(student, "_orig_mod", student)
    for name, p in base.named_parameters():
        if not p.requires_grad:
            continue
        if not torch.isfinite(p.data).all():
            bad.append(name)
            if len(bad) >= 5:
                break
    return bad


def _set_experts_implementation(model: nn.Module, impl: str) -> None:
    """Override the MoE experts forward dispatch on `model`.

    The `transformers.integrations.moe` decorator dispatches each MoE forward
    by reading `self.config._experts_implementation` at every call (see
    `ExpertsInterface.get_interface`), so this assignment takes effect for
    all subsequent forwards without rebuilding the model. The valid values
    registered in transformers v4.x's `ALL_EXPERTS_FUNCTIONS`:

      * `"grouped_mm"`  — default; uses `torch.nn.functional.grouped_mm`.
                          DEADLOCKS on Blackwell sm_100 (see project memory
                          `project_grouped_mm_blackwell.md`). Do NOT use on
                          B200 / GB200 / B300.
      * `"batched_mm"`  — uses `torch.bmm` per expert group with padding to
                          max active count. ~70-90% of grouped_mm's speed,
                          but bmm is universally supported (Hopper +
                          Blackwell). Recommended default on B200.
      * `"sonicmoe"`    — custom kernel registered by the sonicmoe package.
                          Performance unknown; Blackwell-compatibility
                          unknown. Try as fallback if `batched_mm` is too
                          slow or hits an issue.
      * `"eager"`       — Python loop over active experts, one
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
    """Run Router KD.

    ``stage_key`` controls the partial-dir and output-dir names, allowing the
    same code to serve both Stage 2.5 (``stage_key="stage2p5"``) and Stage 5
    (``stage_key="stage5"``).  The config section read is always
    ``stage5_router_kd`` regardless of ``stage_key``.
    """
    s5 = config["stage5_router_kd"]
    cal = config["calibration"]

    # Set MoE forward dispatch (default 'batched_mm' to work around the
    # grouped_mm Blackwell deadlock — see _set_experts_implementation
    # docstring). Env var `EXPERTS_IMPLEMENTATION` overrides the YAML default
    # for quick A/B without redeploying config.
    _experts_impl = os.environ.get(
        "EXPERTS_IMPLEMENTATION", s5.get("experts_implementation", "batched_mm")
    )
    _set_experts_implementation(student, _experts_impl)

    # Vocabulary-level KD does not use merge_map; see ALGORITHM_REFERENCE.md §8.

    # Stage 5 holds teacher (~70 GB BF16) AND student (~50 GB BF16) on cuda
    # at once. On H200 (141 GB) both fit with ~15 GB headroom (per §8 spec).
    # Mitigations are still available for tighter-VRAM hosts:
    #   (A) teacher_load_in_4bit: true  — bitsandbytes NF4, ~17 GB live.
    #   (B) teacher_logits_cache: <path> — sidecar produced by
    #       hf_jobs/precompute_teacher_logits.py; skip live teacher entirely.
    # If both are set, (B) wins.
    teacher = None
    # Teacher is lazily loaded inside `_teacher_state["model"]`.
    teacher_logits_cache = None
    cache_path_cfg = s5.get("teacher_logits_cache")
    if cache_path_cfg:
        cache_path = Path(cache_path_cfg)
        if not cache_path.is_absolute():
            cache_path = artifacts_dir / cache_path
        if cache_path.exists():
            # Spec §8 mutual-exclusion rule: if both teacher_logits_cache and
            # teacher_load_in_4bit are configured, cache wins. Surface the
            # override so an operator who set 4-bit isn't surprised when the
            # cache path supersedes it.
            if s5.get("teacher_load_in_4bit"):
                log.warning(
                    "Stage 5: teacher_load_in_4bit=true is configured but "
                    "teacher_logits_cache=%s exists; per spec §8 'cache wins on "
                    "conflict' the cache supersedes the 4-bit load — 4-bit will not run.",
                    cache_path,
                )
            log.info("Stage 5: loading precomputed teacher logits from %s", cache_path)
            # mmap=True keeps the ~30 GB sidecar memory-mapped instead of
            # materializing the whole thing in CPU RAM. Each per-batch
            # slice pages in only what the loop touches.
            cache_payload = torch.load(cache_path, map_location="cpu", mmap=True)
            teacher_logits_cache = cache_payload
            # Schema check: any future format change must bump this.
            fmt = int(cache_payload.get("format_version", 0))
            if fmt != 1:
                raise RuntimeError(
                    f"Teacher-logits cache format_version={fmt} unsupported "
                    "(this Stage 5 only knows version 1). Regenerate the cache "
                    "or upgrade Stage 5."
                )
            cached_bs = int(cache_payload.get("batch_size", -1))
            if cached_bs != int(s5["batch_size"]):
                log.warning(
                    "Stage 5: teacher_logits_cache batch_size=%d disagrees with "
                    "stage5_router_kd.batch_size=%d. The cache is logically valid "
                    "as long as token order matches; batch grouping is irrelevant "
                    "to KL correctness — proceeding.",
                    cached_bs, int(s5["batch_size"]),
                )
            if int(cache_payload.get("sequence_length", -1)) != int(s5["max_sequence_length"]):
                raise RuntimeError(
                    "Teacher-logits cache sequence_length disagrees with config — "
                    "re-run precompute or align configs."
                )
            # F3 fix: also verify num_samples matches. A cache built with
            # fewer samples than Stage 5 expects would silently return
            # zero-length slices for late batches → degenerate KD signal.
            cache_n = int(cache_payload.get("num_samples", -1))
            cfg_n = int(s5["max_calibration_samples"])
            epochs_cfg = int(s5.get("epochs", 1))
            # Accept caches sized for either single-epoch (cfg_n) or
            # multi-epoch (epochs_cfg * cfg_n) coverage. The training loop
            # indexes into the cache via (epoch * len(batches) + i) *
            # cache_tokens_per_batch, so a cache sized at epochs_cfg * cfg_n
            # is the canonical multi-epoch layout.
            # The multi-epoch + cache combination is hard-rejected later (the
            # student input replays identically across epochs while cache
            # advances — silent KD corruption). So at this point only
            # epochs_cfg=1 with cache_n=cfg_n is valid; reject anything else
            # with a clear message that points at the right config knob.
            if cache_n != cfg_n:
                raise RuntimeError(
                    f"Teacher-logits cache num_samples={cache_n} disagrees with "
                    f"stage5_router_kd.max_calibration_samples={cfg_n}. "
                    "Stage 5 would read past the end of the cache — regenerate or align."
                )
            # Topology check: the cache must be keyed against this student's
            # vocabulary and calibration shape. A mismatch in the trailing
            # logits dim or the (num_samples × sequence_length) token count
            # means the cache was generated for a different student/tokenizer
            # combination and would silently produce a wrong KD signal.
            student_vocab_size = int(getattr(student.config, "vocab_size", -1))
            cache_logits = cache_payload.get("logits")
            if cache_logits is None:
                raise RuntimeError(
                    "Teacher-logits cache missing 'logits' tensor — wrong cache for this student."
                )
            cache_vocab_size = int(cache_logits.shape[-1])
            if cache_vocab_size != student_vocab_size:
                raise RuntimeError(
                    f"Teacher-logits cache vocab_size={cache_vocab_size} does not match "
                    f"student.config.vocab_size={student_vocab_size} — wrong cache for this student."
                )
            cache_seq_len_meta = int(cache_payload.get("sequence_length", -1))
            expected_tokens = cache_n * cache_seq_len_meta
            actual_tokens = int(cache_logits.shape[0]) if cache_logits.dim() >= 1 else -1
            if actual_tokens != expected_tokens:
                raise RuntimeError(
                    f"Teacher-logits cache token count ({actual_tokens}) disagrees with "
                    f"num_samples × sequence_length ({cache_n} × {cache_seq_len_meta} = "
                    f"{expected_tokens}) — wrong cache for this student."
                )
            # F1 fix: verify the cache covers all epochs, not just one pass.
            # With multi-epoch training the token index advances as
            # (epoch * num_batches + i) * cache_tokens_per_batch; a cache that
            # only covers one epoch would be silently re-read from position 0
            # for epochs 2..N, replaying epoch-1 teacher logits against later
            # student batches — a corrupted KD signal.
            if epochs_cfg > 1 and cache_n < epochs_cfg * cfg_n:
                # Hard-fail: training-loop index `(epoch * num_batches + i) *
                # cache_tokens_per_batch` would read past the end of a
                # single-epoch cache for epochs >= 1. Reading past end yields
                # zero-length slices → degenerate (silently zero) KD signal,
                # which silently corrupts router updates. Refuse to proceed.
                raise RuntimeError(
                    f"Stage 5: teacher_logits_cache num_samples={cache_n} covers only "
                    f"{cache_n // max(cfg_n, 1)} epoch(s) of data but "
                    f"stage5_router_kd.epochs={epochs_cfg}. The training loop would "
                    "read past cache end for later epochs, silently corrupting the "
                    "KD signal. Regenerate a multi-epoch cache (num_samples="
                    f"{epochs_cfg * cfg_n}) or set epochs=1."
                )
            log.info("Stage 5: cache covers %d samples, %d sequence_length",
                     cache_payload.get("num_samples"), cache_payload.get("sequence_length"))
        else:
            log.warning("Stage 5: teacher_logits_cache=%s not found at %s — falling back to live teacher",
                        cache_path_cfg, cache_path)

    # --- torch.compile acceleration (spec §8) ---
    # Assigned BEFORE _get_teacher closure so the closure's reference resolves correctly.
    use_compile = bool(s5.get("torch_compile", False))

    if teacher_logits_cache is None:
        # Deferred teacher load: only load the teacher on the first live training
        # batch. On resume, fast-forward iterates without ever touching the teacher —
        # saves ~60s + 70 GB VRAM when resuming deep into training.
        _teacher_state: dict = {"model": None}

        def _get_teacher(student_refs_count: int):
            if _teacher_state["model"] is None:
                load_in_4bit = bool(s5.get("teacher_load_in_4bit", False))
                teacher_repo_override = s5.get("teacher_model_repo") or None
                teacher_name_or_path = (
                    teacher_repo_override
                    if teacher_repo_override
                    else config["model"]["name_or_path"]
                )
                if teacher_repo_override and load_in_4bit:
                    # An override repo is already quantized (e.g. FP8); stacking
                    # bitsandbytes 4-bit on top is incoherent. Honor the override.
                    log.warning(
                        "Stage 5: teacher_model_repo=%s in use; ignoring "
                        "teacher_load_in_4bit (the override repo is already quantized).",
                        teacher_repo_override,
                    )
                    load_in_4bit = False
                if config["model"].get("load_in_4bit", False) and not load_in_4bit and not teacher_repo_override:
                    log.warning(
                        "Stage 5: config['model']['load_in_4bit']=true but "
                        "stage5_router_kd.teacher_load_in_4bit=false. The teacher "
                        "will load in BF16 (~70 GB) and may OOM tighter-VRAM hosts. "
                        "Set teacher_load_in_4bit: true to match."
                    )
                # 4-bit (bitsandbytes) requires a single-device map; honor the
                # caller's device choice from config["model"]["device_map"] if
                # it's a single-device dict (e.g. {"": "cuda:1"}); otherwise
                # default to {"": 0}. Never pin to GPU 0 unconditionally.
                _cfg_dm = config["model"]["device_map"]
                if load_in_4bit:
                    if isinstance(_cfg_dm, dict) and len(_cfg_dm) == 1:
                        _device_map = _cfg_dm
                    else:
                        # Co-locate 4-bit teacher with the student rather than
                        # blindly pinning to GPU 0 — `device` (or the student's
                        # actual placement) is the source of truth so KL forward
                        # doesn't perform a cross-device round-trip per microbatch.
                        if device is not None:
                            _device_map = {"": str(device)}
                        else:
                            try:
                                _student_device = next(student.parameters()).device
                                _device_map = {"": str(_student_device)}
                            except (StopIteration, AttributeError):
                                _device_map = {"": 0}
                else:
                    _device_map = _cfg_dm
                log.info("Loading teacher for KD (first live batch): %s "
                         "(teacher_model_repo=%s, teacher_load_in_4bit=%s, device_map=%s)",
                         teacher_name_or_path, teacher_repo_override, load_in_4bit, _device_map)
                _t, _ = load_model(
                    teacher_name_or_path,
                    revision=config["model"]["revision"],
                    torch_dtype=config["model"]["torch_dtype"],
                    device_map=_device_map,
                    attn_implementation=config["model"]["attn_implementation"],
                    load_in_4bit=load_in_4bit,
                    trust_remote_code=config["model"].get("trust_remote_code", False),
                )
                # Set the MoE experts implementation on the teacher too. The
                # teacher's forward path goes through the same
                # `transformers.integrations.moe._grouped_mm` integration that
                # deadlocks on Blackwell (see project memory
                # `project_grouped_mm_blackwell.md`). Mirror what we applied
                # to the student at run() entry.
                _set_experts_implementation(_t, _experts_impl)
                _t.eval()
                # Vocab-size guard for the live-teacher path. Mirrors the
                # cache-path check at lines 155-159 so a `teacher_model_repo`
                # pointed at a model with a different tokenizer fails fast
                # instead of silently producing a wrong KD signal. Passes by
                # definition on the default path. Unwrap a possible
                # torch.compile wrapper to read .config reliably.
                _student_unwrapped = getattr(student, "_orig_mod", student)
                _teacher_vocab = int(getattr(_t.config, "vocab_size", -1))
                _student_vocab = int(getattr(_student_unwrapped.config, "vocab_size", -1))
                if _teacher_vocab != _student_vocab:
                    raise RuntimeError(
                        f"Teacher (repo={teacher_name_or_path}) vocab_size="
                        f"{_teacher_vocab} does not match student vocab_size="
                        f"{_student_vocab}. Vocabulary-level KD is impossible "
                        "with a tokenizer mismatch."
                    )
                # torch.compile(teacher) is deterministically skipped when an
                # override repo is in use. FP8 weights are not yet fully
                # supported by reduce-overhead; the existing eager-fallback
                # try/except would be a silent slowdown, which the
                # no-speed-compromises rule disallows. Student compile is
                # untouched and still carries the speedup.
                if use_compile and not teacher_repo_override:
                    try:
                        log.info("Stage 5: torch.compile(teacher, mode='default')")
                        _t = torch.compile(_t, mode="default")
                    except Exception as exc:
                        log.warning("Stage 5: torch.compile(teacher) failed (%s) — eager", exc)
                _teacher_state["model"] = _t
                _teacher_refs_count = sum(1 for _ in iter_moe_layers(getattr(_t, "_orig_mod", _t)))
                if _teacher_refs_count != student_refs_count:
                    raise RuntimeError(
                        f"Teacher/student MoE layer count mismatch: "
                        f"{_teacher_refs_count} (teacher) vs {student_refs_count} "
                        f"(student). Vocabulary-level KD requires identical MoE "
                        "topology between teacher and student."
                    )
            return _teacher_state["model"]
    # Freeze non-router parameters BEFORE compiling the student so that the
    # compiled graph is traced with the final requires_grad flags. Compiling
    # before freeze risks the compiler baking in the wrong gradient-enabled
    # state for parameters that are about to be frozen.
    # Sanity check: warn if any parameter name matches BOTH trainable and
    # frozen patterns (frozen_name_patterns is informational only — it is NOT
    # consulted by _freeze_non_routers; freeze is driven entirely by
    # `requires_grad_(any(p in name for p in trainable_name_patterns))`.
    # Names that match only frozen_name_patterns are still correctly frozen
    # because they fail the trainable-pattern check. The patterns list exists
    # solely for the conflict-overlap sanity check below; trainable wins,
    # but a name in both is almost certainly a config bug).
    _frozen_patterns = s5.get("frozen_name_patterns", []) or []
    _trainable_patterns = s5["trainable_name_patterns"]
    if _frozen_patterns:
        _base_for_check = getattr(student, "_orig_mod", student)
        _conflicts = [
            name for name, _ in _base_for_check.named_parameters()
            if any(pat in name for pat in _trainable_patterns)
            and any(pat in name for pat in _frozen_patterns)
        ]
        if _conflicts:
            raise RuntimeError(
                f"Stage 5 config error: {len(_conflicts)} parameter(s) match BOTH "
                f"trainable_name_patterns and frozen_name_patterns (e.g. {_conflicts[:3]}). "
                "Resolve the overlap in stage5_router_kd config."
            )
    _freeze_non_routers(student, _trainable_patterns)

    # Optimizer constructed AFTER freezing so it only receives parameters that
    # have requires_grad=True at construction time.
    # weight_decay is config-driven (default 0.0 to match the pre-2026-05-13
    # baseline). On the 2026-05-13 A0 run (epochs=2, samples=6000) the loss
    # bottomed at step 950 then RISE back to step-1400 levels — clear
    # memorization. weight_decay=0.01 (AdamW default) regularizes router
    # weights to counter that. The paper doesn't specify, but empirics on this
    # model say 0.0 over-fits.
    optim = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=s5["learning_rate"],
        weight_decay=float(s5.get("weight_decay", 0.0)),
    )

    # torch.compile applied AFTER freeze+optimizer construction so the compiled
    # graph reflects the final frozen parameter layout. named_parameters() on
    # the compiled wrapper delegates to the underlying module — the optimizer
    # already holds the correct parameter references before compilation.
    # mode='default' (not 'reduce-overhead'): the latter captures CUDA graphs
    # and replays them. On Blackwell B200 with Qwen3.6-A3B's MoE grouped_mm,
    # CUDA graph replay deadlocked after ~1400 steps of sustained training on
    # the 2026-05-13 A0 run — main thread stuck inside
    # `torch.nn.functional.grouped_mm` via `transformers.integrations.moe._grouped_linear`,
    # faulthandler thread dump confirmed kernel-level hang (not a Python exception).
    # The previous 375-step run completed fine — same code path — so the failure
    # mode is sustained-training + CUDA-graph state accumulation specific to
    # `reduce-overhead`. `mode='default'` keeps TorchDynamo + TorchInductor
    # fusion (≈ 50-70% of reduce-overhead's speedup) but launches kernels
    # individually, avoiding graph replay entirely.
    if use_compile:
        try:
            log.info("Stage 5: torch.compile(student, mode='default')")
            student = torch.compile(student, mode="default")
        except Exception as exc:
            log.warning("Stage 5: torch.compile failed (%s) — falling back to eager mode", exc)
            use_compile = False

    # seed_offset distinguishes Stage 2.5 (post-merge router KD) from
    # Stage 5 (final router KD) so each pass sees a different calibration
    # draw and the routers are not retrained on the identical 3000 sequences.
    if stage_key == "stage2p5":
        _seed_offset = 25
    elif stage_key == "stage5":
        _seed_offset = 5
    else:
        raise ValueError(
            f"Stage 5: unsupported stage_key={stage_key!r}; expected "
            "'stage5' or 'stage2p5'."
        )
    spec = spec_from_config(
        cal,
        num_sequences_override=s5["max_calibration_samples"],
        sequence_length_override=s5["max_sequence_length"],
        seed_offset=_seed_offset,
    )
    calib = build_calibration_tensor(
        tokenizer, spec, cache_dir=artifacts_dir / "_calibration_cache"
    )
    batches = iter_batches(calib, batch_size=s5["batch_size"])
    grad_accum = s5["gradient_accumulation"]
    # Temperature ramp (Move B): linear from start → end over the full
    # optimizer-step count. If start == end, behaves as a constant temperature
    # (rollback / legacy). Falls back to the legacy scalar `kd_temperature` if
    # the start/end keys are absent.
    T_start = float(s5.get("kd_temperature_start", s5.get("kd_temperature", 1.0)))
    T_end = float(s5.get("kd_temperature_end", s5.get("kd_temperature", 1.0)))
    ckpt_every = int(s5.get("checkpoint_every_n_steps", 100))
    if ckpt_every <= 0:
        raise ValueError(
            f"Stage 5: checkpoint_every_n_steps={ckpt_every} disables "
            "checkpointing; spec §8 Resume mandates step-boundary "
            "checkpointing every N optimizer steps. Set a positive integer "
            "(default 100) — disabling resume would silently lose progress "
            "on a long-running stage."
        )

    # Teacher/student MoE layer count sanity check (router structure must match
    # even though we're distilling at vocab level — the student's routers are
    # what we're training).
    # We only need the count for teacher↔student topology checks (line 507);
    # avoid retaining a list of MoE-layer references on a 35B-class model.
    student_refs_count = sum(1 for _ in iter_moe_layers(getattr(student, "_orig_mod", student)))

    # -----------------------------------------------------------------------
    # Crash-resume: find latest checkpoint and restore router + optim state.
    # -----------------------------------------------------------------------
    resume_step = 0
    resume_epoch = 0
    resume_batch_i = -1
    # Captured from a v2 (or later) resume checkpoint and consumed after the
    # LR scheduler / best-tracker locals are constructed below. v1 checkpoints
    # leave these as None — the scheduler is fast-forwarded by replaying
    # scheduler.step() resume_step times; best-tracker re-initializes from +inf.
    _resume_scheduler_state = None
    _resume_best_raw_kl_ema = None
    _resume_best_step = None
    _resume_prev_ema = None

    # `no_resume=True` means "don't pick up existing checkpoints"; it does
    # NOT mean "don't write new ones". Spec §8 mandates step-boundary
    # checkpointing every N optimizer steps — disabling future writes would
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
                    f"(expected 1 or 2) — delete _{stage_key}_partial/ and re-run"
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
                    f"checkpoint — added={added[:5]}{'...' if len(added) > 5 else ''}, "
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
                        "and `device` is None — optimizer state left on its loaded "
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
                        f"Stage 5 resume: gradient_accumulation mismatch — "
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
            log.info("Stage 5: resumed from step %d (epoch %d, batch %d)",
                     resume_step, resume_epoch, resume_batch_i)

    # `train()` enables dropout / batchnorm-train semantics on every submodule.
    # For Qwen3-30B-A3B and the production target (no dropout, RMSNorm only),
    # this is a no-op. For architectures with dropout in attention or MLP, the
    # frozen submodules would still emit dropped activations — a silent
    # train-vs-eval mismatch that could distort the KD signal. Spec §8 requires
    # only `mlp.gate.weight` to be trainable; the rest of the module tree is
    # frozen via _freeze_non_routers but stays in train mode here. If a future
    # architecture variant introduces dropout, set frozen submodules to eval().
    student.train()
    total_steps = (len(batches) // grad_accum) * s5["epochs"]
    total_optim_steps = total_steps  # alias used by scheduler + T-ramp
    remaining_steps = max(0, total_steps - resume_step)

    # --- LR scheduler (Move A) ---
    # Constructed AFTER total_optim_steps is known so the warmup horizon and
    # cosine endpoint align with the real step count (= len(batches)//grad_accum
    # × epochs, matching the value emitted to trackio as total_steps_planned).
    _lr_schedule = str(s5.get("lr_schedule", "none"))
    _warmup_ratio = float(s5.get("warmup_ratio", 0.05))
    _lr_min_ratio = float(s5.get("lr_min_ratio", 0.10))
    warmup_steps = max(1, int(total_optim_steps * _warmup_ratio))

    def _lr_lambda(current_step: int) -> float:
        if _lr_schedule == "none":
            return 1.0
        # Off-by-one: LambdaLR with last_epoch=-1 advances to current_step=0
        # on the first .step() call. Use (current_step + 1) in the warmup
        # branch so step 0 fires at LR = 1/warmup_steps, not 0.
        if current_step < warmup_steps:
            return (current_step + 1) / warmup_steps
        progress = (current_step - warmup_steps) / max(1, total_optim_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return _lr_min_ratio + (1.0 - _lr_min_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, _lr_lambda)

    # Temperature ramp closure (Move B). Uses total_optim_steps so the ramp
    # endpoint coincides with the cosine endpoint.
    def _current_T(global_step: int) -> float:
        if T_start == T_end:
            return T_start
        p = min(max(global_step / max(1, total_optim_steps), 0.0), 1.0)
        return T_start + (T_end - T_start) * p

    # Best-tracker state (Move A). Initialized here so resume-restore below can
    # overwrite when restarting from a v2 checkpoint.
    _best_ema_alpha = float(s5.get("best_metric_ema_alpha", 0.2))
    _save_best = bool(s5.get("save_best", True))
    best_raw_kl_ema = float("inf")
    best_step = -1
    prev_ema = float("inf")

    # --- Resume restore for scheduler + best-tracker ---
    # v2 checkpoint: load scheduler state verbatim, restore best/prev EMA.
    # v1 checkpoint (or no resume): fast-forward scheduler by replaying
    # scheduler.step() resume_step times so the LR curve picks up at the
    # right point. Best/prev EMA stay at +inf (the next log boundary
    # bootstraps cleanly).
    if _resume_scheduler_state is not None:
        scheduler.load_state_dict(_resume_scheduler_state)
    elif resume_step > 0:
        for _ in range(resume_step):
            scheduler.step()
    if _resume_best_raw_kl_ema is not None:
        best_raw_kl_ema = _resume_best_raw_kl_ema
        best_step = _resume_best_step if _resume_best_step is not None else -1
    if _resume_prev_ema is not None:
        prev_ema = _resume_prev_ema

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
        "stage5/config/kd_temperature_start": T_start,
        "stage5/config/kd_temperature_end": T_end,
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
            "grad-accum window (grad_accum=%d) — their gradients are dropped "
            "at each epoch end (×%d epochs).",
            trailing, grad_accum, int(s5["epochs"]),
        )

    # Spec D-protocol-blend / §8: when running multi-epoch KD with a
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
            "teacher_logits_cache is not supported — calibration batches are "
            "produced once and replayed identically across epochs, but the "
            "cache index advances per (epoch, batch), creating a teacher/"
            "student input mismatch in epochs ≥ 2. Either set epochs=1 or "
            "regenerate the cache against a deterministic per-epoch shuffle "
            "schedule that this code path also applies."
        )
    step = resume_step
    optim.zero_grad()
    # Tracks whether the first-batch sanity probe has run yet. A boolean is
    # used (instead of a `step == 0 and i == resume_batch_i + 1` predicate)
    # because `step` is the optimizer-step counter and stays > 0 across any
    # non-trivial resume — a step-based guard silently skips the probe on
    # the resumed path, which is when we need it most (router weights may
    # have been NaN-poisoned in the previous run).
    _first_batch_probed = False
    for epoch in range(s5["epochs"]):
        if epoch < resume_epoch:
            continue
        # Accumulate detached loss tensors across the grad-accum + log windows
        # and pay one .item() sync at log-emission time, instead of paying
        # device→host sync per microbatch (~375/epoch at full scale).
        window_loss_acc: list[torch.Tensor] = []
        # Raw KL = loss / T^2 — invariant under the temperature ramp, so it
        # remains comparable across the run and across runs with different T
        # schedules. Used by the best-checkpoint tracker.
        window_raw_kl_acc: list[torch.Tensor] = []
        for i, batch in enumerate(batches):
            # Fast-forward: skip batches already processed in the resumed run.
            # resume_batch_i is the last batch of the grad-accum window that
            # triggered the checkpoint — the optimizer step has already occurred
            # for that entire window (including batch resume_batch_i itself).
            # Skip 0..resume_batch_i (inclusive) to avoid re-running the already-
            # completed step and triggering a spurious duplicate optimizer step.
            if epoch == resume_epoch and i <= resume_batch_i:
                continue

            if device is not None:
                batch = batch.to(device)

            # --- Vocabulary-level KD (paper 2603.02217, Eq. 3) ---
            # Teacher: full forward pass, get vocabulary logits z_T ∈ ℝ^{B×L×|V|}
            if teacher_logits_cache is not None:
                # Path B: precomputed teacher vocab logits.
                cache_seq_len = int(s5["max_sequence_length"])
                cache_batch_size = int(s5["batch_size"])
                cache_tokens_per_batch = cache_batch_size * cache_seq_len
                # Cache slicing assumes uniform batch shape across the run —
                # any trailing partial batch would misalign subsequent
                # epochs' token_start. Enforce divisibility upfront so the
                # failure mode is a clean error, not silent KD corruption.
                if int(s5["max_calibration_samples"]) % cache_batch_size != 0:
                    raise RuntimeError(
                        f"Stage 5 teacher-logits cache requires "
                        f"max_calibration_samples ({s5['max_calibration_samples']}) "
                        f"divisible by batch_size ({cache_batch_size}); otherwise "
                        "the trailing partial batch misaligns the cache slice "
                        "across subsequent batches/epochs."
                    )
                # F1 fix: incorporate the epoch offset so that epoch N reads
                # the correct slice of the cache instead of wrapping back to
                # position 0 (which would replay epoch-0 teacher logits
                # against epoch-N student batches — wrong KD signal).
                token_start = (epoch * len(batches) + i) * cache_tokens_per_batch
                token_end = token_start + (batch.shape[0] * batch.shape[1])
                teacher_vocab_logits = teacher_logits_cache["logits"][token_start:token_end]
                teacher_vocab_logits = teacher_vocab_logits.to(device=batch.device, dtype=torch.float32)
                teacher_vocab_logits = teacher_vocab_logits.view(batch.shape[0], batch.shape[1], -1)
            else:
                with torch.no_grad():
                    # F2 fix: re-enforce eval mode immediately before every
                    # teacher forward. A single _t.eval() at load time is not
                    # sufficient — framework hooks or torch.compile can silently
                    # transition the model back to train mode, which activates
                    # dropout and produces stochastic KD targets.
                    _teacher = _get_teacher(student_refs_count)
                    _teacher.eval()
                    teacher_out = _teacher(input_ids=batch)
                    teacher_vocab_logits = teacher_out.logits.detach().to(torch.float32)  # [B, L, |V|]
                    del teacher_out  # free the full output object before student backward pass

            # Student: full forward pass with gradients (routers are trainable).
            student_out = student(input_ids=batch)
            student_vocab_logits = student_out.logits.to(torch.float32)  # [B, L, |V|]
            del student_out  # free model output object; student_vocab_logits retains grad_fn

            # KL(teacher ‖ student) over vocabulary, per-token, scaled by τ².
            # Paper Eq. 3: L_RKD = (τ²/N_x) Σ_t KL(p_T^t ‖ p_S^t)
            # Shift logits: predict token t+1 from position t (standard causal LM).
            t_logits_shift = teacher_vocab_logits[:, :-1, :]   # [B, L-1, |V|]
            s_logits_shift = student_vocab_logits[:, :-1, :]   # [B, L-1, |V|]

            # Chunked KL to bound peak memory: at |V|≈150K with B=4, chunk=128,
            # peak intermediate is ~300 MB vs ~1.2 GB for the full sequence.
            # Spec §8 hyperparameter table pins kd_seq_chunk_size=512 on H200
            # (full sequence length — single-shot KL, no chunk loop). Default
            # to 512 here so a missing config can't degrade to a non-spec
            # chunked loop.
            seq_chunk = int(s5.get("kd_seq_chunk_size", 512))
            # Temperature ramps with the optimizer step (piece-wise constant
            # across the grad_accum micro-batches that compose one step).
            T = _current_T(step)
            loss = _chunked_vocab_kl(s_logits_shift, t_logits_shift, T, chunk_size=seq_chunk)

            # --- First-batch sanity probe (added 2026-05-13) ---
            # On the FIRST non-skipped iteration of the run (cold start OR
            # resume), dump teacher/student/loss stats so we can verify the
            # forward path BEFORE the optimizer touches anything. Raises if
            # anything is NaN/Inf — much faster signal than waiting until
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
            window_raw_kl_acc.append(loss.detach() / max(T * T, 1e-12))
            (loss / grad_accum).backward()

            if (i + 1) % grad_accum == 0:
                # Pre-step: compute gradient norm over trainable params.
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
                    # Single device→host sync per log boundary (vs per-microbatch).
                    # The window covers the period since the previous log line —
                    # not a single optimizer step — so the reported loss is the
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

                    # EMA of raw_kl across log boundaries. prev_ema=+inf on
                    # first observation triggers a bootstrap (ema = raw_kl_val).
                    if math.isinf(prev_ema):
                        ema = raw_kl_val
                    else:
                        ema = _best_ema_alpha * raw_kl_val + (1.0 - _best_ema_alpha) * prev_ema
                    prev_ema = ema

                    # Save-best by EMA-smoothed raw KL. +inf seed of
                    # best_raw_kl_ema guarantees the first log boundary always
                    # writes a best.pt, so the run always exports SOMETHING
                    # even if it crashes before any improvement.
                    if _save_best and ema < best_raw_kl_ema:
                        best_raw_kl_ema = ema
                        best_step = step
                        _save_best_router_state(partial_dir, student, step, epoch, ema)

                    current_lr = scheduler.get_last_lr()[0]
                    current_T = _current_T(step)

                    log.info(
                        "  epoch=%d step=%d window_loss=%.6f raw_kl=%.6f "
                        "ema=%.6f best_ema=%.6f@%d lr=%.3e T=%.3f grad_norm=%.4f | %s",
                        epoch, step, loss_val, raw_kl_val, ema, best_raw_kl_ema,
                        best_step, current_lr, current_T, grad_norm, _rt_snap(),
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
                        "stage5/temperature": current_T,
                        "stage5/grad_norm": grad_norm,
                    }
                    _trackio_log(payload)

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
                            "Some router weight went NaN/Inf without surfacing in loss — "
                            "halting to preserve diagnostics."
                        )

                # Periodic checkpoint for crash-resume.
                if partial_dir is not None and ckpt_every > 0 and step % ckpt_every == 0:
                    _save_stage5_checkpoint(
                        partial_dir, step, epoch, i, student, optim,
                        grad_accum=grad_accum,
                        scheduler=scheduler,
                        best_raw_kl_ema=best_raw_kl_ema,
                        best_step=best_step,
                        prev_ema=prev_ema,
                    )
                    # Keep only the two most recent checkpoints to bound disk use.
                    # Sort by step number (ascending) and delete all but the newest two.
                    all_ckpts = sorted(
                        partial_dir.glob("step_*.pt"),
                        key=lambda p: int(p.stem.split("_")[1]),
                    )
                    for old_ckpt in all_ckpts[:-2]:
                        old_ckpt.unlink(missing_ok=True)
        # Trailing-batch accounting is computed once before the epoch loop
        # (see the run-start log.warning above); no per-epoch repeat here.
        optim.zero_grad()

    # --- Best-checkpoint reload (Move A) ---
    # If save_best was active and a best.pt was written during training, swap
    # the trainable (router) params for the best snapshot before export. The
    # bulk of the model (frozen, not in best.pt) stays at its current state —
    # that's the whole point of saving only the trainable subset.
    if _save_best:
        best_path = partial_dir / "best.pt"
        if best_path.exists():
            best_blob = torch.load(best_path, map_location="cpu")
            _base = getattr(student, "_orig_mod", student)
            missing, unexpected = _base.load_state_dict(
                best_blob["router_state"], strict=False
            )
            log.info(
                "Stage %s: reloaded best router state from step=%d "
                "(raw_kl_ema=%.6f); missing=%d (expected — non-router params "
                "not in best), unexpected=%d",
                stage_key, int(best_blob.get("step", -1)),
                float(best_blob.get("raw_kl_ema", float("nan"))),
                len(missing), len(unexpected),
            )
            if unexpected:
                raise RuntimeError(
                    f"Stage {stage_key}: best.pt contains unexpected keys "
                    f"(non-router params leaked into best snapshot): "
                    f"{unexpected[:5]}"
                )
        else:
            log.warning(
                "Stage %s: save_best=true but no best.pt found in %s — "
                "exporting last-step state (best-tracker never fired)",
                stage_key, partial_dir,
            )

    out_dir = artifacts_dir / f"{stage_key}_final"
    save_compressed_checkpoint(
        # Unwrap torch.compile wrapper before save so iter_moe_layers inside
        # save_compressed_checkpoint can find the text tower via attribute lookup.
        getattr(student, "_orig_mod", student), tokenizer, out_dir,
        pipeline_stage=f"{stage_key}_final",
    )
    log.info("Stage %s complete → %s", stage_key, out_dir)
    return out_dir


def _chunked_vocab_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
    chunk_size: int = 128,
) -> torch.Tensor:
    """Compute vocab-level KL(teacher ‖ student) in sequence chunks.

    Processes ``chunk_size`` sequence positions at a time to bound peak
    intermediate memory. At chunk_size=128 with |V|=150K and B=4:
      Peak intermediate per chunk ≈ 4 × 128 × 150K × 4 bytes ≈ 300 MB
      vs ≈1.2 GB for the full sequence at L=512.

    Returns scalar loss = (τ²/N_tokens) × Σ_t KL(teacher_t ‖ student_t).

    Note: n_tokens = B × (L−1) is the per-position-mean denominator (paper
    Eq. 3's N_x for fully-packed sequences with no padding).

    ASSUMPTION: fully-packed sequences (no padding) — see spec §8 N_x note.
    Under this invariant, paper Eq. 3's mask `m_{t+1}=1` everywhere and
    `N_x = Σ_t m_{t+1} = B × (L−1) = n_tokens`, so the `+ ε` zero-mask
    safety constant from paper Eq. 3 is unnecessary. If a future calibration
    source ever introduces padding, this normalization (and the `+ ε`) must
    be revisited.
    """
    B, L, V = student_logits.shape
    total_kl = torch.zeros((), device=student_logits.device, dtype=torch.float32)
    n_tokens = 0
    for start in range(0, L, chunk_size):
        end = min(start + chunk_size, L)
        s_chunk = student_logits[:, start:end, :]
        t_chunk = teacher_logits[:, start:end, :]
        t_p = F.softmax(t_chunk / temperature, dim=-1)
        s_lp = F.log_softmax(s_chunk / temperature, dim=-1)
        chunk_kl = F.kl_div(s_lp, t_p, reduction="none").sum(dim=-1)  # [B, chunk_len]
        total_kl = total_kl + chunk_kl.sum()
        n_tokens += chunk_kl.numel()
        del t_p, s_lp, chunk_kl  # free intermediates eagerly
    return (total_kl / max(n_tokens, 1)) * (temperature ** 2)


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
    payload = {
        "format_version": 2,
        "step": step,
        "epoch": epoch,
        "batch_idx": batch_idx,
        "router_state": router_state,
        "optim_state": optim.state_dict(),
        "gradient_accumulation": grad_accum,
        # Trainable parameter name set; resume validates this matches the
        # current trainable scope so a config change to trainable_name_patterns
        # cannot pair stale moments with the wrong parameters.
        "trainable_param_names": sorted(router_state.keys()),
        # v2 additions (Move A): LR scheduler + best-tracker state. None for
        # legacy code paths that don't pass them; resume tolerates None.
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "best_raw_kl_ema": best_raw_kl_ema,
        "best_step": best_step,
        "prev_ema": prev_ema,
    }
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
    log.info("Stage 5: checkpoint saved at step %d (epoch %d, batch %d)", step, epoch, batch_idx)


def _save_best_router_state(
    partial_dir: Path,
    student: nn.Module,
    step: int,
    epoch: int,
    raw_kl_ema: float,
) -> None:
    """Atomically rewrite best.pt with the trainable (router) params only.

    File size is ~10-50 MB (router weights only) vs ~5 GB for the full
    optim+student checkpoint, so we can afford to rewrite on every
    improvement. The slim payload also keeps the end-of-training reload
    boundaried: only trainable params land via load_state_dict(strict=False).
    """
    unwrapped = getattr(student, "_orig_mod", student)
    router_state = {
        name: p.data.cpu().clone()
        for name, p in unwrapped.named_parameters()
        if p.requires_grad
    }
    payload = {
        "format_version": 1,  # best.pt format; independent of step_*.pt versioning
        "step": int(step),
        "epoch": int(epoch),
        "raw_kl_ema": float(raw_kl_ema),
        "router_state": router_state,
    }
    tmp = partial_dir / "best.pt.tmp"
    final = partial_dir / "best.pt"
    torch.save(payload, tmp)
    fd = os.open(str(tmp), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, final)


def _move_optimizer_state_to_device(optim: torch.optim.Optimizer, device) -> None:
    """Move all optimizer state tensors to the target device.

    Required after load_state_dict() when the checkpoint was saved on CPU
    but the training params live on a CUDA device — otherwise the first
    optimizer step silently mixes CPU and CUDA tensors.
    """
    for state in optim.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)


def _freeze_non_routers(model: nn.Module, trainable_patterns: list[str]) -> None:
    for name, p in model.named_parameters():
        p.requires_grad_(any(pat in name for pat in trainable_patterns))



