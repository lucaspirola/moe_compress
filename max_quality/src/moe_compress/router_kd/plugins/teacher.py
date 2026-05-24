"""Teacher-logits slot plugins (RK-5 of the Router-KD plugin-architecture refactor).

Paper
-----
Hyeon & Do, "Is Retraining-Free Enough? The Necessity of Router
Calibration for Efficient MoE Compression" — arXiv:2603.02217 (§F.3,
Eq. 3, Table 1). audit/spec_compliance/01_papers/2603.02217/source.md.

Two plugins:

- ``TeacherCachePlugin`` — loads per-batch teacher vocabulary logits
  from a precomputed SHA-256-keyed sidecar cache file. Registered
  FIRST in the ``provide_teacher_logits`` ``dispatch_first`` slot so
  it wins if a cache is available.
- ``TeacherLivePlugin`` — loads the teacher model (optionally in 4-bit
  via bitsandbytes per §F.3 fallback), runs the per-batch forward,
  and returns vocabulary logits. Wins ``dispatch_first`` when no
  cache plugin returned a result.

Vocab guard: both plugins assert the teacher and student share the
same vocab dimension (4-bit teacher cannot be smaller than the
student per §F.3 4-bit cache compatibility).

Official code
-------------
**None published.** See :mod:`router_kd.plugins.trainable_scope` for
the negative finding.

Calibration deviation D11 (SHARED) — canonical owner is
:mod:`stage2.plugins.reap_scoring`.

Home of the Router-KD teacher-logits concern: where the per-batch teacher
vocabulary logits come from. RK-5 ships TWO plugins — ``TeacherCachePlugin``
and ``TeacherLivePlugin`` — both implementing the SAME slot hook
``provide_teacher_logits``.

RK-5 is PURE Pattern B — the ``stage5_router_kd.py`` monolith is NOT modified.
Unlike the Pattern-A relocations (RK-2/RK-3/RK-4), the teacher logic this task
owns has nothing standalone to relocate: the teacher-CACHE load + multi-epoch
validation is INLINE ``run()`` code, and ``_get_teacher`` is a closure defined
*inside* ``run()``. There is no module-level function to move. So RK-5
REPRODUCES that logic faithfully in the plugins' hooks and leaves the monolith
byte-for-byte unchanged — byte-identity is trivially preserved (the RK-0 golden
snapshot stays green for free).

Slot contract — ``provide_teacher_logits``
-----------------------------------------
``provide_teacher_logits(ctx, *, input_ids, epoch, batch_index, num_batches)
-> torch.Tensor | None`` returns the teacher vocabulary logits ``[B, L, |V|]``
for one training batch, or ``None`` to DEFER to the next plugin in the
:meth:`PluginRegistry.dispatch_first` chain. ``TeacherCachePlugin`` returns
``None`` on a cache miss (no cache configured / file absent / payload ``None``);
``TeacherLivePlugin`` always returns a tensor — it is the universal fallback.
The RK-8 orchestrator registers ``TeacherCachePlugin`` BEFORE
``TeacherLivePlugin``, so on a cache HIT the cache wins and the live teacher is
never touched; on a MISS the cache defers and the live teacher answers.

Circular-import note (mirror of ``vocab_kd.py`` / ``kd_optimizer.py`` /
``trainable_scope.py``): this module imports only from ``...pipeline.*`` /
``..context`` / ``...utils.*`` / stdlib / torch — NEVER from
``stage5_router_kd`` or ``router_kd.orchestrator`` at any scope (module-top OR
function-local). The monolith re-imports the plugin package at load time, so a
``from ..stage5_router_kd import ...`` here would deadlock the import; nothing
in this module does that. ``_set_experts_implementation`` below is a deliberate
Pattern-B REPRODUCTION of the small monolith helper of the same name — it is
NOT a monolith import, precisely to keep this contract.

Both plugins are registered-but-INERT at RK-5 — no orchestrator walk or test
invokes their ``provide_teacher_logits`` / ``load_teacher_cache`` hooks in the
live pipeline. RK-8 plugs the slot into the Router-KD plugin sequencer and
deletes the monolith ``run()``.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from ..context import PipelineContext
from ...utils.model_io import iter_moe_layers, load_model

log = logging.getLogger(__name__)


def _set_experts_implementation(model: "torch.nn.Module", impl: str) -> None:
    """Override the MoE experts forward dispatch on `model`.

    Pattern-B REPRODUCTION of the ``stage5_router_kd._set_experts_implementation``
    helper (verbatim) — see this module's docstring on why it is reproduced,
    not imported.

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


class TeacherCachePlugin:
    """Router-KD precomputed-teacher-logits plugin (RK-5 — registered-but-INERT).

    Owns "Path B" of the teacher-logits concern: a precomputed teacher-logits
    sidecar produced by ``hf_jobs/precompute_teacher_logits.py``, which lets
    Stage 5 skip the live teacher entirely (~70 GB BF16 VRAM saved). The
    :meth:`load_teacher_cache` hook is the one-time setup that resolves the
    config path, memory-maps the sidecar and runs every schema/topology/
    multi-epoch validation; the :meth:`provide_teacher_logits` slot hook does
    the per-batch slice arithmetic.

    RK-5 is PURE Pattern B: the monolith's inline cache-load block is
    REPRODUCED here (it has nothing standalone to relocate). Both hooks are
    INERT at RK-5 — no orchestrator walk or test invokes them in the live
    pipeline. RK-8 plugs the slot into the Router-KD sequencer.
    """

    name = "teacher_cache"
    paper = (
        "Router KD Eq. 3 — arXiv:2603.02217 (Hyeon & Do); no official code. "
        "Concern: precomputed-teacher-logits cache slot (SHA-256-keyed). "
        "Registered FIRST in dispatch_first(provide_teacher_logits) so it "
        "wins on cache-hit. See module docstring."
    )
    config_key = "stage5_router_kd.teacher_logits_cache"
    # ``config`` / ``student`` / ``artifacts_dir`` drive the one-time cache
    # load + validation; ``teacher_logits_cache`` is the loaded payload the
    # slot hook then slices per batch.
    reads: tuple[str, ...] = (
        "config", "student", "artifacts_dir", "teacher_logits_cache",
    )
    # ``load_teacher_cache`` publishes the validated payload (or None on miss).
    writes: tuple[str, ...] = ("teacher_logits_cache",)
    # Empty: serving cached logits needs no calibration pass.
    provides: tuple[str, ...] = ()

    def is_enabled(self, config: dict) -> bool:
        """True IFF ``stage5_router_kd.teacher_logits_cache`` is configured.

        The cache plugin is dropped from the slot chain when no cache path is
        set — there is nothing for it to serve, so the live teacher answers
        every batch unopposed.
        """
        return bool(config.get("stage5_router_kd", {}).get("teacher_logits_cache"))

    def contribute_artifact(self, ctx: Any) -> dict:
        return {}

    def load_teacher_cache(self, ctx: PipelineContext) -> None:
        """One-time setup hook — load + validate the teacher-logits sidecar.

        INERT at RK-5: no orchestrator walk or test invokes this hook in the
        live pipeline. RK-8 calls it once before the epoch loop. The body
        reproduces the monolith ``run()`` inline cache-load block faithfully —
        dead code at RK-5 but RK-8 + unit tests rely on it.

        Resolves ``stage5_router_kd.teacher_logits_cache`` (relative paths
        resolve against ``artifacts_dir``), and if the file exists,
        memory-maps it (``mmap=True`` — the ~30 GB sidecar is never fully
        materialized in CPU RAM) and runs every schema / num_samples /
        vocab-size / token-count / multi-epoch validation. The validated
        payload is published to ``teacher_logits_cache``; on a missing file
        the slot is set to ``None`` (a clean cache miss → the live teacher
        takes over).
        """
        config = ctx.get("config")
        s5 = config["stage5_router_kd"]
        student = ctx.get("student")
        artifacts_dir = ctx.get("artifacts_dir")

        teacher_logits_cache = None
        cache_path_cfg = s5.get("teacher_logits_cache")
        if cache_path_cfg:
            cache_path = Path(cache_path_cfg)
            if not cache_path.is_absolute():
                cache_path = artifacts_dir / cache_path
            if cache_path.exists():
                # Spec §8 mutual-exclusion rule: if both teacher_logits_cache
                # and teacher_load_in_4bit are configured, cache wins. Surface
                # the override so an operator who set 4-bit isn't surprised
                # when the cache path supersedes it.
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

        ctx.set("teacher_logits_cache", teacher_logits_cache, overwrite=True)

    def provide_teacher_logits(
        self,
        ctx: PipelineContext,
        *,
        input_ids: torch.Tensor,
        epoch: int,
        batch_index: int,
        num_batches: int,
    ) -> "torch.Tensor | None":
        """Slot hook — return the cached teacher logits for one batch, or defer.

        INERT at RK-5: no orchestrator walk or test invokes this hook in the
        live pipeline. RK-8 dispatches it via :meth:`PluginRegistry.dispatch_first`
        ahead of :meth:`TeacherLivePlugin.provide_teacher_logits`. The body
        reproduces the monolith ``run()`` per-batch cache branch faithfully.

        Returns ``None`` (DEFER to the next plugin in the dispatch_first chain)
        on a cache miss — no ``teacher_logits_cache`` slot, or it was published
        as ``None`` (config absent / sidecar file missing). On a HIT it
        reproduces the monolith's per-batch slice arithmetic — the divisibility
        guard, the ``token_start`` epoch-offset index, the slice + dtype/device
        cast + ``[B, L, |V|]`` reshape — and returns the tensor.
        """
        # has()-guarded: an unwritten slot OR a slot holding None is a miss.
        if not ctx.has("teacher_logits_cache"):
            return None
        teacher_logits_cache = ctx.get("teacher_logits_cache")
        if teacher_logits_cache is None:
            return None

        config = ctx.get("config")
        s5 = config["stage5_router_kd"]
        # Path B: precomputed teacher vocab logits.
        cache_seq_len = int(s5["max_sequence_length"])
        cache_batch_size = int(s5["batch_size"])
        cache_tokens_per_batch = cache_batch_size * cache_seq_len
        # Cache slicing assumes uniform batch shape across the run — any
        # trailing partial batch would misalign subsequent epochs'
        # token_start. Enforce divisibility upfront so the failure mode is a
        # clean error, not silent KD corruption.
        if int(s5["max_calibration_samples"]) % cache_batch_size != 0:
            raise RuntimeError(
                f"Stage 5 teacher-logits cache requires "
                f"max_calibration_samples ({s5['max_calibration_samples']}) "
                f"divisible by batch_size ({cache_batch_size}); otherwise "
                "the trailing partial batch misaligns the cache slice "
                "across subsequent batches/epochs."
            )
        # F1 fix: incorporate the epoch offset so that epoch N reads the
        # correct slice of the cache instead of wrapping back to position 0
        # (which would replay epoch-0 teacher logits against epoch-N student
        # batches — wrong KD signal).
        token_start = (epoch * num_batches + batch_index) * cache_tokens_per_batch
        token_end = token_start + (input_ids.shape[0] * input_ids.shape[1])
        teacher_vocab_logits = teacher_logits_cache["logits"][token_start:token_end]
        teacher_vocab_logits = teacher_vocab_logits.to(
            device=input_ids.device, dtype=torch.float32
        )
        teacher_vocab_logits = teacher_vocab_logits.view(
            input_ids.shape[0], input_ids.shape[1], -1
        )
        return teacher_vocab_logits


class TeacherLivePlugin:
    """Router-KD live-teacher-logits plugin (RK-5 — registered-but-INERT).

    Owns "Path A" of the teacher-logits concern: the live teacher model.
    Lazily loads the teacher on the first call (deferred load — on resume the
    fast-forward never touches the teacher, saving ~60 s + ~70 GB VRAM), then
    runs a no-grad forward per batch to produce the vocabulary logits.

    RK-5 is PURE Pattern B: the monolith's ``_get_teacher`` ``run()`` closure
    is REPRODUCED here as :meth:`_load_teacher` (closures cannot be relocated).
    The slot hook is INERT at RK-5 — no orchestrator walk or test invokes it in
    the live pipeline. RK-8 plugs it into the Router-KD sequencer.

    The merge-repair ``_LayerOutputCapture`` branch around the monolith teacher
    forward is OUT of scope for RK-5 — this plugin provides only the
    vocab-logits slot.
    """

    name = "teacher_live"
    paper = (
        "Router KD Eq. 3 — arXiv:2603.02217 (Hyeon & Do); no official code. "
        "Concern: live teacher forward (optionally 4-bit bnb per §F.3 "
        "fallback) + vocab-dim guard. Wins dispatch_first when no cache. "
        "See module docstring."
    )
    config_key = "stage5_router_kd.teacher_model_repo"
    # The live teacher load reads model/config knobs + the student (for
    # device-map co-location and the vocab/topology guards).
    reads: tuple[str, ...] = ("config", "student", "device")
    # The slot hook returns its tensor directly; it publishes no ctx slot.
    writes: tuple[str, ...] = ()
    # Empty: a forward pass of the teacher needs no calibration pass.
    provides: tuple[str, ...] = ()

    def __init__(self) -> None:
        # Lazy-load state: the teacher model, materialized on first use.
        self._teacher = None

    def is_enabled(self, config: dict) -> bool:
        """Always True — the live teacher is the UNIVERSAL fallback.

        Every Router-KD run must be able to produce teacher logits; when no
        cache is configured (or the cache misses) the live teacher answers
        every batch. It is therefore unconditionally enabled and registered
        AFTER ``TeacherCachePlugin`` so the cache wins on a hit.
        """
        return True

    def contribute_artifact(self, ctx: Any) -> dict:
        return {}

    def _load_teacher(self, ctx: PipelineContext):
        """Lazily materialize + validate the live teacher model (cached).

        Pattern-B REPRODUCTION of the monolith ``run()`` ``_get_teacher``
        closure: resolves 4-bit / ``teacher_model_repo`` override, derives the
        device map, calls ``load_model``, applies ``_set_experts_implementation``,
        switches to inference mode, runs the vocab-size + MoE-topology
        ``RuntimeError`` guards, and optionally ``torch.compile``s the teacher.
        Caches the result into ``self._teacher`` — the second call returns it
        directly.
        """
        if self._teacher is not None:
            return self._teacher

        import os

        config = ctx.get("config")
        s5 = config["stage5_router_kd"]
        student = ctx.get("student")
        device = ctx.get("device") if ctx.has("device") else None
        # torch.compile acceleration (spec §8) — config-gated, default off.
        use_compile = bool(s5.get("torch_compile", False))
        # Student MoE-layer count — the topology guard's reference.
        student_refs_count = sum(
            1 for _ in iter_moe_layers(getattr(student, "_orig_mod", student))
        )

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
        # Env var `EXPERTS_IMPLEMENTATION` overrides the YAML default — mirror
        # the monolith run() entry resolution so the teacher's experts impl
        # matches the student's.
        _experts_impl = os.environ.get(
            "EXPERTS_IMPLEMENTATION", s5.get("experts_implementation", "batched_mm")
        )
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
        # Put the teacher in inference mode (disable dropout).
        _t.eval()
        # Vocab-size guard for the live-teacher path. Mirrors the
        # cache-path check so a `teacher_model_repo` pointed at a model
        # with a different tokenizer fails fast instead of silently
        # producing a wrong KD signal. Passes by definition on the default
        # path. Unwrap a possible torch.compile wrapper to read .config
        # reliably.
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
        # Faithful monolith order (`_get_teacher` closure): cache the model
        # FIRST, THEN run the MoE-topology guard. The monolith assigns
        # `_teacher_state["model"] = _t` *before* the layer-count RuntimeError,
        # so on a topology mismatch `_teacher_state["model"]` is already set —
        # this plugin matches that exactly (`self._teacher` set before the
        # guard fires). Do NOT reorder to "guard first": RK-5 is a faithful
        # byte-identical reproduction of the monolith, not an improvement.
        self._teacher = _t
        _teacher_refs_count = sum(1 for _ in iter_moe_layers(getattr(_t, "_orig_mod", _t)))
        if _teacher_refs_count != student_refs_count:
            raise RuntimeError(
                f"Teacher/student MoE layer count mismatch: "
                f"{_teacher_refs_count} (teacher) vs {student_refs_count} "
                f"(student). Vocabulary-level KD requires identical MoE "
                "topology between teacher and student."
            )
        return self._teacher

    def provide_teacher_logits(
        self,
        ctx: PipelineContext,
        *,
        input_ids: torch.Tensor,
        epoch: int,
        batch_index: int,
        num_batches: int,
    ) -> torch.Tensor:
        """Slot hook — run a no-grad teacher forward, return the vocab logits.

        INERT at RK-5: no orchestrator walk or test invokes this hook in the
        live pipeline. RK-8 dispatches it via :meth:`PluginRegistry.dispatch_first`
        AFTER :meth:`TeacherCachePlugin.provide_teacher_logits`, so it only
        runs on a cache miss. The body reproduces the monolith ``run()``
        live-teacher branch (minus the OUT-of-scope merge-repair capture).

        Lazily loads the teacher (:meth:`_load_teacher`), runs a no-grad
        forward, and returns ``out.logits.detach().to(torch.float32)`` — the
        ``[B, L, |V|]`` teacher vocabulary logits. ALWAYS returns a tensor
        (never ``None``): the live teacher is the universal fallback.

        ``epoch`` / ``batch_index`` / ``num_batches`` are accepted for
        slot-signature uniformity with :meth:`TeacherCachePlugin.provide_teacher_logits`
        even though the live path — a stateless forward — ignores them.
        """
        with torch.no_grad():
            teacher = self._load_teacher(ctx)
            out = teacher(input_ids=input_ids)
            return out.logits.detach().to(torch.float32)


__all__ = ["TeacherCachePlugin", "TeacherLivePlugin"]
