"""Eval-environment setup (Stage 6 plugin — live setup_environment hook).

Paper / spec source
--------------------
This plugin owns Stage 6's per-side environment-setup concern, not a
specific paper. It enforces:

- **Dataset revision pinning** — every eval dataset is loaded at a
  canonical SHA-256-keyed revision so cached evals invalidate when a
  dataset is silently updated upstream.
- **Kernel patches** — torch._dynamo.disable on linear-attention
  sublayers + fla→torch fallback swaps for the cu130/Hopper segfault
  cluster on Qwen3.5-MoE GatedDeltaNet blocks.
- **Experts-impl shim** — swap the project's ``FactoredExperts`` for
  the upstream eager experts impl on the eval path so lm-eval and
  HumanEval generation see the same shape as the GPU pipeline.
- **Imatrix corpus build** — concatenate the WikiText-2-train corpus
  for downstream consumption by :mod:`stage6.plugins.imatrix_export`.

The setup choices follow the project Spec §9 (Stage 6 validation gate)
— there is no upstream paper for the Stage-6 gate itself; eval tasks
are paper-anchored at their individual plugins.

Home of the Stage 6 environment-setup concern, extracted from the legacy
``stage6_validate.py`` monolith. This plugin owns the five environment
concerns that the legacy monolith's ``run()`` performed before any eval
family executes:

1. **Dataset revision pinning** — resolve the canonical per-dataset revision
   mapping and, under ``strict_revision_pinning``, fail fast on a misconfigured
   production run (``_resolve_dataset_revisions`` / ``_enforce_revision_pinning``).
2. **Stage 6 kernel patches** — the cu130/Hopper segfault-fix patches applied
   in-place to a Qwen3.5-MoE model (``_apply_stage6_kernel_patches``).
3. **Experts-implementation shim** — override the MoE experts forward dispatch
   (the ``grouped_mm`` Blackwell-deadlock workaround,
   ``_set_experts_implementation_s6``).
4. **imatrix calibration-corpus build** — download the WikiText-2 *train* split
   and write ``calibration_wiki_train.txt`` atomically
   (``_build_imatrix_calibration_corpus`` + the ``_atomic_write_text`` helper).
5. **torch.compile setup** — compile ``model.forward`` for the prefill-dominant
   PPL / lm-eval paths, stashing the pre-compile bound method so the generative
   block can restore eager mode.

Live wiring (S6-8 landed in 17134b2)
------------------------------------
This module's ``EvalEnvironmentPlugin.setup_environment`` is the **live**
Stage 6 environment-setup entry point. ``stage6.orchestrator.run()`` invokes
``walk_phases(("setup_environment",), plugins, run_ctx)`` (orchestrator
L171) and the legacy ``stage6_validate.run()`` is now a thin delegator into
the orchestrator. Tests invoke this hook directly via
``EvalEnvironmentPlugin().setup_environment(ctx)``.

Downstream contract — generative plugins MUST restore on the student side
------------------------------------------------------------------------
The teacher-side restoration of pre-compile forward and the generative
experts-impl switch is implemented in ``teacher_provider.py``
(see L602-616, "Option C: keep compile for prefill-only paths; eager
for generate()"). The legacy monolith ``stage6_validate.py`` (commit
8d95f8e) ALSO restored the student-side equivalents inline. After the
plugin split, that student-side restoration is no longer here — it
belongs to the generative plugins that own ``model.generate()``.

Two ctx slots are populated by ``setup_environment`` for that purpose:

* ``pre_compile_forward`` — the pre-compile bound method, or ``None`` if
  torch.compile was off or failed. ``HumanEvalPlugin`` and
  ``Math500Plugin`` **MUST** assign ``model.forward =
  ctx.get("pre_compile_forward")`` before calling ``model.generate()``
  when the slot is non-None, to dodge the Inductor recompile storm +
  cu130 decode-shape codegen crashes.
* ``experts_implementation_generative`` — the generative-block experts
  implementation (resolved from env var ``EXPERTS_IMPLEMENTATION_GENERATIVE``,
  default ``batched_mm``). ``HumanEvalPlugin`` and ``Math500Plugin`` **MUST**
  call ``_set_experts_implementation_s6(model, ctx.get("experts_implementation_generative"))``
  before ``model.generate()`` if the current ``_experts_implementation`` on
  the model config differs — this matches the teacher-side pattern at
  ``teacher_provider.py`` L607-616 and avoids the ``torch._grouped_mm``
  B=1 decode-shape crash on cu130 when YAML pins ``experts_implementation=grouped_mm``.

Failure to honor either contract is benign while
``experts_implementation=batched_mm`` (the current default) and
``torch_compile=false``, but production qwen36_35b_a3b_30pct.yaml sets
``torch_compile=true`` and Hopper PPL/lm_eval defaults toward
``grouped_mm`` per ``teacher_provider.py`` L142-149.

Circular-import contract (mirror of ``router_kd/plugins/trainable_scope.py``):
this module imports only from ``..context`` / stdlib / torch — NEVER from
``stage6_validate`` or ``stage6.orchestrator`` at any scope (module-top OR
function-local). The legacy monolith re-imports *this* module at load time, so a
``from ..stage6_validate import ...`` here would deadlock the import; nothing
in this module does that.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import torch

from ..context import PipelineContext

log = logging.getLogger(__name__)


# F-C-C-1: Spec §9 — imatrix calibration corpus is the WikiText-2 *train* split,
# written to calibration_wiki_train.txt. The eval-text concat (eval prompts seen
# by the model during PPL/zero-shot/generative) is captured separately to
# eval_text_concat.txt as a debugging side-channel ONLY.
_IMATRIX_CALIB_FILENAME: str = "calibration_wiki_train.txt"


# ---------------------------------------------------------------------------
# Dataset revision pinning (F-C-H-3)
# ---------------------------------------------------------------------------


#: Canonical per-dataset revision keys (Spec §9 line 840).
#:
#: Deviations from earlier drafts:
#:   * ``hellaswag`` and ``arc_challenge`` are intentionally absent: lm-eval
#:     pulls dataset revisions internally and the load path cannot enforce a
#:     SHA at ``simple_evaluate`` time. The cache key invalidates on
#:     ``lm_eval_version`` + lm-eval task config hash changes
#:     (see ``_teacher_cache_key``); precise SHA control requires editing
#:     lm-eval task YAMLs out-of-band.
_CANONICAL_DATASET_REVISION_KEYS = ("wikitext_ppl", "humaneval", "math500")


def _resolve_dataset_revisions(config: dict) -> dict[str, str | None]:
    """Return the per-dataset revision mapping from stage6_validate config.

    Restricted to the canonical 3-key set per Spec §9 line 840:
    {"wikitext_ppl", "humaneval", "math500"}. Any extra keys present in the
    config are dropped with a warning so the cache key is not contaminated
    by operator-only metadata. See ``_CANONICAL_DATASET_REVISION_KEYS`` for
    the deviations record.
    """
    s6 = config.get("stage6_validate", {}) or {}
    raw = s6.get("dataset_revisions") or {}
    if not isinstance(raw, dict):
        log.warning(
            "_resolve_dataset_revisions: dataset_revisions config is not a dict (%r); ignoring",
            type(raw).__name__,
        )
        return {}
    extra = set(raw.keys()) - set(_CANONICAL_DATASET_REVISION_KEYS)
    if extra:
        log.warning(
            "_resolve_dataset_revisions: dropping non-canonical keys %s "
            "(spec restricts to %s).",
            sorted(extra), list(_CANONICAL_DATASET_REVISION_KEYS),
        )
    out: dict[str, str | None] = {}
    for k in _CANONICAL_DATASET_REVISION_KEYS:
        if k not in raw:
            continue
        v = raw[k]
        if v is None:
            out[k] = None
        elif isinstance(v, str):
            out[k] = v
        else:
            raise TypeError(
                f"_resolve_dataset_revisions: revision for {k!r} must be a "
                f"string SHA or null; got {type(v).__name__} (value={v!r}). "
                f"Fix the config under stage6_validate.dataset_revisions."
            )
    return out


def _enforce_revision_pinning(
    config: dict, required_keys: tuple[str, ...] = _CANONICAL_DATASET_REVISION_KEYS,
) -> dict[str, str | None]:
    """Validate dataset_revisions when strict_revision_pinning is on.

    Raises RuntimeError listing every missing/null required key. Returns the
    resolved revisions dict regardless of strict mode (so callers can still
    pass-through whatever revisions are pinned).
    """
    s6 = config.get("stage6_validate", {}) or {}
    revisions = _resolve_dataset_revisions(config)
    strict = bool(s6.get("strict_revision_pinning", True))
    if not strict:
        return revisions
    missing = [k for k in required_keys if not revisions.get(k)]
    if missing:
        raise RuntimeError(
            "Stage 6: strict_revision_pinning=true but dataset_revisions are "
            f"missing or null for: {missing}. Pin each dataset SHA in "
            "configs/<…>.yaml under stage6_validate.dataset_revisions, or "
            "set strict_revision_pinning=false to opt out (NOT for production)."
        )
    return revisions


# ---------------------------------------------------------------------------
# imatrix calibration corpus — WikiText-2 *train* split (F-C-C-1)
# ---------------------------------------------------------------------------


def _atomic_write_text(path: Path, text: str) -> None:
    """Write `text` to `path` atomically per Spec §11 (tmp → fsync → replace → fsync parent)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        # F-CR2-N-1: open read-only solely to fsync — no bytes are written here.
        # O_RDONLY is the most accurate intent (write+append flags were misleading).
        fd = os.open(str(tmp), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    try:
        parent_fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except Exception as exc:  # noqa: BLE001
        log.debug("_atomic_write_text: parent-dir fsync failed (%s); file already on disk", exc)


def _build_imatrix_calibration_corpus(
    artifacts_dir: Path, dataset_revisions: dict[str, str | None],
) -> Path | None:
    """Download WikiText-2 *train* split and write `calibration_wiki_train.txt` atomically.

    Returns the path on success, or None if the dataset cannot be loaded
    (operator can supply the file out-of-band; imatrix run will be skipped
    upstream when no calibration file exists).
    """
    target = artifacts_dir / _IMATRIX_CALIB_FILENAME
    if target.exists() and target.stat().st_size > 0:
        log.info("imatrix calibration: %s already exists — reusing", target)
        return target
    try:
        from datasets import load_dataset
    except Exception as exc:  # noqa: BLE001
        log.warning("imatrix calibration: `datasets` not available (%s); skipping corpus build", exc)
        return None
    revision = dataset_revisions.get("wikitext_ppl")
    log.info(
        "imatrix calibration: downloading Salesforce/wikitext (wikitext-2-raw-v1, train, revision=%s)",
        revision,
    )
    try:
        ds = load_dataset(
            "Salesforce/wikitext", "wikitext-2-raw-v1",
            split="train", revision=revision,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("imatrix calibration: load_dataset(train) failed (%s); skipping corpus build", exc)
        return None
    rows = []
    for row in ds:
        text = row.get("text", "")
        # Preserve empty rows for symmetry with _wikitext2_ppl: both inputs
        # use the canonical HF/lm-eval recipe (empty rows produce the
        # expected paragraph-spacing tokens via the "\n\n" joiner). Filtering
        # empties here would drift imatrix activation statistics from the
        # PPL gate's token distribution beyond the joiner-only alignment
        # the spec describes.
        rows.append(text)
    # Spec §9 line 783: shared "\n\n" joiner across PPL eval and imatrix
    # calibration so imatrix activation statistics see comparable tokens.
    joined = "\n\n".join(rows)
    if not joined.strip():
        log.warning("imatrix calibration: WikiText-2 train split is empty after filtering; skipping write")
        return None
    _atomic_write_text(target, joined)
    log.info("imatrix calibration: wrote %s (%d rows, %d chars)", target, len(rows), len(joined))
    return target


# ---------------------------------------------------------------------------
# MoE experts-implementation shim + Stage 6 kernel patches
# ---------------------------------------------------------------------------


def _set_experts_implementation_s6(model, impl: str) -> None:
    """Override MoE experts forward dispatch on `model` (Stage 6 variant).

    Mirror of `stage5_router_kd._set_experts_implementation`. See that
    function's docstring for the rationale (Blackwell `grouped_mm` deadlock
    workaround) and the registered impl values (`grouped_mm`, `batched_mm`,
    `sonicmoe`, `eager`).
    """
    base = getattr(model, "_orig_mod", model)
    cfg = base.config
    if hasattr(cfg, "text_config"):
        cfg.text_config._experts_implementation = impl
    cfg._experts_implementation = impl
    log.info("Stage 6: MoE experts_implementation = %r", impl)


def _apply_stage6_kernel_patches(m, *, role: str) -> None:
    """Apply the cu130/Hopper segfault-fix patches in-place on a Qwen3.5-MoE
    model. Idempotent: safe to call multiple times. Called once per model
    (student AND teacher both need it before torch.compile / generate).

    The patches are necessary on torch 2.11+cu130 + Triton 3.4 + Hopper because:
      1. Inductor codegen for `constant_pad_nd(pad=4-s87)` indexes OOB inside
         GatedDeltaNet / LinearAttention / MoeMamba submodules — dodge by
         `torch._dynamo.disable`-ing those modules' forwards.
      2. fla/tilelang's `chunk_gated_delta_rule`, `recurrent_gated_delta_rule`,
         and `causal_conv1d_update` are unstable on this stack (SIGSEGV in
         lm_eval, SIGABRT in HumanEval). Swap to torch-native fallbacks
         already shipped in `modeling_qwen3_5_moe.py`.
      3. fla's `FusedRMSNormGated` Triton kernel JIT-recompiles for the
         `(B*1, head_v_dim)` decode shape during generate() and segfaults on
         Hopper + Triton 3.4 (fla-org/flash-linear-attention#734). Swap to
         the pure-torch `Qwen3_5MoeRMSNormGated` (same numerics).

    Speed impact: ~5% of GatedDeltaNet flops run eager + a few RMSNorm calls;
    net <1% wall on the full Stage 6 run.
    """
    _bypass_names = ("GatedDeltaNet", "LinearAttention", "MoeMamba")
    _bypassed = 0
    _skipped = 0
    for _name, _mod in m.named_modules():
        _cls = type(_mod).__name__
        if any(b in _cls for b in _bypass_names):
            # M1 (iter1): sentinel-guard so a second call does not wrap an
            # already-disabled forward (nested torch._dynamo.disable wraps
            # break the descriptor and complicate later restoration).
            if getattr(_mod, "_stage6_dynamo_disabled", False):
                _skipped += 1
                continue
            _mod.forward = torch._dynamo.disable(_mod.forward)
            _mod._stage6_dynamo_disabled = True
            _bypassed += 1
    if _bypassed:
        log.info("Stage 6 [%s]: torch._dynamo.disable on %d linear-attention "
                 "sublayer(s)", role, _bypassed)
    if _skipped:
        log.debug("Stage 6 [%s]: skipped %d linear-attention sublayer(s) "
                  "already marked _stage6_dynamo_disabled", role, _skipped)

    try:
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeRMSNormGated as _TorchRMSNormGated,
            torch_chunk_gated_delta_rule,
            torch_recurrent_gated_delta_rule,
            torch_causal_conv1d_update,
        )
    except ImportError as _exc:
        log.warning("Stage 6 [%s]: torch fallback symbols not found in "
                    "transformers.models.qwen3_5_moe (%s) — fla/tilelang may "
                    "crash later; continuing", role, _exc)
        return

    _gdn_patched = 0
    _norm_patched = 0
    for _name, _mod in m.named_modules():
        if type(_mod).__name__ != "Qwen3_5MoeGatedDeltaNet":
            continue
        if hasattr(_mod, "chunk_gated_delta_rule"):
            _mod.chunk_gated_delta_rule = torch_chunk_gated_delta_rule
        if hasattr(_mod, "recurrent_gated_delta_rule"):
            _mod.recurrent_gated_delta_rule = torch_recurrent_gated_delta_rule
        # Prefill path checks `if causal_conv1d_fn is not None` — None falls
        # through to torch. Decode path calls `causal_conv1d_update` with no
        # None-check, so must set explicitly to the torch fallback.
        if hasattr(_mod, "causal_conv1d_fn"):
            _mod.causal_conv1d_fn = None
        if hasattr(_mod, "causal_conv1d_update"):
            _mod.causal_conv1d_update = torch_causal_conv1d_update
        if (
            hasattr(_mod, "norm")
            and type(_mod.norm).__name__ == "FusedRMSNormGated"
        ):
            _old_norm = _mod.norm
            _new_norm = _TorchRMSNormGated(
                _mod.head_v_dim, eps=_mod.layer_norm_epsilon
            )
            # Move new norm to model device/dtype BEFORE copying weights —
            # ordering matters: if .to() raised after .copy_(), the stranded
            # CPU weight would still be swapped in and crash on next forward.
            _new_norm.to(
                device=_old_norm.weight.device,
                dtype=_old_norm.weight.dtype,
            )
            _new_norm.weight.data.copy_(_old_norm.weight.data)
            _mod.norm = _new_norm
            _norm_patched += 1
        _gdn_patched += 1

    if _gdn_patched:
        log.info("Stage 6 [%s]: forced torch-native fallback for "
                 "chunk_gated_delta_rule + causal_conv1d on %d "
                 "Qwen3_5MoeGatedDeltaNet block(s)", role, _gdn_patched)
    if _norm_patched:
        log.info("Stage 6 [%s]: replaced fla FusedRMSNormGated with "
                 "torch-native Qwen3_5MoeRMSNormGated on %d GDN block(s)",
                 role, _norm_patched)


class EvalEnvironmentPlugin:
    """Stage 6 eval-environment plugin — live ``setup_environment`` owner.

    Owns the Stage 6 environment-setup concern: dataset revision pinning, the
    cu130/Hopper kernel patches, the MoE experts-implementation shim, the
    imatrix calibration-corpus build, and the torch.compile setup. The
    standalone helpers are relocated verbatim above and re-imported by the
    legacy ``stage6_validate.py`` shim; the ordering glue and the
    torch.compile setup live in ``setup_environment`` below.

    Wiring (live since S6-8, commit 17134b2): ``stage6.orchestrator.run``
    invokes ``walk_phases(("setup_environment",), plugins, run_ctx)`` once
    per Stage 6 run. The legacy ``stage6_validate.run()`` is a thin
    delegator into the orchestrator.
    """

    name = "eval_environment"
    paper = (
        "Stage 6 per-side eval environment setup. No upstream paper — Spec §9 "
        "(Stage 6 validation gate); see this module's docstring for the "
        "downstream contract."
    )
    config_key = "stage6_validate.experts_implementation"
    reads: tuple[str, ...] = ("model", "config", "artifacts_dir")
    writes: tuple[str, ...] = (
        "dataset_revisions", "imatrix_calib_path", "use_torch_compile",
        "pre_compile_forward", "experts_impl",
        # C2 (iter1): generative-block experts-impl resolution lives in ctx
        # so HumanEval/Math500 plugins can match teacher_provider.py's
        # L607-616 pattern before calling generate().
        "experts_implementation_generative",
    )
    provides: tuple[str, ...] = ()

    def is_enabled(self, config: dict) -> bool:
        """Always True — environment setup is UNCONDITIONAL.

        Every Stage 6 run must pin dataset revisions, apply the kernel patches,
        set the experts-implementation shim and build the imatrix corpus before
        any eval family runs; ``config_key`` only names *which*
        experts-implementation is used, it never gates the plugin as a whole.
        """
        return True

    def contribute_artifact(self, ctx: Any) -> dict:
        return {}

    def setup_environment(self, ctx: PipelineContext) -> None:
        """Phase hook — Stage 6 eval-environment setup (live).

        Dispatched by ``stage6.orchestrator.run`` via
        ``walk_phases(("setup_environment",), plugins, run_ctx)`` (S6-8,
        commit 17134b2). The legacy ``stage6_validate.run()`` is a thin
        delegator. Tests invoke this hook directly.

        This hook performs, in order:

        1. **Experts-implementation shim** — ``_set_experts_implementation_s6``
           (env var ``EXPERTS_IMPLEMENTATION`` overrides the YAML default
           ``batched_mm``). Also resolves the generative-block override
           ``EXPERTS_IMPLEMENTATION_GENERATIVE`` (default ``batched_mm``) and
           publishes it via ctx; the actual switch is the responsibility of
           the generative plugins (see contract below).
        2. **``model.eval()`` switch** — Stage 5 leaves the model in
           ``train()`` mode; the gate must run every sub-metric in inference
           mode. The legacy monolith ``run()`` flipped the model right after
           the experts-impl shim, so the hook keeps that relative position.
        3. **Strict revision pinning** — ``_enforce_revision_pinning``.
        4. **imatrix calibration-corpus build** —
           ``_build_imatrix_calibration_corpus`` (after mkdir-ing
           ``artifacts_dir``).
        5. **cu130/Hopper kernel patches** — ``_apply_stage6_kernel_patches``
           on the student.
        6. **torch.compile setup** — compile ``model.forward`` for the
           prefill-dominant PPL / lm-eval paths, stashing the pre-compile bound
           method so the generative block can restore eager mode.
        7. **``masking_utils`` patch** — register ``'linear_attention'`` in
           ``transformers.masking_utils.LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING``
           so ``generate()`` on Qwen3.5-MoE does not raise
           ``KeyError: 'linear_attention'``. The legacy monolith applied this
           right after the torch.compile block; the hook keeps that order.

        ctx writes — each result of steps 1/3/4/6 is published via
        ``ctx.set(...)`` to the corresponding slot named in ``writes``. The
        step-2 inference-mode switch, the step-5 kernel patches, and the
        step-7 ``masking_utils`` patch are in-place mutations with no ctx
        slot.

        Downstream contract (HumanEval / MATH-500 plugins MUST honor)
        ------------------------------------------------------------
        ``pre_compile_forward`` and ``experts_implementation_generative``
        are populated for use by the generative-side plugins. Those plugins
        MUST, before calling ``model.generate()``:

        * Restore eager forward when applicable::

              _pre = ctx.get("pre_compile_forward")
              if _pre is not None:
                  model.forward = _pre

          Failure to do this leaves a compiled ``model.forward`` active
          through ``generate()``, which on torch 2.11+cu130 + Triton 3.4
          triggers an Inductor recompile storm on growing
          ``cache_position`` and decode-shape codegen crashes (Hopper).

        * Switch experts dispatch::

              _gen_impl = ctx.get("experts_implementation_generative")
              _cfg = getattr(model, "_orig_mod", model).config
              if _gen_impl != getattr(_cfg, "_experts_implementation", None):
                  _set_experts_implementation_s6(model, _gen_impl)

          Failure to do this leaves whichever PPL/lm_eval experts impl is
          active during generate; ``grouped_mm`` (Hopper PPL/lm_eval
          default per ``teacher_provider.py`` L142-149) crashes on B=1
          decode-shape on cu130.

        The teacher-side equivalents are already implemented at
        ``teacher_provider.py`` L602-616; the student-side equivalents
        used to live in the legacy ``stage6_validate.py`` monolith
        (commit 8d95f8e, "Option C: keep compile for prefill-only paths;
        eager for generate()") and now belong to the generative plugins.

        Intentionally DEFERRED — present in the legacy monolith's
        environment-setup region but NOT this hook's concern:

        * the one-shot **Trackio config emit** (``stage6/config/*`` keys) —
          owned by the S6-7 report plugin;
        * the **eval batch-size parsing / validation** (``ppl_batch_size``,
          ``lm_eval_batch_size``, ``gen_batch_size``) — owned by the
          S6-3/S6-4 eval plugins.
        """
        # Required slots — direct get(): a missing one is a wiring bug and
        # SHOULD raise.
        model = ctx.get("model")
        config = ctx.get("config")
        artifacts_dir = ctx.get("artifacts_dir")
        s6 = config["stage6_validate"]

        # (1) Set MoE forward dispatch (default 'batched_mm' to work around the
        # grouped_mm Blackwell deadlock — see project memory
        # `project_grouped_mm_blackwell.md`). Same shim as stage5_router_kd;
        # env var `EXPERTS_IMPLEMENTATION` overrides YAML for quick A/B.
        experts_impl = os.environ.get(
            "EXPERTS_IMPLEMENTATION", s6.get("experts_implementation", "batched_mm")
        )
        _set_experts_implementation_s6(model, experts_impl)

        # (1b) C2 (iter1): Resolve the generative-block experts-impl override
        # here so the value lives in ctx for HumanEval/Math500 to consume.
        # We do NOT switch the model here — the switch belongs to the
        # generative plugins (see docstring "Downstream contract"), matching
        # teacher_provider.py L607-616.
        # Default is the YAML/env-var default 'batched_mm' (matches the
        # student-side initial impl). teacher_provider.py L142-149 documents
        # a Hopper preference for grouped_mm on PPL/lm_eval; that's a
        # teacher-side choice and does not constrain the generative-block
        # default.
        experts_implementation_generative = os.environ.get(
            "EXPERTS_IMPLEMENTATION_GENERATIVE", "batched_mm"
        )

        # (2) Switch the model to inference mode before any sub-metric runs —
        # Stage 5 leaves it in train() mode. Reproduces the monolith run()'s
        # `model.eval()` call, which sits right after the experts-impl shim.
        model.eval()

        # (3) F-C-H-3: enforce strict revision pinning early — fail fast on a
        # misconfigured production run rather than after expensive teacher
        # loads / evals.
        dataset_revisions = _enforce_revision_pinning(config)

        # (4) F-C-C-1: build the imatrix calibration corpus from WikiText-2
        # *train* split, written to artifacts_dir/calibration_wiki_train.txt.
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        imatrix_calib_path = _build_imatrix_calibration_corpus(
            artifacts_dir, dataset_revisions
        )

        # (5) Apply the cu130/Hopper segfault-fix patches to the student
        # UNCONDITIONALLY (not gated on use_torch_compile). The fla kernel
        # crashes happen during eager generate() regardless of compile state,
        # and the helper is a no-op on models that don't have GatedDeltaNet
        # modules. Mirrors the unconditional teacher-side patch call.
        _apply_stage6_kernel_patches(model, role="student")

        # (6) Optimization #5: torch.compile for prefill-dominant paths.
        # Compile model.forward before evaluations begin; model.generate also
        # benefits since it calls model.forward internally for each prefill
        # step. dynamic=True handles variable-length padded batches from
        # lm-eval. mode='default' (not 'reduce-overhead') avoids the
        # CUDA-graph deadlock in lm-eval's loglikelihood loop.
        use_torch_compile = s6.get("torch_compile", False)
        # If we compile model.forward below, stash the pre-compile bound method
        # here so the generative block (HumanEval/MATH-500) can restore it.
        pre_compile_forward = None
        if use_torch_compile:
            log.info("Stage 6: applying torch.compile(dynamic=True, mode='default') to model.forward")
            try:
                # Capture the pre-compile bound method BEFORE wrapping so the
                # generative block can restore it (Option C: keep compile for
                # prefill-only paths; eager for generate()).
                pre_compile_forward = model.forward
                model.forward = torch.compile(model.forward, dynamic=True, mode="default")
                log.info("Stage 6: torch.compile applied successfully")
            except Exception as exc:
                log.warning("Stage 6: torch.compile failed (%s) — continuing without compilation", exc)
                use_torch_compile = False
                pre_compile_forward = None

        # (7) transformers' LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING is missing an
        # entry for 'linear_attention' in 4.x, but Qwen3.5-MoE's GatedDeltaNet
        # layers register that pattern. create_masks_for_generate (called by
        # generate's prefill path when cache_implementation='static' is active)
        # then raises KeyError: 'linear_attention' at masking_utils.py:1479
        # before the first HumanEval token is produced. Register a passthrough
        # mapping to the same function as 'full_attention' — GatedDeltaNet
        # doesn't consume the attention mask anyway (it derives causality from
        # internal conv1d state via the torch-native fallback we just installed).
        # Same math, same outputs, no quality compromise.
        # M2 (iter1): broadened from `except ImportError` to a generic Exception
        # with a log.warning so a future transformers rename
        # (`LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING` -> something else) or any
        # other non-fatal failure surfaces in the log rather than crashing the
        # whole Stage 6 run. The mapping mutation is process-global; we do not
        # currently restore it (no teardown hook on PipelineContext) — the
        # passthrough is semantically a no-op for non-Qwen3.5-MoE imports, so
        # leaving the entry registered is benign for the rest of the process.
        try:
            from transformers import masking_utils as _mu
            _mapping = getattr(_mu, "LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING", None)
            if isinstance(_mapping, dict) and "linear_attention" not in _mapping:
                if "full_attention" in _mapping:
                    _mapping["linear_attention"] = _mapping["full_attention"]
                    log.info("Stage 6: registered 'linear_attention' → full_attention mask "
                             "in LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING (transformers missing "
                             "entry for Qwen3.5-MoE GatedDeltaNet)")
        except Exception as _exc:  # noqa: BLE001
            log.warning(
                "Stage 6: masking_utils linear_attention passthrough patch "
                "skipped (%s: %s) — generate() on Qwen3.5-MoE may raise "
                "KeyError: 'linear_attention' if transformers has renamed the "
                "mapping.", type(_exc).__name__, _exc,
            )

        ctx.set("experts_impl", experts_impl)
        ctx.set("dataset_revisions", dataset_revisions)
        ctx.set("imatrix_calib_path", imatrix_calib_path)
        ctx.set("use_torch_compile", use_torch_compile)
        # C1 (iter1): pre_compile_forward MUST land in ctx whether or not
        # torch.compile was applied. Generative plugins (HumanEval, Math500)
        # read this slot to restore eager forward before generate() — see
        # the "Downstream contract" section in this method's docstring.
        ctx.set("pre_compile_forward", pre_compile_forward)
        # C2 (iter1): generative experts-impl resolution lives in ctx so
        # HumanEval/Math500 plugins can match teacher_provider.py L607-616.
        ctx.set("experts_implementation_generative", experts_implementation_generative)


__all__ = [
    "_CANONICAL_DATASET_REVISION_KEYS",
    "_resolve_dataset_revisions",
    "_enforce_revision_pinning",
    "_atomic_write_text",
    "_IMATRIX_CALIB_FILENAME",
    "_build_imatrix_calibration_corpus",
    "_set_experts_implementation_s6",
    "_apply_stage6_kernel_patches",
    "EvalEnvironmentPlugin",
]
