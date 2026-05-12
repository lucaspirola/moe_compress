"""Atomic save path for kdr — partial (Phase 3b) and final (Phase 4).

`save_partial` is the Phase 3b / BF16-mode path: vanilla `save_pretrained`
+ tokenizer + optional `compressed_metadata.json` passthrough +
`_SAVE_COMPLETE` sentinel. Atomic via `.tmp` directory + `os.rename`.

`save_kdr_artifact` is the Phase 4 / `da_qad`-mode path: same shape but the
weight serialiser is the compressed-tensors converter from the active
`QuantBackend`. Still stubbed — Phase 4 lands the body.

Atomic-save invariant (LLR-0029): inside a partial dir, `_SAVE_COMPLETE` is
the LAST file written and is empty (zero bytes). Its presence is the
post-rename guarantee that every other file in the dir was committed before
the rename. Resume logic SHALL ignore dirs lacking it.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch.nn as nn
from accelerate import Accelerator
from transformers import PreTrainedTokenizerBase

from ..config import QuantBlock
from ..modes import Mode
from ..quant.interface import QuantBackend
from ..quant.specs import KVQuantSpec, MixedWeightSpec, WeightQuantSpec

log = logging.getLogger(__name__)

# REQ: LLR-0029
SAVE_COMPLETE_SENTINEL = "_SAVE_COMPLETE"
"""Empty file written LAST inside a saved partial dir; its presence is the
post-atomic-rename invariant that the dir is fully committed."""

COMPRESSED_METADATA_FILENAME = "compressed_metadata.json"
"""HLR-0005 / LLR-0019: preserved verbatim from the input student if present."""


# ---------------------------------------------------------------------------
# Async save_partial machinery (LLR-0027 v2)
# ---------------------------------------------------------------------------
#
# Why a module-level single-flight executor:
#
# 1. **Single flight (depth-1 queue)** guarantees monotone partial ordering.
#    If save_partial(step=50) is dispatched and then save_partial(step=60)
#    is dispatched while step=50 is still mid-write, the on-disk and Hub
#    timeline would be racy — partial_step60 could appear before partial_step50.
#    The depth-1 queue forces the second call to auto-join the first.
#
# 2. **Module-level** rather than per-call: the queue must be shared across
#    all save_partial calls in a single run. A run that does 4 saves at
#    steps 10/20/30/40 needs ALL of them to flow through the same executor
#    so each call can auto-join the previous.
#
# 3. **max_workers=1** rather than higher: the rank-0 disk write is bound
#    by sequential I/O to the local SSD; parallelizing would contend for
#    bandwidth with the trainer's own checkpoint reads (none in steady
#    state, but the load-back-round-trip at the end of bootstrap.sh).
#    More importantly, max_workers=1 makes the single-flight invariant
#    structural rather than enforced-by-convention.
#
# Generic-tool notes for adapting to larger setups:
#
# - **Pinned-memory cost**: the CPU state_dict snapshot is `.detach().cpu().clone()`,
#   which costs ~weight-tensor-size in pageable CPU RAM. For an 8B BF16
#   model that's ~17 GB; for 70B that's ~140 GB. If your host CPU RAM is
#   constrained, leave `enable_async_save: false` (sync save reuses the
#   ZeRO-3 consolidation buffer in place) or implement a streamed-to-disk
#   serializer that pages tensors out of the GPU directly to disk without
#   the CPU intermediate.
#
# - **Multi-rank**: under DDP/FSDP/DS the collective `get_state_dict` is
#   ALWAYS synchronous (NCCL is not thread-safe). Only the rank-0
#   post-consolidation disk writes are dispatched to the background
#   thread. The wait_for_everyone at the end of save_partial barriers
#   ranks AFTER the submit returns, so other ranks aren't blocked on
#   rank-0's disk write.
#
# - **Crash safety**: if the trainer crashes with a pending Future, the
#   ThreadPoolExecutor's threads are non-daemon and will block process
#   exit until they finish. Acceptable for crash-on-bug; for crash-on-
#   stop-signal the user can ctrl-c twice.
class _AsyncSaveExecutor:
    """Module-level single-flight executor for async save_partial."""

    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="kdr-save-partial"
        )
        self._pending: Future[None] | None = None

    def submit(self, fn: Any, *args: Any, **kwargs: Any) -> Future[None]:
        # Auto-join the prior future BEFORE submitting the next. Re-raises
        # if the prior write failed — caller sees the exception at the
        # call site of the second save, not somewhere later at random.
        #
        # `try/finally` guarantees `_pending` is cleared even if `.result()`
        # raises. Without this, a failed Future stays referenced in
        # `_pending` and every subsequent `.join()` / `.submit()` would
        # re-raise the same stale exception. The contract is: a Future's
        # exception is reported exactly once, at the next operation after
        # it failed, and then the executor state advances.
        if self._pending is not None:
            try:
                self._pending.result()
            finally:
                self._pending = None
        future = self._executor.submit(fn, *args, **kwargs)
        self._pending = future
        return future

    def join(self) -> None:
        """Block until any pending future completes; re-raise its
        exception if it failed. No-op if no future is pending."""
        if self._pending is not None:
            try:
                self._pending.result()
            finally:
                self._pending = None

    def shutdown(self) -> None:
        """Final teardown — wait for outstanding work and free the thread.
        Currently called only from test teardown; production runs let the
        executor live for the process lifetime (the thread exits at
        process exit anyway)."""
        self.join()
        self._executor.shutdown(wait=True)


# Module-global singleton. Construction is lazy (a thread is created only
# at first .submit). Resetting is exposed for tests via _reset_async_save_executor().
_ASYNC_SAVE = _AsyncSaveExecutor()


def save_partial_join() -> None:
    """Public entry point: flush any pending async save_partial Future.

    Call this before the final save (which must be synchronous, see
    LLR-0027 AC) and any time the caller needs the disk state to reflect
    all in-flight partials (e.g., before uploading from the trainer).

    Re-raises any exception that the background thread saw — this is how
    a failure mode in the disk write surfaces to the trainer rather than
    silently corrupting the partial chain.
    """
    _ASYNC_SAVE.join()


def _reset_async_save_executor() -> None:
    """Test-only: replace the module-global executor with a fresh one
    after a failure test (otherwise the stale Future state would leak
    into the next test).

    This is teardown-safe: any exception raised by a pending Future is
    SWALLOWED here. The test that raised the exception is responsible
    for catching it inline (via `pytest.raises(...)`); the role of this
    function is only to scrub state between tests, not to surface
    failures. A leaked exception from the prior test must not break the
    next test's setup phase.
    """
    global _ASYNC_SAVE  # noqa: PLW0603 — module-global is the contract
    try:
        _ASYNC_SAVE.shutdown()
    except BaseException:  # noqa: BLE001 — see docstring: teardown swallows.
        pass
    _ASYNC_SAVE = _AsyncSaveExecutor()


# REQ: LLR-0027
def partial_dir_name(mode: Mode, step: int) -> str:
    """`kdr_{mode}_partial_step{N}` — embeds both mode and step.

    Mode-prefixing avoids cross-mode resume contamination (a partial saved
    by `bf16` cannot be picked up as a resume seed for `da_qad` which
    has different module wrappers).
    """
    return f"kdr_{mode}_partial_step{step}"


# REQ: LLR-0027
def save_partial(
    student: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    accelerator: Accelerator,
    *,
    artifacts_dir: Path,
    mode: Mode,
    step: int,
    source_metadata_path: Path | None = None,
    extra_metadata: dict[str, Any] | None = None,
    partial: bool = True,
    async_mode: bool = False,
) -> Path:
    """Atomic save of `student` to a partial (or final) dir.

    Layout:
      `<artifacts_dir>/kdr_{mode}_partial_step{step}/` (partial=True), or
      `<artifacts_dir>/kdr_{mode}_recovered/` (partial=False).

    Steps (all rank-0):
      1. Build under `.tmp/` so a half-written dir never appears at the
         final path.
      2. `unwrapped.save_pretrained(tmp_dir, state_dict=…, safe_serialization=True)`
         — `state_dict` comes from `accelerator.get_state_dict(student)`,
         which under ZeRO-3 streams the consolidated tensors INTO CPU memory
         on rank 0 only (other ranks return `{}`).
      3. `tokenizer.save_pretrained(tmp_dir)`.
      4. Copy `compressed_metadata.json` verbatim from `source_metadata_path`
         if it exists (HLR-0005).
      5. Write `extra_metadata` (if provided) into a sidecar
         `kdr_run_metadata.json`.
      6. Atomically rename `tmp_dir` → final dir.
      7. Write empty `_SAVE_COMPLETE` LAST so its presence post-rename is
         the integrity sentinel (LLR-0029).

    All ranks call (`get_state_dict` is collective under DS); only rank 0
    actually writes.

    `async_mode` (LLR-0027 v2): when True (only valid for `partial=True`),
    the rank-0 disk-write phase is dispatched to a single-flight background
    thread; this function returns the target `Path` immediately after the
    collective consolidation barrier. Use `save_partial_join()` to flush
    pending writes before the final save (or to surface background-thread
    exceptions). The CPU state_dict is deep-copied before submission so
    subsequent optimizer steps cannot mutate the snapshot.
    """
    if async_mode and not partial:
        raise ValueError(
            "save_partial(async_mode=True, partial=False) is not supported. "
            "The final save must complete synchronously because its return "
            "path is consumed immediately by the upload step. Call "
            "save_partial_join() to flush pending writes, then call "
            "save_partial(..., partial=False) with async_mode=False."
        )

    accelerator.wait_for_everyone()

    out_name = partial_dir_name(mode, step) if partial else f"kdr_{mode}_recovered"
    out_dir = artifacts_dir / out_name
    tmp_dir = out_dir.parent / f"{out_dir.name}.tmp"

    # Collective: every rank participates in the consolidation. This MUST
    # happen on the main thread regardless of async_mode (NCCL is not
    # thread-safe). Only the post-consolidation rank-0 disk write is
    # dispatched to a background thread when async_mode=True.
    state_dict = accelerator.get_state_dict(student)
    unwrapped = accelerator.unwrap_model(student)

    if accelerator.is_main_process:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        if async_mode:
            # Deep-copy the state_dict to independent CPU tensors so the
            # main thread can return to training immediately and any
            # subsequent optimizer.step() updates to GPU weights cannot
            # mutate the snapshot that the background thread is serializing.
            #
            # `.detach().cpu().clone()` breaks any reference back to the
            # source storage:
            #   - `.detach()` strips autograd context (not strictly needed
            #     post-consolidation but cheap insurance).
            #   - `.cpu()` moves GPU tensors to CPU (no-op on already-CPU
            #     tensors under ZeRO-3 consolidation).
            #   - `.clone()` allocates an independent CPU tensor; even if
            #     `.cpu()` was a no-op it now decouples from any DS-internal
            #     CPU buffer that might be reused on the next collective.
            cpu_state_dict = {
                k: v.detach().cpu().clone() for k, v in state_dict.items()
            }
            # Release the main-thread reference so the original (possibly
            # GPU-backed) tensors can be reaped immediately.
            del state_dict
            _ASYNC_SAVE.submit(
                _write_partial_dir,
                tmp_dir=tmp_dir,
                out_dir=out_dir,
                state_dict=cpu_state_dict,
                unwrapped=unwrapped,
                tokenizer=tokenizer,
                source_metadata_path=source_metadata_path,
                extra_metadata=extra_metadata,
                partial=partial,
                step=step,
            )
        else:
            _write_partial_dir(
                tmp_dir=tmp_dir,
                out_dir=out_dir,
                state_dict=state_dict,
                unwrapped=unwrapped,
                tokenizer=tokenizer,
                source_metadata_path=source_metadata_path,
                extra_metadata=extra_metadata,
                partial=partial,
                step=step,
            )

    accelerator.wait_for_everyone()
    return out_dir


def _write_partial_dir(
    *,
    tmp_dir: Path,
    out_dir: Path,
    state_dict: dict[str, Any],
    unwrapped: nn.Module,
    tokenizer: PreTrainedTokenizerBase | None,
    source_metadata_path: Path | None,
    extra_metadata: dict[str, Any] | None,
    partial: bool,
    step: int,
) -> None:
    """Rank-0 disk-write subroutine for save_partial. Pure I/O.

    Safe to call from the main thread (sync mode) OR from the
    ``_AsyncSaveExecutor`` background thread (async mode). The body
    performs no collective ops, no GPU work, and no shared-state mutation
    beyond writes to its own `tmp_dir`/`out_dir`.

    Preserves the LLR-0029 sentinel-last invariant: every other file is
    written first, atomic rename `tmp_dir → out_dir` flips the dir into
    place, then the empty `_SAVE_COMPLETE` file is `touch`'d LAST inside
    the renamed dir. Resume logic ignores dirs lacking the sentinel.

    Generic-tool note: when adapting for a non-HF model whose
    `save_pretrained` semantics differ, the only contract this function
    needs is "write everything into `tmp_dir`, then `_atomic_replace_dir`,
    then touch sentinel". The state_dict / tokenizer / metadata operations
    can be swapped without changing the atomicity guarantee.
    """
    # Clean any stale `.tmp` from a previous failed save.
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    unwrapped.save_pretrained(
        tmp_dir, state_dict=state_dict, safe_serialization=True
    )
    if tokenizer is not None:
        tokenizer.save_pretrained(tmp_dir)

    # REQ: LLR-0019
    if source_metadata_path is not None and source_metadata_path.exists():
        shutil.copyfile(
            source_metadata_path, tmp_dir / COMPRESSED_METADATA_FILENAME
        )

    if extra_metadata is not None:
        (tmp_dir / "kdr_run_metadata.json").write_text(
            json.dumps(extra_metadata, indent=2, sort_keys=True)
        )

    # Atomic rename — replaces an existing final dir if present.
    _atomic_replace_dir(tmp_dir, out_dir)

    # REQ: LLR-0029
    # Sentinel written LAST, INTO the renamed final dir, EMPTY (zero bytes).
    # Mtime ordering guarantees every other file's mtime ≤ sentinel's.
    sentinel = out_dir / SAVE_COMPLETE_SENTINEL
    sentinel.touch(exist_ok=False)

    log.info(
        "Saved %s checkpoint to %s (step=%d)",
        "PARTIAL" if partial else "FINAL",
        out_dir,
        step,
    )


# REQ: LLR-0018
# REQ: LLR-0019
# REQ: LLR-0020
def save_kdr_artifact(
    model: nn.Module,
    output_dir: Path,
    *,
    backends: list[QuantBackend],
    quant_block: QuantBlock,
    fp32_carve_outs: list[str],
    tokenizer: PreTrainedTokenizerBase | None = None,
    source_metadata_path: Path | None = None,
) -> None:
    """Compressed-tensors final save (Phase 4 / ``da_qad`` mode).

    Mirrors ``save_partial``'s atomicity pattern (LLR-0029): all writes land
    in a sibling ``.tmp`` directory; only after every step succeeds is the
    ``.tmp`` atomically renamed onto ``output_dir``; the sentinel is then
    written LAST, INTO the renamed final dir, so its presence is the
    post-rename guarantee that every other file in the dir was committed.

    Sequence:

      1. Build under ``.tmp/`` so a half-written dir never appears at
         ``output_dir``.
      2. Pick the weight-handling backend (the routed backend whose
         ``QuantBlockSubset`` carries ``weight``); call its ``.save`` to
         emit compressed-tensors safetensors + ``config.json`` (LLR-0021).
      3. Inject the full ``quantization_config`` block into ``config.json``
         (LLR-0020) — covers the K/V cache scheme and FP32 ``ignore`` list
         that the converter would otherwise miss.
      4. Save the tokenizer if provided.
      5. Preserve the input student's ``compressed_metadata.json`` verbatim
         when ``source_metadata_path`` exists (HLR-0005 / LLR-0019).
      6. Atomically rename ``.tmp`` → ``output_dir``.
      7. Write the empty ``_SAVE_COMPLETE`` sentinel last (LLR-0029
         invariant — sentinel is written with ``exist_ok=False`` so a
         stale sentinel from a prior crash + retry surfaces as an error
         rather than masquerading as a successful re-save).

    Args:
        model: the quantized student (post ``apply_quant``).
        output_dir: target directory for the final artifact.
        backends: routes returned by ``factory.partition_and_dispatch``.
        quant_block: original YAML quant block (used to compose the
            ``quantization_config`` payload — LLR-0020).
        fp32_carve_outs: adapter's FP32 carve-out submodule patterns
            (becomes the ``ignore`` list — LLR-0020 AC #3).
        tokenizer: student tokenizer; saved alongside if provided.
        source_metadata_path: input student's ``compressed_metadata.json``
            location for byte-equal passthrough; ``None`` if the input
            lacked the file.

    Raises:
        ValueError: if no backend handles the weight quantizer (the
            converter selection requires it).
    """
    weight_backend = _find_weight_handling_backend(backends)
    if weight_backend is None:
        raise ValueError(
            "save_kdr_artifact: no backend in `backends` handles the weight "
            "quantizer; the compressed-tensors save path requires a "
            "weight-handling backend (typically ModelOpt)."
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir.parent / f"{output_dir.name}.tmp"
    if tmp_dir.exists():
        # Stale `.tmp` from a previous failed save — discard.
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    # 2. Backend writes weights + config.json via the format-specific converter.
    weight_backend.save(model, tmp_dir)

    # 3. Inject the full quantization_config block into config.json.
    _inject_quantization_config(tmp_dir, quant_block, fp32_carve_outs)

    # 4. Tokenizer (separate from the converter's save).
    if tokenizer is not None:
        tokenizer.save_pretrained(tmp_dir)

    # 5. Preserve compressed_metadata.json verbatim if the input had it.
    if source_metadata_path is not None and source_metadata_path.exists():
        shutil.copyfile(source_metadata_path, tmp_dir / COMPRESSED_METADATA_FILENAME)

    # 6. Atomic rename — replaces an existing final dir if present.
    _atomic_replace_dir(tmp_dir, output_dir)

    # REQ: LLR-0029
    # 7. Sentinel written LAST, INTO the renamed final dir, EMPTY (zero bytes).
    #    `exist_ok=False` matches `save_partial` so stale sentinels surface as
    #    errors rather than masking a re-save.
    sentinel = output_dir / SAVE_COMPLETE_SENTINEL
    sentinel.touch(exist_ok=False)

    log.info("save_kdr_artifact: wrote final compressed-tensors checkpoint to %s", output_dir)


def _find_weight_handling_backend(
    backends: list[QuantBackend],
) -> QuantBackend | None:
    """Pick the backend whose dispatched ``QuantBlockSubset`` includes weight.

    Backends store the dispatched subset on ``self._quant_block`` (set inside
    ``apply_quant``). Inspecting it avoids threading the routes through a
    second parameter.
    """
    for b in backends:
        # Both ModelOptBackend and NativeBackend expose ``_quant_block``;
        # ``getattr`` keeps this duck-typed against the Protocol surface.
        sub = getattr(b, "_quant_block", None)
        if sub is not None and getattr(sub, "weight", None) is not None:
            return b
    return None


def _inject_quantization_config(
    output_dir: Path,
    quant_block: QuantBlock,
    fp32_carve_outs: list[str],
) -> None:
    """Patch ``config.json`` with the compressed-tensors ``quantization_config``.

    The backend's converter typically writes a partial ``quantization_config``
    that doesn't fully reflect kdr's recipe — this function overwrites the
    block with the canonical kdr-built payload composed from the YAML.
    """
    cfg_path = output_dir / "config.json"
    if not cfg_path.exists():
        # The converter is expected to produce config.json; if it didn't,
        # write a minimal stub so the output dir is at least loadable as a
        # bare HF dir. The caller's verifier flags any deeper issues.
        cfg: dict[str, Any] = {}
    else:
        cfg = json.loads(cfg_path.read_text())
    cfg["quantization_config"] = _build_quantization_config(quant_block, fp32_carve_outs)
    cfg_path.write_text(json.dumps(cfg, indent=2, sort_keys=True))


def _build_quantization_config(
    quant_block: QuantBlock,
    fp32_carve_outs: list[str],
) -> dict[str, Any]:
    """Compose the HF ``quantization_config`` dict (LLR-0020).

    Schema (compressed-tensors flavoured):

        {
            "quant_method": "compressed-tensors",
            "config_groups": {
                "group_0": {
                    "weights": <WeightArgs>,
                    "input_activations": None,
                    "targets": ["Linear"],
                },
            },
            "kv_cache_scheme": { "key": <KVArgs>, "value": <KVArgs> },
            "ignore": [<fp32 carve-out patterns>],
        }
    """
    if isinstance(quant_block.weight, MixedWeightSpec):
        raise NotImplementedError(
            "save_kdr_artifact's quantization_config emission for MixedWeightSpec "
            "lands in Phase 7.2 Task 6."
        )
    return {
        "quant_method": "compressed-tensors",
        "config_groups": {
            "group_0": {
                "weights": _weight_spec_to_ct(quant_block.weight),
                "input_activations": None,
                "targets": ["Linear"],
            },
        },
        "kv_cache_scheme": {
            "key": _kv_spec_to_ct(quant_block.kv_quant.key),
            "value": _kv_spec_to_ct(quant_block.kv_quant.value),
        },
        "ignore": list(fp32_carve_outs),
    }


def _weight_spec_to_ct(spec: WeightQuantSpec) -> dict[str, Any]:
    """Translate kdr's ``WeightQuantSpec`` to a compressed-tensors-shaped dict."""
    return {
        "num_bits": spec.bits,
        "type": _format_to_ct_type(spec.format),
        "strategy": _granularity_to_ct_strategy(spec.granularity),
        "symmetric": True,
    }


def _kv_spec_to_ct(spec: KVQuantSpec) -> dict[str, Any]:
    """Translate kdr's ``KVQuantSpec`` to a compressed-tensors-shaped dict."""
    return {
        "num_bits": spec.bits,
        "type": _format_to_ct_type(spec.format),
        "strategy": _granularity_to_ct_strategy(spec.granularity),
        "symmetric": True,
    }


def _format_to_ct_type(fmt: str) -> str:
    """Map kdr ``Format`` literal → compressed-tensors ``type`` string."""
    if fmt == "int":
        return "int"
    # ``fp8``, ``nvfp4``, ``mxfp4`` all live under "float" in compressed-tensors.
    return "float"


def _granularity_to_ct_strategy(g: str) -> str:
    """Map kdr ``Granularity`` literal → compressed-tensors ``strategy`` string."""
    # compressed-tensors uses these literal strings; pass through directly
    # except for ``token`` which it spells the same way.
    return g


# ---------------------------------------------------------------------------
# Atomic helpers
# ---------------------------------------------------------------------------


def _atomic_replace_dir(src: Path, dst: Path) -> None:
    """Atomically replace `dst` with `src`. Both must be on the same FS.

    POSIX `rename(2)` (and Python's `os.rename`) refuses to replace a
    non-empty directory. We work around by moving the existing `dst` aside
    first; on rename failure, restore it.
    """
    if dst.exists():
        backup = dst.with_name(dst.name + ".bak")
        if backup.exists():
            shutil.rmtree(backup)
        os.rename(dst, backup)
        try:
            os.rename(src, dst)
        except Exception:
            # Restore the backup on any failure.
            os.rename(backup, dst)
            raise
        shutil.rmtree(backup, ignore_errors=True)
    else:
        os.rename(src, dst)
