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
from pathlib import Path
from typing import Any

import torch.nn as nn
from accelerate import Accelerator
from transformers import PreTrainedTokenizerBase

from ..config import QuantBlock
from ..modes import Mode
from ..quant.interface import QuantBackend
from ..quant.specs import KVQuantSpec, WeightQuantSpec

log = logging.getLogger(__name__)

# REQ: LLR-0029
SAVE_COMPLETE_SENTINEL = "_SAVE_COMPLETE"
"""Empty file written LAST inside a saved partial dir; its presence is the
post-atomic-rename invariant that the dir is fully committed."""

COMPRESSED_METADATA_FILENAME = "compressed_metadata.json"
"""HLR-0005 / LLR-0019: preserved verbatim from the input student if present."""


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
    """
    accelerator.wait_for_everyone()

    out_name = partial_dir_name(mode, step) if partial else f"kdr_{mode}_recovered"
    out_dir = artifacts_dir / out_name
    tmp_dir = out_dir.parent / f"{out_dir.name}.tmp"

    # Collective: every rank participates in the consolidation.
    state_dict = accelerator.get_state_dict(student)
    unwrapped = accelerator.unwrap_model(student)

    if accelerator.is_main_process:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
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

    accelerator.wait_for_everyone()
    return out_dir


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
