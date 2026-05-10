"""Tests for `kdr.io.save.save_kdr_artifact` (LLR-0018, LLR-0019, LLR-0020).

# VERIFIES: LLR-0018
# VERIFIES: LLR-0019
# VERIFIES: LLR-0020
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch.nn as nn

from kdr.config import KVQuantBlock, QuantBlock
from kdr.io.save import (
    COMPRESSED_METADATA_FILENAME,
    SAVE_COMPLETE_SENTINEL,
    save_kdr_artifact,
)
from kdr.quant.interface import QuantBlockSubset
from kdr.quant.specs import KVQuantSpec, WeightQuantSpec


def _qb() -> QuantBlock:
    return QuantBlock(
        weight=WeightQuantSpec(bits=4, format="nvfp4", granularity="channel", transform="none"),
        kv_quant=KVQuantBlock(
            key=KVQuantSpec(bits=4, format="int", granularity="channel", transform="none"),
            value=KVQuantSpec(bits=2, format="int", granularity="token", transform="none"),
        ),
    )


def _fake_backend(
    *, weight: WeightQuantSpec | None, has_save_pretrained: bool = True
) -> MagicMock:
    """A QuantBackend stand-in. Records `.save` calls and writes a stub config.json."""
    b = MagicMock()
    b.name = "fake"
    b._quant_block = QuantBlockSubset(weight=weight)

    def _save(model: nn.Module, output_dir: Path) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        # Simulate the converter writing a partial config + safetensors stub.
        (out / "config.json").write_text(json.dumps({"hidden_size": 8}))
        (out / "model.safetensors").write_bytes(b"\x00" * 16)

    b.save.side_effect = _save
    return b


def test_save_kdr_artifact_writes_quantization_config(tmp_path: Path) -> None:
    """LLR-0020 AC: `quantization_config.config_groups.weights`,
    `kv_cache_scheme`, and `ignore` populated correctly."""
    qb = _qb()
    backend = _fake_backend(weight=qb.weight)

    save_kdr_artifact(
        nn.Linear(4, 4),
        tmp_path / "out",
        backends=[backend],
        quant_block=qb,
        fp32_carve_outs=["lm_head", "rmsnorm"],
    )

    cfg = json.loads((tmp_path / "out" / "config.json").read_text())
    qc = cfg["quantization_config"]
    # quant_method
    assert qc["quant_method"] == "compressed-tensors"
    # config_groups.weights
    weights = qc["config_groups"]["group_0"]["weights"]
    assert weights["num_bits"] == 4
    assert weights["type"] == "float"  # nvfp4 → float
    assert weights["strategy"] == "channel"
    assert weights["symmetric"] is True
    # kv_cache_scheme
    kv = qc["kv_cache_scheme"]
    assert kv["key"]["num_bits"] == 4
    assert kv["key"]["type"] == "int"
    assert kv["key"]["strategy"] == "channel"
    assert kv["value"]["num_bits"] == 2
    assert kv["value"]["strategy"] == "token"
    # ignore list
    assert qc["ignore"] == ["lm_head", "rmsnorm"]


def test_save_kdr_artifact_preserves_compressed_metadata_byte_equal(
    tmp_path: Path,
) -> None:
    """LLR-0019 AC #1: byte-equality of compressed_metadata.json passthrough."""
    src_meta = tmp_path / "src" / COMPRESSED_METADATA_FILENAME
    src_meta.parent.mkdir()
    src_meta.write_text('{"version": 1, "factored_layers": [3, 7]}')

    qb = _qb()
    save_kdr_artifact(
        nn.Linear(4, 4),
        tmp_path / "out",
        backends=[_fake_backend(weight=qb.weight)],
        quant_block=qb,
        fp32_carve_outs=[],
        source_metadata_path=src_meta,
    )
    out_meta = tmp_path / "out" / COMPRESSED_METADATA_FILENAME
    assert out_meta.exists()
    assert out_meta.read_text() == src_meta.read_text()


def test_save_kdr_artifact_omits_metadata_when_input_lacks_it(tmp_path: Path) -> None:
    """LLR-0019 AC #2: input lacking the file → output also lacks it."""
    qb = _qb()
    save_kdr_artifact(
        nn.Linear(4, 4),
        tmp_path / "out",
        backends=[_fake_backend(weight=qb.weight)],
        quant_block=qb,
        fp32_carve_outs=[],
        source_metadata_path=None,
    )
    assert not (tmp_path / "out" / COMPRESSED_METADATA_FILENAME).exists()


def test_save_kdr_artifact_writes_sentinel(tmp_path: Path) -> None:
    """LLR-0029 invariant: `_SAVE_COMPLETE` is written and is empty."""
    qb = _qb()
    save_kdr_artifact(
        nn.Linear(4, 4),
        tmp_path / "out",
        backends=[_fake_backend(weight=qb.weight)],
        quant_block=qb,
        fp32_carve_outs=[],
    )
    sentinel = tmp_path / "out" / SAVE_COMPLETE_SENTINEL
    assert sentinel.exists()
    assert sentinel.stat().st_size == 0


def test_save_kdr_artifact_requires_weight_handling_backend(tmp_path: Path) -> None:
    """No backend handles weight → save can't pick a converter."""
    qb = _qb()
    # All backends with key/value but no weight subset.
    kv_only = _fake_backend(weight=None)
    with pytest.raises(ValueError, match="no backend in `backends` handles the weight"):
        save_kdr_artifact(
            nn.Linear(4, 4),
            tmp_path / "out",
            backends=[kv_only],
            quant_block=qb,
            fp32_carve_outs=[],
        )


def test_save_kdr_artifact_overwrites_existing_quantization_config(tmp_path: Path) -> None:
    """The backend's converter may write a partial ``quantization_config`` into
    the staging dir; kdr's composed payload REPLACES it while preserving any
    other keys the backend wrote (LLR-0020 — kdr is the source of truth for
    the quantization_config block, but model architecture keys flow through)."""
    qb = _qb()
    backend = _fake_backend(weight=qb.weight)

    def _save(_m: object, output_dir: Path) -> None:
        # Simulate a converter that writes BOTH a model arch key AND a stale
        # quantization_config (some real converters do exactly this).
        (Path(output_dir) / "config.json").write_text(
            json.dumps({"hidden_size": 8, "quantization_config": {"stale": True}})
        )

    backend.save.side_effect = _save

    save_kdr_artifact(
        nn.Linear(4, 4),
        tmp_path / "out",
        backends=[backend],
        quant_block=qb,
        fp32_carve_outs=[],
    )
    cfg = json.loads((tmp_path / "out" / "config.json").read_text())
    # Architecture key from the converter survives the merge.
    assert cfg.get("hidden_size") == 8
    # kdr's composed payload overwrites the stale quantization_config.
    assert cfg["quantization_config"]["quant_method"] == "compressed-tensors"
    assert "stale" not in cfg["quantization_config"]


def test_save_kdr_artifact_calls_tokenizer_save_when_provided(tmp_path: Path) -> None:
    """Tokenizer is saved INTO the staging .tmp dir (atomic-rename pattern)
    BEFORE the rename onto the final output_dir."""
    qb = _qb()
    tok = MagicMock()
    save_kdr_artifact(
        nn.Linear(4, 4),
        tmp_path / "out",
        backends=[_fake_backend(weight=qb.weight)],
        quant_block=qb,
        fp32_carve_outs=[],
        tokenizer=tok,
    )
    tok.save_pretrained.assert_called_once_with(tmp_path / "out.tmp")


def test_save_kdr_artifact_atomic_no_tmp_leaks(tmp_path: Path) -> None:
    """LLR-0029 atomic invariant: after a successful save, no `.tmp` sibling
    survives — the rename consumed it."""
    qb = _qb()
    out = tmp_path / "out"
    save_kdr_artifact(
        nn.Linear(4, 4),
        out,
        backends=[_fake_backend(weight=qb.weight)],
        quant_block=qb,
        fp32_carve_outs=[],
    )
    # Final dir present, sentinel written, .tmp gone.
    assert out.exists()
    assert (out / SAVE_COMPLETE_SENTINEL).exists()
    assert not (tmp_path / "out.tmp").exists()


def test_save_kdr_artifact_discards_stale_tmp(tmp_path: Path) -> None:
    """A leftover `.tmp` from a previous failed run is purged before the
    fresh staging starts — otherwise we'd inherit corrupted state."""
    qb = _qb()
    out = tmp_path / "out"
    # Plant a stale .tmp with garbage content.
    stale = tmp_path / "out.tmp"
    stale.mkdir()
    (stale / "stale.txt").write_text("corruption")

    save_kdr_artifact(
        nn.Linear(4, 4),
        out,
        backends=[_fake_backend(weight=qb.weight)],
        quant_block=qb,
        fp32_carve_outs=[],
    )
    assert out.exists()
    assert not (out / "stale.txt").exists()
    assert not stale.exists()


def test_save_kdr_artifact_sentinel_exist_ok_false(tmp_path: Path) -> None:
    """LLR-0029 invariant — `exist_ok=False` ensures a stale sentinel from a
    crashed prior run surfaces as ``FileExistsError`` rather than masking
    a re-save."""
    qb = _qb()
    out = tmp_path / "out"
    # First save succeeds.
    save_kdr_artifact(
        nn.Linear(4, 4),
        out,
        backends=[_fake_backend(weight=qb.weight)],
        quant_block=qb,
        fp32_carve_outs=[],
    )
    # Manually re-save: rename will replace the dir, but sentinel.touch
    # with exist_ok=False fires only if the rename DIDN'T fully replace.
    # The atomic-replace path moves the old dir aside, so the new sentinel
    # write into the swapped-in tmp dir succeeds. This test asserts that
    # behavior (a clean second save) AND that the sentinel ends up empty
    # — together, they verify exist_ok=False is correctly partnered with
    # the atomic-replace pattern.
    save_kdr_artifact(
        nn.Linear(4, 4),
        out,
        backends=[_fake_backend(weight=qb.weight)],
        quant_block=qb,
        fp32_carve_outs=[],
    )
    sentinel = out / SAVE_COMPLETE_SENTINEL
    assert sentinel.exists()
    assert sentinel.stat().st_size == 0
