"""Tests for kdr.tools.kdr_to_gguf (Task 9 / LLR-0059).

# REQ: LLR-0059
"""

from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import ANY, MagicMock, patch

import numpy as np
import pytest

# ─────────────────────────────────────────────────────────────────────────────
# Module loading — install a stub `gguf` so we can import kdr_to_gguf
# ─────────────────────────────────────────────────────────────────────────────


class _FakeQuantType:
    IQ2_XS = "IQ2_XS"
    Q3_K = "Q3_K"
    IQ4_XS = "IQ4_XS"
    Q5_K = "Q5_K"


def _install_gguf_stub() -> types.ModuleType:
    """Install a sentinel `gguf` module on sys.modules and return it."""
    stub = types.ModuleType("gguf")
    stub.GGMLQuantizationType = _FakeQuantType
    stub.GGUFWriter = MagicMock
    sys.modules["gguf"] = stub
    return stub


def _import_kdr_to_gguf() -> Any:
    """Import (or reimport) the converter module under the gguf stub."""
    _install_gguf_stub()
    if "kdr.tools.kdr_to_gguf" in sys.modules:
        return importlib.reload(sys.modules["kdr.tools.kdr_to_gguf"])
    return importlib.import_module("kdr.tools.kdr_to_gguf")


# Import once at collection time so individual tests can refer to symbols.
_kg = _import_kdr_to_gguf()


# ─────────────────────────────────────────────────────────────────────────────
# Format table
# ─────────────────────────────────────────────────────────────────────────────


def test_format_to_gguf_type_has_four_profile_j_entries() -> None:
    """LLR-0059 AC: _FORMAT_TO_GGUF_TYPE covers exactly the four Profile-J formats."""
    assert set(_kg._FORMAT_TO_GGUF_TYPE.keys()) == {
        "iq2_xs",
        "q3_k",
        "iq4_xs",
        "q5_k",
    }
    assert _kg._FORMAT_TO_GGUF_TYPE["iq2_xs"] == _FakeQuantType.IQ2_XS
    assert _kg._FORMAT_TO_GGUF_TYPE["q3_k"] == _FakeQuantType.Q3_K
    assert _kg._FORMAT_TO_GGUF_TYPE["iq4_xs"] == _FakeQuantType.IQ4_XS
    assert _kg._FORMAT_TO_GGUF_TYPE["q5_k"] == _FakeQuantType.Q5_K


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def test_arch_id_strips_forcausallm_and_lowercases() -> None:
    """LLR-0059 AC: _arch_id behaves on the two reference inputs."""
    assert _kg._arch_id("ZayaForCausalLM") == "zaya"
    assert _kg._arch_id("LlamaForCausalLM") == "llama"


def test_to_gguf_tensor_name_maps_embed_and_lm_head() -> None:
    """LLR-0059 AC: pure name translation for the tied-embed names."""
    assert _kg._to_gguf_tensor_name("model.embed_tokens.weight") == "token_embd.weight"
    assert _kg._to_gguf_tensor_name("lm_head.weight") == "token_embd.weight"
    # Pass-through for everything else.
    assert _kg._to_gguf_tensor_name("model.layers.0.mlp.gate_proj.weight") == (
        "model.layers.0.mlp.gate_proj.weight"
    )


def test_match_group_first_match_wins() -> None:
    """LLR-0059 AC: insertion-ordered first-match-wins."""
    config_groups = {
        "group_0": {"targets": ["proj"], "weights": {"num_bits": 4, "strategy": "block"}},
        "group_1": {"targets": ["q_proj"], "weights": {"num_bits": 5, "strategy": "block"}},
    }
    matched = _kg._match_group("attn.q_proj.weight", config_groups)
    assert matched is config_groups["group_0"]


def test_match_group_linear_is_catch_all() -> None:
    """LLR-0059 AC: the literal pattern "Linear" matches every tensor name.

    Mirrors the v0 Uniform-path emission from LLR-0056 where
    `_build_quantization_config` writes `targets: ["Linear"]`.
    """
    config_groups = {
        "group_0": {
            "targets": ["Linear"],
            "weights": {"num_bits": 4, "strategy": "block", "type": "int", "kdr_format": "iq4_xs"},
        },
    }
    matched = _kg._match_group(
        "model.layers.0.attn.q_proj.weight", config_groups
    )
    assert matched is config_groups["group_0"]


def test_match_group_returns_none_on_no_match() -> None:
    config_groups = {
        "group_0": {"targets": ["proj"], "weights": {"num_bits": 4, "strategy": "block"}},
    }
    assert _kg._match_group("attn.weird_layer.weight", config_groups) is None


# ─────────────────────────────────────────────────────────────────────────────
# convert() — mock-driven
# ─────────────────────────────────────────────────────────────────────────────


def _write_artifact(
    tmp_path: Path,
    *,
    config_groups: dict[str, Any],
    tensors: dict[str, np.ndarray],
    ignore: list[str] | None = None,
    tie_word_embeddings: bool = False,
    architectures: list[str] | None = None,
) -> Path:
    """Stage a stand-in kdr-dir at `tmp_path`."""
    out = tmp_path / "art"
    out.mkdir()
    cfg: dict[str, Any] = {
        "_name_or_path": "kdr/test",
        "architectures": architectures or ["Zaya1ForCausalLM"],
        "tie_word_embeddings": tie_word_embeddings,
        "hidden_size": 16,
        "num_attention_heads": 2,
        "num_hidden_layers": 2,
        "vocab_size": 100,
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "config_groups": config_groups,
            "ignore": ignore or [],
        },
    }
    (out / "config.json").write_text(json.dumps(cfg))
    from safetensors.numpy import save_file

    save_file(tensors, str(out / "model.safetensors"))
    return out


def test_convert_dispatches_per_group_and_ignores_carve_outs(tmp_path: Path) -> None:
    """LLR-0059 AC: convert() calls add_tensor with raw_dtype for groups, F16 for ignore."""
    art = _write_artifact(
        tmp_path,
        config_groups={
            "group_0": {
                "weights": {"num_bits": 2, "strategy": "block", "type": "int", "kdr_format": "iq2_xs"},
                "input_activations": None,
                "targets": ["gate_proj"],
            },
        },
        tensors={
            "model.gate_proj.weight": np.zeros((4, 4), dtype=np.float32),
            "model.norm.weight": np.zeros((4,), dtype=np.float32),
        },
        ignore=["norm"],
    )
    with patch.object(_kg.gguf, "GGUFWriter") as mock_writer_cls:
        writer = mock_writer_cls.return_value
        _kg.convert(art, tmp_path / "out.gguf")
        # Architecture passed as the second positional ctor arg.
        mock_writer_cls.assert_called_once_with(str(tmp_path / "out.gguf"), "zaya1")
        # kdr.source_dir written.
        writer.add_kv.assert_any_call("kdr.source_dir", str(art.absolute()))
        # kdr.version written (any string value — version-agnostic).
        writer.add_kv.assert_any_call("kdr.version", ANY)
        # Two add_tensor calls: gate_proj → raw_dtype=IQ2_XS; norm → F16.
        calls = writer.add_tensor.call_args_list
        assert len(calls) == 2
        names_to_calls = {c.args[0]: c for c in calls}
        gate_call = names_to_calls["model.gate_proj.weight"]
        assert gate_call.kwargs.get("raw_dtype") == _FakeQuantType.IQ2_XS
        norm_call = names_to_calls["model.norm.weight"]
        # F16 carve-out: tensor cast via .astype(np.float16), no raw_dtype.
        assert norm_call.kwargs.get("raw_dtype") is None
        assert norm_call.args[1].dtype == np.float16


def test_convert_raises_on_orphan_tensor(tmp_path: Path) -> None:
    """LLR-0059 AC: tensor matching no group AND no ignore → RuntimeError naming it."""
    art = _write_artifact(
        tmp_path,
        config_groups={
            "group_0": {
                "weights": {"num_bits": 4, "strategy": "block", "type": "int", "kdr_format": "iq4_xs"},
                "input_activations": None,
                "targets": ["q_proj"],
            },
        },
        tensors={
            "model.q_proj.weight": np.zeros((4, 4), dtype=np.float32),
            "mystery_tensor": np.zeros((4,), dtype=np.float32),
        },
        ignore=[],
    )
    with patch.object(_kg.gguf, "GGUFWriter"), pytest.raises(
        RuntimeError, match="mystery_tensor"
    ):
        _kg.convert(art, tmp_path / "out.gguf")


def test_convert_dedupes_tied_embed_lm_head(tmp_path: Path) -> None:
    """LLR-0059 AC: tied embed/lm_head → exactly one token_embd.weight tensor."""
    art = _write_artifact(
        tmp_path,
        config_groups={
            "group_0": {
                "weights": {"num_bits": 5, "strategy": "block", "type": "int", "kdr_format": "q5_k"},
                "input_activations": None,
                "targets": ["embed_tokens", "lm_head"],
            },
        },
        tensors={
            "model.embed_tokens.weight": np.zeros((4, 4), dtype=np.float32),
            "lm_head.weight": np.zeros((4, 4), dtype=np.float32),
        },
        tie_word_embeddings=True,
    )
    with patch.object(_kg.gguf, "GGUFWriter") as mock_writer_cls:
        writer = mock_writer_cls.return_value
        _kg.convert(art, tmp_path / "out.gguf")
        # Exactly one add_tensor; named token_embd.weight.
        calls = writer.add_tensor.call_args_list
        assert len(calls) == 1
        assert calls[0].args[0] == "token_embd.weight"


def test_convert_tied_embed_with_only_lm_head_present(tmp_path: Path) -> None:
    """LLR-0059 AC: tied embed with ONLY lm_head.weight present → renamed, not dropped."""
    art = _write_artifact(
        tmp_path,
        config_groups={
            "group_0": {
                "weights": {"num_bits": 5, "strategy": "block", "type": "int", "kdr_format": "q5_k"},
                "input_activations": None,
                "targets": ["lm_head"],
            },
        },
        tensors={
            "lm_head.weight": np.zeros((4, 4), dtype=np.float32),
        },
        tie_word_embeddings=True,
    )
    with patch.object(_kg.gguf, "GGUFWriter") as mock_writer_cls:
        writer = mock_writer_cls.return_value
        _kg.convert(art, tmp_path / "out.gguf")
        calls = writer.add_tensor.call_args_list
        assert len(calls) == 1
        assert calls[0].args[0] == "token_embd.weight"


# ─────────────────────────────────────────────────────────────────────────────
# _guess_fmt
# ─────────────────────────────────────────────────────────────────────────────


def test_guess_fmt_maps_profile_j_bits_to_format() -> None:
    """LLR-0059 AC: _guess_fmt resolves the four Profile-J (bits, block) tuples."""
    cases = [
        (2, "iq2_xs"),
        (3, "q3_k"),
        (4, "iq4_xs"),
        (5, "q5_k"),
    ]
    for bits, expected in cases:
        group = {"weights": {"num_bits": bits, "strategy": "block"}}
        assert _kg._guess_fmt(group) == expected
    # Unsupported tuple surfaces a KeyError mentioning the Phase-7.2 wired format.
    with pytest.raises(KeyError, match=r"Phase-7\.2 wired GGUF format"):
        _kg._guess_fmt({"weights": {"num_bits": 6, "strategy": "block"}})


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def test_main_dispatches_to_convert_with_path_objects(tmp_path: Path) -> None:
    """LLR-0059 AC: main(['--kdr-dir', 'x', '--output', 'y.gguf']) → convert(Path('x'), Path('y.gguf'))."""
    with patch.object(_kg, "convert") as mock_convert:
        rc = _kg.main(["--kdr-dir", "x", "--output", "y.gguf"])
        assert rc == 0
        mock_convert.assert_called_once_with(Path("x"), Path("y.gguf"))


def test_main_returns_nonzero_on_convert_failure() -> None:
    """The CLI surfaces exceptions as non-zero exit + stderr message."""
    with patch.object(_kg, "convert", side_effect=RuntimeError("boom")):
        rc = _kg.main(["--kdr-dir", "x", "--output", "y.gguf"])
        assert rc == 1


# ─────────────────────────────────────────────────────────────────────────────
# gguf-not-installed path
# ─────────────────────────────────────────────────────────────────────────────


def test_module_import_fails_actionably_when_gguf_missing() -> None:
    """LLR-0059 AC: import without `gguf` raises ImportError mentioning `pip install gguf`."""
    saved_gguf = sys.modules.pop("gguf", None)
    saved_kg = sys.modules.pop("kdr.tools.kdr_to_gguf", None)
    # `sys.modules[name] = None` is Python's canonical "this import must
    # fail" sentinel — the import machinery sees None and raises
    # ModuleNotFoundError (an ImportError subclass) without touching
    # finders. Avoids meta_path recursion.
    sys.modules["gguf"] = None  # type: ignore[assignment]
    try:
        with pytest.raises(ImportError, match="pip install gguf"):
            importlib.import_module("kdr.tools.kdr_to_gguf")
    finally:
        if saved_gguf is not None:
            sys.modules["gguf"] = saved_gguf
        else:
            sys.modules.pop("gguf", None)
        if saved_kg is not None:
            sys.modules["kdr.tools.kdr_to_gguf"] = saved_kg
        else:
            _import_kdr_to_gguf()
