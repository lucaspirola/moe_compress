"""Convert a kdr-saved artifact directory into a llama.cpp GGUF file.

# REQ: LLR-0059

Usage::

    python -m kdr.tools.kdr_to_gguf \
        --kdr-dir <path>  \
        --output  <path.gguf>

The tool reads ``<kdr-dir>/config.json`` (specifically the
``quantization_config`` block emitted by ``save_kdr_artifact``, LLR-0056)
plus every ``*.safetensors`` file under the dir, and writes a single
GGUF file at ``--output`` using llama.cpp's official ``gguf`` Python
library.

Per-tensor encoding is driven by ``config.json::quantization_config.config_groups``:
each tensor's name is substring-matched against each group's ``targets``
list (first match wins), and the tensor is encoded with the matching
``GGMLQuantizationType``. Tensors that match no group but are listed
in ``quantization_config.ignore`` (FP32 carve-outs) are emitted as F16.
Any other tensor raises ``RuntimeError`` — no silent F16 fallback per
locked design choice #5.

This tool is a thin Profile-J convenience layer; it does not attempt
to be a general HF→GGUF converter. For non-Profile-J artifacts or
unrecognised architectures, llama.cpp's ``convert_hf_to_gguf.py``
remains the canonical path.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

try:
    import gguf  # type: ignore[import-not-found]
except ImportError as e:  # pragma: no cover — exercised by the gguf-absent test
    raise ImportError(
        "kdr_to_gguf requires the gguf package; install it with "
        "`pip install gguf`"
    ) from e


_FORMAT_TO_GGUF_TYPE: dict[str, Any] = {
    "iq2_xs": gguf.GGMLQuantizationType.IQ2_XS,
    "q3_k": gguf.GGMLQuantizationType.Q3_K,
    "iq4_xs": gguf.GGMLQuantizationType.IQ4_XS,
    "q5_k": gguf.GGMLQuantizationType.Q5_K,
}


_TIED_EMBED_HF_NAMES = ("model.embed_tokens.weight", "lm_head.weight")
_TIED_EMBED_GGUF_NAME = "token_embd.weight"


def _kdr_version() -> str:
    try:
        from importlib.metadata import version

        return version("kdr")
    except Exception:
        return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _arch_id(architectures_entry: str) -> str:
    """Map HF architecture string to GGUF lowercase id.

    ``"ZayaForCausalLM"`` → ``"zaya"``.
    ``"LlamaForCausalLM"`` → ``"llama"``.
    Strips ``ForCausalLM`` / ``Model`` suffixes; lowercases the rest.
    """
    name = architectures_entry
    for suffix in ("ForCausalLM", "Model"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name.lower()


def _to_gguf_tensor_name(hf_name: str) -> str:
    """Pure name translation (no dedup; convert() owns the tied-embed skip)."""
    if hf_name in _TIED_EMBED_HF_NAMES:
        return _TIED_EMBED_GGUF_NAME
    return hf_name


def _match_group(
    tensor_name: str, config_groups: dict[str, Any]
) -> dict[str, Any] | None:
    """Return the first group whose `targets` list has a substring of `tensor_name`."""
    for group in config_groups.values():
        assert isinstance(group, dict)
        for pattern in group.get("targets", []):
            if pattern == "Linear":
                # Uniform-config catch-all: matches every nn.Linear tensor
                # (the safetensors file only contains weight tensors here).
                return group
            if pattern and pattern in tensor_name:
                return group
    return None


def _matches_ignore(tensor_name: str, ignore_patterns: list[str]) -> bool:
    """Substring match against the FP32 carve-out list."""
    return any(p and p in tensor_name for p in ignore_patterns)


def _iter_safetensors(kdr_dir: Path) -> Iterator[tuple[str, np.ndarray]]:
    """Yield (tensor_name, numpy_array) for every tensor in every shard."""
    from safetensors.numpy import load_file  # local import — heavy dep

    shards = sorted(kdr_dir.glob("*.safetensors"))
    if not shards:
        raise RuntimeError(
            f"kdr_to_gguf: no *.safetensors files found under {kdr_dir}. "
            f"Contents: {sorted(p.name for p in kdr_dir.iterdir())}"
        )
    for shard in shards:
        yield from load_file(str(shard)).items()


def _add_hparams(writer: Any, cfg: dict[str, Any]) -> None:
    """Emit the common HF hparams via the gguf library setters.

    Phase 7.2 only requires that ``general.architecture`` and
    ``kdr.source_dir`` are present (those are set elsewhere in
    ``convert``); the additional hparams here are best-effort
    compatibility for llama.cpp loaders that look them up.
    """
    name = cfg.get("_name_or_path") or cfg.get("model_type") or "kdr-recovered"
    writer.add_name(name)
    # The exact setter names vary across gguf versions; guard each call so
    # a missing setter on an older library version does not crash the run.
    for cfg_key, setter_name in (
        ("hidden_size", "add_embedding_length"),
        ("num_attention_heads", "add_head_count"),
        ("num_key_value_heads", "add_head_count_kv"),
        ("num_hidden_layers", "add_block_count"),
        ("intermediate_size", "add_feed_forward_length"),
        ("max_position_embeddings", "add_context_length"),
        ("vocab_size", "add_vocab_size"),
    ):
        value = cfg.get(cfg_key)
        if value is None:
            continue
        setter = getattr(writer, setter_name, None)
        if setter is not None:
            setter(value)


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────


def convert(kdr_dir: Path, output_path: Path) -> None:
    """Read `kdr_dir` (compressed-tensors artifact) → write `output_path` GGUF."""
    cfg_path = kdr_dir / "config.json"
    cfg = json.loads(cfg_path.read_text())
    qcfg = cfg["quantization_config"]
    config_groups = qcfg["config_groups"]
    ignore_patterns: list[str] = list(qcfg.get("ignore", []))
    tie_word_embeddings = bool(cfg.get("tie_word_embeddings", False))

    architectures = cfg.get("architectures") or ["Model"]
    arch = _arch_id(architectures[0])

    writer = gguf.GGUFWriter(str(output_path), arch)
    _add_hparams(writer, cfg)
    writer.add_kv("kdr.source_dir", str(kdr_dir.absolute()))
    writer.add_kv("kdr.version", _kdr_version())

    # Collect tensors first; the tied-embed skip needs to see both names
    # so it can prefer model.embed_tokens.weight over lm_head.weight
    # deterministically (independent of stream order).
    tensors: dict[str, np.ndarray] = {}
    for tensor_name, tensor in _iter_safetensors(kdr_dir):
        tensors[tensor_name] = tensor

    # If tied and both keys are present, prefer embed_tokens.weight and drop
    # lm_head.weight. If only lm_head.weight is present, leave it in place —
    # _to_gguf_tensor_name will rename it to token_embd.weight.
    if tie_word_embeddings and "model.embed_tokens.weight" in tensors:
        tensors.pop("lm_head.weight", None)

    for tensor_name, tensor in tensors.items():
        gguf_name = _to_gguf_tensor_name(tensor_name)
        group = _match_group(tensor_name, config_groups)
        if group is not None:
            group["weights"]["type"]
            # The compressed-tensors writer emits `weights.type` as the
            # high-level family ("int" / "float"); when the YAML's
            # original `format` string is preserved in the spec_map
            # (Profile-J), the more specific format is recoverable from
            # the targets/num_bits. For the Phase-7.2 path the format
            # name is the GGUF key directly; LLR-0056's _build maps
            # Profile-J `format` strings to the same compressed-tensors
            # type. Look up via a small fallback: prefer the explicit
            # format if present.
            fmt_for_gguf = group["weights"].get("kdr_format") or _guess_fmt(group)
            gguf_type = _FORMAT_TO_GGUF_TYPE[fmt_for_gguf]
            writer.add_tensor(gguf_name, tensor, raw_dtype=gguf_type)
        elif _matches_ignore(tensor_name, ignore_patterns):
            writer.add_tensor(gguf_name, tensor.astype(np.float16))
        else:
            raise RuntimeError(
                f"kdr_to_gguf: tensor {tensor_name!r} matched no "
                f"config_groups.targets pattern and is not in the ignore "
                f"list — refusing to silently fall back to F16."
            )

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def _guess_fmt(group: dict[str, Any]) -> str:
    """Recover the Profile-J format key from `(num_bits, type)`.

    `_build_quantization_config` (LLR-0056) writes `weights.type` as
    the compressed-tensors family ("int"/"float"); the original
    Profile-J format (e.g., "iq2_xs") is implied by the
    `(num_bits, granularity)` tuple. This helper inverts that mapping
    for the four wired formats only — anything else is a
    KeyError surfaced to the caller.
    """
    bits = group["weights"]["num_bits"]
    strat = group["weights"].get("strategy", "block")
    # The Profile-J table:
    #   2 bits + block → iq2_xs
    #   3 bits + block → q3_k
    #   4 bits + block → iq4_xs
    #   5 bits + block → q5_k
    mapping = {
        (2, "block"): "iq2_xs",
        (3, "block"): "q3_k",
        (4, "block"): "iq4_xs",
        (5, "block"): "q5_k",
    }
    try:
        return mapping[(bits, strat)]
    except KeyError as e:
        raise KeyError(
            f"kdr_to_gguf: cannot map (num_bits={bits}, strategy={strat!r}) "
            "to a Phase-7.2 wired GGUF format. Only the four Profile-J "
            "formats (IQ2_XS, Q3_K, IQ4_XS, Q5_K) are supported."
        ) from e


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kdr_to_gguf",
        description="Convert a kdr-saved Profile-J artifact into a GGUF file.",
    )
    parser.add_argument(
        "--kdr-dir",
        type=Path,
        required=True,
        help="Path to the kdr artifact directory (contains config.json + safetensors).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination .gguf file path.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        convert(args.kdr_dir, args.output)
    except Exception as e:
        print(f"kdr_to_gguf: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
