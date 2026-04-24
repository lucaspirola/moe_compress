"""Calibration dataset loader.

Produces a batch of tokenized sequences shared across Stages 0, 2, 3, 4, and 5.
Tokenization happens once per pipeline invocation and the result is memoized on
disk by a content-addressed key so re-runs (and `--resume-from-stage`) read from
cache instead of re-shuffling.

Domain mix per spec (Strategy A §Stage 2): 0:0.5:0.5 C4 : Math : Code for
generative tasks. Stage 0 uses a 100-sample C4 slice only.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import torch

log = logging.getLogger(__name__)


@dataclass
class CalibrationSpec:
    num_sequences: int
    sequence_length: int
    seed: int
    domain_mix: dict                # {"c4": 0.0, "math": 0.5, "code": 0.5}
    c4_dataset: str = "allenai/c4"
    c4_subset: str = "en"
    math_dataset: str = "hendrycks/competition_math"
    code_dataset: str = "bigcode/the-stack-smol"

    def cache_key(self, tokenizer_name: str) -> str:
        payload = f"{self.num_sequences}|{self.sequence_length}|{self.seed}|{self.domain_mix}|{tokenizer_name}"
        return hashlib.sha1(payload.encode()).hexdigest()[:16]


def build_calibration_tensor(
    tokenizer,
    spec: CalibrationSpec,
    *,
    cache_dir: str | Path = "./artifacts/_calibration_cache",
) -> torch.LongTensor:
    """Return a ``(num_sequences, sequence_length)`` LongTensor of input ids."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = spec.cache_key(tokenizer.name_or_path)
    cache_file = cache_dir / f"calib_{key}.pt"
    if cache_file.exists():
        log.info("Loading cached calibration tensor from %s", cache_file)
        return torch.load(cache_file, map_location="cpu")

    log.info(
        "Building calibration tensor: %d × %d tokens, mix=%s",
        spec.num_sequences,
        spec.sequence_length,
        spec.domain_mix,
    )
    per_domain = _distribute_counts(spec.num_sequences, spec.domain_mix)
    texts: list[str] = []
    for domain, count in per_domain.items():
        if count <= 0:
            continue
        texts.extend(_stream_texts(domain, count, spec))
    # Shuffle deterministically
    rng = torch.Generator().manual_seed(spec.seed)
    idx = torch.randperm(len(texts), generator=rng).tolist()
    texts = [texts[i] for i in idx]

    input_ids = _tokenize_to_fixed_length(tokenizer, texts, spec.sequence_length, spec.num_sequences)
    torch.save(input_ids, cache_file)
    log.info("Cached calibration tensor: %s (shape=%s)", cache_file, tuple(input_ids.shape))
    return input_ids


def build_super_expert_slice(
    tokenizer,
    spec: CalibrationSpec,
    num_samples: int,
    *,
    cache_dir: str | Path = "./artifacts/_calibration_cache",
) -> torch.LongTensor:
    """C4-only slice for Stage 0 super-expert detection."""
    small_spec = CalibrationSpec(
        num_sequences=num_samples,
        sequence_length=spec.sequence_length,
        seed=spec.seed + 1,          # different seed so it doesn't overlap with full set
        domain_mix={"c4": 1.0, "math": 0.0, "code": 0.0},
        c4_dataset=spec.c4_dataset,
        c4_subset=spec.c4_subset,
    )
    return build_calibration_tensor(tokenizer, small_spec, cache_dir=cache_dir)


def _distribute_counts(total: int, mix: dict[str, float]) -> dict[str, int]:
    """Distribute `total` integers according to float ratios; sum stays ==total."""
    s = sum(mix.values())
    if s <= 0:
        return {k: 0 for k in mix}
    raw = {k: (v / s) * total for k, v in mix.items()}
    out = {k: int(v) for k, v in raw.items()}
    remainder = total - sum(out.values())
    # Assign leftover by largest fractional parts
    fracs = sorted(((raw[k] - out[k], k) for k in raw), reverse=True)
    for i in range(remainder):
        out[fracs[i % len(fracs)][1]] += 1
    return out


def _stream_texts(domain: str, count: int, spec: CalibrationSpec) -> list[str]:
    from datasets import load_dataset

    log.info("Streaming %d %s samples", count, domain)
    if domain == "c4":
        ds = load_dataset(spec.c4_dataset, spec.c4_subset, split="train", streaming=True)
        key = "text"
    elif domain == "math":
        ds = load_dataset(spec.math_dataset, split="train", streaming=True)
        key = "problem"
    elif domain == "code":
        ds = load_dataset(spec.code_dataset, split="train", streaming=True)
        key = "content"
    else:
        raise ValueError(f"Unknown calibration domain: {domain}")

    out: list[str] = []
    for row in ds:
        txt = row.get(key)
        if not txt:
            continue
        out.append(txt)
        if len(out) >= count:
            break
    return out


def _tokenize_to_fixed_length(
    tokenizer, texts: list[str], seq_len: int, num_sequences: int
) -> torch.LongTensor:
    """Tokenize, concat, and chunk into (num_sequences, seq_len) windows."""
    all_ids: list[int] = []
    eos = tokenizer.eos_token_id
    if eos is None:
        eos = 0
    for t in texts:
        ids = tokenizer(t, add_special_tokens=False, truncation=False)["input_ids"]
        all_ids.extend(ids)
        all_ids.append(eos)
        if len(all_ids) >= num_sequences * seq_len:
            break

    need = num_sequences * seq_len
    if len(all_ids) < need:
        # pad with eos
        all_ids.extend([eos] * (need - len(all_ids)))
    else:
        all_ids = all_ids[:need]
    tensor = torch.tensor(all_ids, dtype=torch.long).view(num_sequences, seq_len)
    return tensor


def iter_batches(
    calib_ids: torch.LongTensor, batch_size: int
) -> list[torch.LongTensor]:
    """Split the calibration tensor into simple batch chunks."""
    return [calib_ids[i : i + batch_size] for i in range(0, calib_ids.size(0), batch_size)]
