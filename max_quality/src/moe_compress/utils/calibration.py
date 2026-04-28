"""Calibration dataset loader with per-subset weighted sampling.

Supports two sources (picked via ``CalibrationSpec.source``):

- ``nvidia-cascade`` — pulls from ``nvidia/Nemotron-Cascade-2-SFT-Data``, a
  multi-subset SFT dataset where subsets are domains (math, science, chat,
  instruction_following, conversational_agent, swe, terminal_agent). Rows
  come as OpenAI-style ``messages=[{role, content}, ...]``; we render them
  via the tokenizer's chat template when available so calibration text
  matches the deployment format.

- ``c4-math-code`` — legacy split across allenai/c4, hendrycks/competition_math,
  and bigcode/the-stack-smol; kept for smoke-test compatibility. Uses the
  original ``domain_mix`` dict.

Tokenization happens once per pipeline invocation and is cached on disk by a
content-addressed key so re-runs and ``--resume-from-stage`` skip retokenizing.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch

log = logging.getLogger(__name__)


# Public subset names the nvidia-cascade source accepts. The accompanying
# float is a sanity default only; actual weights come from config.
_CASCADE_SUBSETS = {
    "math": 0.21,
    "science": 0.11,
    "chat": 0.56,
    "instruction_following": 0.033,
    "conversational_agent": 0.0331,
    "swe": 0.02,
    "terminal_agent": 0.0331,
    # Not listed in the default production mix but accepted if requested:
    "safety": 0.0,
}


@dataclass
class CalibrationSpec:
    """Parameters for building one calibration tensor."""
    num_sequences: int
    sequence_length: int
    seed: int
    # New-style: single dataset with subset weights (sums ~1.0).
    # e.g. ``{"math": 0.21, "science": 0.11, "chat": 0.56, ...}``.
    subset_weights: dict[str, float] = field(default_factory=dict)
    source: str = "nvidia-cascade"
    dataset: str = "nvidia/Nemotron-Cascade-2-SFT-Data"
    # Legacy multi-dataset fields (source = "c4-math-code").
    domain_mix: dict[str, float] = field(default_factory=dict)
    c4_dataset: str = "allenai/c4"
    c4_subset: str = "en"
    math_dataset: str = "hendrycks/competition_math"
    code_dataset: str = "bigcode/the-stack-smol"

    def cache_key(self, tokenizer_name: str) -> str:
        payload = json.dumps({
            "num_sequences": self.num_sequences,
            "sequence_length": self.sequence_length,
            "seed": self.seed,
            "source": self.source,
            "dataset": self.dataset,
            "subset_weights": self.subset_weights,
            "domain_mix": self.domain_mix,
            "c4_dataset": self.c4_dataset,
            "c4_subset": self.c4_subset,
            "math_dataset": self.math_dataset,
            "code_dataset": self.code_dataset,
            "tokenizer": tokenizer_name,
        }, sort_keys=True)
        return hashlib.sha1(payload.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_calibration_tensor(
    tokenizer,
    spec: CalibrationSpec,
    *,
    cache_dir: str | Path = "./artifacts/_calibration_cache",
) -> torch.LongTensor:
    """Return a ``(num_sequences, sequence_length)`` LongTensor of input ids."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = spec.cache_key(getattr(tokenizer, "name_or_path", "unknown"))
    cache_file = cache_dir / f"calib_{key}.pt"
    if cache_file.exists():
        log.info("Loading cached calibration tensor from %s", cache_file)
        return torch.load(cache_file, map_location="cpu")

    log.info(
        "Building calibration tensor: %d × %d tokens, source=%s",
        spec.num_sequences, spec.sequence_length, spec.source,
    )
    if spec.source == "nvidia-cascade":
        per = _distribute_counts(spec.num_sequences, spec.subset_weights)
        log.info("  per-subset sequence counts: %s", per)
        texts: list[str] = []
        for subset, count in per.items():
            if count <= 0:
                continue
            # Per-subset seed offset so subsets draw from independent shuffles
            # even when the same base seed is reused across stages.
            seed = spec.seed + int(hashlib.md5(subset.encode()).hexdigest(), 16) % 1_000_000
            texts.extend(_stream_cascade_texts(
                spec.dataset, subset, count, tokenizer, seed=seed,
            ))
    elif spec.source == "c4-math-code":
        per = _distribute_counts(spec.num_sequences, spec.domain_mix)
        texts = []
        for domain, count in per.items():
            if count <= 0:
                continue
            texts.extend(_stream_legacy_texts(domain, count, spec))
    else:
        raise ValueError(f"Unknown calibration source: {spec.source}")

    rng = torch.Generator().manual_seed(spec.seed)
    idx = torch.randperm(len(texts), generator=rng).tolist()
    texts = [texts[i] for i in idx]

    input_ids = _tokenize_to_fixed_length(
        tokenizer, texts, spec.sequence_length, spec.num_sequences,
    )
    torch.save(input_ids, cache_file)
    log.info("Cached calibration tensor: %s (shape=%s)",
             cache_file, tuple(input_ids.shape))
    return input_ids


def build_super_expert_slice(
    tokenizer,
    spec: CalibrationSpec,
    num_samples: int,
    *,
    cache_dir: str | Path = "./artifacts/_calibration_cache",
) -> torch.LongTensor:
    """Small slice for Stage 0. Uses the *same* distribution as the full set
    so super-expert detection sees representative routing, just at lower volume.
    """
    small_spec = CalibrationSpec(
        num_sequences=num_samples,
        sequence_length=spec.sequence_length,
        seed=spec.seed + 1,
        source=spec.source,
        dataset=spec.dataset,
        subset_weights=spec.subset_weights,
        domain_mix=spec.domain_mix,
        c4_dataset=spec.c4_dataset,
        c4_subset=spec.c4_subset,
        math_dataset=spec.math_dataset,
        code_dataset=spec.code_dataset,
    )
    return build_calibration_tensor(tokenizer, small_spec, cache_dir=cache_dir)


def iter_batches(
    calib_ids: torch.LongTensor, batch_size: int,
) -> list[torch.LongTensor]:
    return [calib_ids[i : i + batch_size] for i in range(0, calib_ids.size(0), batch_size)]


def spec_from_config(
    cal_cfg: dict,
    *,
    num_sequences_override: int | None = None,
    sequence_length_override: int | None = None,
    seed_offset: int = 0,
) -> CalibrationSpec:
    """Build a CalibrationSpec from the ``calibration:`` section of the YAML.

    Supports both schemas:
      - new (``source: nvidia-cascade``): reads ``dataset`` + ``subset_weights``
      - legacy (no ``source`` key): reads ``domain_mix`` + c4/math/code datasets

    ``seed_offset`` lets callers derive disjoint sample draws per stage from
    the same base seed (Stage 0 uses +1 via the super_expert slice, Stage 3
    B-cov uses +2, Stage 5 uses +5 — see individual stage code).
    """
    source = cal_cfg.get("source", "c4-math-code")
    seed = int(cal_cfg.get("seed", 0)) + seed_offset
    num_sequences = int(num_sequences_override if num_sequences_override is not None
                        else cal_cfg["num_sequences"])
    sequence_length = int(sequence_length_override if sequence_length_override is not None
                          else cal_cfg["sequence_length"])
    if source == "nvidia-cascade":
        return CalibrationSpec(
            num_sequences=num_sequences,
            sequence_length=sequence_length,
            seed=seed,
            source=source,
            dataset=cal_cfg["dataset"],
            subset_weights=dict(cal_cfg["subset_weights"]),
        )
    # Legacy schema (kept so synthetic-MoE tests work unchanged).
    return CalibrationSpec(
        num_sequences=num_sequences,
        sequence_length=sequence_length,
        seed=seed,
        source="c4-math-code",
        domain_mix=dict(cal_cfg.get("domain_mix", {})),
        c4_dataset=cal_cfg.get("dataset", "allenai/c4"),
        c4_subset=cal_cfg.get("subset", "en"),
        math_dataset=cal_cfg.get("math_dataset", "hendrycks/competition_math"),
        code_dataset=cal_cfg.get("code_dataset", "bigcode/the-stack-smol"),
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _distribute_counts(total: int, weights: dict[str, float]) -> dict[str, int]:
    s = sum(weights.values())
    if s <= 0:
        return {k: 0 for k in weights}
    raw = {k: (v / s) * total for k, v in weights.items()}
    out = {k: int(v) for k, v in raw.items()}
    remainder = total - sum(out.values())
    fracs = sorted(((raw[k] - out[k], k) for k in raw), reverse=True)
    for i in range(remainder):
        out[fracs[i % len(fracs)][1]] += 1
    return out


def _stream_cascade_texts(
    dataset_name: str, subset: str, count: int, tokenizer,
    *, seed: int = 0,
) -> list[str]:
    """Stream `count` non-empty rows from a cascade subset.

    Uses a seeded streaming shuffle so independent calls with different seeds
    produce disjoint row sets (important because Stage 0's super-expert slice
    uses ``seed+1`` vs Stage 2/3/5's ``seed``; without shuffling, both would
    take the first N rows of the subset and overlap entirely).
    """
    from datasets import load_dataset

    if subset not in _CASCADE_SUBSETS:
        log.warning("Unknown cascade subset '%s' — passing through verbatim", subset)
    log.info("Streaming %d %s samples from %s (seed=%d)", count, subset, dataset_name, seed)
    try:
        ds = load_dataset(dataset_name, name=subset, split="train", streaming=True)
    except Exception as err:                          # noqa: BLE001
        log.error("load_dataset(%s, name=%s) failed: %s", dataset_name, subset, err)
        raise

    # buffer_size >> count so the shuffle has meaningful randomness even for
    # streaming datasets whose row count is large.
    ds = ds.shuffle(seed=seed, buffer_size=max(10000, count * 10))

    out: list[str] = []
    for row in ds:
        text = _render_messages(row.get("messages"), tokenizer)
        if not text or not text.strip():
            continue
        out.append(text)
        if len(out) >= count:
            break
    if len(out) < count:
        log.warning("Subset %s only produced %d/%d non-empty rows", subset, len(out), count)
    return out


def _render_messages(messages, tokenizer) -> str | None:
    if not messages:
        return None
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
    except Exception:                                 # noqa: BLE001 — fall back to plain concat
        parts = []
        for m in messages:
            role = m.get("role", "")
            content = m.get("content", "")
            if isinstance(content, list):
                content = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
            parts.append(f"<|{role}|>\n{content}")
        return "\n".join(parts)


def _stream_legacy_texts(domain: str, count: int, spec: CalibrationSpec) -> list[str]:
    from datasets import load_dataset

    log.info("Streaming %d %s samples (legacy source)", count, domain)
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
        raise ValueError(f"Unknown legacy calibration domain: {domain}")

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
    tokenizer, texts: list[str], seq_len: int, num_sequences: int,
) -> torch.LongTensor:
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
        all_ids.extend([eos] * (need - len(all_ids)))
    else:
        all_ids = all_ids[:need]
    return torch.tensor(all_ids, dtype=torch.long).view(num_sequences, seq_len)
