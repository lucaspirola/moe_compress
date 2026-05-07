"""Calibration dataset loader with per-subset weighted sampling.

Supports two sources (picked via ``CalibrationSpec.source``):

- ``nvidia-cascade`` — pulls from ``nvidia/Nemotron-Cascade-2-SFT-Data``, a
  multi-subset SFT dataset where subsets are domains (math, science, chat,
  instruction_following, conversational_agent, swe, terminal_agent, safety).
  Rows come as OpenAI-style ``messages=[{role, content}, ...]``; we render
  them via the tokenizer's chat template when available so calibration text
  matches the deployment format.

- ``c4-math-code`` — legacy split across allenai/c4, hendrycks/competition_math,
  and bigcode/the-stack-smol; kept for smoke-test compatibility. Uses the
  original ``domain_mix`` dict.

Tokenization happens once per pipeline invocation and is cached on disk by a
configuration-addressed key so re-runs and ``--resume-from-stage`` skip retokenizing.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
from dataclasses import dataclass, field, replace as _dc_replace
from pathlib import Path

import torch

log = logging.getLogger(__name__)


# Valid domain names for the legacy c4-math-code source.
_LEGACY_DOMAINS = frozenset({"c4", "math", "code"})

# circuit-breaker fires if rows_seen exceeds this multiple of the requested count
_CIRCUIT_BREAKER_MULTIPLIER = 10

# Public subset names the nvidia-cascade source accepts.  The accompanying
# floats are sanity defaults only — actual weights come from the config's
# ``subset_weights`` dict and do NOT need to sum to 1.0 here.
_CASCADE_SUBSETS = {
    "math": 0.21,
    "science": 0.11,
    "chat": 0.56,
    "instruction_following": 0.034,
    "conversational_agent": 0.033,
    "swe": 0.02,
    "terminal_agent": 0.033,
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
        # All fields (including inactive-source fields) are included intentionally:
        # any field change — even one not active for the current source — will
        # invalidate the cache.  This is deliberate: it avoids silent cache
        # collisions if a field is later added without bumping the hash.
        # bump _schema_version whenever tokenization logic changes
        payload = json.dumps({
            "_schema_version": 1,
            "num_sequences": self.num_sequences,
            "sequence_length": self.sequence_length,
            "seed": self.seed,
            "source": self.source,
            "dataset": self.dataset,
            "subset_weights": {k: float(v) for k, v in self.subset_weights.items()},
            "domain_mix": {k: float(v) for k, v in self.domain_mix.items()},
            "c4_dataset": self.c4_dataset,
            "c4_subset": self.c4_subset,
            "math_dataset": self.math_dataset,
            "code_dataset": self.code_dataset,
            # NOTE: JSON key is "tokenizer" (matches the historical on-disk format); renaming would invalidate existing caches.
            "tokenizer": tokenizer_name,
            # _CIRCUIT_BREAKER_MULTIPLIER is intentionally excluded: it is a
            # performance-tuning constant with no effect on output semantics, so
            # changing it should not invalidate the on-disk cache.
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_calibration_tensor(
    tokenizer,
    spec: CalibrationSpec,
    *,
    cache_dir: str | Path = "./artifacts/_calibration_cache",
) -> torch.Tensor:
    """Return a ``(num_sequences, sequence_length)`` int64 tensor of input ids."""
    if spec.num_sequences <= 0:
        raise ValueError(f"num_sequences must be > 0, got {spec.num_sequences}")
    if spec.sequence_length <= 0:
        raise ValueError(f"sequence_length must be positive, got {spec.sequence_length}")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Use a class-qualified fallback so tokenizers that lack name_or_path
    # (e.g. anonymous or in-memory instances) don't all collapse to the same
    # cache key "unknown" and incorrectly share cached tensors.
    tok_name = (
        getattr(tokenizer, "name_or_path", None)
        or f"{tokenizer.__class__.__module__}.{tokenizer.__class__.__name__}"
    )
    key = spec.cache_key(tok_name)
    cache_file = cache_dir / f"calib_{key}.pt"
    if cache_file.exists():
        log.info("Loading cached calibration tensor from %s", cache_file)
        return torch.load(cache_file, map_location="cpu", weights_only=True)

    log.info(
        "Building calibration tensor: %d x %d tokens, source=%s",
        spec.num_sequences, spec.sequence_length, spec.source,
    )
    texts: list[str] = []
    if spec.source == "nvidia-cascade":
        if not spec.subset_weights:
            raise ValueError(
                "CalibrationSpec.subset_weights must be non-empty for source='nvidia-cascade'"
            )
        for k in spec.subset_weights:
            if k not in _CASCADE_SUBSETS:
                raise ValueError(
                    f"Unknown cascade subset: {k!r}. Valid: {list(_CASCADE_SUBSETS)}"
                )
        per = _distribute_counts(spec.num_sequences, spec.subset_weights)
        log.info("  per-subset sequence counts: %s", per)
        for subset, count in per.items():
            if count <= 0:
                continue
            # Per-subset seed offset so subsets draw from independent shuffles
            # even when the same base seed is reused across stages.
            subset_offset = int(hashlib.md5(subset.encode(), usedforsecurity=False).hexdigest(), 16) % 1_000_000
            subset_seed = (spec.seed + subset_offset) % (2**32)
            texts.extend(_stream_cascade_texts(
                spec.dataset, subset, count, tokenizer, seed=subset_seed,
            ))
    elif spec.source == "c4-math-code":
        if not spec.domain_mix:
            raise ValueError(
                "CalibrationSpec.domain_mix must be non-empty for source='c4-math-code'"
            )
        unknown = set(spec.domain_mix) - _LEGACY_DOMAINS
        if unknown:
            raise ValueError(
                f"Unknown legacy calibration domains: {sorted(unknown)}. "
                f"Valid: {sorted(_LEGACY_DOMAINS)}"
            )
        per = _distribute_counts(spec.num_sequences, spec.domain_mix)
        for domain, count in per.items():
            if count <= 0:
                continue
            # Per-domain seed offset so domains draw from independent shuffles
            # (mirrors the per-subset seed offset used for nvidia-cascade).
            domain_offset = int(hashlib.md5(domain.encode(), usedforsecurity=False).hexdigest(), 16) % 1_000_000
            domain_seed = (spec.seed + domain_offset) % (2**32)
            texts.extend(_stream_legacy_texts(domain, count, spec, seed=domain_seed))
    else:
        # All recognised sources are handled above; anything else is a config error.
        raise ValueError(f"Unknown calibration source: {spec.source}")

    if not texts:
        raise ValueError(
            f"build_calibration_tensor: no texts collected for spec (seed={spec.seed}, "
            f"num_sequences={spec.num_sequences}). Check dataset availability and subset weights."
        )

    # Globally shuffle collected texts so rows from different subsets/domains are
    # interleaved, preventing the tokenizer from seeing one domain at a time.
    # NOTE: this reuses spec.seed directly (no offset), so changing spec.seed
    # affects both per-subset sampling seeds AND this global shuffle order.
    # Per-subset seeds use an md5-derived offset; see seed computation above.
    rng = torch.Generator().manual_seed(spec.seed)
    idx = torch.randperm(len(texts), generator=rng).tolist()
    texts = [texts[i] for i in idx]

    if len(texts) < spec.num_sequences:
        log.warning(
            "Collected %d texts but spec.num_sequences=%d; "
            "calibration tensor may be padded with EOS.",
            len(texts), spec.num_sequences,
        )
    input_ids = _tokenize_to_fixed_length(
        tokenizer, texts, spec.sequence_length, spec.num_sequences,
    )
    tmp_file = cache_file.with_suffix(".tmp")
    torch.save(input_ids, tmp_file)
    os.replace(tmp_file, cache_file)
    log.info("Cached calibration tensor: %s (shape=%s)",
             cache_file, tuple(input_ids.shape))
    return input_ids


def build_super_expert_slice(
    tokenizer,
    spec: CalibrationSpec,
    num_samples: int,
    *,
    cache_dir: str | Path = "./artifacts/_calibration_cache",
) -> torch.Tensor:
    """Small slice for Stage 0. Uses the same weight distribution as the full
    set but an offset seed (``(spec.seed + 1) % (2**32)``) to draw a largely
    non-overlapping sample, so super-expert detection sees representative routing
    at lower volume without reusing the same rows as the main calibration tensor.

    Note on seed offsets: this function applies an internal ``+1`` offset to
    avoid overlap with the full-tensor draw — this is distinct from
    ``spec_from_config``'s ``seed_offset`` parameter, which adjusts the base
    seed at the stage level before the spec reaches this function. The two
    offsets serve different purposes and are applied independently.
    """
    if num_samples <= 0:
        raise ValueError(f"num_samples must be > 0, got {num_samples}")
    if num_samples > spec.num_sequences:
        raise ValueError(
            f"build_super_expert_slice: num_samples={num_samples} exceeds "
            f"spec.num_sequences={spec.num_sequences}"
        )
    # validation delegated to build_calibration_tensor.
    small_spec = _dc_replace(
        spec,
        num_sequences=num_samples,
        seed=(spec.seed + 1) % (2**32),
    )
    return build_calibration_tensor(tokenizer, small_spec, cache_dir=cache_dir)


def iter_batches(
    calib_ids: torch.Tensor, batch_size: int,
) -> list[torch.Tensor]:
    """Return a fully-materialised list of tensor slices.

    Despite the ``iter_`` prefix, all slices are computed eagerly and returned
    as a plain list.

    Warning: all slices share the underlying tensor storage — in-place
    modifications to any batch will corrupt all batches and the source tensor.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
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
      - legacy (source omitted or explicitly ``'c4-math-code'``): reads ``domain_mix`` + c4/math/code datasets

    ``seed_offset`` lets callers derive disjoint sample draws per stage from
    the same base seed (Stage 3 B-cov uses +2, Stage 5 uses +5 — see
    individual stage code).  Note: ``build_super_expert_slice`` applies its
    own internal ``+1`` offset independently; Stage 0 callers typically pass
    ``seed_offset=0`` to ``spec_from_config`` and rely on that internal offset.
    """
    source = cal_cfg.get("source", "nvidia-cascade")
    seed = (int(cal_cfg.get("seed", 0)) + seed_offset) % (2**32)
    for required_key in ("num_sequences", "sequence_length"):
        if required_key not in cal_cfg:
            raise KeyError(f"calibration config is missing required key {required_key!r}")
    num_sequences = int(num_sequences_override if num_sequences_override is not None
                        else cal_cfg["num_sequences"])
    sequence_length = int(sequence_length_override if sequence_length_override is not None
                          else cal_cfg["sequence_length"])
    if num_sequences <= 0:
        raise ValueError(f"calibration config: num_sequences must be > 0, got {num_sequences}")
    if sequence_length <= 0:
        raise ValueError(f"calibration config: sequence_length must be > 0, got {sequence_length}")
    if source == "nvidia-cascade":
        # "dataset" and "subset_weights" are required keys for this source;
        # raise a descriptive error rather than a bare KeyError if absent.
        try:
            dataset = str(cal_cfg["dataset"]).strip()
            subset_weights = dict(cal_cfg["subset_weights"])
        except KeyError as exc:
            raise KeyError(
                f"calibration config is missing required key {exc} for source='nvidia-cascade'"
            ) from exc
        unknown = set(subset_weights) - _CASCADE_SUBSETS.keys()
        if unknown:
            raise ValueError(
                f"spec_from_config: unknown subset_weights keys {sorted(unknown)!r}; "
                f"valid keys are {sorted(_CASCADE_SUBSETS)!r}"
            )
        if any(w < 0 for w in subset_weights.values()):
            raise ValueError(f"subset_weights contains negative values: {subset_weights}")
        if not any(w > 0 for w in subset_weights.values()):
            raise ValueError(f"subset_weights are all zero — at least one weight must be positive: {subset_weights}")
        return CalibrationSpec(
            num_sequences=num_sequences,
            sequence_length=sequence_length,
            seed=seed,
            source=source,
            dataset=dataset,
            subset_weights=subset_weights,
        )
    # Legacy schema (kept so synthetic-MoE tests work unchanged).
    if source != "c4-math-code":
        # Preserve the original source value so the cache key reflects the actual
        # config.  Two configs that differ only by their `source` field must
        # produce different cache keys; normalizing unknown sources to a canonical
        # string defeats that invariant.
        raise ValueError(
            f"spec_from_config: unrecognized source {source!r}. "
            f"Valid sources are 'nvidia-cascade' and 'c4-math-code'."
        )
    if "domain_mix" not in cal_cfg:
        raise KeyError("calibration config is missing required key 'domain_mix'")
    # Warn if cascade-specific keys are present but ignored under this source.
    ignored = {k for k in ("subset_weights", "dataset") if k in cal_cfg}
    if ignored:
        log.warning(
            "spec_from_config: source=%r — the following keys are ignored for "
            "the legacy c4-math-code source: %s",
            source, sorted(ignored),
        )
    # Always use the hardcoded default for c4_dataset here — the cascade
    # "dataset" key is genuinely ignored in the legacy path (as the warning
    # above states).  Legacy callers should not repurpose the cascade "dataset"
    # key; use the dedicated "c4_dataset" config key instead.
    domain_mix_raw = cal_cfg["domain_mix"]
    if not isinstance(domain_mix_raw, dict):
        raise ValueError(
            f"calibration config 'domain_mix' must be a dict mapping subset names to weights, "
            f"got {type(domain_mix_raw).__name__!r}"
        )
    if not domain_mix_raw:
        raise ValueError("calibration config: 'domain_mix' must be non-empty")
    domain_mix = dict(domain_mix_raw)
    if any(w < 0 for w in domain_mix.values()):
        raise ValueError(f"domain_mix contains negative values: {domain_mix}")
    if not any(w > 0 for w in domain_mix.values()):
        raise ValueError(f"domain_mix are all zero — at least one weight must be positive (source={source!r}): {domain_mix}")
    return CalibrationSpec(
        num_sequences=num_sequences,
        sequence_length=sequence_length,
        seed=seed,
        source=source,
        domain_mix=domain_mix,
        c4_dataset=cal_cfg.get("c4_dataset", "allenai/c4"),
        c4_subset=cal_cfg.get("c4_subset", "en"),
        math_dataset=cal_cfg.get("math_dataset", "hendrycks/competition_math"),
        code_dataset=cal_cfg.get("code_dataset", "bigcode/the-stack-smol"),
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _distribute_counts(total: int, weights: dict[str, float]) -> dict[str, int]:
    """Distribute *total* integer counts across keys using the largest-remainder
    algorithm: floor-allocate proportional shares, then assign leftover units
    to keys with the largest fractional remainders.

    Handles the rare case where floating-point accumulation causes a negative
    remainder (overshoot): subtracts 1 from keys with the *smallest* fractional
    remainders — those where float error pushed the raw value just past an integer
    boundary, causing floor() to produce a value one too high.
    """
    if any(v < 0 for v in weights.values()):
        raise ValueError(f"All weights must be non-negative, got: {weights}")
    weight_sum = sum(weights.values())
    if weight_sum <= 0:
        raise ValueError("all subset weights are zero — cannot distribute sequences")
    raw = {k: (v / weight_sum) * total for k, v in weights.items()}
    out = {k: math.floor(v) for k, v in raw.items()}
    remainder = total - sum(out.values())
    # remainder can be negative if float arithmetic causes sum(raw) to slightly
    # exceed total; handle both directions so sum(out.values()) == total exactly.
    if abs(remainder) >= len(weights):
        raise ValueError(
            f"_distribute_counts: remainder {remainder} exceeds key count "
            f"{len(weights)} — float arithmetic error is larger than expected"
        )
    if remainder != 0:
        if remainder > 0:
            # Positive remainder: add 1 to keys with the largest fractional parts.
            fracs = sorted(((raw[k] - out[k], k) for k in raw), reverse=True)
            for i in range(remainder):
                out[fracs[i][1]] += 1
        else:
            # Negative remainder: subtract 1 from keys with the *smallest* fractional
            # parts — these are the keys where floating-point accumulation pushed the raw
            # value just past an integer boundary (frac ≈ 0), making floor() produce a
            # value 1 higher than it should.  Decrementing those keys brings the sum back
            # to `total` while preserving the `>= 0` invariant.
            eligible = sorted(
                ((raw[k] - out[k], k) for k in raw if out[k] > 0),
                reverse=False,
            )
            if len(eligible) < abs(remainder):
                raise ValueError(
                    f"_distribute_counts invariant broken: only {len(eligible)} eligible "
                    f"keys to decrement but need {abs(remainder)}. out={out}"
                )
            for i in range(abs(remainder)):
                out[eligible[i][1]] -= 1
    if not all(v >= 0 for v in out.values()):
        raise ValueError(f"_distribute_counts produced negative counts: {out}")
    if sum(out.values()) != total:
        raise ValueError(
            f"_distribute_counts total mismatch: got {sum(out.values())}, expected {total}"
        )
    return out


def _stream_cascade_texts(
    dataset_name: str, subset: str, count: int, tokenizer,
    *, seed: int = 0,
) -> list[str]:
    """Stream `count` non-empty rows from a cascade subset.

    Uses a seeded streaming shuffle so independent calls with different seeds
    produce largely non-overlapping row sets (important because Stage 0's
    super-expert slice uses ``seed+1`` vs Stage 2/3/5's ``seed``; without
    shuffling, both would take the first N rows of the subset and overlap
    entirely).
    """
    if count <= 0:
        return []

    from datasets import load_dataset

    if subset not in _CASCADE_SUBSETS:
        raise ValueError(f"Unexpected subset {subset!r}; internal error: upstream build_calibration_tensor should have caught this. Valid: {sorted(_CASCADE_SUBSETS)}")
    log.info("Streaming %d %s samples from %s (seed=%d)", count, subset, dataset_name, seed)
    try:
        ds = load_dataset(dataset_name, name=subset, split="train", streaming=True)
    except Exception as err:                          # noqa: BLE001 — broad catch intentional: re-raise with context after logging
        log.error("load_dataset(%s, name=%s) failed: %s", dataset_name, subset, err)
        raise

    # buffer_size >> count so the shuffle has meaningful randomness even for
    # streaming datasets whose row count is large; capped to avoid OOM on
    # large datasets with small count values.
    circuit_limit = _CIRCUIT_BREAKER_MULTIPLIER * count
    ds = ds.shuffle(seed=seed, buffer_size=min(max(10_000, circuit_limit), 200_000))

    out: list[str] = []
    rows_seen = 0
    for row in ds:
        rows_seen += 1
        text = _render_messages(row.get("messages"), tokenizer)
        if text:
            out.append(text)
            if len(out) >= count:
                break
        if rows_seen >= circuit_limit:
            log.warning(
                "_stream_cascade_texts: circuit-breaker fired after %d rows examined for content "
                "in subset=%r (circuit_limit=%d); dataset may be smaller than expected",
                rows_seen, subset, circuit_limit,
            )
            break
    if len(out) < count:
        log.warning("Subset %s only produced %d/%d non-empty rows", subset, len(out), count)
    return out


def _render_messages(messages, tokenizer) -> str | None:
    """Render a messages list to a string: try apply_chat_template, fall back to
    role/content concat, and return None on failure so the caller's empty-row
    filter can discard the row."""
    if not messages:
        return None
    try:
        # Strip here so both paths return consistently trimmed text (the fallback
        # path already strips; keeping both uniform avoids caller-side compensation).
        # Return None (not "") on empty render so both paths share the same sentinel.
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        ).strip()
        return rendered or None
    except Exception as tmpl_exc:                      # noqa: BLE001 — fall back to plain concat
        log.debug("apply_chat_template failed (%s); falling back to plain role/content concat", tmpl_exc)
        try:
            parts = []
            for m in messages:
                role = m.get("role", "")
                content = m.get("content")
                if content is None:
                    content = ""
                elif not isinstance(content, (str, list)):
                    content = str(content)
                if isinstance(content, list):
                    filtered = [c for c in content if isinstance(c, dict)]
                    if not filtered and content:
                        log.debug(
                            "_render_messages: content list had %d item(s) but none were dicts "
                            "(types: %s); text will be empty for this message",
                            len(content), [type(c).__name__ for c in content],
                        )
                    content = " ".join(c.get("text") or "" for c in filtered)
                parts.append(f"{role.upper()}: {content}")
            text = "\n".join(parts)
            # Strip and reject if only role prefixes remain (e.g. "USER:\nASSISTANT:").
            # Tightened from endswith(":") to an uppercase-word-colon pattern so that
            # content lines legitimately ending with ":" (e.g. "The answer is:") are
            # not discarded.
            stripped = text.strip()
            if not stripped or all(
                bool(re.match(r'^[A-Z_]+:\s*$', part.strip()))
                for part in stripped.splitlines() if part.strip()
            ):
                return None
            return stripped
        except (AttributeError, TypeError, KeyError, ValueError) as fmt_exc:
            # Use next(iter(...)) instead of messages[0] to handle non-subscriptable
            # iterables — messages[0] inside an except block would raise an uncaught
            # TypeError if messages is e.g. a generator.
            first_elem = next(iter(messages), None)
            first_type = type(first_elem).__name__
            log.warning(
                "_render_messages: unexpected message format "
                "(messages type=%s, first element type=%s: %s) — skipping row",
                type(messages).__name__, first_type, fmt_exc,
            )
            return None  # let the caller's empty-row filter discard it


def _stream_legacy_texts(
    domain: str, count: int, spec: CalibrationSpec, *, seed: int = 0,
) -> list[str]:
    if count <= 0:
        return []

    from datasets import load_dataset

    log.info("Streaming %d %s samples (legacy source, seed=%d)", count, domain, seed)
    if domain == "c4":
        try:
            ds = load_dataset(spec.c4_dataset, spec.c4_subset, split="train", streaming=True)
        except Exception as err:                      # noqa: BLE001 — broad catch intentional: re-raise with context after logging
            log.error("load_dataset(%s, %s) failed for domain=%s: %s", spec.c4_dataset, spec.c4_subset, domain, err)
            raise
        key = "text"
    elif domain == "math":
        try:
            ds = load_dataset(spec.math_dataset, split="train", streaming=True)
        except Exception as err:                      # noqa: BLE001 — broad catch intentional: re-raise with context after logging
            log.error("load_dataset(%s) failed for domain=%s: %s", spec.math_dataset, domain, err)
            raise
        key = "problem"
    elif domain == "code":
        try:
            ds = load_dataset(spec.code_dataset, split="train", streaming=True)
        except Exception as err:                      # noqa: BLE001 — broad catch intentional: re-raise with context after logging
            log.error("load_dataset(%s) failed for domain=%s: %s", spec.code_dataset, domain, err)
            raise
        key = "content"
    else:
        raise ValueError(f"Unknown legacy calibration domain: {domain}")

    # Shuffle so repeated or concurrent calls with different seeds draw disjoint
    # rows rather than always taking the first N rows of the dataset; capped to
    # avoid OOM on large datasets with small count values.
    circuit_limit = _CIRCUIT_BREAKER_MULTIPLIER * count
    ds = ds.shuffle(seed=seed, buffer_size=min(max(10_000, circuit_limit), 200_000))

    out: list[str] = []
    rows_seen = 0
    for row in ds:
        rows_seen += 1
        txt = row.get(key)
        if isinstance(txt, str) and txt.strip():
            out.append(txt.strip())
            if len(out) >= count:
                break
        if rows_seen >= circuit_limit:
            log.warning(
                "_stream_legacy_texts: circuit-breaker fired after %d rows examined for content "
                "in domain=%r (circuit_limit=%d); dataset may be smaller than expected",
                rows_seen, domain, circuit_limit,
            )
            break
    if len(out) < count:
        log.warning(
            "Domain %s only produced %d/%d non-empty rows", domain, len(out), count
        )
    return out


def _tokenize_to_fixed_length(
    tokenizer, texts: list[str], seq_len: int, num_sequences: int,
) -> torch.Tensor:
    if not texts:
        raise ValueError(
            "_tokenize_to_fixed_length: texts is empty — no calibration data was collected"
        )
    all_ids: list[int] = []
    need = num_sequences * seq_len
    eos = tokenizer.eos_token_id
    if eos is None:
        pad = getattr(tokenizer, "pad_token_id", None)
        # Use explicit `is not None` — pad_token_id=0 is falsy but valid, so
        # `or 0` would incorrectly discard it and always fall back to 0.
        eos = pad if pad is not None else 0
        log.warning("Tokenizer has no eos_token_id; using %d as separator", eos)
    for t in texts:
        # truncation=False is intentional: we concatenate all token streams into
        # a flat list and then slice to exactly num_sequences * seq_len, so
        # per-text truncation would silently discard tokens we need.
        ids = tokenizer(t, add_special_tokens=False, truncation=False)["input_ids"]
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if isinstance(ids, int):
            ids = [ids]
        # Flatten arbitrary-depth nesting — some tokenizers return [[...]] or deeper.
        while ids and isinstance(ids[0], list):
            ids = [tok for sub in ids for tok in sub]
        all_ids.extend(ids)
        all_ids.append(eos)
        if len(all_ids) >= need:
            break
    if len(all_ids) < need:
        shortfall = 1.0 - len(all_ids) / need
        log.warning(
            "_tokenize_to_fixed_length: token pool short by %d tokens (%.2f%% shortfall); "
            "padding with EOS. "
            "Consider reducing num_sequences or seq_len, or increasing dataset coverage "
            "(adding more subsets/domains or raising their weights).",
            need - len(all_ids),
            shortfall * 100,
        )
        all_ids.extend([eos] * (need - len(all_ids)))
    else:
        all_ids = all_ids[:need]
    return torch.tensor(all_ids, dtype=torch.long).view(num_sequences, seq_len)
