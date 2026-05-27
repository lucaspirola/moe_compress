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
from typing import Callable

import torch

log = logging.getLogger(__name__)


# One-shot guard so the enable_thinking fallback warning (L1) fires at most
# once per process. Callers depending on ``<think>...</think>`` preservation
# (e.g. self-traces) get a heads-up when the tokenizer silently drops the
# kwarg; subsequent rows stay quiet to avoid log spam.
_enable_thinking_unsupported_warned = False

# N-C — one-shot guard for the qwen3-pretrain-mix intro log. Without it the
# "broad-instruct-mix (legacy name retained)" banner repeats once per
# build_calibration_tensor call (~6× per pipeline run); the message is
# orientation, not progress, so cap it at one emission per process.
_broad_instruct_mix_intro_logged: bool = False


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
# Corpus registry — pluggable faucet
# ---------------------------------------------------------------------------
#
# To add a new calibration corpus, define two functions and one adapter:
#
#     def _parse_yaml_<name>(cal_cfg, num_sequences, sequence_length, seed) -> CalibrationSpec:
#         ...validate/extract source-specific yaml fields...
#
#     def _stream_texts_<name>(spec, tokenizer) -> list[str]:
#         ...iterate the dataset, return rendered text rows...
#
#     register_corpus(CorpusAdapter(
#         name="<name>", parse_yaml=_parse_yaml_<name>, stream_texts=_stream_texts_<name>,
#     ))
#
# Everything downstream (build_calibration_tensor, spec_from_config) dispatches
# by name through the registry. The cache key already includes the source
# string, so swapping corpora correctly invalidates cached calibration tensors.


@dataclass(frozen=True)
class CorpusAdapter:
    """Source-specific behavior for one calibration corpus.

    - ``parse_yaml`` builds a :class:`CalibrationSpec` from the ``calibration:``
      section of the run YAML. Receives ``(cal_cfg, num_sequences, sequence_length, seed)``
      and is responsible for validating source-specific fields (e.g. that
      ``subset_weights`` is non-empty for nvidia-cascade).
    - ``stream_texts`` produces the list of rendered text rows for the spec.
      Receives ``(spec, tokenizer)``. Order does not matter — the caller
      reshuffles globally before tokenizing.
    """
    name: str
    parse_yaml: Callable[[dict, int, int, int], "CalibrationSpec"]
    stream_texts: Callable[["CalibrationSpec", object], list[str]]


_CORPUS_REGISTRY: dict[str, CorpusAdapter] = {}


def register_corpus(adapter: CorpusAdapter) -> CorpusAdapter:
    """Register a corpus adapter under its ``name``. Idempotent re-registration
    of an equal adapter is allowed (helps in test fixtures and across module
    reloads — ``CorpusAdapter`` is ``frozen=True`` and compares by value, so a
    reload that re-runs the module-bottom ``register_corpus`` calls succeeds);
    registering a different adapter under an existing name raises."""
    existing = _CORPUS_REGISTRY.get(adapter.name)
    if existing is not None and existing != adapter:
        raise ValueError(
            f"Corpus {adapter.name!r} is already registered with a different adapter"
        )
    _CORPUS_REGISTRY[adapter.name] = adapter
    return adapter


def _unregister_corpus(name: str) -> None:
    """Private test affordance — do not call from production code.

    Removes a registered corpus. No-op if not present. The leading underscore
    marks this as a test-only API; production callers should never need to
    unregister a corpus.
    """
    _CORPUS_REGISTRY.pop(name, None)


def get_corpus_adapter(name: str) -> CorpusAdapter:
    """Look up the adapter for ``name`` or raise ``ValueError`` listing the
    registered corpora — gives a much clearer error than a bare ``KeyError``.

    Raises ``ValueError`` (not ``KeyError``) because the lookup name comes from
    config-driven user input, not from internal dict-access.
    """
    adapter = _CORPUS_REGISTRY.get(name)
    if adapter is None:
        raise ValueError(
            f"Unknown calibration source {name!r}; registered: {sorted(_CORPUS_REGISTRY)}"
        )
    return adapter


def registered_corpora() -> list[str]:
    """Snapshot of registered corpus names, sorted."""
    return sorted(_CORPUS_REGISTRY)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def shared_calibration_cache_dir(artifacts_dir: str | Path) -> Path:
    """Resolve the calibration-tensor cache dir to the run-shared ``_shared/``.

    A calibration tensor depends only on the corpus + tokenizer + spec — never
    on which ablation row is running. Caching it under a per-ablation dir means
    every row (and every re-run, since per-ablation dirs get wiped) re-streams
    the corpus from scratch. ``artifacts_dir`` is an ablation row's dir; its
    parent is the ablations root that holds the persistent ``_shared/``.
    """
    return Path(artifacts_dir).parent / "_shared" / "_calibration_cache"


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
    adapter = get_corpus_adapter(spec.source)
    texts = adapter.stream_texts(spec, tokenizer)

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
    adapter = get_corpus_adapter(source)
    return adapter.parse_yaml(cal_cfg, num_sequences, sequence_length, seed)


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


def _render_messages(
    messages,
    tokenizer,
    *,
    enable_thinking: bool = False,
    add_generation_prompt: bool = False,
) -> str | None:
    """Render a messages list to a string: try apply_chat_template, fall back to
    role/content concat, and return None on failure so the caller's empty-row
    filter can discard the row.

    Parameters
    ----------
    enable_thinking
        Pass ``enable_thinking=True`` to ``apply_chat_template`` for Qwen3-
        thinking-style tokenizers. Required for calibration corpora whose
        assistant content carries ``<think>...</think>`` traces — without this
        kwarg the templater may strip the thinking block or fail to insert the
        ``<|im_start|>think\n`` opener. Older tokenizers that don't accept the
        kwarg raise ``TypeError``; we retry without it so the call still
        succeeds (the chat template still applies, just without the explicit
        thinking-mode flag). Default False preserves byte-identical behaviour
        for all callers that don't opt in.
    add_generation_prompt
        When True the template appends the assistant-cursor opener so the
        returned string ends at the position where the model would START
        generating. Useful for teacher-trace generation scripts; False (the
        default) gives the full sequence including the assistant turn(s).

    See ``scripts/build_self_traces_calib.py`` for the canonical caller that
    needs ``enable_thinking=True`` + ``add_generation_prompt=True`` to elicit
    the teacher's thinking + answer trace.
    """
    if not messages:
        return None
    try:
        # Strip here so both paths return consistently trimmed text (the fallback
        # path already strips; keeping both uniform avoids caller-side compensation).
        # Return None (not "") on empty render so both paths share the same sentinel.
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": add_generation_prompt,
        }
        if enable_thinking:
            # Try the kwarg first; older tokenizers raise TypeError on unknown
            # kwargs — fall back without it so the chat template still applies.
            try:
                rendered = tokenizer.apply_chat_template(
                    messages, enable_thinking=True, **kwargs,
                ).strip()
            except TypeError:
                global _enable_thinking_unsupported_warned
                if not _enable_thinking_unsupported_warned:
                    tok_name = getattr(tokenizer, "name_or_path", None) or type(tokenizer).__name__
                    log.warning(
                        "_render_messages: tokenizer %r does not accept "
                        "enable_thinking=True — falling back to plain "
                        "apply_chat_template. Reasoning-mode markers "
                        "(<think>...</think>) may not render correctly. "
                        "This warning fires once per process.",
                        tok_name,
                    )
                    _enable_thinking_unsupported_warned = True
                rendered = tokenizer.apply_chat_template(messages, **kwargs).strip()
        else:
            rendered = tokenizer.apply_chat_template(messages, **kwargs).strip()
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


# ---------------------------------------------------------------------------
# Built-in corpus adapters
# ---------------------------------------------------------------------------


def _parse_yaml_nvidia_cascade(
    cal_cfg: dict, num_sequences: int, sequence_length: int, seed: int,
) -> CalibrationSpec:
    """Yaml → CalibrationSpec for the ``nvidia-cascade`` source."""
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
        raise ValueError(
            f"subset_weights are all zero — at least one weight must be positive: {subset_weights}"
        )
    return CalibrationSpec(
        num_sequences=num_sequences,
        sequence_length=sequence_length,
        seed=seed,
        source="nvidia-cascade",
        dataset=dataset,
        subset_weights=subset_weights,
    )


def _stream_texts_nvidia_cascade(spec: CalibrationSpec, tokenizer) -> list[str]:
    """CalibrationSpec → text rows for the ``nvidia-cascade`` source."""
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
    texts: list[str] = []
    for subset, count in per.items():
        if count <= 0:
            continue
        # Per-subset seed offset so subsets draw from independent shuffles
        # even when the same base seed is reused across stages.
        subset_offset = int(
            hashlib.md5(subset.encode(), usedforsecurity=False).hexdigest(), 16
        ) % 1_000_000
        subset_seed = (spec.seed + subset_offset) % (2**32)
        texts.extend(_stream_cascade_texts(
            spec.dataset, subset, count, tokenizer, seed=subset_seed,
        ))
    return texts


def _parse_yaml_c4_math_code(
    cal_cfg: dict, num_sequences: int, sequence_length: int, seed: int,
) -> CalibrationSpec:
    """Yaml → CalibrationSpec for the legacy ``c4-math-code`` source."""
    if "domain_mix" not in cal_cfg:
        raise KeyError("calibration config is missing required key 'domain_mix'")
    # Warn if cascade-specific keys are present but ignored under this source.
    ignored = {k for k in ("subset_weights", "dataset") if k in cal_cfg}
    if ignored:
        log.warning(
            "spec_from_config: source='c4-math-code' — the following keys are ignored "
            "for the legacy source: %s",
            sorted(ignored),
        )
    domain_mix_raw = cal_cfg["domain_mix"]
    if not isinstance(domain_mix_raw, dict):
        raise ValueError(
            f"calibration config 'domain_mix' must be a dict mapping subset names to weights, "
            f"got {type(domain_mix_raw).__name__!r}"
        )
    if not domain_mix_raw:
        raise ValueError("calibration config: 'domain_mix' must be non-empty")
    domain_mix = dict(domain_mix_raw)
    unknown = set(domain_mix) - _LEGACY_DOMAINS
    if unknown:
        raise ValueError(
            f"calibration config 'domain_mix' has unknown domain keys "
            f"{sorted(unknown)!r}; valid keys are {sorted(_LEGACY_DOMAINS)!r}"
        )
    if any(w < 0 for w in domain_mix.values()):
        raise ValueError(f"domain_mix contains negative values: {domain_mix}")
    if not any(w > 0 for w in domain_mix.values()):
        raise ValueError(
            f"domain_mix are all zero — at least one weight must be positive: {domain_mix}"
        )
    return CalibrationSpec(
        num_sequences=num_sequences,
        sequence_length=sequence_length,
        seed=seed,
        source="c4-math-code",
        domain_mix=domain_mix,
        c4_dataset=cal_cfg.get("c4_dataset", "allenai/c4"),
        c4_subset=cal_cfg.get("c4_subset", "en"),
        math_dataset=cal_cfg.get("math_dataset", "hendrycks/competition_math"),
        code_dataset=cal_cfg.get("code_dataset", "bigcode/the-stack-smol"),
    )


def _stream_texts_c4_math_code(spec: CalibrationSpec, tokenizer) -> list[str]:
    """CalibrationSpec → text rows for the legacy ``c4-math-code`` source."""
    del tokenizer  # rendering happens lower down; the legacy loader is plain text
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
    texts: list[str] = []
    for domain, count in per.items():
        if count <= 0:
            continue
        # Per-domain seed offset so domains draw from independent shuffles
        # (mirrors the per-subset seed offset used for nvidia-cascade).
        domain_offset = int(
            hashlib.md5(domain.encode(), usedforsecurity=False).hexdigest(), 16
        ) % 1_000_000
        domain_seed = (spec.seed + domain_offset) % (2**32)
        texts.extend(_stream_legacy_texts(domain, count, spec, seed=domain_seed))
    return texts


register_corpus(CorpusAdapter(
    name="nvidia-cascade",
    parse_yaml=_parse_yaml_nvidia_cascade,
    stream_texts=_stream_texts_nvidia_cascade,
))


# ---------------------------------------------------------------------------
# tulu3-sft-mix — broad diverse SFT mixture (Tülu 3) for instruct calibration
# ---------------------------------------------------------------------------
# Rationale: nvidia-cascade is a narrow technical SFT slice (math, science,
# chat, IF, conv-agent, swe, terminal-agent). For an instruct-tuned MoE
# like Qwen3.6-A3B, the heal step's cross-domain transfer (e.g. WikiText
# BPT) barely moves when calibration is pool-narrow — empirically observed
# 2026-05-21 (this branch, 100x SH run): in-domain holdout MSE dropped
# 3.3x while WikiText-XD holdout dropped 2%.
#
# Tülu-3 is the most diverse open SFT mix today (WildChat + FLAN +
# ShareGPT + OpenAssistant + math + code). Rows have OpenAI-style
# ``messages=[{role,content}, ...]``; we render via the same
# ``_render_messages`` helper as nvidia-cascade so the chat template gets
# applied (instruct-model calibration requires it per Gemma 3 QAT docs).


def _parse_yaml_tulu3(
    cal_cfg: dict, num_sequences: int, sequence_length: int, seed: int,
) -> CalibrationSpec:
    """Yaml → CalibrationSpec for the ``tulu3-sft-mix`` source.

    Only ``dataset`` is honoured (defaults to ``allenai/tulu-3-sft-mixture``);
    Tülu-3 is already an internal mix so we don't sub-weight by source.
    """
    dataset = str(cal_cfg.get("dataset", "allenai/tulu-3-sft-mixture")).strip()
    return CalibrationSpec(
        num_sequences=num_sequences,
        sequence_length=sequence_length,
        seed=seed,
        source="tulu3-sft-mix",
        dataset=dataset,
    )


def _stream_texts_tulu3(spec: CalibrationSpec, tokenizer) -> list[str]:
    """Stream Tülu-3 rows, render each through the tokenizer's chat template."""
    from datasets import load_dataset

    log.info("Streaming %d tulu3-sft-mix samples from %s (seed=%d)",
             spec.num_sequences, spec.dataset, spec.seed)
    try:
        ds = load_dataset(spec.dataset, split="train", streaming=True)
    except Exception as err:                          # noqa: BLE001
        log.error("load_dataset(%s) failed: %s", spec.dataset, err)
        raise

    circuit_limit = _CIRCUIT_BREAKER_MULTIPLIER * spec.num_sequences
    ds = ds.shuffle(
        seed=spec.seed,
        buffer_size=min(max(10_000, circuit_limit), 200_000),
    )

    out: list[str] = []
    rows_seen = 0
    for row in ds:
        rows_seen += 1
        text = _render_messages(row.get("messages"), tokenizer)
        if text:
            out.append(text)
            if len(out) >= spec.num_sequences:
                break
        if rows_seen >= circuit_limit:
            log.warning(
                "_stream_texts_tulu3: circuit-breaker fired after %d rows examined "
                "for content (circuit_limit=%d); dataset may be smaller than expected",
                rows_seen, circuit_limit,
            )
            break
    if len(out) < spec.num_sequences:
        log.warning("tulu3-sft-mix only produced %d/%d non-empty rows",
                    len(out), spec.num_sequences)
    return out


register_corpus(CorpusAdapter(
    name="tulu3-sft-mix",
    parse_yaml=_parse_yaml_tulu3,
    stream_texts=_stream_texts_tulu3,
))

register_corpus(CorpusAdapter(
    name="c4-math-code",
    parse_yaml=_parse_yaml_c4_math_code,
    stream_texts=_stream_texts_c4_math_code,
))


# ---------------------------------------------------------------------------
# qwen3-pretrain-mix — broad multi-source SFT-target mix for instruct/
# thinking-mode calibration. All sources go through ``apply_chat_template``
# so the instruct-tuned router stays in-distribution (Gemma 3 QAT rule).
# ---------------------------------------------------------------------------
#
# Naming note: the historical name "qwen3-pretrain-mix" is kept as a stable
# config-key identifier; the actual SHAPE of the mix is instruct/SFT-target,
# NOT pretrain-shape. Our compression target is a chat-tuned thinking-mode
# student that emits ``<think>...</think>`` at deploy — calibration must
# match that distribution, with a small raw-text anti-forgetting replay
# slice. The eight subsets all map to domains Qwen3 explicitly documents:
#   * pretraining: "web + STEM + code + math + multilingual + books + synthetic"
#   * post-training SFT: "coding + math + instruction + multilingual +
#       creative writing + QA + role-playing"
# (Qwen has not published the granular percentages for either stage — these
# weights are an instruct-target choice, not a Qwen-published target.)
#
# Empirical motivation (2026-05-21 SH heal experiments):
#   * nvidia-cascade or tulu3 alone: in-domain heal-MSE drops ~70%/layer but
#     WikiText cross-domain MSE only drops 2-11%/layer — too narrow.
#   * qwen3-pretrain-mix (4-subset, pretrain-leaning): WikiText XD transfer
#     improves to 15-19%/layer on the qwen3_mix_seals.txt run.
#   * This expanded 8-subset SFT-target mix is the next iteration: ~90%
#     chat/reasoning + ~10% raw-text replay.
#
# Sub-sources (instruct-target weights, sum = 1.00):
#   tulu3 (30%):        allenai/tulu-3-sft-mixture           — SFT/chat
#   math (15%):         nvidia/OpenMathInstruct-2            — math reasoning
#   code (15%):         nickrosh/Evol-Instruct-Code-80k-v1   — code reasoning
#   qa (10%):           databricks/databricks-dolly-15k      — instruction QA
#   creative (10%):     euclaise/writingprompts              — long-form gen
#   multilingual (10%): CohereForAI/aya_dataset              — 65+ language SFT
#   fineweb (5%):       HuggingFaceFW/fineweb-edu            — raw-text replay
#   papers (5%):        gfissore/arxiv-abstracts-2021        — academic reasoning
#
# Anti-forgetting replay: fineweb-edu + papers are the only non-chat-shaped
# subsets; 10% combined keeps general knowledge / vocabulary stable while
# the other 90% supervise the routers + merged experts on deploy-shaped
# inputs. Drop these to 0% if a future experiment shows pure-SFT is fine.
#
# All datasets are parquet-stored (no datasets>=4.5 script-loader gotcha).

# Best-effort weights (L3): the per-subset streamers
# (_stream_messages_native, _stream_raw_wrapped, _stream_problem_solution,
# _stream_instruction_output) absorb per-row exceptions and continue. If a
# subset's underlying dataset throws on many rows (network blip, schema
# drift, encoding error), the configured weight will UNDER-represent that
# subset in the final mix. _stream_texts_qwen3_pretrain_mix itself also
# catches subset-level failures and skips the subset entirely. These weights
# describe INTENT; emit a token-shortfall warning from
# _tokenize_to_fixed_length to detect material drift.
_QWEN3_MIX_WEIGHTS = {
    "tulu3":        0.30,
    "math":         0.15,
    "code":         0.15,
    "qa":           0.10,
    "creative":     0.10,
    "multilingual": 0.10,
    "fineweb":      0.05,
    "papers":       0.05,
}

# Rough avg-tokens-per-row used for row-count budgeting. Underestimating is
# OK — `_tokenize_to_fixed_length` truncates to exactly the requested token
# budget; overshoot is just unused rows. Underestimating risks running out.
_QWEN3_MIX_AVG_TOKENS = {
    "tulu3":        600,
    "math":         800,
    "code":         300,
    "qa":           400,
    "creative":     600,
    "multilingual": 400,
    "fineweb":      1500,
    "papers":       300,
}

_QWEN3_MIX_DATASET = {
    "tulu3":        "allenai/tulu-3-sft-mixture",
    "math":         "nvidia/OpenMathInstruct-2",
    "code":         "nickrosh/Evol-Instruct-Code-80k-v1",
    "qa":           "databricks/databricks-dolly-15k",
    "creative":     "euclaise/writingprompts",
    "multilingual": "CohereForAI/aya_dataset",
    "fineweb":      "HuggingFaceFW/fineweb-edu",
    "papers":       "gfissore/arxiv-abstracts-2021",
}


def _parse_yaml_qwen3_pretrain_mix(
    cal_cfg: dict, num_sequences: int, sequence_length: int, seed: int,
) -> CalibrationSpec:
    """Yaml → CalibrationSpec for ``qwen3-pretrain-mix``.

    Ignores ``dataset`` / ``subset_weights`` from the YAML — the mix is
    hard-coded above to keep the experimental signal clean. If you need
    a custom mix, register a new adapter.
    """
    return CalibrationSpec(
        num_sequences=num_sequences,
        sequence_length=sequence_length,
        seed=seed,
        source="qwen3-pretrain-mix",
    )


def _make_subset_seed(base_seed: int, subset: str) -> int:
    subset_offset = int(
        hashlib.md5(subset.encode(), usedforsecurity=False).hexdigest(), 16
    ) % 1_000_000
    return (base_seed + subset_offset) % (2**32)


def _shuffled_stream(
    dataset_name: str, count: int, seed: int,
    *, config: str | None = None, split: str = "train",
):
    """Open a streaming HF dataset and return a shuffled iterator + circuit
    limit. Caller iterates and breaks when `count` non-empty rows yielded.

    ``config`` and ``split`` are optional kwargs added for the
    qwen3-pretrain-mix-v2 corpus, whose MoT-math/code/science subsets
    require a non-default config and whose swe_smith subset requires
    ``split="xml"``. Defaults preserve the v1 behavior (config=None →
    ``load_dataset(name, split="train", streaming=True)``) so all
    existing callsites are unaffected.
    """
    from datasets import load_dataset
    try:
        # load_dataset's positional signature is (path, name=None, ...). Passing
        # ``name=None`` is equivalent to omitting it; keep both branches explicit
        # for log-line clarity.
        if config is None:
            ds = load_dataset(dataset_name, split=split, streaming=True)
        else:
            ds = load_dataset(dataset_name, config, split=split, streaming=True)
    except Exception as err:                          # noqa: BLE001
        log.error("load_dataset(%s, config=%r, split=%r) failed: %s",
                  dataset_name, config, split, err)
        raise
    circuit_limit = _CIRCUIT_BREAKER_MULTIPLIER * count
    ds = ds.shuffle(
        seed=seed,
        buffer_size=min(max(10_000, circuit_limit), 200_000),
    )
    return ds, circuit_limit


def _stream_messages_native(dataset_name: str, count: int, tokenizer, seed: int) -> list[str]:
    """For datasets with a native ``messages=[...]`` field (e.g. Tülu-3)."""
    ds, circuit_limit = _shuffled_stream(dataset_name, count, seed)
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
            log.warning("_stream_messages_native: circuit-breaker fired at %d rows on %s",
                        rows_seen, dataset_name)
            break
    if len(out) < count:
        log.warning("%s yielded %d/%d rows", dataset_name, len(out), count)
    return out


def _stream_raw_wrapped(
    dataset_name: str, text_field: str, count: int, tokenizer, seed: int,
    *, user_prompt: str,
) -> list[str]:
    """Plain-text dataset → wrap each row as a chat turn (user prompt +
    text as assistant response), then run through ``apply_chat_template``."""
    ds, circuit_limit = _shuffled_stream(dataset_name, count, seed)
    out: list[str] = []
    rows_seen = 0
    for row in ds:
        rows_seen += 1
        text = (row.get(text_field) or "").strip()
        if text:
            messages = [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": text},
            ]
            rendered = _render_messages(messages, tokenizer)
            if rendered:
                out.append(rendered)
                if len(out) >= count:
                    break
        if rows_seen >= circuit_limit:
            log.warning("_stream_raw_wrapped: circuit-breaker fired at %d rows on %s",
                        rows_seen, dataset_name)
            break
    if len(out) < count:
        log.warning("%s yielded %d/%d rows", dataset_name, len(out), count)
    return out


def _stream_problem_solution(
    dataset_name: str, count: int, tokenizer, seed: int,
    *, problem_field: str, solution_field: str,
) -> list[str]:
    """Problem/solution-pair dataset (e.g. OpenMathInstruct-2) → messages."""
    ds, circuit_limit = _shuffled_stream(dataset_name, count, seed)
    out: list[str] = []
    rows_seen = 0
    for row in ds:
        rows_seen += 1
        problem = (row.get(problem_field) or "").strip()
        solution = (row.get(solution_field) or "").strip()
        if problem and solution:
            messages = [
                {"role": "user", "content": problem},
                {"role": "assistant", "content": solution},
            ]
            rendered = _render_messages(messages, tokenizer)
            if rendered:
                out.append(rendered)
                if len(out) >= count:
                    break
        if rows_seen >= circuit_limit:
            log.warning("_stream_problem_solution: circuit-breaker fired at %d rows on %s",
                        rows_seen, dataset_name)
            break
    if len(out) < count:
        log.warning("%s yielded %d/%d rows", dataset_name, len(out), count)
    return out


def _stream_instruction_output(
    dataset_name: str, count: int, tokenizer, seed: int,
    *,
    instruction_field: str = "instruction",
    input_field: str = "input",
    output_field: str = "output",
) -> list[str]:
    """Alpaca-style ``instruction``/``input``/``output`` dataset → messages.

    Field names default to the canonical Alpaca schema but are overridable so
    the same adapter handles e.g. databricks-dolly-15k (``response`` instead
    of ``output``, ``context`` instead of ``input``).
    """
    ds, circuit_limit = _shuffled_stream(dataset_name, count, seed)
    out: list[str] = []
    rows_seen = 0
    for row in ds:
        rows_seen += 1
        instruction = (row.get(instruction_field) or "").strip()
        input_part = (row.get(input_field) or "").strip()
        output = (row.get(output_field) or "").strip()
        if instruction and output:
            user_content = instruction + (("\n\n" + input_part) if input_part else "")
            messages = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": output},
            ]
            rendered = _render_messages(messages, tokenizer)
            if rendered:
                out.append(rendered)
                if len(out) >= count:
                    break
        if rows_seen >= circuit_limit:
            log.warning("_stream_instruction_output: circuit-breaker fired at %d rows on %s",
                        rows_seen, dataset_name)
            break
    if len(out) < count:
        log.warning("%s yielded %d/%d rows", dataset_name, len(out), count)
    return out


def _stream_texts_qwen3_pretrain_mix(spec: CalibrationSpec, tokenizer) -> list[str]:
    """Multi-source Qwen3-pretrain-distribution mix, all chat-templated."""
    target_total_tokens = spec.num_sequences * spec.sequence_length
    texts: list[str] = []
    # N4: corpus identifier "qwen3-pretrain-mix" is the historical config-compat
    # name; the SHAPE is instruct/SFT-target (see file-top note ~lines 941-980).
    # N-C — fire the orientation banner once per process; per-call repeats add
    # noise without information across the ~6 build_calibration_tensor calls
    # in a pipeline run.
    global _broad_instruct_mix_intro_logged
    if not _broad_instruct_mix_intro_logged:
        log.info(
            "qwen3-pretrain-mix: building broad-instruct-mix (legacy name retained "
            "for config compat) for %d sequences x %d tokens (target_total=%d).",
            spec.num_sequences, spec.sequence_length, target_total_tokens,
        )
        _broad_instruct_mix_intro_logged = True
    for subset, weight in _QWEN3_MIX_WEIGHTS.items():
        target_subset_tokens = int(target_total_tokens * weight)
        avg_tok = _QWEN3_MIX_AVG_TOKENS[subset]
        # 2x oversample — `_tokenize_to_fixed_length` truncates anyway, and
        # we'd rather have leftover rows than fall short on token budget.
        n_rows = max(1, int((target_subset_tokens / avg_tok) * 2.0))
        seed = _make_subset_seed(spec.seed, subset)
        ds_name = _QWEN3_MIX_DATASET[subset]
        log.info("qwen3-pretrain-mix: %s — streaming %d rows from %s "
                 "(weight=%.2f, target_tokens=%d, avg_tok=%d, seed=%d)",
                 subset, n_rows, ds_name, weight, target_subset_tokens, avg_tok, seed)
        try:
            if subset == "tulu3":
                subset_texts = _stream_messages_native(ds_name, n_rows, tokenizer, seed)
            elif subset == "fineweb":
                subset_texts = _stream_raw_wrapped(
                    ds_name, "text", n_rows, tokenizer, seed,
                    user_prompt="Read this passage carefully and reproduce it faithfully.",
                )
            elif subset == "math":
                subset_texts = _stream_problem_solution(
                    ds_name, n_rows, tokenizer, seed,
                    problem_field="problem", solution_field="generated_solution",
                )
            elif subset == "code":
                subset_texts = _stream_instruction_output(ds_name, n_rows, tokenizer, seed)
            elif subset == "qa":
                # databricks-dolly-15k: instruction/context/response (not output).
                subset_texts = _stream_instruction_output(
                    ds_name, n_rows, tokenizer, seed,
                    input_field="context", output_field="response",
                )
            elif subset == "creative":
                # euclaise/writingprompts: prompt/story.
                subset_texts = _stream_problem_solution(
                    ds_name, n_rows, tokenizer, seed,
                    problem_field="prompt", solution_field="story",
                )
            elif subset == "multilingual":
                # CohereForAI/aya_dataset: inputs/targets across 65+ languages.
                subset_texts = _stream_problem_solution(
                    ds_name, n_rows, tokenizer, seed,
                    problem_field="inputs", solution_field="targets",
                )
            elif subset == "papers":
                # gfissore/arxiv-abstracts-2021: title/abstract. "Given a title,
                # write the abstract" exercises academic-style reasoning without
                # the giant full-paper token bill.
                subset_texts = _stream_problem_solution(
                    ds_name, n_rows, tokenizer, seed,
                    problem_field="title", solution_field="abstract",
                )
            else:
                raise ValueError(f"unknown qwen3-pretrain-mix subset {subset!r}")
        except Exception as err:                      # noqa: BLE001
            # One source failing shouldn't tank the whole calibration build;
            # log + continue with what we have. If too many fail,
            # `_tokenize_to_fixed_length` will warn about a short token pool.
            log.error("qwen3-pretrain-mix: subset %s (%s) failed: %s — continuing",
                      subset, ds_name, err)
            continue
        log.info("qwen3-pretrain-mix: %s — yielded %d rows", subset, len(subset_texts))
        texts.extend(subset_texts)
    log.info("qwen3-pretrain-mix: total %d rendered rows across all subsets", len(texts))
    return texts


register_corpus(CorpusAdapter(
    name="qwen3-pretrain-mix",
    parse_yaml=_parse_yaml_qwen3_pretrain_mix,
    stream_texts=_stream_texts_qwen3_pretrain_mix,
))


# ---------------------------------------------------------------------------
# qwen3-pretrain-mix-v2 — reasoning-mode 12-subset hybrid mix (Generate + TF)
# ---------------------------------------------------------------------------
#
# Sibling to ``qwen3-pretrain-mix`` (v1, 8 subsets, all generate-mode). The
# v2 mix narrows-and-expands the source distribution to better cover the
# Qwen3-thinking deploy surface: keeps the 7 instruct/SFT subsets that v1
# already used (tulu3, math, qa, creative, multilingual, fineweb, papers),
# adds 4 reasoning-format-native sources via TEACHER_FORCED policy
# (MoT-math/code/science from open-r1, plus SWE-smith trajectories), and
# adds Glaive function-calling as a GENERATE subset for tool-use coverage.
#
# Policy plumbing: every subset has a fixed policy ("GENERATE" or
# "TEACHER_FORCED"). The build_self_traces_calib{,_vllm}.py scripts read
# this policy to decide whether to run the teacher's generate() loop or
# whether to emit the canonical assistant turn directly into the JSONL.
# The downstream calibration loader (_stream_texts_self_traces) is policy-
# agnostic — it renders+forwards whatever assistant message string ends up
# in the JSONL row. Stage 2/2.5/3 capture cov/router/imatrix stats from
# the rendered tokens regardless of source.
#
# Sole-truth dicts: all per-subset behavior (weight, avg_tokens, dataset,
# config, split, policy) flows from the six ``_QWEN3_MIX_V2_*`` dicts
# below. The iterator dispatches on subset key and looks up policy /
# dataset / config / split / avg_tokens from these dicts; do NOT hardcode
# per-subset choices in iterator/streamer bodies.
#
# Design + plan refs:
#  * tasks/CALIBRATION_MIX_V2_DESIGN.md
#  * tasks/CALIBRATION_MIX_V2_PLAN.md

# 12-subset mix; weights sum to exactly 1.0 (5 GENERATE + 1 function_calling
# GENERATE = 52%, 4 TEACHER_FORCED = 48%).
_QWEN3_MIX_V2_WEIGHTS = {
    "tulu3":            0.11,
    "math":             0.09,
    "qa":               0.05,
    "creative":         0.05,
    "multilingual":     0.08,
    "fineweb":          0.05,
    "papers":           0.05,
    "mot_math":         0.12,
    "mot_code":         0.12,
    "mot_science":      0.08,
    "swe_smith":        0.12,
    "function_calling": 0.08,
}

# Avg tokens per row used for row-count budgeting. Underestimating risks
# running out; overshoot is harmless (truncated by ``_tokenize_to_fixed_length``).
_QWEN3_MIX_V2_AVG_TOKENS = {
    "tulu3":              600,
    "math":               800,
    "qa":                 400,
    "creative":           600,
    "multilingual":       400,
    "fineweb":           1500,
    "papers":             300,
    "mot_math":          3500,
    "mot_code":         15000,
    "mot_science":       2500,
    "swe_smith":         8000,
    "function_calling":   500,
}

_QWEN3_MIX_V2_DATASET = {
    "tulu3":            "allenai/tulu-3-sft-mixture",
    "math":             "nvidia/OpenMathInstruct-2",
    "qa":               "databricks/databricks-dolly-15k",
    "creative":         "euclaise/writingprompts",
    "multilingual":     "CohereForAI/aya_dataset",
    "fineweb":          "HuggingFaceFW/fineweb-edu",
    "papers":           "gfissore/arxiv-abstracts-2021",
    "mot_math":         "open-r1/Mixture-of-Thoughts",
    "mot_code":         "open-r1/Mixture-of-Thoughts",
    "mot_science":      "open-r1/Mixture-of-Thoughts",
    "swe_smith":        "SWE-bench/SWE-smith-trajectories",
    "function_calling": "glaiveai/glaive-function-calling-v2",
}

# Per-subset HF dataset config name (``load_dataset(name, config, ...)``).
# Only the three MoT subsets need a config; all others use the default config
# (``None`` → load_dataset's ``name=None`` positional default).
_QWEN3_MIX_V2_DATASET_CONFIG: dict[str, str | None] = {
    "tulu3":            None,
    "math":             None,
    "qa":               None,
    "creative":         None,
    "multilingual":     None,
    "fineweb":          None,
    "papers":           None,
    "mot_math":         "math",
    "mot_code":         "code",
    "mot_science":      "science",
    "swe_smith":        None,
    "function_calling": None,
}

# Per-subset HF dataset split. Plain ``"train"`` for everything except
# swe_smith which has named splits (we want ``"xml"`` — the Anthropic-XML
# tool-call format that matches Qwen3's <tool_call> template).
_QWEN3_MIX_V2_DATASET_SPLIT = {
    "tulu3":            "train",
    "math":             "train",
    "qa":               "train",
    "creative":         "train",
    "multilingual":     "train",
    "fineweb":          "train",
    "papers":           "train",
    "mot_math":         "train",
    "mot_code":         "train",
    "mot_science":      "train",
    "swe_smith":        "xml",
    "function_calling": "train",
}

# Per-subset policy. "GENERATE" = teacher generates fresh thinking-mode
# response from prompt-only. "TEACHER_FORCED" = use the canonical assistant
# turn from the source dataset (skip generation, emit JSONL row directly).
# Consumed by build_self_traces_calib{,_vllm}.py at row-write time.
_QWEN3_MIX_V2_POLICY = {
    "tulu3":            "GENERATE",
    "math":             "GENERATE",
    "qa":               "GENERATE",
    "creative":         "GENERATE",
    "multilingual":     "GENERATE",
    "fineweb":          "GENERATE",
    "papers":           "GENERATE",
    "mot_math":         "TEACHER_FORCED",
    "mot_code":         "TEACHER_FORCED",
    "mot_science":      "TEACHER_FORCED",
    "swe_smith":        "TEACHER_FORCED",
    "function_calling": "GENERATE",
}

# One-shot guard for the v2 intro banner (mirrors the v1 guard at the
# top of this file; do NOT reuse the v1 flag so v1 and v2 each get to log
# their own orientation banner once per process).
_broad_instruct_mix_v2_intro_logged: bool = False


def _parse_yaml_qwen3_pretrain_mix_v2(
    cal_cfg: dict, num_sequences: int, sequence_length: int, seed: int,
) -> CalibrationSpec:
    """Yaml → CalibrationSpec for ``qwen3-pretrain-mix-v2``.

    Twin of ``_parse_yaml_qwen3_pretrain_mix`` but stamps
    ``source="qwen3-pretrain-mix-v2"`` so the CalibrationSpec cache_key
    correctly invalidates between v1 and v2 runs. Same contract: the
    YAML's ``dataset`` / ``subset_weights`` are ignored — the mix is
    hard-coded in the ``_QWEN3_MIX_V2_*`` dicts above.
    """
    return CalibrationSpec(
        num_sequences=num_sequences,
        sequence_length=sequence_length,
        seed=seed,
        source="qwen3-pretrain-mix-v2",
    )


def _stream_messages_with_config(
    dataset_name: str, config: str | None, split: str,
    count: int, tokenizer, seed: int,
) -> list[str]:
    """Like ``_stream_messages_native`` but routes through ``_shuffled_stream``'s
    ``config``/``split`` kwargs so the MoT-math/code/science subsets can pull
    from their non-default configs. Renders ``row["messages"]`` (a native
    list-of-dicts on these subsets) via ``apply_chat_template`` with
    ``enable_thinking=True`` so the canonical ``<think>...</think>`` block in
    the assistant turn lands in the rendered text intact.
    """
    ds, circuit_limit = _shuffled_stream(
        dataset_name, count, seed, config=config, split=split,
    )
    out: list[str] = []
    rows_seen = 0
    for row in ds:
        rows_seen += 1
        text = _render_messages(
            row.get("messages"), tokenizer, enable_thinking=True,
        )
        if text:
            out.append(text)
            if len(out) >= count:
                break
        if rows_seen >= circuit_limit:
            log.warning(
                "_stream_messages_with_config: circuit-breaker fired at %d "
                "rows on %s (config=%r, split=%r)",
                rows_seen, dataset_name, config, split,
            )
            break
    if len(out) < count:
        log.warning(
            "%s (config=%r, split=%r) yielded %d/%d rows",
            dataset_name, config, split, len(out), count,
        )
    return out


def _stream_swe_smith_xml(
    dataset_name: str, split: str, count: int, tokenizer, seed: int,
) -> list[str]:
    """SWE-smith trajectories — multi-turn flatten to (first_user, first_assistant).

    SWE-smith stores ``messages`` as a JSON-encoded string (NOT a native
    list), so we ``json.loads`` first. The schema is system + user +
    assistant + (tool / user / assistant)*; we keep only the first
    (user, assistant) pair and drop the system header (Qwen3's
    apply_chat_template re-injects its own system preamble at
    calibration consumption time — including SWE-smith's would double
    the agent-bash-tool boilerplate). The first assistant turn carries
    the literal ``<function=...>`` tool-call block as plain string
    content; the chat template renders it as-is (it lives in the
    assistant ``content`` field, not in a separate ``tool_calls`` slot,
    so the template will NOT re-wrap it). Per the design doc §5.4 this
    gives us routing-pattern supervision for tool-use traces, at the
    cost of zero ``<think>`` supervision on this subset (Claude 3.7's
    SWE-smith trajectories were generated outside extended-thinking
    mode).
    """
    ds, circuit_limit = _shuffled_stream(
        dataset_name, count, seed, config=None, split=split,
    )
    out: list[str] = []
    rows_seen = 0
    for row in ds:
        rows_seen += 1
        raw = row.get("messages")
        if not isinstance(raw, str):
            # Schema drift — abort row but keep going (matches v1 per-row
            # tolerance).
            continue
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(parsed, list):
            continue
        # Find first user message and first assistant AFTER it. The xml
        # split's typical layout is [system, user, assistant, ...]; we
        # tolerate any leading system messages and skip them.
        user_msg = None
        assistant_msg = None
        seen_user = False
        for msg in parsed:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            content = msg.get("content")
            if not isinstance(content, str):
                continue
            if role == "user" and user_msg is None:
                user_msg = content
                seen_user = True
            elif role == "assistant" and seen_user and assistant_msg is None:
                assistant_msg = content
                break
        if not user_msg or not assistant_msg:
            continue
        flat = [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]
        rendered = _render_messages(flat, tokenizer, enable_thinking=True)
        if rendered:
            out.append(rendered)
            if len(out) >= count:
                break
        if rows_seen >= circuit_limit:
            log.warning(
                "_stream_swe_smith_xml: circuit-breaker fired at %d rows "
                "on %s (split=%r)", rows_seen, dataset_name, split,
            )
            break
    if len(out) < count:
        log.warning("%s (split=%r) yielded %d/%d rows",
                    dataset_name, split, len(out), count)
    return out


def _stream_glaive_function_calling(
    dataset_name: str, count: int, tokenizer, seed: int,
) -> list[str]:
    """Glaive function-calling v2 — extract first USER turn + system schema.

    Glaive stores both ``system`` and ``chat`` as flat strings; ``chat``
    uses literal ``USER:`` / ``ASSISTANT:`` / ``FUNCTION RESPONSE:``
    markers with ``<|endoftext|>`` separators. We pull the FIRST user
    turn and prepend the system content as a single combined user
    message (Option (i) from plan §B.6 — keeps the iterator tuple shape
    stable; the teacher then generates its own Qwen3-native
    ``<tool_call>...</tool_call>`` response).

    For the calibration corpus path (this helper), we have neither the
    canonical Glaive ASSISTANT turn (GENERATE policy discards it) nor a
    teacher-generated thinking-mode response, so we render the user
    turn alone and let ``_tokenize_to_fixed_length`` keep it as a
    prompt-only calibration row. This matches v1's prompt-only handling
    of subsets where the canonical assistant content is wrong-format
    (e.g., aya / writingprompts at calibration consumption time).
    """
    ds, circuit_limit = _shuffled_stream(
        dataset_name, count, seed, config=None, split="train",
    )
    out: list[str] = []
    rows_seen = 0
    for row in ds:
        rows_seen += 1
        system_raw = row.get("system") or ""
        chat_raw = row.get("chat") or ""
        if not isinstance(system_raw, str) or not isinstance(chat_raw, str):
            continue
        system_text = system_raw.strip()
        if system_text.startswith("SYSTEM:"):
            system_text = system_text[len("SYSTEM:"):].strip()
        # First USER turn in the flat chat string.
        user_text = _extract_glaive_first_user(chat_raw)
        if not user_text:
            continue
        # Option (i): concat system schema + first user turn into one user
        # message (see plan §B.6). The teacher generates its own
        # Qwen3-native <tool_call> response; for calibration purposes we
        # need only the prompt-shaped tokens.
        user_content = (
            f"{system_text}\n\n{user_text}" if system_text else user_text
        )
        messages = [{"role": "user", "content": user_content}]
        rendered = _render_messages(
            messages, tokenizer, add_generation_prompt=True,
        )
        if rendered:
            out.append(rendered)
            if len(out) >= count:
                break
        if rows_seen >= circuit_limit:
            log.warning(
                "_stream_glaive_function_calling: circuit-breaker fired at "
                "%d rows on %s", rows_seen, dataset_name,
            )
            break
    if len(out) < count:
        log.warning("%s yielded %d/%d rows", dataset_name, len(out), count)
    return out


# Markers in Glaive's flat ``chat`` string that separate the first USER turn
# from whatever follows (ASSISTANT response, FUNCTION RESPONSE, next USER).
# Kept as module-level constants so the prompt-iterator in
# build_self_traces_calib.py can import the same set instead of re-inlining
# them. Order doesn't matter — we take min(found index) across them all.
_GLAIVE_TURN_MARKERS = ("ASSISTANT:", "FUNCTION RESPONSE:", "USER:")


def _extract_glaive_first_user(chat: str) -> str | None:
    """Return the first USER turn's content (stripped) from a Glaive flat
    chat string, or None if no USER turn is found.

    Algorithm:
      1. Find the first ``USER:`` occurrence and slice from just after it.
      2. Find the next occurrence of any of ASSISTANT:, FUNCTION RESPONSE:,
         or a SUBSEQUENT USER:, and slice up to the earliest such
         boundary. If none is found, take everything to end-of-string.
      3. Strip whitespace and ``<|endoftext|>`` sentinels.
    """
    if not isinstance(chat, str):
        return None
    start = chat.find("USER:")
    if start < 0:
        return None
    body_start = start + len("USER:")
    # Find earliest subsequent boundary in (ASSISTANT:, FUNCTION RESPONSE:,
    # USER:).
    boundaries = []
    for marker in _GLAIVE_TURN_MARKERS:
        idx = chat.find(marker, body_start)
        if idx >= 0:
            boundaries.append(idx)
    body_end = min(boundaries) if boundaries else len(chat)
    body = chat[body_start:body_end].strip()
    # Strip any stray <|endoftext|> separator at end of turn.
    body = body.replace("<|endoftext|>", "").strip()
    return body or None


def _stream_texts_qwen3_pretrain_mix_v2(spec: CalibrationSpec, tokenizer) -> list[str]:
    """Multi-source Qwen3-thinking-mode reasoning mix (v2).

    12 subsets dispatched by key from ``_QWEN3_MIX_V2_WEIGHTS``. Per-subset
    behavior (weight, avg-tokens, dataset, config, split, policy) flows
    from the six ``_QWEN3_MIX_V2_*`` sole-truth dicts. The calibration
    consumer is policy-agnostic — both GENERATE and TEACHER_FORCED
    subsets land as rendered chat-formatted texts here; the policy
    field is consumed only by the build_self_traces_calib{,_vllm}.py
    scripts when they decide whether to call model.generate() or to
    short-circuit and emit the canonical completion directly.

    See tasks/CALIBRATION_MIX_V2_PLAN.md §B for the per-subset row-
    extraction recipe.
    """
    target_total_tokens = spec.num_sequences * spec.sequence_length
    texts: list[str] = []
    global _broad_instruct_mix_v2_intro_logged
    if not _broad_instruct_mix_v2_intro_logged:
        log.info(
            "qwen3-pretrain-mix-v2: building 12-subset reasoning mix "
            "(7 carryover GENERATE + 4 TEACHER_FORCED + Glaive GENERATE) "
            "for %d sequences x %d tokens (target_total=%d).",
            spec.num_sequences, spec.sequence_length, target_total_tokens,
        )
        _broad_instruct_mix_v2_intro_logged = True

    for subset, weight in _QWEN3_MIX_V2_WEIGHTS.items():
        target_subset_tokens = int(target_total_tokens * weight)
        avg_tok = _QWEN3_MIX_V2_AVG_TOKENS[subset]
        # 2x oversample — mirrors v1; `_tokenize_to_fixed_length` truncates
        # so leftover rows are harmless.
        n_rows = max(1, int((target_subset_tokens / avg_tok) * 2.0))
        seed = _make_subset_seed(spec.seed, subset)
        ds_name = _QWEN3_MIX_V2_DATASET[subset]
        config = _QWEN3_MIX_V2_DATASET_CONFIG[subset]
        split = _QWEN3_MIX_V2_DATASET_SPLIT[subset]
        policy = _QWEN3_MIX_V2_POLICY[subset]
        log.info(
            "qwen3-pretrain-mix-v2: %s — streaming %d rows from %s "
            "(weight=%.2f, target_tokens=%d, avg_tok=%d, seed=%d, "
            "config=%r, split=%r, policy=%s)",
            subset, n_rows, ds_name, weight, target_subset_tokens, avg_tok,
            seed, config, split, policy,
        )
        try:
            if subset == "tulu3":
                subset_texts = _stream_messages_native(
                    ds_name, n_rows, tokenizer, seed,
                )
            elif subset == "fineweb":
                subset_texts = _stream_raw_wrapped(
                    ds_name, "text", n_rows, tokenizer, seed,
                    user_prompt="Read this passage carefully and reproduce it faithfully.",
                )
            elif subset == "math":
                subset_texts = _stream_problem_solution(
                    ds_name, n_rows, tokenizer, seed,
                    problem_field="problem", solution_field="generated_solution",
                )
            elif subset == "qa":
                # databricks-dolly-15k: instruction/context/response.
                subset_texts = _stream_instruction_output(
                    ds_name, n_rows, tokenizer, seed,
                    input_field="context", output_field="response",
                )
            elif subset == "creative":
                # euclaise/writingprompts: prompt/story.
                subset_texts = _stream_problem_solution(
                    ds_name, n_rows, tokenizer, seed,
                    problem_field="prompt", solution_field="story",
                )
            elif subset == "multilingual":
                # CohereForAI/aya_dataset: inputs/targets across 65+ languages.
                subset_texts = _stream_problem_solution(
                    ds_name, n_rows, tokenizer, seed,
                    problem_field="inputs", solution_field="targets",
                )
            elif subset == "papers":
                # gfissore/arxiv-abstracts-2021: title/abstract.
                subset_texts = _stream_problem_solution(
                    ds_name, n_rows, tokenizer, seed,
                    problem_field="title", solution_field="abstract",
                )
            elif subset in ("mot_math", "mot_code", "mot_science"):
                # open-r1/Mixture-of-Thoughts — R1 canonical <think>...</think>
                # traces in messages=[user, assistant] shape.
                subset_texts = _stream_messages_with_config(
                    ds_name, config, split, n_rows, tokenizer, seed,
                )
            elif subset == "swe_smith":
                # SWE-bench/SWE-smith-trajectories xml split — multi-turn
                # tool-call traces (Claude 3.7), flatten to first
                # (user, assistant) pair.
                subset_texts = _stream_swe_smith_xml(
                    ds_name, split, n_rows, tokenizer, seed,
                )
            elif subset == "function_calling":
                subset_texts = _stream_glaive_function_calling(
                    ds_name, n_rows, tokenizer, seed,
                )
            else:
                raise ValueError(
                    f"unknown qwen3-pretrain-mix-v2 subset {subset!r}"
                )
        except Exception as err:                       # noqa: BLE001
            log.error(
                "qwen3-pretrain-mix-v2: subset %s (%s) failed: %s — continuing",
                subset, ds_name, err,
            )
            continue
        log.info("qwen3-pretrain-mix-v2: %s — yielded %d rows",
                 subset, len(subset_texts))
        texts.extend(subset_texts)
    log.info("qwen3-pretrain-mix-v2: total %d rendered rows across all subsets",
             len(texts))
    return texts


register_corpus(CorpusAdapter(
    name="qwen3-pretrain-mix-v2",
    parse_yaml=_parse_yaml_qwen3_pretrain_mix_v2,
    stream_texts=_stream_texts_qwen3_pretrain_mix_v2,
))


# ---------------------------------------------------------------------------
# self-traces — model-self-distillation calibration (FIRST STEP FOR ANY NEW MODEL)
# ---------------------------------------------------------------------------
#
# This corpus is generic — it works for ANY teacher model (Qwen3.x-thinking,
# Llama-3, DeepSeek-R1, Gemma-3, etc.). The corpus name is intentionally
# model-agnostic; the teacher's own outputs become the calibration text.
#
# Why this corpus exists
# ----------------------
# All other calibration sources (nvidia-cascade, tulu3-sft-mix, c4-math-code,
# qwen3-pretrain-mix) render the chat template around the OUTER role headers
# and EOS markers — that's the Gemma 3 QAT rule, and it stops the model from
# "forgetting how to chat" post-compression. But for reasoning-mode models
# (any model that emits ``<think>...</think>`` or analogous CoT markers in
# its assistant turn) the most reasoning-critical token positions live
# INSIDE that block — and none of the above sources have such traces in
# their assistant content. So the routers (Stage 2.5 KD) and merged expert
# weights (Stage 2 SH heal) are never supervised on the token positions
# that matter most at serve time.
#
# Empirical motivation:
#  * Project memory ``project_sh_lr_schedule_xd_result`` (2026-05-21):
#    cross-domain telemetry showed Nemotron-Cascade-only heal moved
#    Nemotron MSE by ~70% but WikiText MSE by only 5-30%. The local-MSE
#    objective was calibration-distribution-bound.
#  * Same logic applies to reasoning-mode: a heal/KD optimized on non-
#    thinking chat text won't transfer to ``<think>...</think>`` token
#    positions at deploy.
#
# When to use self-traces
# -----------------------
# **This is the FIRST step for any new model.** Before running the pipeline
# against a new teacher, you must generate the trace JSONL once. The
# pipeline fails loudly at Stage 1 calibration build if the JSONL is missing
# (see the FileNotFoundError in _stream_texts_self_traces below — it prints
# the exact command for the current model).
#
# Schema
# ------
# JSONL where each row is one of:
#  * ``{"messages": [{"role": "user", "content": ...},
#                    {"role": "assistant", "content": "<think>...</think>final answer"}]}``
#  * Or any other valid ``messages`` shape that ``apply_chat_template``
#    accepts; the loader passes ``enable_thinking=True`` so reasoning-mode
#    tokenizers (Qwen3-thinking, DeepSeek-R1 distill, etc.) render the
#    trace correctly. The kwarg is silently ignored by tokenizers that
#    don't know about it (TypeError fallback).
#
# How to generate it
# ------------------
# One-shot pre-step — see ``max_quality/scripts/build_self_traces_calib.py``.
# Greedy generation under fixed (teacher_repo, revision, prompts,
# max_new_tokens) is deterministic, so the JSONL is reproducible and the
# cache invalidates correctly when any of those change.
#
# YAML
# ----
# .. code-block:: yaml
#
#     calibration:
#       source: self-traces
#       seed: 1337
#       num_sequences: 4000
#       sequence_length: 2048
#       jsonl_path: artifacts/_shared/self_traces.jsonl   # optional;
#       # defaults to ``artifacts/_shared/self_traces.jsonl`` when omitted.
#       # Path is folded into ``CalibrationSpec.dataset`` and hashed into
#       # the cache key so changing the file auto-invalidates the
#       # calibration cache.

# Default self-traces JSONL location. Resolved relative to CWD by
# ``_stream_texts_self_traces`` (M2 documented assumption): the pipeline
# driver invokes from the repo root, so this default lands at
# ``<repo>/artifacts/_shared/self_traces.jsonl``. Operators running from a
# different CWD MUST set ``calibration.jsonl_path`` to an absolute path in
# their YAML to avoid silent FileNotFoundError or pointing at the wrong file.
_DEFAULT_SELF_TRACES_PATH = "artifacts/_shared/self_traces.jsonl"

# Per-process state for the self-traces blacklist, keyed by absolute JSONL
# path so multiple datasets (e.g. different teachers) coexist. Each entry:
#     {
#         "rows":      list[dict]               # parsed JSONL rows in file order
#         "by_domain": dict[str, list[int]]     # domain → row-indices in rows
#         "weights":   dict[str, float]         # empirical domain weights (sum=1)
#         "blacklist": dict[str, set[int]]      # PER-DOMAIN row-indices already
#                                               # served this run (H2 — domains
#                                               # exhaust independently; only the
#                                               # offending domain is whitelisted
#                                               # on quota miss).
#     }
#
# Why per-process: every stage that calls build_calibration_tensor
# (Stage 1 main + ablation_filter, Stage 2, Stage 2.5 / 5, Stage 3, Stage 6alt
# thermo_corpus) reaches the loader through _stream_texts_self_traces. They
# share this module's globals, so the blacklist accumulates across stages
# within ONE python invocation. The on-disk calibration cache is keyed by
# spec.cache_key (which folds in spec.seed) — different stages have distinct
# seed_offsets so each one calls into this loader at least once.
_SELF_TRACES_STATE: dict[str, dict] = {}


def _reset_self_traces_state() -> None:
    """Test affordance — clear the cached JSONL parses + blacklists. Production
    callers don't need this; the state lives for the lifetime of the process.
    """
    _SELF_TRACES_STATE.clear()


def _load_self_traces_state(path) -> dict:
    """Load + cache the JSONL once per path; partition rows by ``domain`` field
    and compute empirical domain weights from row counts. Initialise an empty
    per-domain blacklist map. Subsequent calls return the cached state (and
    its growing per-domain blacklists).
    """
    key = str(Path(path).resolve())
    cached = _SELF_TRACES_STATE.get(key)
    if cached is not None:
        return cached

    rows: list[dict] = []
    n_incomplete_filtered = 0
    n_parsed = 0
    # errors="replace" so a single malformed UTF-8 byte doesn't crash the
    # whole load (M3) — JSONDecodeError-per-line is already absorbed below.
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                log.warning("self-traces: skipping malformed JSONL line: %s", exc)
                continue
            n_parsed += 1
            # Completeness filter (schema_version >=4 produced by
            # build_self_traces_calib.py). A row is kept iff it lacks the
            # `_complete` field (legacy JSONLs without the flag are trusted)
            # OR has `_complete=true`. Truncated reasoning traces (no
            # `</think>` or no EOS) carry `_complete=false` and are dropped
            # here so Stage 2.5 router-KD never sees malformed chat-template
            # tails. Logged once per load as a summary line below.
            if row.get("_complete") is False:
                n_incomplete_filtered += 1
                continue
            rows.append(row)

    if not rows:
        if n_incomplete_filtered > 0:
            # Rows decoded successfully but were all dropped by the
            # `_complete=false` filter — the trace builder produced only
            # truncated reasoning. Re-running with more headroom recovers them.
            msg = (
                f"self-traces: JSONL at {path} produced zero usable rows "
                f"(out of {n_parsed} JSON-decoded, {n_incomplete_filtered} "
                "were filtered as `_complete=false`). Re-run the trace "
                "builder with a larger --max-new-tokens or oversample "
                "--num-prompts."
            )
        else:
            # No rows decoded at all — file is empty or every line failed
            # json.loads. Builder-rerun advice on token budget doesn't apply;
            # operator should verify the path and re-run the builder.
            msg = (
                f"self-traces: JSONL at {path} produced zero usable rows "
                f"({n_parsed} JSON-decoded). The file is empty or every "
                "line failed JSON decoding. Verify the path and re-run the "
                "trace builder from scratch."
            )
        raise ValueError(msg)

    if n_incomplete_filtered > 0:
        log.info(
            "self-traces: filtered %d/%d rows as `_complete=false` "
            "(truncated reasoning traces); %d kept for calibration.",
            n_incomplete_filtered, n_parsed, len(rows),
        )

    by_domain: dict[str, list[int]] = {}
    # N-D — build the inverted index ``domain_of`` once at load time so each
    # draw can look up an index's domain in O(1) rather than rebuilding the
    # O(total_rows) inverse from ``by_domain`` on every call.
    domain_of: dict[int, str] = {}
    for i, row in enumerate(rows):
        d = row.get("domain") or "unknown"
        by_domain.setdefault(d, []).append(i)
        domain_of[i] = d

    total = sum(len(v) for v in by_domain.values())
    weights = {d: len(v) / total for d, v in by_domain.items()}

    log.info(
        "self-traces: loaded %d rows from %s with empirical mix %s",
        len(rows), path,
        {d: f"{w:.2%}" for d, w in sorted(weights.items())},
    )

    # L2 — if NO row carried a ``domain`` field, the mix degenerates to a
    # single "unknown" bucket and every per-domain affordance becomes a no-op.
    # Warn loudly so the operator regenerates with a domain-tagged builder.
    if set(by_domain.keys()) == {"unknown"}:
        log.warning(
            "self-traces: every row at %s lacks a 'domain' field — the "
            "empirical mix degenerated to {'unknown': 1.0} and per-domain "
            "blacklist + quota mechanics are no-ops. Regenerate the JSONL "
            "with a domain-tagged build script (see "
            "max_quality/scripts/build_self_traces_calib.py).",
            path,
        )

    state = {
        "rows": rows,
        "by_domain": by_domain,
        # N-D — cached inverted index for O(1) row-index → domain lookup
        # in _draw_self_traces_indices (avoids rebuilding it per draw).
        "domain_of": domain_of,
        "weights": weights,
        # H2 — per-domain blacklist map; lazy-initialised in
        # _draw_self_traces_indices via setdefault(d, set()).
        "blacklist": {},
    }
    _SELF_TRACES_STATE[key] = state
    return state


def _draw_self_traces_indices(
    state: dict, n_requested: int, seed: int,
) -> list[int]:
    """Draw ``n_requested`` row-indices, respecting the empirical domain mix
    AND the per-domain blacklists.

    Algorithm:
      1. Allocate per-domain quotas via ``_distribute_counts`` so they sum
         exactly to ``n_requested`` (largest-remainder; M1 fix — the prior
         ``max(1, round(...))`` formulation drifted from n_requested and
         over-represented tiny-weight domains at small ``n``).
      2. For each domain, sample ``quota_d`` indices from
         ``by_domain[d]`` after removing that domain's own blacklist set
         (deterministic shuffle, seeded by ``spec.seed`` + stable md5(d)).
      3. If a domain can't fill its quota out of its un-blacklisted pool,
         reset ONLY that domain's blacklist (per-domain contract: a small
         domain exhausting MUST NOT whitelist still-flush domains' un-served
         rows) and retry step 2 once. If the domain's TOTAL pool is smaller
         than its quota, take what's available and log a warning that names
         the offending domain(s).
      4. Update each domain's blacklist with the indices selected from it
         before returning.
    """
    import random as _random

    by_domain = state["by_domain"]
    weights = state["weights"]
    blacklist: dict[str, set[int]] = state["blacklist"]

    # Per-domain quotas via largest-remainder so they sum exactly to
    # n_requested (M1). _distribute_counts can emit zero-quota entries for
    # tiny weights; that's intentional (avoids forcing one row from a domain
    # with negligible weight at small n).
    per_domain_quota = _distribute_counts(n_requested, weights)

    def _domain_rng(d: str) -> _random.Random:
        # PEP 456 randomises ``hash(str)`` per interpreter invocation, so the
        # previous ``hash(d)`` formulation broke the "deterministic shuffle"
        # promise across processes. md5 is stable + matches the offset hash
        # pattern used by ``_make_subset_seed`` (lines 1018-1022).
        domain_offset = int(
            hashlib.md5(d.encode(), usedforsecurity=False).hexdigest(), 16
        ) & 0x7FFFFFFF
        return _random.Random((seed * 1009 + domain_offset) & 0xFFFFFFFF)

    def _try_draw(
        domain_subsets: dict[str, list[int]],
    ) -> tuple[list[int], list[str]]:
        """Returns (selected, shortfall_domains). A non-empty
        ``shortfall_domains`` flags which domain(s) need a per-domain
        blacklist reset (H2 contract: blacklists are PER-DOMAIN; we whitelist
        only the offending domain, not the global pool)."""
        selected_local: list[int] = []
        shortfall: list[str] = []
        for d, quota in per_domain_quota.items():
            pool = domain_subsets.get(d, [])
            if len(pool) < quota:
                shortfall.append(d)
                continue
            rng = _domain_rng(d)
            shuffled = list(pool)
            rng.shuffle(shuffled)
            selected_local.extend(shuffled[:quota])
        return selected_local, shortfall

    # First try: filter each domain pool by ITS OWN blacklist (H2 — per-
    # domain blacklists; a small domain exhausting must not whitelist the
    # rest of the run).
    def _available(blist_map: dict[str, set[int]]) -> dict[str, list[int]]:
        # N-A — ``get`` keeps this a pure read; ``setdefault`` would mutate
        # ``blist_map`` mid-read by inserting empty-set entries for every
        # domain. Harmless (empty sets are idempotent for the ``not in``
        # filter) but clearer to avoid the side effect. ``get`` returns a
        # fresh local empty set for absent keys — no shared-state risk.
        return {
            d: [i for i in idx_list
                if i not in blist_map.get(d, set())]
            for d, idx_list in by_domain.items()
        }

    available = _available(blacklist)
    selected, shortfall = _try_draw(available)

    if shortfall:
        # PER-DOMAIN reset: only the domain(s) that exhausted get whitelisted.
        # Original contract ("only when we blacklist all samples do we whitelist
        # everything") still holds — it just applies per-domain so that no
        # un-served row from a still-flush domain is recycled prematurely.
        for d in shortfall:
            d_total = len(by_domain.get(d, []))
            d_used = len(blacklist.get(d, set()))
            log.info(
                "self-traces: domain %r blacklist exhausted (%d/%d consumed) — "
                "resetting that domain's blacklist and re-drawing it.",
                d, d_used, d_total,
            )
            blacklist[d] = set()
        available = _available(blacklist)
        selected, shortfall = _try_draw(available)
        if shortfall:
            # Even with a fresh per-domain blacklist some domain has fewer rows
            # than its quota — JSONL is undersized for this n_requested at the
            # current mix. Take what's available and name the offenders so the
            # operator knows which domain(s) to grow.
            undersized = [
                (d, len(by_domain.get(d, [])), per_domain_quota[d])
                for d in shortfall
            ]
            details = ", ".join(
                f"{d}={n_rows} rows vs quota {q}"
                for d, n_rows, q in undersized
            )
            log.warning(
                "self-traces: JSONL undersized — domain(s) short of quota even "
                "after per-domain blacklist reset: %s. Regenerate with "
                "--num-prompts large enough that every shortfall domain "
                "exceeds its quota (per-domain bottleneck — global "
                "n_requested*2 isn't sufficient when one domain has small "
                "weight). Filling with all available rows per domain (mix "
                "will be approximated, not exact).",
                details,
            )
            # N-B — rebuilding ``selected`` from scratch here repeats the
            # shuffles ``_try_draw`` already performed for non-shortfall
            # domains. Because ``_domain_rng(d)`` is deterministic per
            # (seed, domain) and non-shortfall blacklists haven't moved
            # since the second ``_try_draw``, the rebuilt selection
            # coincides with what ``_try_draw`` would have produced —
            # the redundant shuffle work is cheap and only fires on this
            # rare undersized-JSONL fallback path. Trade-off: clearer
            # code over reusing the prior partial selection.
            selected = []
            for d, quota in per_domain_quota.items():
                pool = available.get(d, [])
                rng = _domain_rng(d)
                shuffled = list(pool)
                rng.shuffle(shuffled)
                selected.extend(shuffled[:quota])

    # Update each domain's blacklist with the indices we just served.
    # N-D — reuse the cached inverted index from _load_self_traces_state
    # (built once at load time) instead of rebuilding the O(total_rows)
    # inverse on every draw.
    domain_of: dict[int, str] = state["domain_of"]
    selected_by_domain: dict[str, set[int]] = {}
    for i in selected:
        d = domain_of.get(i)
        if d is None:
            continue
        selected_by_domain.setdefault(d, set()).add(i)
    for d, ids in selected_by_domain.items():
        blacklist.setdefault(d, set()).update(ids)
    return selected


def _parse_yaml_self_traces(
    cal_cfg: dict, num_sequences: int, sequence_length: int, seed: int,
) -> CalibrationSpec:
    jsonl_path = str(cal_cfg.get("jsonl_path", _DEFAULT_SELF_TRACES_PATH))
    return CalibrationSpec(
        num_sequences=num_sequences,
        sequence_length=sequence_length,
        seed=seed,
        source="self-traces",
        dataset=jsonl_path,
    )


def _stream_texts_self_traces(
    spec: CalibrationSpec, tokenizer
) -> list[str]:
    """Stream rows from the local JSONL produced by build_self_traces_calib.py.

    Each row carries ``{"messages": [...], "domain": "..."}`` — the loader
    partitions by ``domain``, preserves the empirical domain mix at every
    draw (so a 1000-sample request returns the same percentage split as a
    5000-sample request), and runs PER-DOMAIN row-index blacklists so no
    sample is served twice within one ``run_pipeline.py`` invocation. Each
    domain's blacklist resets independently only when THAT domain can no
    longer satisfy its quota — a tiny domain exhausting does NOT recycle
    still-flush domains' un-served rows. See ``_draw_self_traces_indices``
    for the full contract.

    Path resolution note (M2): a relative ``spec.dataset`` is resolved
    against ``Path.cwd()`` — the pipeline driver invokes from the repo
    root, so the default ``artifacts/_shared/self_traces.jsonl`` lands
    correctly. Operators running from a different directory should set
    ``calibration.jsonl_path`` to an absolute path in YAML to avoid
    relative-path surprises.

    Each selected row's ``messages`` list (already containing the teacher's
    reasoning + answer trace in the assistant turn) is rendered through the
    model's chat template with ``enable_thinking=True`` so reasoning-mode
    tokenizers (Qwen3-thinking, DeepSeek-R1, etc.) keep the in-block markers
    in the output token stream.
    """
    path = Path(spec.dataset)
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        teacher_hint = getattr(tokenizer, "name_or_path", None) or "<TEACHER_REPO>"
        raise FileNotFoundError(
            "self-traces calibration JSONL not found at "
            f"{path}.\n\n"
            "This is the FIRST step for any new model. Generate the traces "
            "with the teacher you want to distil against:\n\n"
            "    python max_quality/scripts/build_self_traces_calib.py \\\n"
            f"        --teacher {teacher_hint} \\\n"
            "        --num-prompts 5000 --max-new-tokens 4096 \\\n"
            f"        --output {path}\n\n"
            "Then re-run the pipeline. The trace JSONL is reusable across "
            "every Stage-2 / 2.5 run for this teacher (regenerate only when "
            "the teacher revision or prompt set changes)."
        )

    state = _load_self_traces_state(path)
    selected = _draw_self_traces_indices(
        state, n_requested=spec.num_sequences, seed=spec.seed,
    )

    out: list[str] = []
    rows = state["rows"]
    for i in selected:
        messages = rows[i].get("messages")
        if not messages:
            continue
        rendered = _render_messages(
            messages, tokenizer,
            enable_thinking=True,
            add_generation_prompt=False,
        )
        if rendered:
            out.append(rendered)

    blacklist_total = sum(len(s) for s in state["blacklist"].values())
    per_domain_bl = {
        d: f"{len(state['blacklist'].get(d, set()))}/{len(state['by_domain'][d])}"
        for d in sorted(state["by_domain"].keys())
    }
    log.info(
        "self-traces: served %d rows from %s (blacklist=%d/%d total, "
        "per-domain=%s, seed=%d, requested=%d)",
        len(out), path,
        blacklist_total, len(state["rows"]),
        per_domain_bl,
        spec.seed, spec.num_sequences,
    )
    return out


register_corpus(CorpusAdapter(
    name="self-traces",
    parse_yaml=_parse_yaml_self_traces,
    stream_texts=_stream_texts_self_traces,
))
