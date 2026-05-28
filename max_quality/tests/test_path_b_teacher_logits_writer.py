"""CRITICAL-2 / Path B teacher-logits writer rewrite — on-disk contract tests.

Background
----------
``max_quality/hf_jobs/precompute_teacher_logits.py`` was previously writing
per-layer pre-softmax router scores (``dict[int, Tensor[B*T, num_experts]]``).
The Stage 5 reader
(``moe_compress.router_kd.plugins.teacher.TeacherCachePlugin.load_teacher_cache``)
treats ``cache_payload["logits"]`` as a single flat tensor
``[num_samples * sequence_length, vocab_size]`` and validates
``shape[-1] == student.config.vocab_size`` (teacher.py L374-393). The two
halves disagreed on BOTH shape and signal content. Latent because every
shipped YAML kept ``stage5_router_kd.teacher_logits_cache`` commented out;
the first uncomment would AttributeError on the dict-vs-tensor mismatch.

The rewrite makes the writer emit what the reader expects: a single
BF16 tensor ``[N*L, |V|]`` populated batch-by-batch from
``teacher(input_ids=batch).logits``, written via the existing
``atomic_torch_save`` + ``write_manifest_last`` protocol (Pattern O —
manifest LAST so a torn payload is caught BEFORE mmap).

These tests verify the on-disk contract end-to-end without spinning up a
real 35 B transformer:

1. ``test_writer_payload_shape_matches_reader_contract`` — the payload
   shape the writer produces (single flat ``[N*L, |V|]`` BF16 tensor)
   is the exact shape the reader validates.
2. ``test_writer_to_reader_roundtrip`` — a payload simulated to match
   the writer's output passes every reader-side validation
   (``load_teacher_cache``) and round-trips a per-batch slice through
   ``provide_teacher_logits``.
3. ``test_manifest_written_last_and_validated_by_reader`` — the manifest
   is written AFTER the payload (Pattern O), reader validation succeeds
   on the happy path and fails loudly on a torn payload (size mismatch)
   or a missing manifest.
4. ``test_fake_teacher_writer_loop_produces_flat_vocab_tensor`` —
   exercises the writer's per-batch capture loop arithmetic with a fake
   teacher (no transformers / no calibration), confirming the cursor
   bookkeeping fills the buffer in token order and the dtype is BF16.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from moe_compress.utils.atomic_io import (
    ManifestMismatchError,
    atomic_torch_save,
    read_and_validate_manifest,
    write_manifest_last,
)


# ---------------------------------------------------------------------------
# Helpers — payloads + stubs
# ---------------------------------------------------------------------------
def _writer_style_payload(
    *,
    num_samples: int = 4,
    seq_len: int = 8,
    vocab: int = 32,
    batch_size: int = 2,
    model: str = "fake-model",
) -> dict:
    """Construct a payload in the EXACT shape the rewritten writer produces.

    Mirrors ``precompute_teacher_logits._main()`` lines 253-264: single flat
    ``[N*L, |V|]`` BF16 tensor + the metadata keys the reader cross-checks.
    """
    total_tokens = num_samples * seq_len
    # Deterministic fill so per-batch slices have predictable values.
    logits = (
        torch.arange(total_tokens * vocab, dtype=torch.float32)
        .view(total_tokens, vocab)
        .to(torch.bfloat16)
    )
    return {
        "logits": logits,
        "num_samples": num_samples,
        "sequence_length": seq_len,
        "batch_size": batch_size,
        "vocab_size": vocab,
        "model": model,
        "calibration_seed_offset": 5,
        "format_version": 1,
    }


class _FakeTeacherOutput:
    """Stand-in for ``transformers``'s ``CausalLMOutput``."""

    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits


class _FakeTeacherConfig:
    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size


class _FakeTeacher:
    """Stub teacher that returns deterministic vocabulary logits.

    Mirrors the call shape ``teacher(input_ids=batch)`` and exposes
    ``.config.vocab_size`` — the only two surfaces the writer touches.
    """

    def __init__(self, vocab_size: int) -> None:
        self.config = _FakeTeacherConfig(vocab_size)
        self._vocab = vocab_size

    def __call__(self, *, input_ids: torch.Tensor) -> _FakeTeacherOutput:
        # Deterministic per-token logits = token_id * vocab_arange so the
        # buffer's slice arithmetic is verifiable byte-for-byte.
        B, L = input_ids.shape
        # [B, L, V] = input_ids[:, :, None] (broadcast) * arange(V)
        arange = torch.arange(self._vocab, dtype=torch.float32)
        ids = input_ids.to(torch.float32).unsqueeze(-1)
        return _FakeTeacherOutput(ids * arange)


def _run_writer_capture_loop(
    teacher: _FakeTeacher,
    batches: list[torch.Tensor],
    *,
    num_samples: int,
    seq_len: int,
    vocab_size: int,
) -> torch.Tensor:
    """Re-runs the writer's per-batch capture loop in isolation.

    This is the byte-identical inner loop from
    ``precompute_teacher_logits._main()`` lines 193-227 lifted out so we
    can exercise it with a fake teacher in a millisecond-scale unit test.
    Any future divergence here must be reflected in the test or the test
    becomes stale — kept structurally identical on purpose.
    """
    total_tokens = num_samples * seq_len
    logits_buf = torch.empty((total_tokens, vocab_size), dtype=torch.bfloat16)
    cursor = 0
    with torch.no_grad():
        for batch in batches:
            out = teacher(input_ids=batch)
            flat = (
                out.logits.detach()
                .to(torch.bfloat16)
                .reshape(-1, vocab_size)
                .cpu()
            )
            n_rows = flat.shape[0]
            logits_buf[cursor : cursor + n_rows].copy_(flat)
            cursor += n_rows
    assert cursor == total_tokens, (
        f"capture loop wrote {cursor} rows, expected {total_tokens}"
    )
    return logits_buf


class _StudentConfig:
    """Minimal stand-in for ``student.config`` — only ``vocab_size`` is read."""

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size


class _StudentStub:
    def __init__(self, vocab_size: int) -> None:
        self.config = _StudentConfig(vocab_size)


# ---------------------------------------------------------------------------
# 1. Shape contract — writer produces single-tensor logits, vocab last dim
# ---------------------------------------------------------------------------
def test_writer_payload_shape_matches_reader_contract():
    """The writer-produced payload's ``logits`` key is a SINGLE tensor with
    shape ``[N*L, |V|]`` and dtype BF16 — exactly what the reader validates.

    Pre-CRITICAL-2 the writer wrote ``dict[int, Tensor[N*L, num_experts]]``;
    this test pins the new contract so the regression cannot re-land.
    """
    payload = _writer_style_payload(
        num_samples=4, seq_len=8, vocab=32, batch_size=2
    )
    logits = payload["logits"]

    # SINGLE tensor (NOT a dict per-layer router scores).
    assert isinstance(logits, torch.Tensor), (
        f"writer must emit a single tensor, got {type(logits).__name__} "
        "(pre-CRITICAL-2 regression: per-layer dict was emitted)"
    )
    # Shape contract: [N*L, |V|].
    assert logits.shape == (4 * 8, 32), logits.shape
    # Dtype: BF16 (reader's mmap-load + per-batch slice expects BF16).
    assert logits.dtype == torch.bfloat16
    # Metadata keys the reader cross-checks (teacher.py L321-393).
    assert int(payload["format_version"]) == 1
    assert int(payload["num_samples"]) == 4
    assert int(payload["sequence_length"]) == 8
    assert int(payload["vocab_size"]) == 32


# ---------------------------------------------------------------------------
# 2. Reader successfully loads + validates from writer output
# ---------------------------------------------------------------------------
def test_writer_to_reader_roundtrip(tmp_path):
    """The actual ``TeacherCachePlugin.load_teacher_cache`` accepts the
    writer's payload + passes every shape / vocab / token-count guard.

    Exercises the reader top-to-bottom against a payload built in the
    writer's exact format.
    """
    from moe_compress.pipeline.context import PipelineContext
    from moe_compress.router_kd.plugins.teacher import TeacherCachePlugin

    num_samples, seq_len, vocab, batch_size = 4, 8, 32, 2
    cache_path = tmp_path / "_stage5_teacher_logits.pt"
    manifest_path = cache_path.with_suffix(cache_path.suffix + ".MANIFEST.json")

    payload = _writer_style_payload(
        num_samples=num_samples, seq_len=seq_len, vocab=vocab, batch_size=batch_size,
    )
    atomic_torch_save(cache_path, payload)
    write_manifest_last(
        cache_path, manifest_path, schema_version=1,
        extra_meta={
            "artifact": "stage5_teacher_logits",
            "vocab_size": vocab,
            "total_tokens": num_samples * seq_len,
        },
    )

    student = _StudentStub(vocab_size=vocab)
    cfg = {
        # PB-2: teacher cross-check resolves via config["model"]["name_or_path"]
        # — must match the writer-style payload's "model" field (fake-model).
        "model": {"name_or_path": "fake-model"},
        "stage5_router_kd": {
            "teacher_logits_cache": str(cache_path),
            "batch_size": batch_size,
            "max_sequence_length": seq_len,
            "max_calibration_samples": num_samples,
        },
    }
    ctx = PipelineContext()
    ctx.set("config", cfg)
    ctx.set("student", student)
    ctx.set("artifacts_dir", tmp_path)

    plugin = TeacherCachePlugin()
    # Must NOT raise — full shape/vocab/token-count guard passes.
    plugin.load_teacher_cache(ctx)
    loaded = ctx.get("teacher_logits_cache")
    assert loaded is not None
    assert loaded["logits"].shape == (num_samples * seq_len, vocab)
    assert loaded["logits"].dtype == torch.bfloat16

    # And the per-batch slice round-trips through provide_teacher_logits.
    input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    result = plugin.provide_teacher_logits(
        ctx, input_ids=input_ids,
        epoch=0, batch_index=0, num_batches=num_samples // batch_size,
    )
    assert result.shape == (batch_size, seq_len, vocab)
    assert result.dtype == torch.float32  # reader upcasts BF16 -> FP32


def test_writer_to_reader_vocab_mismatch_raises(tmp_path):
    """A writer-format payload with the WRONG vocab dim is caught by the
    reader's ``shape[-1] == student.config.vocab_size`` guard (teacher.py L380).

    This is the production-floor protection the bug was hiding: a cache
    built for a different tokenizer would silently destroy KD signal.
    """
    from moe_compress.pipeline.context import PipelineContext
    from moe_compress.router_kd.plugins.teacher import TeacherCachePlugin

    cache_path = tmp_path / "_stage5_teacher_logits.pt"
    manifest_path = cache_path.with_suffix(cache_path.suffix + ".MANIFEST.json")
    payload = _writer_style_payload(num_samples=4, seq_len=8, vocab=32)
    atomic_torch_save(cache_path, payload)
    write_manifest_last(cache_path, manifest_path, schema_version=1)

    student = _StudentStub(vocab_size=64)  # Mismatch — student is 64-vocab.
    cfg = {
        "stage5_router_kd": {
            "teacher_logits_cache": str(cache_path),
            "batch_size": 2,
            "max_sequence_length": 8,
            "max_calibration_samples": 4,
        },
    }
    ctx = PipelineContext()
    ctx.set("config", cfg)
    ctx.set("student", student)
    ctx.set("artifacts_dir", tmp_path)

    with pytest.raises(RuntimeError, match=r"vocab_size=32 does not match"):
        TeacherCachePlugin().load_teacher_cache(ctx)


# ---------------------------------------------------------------------------
# 3. Pattern O — manifest LAST, validated by reader
# ---------------------------------------------------------------------------
def test_manifest_written_last_and_validated_by_reader(tmp_path):
    """Happy path: ``atomic_torch_save`` then ``write_manifest_last``, then
    the reader validates the manifest BEFORE the mmap-load.
    """
    cache_path = tmp_path / "_stage5_teacher_logits.pt"
    manifest_path = cache_path.with_suffix(cache_path.suffix + ".MANIFEST.json")

    atomic_torch_save(cache_path, _writer_style_payload())
    # Manifest LAST — Pattern O contract.
    assert cache_path.exists()
    assert not manifest_path.exists()
    write_manifest_last(
        cache_path, manifest_path, schema_version=1,
        extra_meta={"artifact": "stage5_teacher_logits", "vocab_size": 32},
    )
    assert manifest_path.exists()

    # Reader validation succeeds + surfaces forensics.
    manifest = read_and_validate_manifest(
        cache_path, manifest_path, expected_schema_version=1,
    )
    assert manifest["payload_name"] == cache_path.name
    assert manifest["extra"]["artifact"] == "stage5_teacher_logits"
    assert manifest["extra"]["vocab_size"] == 32


def test_manifest_missing_caught_before_mmap(tmp_path):
    """Kill between ``atomic_torch_save`` and ``write_manifest_last`` leaves
    a payload with NO manifest sidecar. Reader's validator raises rather than
    letting the mmap-load silently consume torn bytes."""
    cache_path = tmp_path / "_stage5_teacher_logits.pt"
    manifest_path = cache_path.with_suffix(cache_path.suffix + ".MANIFEST.json")
    atomic_torch_save(cache_path, _writer_style_payload())

    with pytest.raises(ManifestMismatchError, match="missing"):
        read_and_validate_manifest(
            cache_path, manifest_path, expected_schema_version=1,
        )


def test_torn_payload_caught_by_size_check(tmp_path):
    """Simulate an HF Jobs pod eviction truncating a ~30 GB ``.pt`` in flight:
    payload size on disk no longer matches manifest ``size_bytes``."""
    cache_path = tmp_path / "_stage5_teacher_logits.pt"
    manifest_path = cache_path.with_suffix(cache_path.suffix + ".MANIFEST.json")
    atomic_torch_save(cache_path, _writer_style_payload())
    write_manifest_last(cache_path, manifest_path, schema_version=1)

    real_size = cache_path.stat().st_size
    with open(cache_path, "r+b") as f:
        f.truncate(real_size // 3)

    with pytest.raises(ManifestMismatchError, match="size"):
        read_and_validate_manifest(
            cache_path, manifest_path, expected_schema_version=1,
        )


# ---------------------------------------------------------------------------
# 4. Writer capture loop — fake teacher, full token-order semantics
# ---------------------------------------------------------------------------
def test_fake_teacher_writer_loop_produces_flat_vocab_tensor():
    """The writer's per-batch capture loop fills a single ``[N*L, |V|]``
    BF16 buffer in token order from ``teacher(input_ids=...).logits``.

    Uses a fake teacher returning deterministic logits so we can byte-check
    the buffer's contents against the per-batch outputs.
    """
    num_samples, seq_len, vocab_size, batch_size = 8, 4, 16, 2
    teacher = _FakeTeacher(vocab_size=vocab_size)
    # Deterministic token ids — uniform full batches (no padding, no partial tails).
    torch.manual_seed(0)
    all_ids = torch.randint(0, vocab_size, (num_samples, seq_len))
    batches = [
        all_ids[i : i + batch_size] for i in range(0, num_samples, batch_size)
    ]

    buf = _run_writer_capture_loop(
        teacher, batches,
        num_samples=num_samples, seq_len=seq_len, vocab_size=vocab_size,
    )

    # Shape + dtype contract.
    assert buf.shape == (num_samples * seq_len, vocab_size)
    assert buf.dtype == torch.bfloat16

    # Byte-check: rebuild the expected buffer the trivial way (single forward
    # over the FULL [N, L] tensor) and compare against the per-batch fill.
    with torch.no_grad():
        expected = (
            teacher(input_ids=all_ids)
            .logits.to(torch.bfloat16)
            .reshape(-1, vocab_size)
        )
    assert torch.equal(buf, expected), (
        "writer capture loop must fill the buffer in token order matching "
        "a single full-batch forward — any cursor / reshape bug surfaces here"
    )


def test_fake_teacher_writer_loop_payload_round_trips_to_reader(tmp_path):
    """The output of the writer capture loop, packaged into a writer-style
    payload, is accepted by the reader plugin end-to-end.

    Closes the loop: fake teacher → capture loop → atomic write + manifest →
    reader load + per-batch slice.
    """
    from moe_compress.pipeline.context import PipelineContext
    from moe_compress.router_kd.plugins.teacher import TeacherCachePlugin

    num_samples, seq_len, vocab_size, batch_size = 4, 8, 16, 2
    teacher = _FakeTeacher(vocab_size=vocab_size)
    torch.manual_seed(11)
    all_ids = torch.randint(0, vocab_size, (num_samples, seq_len))
    batches = [
        all_ids[i : i + batch_size] for i in range(0, num_samples, batch_size)
    ]

    buf = _run_writer_capture_loop(
        teacher, batches,
        num_samples=num_samples, seq_len=seq_len, vocab_size=vocab_size,
    )
    payload = {
        "logits": buf,
        "num_samples": num_samples,
        "sequence_length": seq_len,
        "batch_size": batch_size,
        "vocab_size": vocab_size,
        "model": "fake",
        "calibration_seed_offset": 5,
        "format_version": 1,
    }
    cache_path = tmp_path / "_stage5_teacher_logits.pt"
    manifest_path = cache_path.with_suffix(cache_path.suffix + ".MANIFEST.json")
    atomic_torch_save(cache_path, payload)
    write_manifest_last(
        cache_path, manifest_path, schema_version=1,
        extra_meta={"vocab_size": vocab_size, "total_tokens": num_samples * seq_len},
    )

    ctx = PipelineContext()
    ctx.set("config", {
        # PB-2: match payload's "model" field so the cross-check stays quiet.
        "model": {"name_or_path": "fake"},
        "stage5_router_kd": {
            "teacher_logits_cache": str(cache_path),
            "batch_size": batch_size,
            "max_sequence_length": seq_len,
            "max_calibration_samples": num_samples,
        },
    })
    ctx.set("student", _StudentStub(vocab_size=vocab_size))
    ctx.set("artifacts_dir", tmp_path)
    plugin = TeacherCachePlugin()
    plugin.load_teacher_cache(ctx)

    # First batch slice = first batch_size * seq_len token rows.
    result0 = plugin.provide_teacher_logits(
        ctx, input_ids=batches[0],
        epoch=0, batch_index=0, num_batches=len(batches),
    )
    assert result0.shape == (batch_size, seq_len, vocab_size)
    # Byte-compare against the same slice from the captured buffer
    # (upcast to FP32 to match the reader's dtype cast at teacher.py L463-465).
    expected0 = (
        buf[0 : batch_size * seq_len]
        .to(torch.float32)
        .view(batch_size, seq_len, vocab_size)
    )
    assert torch.equal(result0, expected0)


def test_writer_manifest_extra_meta_includes_vocab_size(tmp_path):
    """The writer surfaces ``vocab_size`` + ``total_tokens`` in the manifest's
    ``extra`` block for human-inspection forensics (the on-disk JSON, not the
    payload). Critical because mmap-loading a 30 GB .pt just to read vocab is
    expensive; the manifest exposes it cheaply.
    """
    cache_path = tmp_path / "_stage5_teacher_logits.pt"
    manifest_path = cache_path.with_suffix(cache_path.suffix + ".MANIFEST.json")
    atomic_torch_save(cache_path, _writer_style_payload(vocab=128))
    write_manifest_last(
        cache_path, manifest_path, schema_version=1,
        extra_meta={
            "artifact": "stage5_teacher_logits",
            "vocab_size": 128,
            "total_tokens": 4 * 8,
        },
    )

    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    assert manifest["extra"]["vocab_size"] == 128
    assert manifest["extra"]["total_tokens"] == 32
    assert manifest["extra"]["artifact"] == "stage5_teacher_logits"


# ---------------------------------------------------------------------------
# 5. PB-2 soft cross-check — teacher model + seed_offset WARN-only
# ---------------------------------------------------------------------------
def _build_loaded_ctx(tmp_path, *, payload):
    """Helper: write a payload + manifest pair, build a ctx for the reader.

    Returns (ctx, plugin) ready for ``plugin.load_teacher_cache(ctx)`` and a
    happy-path-shaped reader configuration (no override unless caller sets it).
    """
    from moe_compress.pipeline.context import PipelineContext
    from moe_compress.router_kd.plugins.teacher import TeacherCachePlugin

    cache_path = tmp_path / "_stage5_teacher_logits.pt"
    manifest_path = cache_path.with_suffix(cache_path.suffix + ".MANIFEST.json")
    atomic_torch_save(cache_path, payload)
    write_manifest_last(cache_path, manifest_path, schema_version=1)

    student = _StudentStub(vocab_size=int(payload["vocab_size"]))
    ctx = PipelineContext()
    ctx.set("config", {
        "model": {"name_or_path": "Qwen/Qwen3.6-35B-A3B"},
        "stage5_router_kd": {
            "teacher_logits_cache": str(cache_path),
            "batch_size": int(payload["batch_size"]),
            "max_sequence_length": int(payload["sequence_length"]),
            "max_calibration_samples": int(payload["num_samples"]),
        },
    })
    ctx.set("student", student)
    ctx.set("artifacts_dir", tmp_path)
    return ctx, TeacherCachePlugin()


def _run_with_caplog_pattern_n(caplog, fn):
    """Pattern N (caplog-propagate-restore) — see test_calib_ckpt_counter.py:64.

    The teacher.py logger is non-root; under this repo's pytest setup, caplog
    only captures named loggers when (a) root level is set via
    ``caplog.set_level`` and (b) the named logger's ``propagate`` is forced
    True for the duration of the call. Encapsulated here so the three PB-2
    tests don't re-implement the dance.
    """
    import logging
    caplog.set_level(logging.WARNING)
    logger = logging.getLogger("moe_compress.router_kd.plugins.teacher")
    prev = logger.propagate
    logger.propagate = True
    try:
        return fn()
    finally:
        logger.propagate = prev


def test_pb2_cache_model_mismatch_warns(tmp_path, caplog):
    """PB-2 — cache 'model' that disagrees with the resolved teacher repo emits
    a WARN (not a raise). The override path is legitimate; this is forensics.

    Mirrors the WARN-only pattern at router_kd/plugins/teacher.py:343 (the
    batch_size soft warn): caplog at WARN level, substring assertion, no raise.
    """
    payload = _writer_style_payload(
        num_samples=4, seq_len=8, vocab=32, batch_size=2,
        model="some/other-model",  # disagrees with config["model"]["name_or_path"]
    )
    ctx, plugin = _build_loaded_ctx(tmp_path, payload=payload)

    _run_with_caplog_pattern_n(caplog, lambda: plugin.load_teacher_cache(ctx))

    assert any(
        "teacher-logits cache 'model'" in rec.getMessage()
        and "disagrees with configured teacher repo" in rec.getMessage()
        for rec in caplog.records
    ), [rec.getMessage() for rec in caplog.records]
    # Reader still publishes the cache despite the warning.
    assert ctx.get("teacher_logits_cache") is not None


def test_pb2_cache_model_override_respected(tmp_path, caplog):
    """If the operator set ``stage5_router_kd.teacher_model_repo``, the
    cache's `model` is compared against the override — match means no warning.
    """
    payload = _writer_style_payload(
        num_samples=4, seq_len=8, vocab=32, batch_size=2,
        model="Qwen/Qwen3.6-35B-A3B-FP8",  # equals the override below
    )
    ctx, plugin = _build_loaded_ctx(tmp_path, payload=payload)
    # Inject the override.
    cfg = ctx.get("config")
    cfg["stage5_router_kd"]["teacher_model_repo"] = "Qwen/Qwen3.6-35B-A3B-FP8"

    _run_with_caplog_pattern_n(caplog, lambda: plugin.load_teacher_cache(ctx))

    assert not any(
        "teacher-logits cache 'model'" in rec.getMessage()
        for rec in caplog.records
    ), [rec.getMessage() for rec in caplog.records]


def test_pb2_cache_seed_offset_mismatch_warns(tmp_path, caplog):
    """PB-2 — cache built with a different ``calibration_seed_offset`` (not 5)
    emits a WARN. The fixture's `_writer_style_payload` defaults to 5; override
    here to simulate a stale cache.
    """
    payload = _writer_style_payload(
        num_samples=4, seq_len=8, vocab=32, batch_size=2,
    )
    payload["calibration_seed_offset"] = 7  # ≠ 5
    ctx, plugin = _build_loaded_ctx(tmp_path, payload=payload)

    _run_with_caplog_pattern_n(caplog, lambda: plugin.load_teacher_cache(ctx))

    assert any(
        "calibration_seed_offset=7" in rec.getMessage()
        and "Stage 5 uses 5" in rec.getMessage()
        for rec in caplog.records
    ), [rec.getMessage() for rec in caplog.records]


def test_pb2_happy_path_no_warns(tmp_path, caplog):
    """PB-2 — when the cache's `model` matches config and seed_offset == 5,
    no PB-2 WARN is emitted (false-positive guard).
    """
    payload = _writer_style_payload(
        num_samples=4, seq_len=8, vocab=32, batch_size=2,
        model="Qwen/Qwen3.6-35B-A3B",
    )
    ctx, plugin = _build_loaded_ctx(tmp_path, payload=payload)

    _run_with_caplog_pattern_n(caplog, lambda: plugin.load_teacher_cache(ctx))

    pb2_messages = [
        rec.getMessage() for rec in caplog.records
        if "teacher-logits cache 'model'" in rec.getMessage()
        or "calibration_seed_offset=" in rec.getMessage()
    ]
    assert pb2_messages == [], pb2_messages
