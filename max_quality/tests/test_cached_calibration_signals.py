"""Tests for ``cached_calibration_signals`` -- provider-pair infrastructure.

Covers the schema/atomic-write/load API in
``moe_compress.utils.cached_calibration_signals``:

1. ``sidecar_path`` derivation for both atomic single-file signals and
   per-shard signals (with the ``block_hidden/layer_NNNN`` subpath).
2-7. Round-trip save+load for each of the 6 dataclass payloads. Each
   round-trip verifies tensor-field equality (or ndarray equality), that
   tensor fields land on CPU after a load, and that the sidecar lives at
   the documented path under ``<jsonl_path.parent>/sidecars/``.
8. Schema-version-mismatch: bumping ``SCHEMA_VERSIONS[...]`` after a
   sidecar exists makes the matching ``load_*`` raise ``ValueError``
   carrying the actionable "Delete the sidecar to regenerate" message.
9. Atomic-write crash safety: monkeypatching ``torch.save`` to raise
   mid-write leaves the previous ``final_path`` intact and the ``.tmp``
   file may exist (orphan) but is not promoted via ``os.replace``.
10. Provider-pair ``dispatch_first`` behaviour: cache-hit short-circuits
    to the cache provider; cache-miss falls through to the live provider
    which both writes the sidecar AND populates the ctx slot; the
    Signal-3 partial-cache scenario exercises a cache provider that only
    populates ``ctx.set("sigma_in_cached", ...)`` and falls through to a
    live provider that adds ``ctx.set("C_cross_cov", ...)``.

CPU-only by construction (every tensor allocation defaults to CPU).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from moe_compress.pipeline.context import PipelineContext
from moe_compress.pipeline.registry import PluginRegistry
from moe_compress.utils.cached_calibration_signals import (
    SCHEMA_VERSIONS,
    BaseCacheProvider,
    BaseLiveProvider,
    BlockHiddenPayload,
    CovariancePayload,
    PhaseBPayload,
    RouterKDLogitsPayload,
    RoutingStatsPayload,
    Stage1PerExpertMaxPayload,
    Stage2ProfilePayload,
    Stage2ReapPayload,
    TeacherEvalPayload,
    load_block_hidden,
    load_covariance,
    load_per_expert_max,
    load_phase_b,
    load_reap_scores,
    load_routing_stats,
    load_router_kd_logits,
    load_stage2_profile,
    load_teacher_eval,
    router_kd_logits_dir,
    save_block_hidden,
    save_covariance,
    save_per_expert_max,
    save_phase_b,
    save_reap_scores,
    save_routing_stats,
    save_router_kd_logits,
    save_stage2_profile,
    save_teacher_eval,
    sidecar_path,
)


# ---------------------------------------------------------------------------
# Payload factories -- small deterministic shapes; explicit dtypes so the
# load side has something concrete to assert against.
# ---------------------------------------------------------------------------
def _make_phase_b(n_layers: int = 2, n_experts: int = 3) -> PhaseBPayload:
    return PhaseBPayload(
        schema_version=SCHEMA_VERSIONS["phase_b"],
        n_experts=n_experts,
        n_layers=n_layers,
        per_expert_max=torch.arange(
            n_layers * n_experts, dtype=torch.float32
        ).reshape(n_layers, n_experts),
        routing_freq=torch.full((n_layers, n_experts), 0.25, dtype=torch.float32),
        mean_routing_weight=torch.full(
            (n_layers, n_experts), 0.5, dtype=torch.float32
        ),
        output_reservoir=torch.zeros(
            (n_layers, n_experts, 4, 8), dtype=torch.float32
        ),
    )


def _make_stage2_profile(n_layers: int = 2, n_experts: int = 3) -> Stage2ProfilePayload:
    return Stage2ProfilePayload(
        schema_version=SCHEMA_VERSIONS["stage2_profile"],
        n_experts=n_experts,
        n_layers=n_layers,
        delta_gate=torch.full(
            (n_layers, n_experts, n_experts), 1.0, dtype=torch.float32
        ),
        delta_expert=torch.full(
            (n_layers, n_experts, n_experts), 2.0, dtype=torch.float64
        ),
        a_gate_up=torch.full((n_layers, n_experts, 5), 0.1, dtype=torch.float32),
        a_down=torch.full((n_layers, n_experts, 7), 0.2, dtype=torch.float32),
        token_counts=torch.ones((n_layers, n_experts), dtype=torch.int64) * 13,
    )


def _make_covariance(n_layers: int = 2, n_experts: int = 3, hidden: int = 8) -> CovariancePayload:
    """Dict-valued covariance payload mirroring the on-disk
    ``_stage2_input_covariance.pt`` shape (schema v2).

    Keys: ``(layer_idx, expert_idx, matrix_name)`` -> ``Tensor[d_in, d_in]``.
    Matrix names match the Stage 2 writer convention: ``gate_proj`` (which
    covers both gate and up via the up_proj alias) and ``down_proj``.
    """
    sigma_in: dict = {}
    token_counts: dict = {}
    for li in range(n_layers):
        for e in range(n_experts):
            for name in ("gate_proj", "down_proj"):
                # fp16 to mirror the Stage 2 writer's storage dtype.
                sigma_in[(li, e, name)] = torch.eye(hidden, dtype=torch.float16)
                token_counts[(li, e, name)] = 7
    return CovariancePayload(
        schema_version=SCHEMA_VERSIONS["covariance"],
        n_experts=n_experts,
        n_layers=n_layers,
        sigma_in=sigma_in,
        token_counts=token_counts,
    )


def _make_router_kd(attempt_idx: int = 42, n_tokens: int = 4, top_k: int = 3) -> RouterKDLogitsPayload:
    rng = np.random.default_rng(seed=attempt_idx)
    return RouterKDLogitsPayload(
        schema_version=SCHEMA_VERSIONS["router_kd_logits"],
        token_ids=np.arange(n_tokens, dtype=np.int32),
        top_ids=rng.integers(0, 100, size=(n_tokens, top_k), dtype=np.int32),
        top_logprobs=rng.standard_normal(size=(n_tokens, top_k)).astype(np.float32),
        attempt_idx=attempt_idx,
        top_k=top_k,
    )


def _make_block_hidden(layer_idx: int = 7, n_tokens: int = 5, hidden: int = 8) -> BlockHiddenPayload:
    return BlockHiddenPayload(
        schema_version=SCHEMA_VERSIONS["block_hidden"],
        layer_idx=layer_idx,
        n_prompts_in_subset=2,
        hidden_states=torch.arange(
            n_tokens * hidden, dtype=torch.float32
        ).reshape(n_tokens, hidden).to(torch.bfloat16),
    )


def _make_teacher_eval() -> TeacherEvalPayload:
    return TeacherEvalPayload(
        schema_version=SCHEMA_VERSIONS["teacher_eval"],
        cache_key="0" * 64,  # SHA-256 hex placeholder
        teacher_results={"piqa": {"acc": 0.81}},
        teacher_param_counts={"total": 30_000_000_000},
    )


def _jsonl(tmp_path: Path) -> Path:
    """Conventional JSONL path -- the file itself does not need to exist;
    only its parent directory is consulted by the sidecar layout."""
    return tmp_path / "trace_000123.jsonl"


# ---------------------------------------------------------------------------
# Test 1 -- path derivation for atomic vs sharded signals.
# ---------------------------------------------------------------------------
def test_sidecar_path_atomic_and_sharded(tmp_path):
    jsonl = _jsonl(tmp_path)

    # Atomic single-file signal.
    atomic = sidecar_path(jsonl, "phase_b")
    assert atomic == tmp_path / "sidecars" / "phase_b.pt"

    # Custom suffix passthrough.
    npz = sidecar_path(jsonl, "router_kd_logits/0000007", suffix=".npz")
    assert npz == tmp_path / "sidecars" / "router_kd_logits" / "0000007.npz"

    # Per-shard block_hidden subpath.
    sharded = sidecar_path(jsonl, "block_hidden/layer_0007")
    assert sharded == tmp_path / "sidecars" / "block_hidden" / "layer_0007.pt"

    # router_kd_logits_dir convenience helper.
    assert router_kd_logits_dir(jsonl) == tmp_path / "sidecars" / "router_kd_logits"


# ---------------------------------------------------------------------------
# Test 2-7 -- round-trip for each of the 6 signals.
# ---------------------------------------------------------------------------
def test_phase_b_roundtrip(tmp_path):
    jsonl = _jsonl(tmp_path)
    original = _make_phase_b()
    save_phase_b(original, jsonl)

    expected_path = sidecar_path(jsonl, "phase_b")
    assert expected_path.exists(), "phase_b.pt must land at sidecars/phase_b.pt"
    # No orphan tmp left behind.
    assert not Path(str(expected_path) + ".tmp").exists()

    loaded = load_phase_b(jsonl)
    assert loaded is not None
    assert loaded.schema_version == SCHEMA_VERSIONS["phase_b"]
    assert loaded.n_experts == original.n_experts
    assert loaded.n_layers == original.n_layers
    assert torch.equal(loaded.per_expert_max, original.per_expert_max.cpu())
    assert torch.equal(loaded.routing_freq, original.routing_freq.cpu())
    assert torch.equal(loaded.mean_routing_weight, original.mean_routing_weight.cpu())
    assert torch.equal(loaded.output_reservoir, original.output_reservoir.cpu())
    # All loaded tensors are on CPU.
    assert loaded.per_expert_max.device.type == "cpu"
    assert loaded.output_reservoir.device.type == "cpu"

    # Miss-on-absent: deleting the file gives a clean None.
    expected_path.unlink()
    assert load_phase_b(jsonl) is None


def test_stage2_profile_roundtrip(tmp_path):
    jsonl = _jsonl(tmp_path)
    original = _make_stage2_profile()
    save_stage2_profile(original, jsonl)

    expected_path = sidecar_path(jsonl, "stage2_profile")
    assert expected_path.exists()

    loaded = load_stage2_profile(jsonl)
    assert loaded is not None
    assert loaded.schema_version == SCHEMA_VERSIONS["stage2_profile"]
    assert torch.equal(loaded.delta_gate, original.delta_gate.cpu())
    assert torch.equal(loaded.delta_expert, original.delta_expert.cpu())
    assert torch.equal(loaded.a_gate_up, original.a_gate_up.cpu())
    assert torch.equal(loaded.a_down, original.a_down.cpu())
    assert torch.equal(loaded.token_counts, original.token_counts.cpu())
    # Dtype preserved across the round-trip (float64 stays float64).
    assert loaded.delta_expert.dtype == torch.float64
    assert loaded.token_counts.dtype == torch.int64


def _make_reap_scores(n_layers: int = 2, n_experts: int = 3) -> Stage2ReapPayload:
    return Stage2ReapPayload(
        schema_version=SCHEMA_VERSIONS["reap_scores"],
        n_experts=n_experts,
        n_layers=n_layers,
        reap_scores=torch.arange(
            n_layers * n_experts, dtype=torch.float32
        ).reshape(n_layers, n_experts),
        token_counts=torch.full(
            (n_layers, n_experts), 11, dtype=torch.int64
        ),
    )


def test_reap_scores_roundtrip(tmp_path):
    jsonl = _jsonl(tmp_path)
    original = _make_reap_scores()
    save_reap_scores(original, jsonl)

    expected_path = sidecar_path(jsonl, "reap_scores")
    assert expected_path.exists()
    assert not Path(str(expected_path) + ".tmp").exists()

    loaded = load_reap_scores(jsonl)
    assert loaded is not None
    assert loaded.schema_version == SCHEMA_VERSIONS["reap_scores"]
    assert loaded.n_experts == original.n_experts
    assert loaded.n_layers == original.n_layers
    assert torch.equal(loaded.reap_scores, original.reap_scores.cpu())
    assert torch.equal(loaded.token_counts, original.token_counts.cpu())
    assert loaded.reap_scores.dtype == torch.float32
    assert loaded.token_counts.dtype == torch.int64
    assert loaded.reap_scores.device.type == "cpu"
    assert loaded.token_counts.device.type == "cpu"

    expected_path.unlink()
    assert load_reap_scores(jsonl) is None


def _make_per_expert_max(n_layers: int = 2, n_experts: int = 3) -> Stage1PerExpertMaxPayload:
    return Stage1PerExpertMaxPayload(
        schema_version=SCHEMA_VERSIONS["per_expert_max"],
        n_experts=n_experts,
        n_layers=n_layers,
        per_expert_max=torch.arange(
            n_layers * n_experts, dtype=torch.float32
        ).reshape(n_layers, n_experts),
        token_counts=torch.full(
            (n_layers, n_experts), 7, dtype=torch.int64
        ),
    )


def test_per_expert_max_roundtrip(tmp_path):
    jsonl = _jsonl(tmp_path)
    original = _make_per_expert_max()
    save_per_expert_max(original, jsonl)

    expected_path = sidecar_path(jsonl, "per_expert_max")
    assert expected_path.exists()
    assert not Path(str(expected_path) + ".tmp").exists()

    loaded = load_per_expert_max(jsonl)
    assert loaded is not None
    assert loaded.schema_version == SCHEMA_VERSIONS["per_expert_max"]
    assert loaded.n_experts == original.n_experts
    assert loaded.n_layers == original.n_layers
    assert torch.equal(loaded.per_expert_max, original.per_expert_max.cpu())
    assert torch.equal(loaded.token_counts, original.token_counts.cpu())
    assert loaded.per_expert_max.dtype == torch.float32
    assert loaded.token_counts.dtype == torch.int64
    assert loaded.per_expert_max.device.type == "cpu"
    assert loaded.token_counts.device.type == "cpu"

    expected_path.unlink()
    assert load_per_expert_max(jsonl) is None


def _make_routing_stats(n_layers: int = 2, n_experts: int = 3) -> RoutingStatsPayload:
    return RoutingStatsPayload(
        schema_version=SCHEMA_VERSIONS["routing_stats"],
        n_experts=n_experts,
        n_layers=n_layers,
        freq=torch.arange(
            n_layers * n_experts, dtype=torch.int64
        ).reshape(n_layers, n_experts),
        mean_weight=torch.linspace(
            0.1, 0.9, n_layers * n_experts, dtype=torch.float32,
        ).reshape(n_layers, n_experts),
    )


def test_routing_stats_roundtrip(tmp_path):
    jsonl = _jsonl(tmp_path)
    original = _make_routing_stats()
    save_routing_stats(original, jsonl)

    expected_path = sidecar_path(jsonl, "routing_stats")
    assert expected_path.exists()
    assert not Path(str(expected_path) + ".tmp").exists()

    loaded = load_routing_stats(jsonl)
    assert loaded is not None
    assert loaded.schema_version == SCHEMA_VERSIONS["routing_stats"]
    assert loaded.n_experts == original.n_experts
    assert loaded.n_layers == original.n_layers
    assert torch.equal(loaded.freq, original.freq.cpu())
    assert torch.equal(loaded.mean_weight, original.mean_weight.cpu())
    assert loaded.freq.dtype == torch.int64
    assert loaded.mean_weight.dtype == torch.float32
    assert loaded.freq.device.type == "cpu"
    assert loaded.mean_weight.device.type == "cpu"

    expected_path.unlink()
    assert load_routing_stats(jsonl) is None


def test_covariance_roundtrip(tmp_path):
    """Dict-valued covariance (schema v2) round-trips through save/load.

    Validates every (layer, expert, matrix) key survives, every tensor
    lands on CPU as fp16, and the token_counts dict is preserved as-is.
    """
    jsonl = _jsonl(tmp_path)
    original = _make_covariance()
    save_covariance(original, jsonl)

    assert sidecar_path(jsonl, "covariance").exists()
    loaded = load_covariance(jsonl)
    assert loaded is not None
    assert loaded.schema_version == SCHEMA_VERSIONS["covariance"]
    # Same keys.
    assert set(loaded.sigma_in.keys()) == set(original.sigma_in.keys())
    assert loaded.token_counts == original.token_counts
    # Each tensor matches (after a CPU/fp16 cast on the original side) and
    # lives on CPU as fp16 after the load.
    for key, t in loaded.sigma_in.items():
        assert t.device.type == "cpu"
        assert t.dtype == torch.float16
        assert torch.equal(t, original.sigma_in[key].cpu().to(torch.float16))


def test_router_kd_logits_roundtrip(tmp_path):
    jsonl = _jsonl(tmp_path)
    original = _make_router_kd(attempt_idx=42)
    save_router_kd_logits(original, jsonl)

    # Sharded .npz lives at sidecars/router_kd_logits/0000042.npz.
    expected_path = router_kd_logits_dir(jsonl) / "0000042.npz"
    assert expected_path.exists()
    # No double-extension (.npz.tmp.npz) or orphaned tmp.
    assert not Path(str(expected_path) + ".tmp").exists()
    assert not expected_path.with_suffix(".npz.npz").exists()

    loaded = load_router_kd_logits(jsonl, attempt_idx=42)
    assert loaded is not None
    assert loaded.schema_version == SCHEMA_VERSIONS["router_kd_logits"]
    assert loaded.attempt_idx == 42
    assert loaded.top_k == original.top_k
    np.testing.assert_array_equal(loaded.token_ids, original.token_ids)
    np.testing.assert_array_equal(loaded.top_ids, original.top_ids)
    np.testing.assert_array_equal(loaded.top_logprobs, original.top_logprobs)

    # Different attempt_idx: clean miss.
    assert load_router_kd_logits(jsonl, attempt_idx=99) is None


def test_block_hidden_roundtrip(tmp_path):
    jsonl = _jsonl(tmp_path)
    original = _make_block_hidden(layer_idx=7)
    save_block_hidden(original, jsonl)

    expected_path = (
        tmp_path / "sidecars" / "block_hidden" / "layer_0007.pt"
    )
    assert expected_path.exists()
    # Different layer_idx: clean miss (sharded layout).
    assert load_block_hidden(jsonl, layer_idx=8) is None

    loaded = load_block_hidden(jsonl, layer_idx=7)
    assert loaded is not None
    assert loaded.schema_version == SCHEMA_VERSIONS["block_hidden"]
    assert loaded.layer_idx == 7
    assert loaded.n_prompts_in_subset == 2
    assert loaded.hidden_states.dtype == torch.bfloat16
    assert torch.equal(loaded.hidden_states, original.hidden_states.cpu())


def test_teacher_eval_roundtrip(tmp_path):
    jsonl = _jsonl(tmp_path)
    original = _make_teacher_eval()
    save_teacher_eval(original, jsonl)

    assert sidecar_path(jsonl, "teacher_eval").exists()
    loaded = load_teacher_eval(jsonl)
    assert loaded is not None
    assert loaded.schema_version == SCHEMA_VERSIONS["teacher_eval"]
    assert loaded.cache_key == original.cache_key
    assert loaded.teacher_results == original.teacher_results
    assert loaded.teacher_param_counts == original.teacher_param_counts


# ---------------------------------------------------------------------------
# Test 8 -- schema_version mismatch raises with an actionable message.
# ---------------------------------------------------------------------------
def test_schema_version_mismatch_raises(tmp_path, monkeypatch):
    jsonl = _jsonl(tmp_path)

    # Persist a phase_b sidecar at version 1.
    save_phase_b(_make_phase_b(), jsonl)

    # Bump the central version *after* the write -- mimics a code update.
    bumped = dict(SCHEMA_VERSIONS)
    bumped["phase_b"] = 2
    monkeypatch.setattr(
        "moe_compress.utils.cached_calibration_signals.SCHEMA_VERSIONS",
        bumped,
    )

    with pytest.raises(ValueError) as exc:
        load_phase_b(jsonl)
    msg = str(exc.value)
    assert "schema_version=1" in msg
    assert "expected 2" in msg
    assert "Delete the sidecar to regenerate" in msg

    # The npz path also enforces schema versioning.
    save_router_kd_logits(_make_router_kd(attempt_idx=3), jsonl)
    bumped["router_kd_logits"] = 2
    monkeypatch.setattr(
        "moe_compress.utils.cached_calibration_signals.SCHEMA_VERSIONS",
        bumped,
    )
    with pytest.raises(ValueError) as exc:
        load_router_kd_logits(jsonl, attempt_idx=3)
    assert "router_kd_logits" in str(exc.value)
    assert "Delete the sidecar to regenerate" in str(exc.value)


# ---------------------------------------------------------------------------
# Test 9 -- atomic write: a crash mid-save leaves the previous final path
# intact (and never promotes a partial .tmp to the final path).
# ---------------------------------------------------------------------------
def test_atomic_write_crash_safety(tmp_path, monkeypatch):
    jsonl = _jsonl(tmp_path)

    # Write a known-good first version.
    first = _make_phase_b()
    save_phase_b(first, jsonl)
    expected_path = sidecar_path(jsonl, "phase_b")
    assert expected_path.exists()
    good_bytes = expected_path.read_bytes()

    # Now inject a crash inside torch.save -- the .tmp will be partial,
    # and os.replace must never run.
    def _boom(*args, **kwargs):
        raise RuntimeError("simulated SIGTERM during torch.save")

    monkeypatch.setattr(torch, "save", _boom)

    second = _make_phase_b(n_layers=4, n_experts=5)
    with pytest.raises(RuntimeError, match="simulated SIGTERM"):
        save_phase_b(second, jsonl)

    # The final path is untouched -- bytes match the first write.
    assert expected_path.exists()
    assert expected_path.read_bytes() == good_bytes

    # A tmp file may or may not exist (torch.save may have errored before
    # opening it); if it exists it is an orphan, never the final file.
    tmp = Path(str(expected_path) + ".tmp")
    if tmp.exists():
        assert tmp != expected_path

    # Undo the monkeypatch so a subsequent load (via reload of the
    # original good sidecar) still works.
    monkeypatch.undo()
    reloaded = load_phase_b(jsonl)
    assert reloaded is not None
    assert reloaded.n_layers == first.n_layers
    assert reloaded.n_experts == first.n_experts


# ---------------------------------------------------------------------------
# Synthetic provider pair -- minimal subclasses for dispatch tests.
# ---------------------------------------------------------------------------
class _SyntheticCacheProvider(BaseCacheProvider):
    """Cache-side: returns the loaded payload if the sidecar exists,
    else returns None and does not touch ctx."""

    name = "synthetic_phase_b_cache"
    paper = "Cached-calibration-signals plan (Item 0)"
    config_key = "calibration_v2.synthetic_phase_b.enabled"
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ("phase_b_cached",)
    provides: tuple[str, ...] = ()

    def on_load(self, ctx: PipelineContext, jsonl_path: Path):
        loaded = load_phase_b(jsonl_path)
        if loaded is None:
            return None
        ctx.set("phase_b_cached", loaded)
        return loaded


class _SyntheticLiveProvider(BaseLiveProvider):
    """Live-side: synthesizes a fresh payload, writes a sidecar, populates
    ctx, returns it. ``calls`` counts invocations so tests can assert
    short-circuit / fall-through behaviour."""

    name = "synthetic_phase_b_live"
    paper = "Cached-calibration-signals plan (Item 0)"
    config_key = "calibration_v2.synthetic_phase_b.enabled"
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ("phase_b_cached",)
    provides: tuple[str, ...] = ()

    def __init__(self):
        self.calls = 0

    def on_load(self, ctx: PipelineContext, jsonl_path: Path):
        self.calls += 1
        payload = _make_phase_b()
        save_phase_b(payload, jsonl_path)
        ctx.set("phase_b_cached", payload)
        return payload


class _SyntheticCovCacheProvider(BaseCacheProvider):
    """Signal-3 partial-cache cache side: populates only sigma_in_cached."""

    name = "synthetic_cov_cache"
    paper = "Cached-calibration-signals plan (Item 0)"
    config_key = "calibration_v2.synthetic_cov.enabled"
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ("sigma_in_cached",)
    provides: tuple[str, ...] = ()

    def on_load(self, ctx: PipelineContext, jsonl_path: Path):
        loaded = load_covariance(jsonl_path)
        if loaded is None:
            return None
        ctx.set("sigma_in_cached", loaded)
        return loaded


class _SyntheticCovLiveProvider(BaseLiveProvider):
    """Signal-3 partial-cache live side: writes the sigma_in sidecar AND
    sets ``C_cross_cov`` -- the cross-covariance is *always* recomputed live
    (it depends on the student, which the cache can't know about)."""

    name = "synthetic_cov_live"
    paper = "Cached-calibration-signals plan (Item 0)"
    config_key = "calibration_v2.synthetic_cov.enabled"
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ("sigma_in_cached", "C_cross_cov")
    provides: tuple[str, ...] = ()

    def __init__(self):
        self.calls = 0

    def on_load(self, ctx: PipelineContext, jsonl_path: Path):
        self.calls += 1
        payload = _make_covariance()
        save_covariance(payload, jsonl_path)
        ctx.set("sigma_in_cached", payload)
        # The student-dependent slot, always set on the live path:
        ctx.set("C_cross_cov", torch.eye(8, dtype=torch.float32))
        return payload


# ---------------------------------------------------------------------------
# Test 10 -- provider-pair dispatch via PluginRegistry.dispatch_first.
# Cache miss -> live runs, writes sidecar, populates ctx, returns payload.
# Cache hit  -> cache returns payload, live is never called.
# Partial-cache (Signal 3) -> the live path additionally populates
# ``C_cross_cov``; the cache path leaves it absent.
# ---------------------------------------------------------------------------
def test_provider_pair_dispatch_first(tmp_path):
    jsonl = _jsonl(tmp_path)

    # --- Cache miss: dispatch_first falls through to the live provider. ---
    cache = _SyntheticCacheProvider()
    live = _SyntheticLiveProvider()
    registry = PluginRegistry(plugins=[cache, live])
    ctx = PipelineContext()

    result = PluginRegistry.dispatch_first(
        registry.enabled({}), "on_load", ctx=ctx, jsonl_path=jsonl
    )
    assert result is not None
    assert live.calls == 1, "live provider must run on cache miss"
    # Sidecar was written by the live provider.
    assert sidecar_path(jsonl, "phase_b").exists()
    # ctx populated by the live provider's ctx.set.
    assert ctx.has("phase_b_cached")
    cached_payload = ctx.get("phase_b_cached")
    assert cached_payload.schema_version == SCHEMA_VERSIONS["phase_b"]

    # --- Cache hit on a fresh run: live must NOT run again. ---
    cache_2 = _SyntheticCacheProvider()
    live_2 = _SyntheticLiveProvider()
    registry_2 = PluginRegistry(plugins=[cache_2, live_2])
    ctx_2 = PipelineContext()

    result_2 = PluginRegistry.dispatch_first(
        registry_2.enabled({}), "on_load", ctx=ctx_2, jsonl_path=jsonl
    )
    assert result_2 is not None
    assert live_2.calls == 0, "cache hit must short-circuit the live provider"
    assert ctx_2.has("phase_b_cached")

    # --- Partial-cache scenario (Signal 3): cache populates only
    # sigma_in_cached; live additionally populates C_cross_cov. ---
    # Use a separate JSONL so the covariance sidecar starts absent.
    jsonl_cov = tmp_path / "trace_cov.jsonl"
    cov_cache = _SyntheticCovCacheProvider()
    cov_live = _SyntheticCovLiveProvider()
    cov_registry = PluginRegistry(plugins=[cov_cache, cov_live])
    cov_ctx = PipelineContext()

    # First run -- cache miss: live writes sidecar + sets both slots.
    PluginRegistry.dispatch_first(
        cov_registry.enabled({}), "on_load", ctx=cov_ctx, jsonl_path=jsonl_cov
    )
    assert cov_live.calls == 1
    assert cov_ctx.has("sigma_in_cached")
    assert cov_ctx.has("C_cross_cov")

    # Second run -- cache hit: only the cached slot is set; the
    # student-dependent C_cross_cov is NOT auto-populated by the cache.
    cov_cache_2 = _SyntheticCovCacheProvider()
    cov_live_2 = _SyntheticCovLiveProvider()
    cov_registry_2 = PluginRegistry(plugins=[cov_cache_2, cov_live_2])
    cov_ctx_2 = PipelineContext()

    PluginRegistry.dispatch_first(
        cov_registry_2.enabled({}), "on_load", ctx=cov_ctx_2, jsonl_path=jsonl_cov
    )
    assert cov_live_2.calls == 0, "cov cache hit must short-circuit live"
    assert cov_ctx_2.has("sigma_in_cached")
    assert not cov_ctx_2.has("C_cross_cov"), (
        "C_cross_cov is student-dependent and must NOT be supplied by the "
        "cache provider"
    )
