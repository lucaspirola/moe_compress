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
    OutputReservoirPayload,
    PhaseBPayload,
    RouterKDLogitsPayload,
    RouterLogitsStatsPayload,
    RoutingStatsPayload,
    Stage1PerExpertMaxPayload,
    Stage2ProfilePayloadV3,
    Stage2ReapPayload,
    TeacherEvalPayload,
    load_block_hidden,
    load_covariance,
    load_output_reservoir,
    load_per_expert_max,
    load_phase_b,
    load_reap_scores,
    load_router_logits_stats,
    load_routing_stats,
    load_router_kd_logits,
    load_stage2_profile_v3,
    load_teacher_eval,
    router_kd_logits_dir,
    save_block_hidden,
    save_covariance,
    save_output_reservoir,
    save_per_expert_max,
    save_reap_scores,
    save_router_logits_stats,
    save_routing_stats,
    save_router_kd_logits,
    save_stage2_profile_v3,
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


def _test_save_phase_b(payload: PhaseBPayload, jsonl_path: Path) -> None:
    """Test-local stand-in for the deleted public ``save_phase_b`` writer.

    NIT-3 (audit/calibration-completeness): the production writer
    ``save_phase_b`` was removed -- the combined Phase-B payload was
    superseded by per-signal sidecars. ``PhaseBPayload`` + ``load_phase_b``
    are still retained for legacy read support and are exercised below
    via this private test helper. Mirrors the deleted writer byte-for-byte
    so the substrate tests (schema-mismatch / atomic-write-crash-safety /
    F-H-7 new-path-shadowing) keep using phase_b as their regression target.
    """
    from dataclasses import replace as _dc_replace
    from moe_compress.utils.cached_calibration_signals import (
        _atomic_torch_save as _ccs_atomic_torch_save,
    )
    cpu_payload = _dc_replace(
        payload,
        per_expert_max=payload.per_expert_max.detach().cpu(),
        routing_freq=payload.routing_freq.detach().cpu(),
        mean_routing_weight=payload.mean_routing_weight.detach().cpu(),
        output_reservoir=payload.output_reservoir.detach().cpu(),
    )
    _ccs_atomic_torch_save(cpu_payload, sidecar_path(jsonl_path, "phase_b"))


def _make_stage2_profile(
    n_layers: int = 2, n_experts: int = 3, *, hidden: int = 8, d_int: int = 4,
) -> Stage2ProfilePayloadV3:
    """Build a deterministic Stage 2 profile-pass payload (schema v3).

    Tiny shapes — the focus is on schema/IO round-trip, not numerics.
    """
    gate_logit_profiles: dict[int, list[tuple[int, torch.Tensor]]] = {}
    neuron_act_sum: dict = {}
    neuron_act_count: dict = {}
    cov_acc: dict = {}
    cov_token_count: dict = {}
    layer_input_reservoir: list = []
    T_b = 5
    for lr in range(n_layers):
        gate_logit_profiles[lr] = [
            (T_b * b, torch.full((T_b, n_experts), 0.1 * (b + 1), dtype=torch.float32))
            for b in range(2)
        ]
        for e in range(n_experts):
            neuron_act_sum[(lr, e)] = torch.full((d_int,), 0.3, dtype=torch.float32)
            neuron_act_count[(lr, e)] = 11
            for m in ("gate_proj", "down_proj"):
                cov_acc[(lr, e, m)] = torch.eye(hidden, dtype=torch.float16)
                cov_token_count[(lr, e, m)] = 7
        layer_input_reservoir.append(torch.zeros((8, hidden), dtype=torch.bfloat16))
    return Stage2ProfilePayloadV3(
        format_version=3,
        schema_version=SCHEMA_VERSIONS["stage2_profile"],
        model_hash="deadbeef",
        n_layers=n_layers,
        n_experts=n_experts,
        top_k=2,
        cov_storage_dtype="float16",
        total_tokens_per_layer=torch.full((n_layers,), 2 * T_b, dtype=torch.int64),
        gate_logit_profiles=gate_logit_profiles,
        sim_tensor=torch.zeros((n_layers, n_experts, n_experts), dtype=torch.float64),
        neuron_act_sum=neuron_act_sum,
        neuron_act_count=neuron_act_count,
        cov_acc=cov_acc,
        cov_token_count=cov_token_count,
        layer_input_reservoir=layer_input_reservoir,
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
    stem = jsonl.stem  # "trace_000123"

    # Atomic single-file signal — F-H-7: namespaced by JSONL stem.
    atomic = sidecar_path(jsonl, "phase_b")
    assert atomic == tmp_path / "sidecars" / stem / "phase_b.pt"

    # Custom suffix passthrough.
    npz = sidecar_path(jsonl, "router_kd_logits/0000007", suffix=".npz")
    assert npz == tmp_path / "sidecars" / stem / "router_kd_logits" / "0000007.npz"

    # Per-shard block_hidden subpath.
    sharded = sidecar_path(jsonl, "block_hidden/layer_0007")
    assert sharded == tmp_path / "sidecars" / stem / "block_hidden" / "layer_0007.pt"

    # router_kd_logits_dir convenience helper.
    assert router_kd_logits_dir(jsonl) == tmp_path / "sidecars" / stem / "router_kd_logits"


# ---------------------------------------------------------------------------
# Test 2-7 -- round-trip for each of the 6 signals.
# ---------------------------------------------------------------------------
# NIT-3 (audit/calibration-completeness): ``test_phase_b_roundtrip`` was
# deleted alongside the public ``save_phase_b`` writer. ``load_phase_b``
# remains under test via the FH7 backward-compat suite below and the
# atomic-write / schema-mismatch substrate tests (which now write via
# the private ``_test_save_phase_b`` helper).


def test_stage2_profile_roundtrip(tmp_path):
    jsonl = _jsonl(tmp_path)
    original = _make_stage2_profile()
    save_stage2_profile_v3(original, jsonl)

    expected_path = sidecar_path(jsonl, "stage2_profile")
    assert expected_path.exists()

    loaded = load_stage2_profile_v3(jsonl)
    assert loaded is not None
    assert loaded.schema_version == SCHEMA_VERSIONS["stage2_profile"]
    assert loaded.format_version == 3
    assert loaded.cov_storage_dtype == "float16"
    assert torch.equal(
        loaded.total_tokens_per_layer, original.total_tokens_per_layer.cpu()
    )
    assert torch.equal(loaded.sim_tensor, original.sim_tensor.cpu())
    # gate_logit_profiles preserves the list-of-tuples shape verbatim.
    assert set(loaded.gate_logit_profiles) == set(original.gate_logit_profiles)
    for lr, batches in original.gate_logit_profiles.items():
        loaded_batches = loaded.gate_logit_profiles[lr]
        assert len(loaded_batches) == len(batches)
        for (lof, lt), (oof, ot) in zip(loaded_batches, batches):
            assert int(lof) == int(oof)
            assert torch.equal(lt, ot.cpu())
    # Dtype preserved across the round-trip (float64 stays float64).
    assert loaded.sim_tensor.dtype == torch.float64
    assert loaded.total_tokens_per_layer.dtype == torch.int64


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


def _make_router_logits_stats(
    n_layers: int = 2, n_experts: int = 3,
) -> RouterLogitsStatsPayload:
    return RouterLogitsStatsPayload(
        schema_version=SCHEMA_VERSIONS["router_logits_stats"],
        n_experts=n_experts,
        n_layers=n_layers,
        score_sink_sum=torch.arange(
            n_layers * n_experts, dtype=torch.float32
        ).reshape(n_layers, n_experts),
        score_normal_sum=torch.linspace(
            0.0, 1.0, n_layers * n_experts, dtype=torch.float32,
        ).reshape(n_layers, n_experts),
        fire_on_sink=torch.arange(
            1, n_layers * n_experts + 1, dtype=torch.int64
        ).reshape(n_layers, n_experts),
        n_sink_tokens=torch.tensor(
            [4 * (i + 1) for i in range(n_layers)], dtype=torch.int64,
        ),
        n_normal_tokens=torch.tensor(
            [16 * (i + 1) for i in range(n_layers)], dtype=torch.int64,
        ),
        bos_token_id=151643,
    )


def test_router_logits_stats_roundtrip(tmp_path):
    """Per-(layer, expert) sink-vs-normal router-score aggregates round-trip
    through save/load with byte-identical tensors and the BOS id preserved.

    Also exercises the bos_token_id=None branch (writer didn't capture
    the BOS) -- the round-trip must preserve the None as-is, not coerce
    to 0 or any other sentinel."""
    jsonl = _jsonl(tmp_path)
    original = _make_router_logits_stats()
    save_router_logits_stats(original, jsonl)

    expected_path = sidecar_path(jsonl, "router_logits_stats")
    assert expected_path.exists()
    assert not Path(str(expected_path) + ".tmp").exists()

    loaded = load_router_logits_stats(jsonl)
    assert loaded is not None
    assert loaded.schema_version == SCHEMA_VERSIONS["router_logits_stats"]
    assert loaded.n_experts == original.n_experts
    assert loaded.n_layers == original.n_layers
    assert torch.equal(loaded.score_sink_sum, original.score_sink_sum.cpu())
    assert torch.equal(loaded.score_normal_sum, original.score_normal_sum.cpu())
    assert torch.equal(loaded.fire_on_sink, original.fire_on_sink.cpu())
    assert torch.equal(loaded.n_sink_tokens, original.n_sink_tokens.cpu())
    assert torch.equal(loaded.n_normal_tokens, original.n_normal_tokens.cpu())
    assert loaded.bos_token_id == 151643
    # Dtypes survive the cast in save_router_logits_stats.
    assert loaded.score_sink_sum.dtype == torch.float32
    assert loaded.score_normal_sum.dtype == torch.float32
    assert loaded.fire_on_sink.dtype == torch.int64
    assert loaded.n_sink_tokens.dtype == torch.int64
    assert loaded.n_normal_tokens.dtype == torch.int64
    # All loaded tensors are on CPU (device-agnostic sidecar contract).
    assert loaded.score_sink_sum.device.type == "cpu"
    assert loaded.fire_on_sink.device.type == "cpu"

    # bos_token_id=None branch -- writer didn't capture it. The None must
    # survive the round-trip (not get coerced to 0 by int(None)).
    none_bos = RouterLogitsStatsPayload(
        schema_version=SCHEMA_VERSIONS["router_logits_stats"],
        n_experts=2,
        n_layers=1,
        score_sink_sum=torch.zeros((1, 2), dtype=torch.float32),
        score_normal_sum=torch.zeros((1, 2), dtype=torch.float32),
        fire_on_sink=torch.zeros((1, 2), dtype=torch.int64),
        n_sink_tokens=torch.zeros((1,), dtype=torch.int64),
        n_normal_tokens=torch.zeros((1,), dtype=torch.int64),
        bos_token_id=None,
    )
    jsonl2 = tmp_path / "trace_none_bos.jsonl"
    save_router_logits_stats(none_bos, jsonl2)
    loaded_none = load_router_logits_stats(jsonl2)
    assert loaded_none is not None
    assert loaded_none.bos_token_id is None

    expected_path.unlink()
    assert load_router_logits_stats(jsonl) is None


def _make_output_reservoir(
    n_layers: int = 2, n_experts: int = 3,
    max_tokens: int = 4, hidden_dim: int = 5,
) -> OutputReservoirPayload:
    # Deterministic small payload; reservoir is fp32 on the writer side
    # and gets cast to bf16 inside save_output_reservoir.
    reservoir = torch.arange(
        n_layers * n_experts * max_tokens * hidden_dim, dtype=torch.float32,
    ).reshape(n_layers, n_experts, max_tokens, hidden_dim)
    valid_count = torch.tensor(
        [[max_tokens, max_tokens - 1, 0],
         [max_tokens - 2, max_tokens, 1]],
        dtype=torch.int64,
    )
    total_seen = torch.tensor(
        [[max_tokens, max_tokens - 1, 0],
         [max_tokens * 3, max_tokens * 5, 1]],
        dtype=torch.int64,
    )
    return OutputReservoirPayload(
        schema_version=SCHEMA_VERSIONS["output_reservoir"],
        n_experts=n_experts,
        n_layers=n_layers,
        reservoir=reservoir,
        valid_count=valid_count,
        total_seen=total_seen,
        max_tokens=max_tokens,
    )


def test_output_reservoir_roundtrip(tmp_path):
    """Per-(layer, expert) output reservoir round-trips through save/load.

    Validates: (a) the sidecar lands at the documented path, (b) the
    reservoir tensor survives the bf16 cast (equal modulo dtype precision),
    (c) valid_count + total_seen are preserved as int64 on CPU, and
    (d) the max_tokens scalar field is preserved as Python int.
    """
    jsonl = _jsonl(tmp_path)
    original = _make_output_reservoir()
    save_output_reservoir(original, jsonl)

    expected_path = sidecar_path(jsonl, "output_reservoir")
    assert expected_path.exists()
    # tmp file does not leak after the atomic os.replace.
    assert not Path(str(expected_path) + ".tmp").exists()

    loaded = load_output_reservoir(jsonl)
    assert loaded is not None
    assert loaded.schema_version == SCHEMA_VERSIONS["output_reservoir"]
    assert loaded.n_experts == original.n_experts
    assert loaded.n_layers == original.n_layers
    assert loaded.max_tokens == original.max_tokens
    # Reservoir is byte-identical to the bf16-cast original (the writer
    # casts via .to(dtype=torch.bfloat16); compare against the same cast).
    expected_reservoir = original.reservoir.cpu().to(torch.bfloat16)
    assert torch.equal(loaded.reservoir, expected_reservoir)
    assert loaded.reservoir.dtype == torch.bfloat16
    assert loaded.reservoir.device.type == "cpu"
    assert torch.equal(loaded.valid_count, original.valid_count.cpu())
    assert torch.equal(loaded.total_seen, original.total_seen.cpu())
    assert loaded.valid_count.dtype == torch.int64
    assert loaded.total_seen.dtype == torch.int64
    assert loaded.valid_count.device.type == "cpu"
    assert loaded.total_seen.device.type == "cpu"
    assert isinstance(loaded.max_tokens, int)

    expected_path.unlink()
    assert load_output_reservoir(jsonl) is None


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
        tmp_path / "sidecars" / jsonl.stem / "block_hidden" / "layer_0007.pt"
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
    _test_save_phase_b(_make_phase_b(), jsonl)

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
    _test_save_phase_b(first, jsonl)
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
        _test_save_phase_b(second, jsonl)

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
        _test_save_phase_b(payload, jsonl_path)
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


# ---------------------------------------------------------------------------
# MEDIUM-7 — F-H-7 backward-compat: legacy non-namespaced sidecar fallback.
# ---------------------------------------------------------------------------
def _write_legacy_phase_b(jsonl_path: Path, payload: PhaseBPayload) -> Path:
    """Write a PhaseBPayload at the LEGACY (pre-F-H-7) non-namespaced
    path: <jsonl.parent>/sidecars/phase_b.pt. Used by the backward-compat
    tests below to seed disk state that simulates a pre-F-H-7 run."""
    from moe_compress.utils.cached_calibration_signals import _legacy_sidecar_path
    legacy = _legacy_sidecar_path(jsonl_path, "phase_b")
    legacy.parent.mkdir(parents=True, exist_ok=True)
    # Move tensor fields to CPU as save_phase_b would; we bypass the
    # public writer because save_phase_b writes to the NEW namespaced
    # path, but the test needs the legacy layout on disk.
    from dataclasses import replace as _replace
    cpu_payload = _replace(
        payload,
        per_expert_max=payload.per_expert_max.detach().cpu(),
        routing_freq=payload.routing_freq.detach().cpu(),
        mean_routing_weight=payload.mean_routing_weight.detach().cpu(),
        output_reservoir=payload.output_reservoir.detach().cpu(),
    )
    torch.save(cpu_payload, legacy)
    return legacy


def _with_ccs_logger_propagate(fn):
    """Pattern N (caplog-propagate-restore) wrapper for cached_calibration_signals.

    The module uses ``log = logging.getLogger(__name__)`` — a non-root logger
    — so pytest's caplog needs the logger to propagate=True for records to
    bubble up to the root LogCaptureHandler. See [[caplog-propagate-restore]].
    """
    import logging as _lg
    _logger = _lg.getLogger("moe_compress.utils.cached_calibration_signals")
    prev = _logger.propagate
    _logger.propagate = True
    try:
        return fn()
    finally:
        _logger.propagate = prev


def test_fh7_legacy_single_stem_fallback_warns_once(tmp_path, caplog):
    """MEDIUM-7 + HIGH-4: a single-stem legacy sidecar is consumed with a
    ONE-SHOT WARNING. Multiple successive loads emit the warning AT MOST
    ONCE per (legacy_path, signal_name) — HIGH-4's dedupe contract.
    """
    import logging
    import moe_compress.utils.cached_calibration_signals as ccs

    # Reset the dedupe set so this test is independent of prior tests.
    ccs._already_warned_legacy_paths.clear()

    jsonl = tmp_path / "trace_solo.jsonl"
    jsonl.touch()  # ensure the parent dir counts exactly one .jsonl stem
    legacy = _write_legacy_phase_b(jsonl, _make_phase_b())
    assert legacy.exists()
    # Sanity: new-style path does NOT exist.
    assert not sidecar_path(jsonl, "phase_b").exists()

    caplog.set_level(logging.WARNING, logger="moe_compress.utils.cached_calibration_signals")

    def _do_loads():
        p1 = load_phase_b(jsonl)
        p2 = load_phase_b(jsonl)
        p3 = load_phase_b(jsonl)
        assert p1 is not None and p2 is not None and p3 is not None

    _with_ccs_logger_propagate(_do_loads)
    # Exactly ONE warning recorded across three reads (HIGH-4 dedupe).
    warns = [r for r in caplog.records
             if r.levelno >= logging.WARNING and "F-H-7" in r.getMessage()]
    assert len(warns) == 1, (
        f"expected exactly 1 WARNING (one-shot dedupe), got {len(warns)}: "
        f"{[r.getMessage() for r in warns]}"
    )


def test_fh7_legacy_multi_stem_refuses_with_error(tmp_path, caplog):
    """MEDIUM-7: with >1 JSONL stem in the parent dir, the legacy
    fallback is REFUSED — log.error + None return. The error message
    is operator-actionable (mentions the .jsonl.tmp participation; see
    MEDIUM-6).
    """
    import logging
    import moe_compress.utils.cached_calibration_signals as ccs

    ccs._already_warned_legacy_paths.clear()

    jsonl_a = tmp_path / "trace_alpha.jsonl"
    jsonl_b = tmp_path / "trace_beta.jsonl"
    jsonl_a.touch()
    jsonl_b.touch()
    # Two stems → ambiguous → refuse.
    _write_legacy_phase_b(jsonl_a, _make_phase_b())

    caplog.set_level(logging.ERROR, logger="moe_compress.utils.cached_calibration_signals")
    result_box: dict = {}

    def _do_load():
        result_box["r"] = load_phase_b(jsonl_a)

    _with_ccs_logger_propagate(_do_load)
    assert result_box["r"] is None, "multi-stem legacy fallback must return None"
    errs = [r for r in caplog.records
            if r.levelno >= logging.ERROR and "F-H-7" in r.getMessage()]
    assert len(errs) >= 1, "expected ERROR log on multi-stem refusal"
    # MEDIUM-6: error mentions .jsonl.tmp participation.
    assert any(".jsonl.tmp" in r.getMessage() for r in errs), (
        "ERROR must mention .jsonl.tmp participation per MEDIUM-6"
    )


def test_fh7_new_path_exists_ignores_legacy(tmp_path, caplog):
    """MEDIUM-7: when the new-style namespaced path exists, the legacy
    file is ignored entirely — no WARNING, no ERROR, no log emission
    relating to F-H-7 backward compat.
    """
    import logging
    import moe_compress.utils.cached_calibration_signals as ccs

    ccs._already_warned_legacy_paths.clear()

    jsonl = tmp_path / "trace_x.jsonl"
    payload = _make_phase_b()
    # Seed BOTH the new-style path (via the test-local writer) and the
    # legacy path (via the helper).
    _test_save_phase_b(payload, jsonl)
    _write_legacy_phase_b(jsonl, payload)
    assert sidecar_path(jsonl, "phase_b").exists()

    caplog.set_level(logging.DEBUG, logger="moe_compress.utils.cached_calibration_signals")
    loaded_box: dict = {}

    def _do_load():
        loaded_box["v"] = load_phase_b(jsonl)

    _with_ccs_logger_propagate(_do_load)
    assert loaded_box["v"] is not None
    # No F-H-7 log records at all.
    fh7 = [r for r in caplog.records if "F-H-7" in r.getMessage()]
    assert fh7 == [], (
        f"new-path hit must NOT touch F-H-7 backward-compat code path; "
        f"got {[r.getMessage() for r in fh7]}"
    )


def test_fh7_router_kd_legacy_single_stem_one_shot_warn(tmp_path, caplog):
    """MEDIUM-7 (router_kd variant): the legacy router_kd_logits dir
    fallback (load_router_kd_logits) also dedupes WARN emissions per
    HIGH-4. Multiple loads → at most one warning.
    """
    import logging
    import moe_compress.utils.cached_calibration_signals as ccs
    from moe_compress.utils.cached_calibration_signals import (
        RouterKDLogitsPayload,
        SCHEMA_VERSIONS,
        _legacy_router_kd_logits_dir,
        save_router_kd_logits,
    )

    ccs._already_warned_legacy_paths.clear()

    jsonl = tmp_path / "trace_router.jsonl"
    jsonl.touch()
    # Seed a legacy shard.
    payload = RouterKDLogitsPayload(
        schema_version=SCHEMA_VERSIONS["router_kd_logits"],
        token_ids=np.arange(4, dtype=np.int32),
        top_ids=np.zeros((4, 3), dtype=np.int32),
        top_logprobs=np.zeros((4, 3), dtype=np.float32),
        attempt_idx=7,
        top_k=3,
    )
    # Use the public writer to create a *new-namespaced* shard, then
    # move it to the legacy location.
    save_router_kd_logits(payload, jsonl)
    new_shard = router_kd_logits_dir(jsonl) / "0000007.npz"
    assert new_shard.exists()
    legacy_dir = _legacy_router_kd_logits_dir(jsonl)
    legacy_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.move(str(new_shard), str(legacy_dir / "0000007.npz"))
    # Remove the now-empty new-style dir to simulate a pre-F-H-7 layout.
    new_shard.parent.rmdir()

    caplog.set_level(logging.WARNING, logger="moe_compress.utils.cached_calibration_signals")

    def _do_loads():
        r1 = load_router_kd_logits(jsonl, attempt_idx=7)
        r2 = load_router_kd_logits(jsonl, attempt_idx=7)
        r3 = load_router_kd_logits(jsonl, attempt_idx=7)
        assert r1 is not None and r2 is not None and r3 is not None

    _with_ccs_logger_propagate(_do_loads)
    warns = [r for r in caplog.records
             if r.levelno >= logging.WARNING and "F-H-7" in r.getMessage()]
    assert len(warns) == 1, (
        f"router_kd_logits legacy WARN must be one-shot; got {len(warns)}"
    )
