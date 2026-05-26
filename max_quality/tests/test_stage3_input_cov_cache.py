"""Tests for ``Stage3InputCovCacheProvider`` (Item 1 reader, Stage 3 side).

Covers:

1. ``test_load_miss`` -- no sidecar → ``load_covariance`` returns None and
   ``on_load`` returns None (does not raise).
2. ``test_load_hit`` -- sidecar present → ``on_load`` returns the
   dict-shaped CovariancePayload with byte-identical contents (modulo
   the fp16 cast the writer applies).
3. ``test_schema_mismatch_raises`` -- forced schema=99 → ``load_covariance``
   raises ValueError with the actionable 'Delete the sidecar' message.
4. ``test_payload_shape_matches_stage2_writer`` -- the cached payload's
   ``sigma_in`` dict has the same keying convention as the legacy
   ``_stage2_input_covariance.pt`` file's "covariance" dict:
   ``(layer_idx, expert_idx, matrix_name)`` → ``Tensor[d_in, d_in]``.
5. ``test_orchestrator_can_consume_cached_payload`` -- the cache hit's
   ``payload.sigma_in`` is a drop-in for ``A_cov`` in the Stage 3
   orchestrator: ``_cov_lookup`` resolves keys against it identically.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from moe_compress.pipeline.context import PipelineContext
from moe_compress.stage3.plugins.input_cov_cache import (
    Stage3InputCovCacheProvider,
)
from moe_compress.utils.cached_calibration_signals import (
    SCHEMA_VERSIONS,
    CovariancePayload,
    load_covariance,
    save_covariance,
    sidecar_path,
)


def _make_payload(n_layers: int = 2, n_experts: int = 3, d_in: int = 4) -> CovariancePayload:
    sigma_in: dict = {}
    token_counts: dict = {}
    for li in range(n_layers):
        for e in range(n_experts):
            # Stage 2 writer convention: gate_proj only (up_proj is aliased
            # via _cov_lookup's gate→up fallback). Add a down_proj entry too
            # so the payload matches the on-disk Stage 2 file's shape.
            for name in ("gate_proj", "down_proj"):
                sigma_in[(li, e, name)] = torch.eye(d_in, dtype=torch.float16) * (li + 1)
                token_counts[(li, e, name)] = 11 * (e + 1)
    return CovariancePayload(
        schema_version=SCHEMA_VERSIONS["covariance"],
        n_experts=n_experts,
        n_layers=n_layers,
        sigma_in=sigma_in,
        token_counts=token_counts,
    )


def _jsonl(tmp_path: Path) -> Path:
    return tmp_path / "trace_0001.jsonl"


# ---------------------------------------------------------------------------
# Test 1 -- miss
# ---------------------------------------------------------------------------


def test_load_miss(tmp_path):
    jsonl = _jsonl(tmp_path)
    provider = Stage3InputCovCacheProvider()
    ctx = PipelineContext()
    assert provider.on_load(ctx, jsonl) is None


# ---------------------------------------------------------------------------
# Test 2 -- hit
# ---------------------------------------------------------------------------


def test_load_hit(tmp_path):
    jsonl = _jsonl(tmp_path)
    payload = _make_payload()
    save_covariance(payload, jsonl)
    assert sidecar_path(jsonl, "covariance").exists()

    provider = Stage3InputCovCacheProvider()
    ctx = PipelineContext()
    result = provider.on_load(ctx, jsonl)
    assert result is not None
    assert result.schema_version == SCHEMA_VERSIONS["covariance"]
    assert result.n_layers == payload.n_layers
    assert result.n_experts == payload.n_experts
    # Same keys, fp16 on CPU.
    assert set(result.sigma_in.keys()) == set(payload.sigma_in.keys())
    for key, t in result.sigma_in.items():
        assert t.device.type == "cpu"
        assert t.dtype == torch.float16
        assert torch.equal(t, payload.sigma_in[key].cpu().to(torch.float16))


# ---------------------------------------------------------------------------
# Test 3 -- schema mismatch raises
# ---------------------------------------------------------------------------


def test_schema_mismatch_raises(tmp_path, monkeypatch):
    jsonl = _jsonl(tmp_path)
    save_covariance(_make_payload(), jsonl)

    # Bump the central version AFTER the write, mimicking a code upgrade.
    bumped = dict(SCHEMA_VERSIONS)
    bumped["covariance"] = 99
    monkeypatch.setattr(
        "moe_compress.utils.cached_calibration_signals.SCHEMA_VERSIONS",
        bumped,
    )

    provider = Stage3InputCovCacheProvider()
    with pytest.raises(ValueError) as exc:
        provider.on_load(PipelineContext(), jsonl)
    msg = str(exc.value)
    # The on-disk file is v2; expected v99 after the monkeypatch.
    assert "schema_version=2" in msg
    assert "expected 99" in msg
    assert "Delete the sidecar to regenerate" in msg


def test_schema_mismatch_propagates_through_dispatch_first(tmp_path, monkeypatch):
    """Schema-mismatch ValueError must propagate out of dispatch_first.

    The orchestrator's try/except around the cache lookup narrowed to
    ``(FileNotFoundError, OSError)`` -- a ``ValueError`` from
    ``_check_schema`` MUST escape so the user sees the actionable
    "Delete the sidecar to regenerate" message instead of silently
    falling back to a stale legacy ``_stage2_input_covariance.pt``.

    Mirror of the orchestrator's wiring: dispatch_first against the
    cache-provider-only plugin list, with a forced schema bump.
    """
    from moe_compress.pipeline.registry import PluginRegistry

    jsonl = _jsonl(tmp_path)
    save_covariance(_make_payload(), jsonl)

    bumped = dict(SCHEMA_VERSIONS)
    bumped["covariance"] = 99
    monkeypatch.setattr(
        "moe_compress.utils.cached_calibration_signals.SCHEMA_VERSIONS",
        bumped,
    )

    plugins = [Stage3InputCovCacheProvider()]
    with pytest.raises(ValueError) as exc:
        PluginRegistry.dispatch_first(
            plugins, "on_load", PipelineContext(), jsonl,
        )
    msg = str(exc.value)
    assert "schema_version=2" in msg
    assert "expected 99" in msg
    assert "Delete the sidecar to regenerate" in msg


# ---------------------------------------------------------------------------
# Test 4 -- payload shape matches Stage 2 writer
# ---------------------------------------------------------------------------


def test_payload_shape_matches_stage2_writer(tmp_path):
    """The cached sigma_in dict must use the same key tuple ``(layer_idx,
    expert_idx, matrix_name)`` as the Stage 2 writer's "covariance" field.

    This is the contract that lets the Stage 3 orchestrator use the cached
    payload as a drop-in for ``A_cov`` (the dict loaded from
    ``_stage2_input_covariance.pt``).
    """
    jsonl = _jsonl(tmp_path)
    payload = _make_payload(n_layers=2, n_experts=3, d_in=5)
    save_covariance(payload, jsonl)

    loaded = load_covariance(jsonl)
    assert loaded is not None

    # Every key is a 3-tuple of (int, int, str).
    for key in loaded.sigma_in.keys():
        assert isinstance(key, tuple)
        assert len(key) == 3
        li, e, name = key
        assert isinstance(li, int)
        assert isinstance(e, int)
        assert isinstance(name, str)
        assert name in ("gate_proj", "down_proj", "up_proj")

    # Every value is a 2-D square tensor of shape [d_in, d_in].
    for key, t in loaded.sigma_in.items():
        assert t.ndim == 2
        assert t.shape[0] == t.shape[1] == 5


# ---------------------------------------------------------------------------
# Test 5 -- orchestrator-integration smoke
# ---------------------------------------------------------------------------


def test_orchestrator_can_consume_cached_payload(tmp_path):
    """The cache hit's ``payload.sigma_in`` is a drop-in for ``A_cov``:
    ``_cov_lookup`` resolves the same keys against it as it does against
    the legacy Stage 2 file's "covariance" dict.

    The provider now also populates ``ctx.A_cov`` directly (M1 fix:
    uniform with Stage 4) so the orchestrator reads through the ctx
    rather than the returned payload.
    """
    from moe_compress.stage3.plugins.aa_svd_factor import _cov_lookup

    jsonl = _jsonl(tmp_path)
    payload = _make_payload(n_layers=2, n_experts=3, d_in=4)
    save_covariance(payload, jsonl)

    provider = Stage3InputCovCacheProvider()
    ctx = PipelineContext()
    result = provider.on_load(ctx, jsonl)
    assert result is not None
    # M1 fix: on-hit provider populates ctx.A_cov for dispatch_first.
    assert ctx.has("A_cov")

    A_cov = ctx.get("A_cov")
    # Same dict as result.sigma_in.
    assert A_cov is result.sigma_in

    # Direct hit: (0, 1, "gate_proj") is in the dict.
    t = _cov_lookup(A_cov, layer_idx=0, expert_idx=1, matrix_name="gate_proj")
    assert t is not None
    assert t.shape == (4, 4)

    # gate→up fallback: lookup as "up_proj" must resolve to the
    # gate_proj entry (matches the live ``_cov_lookup`` contract used by
    # ``aa_svd_factor`` to serve up_proj covariance via gate_proj).
    t_up = _cov_lookup(A_cov, layer_idx=0, expert_idx=1, matrix_name="up_proj")
    assert t_up is not None
    assert torch.equal(t_up, t)

    # Miss: nonexistent (layer, expert) pair → None.
    t_miss = _cov_lookup(A_cov, layer_idx=99, expert_idx=0, matrix_name="gate_proj")
    assert t_miss is None
