"""Tests for ``Stage4InputCovCacheProvider`` (Item 1 reader, Stage 4 side).

Covers:

1. ``test_load_miss`` -- no sidecar → ``on_load`` returns None, no ctx
   mutation.
2. ``test_load_hit_populates_ctx`` -- sidecar present → ``on_load``
   returns the payload AND sets ``ctx.A_cov`` + ``ctx.a_storage_dtype``.
3. ``test_schema_mismatch_raises`` -- forced schema=99 → ``load_covariance``
   raises ValueError with the actionable 'Delete the sidecar' message.
4. ``test_integration_cache_hit_skips_disk`` -- with the cache provider
   running BEFORE EoraInputsPlugin and a sidecar in place, the live
   load_eora_inputs sees ``ctx.has("A_cov")`` and does NOT touch
   ``_stage2_input_covariance.pt`` (even if a stale one exists, it is
   not opened on the hot path).
5. ``test_cache_miss_falls_back_to_disk`` -- no sidecar AND an on-disk
   ``_stage2_input_covariance.pt`` → live load picks the disk version.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from moe_compress.pipeline.context import PipelineContext
from moe_compress.pipeline.registry import PluginRegistry
from moe_compress.stage4.plugins.eora_inputs import EoraInputsPlugin
from moe_compress.stage4.plugins.input_cov_cache import (
    Stage4InputCovCacheProvider,
)
from moe_compress.utils.cached_calibration_signals import (
    SCHEMA_VERSIONS,
    CovariancePayload,
    save_covariance,
    sidecar_path,
)


def _make_payload(n_layers: int = 2, n_experts: int = 3, d_in: int = 4) -> CovariancePayload:
    sigma_in: dict = {}
    token_counts: dict = {}
    for li in range(n_layers):
        for e in range(n_experts):
            for name in ("gate_proj", "down_proj"):
                sigma_in[(li, e, name)] = torch.eye(d_in, dtype=torch.float16) * (e + 1)
                token_counts[(li, e, name)] = 5
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
    provider = Stage4InputCovCacheProvider()
    ctx = PipelineContext()
    result = provider.on_load(ctx, jsonl)
    assert result is None
    assert not ctx.has("A_cov")
    assert not ctx.has("a_storage_dtype")


# ---------------------------------------------------------------------------
# Test 2 -- hit populates ctx
# ---------------------------------------------------------------------------


def test_load_hit_populates_ctx(tmp_path):
    jsonl = _jsonl(tmp_path)
    payload = _make_payload()
    save_covariance(payload, jsonl)
    assert sidecar_path(jsonl, "covariance").exists()

    provider = Stage4InputCovCacheProvider()
    ctx = PipelineContext()
    result = provider.on_load(ctx, jsonl)
    assert result is not None
    assert ctx.has("A_cov")
    assert ctx.has("a_storage_dtype")
    assert ctx.get("a_storage_dtype") == torch.float16
    # ctx.A_cov is the same dict as payload.sigma_in (post fp16 cast).
    A_cov = ctx.get("A_cov")
    assert set(A_cov.keys()) == set(payload.sigma_in.keys())
    for key, t in A_cov.items():
        assert t.device.type == "cpu"
        assert t.dtype == torch.float16


# ---------------------------------------------------------------------------
# Test 3 -- schema mismatch raises
# ---------------------------------------------------------------------------


def test_schema_mismatch_raises(tmp_path, monkeypatch):
    jsonl = _jsonl(tmp_path)
    save_covariance(_make_payload(), jsonl)

    bumped = dict(SCHEMA_VERSIONS)
    bumped["covariance"] = 99
    monkeypatch.setattr(
        "moe_compress.utils.cached_calibration_signals.SCHEMA_VERSIONS",
        bumped,
    )

    provider = Stage4InputCovCacheProvider()
    with pytest.raises(RuntimeError) as exc:
        provider.on_load(PipelineContext(), jsonl)
    msg = str(exc.value)
    assert "manifest validation FAILED" in msg
    assert "schema_version=2" in msg
    assert "expected 99" in msg
    assert "re-run calibration" in msg


# ---------------------------------------------------------------------------
# Test 4 -- integration: cache hit short-circuits the live disk load
# ---------------------------------------------------------------------------


def test_integration_cache_hit_skips_disk(tmp_path, monkeypatch):
    """With the cache hit + cache provider registered FIRST, the live
    ``EoraInputsPlugin.load_eora_inputs`` short-circuits its on-disk
    ``_stage2_input_covariance.pt`` load via ``ctx.has("A_cov")``.

    Verification: install a torch.load spy that fails the test if the
    stale on-disk file is touched.
    """
    jsonl = _jsonl(tmp_path)
    payload = _make_payload(n_layers=1, n_experts=2, d_in=3)
    save_covariance(payload, jsonl)

    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    stale_path = artifacts_dir / "_stage2_input_covariance.pt"
    # Stale on-disk file the live path would otherwise read.
    torch.save({"format_version": 1, "covariance": {}, "tokens": {}}, stale_path)

    cache = Stage4InputCovCacheProvider()
    # Manually invoke the cache (mirroring orchestrator dispatch).
    ctx = PipelineContext()
    ctx.set("artifacts_dir", artifacts_dir)
    ctx.set("config", {"stage2_reap_ream": {"covariance_storage_dtype": "float16"}})

    # Mock model: a minimal stand-in returning an empty iterator of MoE layers.
    # Minimal model stub with a discoverable text tower (.layers).
    # iter_moe_layers walks model.layers, and _find_text_tower picks
    # the candidate whose .layers is a non-empty ModuleList -- so we
    # plant a sentinel non-MoE layer that _is_moe_layer rejects, which
    # leaves iter_moe_layers yielding nothing.
    import torch.nn as nn
    class _StubLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = None  # rejected by _is_moe_layer (no .experts attr)
    stub_model = nn.Module()
    stub_model.layers = nn.ModuleList([_StubLayer()])
    ctx.set("model", stub_model)

    # Install the torch.load spy. The cache provider has already loaded
    # the sidecar via cached_calibration_signals.load_covariance (uses
    # torch.load internally) BEFORE we install the spy. After install,
    # the only torch.load that should fire is from the
    # ``_stage3_original_weights.pt`` path (originals); if A_cov is
    # cache-hit, the ``_stage2_input_covariance.pt`` torch.load must
    # NOT fire.
    cache_result = cache.on_load(ctx, jsonl)
    assert cache_result is not None
    assert ctx.has("A_cov")

    # Write a placeholder originals file so the live plugin can proceed.
    torch.save({}, artifacts_dir / "_stage3_original_weights.pt")

    loaded_paths: list[str] = []
    real_torch_load = torch.load

    def _spy_load(path, *args, **kwargs):
        loaded_paths.append(str(path))
        return real_torch_load(path, *args, **kwargs)

    monkeypatch.setattr(torch, "load", _spy_load)

    live = EoraInputsPlugin()
    live.load_eora_inputs(ctx)

    # Assert: the stale Stage 2 cov file was NOT opened.
    assert not any("_stage2_input_covariance.pt" in p for p in loaded_paths), (
        f"unexpected torch.load on cache hit: {loaded_paths}"
    )


# ---------------------------------------------------------------------------
# Test 5 -- cache miss falls back to disk
# ---------------------------------------------------------------------------


def test_cache_miss_falls_back_to_disk(tmp_path):
    """No sidecar → live ``EoraInputsPlugin.load_eora_inputs`` reads
    ``_stage2_input_covariance.pt`` and populates A_cov from it.
    """
    jsonl = _jsonl(tmp_path)  # no sidecar saved

    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    # Plant a Stage-2-shaped on-disk file.
    disk_payload = {
        "format_version": 1,
        "covariance": {
            (0, 0, "gate_proj"): torch.eye(3, dtype=torch.float16) * 2,
        },
        "tokens": {(0, 0, "gate_proj"): 9},
    }
    torch.save(disk_payload, artifacts_dir / "_stage2_input_covariance.pt")
    torch.save({}, artifacts_dir / "_stage3_original_weights.pt")

    cache = Stage4InputCovCacheProvider()
    ctx = PipelineContext()
    ctx.set("artifacts_dir", artifacts_dir)
    ctx.set("config", {"stage2_reap_ream": {"covariance_storage_dtype": "float16"}})

    # Minimal model stub with a discoverable text tower (.layers).
    # iter_moe_layers walks model.layers, and _find_text_tower picks
    # the candidate whose .layers is a non-empty ModuleList -- so we
    # plant a sentinel non-MoE layer that _is_moe_layer rejects, which
    # leaves iter_moe_layers yielding nothing.
    import torch.nn as nn
    class _StubLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = None  # rejected by _is_moe_layer (no .experts attr)
    stub_model = nn.Module()
    stub_model.layers = nn.ModuleList([_StubLayer()])
    ctx.set("model", stub_model)

    # Cache miss.
    assert cache.on_load(ctx, jsonl) is None
    assert not ctx.has("A_cov")

    # Live load reads from disk.
    live = EoraInputsPlugin()
    live.load_eora_inputs(ctx)
    assert ctx.has("A_cov")
    A_cov = ctx.get("A_cov")
    assert (0, 0, "gate_proj") in A_cov
    assert torch.equal(
        A_cov[(0, 0, "gate_proj")],
        torch.eye(3, dtype=torch.float16) * 2,
    )


# ---------------------------------------------------------------------------
# Test 6 (S-2) -- Reader 2: torn on-disk payload fails loudly via manifest
# ---------------------------------------------------------------------------


def test_stage4_cache_miss_torn_disk_payload_fails_loudly(tmp_path):
    """S-2: end-to-end through Reader 2's cache-miss branch.

    Plant a Stage 2 cov via the real writer (which emits a manifest),
    then truncate the .pt to simulate a SIGKILL mid-write. The manifest
    still vouches for the original size, so reading via
    ``EoraInputsPlugin.load_eora_inputs`` MUST raise loudly with the
    'delete + re-run Stage 2' actionable signature instead of silently
    consuming a partial multi-GB payload.
    """
    import threading

    from moe_compress.stage2.shared_io import _save_covariance

    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()

    # Minimal fake of InputCovarianceAccumulator that _save_covariance
    # consumes: _lock + .covariance + .token_count.
    class _Fake:
        def __init__(self) -> None:
            self._lock = threading.Lock()
            self.covariance = {
                (0, 0, "gate_proj"): torch.eye(3, dtype=torch.float16),
                (0, 1, "gate_proj"): torch.eye(3, dtype=torch.float16) * 2,
            }
            self.token_count = {k: 5 for k in self.covariance}

    cov_path = artifacts_dir / "_stage2_input_covariance.pt"
    _save_covariance(_Fake(), cov_path)

    # Plant a stage3 originals placeholder (load_eora_inputs needs it past
    # the cov branch we're testing).
    torch.save({}, artifacts_dir / "_stage3_original_weights.pt")

    # Truncate the cov payload — manifest still says the old size, so
    # read_and_validate_manifest must raise ManifestMismatchError.
    real_size = cov_path.stat().st_size
    with open(cov_path, "r+b") as f:
        f.truncate(real_size // 2)

    ctx = PipelineContext()
    ctx.set("artifacts_dir", artifacts_dir)
    ctx.set("config", {"stage2_reap_ream": {"covariance_storage_dtype": "float16"}})

    # Minimal model stub (same pattern as the other tests in this file).
    import torch.nn as nn

    class _StubLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mlp = None  # rejected by _is_moe_layer

    stub_model = nn.Module()
    stub_model.layers = nn.ModuleList([_StubLayer()])
    ctx.set("model", stub_model)

    live = EoraInputsPlugin()
    with pytest.raises(RuntimeError, match="re-run Stage 2"):
        live.load_eora_inputs(ctx)
