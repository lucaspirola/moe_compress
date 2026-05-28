"""Tests for the W-1 Wanda scalar_row calibration sidecar (audit/PLAN_W1).

Covers (plan §9):

* T3 -- cache provider HIT / MISS / schema mismatch behaviour.
* T4 -- end-to-end short-circuit in ``WandaIntraExpertScorePlugin``:
  on cache HIT the plugin skips its per-layer calibration sweep
  (instrument_experts NEVER called); the resulting score map equals
  the live-accumulator score map for the same scalar_row inputs.
* T5 -- Pattern O write order: manifest is the LAST artifact written
  (mock ``os.replace`` to capture the rename sequence).
* T6 -- checkpoint kill + resume byte-equality: the post-resume
  accumulator state matches a single-segment reference run at
  near-byte-equality (``torch.allclose(rtol=0, atol=1e-5)`` for
  fp32 sigma sums; ``torch.equal`` for int64 token counts); a final
  dump + manifest re-load round-trips through the cache reader.
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from moe_compress.pipeline.context import PipelineContext
from moe_compress.stage3.plugins.wanda_intra_expert_score import (
    _WandaScalarRowAccumulator,
    WandaIntraExpertScorePlugin,
)
from moe_compress.stage3.plugins.wanda_scalar_row_cache import (
    Stage3WandaScalarRowCacheProvider,
)
from moe_compress.utils.atomic_io import ManifestMismatchError
from moe_compress.utils.cached_calibration_signals import (
    SCHEMA_VERSIONS,
    WandaScalarRowPayload,
    load_wanda_scalar_row,
    save_wanda_scalar_row,
    sidecar_path,
)
from moe_compress.utils.model_io import iter_moe_layers


# ==========================================================================
# Helpers
# ==========================================================================


def _make_payload(n_layers: int = 2, n_experts: int = 3, d_in: int = 5,
                  seed: int = 7) -> WandaScalarRowPayload:
    """Construct a synthetic WandaScalarRowPayload for round-trip tests."""
    torch.manual_seed(seed)
    sigma: dict = {}
    counts: dict = {}
    for li in range(n_layers):
        for e in range(n_experts):
            for name in ("gate_proj",):  # writer only emits gate_proj
                key = (li, e, name)
                sigma[key] = torch.rand(d_in, dtype=torch.float32)
                counts[key] = (li + 1) * (e + 1)
    return WandaScalarRowPayload(
        schema_version=SCHEMA_VERSIONS["wanda_scalar_row"],
        n_experts=n_experts,
        n_layers=n_layers,
        sigma_x_g_squared=sigma,
        token_counts=counts,
    )


# ==========================================================================
# T3 -- cache HIT / MISS / schema mismatch
# ==========================================================================


def test_cache_hit_populates_ctx(tmp_path):
    """save_wanda_scalar_row -> load_wanda_scalar_row round-trips
    bit-equality through fp32; the cache provider populates
    ``ctx['stage3.wanda_scalar_row']`` on hit.
    """
    jsonl = tmp_path / "trace.jsonl"
    payload = _make_payload()
    save_wanda_scalar_row(payload, jsonl)
    # Sidecar + manifest both written.
    sidecar = sidecar_path(jsonl, "wanda_scalar_row")
    manifest = sidecar.with_suffix(sidecar.suffix + ".MANIFEST.json")
    assert sidecar.exists(), f"sidecar not at {sidecar}"
    assert manifest.exists(), f"manifest not at {manifest}"

    ctx = PipelineContext()
    provider = Stage3WandaScalarRowCacheProvider()
    out = provider.on_load(ctx, jsonl)
    assert out is not None
    assert ctx.has("stage3.wanda_scalar_row")
    cached = ctx.get("stage3.wanda_scalar_row")
    # Round-trip equality.
    assert cached.schema_version == payload.schema_version
    assert cached.n_layers == payload.n_layers
    assert cached.n_experts == payload.n_experts
    for key, expected in payload.sigma_x_g_squared.items():
        got = cached.sigma_x_g_squared[key]
        assert torch.equal(got, expected.to(torch.float32)), (
            f"sigma_x_g_squared[{key}] mismatch"
        )
    for key, count in payload.token_counts.items():
        assert cached.token_counts[key] == count


def test_cache_miss_returns_none(tmp_path):
    """No sidecar on disk -> on_load returns None and ctx untouched."""
    jsonl = tmp_path / "trace.jsonl"
    ctx = PipelineContext()
    provider = Stage3WandaScalarRowCacheProvider()
    out = provider.on_load(ctx, jsonl)
    assert out is None
    assert not ctx.has("stage3.wanda_scalar_row")


def test_schema_mismatch_raises(tmp_path):
    """Sidecar with the wrong schema_version raises ValueError with the
    actionable 'Delete the sidecar to regenerate' message.
    """
    jsonl = tmp_path / "trace.jsonl"
    payload = _make_payload()
    save_wanda_scalar_row(payload, jsonl)

    # Mutate the payload's schema_version on disk.
    sidecar = sidecar_path(jsonl, "wanda_scalar_row")
    bad = torch.load(sidecar, map_location="cpu", weights_only=False)
    bad.schema_version = 99
    torch.save(bad, sidecar)
    # Re-write the manifest with the bad version so we hit the schema
    # check at the dataclass level (not the manifest cross-check).
    from moe_compress.utils.atomic_io import write_manifest_last
    manifest = sidecar.with_suffix(sidecar.suffix + ".MANIFEST.json")
    manifest.unlink()
    write_manifest_last(sidecar, manifest, schema_version=99,
                        compute_sha256=False)
    # The manifest validates against SCHEMA_VERSIONS["wanda_scalar_row"]=1,
    # so the bad on-disk manifest raises ManifestMismatchError before the
    # payload-side schema check runs. Either error path is acceptable —
    # both are actionable.
    with pytest.raises((ValueError, ManifestMismatchError)):
        load_wanda_scalar_row(jsonl)


def test_missing_manifest_raises(tmp_path):
    """L-1 plan fold: green-field sidecar requires a manifest. A
    payload-without-manifest layout raises ManifestMismatchError
    (no silent torn-write read).
    """
    jsonl = tmp_path / "trace.jsonl"
    payload = _make_payload()
    save_wanda_scalar_row(payload, jsonl)
    sidecar = sidecar_path(jsonl, "wanda_scalar_row")
    manifest = sidecar.with_suffix(sidecar.suffix + ".MANIFEST.json")
    manifest.unlink()
    with pytest.raises(ManifestMismatchError):
        load_wanda_scalar_row(jsonl)


# ==========================================================================
# T5 -- Pattern O write order (manifest LAST)
# ==========================================================================


def test_writer_emits_manifest_after_payload(tmp_path):
    """Mock os.replace to capture the rename order. The manifest's
    rename MUST be observed AFTER the payload's rename (Pattern O).
    """
    jsonl = tmp_path / "trace.jsonl"
    payload = _make_payload(n_layers=1, n_experts=2, d_in=3)

    rename_order: list[str] = []

    # Capture the two ``os.replace`` calls that materialise the sidecar
    # + manifest. The atomic_io helpers use ``os.replace`` for the
    # tmp->final rename; we patch the symbol used inside the helpers.
    import moe_compress.utils.atomic_io as _aio
    real_replace = _aio.os.replace

    def _spy_replace(src, dst):
        rename_order.append(str(dst))
        return real_replace(src, dst)

    with patch.object(_aio.os, "replace", side_effect=_spy_replace):
        save_wanda_scalar_row(payload, jsonl)

    # Both renames observed; manifest rename must come AFTER the
    # payload rename.
    sidecar = sidecar_path(jsonl, "wanda_scalar_row")
    manifest = sidecar.with_suffix(sidecar.suffix + ".MANIFEST.json")
    sidecar_str = str(sidecar)
    manifest_str = str(manifest)
    # rename_order may contain parent-dir fsync targets that aren't
    # ``os.replace``; filter to just the actual file renames.
    # (Pattern O's parent-dir sync mechanism is ``os.fsync(fd)`` on
    # the parent directory handle -- NOT ``os.replace`` -- so the
    # spy here genuinely only observes file renames; the filter is
    # defensive in case the helper ever grows an intermediate
    # ``os.replace`` for some other artifact in the same tmp dir.)
    file_renames = [r for r in rename_order
                    if r in (sidecar_str, manifest_str)]
    assert sidecar_str in file_renames, (
        f"Sidecar rename not observed; renames: {file_renames}"
    )
    assert manifest_str in file_renames, (
        f"Manifest rename not observed; renames: {file_renames}"
    )
    sidecar_idx = file_renames.index(sidecar_str)
    manifest_idx = file_renames.index(manifest_str)
    assert manifest_idx > sidecar_idx, (
        f"Manifest at idx {manifest_idx} not AFTER sidecar at idx "
        f"{sidecar_idx}; order: {file_renames}"
    )


# ==========================================================================
# T4 -- end-to-end short-circuit in WandaIntraExpertScorePlugin
# ==========================================================================


def _make_score_ctx_with_cache(model, batches, tmp_path,
                               payload: WandaScalarRowPayload) -> PipelineContext:
    """Build a ctx for collect_wanda_scores with the cache pre-populated."""
    config = {
        "stage3": {
            "wanda_intra_expert": {
                "enabled": True,
                "write_sidecar": False,
                "score_dtype": "float32",
                "scalar_row_dtype": "float32",
            }
        }
    }
    ctx = PipelineContext()
    ctx.set("model", model)
    ctx.set("moe_layers", list(iter_moe_layers(model)))
    ctx.set("batches", batches)
    ctx.set("device", None)
    ctx.set("config", config)
    ctx.set("artifacts_dir", tmp_path)
    ctx.set("stage3.wanda_scalar_row", payload)
    return ctx


def _make_payload_from_live_acc(
    moe_layers, scalar_row_dtype: torch.dtype = torch.float32,
) -> WandaScalarRowPayload:
    """Construct a payload that matches what a live calibration sweep would
    have built — one entry per (layer, expert, 'gate_proj') with d_in
    matching the bank's gate_proj input width.
    """
    from moe_compress.utils.model_io import build_banks
    sigma: dict = {}
    counts: dict = {}
    n_experts = 0
    for ref in moe_layers:
        banks = build_banks(ref)
        d_in = banks["gate_proj"].get(0).shape[1]
        for e in range(ref.num_routed_experts):
            sigma[(ref.layer_idx, e, "gate_proj")] = torch.rand(
                d_in, dtype=torch.float32,
            ) + 0.1
            counts[(ref.layer_idx, e, "gate_proj")] = 4
        n_experts = max(n_experts, ref.num_routed_experts)
    return WandaScalarRowPayload(
        schema_version=SCHEMA_VERSIONS["wanda_scalar_row"],
        n_experts=n_experts,
        n_layers=len(moe_layers),
        sigma_x_g_squared=sigma,
        token_counts=counts,
    )


def test_plugin_consumes_cache_sidecar(tiny_model, tmp_path):
    """On cache HIT, instrument_experts is NEVER called; the
    score map equals the live-accumulator output for the same scalar_row.
    """
    moe_layers = list(iter_moe_layers(tiny_model))
    payload = _make_payload_from_live_acc(moe_layers)
    batches = [torch.randint(0, 32, (1, 4), dtype=torch.long)]
    ctx = _make_score_ctx_with_cache(tiny_model, batches, tmp_path, payload)

    # Spy on instrument_experts; collect_wanda_scores uses the symbol
    # bound at the wanda module's TOP via "from ...utils.activation_hooks
    # import instrument_experts" — patch THAT binding.
    import moe_compress.stage3.plugins.wanda_intra_expert_score as _wmod
    calls: list = []
    real_instrument = _wmod.instrument_experts

    def _spy_instrument(*args, **kwargs):
        calls.append((args, kwargs))
        return real_instrument(*args, **kwargs)

    with patch.object(_wmod, "instrument_experts",
                      side_effect=_spy_instrument) as _:
        plugin = WandaIntraExpertScorePlugin()
        plugin.collect_wanda_scores(ctx)

    assert calls == [], (
        f"Cache HIT must skip instrument_experts entirely; got "
        f"{len(calls)} call(s)"
    )

    score_map = ctx.get("stage3.wanda_intra_expert_score")
    assert isinstance(score_map, dict)
    # At least one layer/expert/matrix populated.
    n_entries = sum(
        len(per_e) for per_l in score_map.values()
        for per_e in per_l.values()
    )
    assert n_entries > 0, "expected at least one score entry from cached payload"

    # Cross-check: a live-accumulator path using the same payload
    # produces an IDENTICAL score map (anchors the from_payload L-2
    # contract).
    live_acc = _WandaScalarRowAccumulator(scalar_row_dtype=torch.float32)
    for key, sigma in payload.sigma_x_g_squared.items():
        live_acc._cpu[key] = sigma.to(torch.float32)
        live_acc._nsamples[key] = int(payload.token_counts[key])
    from moe_compress.stage3.plugins.wanda_intra_expert_score import (
        _compute_scores,
    )
    live_map = _compute_scores(
        moe_layers, live_acc, score_dtype=torch.float32,
    )
    # Same set of keys.
    assert set(score_map.keys()) == set(live_map.keys())
    for li in score_map:
        assert set(score_map[li].keys()) == set(live_map[li].keys()), (
            f"layer {li}: cached vs live expert sets differ"
        )
        for e in score_map[li]:
            assert set(score_map[li][e].keys()) == set(live_map[li][e].keys())
            for m in score_map[li][e]:
                torch.testing.assert_close(
                    score_map[li][e][m], live_map[li][e][m],
                    rtol=1e-5, atol=1e-6,
                )


# ==========================================================================
# H-1 -- orchestrator wiring: the V2 cache dispatch must promote
# ctx['stage3.wanda_scalar_row'] onto the live run_ctx that
# walk_phases(('collect_wanda_scores',), ...) consumes.
#
# This test EXISTS because of the original reviewer's C-1 finding:
# the first version of the orchestrator wired the wanda cache
# provider against a throwaway ``_cache_ctx`` that was discarded
# before run_ctx construction, so ``run_ctx.has('stage3.wanda_scalar_row')``
# always returned False and the consumer plugin always fell through
# to its per-layer calibration sweep. The fix added a second
# ``dispatch_first`` call on ``run_ctx`` mirroring
# ``Stage3BlockHiddenCacheProvider``'s pattern. This test asserts
# the wiring end-to-end and would have caught C-1 had it been in
# place before C-1 shipped green.
# ==========================================================================


class _IntegrationTokenizer:
    name_or_path = "tiny-tokenizer"
    eos_token_id = 0

    def __call__(self, text, *_, **__):
        return {"input_ids": [min(ord(c) % 32, 31) for c in (text or " ")]}

    def save_pretrained(self, *_args, **_kwargs):
        return None


def _noop_save_integration(model, tokenizer, path, **kwargs):
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


def test_orchestrator_promotes_wanda_cache_to_run_ctx(
    tiny_model, tiny_config, tmp_path, monkeypatch,
):
    """H-1 integration test (audit/PLAN_W1).

    Drive the real stage3 orchestrator's dispatch logic with a tiny
    model + pre-written W-1 sidecar; assert that on cache HIT the
    slot lands on ``run_ctx`` BEFORE ``walk_phases(('collect_wanda_scores',))``
    fires. The previous bug (C-1) would manifest as
    ``run_ctx.has('stage3.wanda_scalar_row') is False`` at the
    capture point and is therefore caught directly by this test.

    Implementation notes:

    * The orchestrator computes the calibration JSONL path as
      ``Path.cwd() / cal.get('jsonl_path', _DEFAULT_SELF_TRACES_PATH)``;
      we set ``calibration.jsonl_path`` to an absolute path under
      ``tmp_path`` so the orchestrator and the test write/read the
      same sidecar location.
    * We monkeypatch ``walk_phases`` to capture the live ``ctx`` at
      every walk -- the relevant one is the
      ``('collect_wanda_scores',)`` walk -- and short-circuit it
      to avoid running the full SVD pipeline. The orchestrator
      executes its preamble (incl. the V2 cache dispatch we are
      testing) and is interrupted by an exception we raise from a
      later phase so the orchestrator's finalize block never runs.
    """
    # Avoid the stage1+2 prerequisites: drive the orchestrator's
    # dispatch logic directly via the same module that ``run`` uses.
    from moe_compress.stage3 import orchestrator as stage3_orchestrator
    from moe_compress.utils import calibration as cal_mod
    from moe_compress.utils import model_io as mio
    from moe_compress.budget.solver import BudgetDecomposition
    from moe_compress.tools import phase_walker as _phase_walker_mod

    # Calibration JSONL path = tmp_path / "traces.jsonl"; write the
    # W-1 sidecar to ``tmp_path/sidecars/wanda_scalar_row.pt``.
    jsonl = tmp_path / "traces.jsonl"
    payload = _make_payload(n_layers=1, n_experts=2, d_in=4)
    save_wanda_scalar_row(payload, jsonl)
    # Sanity: sidecar + manifest both written.
    sidecar = sidecar_path(jsonl, "wanda_scalar_row")
    manifest = sidecar.with_suffix(sidecar.suffix + ".MANIFEST.json")
    assert sidecar.exists()
    assert manifest.exists()

    # Point the orchestrator at the absolute sidecar path so the
    # ``Path.cwd() / jsonl_path`` join in the orchestrator yields
    # the same absolute path.
    config = copy_config_with_wanda_jsonl(tiny_config, jsonl)

    # Stub the heavy loaders so the orchestrator doesn't try real I/O.
    def _fake_build(tokenizer, spec, cache_dir=None):
        torch.manual_seed(spec.seed)
        return torch.randint(0, 32, (spec.num_sequences, spec.sequence_length),
                             dtype=torch.long)

    monkeypatch.setattr(cal_mod, "build_calibration_tensor", _fake_build)
    monkeypatch.setattr(mio, "save_compressed_checkpoint", _noop_save_integration)

    # Capture the ctx at the collect_wanda_scores walk and short-circuit
    # the orchestrator before it does real SVD work.
    captured: dict = {"saw_wanda_walk": False, "ctx_has_slot": None}

    real_walk_phases = _phase_walker_mod.walk_phases

    class _StopHere(Exception):
        pass

    def _spy_walk_phases(phases, plugins, ctx, *args, **kwargs):
        if phases == ("collect_wanda_scores",):
            captured["saw_wanda_walk"] = True
            captured["ctx_has_slot"] = ctx.has("stage3.wanda_scalar_row")
            # Stop the orchestrator here -- everything after is out of
            # scope for this wiring test.
            raise _StopHere
        return real_walk_phases(phases, plugins, ctx, *args, **kwargs)

    monkeypatch.setattr(stage3_orchestrator, "walk_phases", _spy_walk_phases)

    decomp = BudgetDecomposition(
        total_reduction_ratio=0.2,
        expert_prune_ratio=0.5,
        svd_rank_ratio=0.14,
        global_expert_budget=4,
        min_experts_per_layer=2,
        blacklisted_experts={},
    )

    # The orchestrator's covariance phase needs a Stage 2 cov artifact;
    # write a minimal one so the no-cache path doesn't fail before our
    # spy fires. We don't actually use the values -- the test only
    # cares about the ctx state at the wanda walk.
    art = tmp_path
    stage2_cov_path = art / "_stage2_input_covariance.pt"
    torch.save({"covariance": {}}, stage2_cov_path)

    with pytest.raises(_StopHere):
        stage3_orchestrator.run(
            tiny_model, _IntegrationTokenizer(), config, art, decomp,
            device=None, no_resume=True,
        )

    assert captured["saw_wanda_walk"], (
        "orchestrator never reached the collect_wanda_scores walk "
        "(spy not triggered) -- preamble likely raised before the "
        "wanda dispatch could fire"
    )
    assert captured["ctx_has_slot"] is True, (
        "C-1 REGRESSION: run_ctx.has('stage3.wanda_scalar_row') was "
        f"{captured['ctx_has_slot']} at the collect_wanda_scores walk. "
        "The orchestrator's V2 wanda cache dispatch is NOT promoting "
        "the slot onto run_ctx; the consumer plugin will always fall "
        "through to its per-layer calibration sweep."
    )


def copy_config_with_wanda_jsonl(tiny_config: dict, jsonl: Path) -> dict:
    """Return a deep-copy of tiny_config with calibration.jsonl_path
    set to the absolute ``jsonl`` and wanda_intra_expert enabled."""
    import copy as _copy
    cfg = _copy.deepcopy(tiny_config)
    cfg.setdefault("calibration", {})["jsonl_path"] = str(jsonl)
    cfg.setdefault("stage3", {}).setdefault("wanda_intra_expert", {})[
        "enabled"
    ] = True
    return cfg


# ==========================================================================
# T6 -- checkpoint kill + resume byte-equality
# ==========================================================================


def _reload_wsr(env: dict) -> object:
    """Reload vllm.calibration_wanda_scalar_row with a fresh env so the
    import-time _CAPTURE_WANDA_SCALAR_ROW gate is re-sampled."""
    sys.modules.pop("vllm.calibration_wanda_scalar_row", None)
    sys.modules.pop("vllm.calibration_hooks", None)
    for key in (
        "VLLM_CALIB_CAPTURE_WANDA_SCALAR_ROW",
        "VLLM_CALIB_CAPTURE_ROUTER",
        "VLLM_CALIB_CAPTURE_EXPERT",
    ):
        os.environ.pop(key, None)
    for k, v in env.items():
        os.environ[k] = v
    importlib.import_module("vllm.calibration_hooks")
    return importlib.import_module("vllm.calibration_wanda_scalar_row")


def _seed_layer(wsr, layer_idx: int, rank: int, n_experts: int) -> None:
    wsr._LAYER_ID_TO_RANK[layer_idx] = rank
    if rank + 1 > wsr._N_LAYERS:
        wsr._N_LAYERS = rank + 1
    if n_experts > wsr._N_EXPERTS:
        wsr._N_EXPERTS = n_experts


def test_checkpoint_resume_byte_equal(tmp_path):
    """T6 (plan §9, H-1 plan-reviewer-v1 fold). A reference run of
    2 * n_chunks chunks must equal a killed+resumed run that does
    n_chunks chunks, checkpoints, reloads the module, and processes
    the remaining n_chunks chunks. Mirrors REAP's two_segment_additivity
    test at vllm_calibration_hooks.patch:3974-4039.
    """
    pytest.importorskip("vllm.calibration_hooks")

    n_chunks_per_seg = 4
    n_experts = 4
    top_k = 2
    d_in = 6

    # Deterministic data.
    torch.manual_seed(7)
    chunks_hs = [
        torch.randn(3, d_in, dtype=torch.float32)
        for _ in range(2 * n_chunks_per_seg)
    ]
    chunks_tw = [
        torch.rand(3, top_k, dtype=torch.float32) + 0.05
        for _ in range(2 * n_chunks_per_seg)
    ]
    chunks_ids = [
        torch.randint(0, n_experts, (3, top_k), dtype=torch.int64)
        for _ in range(2 * n_chunks_per_seg)
    ]

    # ---- Reference: single uninterrupted run ----------------------------
    ref = _reload_wsr({
        "VLLM_CALIB_CAPTURE_WANDA_SCALAR_ROW": "1",
        "VLLM_CALIB_CAPTURE_ROUTER": "1",
        "VLLM_CALIB_CAPTURE_EXPERT": "1",
    })
    _seed_layer(ref, layer_idx=0, rank=0, n_experts=n_experts)
    for hs, tw, ids in zip(chunks_hs, chunks_tw, chunks_ids):
        ref._on_router(layer_idx=0,
                       router_logits=torch.zeros(hs.shape[0], n_experts),
                       topk_weights=tw, topk_ids=ids)
        ref._on_expert_in(layer_idx=0, hidden_states=hs, topk_ids=ids)
    expected_sum = {
        k: v.clone() for k, v in ref._WANDA_SCALAR_ROW_SUM.items()
    }
    expected_counts = dict(ref._WANDA_TOKEN_COUNTS)

    # ---- Killed + resumed run -------------------------------------------
    seg = _reload_wsr({
        "VLLM_CALIB_CAPTURE_WANDA_SCALAR_ROW": "1",
        "VLLM_CALIB_CAPTURE_ROUTER": "1",
        "VLLM_CALIB_CAPTURE_EXPERT": "1",
    })
    _seed_layer(seg, layer_idx=0, rank=0, n_experts=n_experts)
    for hs, tw, ids in zip(chunks_hs[:n_chunks_per_seg],
                           chunks_tw[:n_chunks_per_seg],
                           chunks_ids[:n_chunks_per_seg]):
        seg._on_router(layer_idx=0,
                       router_logits=torch.zeros(hs.shape[0], n_experts),
                       topk_weights=tw, topk_ids=ids)
        seg._on_expert_in(layer_idx=0, hidden_states=hs, topk_ids=ids)
    seg.set_n_prompts_accumulated(7)
    ckpt = str(tmp_path / "wsr.ckpt")
    seg.dump_wanda_scalar_row_checkpoint(ckpt)

    # Simulate process death + restart by reloading the module from
    # scratch.
    seg2 = _reload_wsr({
        "VLLM_CALIB_CAPTURE_WANDA_SCALAR_ROW": "1",
        "VLLM_CALIB_CAPTURE_ROUTER": "1",
        "VLLM_CALIB_CAPTURE_EXPERT": "1",
    })
    loaded_prompts = seg2.load_wanda_scalar_row_checkpoint(ckpt)
    assert loaded_prompts == 7
    # Second half: process the remaining chunks.
    for hs, tw, ids in zip(chunks_hs[n_chunks_per_seg:],
                           chunks_tw[n_chunks_per_seg:],
                           chunks_ids[n_chunks_per_seg:]):
        seg2._on_router(layer_idx=0,
                        router_logits=torch.zeros(hs.shape[0], n_experts),
                        topk_weights=tw, topk_ids=ids)
        seg2._on_expert_in(layer_idx=0, hidden_states=hs, topk_ids=ids)

    # ---- Near-byte-equality (rtol=0, atol=1e-5) -------------------------
    # (Note: REAP's two_segment_additivity test uses the same bound at
    # patch:4038; "byte-equality" in the plan is shorthand for sub-ULP
    # drift admissible in principle but absent for fp32 sum-of-squares
    # additivity in practice.)
    assert set(seg2._WANDA_SCALAR_ROW_SUM.keys()) == set(expected_sum.keys())
    for k, expected in expected_sum.items():
        got = seg2._WANDA_SCALAR_ROW_SUM[k]
        assert torch.allclose(got, expected, rtol=0, atol=1e-5), (
            f"sum mismatch at {k}: got {got}, expected {expected}"
        )
    for k, expected_count in expected_counts.items():
        assert seg2._WANDA_TOKEN_COUNTS[k] == expected_count, (
            f"count mismatch at {k}"
        )

    # ---- Final-dump + manifest re-load integrity -------------------------
    jsonl = tmp_path / "trace.jsonl"
    seg2.dump_wanda_scalar_row(jsonl)
    loaded = load_wanda_scalar_row(jsonl)
    assert loaded is not None
    # The dump emits the running MEAN (sum / count); cross-check that:
    for k, expected_sum_t in expected_sum.items():
        count = expected_counts[k]
        expected_mean = expected_sum_t / float(max(1, count))
        got = loaded.sigma_x_g_squared[k]
        assert torch.allclose(got, expected_mean, rtol=0, atol=1e-5), (
            f"final dump mean mismatch at {k}"
        )
        assert loaded.token_counts[k] == count
