"""S3-4 — Swift-SVD+ α-search plugin extraction tests.

Verifies the pure-relocation of the eight Swift-SVD+ α-selection symbols out of
the ``stage3_svd.py`` monolith into ``stage3/plugins/swift_svd_alpha.py``:

* ``_snapshot_originals`` / ``_build_wikitext2_validation`` /
  ``_evaluate_wikitext2_ppl`` / ``_factor_model_at_ranks`` /
  ``_restore_fused_experts`` / ``_swift_svd_plus_alpha_search_validation`` /
  ``_swift_svd_plus_alpha_search`` / ``_redistribute_ranks_swift_svd_plus``;
* the plugin module exposes the relocated symbols;
* the monolith RE-IMPORTS them (identity, not copy) so ``run()`` and external
  callers keep their import paths;
* ``SwiftSvdAlphaPlugin`` satisfies the universal ``PipelinePlugin`` Protocol,
  carries correct metadata, is unconditionally enabled, and exposes the (S3-7)
  ``select_alpha`` phase hook;
* the empty-tensor PPL guard and the rank-redistribution budget conservation
  still behave as in the monolith.

The byte-identical behavioral gate is the S3-0 golden snapshot
(``test_stage3_golden_snapshot.py``); this file only checks the relocation
plumbing plus a couple of pure-unit logic assertions. The model-driven paths —
``_snapshot_originals`` / ``_factor_model_at_ranks`` / ``_restore_fused_experts``
/ ``_swift_svd_plus_alpha_search_validation`` and the AA-SVD-core lazy-import
escape — need a real fused-experts model and are covered by the smoke / golden
suites rather than re-exercised here.
"""
from __future__ import annotations

import torch


_SWIFT_SVD_SYMBOLS = (
    "_snapshot_originals",
    "_build_wikitext2_validation",
    "_evaluate_wikitext2_ppl",
    "_factor_model_at_ranks",
    "_restore_fused_experts",
    "_swift_svd_plus_alpha_search_validation",
    "_swift_svd_plus_alpha_search",
    "_redistribute_ranks_swift_svd_plus",
)


def test_swift_svd_module_imports():
    """The 8 relocated functions + ``SwiftSvdAlphaPlugin`` import from the
    plugin module."""
    from moe_compress.stage3.plugins import swift_svd_alpha
    from moe_compress.stage3.plugins.swift_svd_alpha import SwiftSvdAlphaPlugin

    for name in _SWIFT_SVD_SYMBOLS:
        assert callable(getattr(swift_svd_alpha, name)), name
    assert isinstance(SwiftSvdAlphaPlugin, type)


def test_monolith_reexports_swift_svd_symbols():
    """The monolith re-imports the relocated symbols — identity, not copy.

    ``IS`` identity proves ``stage3_svd`` holds the *same* objects as the
    plugin module (a re-import), not independent copies that could drift.
    """
    import moe_compress.stage3_svd as monolith
    import moe_compress.stage3.plugins.swift_svd_alpha as plugin

    for name in _SWIFT_SVD_SYMBOLS:
        assert getattr(monolith, name) is getattr(plugin, name), name


def test_plugin_satisfies_protocol():
    """``SwiftSvdAlphaPlugin`` structurally satisfies ``PipelinePlugin``."""
    from moe_compress.pipeline.plugin import PipelinePlugin
    from moe_compress.stage3.plugins.swift_svd_alpha import SwiftSvdAlphaPlugin

    assert isinstance(SwiftSvdAlphaPlugin(), PipelinePlugin)


def test_plugin_metadata():
    """Plugin metadata — name / paper id / config_key / tuple-typed fields."""
    from moe_compress.stage3.plugins.swift_svd_alpha import SwiftSvdAlphaPlugin

    plugin = SwiftSvdAlphaPlugin()
    assert plugin.name == "swift_svd_alpha"
    assert "2604.01609" in plugin.paper
    assert plugin.config_key == "stage3_svd.swift_svd_plus.alpha_grid"
    assert isinstance(plugin.reads, tuple)
    assert isinstance(plugin.writes, tuple)
    assert isinstance(plugin.provides, tuple)


def test_plugin_is_enabled_unconditional():
    """Swift-SVD+ α selection is UNCONDITIONAL — ``is_enabled`` always True.

    An ``alpha_grid`` of length ≤ 1 yields the uniform path; it does not
    disable the phase. ``config_key`` only parametrises the grid.
    """
    from moe_compress.stage3.plugins.swift_svd_alpha import SwiftSvdAlphaPlugin

    plugin = SwiftSvdAlphaPlugin()
    assert plugin.is_enabled({}) is True
    assert plugin.is_enabled(
        {"stage3_svd": {"swift_svd_plus": {"alpha_grid": [0.0, 0.5, 1.0]}}}
    ) is True


def test_plugin_has_select_alpha_hook():
    """The S3-7 phase hook ``select_alpha`` is present and callable."""
    from moe_compress.stage3.plugins.swift_svd_alpha import SwiftSvdAlphaPlugin

    plugin = SwiftSvdAlphaPlugin()
    assert callable(getattr(plugin, "select_alpha", None))


def test_evaluate_wikitext2_ppl_empty_tensor():
    """``_evaluate_wikitext2_ppl`` returns ``inf`` for an empty validation
    tensor — the ``numel() == 0`` guard. Pure, no model is touched.
    """
    from moe_compress.stage3.plugins.swift_svd_alpha import _evaluate_wikitext2_ppl

    empty = torch.empty((0, 0), dtype=torch.long)
    # model is never reached: the numel()==0 guard returns before any forward.
    ppl = _evaluate_wikitext2_ppl(None, empty, device=None)
    assert ppl == float("inf")


class _FakeBank:
    """Minimal duck-typed expert bank for ``_redistribute_ranks_swift_svd_plus``
    — only ``.get(e)`` is touched (``A_cov=None`` skips ``.shape()``)."""

    def __init__(self, weights):
        self._w = weights  # list of [d_out, d_in] tensors

    def get(self, e):
        return self._w[e]


class _FakeRef:
    """Minimal MoE layer ref — ``_redistribute_ranks_swift_svd_plus`` only
    filters ``moe_layers`` by ``.layer_idx``."""

    def __init__(self, layer_idx):
        self.layer_idx = layer_idx


def test_redistribute_ranks_budget_conservation(monkeypatch):
    """``_redistribute_ranks_swift_svd_plus`` conserves the per-group rank
    budget: sum of per-expert ranks == base_rank × n_experts.

    ``A_cov=None`` takes the raw-``svdvals`` branch — the ``_cov_lookup``
    whitening path (and thus the AA-SVD-core lazy import) is not exercised, so
    no model is needed. ``build_banks`` is monkeypatched to a stub returning
    synthetic banks.
    """
    from moe_compress.stage3.plugins import swift_svd_alpha
    from moe_compress.stage3.plugins.swift_svd_alpha import (
        _GroupStats,
        _redistribute_ranks_swift_svd_plus,
    )

    torch.manual_seed(0)
    n_experts, d_out, d_in = 5, 32, 24
    # One (layer, matrix) group. Square-ish so the cap min(d_out,d_in)-1 = 23
    # comfortably exceeds the per-expert ranks at base_rank=8.
    layer_idx, matrix = 0, "gate_proj"

    fake_weights = [torch.randn(d_out, d_in) for _ in range(n_experts)]
    fake_banks = {matrix: _FakeBank(fake_weights)}

    def _stub_build_banks(ref):
        # ref is the single-element list's [0]; ignore it, return the stub.
        return fake_banks

    monkeypatch.setattr(swift_svd_alpha, "build_banks", _stub_build_banks)

    group_stats = {
        (layer_idx, matrix): _GroupStats(
            d_out=d_out,
            d_in=d_in,
            n_experts=n_experts,
            singular_values_mean=torch.ones(min(d_out, d_in)),
            effective_rank=float(min(d_out, d_in)) / 2.0,
            omega=n_experts * (d_out + d_in),
        ),
    }
    base_rank = 8
    base_ranks = {(layer_idx, matrix): base_rank}
    moe_layers = [_FakeRef(layer_idx)]
    alpha_by_type = {"all": 0.5}

    out = _redistribute_ranks_swift_svd_plus(
        moe_layers, group_stats, base_ranks, alpha_by_type, A_cov=None,
    )

    # One entry per (layer, matrix, expert).
    assert set(out.keys()) == {
        (layer_idx, matrix, e) for e in range(n_experts)
    }
    # Budget conservation: total per-expert rank == base_rank × n_experts.
    assert sum(out.values()) == base_rank * n_experts
    # Every expert keeps at least the δ=0.5 rank floor and stays under the cap.
    cap = min(d_out, d_in) - 1
    assert all(1 <= r <= cap for r in out.values())


# ---------------------------------------------------------------------------
# A2 — eigh-decomp spill cache across alpha candidates
# ---------------------------------------------------------------------------


def test_a2_eigh_decomp_roundtrip():
    """A2: ``_EighDecomp`` serialize→load preserves every field bit-for-bit;
    the ``None`` ValueError-sentinel round-trips as ``None``."""
    from moe_compress.stage3.plugins.aa_svd_factor import (
        _EighDecomp, _precompute_eigh,
    )
    from moe_compress.stage3.plugins.swift_svd_alpha import (
        _serialize_eigh_decomp, _deserialize_eigh_decomp, _EIGH_DECOMP_FIELDS,
    )

    torch.manual_seed(7)
    d = 12
    X = torch.randn(40, d)
    B = X.T @ X  # PSD Gram with positive eigenvalues
    decomp = _precompute_eigh(B, None, None, device=torch.device("cpu"))

    payload = _serialize_eigh_decomp(decomp)
    back = _deserialize_eigh_decomp(payload, _EighDecomp)
    for f in _EIGH_DECOMP_FIELDS:
        a, b = getattr(decomp, f), getattr(back, f)
        if isinstance(a, torch.Tensor):
            assert torch.equal(a, b), f"field {f} differs after round-trip"
        else:
            assert a == b, f"scalar field {f} differs after round-trip"

    # None sentinel round-trips as None.
    assert _serialize_eigh_decomp(None) is None
    assert _deserialize_eigh_decomp(None, _EighDecomp) is None


def _build_cov_for_layers(moe_layers, *, cross, seed):
    """Populate + finalize a B (and optional C) accumulator with random PSD
    covariances for every (layer, expert) so the eigh path engages."""
    from moe_compress.utils.activation_hooks import InputCovarianceAccumulator
    from moe_compress.utils.model_io import MATRIX_NAMES

    torch.manual_seed(seed)
    B = InputCovarianceAccumulator(); B.set_storage_dtype(torch.float32)
    C = None
    if cross:
        C = InputCovarianceAccumulator(); C.set_storage_dtype(torch.float32)
    for ref in moe_layers:
        ex = ref.experts_module
        d_hid = ex.gate_up_proj.shape[-1]
        d_int = ex.gate_up_proj.shape[1] // 2
        for e in range(ref.num_routed_experts):
            # gate_proj (up_proj aliases to it inside the accumulator).
            xg = torch.randn(32, d_hid)
            B.update(ref.layer_idx, e, "gate_proj", xg)
            # down_proj input is intermediate-dim.
            xd = torch.randn(32, d_int)
            B.update(ref.layer_idx, e, "down_proj", xd)
            if C is not None:
                # cross-cov X_pre^T @ X_post (gate_proj only, per the pipeline).
                xpre = torch.randn(32, d_hid)
                C.update_cross(
                    ref.layer_idx, e, "gate_proj", xpre.T @ xg, n_tokens=32,
                )
        B.finalize_layer(ref.layer_idx)
        if C is not None:
            C.finalize_layer(ref.layer_idx)
    return B, C


def _spill_cov(acc, moe_layers, dir_path):
    for ref in moe_layers:
        acc.spill_layer_to_disk(ref.layer_idx, dir_path)


def _factored_uv_snapshot(moe_layers):
    """Snapshot every installed FactoredExperts U/V tensor for comparison."""
    from moe_compress.utils.model_io import MATRIX_NAMES
    snap = {}
    for ref in moe_layers:
        ex = ref.experts_module
        for e in range(ref.num_routed_experts):
            for name in MATRIX_NAMES:
                snap[(ref.layer_idx, e, name, "U")] = getattr(ex, f"{name}_U")[e].clone()
                snap[(ref.layer_idx, e, name, "V")] = getattr(ex, f"{name}_V")[e].clone()
    return snap


def _run_cache_test(model_factory, tmp_path, cross):
    """A2: factoring with the per-layer eigh spill cache ENABLED produces
    BYTE-IDENTICAL FactoredExperts U/V (``torch.equal``) to factoring with the
    cache DISABLED, across two distinct rank allocations (mimicking the
    alpha-major candidate loop where candidate 0 fills the cache and candidate
    1 reads it)."""
    import copy
    from moe_compress.stage3.plugins.swift_svd_alpha import (
        _snapshot_originals, _factor_model_at_ranks, _restore_fused_experts,
    )
    from moe_compress.utils.model_io import iter_moe_layers, MATRIX_NAMES

    device = torch.device("cpu")
    bdir = tmp_path / "bcov"; bdir.mkdir()
    cdir = (tmp_path / "ccov"); cdir.mkdir() if cross else None
    ccov_dir = cdir if cross else None

    def make():
        m = copy.deepcopy(model_factory).eval()
        return m, list(iter_moe_layers(m))

    # Two rank allocations to mimic two alpha candidates.
    def ranks_for(moe, scale):
        ex0 = moe[0].experts_module
        d_int = ex0.gate_up_proj.shape[1] // 2
        d_hid = ex0.gate_up_proj.shape[-1]
        base = {}
        for ref in moe:
            for name in MATRIX_NAMES:
                cap = min(
                    (d_int if name != "down_proj" else d_hid),
                    (d_hid if name != "down_proj" else d_int),
                )
                base[(ref.layer_idx, name)] = max(1, min(scale, cap - 1))
        return base, {}

    results = {}
    for use_cache in (False, True):
        m, moe = make()
        originals = _snapshot_originals(moe)
        Bacc, Cacc = _build_cov_for_layers(moe, cross=cross, seed=3)
        _spill_cov(Bacc, moe, bdir)
        if cross:
            _spill_cov(Cacc, moe, ccov_dir)
        # Drop in-memory cov so factor must load from spill (matches prod).
        for ref in moe:
            Bacc.unload_layer(ref.layer_idx)
            if cross:
                Cacc.unload_layer(ref.layer_idx)

        cache_dir = str(tmp_path / f"eigh_cache_{cross}") if use_cache else None
        if use_cache:
            import shutil
            shutil.rmtree(cache_dir, ignore_errors=True)

        snaps = []
        for ci, scale in enumerate((2, 3)):
            base_ranks, per_expert = ranks_for(moe, scale)
            _factor_model_at_ranks(
                m, moe, originals, per_expert, base_ranks,
                {}, Bacc, bdir, Cacc, ccov_dir,
                device=device, storage_dtype=torch.float32,
                gate_up_decomp_cache_dir=cache_dir,
            )
            snaps.append(_factored_uv_snapshot(moe))
            _restore_fused_experts(m, moe, originals, device=device)
        results[use_cache] = snaps

    # Cache-on must match cache-off for BOTH candidates, every key.
    for ci in range(2):
        off = results[False][ci]
        on = results[True][ci]
        assert set(off) == set(on)
        for k in off:
            assert torch.equal(off[k], on[k]), \
                f"candidate {ci}: cache-on != cache-off at {k}"


def test_a2_eigh_cache_byte_identical_nocross(tiny_model, tmp_path):
    _run_cache_test(tiny_model, tmp_path, cross=False)


def test_a2_eigh_cache_byte_identical_cross(tiny_model, tmp_path):
    _run_cache_test(tiny_model, tmp_path, cross=True)
