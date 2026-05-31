"""Per-item fast unit tests for the Stage-3 Tier-1 (byte-identical) speedups.

These are the PRIMARY correctness gate for items 2, 8, 9, 10 (the end-to-end
Stage-3 golden's ``alpha_grid=[0.5]`` skips the α paths, so item 2 is invisible
to it). Item 3 is DEMOTED out of Tier-1; its disproof test codifies WHY a naive
``_group_stat`` reuse is NOT byte-safe.

No production code is monkeypatched: tests build real fused-experts modules and
real ``MoELayerRef`` / ``InputCovarianceAccumulator`` objects and exercise the
production functions directly. LAPACK (cholesky / svdvals / eigh) is required
for the item-2 / item-3 tensors; the suite is seeded on a LAPACK-enabled build.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

try:
    import torch
    import torch.nn as nn
except Exception as e:  # pragma: no cover - import guard
    pytest.skip(f"torch unavailable: {e}", allow_module_level=True)


# ---------------------------------------------------------------------------
# Shared helpers — a real fused experts module + MoELayerRef so build_banks()
# (production) runs unpatched.
# ---------------------------------------------------------------------------


class _FusedExperts(nn.Module):
    """Minimal fused-layout experts module matching ``_is_fused_experts``.

    ``gate_up_proj`` : [N, 2·d_int, d_hid]  (gate = first half, up = second)
    ``down_proj``    : [N, d_hid, d_int]
    """

    def __init__(self, n_experts, d_int, d_hid, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.num_experts = n_experts
        self.gate_up_proj = nn.Parameter(
            torch.randn(n_experts, 2 * d_int, d_hid, generator=g)
        )
        self.down_proj = nn.Parameter(
            torch.randn(n_experts, d_hid, d_int, generator=g)
        )


def _make_layer_ref(layer_idx, experts):
    from moe_compress.utils.model_io import MoELayerRef

    dummy = nn.Identity()
    return MoELayerRef(
        layer_idx=layer_idx,
        layer_module=dummy,
        mlp=dummy,
        router=dummy,
        experts_module=experts,
        shared_expert=None,
        layer_type="unknown",
    )


def _build_acov(layer_idx, experts, d_hid, d_int, seed=1):
    """Build a per-(layer,expert,matrix) SPD covariance dict for the bank dims.

    gate_proj / up_proj inputs are [d_hid] (cov [d_hid, d_hid]); down_proj
    input is [d_int] (cov [d_int, d_int]). Per-expert distinct so the spectra
    are non-degenerate.
    """
    A_cov = {}
    n = experts.num_experts
    for e in range(n):
        g = torch.Generator().manual_seed(seed + e)
        for name, dim in (("gate_proj", d_hid), ("up_proj", d_hid),
                          ("down_proj", d_int)):
            M = torch.randn(dim, dim, generator=g)
            A_cov[(layer_idx, e, name)] = (M @ M.T) + torch.eye(dim)
    return A_cov


# ---------------------------------------------------------------------------
# Item 2 — grouped_svs cache equals recompute (byte-identity precondition).
# ---------------------------------------------------------------------------


def _group_stats_for(layer_idx, experts, d_int, d_hid):
    from moe_compress.stage3.plugins.d_rank_allocate import _GroupStats

    n = experts.num_experts
    dims = {"gate_proj": (d_int, d_hid), "up_proj": (d_int, d_hid),
            "down_proj": (d_hid, d_int)}
    gs = {}
    for name, (d_out, d_in) in dims.items():
        gs[(layer_idx, name)] = _GroupStats(
            d_out=d_out, d_in=d_in, n_experts=n,
            singular_values_mean=torch.ones(min(d_out, d_in)),
            effective_rank=float(min(d_out, d_in)) / 2.0,
            omega=n * (d_out + d_in),
        )
    return gs


def test_grouped_svs_cache_precondition_torch_equal():
    """MANDATED precondition: the proxy's per-expert spectrum (materialised via
    ``M_A = W @ L_A; svdvals(M_A)``) is ``torch.equal`` to the redistribute's
    inline ``svdvals(W @ L_A)``. Any latent rounding difference from the
    temporary would silently shift rank_map.json — this gate proves there is
    none before the cache is wired.
    """
    from moe_compress.stage3.plugins.swift_svd_alpha import (
        _swift_svd_plus_alpha_search,
    )

    layer_idx, n, d_int, d_hid = 0, 4, 6, 8
    experts = _FusedExperts(n, d_int, d_hid, seed=3)
    ref = _make_layer_ref(layer_idx, experts)
    A_cov = _build_acov(layer_idx, experts, d_hid, d_int)
    group_stats = _group_stats_for(layer_idx, experts, d_int, d_hid)
    base_ranks = {k: 3 for k in group_stats}

    # Proxy spectra (materialises M_A then svdvals(M_A)).
    _, grouped_svs = _swift_svd_plus_alpha_search(
        [ref], group_stats, base_ranks, [0.0, 0.5, 1.0],
        per_group_type=True, A_cov=A_cov, return_svs=True,
    )

    # Inline recompute exactly as _redistribute_ranks_swift_svd_plus does
    # (svdvals(W @ L_A), no M_A temporary). Tier-2 §2.2/C1: the producer now
    # emits CPU-fp64 spectra, so this inline recompute is CPU-fp64 in lockstep
    # for the torch.equal precondition to hold bit-exact.
    from moe_compress.utils.model_io import build_banks

    banks = build_banks(ref)
    for name in ("gate_proj", "up_proj", "down_proj"):
        for e in range(n):
            W = banks[name].get(e).detach().to(device="cpu", dtype=torch.float64)
            A = A_cov[(layer_idx, e, name)].to(device="cpu", dtype=torch.float64)
            A = 0.5 * (A + A.T)
            ev, evec = torch.linalg.eigh(A)
            keep = ev > ev.max() * 1e-6
            L_A = evec[:, keep] * ev[keep].clamp_min(1e-12).sqrt().unsqueeze(0)
            inline = torch.linalg.svdvals(W @ L_A)
            assert torch.equal(grouped_svs[name][(layer_idx, e)], inline), (
                f"proxy spectrum != inline recompute for {name} expert {e}"
            )


def test_grouped_svs_cache_equals_recompute():
    """``_redistribute_ranks_swift_svd_plus(grouped_svs_cache=cache)`` returns a
    rank dict ``==`` the ``grouped_svs_cache=None`` recompute path, proving the
    cache wiring is byte-identical end-to-end (not just at the spectrum level).
    """
    from moe_compress.stage3.plugins.swift_svd_alpha import (
        _swift_svd_plus_alpha_search,
        _redistribute_ranks_swift_svd_plus,
    )

    layer_idx, n, d_int, d_hid = 0, 5, 7, 9
    experts = _FusedExperts(n, d_int, d_hid, seed=11)
    ref = _make_layer_ref(layer_idx, experts)
    A_cov = _build_acov(layer_idx, experts, d_hid, d_int)
    group_stats = _group_stats_for(layer_idx, experts, d_int, d_hid)
    base_ranks = {k: 3 for k in group_stats}

    alpha_by_type, cache = _swift_svd_plus_alpha_search(
        [ref], group_stats, base_ranks, [0.0, 0.5, 1.0],
        per_group_type=True, A_cov=A_cov, return_svs=True,
    )

    out_recompute = _redistribute_ranks_swift_svd_plus(
        [ref], group_stats, base_ranks, alpha_by_type,
        grouped_svs_cache=None, A_cov=A_cov,
    )
    out_cached = _redistribute_ranks_swift_svd_plus(
        [ref], group_stats, base_ranks, alpha_by_type,
        grouped_svs_cache=cache, A_cov=A_cov,
    )
    assert out_cached == out_recompute


def test_return_svs_false_keeps_base_return_type():
    """``return_svs=False`` (default) keeps the base return type — a plain
    dict, NOT a tuple — so the re-export and pin-test contracts are intact.
    """
    from moe_compress.stage3.plugins.swift_svd_alpha import (
        _swift_svd_plus_alpha_search,
    )

    layer_idx, n, d_int, d_hid = 0, 3, 5, 6
    experts = _FusedExperts(n, d_int, d_hid, seed=7)
    ref = _make_layer_ref(layer_idx, experts)
    A_cov = _build_acov(layer_idx, experts, d_hid, d_int)
    group_stats = _group_stats_for(layer_idx, experts, d_int, d_hid)
    base_ranks = {k: 2 for k in group_stats}

    res = _swift_svd_plus_alpha_search(
        [ref], group_stats, base_ranks, [0.0, 0.5, 1.0],
        per_group_type=True, A_cov=A_cov,
    )
    assert isinstance(res, dict) and not isinstance(res, tuple)

    res_global = _swift_svd_plus_alpha_search(
        [ref], group_stats, base_ranks, [0.0, 0.5, 1.0],
        per_group_type=False, A_cov=A_cov,
    )
    assert isinstance(res_global, dict) and set(res_global) == {"all"}


# ---------------------------------------------------------------------------
# Item 3 — DISPROOF: group-avg+cholesky spectra != per-expert+eigh spectra.
# ---------------------------------------------------------------------------


def test_group_stat_vs_swift_spectra_differ():
    """Codifies the load-bearing finding: ``svdvals(cholesky(mean_A) @ W.T)``
    (the _group_stat group-averaged Cholesky-whitened spectrum) is NOT
    ``torch.allclose`` to ``svdvals(W @ eigh_factor(A_e))`` (Swift-SVD's
    per-expert eigh-whitened spectrum). This prevents a future "obvious"
    publish-and-consume reuse from silently regressing the golden.
    """
    torch.manual_seed(0)
    d_out, d_in, n = 6, 5, 4
    W = torch.randn(d_out, d_in)
    # Per-expert SPD covariances with a non-trivial group average.
    A_es = [torch.randn(d_in, d_in) for _ in range(n)]
    A_es = [(M @ M.T) + torch.eye(d_in) for M in A_es]
    A_g = torch.stack(A_es).mean(0)

    # _group_stat path: group-averaged Cholesky factor, svdvals(L_A @ W.T).
    # Tier-2 §2.4/M2: _group_stat now keeps the spectrum CPU-fp64 end-to-end
    # (the prior `.float()` cast on L_A is removed); mirror that here so this
    # disproof test stays an honest shadow of the production path. The
    # `not torch.allclose` inequality compares two structurally different
    # operators (group-avg Cholesky vs per-expert eigh) and is independent of
    # fp32 vs fp64, so it still holds.
    L_chol = torch.linalg.cholesky(A_g.double() + 1e-6 * torch.eye(d_in).double())
    s_group = torch.linalg.svdvals(L_chol @ W.double().T)

    # Swift-SVD path: per-expert eigh factor, svdvals(W @ L_eigh). Also CPU-fp64
    # to mirror the changed Swift producer (§2.2) and to share dtype with
    # ``s_group`` for the allclose comparison below.
    A0 = 0.5 * (A_es[0].double() + A_es[0].double().T)
    ev, evec = torch.linalg.eigh(A0)
    keep = ev > ev.max() * 1e-6
    L_eigh = evec[:, keep] * ev[keep].clamp_min(1e-12).sqrt().unsqueeze(0)
    s_swift = torch.linalg.svdvals(W.double() @ L_eigh)

    assert not torch.allclose(s_group, s_swift), (
        "group-avg Cholesky spectrum unexpectedly matches per-expert eigh "
        "spectrum — Item 3 reuse would NOT be byte-safe as claimed; re-audit."
    )


# ---------------------------------------------------------------------------
# Item 8 — sharded save overlap produces byte-identical output.
# ---------------------------------------------------------------------------


def test_save_state_dict_sharded_overlap_identical(tmp_path):
    """The overlapped (pooled) ``_save_state_dict_sharded`` writes byte-identical
    shard files + ``model.safetensors.index.json`` vs. a known-good serial
    reference. Forces multiple shards via a tiny ``max_shard_size_bytes``.
    """
    pytest.importorskip("safetensors")
    from safetensors.torch import save_file
    from moe_compress.utils.model_io import (
        _save_state_dict_sharded,
        save_json_artifact,
    )

    torch.manual_seed(0)
    state = {f"w{i}": torch.randn(8, 8) for i in range(7)}
    # Each tensor is 8*8*4 = 256 bytes; cap forces ~1 tensor per shard.
    cap = 300

    # Serial reference: replicate the pre-Tier-1 loop exactly.
    ref_dir = tmp_path / "ref"
    ref_dir.mkdir()
    shards = [{}]
    cur = 0
    for k, t in state.items():
        nb = t.element_size() * t.numel()
        if cur + nb > cap and shards[-1]:
            shards.append({})
            cur = 0
        shards[-1][k] = t
        cur += nb
    n_shards = len(shards)
    wmap = {}
    total = 0
    for i, shard in enumerate(shards):
        name = f"model-{i + 1:05d}-of-{n_shards:05d}.safetensors"
        cpu = {k: v.detach().cpu().contiguous() for k, v in shard.items()}
        save_file(cpu, str(ref_dir / name))
        for k in shard:
            wmap[k] = name
        total += sum(t.element_size() * t.numel() for t in shard.values())
    # Use the SAME index serializer production uses so the byte-comparison
    # below isolates the shard-write overlap change, not JSON formatting.
    save_json_artifact(
        {"metadata": {"total_size": total}, "weight_map": wmap},
        ref_dir / "model.safetensors.index.json",
    )

    # Production overlapped writer.
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _save_state_dict_sharded(state, out_dir, max_shard_size_bytes=cap)

    ref_files = sorted(p.name for p in ref_dir.iterdir())
    out_files = sorted(p.name for p in out_dir.iterdir())
    assert out_files == ref_files, (out_files, ref_files)
    for name in ref_files:
        assert (out_dir / name).read_bytes() == (ref_dir / name).read_bytes(), name


# ---------------------------------------------------------------------------
# Item 9 — bcov prefetch matches serial accumulate.
# ---------------------------------------------------------------------------


def _spill_layer(acc, layer_idx, dir_path, seed):
    """Populate + spill one layer's covariance via the production spill path."""
    g = torch.Generator().manual_seed(seed)
    # Two experts, one matrix each — small SPD-ish accumulations.
    for e in range(2):
        x = torch.randn(4, 6, generator=g)
        acc.update(layer_idx, e, "gate_proj", x)
    acc.finalize_layer(layer_idx)
    acc.spill_layer_to_disk(layer_idx, dir_path)


def test_bcov_prefetch_matches_serial(tmp_path):
    """Loading 3 per-layer spills via ``BcovLayerPrefetcher`` (consume +
    depth-1 prefetch) yields key-for-key ``torch.equal`` ``covariance`` /
    ``token_count`` vs. serial ``load_layer_from_disk``.
    """
    from moe_compress.utils.activation_hooks import (
        InputCovarianceAccumulator,
        BcovLayerPrefetcher,
    )

    spill_dir = tmp_path / "spill"
    # Producer accumulator writes 3 layer spills.
    producer = InputCovarianceAccumulator()
    for li in (0, 1, 2):
        _spill_layer(producer, li, spill_dir, seed=100 + li)

    # Serial consumer.
    serial = InputCovarianceAccumulator()
    for li in (0, 1, 2):
        assert serial.load_layer_from_disk(li, spill_dir) is True

    # Prefetched consumer — mirror the factor-loop drive pattern.
    pref = InputCovarianceAccumulator()
    layers = [0, 1, 2]
    pf = BcovLayerPrefetcher(pref, spill_dir)
    try:
        for i, li in enumerate(layers):
            assert pf.consume(li) is True
            if i + 1 < len(layers):
                pf.prefetch(layers[i + 1])
    finally:
        pf.shutdown()

    assert serial.covariance.keys() == pref.covariance.keys()
    for k in serial.covariance:
        assert torch.equal(serial.covariance[k], pref.covariance[k]), k
    assert dict(serial.token_count) == dict(pref.token_count)


def test_bcov_prefetch_missing_spill_returns_false(tmp_path):
    """A prefetched-but-absent spill (resume case) consumes as ``False`` — same
    semantics as the serial loader's missing-file path."""
    from moe_compress.utils.activation_hooks import (
        InputCovarianceAccumulator,
        BcovLayerPrefetcher,
    )

    spill_dir = tmp_path / "spill"
    spill_dir.mkdir()
    acc = InputCovarianceAccumulator()
    pf = BcovLayerPrefetcher(acc, spill_dir)
    try:
        pf.prefetch(0)
        assert pf.consume(0) is False
    finally:
        pf.shutdown()


# ---------------------------------------------------------------------------
# Item 10 — originals manifest with sha256=None validates; payload identical.
# ---------------------------------------------------------------------------


def test_originals_manifest_no_sha_validates(tmp_path):
    """``write_manifest_last(compute_sha256=False)`` yields a manifest that
    ``read_and_validate_manifest`` passes with ``sha256 is None``, and the
    payload ``.pt`` bytes are identical to a ``compute_sha256=True`` write of
    the same dict (only the manifest's sha field differs).
    """
    from moe_compress.utils.atomic_io import (
        atomic_torch_save,
        write_manifest_last,
        read_and_validate_manifest,
    )

    payload = {"a": torch.arange(10), "b": torch.ones(3, 3)}

    # Identical payload FILENAME in two dirs so the .pt zip's embedded
    # filename does not perturb the byte comparison — the only intended
    # variable is the compute_sha256 flag.
    d_nosha = tmp_path / "nosha"
    d_sha = tmp_path / "sha"
    d_nosha.mkdir()
    d_sha.mkdir()

    # No-sha write (the Tier-1 item-10 behaviour).
    p1 = d_nosha / "_stage3_original_weights.pt"
    m1 = d_nosha / "_stage3_original_weights.pt.MANIFEST.json"
    atomic_torch_save(p1, payload)
    write_manifest_last(p1, m1, schema_version=1, compute_sha256=False)

    # With-sha write of the SAME dict for payload-byte comparison.
    p2 = d_sha / "_stage3_original_weights.pt"
    m2 = d_sha / "_stage3_original_weights.pt.MANIFEST.json"
    atomic_torch_save(p2, payload)
    write_manifest_last(p2, m2, schema_version=1, compute_sha256=True)

    # Payload bytes identical (only the manifest sha field differs).
    assert p1.read_bytes() == p2.read_bytes()

    man = json.loads(m1.read_text())
    assert man["sha256"] is None

    # Validates with the default require_sha256=False (Stage 4's call shape).
    read_and_validate_manifest(p1, m1, expected_schema_version=1)
