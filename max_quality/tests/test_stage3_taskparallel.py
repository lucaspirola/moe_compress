"""Stage 3 task-parallel levers — equivalence + resolution + determinism.

Two RESULT-PRESERVING, default-OFF multi-GPU levers in Stage 3:

* **Lever 2 (per-expert SVD factor)** — ``aa_svd_factor.factor_layer`` fans the
  per-expert AA-SVD solve across worker devices via the SAME EoRA concurrency
  engine (``_run_expert_bands`` / ``_resolve_worker_devices``), assembling rows
  in ascending-e on the main thread. Default ``factor_workers=1`` ⇒ inline
  serial, byte-identical.

* **Lever 1 (α-grid)** — ``swift_svd_alpha`` distributes the 11-candidate
  WikiText-2 PPL grid across process-spawn replicas; the parent argmins on a
  completion-order-independent ``(grid_idx, alpha, ppl)`` fold. Default
  ``alpha_workers`` absent ⇒ serial.

All run WITHOUT a real multi-GPU box: worker devices are CPU via the
``factor_worker_devices`` / ``eora_worker_devices`` ctx seam (no monkeypatch —
project rule). Live ≥2-GPU validation is deferred to a real box.
"""
from __future__ import annotations

import torch


# --------------------------------------------------------------------------- #
# Lever 2 — T2.0: the EoRA concurrency engine is importable for reuse.
# --------------------------------------------------------------------------- #
def test_factor_engine_imports():
    """Lever 2 reuses the landed EoRA band engine in place (Q1 option (a)).

    ``_run_expert_bands`` + ``_resolve_worker_devices`` import from
    ``stage4.plugins.eora_compensation`` — a deliberate stage3→stage4 reuse of
    the stage-agnostic concurrency engine (the EoRA file + its golden are left
    untouched; Lever 2 only IMPORTS them).
    """
    from moe_compress.stage4.plugins.eora_compensation import (
        _resolve_worker_devices,
        _run_expert_bands,
    )

    assert callable(_run_expert_bands)
    assert callable(_resolve_worker_devices)


# --------------------------------------------------------------------------- #
# Lever 2 fixture: a tiny single-expert AA-SVD case (originals + B/A/C cov).
# --------------------------------------------------------------------------- #
def _make_spd(d: int, seed: int) -> torch.Tensor:
    """Well-conditioned SPD covariance [d, d] (fp32, CPU)."""
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(4 * d, d, generator=g, dtype=torch.float32)
    return (X.T @ X) / X.shape[0] + torch.eye(d)


def _tile_case(layer_idx=0, e=0, hidden=8, inter=12, k=2, seed=0):
    """Build the inputs ``_factor_expert_tile`` consumes for one expert.

    Returns ``(originals, A_cov, B_cov, C_cov, per_expert_ranks, ranks_layer)``
    where the cov dicts are plain ``{(layer, expert, name): tensor}`` maps
    (the tile reads ``B_acc.covariance`` / ``C_acc.covariance`` style dicts,
    NOT the accumulator objects — the main thread owns load/unload).
    gate_proj / up_proj share the hidden-dim B/A/C (cov_lookup up→gate
    fallback); down_proj gets its own inter-dim B/A.
    """
    g = torch.Generator().manual_seed(seed)
    originals = {}
    for name in ("gate_proj", "up_proj", "down_proj"):
        d_out, d_in = (inter, hidden) if name != "down_proj" else (hidden, inter)
        originals[(layer_idx, e, name)] = torch.randn(
            d_out, d_in, generator=g, dtype=torch.float32
        )
    A_cov = {
        (layer_idx, e, "gate_proj"): _make_spd(hidden, seed=100),
        (layer_idx, e, "down_proj"): _make_spd(inter, seed=200),
    }
    B_cov = {
        (layer_idx, e, "gate_proj"): _make_spd(hidden, seed=300),
        (layer_idx, e, "down_proj"): _make_spd(inter, seed=400),
    }
    C_cov = {
        (layer_idx, e, "gate_proj"): _make_spd(hidden, seed=500),
        (layer_idx, e, "down_proj"): _make_spd(inter, seed=600),
    }
    ranks_layer = {"gate_proj": k, "up_proj": k, "down_proj": k}
    per_expert_ranks = {
        (layer_idx, "gate_proj", e): k,
        (layer_idx, "up_proj", e): k,
        (layer_idx, "down_proj", e): k,
    }
    return originals, A_cov, B_cov, C_cov, per_expert_ranks, ranks_layer


def test_factor_expert_tile_pure():
    """``_factor_expert_tile`` on two CPU 'devices' gives identical tiles.

    Pure per-expert AA-SVD solve — same inputs on two target "devices" must give
    bit-identical ``(U_k, V_k)`` and equal ``(k_eff, k)`` per matrix. Mirrors
    ``test_solve_expert_tile_pure`` (test_stage4_multigpu.py)."""
    from moe_compress.stage3.plugins.aa_svd_factor import _factor_expert_tile

    originals, A_cov, B_cov, C_cov, per_expert_ranks, ranks_layer = _tile_case()

    out0 = _factor_expert_tile(
        0, 0, torch.device("cpu"),
        originals=originals, A_cov=A_cov, B_cov=B_cov, C_cov=C_cov,
        per_expert_ranks=per_expert_ranks, ranks_layer=ranks_layer,
        B_cov_dtype=torch.float32,
    )
    out1 = _factor_expert_tile(
        0, 0, torch.device("cpu"),
        originals=originals, A_cov=A_cov, B_cov=B_cov, C_cov=C_cov,
        per_expert_ranks=per_expert_ranks, ranks_layer=ranks_layer,
        B_cov_dtype=torch.float32,
    )
    assert set(out0.keys()) == {"gate_proj", "up_proj", "down_proj"}
    for name in ("gate_proj", "up_proj", "down_proj"):
        U0, V0, keff0, k0, err0 = out0[name]
        U1, V1, keff1, k1, err1 = out1[name]
        assert torch.equal(U0, U1)
        assert torch.equal(V0, V1)
        assert keff0 == keff1
        assert k0 == k1 == 2
        # Padded to the per-layer slot width.
        assert U0.shape[1] == ranks_layer[name]
        assert V0.shape[0] == ranks_layer[name]


def test_factor_expert_tile_threads_B_cov_dtype():
    """N1: ``B_cov_dtype`` must reach the tile's eigh noise floor.

    A bf16 ``B_cov_dtype`` raises the relative noise floor vs fp32; on a
    cov whose spectrum has directions between the two floors the kept-rank
    (k_eff) differs. The tile must thread the dtype through (not silently use
    the fp32 default), so the two dtypes can disagree — proving the arg is
    live, not ignored."""
    from moe_compress.stage3.plugins.aa_svd_factor import _factor_expert_tile

    # A cov with a wide eigenvalue spread so the bf16 floor (~7.8e-3) and the
    # fp32 floor (~1e-6) keep DIFFERENT numbers of directions.
    g = torch.Generator().manual_seed(7)
    hidden, inter, k = 8, 12, 6
    Q, _ = torch.linalg.qr(torch.randn(hidden, hidden, generator=g))
    # Geometric spectrum spanning the gap between fp32 and bf16 floors.
    lam = torch.logspace(0, -5, hidden)
    B = (Q * lam.unsqueeze(0)) @ Q.T
    B = 0.5 * (B + B.T)
    originals = {
        (0, 0, n): torch.randn((inter, hidden) if n != "down_proj" else (hidden, inter),
                               generator=g, dtype=torch.float32)
        for n in ("gate_proj", "up_proj", "down_proj")
    }
    B_cov = {(0, 0, "gate_proj"): B, (0, 0, "down_proj"): _make_spd(inter, 9)}
    A_cov = {(0, 0, "gate_proj"): B.clone(), (0, 0, "down_proj"): _make_spd(inter, 9)}
    ranks_layer = {"gate_proj": k, "up_proj": k, "down_proj": k}
    pe = {(0, n, 0): k for n in ("gate_proj", "up_proj", "down_proj")}

    common = dict(originals=originals, A_cov=A_cov, B_cov=B_cov, C_cov=None,
                  per_expert_ranks=pe, ranks_layer=ranks_layer)
    out_fp32 = _factor_expert_tile(0, 0, torch.device("cpu"),
                                   B_cov_dtype=torch.float32, **common)
    out_bf16 = _factor_expert_tile(0, 0, torch.device("cpu"),
                                   B_cov_dtype=torch.bfloat16, **common)
    # The bf16 floor truncates more directions ⇒ a smaller (or equal) k_eff on
    # gate_proj. Assert it is never LARGER (the dtype is genuinely consumed);
    # with this spectrum it is strictly smaller.
    assert out_bf16["gate_proj"][2] <= out_fp32["gate_proj"][2]
    assert out_bf16["gate_proj"][2] < out_fp32["gate_proj"][2]


# --------------------------------------------------------------------------- #
# Lever 2 — factor_layer end-to-end fixture (fused experts + stub cov accs).
# --------------------------------------------------------------------------- #
class _StubCovAcc:
    """Minimal stand-in for InputCovarianceAccumulator: a pre-populated
    ``.covariance`` dict + no-op load/unload (the cov is already resident, the
    test owns it, not disk). Mirrors the accumulator surface factor_layer
    touches (``.covariance`` / ``load_layer_from_disk`` → True / ``unload_layer``)."""

    def __init__(self, covariance):
        self.covariance = covariance

    def load_layer_from_disk(self, layer_idx, dir_path):
        return True

    def unload_layer(self, layer_idx):
        return None


def _make_fused_experts(n_experts, hidden, inter, originals, *, dtype=torch.float32):
    """A fused Qwen3-style experts module build_banks recognises: gate_up_proj
    [N, 2·inter, hidden] (gate=first half, up=second half) + down_proj
    [N, hidden, inter], filled from ``originals``."""
    import torch.nn as nn
    gate_up = torch.zeros(n_experts, 2 * inter, hidden, dtype=dtype)
    down = torch.zeros(n_experts, hidden, inter, dtype=dtype)
    for e in range(n_experts):
        gate_up[e, :inter] = originals[(0, e, "gate_proj")].to(dtype)
        gate_up[e, inter:] = originals[(0, e, "up_proj")].to(dtype)
        down[e] = originals[(0, e, "down_proj")].to(dtype)
    ex = nn.Module()
    ex.gate_up_proj = nn.Parameter(gate_up, requires_grad=False)
    ex.down_proj = nn.Parameter(down, requires_grad=False)
    ex.num_experts = n_experts
    return ex


def _factor_case(n_experts=6, hidden=8, inter=12, k=2, seed=0, *, cross_cov=True):
    """A tiny MoE layer + the factor_layer inputs (originals + A/B/C cov accs).

    Returns ``(build_ctx, n_experts, ranks_layer)`` where ``build_ctx(workers,
    worker_devices, per_expert)`` constructs a fresh PipelineContext driving a
    FRESH fused-experts MoELayerRef (factor_layer swaps in FactoredExperts, so
    each run needs its own model)."""
    import torch.nn as nn
    from moe_compress.pipeline.context import PipelineContext
    from moe_compress.utils.model_io import MoELayerRef

    g = torch.Generator().manual_seed(seed)
    originals = {}
    for e in range(n_experts):
        for name in ("gate_proj", "up_proj", "down_proj"):
            d_out, d_in = (inter, hidden) if name != "down_proj" else (hidden, inter)
            originals[(0, e, name)] = torch.randn(
                d_out, d_in, generator=g, dtype=torch.float32)

    A_cov, B_cov, C_cov = {}, {}, {}
    for e in range(n_experts):
        A_cov[(0, e, "gate_proj")] = _make_spd(hidden, 100 + e)
        A_cov[(0, e, "down_proj")] = _make_spd(inter, 200 + e)
        B_cov[(0, e, "gate_proj")] = _make_spd(hidden, 300 + e)
        B_cov[(0, e, "down_proj")] = _make_spd(inter, 400 + e)
        C_cov[(0, e, "gate_proj")] = _make_spd(hidden, 500 + e)
        C_cov[(0, e, "down_proj")] = _make_spd(inter, 600 + e)

    base_ranks = {(0, name): k for name in ("gate_proj", "up_proj", "down_proj")}

    def build_ctx(*, factor_workers=1, worker_devices=None, per_expert=None):
        ex = _make_fused_experts(n_experts, hidden, inter, originals)
        mlp = nn.Module()
        mlp.experts = ex
        ref = MoELayerRef(
            layer_idx=0, layer_module=nn.Identity(), mlp=mlp,
            router=nn.Identity(), experts_module=ex, shared_expert=None,
            layer_type="full_attention",
        )
        ctx = PipelineContext()
        ctx.set("layer_ref", ref)
        ctx.set("ranks", dict(base_ranks))
        ctx.set("B_acc", _StubCovAcc(B_cov))
        ctx.set("B_cov_dtype", torch.float32)
        ctx.set("rank_map", {})
        ctx.set("device", torch.device("cpu"))
        ctx.set("originals", originals)
        ctx.set("bcov_spill_dir", None)
        if per_expert is not None:
            ctx.set("per_expert_ranks", per_expert)
        if A_cov is not None:
            ctx.set("A_cov", A_cov)
        if cross_cov:
            ctx.set("C_acc", _StubCovAcc(C_cov))
            ctx.set("ccov_spill_dir", None)
        ctx.set("factor_workers", factor_workers)
        if worker_devices is not None:
            ctx.set("factor_worker_devices", worker_devices)
        return ctx, ref

    return build_ctx, n_experts, {n: k for n in ("gate_proj", "up_proj", "down_proj")}


def _run_factor(build_ctx, **kw):
    """Run factor_layer once; return (factored_experts, rank_map)."""
    from moe_compress.stage3.plugins.aa_svd_factor import AaSvdFactorPlugin
    ctx, ref = build_ctx(**kw)
    AaSvdFactorPlugin().factor_layer(ctx)
    return ref.experts_module, dict(ctx.get("rank_map"))


def test_factor_taskparallel_equivalence():
    """factor_layer: serial == W=2 (two CPU worker devices).

    Integer ranks (rank_map) EXACTLY equal; float factor tensors within
    rtol=1e-5/atol=1e-6. Mirror of test_eora_taskparallel_equivalence."""
    build_ctx, n, _ = _factor_case()
    fe_serial, rm_serial = _run_factor(build_ctx, factor_workers=1)
    fe_par, rm_par = _run_factor(
        build_ctx, factor_workers=2, worker_devices=["cpu", "cpu"])

    assert rm_serial == rm_par
    assert fe_serial.ranks == fe_par.ranks
    assert fe_serial.effective_ranks == fe_par.effective_ranks
    for name in ("gate_proj", "up_proj", "down_proj"):
        for proj in ("U", "V"):
            a = getattr(fe_serial, f"{name}_{proj}").data
            b = getattr(fe_par, f"{name}_{proj}").data
            assert a.shape == b.shape
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)


def test_factor_taskparallel_equivalence_per_expert_ranks():
    """Same equivalence with NON-UNIFORM per-expert ranks (the real Lever-2
    exerciser: multi-α drives differing k per expert → zero-pad-to-slot path)."""
    build_ctx, n, _ = _factor_case(n_experts=6, k=3)
    # Per-expert ranks: alternate 1/3 so the per-layer slot (max=3) forces
    # zero-padding on the rank-1 experts.
    per_expert = {}
    for e in range(n):
        ke = 1 if e % 2 == 0 else 3
        for name in ("gate_proj", "up_proj", "down_proj"):
            per_expert[(0, name, e)] = ke
    fe_s, rm_s = _run_factor(build_ctx, factor_workers=1, per_expert=per_expert)
    fe_p, rm_p = _run_factor(
        build_ctx, factor_workers=2, worker_devices=["cpu", "cpu"],
        per_expert=per_expert)
    assert rm_s == rm_p
    assert fe_s.ranks == fe_p.ranks and fe_s.effective_ranks == fe_p.effective_ranks
    for name in ("gate_proj", "up_proj", "down_proj"):
        for proj in ("U", "V"):
            torch.testing.assert_close(
                getattr(fe_s, f"{name}_{proj}").data,
                getattr(fe_p, f"{name}_{proj}").data, rtol=1e-5, atol=1e-6)


# --------------------------------------------------------------------------- #
# Lever 2 — T2.3: determinism + C1 multi-band (really-threaded) proof.
# --------------------------------------------------------------------------- #
def test_factor_gather_order_deterministic():
    """The W=3 gather is deterministic — run twice, identical bytes."""
    build_ctx, n, _ = _factor_case(n_experts=9)
    fe1, rm1 = _run_factor(
        build_ctx, factor_workers=3, worker_devices=["cpu", "cpu", "cpu"])
    fe2, rm2 = _run_factor(
        build_ctx, factor_workers=3, worker_devices=["cpu", "cpu", "cpu"])
    assert rm1 == rm2 and fe1.ranks == fe2.ranks
    for name in ("gate_proj", "up_proj", "down_proj"):
        for proj in ("U", "V"):
            assert torch.equal(
                getattr(fe1, f"{name}_{proj}").data,
                getattr(fe2, f"{name}_{proj}").data)


def test_factor_concurrent_exact_equals_serial_cpu():
    """W=3 concurrent path is BIT-identical to serial on CPU AND really threaded.

    9 experts → 3 full bands of 3 under W=3. torch.equal (CPU eigh/svd is
    deterministic): the engine only changes WHEN pure tiles run, not WHAT they
    compute, and assembles ascending-e on the main thread. The
    _LAST_BAND_COUNT==3 / _LAST_RAN_THREADED guard is C1: ["cpu","cpu","cpu"]
    MUST yield 3 real bands (banded by ordinal w, not device object)."""
    from moe_compress.stage4.plugins import eora_compensation as _eng
    build_ctx, n, _ = _factor_case(n_experts=9)
    fe_serial, rm_s = _run_factor(build_ctx, factor_workers=1)
    fe_par, rm_p = _run_factor(
        build_ctx, factor_workers=3, worker_devices=["cpu", "cpu", "cpu"])
    assert _eng._LAST_BAND_COUNT == 3, (
        f"expected 3 bands (1 thread/worker), got {_eng._LAST_BAND_COUNT} — "
        "band-by-device-object bug would collapse CPU workers to 1 band")
    assert _eng._LAST_RAN_THREADED is True
    assert rm_s == rm_p and fe_serial.ranks == fe_par.ranks
    assert fe_serial.effective_ranks == fe_par.effective_ranks
    for name in ("gate_proj", "up_proj", "down_proj"):
        for proj in ("U", "V"):
            assert torch.equal(
                getattr(fe_serial, f"{name}_{proj}").data,
                getattr(fe_par, f"{name}_{proj}").data), \
                f"{name}_{proj} bytes differ serial vs concurrent"


def test_factor_single_worker_is_one_band_not_threaded():
    """factor_workers=1 ⇒ exactly one band, never the threaded branch (M2)."""
    from moe_compress.stage4.plugins import eora_compensation as _eng
    build_ctx, n, _ = _factor_case(n_experts=6)
    _run_factor(build_ctx, factor_workers=1)
    assert _eng._LAST_BAND_COUNT == 1
    assert _eng._LAST_RAN_THREADED is False


# --------------------------------------------------------------------------- #
# Lever 2 — T2.4: worker-count resolution clamps correctly.
# --------------------------------------------------------------------------- #
def test_factor_workers_resolution():
    """``_resolve_factor_workers`` clamps to min(requested, device_count), floor 1.

    Clone of ``_resolve_eora_workers`` keyed ``factor_workers``: multi_gpu
    absent ⇒ 1; <=1 ⇒ 1 regardless of GPUs; requested-8 clamps to
    min(8, device_count()) (CI: 0 GPUs ⇒ 1)."""
    from moe_compress.stage3.orchestrator import _resolve_factor_workers

    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    assert _resolve_factor_workers({}) == 1
    assert _resolve_factor_workers({"multi_gpu": {"factor_workers": 1}}) == 1
    assert _resolve_factor_workers({"multi_gpu": {"factor_workers": 0}}) == 1

    requested = 8
    got = _resolve_factor_workers({"multi_gpu": {"factor_workers": requested}})
    if n_gpu < 2:
        assert got == 1
    else:
        assert got == min(requested, n_gpu)
        assert got >= 1
