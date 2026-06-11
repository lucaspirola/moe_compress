"""Stage 4 N-GPU (task-parallel EoRA) — equivalence + worker-resolution tests.

Covers the multi-GPU Stage-4 lever 1 (``tasks/PLAN_MULTIGPU_STAGE4.md`` §7):

* ``test_solve_expert_tile_pure`` — the extracted per-expert solve helper is a
  pure device-relocation: the same inputs on two (CPU) target "devices" give
  identical ``(Uc, Vc, take_eff)``.
* ``test_eora_workers_resolution`` — ``_resolve_eora_workers`` clamps to
  ``min(requested, device_count())`` with a floor of 1, returns 1 when
  ``multi_gpu`` is absent or there is <2 GPUs (CI has 0 — drives the real
  resolver, no monkeypatch).
* ``test_eora_taskparallel_equivalence`` (the load-bearing one) — runs the real
  ``compensate_layer`` per-expert gather on a tiny ``FactoredExperts`` fixture
  TWICE: once serial (``eora_workers=1``), once with ``eora_workers=2`` whose
  two "worker devices" are both CPU (the CI stand-in for a 2nd GPU, injected via
  the ``eora_worker_devices`` ctx seam). Asserts integer ranks EXACTLY equal and
  float factor tensors within ``rtol=1e-5, atol=1e-6`` (CPU-vs-CPU is
  bit-identical in practice; the tolerance is stated for cross-arch
  generalization — never ``atol=0`` across nominal backends).
* ``test_eora_gather_order_deterministic`` — the W=2 path run twice yields
  identical output (fixed contiguous expert→worker bands + ascending-e gather).

All run WITHOUT a real multi-GPU box: the second "device" is CPU via the
``eora_worker_devices`` ctx seam (no monkeypatch — project rule).
"""
from __future__ import annotations

import pytest
import torch

from moe_compress.pipeline.context import PipelineContext
from moe_compress.stage4.orchestrator import _resolve_eora_workers
from moe_compress.stage4.plugins.eora_compensation import (
    EoraCompensationPlugin,
    _compute_eora_factors,
    _eigh_spectrum,
    _solve_expert_tile,
)
from moe_compress.utils.model_io import MoELayerRef, FactoredExperts


# --------------------------------------------------------------------------- #
# Fixture builders
# --------------------------------------------------------------------------- #
def _make_cov(d_in: int, seed: int) -> torch.Tensor:
    """A well-conditioned SPD covariance [d_in, d_in] (fp32, CPU)."""
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(4 * d_in, d_in, generator=g, dtype=torch.float32)
    return (X.T @ X) / X.shape[0] + torch.eye(d_in)


def _build_case(n_experts=6, hidden=8, inter=12, rank=2, seed=0):
    """A tiny factored layer + the EoRA inputs (originals + A_cov) it consumes.

    Returns ``(make_fe, ref_factory, originals, A_cov, config)`` where
    ``make_fe`` builds a FRESH FactoredExperts (compensate_layer mutates it
    in place via widen_rank, so each run needs its own) and ``ref_factory(fe)``
    wraps it in a MoELayerRef at layer_idx=0.
    """
    g = torch.Generator().manual_seed(seed)
    ranks = {"gate_proj": rank, "up_proj": rank, "down_proj": rank}

    # Stable random U/V seeds so every fresh FactoredExperts is identical.
    init = {}
    for name in ("gate_proj", "up_proj", "down_proj"):
        d_out, d_in = (inter, hidden) if name != "down_proj" else (hidden, inter)
        init[f"{name}_U"] = torch.randn(n_experts, d_out, rank, generator=g, dtype=torch.float32)
        init[f"{name}_V"] = torch.randn(n_experts, rank, d_in, generator=g, dtype=torch.float32)

    def make_fe():
        fe = FactoredExperts(
            num_experts=n_experts, hidden_dim=hidden, intermediate_dim=inter,
            ranks=dict(ranks), dtype=torch.float32, device="cpu",
        )
        for name in ("gate_proj", "up_proj", "down_proj"):
            getattr(fe, f"{name}_U").data.copy_(init[f"{name}_U"])
            getattr(fe, f"{name}_V").data.copy_(init[f"{name}_V"])
        return fe

    def ref_factory(fe):
        return MoELayerRef(
            layer_idx=0, layer_module=torch.nn.Identity(), mlp=torch.nn.Identity(),
            router=torch.nn.Identity(), experts_module=fe, shared_expert=None,
            layer_type="full_attention",
        )

    # originals: a full-rank target weight per (layer, expert, matrix).
    originals = {}
    for e in range(n_experts):
        for name in ("gate_proj", "up_proj", "down_proj"):
            d_out, d_in = (inter, hidden) if name != "down_proj" else (hidden, inter)
            originals[(0, e, name)] = torch.randn(
                d_out, d_in, generator=g, dtype=torch.float32
            )

    # A_cov: gate_proj covariance per expert (up_proj reuses gate's; down_proj
    # gets its own under (0,e,"down_proj")).
    A_cov = {}
    for e in range(n_experts):
        A_cov[(0, e, "gate_proj")] = _make_cov(hidden, seed=100 + e)
        A_cov[(0, e, "down_proj")] = _make_cov(inter, seed=200 + e)

    config = {"stage4_eora": {"compensation_budget_pct": 0.5, "eigenspace_rank_cap": 4}}
    return make_fe, ref_factory, originals, A_cov, config


def _run_compensate(make_fe, ref_factory, originals, A_cov, config,
                    *, eora_workers, worker_devices=None):
    """Run compensate_layer once; return (fe, rank_map, compensated_params)."""
    fe = make_fe()
    ref = ref_factory(fe)
    ctx = PipelineContext()
    ctx.set("layer_ref", ref)
    ctx.set("originals", originals)
    ctx.set("A_cov", A_cov)
    ctx.set("a_storage_dtype", torch.float32)
    ctx.set("config", config)
    # stage3_ranks snapshot — the pre-widen ranks (double-widen guard).
    ctx.set("stage3_ranks", {0: dict(fe.ranks)})
    ctx.set("rank_map", {})
    ctx.set("compensated_params", 0)
    ctx.set("eora_workers", eora_workers)
    if worker_devices is not None:
        ctx.set("eora_worker_devices", worker_devices)
    EoraCompensationPlugin().compensate_layer(ctx)
    return fe, dict(ctx.get("rank_map")), int(ctx.get("compensated_params"))


# --------------------------------------------------------------------------- #
# Unit: the extracted helper is a pure device-relocation
# --------------------------------------------------------------------------- #
def test_solve_expert_tile_pure():
    """``_solve_expert_tile`` on two CPU 'devices' gives identical tiles."""
    d_out, d_in, r = 12, 8, 2
    g = torch.Generator().manual_seed(7)
    W_orig = torch.randn(d_out, d_in, generator=g, dtype=torch.float32)
    U_e = torch.randn(d_out, r, generator=g, dtype=torch.float32)
    V_e = torch.randn(r, d_in, generator=g, dtype=torch.float32)
    A = _make_cov(d_in, seed=3)

    out0 = _solve_expert_tile(
        "gate_proj", 0, 0, W_orig, U_e, V_e, A, d_in, r,
        torch.device("cpu"), torch.float32, gate_spectrum=None,
    )
    out1 = _solve_expert_tile(
        "gate_proj", 0, 0, W_orig, U_e, V_e, A, d_in, r,
        torch.device("cpu"), torch.float32, gate_spectrum=None,
    )
    Uc0, Vc0, te0, *_ = out0
    Uc1, Vc1, te1, *_ = out1
    assert te0 == te1
    assert torch.equal(Uc0, Uc1)
    assert torch.equal(Vc0, Vc1)
    # The gate-pass returns a spectrum to retain; up_proj returns None.
    assert out0[5] is not None
    up = _solve_expert_tile(
        "up_proj", 0, 0, W_orig, U_e, V_e, A, d_in, r,
        torch.device("cpu"), torch.float32, gate_spectrum=out0[5],
    )
    assert up[5] is None


# --------------------------------------------------------------------------- #
# Unit: worker resolution clamps correctly
# --------------------------------------------------------------------------- #
def test_eora_workers_resolution():
    """``_resolve_eora_workers`` clamps to min(requested, device_count), floor 1."""
    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0

    # multi_gpu absent ⇒ 1.
    assert _resolve_eora_workers({}) == 1
    # eora_workers <= 1 ⇒ 1 regardless of GPUs.
    assert _resolve_eora_workers({"multi_gpu": {"eora_workers": 1}}) == 1
    assert _resolve_eora_workers({"multi_gpu": {"eora_workers": 0}}) == 1

    requested = 8
    got = _resolve_eora_workers({"multi_gpu": {"eora_workers": requested}})
    if n_gpu < 2:
        # graceful 1-GPU / CPU degrade (CI path).
        assert got == 1
    else:
        assert got == min(requested, n_gpu)
        assert got >= 1


# --------------------------------------------------------------------------- #
# Equivalence: serial vs W=2 (CPU stand-in for the 2nd device)
# --------------------------------------------------------------------------- #
def test_eora_taskparallel_equivalence():
    """compensate_layer: serial == W=2 (two CPU worker devices)."""
    case = _build_case()
    fe_serial, rm_serial, cp_serial = _run_compensate(*case, eora_workers=1)
    fe_par, rm_par, cp_par = _run_compensate(
        *case, eora_workers=2, worker_devices=["cpu", "cpu"],
    )

    # Integer ranks / param counts — EXACTLY equal (same backend → same n_keep).
    assert fe_serial.ranks == fe_par.ranks
    assert fe_serial.effective_ranks == fe_par.effective_ranks
    assert rm_serial == rm_par
    assert cp_serial == cp_par

    # Float factor tensors — within tolerance (never atol=0 across backends).
    for name in ("gate_proj", "up_proj", "down_proj"):
        for proj in ("U", "V"):
            a = getattr(fe_serial, f"{name}_{proj}").data
            b = getattr(fe_par, f"{name}_{proj}").data
            assert a.shape == b.shape
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)


def test_eora_gather_order_deterministic():
    """The W=2 gather is deterministic — run twice, identical output."""
    case = _build_case()
    fe1, rm1, cp1 = _run_compensate(
        *case, eora_workers=2, worker_devices=["cpu", "cpu"],
    )
    fe2, rm2, cp2 = _run_compensate(
        *case, eora_workers=2, worker_devices=["cpu", "cpu"],
    )
    assert rm1 == rm2 and cp1 == cp2
    assert fe1.ranks == fe2.ranks
    for name in ("gate_proj", "up_proj", "down_proj"):
        for proj in ("U", "V"):
            a = getattr(fe1, f"{name}_{proj}").data
            b = getattr(fe2, f"{name}_{proj}").data
            assert torch.equal(a, b)


# --------------------------------------------------------------------------- #
# HIGH regression: gate-pass spectrum on a DIFFERENT device than up-pass delta
# --------------------------------------------------------------------------- #
def _drop_up_originals(originals, experts):
    """Return a copy of ``originals`` with the up_proj entry removed for the
    given experts — makes the gate_proj and up_proj eligibility SETS differ in
    length, which (under W>1) bands a shared expert ``e`` to a DIFFERENT worker
    device on the gate vs up pass. That is the trigger for the cross-device
    gate-spectrum bug."""
    return {k: v for k, v in originals.items()
            if not (k[2] == "up_proj" and k[1] in experts)}


def test_eora_differing_eligibility_bands_diverge():
    """With gate-eligible != up-eligible, compensate_layer still completes and
    matches serial (CPU). On real multi-GPU the per-matrix bands would assign
    expert e to different devices across the gate/up passes — this is the case
    whose gate-spectrum reuse must NOT cross a device boundary."""
    make_fe, ref_factory, originals, A_cov, config = _build_case(n_experts=6)
    # Drop up_proj originals for experts {1,4}: gate has 6 eligible, up has 4,
    # so contiguous banding (W=3) maps e.g. expert 5 to band 2 in gate but a
    # different band in up — divergent device_of per matrix.
    originals_skew = _drop_up_originals(originals, {1, 4})

    serial = _run_compensate(
        make_fe, ref_factory, originals_skew, A_cov, config, eora_workers=1,
    )
    par = _run_compensate(
        make_fe, ref_factory, originals_skew, A_cov, config,
        eora_workers=3, worker_devices=["cpu", "cpu", "cpu"],
    )
    fe_s, rm_s, cp_s = serial
    fe_p, rm_p, cp_p = par
    assert fe_s.ranks == fe_p.ranks
    assert rm_s == rm_p and cp_s == cp_p
    for name in ("gate_proj", "up_proj", "down_proj"):
        for proj in ("U", "V"):
            torch.testing.assert_close(
                getattr(fe_s, f"{name}_{proj}").data,
                getattr(fe_p, f"{name}_{proj}").data,
                rtol=1e-5, atol=1e-6,
            )


def test_compute_eora_factors_relocates_cross_device_spectrum():
    """_compute_eora_factors must relocate a SUPPLIED spectrum to delta's device
    before the projection matmul. Reproduces the gate(device A)→up(device B)
    reuse path on two real CUDA devices; would RuntimeError before the fix.

    Skipped without >=2 CUDA devices (CI has 0). The CPU path is covered by
    test_eora_differing_eligibility_bands_diverge above."""
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("needs >=2 CUDA devices")
    d_out, d_in, r = 12, 8, 2
    g = torch.Generator().manual_seed(11)
    A = _make_cov(d_in, seed=5)
    # Gate pass: spectrum computed on cuda:0.
    spectrum = _eigh_spectrum(A, d_in, torch.device("cuda:0"), torch.float32)
    assert spectrum is not None and spectrum[0].device.type == "cuda"
    # Up pass: delta on cuda:1; supplying the cuda:0 spectrum must NOT crash.
    delta = torch.randn(d_out, d_in, generator=g, dtype=torch.float32).to("cuda:1")
    Uc, Vc, take_eff = _compute_eora_factors(
        delta, A, r, torch.device("cuda:1"),
        storage_dtype=torch.float32, spectrum=spectrum,
    )
    assert Uc.device == torch.device("cuda:1")
    assert Vc.device == torch.device("cuda:1")
