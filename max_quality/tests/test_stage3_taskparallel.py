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
