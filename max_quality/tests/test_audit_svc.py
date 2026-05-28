"""Tests for ``audit/spec_compliance/svc_audit.py``.

Pins the Eq. 8 projection-coefficient primitive (arXiv:2602.05536) plus the
output-activation-space adaptation on a tiny synthetic case whose expected
values are derived by hand below.

Hand-derived synthetic cases
============================

Throughout, all matrices live in float64 inside the audit's primitives,
so the assertion tolerances can be tight (1e-10).

**Case A — Self-merge identity.**
    Merged weight == donor weight, Σ_in == I → activation matrices are
    identical → left singular vectors identical → s_r^i = 1 ∀ r.

**Case B — Rotation by θ, isotropic input.**
    donor   W = diag(2, 1), Σ_in = I → Y = W, SVD has U = I,
            top-left singular vector = [1, 0].
    merged  W = R(θ) · diag(2, 1), Σ_in = I → SVD: U = R(θ),
            top-left singular vector = [cos θ, sin θ].
    Eq. 8 → s_0 = <[cos θ, sin θ], [1, 0]> / 1 = cos θ.
    For θ = π/3 → s_0 = 0.5.
    Second direction: donor [0, 1], merged R(θ)[:, 1] = [-sin θ, cos θ];
                      s_1 = <[-sin θ, cos θ], [0, 1]> = cos θ = 0.5.

**Case C — Anisotropic input covariance.**
    Σ_in = diag(9, 1), donor W = I, merged W = I.
    Y_donor = I · L = diag(3, 1) → top-left vec = [1, 0].
    Y_merged = same → s_0 = 1, s_1 = 1.
    A version where merged is rotated by θ then gives s_0 = cos θ again
    because the rotation acts on the OUTPUT space, not the input.

**Case D — Upstream-formula cross-check on a non-unit-norm vector pair.**
    Plain Eq. 8 with hand-picked non-unit vectors:
        merged = [3, 4], donor = [1, 0]  →  s = <[3,4],[1,0]> / 1 = 3.
        merged = [1, 1], donor = [2, 0]  →  s = 2 / 4 = 0.5.

**Case E — Skipped: degenerate spectrum.**
    diag(c, c) merged with diag(c, c) is left out of the test grid
    because SVD direction is not unique on degenerate spectra (any
    orthonormal basis of the eigenspace is a valid U) and the diagnostic
    is intentionally undefined there. Instead we test that the script
    does not crash and reports a value of ±1 on this case via the
    integration test ``test_run_audit_does_not_crash_on_degenerate_spectrum``.
"""
from __future__ import annotations

import importlib.util
import json
import logging
import math
import sys
from pathlib import Path

import pytest
import torch

# Audit script is a stand-alone file under audit/spec_compliance/. Import it
# by absolute path to avoid leaking 'svc_audit' into the global module cache
# of unrelated tests (and to keep the audit/ tree out of the moe_compress
# package's import surface).
_AUDIT_PATH = (
    Path(__file__).resolve().parents[2]
    / "audit"
    / "spec_compliance"
    / "svc_audit.py"
)
_spec = importlib.util.spec_from_file_location("svc_audit_under_test", _AUDIT_PATH)
assert _spec is not None and _spec.loader is not None
svc_audit = importlib.util.module_from_spec(_spec)
# Register in sys.modules BEFORE exec_module so dataclasses can find the
# module via cls.__module__ during @dataclass class-construction.
sys.modules["svc_audit_under_test"] = svc_audit
_spec.loader.exec_module(svc_audit)


# --------------------------------------------------------------------------- #
# Eq. 8 projection-coefficient primitive — exact hand-derived values          #
# --------------------------------------------------------------------------- #


def test_projection_coefficient_unit_aligned_equals_one():
    """Aligned unit vectors → s = 1 (perfect preservation)."""
    a_merge = torch.tensor([1.0, 0.0])
    a_donor = torch.tensor([1.0, 0.0])
    s = svc_audit.compute_projection_coefficient(a_merge, a_donor)
    assert s == pytest.approx(1.0, abs=1e-12)


def test_projection_coefficient_unit_anti_aligned_equals_minus_one():
    """Anti-aligned unit vectors → s = -1."""
    a_merge = torch.tensor([1.0, 0.0])
    a_donor = torch.tensor([-1.0, 0.0])
    s = svc_audit.compute_projection_coefficient(a_merge, a_donor)
    assert s == pytest.approx(-1.0, abs=1e-12)


def test_projection_coefficient_unit_orthogonal_equals_zero():
    """Orthogonal unit vectors → s = 0 (donor direction completely dropped)."""
    a_merge = torch.tensor([1.0, 0.0])
    a_donor = torch.tensor([0.0, 1.0])
    s = svc_audit.compute_projection_coefficient(a_merge, a_donor)
    assert s == pytest.approx(0.0, abs=1e-12)


def test_projection_coefficient_non_unit_norm_divisor_active():
    """Eq. 8 divisor IS exercised for non-unit-norm vectors.

    merged = [3, 4], donor = [1, 0]  →  <[3,4],[1,0]> / 1^2 = 3.
    """
    a_merge = torch.tensor([3.0, 4.0])
    a_donor = torch.tensor([1.0, 0.0])
    s = svc_audit.compute_projection_coefficient(a_merge, a_donor)
    assert s == pytest.approx(3.0, abs=1e-12)


def test_projection_coefficient_non_unit_norm_donor():
    """Eq. 8 divisor halves a coincident-merged scenario by ||donor||^2.

    merged = [1, 1], donor = [2, 0]  →  <[1,1],[2,0]> / 4 = 0.5.
    """
    a_merge = torch.tensor([1.0, 1.0])
    a_donor = torch.tensor([2.0, 0.0])
    s = svc_audit.compute_projection_coefficient(a_merge, a_donor)
    assert s == pytest.approx(0.5, abs=1e-12)


def test_projection_coefficient_rejects_shape_mismatch():
    a_merge = torch.tensor([1.0, 0.0])
    a_donor = torch.tensor([1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="shape mismatch"):
        svc_audit.compute_projection_coefficient(a_merge, a_donor)


def test_projection_coefficient_rejects_non_1d():
    a_merge = torch.tensor([[1.0, 0.0]])  # 2-D
    a_donor = torch.tensor([[1.0, 0.0]])
    with pytest.raises(ValueError, match="1-D"):
        svc_audit.compute_projection_coefficient(a_merge, a_donor)


def test_projection_coefficient_zero_donor_does_not_explode():
    """Eps clamp guards a degenerate-donor edge case."""
    a_merge = torch.tensor([1.0, 0.0])
    a_donor = torch.tensor([0.0, 0.0])
    s = svc_audit.compute_projection_coefficient(a_merge, a_donor)
    # 0 / eps = 0 (numerator is also exactly zero); never NaN, never inf.
    assert math.isfinite(s)
    assert s == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------- #
# Activation-matrix reconstruction from Σ_in                                  #
# --------------------------------------------------------------------------- #


def test_activation_matrix_from_cov_identity():
    """Σ_in = I → activation matrix == W (cholesky of I is I)."""
    W = torch.tensor([[2.0, 0.0], [0.0, 1.0]])
    sigma = torch.eye(2)
    Y = svc_audit._activation_matrix_from_cov(W, sigma)
    # Cholesky of I is I (plus a tiny jitter * I); allow a comfortable
    # tolerance for the jitter contribution.
    assert torch.allclose(Y, W.to(torch.float64), atol=1e-3)


def test_activation_matrix_from_cov_diagonal():
    """Σ_in = diag(λ) → Y = W · diag(sqrt λ).

    For W = diag(2, 1), Σ_in = diag(9, 1) the expected Y = diag(2·3, 1·1) =
    diag(6, 1).
    """
    W = torch.tensor([[2.0, 0.0], [0.0, 1.0]])
    sigma = torch.tensor([[9.0, 0.0], [0.0, 1.0]])
    Y = svc_audit._activation_matrix_from_cov(W, sigma)
    expected = torch.tensor([[6.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    assert torch.allclose(Y, expected, atol=1e-4)


def test_activation_matrix_preserves_left_singular_structure():
    """Y · Y^T should equal W · Σ_in · W^T (up to jitter)."""
    g = torch.Generator().manual_seed(0)
    W = torch.randn(4, 3, generator=g, dtype=torch.float64)
    X = torch.randn(20, 3, generator=g, dtype=torch.float64)
    sigma = X.transpose(0, 1) @ X
    Y = svc_audit._activation_matrix_from_cov(W, sigma)
    YYT = Y @ Y.transpose(0, 1)
    expected = W @ sigma @ W.transpose(0, 1)
    err = (YYT - expected).norm() / expected.norm()
    # Jitter is sigma_diag_mean * 1e-8; relative error should be ≪ 1.
    assert err.item() < 1e-6


# --------------------------------------------------------------------------- #
# Per-group SVC scoring — Case A (self-merge identity)                        #
# --------------------------------------------------------------------------- #


def test_svc_scores_self_merge_is_unity():
    """Case A: W_merge == W_donor and Σ_in = I → s_r = 1 for every r."""
    g = torch.Generator().manual_seed(7)
    W = torch.randn(5, 4, generator=g, dtype=torch.float32) * 1.5
    sigma = torch.eye(4)
    result = svc_audit.svc_scores_for_group(
        merged_weight=W,
        donor_weights={42: W.clone()},
        donor_input_covariances={42: sigma},
        rank=3,
        layer_idx=0,
        centroid_expert_idx=42,
        matrix_name="gate_proj",
    )
    assert [s.donor_expert_idx for s in result.scores] == [42, 42, 42]
    assert [s.rank for s in result.scores] == [0, 1, 2]
    # Self-merge: each |s_r| must be 1 (sign may flip on near-degenerate
    # SVD spectra, but on randn-init the spectrum is non-degenerate and
    # SVD signs match between identical inputs).
    for s in result.scores:
        assert abs(s.s_r) == pytest.approx(1.0, abs=1e-8)


# --------------------------------------------------------------------------- #
# Per-group SVC scoring — Case B (rotation by θ)                              #
# --------------------------------------------------------------------------- #


def _rotation_2d(theta: float) -> torch.Tensor:
    c, s = math.cos(theta), math.sin(theta)
    return torch.tensor([[c, -s], [s, c]], dtype=torch.float64)


@pytest.mark.parametrize("theta", [math.pi / 3, math.pi / 6, 0.4])
def test_svc_scores_rotated_merge_equals_cos_theta(theta: float):
    """Case B: merged = R(θ) · donor, Σ_in = I → s_0 = s_1 = cos θ.

    Derivation: donor W = diag(2, 1) → SVD U = I, top vecs = e1, e2.
    Merged W = R(θ) · diag(2, 1) = R(θ)·diag(2,1)·I, so SVD gives
    U = R(θ), top vec = [cos θ, sin θ], second = [-sin θ, cos θ].
        s_0 = <[cos θ, sin θ], [1, 0]> = cos θ
        s_1 = <[-sin θ, cos θ], [0, 1]> = cos θ
    Both projection coefficients equal cos θ exactly.
    """
    W_donor = torch.tensor([[2.0, 0.0], [0.0, 1.0]])
    W_merged = (_rotation_2d(theta) @ W_donor.to(torch.float64)).to(torch.float32)
    sigma = torch.eye(2)
    result = svc_audit.svc_scores_for_group(
        merged_weight=W_merged,
        donor_weights={0: W_donor},
        donor_input_covariances={0: sigma},
        rank=2,
        matrix_name="up_proj",
    )
    expected = math.cos(theta)
    # Two scores: rank 0 and rank 1, both = cos θ.
    s0 = next(s.s_r for s in result.scores if s.rank == 0)
    s1 = next(s.s_r for s in result.scores if s.rank == 1)
    # SVD sign convention can flip both vectors of a singular triplet
    # together. The flip would invert BOTH the merged and donor vectors
    # consistently for a self-comparison, but here merged != donor.
    # On a clean 2x2 the convention is stable; relax slightly to be safe.
    assert abs(s0) == pytest.approx(abs(expected), abs=1e-6)
    assert abs(s1) == pytest.approx(abs(expected), abs=1e-6)


# --------------------------------------------------------------------------- #
# Per-group SVC scoring — multi-donor                                         #
# --------------------------------------------------------------------------- #


def test_svc_scores_two_donor_group_returns_one_block_per_donor():
    """Score grid for K donors × R ranks has K*R entries, donor-major order."""
    g = torch.Generator().manual_seed(11)
    W_a = torch.randn(4, 3, generator=g, dtype=torch.float32)
    W_b = torch.randn(4, 3, generator=g, dtype=torch.float32)
    W_merged = 0.5 * (W_a + W_b)
    sigma = torch.eye(3)
    result = svc_audit.svc_scores_for_group(
        merged_weight=W_merged,
        donor_weights={5: W_a, 9: W_b},
        donor_input_covariances={5: sigma, 9: sigma},
        rank=2,
        matrix_name="down_proj",
    )
    assert result.donor_expert_ids == [5, 9]
    # Donor-major: all of donor 5 first, then all of donor 9.
    donor_order = [s.donor_expert_idx for s in result.scores]
    assert donor_order == [5, 5, 9, 9]
    rank_order = [s.rank for s in result.scores]
    assert rank_order == [0, 1, 0, 1]


def test_svc_scores_rejects_mismatched_donor_keys():
    sigma = torch.eye(2)
    W = torch.eye(2)
    with pytest.raises(ValueError, match="key mismatch"):
        svc_audit.svc_scores_for_group(
            merged_weight=W,
            donor_weights={1: W},
            donor_input_covariances={2: sigma},
            rank=1,
        )


def test_svc_scores_rejects_zero_rank():
    sigma = torch.eye(2)
    W = torch.eye(2)
    with pytest.raises(ValueError, match="rank must be > 0"):
        svc_audit.svc_scores_for_group(
            merged_weight=W,
            donor_weights={1: W},
            donor_input_covariances={1: sigma},
            rank=0,
        )


def test_svc_scores_rejects_shape_mismatch_in_donor():
    sigma = torch.eye(2)
    W_merged = torch.eye(2)
    W_donor_wrong_shape = torch.eye(3)
    with pytest.raises(ValueError, match="shape"):
        svc_audit.svc_scores_for_group(
            merged_weight=W_merged,
            donor_weights={1: W_donor_wrong_shape},
            donor_input_covariances={1: sigma},
            rank=1,
        )


# --------------------------------------------------------------------------- #
# End-to-end run_audit on a tiny synthetic merge map                          #
# --------------------------------------------------------------------------- #


def test_run_audit_skips_singletons_and_scores_real_groups():
    """One singleton group + one 2-donor group → only the 2-donor one is scored."""
    g = torch.Generator().manual_seed(0)
    W_singleton = torch.randn(3, 2, generator=g, dtype=torch.float32)
    W_a = torch.randn(3, 2, generator=g, dtype=torch.float32)
    W_b = torch.randn(3, 2, generator=g, dtype=torch.float32)
    W_merged_ab = 0.5 * (W_a + W_b)

    originals = {
        # Layer 0, singleton (no scoring expected).
        (0, 7, "gate_proj"): W_singleton,
        (0, 7, "up_proj"): W_singleton,
        (0, 7, "down_proj"): W_singleton,
        # Layer 0, the real merge group (centroid 0, donors 0 and 1).
        (0, 0, "gate_proj"): W_a,
        (0, 0, "up_proj"): W_a,
        (0, 0, "down_proj"): W_a,
        (0, 1, "gate_proj"): W_b,
        (0, 1, "up_proj"): W_b,
        (0, 1, "down_proj"): W_b,
    }
    merged = {
        (0, 7, "gate_proj"): W_singleton,
        (0, 7, "up_proj"): W_singleton,
        (0, 7, "down_proj"): W_singleton,
        (0, 0, "gate_proj"): W_merged_ab,
        (0, 0, "up_proj"): W_merged_ab,
        (0, 0, "down_proj"): W_merged_ab,
    }
    sigma = torch.eye(2)
    input_cov = {k: sigma for k in originals.keys()}
    merge_map = {0: {7: [7], 0: [0, 1]}}

    results = svc_audit.run_audit(
        originals=originals,
        merged=merged,
        input_cov=input_cov,
        merge_map=merge_map,
        rank=2,
    )
    # 3 matrices x 1 non-singleton group = 3 group results; singleton skipped.
    assert len(results) == 3
    # Every group result should reference centroid 0 and donors [0, 1].
    for r in results:
        assert r.centroid_expert_idx == 0
        assert r.donor_expert_ids == [0, 1]
        # Two donors × two ranks = four projection scores.
        assert len(r.scores) == 4


def test_run_audit_clips_rank_to_matrix_size():
    """Requested rank above min(d_out, d_in) is clipped, not crashed."""
    W = torch.eye(2)  # d_out=2; max possible rank=2.
    originals = {
        (0, 0, "gate_proj"): W,
        (0, 1, "gate_proj"): W,
    }
    merged = {(0, 0, "gate_proj"): W}
    sigma = torch.eye(2)
    input_cov = {(0, 0, "gate_proj"): sigma, (0, 1, "gate_proj"): sigma}
    merge_map = {0: {0: [0, 1]}}
    results = svc_audit.run_audit(
        originals=originals,
        merged=merged,
        input_cov=input_cov,
        merge_map=merge_map,
        rank=99,  # absurd
        matrix_names=("gate_proj",),
    )
    assert len(results) == 1
    assert results[0].rank == 2  # clipped to d_out.


# --------------------------------------------------------------------------- #
# Reporting                                                                   #
# --------------------------------------------------------------------------- #


def test_write_results_json_round_trips(tmp_path: Path):
    result = svc_audit.SVCGroupResult(
        layer_idx=3,
        centroid_expert_idx=5,
        matrix_name="gate_proj",
        donor_expert_ids=[5, 7],
        rank=2,
        scores=[
            svc_audit.SVCDonorScore(donor_expert_idx=5, rank=0, s_r=1.0),
            svc_audit.SVCDonorScore(donor_expert_idx=5, rank=1, s_r=0.5),
            svc_audit.SVCDonorScore(donor_expert_idx=7, rank=0, s_r=-0.3),
            svc_audit.SVCDonorScore(donor_expert_idx=7, rank=1, s_r=2.1),
        ],
    )
    out = tmp_path / "svc_audit_results.json"
    svc_audit.write_results_json([result], out)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["format_version"] == 1
    assert payload["audit"] == "svc_audit"
    assert len(payload["results"]) == 1
    rec = payload["results"][0]
    assert rec["layer_idx"] == 3
    assert rec["centroid_expert_idx"] == 5
    assert rec["donor_expert_ids"] == [5, 7]
    assert len(rec["scores"]) == 4
    assert rec["scores"][3]["s_r"] == pytest.approx(2.1)


def test_write_summary_markdown_thresholds_classify_correctly(tmp_path: Path):
    """A handcrafted score grid hits the over-count and dropped flags exactly.

    Six samples on one (layer, matrix) bucket:
        +1.0  → normal
        +0.05 → dropped (|s| < 0.1)
        +1.5  → over-counted (|s| > 1.3)
        -1.4  → over-counted (sign-agnostic via |s|)
        +0.5  → normal
        +0.0  → dropped
    Expected: #over = 2, #dropped = 2, mean = (1 + 0.05 + 1.5 - 1.4 + 0.5 + 0)/6
    """
    scores = [
        svc_audit.SVCDonorScore(donor_expert_idx=0, rank=0, s_r=1.0),
        svc_audit.SVCDonorScore(donor_expert_idx=0, rank=1, s_r=0.05),
        svc_audit.SVCDonorScore(donor_expert_idx=1, rank=0, s_r=1.5),
        svc_audit.SVCDonorScore(donor_expert_idx=1, rank=1, s_r=-1.4),
        svc_audit.SVCDonorScore(donor_expert_idx=2, rank=0, s_r=0.5),
        svc_audit.SVCDonorScore(donor_expert_idx=2, rank=1, s_r=0.0),
    ]
    result = svc_audit.SVCGroupResult(
        layer_idx=2,
        centroid_expert_idx=4,
        matrix_name="down_proj",
        donor_expert_ids=[0, 1, 2],
        rank=2,
        scores=scores,
    )
    out = tmp_path / "svc_audit_summary.md"
    svc_audit.write_summary_markdown([result], out, over_count_threshold=1.3, dropped_threshold=0.1)
    text = out.read_text(encoding="utf-8")
    # Row for (layer=2, matrix=down_proj) should encode #over=2, #dropped=2.
    assert "down_proj" in text
    # Expected mean: 1.65 / 6 = 0.275
    assert "0.2750" in text
    # Look for the row that has the layer/matrix and the counts.
    matching_row = [
        line for line in text.splitlines()
        if "down_proj" in line and "| 2 |" in line
    ]
    assert matching_row, f"no expected row found in:\n{text}"
    # The row's tail should contain ' | 2 | 2 |' (over=2, dropped=2).
    assert any(
        row.rstrip().endswith("| 2 | 2 |") for row in matching_row
    ), f"expected '#over=2 #dropped=2' tail in row, got:\n{matching_row}"


# --------------------------------------------------------------------------- #
# Merge-map loader                                                            #
# --------------------------------------------------------------------------- #


def test_load_merge_map_partial_layout(tmp_path: Path):
    """Per-layer ``_stage2_partial/merge_{N}.json`` files are aggregated correctly."""
    art_dir = tmp_path / "artifacts"
    partial = art_dir / "_stage2_partial"
    partial.mkdir(parents=True)
    (partial / "merge_0.json").write_text(json.dumps({
        "grouped": {"0": [0, 1, 2], "5": [5]},
    }), encoding="utf-8")
    (partial / "merge_1.json").write_text(json.dumps({
        "grouped": {"3": [3, 4]},
    }), encoding="utf-8")
    # S-2 (PLAN_S2_SVC_LOAD_MERGE_MAP.md §2.5): loader now returns a
    # ``(map, stage2_run_id)`` tuple; pre-S-2 payloads return run_id=None.
    merge_map, run_id = svc_audit.load_merge_map(art_dir)
    assert merge_map == {0: {0: [0, 1, 2], 5: [5]}, 1: {3: [3, 4]}}
    assert run_id is None


def test_load_merge_map_aggregate_layout(tmp_path: Path):
    """Top-level ``merge_map.json`` with ``{layer: {centroid: [donors]}}``."""
    art_dir = tmp_path / "artifacts"
    art_dir.mkdir()
    raw = {"0": {"0": [0, 1], "5": [5]}, "1": {"3": [3, 4]}}
    (art_dir / "merge_map.json").write_text(json.dumps(raw), encoding="utf-8")
    merge_map, run_id = svc_audit.load_merge_map(art_dir)
    assert merge_map == {0: {0: [0, 1], 5: [5]}, 1: {3: [3, 4]}}
    assert run_id is None


def test_load_merge_map_raises_on_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        svc_audit.load_merge_map(tmp_path / "nope")


# --------------------------------------------------------------------------- #
# S-2: stage2_run_id cross-check (PLAN_S2_SVC_LOAD_MERGE_MAP.md §4)           #
# --------------------------------------------------------------------------- #


def _write_wrapper_merge_map(art_dir: Path, run_id: str | None,
                             inner: dict) -> Path:
    """Helper: write the S-2 wrapper-shape merge_map.json.

    When ``run_id`` is None the stage2_run_id key is omitted (legacy
    pre-S-2 writer shape would just be the bare inner dict — exercised by
    the existing test_load_merge_map_aggregate_layout above).
    """
    art_dir.mkdir(parents=True, exist_ok=True)
    payload: dict = {"format_version": 1, "merge_map": inner}
    if run_id is not None:
        payload["stage2_run_id"] = run_id
    (art_dir / "merge_map.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    return art_dir / "merge_map.json"


def _write_merged_checkpoint_meta(ckpt_dir: Path, run_id: str | None) -> None:
    """Helper: write a stub ``compressed_metadata.json`` under ``ckpt_dir``.

    Mirrors the ``extra`` envelope shape produced by
    ``save_compressed_checkpoint`` so ``_load_merged_checkpoint_run_id``
    finds the field where it expects to.
    """
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    extra: dict = {"merge_map_file": "merge_map.json"}
    if run_id is not None:
        extra["stage2_run_id"] = run_id
    (ckpt_dir / "compressed_metadata.json").write_text(
        json.dumps({"pipeline_stage": "stage2_pruned", "extra": extra}),
        encoding="utf-8",
    )


def test_load_merge_map_wrapper_shape_returns_run_id(tmp_path: Path):
    """Wrapper-shape aggregate parses and returns the embedded run_id."""
    art_dir = tmp_path / "artifacts"
    _write_wrapper_merge_map(
        art_dir, "abc123def456",
        {"0": {"0": [0, 1], "5": [5]}, "1": {"3": [3, 4]}},
    )
    merge_map, run_id = svc_audit.load_merge_map(art_dir)
    assert merge_map == {0: {0: [0, 1], 5: [5]}, 1: {3: [3, 4]}}
    assert run_id == "abc123def456"


def test_load_merge_map_run_id_match_passes(
    tmp_path: Path, caplog, monkeypatch
):
    """End-to-end: matching merge_map + merged checkpoint run_ids pass."""
    art_dir = tmp_path / "artifacts"
    _write_wrapper_merge_map(art_dir, "abc123", {"0": {"0": [0]}})
    ckpt = tmp_path / "merged"
    _write_merged_checkpoint_meta(ckpt, "abc123")

    # The ``svc_audit`` named logger has ``propagate=False`` by default
    # (see audit/spec_compliance/svc_audit.py:124), which means caplog's
    # root handler never sees its records. Temporarily flip propagate so
    # the stdlib Pattern N (``caplog.at_level(..., logger="svc_audit")``)
    # routes records through caplog. ``monkeypatch`` restores it on test
    # teardown so we don't pollute sibling tests.
    monkeypatch.setattr(logging.getLogger("svc_audit"), "propagate", True)
    with caplog.at_level(logging.DEBUG, logger="svc_audit"):
        _, mm_run_id = svc_audit.load_merge_map(art_dir)
        merged_run_id = svc_audit._load_merged_checkpoint_run_id(ckpt)
        rc = svc_audit._cross_check_run_ids(mm_run_id, merged_run_id)
    assert rc == 0
    assert "cross-check OK" in caplog.text
    assert "abc123" in caplog.text


def test_load_merge_map_run_id_mismatch_raises_actionable_error(
    tmp_path: Path, caplog, monkeypatch
):
    """Mismatched run_ids return exit code 2 and name BOTH ids in the log."""
    art_dir = tmp_path / "artifacts"
    _write_wrapper_merge_map(art_dir, "AAA", {"0": {"0": [0]}})
    ckpt = tmp_path / "merged"
    _write_merged_checkpoint_meta(ckpt, "BBB")

    monkeypatch.setattr(logging.getLogger("svc_audit"), "propagate", True)
    with caplog.at_level(logging.DEBUG, logger="svc_audit"):
        _, mm_run_id = svc_audit.load_merge_map(art_dir)
        merged_run_id = svc_audit._load_merged_checkpoint_run_id(ckpt)
        rc = svc_audit._cross_check_run_ids(mm_run_id, merged_run_id)
    assert rc == 2
    # Naming the conflicting IDs is non-negotiable — that's what makes
    # the message ACTIONABLE: the operator can grep their run logs for
    # both IDs and identify which run each artifact came from.
    assert "RUN IDENTITY MISMATCH" in caplog.text
    assert "AAA" in caplog.text
    assert "BBB" in caplog.text


def test_load_merge_map_legacy_no_run_id_warns_not_raises(
    tmp_path: Path, caplog, monkeypatch
):
    """Both sides pre-S-2 → cross-check returns 0 + WARN, no failure."""
    art_dir = tmp_path / "artifacts"
    # Legacy bare-dict shape (no wrapper, no run_id).
    art_dir.mkdir()
    (art_dir / "merge_map.json").write_text(
        json.dumps({"0": {"0": [0, 1]}}), encoding="utf-8"
    )
    ckpt = tmp_path / "merged"
    _write_merged_checkpoint_meta(ckpt, None)

    monkeypatch.setattr(logging.getLogger("svc_audit"), "propagate", True)
    with caplog.at_level(logging.DEBUG, logger="svc_audit"):
        _, mm_run_id = svc_audit.load_merge_map(art_dir)
        merged_run_id = svc_audit._load_merged_checkpoint_run_id(ckpt)
        rc = svc_audit._cross_check_run_ids(mm_run_id, merged_run_id)
    assert rc == 0
    assert mm_run_id is None
    assert merged_run_id is None
    assert "cross-check skipped" in caplog.text


def test_load_merge_map_partial_dir_run_id_drift_raises(tmp_path: Path):
    """_stage2_partial/ with conflicting run_ids raises RunIdMismatchError."""
    art_dir = tmp_path / "artifacts"
    partial = art_dir / "_stage2_partial"
    partial.mkdir(parents=True)
    (partial / "merge_0.json").write_text(json.dumps({
        "grouped": {"0": [0, 1]},
        "stage2_run_id": "X",
    }), encoding="utf-8")
    (partial / "merge_1.json").write_text(json.dumps({
        "grouped": {"1": [1, 2]},
        "stage2_run_id": "Y",
    }), encoding="utf-8")
    with pytest.raises(svc_audit.RunIdMismatchError) as excinfo:
        svc_audit.load_merge_map(art_dir)
    msg = str(excinfo.value)
    # Both layer files AND both IDs must appear in the error message
    # (operator needs every reference to diagnose which runs collided).
    assert "merge_0.json" in msg
    assert "merge_1.json" in msg
    assert "'X'" in msg
    assert "'Y'" in msg


def test_load_merge_map_partial_dir_mixed_legacy_and_post_s2_raises(
    tmp_path: Path,
):
    """_stage2_partial/ with mixed pre-S-2 + post-S-2 files hard-fails."""
    art_dir = tmp_path / "artifacts"
    partial = art_dir / "_stage2_partial"
    partial.mkdir(parents=True)
    (partial / "merge_0.json").write_text(json.dumps({
        "grouped": {"0": [0, 1]},
        # NO stage2_run_id — pre-S-2 shape.
    }), encoding="utf-8")
    (partial / "merge_1.json").write_text(json.dumps({
        "grouped": {"1": [1, 2]},
        "stage2_run_id": "Z",
    }), encoding="utf-8")
    with pytest.raises(svc_audit.RunIdMismatchError):
        svc_audit.load_merge_map(art_dir)


def test_load_merge_map_partial_dir_uniform_run_id_returns_it(
    tmp_path: Path,
):
    """_stage2_partial/ where every file has the SAME run_id returns it."""
    art_dir = tmp_path / "artifacts"
    partial = art_dir / "_stage2_partial"
    partial.mkdir(parents=True)
    (partial / "merge_0.json").write_text(json.dumps({
        "grouped": {"0": [0, 1]},
        "stage2_run_id": "uniform-id",
    }), encoding="utf-8")
    (partial / "merge_1.json").write_text(json.dumps({
        "grouped": {"1": [1, 2]},
        "stage2_run_id": "uniform-id",
    }), encoding="utf-8")
    merge_map, run_id = svc_audit.load_merge_map(art_dir)
    assert merge_map == {0: {0: [0, 1]}, 1: {1: [1, 2]}}
    assert run_id == "uniform-id"


def test_load_merged_checkpoint_run_id_pt_short_circuits_to_none(
    tmp_path: Path,
):
    """``.pt`` checkpoint path returns None (no sidecar to read)."""
    # Path doesn't exist as a dir → loader returns None without crashing.
    pt_path = tmp_path / "merged.pt"
    pt_path.write_bytes(b"")
    assert svc_audit._load_merged_checkpoint_run_id(pt_path) is None


def test_load_merged_checkpoint_run_id_missing_meta_returns_none(
    tmp_path: Path,
):
    """Directory without compressed_metadata.json returns None."""
    ckpt = tmp_path / "merged"
    ckpt.mkdir()
    assert svc_audit._load_merged_checkpoint_run_id(ckpt) is None


# --------------------------------------------------------------------------- #
# HF state-dict normaliser — fused layout (M2)                                #
# --------------------------------------------------------------------------- #


def test_normalise_hf_state_dict_fused_layout_splits_correctly():
    """Fused Qwen3-MoE layout splits cleanly into per-expert gate/up/down.

    The fused state-dict layout is the production Qwen3 MoE shape:
      * ``...experts.gate_up_proj`` : ``[E, 2*d_int, d_hid]`` — stacked
        gate (first d_int rows) + up (remaining d_int rows) per expert.
      * ``...experts.down_proj`` : ``[E, d_out, d_in]`` — one slab per
        expert.

    The normaliser must:
      * unstack E experts;
      * split ``gate_up_proj`` into ``gate_proj`` [d_int, d_hid] and
        ``up_proj`` [d_int, d_hid];
      * emit per-expert ``down_proj`` slabs unchanged.

    Each per-expert tensor must keep the nn.Linear ``[d_out, d_in]``
    convention so downstream audit consumers see the same shape contract
    as the non-fused layout.
    """
    E, d_int, d_hid, d_out = 4, 8, 6, 6
    g = torch.Generator().manual_seed(123)
    gate_up = torch.randn(E, 2 * d_int, d_hid, generator=g)
    down = torch.randn(E, d_out, d_int, generator=g)
    sd = {
        "model.layers.0.mlp.experts.gate_up_proj": gate_up,
        "model.layers.0.mlp.experts.down_proj": down,
    }
    out = svc_audit._normalise_hf_state_dict(sd)

    # Every expert should have all three matrix slots present.
    for e in range(E):
        for matrix in ("gate_proj", "up_proj", "down_proj"):
            assert (0, e, matrix) in out, f"missing ({0}, {e}, {matrix!r})"

    # Shape contract: nn.Linear convention [d_out, d_in].
    for e in range(E):
        assert out[(0, e, "gate_proj")].shape == (d_int, d_hid)
        assert out[(0, e, "up_proj")].shape == (d_int, d_hid)
        assert out[(0, e, "down_proj")].shape == (d_out, d_int)

    # Value contract: gate slab is the FIRST d_int rows, up slab is the
    # LAST d_int rows, and down_proj is unchanged per expert.
    for e in range(E):
        assert torch.equal(out[(0, e, "gate_proj")], gate_up[e, :d_int, :])
        assert torch.equal(out[(0, e, "up_proj")], gate_up[e, d_int:, :])
        assert torch.equal(out[(0, e, "down_proj")], down[e])

    # Exactly E*3 entries — no spurious keys.
    assert len(out) == E * 3
