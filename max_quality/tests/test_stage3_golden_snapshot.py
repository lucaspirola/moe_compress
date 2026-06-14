"""Byte-identical golden snapshot for Stage 3 artifacts.

This test pins the bytes-on-disk of the ``rank_map.json`` artifact produced by
``stage3_svd.run()`` on the ``tiny_model`` fixture. It exists so every later
sub-task of the Stage 3 plugin refactor (S3-1..S3-8) can be measured against an
immutable byte-identical target.

Determinism caveat (section 4.6 of the sub-task plan)
-----------------------------------------------------
The regen step (``MOE_REGEN_GOLDEN=1``) and the verify step (no env var)
MUST be executed on the same machine, with the same Python/torch wheel and
the same conda/venv environment. PyTorch CPU ops are bit-identical only
under those conditions; a different wheel or platform may produce different
float reprs in the JSON. If the goldens are seeded on machine A and the
suite is then run on machine B, drift is expected and is NOT a real
regression.

The snapshot is captured with ``block_refine`` OFF and ``alpha_grid`` length
1 (see ``tiny_config["stage3_svd"]``) — no L-BFGS refine, no PPL α-search —
so ``rank_map.json`` carries only integer ranks and ``alpha_by_type`` is
``null``.

First-run seeding workflow
--------------------------
1. ``MOE_REGEN_GOLDEN=1 pytest max_quality/tests/test_stage3_golden_snapshot.py -v``
   - test skips with reason "Regenerated goldens — inspect ``git diff`` then commit."
   - two new files appear under ``max_quality/tests/golden/stage3/``.
2. ``pytest max_quality/tests/test_stage3_golden_snapshot.py -v`` (no env var).
   - test must pass.
3. ``git add`` the two goldens + the ``.gitkeep`` and commit.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path

import pytest

try:
    import torch  # noqa: F401
    from moe_compress import stage1, stage3_svd
    from moe_compress.stage2 import orchestrator as stage2_reap_ream
    from moe_compress.budget.solver import BudgetDecomposition
except Exception as e:  # pragma: no cover - import-time guard
    pytest.skip(f"Stage 3 imports unavailable: {e}", allow_module_level=True)


REGEN = os.environ.get("MOE_REGEN_GOLDEN") == "1"


class _TinyTokenizer:
    """Mirror of the tokenizer used by ``test_smoke_stage3.py``.

    Redeclared locally on purpose: tests in this codebase do not import from
    each other, and coupling the snapshot to that test file would create an
    implicit cross-test dependency that the snapshot is meant to avoid.
    """

    name_or_path = "tiny-tokenizer"
    eos_token_id = 0

    def __call__(self, text, *_, **__):
        return {"input_ids": [min(ord(c) % 32, 31) for c in (text or " ")]}

    def save_pretrained(self, *_args, **_kwargs):
        return None


def _noop_save(model, tokenizer, path, **kwargs):
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


@pytest.fixture(params=["fp32", "bf16"])
def patched_stage3(request, monkeypatch, tiny_config, tiny_config_bf16):
    """Patch the stage-2 + stage-3 calibration loaders and the checkpoint saver.

    Replaces ``build_calibration_tensor`` / ``build_super_expert_slice`` with
    seeded fakes on the ``utils.calibration`` source module and on the
    ``stage2.orchestrator`` / ``stage3_svd`` modules (which bind the names by
    direct import), and stubs ``save_compressed_checkpoint`` to a no-op. Stage 1
    imports ``build_calibration_tensor`` by direct name into its own modules and
    is intentionally left on the real loader — matching the established
    ``test_smoke_stage3.py`` / ``test_stage1_golden_snapshot.py`` pattern; the
    golden is reproducible on a fixed machine + wheel + dataset cache (see the
    module-level determinism caveat).

    Parametrized over ``fp32`` (default) and ``bf16`` covariance storage so the
    eigh-based AA-SVD path is exercised under bf16 quantization end-to-end —
    defense in depth for the bf16 covariance bug fixed in §6.5.

    Redeclared locally (not imported from ``test_smoke_stage3.py``): the
    snapshot must not depend on another test module.
    """
    tiny_config = tiny_config_bf16 if request.param == "bf16" else tiny_config
    from moe_compress.utils import calibration as cal_mod

    def _fake_build(tokenizer, spec, cache_dir=None):
        torch.manual_seed(spec.seed)
        return torch.randint(0, 32, (spec.num_sequences, spec.sequence_length),
                             dtype=torch.long)

    def _fake_slice(tokenizer, spec, num_samples, cache_dir=None):
        torch.manual_seed(spec.seed + 1)
        return torch.randint(0, 32, (num_samples, spec.sequence_length),
                             dtype=torch.long)

    monkeypatch.setattr(cal_mod, "build_calibration_tensor", _fake_build)
    monkeypatch.setattr(cal_mod, "build_super_expert_slice", _fake_slice)
    monkeypatch.setattr(stage2_reap_ream, "build_calibration_tensor", _fake_build)
    monkeypatch.setattr(stage3_svd, "build_calibration_tensor", _fake_build)

    from moe_compress.utils import model_io as mio
    monkeypatch.setattr(mio, "save_compressed_checkpoint", _noop_save)
    monkeypatch.setattr(stage2_reap_ream, "save_compressed_checkpoint", _noop_save)
    monkeypatch.setattr(stage3_svd, "save_compressed_checkpoint", _noop_save)

    return tiny_config


@pytest.fixture
def stage3_case(request):
    """The parametrized case label (``"fp32"`` / ``"bf16"``) for ``patched_stage3``.

    Sourced from the same parametrization that drives ``patched_stage3`` so the
    golden filename always matches the config actually exercised. Must only be
    requested by a test that also consumes ``patched_stage3`` — it reads that
    fixture's parametrization off the test call spec.
    """
    return request.node.callspec.params["patched_stage3"]


def _run_stages_012(model, config, tmp_path):
    """Run Stages 1→2 to get a post-prune model + Stage 2 covariance artifact.

    Returns the ``BudgetDecomposition`` that Stage 3 consumes.
    """
    decomp = BudgetDecomposition(
        total_reduction_ratio=0.2,
        expert_prune_ratio=0.5,
        svd_rank_ratio=0.14,
        global_expert_budget=4,
        min_experts_per_layer=2,
        blacklisted_experts={},
    )
    stage1.run(model, _TinyTokenizer(), config, tmp_path, decomp)
    stage2_reap_ream.run(
        model, _TinyTokenizer(), config, tmp_path, device=None,
    )
    return decomp


def test_stage3_rank_map_byte_identical(tiny_model, patched_stage3, stage3_case,
                                        tmp_path):
    decomp = _run_stages_012(tiny_model, patched_stage3, tmp_path)
    stage3_svd.run(
        tiny_model, _TinyTokenizer(), patched_stage3, tmp_path, decomp, device=None,
    )

    produced = tmp_path / "stage3_svd" / "rank_map.json"
    assert produced.exists(), f"Stage 3 did not produce rank_map.json at {produced}"

    golden = (
        Path(__file__).resolve().parent
        / "golden" / "stage3" / f"rank_map.{stage3_case}.json"
    )

    if REGEN:
        golden.parent.mkdir(parents=True, exist_ok=True)
        golden.write_bytes(produced.read_bytes())
        pytest.skip("Regenerated goldens — inspect `git diff` then commit.")

    if not golden.exists():
        pytest.fail(
            f"Golden snapshot missing: {golden}\n"
            f"This must be seeded once. Run:\n"
            f"  MOE_REGEN_GOLDEN=1 pytest max_quality/tests/test_stage3_golden_snapshot.py\n"
            f"then `git diff` and commit the resulting JSON files."
        )

    if produced.read_bytes() != golden.read_bytes():
        pytest.fail(
            "Stage 3 golden snapshot drift detected:\n"
            f"  rank_map.{stage3_case}.json: produced={produced}  golden={golden}\n"
            "If intentional, re-run with MOE_REGEN_GOLDEN=1 and commit the new bytes."
        )


# ---------------------------------------------------------------------------
# Additive α-path golden variant (Tier-1 items 2/3 end-to-end coverage).
#
# The pinned golden above uses ``alpha_grid=[0.5]`` (length 1) → the uniform
# path, which never enters ``_swift_svd_plus_alpha_search`` /
# ``_redistribute_ranks_swift_svd_plus`` / the ``grouped_svs`` cache (item 2's
# target). This ADDITIVE variant flips ``alpha_grid=[0.0, 0.5, 1.0]`` with
# ``validation_samples=0`` so ``select_alpha`` takes the offline spectral-proxy
# branch (iii): the proxy runs with ``return_svs=True`` and threads its
# ``grouped_svs`` cache into the redistribute. No model forward / network — the
# WikiText-2 PPL grid (validation_samples>0) is intentionally NOT exercised.
#
# It pins a SEPARATE golden file (``rank_map.alpha.{case}.json``); the existing
# pinned goldens are left immutable.
# ---------------------------------------------------------------------------


@pytest.fixture(params=["fp32", "bf16"])
def patched_stage3_alpha(request, monkeypatch, tiny_config, tiny_config_bf16):
    """Same patching as ``patched_stage3`` but with an α-grid > 1 + offline
    spectral-proxy (``validation_samples=0``) so the α redistribution + the
    ``grouped_svs`` cache path actually run."""
    base = tiny_config_bf16 if request.param == "bf16" else tiny_config
    cfg = copy.deepcopy(base)
    sp = cfg["stage3_svd"]["swift_svd_plus"]
    sp["alpha_grid"] = [0.0, 0.5, 1.0]
    sp["validation_samples"] = 0  # offline: spectral-proxy branch (iii)
    sp["per_group_type"] = True

    from moe_compress.utils import calibration as cal_mod

    def _fake_build(tokenizer, spec, cache_dir=None):
        torch.manual_seed(spec.seed)
        return torch.randint(0, 32, (spec.num_sequences, spec.sequence_length),
                             dtype=torch.long)

    def _fake_slice(tokenizer, spec, num_samples, cache_dir=None):
        torch.manual_seed(spec.seed + 1)
        return torch.randint(0, 32, (num_samples, spec.sequence_length),
                             dtype=torch.long)

    monkeypatch.setattr(cal_mod, "build_calibration_tensor", _fake_build)
    monkeypatch.setattr(cal_mod, "build_super_expert_slice", _fake_slice)
    monkeypatch.setattr(stage2_reap_ream, "build_calibration_tensor", _fake_build)
    monkeypatch.setattr(stage3_svd, "build_calibration_tensor", _fake_build)

    from moe_compress.utils import model_io as mio
    monkeypatch.setattr(mio, "save_compressed_checkpoint", _noop_save)
    monkeypatch.setattr(stage2_reap_ream, "save_compressed_checkpoint", _noop_save)
    monkeypatch.setattr(stage3_svd, "save_compressed_checkpoint", _noop_save)

    # F1 coverage: the tiny-model Stage-2 covariance is near-singular, so the
    # α-path's ``torch.linalg.cholesky`` raises and the golden would exercise
    # only the rank-zero FALLBACK (``svdvals(W)``), NOT the Cholesky whitening
    # the fix is about. Re-condition the loaded A_cov to a guaranteed-SPD matrix
    # of the SAME shape (``M @ M.T + dim·I`` — strictly positive-definite, like
    # ``_build_acov`` in the Tier-1 tests) so Stage-3's α-path runs real Cholesky
    # end-to-end. This wraps the real loader (preserving every key/shape) and
    # touches ONLY the α-variant fixture; the non-α fixture/golden and the ×0.02
    # weight scale are untouched.
    from moe_compress.stage3 import orchestrator as s3_orch
    _real_loader = s3_orch._load_stage2_covariance

    # Process-STABLE seed from the key contents (NOT Python ``hash()``, which is
    # salted per-process via PYTHONHASHSEED → would make the golden
    # non-reproducible between the regen run and the verify run).
    _MATRIX_SEED = {"gate_proj": 0, "up_proj": 1, "down_proj": 2}

    def _stable_seed(key):
        layer_idx, expert_idx, name = key
        return 1009 + 131 * int(layer_idx) + 17 * int(expert_idx) + _MATRIX_SEED.get(name, 7)

    def _spd_loader(path):
        cov = _real_loader(path)
        spd = {}
        for key, mat in cov.items():
            mat = mat.to(torch.float64)
            dim = mat.shape[-1]
            g = torch.Generator().manual_seed(_stable_seed(key))
            M = torch.randn(dim, dim, generator=g, dtype=torch.float64)
            spd[key] = ((M @ M.T) + dim * torch.eye(dim, dtype=torch.float64)).to(mat.dtype)
        return spd

    monkeypatch.setattr(s3_orch, "_load_stage2_covariance", _spd_loader)

    return cfg


@pytest.fixture
def stage3_alpha_case(request):
    """Case label (``"fp32"`` / ``"bf16"``) for ``patched_stage3_alpha``."""
    return request.node.callspec.params["patched_stage3_alpha"]


# Tier-2 §5c: the zero-pad fix in aa_svd_factor.factor_layer (pad each expert's
# U_k/V_k up to the per-layer slot width before set_factors) removes the
# pre-existing ValueError crash on the non-uniform per-expert path, so this test
# no longer xfails. The α-grid goldens (rank_map.alpha.{fp32,bf16}.json) are a
# human-gated bless: mint them via MOE_REGEN_GOLDEN=1 after reviewing the
# Tier-2 rank diff. Until they are committed, this test fails with
# "golden missing" (expected on the impl branch; green once the goldens land).
def test_stage3_rank_map_alpha_variant_byte_identical(
    tiny_model, patched_stage3_alpha, stage3_alpha_case, tmp_path, caplog
):
    # F1 coverage proof: reset the warn-once latch and capture WARNING-level
    # logs so we can assert the α-path took the REAL Cholesky branch (not the
    # rank-zero fallback) on the re-conditioned, guaranteed-SPD A_cov.
    import logging
    from moe_compress.stage3.plugins.swift_svd_alpha import (
        _reset_raw_svd_fallback_warning,
    )
    _reset_raw_svd_fallback_warning()

    decomp = _run_stages_012(tiny_model, patched_stage3_alpha, tmp_path)
    with caplog.at_level(logging.WARNING):
        stage3_svd.run(
            tiny_model, _TinyTokenizer(), patched_stage3_alpha, tmp_path, decomp,
            device=None,
        )

    # The whole point of F1 is the Cholesky whitening path; assert it actually
    # ran end-to-end here (the re-conditioned SPD A_cov never triggers the
    # cholesky LinAlgError fallback). If this fires, the golden would be pinning
    # the fallback svdvals(W), proving nothing about the fix.
    fallback_msgs = [
        r.getMessage() for r in caplog.records
        if "D-raw-svd-fallback" in r.getMessage()
    ]
    assert not fallback_msgs, (
        "α-path took the rank-zero/Cholesky-failed FALLBACK instead of the "
        f"Cholesky whitening path — golden does not exercise the fix:\n{fallback_msgs}"
    )

    produced = tmp_path / "stage3_svd" / "rank_map.json"
    assert produced.exists(), f"Stage 3 did not produce rank_map.json at {produced}"

    golden = (
        Path(__file__).resolve().parent
        / "golden" / "stage3" / f"rank_map.alpha.{stage3_alpha_case}.json"
    )

    if REGEN:
        golden.parent.mkdir(parents=True, exist_ok=True)
        golden.write_bytes(produced.read_bytes())
        pytest.skip("Regenerated α-variant goldens — inspect `git diff` then commit.")

    if not golden.exists():
        pytest.fail(
            f"α-variant golden snapshot missing: {golden}\n"
            f"Seed once with:\n"
            f"  MOE_REGEN_GOLDEN=1 pytest max_quality/tests/test_stage3_golden_snapshot.py\n"
            f"then `git diff` and commit the resulting JSON files."
        )

    if produced.read_bytes() != golden.read_bytes():
        pytest.fail(
            "Stage 3 α-variant golden snapshot drift detected:\n"
            f"  rank_map.alpha.{stage3_alpha_case}.json: produced={produced}  "
            f"golden={golden}\n"
            "If intentional, re-run with MOE_REGEN_GOLDEN=1 and commit the new bytes."
        )
