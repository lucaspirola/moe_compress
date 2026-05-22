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
