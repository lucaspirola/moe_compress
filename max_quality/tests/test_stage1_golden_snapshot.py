"""Byte-identical golden snapshot for Stage 1 artifacts.

This test pins the bytes-on-disk of the three JSON artifacts produced by
``stage1.run()`` on the ``tiny_model`` fixture. It exists so every later
sub-task of the Stage 1 plugin refactor can be measured against an immutable
byte-identical target.

Determinism caveat (section 4.6 of the sub-task plan)
-----------------------------------------------------
The regen step (``MOE_REGEN_GOLDEN=1``) and the verify step (no env var)
MUST be executed on the same machine, with the same Python/torch wheel and
the same conda/venv environment. PyTorch CPU ops are bit-identical only
under those conditions; a different wheel or platform may produce different
float reprs in the JSON. If the goldens are seeded on machine A and the
suite is then run on machine B, drift is expected and is NOT a real
regression.

First-run seeding workflow
--------------------------
1. ``MOE_REGEN_GOLDEN=1 pytest max_quality/tests/test_stage1_golden_snapshot.py -v``
   - test skips with reason "Regenerated goldens — inspect ``git diff`` then commit."
   - three new files appear under ``max_quality/tests/golden/stage1/``.
2. ``pytest max_quality/tests/test_stage1_golden_snapshot.py -v`` (no env var).
   - test must pass.
3. ``git add`` the three goldens + the ``.gitkeep`` and commit.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

try:
    import torch  # noqa: F401
    from moe_compress import stage1
    from moe_compress.budget.solver import BudgetDecomposition
except Exception as e:  # pragma: no cover - import-time guard
    pytest.skip(f"Stage 1 imports unavailable: {e}", allow_module_level=True)


REGEN = os.environ.get("MOE_REGEN_GOLDEN") == "1"


class _TinyTokenizer:
    """Mirror of the tokenizer used by ``test_stage1_e2e.py``.

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


def test_stage1_artifacts_byte_identical(tiny_model, tiny_config, tmp_path):
    decomp = BudgetDecomposition(
        total_reduction_ratio=0.20,
        expert_prune_ratio=0.25,
        svd_rank_ratio=0.0,
        global_expert_budget=5,
        min_experts_per_layer=2,
        blacklisted_experts={},
    )
    stage1.run(tiny_model, _TinyTokenizer(), tiny_config, tmp_path, decomp)

    artifacts = [
        "stage1_blacklist.json",
        "stage1_budgets.json",
        "stage1_ablation_filter.json",
    ]
    golden_dir = Path(__file__).resolve().parent / "golden" / "stage1"
    drift = []
    for name in artifacts:
        produced = tmp_path / name
        golden = golden_dir / name
        assert produced.exists(), f"Stage 1 did not produce {name} at {produced}"
        if REGEN:
            golden.parent.mkdir(parents=True, exist_ok=True)
            golden.write_bytes(produced.read_bytes())
            continue
        if not golden.exists():
            pytest.fail(
                f"Golden snapshot missing: {golden}\n"
                f"This must be seeded once. Run:\n"
                f"  MOE_REGEN_GOLDEN=1 pytest max_quality/tests/test_stage1_golden_snapshot.py\n"
                f"then `git diff` and commit the resulting JSON files."
            )
        if produced.read_bytes() != golden.read_bytes():
            drift.append((name, produced, golden))
    if REGEN:
        pytest.skip("Regenerated goldens — inspect `git diff` then commit.")
    if drift:
        msg = "\n".join(
            f"  {n}: produced={p}  golden={g}" for n, p, g in drift
        )
        pytest.fail(
            "Stage 1 golden snapshot drift detected:\n"
            + msg
            + "\nIf intentional, re-run with MOE_REGEN_GOLDEN=1 and commit the new bytes."
        )
