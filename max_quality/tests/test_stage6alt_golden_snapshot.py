"""Byte-identical golden snapshot for the Stage 6alt ``stage6alt_eval.json`` artifact.

This test (S6A-0) pins the bytes-on-disk of the ``stage6alt_eval.json`` artifact
produced by ``stage6alt_thermometer.run()`` on the ``tiny_model`` fixture,
BEFORE the Stage 6alt plugin decomposition. Every later sub-task of the
Stage 6alt refactor (S6A-*) can be measured against this immutable byte-
identical target.

Why a plain byte compare is safe here
-------------------------------------
The snapshot is captured with the six heavy module-top helpers patched out
on the ``stage6alt_thermometer`` monolith (no real corpus build, no real
forward pass, no lm-eval, and a pre-baked teacher-cache hit). Under those
conditions ``stage6alt_eval.json`` carries only fixed integers, exact
floats (``3.0``, ``2.5`` → ``bpt_gap = 0.5``), ``None``s, strings and
nested dicts — there are no computed metrics that could drift across
machines. ``save_json_artifact`` writes with ``indent=2, sort_keys=True``,
so the bytes are fully stable.

BLOCKER fix — pinned ``teacher_cache.path``
-------------------------------------------
The monolith's ``run()`` writes ``teacher_cache.path`` as
``str(therm.get("teacher_cache_path") or artifacts_dir/"thermometer_teacher_cache.json")``.
``artifacts_dir`` is pytest's ``tmp_path`` and therefore volatile across
runs (``/tmp/pytest-of-USER/pytest-N/.../``), which would make a byte
compare flap. The fixture pins ``thermometer.teacher_cache_path`` to a
fixed sentinel string (``/dev/null/stub_teacher_cache.json``) so the
JSON field stays stable. The path is never opened — the
``_load_thermo_teacher_cache`` patch returns a cache HIT unconditionally.

Determinism caveat
------------------
The regen step (``MOE_REGEN_GOLDEN=1``) and the verify step (no env var)
must be run on the same machine / Python+torch wheel / venv. Because
this artifact contains only fixed values the practical drift surface is
tiny, but the discipline matches the sibling Stage 6 / Router-KD snapshots.

First-run seeding workflow
--------------------------
1. ``MOE_REGEN_GOLDEN=1 pytest max_quality/tests/test_stage6alt_golden_snapshot.py -v``
   - test skips with reason "Regenerated golden — inspect ``git diff`` then commit."
   - a new file appears at ``max_quality/tests/golden/stage6alt/stage6alt_eval.json``.
2. ``pytest max_quality/tests/test_stage6alt_golden_snapshot.py -v`` (no env var).
   - test must pass.
3. ``git add`` the golden + the ``.gitkeep`` and commit.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest

try:
    import torch
    from moe_compress import stage6alt_thermometer
except Exception as e:  # pragma: no cover - import-time guard
    pytest.skip(f"Stage 6alt imports unavailable: {e}", allow_module_level=True)


REGEN = os.environ.get("MOE_REGEN_GOLDEN") == "1"


class _TinyTokenizer:
    """Mirror of the tokenizer used by the sibling Stage 6 snapshot.

    Redeclared locally on purpose: tests in this codebase do not import from
    each other, and coupling the snapshot to another test file would create an
    implicit cross-test dependency that the snapshot is meant to avoid.
    """

    name_or_path = "tiny-tokenizer"
    eos_token_id = 0

    def __call__(self, text, *_, **__):
        return {"input_ids": [min(ord(c) % 32, 31) for c in (text or " ")]}

    def save_pretrained(self, *_args, **_kwargs):
        return None


@pytest.fixture
def patched_stage6alt(monkeypatch, tiny_config):
    """Patch Stage 6alt so ``run()`` completes CPU-only with no real evals.

    Six things are neutralized on the post-S6A-6 plugin modules (HAZARD
    H3 — the orchestrator flip moved the call sites off
    ``stage6alt_thermometer`` and onto the plugins). Each patch targets
    the module that *owns the call site*, so ``monkeypatch.setattr`` on
    the plugin module attribute bites the live invocation:

    * ``_set_experts_implementation_s6`` on
      ``stage6alt.plugins.thermo_environment`` → no-op (the tiny_model
      has no Qwen3_5 fused-experts implementation switch to twiddle).
    * ``_apply_stage6_kernel_patches`` on
      ``stage6alt.plugins.thermo_environment`` → no-op (no fla /
      GatedDeltaNet on the tiny_model).
    * ``_build_thermo_corpus`` on ``stage6alt.plugins.thermo_corpus``
      → returns a constant ``(calib_ids, corpus_meta, corpus_id)`` so
      no dataset is loaded. ``corpus_meta`` mirrors the nemotron branch
      of the monolith (``name, num_sequences, sequence_length,
      effective_seed, seed_offset, subset_weights``) — the exact dict
      shape that lands in the JSON.
    * ``_bpt_from_nll`` on ``stage6alt.plugins.bpt_metric`` → returns
      ``(3.0, None)``: a finite student BPT and no argmax (forces
      ``top1_agreement`` to ``None``, which is the desired fixed shape).
    * ``_lm_eval_subset`` on ``stage6alt.plugins.zero_shot_subset``
      → returns all-None metrics (avoids any lm-eval import / harness
      call).
    * ``_load_thermo_teacher_cache`` on
      ``stage6alt.plugins.thermo_teacher_provider`` → returns a
      pre-baked teacher dict (cache HIT, ``teacher_bpt=2.5``, all other
      metrics None, no per-token argmax) so the teacher-load branch is
      bypassed entirely.

    Additionally the config's ``thermometer`` sub-dict is overlaid with a
    fixed ``teacher_cache_path`` (``/dev/null/stub_teacher_cache.json``)
    so the JSON's ``teacher_cache.path`` field does not embed pytest's
    volatile ``tmp_path`` — the BLOCKER fix (preserved from S6A-0) that
    lets the snapshot be byte-stable across runs.

    Returns the deep-copied, overlaid config.
    """
    # Function-local imports of the plugin modules — the H3 patches now
    # target the plugin modules that own the call sites (post-S6A-6
    # flip), not the legacy ``stage6alt_thermometer`` monolith. Importing
    # at function scope mirrors the test-isolation discipline used by
    # the sibling stage6 golden.
    from moe_compress.stage6alt.plugins import (
        bpt_metric,
        thermo_corpus,
        thermo_environment,
        thermo_teacher_provider,
        zero_shot_subset,
    )

    cfg = copy.deepcopy(tiny_config)
    s6 = cfg["stage6_validate"]
    s6["mode"] = "thermometer"
    s6["thermometer"] = {
        "corpus": "nemotron",
        "num_sequences": 4,
        "sequence_length": 16,
        "bpt_batch_size": 4,
        "lm_eval_batch_size": "auto:4",
        "arc_easy_limit": 2,
        "hellaswag_limit": 2,
        # BLOCKER fix: pin the teacher_cache.path JSON field to a fixed
        # string. Never opened — _load_thermo_teacher_cache is patched
        # to return a HIT unconditionally.
        "teacher_cache_path": "/dev/null/stub_teacher_cache.json",
    }

    # 1. Kernel/impl switches → no-op (on the thermo_environment plugin
    # module, which is what the live ``setup_thermo_environment`` hook
    # resolves via ``from ...stage6.plugins.eval_environment import ...``
    # re-exported at module scope).
    monkeypatch.setattr(
        thermo_environment, "_set_experts_implementation_s6",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        thermo_environment, "_apply_stage6_kernel_patches",
        lambda *a, **k: None,
    )

    # 2. Corpus build → constant (calib_ids, corpus_meta, corpus_id).
    _calib = torch.zeros(4, 16, dtype=torch.long)
    _corpus_meta = {
        "name": "nemotron",
        "num_sequences": 4,
        "sequence_length": 16,
        "effective_seed": 0,
        "seed_offset": 715,
        "subset_weights": {
            "math": 0.35, "swe": 0.25, "chat": 0.25, "science": 0.15,
        },
    }
    monkeypatch.setattr(
        thermo_corpus, "_build_thermo_corpus",
        lambda *a, **k: (_calib, _corpus_meta, "nemotron:stub"),
    )

    # 3. BPT → (3.0, None). Finite BPT + no argmax → top1_agreement None.
    monkeypatch.setattr(
        bpt_metric, "_bpt_from_nll",
        lambda *a, **k: (3.0, None),
    )

    # 4. lm-eval subset → all-None.
    monkeypatch.setattr(
        zero_shot_subset, "_lm_eval_subset",
        lambda *a, **k: {
            "arc_easy_acc_norm": None,
            "hellaswag_acc_norm": None,
            "acc_norm_sum": None,
        },
    )

    # 5. Teacher cache → HIT with teacher_bpt=2.5, no argmax.
    monkeypatch.setattr(
        thermo_teacher_provider, "_load_thermo_teacher_cache",
        lambda *a, **k: {
            "teacher_bpt": 2.5,
            "teacher_arc_easy_acc_norm": None,
            "teacher_hellaswag_acc_norm": None,
            "teacher_acc_norm_sum": None,
            "teacher_argmax": None,
        },
    )

    return cfg


def test_stage6alt_eval_snapshot(tiny_model, patched_stage6alt, tmp_path, monkeypatch):
    cfg = patched_stage6alt

    # Defensive: stage 6alt's _bpt_from_nll guards on
    # model.config._attn_implementation == "eager"; the helper itself is
    # patched away, but we pin the attribute too so any future S6A-* code
    # path that reads it sees a sane value.
    monkeypatch.setattr(
        tiny_model.config, "_attn_implementation", "eager", raising=False,
    )

    produced = stage6alt_thermometer.run(
        tiny_model, _TinyTokenizer(), cfg, tmp_path, device=None,
    )
    assert produced == tmp_path / "stage6alt_eval.json"
    assert (tmp_path / "stage6alt_eval.json").is_file(), (
        "Stage 6alt did not produce stage6alt_eval.json"
    )

    golden = (
        Path(__file__).resolve().parent
        / "golden" / "stage6alt" / "stage6alt_eval.json"
    )
    produced_bytes = (tmp_path / "stage6alt_eval.json").read_bytes()

    if REGEN:
        golden.parent.mkdir(parents=True, exist_ok=True)
        golden.write_bytes(produced_bytes)
        pytest.skip("Regenerated golden — inspect `git diff` then commit.")

    if not golden.exists():
        pytest.fail(
            f"Golden snapshot missing: {golden}\n"
            "This must be seeded once. Run:\n"
            "  MOE_REGEN_GOLDEN=1 pytest "
            "max_quality/tests/test_stage6alt_golden_snapshot.py\n"
            "then `git diff` and commit the resulting JSON file."
        )

    golden_bytes = golden.read_bytes()
    if produced_bytes != golden_bytes:
        pytest.fail(
            "Stage 6alt golden snapshot drift detected:\n"
            f"  stage6alt_eval.json: produced={tmp_path / 'stage6alt_eval.json'}  "
            f"golden={golden}\n"
            "If intentional, re-run with MOE_REGEN_GOLDEN=1 and commit the new bytes."
        )

    # --- non-vacuousness asserts: the artifact carries the expected shape ---
    payload = json.loads(produced_bytes)
    assert payload["stage"] == "6alt"
    assert payload["mode"] == "thermometer"
    for key in (
        "student_bpt", "teacher_bpt", "bpt_gap", "corpus", "teacher_cache",
    ):
        assert key in payload, f"missing key in stage6alt_eval.json: {key}"
    assert payload["teacher_cache"]["hit"] is True
