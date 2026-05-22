"""Byte-identical golden snapshot for the Stage 6 ``stage6_eval.json`` artifact.

This test (S6-0) pins the bytes-on-disk of the ``stage6_eval.json`` artifact
produced by ``stage6_validate.run()`` on the ``tiny_model`` fixture, BEFORE the
Stage 6 plugin decomposition. Every later sub-task of the Stage 6 refactor
(S6-*) can be measured against this immutable byte-identical target.

Why a plain byte compare is safe here
-------------------------------------
The snapshot is captured with ALL eval families disabled
(``wikitext2`` / ``zero_shot`` / ``generative`` — all ``enabled: False`` in the
``tiny_config`` fixture) AND with a pre-baked teacher-cache hit. Under those
conditions ``stage6_eval.json`` carries only integers, booleans, empty dicts
and strings — there are no computed float metrics — so the artifact is fully
byte-stable. No float tolerance and no dtype parametrization are needed (unlike
the Stage 3 / Router-KD snapshots).

NaN-path note
-------------
With empty ``student`` / ``teacher`` result dicts, ``_deltas`` produces an
empty ``delta`` dict and never exercises its non-finite (NaN) skip path. This
golden therefore does NOT cover the ``_deltas`` NaN branch; a later S6-* task
should add a separate focused unit test for that path if it is refactored.

Determinism caveat
------------------
The regen step (``MOE_REGEN_GOLDEN=1``) and the verify step (no env var) must
be run on the same machine / Python+torch wheel / venv. Because this artifact
contains only integers/booleans/strings the practical drift surface is tiny,
but the discipline matches the sibling Stage 3 / Router-KD snapshots.

First-run seeding workflow
--------------------------
1. ``MOE_REGEN_GOLDEN=1 pytest max_quality/tests/test_stage6_golden_snapshot.py -v``
   - test skips with reason "Regenerated golden — inspect ``git diff`` then commit."
   - a new file appears at ``max_quality/tests/golden/stage6/stage6_eval.json``.
2. ``pytest max_quality/tests/test_stage6_golden_snapshot.py -v`` (no env var).
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
    import torch  # noqa: F401
    from moe_compress import stage6_validate
except Exception as e:  # pragma: no cover - import-time guard
    pytest.skip(f"Stage 6 imports unavailable: {e}", allow_module_level=True)


REGEN = os.environ.get("MOE_REGEN_GOLDEN") == "1"


class _TinyTokenizer:
    """Mirror of the tokenizer used by the sibling Stage 3 / Router-KD snapshots.

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
def patched_stage6(monkeypatch, tiny_config):
    """Patch Stage 6 so ``run()`` completes CPU-only with no real evals.

    Four things are neutralized:

    * ``strict_revision_pinning`` is turned OFF on the ``stage6_validate`` config
      sub-dict so ``_enforce_revision_pinning`` returns instead of raising on the
      tiny config (which pins no dataset SHAs).
    * ``_build_imatrix_calibration_corpus`` is patched to a no-op returning
      ``None`` — it is called unconditionally near the top of ``run()`` and would
      otherwise hit the network for the WikiText-2 train split.
    * The teacher eval cache is ENABLED on the config sub-dict and
      ``_load_teacher_cache`` is patched to return a permanent pre-baked hit
      ``{"results": {}, "param_counts": {"total": 0, "expert": 0}}``. This forces
      the cache-hit branch, bypassing teacher loading, the background preload
      thread and the GGUF/imatrix pipeline entirely.
    * ``_trackio_log`` is patched on the orchestrator module so the test can
      assert the run emitted at least one payload.

    ``imatrix.enabled`` is also set to ``False`` (belt-and-suspenders); on the
    cache-hit path no background GGUF thread is started anyway, and
    ``_generate_imatrix``'s own ``enabled`` guard returns early.

    HAZARD H3 — post-S6-8 patch surface
    -----------------------------------
    Pre-S6-8 the three patches targeted attributes on ``stage6_validate``
    (the monolith bound them all at module top). Post-S6-8 the orchestrator
    body lives at ``stage6.orchestrator`` and the eval-environment helper
    lives at ``stage6.plugins.eval_environment``; the patches must repoint to
    the module each name is actually resolved from at call time:

    * ``_build_imatrix_calibration_corpus`` is called by
      ``EvalEnvironmentPlugin.setup_environment`` from its OWN module scope
      → patch ``stage6.plugins.eval_environment._build_imatrix_calibration_corpus``.
    * ``_load_teacher_cache`` is imported by the orchestrator at module top
      and called directly in its preamble (HAZARD H3) → patch
      ``stage6.orchestrator._load_teacher_cache``.
    * ``_trackio_log`` is imported by the orchestrator at module top and used
      for the one-shot Stage 6 config emit (HAZARD H3) → patch
      ``stage6.orchestrator._trackio_log``.

    Returns ``(deep_copied_config, captured_list)``.
    """
    # Function-local imports so the H3 patch targets are resolved exactly
    # where the orchestrator + plugin module attribute lookup will resolve
    # them at call time. The orchestrator binds _load_teacher_cache /
    # _trackio_log via `from ... import` at module top, and the plugin binds
    # _build_imatrix_calibration_corpus inside its own module — those are the
    # attributes that need to be replaced for the patches to take effect.
    from moe_compress.stage6 import orchestrator as _s6_orch
    from moe_compress.stage6.plugins import eval_environment as _eval_env_mod

    cfg = copy.deepcopy(tiny_config)
    s6 = cfg["stage6_validate"]
    s6["strict_revision_pinning"] = False
    s6["teacher_eval_cache"] = {"enabled": True}
    s6["imatrix"] = {"enabled": False}

    monkeypatch.setattr(
        _eval_env_mod, "_build_imatrix_calibration_corpus",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        _s6_orch, "_load_teacher_cache",
        lambda *a, **k: {
            "results": {},
            "param_counts": {"total": 0, "expert": 0},
        },
    )

    captured: list[dict] = []
    monkeypatch.setattr(
        _s6_orch, "_trackio_log",
        lambda payload: captured.append(dict(payload)),
    )

    return cfg, captured


def test_stage6_eval_snapshot(tiny_model, patched_stage6, tmp_path, monkeypatch):
    cfg, captured = patched_stage6

    # Defensive: stage 6 forces attn_implementation="eager" for the teacher;
    # the student tiny-model has no real attention impl, but pin the attribute
    # so any code reading model.config._attn_implementation sees a sane value.
    monkeypatch.setattr(
        tiny_model.config, "_attn_implementation", "eager", raising=False,
    )

    produced = stage6_validate.run(
        tiny_model, _TinyTokenizer(), cfg, tmp_path, device=None,
    )
    assert produced == tmp_path / "stage6_eval.json"
    assert (tmp_path / "stage6_eval.json").is_file(), (
        "Stage 6 did not produce stage6_eval.json"
    )

    golden = (
        Path(__file__).resolve().parent / "golden" / "stage6" / "stage6_eval.json"
    )
    produced_bytes = (tmp_path / "stage6_eval.json").read_bytes()

    if REGEN:
        golden.parent.mkdir(parents=True, exist_ok=True)
        golden.write_bytes(produced_bytes)
        assert captured, (
            "Stage 6 emitted no _trackio_log payloads — the capture list is "
            "empty; fix before committing the golden."
        )
        pytest.skip("Regenerated golden — inspect `git diff` then commit.")

    if not golden.exists():
        pytest.fail(
            f"Golden snapshot missing: {golden}\n"
            "This must be seeded once. Run:\n"
            "  MOE_REGEN_GOLDEN=1 pytest "
            "max_quality/tests/test_stage6_golden_snapshot.py\n"
            "then `git diff` and commit the resulting JSON file."
        )

    golden_bytes = golden.read_bytes()
    if produced_bytes != golden_bytes:
        pytest.fail(
            "Stage 6 golden snapshot drift detected:\n"
            f"  stage6_eval.json: produced={tmp_path / 'stage6_eval.json'}  "
            f"golden={golden}\n"
            "If intentional, re-run with MOE_REGEN_GOLDEN=1 and commit the new bytes."
        )

    # --- non-vacuousness asserts: the artifact carries the expected shape ---
    payload = json.loads(produced_bytes)
    assert "overall_pass" in payload
    assert "measured_reduction" in payload
    assert "thresholds" in payload
    assert len(captured) >= 1
