"""Golden snapshot for the unified Router-KD module (Stage 2.5 + Stage 5).

``stage5_router_kd`` is a single module that serves BOTH Stage 2.5 and Stage 5,
selected by the keyword-only ``stage_key`` parameter (``"stage2p5"`` /
``"stage5"``). This test (RK-0) captures a golden baseline of its observable
outputs BEFORE the plugin decomposition, so every later sub-task of the
Router-KD refactor can be measured against an immutable target.

Two artifacts are pinned, with two different strategies:

* ``compressed_metadata.json`` (under ``{stage_key}_final/``) — BYTE-pinned via a
  raw ``read_bytes()`` compare. It carries only integer/string content
  (``pipeline_stage`` and friends), so a byte-identical compare is safe and
  catches any drift in the saved metadata.
* the loss trace — TOLERANCE-pinned with ``math.isclose(rel_tol=1e-5,
  abs_tol=1e-7)``. Router-KD is a TRAINING stage (AdamW, KD loss, 1 epoch);
  the loss trace carries computed float KD-loss values (``loss`` / ``raw_kl``)
  that are not guaranteed bit-identical the way pure integer artifacts are.
  Integer ``step`` values are still compared exactly. There is NO loss-trace
  disk artifact — loss values are emitted only through ``_trackio_log`` calls,
  so the test captures them by monkeypatching ``stage5_router_kd._trackio_log``.

Determinism caveat (same machine / wheel / venv)
------------------------------------------------
The regen step (``MOE_REGEN_GOLDEN=1``) and the verify step (no env var) MUST
be executed on the same machine, with the same Python/torch wheel and the same
conda/venv environment. PyTorch CPU ops — and the AdamW float math in this
training stage — are reproducible only under those conditions. Goldens seeded
on machine A and verified on machine B may drift; that is NOT a real
regression. Do NOT widen the tolerance to mask a same-machine drift — that
indicates a genuinely unpinned RNG source and must be fixed at the seeding.

First-run seeding workflow
--------------------------
1. ``MOE_REGEN_GOLDEN=1 pytest max_quality/tests/test_router_kd_golden_snapshot.py -v``
   - both params skip with reason "Regenerated goldens — inspect ``git diff`` then commit."
   - four new files appear under ``max_quality/tests/golden/router_kd/``.
2. ``pytest max_quality/tests/test_router_kd_golden_snapshot.py -v`` (no env var).
   - both params must pass.
3. ``git add`` the four goldens + the ``.gitkeep`` and commit.
"""

from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path

import pytest

try:
    import torch  # noqa: F401
    from moe_compress import stage1, stage5_router_kd
    from moe_compress.stage2 import orchestrator as stage2_reap_ream
    from moe_compress.budget.solver import BudgetDecomposition
except Exception as e:  # pragma: no cover - import-time guard
    pytest.skip(f"Router-KD imports unavailable: {e}", allow_module_level=True)


REGEN = os.environ.get("MOE_REGEN_GOLDEN") == "1"


class _TinyTokenizer:
    """Mirror of the tokenizer used by ``test_smoke_stage5_resume.py``.

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


def _prepare_model_and_merge_map(model, config, tmp_path, monkeypatch):
    """Run stages 1+2 and write an identity merge_map.json at stage2_pruned/.

    Copied from ``test_smoke_stage5_resume.py``: Stage 2's real merge_map maps
    new_idx → [original_expert_ids], but with teacher == student (same
    post-stage-2 model) ``_pool_teacher_logits`` would index original expert
    IDs into a tensor whose last dim is num_new_experts — an out-of-bounds
    error. A trivial identity map (each expert maps to itself) avoids the
    pooling step entirely.
    """
    from moe_compress.utils import calibration as cal_mod
    from moe_compress.utils.model_io import iter_moe_layers

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

    from moe_compress.utils import model_io as mio
    monkeypatch.setattr(mio, "save_compressed_checkpoint", _noop_save)
    monkeypatch.setattr(stage2_reap_ream, "save_compressed_checkpoint", _noop_save)

    decomp = BudgetDecomposition(
        total_reduction_ratio=0.2,
        expert_prune_ratio=0.5,
        svd_rank_ratio=0.14,
        global_expert_budget=4,
        min_experts_per_layer=2,
        blacklisted_experts={},
    )
    stage1.run(model, _TinyTokenizer(), config, tmp_path, decomp)
    stage2_reap_ream.run(model, _TinyTokenizer(), config, tmp_path, device=None)

    moe_layer_refs = list(iter_moe_layers(model))
    trivial_map = {
        str(ref.layer_idx): {str(i): [i] for i in range(ref.num_routed_experts)}
        for ref in moe_layer_refs
    }
    (tmp_path / "stage2_pruned").mkdir(parents=True, exist_ok=True)
    (tmp_path / "stage2_pruned" / "merge_map.json").write_text(json.dumps(trivial_map))


@pytest.fixture
def patched_router_kd(monkeypatch, tiny_config):
    """Patch calibration loaders, the stage-2 saver, and Router-KD's teacher.

    Replaces ``build_calibration_tensor`` / ``build_super_expert_slice`` with
    seeded fakes on the ``utils.calibration`` source module and on the modules
    that bind those names by direct import (``stage2.orchestrator`` and — after
    RK-8 — ``router_kd.orchestrator``, which is the real Router-KD phase
    sequencer; it binds only ``build_calibration_tensor``).

    ``save_compressed_checkpoint`` is stubbed to a no-op on ``utils.model_io``
    and ``stage2.orchestrator`` ONLY — ``router_kd.orchestrator`` calls the REAL
    ``save_compressed_checkpoint`` so the ``{stage_key}_final/compressed_metadata.json``
    artifact this golden pins is actually produced.

    ``load_model`` is patched on ``router_kd.plugins.teacher`` (where the live
    teacher plugin binds it) so the teacher == the student (same Python object):
    the tiny model has no real teacher checkpoint to load. KL divergence
    converges to zero once teacher and student share weights, which is fine —
    this golden pins whatever the deterministic run produces.

    ``router_kd.orchestrator._trackio_log`` is patched to a capture closure that
    appends every emitted ``payload`` dict to ``captured`` — the loss trace is
    read off this list (there is no loss-trace disk artifact).

    Returns ``(tiny_config_unchanged, captured_list)``.
    """
    from moe_compress.utils import calibration as cal_mod
    from moe_compress.router_kd import orchestrator as rk_orchestrator

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
    # RK-8: the real Router-KD orchestrator binds build_calibration_tensor by
    # direct import — patch it there.
    monkeypatch.setattr(rk_orchestrator, "build_calibration_tensor", _fake_build)
    # router_kd.orchestrator binds only build_calibration_tensor by direct
    # import; patch build_super_expert_slice on it only if it is bound there.
    if hasattr(rk_orchestrator, "build_super_expert_slice"):
        monkeypatch.setattr(rk_orchestrator, "build_super_expert_slice", _fake_slice)

    from moe_compress.utils import model_io as mio
    monkeypatch.setattr(mio, "save_compressed_checkpoint", _noop_save)
    monkeypatch.setattr(stage2_reap_ream, "save_compressed_checkpoint", _noop_save)
    # NOTE: router_kd.orchestrator.save_compressed_checkpoint is intentionally
    # NOT patched — the {stage_key}_final/compressed_metadata.json it writes is
    # the byte-pinned artifact under test.

    captured: list[dict] = []

    def _capture_trackio(payload):
        captured.append(dict(payload))

    # RK-8: the Router-KD orchestrator emits trackio from its own module.
    monkeypatch.setattr(rk_orchestrator, "_trackio_log", _capture_trackio)

    return tiny_config, captured


def _load_student_factory(student, tokenizer, monkeypatch):
    """Patch load_model so teacher == student.

    RK-8: the live-teacher plugin (``router_kd.plugins.teacher``) binds
    ``load_model`` by direct import — patch it there. ``utils.model_io`` is
    also patched so any other consumer of the source name sees the stub.
    """
    from moe_compress.utils import model_io as mio
    from moe_compress.router_kd.plugins import teacher as rk_teacher

    def _load_student(*_args, **_kwargs):
        return student, tokenizer

    monkeypatch.setattr(mio, "load_model", _load_student)
    monkeypatch.setattr(rk_teacher, "load_model", _load_student)


@pytest.mark.parametrize("stage_id", ["stage2p5", "stage5"])
def test_router_kd_golden(tiny_model, patched_router_kd, stage_id, tmp_path, monkeypatch):
    base_config, captured = patched_router_kd

    cfg = copy.deepcopy(base_config)
    # Router-KD is a 1-epoch training stage; confirm the golden is captured at
    # epochs == 1 (the tiny_config default), and force a per-step loss emit so
    # the loss trace is guaranteed non-empty.
    assert cfg["stage5_router_kd"]["epochs"] == 1, (
        "Router-KD golden must be captured with epochs == 1"
    )
    cfg["logging"]["log_every_n_steps"] = 1

    # The REAL save_compressed_checkpoint writes compressed_metadata.json (the
    # artifact under test) and then calls model.config.save_pretrained(). The
    # tiny-model fixture's _TinyConfig has no such method; attach a no-op so the
    # real saver runs to completion. The metadata JSON is written BEFORE this
    # call, so the no-op does not affect the pinned artifact.
    monkeypatch.setattr(
        tiny_model.config, "save_pretrained", lambda *a, **k: None, raising=False,
    )

    # Prep is identical for both stage_ids — the only difference is the
    # stage_key argument passed to stage5_router_kd.run below.
    _prepare_model_and_merge_map(tiny_model, cfg, tmp_path, monkeypatch)
    _load_student_factory(tiny_model, _TinyTokenizer(), monkeypatch)

    stage5_router_kd.run(
        tiny_model, _TinyTokenizer(), cfg, tmp_path, device=None, stage_key=stage_id,
    )

    produced_meta = tmp_path / f"{stage_id}_final" / "compressed_metadata.json"
    assert produced_meta.exists(), (
        f"Router-KD (stage_key={stage_id}) did not produce {produced_meta}"
    )

    # Loss trace: the ordered list of KD-loss log-window payloads, read off the
    # captured _trackio_log calls. Each "stage5/loss" value is the mean over a
    # log_every_n_steps window; this test forces log_every_n_steps == 1, so each
    # payload is exactly one optimizer step. Filter on the presence of
    # "stage5/loss" (the training-window payload — the one-shot run-level config
    # emit lacks that key) — payload keys verified against the
    # _trackio_log(payload) site in stage5_router_kd.py.
    produced_trace = [
        {"step": p["stage5/step"], "loss": p["stage5/loss"], "raw_kl": p["stage5/raw_kl"]}
        for p in captured
        if "stage5/loss" in p
    ]

    golden_dir = Path(__file__).resolve().parent / "golden" / "router_kd"
    golden_meta = golden_dir / f"compressed_metadata.{stage_id}.json"
    golden_trace = golden_dir / f"loss_trace.{stage_id}.json"

    if REGEN:
        golden_dir.mkdir(parents=True, exist_ok=True)
        golden_meta.write_bytes(produced_meta.read_bytes())
        golden_trace.write_text(json.dumps(produced_trace, indent=2))
        assert produced_trace, (
            f"Router-KD (stage_key={stage_id}) loss trace is EMPTY — "
            "log_every_n_steps did not produce per-step emits; fix before commit."
        )
        pytest.skip("Regenerated goldens — inspect `git diff` then commit.")

    if not golden_meta.exists() or not golden_trace.exists():
        pytest.fail(
            f"Golden snapshot missing for stage_key={stage_id}:\n"
            f"  {golden_meta}\n"
            f"  {golden_trace}\n"
            "These must be seeded once. Run:\n"
            "  MOE_REGEN_GOLDEN=1 pytest max_quality/tests/test_router_kd_golden_snapshot.py\n"
            "then `git diff` and commit the resulting JSON files."
        )

    # --- compressed_metadata.json: byte-identical ---
    if produced_meta.read_bytes() != golden_meta.read_bytes():
        pytest.fail(
            "Router-KD golden snapshot drift detected:\n"
            f"  compressed_metadata.{stage_id}.json: "
            f"produced={produced_meta}  golden={golden_meta}\n"
            "If intentional, re-run with MOE_REGEN_GOLDEN=1 and commit the new bytes."
        )

    # --- loss trace: tolerance-pinned ---
    golden_rows = json.loads(golden_trace.read_text())
    if len(produced_trace) != len(golden_rows):
        pytest.fail(
            f"Router-KD loss trace length drift (stage_key={stage_id}): "
            f"produced {len(produced_trace)} rows, golden {len(golden_rows)} rows."
        )
    for prod, gold in zip(produced_trace, golden_rows):
        if prod["step"] != gold["step"]:
            pytest.fail(
                f"Router-KD loss trace step drift (stage_key={stage_id}): "
                f"produced step={prod['step']}, golden step={gold['step']}."
            )
        for field in ("loss", "raw_kl"):
            if not math.isclose(prod[field], gold[field],
                                rel_tol=1e-5, abs_tol=1e-7):
                pytest.fail(
                    f"Router-KD loss trace drift (stage_key={stage_id}) at "
                    f"step={gold['step']}, field={field}: "
                    f"produced={prod[field]!r}  golden={gold[field]!r}\n"
                    "If intentional, re-run with MOE_REGEN_GOLDEN=1 and commit. "
                    "If on the same machine/wheel/venv, this is an unpinned RNG "
                    "source — fix the seeding, do NOT widen the tolerance."
                )
