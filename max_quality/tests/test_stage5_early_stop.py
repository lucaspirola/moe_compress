"""CPU-only tests for Stage 2.5 / Stage 5 router-KD early stopping.

Early stopping (2026-05-17 overfit fix) is a config-gated patience mechanism
in `stage5_router_kd.run()`. The S0 live run showed the pure teacher↔student
raw_kl rising 7× over the back half of a multi-epoch schedule while save-best
pinned the export at an early step — so ~88% of the schedule was discarded
compute. `early_stop_patience > 0` ends the run once the raw_kl EMA stops
improving, automatically.

These tests exercise the real `run()` code path on the tiny synthetic model
(student acts as its own teacher — see test_smoke_stage5_resume for the
wiring rationale) and verify:
  * with `early_stop_patience > 0` and a non-improving metric, the run stops
    BEFORE the full optimizer-step schedule;
  * with `early_stop_patience == 0` (default), the run walks the full
    schedule — i.e. the flag-off path is unchanged from pre-fix `main`;
  * the early-stop state survives a crash-resume (no_improve_windows /
    es_ref_ema are persisted in and restored from the checkpoint).
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch

from moe_compress import stage5_router_kd

# Reuse the fixtures + helpers proven by the Stage 5 resume smoke tests.
# `tests/` is a package but pytest's default import mode does not put it on
# sys.path for bare sibling imports — add the directory explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_smoke_stage5_resume import (  # noqa: E402
    _make_stage5_config,
    _prepare_model_and_merge_map,
    _run_stage5_self_kd,
    patched_stage5,  # noqa: F401  (pytest fixture)
)


def _cfg_for_early_stop(
    base_config: dict,
    *,
    n_samples: int,
    patience: int,
    log_every: int = 1,
) -> dict:
    """Stage-5 test config with a knownstep count and tunable early-stop.

    batch_size=1, grad_accum=1 → n_samples optimizer steps. log_every=1 makes
    every step its own log window, so the early-stop counter advances once per
    step and the test stays fast and deterministic.
    """
    cfg = _make_stage5_config(base_config, ckpt_every=1)
    cfg = copy.deepcopy(cfg)
    s5 = cfg["stage5_router_kd"]
    s5["max_calibration_samples"] = n_samples
    s5["early_stop_patience"] = patience
    # Constant temperature so raw_kl == loss and the metric trajectory is the
    # genuine teacher↔student signal, not a T-ramp artefact.
    s5["kd_temperature_start"] = 1.0
    s5["kd_temperature_end"] = 1.0
    cfg["logging"] = dict(cfg.get("logging", {}))
    cfg["logging"]["log_every_n_steps"] = log_every
    return cfg


def _latest_ckpt_step(tmp_path) -> int:
    ckpts = sorted(
        (tmp_path / "_stage5_partial").glob("step_*.pt"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    assert ckpts, "no Stage 5 checkpoint written"
    return int(ckpts[-1].stem.split("_")[1])


def test_early_stop_triggers_before_full_schedule(
    tiny_model, patched_stage5, tmp_path, monkeypatch
):
    """With patience>0 and a non-improving metric the run stops early.

    Self-KD: teacher == student, so after the first optimizer step KL → 0 and
    the raw_kl EMA flatlines — it never improves on its running minimum, so the
    no-improve counter climbs to `patience` and training breaks well before the
    20-step schedule end.
    """
    _prepare_model_and_merge_map(tiny_model, patched_stage5, tmp_path, monkeypatch)
    state = copy.deepcopy(tiny_model.state_dict())

    n_samples, patience = 20, 3
    cfg = _cfg_for_early_stop(patched_stage5, n_samples=n_samples, patience=patience)
    tiny_model.load_state_dict(state)

    _run_stage5_self_kd(tiny_model, cfg, tmp_path, monkeypatch)

    last_step = _latest_ckpt_step(tmp_path)
    assert last_step < n_samples, (
        f"early_stop_patience={patience} should have stopped the run before the "
        f"full {n_samples}-step schedule, but it ran to step {last_step}"
    )


def test_early_stop_disabled_runs_full_schedule(
    tiny_model, patched_stage5, tmp_path, monkeypatch
):
    """patience=0 (the default) → the run walks the full schedule unchanged.

    This is the flag-OFF guarantee: with early_stop_patience=0 the loop is
    byte-identical to pre-2026-05-17 `main`, so it must reach the last step.
    """
    _prepare_model_and_merge_map(tiny_model, patched_stage5, tmp_path, monkeypatch)
    state = copy.deepcopy(tiny_model.state_dict())

    n_samples = 8
    cfg = _cfg_for_early_stop(patched_stage5, n_samples=n_samples, patience=0)
    tiny_model.load_state_dict(state)

    _run_stage5_self_kd(tiny_model, cfg, tmp_path, monkeypatch)

    last_step = _latest_ckpt_step(tmp_path)
    assert last_step == n_samples, (
        f"with early_stop_patience=0 the run must complete all {n_samples} "
        f"steps, but the last checkpoint is at step {last_step}"
    )


def test_early_stop_state_persisted_in_checkpoint(
    tiny_model, patched_stage5, tmp_path, monkeypatch
):
    """no_improve_windows / es_ref_ema are written into the resume checkpoint.

    Without persistence a crash-resume would reset patience and re-run the
    discarded back half — defeating the fix. Every checkpoint must carry both
    fields (None is acceptable only when early stopping never advanced; here
    patience>0 so they are populated).
    """
    _prepare_model_and_merge_map(tiny_model, patched_stage5, tmp_path, monkeypatch)
    state = copy.deepcopy(tiny_model.state_dict())

    cfg = _cfg_for_early_stop(patched_stage5, n_samples=20, patience=3)
    tiny_model.load_state_dict(state)

    _run_stage5_self_kd(tiny_model, cfg, tmp_path, monkeypatch)

    last_step = _latest_ckpt_step(tmp_path)
    payload = torch.load(
        tmp_path / "_stage5_partial" / f"step_{last_step}.pt", map_location="cpu"
    )
    assert "no_improve_windows" in payload, "checkpoint missing no_improve_windows"
    assert "es_ref_ema" in payload, "checkpoint missing es_ref_ema"
    assert payload["no_improve_windows"] is not None
    assert payload["es_ref_ema"] is not None
    # The early-stop break fires when the counter reaches `patience`.
    assert int(payload["no_improve_windows"]) >= 3


def test_early_stop_state_restored_on_resume(
    tiny_model, patched_stage5, tmp_path, monkeypatch
):
    """A crash-resume must RESTORE the early-stop counter, not reset it to 0.

    Run 1 trains until early-stop fires at step S1 — its final checkpoint
    carries ``no_improve_windows == patience``. Run 2 resumes from that
    checkpoint: if ``no_improve_windows`` (and ``es_ref_ema``) are restored, the
    counter is already at ``patience``, so the first still-plateaued window
    pushes it over and run 2 stops ~1 step later. If either field were reset on
    resume, run 2 would re-train ~``patience`` more non-improving windows.
    Asserting run 2 stops below ``S1 + patience`` proves the resume restore
    path — a typo in a restore key re-counts from zero and fails this. (Test 3
    only checks the fields are *written*; this checks they are *read back*.)
    """
    _prepare_model_and_merge_map(tiny_model, patched_stage5, tmp_path, monkeypatch)
    state = copy.deepcopy(tiny_model.state_dict())

    n_samples, patience = 40, 4
    cfg = _cfg_for_early_stop(patched_stage5, n_samples=n_samples, patience=patience)

    # --- Run 1: trains until early-stop fires. ---
    tiny_model.load_state_dict(state)
    _run_stage5_self_kd(tiny_model, cfg, tmp_path, monkeypatch)
    s1 = _latest_ckpt_step(tmp_path)
    assert s1 < n_samples, "run 1 should have early-stopped before the schedule end"
    v1 = int(torch.load(
        tmp_path / "_stage5_partial" / f"step_{s1}.pt", map_location="cpu"
    )["no_improve_windows"])
    assert v1 >= patience, (
        f"run 1 early-stopped — its checkpoint counter should be ≥{patience}, got {v1}"
    )

    # --- Run 2: resume from the early-stopped checkpoint. ---
    tiny_model.load_state_dict(state)
    _run_stage5_self_kd(tiny_model, cfg, tmp_path, monkeypatch)
    s2 = _latest_ckpt_step(tmp_path)
    assert s2 < s1 + patience, (
        f"resume must RESTORE no_improve_windows (was {v1} at step {s1}) — run 2 "
        f"ran to step {s2}; a counter reset to 0 on resume would re-train "
        f"~{patience} non-improving windows and reach step ~{s1 + patience}"
    )
