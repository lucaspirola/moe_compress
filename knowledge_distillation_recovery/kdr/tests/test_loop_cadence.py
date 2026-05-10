"""Tests for `kdr.training.loop` cadence dispatch (LLR-0049).

LLR-0049 AC #5: a 50-step run with `eval_every_n_steps=10` invokes eval
EXACTLY 5 times (steps 10, 20, 30, 40, 50) — NOT 6 (no step-0 baseline eval).

We don't run a real distillation loop here — that requires real models and
GPU. Instead we verify the cadence guard directly: simulate the optimizer-
step dispatch and count eval invocations.

# VERIFIES: LLR-0049
"""

from __future__ import annotations


def _eval_steps_for_run(total_steps: int, eval_every: int) -> list[int]:
    """Mirror the loop's eval-cadence guard from `_LoopState._commit_window`:

        if step > 0 and step % eval_every == 0: eval()

    Step 0 explicitly does NOT trigger eval (LLR-0049 AC #3) — the range
    starts at 0 here so the helper actually exercises the boundary, not at
    1 (which would mask a regression that fired on step 0).
    """
    fired: list[int] = []
    for step in range(0, total_steps + 1):
        if step > 0 and step % eval_every == 0:
            fired.append(step)
    return fired


def test_eval_cadence_50_steps_every_10_fires_5_times() -> None:
    """LLR-0049 AC #5 verbatim: 50 steps * eval_every=10 -> 5 eval calls
    at steps 10, 20, 30, 40, 50 — never 6 (no step-0)."""
    fired = _eval_steps_for_run(total_steps=50, eval_every=10)
    assert fired == [10, 20, 30, 40, 50]
    assert len(fired) == 5


def test_eval_cadence_skips_step_0() -> None:
    """Even with eval_every=1 the step-0 dispatch must not fire (a baseline
    eval is a separate concern, out of scope per LLR-0049)."""
    fired = _eval_steps_for_run(total_steps=3, eval_every=1)
    assert fired == [1, 2, 3]
    assert 0 not in fired


def test_eval_cadence_irregular() -> None:
    """7 steps, eval_every=3 → fires at 3 and 6 only."""
    fired = _eval_steps_for_run(total_steps=7, eval_every=3)
    assert fired == [3, 6]


def test_loop_dispatch_uses_step_modulo_guard() -> None:
    """Structural check: the loop's commit-window method contains the exact
    LLR-0049 dispatch guard so future refactors don't regress it."""
    import inspect

    from kdr.training import loop

    src = inspect.getsource(loop._LoopState._commit_window)
    # Required substrings for the eval-cadence dispatch guard.
    assert "self.step > 0" in src
    assert "self.step % self.dconf.eval_every_n_steps == 0" in src


def test_loop_save_cadence_zero_disables_partials() -> None:
    """save_every_n_steps=0 means no partial saves (final-only smoke runs).
    Verified by the loop's save guard: `if save_every > 0 and ...`"""
    import inspect

    from kdr.training import loop

    src = inspect.getsource(loop._LoopState._commit_window)
    assert "self.dconf.save_every_n_steps > 0" in src


# REQ: VERIFIES: LLR-0049
def test_real_loop_state_eval_fires_5_times_in_50_step_run() -> None:
    """LLR-0049 AC #5 (real version): construct an actual `_LoopState`,
    invoke `_commit_window` 50 times with patched `eval_run` / `save_partial`
    / accelerator collectives, and assert eval was invoked exactly 5 times
    at steps 10, 20, 30, 40, 50 — NOT 6 (no step-0 eval)."""
    from pathlib import Path
    from unittest.mock import MagicMock, patch

    import torch
    import torch.nn as nn

    from kdr.training import loop
    from kdr.training.loop import _LoopState

    # Build a `_LoopState` with the minimum that `_commit_window` touches.
    config = MagicMock()
    config.mode = "bf16"
    config.distillation = MagicMock()
    config.distillation.total_tokens = 50_000_000
    config.distillation.per_device_batch_size = 1
    config.distillation.gradient_accumulation = 1
    config.distillation.sequence_length = 1024
    config.distillation.warmup_steps = 5
    config.distillation.learning_rate = 1e-4
    config.distillation.min_learning_rate = 1e-6
    config.distillation.weight_decay = 0.0
    config.distillation.grad_clip_norm = 1.0
    config.distillation.log_every_n_steps = 1000  # silence
    config.distillation.eval_every_n_steps = 10
    config.distillation.save_every_n_steps = 0
    config.eval = MagicMock()

    accel = MagicMock()
    accel.num_processes = 1
    accel.is_main_process = True
    accel.device = torch.device("cpu")
    # `clip_grad_norm_` returns a tensor in real life; MagicMock default is fine.
    accel.clip_grad_norm_ = MagicMock()

    student = nn.Linear(4, 4)
    optim = torch.optim.SGD(student.parameters(), lr=1e-4)

    state = _LoopState(
        config=config,
        accelerator=accel,
        artifacts_dir=Path("/tmp/unused"),
        teacher=MagicMock(),
        student=student,
        tokenizer=MagicMock(),
        optim=optim,
        batches=[],
        resume_step=0,
        source_metadata_path=None,
    )

    eval_calls: list[int] = []

    def _capture_eval(*args: object, **kw: object) -> None:
        eval_calls.append(state.step)

    with patch.object(loop, "eval_run", side_effect=_capture_eval), patch.object(
        loop, "is_deepspeed", return_value=False
    ):
        # Drive 50 commit-window invocations. We intentionally skip the
        # forward/backward path (which would require real models on the
        # device) — `_commit_window` only needs the step counter and the
        # cadence-guard arithmetic to fire. `optim.SGD.step()` no-ops on
        # `grad is None` parameters, so the call sequence is safe.
        for _ in range(50):
            state._commit_window()

    assert eval_calls == [10, 20, 30, 40, 50], (
        f"Expected eval at steps [10,20,30,40,50]; got {eval_calls}."
    )
    assert len(eval_calls) == 5
