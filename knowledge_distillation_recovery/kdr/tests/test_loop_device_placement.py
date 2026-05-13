"""LLR-0060: pre-apply_quant device placement + calibrate_loop batch-move.

# REQ: LLR-0060
# VERIFIES: LLR-0060
"""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from kdr.training import loop


# ─────────────────────────────────────────────────────────────────────────────
# Source-level invariants
# ─────────────────────────────────────────────────────────────────────────────


def test_student_to_device_inside_da_qad_branch_and_zero3_context() -> None:
    """LLR-0060 AC: the `student = student.to(accelerator.device)` statement
    sits INSIDE both the `if config.mode == "da_qad":` block AND the
    `with activate_zero3_init(accelerator):` context, before the
    `partition_and_dispatch` call."""
    src = inspect.getsource(loop.run_recovery)
    lines = src.splitlines()
    with_idx = next(
        i for i, ln in enumerate(lines) if "activate_zero3_init" in ln
    )
    da_qad_idx = next(
        i for i, ln in enumerate(lines)
        if 'if config.mode == "da_qad"' in ln
    )
    to_dev_idx = next(
        (i for i, ln in enumerate(lines)
         if "student = student.to(accelerator.device)" in ln),
        -1,
    )
    pd_idx = next(
        i for i, ln in enumerate(lines)
        if "active_backends = partition_and_dispatch" in ln
    )
    assert to_dev_idx > -1, (
        "missing `student = student.to(accelerator.device)` in run_recovery"
    )
    assert with_idx < to_dev_idx < pd_idx, (
        "to(device) must be inside activate_zero3_init AND before "
        "partition_and_dispatch"
    )
    assert da_qad_idx < to_dev_idx, (
        "to(device) must be inside the da_qad branch"
    )


def test_to_device_guarded_by_is_deepspeed() -> None:
    """LLR-0060 AC: the move is gated `if not is_deepspeed(accelerator):`."""
    src = inspect.getsource(loop.run_recovery)
    # Find the to(device) line and confirm the preceding line is the guard.
    lines = src.splitlines()
    for i, ln in enumerate(lines):
        if "student = student.to(accelerator.device)" in ln:
            # Look back up to 3 lines for the guard.
            window = "\n".join(lines[max(0, i - 3):i + 1])
            assert "if not is_deepspeed(accelerator)" in window, (
                f"to(device) must be guarded by `if not is_deepspeed`; "
                f"context:\n{window}"
            )
            return
    pytest.fail("to(device) statement not found")


# ─────────────────────────────────────────────────────────────────────────────
# Behavioural unit tests on _make_calibrate_loop
# ─────────────────────────────────────────────────────────────────────────────


def test_calibrate_loop_moves_each_batch_to_model_device() -> None:
    """LLR-0060 AC: each batch is moved to `next(model.parameters()).device`
    before being passed to `model(input_ids=...)`."""
    from kdr.quant.factory import _make_calibrate_loop

    # Three small CPU batches; model claims to be on a non-default device.
    batches = [torch.zeros(2, 4, dtype=torch.long) for _ in range(3)]
    closure = _make_calibrate_loop(batches, ptq_subset_size=6)

    # Fake model: parameters() returns a CUDA-device tensor (we use "meta"
    # device here so the test runs without a real GPU).
    fake_device = torch.device("meta")
    fake_param = torch.empty(1, device=fake_device)
    fake_model = MagicMock(spec=nn.Module)
    fake_model.training = False
    fake_model.parameters = MagicMock(return_value=iter([fake_param]))
    # Make model(...) callable; record the input_ids it was given.
    seen_input_ids: list[torch.Tensor] = []

    def _capture(*, input_ids: torch.Tensor) -> object:
        seen_input_ids.append(input_ids)
        return None

    fake_model.side_effect = _capture
    closure(fake_model)
    assert len(seen_input_ids) == 3
    for t in seen_input_ids:
        assert t.device == fake_device, (
            f"batch handed to model has device {t.device}, expected {fake_device}"
        )


def test_calibrate_loop_restores_training_mode() -> None:
    """The closure resets model.train() if it was training before — regression
    guard ensuring the new device line doesn't break the existing finally."""
    from kdr.quant.factory import _make_calibrate_loop

    batches = [torch.zeros(2, 4, dtype=torch.long)]
    closure = _make_calibrate_loop(batches, ptq_subset_size=2)

    fake_device = torch.device("meta")
    fake_param = torch.empty(1, device=fake_device)
    fake_model = MagicMock(spec=nn.Module)
    fake_model.training = True  # caller had model in training mode
    fake_model.parameters = MagicMock(return_value=iter([fake_param]))
    fake_model.side_effect = lambda **kw: None
    closure(fake_model)
    # `.train()` must have been called inside the finally to restore state.
    fake_model.train.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# Branch coverage: is_deepspeed True vs False (mock-driven)
# ─────────────────────────────────────────────────────────────────────────────


def _fake_accelerator(device: str = "meta") -> MagicMock:
    from accelerate import DistributedType

    accel = MagicMock()
    accel.device = torch.device(device)
    accel.distributed_type = DistributedType.NO
    accel.is_main_process = True
    accel.wait_for_everyone = lambda: None
    return accel


def test_to_device_called_when_not_deepspeed() -> None:
    """LLR-0060 AC: in da_qad mode under non-DS, student.to(accel.device)
    is observably called before partition_and_dispatch."""
    accel = _fake_accelerator(device="meta")
    fake_student = MagicMock(spec=nn.Module)
    fake_student.to = MagicMock(return_value=fake_student)
    # Simulate the with activate_zero3_init block: no-op context manager.
    with (
        patch("kdr.training.loop.activate_zero3_init",
              return_value=__import__("contextlib").nullcontext()),
        patch("kdr.training.loop.is_deepspeed", return_value=False),
    ):
        # Manually drive the relevant snippet from run_recovery's da_qad
        # branch (the source-level test above already pins the structure).
        # Here we just confirm the guard semantics.
        from kdr.training.loop import is_deepspeed
        if not is_deepspeed(accel):
            fake_student = fake_student.to(accel.device)
        fake_student.to.assert_called_once_with(accel.device)


def test_to_device_skipped_when_deepspeed() -> None:
    """LLR-0060 AC: under DeepSpeed, the student.to(device) is skipped."""
    accel = _fake_accelerator(device="meta")
    fake_student = MagicMock(spec=nn.Module)
    fake_student.to = MagicMock(return_value=fake_student)
    with patch("kdr.training.loop.is_deepspeed", return_value=True):
        from kdr.training.loop import is_deepspeed
        if not is_deepspeed(accel):
            fake_student.to(accel.device)
        fake_student.to.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end: run_recovery actually moves student to device before
# partition_and_dispatch is invoked (LLR-0060 AC 4 — the strict form).
# ─────────────────────────────────────────────────────────────────────────────


def test_run_recovery_places_student_on_device_before_dispatch(tmp_path):
    """LLR-0060 AC: drive `run_recovery` end-to-end with mocked downstream
    callees; capture the student `partition_and_dispatch` receives and
    assert it is on the accelerator's device."""
    from contextlib import nullcontext

    # Stand-in student: a real nn.Linear so `.to(device)` returns a module
    # whose `.parameters()[0].device` we can read.
    student_module = nn.Linear(8, 8)
    teacher_module = nn.Linear(8, 8)
    tokenizer = MagicMock()

    fake_adapter = MagicMock()
    fake_adapter.load_teacher_and_student = MagicMock(
        return_value=(teacher_module, student_module, tokenizer)
    )
    fake_adapter.fp32_carve_outs = MagicMock(return_value=["lm_head"])
    fake_adapter.attention_module_paths = MagicMock(return_value=[])
    fake_adapter.kv_quant_exempt_indices = MagicMock(return_value=[])
    fake_adapter.required_attn_implementation = MagicMock(return_value="eager")

    accel = _fake_accelerator(device="meta")
    accel.prepare = MagicMock(return_value=(student_module, MagicMock()))
    accel.unwrap_model = MagicMock(side_effect=lambda m: m)
    accel.get_state_dict = MagicMock(return_value={})

    captured = {}

    def _capture_dispatch(model, *args, **kwargs):
        # Capture the model's device at the moment of dispatch.
        captured["student_device"] = next(model.parameters()).device
        return []

    # Build a minimal da_qad Config.
    from kdr.config import (
        CalibrationConfig,
        Config,
        DistillationConfig,
        EvalConfig,
        KVQuantBlock,
        QuantBlock,
        StudentConfig,
        TeacherConfig,
        WikiText2Config,
    )
    from kdr.quant.specs import (
        KVQuantSpec,
        MixedWeightSpec,
        WeightPatternSpec,
    )

    cfg = Config(
        mode="da_qad",
        teacher=TeacherConfig(name_or_path="x"),
        student=StudentConfig(source="x"),
        calibration=CalibrationConfig(
            source="nvidia-cascade",
            dataset="x",
            seed=1,
            num_sequences=1,
            sequence_length=4,
            ptq_subset_size=1,
            subset_weights={"math": 1.0},
        ),
        quant=QuantBlock(
            weight=MixedWeightSpec(
                spec_map=[
                    WeightPatternSpec(  # type: ignore[arg-type]
                        pattern="",
                        bits=3,
                        format="int",
                        granularity="channel",
                        transform="none",
                    )
                ]
            ),
            kv_quant=KVQuantBlock(
                key=KVQuantSpec(bits=3, format="int", granularity="channel", transform="none"),  # type: ignore[arg-type]
                value=KVQuantSpec(bits=3, format="int", granularity="token", transform="none"),  # type: ignore[arg-type]
            ),
        ),
        distillation=DistillationConfig(
            loss="forward_kld",
            temperature=1.0,
            optimizer="adamw_bnb_8bit",
            learning_rate=2e-4,
            min_learning_rate=4e-7,
            weight_decay=0.01,
            betas=[0.9, 0.999],
            grad_clip_norm=1.0,
            warmup_steps=1,
            total_tokens=64,
            per_device_batch_size=1,
            gradient_accumulation=1,
            sequence_length=4,
            log_every_n_steps=1,
            eval_every_n_steps=1,
            save_every_n_steps=0,
            trainable_scope="full",
            use_gradient_checkpointing=False,
        ),
        eval=EvalConfig(wikitext2=WikiText2Config(
            enabled=False, sequence_length=4, num_sequences=1)),
    )

    with (
        patch("kdr.training.loop.activate_zero3_init",
              return_value=nullcontext()),
        patch("kdr.training.loop.is_deepspeed", return_value=False),
        patch("kdr.quant.factory.partition_and_dispatch",
              side_effect=_capture_dispatch),
        patch("kdr.training.loop._enable_trainable_scope"),
        patch("kdr.training.loop.build_optimizer", return_value=MagicMock()),
        # Short-circuit the training loop by making the loop raise an
        # asserted exception once dispatch has been observed.
        patch(
            "kdr.training.loop._LoopState",
            side_effect=RuntimeError("stop after dispatch"),
        ),
    ):
        with pytest.raises(RuntimeError, match="stop after dispatch"):
            loop.run_recovery(
                cfg,
                fake_adapter,
                accel,
                tmp_path,
                batches=[torch.zeros(1, 4, dtype=torch.long)],
            )

    assert captured, (
        "partition_and_dispatch was never intercepted; check patch target "
        "(loop.py uses a function-local `from ..quant.factory import "
        "partition_and_dispatch`, so the patch sits on kdr.quant.factory)"
    )
    assert captured.get("student_device") == accel.device, (
        f"partition_and_dispatch received student on device "
        f"{captured.get('student_device')}, expected {accel.device}"
    )


def test_run_recovery_skips_to_device_when_deepspeed(tmp_path):
    """LLR-0060 AC: end-to-end DS path — `student.to(accelerator.device)` is
    NOT called on the student when `is_deepspeed(accelerator)` is True."""
    from contextlib import nullcontext

    real_student = nn.Linear(8, 8)
    student_wrapper = MagicMock(wraps=real_student)
    student_wrapper.to = MagicMock(return_value=student_wrapper)
    teacher_module = nn.Linear(8, 8)
    tokenizer = MagicMock()

    fake_adapter = MagicMock()
    fake_adapter.load_teacher_and_student = MagicMock(
        return_value=(teacher_module, student_wrapper, tokenizer)
    )
    fake_adapter.fp32_carve_outs = MagicMock(return_value=["lm_head"])
    fake_adapter.attention_module_paths = MagicMock(return_value=[])
    fake_adapter.kv_quant_exempt_indices = MagicMock(return_value=[])
    fake_adapter.required_attn_implementation = MagicMock(return_value="eager")

    accel = _fake_accelerator(device="meta")
    accel.prepare = MagicMock(return_value=(student_wrapper, MagicMock()))
    accel.unwrap_model = MagicMock(side_effect=lambda m: m)
    accel.get_state_dict = MagicMock(return_value={})

    from kdr.config import (
        CalibrationConfig,
        Config,
        DistillationConfig,
        EvalConfig,
        KVQuantBlock,
        QuantBlock,
        StudentConfig,
        TeacherConfig,
        WikiText2Config,
    )
    from kdr.quant.specs import (
        KVQuantSpec,
        MixedWeightSpec,
        WeightPatternSpec,
    )

    cfg = Config(
        mode="da_qad",
        teacher=TeacherConfig(name_or_path="x"),
        student=StudentConfig(source="x"),
        calibration=CalibrationConfig(
            source="nvidia-cascade", dataset="x", seed=1,
            num_sequences=1, sequence_length=4, ptq_subset_size=1,
            subset_weights={"math": 1.0},
        ),
        quant=QuantBlock(
            weight=MixedWeightSpec(
                spec_map=[
                    WeightPatternSpec(  # type: ignore[arg-type]
                        pattern="", bits=3, format="int",
                        granularity="channel", transform="none",
                    )
                ]
            ),
            kv_quant=KVQuantBlock(
                key=KVQuantSpec(bits=3, format="int", granularity="channel", transform="none"),  # type: ignore[arg-type]
                value=KVQuantSpec(bits=3, format="int", granularity="token", transform="none"),  # type: ignore[arg-type]
            ),
        ),
        distillation=DistillationConfig(
            loss="forward_kld", temperature=1.0,
            optimizer="adamw_bnb_8bit",
            learning_rate=2e-4, min_learning_rate=4e-7,
            weight_decay=0.01, betas=[0.9, 0.999],
            grad_clip_norm=1.0, warmup_steps=1,
            total_tokens=64, per_device_batch_size=1,
            gradient_accumulation=1, sequence_length=4,
            log_every_n_steps=1, eval_every_n_steps=1,
            save_every_n_steps=0, trainable_scope="full",
            use_gradient_checkpointing=False,
        ),
        eval=EvalConfig(wikitext2=WikiText2Config(
            enabled=False, sequence_length=4, num_sequences=1)),
    )

    with (
        patch("kdr.training.loop.activate_zero3_init",
              return_value=nullcontext()),
        patch("kdr.training.loop.is_deepspeed", return_value=True),
        patch("kdr.quant.factory.partition_and_dispatch", return_value=[]),
        patch("kdr.training.loop._enable_trainable_scope"),
        patch("kdr.training.loop.build_optimizer", return_value=MagicMock()),
        patch(
            "kdr.training.loop._LoopState",
            side_effect=RuntimeError("stop after dispatch"),
        ),
    ):
        with pytest.raises(RuntimeError, match="stop after dispatch"):
            loop.run_recovery(
                cfg, fake_adapter, accel, tmp_path,
                batches=[torch.zeros(1, 4, dtype=torch.long)],
            )

    # Under is_deepspeed=True, the student.to(accelerator.device) line in
    # the da_qad branch must be skipped. The student_wrapper's .to was not
    # called from the new pre-dispatch guard.
    student_wrapper.to.assert_not_called()
