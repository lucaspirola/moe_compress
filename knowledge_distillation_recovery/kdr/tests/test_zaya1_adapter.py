"""Tests for `kdr.adapters.zaya1_8b.Zaya1Adapter`.

We mock `from_pretrained` so the test doesn't download any model. The
goal is to verify the adapter's contract surface: load order, freezing,
FP8 lm_head carve-out, layer-count probe.

# VERIFIES: LLR-0023
# VERIFIES: LLR-0004
# VERIFIES: LLR-0005
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from kdr.adapters.zaya1_8b import Zaya1Adapter
from kdr.config import StudentConfig, TeacherConfig


def _fake_pretrained_model(*, dtype: torch.dtype = torch.bfloat16) -> nn.Module:
    """A tiny model that satisfies the adapter's expectations: parameters
    that can be frozen + a `lm_head` linear + a `model.layers` list."""

    class Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            inner = nn.Module()
            inner.layers = nn.ModuleList(  # type: ignore[assignment]
                [nn.Linear(4, 4) for _ in range(40)]
            )
            self.model = inner
            self.lm_head = nn.Linear(4, 17, bias=False)
            self.lm_head.weight.data = self.lm_head.weight.data.to(dtype)

    return Tiny()


def _fake_accelerator() -> MagicMock:
    a = MagicMock()
    a.is_main_process = True
    return a


@pytest.fixture
def teacher_cfg() -> TeacherConfig:
    return TeacherConfig(
        name_or_path="Zyphra/ZAYA1-reasoning-base",
        revision="main",
        torch_dtype="bfloat16",
        attn_implementation="sdpa",
    )


@pytest.fixture
def student_cfg() -> StudentConfig:
    return StudentConfig(
        source="Zyphra/ZAYA1-reasoning-base",
        torch_dtype="bfloat16",
        attn_implementation="sdpa",
    )


# REQ: VERIFIES: LLR-0023
def test_load_teacher_first_then_student(
    teacher_cfg: TeacherConfig, student_cfg: StudentConfig
) -> None:
    """Adapter must load teacher BEFORE student. We capture the call order
    and assert it."""
    call_order: list[str] = []

    def fake_from_pretrained(name_or_path: str, **kw: Any) -> nn.Module:
        # The first call is the teacher; second is the student.
        call_order.append(name_or_path)
        return _fake_pretrained_model()

    fake_tok = MagicMock()
    fake_tok.pad_token_id = None
    fake_tok.eos_token = "<eos>"
    fake_tok.pad_token = None

    with patch(
        "kdr.adapters.zaya1_8b.AutoModelForCausalLM.from_pretrained",
        side_effect=fake_from_pretrained,
    ), patch(
        "kdr.adapters.zaya1_8b.AutoTokenizer.from_pretrained",
        return_value=fake_tok,
    ):
        teacher, student, tok = Zaya1Adapter().load_teacher_and_student(
            _fake_accelerator(), teacher_cfg=teacher_cfg, student_cfg=student_cfg
        )
    assert len(call_order) == 2
    # Both come from same self-distillation source by default.
    assert call_order[0] == teacher_cfg.name_or_path
    assert call_order[1] == student_cfg.source
    assert teacher is not student
    assert tok is fake_tok


# REQ: VERIFIES: LLR-0004
def test_teacher_frozen_after_load(
    teacher_cfg: TeacherConfig, student_cfg: StudentConfig
) -> None:
    """LLR-0004 AC: every teacher param has requires_grad=False after load."""
    fake_tok = MagicMock()
    fake_tok.pad_token_id = None
    fake_tok.eos_token = "<eos>"

    with patch(
        "kdr.adapters.zaya1_8b.AutoModelForCausalLM.from_pretrained",
        side_effect=lambda *a, **k: _fake_pretrained_model(),
    ), patch(
        "kdr.adapters.zaya1_8b.AutoTokenizer.from_pretrained",
        return_value=fake_tok,
    ):
        teacher, _, _ = Zaya1Adapter().load_teacher_and_student(
            _fake_accelerator(), teacher_cfg=teacher_cfg, student_cfg=student_cfg
        )
    assert all(not p.requires_grad for p in teacher.parameters())


# REQ: VERIFIES: LLR-0004
def test_teacher_no_grad_after_backward(
    teacher_cfg: TeacherConfig, student_cfg: StudentConfig
) -> None:
    """LLR-0004 AC: a backward pass leaves teacher params with `grad is None`."""
    fake_tok = MagicMock()
    fake_tok.pad_token_id = None
    fake_tok.eos_token = "<eos>"

    with patch(
        "kdr.adapters.zaya1_8b.AutoModelForCausalLM.from_pretrained",
        side_effect=lambda *a, **k: _fake_pretrained_model(),
    ), patch(
        "kdr.adapters.zaya1_8b.AutoTokenizer.from_pretrained",
        return_value=fake_tok,
    ):
        teacher, student, _ = Zaya1Adapter().load_teacher_and_student(
            _fake_accelerator(), teacher_cfg=teacher_cfg, student_cfg=student_cfg
        )
    # Backward through the student; gradients should NOT touch the teacher.
    # Use the student's lm_head dtype to avoid matmul dtype mismatch.
    head_dtype = student.lm_head.weight.dtype
    s_loss = student.lm_head(torch.randn(4, dtype=head_dtype)).sum()
    s_loss.backward()
    for p in teacher.parameters():
        assert p.grad is None


# REQ: VERIFIES: LLR-0005
def test_lm_head_cast_to_bf16_for_fp8_teacher(student_cfg: StudentConfig) -> None:
    """LLR-0005 AC: when teacher dtype is fp8_e4m3fn, lm_head is BF16."""
    fp8_teacher_cfg = TeacherConfig(
        name_or_path="Zyphra/ZAYA1-reasoning-base",
        revision="main",
        torch_dtype="float8_e4m3fn",
        attn_implementation="sdpa",
    )

    fake_tok = MagicMock()
    fake_tok.pad_token_id = None
    fake_tok.eos_token = "<eos>"

    with patch(
        "kdr.adapters.zaya1_8b.AutoModelForCausalLM.from_pretrained",
        side_effect=lambda *a, **k: _fake_pretrained_model(dtype=torch.float32),
    ), patch(
        "kdr.adapters.zaya1_8b.AutoTokenizer.from_pretrained",
        return_value=fake_tok,
    ):
        teacher, _, _ = Zaya1Adapter().load_teacher_and_student(
            _fake_accelerator(),
            teacher_cfg=fp8_teacher_cfg,
            student_cfg=student_cfg,
        )
    assert teacher.lm_head.weight.dtype == torch.bfloat16


# REQ: VERIFIES: LLR-0005
def test_lm_head_unchanged_for_bf16_teacher(
    teacher_cfg: TeacherConfig, student_cfg: StudentConfig
) -> None:
    """The BF16 carve-out only fires for FP8 teachers — float16 / bf16
    teachers' lm_head is NOT touched."""
    fake_tok = MagicMock()
    fake_tok.pad_token_id = None
    fake_tok.eos_token = "<eos>"

    with patch(
        "kdr.adapters.zaya1_8b.AutoModelForCausalLM.from_pretrained",
        side_effect=lambda *a, **k: _fake_pretrained_model(dtype=torch.float16),
    ), patch(
        "kdr.adapters.zaya1_8b.AutoTokenizer.from_pretrained",
        return_value=fake_tok,
    ):
        teacher, _, _ = Zaya1Adapter().load_teacher_and_student(
            _fake_accelerator(), teacher_cfg=teacher_cfg, student_cfg=student_cfg
        )
    # bf16 teacher: lm_head dtype == float16 (unchanged from our fake).
    assert teacher.lm_head.weight.dtype == torch.float16


# REQ: VERIFIES: LLR-0023
def test_kv_quant_exempt_indices_empty_for_zaya1(
    teacher_cfg: TeacherConfig,
) -> None:
    """ZAYA1 has no SSM layers; the exempt list is empty regardless of the
    model passed in (Phase 5 may revisit if the architecture surveying says
    otherwise)."""
    adapter = Zaya1Adapter()
    assert adapter.kv_quant_exempt_indices(MagicMock()) == []


# REQ: VERIFIES: LLR-0023
def test_load_logs_layer_count_in_expected_set(
    teacher_cfg: TeacherConfig, student_cfg: StudentConfig
) -> None:
    """LLR-0023 AC: 'asserts the value is in {40, 80}'. The fake model has
    40 layers; the adapter must log that count via its `model.model.layers`
    probe so downstream LLRs can consume the empirical figure."""
    fake_tok = MagicMock()
    fake_tok.pad_token_id = None
    fake_tok.eos_token = "<eos>"

    with patch(
        "kdr.adapters.zaya1_8b.AutoModelForCausalLM.from_pretrained",
        side_effect=lambda *a, **k: _fake_pretrained_model(),
    ), patch(
        "kdr.adapters.zaya1_8b.AutoTokenizer.from_pretrained",
        return_value=fake_tok,
    ), patch("kdr.adapters.zaya1_8b.log") as mock_log:
        Zaya1Adapter().load_teacher_and_student(
            _fake_accelerator(), teacher_cfg=teacher_cfg, student_cfg=student_cfg
        )
        # Find the layer-count log message among the info() calls.
        layer_count_messages = [
            call
            for call in mock_log.info.call_args_list
            if "model.model.layers count" in str(call)
        ]
        assert len(layer_count_messages) == 1, (
            f"Expected exactly one layer-count log; got {layer_count_messages}."
        )
        # The logged count comes from the fake model's 40-layer ModuleList.
        # Per LLR-0023 the value must be in {40, 80}.
        # `call.args` is `(format_str, *positional_substitutions)`; the
        # layer count is the first substitution after the format string.
        msg_args = layer_count_messages[0].args
        logged_count = msg_args[1]
        assert logged_count in {40, 80}, (
            f"LLR-0023 requires layer count in {{40, 80}}; got {logged_count}."
        )
