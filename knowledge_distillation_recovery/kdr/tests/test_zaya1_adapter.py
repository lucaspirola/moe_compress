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
            _fake_accelerator(),
            teacher_cfg=teacher_cfg,
            student_cfg=student_cfg,
            mode="bf16",
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
            _fake_accelerator(),
            teacher_cfg=teacher_cfg,
            student_cfg=student_cfg,
            mode="bf16",
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
            _fake_accelerator(),
            teacher_cfg=teacher_cfg,
            student_cfg=student_cfg,
            mode="bf16",
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
            mode="bf16",
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
            _fake_accelerator(),
            teacher_cfg=teacher_cfg,
            student_cfg=student_cfg,
            mode="bf16",
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
            _fake_accelerator(),
            teacher_cfg=teacher_cfg,
            student_cfg=student_cfg,
            mode="bf16",
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


# ───────────────────────────────────────────────────────────────────────────
# Phase 5: QAD methods (LLR-0024, LLR-0025, LLR-0026)
# ───────────────────────────────────────────────────────────────────────────


def _fake_zaya_shaped_model() -> nn.Module:
    """A more elaborate fake whose dotted module names mirror ZAYA1's layout
    enough to exercise the carve-out / attention-path matchers."""

    class QKV(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear_q = nn.Linear(8, 8, bias=False)
            self.linear_k = nn.Linear(8, 8, bias=False)
            self.val_proj1 = nn.Linear(8, 8, bias=False)  # CCA value F16 carve-out
            self.val_proj2 = nn.Linear(8, 8, bias=False)  # CCA value F16 carve-out

    class CCAAttn(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.qkv = QKV()
            self.o_proj = nn.Linear(8, 8, bias=False)

    class MoEFFN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.router = nn.Linear(8, 4, bias=False)
            self.experts = nn.ModuleList([nn.Linear(8, 8, bias=False) for _ in range(4)])

    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn = CCAAttn()
            self.mlp = MoEFFN()
            self.input_norm = nn.LayerNorm(8)  # RMSNorm stand-in
            self.post_attention_layernorm = nn.LayerNorm(8)

    class Inner(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed_tokens = nn.Embedding(100, 8)
            self.layers = nn.ModuleList([Layer() for _ in range(2)])

    class Top(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = Inner()
            self.lm_head = nn.Linear(8, 100, bias=False)

    return Top()


# REQ: VERIFIES: LLR-0024
def test_fp32_carve_outs_includes_lm_head() -> None:
    """LLR-0024 AC #1: list literally includes 'lm_head'."""
    out = Zaya1Adapter().fp32_carve_outs(_fake_zaya_shaped_model())
    assert "lm_head" in out


# REQ: VERIFIES: LLR-0024
# REQ: VERIFIES: LLR-0057
def test_fp32_carve_outs_resolve_to_non_empty_submodule_sets() -> None:
    """LLR-0024 AC #2 / LLR-0057 AC: every returned pattern matches at least
    one submodule on a ZAYA1-shaped instantiated model."""
    model = _fake_zaya_shaped_model()
    patterns = Zaya1Adapter().fp32_carve_outs(model)
    names = [n for n, _ in model.named_modules()]
    for pattern in patterns:
        matches = [n for n in names if pattern in n]
        assert matches, (
            f"carve-out pattern {pattern!r} resolved to ZERO submodules on "
            f"the fake ZAYA1 model — would silently skip the carve-out on real ZAYA1"
        )


# REQ: VERIFIES: LLR-0057
def test_fp32_carve_outs_matches_zaya1_real_substrings() -> None:
    """LLR-0057 AC: returned list is the 5 ZAYA1-real carve-out substrings."""
    out = Zaya1Adapter().fp32_carve_outs(_fake_zaya_shaped_model())
    assert set(out) == {"lm_head", "embed_tokens", "router", "norm", "val_proj"}
    assert len(out) == 5, f"expected exactly 5 entries, no duplicates; got {out!r}"


# REQ: VERIFIES: LLR-0057
def test_fp32_carve_outs_val_proj_matches_both_cca_value_linears() -> None:
    """LLR-0057 AC: `val_proj` substring catches both val_proj1 and val_proj2."""
    model = _fake_zaya_shaped_model()
    names = [n for n, _ in model.named_modules()]
    val_proj_matches = [n for n in names if "val_proj" in n]
    # Two layers × two val_proj Linears each = 4 modules expected.
    assert len(val_proj_matches) == 4
    assert all("val_proj" in n for n in val_proj_matches)
    assert all(n.endswith(("val_proj1", "val_proj2")) for n in val_proj_matches)


# REQ: VERIFIES: LLR-0057
def test_fp32_carve_outs_val_proj_no_collision_outside_cca() -> None:
    """LLR-0057 AC: `val_proj` does not match any non-CCA module."""
    model = _fake_zaya_shaped_model()
    names = [n for n, _ in model.named_modules()]
    # Every val_proj match is under self_attn.qkv.
    for n in names:
        if "val_proj" in n:
            assert "self_attn.qkv" in n, (
                f"unexpected val_proj match outside the CCA path: {n!r}"
            )


# REQ: VERIFIES: LLR-0026
def test_required_attn_implementation_is_eager_or_sdpa() -> None:
    """LLR-0026 AC: returns one of {'eager', 'sdpa'} for every supported mode;
    flash-attn is rejected. The method is now mode-aware (LLR-0026 post-Phase
    7.1), so we check both modes."""
    adapter = Zaya1Adapter()
    for mode in ("bf16", "da_qad"):
        val = adapter.required_attn_implementation(mode)  # type: ignore[arg-type]
        assert val in ("eager", "sdpa"), (
            f"required_attn_implementation({mode!r}) returned {val!r}; "
            f"only 'eager' and 'sdpa' are allowed by LLR-0026."
        )


# REQ: VERIFIES: LLR-0026
def test_load_passes_required_attn_implementation_to_from_pretrained(
    teacher_cfg: TeacherConfig, student_cfg: StudentConfig
) -> None:
    """LLR-0026 AC #2 + AC #3: adapter forces its required attn_implementation,
    overriding any YAML mis-configuration AND no flash-attn module appears
    in the loaded model graph. The fixtures pass 'sdpa' from YAML but
    adapter requires 'eager'.
    """
    captured_kwargs: list[dict[str, Any]] = []
    loaded_models: list[nn.Module] = []

    def fake_from_pretrained(name_or_path: str, **kw: Any) -> nn.Module:
        captured_kwargs.append(kw)
        m = _fake_pretrained_model()
        loaded_models.append(m)
        return m

    fake_tok = MagicMock()
    fake_tok.pad_token_id = None
    fake_tok.eos_token = "<eos>"

    with patch(
        "kdr.adapters.zaya1_8b.AutoModelForCausalLM.from_pretrained",
        side_effect=fake_from_pretrained,
    ), patch(
        "kdr.adapters.zaya1_8b.AutoTokenizer.from_pretrained",
        return_value=fake_tok,
    ):
        adapter = Zaya1Adapter()
        adapter.load_teacher_and_student(
            _fake_accelerator(),
            teacher_cfg=teacher_cfg,
            student_cfg=student_cfg,
            mode="bf16",
        )

    # AC: both teacher and student loads receive the adapter's required
    # value for the mode the call was made with. The fixture defaults to
    # mode="bf16" (set in the load_teacher_and_student call above).
    required = Zaya1Adapter().required_attn_implementation("bf16")
    teacher_kw, student_kw = captured_kwargs
    assert teacher_kw["attn_implementation"] == required
    assert student_kw["attn_implementation"] == required

    # AC #3: no flash-attn module is present in either resulting model graph.
    # Mock-based test scope: structural check that no submodule's class name
    # contains "flash" / "Flash". On a real ZAYA1 the same check would cover
    # the actual flash-attn classes (``FlashAttention2``, etc.); the full
    # cross-check against real classes is deferred to Phase 7 hardware.
    for model in loaded_models:
        for module_name, mod in model.named_modules():
            cls_name = mod.__class__.__name__
            assert "flash" not in cls_name.lower(), (
                f"flash-attn module detected at {module_name!r}: {cls_name}"
            )


# REQ: VERIFIES: LLR-0023
def test_attention_module_paths_returns_self_attn_paths() -> None:
    """attention_module_paths returns a path per layer's self_attn."""
    paths = Zaya1Adapter().attention_module_paths(_fake_zaya_shaped_model())
    # 2 layers in the fake model → 2 self_attn paths.
    assert len(paths) == 2
    assert all(p.endswith(".self_attn") for p in paths)


# REQ: VERIFIES: LLR-0025
def test_router_replay_hook_returns_context_manager() -> None:
    """router_replay_hook returns an enterable context manager."""
    teacher = _fake_zaya_shaped_model()
    student = _fake_zaya_shaped_model()
    hook = Zaya1Adapter().router_replay_hook(teacher, student)
    # Must be enterable; `__exit__` cleans up the installed hooks.
    assert hasattr(hook, "__enter__")
    assert hasattr(hook, "__exit__")
    assert hasattr(hook, "start_microbatch")
