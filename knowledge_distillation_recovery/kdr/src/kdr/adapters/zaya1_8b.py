"""ZAYA1-8B adapter (LLR-0023, LLR-0024, LLR-0025, LLR-0026).

Architecture properties (cross-checked against arxiv:2605.05365 §II-A):

- 40 transformer layers, all CCA + MoE FFN. No SSM, no MoD.
- 16 experts, top-1 routing. 760M active / 8.4B total.
- CCA = Compressed Convolutional Attention with CCGQA, 2x query / 8x KV
  compression.
- EDA = router-only Exponential Depth Averaging (NOT a layer block).
- Custom architecture (`model_type: "zaya"`); requires Zyphra's
  transformers fork (`Zyphra/transformers@zaya1`) or `trust_remote_code=True`.

Phase 3b lands `load_teacher_and_student` (the only adapter method the
BF16 loop needs); Phase 5 lands the QAD-mode methods (`fp32_carve_outs`,
`router_replay_hook`, etc.) once an actual ZAYA1 hookable forward is
available.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal, cast

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from collections.abc import Sized
from accelerate import Accelerator
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)

from ..config import StudentConfig, TeacherConfig

log = logging.getLogger(__name__)


class Zaya1Adapter:
    """`ModelAdapter` Protocol implementation for ZAYA1-8B."""

    name: str = "zaya1_8b"

    # REQ: LLR-0023
    def load_teacher_and_student(
        self,
        accelerator: Accelerator,
        *,
        teacher_cfg: TeacherConfig,
        student_cfg: StudentConfig,
    ) -> tuple[nn.Module, nn.Module, PreTrainedTokenizerBase]:
        """Load (teacher, student, tokenizer).

        Order is teacher → student. Under ZeRO-3 the training loop has
        opened `activate_zero3_init` before calling this method (LLR-0048
        steps 2 and 3); the adapter itself is context-unaware. Both
        `from_pretrained` calls happen inside that already-active context.

        The teacher is frozen post-load (LLR-0004). When the teacher's
        dtype is FP8 (`float8_e4m3fn`), the lm_head linear is cast back to
        BF16 (LLR-0005) — FP8 lm_head produces unstable softmax outputs at
        the vocab tail; BF16 lm_head with FP8 body weights is the documented
        stable configuration on H200.
        """
        teacher = self._load_one(
            name_or_path=teacher_cfg.name_or_path,
            revision=teacher_cfg.revision,
            torch_dtype=teacher_cfg.torch_dtype,
            attn_implementation=teacher_cfg.attn_implementation,
            role="TEACHER",
        )
        # REQ: LLR-0004
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        # Defensive post-condition.
        if any(p.requires_grad for p in teacher.parameters()):
            raise RuntimeError(
                "Zaya1Adapter: teacher has requires_grad=True params after "
                "freeze — gradient flow into the teacher would break the "
                "soft-target contract."
            )

        # REQ: LLR-0005
        if teacher_cfg.torch_dtype == "float8_e4m3fn":
            self._cast_lm_head_to_bf16(teacher)

        student = self._load_one(
            name_or_path=student_cfg.source,
            revision="main",
            torch_dtype=student_cfg.torch_dtype,
            attn_implementation=student_cfg.attn_implementation,
            role="STUDENT",
        )

        tokenizer = AutoTokenizer.from_pretrained(
            student_cfg.source, trust_remote_code=True
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Operative layer count: paper Table I says 40, HF config says 80
        # depending on sub-block convention. Log the empirical count so
        # downstream review can confirm it matches what the model loader
        # actually built.
        if accelerator.is_main_process:
            n_layers: int | None = None
            try:
                inner = student.model
                layers = getattr(inner, "layers", None)
                if layers is not None:
                    # `layers` is an `nn.ModuleList`; `len(...)` is supported
                    # but mypy types it as `Any | Tensor | Module` because
                    # `model` came back from `AutoModelForCausalLM` typed
                    # loosely. Cast to `Sized` so `len()` typechecks.
                    n_layers = len(cast("Sized", layers))
            except AttributeError:
                n_layers = None
            log.info(
                "Zaya1Adapter: model.model.layers count = %s (paper Table I "
                "says 40 transformer layers; HF config may say 80).",
                n_layers,
            )

        return teacher, student, tokenizer

    @staticmethod
    def _load_one(
        *,
        name_or_path: str,
        revision: str,
        torch_dtype: str,
        attn_implementation: str,
        role: str,
    ) -> nn.Module:
        """Single `from_pretrained` invocation, with `trust_remote_code=True`
        to cover both fork and stock-transformers paths.

        We always pass `trust_remote_code=True`. When the Zyphra fork
        (`Zyphra/transformers@zaya1`) is installed, `ZayaForCausalLM` is
        registered with `AutoModelForCausalLM` and the trust flag is
        unused; when stock transformers is installed, the flag triggers
        loading from the repo's modeling code.
        """
        log.info(
            "Loading %s %s (dtype=%s, attn=%s, revision=%s)",
            role,
            name_or_path,
            torch_dtype,
            attn_implementation,
            revision,
        )
        dtype: Any = "auto" if torch_dtype == "auto" else getattr(torch, torch_dtype)
        # `torch_dtype=` is the cross-version-stable kwarg name. Transformers
        # 4.51+ accepts `dtype=` as an alias and the BC `torch_dtype=` path
        # remains; transformers 4.46-4.50 only accepts `torch_dtype=` (an
        # unknown `dtype=` would be silently swallowed and the model would
        # load in float32 — for an 8B-param student that's a 2x VRAM blow-up).
        # We pin to `torch_dtype=` so the call works on the full pyproject
        # `transformers>=4.46.0` range.
        model = AutoModelForCausalLM.from_pretrained(
            name_or_path,
            revision=revision,
            torch_dtype=dtype,
            attn_implementation=attn_implementation,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        return cast(nn.Module, model)

    @staticmethod
    def _cast_lm_head_to_bf16(model: nn.Module) -> None:
        """Cast `model.lm_head` back to BF16 even when the body is FP8."""
        head = getattr(model, "lm_head", None)
        if head is None:
            log.warning(
                "Zaya1Adapter: teacher has no `lm_head` attr — skipping "
                "FP8 lm_head BF16 carve-out."
            )
            return
        head.to(dtype=torch.bfloat16)
        # Post-condition: weight dtype is bf16.
        weight = getattr(head, "weight", None)
        if weight is not None and weight.dtype != torch.bfloat16:
            raise RuntimeError(
                f"Zaya1Adapter: lm_head BF16 cast failed — weight.dtype is "
                f"{weight.dtype} (expected torch.bfloat16)."
            )

    def attention_module_paths(self, model: nn.Module) -> list[str]:
        """Module paths to ZAYA1's CCA blocks (verified at instantiation in Phase 5)."""
        raise NotImplementedError("Phase 5: Zaya1Adapter.attention_module_paths")

    def kv_quant_exempt_indices(self, model: nn.Module) -> list[int]:
        """Empty for ZAYA1 — no SSM layers per arxiv:2605.05365 Table I."""
        return []

    # REQ: LLR-0024
    def fp32_carve_outs(self, model: nn.Module) -> list[str]:
        """`lm_head`, routing softmax, CCA cache state, RMSNorm, residual adds
        per arxiv:2605.05365 §IV-D. Phase 5 verifies the exact wildcards
        against an instantiated ZAYA1 module tree."""
        raise NotImplementedError("Phase 5: Zaya1Adapter.fp32_carve_outs")

    # REQ: LLR-0026
    def required_attn_implementation(self) -> Literal["eager", "sdpa"]:
        """ZAYA1 uses CCA (not flash-attn-compatible). Phase 5 confirms which
        of `eager` / `sdpa` the Zyphra fork supports for hookability."""
        raise NotImplementedError("Phase 5: Zaya1Adapter.required_attn_implementation")

    # REQ: LLR-0025
    def router_replay_hook(self, teacher: nn.Module, student: nn.Module) -> object:
        """Pin student's MoE expert assignments to teacher's per
        arxiv:2605.05365 §IV-C. Phase 5 lands the implementation."""
        raise NotImplementedError("Phase 5: Zaya1Adapter.router_replay_hook")
