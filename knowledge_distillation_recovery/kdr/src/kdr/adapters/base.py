"""ModelAdapter Protocol (LLR-0022).

Each supported model family implements this Protocol. The trainer iterates
the Protocol's outputs without knowing which model family is loaded.

Context ownership (LLR-0023): the adapter does NOT enter or exit the
`activate_zero3_init` context. The training loop opens the context BEFORE
calling `load_teacher_and_student` and keeps it open until after
`mtq.quantize` returns (per LLR-0048). The adapter performs both
`from_pretrained` calls inside the already-active context.
"""

from __future__ import annotations

from typing import Literal, Protocol

import torch.nn as nn
from accelerate import Accelerator
from transformers import PreTrainedTokenizerBase

from ..config import StudentConfig, TeacherConfig
from .router_replay import RouterReplayHookProtocol


# REQ: LLR-0022
class ModelAdapter(Protocol):
    """Per-model-family knowledge: loaders, exempt indices, FP32 carve-outs."""

    name: str

    def load_teacher_and_student(
        self,
        accelerator: Accelerator,
        *,
        teacher_cfg: TeacherConfig,
        student_cfg: StudentConfig,
    ) -> tuple[nn.Module, nn.Module, PreTrainedTokenizerBase]:
        """Load `(teacher, student, tokenizer)` from the configured sources.

        Self-distillation usually means `teacher_cfg.name_or_path ==
        student_cfg.source` but kdr does not enforce it — the adapter is
        free to load from different repos as long as the tokenizers are
        compatible (the trainer asserts this independently).

        The teacher is loaded BEFORE the student. Under ZeRO-3 the training
        loop has opened `activate_zero3_init` before calling — both
        `from_pretrained` calls happen inside that context (LLR-0048 steps
        2 and 3). On exit the teacher's parameters all have
        `requires_grad=False` (LLR-0004); the student is left in the state
        the loop later modifies via `requires_grad_(True)`.

        The returned tokenizer is the STUDENT's tokenizer; the trainer
        independently checks teacher/student tokenizer compatibility and
        uses the student's for calibration and the saved checkpoint.
        """
        ...

    def attention_module_paths(self, model: nn.Module) -> list[str]:
        """Dotted paths to attention modules whose K/V outputs receive hooks.

        Used by the K/V quantizer hooks. Path strings (rather than
        `nn.Module` class names) survive HF version refactors.
        """
        ...

    def kv_quant_exempt_indices(self, model: nn.Module) -> list[int]:
        """Layer indices whose K/V are NOT quantized.

        Empty list for pure-attention models like ZAYA1-8B (no SSM layers).
        Non-empty for hybrid Mamba-2 / Transformer models, where SSM layers
        keep their hidden state at FP8 minimum (Ch3 spec §3.4.2).
        """
        ...

    def fp32_carve_outs(self, model: nn.Module) -> list[str]:
        """Submodule path patterns excluded from weight fake-quant.

        For ZAYA1-8B these are `lm_head`, routing softmax modules, CCA
        cache state, RMSNorm modules, and residual-stream addition points
        (arxiv:2605.05365 §IV-D).
        """
        ...

    def required_attn_implementation(self) -> Literal["eager", "sdpa"]:
        """The HF `attn_implementation` value the adapter requires.

        Flash-attn is rejected because it fuses K/V projection and never
        exposes post-projection K/V tensors at a Python hook boundary,
        silently no-opping the KV simulator.
        """
        ...

    def router_replay_hook(
        self, teacher: nn.Module, student: nn.Module
    ) -> RouterReplayHookProtocol:
        """Hook context that pins the student's MoE expert assignments to
        those produced by the teacher on the same input batch.

        Critical for MoE QAD stability per arxiv:2605.05365 §IV-C ("the
        single most important MoE-specific change"). For non-MoE models
        this returns a :class:`NoOpReplayContextManager` so the training
        loop's call sites stay polymorphic.
        """
        ...
