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
from ..modes import Mode
from .router_replay import RouterReplayHookProtocol

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
        mode: Mode,
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

        `mode` (LLR-0026): selects the attention backend forwarded to
        `from_pretrained(..., attn_implementation=...)` via
        :meth:`required_attn_implementation`. `bf16` uses SDPA; `da_qad`
        uses eager for K/V hookability.
        """
        # LLR-0026 AC: adapter forces its required attn_implementation
        # (mode-aware), overriding the YAML's value. In `da_qad` CCA needs
        # `eager` for hookability; in `bf16` SDPA is selected for speed.
        # Silently honoring a YAML mis-configuration would either no-op
        # KV-quant (da_qad) or lose ~2-3× throughput (bf16) — see LLR-0026.
        required_attn = self.required_attn_implementation(mode)
        teacher = self._load_one(
            name_or_path=teacher_cfg.name_or_path,
            revision=teacher_cfg.revision,
            torch_dtype=teacher_cfg.torch_dtype,
            attn_implementation=required_attn,
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
            attn_implementation=required_attn,
            role="STUDENT",
        )

        # Zyphra/ZAYA1-8B ships a generation_config with `top_p=0.95` and
        # `top_k=-1` alongside `do_sample=False`. Newer transformers
        # (>= 5.x) validates this combination at save time and raises
        # `ValueError: GenerationConfig is invalid` because top_p/top_k are
        # sample-only flags. Unsetting them on the student preserves greedy
        # decoding (do_sample=False) and unblocks save_partial / final save
        # without altering inference behavior on the published checkpoint.
        if getattr(student, "generation_config", None) is not None:
            student.generation_config.top_p = None
            student.generation_config.top_k = None

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
    def _patch_zaya_router_mlp_expansion_list() -> None:
        """Workaround for upstream `Zyphra/transformers@zaya1` config-vs-code mismatch.

        `Zyphra/ZAYA1-reasoning-base/config.json` ships `zaya_mlp_expansion`
        as a list (e.g. ``[0, 256, 0, 256, ...]`` of length 80), but the
        fork's `ZayaModel.__init__` passes the WHOLE list down through
        `ZayaDecoderMLPLayer` → `ZayaBlock` → `ZayaRouter` without indexing
        it (verified against zaya1-branch source). `ZayaRouter.__init__`
        then does `self.mlp_expansion = int(mlp_expansion)` and raises
        `TypeError: int() argument must be a string, a bytes-like object
        or a real number, not 'list'`.

        Every router in the model gets the same list. Routers are only
        instantiated for MoE layers (odd `layer_n`); for those, the
        appropriate `mlp_expansion` is the UNIQUE NONZERO value in the
        list. (For ZAYA1-reasoning-base: 256.) Indexing by `layer_n`
        would also work for layers in range, but the list (length 80) is
        shorter than `num_hidden_layers` (120), so layer_n>=80 gets
        IndexError — the homogeneous-nonzero approach handles all 60 MoE
        layers uniformly.

        Idempotent (sentinel attribute), no-ops when the Zyphra fork isn't
        installed (e.g. local kdr venv with stock transformers). For
        heterogeneous-nonzero configs (none seen in the wild yet), the
        patch falls back to `layer_number` indexing with a bounds check.
        """
        try:
            import transformers.models.zaya.modeling_zaya as _zaya_mod
        except ImportError:
            return
        cls = _zaya_mod.ZayaRouter
        if getattr(cls, "_kdr_patched_mlp_expansion_list", False):
            return
        _orig = cls.__init__

        def _patched(  # type: ignore[no-untyped-def]
            self,
            config: Any,
            layer_n: int,
            num_moe_experts: int,
            moe_router_topk: int,
            mlp_expansion: Any,
            hidden_size: int | None = None,
            layer_number: int | None = None,
        ) -> None:
            if isinstance(mlp_expansion, (list, tuple)):
                unique_nonzero = {v for v in mlp_expansion if v}
                if len(unique_nonzero) == 1:
                    mlp_expansion = unique_nonzero.pop()
                elif len(unique_nonzero) > 1:
                    ln = layer_number if layer_number is not None else 0
                    if 0 <= ln < len(mlp_expansion) and mlp_expansion[ln]:
                        mlp_expansion = mlp_expansion[ln]
                    else:
                        raise ValueError(
                            f"ZayaRouter: cannot resolve mlp_expansion at "
                            f"layer_number={ln} from heterogeneous list of "
                            f"length {len(mlp_expansion)}"
                        )
                else:
                    raise ValueError(
                        "ZayaRouter: zaya_mlp_expansion list is all zeros; "
                        "router should not be instantiated"
                    )
            _orig(
                self, config, layer_n, num_moe_experts, moe_router_topk,
                mlp_expansion, hidden_size=hidden_size,
                layer_number=layer_number,
            )

        cls.__init__ = _patched
        cls._kdr_patched_mlp_expansion_list = True

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
        # Apply the mlp_expansion-list patch before any model load.
        Zaya1Adapter._patch_zaya_router_mlp_expansion_list()
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
        """Dotted paths to each transformer layer's CCA self-attention block.

        Walks ``model.named_modules()`` collecting paths whose last segment
        is ``self_attn``. Stock HF decoders and ZAYA1's ``ZayaForCausalLM``
        both use this naming convention; the segment-suffix match avoids
        false positives from intermediate submodules (``self_attn.q_proj`` etc.)
        while remaining family-portable.

        ZAYA1's actual CCA layout has the K/V projection inside a
        ``kv_compressor`` submodule under each ``self_attn``; the Native
        backend's KV hooks attach to the ``self_attn`` parent and intercept
        the ``(k, v)`` tuple output. Phase 7 validates the exact tuple shape
        on real ZAYA1.
        """
        return [name for name, _ in model.named_modules() if name.endswith(".self_attn")]

    def kv_quant_exempt_indices(self, model: nn.Module) -> list[int]:
        """Empty for ZAYA1 — no SSM layers per arxiv:2605.05365 Table I."""
        return []

    # REQ: LLR-0024
    def fp32_carve_outs(self, model: nn.Module) -> list[str]:
        """ZAYA1 §IV-D FP32 carve-out list.

        Returns substring patterns matched anywhere in dotted module names
        (the Native backend's matcher and modelopt's ``ignore`` list both
        consume substrings). Mirrors arxiv:2605.05365 §IV-D's enumerated
        carve-outs:

          * ``lm_head``      — LM head matmul (LLR-0024 AC #1 explicitly
            requires this entry)
          * ``router``       — routing softmax modules (also covers
            ``moe.router`` / ``router_layer`` / similar variants)
          * ``norm``         — RMSNorm (matches ``rmsnorm``, ``input_norm``,
            ``post_attention_layernorm``, etc.)
          * ``cca_cache``    — CCA cache state (ZAYA1-specific buffer)
          * ``embed_tokens`` — input token embedding (high-precision matmul,
            shares the LM-head's numerical sensitivity at the vocab tail)

        Residual additions are NOT carved out by module name — they are
        tensor ``+`` operations inside layer forwards, not standalone
        submodules. Modelopt's globally-disabled ``*input_quantizer``
        (set in ``config_map.quant_block_to_modelopt_config``) handles them
        by leaving every activation full-precision.

        ``model`` is unused at the substring level — the patterns address
        any module containing the substring; the parameter is retained for
        the Protocol shape and for future per-instance overrides.
        """
        del model  # Patterns are substring-based; no per-instance inspection needed.
        # REQ: LLR-0024
        return ["lm_head", "embed_tokens", "router", "norm", "cca_cache"]

    # REQ: LLR-0026
    def required_attn_implementation(self, mode: Mode) -> Literal["eager", "sdpa"]:
        """Mode-aware attention backend selection (LLR-0026).

        - ``mode == "da_qad"`` → ``"eager"``. CCA's convolutional
          downprojector is not flash-attn-compatible, and ``eager`` is the
          ATTN backend that exposes post-projection K/V tensors at a Python
          hook boundary. ``sdpa`` would also expose K/V to a forward hook on
          ``self_attn``, but the Zyphra fork's CCA layer routes through a
          custom function (``cca_eager_attention_forward`` per the fork's
          source) that is only wired under ``eager``. Phase 7.1 confirmed
          on real ZAYA1-8B.
        - ``mode == "bf16"`` → ``"sdpa"``. Pure-BF16 self/teacher-
          distillation places no hooks (the trainer installs
          ``NoOpReplayContextManager()`` at loop.py:160 in BF16 mode), so the
          eager-only constraint does not apply. SDPA produces numerically
          equivalent output to eager at BF16 precision (agreement an order
          of magnitude below the BF16 forward's own non-determinism) and is
          ~2-3× faster on Hopper/Blackwell, which dominates wall-time for
          80-layer 8B models. Phase 7.1's 200-step variant spent ~33 s/step
          on H200 entirely because eager was forced unnecessarily.

        Flash-attn is rejected in both modes: it fuses K/V projection and
        never exposes post-projection K/V tensors at a Python hook, silently
        no-opping the KV simulator in da_qad and offering no measurable
        speedup over SDPA on Hopper/Blackwell.

        Generic-tool guidance: when adding a new mode (e.g., FP8 / NVFP4
        quant variants), the choice is driven by two questions —
        (1) Will any hooks be placed at the post-K/V-projection boundary?
        If yes, ``eager`` is required because hooks must see post-projection
        tensors. (2) Is the model architecture's custom attention routine
        wired only under one HF backend? (e.g., ZAYA1's CCA is eager-only
        in the published fork.) If yes, the choice is constrained by the
        fork rather than by the mode. For a new fork that wires its custom
        attention under SDPA, both BF16 AND quant-with-no-K/V-hooks paths
        can use SDPA — only K/V-hooking paths are constrained to eager.
        """
        if mode == "bf16":
            return "sdpa"
        if mode == "da_qad":
            return "eager"
        # Defensive: an unknown mode must NOT silently default. Future modes
        # need an explicit branch here so the genericness rationale above is
        # actually applied per-mode rather than papered over with a fallback.
        raise ValueError(
            f"Zaya1Adapter.required_attn_implementation: unsupported mode "
            f"{mode!r}. Add an explicit branch in this method for any new "
            f"mode, after deciding eager-vs-sdpa per the docstring's "
            f"generic-tool guidance."
        )

    # REQ: LLR-0025
    def router_replay_hook(
        self, teacher: nn.Module, student: nn.Module
    ) -> RouterReplayHookProtocol:
        """Returns a context manager pinning student MoE assignments to teacher's.

        Implementation lives in :mod:`.router_replay`. ZAYA1's MoE layer uses
        the standard HF ``router`` last-segment naming; the context manager
        walks both teacher and student for submodules whose dotted path
        ends with ``.router`` (or equals ``router``) and installs
        capture/replay hooks (LLR-0025).
        """
        from .router_replay import RouterReplayContextManager

        return RouterReplayContextManager(teacher, student, router_path_pattern="router")
