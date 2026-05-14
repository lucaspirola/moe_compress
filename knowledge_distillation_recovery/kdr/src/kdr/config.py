"""Top-level YAML config models for kdr (LLR-0006, LLR-0010, LLR-0011, LLR-0041, LLR-0049).

Loading a kdr YAML through `Config.model_validate(yaml.safe_load(...))` either
returns a fully-typed `Config` instance or raises a Pydantic `ValidationError`
with a precise field path. There is no silent coercion and no silent extra
field acceptance — see HLR-0012.
"""

# REQ: LLR-0010
# REQ: LLR-0011

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .modes import Mode
from .quant.specs import KVQuantSpec, MixedWeightSpec, UniformWeightSpec


class TeacherConfig(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    name_or_path: str
    revision: str = "main"
    torch_dtype: Literal["auto", "bfloat16", "float16", "float8_e4m3fn"] = "bfloat16"
    attn_implementation: Literal["eager", "sdpa"] = "sdpa"


class StudentConfig(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    source: str
    torch_dtype: Literal["bfloat16", "float16"] = "bfloat16"
    attn_implementation: Literal["eager", "sdpa"] = "sdpa"


# REQ: LLR-0041
class CalibrationConfig(BaseModel):
    """LLR-0041."""

    model_config = ConfigDict(strict=True, extra="forbid")

    source: str
    dataset: str
    seed: int
    num_sequences: int = Field(..., gt=0)
    sequence_length: int = Field(..., gt=0)
    subset_weights: dict[str, float]
    ptq_subset_size: int = Field(default=256, gt=0)


class KVQuantBlock(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    key: KVQuantSpec
    value: KVQuantSpec


class QuantBlock(BaseModel):
    """Top-level `quant:` YAML block, used only in `da_qad` mode.

    `weight` accepts two shapes:
      - `UniformWeightSpec` — single global weight spec (the pre-Phase-7.2
        shape; all existing YAMLs use this).
      - `MixedWeightSpec` — per-module-pattern specs (Profile J et al.).

    Pydantic discriminates by structural shape: `spec_map` field present →
    `MixedWeightSpec`; otherwise `UniformWeightSpec`.
    """

    model_config = ConfigDict(strict=True, extra="forbid")
    weight: UniformWeightSpec | MixedWeightSpec
    kv_quant: KVQuantBlock


class DistillationConfig(BaseModel):
    """LLR-0049 owns `eval_every_n_steps`; LLR-0010 owns `optimizer`.

    **The tokens-per-step invariant.** The paper-relevant training quantity
    is the total tokens consumed, not how those tokens are partitioned across
    micro-batches. Concretely:

        tokens_per_step = per_device_batch_size × world × gradient_accumulation × sequence_length
        total_steps     = total_tokens // tokens_per_step

    Implementations are free to trade off `per_device_batch_size` against
    `gradient_accumulation` at constant `tokens_per_step` without changing
    the optimizer's trajectory (modulo BF16 non-determinism in the reduction
    order across micro-batches, which is below the BF16 forward's own
    non-determinism floor). This is the basis of the `auto_batch_size`
    optimization: at small `per_device_batch_size` the GPU is launch-bound
    rather than compute-bound (we observed ~44% util at bs=1 on H200 in
    Phase 7.1), so probing for the largest bs that fits in VRAM and
    reducing `gradient_accumulation` proportionally typically buys 1.5-2×
    wall-time on Hopper/Blackwell at no quality cost.

    **Scaling for bigger models.** The probe is bounded above by VRAM. For a
    70B model in BF16 on a single 141 GB H200, the probe will return
    `per_device_batch_size=1` and the user is on the manual `gradient_accumulation`
    path. For a 32B model the probe typically returns 2; for 8B / 13B it
    typically returns 4-8. Multi-GPU FSDP/ZeRO-3 changes the per-rank VRAM
    budget but not the per-rank probe semantics.

    **Mutation under `auto_batch_size`.** When the trainer's Stage-4.5 probe
    rebalances `per_device_batch_size` and `gradient_accumulation`, it
    writes directly to those attributes after construction. `validate_assignment=True`
    is enabled so Pydantic re-runs every field validator on the write —
    a probe that ever produced `bs=0` or `ga=0` would surface immediately
    rather than corrupting `tokens_per_step` silently. Direct mutation
    elsewhere in the codebase must continue to satisfy the field
    constraints; constraints can no longer be bypassed by post-validate
    attribute assignment.
    """

    model_config = ConfigDict(strict=True, extra="forbid", validate_assignment=True)

    loss: Literal["forward_kld"] = "forward_kld"
    temperature: float = 1.0
    # Optional linear-ramp on temperature, borrowed from max_quality
    # stage-5 KD (acts as soft-to-hard curriculum: high T early smooths
    # the teacher distribution so STE updates aren't chasing sharp peaks
    # while codebook assignments are still settling; T → endpoint as the
    # student approaches a basin). When set, the per-step temperature is
    # linearly interpolated from `temperature_start` at step 0 to
    # `temperature` (the endpoint) at `total_steps - 1`. When None
    # (default) the temperature is held constant at `temperature` — the
    # backward-compatible behaviour for every existing config.
    temperature_start: float | None = Field(default=None, gt=0)
    optimizer: Literal["adamw_bnb_8bit", "deepspeed_cpu_adam"]
    learning_rate: float = Field(..., gt=0)
    min_learning_rate: float = Field(..., gt=0)
    weight_decay: float = Field(..., ge=0)
    # Pydantic v2 strict does not auto-coerce list -> tuple. YAML produces a
    # list of two floats; using `list[float]` with length constraint is the
    # idiomatic strict-mode shape.
    betas: list[float] = Field(..., min_length=2, max_length=2)
    grad_clip_norm: float = Field(..., gt=0)
    warmup_steps: int = Field(..., ge=0)
    total_tokens: int = Field(..., gt=0)
    per_device_batch_size: int = Field(..., gt=0)
    gradient_accumulation: int = Field(..., gt=0)
    sequence_length: int = Field(..., gt=0)
    log_every_n_steps: int = Field(..., gt=0)
    eval_every_n_steps: int = Field(..., gt=0)
    # `0` means "no partial saves" — only the final checkpoint is written at
    # end-of-training. Per HLR-0007 / LLR-0027 this is the smoke-tier
    # behaviour (smoke runs are too short to benefit from periodic saves).
    save_every_n_steps: int = Field(..., ge=0, description="0 = skip partial saves; final-only")
    trainable_scope: Literal["full", "experts_only", "factored_only"]
    use_gradient_checkpointing: bool = True

    @model_validator(mode="after")
    def _validate_temperature_curriculum(self) -> "DistillationConfig":
        """Ensure the temperature ramp is soft → hard (start > end).

        The curriculum's purpose is to start with a softer teacher target
        (high T) and tighten to a sharp target (lower T) as the student
        moves into a basin. A config with `temperature_start <= temperature`
        would invert the curriculum (cold → hot), which silently changes
        the regularisation story without surfacing as a runtime error.
        Catch the typo at config-load time instead.
        """
        if (
            self.temperature_start is not None
            and self.temperature_start <= self.temperature
        ):
            raise ValueError(
                f"temperature_start ({self.temperature_start}) must be "
                f"strictly greater than temperature ({self.temperature}) "
                "to describe a soft-to-hard curriculum. Unset "
                "temperature_start for constant-temperature training."
            )
        return self

    # Phase 7+ throughput optimization. See `_probe_max_batch_size` in
    # `kdr/training/loop.py` for the algorithm and the class docstring above
    # for the tokens-per-step invariant.
    #
    # When True, the trainer probes for the largest `per_device_batch_size`
    # that fits VRAM at training start (one-time, ~30 s on H200), then
    # rewrites `per_device_batch_size` and `gradient_accumulation` to that
    # probed value and the matching divisor (preserving `tokens_per_step`
    # exactly). Disabled by default so smoke configs and existing setups see
    # zero behavior change. Set to True in real-recovery configs where the
    # 1.5-2× wall-time win is worth the one-time probe cost.
    #
    # Generic-tool note: for any model where `per_device_batch_size > 1` is
    # already manually set to fill VRAM, the probe will leave that value
    # alone (it caps at the original `per_device_batch_size * gradient_accumulation`
    # product). The probe is meant as a convenience for the "I don't know
    # what fits, pick the best" case, not as a forced override.
    auto_batch_size: bool = Field(
        default=False,
        description=(
            "Phase 7+ throughput optimization. If True, probe at trainer "
            "init for the largest per_device_batch_size that fits VRAM, "
            "then reduce gradient_accumulation proportionally to preserve "
            "tokens_per_step. Default off."
        ),
    )

    # Phase 7+ throughput optimization (LLR-0027 v2). See
    # `kdr/io/save.py:_AsyncSaveExecutor` for the algorithm.
    #
    # When True, the rank-0 disk-write phase of save_partial (state_dict
    # serialization + tokenizer + metadata + atomic rename + sentinel)
    # runs in a single-flight background thread, so the trainer's next
    # forward+backward starts as soon as the collective `get_state_dict`
    # returns. On an 8B BF16 model, save_partial stalls the trainer
    # ~10-30 s per partial in sync mode; async mode reduces the stall to
    # ~the consolidation time alone (~1-2 s on H200). The trade-off is
    # ~weight-tensor-size of additional CPU RAM held in the snapshot
    # while the background thread serializes (~17 GB for 8B BF16,
    # ~140 GB for 70B BF16).
    #
    # Default off so the sync path is the audit-friendly baseline.
    # Real recovery configs (e.g., `zaya1_8b_bf16_recovery.yaml`) opt in.
    enable_async_save: bool = Field(
        default=False,
        description=(
            "Phase 7+ throughput optimization. If True, save_partial "
            "dispatches the rank-0 disk write to a background thread; the "
            "next training step starts immediately after the collective "
            "get_state_dict completes. Costs ~weight-tensor-size of CPU "
            "RAM per pending snapshot. Default off."
        ),
    )


class WikiText2Config(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    enabled: bool = True
    sequence_length: int = Field(default=2048, gt=0)
    num_sequences: int = Field(default=64, gt=0)


class EvalConfig(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    wikitext2: WikiText2Config = WikiText2Config()


class Config(BaseModel):
    """Root kdr YAML schema.

    `quant` is required iff `mode == "da_qad"`. The Pydantic model permits
    `quant` to be None for `bf16` mode; the trainer's mode dispatch (LLR-0007)
    asserts the consistency at runtime.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    mode: Mode
    teacher: TeacherConfig
    student: StudentConfig
    calibration: CalibrationConfig
    quant: QuantBlock | None = None
    distillation: DistillationConfig
    eval: EvalConfig = EvalConfig()
