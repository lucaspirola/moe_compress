"""Top-level YAML config models for kdr (LLR-0006, LLR-0011, LLR-0041, LLR-0049).

Loading a kdr YAML through `Config.model_validate(yaml.safe_load(...))` either
returns a fully-typed `Config` instance or raises a Pydantic `ValidationError`
with a precise field path. There is no silent coercion and no silent extra
field acceptance — see HLR-0012.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .modes import Mode
from .quant.specs import KVQuantSpec, WeightQuantSpec


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
    """Top-level `quant:` YAML block, used only in `da_qad` mode."""

    model_config = ConfigDict(strict=True, extra="forbid")

    weight: WeightQuantSpec
    kv_quant: KVQuantBlock


class DistillationConfig(BaseModel):
    """LLR-0049 owns `eval_every_n_steps`; LLR-0010 owns `optimizer`."""

    model_config = ConfigDict(strict=True, extra="forbid")

    loss: Literal["forward_kld"] = "forward_kld"
    temperature: float = 1.0
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
