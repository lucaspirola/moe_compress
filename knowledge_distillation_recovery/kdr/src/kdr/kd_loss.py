"""Forward KLD distillation loss (LLR-0001, LLR-0002, LLR-0003).

Direct port from `structural_recovery/distillation.py:43-86`. Phase 3a
verifies bit-equal numerical parity with that source.
"""

from __future__ import annotations

import threading

import torch
import torch.nn as nn

# REQ: LLR-0002
# Module-level cache so we don't construct LogitsDistillationLoss per
# microbatch. Keyed by temperature; thread-safe via _CACHE_LOCK because the
# DataLoader workers may race on first access from rank 0 before the loop's
# main thread populates it.
_KLD_LOSS_CACHE: dict[float, nn.Module] = {}
_CACHE_LOCK = threading.Lock()


def _get_kld_loss_fn(temperature: float) -> nn.Module:
    """Return the cached `LogitsDistillationLoss(temperature, batchmean)` instance.

    Modelopt is imported lazily so kdr is importable in environments without
    the modelopt wheel (e.g. local dev that doesn't need to run the loss).
    """
    fn = _KLD_LOSS_CACHE.get(temperature)
    if fn is None:
        with _CACHE_LOCK:
            fn = _KLD_LOSS_CACHE.get(temperature)
            if fn is None:
                from modelopt.torch.distill.losses import LogitsDistillationLoss

                fn = LogitsDistillationLoss(temperature=temperature, reduction="batchmean")
                _KLD_LOSS_CACHE[temperature] = fn
    return fn


# REQ: LLR-0001
def forward_kld_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Forward KL: `KLD(p_teacher || p_student)` averaged per token.

    Delegates to `modelopt.torch.distill.LogitsDistillationLoss` (the loss
    class QUALITY_RECOVERY_GUIDE.md §1.8.2 names as canonical).

    Reduction is `"batchmean"` after reshaping to `[B*T, V]` so PyTorch
    divides by the token count (`B*T`), giving per-token mean. ModelOpt's
    default `"mean"` would divide by `B*T*V` (an extra factor of vocab size,
    ~150k) and silently collapse the gradient signal.

    Both inputs are upcast to fp32 before the softmax so the loss is
    numerically stable on bf16 logits — the underlying model can still run
    in bf16.

    Pad/mask contract: the calibration tensor produced by
    `moe_compress.utils.calibration._tokenize_to_fixed_length` is fully
    packed (concatenated streams separated by EOS, hard 5%-shortage cap).
    Every position is a real token, so per-position averaging is correct
    without an attention_mask. If a future call site feeds pad-bearing
    sequences this contract is violated — assert at the boundary there.
    """
    # REQ: LLR-0003
    if student_logits.shape[-1] != teacher_logits.shape[-1]:
        raise ValueError(
            f"forward_kld_loss: vocab mismatch — student V={student_logits.shape[-1]} "
            f"vs teacher V={teacher_logits.shape[-1]}. Same-tokenizer distillation "
            "is required."
        )
    vocab = student_logits.shape[-1]
    s = student_logits.reshape(-1, vocab).float()
    t = teacher_logits.reshape(-1, vocab).float()
    # `LogitsDistillationLoss.forward(student, teacher)`: per modelopt's API,
    # the FIRST positional argument is the predicted distribution Q (student)
    # and the SECOND is the target distribution P (teacher). The function
    # computes `KLD(p_teacher || p_student)` — i.e. forward KL with teacher
    # as the target. The loss is multiplied internally by T**2 to compensate
    # for gradient scaling under temperature softmax.
    result: torch.Tensor = _get_kld_loss_fn(temperature)(s, t)
    return result
