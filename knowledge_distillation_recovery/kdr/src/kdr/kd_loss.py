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


class _NativeKLDLoss(nn.Module):
    """Pure-torch parity with `modelopt.torch.distill.LogitsDistillationLoss`.

    Used when modelopt is unavailable (e.g. the BF16-only smoke whose image
    deliberately doesn't ship modelopt — see
    `tests/test_loop_dispatch.py::test_bf16_mode_does_not_import_modelopt`).
    Numerics match modelopt's class at any T by the parity test in
    `tests/test_kld_loss.py`: forward-KL with `batchmean` reduction and an
    internal `T**2` scaling to undo the temperature-softmax gradient shrink.
    """

    def __init__(self, temperature: float):
        super().__init__()
        self.T = float(temperature)

    def forward(self, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        T = self.T
        # `dtype=torch.float32` lets log_softmax/softmax accept bf16 logits
        # and accumulate the reduction in fp32 *internally*, without forcing
        # the caller to materialise a `(B*T, V)` fp32 tensor (~1.2 GB at
        # V=150k, B*T=2048). Output is fp32; numerics identical to an
        # explicit `.float()` upcast at the boundary.
        s_lp = torch.nn.functional.log_softmax(student / T, dim=-1, dtype=torch.float32)
        t_p = torch.nn.functional.softmax(teacher / T, dim=-1, dtype=torch.float32)
        return torch.nn.functional.kl_div(s_lp, t_p, reduction="batchmean") * (T * T)


def _get_kld_loss_fn(temperature: float) -> nn.Module:
    """Return the cached `LogitsDistillationLoss(temperature, batchmean)` instance.

    Modelopt is imported lazily so kdr is importable in environments without
    the modelopt wheel. If the import fails at call time we fall back to
    `_NativeKLDLoss` (bit-equal at the formula level — modelopt currently
    wraps the same `F.kl_div` reduction with the same `T**2` scaling).
    """
    fn = _KLD_LOSS_CACHE.get(temperature)
    if fn is None:
        with _CACHE_LOCK:
            fn = _KLD_LOSS_CACHE.get(temperature)
            if fn is None:
                try:
                    from modelopt.torch.distill.losses import LogitsDistillationLoss

                    fn = LogitsDistillationLoss(temperature=temperature, reduction="batchmean")
                except ImportError:
                    fn = _NativeKLDLoss(temperature=temperature)
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
    fn = _get_kld_loss_fn(temperature)
    if isinstance(fn, _NativeKLDLoss):
        # Native path's `log_softmax(..., dtype=torch.float32)` does the
        # fp32 reduction internally; passing bf16 logits avoids
        # materialising a `(B*T, V)` fp32 tensor on every micro.
        s = student_logits.reshape(-1, vocab)
        t = teacher_logits.reshape(-1, vocab)
    else:
        # Modelopt's `LogitsDistillationLoss` expects fp32 logits (its
        # legacy contract). Keep the explicit upcast on that branch.
        s = student_logits.reshape(-1, vocab).float()
        t = teacher_logits.reshape(-1, vocab).float()
    # `LogitsDistillationLoss.forward(student, teacher)`: per modelopt's API,
    # the FIRST positional argument is the predicted distribution Q (student)
    # and the SECOND is the target distribution P (teacher). The function
    # computes `KLD(p_teacher || p_student)` — i.e. forward KL with teacher
    # as the target. The loss is multiplied internally by T**2 to compensate
    # for gradient scaling under temperature softmax.
    result: torch.Tensor = fn(s, t)
    return result
