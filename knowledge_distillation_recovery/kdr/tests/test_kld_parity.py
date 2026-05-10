"""Phase 3a numerical-parity test (plan §"Phase 3a — FKLD numerical parity").

Asserts `kdr.kd_loss.forward_kld_loss` is bit-equal to
`structural_recovery.distillation.forward_kld_loss` on 100 random
`[B,T,V]` inputs at three temperatures. Catches reduction/axis bugs
without burning training time. **Land before any quant simulator.**

Both implementations delegate to the same modelopt
`LogitsDistillationLoss(reduction='batchmean')` instance — they SHOULD be
bit-equal by construction. This test is the regression gate.

Modelopt and structural_recovery are required to be installed; the test
is skipped otherwise.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

# Lazy: only run if both deps are present.
pytest.importorskip("modelopt.torch.distill.losses")

# structural_recovery's `src/` may not be installed; add it to sys.path so
# `import structural_recovery.distillation` works at test time.
_STRUCTURAL_SRC = (
    Path(__file__).resolve().parent.parent.parent / "structural_recovery" / "src"
)
if _STRUCTURAL_SRC.is_dir() and str(_STRUCTURAL_SRC) not in sys.path:
    sys.path.insert(0, str(_STRUCTURAL_SRC))

structural_recovery = pytest.importorskip("structural_recovery.distillation")

# These imports run AFTER `pytest.importorskip` so the whole module skips
# cleanly when modelopt / structural_recovery are absent. E402 is therefore
# expected and intentional for this file.
from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from kdr.kd_loss import forward_kld_loss as kdr_loss  # noqa: E402


@settings(
    max_examples=100,
    deadline=None,  # KL on small tensors is fast but not deterministic-time
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    batch=st.integers(min_value=1, max_value=4),
    seq=st.integers(min_value=1, max_value=8),
    vocab=st.integers(min_value=4, max_value=64),
    temperature=st.sampled_from([0.5, 1.0, 2.0]),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
# VERIFIES: LLR-0001
def test_bit_equal_to_structural_recovery(
    batch: int, seq: int, vocab: int, temperature: float, seed: int
) -> None:
    """LLR-0001 verification: kdr's wrapper produces bit-equal output to the
    structural_recovery port for the same `[B, T, V]` input."""
    gen = torch.Generator().manual_seed(seed)
    student = torch.randn(batch, seq, vocab, generator=gen)
    teacher = torch.randn(batch, seq, vocab, generator=gen)

    kdr_out = kdr_loss(student, teacher, temperature=temperature)
    sr_out = structural_recovery.forward_kld_loss(student, teacher, temperature=temperature)

    # Both implementations should produce bit-equal outputs because they
    # delegate to the same modelopt `LogitsDistillationLoss` instance with
    # the same config, on the same input. Anything other than exact equality
    # indicates wrapper drift.
    assert torch.equal(kdr_out, sr_out), (
        f"FKLD parity failure on B={batch}, T={seq}, V={vocab}, temp={temperature}, "
        f"seed={seed}: kdr={kdr_out.item()}, structural_recovery={sr_out.item()}, "
        f"diff={abs(kdr_out.item() - sr_out.item())}"
    )
