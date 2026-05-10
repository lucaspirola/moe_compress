"""Tests for `kdr.eval.quick` (LLR-0036, LLR-0037, LLR-0038).

The wikitext2_ppl test uses a tiny CPU model + a stub `load_dataset` so the
test never hits the network. The log-samples test verifies the
GatheredParameters wrap path doesn't fire when zero3 is False.

# VERIFIES: LLR-0036
# VERIFIES: LLR-0037
# VERIFIES: LLR-0038
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from kdr.config import EvalConfig, WikiText2Config


class _TinyLM(nn.Module):
    """Returns logits with shape [B, T, V]; mimics HF causal-LM output."""

    def __init__(self, vocab: int = 17) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab, 4)
        self.head = nn.Linear(4, vocab, bias=False)

    def forward(self, *, input_ids: torch.Tensor) -> Any:
        h = self.emb(input_ids)
        logits = self.head(h)
        out = MagicMock()
        out.logits = logits
        return out


def _fake_tokenizer(vocab: int = 17) -> MagicMock:
    tok = MagicMock()
    # The eval calls `tokenizer(text, ...)` and indexes `["input_ids"]`.
    tok.return_value = {"input_ids": list(range(vocab)) * 50}
    tok.eos_token_id = vocab - 1
    return tok


def _fake_accelerator() -> MagicMock:
    accel = MagicMock()
    accel.is_main_process = True
    accel.device = torch.device("cpu")
    # `unwrap_model` returns the input by default; tests that need to
    # capture/replace the unwrapped model can override `.side_effect`.
    accel.unwrap_model.side_effect = lambda m: m
    return accel


def _fake_dataset(rows: int = 300) -> list[dict[str, str]]:
    """Produces enough non-empty rows that `load_dataset(...)` returns a
    list-like the eval code can `for r in ds` over."""
    return [{"text": f"row {i} of fake corpus"} for i in range(rows)]


def test_wikitext2_ppl_returns_finite_float_on_tiny_model() -> None:
    from kdr.eval.quick import wikitext2_ppl

    accel = _fake_accelerator()
    student = _TinyLM(vocab=17)
    student.eval()
    cfg = WikiText2Config(enabled=True, sequence_length=8, num_sequences=4)

    with patch(
        "datasets.load_dataset",
        return_value=_fake_dataset(),
    ):
        ppl = wikitext2_ppl(student, _fake_tokenizer(17), cfg, accel)

    assert ppl > 0
    assert ppl != float("inf")
    assert ppl == pytest.approx(ppl, rel=1e-9)  # not NaN


# REQ: VERIFIES: LLR-0036
def test_wikitext2_ppl_does_not_construct_distributed_sampler() -> None:
    """Collective forward — no DistributedSampler. We can't simulate world>1
    in a unit test, but we can verify the function doesn't *construct* one
    (a comment mentioning the name is permitted)."""
    import inspect
    import re

    from kdr.eval import quick

    src = inspect.getsource(quick.wikitext2_ppl)
    # Strip comments & docstrings before grepping so the structural check
    # ignores a NOTE comment mentioning the symbol.
    code_lines: list[str] = []
    in_doc = False
    for line in src.splitlines():
        stripped = line.lstrip()
        if stripped.startswith('"""'):
            in_doc = not in_doc
            continue
        if in_doc:
            continue
        # drop inline comments
        code_lines.append(re.sub(r"#.*$", "", line))
    code = "\n".join(code_lines)
    assert "DistributedSampler(" not in code, (
        "wikitext2_ppl must not construct a DistributedSampler (LLR-0036)."
    )


# REQ: VERIFIES: LLR-0038
def test_run_returns_none_when_disabled() -> None:
    from kdr.eval.quick import run

    accel = _fake_accelerator()
    cfg = EvalConfig(wikitext2=WikiText2Config(enabled=False, sequence_length=8, num_sequences=4))
    assert run(MagicMock(), MagicMock(), cfg, accel) is None


# REQ: VERIFIES: LLR-0038
def test_run_swallows_exceptions_from_eval() -> None:
    """Eval is a diagnostic, not a gate — any exception logs a warning and
    returns None rather than aborting training."""
    from kdr.eval import quick

    accel = _fake_accelerator()
    cfg = EvalConfig(wikitext2=WikiText2Config(enabled=True, sequence_length=8, num_sequences=4))
    with patch.object(
        quick, "wikitext2_ppl", side_effect=RuntimeError("simulated")
    ):
        result = quick.run(MagicMock(), MagicMock(), cfg, accel)
    assert result is None


# REQ: VERIFIES: LLR-0037
def test_log_samples_no_op_for_empty_prompts() -> None:
    """`log_samples([])` returns immediately without touching the model."""
    from kdr.eval.quick import log_samples

    student = MagicMock()
    log_samples(student, MagicMock(), [], _fake_accelerator())
    student.eval.assert_not_called()
    student.train.assert_not_called()


# REQ: VERIFIES: LLR-0037
def test_log_samples_skips_gather_when_not_zero3() -> None:
    """Non-DS path: `.generate` is called directly without GatheredParameters
    (which would import `deepspeed`, not present in the kdr venv)."""
    from kdr.eval.quick import log_samples

    accel = _fake_accelerator()
    # Override unwrap_model to return a generator-equipped fake model.
    fake_unwrapped = MagicMock()
    fake_unwrapped.generate.return_value = torch.zeros(1, 5, dtype=torch.long)
    accel.unwrap_model.side_effect = lambda m: fake_unwrapped

    student = MagicMock()
    student.parameters.return_value = []
    tok = MagicMock()
    tok.return_value = MagicMock(input_ids=torch.zeros(1, 1, dtype=torch.long))
    tok.decode.return_value = "out"
    tok.eos_token_id = 0

    with patch("kdr.eval.quick.is_zero3", return_value=False):
        log_samples(student, tok, ["hello"], accel)

    student.eval.assert_called_once()
    student.train.assert_called_once()
    fake_unwrapped.generate.assert_called_once()
