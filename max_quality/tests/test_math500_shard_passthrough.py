"""Task 7 — MATH-500 eval-shard passthrough (default OFF = in-process verbatim)."""
from __future__ import annotations

import pytest

from moe_compress.stage6.plugins import math500


_CFG = {"max_new_tokens": 8, "num_samples": 2}


def _prebuilt(tok):
    return {
        "tokenizer": tok,
        "raw_problems": ["1+1", "2+2"],
        "prompts": ["P0", "P1"],
        "answers": ["2", "4"],
    }


def test_math500_no_shard_calls_generate_inprocess(monkeypatch):
    tok = object()
    calls = {"gen": 0, "dp": 0}
    monkeypatch.setattr(
        math500, "_generate_batched",
        lambda *a, **k: (calls.__setitem__("gen", calls["gen"] + 1) or ["C0", "C1"]),
    )
    from moe_compress.tools import eval_shard
    monkeypatch.setattr(
        eval_shard, "run_dp_generate",
        lambda *a, **k: (calls.__setitem__("dp", calls["dp"] + 1) or []),
    )
    monkeypatch.setattr(math500, "_check_math", lambda *a, **k: False)

    res = math500._math500(object(), tok, _CFG, prebuilt=_prebuilt(tok), eval_shard=None)
    assert calls["gen"] == 1
    assert calls["dp"] == 0
    assert res == 0.0
