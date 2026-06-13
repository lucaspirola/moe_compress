"""Task 7 — WikiText-PPL eval-shard passthrough (default OFF = in-process verbatim)."""
from __future__ import annotations

import pytest
import torch

from moe_compress.stage6.plugins import wikitext_ppl


class _StubModel:
    class _Cfg:
        _attn_implementation = "eager"
    config = _Cfg()

    def __call__(self, input_ids=None, labels=None):
        class _Out:
            loss = torch.tensor(1.0)
        return _Out()

    def parameters(self):
        yield torch.zeros(1)


_CFG = {"sequence_length": 2048}


def _prebuilt(tok):
    chunks = torch.randint(0, 50, (4, 2048), dtype=torch.long)
    return {"tokenizer": tok, "chunks": chunks}


def test_wikitext_no_shard_forward_inprocess(monkeypatch):
    tok = object()
    calls = {"dp": 0}
    from moe_compress.tools import eval_shard
    monkeypatch.setattr(
        eval_shard, "run_dp_ppl",
        lambda *a, **k: (calls.__setitem__("dp", calls["dp"] + 1) or 1.0),
    )
    res = wikitext_ppl._wikitext2_ppl(
        _StubModel(), tok, _CFG, prebuilt=_prebuilt(tok), eval_shard=None,
    )
    assert calls["dp"] == 0
    # in-process loop ran (finite PPL from the stub loss=1.0)
    import math
    assert math.isfinite(res)
