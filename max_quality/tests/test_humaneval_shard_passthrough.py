"""Task 7 — HumanEval eval-shard passthrough (default OFF = in-process verbatim).

With eval_shard disabled (or absent), `_humaneval` must call `_generate_batched`
exactly once in-process — NO spawn, NO run_dp_generate — and return the
unchanged result. This is the byte-identical default-path guarantee.
"""
from __future__ import annotations

import pytest

from moe_compress.stage6.plugins import humaneval as hе_mod  # noqa
from moe_compress.stage6.plugins import humaneval


_CFG = {"max_new_tokens": 8, "exec_timeout_secs": 1}


def _prebuilt(tok):
    # bypass dataset load via the prebuilt artifact (tokenizer-identity guard)
    return {
        "tokenizer": tok,
        "raw_prompts": ["def f(): pass", "def g(): pass"],
        "prompts": ["P0", "P1"],
        "tests": ["assert True", "assert True"],
        "entry_points": ["f", "g"],
    }


def test_humaneval_no_shard_calls_generate_inprocess(monkeypatch):
    tok = object()
    calls = {"gen": 0, "dp": 0}

    def _fake_gen(model, tokenizer, prompts, **kw):
        calls["gen"] += 1
        return [f"C{i}" for i in range(len(prompts))]

    def _fake_dp(*a, **k):
        calls["dp"] += 1
        return []

    monkeypatch.setattr(humaneval, "_generate_batched", _fake_gen)
    # run_dp_generate is imported lazily inside the branch; patch on eval_shard.
    from moe_compress.tools import eval_shard
    monkeypatch.setattr(eval_shard, "run_dp_generate", _fake_dp)
    # score = trivial: every problem fails grading but the metric is computed.
    monkeypatch.setattr(humaneval, "_score_all_humaneval", lambda *a, **k: 0)

    res = humaneval._humaneval(
        object(), tok, _CFG, prebuilt=_prebuilt(tok), eval_shard=None,
    )
    assert calls["gen"] == 1
    assert calls["dp"] == 0
    assert res == 0.0


def test_humaneval_disabled_cfg_calls_generate_inprocess(monkeypatch):
    tok = object()
    calls = {"gen": 0, "dp": 0}
    monkeypatch.setattr(
        humaneval, "_generate_batched",
        lambda *a, **k: (calls.__setitem__("gen", calls["gen"] + 1) or ["C0", "C1"]),
    )
    from moe_compress.tools import eval_shard
    monkeypatch.setattr(
        eval_shard, "run_dp_generate",
        lambda *a, **k: (calls.__setitem__("dp", calls["dp"] + 1) or []),
    )
    monkeypatch.setattr(humaneval, "_score_all_humaneval", lambda *a, **k: 0)

    # eval_shard dict present but the config is disabled → still in-process.
    es = {
        "cfg": eval_shard.EvalShardConfig.from_dict({"enabled": False}),
        "tmp_dir": "/nonexistent",
        "experts_impl_generative": "batched_mm",
        "out_dir": "/nonexistent",
    }
    humaneval._humaneval(object(), tok, _CFG, prebuilt=_prebuilt(tok), eval_shard=es)
    assert calls["gen"] == 1
    assert calls["dp"] == 0
