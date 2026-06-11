"""Auto-batch wiring of ``ma_detection`` phase-a (rework: size_batch + run_with_oom_backoff).

Drives the real ``MADetectionPlugin().run`` over the tiny model fixture and
asserts the control flow:

  * Default-OFF (no ``auto_batch`` config): NEITHER ``resolve_batch`` NOR
    ``run_with_oom_backoff`` is called — the original single-run path. This is
    the inert path the golden snapshot depends on.
  * Enabled (``auto_batch.enabled=true``): ``resolve_batch`` IS invoked with
    ``fixed_batch=_PHASE_A_BATCH_SIZE`` + ``FidelityClass.BATCH_INVARIANT`` and
    the real pass runs through ``run_with_oom_backoff``.

GPU-free: the tiny model runs on CPU; ``resolve_batch`` is spied to return the
floor without a CUDA cost probe, and ``run_with_oom_backoff`` is spied to record
the call while delegating to the real implementation.
"""

from __future__ import annotations

from moe_compress.stage1.plugins import ma_detection as M
from moe_compress.stage1.plugins.ma_detection import MADetectionPlugin
from moe_compress.utils.auto_batch import FidelityClass
from moe_compress.pipeline.context import PipelineContext


class _TinyTokenizer:
    name_or_path = "tiny-tokenizer"
    eos_token_id = 0

    def __call__(self, text, *_, **__):
        return {"input_ids": [min(ord(c) % 32, 31) for c in (text or " ")]}

    def save_pretrained(self, *_args, **_kwargs):
        return None


def _ctx(tiny_model, tiny_config, tmp_path) -> PipelineContext:
    ctx = PipelineContext()
    ctx.set("model", tiny_model)
    ctx.set("tokenizer", _TinyTokenizer())
    ctx.set("config", tiny_config)
    ctx.set("artifacts_dir", tmp_path)
    ctx.set("device", None)
    return ctx


def test_default_off_neither_resolver_nor_backoff_called(tiny_model, tiny_config, tmp_path, monkeypatch):
    # No auto_batch block at all -> default-off path: original single run.
    tiny_config["stage1_grape"].pop("auto_batch", None)

    resolve_calls = {"n": 0}
    backoff_calls = {"n": 0}

    def _resolve_spy(*a, **k):
        resolve_calls["n"] += 1
        return M.resolve_batch(*a, **k)

    def _backoff_spy(*a, **k):
        backoff_calls["n"] += 1
        return M.run_with_oom_backoff(*a, **k)

    monkeypatch.setattr(M, "resolve_batch", _resolve_spy)
    monkeypatch.setattr(M, "run_with_oom_backoff", _backoff_spy)

    MADetectionPlugin().run(_ctx(tiny_model, tiny_config, tmp_path))

    assert resolve_calls["n"] == 0       # resolver never touched when off
    assert backoff_calls["n"] == 0       # nor the OOM-backoff wrapper


def test_enabled_invokes_resolver_and_runs_through_backoff(tiny_model, tiny_config, tmp_path, monkeypatch):
    tiny_config["stage1_grape"]["auto_batch"] = {"enabled": True}

    seen = {}

    def fake_resolve(cost_probe_fn, fixed_batch, fidelity_class, cfg, mem=None):
        # Record the call; return the floor (skips the real CUDA cost probe).
        seen["fixed_batch"] = fixed_batch
        seen["fidelity_class"] = fidelity_class
        return fixed_batch

    real_backoff = M.run_with_oom_backoff
    backoff = {"start": None, "floor": None, "n": 0}

    def spy_backoff(run_fn, start_batch, floor):
        backoff["n"] += 1
        backoff["start"] = start_batch
        backoff["floor"] = floor
        return real_backoff(run_fn, start_batch=start_batch, floor=floor)

    monkeypatch.setattr(M, "resolve_batch", fake_resolve)
    monkeypatch.setattr(M, "run_with_oom_backoff", spy_backoff)

    ctx = _ctx(tiny_model, tiny_config, tmp_path)
    MADetectionPlugin().run(ctx)

    assert seen["fixed_batch"] == M._PHASE_A_BATCH_SIZE
    assert seen["fidelity_class"] == FidelityClass.BATCH_INVARIANT
    assert backoff["n"] == 1                       # the real pass ran through the wrapper
    assert backoff["floor"] == M._PHASE_A_BATCH_SIZE
    assert backoff["start"] == M._PHASE_A_BATCH_SIZE  # resolver returned the floor
    # the pass still produced the four output slots
    assert isinstance(ctx.get("L"), set)
