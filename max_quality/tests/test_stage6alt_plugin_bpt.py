"""S6A-3 -- Stage 6alt thermometer BPT-metric + zero-shot-subset plugin tests.

Verifies the S6A-3 ``BptMetricPlugin`` + ``ZeroShotSubsetPlugin``
scaffolding in ``stage6alt/plugins/bpt_metric.py`` and
``stage6alt/plugins/zero_shot_subset.py``:

* the two Pattern-A helper functions (``_bpt_from_nll`` /
  ``_lm_eval_subset``) import from their respective plugin modules;
* the ``stage6alt_thermometer`` monolith re-exports the SAME relocated
  function objects (the ``# noqa: F401`` re-import block is load-bearing
  -- the S6A-0 golden snapshot uses ``monkeypatch.setattr`` against the
  monolith namespace, which only keeps biting if the function objects
  there are ``is``-identical to the plugin-module ones);
* both plugins satisfy the universal ``PipelinePlugin`` Protocol, carry
  the declared metadata, are unconditionally enabled, and expose their
  respective S6A-6 phase hooks (``compute_bpt`` and
  ``compute_zero_shot_subset``);
* neither plugin module imports the ``stage6alt_thermometer`` monolith
  or ``stage6alt.orchestrator`` at any scope (the circular-import
  contract -- mirrors the AST guard in ``test_stage6alt_plugin_corpus``);
* ``_bpt_from_nll`` directly exercises the eager-attention guard, the
  finite-loss happy path, the ``collect_argmax`` shape contract, and
  both inf-return failure paths (None loss every batch / per-batch
  exception every batch);
* ``_lm_eval_subset`` directly exercises the success path (sum present)
  and the lm-eval-unavailable path (all-None);
* the ``compute_bpt`` and ``compute_zero_shot_subset`` hooks write the
  declared ctx slots from their helper-return values.

S6A-3 is a PURE Pattern-A relocation slice (the helpers move; the
monolith re-imports them) plus Pattern-B inert hooks (the monolith
``run()`` is NOT modified for them; an S6A-6 sub-task wires them live).
The byte-identical behavioral gate is the S6A-0 golden snapshot
(``test_stage6alt_golden_snapshot.py``); this file only checks the
relocation plumbing and the relocated logic.
"""
from __future__ import annotations

import ast
import sys
import types
from pathlib import Path

import pytest

try:
    import torch  # noqa: F401
    from moe_compress.stage6alt.plugins import (  # noqa: F401
        bpt_metric,
        zero_shot_subset,
    )
except Exception as e:  # pragma: no cover - import-time guard
    pytest.skip(
        f"Stage 6alt bpt-metric / zero-shot-subset imports unavailable: {e}",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    """Minimal tokenizer placeholder.

    ``_lm_eval_subset`` calls ``_lm_eval_tasks(model, tokenizer, ...)``;
    the harness wrapper is what consumes the tokenizer. In the
    inject-fake-lm_eval tests below the tokenizer is never inspected, so
    a plain object suffices -- this class exists for parity with the
    discipline of ``test_stage6alt_plugin_corpus.py``.
    """

    name_or_path = "fake-tokenizer"


def _fake_lm_eval_module(arc_acc: float | None, hsw_acc: float | None):
    """Build a fake ``lm_eval`` module whose ``simple_evaluate`` returns the
    given per-task accuracies.

    Mirrors the public ``lm_eval`` API surface that
    ``stage6.plugins.zero_shot_lm_eval._lm_eval_tasks`` consumes: a
    ``simple_evaluate`` callable returning a dict with a ``results`` slot
    keyed by task name, each value carrying the metric numbers under their
    lm-eval-shaped keys. ``acc_norm,none`` is the key the wrapper extracts
    as ``<task>_acc``; setting it to ``None`` lets the wrapper record the
    metric as missing.
    """
    mod = types.ModuleType("lm_eval")

    def simple_evaluate(*, model, tasks, batch_size=None, limit=None, **_kwargs):
        results: dict[str, dict[str, float | None]] = {}
        for t in tasks:
            if t == "arc_easy":
                results[t] = {"acc_norm,none": arc_acc}
            elif t == "hellaswag":
                results[t] = {"acc_norm,none": hsw_acc}
        return {"results": results}

    mod.simple_evaluate = simple_evaluate
    # ``_lm_eval_tasks`` references ``lm_eval.models.huggingface.HFLM`` via
    # ``importlib`` -- pre-populate a submodule so the import resolves.
    models_mod = types.ModuleType("lm_eval.models")
    hf_mod = types.ModuleType("lm_eval.models.huggingface")

    class _StubHFLM:
        def __init__(self, *_a, **_k):
            pass

    hf_mod.HFLM = _StubHFLM
    models_mod.huggingface = hf_mod
    mod.models = models_mod
    return mod, models_mod, hf_mod


# ---------------------------------------------------------------------------
# Tests -- module imports + Pattern-A re-export identity
# ---------------------------------------------------------------------------


def test_module_imports():
    """Both Pattern-A helpers + both plugin classes import from their modules."""
    from moe_compress.stage6alt.plugins.bpt_metric import (
        BptMetricPlugin,
        _bpt_from_nll,
    )
    from moe_compress.stage6alt.plugins.zero_shot_subset import (
        ZeroShotSubsetPlugin,
        _lm_eval_subset,
    )

    assert isinstance(BptMetricPlugin, type)
    assert isinstance(ZeroShotSubsetPlugin, type)
    assert callable(_bpt_from_nll)
    assert callable(_lm_eval_subset)


def test_monolith_reexports_pattern_a_functions():
    """The monolith re-exports the SAME relocated FUNCTION objects.

    Load-bearing for the S6A-0 golden snapshot: it does
    ``monkeypatch.setattr(stage6alt_thermometer, "_bpt_from_nll", ...)``
    and same for ``_lm_eval_subset``. That patch-by-attribute only keeps
    biting if the function objects on the monolith namespace are
    ``is``-identical to the relocated ones.
    """
    from moe_compress import stage6alt_thermometer
    from moe_compress.stage6alt.plugins import bpt_metric, zero_shot_subset

    assert stage6alt_thermometer._bpt_from_nll is bpt_metric._bpt_from_nll
    assert stage6alt_thermometer._lm_eval_subset is zero_shot_subset._lm_eval_subset


# ---------------------------------------------------------------------------
# Tests -- Protocol conformance + metadata + is_enabled + hooks
# ---------------------------------------------------------------------------


def test_plugins_satisfy_protocol():
    """Both plugins structurally satisfy ``PipelinePlugin``."""
    from moe_compress.pipeline.plugin import PipelinePlugin
    from moe_compress.stage6alt.plugins.bpt_metric import BptMetricPlugin
    from moe_compress.stage6alt.plugins.zero_shot_subset import (
        ZeroShotSubsetPlugin,
    )

    assert isinstance(BptMetricPlugin(), PipelinePlugin)
    assert isinstance(ZeroShotSubsetPlugin(), PipelinePlugin)


def test_bpt_metric_metadata():
    """BptMetricPlugin -- name / config_key / tuple-typed reads/writes/provides."""
    from moe_compress.stage6alt.plugins.bpt_metric import BptMetricPlugin

    plugin = BptMetricPlugin()
    assert plugin.name == "bpt_metric"
    assert plugin.config_key == "stage6_validate.thermometer"
    assert isinstance(plugin.reads, tuple)
    assert isinstance(plugin.writes, tuple)
    assert isinstance(plugin.provides, tuple)
    assert plugin.reads == ("model", "calib_ids", "config")
    assert plugin.writes == ("student_bpt", "student_argmax")
    assert plugin.provides == ()


def test_zero_shot_subset_metadata():
    """ZeroShotSubsetPlugin -- name / config_key / tuple-typed reads/writes/provides."""
    from moe_compress.stage6alt.plugins.zero_shot_subset import (
        ZeroShotSubsetPlugin,
    )

    plugin = ZeroShotSubsetPlugin()
    assert plugin.name == "zero_shot_subset"
    assert plugin.config_key == "stage6_validate.thermometer"
    assert isinstance(plugin.reads, tuple)
    assert isinstance(plugin.writes, tuple)
    assert isinstance(plugin.provides, tuple)
    assert plugin.reads == ("model", "tokenizer", "config")
    assert plugin.writes == (
        "student_arc_easy_acc_norm",
        "student_hellaswag_acc_norm",
        "student_acc_norm_sum",
    )
    assert plugin.provides == ()


def test_plugins_is_enabled_unconditional():
    """Both plugins are UNCONDITIONALLY enabled -- ``is_enabled`` always True.

    Every thermometer run must score BPT and the ARC-Easy + HellaSwag
    subset; ``config_key`` only names the thermometer config sub-tree,
    it never gates the plugin as a whole.
    """
    from moe_compress.stage6alt.plugins.bpt_metric import BptMetricPlugin
    from moe_compress.stage6alt.plugins.zero_shot_subset import (
        ZeroShotSubsetPlugin,
    )

    for cls in (BptMetricPlugin, ZeroShotSubsetPlugin):
        plugin = cls()
        assert plugin.is_enabled({}) is True
        assert plugin.is_enabled({"stage6_validate": {}}) is True
        assert plugin.is_enabled({
            "stage6_validate": {"thermometer": {"bpt_batch_size": 4}}
        }) is True


def test_plugins_have_phase_hooks():
    """The S6A-6 phase hooks ``compute_bpt`` / ``compute_zero_shot_subset`` exist."""
    from moe_compress.stage6alt.plugins.bpt_metric import BptMetricPlugin
    from moe_compress.stage6alt.plugins.zero_shot_subset import (
        ZeroShotSubsetPlugin,
    )

    bpt_plugin = BptMetricPlugin()
    zs_plugin = ZeroShotSubsetPlugin()
    assert callable(getattr(bpt_plugin, "compute_bpt", None))
    assert callable(getattr(zs_plugin, "compute_zero_shot_subset", None))


# ---------------------------------------------------------------------------
# Tests -- circular-import AST guards
# ---------------------------------------------------------------------------


def _ast_guard_no_monolith_import(mod):
    """Shared AST walk used by the per-module circular-import guards.

    The plugin docstrings forbid importing the ``stage6alt_thermometer``
    monolith (or the ``stage6alt.orchestrator``) at any scope -- module-top
    OR function-local -- since either would risk an import cycle (the
    monolith re-imports these modules at load time). Parse the source
    with ``ast`` and walk the FULL tree so a function-local
    ``import stage6alt_thermometer`` cannot slip past.

    For ``ImportFrom`` both halves are checked: ``node.module`` AND
    ``node.names`` -- so the cycle-causing
    ``from moe_compress import stage6alt_thermometer`` form is also caught.
    Each alias's ``asname`` is checked alongside its ``name`` so a renamed
    import (``import stage6alt_thermometer as x``) cannot slip past either.
    """
    src = Path(mod.__file__).read_text()
    tree = ast.parse(src)
    forbidden = (
        "stage6alt_thermometer",
        "stage6alt.orchestrator",
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not any(
                    f in alias.name or f in (alias.asname or "")
                    for f in forbidden
                ), f"forbidden import at any scope: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            mod_name = node.module or ""
            assert not any(f in mod_name for f in forbidden), (
                f"forbidden import-from at any scope: {mod_name}"
            )
            for alias in node.names:
                assert not any(
                    f in alias.name or f in (alias.asname or "")
                    for f in forbidden
                ), (
                    f"forbidden import-from name at any scope: "
                    f"from {mod_name} import {alias.name}"
                )


def test_no_monolith_import_bpt_metric():
    """``bpt_metric`` never imports ``stage6alt_thermometer`` / orchestrator."""
    from moe_compress.stage6alt.plugins import bpt_metric as mod

    _ast_guard_no_monolith_import(mod)


def test_no_monolith_import_zero_shot_subset():
    """``zero_shot_subset`` never imports ``stage6alt_thermometer`` / orchestrator."""
    from moe_compress.stage6alt.plugins import zero_shot_subset as mod

    _ast_guard_no_monolith_import(mod)


# ---------------------------------------------------------------------------
# Tests -- relocated _bpt_from_nll logic
# ---------------------------------------------------------------------------


def test_bpt_from_nll_requires_eager_attention(tiny_model):
    """Non-eager ``_attn_implementation`` raises ``RuntimeError``.

    Batch-size-invariant NLL requires eager attention (the same guard as
    Stage 6's ``_wikitext2_ppl`` / ``_lm_eval_tasks``). The helper inspects
    ``model.config._attn_implementation`` and aborts if it is not the
    Stage-6 contract value.
    """
    from moe_compress.stage6alt.plugins.bpt_metric import _bpt_from_nll

    # The tiny_model config has no ``_attn_implementation`` attribute by
    # default; the helper's ``getattr(..., None)`` then short-circuits the
    # equality check and raises.
    calib = torch.zeros(2, 4, dtype=torch.long)
    with pytest.raises(RuntimeError, match="batch-size-invariant NLL requires eager"):
        _bpt_from_nll(tiny_model, calib, device=None, batch_size=1)


def test_bpt_from_nll_finite_path(tiny_model):
    """Happy path: returns a positive finite BPT float on a tiny model."""
    from moe_compress.stage6alt.plugins.bpt_metric import _bpt_from_nll

    tiny_model.config._attn_implementation = "eager"
    calib = torch.zeros(2, 4, dtype=torch.long)

    bpt = _bpt_from_nll(tiny_model, calib, device=None, batch_size=1)

    assert isinstance(bpt, float)
    assert bpt > 0.0
    import math
    assert math.isfinite(bpt)


def test_bpt_from_nll_collect_argmax_shape(tiny_model):
    """``collect_argmax=True`` returns ``(float, tensor)`` with the labeled shape."""
    from moe_compress.stage6alt.plugins.bpt_metric import _bpt_from_nll

    tiny_model.config._attn_implementation = "eager"
    calib = torch.zeros(2, 4, dtype=torch.long)

    bpt, argmax = _bpt_from_nll(
        tiny_model, calib, device=None, batch_size=1, collect_argmax=True,
    )

    assert isinstance(bpt, float)
    assert isinstance(argmax, torch.Tensor)
    # Shape (num_seqs, seq_len-1): logits[:, :-1].argmax(-1) on a (2, 4) input.
    assert argmax.shape == (2, 3)
    assert argmax.dtype == torch.long


def test_bpt_from_nll_none_loss_returns_inf(tiny_model, monkeypatch):
    """A model whose ``forward`` returns ``loss=None`` every batch -> ``inf``.

    The helper logs ``None loss; skipping batch`` and -- because every
    batch is skipped -- returns ``float("inf")``: a loud failure that
    prevents a partial-corpus number from corrupting a directional
    comparison.
    """
    from moe_compress.stage6alt.plugins.bpt_metric import _bpt_from_nll

    tiny_model.config._attn_implementation = "eager"

    class _Out:
        loss = None
        logits = torch.zeros(1, 4, 32)

    def _forward(*_args, **_kwargs):
        return _Out()

    monkeypatch.setattr(tiny_model, "forward", _forward)

    calib = torch.zeros(2, 4, dtype=torch.long)
    bpt = _bpt_from_nll(tiny_model, calib, device=None, batch_size=1)

    assert bpt == float("inf")


def test_bpt_from_nll_exception_returns_inf(tiny_model, monkeypatch):
    """A model whose ``forward`` raises every batch -> ``inf``.

    The helper catches per-batch exceptions, logs a warning, and skips
    the batch. With every batch skipped the return value is
    ``float("inf")``.
    """
    from moe_compress.stage6alt.plugins.bpt_metric import _bpt_from_nll

    tiny_model.config._attn_implementation = "eager"

    def _boom(*_args, **_kwargs):
        raise RuntimeError("forward exploded")

    monkeypatch.setattr(tiny_model, "forward", _boom)

    calib = torch.zeros(2, 4, dtype=torch.long)
    bpt = _bpt_from_nll(tiny_model, calib, device=None, batch_size=1)

    assert bpt == float("inf")


# ---------------------------------------------------------------------------
# Tests -- relocated _lm_eval_subset logic
# ---------------------------------------------------------------------------


def test_lm_eval_subset_returns_three_keys_and_sum(monkeypatch):
    """Success path: ARC=0.55, HellaSwag=0.65 -> sum=1.2 and three keys present.

    Injects a fake ``lm_eval`` module via ``sys.modules`` so the helper's
    underlying ``_lm_eval_tasks`` wrapper (from
    ``stage6.plugins.zero_shot_lm_eval``) takes the real-harness branch but
    against a stub returning the canned per-task metrics. The model carries
    an ``_attn_implementation == "eager"`` config so the wrapper's eager-attn
    guard passes.
    """
    from moe_compress.stage6alt.plugins.zero_shot_subset import _lm_eval_subset

    fake_lm_eval, fake_models, fake_hf = _fake_lm_eval_module(0.55, 0.65)
    monkeypatch.setitem(sys.modules, "lm_eval", fake_lm_eval)
    monkeypatch.setitem(sys.modules, "lm_eval.models", fake_models)
    monkeypatch.setitem(sys.modules, "lm_eval.models.huggingface", fake_hf)

    class _ModelCfg:
        _attn_implementation = "eager"

    class _Model:
        config = _ModelCfg()

    model = _Model()
    tokenizer = _FakeTokenizer()
    result = _lm_eval_subset(
        model, tokenizer,
        arc_limit=10, hellaswag_limit=10, batch_size=1,
    )

    assert set(result.keys()) == {
        "arc_easy_acc_norm", "hellaswag_acc_norm", "acc_norm_sum",
    }
    assert result["arc_easy_acc_norm"] == pytest.approx(0.55)
    assert result["hellaswag_acc_norm"] == pytest.approx(0.65)
    assert result["acc_norm_sum"] == pytest.approx(1.2)


def test_lm_eval_subset_harness_unavailable(monkeypatch):
    """Harness-unavailable path: all three metrics + sum are ``None``.

    With ``sys.modules["lm_eval"] = None`` the underlying ``_lm_eval_tasks``
    wrapper records the harness as missing and yields ``None`` for every
    per-task metric; the subset wrapper then leaves all three result-dict
    entries -- and the sum -- as ``None``.
    """
    from moe_compress.stage6alt.plugins.zero_shot_subset import _lm_eval_subset

    monkeypatch.setitem(sys.modules, "lm_eval", None)

    model = object()
    tokenizer = _FakeTokenizer()
    result = _lm_eval_subset(
        model, tokenizer,
        arc_limit=10, hellaswag_limit=10, batch_size=1,
    )

    assert result["arc_easy_acc_norm"] is None
    assert result["hellaswag_acc_norm"] is None
    assert result["acc_norm_sum"] is None


# ---------------------------------------------------------------------------
# Tests -- inert phase hooks
# ---------------------------------------------------------------------------


def test_compute_bpt_hook(monkeypatch):
    """The inert ``compute_bpt`` hook writes ``student_bpt`` / ``student_argmax``.

    Patches ``_bpt_from_nll`` on the ``bpt_metric`` module to a stub
    returning a fixed ``(3.14, None)`` pair; the hook must lift those two
    values into ``student_bpt`` / ``student_argmax`` ctx slots without
    inspecting them further.
    """
    from moe_compress.pipeline.context import PipelineContext
    from moe_compress.stage6alt.plugins import bpt_metric
    from moe_compress.stage6alt.plugins.bpt_metric import BptMetricPlugin

    monkeypatch.setattr(
        bpt_metric, "_bpt_from_nll",
        lambda *a, **k: (3.14, None),
    )

    plugin = BptMetricPlugin()
    ctx = PipelineContext()
    ctx.set("model", object())
    ctx.set("calib_ids", torch.zeros(2, 4, dtype=torch.long))
    ctx.set("config", {"stage6_validate": {"thermometer": {"bpt_batch_size": 4}}})

    plugin.compute_bpt(ctx)

    assert ctx.get("student_bpt") == 3.14
    assert ctx.get("student_argmax") is None


def test_compute_zero_shot_subset_hook(monkeypatch):
    """The inert ``compute_zero_shot_subset`` hook writes 3 ctx slots from the helper return.

    Patches ``_lm_eval_subset`` on the ``zero_shot_subset`` module to a stub
    returning a canned dict; the hook must lift those three values into
    ``student_arc_easy_acc_norm`` / ``student_hellaswag_acc_norm`` /
    ``student_acc_norm_sum`` ctx slots without inspecting them further.
    """
    from moe_compress.pipeline.context import PipelineContext
    from moe_compress.stage6alt.plugins import zero_shot_subset
    from moe_compress.stage6alt.plugins.zero_shot_subset import (
        ZeroShotSubsetPlugin,
    )

    canned = {
        "arc_easy_acc_norm": 0.42,
        "hellaswag_acc_norm": 0.77,
        "acc_norm_sum": 1.19,
    }
    monkeypatch.setattr(
        zero_shot_subset, "_lm_eval_subset",
        lambda *a, **k: canned,
    )

    plugin = ZeroShotSubsetPlugin()
    ctx = PipelineContext()
    ctx.set("model", object())
    ctx.set("tokenizer", _FakeTokenizer())
    ctx.set("config", {
        "stage6_validate": {"thermometer": {
            "arc_easy_limit": 2, "hellaswag_limit": 2,
            "lm_eval_batch_size": "auto:4",
        }}
    })

    plugin.compute_zero_shot_subset(ctx)

    assert ctx.get("student_arc_easy_acc_norm") == pytest.approx(0.42)
    assert ctx.get("student_hellaswag_acc_norm") == pytest.approx(0.77)
    assert ctx.get("student_acc_norm_sum") == pytest.approx(1.19)


# ---------------------------------------------------------------------------
# Item-3b — BPT argmax B/L chunking is BIT-IDENTICAL to the unchunked argmax
# ---------------------------------------------------------------------------


def test_bpt_argmax_chunked_equals_unchunked(tiny_model):
    """The chunked-over-B argmax must be torch.equal to the single-shot argmax.

    Item-3b chunks only the B/L dims (vocab axis kept WHOLE), so the result is
    bit-identical regardless of chunk size: argmax has no float accumulation
    and first-index tie-break is preserved. We score the SAME corpus with
    batch_size=1 (so the whole corpus passes through a single forward per seq)
    at several argmax_chunk_b values and assert identical argmax tensors, and
    that the BPT float is unchanged."""
    import torch as _torch

    from moe_compress.stage6alt.plugins.bpt_metric import _bpt_from_nll

    tiny_model.config._attn_implementation = "eager"
    # 4 sequences x len 6 over the tiny vocab (32); distinct ids so the per-
    # position argmax is non-trivial and order-sensitive.
    calib = _torch.arange(4 * 6, dtype=_torch.long).reshape(4, 6) % 32

    # Reference: chunk_b large enough to be a single (B, L-1, V) argmax.
    bpt_full, argmax_full = _bpt_from_nll(
        tiny_model, calib, device=None, batch_size=4,
        collect_argmax=True, argmax_chunk_b=64,
    )
    for cb in (1, 2, 3):
        bpt_c, argmax_c = _bpt_from_nll(
            tiny_model, calib, device=None, batch_size=4,
            collect_argmax=True, argmax_chunk_b=cb,
        )
        assert _torch.equal(argmax_c, argmax_full), (
            f"argmax differs at argmax_chunk_b={cb}"
        )
        assert bpt_c == bpt_full
        assert argmax_c.shape == (4, 5)
        assert argmax_c.dtype == _torch.long
