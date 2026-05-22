"""RK-5 — Router-KD teacher-logits slot-plugin tests.

Verifies the RK-5 ``TeacherCachePlugin`` / ``TeacherLivePlugin`` scaffolding in
``router_kd/plugins/teacher.py``:

* both classes import from the plugin module;
* both structurally satisfy the universal ``PipelinePlugin`` Protocol and carry
  correct metadata (name / paper id / config_key / tuple-typed fields);
* ``TeacherCachePlugin.is_enabled`` is gated on ``teacher_logits_cache``;
  ``TeacherLivePlugin.is_enabled`` is unconditionally True;
* both expose the ``provide_teacher_logits`` slot hook;
* the ``dispatch_first`` slot semantics: the CACHE plugin wins on a hit, the
  LIVE plugin wins on a miss, and the cache plugin DEFERS (returns ``None``)
  on a miss;
* the per-batch cache-slice arithmetic (the ``token_start`` epoch offset);
* the registry construction order ``("teacher_cache", "teacher_live")``;
* the module never imports the ``stage5_router_kd`` monolith or
  ``router_kd.orchestrator`` at any scope (the circular-import contract);
* ``teacher.py`` reproduces a private ``_set_experts_implementation`` (RK-5 is
  PURE Pattern B — the helper is reproduced, not imported from the monolith).

RK-5 is PURE Pattern B: the monolith ``stage5_router_kd.py`` is NOT modified
(the cache-load is inline ``run()`` code, ``_get_teacher`` is a ``run()``
closure — nothing standalone to relocate), so byte-identity is trivially
preserved and the RK-0 golden snapshot stays green for free.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch


# ---------------------------------------------------------------------------
# Synthetic cache-payload + config builders
# ---------------------------------------------------------------------------


def _make_cfg(*, batch_size=2, seq_len=4, num_samples=8, vocab=32, epochs=1,
              cache_path=None):
    """A minimal ``config`` dict carrying the keys the slot hooks read.

    ``cache_path`` sets ``stage5_router_kd.teacher_logits_cache`` — needed so
    ``TeacherCachePlugin.is_enabled`` keeps the cache plugin in
    ``registry.enabled(cfg)`` (its ``is_enabled`` is gated on that config key).
    """
    s5 = {
        "batch_size": batch_size,
        "max_sequence_length": seq_len,
        "max_calibration_samples": num_samples,
        "epochs": epochs,
    }
    if cache_path is not None:
        s5["teacher_logits_cache"] = cache_path
    return {
        "stage5_router_kd": s5,
        "model": {
            "name_or_path": "tiny", "revision": "main",
            "torch_dtype": "float32", "device_map": "cpu",
            "attn_implementation": "sdpa",
            "load_in_4bit": False, "trust_remote_code": False,
        },
    }


def _make_cache_payload(*, batch_size=2, seq_len=4, num_samples=8, vocab=32,
                        arange=False):
    """A format_version=1 teacher-logits cache payload.

    ``logits`` is shaped ``[num_samples * seq_len, vocab]``. With ``arange``
    the tensor is filled with ``arange`` so a slice's exact values are
    predictable for the slice-arithmetic test.
    """
    n_tokens = num_samples * seq_len
    if arange:
        logits = torch.arange(n_tokens * vocab, dtype=torch.float32).view(
            n_tokens, vocab
        )
    else:
        torch.manual_seed(7)
        logits = torch.randn(n_tokens, vocab)
    return {
        "format_version": 1,
        "batch_size": batch_size,
        "sequence_length": seq_len,
        "num_samples": num_samples,
        "logits": logits,
    }


# ---------------------------------------------------------------------------
# Module import + Protocol conformance + metadata
# ---------------------------------------------------------------------------


def test_teacher_module_imports():
    """Both RK-5 plugin classes import from the plugin module."""
    from moe_compress.router_kd.plugins.teacher import (
        TeacherCachePlugin,
        TeacherLivePlugin,
    )

    assert isinstance(TeacherCachePlugin, type)
    assert isinstance(TeacherLivePlugin, type)


def test_cache_plugin_satisfies_protocol():
    """``TeacherCachePlugin`` structurally satisfies ``PipelinePlugin``."""
    from moe_compress.pipeline.plugin import PipelinePlugin
    from moe_compress.router_kd.plugins.teacher import TeacherCachePlugin

    assert isinstance(TeacherCachePlugin(), PipelinePlugin)


def test_live_plugin_satisfies_protocol():
    """``TeacherLivePlugin`` structurally satisfies ``PipelinePlugin``."""
    from moe_compress.pipeline.plugin import PipelinePlugin
    from moe_compress.router_kd.plugins.teacher import TeacherLivePlugin

    assert isinstance(TeacherLivePlugin(), PipelinePlugin)


def test_cache_plugin_metadata():
    """``TeacherCachePlugin`` metadata — name / paper id / config_key / tuples."""
    from moe_compress.router_kd.plugins.teacher import TeacherCachePlugin

    plugin = TeacherCachePlugin()
    assert plugin.name == "teacher_cache"
    assert "2603.02217" in plugin.paper
    assert plugin.config_key == "stage5_router_kd.teacher_logits_cache"
    assert isinstance(plugin.reads, tuple)
    assert isinstance(plugin.writes, tuple)
    assert isinstance(plugin.provides, tuple)
    assert "teacher_logits_cache" in plugin.writes


def test_live_plugin_metadata():
    """``TeacherLivePlugin`` metadata — name / paper id / config_key / tuples."""
    from moe_compress.router_kd.plugins.teacher import TeacherLivePlugin

    plugin = TeacherLivePlugin()
    assert plugin.name == "teacher_live"
    assert "2603.02217" in plugin.paper
    assert plugin.config_key == "stage5_router_kd.teacher_model_repo"
    assert isinstance(plugin.reads, tuple)
    assert isinstance(plugin.writes, tuple)
    assert isinstance(plugin.provides, tuple)


def test_cache_plugin_is_enabled_gated_on_cache_path():
    """``TeacherCachePlugin.is_enabled`` is gated on ``teacher_logits_cache``.

    Disabled when unset / empty; enabled when a cache path is configured.
    """
    from moe_compress.router_kd.plugins.teacher import TeacherCachePlugin

    plugin = TeacherCachePlugin()
    assert plugin.is_enabled({}) is False
    assert plugin.is_enabled({"stage5_router_kd": {}}) is False
    assert plugin.is_enabled(
        {"stage5_router_kd": {"teacher_logits_cache": ""}}
    ) is False
    assert plugin.is_enabled(
        {"stage5_router_kd": {"teacher_logits_cache": "/tmp/cache.pt"}}
    ) is True


def test_live_plugin_is_enabled_unconditional():
    """``TeacherLivePlugin.is_enabled`` is unconditionally True (the fallback)."""
    from moe_compress.router_kd.plugins.teacher import TeacherLivePlugin

    plugin = TeacherLivePlugin()
    assert plugin.is_enabled({}) is True
    assert plugin.is_enabled(
        {"stage5_router_kd": {"teacher_model_repo": "some/repo"}}
    ) is True


def test_both_plugins_have_provide_teacher_logits_hook():
    """Both plugins expose a callable ``provide_teacher_logits`` slot hook."""
    from moe_compress.router_kd.plugins.teacher import (
        TeacherCachePlugin,
        TeacherLivePlugin,
    )

    assert callable(getattr(TeacherCachePlugin(), "provide_teacher_logits", None))
    assert callable(getattr(TeacherLivePlugin(), "provide_teacher_logits", None))


# ---------------------------------------------------------------------------
# dispatch_first slot semantics
# ---------------------------------------------------------------------------


def test_cache_plugin_registry_order():
    """The registry preserves construction order ``(teacher_cache, teacher_live)``."""
    from moe_compress.pipeline.registry import PluginRegistry
    from moe_compress.router_kd.plugins.teacher import (
        TeacherCachePlugin,
        TeacherLivePlugin,
    )

    reg = PluginRegistry([TeacherCachePlugin(), TeacherLivePlugin()])
    assert reg.names() == ("teacher_cache", "teacher_live")


def test_cache_wins_under_dispatch_first(tiny_model):
    """The master-plan check: on a cache HIT, ``dispatch_first`` returns the
    CACHE plugin's tensor (the live teacher is never touched).

    Seeds a ``PipelineContext`` with a validated in-memory cache payload so the
    cache hook hits; the live plugin is also in the registry but must not win.
    """
    from moe_compress.pipeline.context import PipelineContext
    from moe_compress.pipeline.registry import PluginRegistry
    from moe_compress.router_kd.plugins.teacher import (
        TeacherCachePlugin,
        TeacherLivePlugin,
    )

    # cache_path set so TeacherCachePlugin.is_enabled keeps it in enabled(cfg);
    # the validated payload is seeded directly (load_teacher_cache already ran).
    cfg = _make_cfg(
        batch_size=2, seq_len=4, num_samples=8, vocab=32,
        cache_path="/tmp/teacher_cache.pt",
    )
    payload = _make_cache_payload(
        batch_size=2, seq_len=4, num_samples=8, vocab=32, arange=True
    )
    ctx = PipelineContext()
    ctx.set("config", cfg)
    ctx.set("teacher_logits_cache", payload)

    reg = PluginRegistry([TeacherCachePlugin(), TeacherLivePlugin()])
    input_ids = torch.zeros(2, 4, dtype=torch.long)
    result = PluginRegistry.dispatch_first(
        reg.enabled(cfg),
        "provide_teacher_logits",
        ctx,
        input_ids=input_ids,
        epoch=0,
        batch_index=0,
        num_batches=4,
    )

    # The cache plugin produced this — the known arange slice [0:8] reshaped.
    expected = payload["logits"][0:8].to(torch.float32).view(2, 4, -1)
    assert torch.equal(result, expected)


def test_live_wins_on_cache_miss(tiny_model):
    """On a cache MISS, ``dispatch_first`` falls through to the LIVE plugin.

    The cache slot holds ``None`` (miss) so ``TeacherCachePlugin`` defers; the
    live plugin's ``_teacher`` is pre-injected with the conftest tiny model so
    no real ``load_model`` runs.
    """
    from moe_compress.pipeline.context import PipelineContext
    from moe_compress.pipeline.registry import PluginRegistry
    from moe_compress.router_kd.plugins.teacher import (
        TeacherCachePlugin,
        TeacherLivePlugin,
    )

    # cache_path set so the cache plugin stays in enabled(cfg) and is exercised
    # in the dispatch_first chain; the payload is None so it DEFERS at runtime.
    cfg = _make_cfg(
        batch_size=2, seq_len=4, num_samples=8, vocab=32,
        cache_path="/tmp/teacher_cache.pt",
    )
    ctx = PipelineContext()
    ctx.set("config", cfg)
    ctx.set("teacher_logits_cache", None)  # explicit miss

    live = TeacherLivePlugin()
    live._teacher = tiny_model  # no real load_model — inject the tiny model
    reg = PluginRegistry([TeacherCachePlugin(), live])

    input_ids = torch.randint(0, 32, (2, 4))
    result = PluginRegistry.dispatch_first(
        reg.enabled(cfg),
        "provide_teacher_logits",
        ctx,
        input_ids=input_ids,
        epoch=0,
        batch_index=0,
        num_batches=4,
    )

    # M-3: the plugin runs the teacher forward inside torch.no_grad(); compute
    # `expected` under the same context so the comparison has exact context
    # parity (and stays correct if the model ever gains dropout / BN).
    with torch.no_grad():
        expected = tiny_model(input_ids=input_ids).logits.detach().to(torch.float32)
    assert torch.equal(result, expected)
    assert result.dtype == torch.float32


def test_cache_plugin_defers_on_miss():
    """``TeacherCachePlugin.provide_teacher_logits`` returns ``None`` on a miss.

    No ``teacher_logits_cache`` slot at all → a clean defer (``None``).
    """
    from moe_compress.pipeline.context import PipelineContext
    from moe_compress.router_kd.plugins.teacher import TeacherCachePlugin

    cfg = _make_cfg()
    ctx = PipelineContext()
    ctx.set("config", cfg)

    plugin = TeacherCachePlugin()
    result = plugin.provide_teacher_logits(
        ctx,
        input_ids=torch.zeros(2, 4, dtype=torch.long),
        epoch=0,
        batch_index=0,
        num_batches=4,
    )
    assert result is None


def test_cache_plugin_defers_on_none_payload():
    """A ``teacher_logits_cache`` slot holding ``None`` is also a miss → defer."""
    from moe_compress.pipeline.context import PipelineContext
    from moe_compress.router_kd.plugins.teacher import TeacherCachePlugin

    cfg = _make_cfg()
    ctx = PipelineContext()
    ctx.set("config", cfg)
    ctx.set("teacher_logits_cache", None)

    plugin = TeacherCachePlugin()
    result = plugin.provide_teacher_logits(
        ctx,
        input_ids=torch.zeros(2, 4, dtype=torch.long),
        epoch=0,
        batch_index=0,
        num_batches=4,
    )
    assert result is None


def test_cache_slice_arithmetic():
    """The per-batch cache slice matches ``token_start:token_end`` for several
    ``(epoch, batch_index)`` pairs.

    ``logits`` is an ``arange``-filled tensor so the slice's exact values are
    predictable. ``token_start = (epoch * num_batches + batch_index) *
    cache_tokens_per_batch`` with ``cache_tokens_per_batch = batch_size *
    seq_len``.
    """
    from moe_compress.pipeline.context import PipelineContext
    from moe_compress.router_kd.plugins.teacher import TeacherCachePlugin

    batch_size, seq_len, vocab = 2, 4, 32
    num_batches = 4
    # Multi-epoch coverage: num_samples sized so epochs * batches all fit.
    num_samples = batch_size * num_batches  # = 8 (one epoch of tokens)
    cfg = _make_cfg(
        batch_size=batch_size, seq_len=seq_len,
        num_samples=num_samples, vocab=vocab,
    )
    payload = _make_cache_payload(
        batch_size=batch_size, seq_len=seq_len,
        num_samples=num_samples, vocab=vocab, arange=True,
    )
    ctx = PipelineContext()
    ctx.set("config", cfg)
    ctx.set("teacher_logits_cache", payload)

    plugin = TeacherCachePlugin()
    input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    tokens_per_batch = batch_size * seq_len

    for epoch, batch_index in [(0, 0), (0, 1), (0, 3)]:
        result = plugin.provide_teacher_logits(
            ctx, input_ids=input_ids,
            epoch=epoch, batch_index=batch_index, num_batches=num_batches,
        )
        token_start = (epoch * num_batches + batch_index) * tokens_per_batch
        token_end = token_start + tokens_per_batch
        expected = (
            payload["logits"][token_start:token_end]
            .to(torch.float32)
            .view(batch_size, seq_len, -1)
        )
        assert torch.equal(result, expected), (
            f"slice mismatch for epoch={epoch} batch_index={batch_index}"
        )
        assert result.shape == (batch_size, seq_len, vocab)


def test_cache_slice_arithmetic_multi_epoch():
    """M-2: the cache slice incorporates the EPOCH offset for epoch >= 1.

    The slice index is ``token_start = (epoch * num_batches + batch_index) *
    cache_tokens_per_batch`` — the ``epoch * num_batches`` term advances the
    window one full epoch of batches per epoch. A regression that dropped or
    miscomputed the epoch term would still pass ``test_cache_slice_arithmetic``
    (epoch=0 makes the term vanish) but corrupt epoch-N batches by replaying
    epoch-0 teacher logits. This test guards the F1 fix.

    The cache is built LARGER — ``num_samples = 2 * batch_size * num_batches``
    so it holds 2 full epochs of tokens — then the returned slice is asserted
    for ``(epoch=1, batch_index=0)`` and ``(epoch=1, batch_index=<last>)``.
    """
    from moe_compress.pipeline.context import PipelineContext
    from moe_compress.router_kd.plugins.teacher import TeacherCachePlugin

    batch_size, seq_len, vocab = 2, 4, 32
    num_batches = 4
    # 2-epoch cache: num_samples sized so two full epochs of batches fit.
    num_samples = 2 * batch_size * num_batches  # = 16 (two epochs of tokens)
    cfg = _make_cfg(
        batch_size=batch_size, seq_len=seq_len,
        num_samples=num_samples, vocab=vocab,
    )
    payload = _make_cache_payload(
        batch_size=batch_size, seq_len=seq_len,
        num_samples=num_samples, vocab=vocab, arange=True,
    )
    ctx = PipelineContext()
    ctx.set("config", cfg)
    ctx.set("teacher_logits_cache", payload)

    plugin = TeacherCachePlugin()
    input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    tokens_per_batch = batch_size * seq_len

    # epoch=1, first and last batch: the epoch term must shift the window by a
    # full epoch of batches (epoch * num_batches * tokens_per_batch tokens).
    for epoch, batch_index in [(1, 0), (1, num_batches - 1)]:
        result = plugin.provide_teacher_logits(
            ctx, input_ids=input_ids,
            epoch=epoch, batch_index=batch_index, num_batches=num_batches,
        )
        token_start = (epoch * num_batches + batch_index) * tokens_per_batch
        token_end = token_start + tokens_per_batch
        expected = (
            payload["logits"][token_start:token_end]
            .to(torch.float32)
            .view(batch_size, seq_len, -1)
        )
        assert torch.equal(result, expected), (
            f"slice mismatch for epoch={epoch} batch_index={batch_index}"
        )
        assert result.shape == (batch_size, seq_len, vocab)
        # The epoch-1 window must NOT alias the epoch-0 window — proves the
        # epoch offset is actually applied (a dropped epoch term would make
        # epoch=1,batch=0 return the epoch=0,batch=0 slice).
        epoch0_start = batch_index * tokens_per_batch
        assert token_start != epoch0_start
        epoch0_slice = (
            payload["logits"][epoch0_start:epoch0_start + tokens_per_batch]
            .to(torch.float32)
            .view(batch_size, seq_len, -1)
        )
        assert not torch.equal(result, epoch0_slice), (
            f"epoch={epoch} batch_index={batch_index} aliased the epoch-0 "
            "slice — the epoch offset was not applied"
        )


def test_cache_slice_divisibility_guard():
    """A ``max_calibration_samples`` not divisible by ``batch_size`` raises.

    Reproduces the monolith's per-batch divisibility ``RuntimeError``.
    """
    from moe_compress.pipeline.context import PipelineContext
    from moe_compress.router_kd.plugins.teacher import TeacherCachePlugin

    # num_samples=7 is not divisible by batch_size=2.
    cfg = _make_cfg(batch_size=2, seq_len=4, num_samples=7, vocab=32)
    payload = _make_cache_payload(
        batch_size=2, seq_len=4, num_samples=7, vocab=32, arange=True
    )
    ctx = PipelineContext()
    ctx.set("config", cfg)
    ctx.set("teacher_logits_cache", payload)

    plugin = TeacherCachePlugin()
    with pytest.raises(RuntimeError, match="divisible by batch_size"):
        plugin.provide_teacher_logits(
            ctx,
            input_ids=torch.zeros(2, 4, dtype=torch.long),
            epoch=0, batch_index=0, num_batches=4,
        )


# ---------------------------------------------------------------------------
# Circular-import + Pattern-B contract
# ---------------------------------------------------------------------------


def test_no_monolith_import_at_any_scope():
    """The module never imports ``stage5_router_kd`` / ``router_kd.orchestrator``.

    The module docstring's contract says NEVER import the monolith (or the
    orchestrator) at any scope — module-top OR function-local — since either
    would risk an import cycle (the monolith re-imports the plugin package at
    load time). Parse the source with ``ast`` and walk the FULL tree so a
    function-local ``import stage5_router_kd`` cannot slip past.

    For ``ImportFrom`` both halves are checked: ``node.module`` (the
    ``from X import ...`` package) AND ``node.names`` (the imported symbols) —
    so the cycle-causing ``from moe_compress import stage5_router_kd`` form
    (``module="moe_compress"``, name ``stage5_router_kd``) is also caught.
    """
    from moe_compress.router_kd.plugins import teacher as mod

    src = Path(mod.__file__).read_text()
    tree = ast.parse(src)
    forbidden = ("stage5_router_kd", "router_kd.orchestrator", "orchestrator")
    for node in ast.walk(tree):  # any nesting level, not just module-top
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not any(f in alias.name for f in forbidden), (
                    f"forbidden import at any scope: {alias.name}"
                )
        elif isinstance(node, ast.ImportFrom):
            mod_name = node.module or ""
            assert not any(f in mod_name for f in forbidden), (
                f"forbidden import-from at any scope: {mod_name}"
            )
            for alias in node.names:
                assert not any(f in alias.name for f in forbidden), (
                    f"forbidden import-from name at any scope: "
                    f"from {mod_name} import {alias.name}"
                )


def test_live_plugin_reproduces_set_experts_implementation():
    """``teacher.py`` defines a private ``_set_experts_implementation``.

    RK-5 is PURE Pattern B: the small monolith helper is REPRODUCED in the
    plugin module (not imported from ``stage5_router_kd``). Confirm the module
    defines a module-level ``_set_experts_implementation`` function.
    """
    from moe_compress.router_kd.plugins import teacher as mod

    assert hasattr(mod, "_set_experts_implementation")
    assert callable(mod._set_experts_implementation)
    src = Path(mod.__file__).read_text()
    tree = ast.parse(src)
    defined = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    }
    assert "_set_experts_implementation" in defined
