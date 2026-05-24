"""S6-5 — Stage 6 teacher-provider plugin extraction tests.

Verifies the S6-5 ``TeacherProviderPlugin`` scaffolding in
``stage6/plugins/teacher_provider.py``:

* the 6 Pattern-A symbols (``TEACHER_CACHE_FORMAT_VERSION``,
  ``_safe_pkg_version``, ``_teacher_cache_key``, ``_load_teacher_cache``,
  ``_save_teacher_cache``, ``_preload_teacher_to_cpu``) import from the
  plugin module;
* the ``stage6_validate`` monolith re-exports the SAME relocated function
  objects (the ``# noqa: F401`` re-import block is load-bearing);
* ``TeacherProviderPlugin`` satisfies the universal ``PipelinePlugin``
  Protocol, carries correct metadata, is unconditionally enabled, and
  exposes the (S6-8) ``provide_teacher_side`` phase hook;
* the module never imports the ``stage6_validate`` monolith or
  ``stage6.orchestrator`` at any scope (the circular-import contract);
* the relocated cache-key / load / save helpers behave correctly;
* the inert ``provide_teacher_side`` hook short-circuits on a cache hit.

S6-5 covers a MIXED pattern: the 6 standalone symbols are relocated
verbatim (the monolith re-imports them); the teacher-side block — inline
``run()`` code in the monolith — is reproduced in the inert
``provide_teacher_side`` hook (the monolith ``run()`` is NOT modified for
it). The byte-identical behavioral gate is the S6-0 golden snapshot
(``test_stage6_golden_snapshot.py``); this file only checks the relocation
plumbing and the relocated logic.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

try:
    import torch  # noqa: F401
    from moe_compress.stage6.plugins import teacher_provider  # noqa: F401
except Exception as e:  # pragma: no cover - import-time guard
    pytest.skip(
        f"Stage 6 teacher-provider imports unavailable: {e}",
        allow_module_level=True,
    )


def test_teacher_provider_module_imports():
    """All 6 Pattern-A symbols + ``TeacherProviderPlugin`` import from the module."""
    from moe_compress.stage6.plugins.teacher_provider import (
        TEACHER_CACHE_FORMAT_VERSION,
        TeacherProviderPlugin,
        _load_teacher_cache,
        _preload_teacher_to_cpu,
        _safe_pkg_version,
        _save_teacher_cache,
        _teacher_cache_key,
    )

    assert isinstance(TeacherProviderPlugin, type)
    assert TEACHER_CACHE_FORMAT_VERSION == 1
    for fn in (
        _load_teacher_cache,
        _preload_teacher_to_cpu,
        _safe_pkg_version,
        _save_teacher_cache,
        _teacher_cache_key,
    ):
        assert callable(fn)


def test_monolith_reexports_pattern_a_functions():
    """The monolith re-exports the SAME relocated FUNCTION objects.

    Proves the ``# noqa: F401`` re-import block in ``stage6_validate.py`` keeps
    ``run()`` and external callers/tests (notably
    ``test_teacher_eval_cache_key_invariant``) on their original import path.
    Only the 5 functions are ``is``-identity checked (the int constant is an
    immutable re-import).
    """
    from moe_compress import stage6_validate
    from moe_compress.stage6.plugins import teacher_provider

    for name in (
        "_safe_pkg_version",
        "_teacher_cache_key",
        "_load_teacher_cache",
        "_save_teacher_cache",
        "_preload_teacher_to_cpu",
    ):
        assert getattr(stage6_validate, name) is getattr(teacher_provider, name), (
            f"monolith re-export mismatch for {name}"
        )


def test_plugin_satisfies_protocol():
    """``TeacherProviderPlugin`` structurally satisfies ``PipelinePlugin``."""
    from moe_compress.pipeline.plugin import PipelinePlugin
    from moe_compress.stage6.plugins.teacher_provider import TeacherProviderPlugin

    assert isinstance(TeacherProviderPlugin(), PipelinePlugin)


def test_plugin_metadata():
    """Plugin metadata — name / config_key / tuple-typed fields / writes slots."""
    from moe_compress.stage6.plugins.teacher_provider import TeacherProviderPlugin

    plugin = TeacherProviderPlugin()
    assert plugin.name == "teacher_provider"
    assert plugin.config_key == "stage6_validate.teacher_eval_cache"
    assert isinstance(plugin.reads, tuple)
    assert isinstance(plugin.writes, tuple)
    assert isinstance(plugin.provides, tuple)
    assert plugin.provides == ()
    for slot in ("teacher_results", "teacher_param_counts"):
        assert slot in plugin.writes, f"missing writes slot: {slot}"


def test_plugin_is_enabled_unconditional():
    """Teacher-side is UNCONDITIONAL — ``is_enabled`` always True.

    Every Stage 6 run must produce teacher metrics (either from the cache or
    from a fresh teacher load + eval); ``config_key`` only names where the
    cache lives, it never gates the plugin as a whole. The hook itself
    contains the internal cache-hit shortcut and the per-sub-metric
    ``enabled`` guards.
    """
    from moe_compress.stage6.plugins.teacher_provider import TeacherProviderPlugin

    plugin = TeacherProviderPlugin()
    assert plugin.is_enabled({}) is True
    assert plugin.is_enabled({"stage6_validate": {}}) is True


def test_plugin_has_provide_teacher_side_hook():
    """The S6-8 phase hook ``provide_teacher_side`` is present and callable."""
    from moe_compress.stage6.plugins.teacher_provider import TeacherProviderPlugin

    plugin = TeacherProviderPlugin()
    assert callable(getattr(plugin, "provide_teacher_side", None))


def test_no_monolith_import_at_any_scope():
    """The module never imports ``stage6_validate`` / ``stage6.orchestrator``.

    The module docstring's contract says NEVER import the monolith (or the
    orchestrator) at any scope — module-top OR function-local — since either
    would risk an import cycle (the monolith re-imports *this* module at load
    time). Parse the source with ``ast`` and walk the FULL tree so a
    function-local ``import stage6_validate`` cannot slip past.

    For ``ImportFrom`` both halves are checked: ``node.module`` (the
    ``from X import ...`` package) AND ``node.names`` (the imported symbols)
    — so the cycle-causing ``from moe_compress import stage6_validate`` form
    (``module="moe_compress"``, name ``stage6_validate``) is also caught.

    Each alias's ``asname`` is checked alongside its ``name`` so a renamed
    import (``import stage6_validate as x`` or ``from m import y as
    orchestrator``) cannot slip past the name check either.
    """
    from moe_compress.stage6.plugins import teacher_provider as mod

    src = Path(mod.__file__).read_text()
    tree = ast.parse(src)
    forbidden = ("stage6_validate", "stage6.orchestrator", "orchestrator")
    for node in ast.walk(tree):  # any nesting level, not just module-top
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
            # Also inspect the imported NAMES: ``from moe_compress import
            # stage6_validate`` carries the monolith as an ``alias.name``, not
            # in ``node.module`` — without this it would slip past undetected.
            # The ``asname`` is checked too so a ``from m import y as
            # orchestrator`` rename is caught.
            for alias in node.names:
                assert not any(
                    f in alias.name or f in (alias.asname or "")
                    for f in forbidden
                ), (
                    f"forbidden import-from name at any scope: "
                    f"from {mod_name} import {alias.name}"
                )


def test_teacher_cache_key_deterministic(tiny_config):
    """``_teacher_cache_key`` returns the same SHA for the same config."""
    from moe_compress.stage6.plugins.teacher_provider import _teacher_cache_key

    k1 = _teacher_cache_key(tiny_config)
    k2 = _teacher_cache_key(tiny_config)
    assert k1 == k2


def test_teacher_cache_key_is_string_of_length_64(tiny_config):
    """``_teacher_cache_key`` returns a SHA-256 hex digest (64 hex chars)."""
    from moe_compress.stage6.plugins.teacher_provider import _teacher_cache_key

    k = _teacher_cache_key(tiny_config)
    assert isinstance(k, str)
    assert len(k) == 64
    int(k, 16)  # raises if not hex


def test_load_teacher_cache_miss_no_file(tmp_path):
    """``_load_teacher_cache`` returns ``None`` when the cache file is absent."""
    from moe_compress.stage6.plugins.teacher_provider import _load_teacher_cache

    cache_path = tmp_path / "teacher_eval_cache.json"
    assert not cache_path.exists()
    assert _load_teacher_cache(cache_path, "any-key") is None


def test_load_teacher_cache_key_mismatch(tmp_path):
    """``_load_teacher_cache`` returns ``None`` on a key mismatch.

    Writes a valid JSON payload with a known cache_key, then asks for a
    different cache_key — the loader must reject (returns ``None``) and not
    return the stale results.
    """
    from moe_compress.stage6.plugins.teacher_provider import (
        TEACHER_CACHE_FORMAT_VERSION,
        _load_teacher_cache,
    )

    cache_path = tmp_path / "teacher_eval_cache.json"
    cache_path.write_text(json.dumps({
        "cache_key": "stored-key",
        "teacher_results": {"wikitext2_ppl": 1.0},
        "teacher_param_counts": {"total": 1, "expert": 1},
        "format_version": TEACHER_CACHE_FORMAT_VERSION,
    }))
    assert _load_teacher_cache(cache_path, "different-key") is None


def test_load_teacher_cache_version_mismatch(tmp_path):
    """``_load_teacher_cache`` returns ``None`` on a format_version mismatch.

    A future schema bump (e.g. ``format_version=999``) must trigger
    re-evaluation rather than silently feed wrong values from an older
    on-disk format.
    """
    from moe_compress.stage6.plugins.teacher_provider import _load_teacher_cache

    cache_path = tmp_path / "teacher_eval_cache.json"
    cache_path.write_text(json.dumps({
        "cache_key": "k",
        "teacher_results": {"wikitext2_ppl": 1.0},
        "teacher_param_counts": {"total": 1, "expert": 1},
        "format_version": 999,
    }))
    assert _load_teacher_cache(cache_path, "k") is None


def test_load_teacher_cache_hit(tmp_path):
    """``_load_teacher_cache`` returns a dict with the right shape on a HIT."""
    from moe_compress.stage6.plugins.teacher_provider import (
        TEACHER_CACHE_FORMAT_VERSION,
        _load_teacher_cache,
    )

    cache_path = tmp_path / "teacher_eval_cache.json"
    payload = {
        "cache_key": "k",
        "teacher_results": {"wikitext2_ppl": 1.2345},
        "teacher_param_counts": {"total": 42, "expert": 10},
        "format_version": TEACHER_CACHE_FORMAT_VERSION,
    }
    cache_path.write_text(json.dumps(payload))
    loaded = _load_teacher_cache(cache_path, "k")
    assert loaded is not None
    assert loaded["results"] == {"wikitext2_ppl": 1.2345}
    assert loaded["param_counts"] == {"total": 42, "expert": 10}


def test_save_teacher_cache_roundtrip(tmp_path):
    """``_save_teacher_cache`` then ``_load_teacher_cache`` round-trips cleanly."""
    from moe_compress.stage6.plugins.teacher_provider import (
        _load_teacher_cache,
        _save_teacher_cache,
    )

    cache_path = tmp_path / "teacher_eval_cache.json"
    results = {"wikitext2_ppl": 2.5, "humaneval_pass_at_1": 0.7}
    pc = {"total": 100, "expert": 30}
    _save_teacher_cache(cache_path, "k1", results, teacher_param_counts=pc)
    assert cache_path.exists()
    loaded = _load_teacher_cache(cache_path, "k1")
    assert loaded is not None
    assert loaded["results"] == results
    assert loaded["param_counts"] == pc


def test_save_teacher_cache_atomic_write(tmp_path):
    """After a successful save no ``.tmp`` file remains.

    ``_save_teacher_cache`` writes to ``<file>.json.tmp`` then ``os.replace``s
    onto the target. After success the temp file must NOT exist (replace
    moves it, no copy left behind).
    """
    from moe_compress.stage6.plugins.teacher_provider import _save_teacher_cache

    cache_path = tmp_path / "teacher_eval_cache.json"
    _save_teacher_cache(cache_path, "k", {"wikitext2_ppl": 1.0})
    tmp_file = cache_path.with_suffix(cache_path.suffix + ".tmp")
    assert cache_path.exists()
    assert not tmp_file.exists()


def test_preload_teacher_to_cpu_4bit_skips():
    """``_preload_teacher_to_cpu`` short-circuits when ``load_in_4bit=True``.

    4-bit quantisation requires CUDA, so the CPU preload would crash. The
    function must return early without touching the queue (no put_nowait,
    no network, no load_model call).
    """
    import queue as _queue

    from moe_compress.stage6.plugins.teacher_provider import (
        _preload_teacher_to_cpu,
    )

    q: _queue.Queue = _queue.Queue(maxsize=1)
    config = {"model": {"load_in_4bit": True, "name_or_path": "tiny"}}
    _preload_teacher_to_cpu(config, q)  # must NOT raise, must NOT load anything
    assert q.empty()


def test_provide_teacher_side_hook_cache_hit(tmp_path, monkeypatch, tiny_config):
    """Cache-HIT path short-circuits before any eval function is called.

    With ``cached_teacher_results`` pre-set on the ctx, the hook MUST take
    the early-return path that writes ``teacher_results`` /
    ``teacher_param_counts`` and return without calling any of the four
    teacher-side eval functions. We monkeypatch all 4 evals on their plugin
    modules to raise — if the hook accidentally invokes one, the test
    fails loudly.
    """
    from moe_compress.pipeline.context import PipelineContext
    from moe_compress.stage6.plugins import (
        humaneval as _humaneval_mod,
        math500 as _math500_mod,
        teacher_provider as _tp_mod,
        wikitext_ppl as _wt_mod,
        zero_shot_lm_eval as _zs_mod,
    )

    def _explode(*a, **kw):  # noqa: ANN001, ANN003
        raise AssertionError(
            "teacher-side eval function was called on the cache-HIT path"
        )

    monkeypatch.setattr(_wt_mod, "_wikitext2_ppl", _explode)
    monkeypatch.setattr(_zs_mod, "_lm_eval_tasks", _explode)
    monkeypatch.setattr(_humaneval_mod, "_humaneval", _explode)
    monkeypatch.setattr(_math500_mod, "_math500", _explode)
    # The hook imported the four eval helpers into its own module namespace
    # via ``from .wikitext_ppl import _wikitext2_ppl`` etc.; the bindings
    # the hook actually invokes are those in ``_tp_mod``. Patch them too so
    # an accidental invocation still trips the explode().
    monkeypatch.setattr(_tp_mod, "_wikitext2_ppl", _explode)
    monkeypatch.setattr(_tp_mod, "_lm_eval_tasks", _explode)
    monkeypatch.setattr(_tp_mod, "_humaneval", _explode)
    monkeypatch.setattr(_tp_mod, "_math500", _explode)

    cached_results = {"wikitext2_ppl": 3.14, "humaneval_pass_at_1": 0.42}
    cached_pc = {"total": 99, "expert": 33}

    plugin = _tp_mod.TeacherProviderPlugin()
    ctx = PipelineContext()
    ctx.set("config", tiny_config)
    ctx.set("tokenizer", object())  # unused on cache-hit path
    ctx.set("artifacts_dir", tmp_path)
    ctx.set("cached_teacher_results", cached_results)
    ctx.set("cached_teacher_param_counts", cached_pc)

    plugin.provide_teacher_side(ctx)

    assert ctx.get("teacher_results") is cached_results
    assert ctx.get("teacher_param_counts") is cached_pc
