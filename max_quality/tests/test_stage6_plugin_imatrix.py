"""S6-6 — Stage 6 imatrix-export plugin extraction tests.

Verifies the S6-6 ``ImatrixExportPlugin`` scaffolding in
``stage6/plugins/imatrix_export.py``:

* the 5 Pattern-A function symbols (``_background_gguf_convert``,
  ``_write_eval_text_concat``, ``_run_llama_imatrix_with_prebuilt_gguf``,
  ``_generate_imatrix``, ``_find_llama_cpp_dir``) + the module-local
  ``_EVAL_TEXT_CONCAT_FILENAME`` constant import from the plugin module;
* the ``stage6_validate`` monolith re-exports the SAME relocated function
  objects (the ``# noqa: F401`` re-import block is load-bearing);
* ``ImatrixExportPlugin`` satisfies the universal ``PipelinePlugin``
  Protocol, carries correct metadata, is gated on
  ``stage6_validate.imatrix.enabled`` and exposes the (S6-8)
  ``start_gguf_convert`` + ``export_imatrix`` phase hooks;
* the module never imports the ``stage6_validate`` monolith or
  ``stage6.orchestrator`` at any scope (the circular-import contract);
* the relocated ``_write_eval_text_concat`` produces the file with the
  correct name and contents (deterministic non-subprocess code path);
* the inert hooks short-circuit on the disabled-imatrix and
  no-gguf-thread paths without spawning any subprocess.

S6-6 covers a MIXED pattern: the 5 standalone functions + 1 constant are
relocated verbatim (the monolith re-imports the 5 functions; the constant
is module-local in the plugin since only relocated functions reference
it); the imatrix kickoff + late-phase dispatch — both inline ``run()``
code in the monolith — is reproduced in the inert ``start_gguf_convert``
and ``export_imatrix`` hooks (the monolith ``run()`` is NOT modified for
them). The byte-identical behavioral gate is the S6-0 golden snapshot
(``test_stage6_golden_snapshot.py``); this file only checks the
relocation plumbing and the relocated logic.
"""
from __future__ import annotations

import ast
import subprocess as _subprocess_mod
from pathlib import Path

import pytest

try:
    import torch  # noqa: F401
    from moe_compress.stage6.plugins import imatrix_export  # noqa: F401
except Exception as e:  # pragma: no cover - import-time guard
    pytest.skip(
        f"Stage 6 imatrix-export imports unavailable: {e}",
        allow_module_level=True,
    )


def test_imatrix_export_module_imports():
    """All 5 Pattern-A functions + the constant + ``ImatrixExportPlugin`` import."""
    from moe_compress.stage6.plugins.imatrix_export import (
        ImatrixExportPlugin,
        _EVAL_TEXT_CONCAT_FILENAME,
        _background_gguf_convert,
        _find_llama_cpp_dir,
        _generate_imatrix,
        _run_llama_imatrix_with_prebuilt_gguf,
        _write_eval_text_concat,
    )

    assert isinstance(ImatrixExportPlugin, type)
    # Constant verbatim copy of the monolith value.
    assert _EVAL_TEXT_CONCAT_FILENAME == "eval_text_concat.txt"
    for fn in (
        _background_gguf_convert,
        _find_llama_cpp_dir,
        _generate_imatrix,
        _run_llama_imatrix_with_prebuilt_gguf,
        _write_eval_text_concat,
    ):
        assert callable(fn)


def test_monolith_reexports_pattern_a_functions():
    """The monolith re-exports the SAME relocated FUNCTION objects.

    Proves the ``# noqa: F401`` re-import block in ``stage6_validate.py``
    keeps ``run()`` and external callers/tests on their original import
    path. Only the 5 functions are ``is``-identity checked — the str
    constant is intentionally NOT re-imported by the monolith (only the
    relocated ``_write_eval_text_concat`` references it, and that
    function resolves it from the plugin module's module-local copy).
    """
    from moe_compress import stage6_validate
    from moe_compress.stage6.plugins import imatrix_export

    for name in (
        "_background_gguf_convert",
        "_write_eval_text_concat",
        "_run_llama_imatrix_with_prebuilt_gguf",
        "_generate_imatrix",
        "_find_llama_cpp_dir",
    ):
        assert getattr(stage6_validate, name) is getattr(imatrix_export, name), (
            f"monolith re-export mismatch for {name}"
        )
    # The constant must NOT be re-exported on the monolith (S6-6 contract):
    # surviving run() code never references it directly.
    assert not hasattr(stage6_validate, "_EVAL_TEXT_CONCAT_FILENAME"), (
        "_EVAL_TEXT_CONCAT_FILENAME should NOT be re-exported on the "
        "monolith — only the relocated _write_eval_text_concat references "
        "it, and it resolves from the plugin module's module-local copy."
    )


def test_plugin_satisfies_protocol():
    """``ImatrixExportPlugin`` structurally satisfies ``PipelinePlugin``."""
    from moe_compress.pipeline.plugin import PipelinePlugin
    from moe_compress.stage6.plugins.imatrix_export import ImatrixExportPlugin

    assert isinstance(ImatrixExportPlugin(), PipelinePlugin)


def test_plugin_metadata():
    """Plugin metadata — name / config_key / tuple-typed fields / writes slots."""
    from moe_compress.stage6.plugins.imatrix_export import ImatrixExportPlugin

    plugin = ImatrixExportPlugin()
    assert plugin.name == "imatrix_export"
    assert plugin.config_key == "stage6_validate.imatrix.enabled"
    assert isinstance(plugin.reads, tuple)
    assert isinstance(plugin.writes, tuple)
    assert isinstance(plugin.provides, tuple)
    assert plugin.provides == ()
    for slot in ("gguf_thread", "gguf_result", "imatrix_skipped"):
        assert slot in plugin.writes, f"missing writes slot: {slot}"


def test_plugin_is_enabled_gating():
    """``is_enabled`` defaults to ``True`` to match the monolith's per-call-site
    ``s6.get("imatrix", {}).get("enabled", True)`` default — a Stage 6 config
    that omits the ``imatrix`` subdict triggers the pipeline exactly as the
    monolith does. Explicit ``False`` disables it."""
    from moe_compress.stage6.plugins.imatrix_export import ImatrixExportPlugin

    plugin = ImatrixExportPlugin()
    # Default: empty config / missing subdict → True (matches monolith default).
    assert plugin.is_enabled({}) is True
    assert plugin.is_enabled({"stage6_validate": {}}) is True
    assert plugin.is_enabled({"stage6_validate": {"imatrix": {}}}) is True
    # Explicit True.
    assert (
        plugin.is_enabled({"stage6_validate": {"imatrix": {"enabled": True}}}) is True
    )
    # Explicit False.
    assert (
        plugin.is_enabled({"stage6_validate": {"imatrix": {"enabled": False}}}) is False
    )


def test_plugin_has_both_phase_hooks():
    """Both S6-8 phase hooks (``start_gguf_convert`` + ``export_imatrix``) present."""
    from moe_compress.stage6.plugins.imatrix_export import ImatrixExportPlugin

    plugin = ImatrixExportPlugin()
    assert callable(getattr(plugin, "start_gguf_convert", None))
    assert callable(getattr(plugin, "export_imatrix", None))


def test_no_monolith_import_at_any_scope():
    """The module never imports ``stage6_validate`` / ``stage6.orchestrator``.

    The module docstring's contract says NEVER import the monolith (or
    the orchestrator) at any scope — module-top OR function-local — since
    either would risk an import cycle (the monolith re-imports *this*
    module at load time). Parse the source with ``ast`` and walk the FULL
    tree so a function-local ``import stage6_validate`` cannot slip past.

    For ``ImportFrom`` both halves are checked: ``node.module`` (the
    ``from X import ...`` package) AND ``node.names`` (the imported
    symbols) — so the cycle-causing
    ``from moe_compress import stage6_validate`` form is also caught.

    Each alias's ``asname`` is checked alongside its ``name`` so a
    renamed import cannot slip past the name check either.
    """
    from moe_compress.stage6.plugins import imatrix_export as mod

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
            for alias in node.names:
                assert not any(
                    f in alias.name or f in (alias.asname or "")
                    for f in forbidden
                ), (
                    f"forbidden import-from name at any scope: "
                    f"from {mod_name} import {alias.name}"
                )


def test_write_eval_text_concat_writes_named_file(tmp_path):
    """``_write_eval_text_concat`` writes the exact filename + joined content.

    Deterministic non-subprocess code path: pass a small list of strings,
    confirm the destination file exists at
    ``<artifacts_dir>/eval_text_concat.txt``, the empty/whitespace entries
    are dropped, and the surviving ones are joined with ``\\n\\n``.
    """
    from moe_compress.stage6.plugins.imatrix_export import (
        _EVAL_TEXT_CONCAT_FILENAME,
        _write_eval_text_concat,
    )

    texts = ["alpha", "", "  ", "beta\n", "  gamma  "]
    out = _write_eval_text_concat(texts, tmp_path)
    assert out == tmp_path / _EVAL_TEXT_CONCAT_FILENAME
    assert out.exists()
    body = out.read_text()
    # Each non-empty entry stripped, joined with a blank line.
    assert body == "alpha\n\nbeta\n\ngamma"


def test_find_llama_cpp_dir_no_candidates(monkeypatch):
    """``_find_llama_cpp_dir`` returns ``None`` when nothing resolves.

    Clear ``LLAMA_CPP_DIR``, stub ``shutil.which`` to return ``None`` and
    pass no override — the function should fall through all three
    candidate sources and return ``None`` (no exception, no false positive).
    """
    import shutil

    from moe_compress.stage6.plugins.imatrix_export import _find_llama_cpp_dir

    monkeypatch.delenv("LLAMA_CPP_DIR", raising=False)
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    assert _find_llama_cpp_dir(None) is None


def test_start_gguf_convert_short_circuits_when_disabled(tmp_path, monkeypatch):
    """``start_gguf_convert`` honours the ``imatrix.enabled=False`` short-circuit.

    With ``imatrix.enabled=False`` the hook must NOT spawn a thread or
    invoke subprocess. We monkeypatch ``subprocess.run`` to raise so an
    accidental invocation crashes the test, then assert
    ``ctx.gguf_thread is None``.
    """
    from moe_compress.pipeline.context import PipelineContext
    from moe_compress.stage6.plugins import imatrix_export as _ie_mod
    from moe_compress.stage6.plugins.imatrix_export import ImatrixExportPlugin

    def _explode(*_a, **_kw):
        raise AssertionError("subprocess was invoked on the disabled-imatrix path")

    monkeypatch.setattr(_subprocess_mod, "run", _explode)
    monkeypatch.setattr(_ie_mod.subprocess, "run", _explode)

    config = {"stage6_validate": {"imatrix": {"enabled": False}}}
    plugin = ImatrixExportPlugin()
    ctx = PipelineContext()
    ctx.set("config", config)
    ctx.set("artifacts_dir", tmp_path)

    plugin.start_gguf_convert(ctx)

    assert ctx.get("gguf_thread") is None
    assert ctx.get("gguf_result") == {}


def test_export_imatrix_no_thread_invokes_sequential_fallback(tmp_path, monkeypatch):
    """``export_imatrix`` with no gguf_thread + cache-MISS → sequential fallback.

    When ``gguf_thread`` is None (start hook short-circuited) AND
    ``cached_teacher_results`` is None (no cache hit) AND ``f16_path`` is
    missing, the hook should land in the final ``else:`` branch and call
    ``_generate_imatrix`` — whose internal ``enabled`` guard then
    short-circuits because ``imatrix.enabled=False``. The test stubs
    ``_generate_imatrix`` on the plugin module to record the call and
    verifies ``imatrix_skipped`` is NOT set (since the hook took the
    non-skip branch).
    """
    from moe_compress.pipeline.context import PipelineContext
    from moe_compress.stage6.plugins import imatrix_export as _ie_mod
    from moe_compress.stage6.plugins.imatrix_export import ImatrixExportPlugin

    calls: list[tuple] = []

    def _record(eval_text_concat, icfg, artifacts_dir):  # noqa: ANN001
        calls.append((list(eval_text_concat), dict(icfg), Path(artifacts_dir)))

    monkeypatch.setattr(_ie_mod, "_generate_imatrix", _record)

    config = {"stage6_validate": {"imatrix": {"enabled": False}}}
    plugin = ImatrixExportPlugin()
    ctx = PipelineContext()
    ctx.set("config", config)
    ctx.set("artifacts_dir", tmp_path)
    ctx.set("eval_text_concat", ["hi"])
    # No gguf_thread set → defaults to None; no cached_teacher_results
    # set → defaults to None (cache MISS path).

    plugin.export_imatrix(ctx)

    assert len(calls) == 1
    assert calls[0][0] == ["hi"]
    # The icfg dict passed in is exactly s6["imatrix"] — guards against a
    # regression where the hook accidentally passes an empty dict or the
    # wrong sub-key.
    assert calls[0][1] == {"enabled": False}
    assert calls[0][2] == tmp_path
    # imatrix_skipped should NOT be on ctx (the skip branch is only taken
    # when the bg thread is still alive after the timeout).
    assert not ctx.has("imatrix_skipped")


def test_export_imatrix_timed_out_thread_sets_skipped_sentinel(tmp_path, monkeypatch):
    """``export_imatrix`` with a still-alive gguf_thread → ``imatrix_skipped=True``.

    Simulate the F-CR2-M-1 timeout race: a fake ``gguf_thread`` whose
    ``join()`` returns immediately but ``is_alive()`` returns True. The
    hook must set ``ctx.imatrix_skipped=True``, write the
    ``eval_text_concat.txt`` debug artifact, and MUST NOT invoke any
    subprocess. We monkeypatch ``subprocess.run`` to raise, plus stub the
    plugin-module-local trackio shim to a no-op so the test stays offline.
    """
    from moe_compress.pipeline.context import PipelineContext
    from moe_compress.stage6.plugins import imatrix_export as _ie_mod
    from moe_compress.stage6.plugins.imatrix_export import (
        _EVAL_TEXT_CONCAT_FILENAME,
        ImatrixExportPlugin,
    )

    def _explode(*_a, **_kw):
        raise AssertionError("subprocess was invoked on the imatrix-skipped path")

    monkeypatch.setattr(_subprocess_mod, "run", _explode)
    monkeypatch.setattr(_ie_mod.subprocess, "run", _explode)
    # Silence trackio so the test stays self-contained.
    monkeypatch.setattr(_ie_mod, "_trackio_log", lambda _d: None)
    # Also make _generate_imatrix and _run_llama_imatrix_with_prebuilt_gguf
    # blow up so any accidental fall-through is caught.
    monkeypatch.setattr(_ie_mod, "_generate_imatrix", _explode)
    monkeypatch.setattr(
        _ie_mod, "_run_llama_imatrix_with_prebuilt_gguf", _explode
    )

    class _FakeThread:
        def join(self, timeout=None):  # noqa: ARG002
            return None

        def is_alive(self):
            return True

    plugin = ImatrixExportPlugin()
    ctx = PipelineContext()
    ctx.set(
        "config",
        {"stage6_validate": {"imatrix": {"enabled": True}}},
    )
    ctx.set("artifacts_dir", tmp_path)
    ctx.set("eval_text_concat", ["one", "two"])
    ctx.set("gguf_thread", _FakeThread())
    ctx.set("gguf_result", {})

    plugin.export_imatrix(ctx)

    assert ctx.get("imatrix_skipped") is True
    # The eval-text concat debug artifact must be written on the
    # skipped-imatrix path.
    out = tmp_path / _EVAL_TEXT_CONCAT_FILENAME
    assert out.exists()
    assert out.read_text() == "one\n\ntwo"


def test_background_gguf_convert_skips_when_disabled(tmp_path, monkeypatch):
    """``_background_gguf_convert`` honours its ``icfg.enabled=False`` guard.

    Deterministic non-subprocess code path: pass ``icfg={"enabled": False}``
    and an empty result dict. The function must return immediately
    without invoking subprocess (so monkeypatching it to raise is safe)
    and without touching the result dict.
    """
    from moe_compress.stage6.plugins import imatrix_export as _ie_mod

    def _explode(*_a, **_kw):
        raise AssertionError("subprocess was invoked on the disabled-icfg path")

    monkeypatch.setattr(_subprocess_mod, "run", _explode)
    monkeypatch.setattr(_ie_mod.subprocess, "run", _explode)

    result: dict = {}
    _ie_mod._background_gguf_convert({"enabled": False}, tmp_path, result)
    assert result == {}


def test_generate_imatrix_writes_concat_then_skips_when_disabled(tmp_path, monkeypatch):
    """``_generate_imatrix`` writes eval_text_concat unconditionally, then skips.

    Deterministic non-subprocess code path: with ``icfg.enabled=False``,
    ``_generate_imatrix`` MUST still call ``_write_eval_text_concat`` (the
    debug side-channel is unconditional per spec §9), then return without
    invoking subprocess.
    """
    from moe_compress.stage6.plugins import imatrix_export as _ie_mod
    from moe_compress.stage6.plugins.imatrix_export import (
        _EVAL_TEXT_CONCAT_FILENAME,
        _generate_imatrix,
    )

    def _explode(*_a, **_kw):
        raise AssertionError("subprocess was invoked on the disabled-icfg path")

    monkeypatch.setattr(_subprocess_mod, "run", _explode)
    monkeypatch.setattr(_ie_mod.subprocess, "run", _explode)

    _generate_imatrix(["hello"], {"enabled": False}, tmp_path)

    out = tmp_path / _EVAL_TEXT_CONCAT_FILENAME
    assert out.exists()
    assert out.read_text() == "hello"
