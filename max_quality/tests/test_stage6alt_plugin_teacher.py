"""S6A-4 — Stage 6alt thermometer teacher-provider plugin extraction tests.

Verifies the S6A-4 ``ThermoTeacherProviderPlugin`` scaffolding in
``stage6alt/plugins/thermo_teacher_provider.py``:

* the 4 Pattern-A symbols (``THERMO_TEACHER_CACHE_FORMAT_VERSION``,
  ``_thermo_teacher_cache_key``, ``_load_thermo_teacher_cache``,
  ``_save_thermo_teacher_cache``) import from the plugin module;
* the ``stage6alt_thermometer`` monolith re-exports the SAME relocated
  function objects (the ``# noqa: F401`` re-import block is load-bearing
  — the S6A-0 golden snapshot uses ``monkeypatch.setattr`` against the
  monolith namespace, which only keeps biting if the function objects
  there are ``is``-identical to the plugin-module ones);
* ``ThermoTeacherProviderPlugin`` satisfies the universal
  ``PipelinePlugin`` Protocol, carries the declared metadata, is
  unconditionally enabled, and exposes the (S6A-6)
  ``provide_thermo_teacher_side`` phase hook;
* the module never imports the ``stage6alt_thermometer`` monolith or
  ``stage6alt.orchestrator`` at any scope (the circular-import contract);
* the relocated cache-key / load / save helpers behave correctly:
  ``_thermo_teacher_cache_key`` is deterministic, returns a 64-char hex
  digest, and varies with ``corpus_id`` and the model name;
  ``_load_thermo_teacher_cache`` returns ``None`` on a missing file,
  a key mismatch, and a version mismatch; on a HIT it returns the RAW
  ``teacher_results`` dict (NOT a wrapper carrying ``{"results": ...,
  "param_counts": ...}`` like S6-5's ``_load_teacher_cache``); the
  save/load pair round-trips; the save is atomic (no ``.tmp`` leftover);
* the inert ``provide_thermo_teacher_side`` hook short-circuits on a
  cache hit — neither ``_bpt_from_nll`` nor ``_lm_eval_subset`` is
  invoked, the cache-hit ctx writes happen, and the helper-load is
  bypassed.

S6A-4 covers a MIXED pattern: the 4 standalone symbols are relocated
verbatim (the monolith re-imports them); the teacher block — inline
``run()`` code in the monolith — is reproduced in the inert
``provide_thermo_teacher_side`` hook (the monolith ``run()`` is NOT
modified for it). The byte-identical behavioral gate is the S6A-0
golden snapshot (``test_stage6alt_golden_snapshot.py``); this file
only checks the relocation plumbing and the relocated logic.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

try:
    import torch  # noqa: F401
    from moe_compress.stage6alt.plugins import thermo_teacher_provider  # noqa: F401
except Exception as e:  # pragma: no cover - import-time guard
    pytest.skip(
        f"Stage 6alt thermo_teacher_provider imports unavailable: {e}",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------


def _make_thermo_config(tiny_config) -> dict:
    """Augment ``tiny_config`` with a minimal thermometer sub-tree.

    The teacher-cache key depends on ``stage6_validate.thermometer``
    (arc_easy_limit, hellaswag_limit) plus ``model`` (name_or_path,
    revision, torch_dtype). The tiny_config fixture already provides
    ``model`` and ``stage6_validate``; we just add the thermometer
    sub-dict so the key computation has the limits it expects.
    """
    cfg = dict(tiny_config)
    cfg["stage6_validate"] = dict(cfg["stage6_validate"])
    cfg["stage6_validate"]["thermometer"] = {
        "bpt_batch_size": 4,
        "arc_easy_limit": 5,
        "hellaswag_limit": 7,
        "lm_eval_batch_size": "auto:4",
    }
    return cfg


# ---------------------------------------------------------------------------
# Tests — module imports + Pattern-A re-export identity
# ---------------------------------------------------------------------------


def test_thermo_teacher_provider_module_imports():
    """All 4 Pattern-A symbols + ``ThermoTeacherProviderPlugin`` import."""
    from moe_compress.stage6alt.plugins.thermo_teacher_provider import (
        THERMO_TEACHER_CACHE_FORMAT_VERSION,
        ThermoTeacherProviderPlugin,
        _load_thermo_teacher_cache,
        _save_thermo_teacher_cache,
        _thermo_teacher_cache_key,
    )

    assert isinstance(ThermoTeacherProviderPlugin, type)
    assert THERMO_TEACHER_CACHE_FORMAT_VERSION == 2
    for fn in (
        _load_thermo_teacher_cache,
        _save_thermo_teacher_cache,
        _thermo_teacher_cache_key,
    ):
        assert callable(fn)


def test_monolith_reexports_pattern_a_functions():
    """The monolith re-exports the SAME relocated FUNCTION objects.

    Load-bearing for the S6A-0 golden snapshot: it does
    ``monkeypatch.setattr(stage6alt_thermometer, "_load_thermo_teacher_cache", ...)``
    and friends. That patch-by-attribute only keeps biting if the
    function objects on the monolith namespace are ``is``-identical to
    the relocated ones. Only the 3 functions are ``is``-identity checked
    (the int constant is an immutable re-import).
    """
    from moe_compress import stage6alt_thermometer
    from moe_compress.stage6alt.plugins import thermo_teacher_provider

    for name in (
        "_thermo_teacher_cache_key",
        "_load_thermo_teacher_cache",
        "_save_thermo_teacher_cache",
    ):
        assert getattr(stage6alt_thermometer, name) is getattr(
            thermo_teacher_provider, name
        ), f"monolith re-export mismatch for {name}"


# ---------------------------------------------------------------------------
# Tests — Protocol conformance + metadata + is_enabled + hooks
# ---------------------------------------------------------------------------


def test_plugin_satisfies_protocol():
    """``ThermoTeacherProviderPlugin`` structurally satisfies ``PipelinePlugin``."""
    from moe_compress.pipeline.plugin import PipelinePlugin
    from moe_compress.stage6alt.plugins.thermo_teacher_provider import (
        ThermoTeacherProviderPlugin,
    )

    assert isinstance(ThermoTeacherProviderPlugin(), PipelinePlugin)


def test_plugin_metadata():
    """Plugin metadata — name / config_key / tuple-typed fields / writes slots."""
    from moe_compress.stage6alt.plugins.thermo_teacher_provider import (
        ThermoTeacherProviderPlugin,
    )

    plugin = ThermoTeacherProviderPlugin()
    assert plugin.name == "thermo_teacher_provider"
    assert plugin.config_key == "stage6_validate.thermometer"
    assert isinstance(plugin.reads, tuple)
    assert isinstance(plugin.writes, tuple)
    assert isinstance(plugin.provides, tuple)
    assert plugin.provides == ()
    for slot in (
        "teacher_results",
        "teacher_bpt",
        "teacher_argmax",
        "teacher_cache_hit",
        "teacher_cache_path",
        "teacher_cache_key",
    ):
        assert slot in plugin.writes, f"missing writes slot: {slot}"


def test_plugin_is_enabled_unconditional():
    """Teacher-cache provider is UNCONDITIONAL — ``is_enabled`` always True.

    Every thermometer run must produce teacher results (either from the
    cache or from a fresh teacher load + score). ``config_key`` only
    names the thermometer config sub-tree; it never gates the plugin as
    a whole. The hook itself contains the internal cache-hit shortcut.
    """
    from moe_compress.stage6alt.plugins.thermo_teacher_provider import (
        ThermoTeacherProviderPlugin,
    )

    plugin = ThermoTeacherProviderPlugin()
    assert plugin.is_enabled({}) is True
    assert plugin.is_enabled({"stage6_validate": {}}) is True
    assert plugin.is_enabled({
        "stage6_validate": {"thermometer": {"bpt_batch_size": 4}}
    }) is True


def test_plugin_has_provide_thermo_teacher_side_hook():
    """The S6A-6 phase hook ``provide_thermo_teacher_side`` is present and callable."""
    from moe_compress.stage6alt.plugins.thermo_teacher_provider import (
        ThermoTeacherProviderPlugin,
    )

    plugin = ThermoTeacherProviderPlugin()
    assert callable(getattr(plugin, "provide_thermo_teacher_side", None))


# ---------------------------------------------------------------------------
# Tests — circular-import AST guard
# ---------------------------------------------------------------------------


def test_no_monolith_import_at_any_scope():
    """The module never imports ``stage6alt_thermometer`` / orchestrator.

    The plugin docstring's contract says NEVER import the monolith (or the
    orchestrator) at any scope — module-top OR function-local — since
    either would risk an import cycle (the monolith re-imports *this*
    module at load time). Parse the source with ``ast`` and walk the
    FULL tree so a function-local ``import stage6alt_thermometer``
    cannot slip past.

    For ``ImportFrom`` both halves are checked: ``node.module`` AND
    ``node.names`` — so the cycle-causing
    ``from moe_compress import stage6alt_thermometer`` form is also
    caught. Each alias's ``asname`` is checked alongside its ``name``
    so a renamed import (``import stage6alt_thermometer as x``) cannot
    slip past either.
    """
    from moe_compress.stage6alt.plugins import thermo_teacher_provider as mod

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


# ---------------------------------------------------------------------------
# Tests — relocated cache-key / load / save helpers
# ---------------------------------------------------------------------------


def test_thermo_teacher_cache_key_deterministic(tiny_config):
    """``_thermo_teacher_cache_key`` returns the same SHA for the same input."""
    from moe_compress.stage6alt.plugins.thermo_teacher_provider import (
        _thermo_teacher_cache_key,
    )

    cfg = _make_thermo_config(tiny_config)
    k1 = _thermo_teacher_cache_key(cfg, "corpus-abc")
    k2 = _thermo_teacher_cache_key(cfg, "corpus-abc")
    assert k1 == k2


def test_thermo_teacher_cache_key_hex64_and_varies(tiny_config):
    """``_thermo_teacher_cache_key`` returns 64 hex chars; varies with inputs.

    Two different ``corpus_id`` values, and two different ``model.name_or_path``
    values, must each produce different keys — otherwise a sweep mixing
    corpora or teachers would collide on the same cache entry.
    """
    from moe_compress.stage6alt.plugins.thermo_teacher_provider import (
        _thermo_teacher_cache_key,
    )

    cfg = _make_thermo_config(tiny_config)
    k = _thermo_teacher_cache_key(cfg, "corpus-1")
    assert isinstance(k, str)
    assert len(k) == 64
    int(k, 16)  # raises if not hex

    # Different corpus_id → different key.
    k2 = _thermo_teacher_cache_key(cfg, "corpus-2")
    assert k2 != k

    # Different model.name_or_path → different key.
    cfg2 = dict(cfg)
    cfg2["model"] = dict(cfg["model"])
    cfg2["model"]["name_or_path"] = "other-teacher"
    k3 = _thermo_teacher_cache_key(cfg2, "corpus-1")
    assert k3 != k


def test_load_thermo_teacher_cache_miss_no_file(tmp_path):
    """``_load_thermo_teacher_cache`` returns ``None`` when the file is absent."""
    from moe_compress.stage6alt.plugins.thermo_teacher_provider import (
        _load_thermo_teacher_cache,
    )

    cache_path = tmp_path / "thermometer_teacher_cache.json"
    assert not cache_path.exists()
    assert _load_thermo_teacher_cache(cache_path, "any-key") is None


def test_load_thermo_teacher_cache_key_mismatch(tmp_path):
    """``_load_thermo_teacher_cache`` returns ``None`` on a key mismatch.

    Writes a valid JSON payload with a known cache_key, then asks for a
    different cache_key — the loader must reject (returns ``None``) and not
    return the stale results.
    """
    from moe_compress.stage6alt.plugins.thermo_teacher_provider import (
        THERMO_TEACHER_CACHE_FORMAT_VERSION,
        _load_thermo_teacher_cache,
    )

    cache_path = tmp_path / "thermometer_teacher_cache.json"
    cache_path.write_text(json.dumps({
        "format_version": THERMO_TEACHER_CACHE_FORMAT_VERSION,
        "cache_key": "stored-key",
        "teacher_results": {"teacher_bpt": 1.0},
    }))
    assert _load_thermo_teacher_cache(cache_path, "different-key") is None


def test_load_thermo_teacher_cache_version_mismatch(tmp_path):
    """``_load_thermo_teacher_cache`` returns ``None`` on a version mismatch.

    A future schema bump (e.g. ``format_version=999``) must trigger
    re-evaluation rather than silently feed wrong values from an older
    on-disk format.
    """
    from moe_compress.stage6alt.plugins.thermo_teacher_provider import (
        _load_thermo_teacher_cache,
    )

    cache_path = tmp_path / "thermometer_teacher_cache.json"
    cache_path.write_text(json.dumps({
        "format_version": 999,
        "cache_key": "k",
        "teacher_results": {"teacher_bpt": 1.0},
    }))
    assert _load_thermo_teacher_cache(cache_path, "k") is None


def test_load_thermo_teacher_cache_hit_returns_raw_dict(tmp_path):
    """``_load_thermo_teacher_cache`` returns the RAW teacher_results dict on a HIT.

    Unlike S6-5's ``_load_teacher_cache`` (which wraps the on-disk payload
    as ``{"results": ..., "param_counts": ...}``), the thermometer loader
    returns ``data["teacher_results"]`` directly — no wrapper. The result
    must equal the original dict exactly.
    """
    from moe_compress.stage6alt.plugins.thermo_teacher_provider import (
        THERMO_TEACHER_CACHE_FORMAT_VERSION,
        _load_thermo_teacher_cache,
    )

    cache_path = tmp_path / "thermometer_teacher_cache.json"
    teacher_results = {
        "teacher_bpt": 1.2345,
        "teacher_arc_easy_acc_norm": 0.55,
        "teacher_hellaswag_acc_norm": 0.65,
        "teacher_acc_norm_sum": 1.2,
        "teacher_argmax": None,
    }
    cache_path.write_text(json.dumps({
        "format_version": THERMO_TEACHER_CACHE_FORMAT_VERSION,
        "cache_key": "k",
        "teacher_results": teacher_results,
    }))
    loaded = _load_thermo_teacher_cache(cache_path, "k")
    # Must be the RAW dict, not a wrapper. No "results" or "param_counts"
    # keys are introduced by the loader.
    assert loaded == teacher_results
    assert "results" not in loaded
    assert "param_counts" not in loaded


def test_save_thermo_teacher_cache_roundtrip(tmp_path):
    """``_save_thermo_teacher_cache`` then ``_load_thermo_teacher_cache`` round-trips."""
    from moe_compress.stage6alt.plugins.thermo_teacher_provider import (
        _load_thermo_teacher_cache,
        _save_thermo_teacher_cache,
    )

    cache_path = tmp_path / "thermometer_teacher_cache.json"
    teacher_results = {
        "teacher_bpt": 2.5,
        "teacher_arc_easy_acc_norm": 0.42,
        "teacher_hellaswag_acc_norm": 0.77,
        "teacher_acc_norm_sum": 1.19,
        "teacher_argmax": [[1, 2, 3], [4, 5, 6]],
    }
    _save_thermo_teacher_cache(cache_path, "k1", teacher_results)
    assert cache_path.exists()
    loaded = _load_thermo_teacher_cache(cache_path, "k1")
    assert loaded == teacher_results


def test_save_thermo_teacher_cache_atomic_write(tmp_path):
    """After a successful save no ``.tmp`` file remains.

    ``_save_thermo_teacher_cache`` writes to
    ``<file>.json.tmp`` then ``os.replace``s onto the target. After
    success the temp file must NOT exist (replace moves it; no copy left
    behind).
    """
    from moe_compress.stage6alt.plugins.thermo_teacher_provider import (
        _save_thermo_teacher_cache,
    )

    cache_path = tmp_path / "thermometer_teacher_cache.json"
    _save_thermo_teacher_cache(cache_path, "k", {"teacher_bpt": 1.0})
    tmp_file = cache_path.with_suffix(cache_path.suffix + ".tmp")
    assert cache_path.exists()
    assert not tmp_file.exists()


# ---------------------------------------------------------------------------
# Tests — inert ``provide_thermo_teacher_side`` hook (cache-HIT short-circuit)
# ---------------------------------------------------------------------------


def test_provide_thermo_teacher_side_cache_hit(tmp_path, monkeypatch, tiny_config):
    """Cache-HIT path short-circuits before any teacher-load/eval is performed.

    With ``_load_thermo_teacher_cache`` patched to return a canned hit on
    the plugin module, the hook MUST take the early-return path that
    publishes ``teacher_results`` / ``teacher_cache_hit=True`` /
    ``teacher_cache_path`` / ``teacher_cache_key`` and return without
    calling either of the two teacher-side eval helpers
    (``_bpt_from_nll`` / ``_lm_eval_subset``) or hitting the
    ``load_model`` path. We monkeypatch both eval helpers on the plugin
    module to raise — if the hook accidentally invokes one the test
    fails loudly.
    """
    from moe_compress.pipeline.context import PipelineContext
    from moe_compress.stage6alt.plugins import thermo_teacher_provider as _tp_mod

    def _explode(*a, **kw):  # noqa: ANN001, ANN003
        raise AssertionError(
            "teacher-side eval helper was called on the cache-HIT path"
        )

    # Patch the two eval helpers on the plugin's module namespace so an
    # accidental invocation trips the explode().
    monkeypatch.setattr(_tp_mod, "_bpt_from_nll", _explode)
    monkeypatch.setattr(_tp_mod, "_lm_eval_subset", _explode)

    canned_results = {
        "teacher_bpt": 3.14,
        "teacher_arc_easy_acc_norm": 0.42,
        "teacher_hellaswag_acc_norm": 0.77,
        "teacher_acc_norm_sum": 1.19,
        "teacher_argmax": None,
    }
    monkeypatch.setattr(
        _tp_mod, "_load_thermo_teacher_cache",
        lambda cache_path, cache_key: canned_results,
    )

    cfg = _make_thermo_config(tiny_config)
    plugin = _tp_mod.ThermoTeacherProviderPlugin()
    ctx = PipelineContext()
    ctx.set("config", cfg)
    ctx.set("tokenizer", object())  # unused on cache-hit path
    ctx.set("calib_ids", torch.zeros(2, 4, dtype=torch.long))
    ctx.set("artifacts_dir", tmp_path)
    ctx.set("corpus_id", "corpus-xyz")

    plugin.provide_thermo_teacher_side(ctx)

    assert ctx.get("teacher_results") is canned_results
    assert ctx.get("teacher_cache_hit") is True
    # Cache path/key are derived; just check they are present + correctly
    # typed (Path / str) without re-deriving them here.
    assert isinstance(ctx.get("teacher_cache_path"), Path)
    assert isinstance(ctx.get("teacher_cache_key"), str)
    assert len(ctx.get("teacher_cache_key")) == 64
