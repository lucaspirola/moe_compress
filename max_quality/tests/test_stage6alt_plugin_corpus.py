"""S6A-2 — Stage 6alt thermometer environment + corpus plugin extraction tests.

Verifies the S6A-2 ``ThermoEnvironmentPlugin`` + ``ThermoCorpusPlugin``
scaffolding in ``stage6alt/plugins/thermo_environment.py`` and
``stage6alt/plugins/thermo_corpus.py``:

* the five Pattern-A symbols (``THERMO_SEED_OFFSET``,
  ``_DEFAULT_SUBSET_WEIGHTS``, ``_thermo_corpus_spec``,
  ``_thermo_wikitext_tensor``, ``_build_thermo_corpus``) import from the
  ``thermo_corpus`` plugin module;
* the ``stage6alt_thermometer`` monolith re-exports the SAME relocated
  function objects (the ``# noqa: F401`` re-import block is load-bearing);
* both plugins satisfy the universal ``PipelinePlugin`` Protocol, carry
  correct metadata, are unconditionally enabled, and expose their
  respective S6A-6 phase hooks;
* neither plugin module imports the ``stage6alt_thermometer`` monolith
  or ``stage6alt.orchestrator`` at any scope (the circular-import contract);
* the relocated ``_build_thermo_corpus`` keeps both branches (nemotron +
  wikitext), the invalid-corpus error path, and the ``THERMO_SEED_OFFSET``
  application;
* the ``setup_thermo_environment`` and ``build_corpus`` hooks write the
  declared ctx slots.

S6A-2 covers a MIXED pattern: the five corpus symbols are relocated
verbatim (the monolith re-imports them); the inline environment-setup
block and the corpus-build call site — inline ``run()`` code in the
monolith — are reproduced in the inert ``setup_thermo_environment`` /
``build_corpus`` hooks (the monolith ``run()`` is NOT modified for them).
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
        thermo_corpus,
        thermo_environment,
    )
except Exception as e:  # pragma: no cover - import-time guard
    pytest.skip(
        f"Stage 6alt thermo-corpus / thermo-environment imports unavailable: {e}",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    """Minimal tokenizer: maps every character to a small-vocab token id.

    Mirrors the discipline in ``test_stage6_plugin_wikitext.py`` —
    ``_thermo_wikitext_tensor`` calls ``tokenizer(text, add_special_tokens=...,
    return_tensors=None)["input_ids"]`` and gets back a flat list of ints.
    """

    name_or_path = "fake-tokenizer"

    def __call__(self, text, add_special_tokens=True, return_tensors=None):
        ids = [(ord(c) % 31) + 1 for c in text]
        return {"input_ids": ids}


def _fake_datasets_module(rows):
    """Build a fake ``datasets`` module whose ``load_dataset`` yields ``rows``.

    Each row is a ``{"text": ...}`` dict, matching the wikitext-2-raw-v1 schema
    that ``_thermo_wikitext_tensor`` iterates. The module is injected via
    ``sys.modules`` because the ``from datasets import load_dataset`` inside
    the relocated helper is function-local.
    """
    mod = types.ModuleType("datasets")

    def load_dataset(*_args, **_kwargs):
        return [{"text": r} for r in rows]

    mod.load_dataset = load_dataset
    return mod


def _calib_cfg() -> dict:
    """Minimal ``calibration:`` config slice required by ``spec_from_config``.

    The thermometer's corpus_spec only ever consults this through
    ``spec_from_config`` (nvidia-cascade adapter); the seed knob is what the
    seed-offset test verifies.
    """
    return {
        "calibration": {
            "source": "nvidia-cascade",
            "dataset": "nvidia/Nemotron-Cascade-2-SFT-Data",
            "subset_weights": {
                "math": 0.25, "swe": 0.25, "chat": 0.25, "science": 0.25,
            },
            "seed": 1234,
            "num_sequences": 8,
            "sequence_length": 16,
        }
    }


# ---------------------------------------------------------------------------
# Tests — module imports + Pattern-A re-export identity
# ---------------------------------------------------------------------------


def test_module_imports():
    """All 5 Pattern-A symbols + both plugin classes import from the module."""
    from moe_compress.stage6alt.plugins.thermo_corpus import (
        THERMO_SEED_OFFSET,
        ThermoCorpusPlugin,
        _DEFAULT_SUBSET_WEIGHTS,
        _build_thermo_corpus,
        _thermo_corpus_spec,
        _thermo_wikitext_tensor,
    )
    from moe_compress.stage6alt.plugins.thermo_environment import (
        ThermoEnvironmentPlugin,
    )

    assert isinstance(ThermoCorpusPlugin, type)
    assert isinstance(ThermoEnvironmentPlugin, type)
    assert THERMO_SEED_OFFSET == 715
    assert _DEFAULT_SUBSET_WEIGHTS == {
        "math": 0.35, "swe": 0.25, "chat": 0.25, "science": 0.15,
    }
    for fn in (_build_thermo_corpus, _thermo_corpus_spec, _thermo_wikitext_tensor):
        assert callable(fn)


def test_monolith_reexports_pattern_a_corpus_functions():
    """The monolith re-exports the SAME relocated FUNCTION objects.

    Proves the ``# noqa: F401`` re-import block in
    ``stage6alt_thermometer.py`` keeps ``run()`` and external callers/tests
    (notably ``stage2/orchestrator.py``'s xD calibration that imports
    ``_thermo_wikitext_tensor``) on their original import path. Only the
    3 functions are ``is``-identity checked; the 2 constants are immutable
    re-imports.
    """
    from moe_compress import stage6alt_thermometer
    from moe_compress.stage6alt.plugins import thermo_corpus

    for name in (
        "_thermo_corpus_spec",
        "_thermo_wikitext_tensor",
        "_build_thermo_corpus",
    ):
        assert getattr(stage6alt_thermometer, name) is getattr(thermo_corpus, name), (
            f"monolith re-export mismatch for {name}"
        )


# ---------------------------------------------------------------------------
# Tests — Protocol conformance + metadata + is_enabled + hooks
# ---------------------------------------------------------------------------


def test_plugins_satisfy_protocol():
    """Both plugins structurally satisfy ``PipelinePlugin``."""
    from moe_compress.pipeline.plugin import PipelinePlugin
    from moe_compress.stage6alt.plugins.thermo_corpus import ThermoCorpusPlugin
    from moe_compress.stage6alt.plugins.thermo_environment import (
        ThermoEnvironmentPlugin,
    )

    assert isinstance(ThermoEnvironmentPlugin(), PipelinePlugin)
    assert isinstance(ThermoCorpusPlugin(), PipelinePlugin)


def test_thermo_environment_metadata():
    """ThermoEnvironmentPlugin — name / config_key / tuple-typed fields / writes slots."""
    from moe_compress.stage6alt.plugins.thermo_environment import (
        ThermoEnvironmentPlugin,
    )

    plugin = ThermoEnvironmentPlugin()
    assert plugin.name == "thermo_environment"
    assert plugin.config_key == "stage6_validate.thermometer"
    assert isinstance(plugin.reads, tuple)
    assert isinstance(plugin.writes, tuple)
    assert isinstance(plugin.provides, tuple)
    assert plugin.reads == ("model", "config")
    assert plugin.writes == ("experts_impl",)
    assert plugin.provides == ()


def test_thermo_corpus_metadata():
    """ThermoCorpusPlugin — name / config_key / tuple-typed fields / writes slots."""
    from moe_compress.stage6alt.plugins.thermo_corpus import ThermoCorpusPlugin

    plugin = ThermoCorpusPlugin()
    assert plugin.name == "thermo_corpus"
    assert plugin.config_key == "stage6_validate.thermometer.corpus"
    assert isinstance(plugin.reads, tuple)
    assert isinstance(plugin.writes, tuple)
    assert isinstance(plugin.provides, tuple)
    assert plugin.reads == ("model", "tokenizer", "config", "artifacts_dir")
    assert plugin.writes == ("calib_ids", "corpus_meta", "corpus_id")
    assert plugin.provides == ()


def test_plugins_is_enabled_unconditional():
    """Both plugins are UNCONDITIONALLY enabled — ``is_enabled`` always True.

    Every thermometer run must set up the env and build an eval corpus;
    ``config_key`` only names the relevant config sub-tree, it never gates
    the plugin as a whole.
    """
    from moe_compress.stage6alt.plugins.thermo_corpus import ThermoCorpusPlugin
    from moe_compress.stage6alt.plugins.thermo_environment import (
        ThermoEnvironmentPlugin,
    )

    for cls in (ThermoEnvironmentPlugin, ThermoCorpusPlugin):
        plugin = cls()
        assert plugin.is_enabled({}) is True
        assert plugin.is_enabled({"stage6_validate": {}}) is True
        assert plugin.is_enabled({
            "stage6_validate": {"thermometer": {"corpus": "wikitext"}}
        }) is True


def test_plugins_have_phase_hooks():
    """The S6A-6 phase hooks ``setup_thermo_environment`` / ``build_corpus`` exist."""
    from moe_compress.stage6alt.plugins.thermo_corpus import ThermoCorpusPlugin
    from moe_compress.stage6alt.plugins.thermo_environment import (
        ThermoEnvironmentPlugin,
    )

    env_plugin = ThermoEnvironmentPlugin()
    corpus_plugin = ThermoCorpusPlugin()
    assert callable(getattr(env_plugin, "setup_thermo_environment", None))
    assert callable(getattr(corpus_plugin, "build_corpus", None))


# ---------------------------------------------------------------------------
# Tests — circular-import AST guards
# ---------------------------------------------------------------------------


def _ast_guard_no_monolith_import(mod):
    """Shared AST walk used by the per-module circular-import guards.

    The plugin docstrings forbid importing the ``stage6alt_thermometer``
    monolith (or the ``stage6alt.orchestrator``) at any scope — module-top
    OR function-local — since either would risk an import cycle (the
    monolith re-imports these modules at load time). Parse the source
    with ``ast`` and walk the FULL tree so a function-local
    ``import stage6alt_thermometer`` cannot slip past.

    For ``ImportFrom`` both halves are checked: ``node.module`` AND
    ``node.names`` — so the cycle-causing
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


def test_no_monolith_import_thermo_environment():
    """``thermo_environment`` never imports ``stage6alt_thermometer`` / orchestrator."""
    from moe_compress.stage6alt.plugins import thermo_environment as mod

    _ast_guard_no_monolith_import(mod)


def test_no_monolith_import_thermo_corpus():
    """``thermo_corpus`` never imports ``stage6alt_thermometer`` / orchestrator."""
    from moe_compress.stage6alt.plugins import thermo_corpus as mod

    _ast_guard_no_monolith_import(mod)


# ---------------------------------------------------------------------------
# Tests — relocated corpus-build logic
# ---------------------------------------------------------------------------


def test_build_thermo_corpus_nemotron_path(monkeypatch, tmp_path):
    """Nemotron branch: returns ``(tensor, dict, str)``; meta name + id prefix.

    ``build_calibration_tensor`` is monkey-patched to a stub returning a
    fixed tensor — the test exercises ``_build_thermo_corpus``'s orchestration
    (spec build, cache_dir resolution, meta assembly, id formatting) without
    hitting the real corpus loader.
    """
    from moe_compress.stage6alt.plugins import thermo_corpus
    from moe_compress.stage6alt.plugins.thermo_corpus import _build_thermo_corpus

    fake_calib = torch.zeros(4, 8, dtype=torch.long)
    monkeypatch.setattr(
        thermo_corpus, "build_calibration_tensor",
        lambda tokenizer, spec, *, cache_dir: fake_calib,
    )

    cfg = _calib_cfg()
    cfg["stage6_validate"] = {
        "thermometer": {
            "corpus": "nemotron",
            "num_sequences": 4,
            "sequence_length": 8,
        }
    }

    calib, meta, cid = _build_thermo_corpus(cfg, _FakeTokenizer(), tmp_path)

    assert isinstance(calib, torch.Tensor)
    assert calib.shape == (4, 8)
    assert isinstance(meta, dict)
    assert isinstance(cid, str)
    assert meta["name"] == "nemotron"
    assert meta["seed_offset"] == 715
    assert cid.startswith("nemotron:")


def test_build_thermo_corpus_wikitext_path(monkeypatch, tmp_path):
    """Wikitext branch: returns ``(tensor, dict, str)``; meta name + id prefix.

    Injects a fake ``datasets`` module via ``sys.modules`` because the
    ``from datasets import load_dataset`` inside ``_thermo_wikitext_tensor``
    is function-local. The fake corpus has just enough characters to yield
    one full 8-token chunk after the ``_FakeTokenizer`` maps each char to
    one token id.
    """
    from moe_compress.stage6alt.plugins.thermo_corpus import _build_thermo_corpus

    # 20 chars per row → 200 chars total → 25 full 8-token chunks.
    rows = ["the quick brown fox " for _ in range(10)]
    monkeypatch.setitem(sys.modules, "datasets", _fake_datasets_module(rows))

    cfg = _calib_cfg()
    cfg["stage6_validate"] = {
        "thermometer": {
            "corpus": "wikitext",
            "num_sequences": 4,
            "sequence_length": 8,
        }
    }

    calib, meta, cid = _build_thermo_corpus(cfg, _FakeTokenizer(), tmp_path)

    assert isinstance(calib, torch.Tensor)
    assert calib.dim() == 2
    assert calib.shape[1] == 8
    assert isinstance(meta, dict)
    assert isinstance(cid, str)
    assert meta["name"] == "wikitext"
    assert cid.startswith("wikitext:")


def test_build_thermo_corpus_invalid_corpus(tmp_path):
    """A bogus ``thermometer.corpus`` value raises ``ValueError``."""
    from moe_compress.stage6alt.plugins.thermo_corpus import _build_thermo_corpus

    cfg = _calib_cfg()
    cfg["stage6_validate"] = {"thermometer": {"corpus": "bogus"}}

    with pytest.raises(ValueError, match="thermometer.corpus must be"):
        _build_thermo_corpus(cfg, _FakeTokenizer(), tmp_path)


def test_thermo_corpus_spec_seed_offset():
    """``_thermo_corpus_spec``'s seed differs from the base by ``THERMO_SEED_OFFSET``.

    Bumps the held-out draw away from Stage 2/2.5's training draw so the
    thermometer's eval sequences are disjoint. The implementation routes
    through ``spec_from_config(seed_offset=THERMO_SEED_OFFSET)``.
    """
    from moe_compress.stage6alt.plugins.thermo_corpus import (
        THERMO_SEED_OFFSET,
        _thermo_corpus_spec,
    )
    from moe_compress.utils.calibration import CalibrationSpec, spec_from_config

    cfg = _calib_cfg()
    cfg["stage6_validate"] = {
        "thermometer": {"num_sequences": 4, "sequence_length": 16}
    }

    spec = _thermo_corpus_spec(cfg)
    assert isinstance(spec, CalibrationSpec)

    base_spec = spec_from_config(
        cfg["calibration"],
        num_sequences_override=4,
        sequence_length_override=16,
        seed_offset=0,
    )
    assert spec.seed == (base_spec.seed + THERMO_SEED_OFFSET) % (2**32)


# ---------------------------------------------------------------------------
# Tests — inert phase hooks
# ---------------------------------------------------------------------------


def test_setup_thermo_environment_hook(tiny_model):
    """The inert ``setup_thermo_environment`` hook writes ``experts_impl``.

    Stage 6's ``_set_experts_implementation_s6`` mutates ``model.config``
    in place (sets ``_experts_implementation`` on both the top-level and
    nested ``text_config``); ``_apply_stage6_kernel_patches`` is a no-op on
    a tiny model with no GatedDeltaNet / fla modules. The hook resolves
    the YAML default ``batched_mm`` and writes it to ctx.
    """
    from moe_compress.pipeline.context import PipelineContext
    from moe_compress.stage6alt.plugins.thermo_environment import (
        ThermoEnvironmentPlugin,
    )

    plugin = ThermoEnvironmentPlugin()
    ctx = PipelineContext()
    ctx.set("model", tiny_model)
    ctx.set("config", {"stage6_validate": {}})

    plugin.setup_thermo_environment(ctx)

    assert ctx.get("experts_impl") == "batched_mm"


def test_build_corpus_hook(monkeypatch, tmp_path):
    """The inert ``build_corpus`` hook writes 3 ctx slots from the helper return.

    Patches ``_build_thermo_corpus`` on the ``thermo_corpus`` module to a stub
    returning a fixed ``(tensor, meta, id)`` triple; the hook must lift those
    three values into ``calib_ids`` / ``corpus_meta`` / ``corpus_id`` ctx
    slots without inspecting them further.
    """
    from moe_compress.pipeline.context import PipelineContext
    from moe_compress.stage6alt.plugins import thermo_corpus
    from moe_compress.stage6alt.plugins.thermo_corpus import ThermoCorpusPlugin

    stub_tensor = torch.zeros(2, 4, dtype=torch.long)
    stub_meta = {"name": "stub", "num_sequences": 2, "sequence_length": 4}
    monkeypatch.setattr(
        thermo_corpus, "_build_thermo_corpus",
        lambda *a, **k: (stub_tensor, stub_meta, "stub:id"),
    )

    plugin = ThermoCorpusPlugin()
    ctx = PipelineContext()
    ctx.set("config", {"stage6_validate": {"thermometer": {}}})
    ctx.set("tokenizer", _FakeTokenizer())
    ctx.set("artifacts_dir", tmp_path)

    plugin.build_corpus(ctx)

    assert ctx.get("calib_ids") is stub_tensor
    assert ctx.get("corpus_meta") is stub_meta
    assert ctx.get("corpus_id") == "stub:id"
