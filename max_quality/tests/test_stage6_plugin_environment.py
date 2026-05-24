"""S6-2 — Stage 6 eval-environment plugin extraction tests.

Verifies the S6-2 ``EvalEnvironmentPlugin`` scaffolding in
``stage6/plugins/eval_environment.py``:

* the eight Pattern-A symbols (``_CANONICAL_DATASET_REVISION_KEYS``,
  ``_resolve_dataset_revisions``, ``_enforce_revision_pinning``,
  ``_atomic_write_text``, ``_IMATRIX_CALIB_FILENAME``,
  ``_build_imatrix_calibration_corpus``, ``_set_experts_implementation_s6``,
  ``_apply_stage6_kernel_patches``) import from the plugin module;
* the ``stage6_validate`` monolith re-exports the SAME relocated function
  objects (the ``# noqa: F401`` re-import block is load-bearing);
* ``EvalEnvironmentPlugin`` satisfies the universal ``PipelinePlugin``
  Protocol, carries correct metadata, is unconditionally enabled, and exposes
  the (S6-8) ``setup_environment`` phase hook;
* the module never imports the ``stage6_validate`` monolith or
  ``stage6.orchestrator`` at any scope (the circular-import contract);
* the relocated revision-pinning / imatrix-corpus logic behaves correctly;
* the ``setup_environment`` hook populates every ``writes`` ctx slot.

S6-2 covers a MIXED pattern: the eight standalone symbols are relocated
verbatim (the monolith re-imports them); the torch.compile setup and the
ordering glue — inline ``run()`` code in the monolith — are reproduced in the
inert ``setup_environment`` hook (the monolith ``run()`` is NOT modified for
it). The byte-identical behavioral gate is the S6-0 golden snapshot
(``test_stage6_golden_snapshot.py``); this file only checks the relocation
plumbing and the relocated logic.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

try:
    import torch  # noqa: F401
    from moe_compress.stage6.plugins import eval_environment  # noqa: F401
except Exception as e:  # pragma: no cover - import-time guard
    pytest.skip(
        f"Stage 6 eval-environment imports unavailable: {e}",
        allow_module_level=True,
    )


def test_eval_environment_module_imports():
    """All 8 Pattern-A symbols + ``EvalEnvironmentPlugin`` import from the module."""
    from moe_compress.stage6.plugins.eval_environment import (
        EvalEnvironmentPlugin,
        _CANONICAL_DATASET_REVISION_KEYS,
        _IMATRIX_CALIB_FILENAME,
        _apply_stage6_kernel_patches,
        _atomic_write_text,
        _build_imatrix_calibration_corpus,
        _enforce_revision_pinning,
        _resolve_dataset_revisions,
        _set_experts_implementation_s6,
    )

    assert isinstance(EvalEnvironmentPlugin, type)
    assert _CANONICAL_DATASET_REVISION_KEYS == (
        "wikitext_ppl", "humaneval", "math500",
    )
    assert _IMATRIX_CALIB_FILENAME == "calibration_wiki_train.txt"
    for fn in (
        _apply_stage6_kernel_patches,
        _atomic_write_text,
        _build_imatrix_calibration_corpus,
        _enforce_revision_pinning,
        _resolve_dataset_revisions,
        _set_experts_implementation_s6,
    ):
        assert callable(fn)


def test_monolith_reexports_pattern_a_functions():
    """The monolith re-exports the SAME relocated FUNCTION objects.

    Proves the ``# noqa: F401`` re-import block in ``stage6_validate.py`` keeps
    ``run()`` and external callers/tests on their original import path. Only
    the 6 functions are ``is``-identity checked (the 2 constants are immutable
    re-declarations / re-imports).
    """
    from moe_compress import stage6_validate
    from moe_compress.stage6.plugins import eval_environment

    for name in (
        "_resolve_dataset_revisions",
        "_enforce_revision_pinning",
        "_atomic_write_text",
        "_build_imatrix_calibration_corpus",
        "_set_experts_implementation_s6",
        "_apply_stage6_kernel_patches",
    ):
        assert getattr(stage6_validate, name) is getattr(eval_environment, name), (
            f"monolith re-export mismatch for {name}"
        )


def test_plugin_satisfies_protocol():
    """``EvalEnvironmentPlugin`` structurally satisfies ``PipelinePlugin``."""
    from moe_compress.pipeline.plugin import PipelinePlugin
    from moe_compress.stage6.plugins.eval_environment import EvalEnvironmentPlugin

    assert isinstance(EvalEnvironmentPlugin(), PipelinePlugin)


def test_plugin_metadata():
    """Plugin metadata — name / config_key / tuple-typed fields / writes slots."""
    from moe_compress.stage6.plugins.eval_environment import EvalEnvironmentPlugin

    plugin = EvalEnvironmentPlugin()
    assert plugin.name == "eval_environment"
    assert plugin.config_key == "stage6_validate.experts_implementation"
    assert isinstance(plugin.reads, tuple)
    assert isinstance(plugin.writes, tuple)
    assert isinstance(plugin.provides, tuple)
    for slot in (
        "dataset_revisions", "imatrix_calib_path", "use_torch_compile",
        "pre_compile_forward", "experts_impl",
    ):
        assert slot in plugin.writes, f"missing writes slot: {slot}"


def test_plugin_is_enabled_unconditional():
    """Environment setup is UNCONDITIONAL — ``is_enabled`` always True.

    Every Stage 6 run must pin revisions, apply kernel patches, set the
    experts-implementation shim and build the imatrix corpus; ``config_key``
    only names which experts-implementation is used, it never gates the plugin.
    """
    from moe_compress.stage6.plugins.eval_environment import EvalEnvironmentPlugin

    plugin = EvalEnvironmentPlugin()
    assert plugin.is_enabled({}) is True
    assert plugin.is_enabled({"stage6_validate": {}}) is True


def test_plugin_has_setup_environment_hook():
    """The S6-8 phase hook ``setup_environment`` is present and callable."""
    from moe_compress.stage6.plugins.eval_environment import EvalEnvironmentPlugin

    plugin = EvalEnvironmentPlugin()
    assert callable(getattr(plugin, "setup_environment", None))


def test_no_monolith_import_at_any_scope():
    """The module never imports ``stage6_validate`` / ``stage6.orchestrator``.

    The module docstring's contract says NEVER import the monolith (or the
    orchestrator) at any scope — module-top OR function-local — since either
    would risk an import cycle (the monolith re-imports *this* module at load
    time). Parse the source with ``ast`` and walk the FULL tree so a
    function-local ``import stage6_validate`` cannot slip past.

    For ``ImportFrom`` both halves are checked: ``node.module`` (the
    ``from X import ...`` package) AND ``node.names`` (the imported symbols) —
    so the cycle-causing ``from moe_compress import stage6_validate`` form
    (``module="moe_compress"``, name ``stage6_validate``) is also caught.

    Each alias's ``asname`` is checked alongside its ``name`` so a renamed
    import (``import stage6_validate as x`` or ``from m import y as
    orchestrator``) cannot slip past the name check either.
    """
    from moe_compress.stage6.plugins import eval_environment as mod

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


def test_resolve_dataset_revisions_canonical_keys():
    """``_resolve_dataset_revisions`` keeps canonical keys, drops the rest.

    Canonical keys (``wikitext_ppl``/``humaneval``/``math500``) survive; an
    extra key is dropped with a warning; a non-dict ``dataset_revisions``
    value resolves to ``{}``.
    """
    from moe_compress.stage6.plugins.eval_environment import (
        _resolve_dataset_revisions,
    )

    # Canonical keys kept, extra key dropped.
    out = _resolve_dataset_revisions({
        "stage6_validate": {
            "dataset_revisions": {
                "wikitext_ppl": "abc123",
                "humaneval": "def456",
                "math500": None,
                "bogus_key": "should_drop",
            }
        }
    })
    assert out == {"wikitext_ppl": "abc123", "humaneval": "def456", "math500": None}

    # Non-dict dataset_revisions → empty mapping.
    assert _resolve_dataset_revisions(
        {"stage6_validate": {"dataset_revisions": ["not", "a", "dict"]}}
    ) == {}

    # No stage6_validate config at all → empty mapping.
    assert _resolve_dataset_revisions({}) == {}


def test_enforce_revision_pinning_strict_mode():
    """``_enforce_revision_pinning`` raises only when strict + a key missing.

    Non-strict mode returns the resolved dict without raising; strict mode
    raises ``RuntimeError`` when a required key is missing/null and returns
    the dict when every required key is pinned.
    """
    from moe_compress.stage6.plugins.eval_environment import (
        _enforce_revision_pinning,
    )

    # Non-strict: missing keys are fine, no raise.
    out = _enforce_revision_pinning({
        "stage6_validate": {"strict_revision_pinning": False}
    })
    assert isinstance(out, dict)

    # Strict + missing required key → RuntimeError.
    with pytest.raises(RuntimeError, match="strict_revision_pinning"):
        _enforce_revision_pinning({
            "stage6_validate": {
                "strict_revision_pinning": True,
                "dataset_revisions": {"wikitext_ppl": "abc"},  # humaneval/math500 missing
            }
        })

    # Strict + all required keys pinned → returns the dict.
    full = _enforce_revision_pinning({
        "stage6_validate": {
            "strict_revision_pinning": True,
            "dataset_revisions": {
                "wikitext_ppl": "abc", "humaneval": "def", "math500": "ghi",
            },
        }
    })
    assert full == {"wikitext_ppl": "abc", "humaneval": "def", "math500": "ghi"}


def test_build_imatrix_calibration_corpus_reuse(tmp_path):
    """``_build_imatrix_calibration_corpus`` reuses an existing non-empty file.

    A pre-written non-empty ``calibration_wiki_train.txt`` triggers the
    reuse early-return — the function returns the existing path WITHOUT any
    network / ``load_dataset`` call.
    """
    from moe_compress.stage6.plugins.eval_environment import (
        _IMATRIX_CALIB_FILENAME,
        _build_imatrix_calibration_corpus,
    )

    existing = tmp_path / _IMATRIX_CALIB_FILENAME
    existing.write_text("some pre-existing calibration text", encoding="utf-8")

    result = _build_imatrix_calibration_corpus(tmp_path, {})
    assert result == existing
    assert result.exists()
    assert result.stat().st_size > 0


def test_setup_environment_hook_basic(tiny_model, tmp_path):
    """The inert ``setup_environment`` hook populates every ``writes`` ctx slot.

    With ``torch_compile=False`` and ``strict_revision_pinning=False`` the
    hook runs end-to-end on the ``tiny_model`` fixture (CPU-only) and writes
    all 5 declared ctx slots: the experts-impl default, no compile, no
    pre-compile method, a revisions dict and a None-or-Path imatrix path.
    """
    from moe_compress.pipeline.context import PipelineContext
    from moe_compress.stage6.plugins.eval_environment import EvalEnvironmentPlugin

    plugin = EvalEnvironmentPlugin()
    ctx = PipelineContext()
    ctx.set("model", tiny_model)
    ctx.set("config", {
        "stage6_validate": {
            "strict_revision_pinning": False,
            "torch_compile": False,
        }
    })
    # artifacts_dir is mkdir-ed inside the hook (and is where the imatrix
    # corpus would be written); a fresh pytest tmp_path keeps the test
    # hermetic and leaves no trace.
    ctx.set("artifacts_dir", tmp_path)

    plugin.setup_environment(ctx)

    assert ctx.get("experts_impl") == "batched_mm"
    assert ctx.get("use_torch_compile") is False
    assert ctx.get("pre_compile_forward") is None
    assert isinstance(ctx.get("dataset_revisions"), dict)
    imatrix = ctx.get("imatrix_calib_path")
    assert imatrix is None or isinstance(imatrix, Path)


def test_setup_environment_hook_switches_model_to_inference_mode(
    tiny_model, tmp_path,
):
    """Hook step 2 — the model is flipped out of ``train()`` mode.

    Stage 5 leaves the model in training mode; the gate must run every
    sub-metric in inference mode. Put the fixture model explicitly in
    training mode first, then assert the hook flips it.
    """
    from moe_compress.pipeline.context import PipelineContext
    from moe_compress.stage6.plugins.eval_environment import EvalEnvironmentPlugin

    tiny_model.train()
    assert tiny_model.training is True  # precondition: starts in train mode

    plugin = EvalEnvironmentPlugin()
    ctx = PipelineContext()
    ctx.set("model", tiny_model)
    ctx.set("config", {
        "stage6_validate": {
            "strict_revision_pinning": False,
            "torch_compile": False,
        }
    })
    ctx.set("artifacts_dir", tmp_path)

    plugin.setup_environment(ctx)

    assert tiny_model.training is False, (
        "setup_environment must flip the model out of train() mode"
    )


def test_setup_environment_hook_registers_linear_attention_mask(
    tiny_model, tmp_path, monkeypatch,
):
    """Hook step 7 — the ``masking_utils`` kernel patch is genuinely applied.

    The hook registers ``'linear_attention'`` in transformers'
    ``LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING`` (a global dict) so
    ``generate()`` on Qwen3.5-MoE does not raise ``KeyError:
    'linear_attention'``. This is a process-global mutation, so the test
    swaps in a fresh copy of the mapping via ``monkeypatch`` — the patch is
    exercised for real but restored automatically, never leaking to other
    tests.
    """
    from moe_compress.pipeline.context import PipelineContext
    from moe_compress.stage6.plugins.eval_environment import EvalEnvironmentPlugin

    try:
        from transformers import masking_utils as _mu
    except ImportError:
        pytest.skip("transformers.masking_utils unavailable")
    orig = getattr(_mu, "LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING", None)
    if not isinstance(orig, dict) or "full_attention" not in orig:
        pytest.skip("LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING shape unexpected")

    # Fresh isolated copy WITHOUT 'linear_attention' so the patch branch is
    # forced to fire; monkeypatch restores the real global dict on teardown.
    isolated = {k: v for k, v in orig.items() if k != "linear_attention"}
    monkeypatch.setattr(
        _mu, "LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING", isolated,
    )
    assert "linear_attention" not in isolated  # precondition

    plugin = EvalEnvironmentPlugin()
    ctx = PipelineContext()
    ctx.set("model", tiny_model)
    ctx.set("config", {
        "stage6_validate": {
            "strict_revision_pinning": False,
            "torch_compile": False,
        }
    })
    ctx.set("artifacts_dir", tmp_path)

    plugin.setup_environment(ctx)

    mapping = _mu.LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING
    assert "linear_attention" in mapping, (
        "setup_environment must register the 'linear_attention' mask entry"
    )
    assert mapping["linear_attention"] is mapping["full_attention"], (
        "'linear_attention' must map to the same function as 'full_attention'"
    )
