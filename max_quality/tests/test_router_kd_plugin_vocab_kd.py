"""RK-4 — Router-KD vocab-KD KD-loss plugin extraction tests.

Verifies the RK-4 ``VocabKdPlugin`` scaffolding in
``router_kd/plugins/vocab_kd.py``:

* the five relocated functions (``_chunked_vocab_kl`` / ``_combine_kd_loss`` /
  ``_log_first_batch_sanity`` / ``_dump_nan_diagnostics`` /
  ``_check_param_sanity``) and ``VocabKdPlugin`` import from the plugin module;
* the ``stage5_router_kd`` monolith re-exports the SAME function objects (the
  ``# noqa: F401`` re-import block is load-bearing — ``test_stage5_merge_repair``
  imports the kernel from the monolith);
* ``VocabKdPlugin`` satisfies the universal ``PipelinePlugin`` Protocol,
  carries correct metadata, is unconditionally enabled, and exposes the (RK-8)
  ``compute_kd_loss`` phase hook;
* the module never imports the ``stage5_router_kd`` monolith or
  ``router_kd.orchestrator`` at any scope (the circular-import contract);
* the relocated KD-loss logic itself: the chunked vocab-KL kernel, the τ²
  temperature scaling, chunk-size invariance, gradient flow, the loss combiner
  flag-off identity, the NaN sanity probes, and the ``compute_kd_loss`` hook.

RK-4 is a PURE Pattern A relocation: the five functions are relocated verbatim
(the monolith re-imports them) and nothing is reproduced inline. The
byte-identical behavioral gate is the RK-0 golden snapshot
(``test_router_kd_golden_snapshot.py``); this file only checks the relocation
plumbing and the relocated logic.
"""
from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest
import torch


def test_vocab_kd_module_imports():
    """All six RK-4 symbols import from the plugin module."""
    from moe_compress.router_kd.plugins.vocab_kd import (
        VocabKdPlugin,
        _chunked_vocab_kl,
        _combine_kd_loss,
        _log_first_batch_sanity,
        _dump_nan_diagnostics,
        _check_param_sanity,
    )

    assert isinstance(VocabKdPlugin, type)
    assert callable(_chunked_vocab_kl)
    assert callable(_combine_kd_loss)
    assert callable(_log_first_batch_sanity)
    assert callable(_dump_nan_diagnostics)
    assert callable(_check_param_sanity)


def test_monolith_reexports_kernel():
    """The monolith re-exports the SAME kernel function objects.

    Proves the ``# noqa: F401`` re-import block in ``stage5_router_kd.py``
    keeps ``run()`` and ``test_stage5_merge_repair.py`` on their original
    import path for the vocab-KL kernel + loss combiner.
    """
    from moe_compress import stage5_router_kd
    from moe_compress.router_kd.plugins import vocab_kd

    assert stage5_router_kd._chunked_vocab_kl is vocab_kd._chunked_vocab_kl
    assert stage5_router_kd._combine_kd_loss is vocab_kd._combine_kd_loss


def test_monolith_reexports_nan_probes():
    """The monolith re-exports the SAME three NaN-probe function objects."""
    from moe_compress import stage5_router_kd
    from moe_compress.router_kd.plugins import vocab_kd

    assert (
        stage5_router_kd._log_first_batch_sanity
        is vocab_kd._log_first_batch_sanity
    )
    assert (
        stage5_router_kd._dump_nan_diagnostics
        is vocab_kd._dump_nan_diagnostics
    )
    assert (
        stage5_router_kd._check_param_sanity
        is vocab_kd._check_param_sanity
    )


def test_plugin_satisfies_protocol():
    """``VocabKdPlugin`` structurally satisfies ``PipelinePlugin``."""
    from moe_compress.pipeline.plugin import PipelinePlugin
    from moe_compress.router_kd.plugins.vocab_kd import VocabKdPlugin

    assert isinstance(VocabKdPlugin(), PipelinePlugin)


def test_plugin_metadata():
    """Plugin metadata — name / paper id / config_key / tuple-typed fields."""
    from moe_compress.router_kd.plugins.vocab_kd import VocabKdPlugin

    plugin = VocabKdPlugin()
    assert plugin.name == "vocab_kd"
    assert "2603.02217" in plugin.paper
    assert plugin.config_key == "stage5_router_kd.kd_temperature"
    assert isinstance(plugin.reads, tuple)
    assert isinstance(plugin.writes, tuple)
    assert isinstance(plugin.provides, tuple)
    assert "kd_loss" in plugin.writes
    assert "vocab_kl" in plugin.writes


def test_plugin_is_enabled_unconditional():
    """Computing the KD loss is UNCONDITIONAL — ``is_enabled`` always True.

    Every Router-KD run distills via the vocab-KL loss; ``config_key`` only
    names the distillation temperature, it never gates the plugin as a whole.
    """
    from moe_compress.router_kd.plugins.vocab_kd import VocabKdPlugin

    plugin = VocabKdPlugin()
    assert plugin.is_enabled({}) is True
    assert plugin.is_enabled(
        {"stage5_router_kd": {"kd_temperature": 2.0}}
    ) is True


def test_plugin_has_compute_kd_loss_hook():
    """The RK-8 phase hook ``compute_kd_loss`` is present and callable."""
    from moe_compress.router_kd.plugins.vocab_kd import VocabKdPlugin

    plugin = VocabKdPlugin()
    assert callable(getattr(plugin, "compute_kd_loss", None))


def test_no_monolith_import_at_any_scope():
    """The module never imports ``stage5_router_kd`` / ``router_kd.orchestrator``.

    The module docstring's contract says NEVER import the monolith (or the
    orchestrator) at any scope — module-top OR function-local — since either
    would risk an import cycle (the monolith re-imports *this* module at load
    time). Parse the source with ``ast`` and walk the FULL tree so a
    function-local ``import stage5_router_kd`` cannot slip past.

    For ``ImportFrom`` both halves are checked: ``node.module`` (the
    ``from X import ...`` package) AND ``node.names`` (the imported symbols) —
    so the cycle-causing ``from moe_compress import stage5_router_kd`` form
    (``module="moe_compress"``, name ``stage5_router_kd``) is also caught.
    """
    from moe_compress.router_kd.plugins import vocab_kd as mod

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
            # Also inspect the imported NAMES: ``from moe_compress import
            # stage5_router_kd`` carries the monolith as an ``alias.name``, not
            # in ``node.module`` — without this it would slip past undetected.
            for alias in node.names:
                assert not any(f in alias.name for f in forbidden), (
                    f"forbidden import-from name at any scope: "
                    f"from {mod_name} import {alias.name}"
                )


# ---------------------------------------------------------------------------
# _chunked_vocab_kl — the relocated vocab-KL kernel
# ---------------------------------------------------------------------------


def test_chunked_vocab_kl_zero_for_identical_logits():
    """KL(teacher ‖ student) is ~0 when student logits equal teacher logits."""
    from moe_compress.router_kd.plugins.vocab_kd import _chunked_vocab_kl

    torch.manual_seed(0)
    logits = torch.randn(2, 8, 16)
    kl = _chunked_vocab_kl(logits.clone(), logits.clone(), 1.0, chunk_size=4)
    assert math.isclose(float(kl), 0.0, abs_tol=1e-6)


def test_chunked_vocab_kl_temperature_scaling():
    """The kernel scales the KL by τ² (the explicit ``temperature ** 2``).

    Compare τ=1 against τ=2 with the SAME logits: the τ² prefactor is part of
    the returned value, so a smoke check confirms the kernel multiplies it in
    (the softmax-temperature also reshapes the distributions, so this is a
    presence check, not an exact 4× equality).
    """
    from moe_compress.router_kd.plugins.vocab_kd import _chunked_vocab_kl

    torch.manual_seed(1)
    s = torch.randn(2, 6, 12)
    t = torch.randn(2, 6, 12)
    kl_t1 = _chunked_vocab_kl(s, t, 1.0, chunk_size=512)
    kl_t2 = _chunked_vocab_kl(s, t, 2.0, chunk_size=512)
    # Both finite and non-negative; τ=2 differs from τ=1 (the τ² prefactor +
    # softened distributions both apply).
    assert torch.isfinite(kl_t1) and torch.isfinite(kl_t2)
    assert float(kl_t1) >= 0.0 and float(kl_t2) >= 0.0
    assert not math.isclose(float(kl_t1), float(kl_t2), rel_tol=1e-6)


def test_chunked_vocab_kl_chunk_size_invariance():
    """The KL value is invariant to ``chunk_size`` — chunking only bounds memory.

    chunk_size=2 (a multi-chunk loop) and chunk_size=512 (single shot) on the
    same logits must agree to floating-point tolerance.
    """
    from moe_compress.router_kd.plugins.vocab_kd import _chunked_vocab_kl

    torch.manual_seed(2)
    s = torch.randn(3, 10, 20)
    t = torch.randn(3, 10, 20)
    kl_small = _chunked_vocab_kl(s, t, 1.3, chunk_size=2)
    kl_full = _chunked_vocab_kl(s, t, 1.3, chunk_size=512)
    assert math.isclose(float(kl_small), float(kl_full), rel_tol=1e-5)


def test_chunked_vocab_kl_grad_flows():
    """Gradient flows back to the student logits through the chunked KL."""
    from moe_compress.router_kd.plugins.vocab_kd import _chunked_vocab_kl

    torch.manual_seed(3)
    s = torch.randn(2, 8, 16, requires_grad=True)
    t = torch.randn(2, 8, 16)
    kl = _chunked_vocab_kl(s, t, 1.0, chunk_size=4)
    kl.backward()
    assert s.grad is not None
    assert torch.isfinite(s.grad).all()
    assert float(s.grad.abs().sum()) > 0.0


# ---------------------------------------------------------------------------
# _combine_kd_loss — the relocated loss combiner
# ---------------------------------------------------------------------------


def test_combine_kd_loss_flag_off_identity():
    """``mse_term=None`` → returns the EXACT ``kl_loss`` object (flag-off path).

    The flag-off loss must be byte-identical to pre-Direction-E ``main``; a
    non-zero ``mse_weight`` must not change that.
    """
    from moe_compress.router_kd.plugins.vocab_kd import _combine_kd_loss

    kl_loss = torch.tensor(1.5)
    loss = _combine_kd_loss(kl_loss, None, mse_weight=3.0)
    assert loss is kl_loss  # SAME object — no MSE term, no new graph node


def test_combine_kd_loss_with_mse():
    """``mse_term`` present → ``kl + mse_weight * mse_term``."""
    from moe_compress.router_kd.plugins.vocab_kd import _combine_kd_loss

    kl_loss = torch.tensor(2.0)
    mse_term = torch.tensor(4.0)
    for w in (0.0, 0.5, 1.0, 3.0):
        loss = _combine_kd_loss(kl_loss, mse_term, mse_weight=w)
        assert torch.allclose(loss, torch.tensor(2.0 + w * 4.0))


# ---------------------------------------------------------------------------
# NaN sanity probes
# ---------------------------------------------------------------------------


def test_log_first_batch_sanity_passes_on_finite():
    """``_log_first_batch_sanity`` does not raise on all-finite tensors."""
    from moe_compress.router_kd.plugins.vocab_kd import _log_first_batch_sanity

    teacher = torch.randn(1, 4, 8)
    student = torch.randn(1, 4, 8)
    loss = torch.tensor(0.5)
    # Must complete without raising.
    _log_first_batch_sanity(teacher, student, loss)


def test_log_first_batch_sanity_raises_on_nan_loss():
    """``_log_first_batch_sanity`` raises ``RuntimeError`` on a non-finite loss."""
    from moe_compress.router_kd.plugins.vocab_kd import _log_first_batch_sanity

    teacher = torch.randn(1, 4, 8)
    student = torch.randn(1, 4, 8)
    nan_loss = torch.tensor(float("nan"))
    with pytest.raises(RuntimeError, match="first-batch sanity FAILED"):
        _log_first_batch_sanity(teacher, student, nan_loss)


def test_check_param_sanity_clean(tiny_model):
    """``_check_param_sanity`` returns ``[]`` when all trainable params finite."""
    from moe_compress.router_kd.plugins.vocab_kd import _check_param_sanity

    for p in tiny_model.parameters():
        p.requires_grad_(True)
    assert _check_param_sanity(tiny_model, step=0) == []


def test_check_param_sanity_detects_nan(tiny_model):
    """``_check_param_sanity`` reports the name of a NaN-containing param."""
    from moe_compress.router_kd.plugins.vocab_kd import _check_param_sanity

    for p in tiny_model.parameters():
        p.requires_grad_(True)
    # Poison the first trainable parameter.
    name0, p0 = next(iter(tiny_model.named_parameters()))
    with torch.no_grad():
        p0.data[(0,) * p0.dim()] = float("nan")
    bad = _check_param_sanity(tiny_model, step=0)
    assert name0 in bad


def test_dump_nan_diagnostics_smoke(tiny_model):
    """``_dump_nan_diagnostics`` runs end-to-end on a NaN loss without raising.

    It is a DIAGNOSTIC: on a non-finite loss it LOGS teacher/student stats and
    the (internal) ``_check_param_sanity`` scan — it must NOT raise. This proves
    the relocated function + its internal ``_check_param_sanity`` call execute
    end-to-end. The call uses every keyword-only argument of the signature
    ``(*, loss, teacher_logits, student_logits, student, epoch, step, batch_i)``.
    """
    from moe_compress.router_kd.plugins.vocab_kd import _dump_nan_diagnostics

    for p in tiny_model.parameters():
        p.requires_grad_(True)
    teacher_logits = torch.randn(2, 4, 8)
    student_logits = torch.randn(2, 4, 8)
    nan_loss = torch.tensor(float("nan"))
    result = _dump_nan_diagnostics(
        loss=nan_loss,
        teacher_logits=teacher_logits,
        student_logits=student_logits,
        student=tiny_model,
        epoch=0,
        step=0,
        batch_i=0,
    )
    # Pure diagnostic: logs only, returns None, never raises.
    assert result is None


# ---------------------------------------------------------------------------
# compute_kd_loss — the (RK-8) phase hook
# ---------------------------------------------------------------------------


def test_compute_kd_loss_no_mse():
    """No merge-repair slots → ``kd_loss`` IS the raw ``vocab_kl`` tensor.

    The hook reads the shifted teacher/student logits + config, computes the
    chunked vocab-KL, and — with no merge-repair MSE slots — publishes the
    exact KL tensor as ``kd_loss`` (the flag-off identity).
    """
    from moe_compress.pipeline.context import PipelineContext
    from moe_compress.router_kd.plugins.vocab_kd import VocabKdPlugin

    torch.manual_seed(4)
    plugin = VocabKdPlugin()
    ctx = PipelineContext()
    ctx.set("teacher_logits", torch.randn(2, 7, 16))
    ctx.set("student_logits", torch.randn(2, 7, 16, requires_grad=True))
    ctx.set("config", {"stage5_router_kd": {"kd_temperature": 1.0}})

    plugin.compute_kd_loss(ctx)

    kd_loss = ctx.get("kd_loss")
    vocab_kl = ctx.get("vocab_kl")
    assert torch.isfinite(kd_loss).all()
    assert torch.isfinite(vocab_kl).all()
    # Flag-off path: _combine_kd_loss returns the exact kl_loss object.
    assert kd_loss is vocab_kl


def test_compute_kd_loss_with_mse():
    """Merge-repair slots present → ``kd_loss`` = ``vocab_kl + w * mse_term``.

    With ``merge_repair_mse_term`` / ``merge_repair_mse_weight`` published the
    hook combines them; ``kd_loss`` then differs from the raw ``vocab_kl``.
    """
    from moe_compress.pipeline.context import PipelineContext
    from moe_compress.router_kd.plugins.vocab_kd import VocabKdPlugin

    torch.manual_seed(5)
    plugin = VocabKdPlugin()
    ctx = PipelineContext()
    ctx.set("teacher_logits", torch.randn(2, 7, 16))
    ctx.set("student_logits", torch.randn(2, 7, 16, requires_grad=True))
    ctx.set("config", {"stage5_router_kd": {"kd_temperature": 1.0}})
    ctx.set("merge_repair_mse_term", torch.tensor(0.25))
    ctx.set("merge_repair_mse_weight", 2.0)

    plugin.compute_kd_loss(ctx)

    kd_loss = ctx.get("kd_loss")
    vocab_kl = ctx.get("vocab_kl")
    assert torch.isfinite(kd_loss).all()
    assert torch.isfinite(vocab_kl).all()
    # With-mse path: kd_loss = vocab_kl + 2.0 * 0.25 — differs from vocab_kl.
    assert kd_loss is not vocab_kl
    assert torch.allclose(kd_loss, vocab_kl + 2.0 * 0.25)
