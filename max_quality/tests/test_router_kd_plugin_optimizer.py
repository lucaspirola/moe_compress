"""RK-3 — Router-KD KD-optimizer plugin extraction tests.

Verifies the RK-3 ``KdOptimizerPlugin`` scaffolding in
``router_kd/plugins/kd_optimizer.py``:

* ``_move_optimizer_state_to_device`` and ``KdOptimizerPlugin`` import from the
  plugin module;
* the ``stage5_router_kd`` monolith re-exports the SAME
  ``_move_optimizer_state_to_device`` object (the ``# noqa: F401`` re-import
  block is load-bearing);
* ``KdOptimizerPlugin`` satisfies the universal ``PipelinePlugin`` Protocol,
  carries correct metadata, is unconditionally enabled, and exposes the (RK-8)
  ``build_optimizer`` phase hook;
* the module never imports the ``stage5_router_kd`` monolith or
  ``router_kd.orchestrator`` at any scope (the circular-import contract);
* ``_move_optimizer_state_to_device`` moves optimizer state tensors;
* the ``build_optimizer`` hook reproduces the monolith's single-group /
  split-param-group AdamW construction and the warmup+cosine ``_lr_lambda``.

RK-3 covers a MIXED pattern: ``_move_optimizer_state_to_device`` is relocated
verbatim (the monolith re-imports it); the split-param-group AdamW + the
``_lr_lambda`` LR scheduler — inline ``run()`` code in the monolith — are
reproduced in the inert hook (the monolith ``run()`` is NOT modified for them).
The byte-identical behavioral gate is the RK-0 golden snapshot
(``test_router_kd_golden_snapshot.py``); this file only checks the relocation
plumbing and the relocated/reproduced logic.
"""
from __future__ import annotations

import ast
import math
from pathlib import Path

import torch


def test_kd_optimizer_module_imports():
    """``KdOptimizerPlugin`` / ``_move_optimizer_state_to_device`` import."""
    from moe_compress.router_kd.plugins.kd_optimizer import (
        KdOptimizerPlugin,
        _move_optimizer_state_to_device,
    )

    assert isinstance(KdOptimizerPlugin, type)
    assert callable(_move_optimizer_state_to_device)


def test_monolith_reexports_move_optimizer_state():
    """The monolith re-exports the SAME ``_move_optimizer_state_to_device``.

    Proves the ``# noqa: F401`` re-import block in ``stage5_router_kd.py``
    keeps ``run()``'s two resume-path call sites on their original import path.
    """
    from moe_compress import stage5_router_kd
    from moe_compress.router_kd.plugins import kd_optimizer

    assert (
        stage5_router_kd._move_optimizer_state_to_device
        is kd_optimizer._move_optimizer_state_to_device
    )


def test_plugin_satisfies_protocol():
    """``KdOptimizerPlugin`` structurally satisfies ``PipelinePlugin``."""
    from moe_compress.pipeline.plugin import PipelinePlugin
    from moe_compress.router_kd.plugins.kd_optimizer import KdOptimizerPlugin

    assert isinstance(KdOptimizerPlugin(), PipelinePlugin)


def test_plugin_metadata():
    """Plugin metadata — name / paper id / config_key / tuple-typed fields."""
    from moe_compress.router_kd.plugins.kd_optimizer import KdOptimizerPlugin

    plugin = KdOptimizerPlugin()
    assert plugin.name == "kd_optimizer"
    assert "2603.02217" in plugin.paper
    assert plugin.config_key == "stage5_router_kd.learning_rate"
    assert isinstance(plugin.reads, tuple)
    assert isinstance(plugin.writes, tuple)
    assert isinstance(plugin.provides, tuple)
    assert "optimizer" in plugin.writes
    assert "lr_scheduler" in plugin.writes


def test_plugin_is_enabled_unconditional():
    """Building the optimizer is UNCONDITIONAL — ``is_enabled`` always True.

    Every Router-KD run must construct an optimizer + LR scheduler before
    training; ``config_key`` only names the learning rate, it never gates the
    plugin as a whole.
    """
    from moe_compress.router_kd.plugins.kd_optimizer import KdOptimizerPlugin

    plugin = KdOptimizerPlugin()
    assert plugin.is_enabled({}) is True
    assert plugin.is_enabled(
        {"stage5_router_kd": {"learning_rate": 1e-4}}
    ) is True


def test_plugin_has_build_optimizer_hook():
    """The RK-8 phase hook ``build_optimizer`` is present and callable."""
    from moe_compress.router_kd.plugins.kd_optimizer import KdOptimizerPlugin

    plugin = KdOptimizerPlugin()
    assert callable(getattr(plugin, "build_optimizer", None))


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
    from moe_compress.router_kd.plugins import kd_optimizer as mod

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


def test_move_optimizer_state_to_device(tiny_model):
    """``_move_optimizer_state_to_device`` moves every state tensor.

    Build an AdamW over the tiny model, run one ``.step()`` to populate
    ``optim.state`` with exp-avg buffers, then move the state to CPU and assert
    every tensor lives on the target device.
    """
    from moe_compress.router_kd.plugins.kd_optimizer import (
        _move_optimizer_state_to_device,
    )

    params = [p for p in tiny_model.parameters()]
    optim = torch.optim.AdamW(params, lr=1e-3)
    for p in params:
        p.grad = torch.ones_like(p)
    optim.step()
    assert optim.state, "expected populated optimizer state after .step()"

    target = torch.device("cpu")
    _move_optimizer_state_to_device(optim, target)
    for state in optim.state.values():
        for v in state.values():
            if isinstance(v, torch.Tensor):
                assert v.device == target, f"state tensor not on {target}: {v.device}"


def test_build_optimizer_single_group(tiny_model):
    """No ``merge_repair_grad_handles`` → single-group AdamW + LambdaLR.

    The hook reads ``student`` + ``config`` (with ``learning_rate`` /
    ``weight_decay``) and ``total_optim_steps``; the result is a one-group
    ``AdamW`` carrying the configured lr / weight_decay and a ``LambdaLR``.
    """
    from moe_compress.pipeline.context import PipelineContext
    from moe_compress.router_kd.plugins.kd_optimizer import KdOptimizerPlugin

    plugin = KdOptimizerPlugin()
    ctx = PipelineContext()
    ctx.set("student", tiny_model)
    ctx.set("config", {
        "stage5_router_kd": {
            "learning_rate": 3e-4,
            "weight_decay": 0.02,
        }
    })
    ctx.set("total_optim_steps", 100)

    plugin.build_optimizer(ctx)

    optim = ctx.get("optimizer")
    assert isinstance(optim, torch.optim.AdamW)
    assert len(optim.param_groups) == 1
    assert optim.param_groups[0]["lr"] == 3e-4
    assert optim.param_groups[0]["weight_decay"] == 0.02

    scheduler = ctx.get("lr_scheduler")
    assert isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR)


def test_build_optimizer_split_groups(tiny_model):
    """``merge_repair_grad_handles`` present → two AdamW param groups.

    With a ``merge_repair_grad_handles`` slot keyed by expert-param ids, the
    hook builds two groups in the monolith order: router group first
    (``weight_decay=_wd``), expert group second (``weight_decay=0.0``).
    """
    from moe_compress.pipeline.context import PipelineContext
    from moe_compress.router_kd.plugins.kd_optimizer import KdOptimizerPlugin

    # Pick an arbitrary trainable param to act as the "expert" tensor.
    trainable = [p for p in tiny_model.parameters() if p.requires_grad]
    assert trainable, "tiny_model has no trainable params"
    expert_param = trainable[0]

    plugin = KdOptimizerPlugin()
    ctx = PipelineContext()
    ctx.set("student", tiny_model)
    ctx.set("config", {
        "stage5_router_kd": {
            "learning_rate": 3e-4,
            "weight_decay": 0.05,
        }
    })
    ctx.set("total_optim_steps", 100)
    # merge_repair_grad_handles is iterated for its keys (expert-param ids).
    ctx.set("merge_repair_grad_handles", {id(expert_param): object()})

    plugin.build_optimizer(ctx)

    optim = ctx.get("optimizer")
    assert isinstance(optim, torch.optim.AdamW)
    assert len(optim.param_groups) == 2
    # Monolith order: router group (weight_decay=_wd) then expert group (0.0).
    assert optim.param_groups[0]["weight_decay"] == 0.05
    assert optim.param_groups[1]["weight_decay"] == 0.0
    # The designated expert param landed in the expert (second) group.
    expert_ids = {id(p) for p in optim.param_groups[1]["params"]}
    assert id(expert_param) in expert_ids


def test_lr_lambda_warmup_and_decay(tiny_model):
    """The reproduced ``_lr_lambda`` matches the monolith warmup+cosine shape.

    With ``lr_schedule="cosine"``: step 0 → ``1/warmup_steps`` (the load-bearing
    ``+1`` off-by-one), ``warmup_steps-1`` → near 1.0, the final step → near
    ``lr_min_ratio``. With ``lr_schedule="none"`` the lambda is a constant 1.0.
    """
    from moe_compress.pipeline.context import PipelineContext
    from moe_compress.router_kd.plugins.kd_optimizer import KdOptimizerPlugin

    plugin = KdOptimizerPlugin()
    total_optim_steps = 200
    warmup_ratio = 0.10
    lr_min_ratio = 0.10
    warmup_steps = max(1, int(total_optim_steps * warmup_ratio))  # = 20

    # --- cosine schedule ---
    ctx = PipelineContext()
    ctx.set("student", tiny_model)
    ctx.set("config", {
        "stage5_router_kd": {
            "learning_rate": 3e-4,
            "lr_schedule": "cosine",
            "warmup_ratio": warmup_ratio,
            "lr_min_ratio": lr_min_ratio,
        }
    })
    ctx.set("total_optim_steps", total_optim_steps)
    plugin.build_optimizer(ctx)
    lr_lambda = ctx.get("lr_scheduler").lr_lambdas[0]

    # Step 0 fires at LR = 1/warmup_steps (the (current_step + 1) off-by-one).
    assert lr_lambda(0) == 1.0 / warmup_steps
    # Last warmup step reaches the full multiplier.
    assert math.isclose(lr_lambda(warmup_steps - 1), 1.0, rel_tol=1e-9)
    # Final step decays to lr_min_ratio.
    assert math.isclose(
        lr_lambda(total_optim_steps), lr_min_ratio, rel_tol=1e-9
    )

    # --- none schedule → constant 1.0 ---
    ctx_none = PipelineContext()
    ctx_none.set("student", tiny_model)
    ctx_none.set("config", {
        "stage5_router_kd": {
            "learning_rate": 3e-4,
            "lr_schedule": "none",
        }
    })
    ctx_none.set("total_optim_steps", total_optim_steps)
    plugin.build_optimizer(ctx_none)
    lr_lambda_none = ctx_none.get("lr_scheduler").lr_lambdas[0]
    for step in (0, 5, warmup_steps, total_optim_steps):
        assert lr_lambda_none(step) == 1.0
