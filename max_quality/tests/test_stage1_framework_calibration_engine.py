"""Unit tests for ``moe_compress.stage1._framework.calibration_engine``.

Covers:

- Registration semantics (duplicate name, empty kinds, non-HookSpec, non-HookKind values).
- One-shot lifecycle (``register_accumulator`` after ``run`` raises; ``run`` twice raises).
- ``required_hook_kinds`` and ``names`` introspection.
- ``run`` against the synthetic ``tiny_model`` fixture: no-op behaviour with an
  empty registry, DOWN_PROJ multiplexing, ROUTER_LOGITS_PER_BATCH wiring and
  per-batch drain, INPUT_IDS_PER_BATCH pre-hook capture, the sink-token combo,
  ``per_batch_hooks`` chaining, and teardown of every installed torch hook.

The single most important test is :func:`test_engine_down_proj_matches_legacy`:
it asserts that running the engine end-to-end produces the byte-identical
``DownProjMaxAccumulator.per_expert_max`` dict you would get from the legacy
inline ``contextlib.ExitStack`` + ``instrument_experts`` + ``run_calibration``
path. This is the byte-equivalence anchor that lets sub-task 10 swap the
inline plumbing for the engine without disturbing the Stage-1 golden snapshot.
"""
from __future__ import annotations

import contextlib

import pytest
import torch

from moe_compress.stage1._framework.calibration_engine import (
    AccumulatorRegistration,
    CalibrationEngine,
    HookKind,
    HookSpec,
    PerBatchContext,
)
from moe_compress.utils.activation_hooks import (
    DownProjMaxAccumulator,
    instrument_experts,
    run_calibration,
)
from moe_compress.utils.model_io import iter_moe_layers


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _RecorderAcc:
    """Per-expert event recorder — captures ``(layer, expert, shape, ctx-keys)``."""

    def __init__(self) -> None:
        self.events: list[tuple[int, int, tuple[int, ...], tuple[str, ...]]] = []

    def cb(self, li, e, tensor, ctx):
        self.events.append(
            (li, e, tuple(tensor.shape), tuple(sorted(ctx)))
        )


class _PerBatchRecorder:
    """Per-batch handler recorder — captures (batch_idx, has_router, has_ids)."""

    def __init__(self) -> None:
        self.events: list[tuple[int, bool, bool]] = []
        self.router_lengths: list[dict[int, int]] = []
        self.input_id_shapes: list[tuple[int, ...] | None] = []

    def cb(self, pbc: PerBatchContext) -> None:
        self.events.append(
            (
                pbc.batch_idx,
                pbc.router_logits_storage is not None,
                pbc.input_ids is not None,
            )
        )
        if pbc.router_logits_storage is not None:
            self.router_lengths.append(
                {li: len(v) for li, v in pbc.router_logits_storage.items()}
            )
        else:
            self.router_lengths.append({})
        if pbc.input_ids is not None:
            self.input_id_shapes.append(tuple(pbc.input_ids.shape))
        else:
            self.input_id_shapes.append(None)


def _make_batches(num_batches: int, batch: int, seq: int, vocab: int = 32):
    """Deterministic batches for the engine vs. legacy comparison."""
    torch.manual_seed(123)
    return [torch.randint(0, vocab, (batch, seq)) for _ in range(num_batches)]


# ---------------------------------------------------------------------------
# Section 5.4 — registration-level error semantics (no model)
# ---------------------------------------------------------------------------


def test_register_rejects_duplicate_name():
    engine = CalibrationEngine()
    spec = HookSpec(kinds=frozenset({HookKind.DOWN_PROJ}))
    engine.register_accumulator("foo", object(), spec)
    with pytest.raises(KeyError) as exc:
        engine.register_accumulator("foo", object(), spec)
    assert "foo" in str(exc.value)


def test_register_rejects_non_hookspec():
    engine = CalibrationEngine()
    with pytest.raises(TypeError):
        engine.register_accumulator(
            "foo", object(), {"kinds": frozenset({HookKind.DOWN_PROJ})}  # type: ignore[arg-type]
        )


def test_register_rejects_empty_kinds():
    engine = CalibrationEngine()
    with pytest.raises(ValueError) as exc:
        engine.register_accumulator("foo", object(), HookSpec(kinds=frozenset()))
    assert "empty" in str(exc.value).lower()


def test_register_rejects_non_hookkind_in_kinds():
    engine = CalibrationEngine()
    # frozenset({str}) bypasses Enum type-checking at construction.
    bogus = HookSpec(kinds=frozenset({"not_a_kind"}))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        engine.register_accumulator("foo", object(), bogus)


def test_register_after_run_raises(tiny_model):
    engine = CalibrationEngine()
    moe_layers = list(iter_moe_layers(tiny_model))
    batches = _make_batches(num_batches=1, batch=1, seq=4)
    # Empty-registration run() is a deliberate no-op.
    engine.run(tiny_model, batches, moe_layers=moe_layers)
    with pytest.raises(RuntimeError) as exc:
        engine.register_accumulator(
            "after",
            object(),
            HookSpec(kinds=frozenset({HookKind.DOWN_PROJ})),
        )
    assert "after run" in str(exc.value).lower()


def test_required_hook_kinds_union():
    engine = CalibrationEngine()
    engine.register_accumulator(
        "a", object(), HookSpec(kinds=frozenset({HookKind.DOWN_PROJ}))
    )
    engine.register_accumulator(
        "b",
        object(),
        HookSpec(
            kinds=frozenset({HookKind.DOWN_PROJ, HookKind.ROUTER_LOGITS_PER_BATCH})
        ),
    )
    assert engine.required_hook_kinds() == frozenset(
        {HookKind.DOWN_PROJ, HookKind.ROUTER_LOGITS_PER_BATCH}
    )


def test_names_returns_registration_order():
    engine = CalibrationEngine()
    spec = HookSpec(kinds=frozenset({HookKind.DOWN_PROJ}))
    engine.register_accumulator("c", object(), spec)
    engine.register_accumulator("a", object(), spec)
    engine.register_accumulator("b", object(), spec)
    assert engine.names() == ("c", "a", "b")
    assert len(engine) == 3


# ---------------------------------------------------------------------------
# Section 5.3 — tests that exercise ``run()`` against ``tiny_model``
# ---------------------------------------------------------------------------


def _forward_is_patched(experts_module) -> bool:
    """True iff ``instrument_experts`` has wrapped ``experts_module.forward``.

    Each access of ``experts_module.forward`` returns a *fresh* bound method,
    so ``is`` comparison cannot detect patching. The wrapper marks the
    underlying function with ``_instrument_experts_patched = True`` (see
    ``utils/activation_hooks.py:1339``); we read that marker through the
    bound method's ``__func__``.
    """
    fwd = experts_module.forward
    underlying = getattr(fwd, "__func__", fwd)
    return bool(getattr(underlying, "_instrument_experts_patched", False))


def test_engine_no_registrations_is_noop(tiny_model, monkeypatch):
    """Bare engine + run() must not patch any expert forward, must not install
    any forward-pre-hook, and must call ``model(input_ids=batch)`` once per batch."""
    engine = CalibrationEngine()
    moe_layers = list(iter_moe_layers(tiny_model))

    pre_pre_hooks = dict(tiny_model._forward_pre_hooks)

    # Count how many forwards the model receives.
    call_counter = {"n": 0}
    original_forward = tiny_model.forward

    def _spy_forward(*args, **kwargs):
        call_counter["n"] += 1
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(tiny_model, "forward", _spy_forward)

    batches = _make_batches(num_batches=3, batch=1, seq=4)
    engine.run(tiny_model, batches, moe_layers=moe_layers)

    assert call_counter["n"] == 3
    # No expert forward was patched.
    for layer in tiny_model.model.layers:
        assert not _forward_is_patched(layer.mlp.experts)
    # No new pre-hooks left behind.
    assert dict(tiny_model._forward_pre_hooks) == pre_pre_hooks


def test_engine_down_proj_matches_legacy(tiny_model):
    """**Golden-equivalence anchor.**

    Running the engine with a single ``DownProjMaxAccumulator`` registered
    under ``HookKind.DOWN_PROJ`` must produce the byte-identical
    ``per_expert_max`` dict that the legacy inline ExitStack +
    ``instrument_experts({"down": cb})`` path produces on the same batches.
    """
    moe_layers = list(iter_moe_layers(tiny_model))
    batches = _make_batches(num_batches=4, batch=1, seq=8)

    # Path A: engine.
    acc_a = DownProjMaxAccumulator()
    engine = CalibrationEngine()
    engine.register_accumulator(
        "max",
        acc_a,
        HookSpec(
            kinds=frozenset({HookKind.DOWN_PROJ}),
            expert_callback=lambda li, e, t, _ctx: acc_a.update(li, e, t),
        ),
    )
    engine.run(tiny_model, batches, moe_layers=moe_layers)
    acc_a.finalize()

    # Path B: legacy inline ExitStack.
    acc_b = DownProjMaxAccumulator()
    with contextlib.ExitStack() as stack:
        for ref in moe_layers:
            stack.enter_context(
                instrument_experts(
                    ref,
                    {"down": lambda li, e, t, _ctx: acc_b.update(li, e, t)},
                )
            )
        run_calibration(tiny_model, batches, device=None)
    acc_b.finalize()

    assert acc_a.per_expert_max.keys() == acc_b.per_expert_max.keys()
    for key in acc_a.per_expert_max:
        # Both paths apply identical ops (torch.maximum on the same tensors in
        # the same order); bytes match.
        assert acc_a.per_expert_max[key] == acc_b.per_expert_max[key], (
            f"engine vs legacy mismatch at {key}: "
            f"{acc_a.per_expert_max[key]!r} != {acc_b.per_expert_max[key]!r}"
        )


def test_engine_down_proj_multiplexes_two_callbacks(tiny_model):
    """Two DOWN_PROJ accumulators must each see every per-expert event in the
    same order — the engine builds one fused ``down`` callback that fans out."""
    moe_layers = list(iter_moe_layers(tiny_model))
    batches = _make_batches(num_batches=2, batch=1, seq=4)

    rec_a = _RecorderAcc()
    rec_b = _RecorderAcc()
    engine = CalibrationEngine()
    engine.register_accumulator(
        "rec_a",
        rec_a,
        HookSpec(
            kinds=frozenset({HookKind.DOWN_PROJ}),
            expert_callback=rec_a.cb,
        ),
    )
    engine.register_accumulator(
        "rec_b",
        rec_b,
        HookSpec(
            kinds=frozenset({HookKind.DOWN_PROJ}),
            expert_callback=rec_b.cb,
        ),
    )
    engine.run(tiny_model, batches, moe_layers=moe_layers)

    assert rec_a.events  # something fired
    assert rec_a.events == rec_b.events


def test_engine_router_logits_storage_is_drained_per_batch(tiny_model):
    """Per-batch handler sees exactly one tensor per layer; the engine clears
    the storage after the handler returns."""
    moe_layers = list(iter_moe_layers(tiny_model))
    batches = _make_batches(num_batches=3, batch=1, seq=4)

    rec = _PerBatchRecorder()
    engine = CalibrationEngine()
    engine.register_accumulator(
        "router_only",
        rec,
        HookSpec(
            kinds=frozenset({HookKind.ROUTER_LOGITS_PER_BATCH}),
            per_batch=rec.cb,
        ),
    )
    engine.run(tiny_model, batches, moe_layers=moe_layers)

    assert len(rec.events) == 3
    for idx, (b_idx, has_router, has_ids) in enumerate(rec.events):
        assert b_idx == idx
        assert has_router is True
        assert has_ids is False
    # Each per-batch event saw exactly one tensor per layer.
    for lengths in rec.router_lengths:
        assert set(lengths.keys()) == {ref.layer_idx for ref in moe_layers}
        for v in lengths.values():
            assert v == 1


def test_engine_input_ids_pre_hook_fires(tiny_model):
    """Per-batch handler must see the captured ``input_ids`` with shape (B, T)."""
    moe_layers = list(iter_moe_layers(tiny_model))
    batches = _make_batches(num_batches=1, batch=2, seq=5)

    rec = _PerBatchRecorder()
    engine = CalibrationEngine()
    engine.register_accumulator(
        "ids_only",
        rec,
        HookSpec(
            kinds=frozenset({HookKind.INPUT_IDS_PER_BATCH}),
            per_batch=rec.cb,
        ),
    )
    engine.run(tiny_model, batches, moe_layers=moe_layers)

    assert len(rec.events) == 1
    b_idx, has_router, has_ids = rec.events[0]
    assert b_idx == 0
    assert has_router is False
    assert has_ids is True
    assert rec.input_id_shapes[0] == (2, 5)


def test_engine_sink_token_style_three_kinds(tiny_model):
    """Sink-token pattern: one accumulator declares both
    ROUTER_LOGITS_PER_BATCH + INPUT_IDS_PER_BATCH and reads both inside its
    ``per_batch`` handler — the integration anchor for sub-task 7."""
    moe_layers = list(iter_moe_layers(tiny_model))
    batches = _make_batches(num_batches=1, batch=2, seq=4)

    captured: dict[str, object] = {}

    def _handler(pbc: PerBatchContext) -> None:
        captured["input_shape"] = tuple(pbc.input_ids.shape)
        layer_idx = moe_layers[0].layer_idx
        logits = pbc.router_logits_storage[layer_idx][-1]
        captured["router_shape"] = tuple(logits.shape)

    spec = HookSpec(
        kinds=frozenset(
            {HookKind.ROUTER_LOGITS_PER_BATCH, HookKind.INPUT_IDS_PER_BATCH}
        ),
        per_batch=_handler,
    )
    engine = CalibrationEngine()
    engine.register_accumulator("sink_like", object(), spec)
    engine.run(tiny_model, batches, moe_layers=moe_layers)

    B, T = 2, 4
    num_experts = moe_layers[0].num_routed_experts
    assert captured["input_shape"] == (B, T)
    assert captured["router_shape"] == (B * T, num_experts)


def test_engine_per_batch_hooks_chain_in_order(tiny_model):
    """``per_batch_hooks`` (the integer-batch-index chain) fires in iteration
    order, once per batch — exactly like the legacy progress callback."""
    moe_layers = list(iter_moe_layers(tiny_model))
    batches = _make_batches(num_batches=2, batch=1, seq=4)

    log: list[tuple[str, int]] = []
    rec = _RecorderAcc()  # need at least one expert callback to keep the path lit
    engine = CalibrationEngine()
    engine.register_accumulator(
        "rec",
        rec,
        HookSpec(
            kinds=frozenset({HookKind.DOWN_PROJ}),
            expert_callback=rec.cb,
        ),
    )

    engine.run(
        tiny_model,
        batches,
        moe_layers=moe_layers,
        per_batch_hooks=[
            lambda i: log.append(("a", i)),
            lambda i: log.append(("b", i)),
        ],
    )

    assert log == [("a", 0), ("b", 0), ("a", 1), ("b", 1)]


def test_engine_tears_down_all_hooks_on_exit(tiny_model):
    """After ``run`` returns:

    a) every ``layer.mlp.experts.forward`` is the original (no leaked
       ``_instrument_experts_patched`` marker visible from the live forward),
    b) ``tiny_model._forward_pre_hooks`` matches its baseline (input_ids hook
       removed),
    c) ``engine._has_run is True``.
    """
    moe_layers = list(iter_moe_layers(tiny_model))
    batches = _make_batches(num_batches=1, batch=1, seq=4)
    pre_pre_hooks = dict(tiny_model._forward_pre_hooks)

    rec = _PerBatchRecorder()
    engine = CalibrationEngine()
    engine.register_accumulator(
        "rec",
        rec,
        HookSpec(
            kinds=frozenset(
                {HookKind.DOWN_PROJ, HookKind.INPUT_IDS_PER_BATCH}
            ),
            expert_callback=lambda *a: None,
            per_batch=rec.cb,
        ),
    )
    engine.run(tiny_model, batches, moe_layers=moe_layers)

    for layer in tiny_model.model.layers:
        assert not _forward_is_patched(layer.mlp.experts)
    assert dict(tiny_model._forward_pre_hooks) == pre_pre_hooks
    assert engine._has_run is True


def _capture_input_ids_pre_hook(engine_cls, tiny_model):
    """Run the engine with an INPUT_IDS_PER_BATCH spec and return the
    pre-forward-hook closure the engine installs on the model.

    The engine registers the closure via ``register_forward_pre_hook`` and
    removes it on ExitStack teardown. To exercise the closure's positional-
    args and 1-D branches without going through ``run_calibration`` (which
    always passes ``input_ids`` as a 2-D kwarg), we intercept the registration
    here and return the captured closure.
    """
    captured: dict[str, object] = {}
    original_register = tiny_model.register_forward_pre_hook

    def _intercept(fn, *args, **kwargs):
        captured["fn"] = fn
        return original_register(fn, *args, **kwargs)

    tiny_model.register_forward_pre_hook = _intercept  # type: ignore[method-assign]
    try:
        moe_layers = list(iter_moe_layers(tiny_model))
        batches = _make_batches(num_batches=1, batch=1, seq=4)
        rec = _PerBatchRecorder()
        engine = engine_cls()
        engine.register_accumulator(
            "ids_only",
            rec,
            HookSpec(
                kinds=frozenset({HookKind.INPUT_IDS_PER_BATCH}),
                per_batch=rec.cb,
            ),
        )
        engine.run(tiny_model, batches, moe_layers=moe_layers)
    finally:
        # Restore the bound method by deleting the instance attribute.
        del tiny_model.register_forward_pre_hook
    return captured["fn"]


def test_engine_input_ids_pre_hook_handles_1d_input(tiny_model):
    """The captured ``input_ids`` may be 1-D ``[T]`` (no batch dim).

    The engine intentionally does NOT unsqueeze — the PerBatchContext
    docstring contract is that consumers handle the 1-D case themselves
    (mirrors the legacy ``_phase_b_per_batch_cb``). This test exercises the
    1-D branch through the actual installed pre-hook closure.
    """
    fn = _capture_input_ids_pre_hook(CalibrationEngine, tiny_model)
    # The closure mutates an internal cell; we observe its effect by calling
    # the closure twice (1-D then 2-D) and reading what it stored each time
    # via a probe call that triggers another capture. Direct introspection of
    # the cell is the only reliable signal because the engine has already
    # torn the hook down.
    #
    # Cells of the closure: locate the ``current_input_ids`` list cell.
    cells = [c.cell_contents for c in fn.__closure__ or ()]
    storage = next(c for c in cells if isinstance(c, list))

    # 1-D input via kwargs.
    ids_1d = torch.tensor([1, 2, 3, 4])
    fn(tiny_model, (), {"input_ids": ids_1d})
    assert storage[0] is not None
    assert storage[0].dim() == 1
    assert storage[0].shape == (4,)
    # The engine does NOT unsqueeze; consumers must do it themselves.
    assert tuple(storage[0].shape) == (4,)


def test_engine_input_ids_pre_hook_handles_positional_arg(tiny_model):
    """The captured ``input_ids`` branch ``args[0]`` fires only when a
    caller passes ``input_ids`` positionally. ``run_calibration`` always uses
    a kwarg, so we drive the closure directly to keep the branch live.
    """
    fn = _capture_input_ids_pre_hook(CalibrationEngine, tiny_model)
    cells = [c.cell_contents for c in fn.__closure__ or ()]
    storage = next(c for c in cells if isinstance(c, list))

    # Positional input — args[0] branch.
    ids_pos = torch.tensor([[5, 6, 7, 8]])
    fn(tiny_model, (ids_pos,), {})
    assert storage[0] is not None
    assert storage[0].shape == (1, 4)
    assert torch.equal(storage[0], ids_pos)

    # Mixed: kwargs takes precedence when present.
    other = torch.tensor([[9, 9]])
    fn(tiny_model, (ids_pos,), {"input_ids": other})
    assert torch.equal(storage[0], other)

    # Neither: storage gets None.
    fn(tiny_model, (), {"input_ids": None})
    assert storage[0] is None


def test_engine_run_twice_raises(tiny_model):
    moe_layers = list(iter_moe_layers(tiny_model))
    batches = _make_batches(num_batches=1, batch=1, seq=4)

    engine = CalibrationEngine()
    engine.register_accumulator(
        "rec",
        DownProjMaxAccumulator(),
        HookSpec(
            kinds=frozenset({HookKind.DOWN_PROJ}),
            expert_callback=lambda *a: None,
        ),
    )
    engine.run(tiny_model, batches, moe_layers=moe_layers)
    with pytest.raises(RuntimeError) as exc:
        engine.run(tiny_model, batches, moe_layers=moe_layers)
    assert "already run" in str(exc.value).lower()
