"""Tier-1 Stage-5 optimization levers (impl/stage5-opt).

Proves the three Tier-1 levers are byte-/behaviour-safe:

* Lever A (``use_cache=False``): the student/teacher forwards are bit-identical
  with and without ``use_cache=False`` — a direct ``torch.equal`` re-proof of
  the diff=0.0 claim (the golden byte/trace pass in
  ``test_router_kd_golden_snapshot.py`` is the project-level proof; this is the
  isolated unit proof).
* Lever C (async checkpoint): the synchronous deep-CPU-copy of the optimizer
  state makes the background write independent of a subsequent ``optim.step()``
  (torn-snapshot guard); a resume round-trip reloads the exact snapshot; the
  ``STAGE5_ASYNC_CKPT=0`` kill-switch forces the synchronous path; and a
  writer-thread error is re-raised on the training thread.

Lever B (grad-norm gating) is golden-neutral by construction (``grad_norm`` is
not in the pinned trace) and is proven via the unchanged golden + the
window-boundary parity argument in the plan; no separate numeric pin here.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from moe_compress.router_kd.orchestrator import (
    _async_ckpt_enabled,
    _deep_cpu_copy,
    _save_stage5_checkpoint,
    _Stage5CheckpointWriter,
)


# --------------------------------------------------------------------------- #
# Lever A — use_cache=False is bit-identical on a real HF-style causal LM      #
# --------------------------------------------------------------------------- #
def test_lever_a_use_cache_false_logits_bit_identical():
    """A single full-sequence forward yields identical logits with/without the
    KV cache. The cache is never read on a non-incremental forward, so
    suppressing its allocation cannot change the logits math."""
    pytest.importorskip("transformers")
    from transformers import AutoConfig, AutoModelForCausalLM

    torch.manual_seed(0)
    cfg = AutoConfig.for_model(
        "llama",
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
    )
    model = AutoModelForCausalLM.from_config(cfg)
    model.eval()

    input_ids = torch.randint(0, 64, (2, 16))
    with torch.no_grad():
        out_default = model(input_ids=input_ids)  # HF default use_cache=True
        out_nocache = model(input_ids=input_ids, use_cache=False)

    assert torch.equal(out_default.logits, out_nocache.logits), (
        "use_cache=False changed the logits — Lever A is NOT byte-identical. "
        f"max-abs-diff="
        f"{(out_default.logits - out_nocache.logits).abs().max().item():.3e}"
    )


# --------------------------------------------------------------------------- #
# Lever B — gate predicate is window-boundary identical to the consumption    #
# --------------------------------------------------------------------------- #
def test_lever_b_gate_matches_consumption_window():
    """Lever B moves the grad-norm computation behind the log window using the
    PRE-increment predicate ``(step + 1) % N == 0`` (the call stays before
    optim.step()/zero_grad(), so it reads the same populated grads). The value
    is consumed inside ``if step % N == 0:`` AFTER ``step += 1``. Because
    ``step += 1`` runs between the two predicates, the pre-increment gate fires
    on exactly the steps whose post-increment value is a window boundary — so
    grad_norm is computed iff it will be consumed, and on those steps it is the
    same grads -> bit-identical reported value. This asserts that equivalence
    over a range of steps for several window sizes."""
    for n in (1, 2, 5, 50):
        for step_before_increment in range(0, 200):
            computed = (step_before_increment + 1) % n == 0
            step_after_increment = step_before_increment + 1
            consumed = step_after_increment % n == 0
            assert computed == consumed, (
                f"gate mismatch at step={step_before_increment} N={n}: "
                f"computed={computed} consumed={consumed}"
            )


# --------------------------------------------------------------------------- #
# Lever C — kill-switch                                                        #
# --------------------------------------------------------------------------- #
def test_lever_c_killswitch_default_enabled(monkeypatch):
    monkeypatch.delenv("STAGE5_ASYNC_CKPT", raising=False)
    assert _async_ckpt_enabled() is True


def test_lever_c_killswitch_disabled(monkeypatch):
    monkeypatch.setenv("STAGE5_ASYNC_CKPT", "0")
    assert _async_ckpt_enabled() is False


def test_lever_c_killswitch_synchronous_write(tmp_path):
    """With writer=None (the kill-switch / no-writer path) the checkpoint is
    written synchronously and is on disk the moment the call returns."""
    model = nn.Linear(4, 4)
    for p in model.parameters():
        p.requires_grad_(True)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # Populate optimizer moment tensors.
    model(torch.randn(2, 4)).sum().backward()
    optim.step()

    _save_stage5_checkpoint(
        tmp_path, step=1, epoch=0, batch_idx=0, student=model, optim=optim,
        scheduler=None, writer=None,
    )
    final = tmp_path / "step_1.pt"
    assert final.exists(), "synchronous path did not write step_1.pt before return"
    payload = torch.load(final, map_location="cpu", weights_only=False)
    assert payload["format_version"] == 2
    assert "optim_state" in payload and "router_state" in payload


# --------------------------------------------------------------------------- #
# Lever C — deep CPU copy / torn-snapshot guard                               #
# --------------------------------------------------------------------------- #
def test_deep_cpu_copy_shares_no_storage():
    """_deep_cpu_copy must clone every tensor (no shared storage) so a later
    in-place mutation of the source cannot leak into the copy."""
    src = {"a": torch.ones(3), "b": [torch.zeros(2), {"c": torch.full((2,), 5.0)}]}
    cp = _deep_cpu_copy(src)
    # Mutate the source in place.
    src["a"].add_(100.0)
    src["b"][0].add_(100.0)
    src["b"][1]["c"].add_(100.0)
    assert torch.equal(cp["a"], torch.ones(3))
    assert torch.equal(cp["b"][0], torch.zeros(2))
    assert torch.equal(cp["b"][1]["c"], torch.full((2,), 5.0))


def test_lever_c_async_write_independent_of_subsequent_optim_step(tmp_path):
    """The torn-snapshot guard: enqueue an async save, IMMEDIATELY mutate the
    router params + optimizer moment tensors via another optim.step(), join the
    writer, reload, and assert the reloaded state equals the PRE-mutation
    snapshot. If the deep CPU copy were missing, torch.save would race the
    mutation and the reload would show the post-mutation values."""
    torch.manual_seed(0)
    model = nn.Linear(4, 4)
    optim = torch.optim.AdamW(model.parameters(), lr=0.5)

    # One step to populate moment tensors with non-trivial values.
    model(torch.randn(8, 4)).pow(2).sum().backward()
    optim.step()

    # Snapshot the state we expect to read back from the checkpoint.
    pre_router = {n: p.detach().clone() for n, p in model.named_parameters()}
    pre_optim = _deep_cpu_copy(optim.state_dict())

    writer = _Stage5CheckpointWriter()
    try:
        _save_stage5_checkpoint(
            tmp_path, step=10, epoch=0, batch_idx=0, student=model, optim=optim,
            scheduler=None, writer=writer,
        )
        # Race window: mutate params + moments before the writer has serialized.
        for _ in range(5):
            optim.zero_grad()
            model(torch.randn(8, 4)).pow(2).sum().backward()
            optim.step()
        writer.join()
    finally:
        writer.close()

    # Confirm the params/moments actually moved (the race window was real).
    post_router = {n: p.detach().clone() for n, p in model.named_parameters()}
    assert any(
        not torch.equal(pre_router[n], post_router[n]) for n in pre_router
    ), "optim.step() did not mutate params — the race window is vacuous"

    payload = torch.load(tmp_path / "step_10.pt", map_location="cpu",
                         weights_only=False)
    for n, expected in pre_router.items():
        assert torch.equal(payload["router_state"][n], expected), (
            f"router_state[{n}] reflects the POST-mutation value — torn snapshot"
        )
    # Optimizer exp_avg / exp_avg_sq must match the pre-mutation snapshot.
    for pid, st in pre_optim["state"].items():
        for key, val in st.items():
            if torch.is_tensor(val):
                got = payload["optim_state"]["state"][pid][key]
                assert torch.equal(got, val), (
                    f"optim_state[{pid}][{key}] reflects POST-mutation — torn"
                )


def test_lever_c_resume_roundtrip(tmp_path):
    """Async save -> reload restores router + optimizer + scheduler exactly."""
    torch.manual_seed(0)
    model = nn.Linear(4, 4)
    optim = torch.optim.AdamW(model.parameters(), lr=0.1)
    sched = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=0.5)
    model(torch.randn(8, 4)).pow(2).sum().backward()
    optim.step()
    sched.step()

    expect_router = {n: p.detach().clone() for n, p in model.named_parameters()}
    expect_optim = _deep_cpu_copy(optim.state_dict())
    expect_sched = sched.state_dict()

    writer = _Stage5CheckpointWriter()
    try:
        _save_stage5_checkpoint(
            tmp_path, step=7, epoch=2, batch_idx=3, student=model, optim=optim,
            scheduler=sched, best_step=5, writer=writer,
        )
        writer.join()
    finally:
        writer.close()

    payload = torch.load(tmp_path / "step_7.pt", map_location="cpu",
                         weights_only=False)
    assert payload["step"] == 7 and payload["epoch"] == 2 and payload["batch_idx"] == 3
    for n, exp in expect_router.items():
        assert torch.equal(payload["router_state"][n], exp)
    assert payload["scheduler_state"]["last_epoch"] == expect_sched["last_epoch"]
    # Rebuild an optimizer and load to prove the saved state is loadable.
    model2 = nn.Linear(4, 4)
    optim2 = torch.optim.AdamW(model2.parameters(), lr=0.1)
    optim2.load_state_dict(payload["optim_state"])
    for pid, st in expect_optim["state"].items():
        for key, val in st.items():
            if torch.is_tensor(val):
                assert torch.equal(optim2.state_dict()["state"][pid][key], val)


def test_lever_c_single_writer_no_overlap(tmp_path):
    """Two queued step_*.pt writes never overlap (maxsize=1 put blocks), and
    the prune keeps the newest two after join."""
    model = nn.Linear(4, 4)
    optim = torch.optim.AdamW(model.parameters(), lr=0.1)
    model(torch.randn(8, 4)).pow(2).sum().backward()
    optim.step()

    writer = _Stage5CheckpointWriter()
    try:
        for step in (1, 2, 3):
            _save_stage5_checkpoint(
                tmp_path, step=step, epoch=0, batch_idx=step, student=model,
                optim=optim, scheduler=None, writer=writer,
            )
        writer.join()
    finally:
        writer.close()
    remaining = sorted(p.name for p in tmp_path.glob("step_*.pt"))
    assert remaining == ["step_2.pt", "step_3.pt"], remaining
    assert not list(tmp_path.glob("*.tmp")), "stale .tmp left after writes"


def test_lever_c_writer_error_reraised_on_training_thread(tmp_path):
    """A torch.save failure on the worker thread is re-raised on the training
    thread (at submit or join), not swallowed."""
    model = nn.Linear(4, 4)
    optim = torch.optim.AdamW(model.parameters(), lr=0.1)
    model(torch.randn(8, 4)).pow(2).sum().backward()
    optim.step()

    writer = _Stage5CheckpointWriter()
    # Point the write at a path whose parent does not exist -> the worker raises.
    bad_dir = tmp_path / "does_not_exist"
    raised = False
    try:
        _save_stage5_checkpoint(
            bad_dir, step=1, epoch=0, batch_idx=0, student=model,
            optim=optim, scheduler=None, writer=writer,
        )
        writer.join()  # error surfaces here if not at a later submit
    except Exception:
        raised = True
    finally:
        try:
            writer.close()
        except Exception:
            raised = True
    assert raised, "writer-thread error was swallowed, not re-raised"
