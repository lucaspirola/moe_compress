"""Tests for the Stage 2 phase walk + the LegacyAdapter wiring.

Three layers of coverage:
  1. Pipeline contract — ``walk_phases`` order, hook dispatch, partial_dir
     threading via the ``partial_dir`` context slot.
  2. LegacyAdapter unit — each hook moves the right ctx state.
  3. End-to-end smoke — Stage 2 via the plugin pipeline produces a valid
     ``merge_map.json`` + a checkpoint directory. (The legacy escape hatch
     was removed in Task 18; the byte-identical gate it backed is retired.)
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from moe_compress import stage1
from moe_compress import stage2_reap_ream as _stage2_monolith
from moe_compress.stage2 import orchestrator as stage2_reap_ream
from moe_compress.budget.solver import BudgetDecomposition
from moe_compress.pipeline.context import PipelineContext
from moe_compress.stage2.orchestrator import (
    _STAGE2_LAYER_PHASES,
    _STAGE2_POST_ASSIGN_PHASES,
    _STAGE2_PRE_ASSIGN_PHASES,
    _run_assignment,
)
from moe_compress.stage2.plugins.legacy_adapter import LegacyAdapter
from moe_compress.pipeline.plugin import PipelinePlugin
from moe_compress.tools.phase_walker import walk_phases
from moe_compress.utils.model_io import iter_moe_layers


def test_legacy_adapter_structural_conformance():
    """LegacyAdapter satisfies the universal PipelinePlugin contract.

    LegacyAdapter cannot be bare-instantiated (its __init__ takes ~39
    keyword-only args), so its conformance is asserted at the class level:
    every PipelinePlugin metadata attribute is present with the right type,
    and the two universal core methods exist.
    """
    for attr in ("name", "paper", "config_key", "reads", "writes", "provides"):
        assert hasattr(LegacyAdapter, attr), f"LegacyAdapter missing {attr!r}"
    assert isinstance(LegacyAdapter.name, str)
    assert isinstance(LegacyAdapter.paper, str)
    assert isinstance(LegacyAdapter.config_key, str)
    assert isinstance(LegacyAdapter.reads, tuple)
    assert isinstance(LegacyAdapter.writes, tuple)
    assert isinstance(LegacyAdapter.provides, tuple)
    assert callable(getattr(LegacyAdapter, "is_enabled", None))
    assert callable(getattr(LegacyAdapter, "contribute_artifact", None))
    # The class object structurally satisfies the runtime_checkable Protocol.
    assert isinstance(LegacyAdapter, PipelinePlugin)


def _make_run_ctx(*, model, tokenizer, config, artifacts_dir,
                  partial_dir, device):
    """Build a root PipelineContext with the six Stage-2 run-scope slots."""
    rc = PipelineContext()
    rc.set("model", model)
    rc.set("tokenizer", tokenizer)
    rc.set("config", config)
    rc.set("artifacts_dir", artifacts_dir)
    rc.set("partial_dir", partial_dir)
    rc.set("device", device)
    return rc


def _make_layer_ctx(root, *, layer_idx, layer_ref, n_experts, target,
                    blacklist=()):
    """Open a per-layer child scope and populate the orchestrator's slots."""
    ctx = root.child()
    ctx.set("layer_idx", layer_idx)
    ctx.set("layer_ref", layer_ref)
    ctx.set("n_experts", n_experts)
    ctx.set("target", target)
    ctx.set("blacklist", tuple(blacklist))
    return ctx


class _TinyTokenizer:
    name_or_path = "tiny-tokenizer"
    eos_token_id = 0

    def __call__(self, text, *_, **__):
        return {"input_ids": [min(ord(c) % 32, 31) for c in (text or " ")]}

    def save_pretrained(self, *_args, **_kwargs):
        return None


def _noop_save(model, tokenizer, path, **kwargs):
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


@pytest.fixture
def patched_stage2(monkeypatch, tiny_config):
    from moe_compress.utils import calibration as cal_mod

    def _fake_build(tokenizer, spec, cache_dir=None):
        torch.manual_seed(spec.seed)
        return torch.randint(0, 32, (spec.num_sequences, spec.sequence_length),
                             dtype=torch.long)

    def _fake_slice(tokenizer, spec, num_samples, cache_dir=None):
        torch.manual_seed(spec.seed + 1)
        return torch.randint(0, 32, (num_samples, spec.sequence_length),
                             dtype=torch.long)

    monkeypatch.setattr(cal_mod, "build_calibration_tensor", _fake_build)
    monkeypatch.setattr(cal_mod, "build_super_expert_slice", _fake_slice)
    monkeypatch.setattr(stage2_reap_ream, "build_calibration_tensor", _fake_build)

    from moe_compress.utils import model_io as mio
    monkeypatch.setattr(mio, "save_compressed_checkpoint", _noop_save)
    monkeypatch.setattr(stage2_reap_ream, "save_compressed_checkpoint", _noop_save)
    return tiny_config


def _run_stage1(model, config, tmp_path):
    tokenizer = _TinyTokenizer()
    decomp = BudgetDecomposition(
        total_reduction_ratio=0.2,
        expert_prune_ratio=0.5,
        svd_rank_ratio=0.14,
        global_expert_budget=4,
        min_experts_per_layer=2,
        blacklisted_experts={},
    )
    stage1.run(model, tokenizer, config, tmp_path, decomp)


# ---------------------------------------------------------------------------
# Layer 1: pipeline contract
# ---------------------------------------------------------------------------


class _CountingPlugin:
    """Records every phase invocation in declaration order."""

    name = "counting"

    def __init__(self):
        self.calls: list[str] = []

    def on_run_setup(self, run_ctx):
        self.calls.append("on_run_setup")

    def on_run_teardown(self, run_ctx):
        self.calls.append("on_run_teardown")

    def on_layer_setup(self, ctx):
        self.calls.append("on_layer_setup")

    def on_profile(self, ctx):
        self.calls.append("on_profile")

    def on_score(self, ctx):
        self.calls.append("on_score")

    def compute_assignment(self, ctx):
        self.calls.append("compute_assignment")

    def pre_merge_snapshot(self, ctx):
        self.calls.append("pre_merge_snapshot")

    def merge(self, ctx):
        self.calls.append("merge")

    def post_merge(self, ctx):
        self.calls.append("post_merge")

    def write_artifacts(self, ctx):
        self.calls.append("write_artifacts")
        return {}

    def on_layer_teardown(self, ctx):
        self.calls.append("on_layer_teardown")


def test_run_layer_visits_each_phase_in_canonical_order(tmp_path):
    """The per-layer walk visits every phase in declared order, once per plugin.

    S2-5: the per-layer loop no longer walks ``compute_assignment`` as a plain
    ``walk_phases`` phase — the bump loop is the explicit ``_run_assignment``
    driver. The phase walks are now the pre-assign and post-assign halves, so
    ``_CountingPlugin`` records every phase EXCEPT ``compute_assignment``.
    """
    plugin = _CountingPlugin()
    plugins = [plugin]
    run_ctx = _make_run_ctx(
        model=object(), tokenizer=object(), config={},
        artifacts_dir=tmp_path, partial_dir=tmp_path, device="cpu",
    )
    layer_ctx = _make_layer_ctx(run_ctx, layer_idx=0, layer_ref=object(),
                                n_experts=4, target=2)
    walk_phases(("on_run_setup",), plugins, run_ctx)
    walk_phases(_STAGE2_PRE_ASSIGN_PHASES, plugins, layer_ctx)
    walk_phases(_STAGE2_POST_ASSIGN_PHASES, plugins, layer_ctx)
    walk_phases(("on_run_teardown",), plugins, run_ctx)
    assert plugin.calls == [
        "on_run_setup",
        *list(_STAGE2_PRE_ASSIGN_PHASES),
        *list(_STAGE2_POST_ASSIGN_PHASES),
        "on_run_teardown",
    ]
    # The compound bump-loop slot is driven by ``_run_assignment``, not the
    # plain phase walk — it must not appear in the walked-phase calls.
    assert "compute_assignment" not in plugin.calls


def test_phases_tuple_matches_t6_canonical_order():
    """The phase tuples are the canonical Stage-2 execution order.

    S2-5: the schedule is split into the pre-assign and post-assign halves;
    ``_STAGE2_LAYER_PHASES`` is the derived 9-tuple back-compat constant with
    the compound ``compute_assignment`` slot wedged between the two halves.
    """
    assert _STAGE2_PRE_ASSIGN_PHASES == (
        "on_layer_setup",
        "on_profile",
        "on_score",
    )
    assert _STAGE2_POST_ASSIGN_PHASES == (
        "pre_merge_snapshot",
        "merge",
        "post_merge",
        "write_artifacts",
        "on_layer_teardown",
    )
    # The derived back-compat 9-tuple = pre-assign + compute_assignment + post.
    assert _STAGE2_LAYER_PHASES == (
        _STAGE2_PRE_ASSIGN_PHASES
        + ("compute_assignment",)
        + _STAGE2_POST_ASSIGN_PHASES
    )
    assert _STAGE2_LAYER_PHASES == (
        "on_layer_setup",
        "on_profile",
        "on_score",
        "compute_assignment",
        "pre_merge_snapshot",
        "merge",
        "post_merge",
        "write_artifacts",
        "on_layer_teardown",
    )


def test_write_artifacts_reads_partial_dir_from_context(tmp_path):
    """write_artifacts reads partial_dir off the per-layer context slot.

    The orchestrator stores ``partial_dir`` on the run-scope context; a layer
    child inherits it. ``walk_phases`` passes only ``ctx`` — the plugin must
    pull the value via ``ctx.get("partial_dir")``.
    """
    seen: dict[str, object] = {}

    class _SnoopPlugin:
        name = "snoop"

        def write_artifacts(self, ctx):
            seen["partial_dir"] = ctx.get("partial_dir")
            return {}

    custom_partial = tmp_path / "my_partial"
    run_ctx = PipelineContext()
    run_ctx.set("partial_dir", custom_partial)
    child = _make_layer_ctx(run_ctx, layer_idx=0, layer_ref=object(),
                            n_experts=4, target=2)
    walk_phases(("write_artifacts",), [_SnoopPlugin()], child)
    assert seen["partial_dir"] == custom_partial


def test_write_artifacts_reads_partial_dir_none_in_no_resume_mode(tmp_path):
    """In no-resume mode partial_dir is None; write_artifacts must see None."""
    seen: dict[str, object] = {"partial_dir": "<sentinel>"}

    class _SnoopPlugin:
        name = "snoop"

        def write_artifacts(self, ctx):
            seen["partial_dir"] = ctx.get("partial_dir")
            return {}

    run_ctx = PipelineContext()
    run_ctx.set("partial_dir", None)
    child = _make_layer_ctx(run_ctx, layer_idx=0, layer_ref=object(),
                            n_experts=4, target=2)
    walk_phases(("write_artifacts",), [_SnoopPlugin()], child)
    assert seen["partial_dir"] is None


# ---------------------------------------------------------------------------
# Layer 2: LegacyAdapter unit coverage
# ---------------------------------------------------------------------------


def _build_minimal_adapter(model, tiny_config, tmp_path, *, moe_layers,
                           cov_acc=None, merge_map=None, mean_costs=None):
    from moe_compress.utils.activation_hooks import InputCovarianceAccumulator
    s2 = tiny_config["stage2_reap_ream"]
    if cov_acc is None:
        cov_acc = InputCovarianceAccumulator()
    if merge_map is None:
        merge_map = {}
    if mean_costs is None:
        mean_costs = []
    return LegacyAdapter(
        s2_cfg=s2, heal_cfg=stage2_reap_ream._HealConfig(s2),
        heal_device=None, xd_batches=None, batches=[],
        model=model,
        cov_acc=cov_acc, merge_map=merge_map,
        layer_mean_costs=mean_costs, partial_dir=tmp_path,
        max_group_cap=0, cost_sigma=float("inf"),
        cost_bump_ratio=0.1, min_active_tokens=1,
        assignment_solver="greedy", cost_alignment_cfg="pre",
        cost_output_token_cap=8, cost_whitening="none",
        cost_asymmetric=False, cost_topk_filter=2,
        capacity_util_threshold=0.0, em_refinement_rounds=0,
        em_convergence_break=True, two_opt_refine=False,
        sinkhorn_epsilon_init=1.0, sinkhorn_epsilon_final=0.01,
        sinkhorn_iters=10, skip_merge_percentile=100.0,
        expert_distill_steps=0, expert_distill_lr=1e-4,
        expert_distill_betas=(0.9, 0.95),
        expert_distill_token_cap=8,
        expert_distill_skip_singletons=True,
        expert_distill_plateau_steps=2,
        expert_distill_plateau_eps=1e-4,
        per_layer_target={ref.layer_idx: 2 for ref in moe_layers},
        blacklist={},
        artifacts_dir=tmp_path, device=None,
    )


def test_legacy_adapter_on_layer_setup_populates_accumulators(tiny_model, patched_stage2,
                                                              tmp_path):
    """on_layer_setup creates ream_acc + perm_cache; layer_input_acc None in default mode.

    Updated for T7: ``ctx.reap_acc`` is now created by ``ReapScoringPlugin.on_layer_setup``,
    which is registered before the LegacyAdapter in ``stage2_reap_ream.run``.
    We invoke it here to mirror the production wiring, then check that
    LegacyAdapter does NOT overwrite it.
    """
    _run_stage1(tiny_model, patched_stage2, tmp_path)
    from moe_compress.utils.activation_hooks import ReamCostAccumulator, ReapAccumulator
    from moe_compress.stage2.permutation_align import _PermAlignCache
    from moe_compress.stage2.plugins.reap_scoring import ReapScoringPlugin

    moe_layers = list(iter_moe_layers(tiny_model))
    adapter = _build_minimal_adapter(tiny_model, patched_stage2, tmp_path,
                                     moe_layers=moe_layers)
    ctx = _make_layer_ctx(PipelineContext(),
                          layer_idx=moe_layers[0].layer_idx,
                          layer_ref=moe_layers[0],
                          n_experts=moe_layers[0].num_routed_experts, target=2)
    ReapScoringPlugin().on_layer_setup(ctx)
    adapter.on_layer_setup(ctx)
    assert isinstance(ctx.get("reap_acc"), ReapAccumulator)
    assert isinstance(ctx.get("ream_acc"), ReamCostAccumulator)
    assert isinstance(ctx.get("perm_cache"), _PermAlignCache)
    # default mode: expert_distill_steps=0 and cost_alignment_cfg="pre" → no input capture.
    assert ctx.get("layer_input_acc") is None


def test_legacy_adapter_on_layer_teardown_clears_state(tiny_model, patched_stage2,
                                                       tmp_path):
    """on_layer_teardown drops the per-layer accumulators."""
    _run_stage1(tiny_model, patched_stage2, tmp_path)
    from moe_compress.stage2.plugins.reap_scoring import ReapScoringPlugin

    moe_layers = list(iter_moe_layers(tiny_model))
    adapter = _build_minimal_adapter(tiny_model, patched_stage2, tmp_path,
                                     moe_layers=moe_layers)
    ctx = _make_layer_ctx(PipelineContext(),
                          layer_idx=moe_layers[0].layer_idx,
                          layer_ref=moe_layers[0],
                          n_experts=moe_layers[0].num_routed_experts, target=2)
    ReapScoringPlugin().on_layer_setup(ctx)
    adapter.on_layer_setup(ctx)
    assert ctx.get("reap_acc") is not None and ctx.get("ream_acc") is not None
    # teardown nulls all 8 per-layer slots unconditionally (overwrite=True is
    # an upsert), so no pre-seed is needed even for slots the pre-merge / merge
    # phases would normally write.
    adapter.on_layer_teardown(ctx)
    assert ctx.get("reap_acc") is None
    assert ctx.get("ream_acc") is None
    assert ctx.get("perm_cache") is None
    assert ctx.get("layer_input_acc") is None
    assert ctx.get("pre_merge_weights") is None
    assert ctx.get("distill_state") is None


# ---------------------------------------------------------------------------
# Layer 2b: _run_assignment driver coverage (S2-5)
# ---------------------------------------------------------------------------


# The five fine-grained assignment slots ``_run_assignment`` drives per bump,
# in canonical call order. ``select_alignment`` (S2-10 — the capacity gate),
# ``compute_cost`` / ``apply_cost_mask`` / ``solve_assignment`` are
# single-winner ``dispatch_first`` slots; ``refine_assignment`` is a CHAIN
# (S2-9 — two-opt then EM, every enabled plugin runs). The probe below is one
# chain link, so it still records exactly one ``refine_assignment`` call per
# bump and the canonical five-slot order holds.
_ASSIGNMENT_SLOTS = (
    "select_alignment",
    "compute_cost",
    "apply_cost_mask",
    "solve_assignment",
    "refine_assignment",
)


def test_legacy_adapter_exposes_four_assignment_slots():
    """LegacyAdapter exposes the four fine-grained cost/mask/solve/refine
    assignment slots and no longer carries the monolithic ``compute_assignment``
    hook (S2-5).

    S2-10: ``select_alignment`` (the capacity gate) is NOT a LegacyAdapter slot
    — it is serviced by ``CapacityGatePlugin`` — so it is excluded here.
    """
    for slot in ("compute_cost", "apply_cost_mask",
                 "solve_assignment", "refine_assignment"):
        assert callable(getattr(LegacyAdapter, slot, None)), (
            f"LegacyAdapter missing assignment slot {slot!r}"
        )
    assert not hasattr(LegacyAdapter, "select_alignment"), (
        "LegacyAdapter must NOT declare select_alignment — the capacity gate "
        "is serviced by CapacityGatePlugin (S2-10)"
    )
    assert not hasattr(LegacyAdapter, "compute_assignment"), (
        "LegacyAdapter.compute_assignment must be gone after the S2-5 "
        "decomposition — the bump loop now lives in orchestrator._run_assignment"
    )


def _make_capacity_gate():
    """Build a CapacityGatePlugin with the same gate knobs ``_build_minimal_adapter``
    passes the LegacyAdapter (max_group_cap=0 / capacity_util_threshold=0.0 /
    cost_alignment_cfg="pre" / cost_asymmetric=False). S2-10: the hand-built
    plugin lists need it to service the ``select_alignment`` slot.
    """
    from moe_compress.stage2.plugins.capacity_gate import CapacityGatePlugin
    return CapacityGatePlugin(
        max_group_cap=0, capacity_util_threshold=0.0,
        cost_alignment_cfg="pre", cost_asymmetric=False,
    )


def _prepare_layer_for_assignment(tiny_model, patched_stage2, tmp_path):
    """Run stage1 + the pre-assign phases so a layer ctx is ready for
    ``_run_assignment``. Returns ``(adapter, plugins, ctx)``."""
    from moe_compress.stage2.plugins.reap_scoring import ReapScoringPlugin

    _run_stage1(tiny_model, patched_stage2, tmp_path)
    moe_layers = list(iter_moe_layers(tiny_model))
    adapter = _build_minimal_adapter(tiny_model, patched_stage2, tmp_path,
                                     moe_layers=moe_layers)
    # S2-10: CapacityGatePlugin services the select_alignment slot — without it
    # _run_assignment's ``assert _alignment is not None`` would fire.
    plugins = [ReapScoringPlugin(), _make_capacity_gate(), adapter]
    layer_ref = moe_layers[0]
    ctx = _make_layer_ctx(PipelineContext(),
                          layer_idx=layer_ref.layer_idx,
                          layer_ref=layer_ref,
                          n_experts=layer_ref.num_routed_experts, target=2)
    walk_phases(_STAGE2_PRE_ASSIGN_PHASES, plugins, ctx)
    return adapter, plugins, ctx


def test_run_assignment_dispatches_slots(tiny_model, patched_stage2, tmp_path):
    """`_run_assignment` drives the four assignment slots, in canonical order,
    once per bump iteration.

    A probe plugin is inserted ahead of the LegacyAdapter: it records every
    slot call (in order) and delegates to the adapter's real slot method so the
    bump loop still produces a valid grouping. The adapter stays in the plugin
    list so ``_run_assignment``'s ``isinstance(p, LegacyAdapter)`` lookup for
    the run-scope scratchpad still resolves.

    S2-9: ``refine_assignment`` is now a CHAIN — the probe is one chain link
    and its ``refine_assignment`` delegates to ``adapter.refine_assignment``
    (the neutered dead fallback, returns ``None``). The chain loop handles the
    ``None`` cleanly, so the probe still records exactly one
    ``refine_assignment`` call per bump and the canonical slot order holds.

    S2-10: ``select_alignment`` (the capacity gate) is the first slot
    ``_run_assignment`` dispatches per bump — the probe delegates it to a real
    ``CapacityGatePlugin`` so the gate slots are published before
    ``compute_cost`` reads them back.
    """
    adapter, plugins, ctx = _prepare_layer_for_assignment(
        tiny_model, patched_stage2, tmp_path)

    calls: list[str] = []
    gate = _make_capacity_gate()

    class _SlotProbe:
        name = "slot_probe"

        def select_alignment(self, ctx):
            calls.append("select_alignment")
            return gate.select_alignment(ctx)

        def compute_cost(self, ctx):
            calls.append("compute_cost")
            return adapter.compute_cost(ctx)

        def apply_cost_mask(self, ctx, delta):
            calls.append("apply_cost_mask")
            return adapter.apply_cost_mask(ctx, delta)

        def solve_assignment(self, ctx, delta):
            calls.append("solve_assignment")
            return adapter.solve_assignment(ctx, delta)

        def refine_assignment(self, ctx, asg, delta):
            calls.append("refine_assignment")
            return adapter.refine_assignment(ctx, asg, delta)

    # Probe first so dispatch_first lands on it; adapter still present for the
    # isinstance lookup + run-scope state.
    from moe_compress.stage2.plugins.reap_scoring import ReapScoringPlugin
    probed = [_SlotProbe(), ReapScoringPlugin(), adapter]
    # ReapScoringPlugin.on_score already ran in _prepare_layer_for_assignment;
    # we reuse the same ctx (scores/freq already published).
    _run_assignment(probed, ctx)

    # At least one bump iteration ran the success branch (the tiny model is
    # feasible with max_group_cap=0), so every slot fired.
    assert calls, "no assignment slots were dispatched"
    n_slots = len(_ASSIGNMENT_SLOTS)
    n_bumps = len(calls) // n_slots
    assert n_bumps >= 1
    # Per-bump invocation: the recorded calls are whole repeats of the
    # canonical five-slot order.
    assert calls == list(_ASSIGNMENT_SLOTS) * n_bumps, (
        f"slot calls not in canonical per-bump order: {calls}"
    )


def test_run_assignment_publishes_layer_output_slots(tiny_model, patched_stage2,
                                                     tmp_path):
    """`_run_assignment` publishes every per-layer output slot the post-assign
    phases consume — all resolve to non-sentinel values after the driver runs."""
    adapter, plugins, ctx = _prepare_layer_for_assignment(
        tiny_model, patched_stage2, tmp_path)

    _run_assignment(plugins, ctx)

    # The ~16 output slots the retired compute_assignment used to set; every
    # one must resolve (ctx.get raises KeyError on an unset slot).
    output_slots = (
        "protected", "ream_centroid_ids", "ream_noncentroid_ids",
        "assignment", "delta", "grouped", "mean_assigned_cost", "n_protected",
        "assigned_cost", "n_assigned", "b_fail", "c_fail", "em_rounds_done",
        "effective_cost_alignment", "effective_cost_asymmetric",
        "capacity_util_value", "effective_target",
    )
    for slot in output_slots:
        ctx.get(slot)  # raises KeyError if the slot was never published
    # Spot-check structural sanity of a few slots.
    assert isinstance(ctx.get("grouped"), dict)
    assert isinstance(ctx.get("ream_centroid_ids"), tuple)
    assert isinstance(ctx.get("protected"), tuple)
    assert isinstance(ctx.get("b_fail"), bool)
    assert isinstance(ctx.get("c_fail"), bool)
    assert isinstance(ctx.get("effective_target"), int)


# ---------------------------------------------------------------------------
# Layer 3: end-to-end smoke test
# ---------------------------------------------------------------------------


def test_stage2_pipeline_produces_valid_outputs(tiny_model, patched_stage2,
                                                tmp_path):
    """Stage 2 via the plugin pipeline runs end-to-end and produces a valid
    ``merge_map.json`` + a checkpoint directory.

    Task 18 removed the ``MOE_STAGE2_LEGACY_LOOP`` escape hatch, so the
    pipeline path is the only path. The byte-identical-vs-legacy gate this
    test used to be is retired: the property it guarded was verified
    continuously across T6-T17 while both paths still existed.
    """
    _run_stage1(tiny_model, patched_stage2, tmp_path)

    out_dir = stage2_reap_ream.run(
        tiny_model, _TinyTokenizer(), patched_stage2, tmp_path,
        device=None, no_resume=True,
    )

    # 1. run() returns the checkpoint dir.
    assert out_dir == tmp_path / "stage2_pruned"
    assert out_dir.is_dir()

    # 2. merge_map.json exists and is structurally valid: keys are layer
    #    indices, values map centroid -> member-list.
    merge_map_path = out_dir / "merge_map.json"
    assert merge_map_path.is_file()
    merge_map = json.loads(merge_map_path.read_text())
    assert isinstance(merge_map, dict)
    for layer_key, groups in merge_map.items():
        int(layer_key)  # layer indices are int-coercible
        assert isinstance(groups, dict)
        for centroid, members in groups.items():
            int(centroid)
            assert isinstance(members, list)

    # 3. every MoE layer still has a non-degenerate router after the pass
    #    (a coarse "no layer silently blew up to zero experts" check —
    #    exact target-count correctness is covered by the solver/merge
    #    tests in the full gate suite).
    for ref in iter_moe_layers(tiny_model):
        assert ref.router.weight.shape[0] >= 1


def test_stage2_pipeline_path_handles_resume(tiny_model, patched_stage2,
                                             tmp_path, monkeypatch):
    """Resume path through the pipeline still skips completed layers.

    Crashes mid-pipeline-run after layer 0, then re-runs from the same
    pre-Stage-2 snapshot. Layer 0 must be skipped (re-applied from
    ``_stage2_partial``) and the final ``merge_map`` must match a clean run.

    Mirrors the structure of ``test_smoke_stage2_resume.py::
    test_stage2_resume_skips_completed_layers`` but pins the pipeline path
    explicitly: a crash inside the pipeline still drops partial-checkpoint
    files that the next ``run()`` invocation picks up.
    """
    _run_stage1(tiny_model, patched_stage2, tmp_path)

    # Snapshot pre-Stage-2 model so the post-crash re-run starts from the
    # same expert layout as the original (the crashed run already mutated
    # layer 0 in place).
    pre_s2 = copy.deepcopy(tiny_model)

    moe_layers = list(iter_moe_layers(tiny_model))
    assert len(moe_layers) >= 2, "Need at least 2 MoE layers for this test"

    # First run: crash after layer 0 fully processed.
    # ``_profile_layer`` is dispatched by ``LegacyAdapter.on_profile`` via the
    # ``moe_compress.stage2_reap_ream`` namespace (so monkeypatching is
    # observable) — patch that module, not the slim orchestrator.
    original_profile = _stage2_monolith._profile_layer
    call_count = [0]

    def _crashing_profile(model, layer_ref, batches, reap_acc, cov_acc, ream_acc, **kwargs):
        call_count[0] += 1
        if call_count[0] > 1:
            raise RuntimeError("simulated crash after layer 0")
        return original_profile(model, layer_ref, batches, reap_acc, cov_acc, ream_acc, **kwargs)

    monkeypatch.setattr(_stage2_monolith, "_profile_layer", _crashing_profile)
    with pytest.raises(RuntimeError, match="simulated crash"):
        stage2_reap_ream.run(tiny_model, _TinyTokenizer(), patched_stage2,
                             tmp_path, device=None)

    # Restore profile, restart from the pre-stage-2 snapshot (the in-memory
    # `tiny_model` has been partially mutated by the crashed run).
    monkeypatch.setattr(_stage2_monolith, "_profile_layer", original_profile)
    resume_model = copy.deepcopy(pre_s2)
    stage2_reap_ream.run(resume_model, _TinyTokenizer(), patched_stage2,
                         tmp_path, device=None)

    # Clean baseline tree.
    clean_dir = tmp_path.parent / (tmp_path.name + "_clean")
    clean_dir.mkdir()
    clean_model = copy.deepcopy(pre_s2)
    _run_stage1(clean_model, patched_stage2, clean_dir)
    stage2_reap_ream.run(clean_model, _TinyTokenizer(), patched_stage2,
                         clean_dir, device=None, no_resume=True)

    resume_map = json.loads((tmp_path / "stage2_pruned" / "merge_map.json").read_text())
    clean_map = json.loads((clean_dir / "stage2_pruned" / "merge_map.json").read_text())
    assert resume_map == clean_map, (
        f"resume merge_map mismatch:\n  resume={resume_map}\n  clean={clean_map}"
    )
