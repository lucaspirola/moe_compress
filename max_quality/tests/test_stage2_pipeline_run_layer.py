"""Tests for Stage2Pipeline.run_layer + the LegacyAdapter wiring (Task 6).

Three layers of coverage:
  1. Pipeline contract — phase-walk order, hook dispatch, partial_dir threading.
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
from moe_compress.stage2._framework import (
    PipelineContext,
    Stage2Pipeline,
)
from moe_compress.stage2.plugins.legacy_adapter import LegacyAdapter
from moe_compress.pipeline.plugin import PipelinePlugin
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

    def write_artifacts(self, ctx, partial_dir):
        self.calls.append("write_artifacts")
        return {}

    def on_layer_teardown(self, ctx):
        self.calls.append("on_layer_teardown")


def test_run_layer_visits_each_phase_in_canonical_order(tmp_path):
    """Stage2Pipeline.run_layer visits every phase in declared order, once per plugin."""
    plugin = _CountingPlugin()
    pipeline = Stage2Pipeline(plugins=[plugin])
    run_ctx = _make_run_ctx(
        model=object(), tokenizer=object(), config={},
        artifacts_dir=tmp_path, partial_dir=tmp_path, device="cpu",
    )
    pipeline.run_setup(run_ctx)
    pipeline.run_layer(_make_layer_ctx(run_ctx, layer_idx=0, layer_ref=object(),
                                       n_experts=4, target=2))
    pipeline.run_teardown(run_ctx)
    assert plugin.calls == [
        "on_run_setup",
        *list(Stage2Pipeline.phases),
        "on_run_teardown",
    ]


def test_phases_tuple_matches_t6_canonical_order():
    """The T7 phase tuple is the 9-element execution order (bump-loop is compound).

    Updated for T7: ``on_score`` is inserted between ``on_profile`` and
    ``compute_assignment`` to let ReapScoringPlugin publish ctx.scores/freq
    before LegacyAdapter.compute_assignment reads them.
    """
    assert Stage2Pipeline.phases == (
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


def test_run_layer_passes_partial_dir_to_write_artifacts(tmp_path):
    """write_artifacts receives partial_dir from any plugin that exposes it."""
    seen: dict[str, object] = {}

    class _SnoopPlugin:
        name = "snoop"

        def __init__(self, partial_dir):
            self.partial_dir = partial_dir

        def write_artifacts(self, ctx, partial_dir):
            seen["partial_dir"] = partial_dir
            return {}

    custom_partial = tmp_path / "my_partial"
    plugin = _SnoopPlugin(partial_dir=custom_partial)
    pipeline = Stage2Pipeline(plugins=[plugin])
    pipeline.run_layer(_make_layer_ctx(PipelineContext(), layer_idx=0,
                                       layer_ref=object(),
                                       n_experts=4, target=2))
    assert seen["partial_dir"] == custom_partial


def test_run_layer_threads_partial_dir_none_when_no_plugin_exposes_it(tmp_path):
    """A plugin without `partial_dir` attr still receives None at write_artifacts."""
    seen: dict[str, object] = {"partial_dir": "<sentinel>"}

    class _SnoopPlugin:
        name = "snoop"

        def write_artifacts(self, ctx, partial_dir):
            seen["partial_dir"] = partial_dir
            return {}

    plugin = _SnoopPlugin()
    pipeline = Stage2Pipeline(plugins=[plugin])
    pipeline.run_layer(_make_layer_ctx(PipelineContext(), layer_idx=0,
                                       layer_ref=object(),
                                       n_experts=4, target=2))
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
