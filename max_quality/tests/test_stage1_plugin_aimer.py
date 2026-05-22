"""Unit tests for ``moe_compress.stage1.plugins.aimer`` (sub-task 6).

Verifies:

1. The plugin's class-level Protocol attributes match the registry
   contract (``PipelinePlugin``).
2. ``is_enabled`` reads
   ``config["stage1_grape"]["super_expert_detection"]["aimer_enabled"]``
   (default True).
3. ``run`` populates the two write slots ``aimer_scores`` and
   ``bottom_pct_by_layer`` — including the disabled-path which still
   writes empty dicts (matching the legacy inline behaviour).
4. ``contribute_artifact`` returns the canonical three-key payload
   byte-equivalent to the legacy inline ``aimer_payload`` literal
   (pre-sub-task-6).
5. Missing required slots cause ``KeyError`` so the orchestrator
   (sub-task 10) gets a clear contract violation rather than a silent
   misbehaviour.
6. The relocated ``_get_expert_down_proj_weight`` helper covers all
   four MoE-expert-layout variants + the unknown-layout raise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from moe_compress.pipeline.candidates import CandidateBag
from moe_compress.pipeline.plugin import PipelinePlugin
from moe_compress.pipeline.context import PipelineContext
from moe_compress.stage1.plugins.aimer import (
    AimerDetectorPlugin,
    _get_expert_down_proj_weight,
)


# ---------------------------------------------------------------------------
# Test fakes — MoE-layer stubs that the plugin's ``run`` iterates.
# ---------------------------------------------------------------------------


@dataclass
class _FakeExperts:
    """Stub ``experts_module`` — fused-parameter layout by default."""

    down_proj: torch.nn.Parameter


@dataclass
class _FakeRef:
    layer_idx: int
    num_routed_experts: int
    experts_module: object


def _fake_ref(layer_idx: int = 0, n: int = 4, d_hid: int = 8, d_int: int = 16, seed: int = 0) -> _FakeRef:
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(n, d_hid, d_int, generator=g)
    return _FakeRef(
        layer_idx=layer_idx,
        num_routed_experts=n,
        experts_module=_FakeExperts(down_proj=torch.nn.Parameter(w)),
    )


def _default_config(aimer_enabled: bool = True, aimer_bottom_pct: float = 0.5) -> dict:
    return {
        "stage1_grape": {
            "super_expert_detection": {
                "aimer_enabled": aimer_enabled,
                "aimer_bottom_pct": aimer_bottom_pct,
                "aimer_layer_max_fraction": 0.1,
            }
        }
    }


def _populated_ctx(
    moe_layers=None,
    L=None,
    config=None,
    per_expert_max=None,
    a_max=None,
    candidate_bag=None,
) -> PipelineContext:
    ctx = PipelineContext()
    ctx.set("moe_layers", moe_layers if moe_layers is not None else [_fake_ref()])
    ctx.set("L", L if L is not None else {0})
    ctx.set("config", config if config is not None else _default_config())
    # Sub-task 8: the extended ``run`` reads ``max_acc`` / ``a_max`` /
    # ``candidate_bag`` for the candidate-add step. Defaults are chosen so
    # the layer-max gate skips every layer (layer_expert_max=0.0 <=
    # aimer_layer_max_fraction * a_max=0.0) — no spurious candidates for
    # tests that only assert ``aimer_scores`` / ``bottom_pct_by_layer``.
    ctx.set(
        "max_acc",
        SimpleNamespace(
            per_expert_max=per_expert_max if per_expert_max is not None else {}
        ),
    )
    ctx.set("a_max", a_max if a_max is not None else 0.0)
    ctx.set(
        "candidate_bag",
        candidate_bag if candidate_bag is not None else CandidateBag(),
    )
    return ctx


# ---------------------------------------------------------------------------
# Protocol-attribute contract
# ---------------------------------------------------------------------------


def test_plugin_protocol_attributes():
    p = AimerDetectorPlugin()
    assert p.name == "aimer"
    assert p.paper.startswith("AIMER")
    assert p.config_key == "stage1_grape.super_expert_detection.aimer_enabled"
    assert p.reads == (
        "moe_layers",
        "L",
        "config",
        "max_acc",
        "a_max",
        "candidate_bag",
    )
    assert p.writes == ("aimer_scores", "bottom_pct_by_layer", "candidate_bag")
    assert p.provides == ()


def test_plugin_is_runtime_checkable_pipelineplugin():
    assert isinstance(AimerDetectorPlugin(), PipelinePlugin)


# ---------------------------------------------------------------------------
# ``is_enabled`` config branching
# ---------------------------------------------------------------------------


def test_plugin_is_enabled_default_true():
    assert AimerDetectorPlugin().is_enabled({}) is True


def test_plugin_is_enabled_explicit_false():
    cfg = {"stage1_grape": {"super_expert_detection": {"aimer_enabled": False}}}
    assert AimerDetectorPlugin().is_enabled(cfg) is False


def test_plugin_is_enabled_explicit_true():
    cfg = {"stage1_grape": {"super_expert_detection": {"aimer_enabled": True}}}
    assert AimerDetectorPlugin().is_enabled(cfg) is True


# ---------------------------------------------------------------------------
# ``run`` — populate the two write slots
# ---------------------------------------------------------------------------


def test_plugin_run_writes_aimer_scores_and_bottom_pct():
    plugin = AimerDetectorPlugin()
    ref = _fake_ref(layer_idx=0, n=4, seed=42)
    ctx = _populated_ctx(moe_layers=[ref])

    plugin.run(ctx)

    scores = ctx.get("aimer_scores")
    bottom = ctx.get("bottom_pct_by_layer")

    # Four (layer, expert) keys with floats in (0, 1].
    assert set(scores.keys()) == {(0, 0), (0, 1), (0, 2), (0, 3)}
    for k, v in scores.items():
        assert isinstance(v, float)
        assert 0.0 < v <= 1.0, f"AIMER score for {k} out of (0, 1]: {v}"

    # bottom_pct=0.5, n=4 → k=round(4*0.5)=2 experts kept per layer (lowest first).
    assert set(bottom.keys()) == {0}
    assert len(bottom[0]) == 2
    # Lowest-score-first ordering.
    layer_scores = sorted(scores.items(), key=lambda kv: kv[1])
    expected_two_lowest = [layer_scores[0][0][1], layer_scores[1][0][1]]
    assert bottom[0] == expected_two_lowest


def test_plugin_run_disabled_writes_empty_dicts():
    plugin = AimerDetectorPlugin()
    ctx = _populated_ctx(config=_default_config(aimer_enabled=False))

    plugin.run(ctx)

    assert ctx.get("aimer_scores") == {}
    assert ctx.get("bottom_pct_by_layer") == {}


def test_plugin_run_handles_empty_moe_layers():
    plugin = AimerDetectorPlugin()
    ctx = _populated_ctx(moe_layers=[], L=set())

    plugin.run(ctx)

    assert ctx.get("aimer_scores") == {}
    assert ctx.get("bottom_pct_by_layer") == {}


def test_plugin_run_multi_layer_writes_all_layers():
    plugin = AimerDetectorPlugin()
    refs = [_fake_ref(layer_idx=0, n=4, seed=0), _fake_ref(layer_idx=3, n=4, seed=1)]
    ctx = _populated_ctx(moe_layers=refs, L={0, 3})

    plugin.run(ctx)

    scores = ctx.get("aimer_scores")
    assert {li for (li, _e) in scores} == {0, 3}
    bottom = ctx.get("bottom_pct_by_layer")
    assert set(bottom.keys()) == {0, 3}


# ---------------------------------------------------------------------------
# ``run`` — sub-task-8 candidate-add step (shared CandidateBag, "aimer" tag)
# ---------------------------------------------------------------------------


def test_plugin_run_adds_candidates_with_aimer_tag():
    """Mirror the AIMER branch of the legacy ``_collect_candidates``.

    200 fillers + one low-AIMER outlier (expert 5) in layer 0; with
    ``aimer_bottom_pct`` small enough that only expert 5 is bottom-pct
    selected, and a per-layer activation max well above the gate
    threshold, the bag must carry exactly ``{0: [5]}`` under the
    ``"aimer"`` tag.
    """
    plugin = AimerDetectorPlugin()

    # 201 experts in a single layer-0 MoE ref; expert 5's down_proj weight
    # is highly concentrated (one large element, rest zero) → AIMER score
    # (l1 / (sqrt(n)*l2)) is driven toward its minimum. AIMER is scale-
    # invariant, so a sparse weight — not a small one — yields the low score.
    n = 201
    g = torch.Generator().manual_seed(7)
    w = torch.randn(n, 8, 16, generator=g)
    w[5] = torch.zeros(8, 16)
    w[5][0, 0] = 1.0  # one non-zero element → maximally concentrated
    ref = _FakeRef(
        layer_idx=0,
        num_routed_experts=n,
        experts_module=_FakeExperts(down_proj=torch.nn.Parameter(w)),
    )

    # bottom_pct small → k = max(1, round(201 * 0.005)) = 1 → only the lowest.
    config = _default_config(aimer_enabled=True, aimer_bottom_pct=0.005)
    # per_expert_max: layer-0 max = 100.0; a_max=100.0; gate threshold = 10.0.
    per_expert_max = {(0, e): 1.0 for e in range(n)}
    per_expert_max[(0, 5)] = 100.0
    bag = CandidateBag()
    ctx = _populated_ctx(
        moe_layers=[ref],
        L={0},
        config=config,
        per_expert_max=per_expert_max,
        a_max=100.0,
        candidate_bag=bag,
    )

    plugin.run(ctx)

    assert bag.by_tag("aimer") == {0: [5]}


def test_plugin_run_disabled_adds_no_candidates():
    plugin = AimerDetectorPlugin()
    bag = CandidateBag()
    ctx = _populated_ctx(
        config=_default_config(aimer_enabled=False),
        candidate_bag=bag,
    )

    plugin.run(ctx)

    assert len(bag) == 0


def test_plugin_run_gates_on_layer_expert_max():
    """When ``layer_expert_max <= aimer_layer_max_fraction * a_max`` the
    layer's AIMER candidates are dropped even though bottom-pct picked
    them. Set fraction=0.5, a_max=10.0 → threshold 5.0; layer max 4.0
    fails the gate."""
    plugin = AimerDetectorPlugin()

    n = 4
    g = torch.Generator().manual_seed(3)
    w = torch.randn(n, 8, 16, generator=g)
    ref = _FakeRef(
        layer_idx=0,
        num_routed_experts=n,
        experts_module=_FakeExperts(down_proj=torch.nn.Parameter(w)),
    )

    config = {
        "stage1_grape": {
            "super_expert_detection": {
                "aimer_enabled": True,
                "aimer_bottom_pct": 0.5,  # picks 2 experts → non-empty bottom-pct
                "aimer_layer_max_fraction": 0.5,
            }
        }
    }
    # Layer-0 activation max is 4.0 < 0.5 * 10.0 = 5.0 → gate fails.
    per_expert_max = {(0, 0): 4.0, (0, 1): 3.0, (0, 2): 2.0, (0, 3): 1.0}
    bag = CandidateBag()
    ctx = _populated_ctx(
        moe_layers=[ref],
        L={0},
        config=config,
        per_expert_max=per_expert_max,
        a_max=10.0,
        candidate_bag=bag,
    )

    plugin.run(ctx)

    # bottom_pct_by_layer is non-empty (the precondition for the gate to matter).
    assert ctx.get("bottom_pct_by_layer")
    # ...but the layer-max gate dropped every candidate.
    assert len(bag) == 0


# ---------------------------------------------------------------------------
# ``contribute_artifact`` — three-key block, byte-equivalent
# ---------------------------------------------------------------------------


def _ctx_with_artifact_inputs(
    aimer_scores: dict | None = None,
    bottom_pct_by_layer: dict | None = None,
    candidates: dict | None = None,
) -> PipelineContext:
    ctx = PipelineContext()
    ctx.set(
        "aimer_scores",
        aimer_scores if aimer_scores is not None else {(0, 0): 0.5, (0, 1): 0.7},
    )
    ctx.set(
        "bottom_pct_by_layer",
        bottom_pct_by_layer if bottom_pct_by_layer is not None else {0: [0]},
    )
    ctx.set(
        "candidates",
        candidates
        if candidates is not None
        else {(0, 1): ["aimer"], (0, 2): ["sink_token", "aimer"], (0, 3): ["phase_c"]},
    )
    return ctx


def test_plugin_contribute_artifact_three_keys():
    plugin = AimerDetectorPlugin()
    ctx = _ctx_with_artifact_inputs()

    payload = plugin.contribute_artifact(ctx)

    assert set(payload.keys()) == {"scores", "bottom_pct_per_layer", "candidates"}
    assert isinstance(payload["scores"], dict)
    assert isinstance(payload["bottom_pct_per_layer"], dict)
    assert isinstance(payload["candidates"], dict)


def test_plugin_contribute_artifact_scores_keying():
    plugin = AimerDetectorPlugin()
    ctx = _ctx_with_artifact_inputs()

    payload = plugin.contribute_artifact(ctx)

    import re
    for key, value in payload["scores"].items():
        assert re.match(r"^L\d+E\d+$", key), f"unexpected key {key!r}"
        assert value is None or isinstance(value, float)


def test_plugin_contribute_artifact_scrubs_nan_inf():
    plugin = AimerDetectorPlugin()
    scores = {
        (0, 0): 0.5,
        (0, 1): float("nan"),
        (0, 2): float("inf"),
        (0, 3): float("-inf"),
    }
    ctx = _ctx_with_artifact_inputs(
        aimer_scores=scores,
        bottom_pct_by_layer={0: [0]},
        candidates={},
    )

    payload = plugin.contribute_artifact(ctx)

    assert payload["scores"]["L0E0"] == 0.5
    assert payload["scores"]["L0E1"] is None
    assert payload["scores"]["L0E2"] is None
    assert payload["scores"]["L0E3"] is None


def test_plugin_contribute_artifact_candidates_inversion():
    plugin = AimerDetectorPlugin()
    candidates = {
        (0, 1): ["aimer"],
        (0, 2): ["sink_token", "aimer"],
        (0, 3): ["phase_c"],  # no aimer tag → excluded
        (1, 0): ["aimer"],
    }
    ctx = _ctx_with_artifact_inputs(
        aimer_scores={},
        bottom_pct_by_layer={},
        candidates=candidates,
    )

    payload = plugin.contribute_artifact(ctx)

    assert payload["candidates"] == {"0": [1, 2], "1": [0]}


def test_plugin_contribute_artifact_bottom_pct_per_layer_keying():
    plugin = AimerDetectorPlugin()
    ctx = _ctx_with_artifact_inputs(
        aimer_scores={},
        bottom_pct_by_layer={0: [3, 1], 7: [0]},
        candidates={},
    )

    payload = plugin.contribute_artifact(ctx)

    assert payload["bottom_pct_per_layer"] == {"0": [3, 1], "7": [0]}


def test_plugin_contribute_artifact_byte_equivalent_to_legacy_inline():
    """Byte-anchor — mirror the inline ``aimer_payload`` literal exactly.

    Legacy logic (pre-sub-task-6):

        aimer_payload = {
            "scores": {
                f"L{li}E{e}": safe_float(v)
                for (li, e), v in aimer_scores.items()
            },
            "bottom_pct_per_layer": {
                str(li): list(exps) for li, exps in bottom_pct_by_layer.items()
            },
            "candidates": _candidates_by_provenance("aimer"),
        }
    """
    from moe_compress.pipeline.safe_json import safe_float

    aimer_scores = {(0, 0): 0.5, (0, 1): float("nan"), (1, 2): 0.99}
    bottom_pct_by_layer = {0: [1], 1: [2]}
    candidates = {
        (0, 0): ["aimer", "phase_c"],
        (0, 1): ["aimer"],
        (1, 2): ["aimer"],
    }

    # Independent re-build of the legacy literal.
    def _candidates_by_provenance(tag: str) -> dict[str, list[int]]:
        out: dict[int, list[int]] = {}
        for (li, e), tags in candidates.items():
            if tag in tags:
                out.setdefault(int(li), []).append(int(e))
        return {str(li): sorted(es) for li, es in out.items()}

    expected = {
        "scores": {
            f"L{li}E{e}": safe_float(v) for (li, e), v in aimer_scores.items()
        },
        "bottom_pct_per_layer": {
            str(li): list(exps) for li, exps in bottom_pct_by_layer.items()
        },
        "candidates": _candidates_by_provenance("aimer"),
    }

    plugin = AimerDetectorPlugin()
    ctx = _ctx_with_artifact_inputs(
        aimer_scores=aimer_scores,
        bottom_pct_by_layer=bottom_pct_by_layer,
        candidates=candidates,
    )
    payload = plugin.contribute_artifact(ctx)

    assert payload == expected


# ---------------------------------------------------------------------------
# Missing-slot errors — KeyError per slot
# ---------------------------------------------------------------------------


def test_plugin_run_rejects_missing_moe_layers():
    plugin = AimerDetectorPlugin()
    ctx = PipelineContext()
    ctx.set("L", set())
    ctx.set("config", _default_config())

    with pytest.raises(KeyError, match="moe_layers"):
        plugin.run(ctx)


def test_plugin_run_rejects_missing_L():
    plugin = AimerDetectorPlugin()
    ctx = PipelineContext()
    ctx.set("moe_layers", [])
    ctx.set("config", _default_config())

    with pytest.raises(KeyError, match="'L'"):
        plugin.run(ctx)


def test_plugin_run_rejects_missing_config():
    plugin = AimerDetectorPlugin()
    ctx = PipelineContext()
    ctx.set("moe_layers", [])
    ctx.set("L", set())

    with pytest.raises(KeyError, match="config"):
        plugin.run(ctx)


def _ctx_with_aimer_candidates_minus(skip_slot: str) -> PipelineContext:
    """Build a run-ready ctx that produces a non-empty ``bottom_pct_by_layer``
    (so the candidate-add block executes) but omits one of the three new
    sub-task-8 read slots (``max_acc`` / ``a_max`` / ``candidate_bag``)."""
    g = torch.Generator().manual_seed(11)
    w = torch.randn(4, 8, 16, generator=g)
    ref = _FakeRef(
        layer_idx=0,
        num_routed_experts=4,
        experts_module=_FakeExperts(down_proj=torch.nn.Parameter(w)),
    )
    ctx = PipelineContext()
    ctx.set("moe_layers", [ref])
    ctx.set("L", {0})
    ctx.set("config", _default_config(aimer_enabled=True, aimer_bottom_pct=0.5))
    if skip_slot != "max_acc":
        ctx.set("max_acc", SimpleNamespace(per_expert_max={(0, 0): 100.0}))
    if skip_slot != "a_max":
        ctx.set("a_max", 100.0)
    if skip_slot != "candidate_bag":
        ctx.set("candidate_bag", CandidateBag())
    return ctx


def test_plugin_run_rejects_missing_max_acc():
    plugin = AimerDetectorPlugin()
    ctx = _ctx_with_aimer_candidates_minus("max_acc")

    with pytest.raises(KeyError, match="max_acc"):
        plugin.run(ctx)


def test_plugin_run_rejects_missing_a_max():
    plugin = AimerDetectorPlugin()
    ctx = _ctx_with_aimer_candidates_minus("a_max")

    with pytest.raises(KeyError, match="a_max"):
        plugin.run(ctx)


def test_plugin_run_rejects_missing_candidate_bag():
    plugin = AimerDetectorPlugin()
    ctx = _ctx_with_aimer_candidates_minus("candidate_bag")

    with pytest.raises(KeyError, match="candidate_bag"):
        plugin.run(ctx)


def test_plugin_contribute_artifact_rejects_missing_candidates():
    plugin = AimerDetectorPlugin()
    ctx = PipelineContext()
    ctx.set("aimer_scores", {})
    ctx.set("bottom_pct_by_layer", {})

    with pytest.raises(KeyError, match="candidates"):
        plugin.contribute_artifact(ctx)


def test_plugin_contribute_artifact_rejects_missing_aimer_scores():
    plugin = AimerDetectorPlugin()
    ctx = PipelineContext()
    ctx.set("candidates", {})
    ctx.set("bottom_pct_by_layer", {})

    with pytest.raises(KeyError, match="aimer_scores"):
        plugin.contribute_artifact(ctx)


def test_plugin_contribute_artifact_rejects_missing_bottom_pct_by_layer():
    plugin = AimerDetectorPlugin()
    ctx = PipelineContext()
    ctx.set("candidates", {})
    ctx.set("aimer_scores", {})

    with pytest.raises(KeyError, match="bottom_pct_by_layer"):
        plugin.contribute_artifact(ctx)


# ---------------------------------------------------------------------------
# ``_get_expert_down_proj_weight`` — four layout variants + raise
# ---------------------------------------------------------------------------


def test_get_expert_down_proj_weight_per_expert_modulelist():
    class _PerExpertModule(torch.nn.Module):
        def __init__(self, d_hid: int, d_int: int) -> None:
            super().__init__()
            self.down_proj = torch.nn.Linear(d_int, d_hid, bias=False)

    experts = torch.nn.ModuleList([_PerExpertModule(8, 16) for _ in range(3)])
    ref = _FakeRef(layer_idx=0, num_routed_experts=3, experts_module=experts)

    w = _get_expert_down_proj_weight(ref, 1)

    assert w.shape == (8, 16)
    assert w is experts[1].down_proj.weight


def test_get_expert_down_proj_weight_fused_parameter():
    n, d_hid, d_int = 4, 8, 16
    fused = torch.nn.Parameter(torch.randn(n, d_hid, d_int))
    experts = _FakeExperts(down_proj=fused)
    ref = _FakeRef(layer_idx=0, num_routed_experts=n, experts_module=experts)

    w = _get_expert_down_proj_weight(ref, 2)

    assert w.shape == (d_hid, d_int)
    assert torch.equal(w, fused[2])


def test_get_expert_down_proj_weight_fused_linear_module():
    n, d_hid, d_int = 4, 8, 16

    class _FusedLinear:
        def __init__(self) -> None:
            self.weight = torch.randn(n, d_hid, d_int)

    class _Experts:
        def __init__(self) -> None:
            self.down_proj = _FusedLinear()

    experts = _Experts()
    ref = _FakeRef(layer_idx=0, num_routed_experts=n, experts_module=experts)

    w = _get_expert_down_proj_weight(ref, 0)

    assert w.shape == (d_hid, d_int)
    assert torch.equal(w, experts.down_proj.weight[0])


def test_get_expert_down_proj_weight_factored_uv():
    n, d_hid, r, d_int = 4, 8, 4, 16

    class _Experts:
        def __init__(self) -> None:
            self.down_proj_U = torch.nn.Parameter(torch.randn(n, d_hid, r))
            self.down_proj_V = torch.nn.Parameter(torch.randn(n, r, d_int))

    experts = _Experts()
    ref = _FakeRef(layer_idx=0, num_routed_experts=n, experts_module=experts)

    w = _get_expert_down_proj_weight(ref, 3)

    assert w.shape == (d_hid, d_int)
    assert torch.allclose(w, experts.down_proj_U[3] @ experts.down_proj_V[3])


def test_get_expert_down_proj_weight_callable_accessor():
    d_hid, d_int = 8, 16
    expected = torch.zeros(d_hid, d_int)

    class _Experts:
        def down_proj_weight(self, idx: int) -> torch.Tensor:
            return expected

    experts = _Experts()
    ref = _FakeRef(layer_idx=0, num_routed_experts=4, experts_module=experts)

    w = _get_expert_down_proj_weight(ref, 2)

    assert w is expected


def test_get_expert_down_proj_weight_raises_when_unknown():
    ref = _FakeRef(layer_idx=7, num_routed_experts=4, experts_module=object())

    with pytest.raises(AttributeError, match="expert 0"):
        _get_expert_down_proj_weight(ref, 0)

    with pytest.raises(AttributeError, match="layer 7"):
        _get_expert_down_proj_weight(ref, 0)
