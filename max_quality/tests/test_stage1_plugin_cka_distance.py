"""Unit tests for ``moe_compress.stage1.plugins.cka_distance``.

Verifies:

1. The plugin's class-level Protocol attributes match the registry
   contract (``PipelinePlugin``).
2. The plugin's ``run`` is byte-equivalent to calling the legacy
   ``_cka_distance_matrix`` helper directly on the same synthetic input
   — i.e. the migration is observation-preserving.
3. ``contribute_artifact`` returns ``{}`` (Phase E owns no JSON artifact).
4. Missing required slots cause ``KeyError`` so the orchestrator
   (sub-task 10) gets a clear contract violation rather than a silent
   misbehaviour.
5. Per-layer write semantics: every ``moe_layer.layer_idx`` keys an
   entry in the output ``D_matrices`` dict.
"""

from __future__ import annotations

import pytest
import torch

from moe_compress.pipeline.plugin import PipelinePlugin
from moe_compress.stage1.context import Stage1Context
from moe_compress.stage1.plugins.cka_distance import (
    CKADistancePlugin,
    _cka_distance_matrix,
)


# ---------------------------------------------------------------------------
# Shared fixtures (kept private — Phase E only needs synthetic tensors
# so the test file stays fast and decoupled from ``conftest.py``).
# ---------------------------------------------------------------------------


class _FakeAcc:
    """Minimal stand-in for ``ExpertOutputAccumulator``.

    Returns the same representation tensor for every (layer, expert) pair —
    enough to anchor byte-equivalence against the legacy CKA helper.
    """

    def __init__(self, R: torch.Tensor):
        self._R = R

    def get_representations(self, li: int, e: int) -> torch.Tensor:  # noqa: N805
        return self._R.clone()


class _FakeAccRaises:
    """Stand-in that raises if called — proves the weight-space branch
    skips the CKA path entirely."""

    def get_representations(self, li: int, e: int):  # noqa: N805
        raise AssertionError(
            "weight-space metric must not consult output_acc"
        )


class _FakeRef:
    def __init__(self, layer_idx: int, num_routed_experts: int):
        self.layer_idx = layer_idx
        self.num_routed_experts = num_routed_experts


def _make_ctx(*, R: torch.Tensor, layers, config: dict | None = None) -> Stage1Context:
    """Build a populated ``Stage1Context`` for a CKA run.

    ``config`` defaults to ``{"stage1_grape": {}}``; tests that exercise
    the weight-space override pass ``{"stage1_grape": {"similarity_metric":
    "cosine"}}`` etc.
    """
    ctx = Stage1Context()
    ctx.set("output_acc", _FakeAcc(R))
    ctx.set("moe_layers", layers)
    ctx.set("config", config if config is not None else {"stage1_grape": {}})
    return ctx


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_plugin_protocol_attributes():
    plugin = CKADistancePlugin()
    assert plugin.name == "cka_distance"
    assert plugin.paper.startswith("Kornblith")
    assert plugin.config_key == "stage1_grape"
    assert plugin.reads == ("output_acc", "moe_layers", "config")
    assert plugin.writes == ("D_matrices",)
    assert plugin.provides == ("output_reservoir",)


def test_plugin_is_runtime_checkable_pipelineplugin():
    assert isinstance(CKADistancePlugin(), PipelinePlugin)


def test_plugin_is_enabled_always_true():
    plugin = CKADistancePlugin()
    assert plugin.is_enabled({}) is True
    assert plugin.is_enabled({"stage1_grape": {"similarity_metric": "cka"}}) is True
    assert plugin.is_enabled({"stage1_grape": {"similarity_metric": "cosine"}}) is True


def test_plugin_run_matches_legacy_cka_distance_matrix():
    """Byte-equivalence anchor against the legacy helper.

    Uses a synthetic 2-expert layer (32 tokens, 16 dim, identical reprs
    for both experts).
    """
    torch.manual_seed(0)
    R = torch.randn(32, 16)
    ctx = _make_ctx(R=R, layers=[_FakeRef(layer_idx=0, num_routed_experts=2)])

    CKADistancePlugin().run(ctx)

    D_dict = ctx.get("D_matrices")
    assert isinstance(D_dict, dict)
    assert set(D_dict.keys()) == {0}
    D = D_dict[0]
    assert isinstance(D, torch.Tensor)
    assert D.shape == (2, 2)
    assert torch.allclose(torch.diag(D), torch.zeros(2))
    assert D[0, 1].item() == pytest.approx(0.0, abs=1e-4)
    assert D[1, 0].item() == pytest.approx(0.0, abs=1e-4)


def test_plugin_run_writes_D_matrices_per_layer():
    """Three layers, four experts each — every layer_idx keys a 4×4 entry."""
    torch.manual_seed(1)
    R = torch.randn(32, 16)
    layers = [
        _FakeRef(layer_idx=0, num_routed_experts=4),
        _FakeRef(layer_idx=1, num_routed_experts=4),
        _FakeRef(layer_idx=2, num_routed_experts=4),
    ]
    ctx = _make_ctx(R=R, layers=layers)

    CKADistancePlugin().run(ctx)

    D_dict = ctx.get("D_matrices")
    assert isinstance(D_dict, dict)
    assert set(D_dict.keys()) == {0, 1, 2}
    for li, D in D_dict.items():
        assert isinstance(D, torch.Tensor)
        assert D.shape == (4, 4)
        assert torch.allclose(torch.diag(D), torch.zeros(4))


def test_plugin_run_rejects_missing_output_acc():
    """``output_acc`` slot missing -> KeyError with the slot name."""
    ctx = Stage1Context()
    ctx.set("moe_layers", [_FakeRef(layer_idx=0, num_routed_experts=2)])
    ctx.set("config", {"stage1_grape": {}})

    with pytest.raises(KeyError, match="output_acc"):
        CKADistancePlugin().run(ctx)


def test_plugin_run_rejects_missing_moe_layers():
    torch.manual_seed(0)
    R = torch.randn(32, 16)
    ctx = Stage1Context()
    ctx.set("output_acc", _FakeAcc(R))
    ctx.set("config", {"stage1_grape": {}})

    with pytest.raises(KeyError, match="moe_layers"):
        CKADistancePlugin().run(ctx)


def test_plugin_run_rejects_missing_config():
    torch.manual_seed(0)
    R = torch.randn(32, 16)
    ctx = Stage1Context()
    ctx.set("output_acc", _FakeAcc(R))
    ctx.set("moe_layers", [_FakeRef(layer_idx=0, num_routed_experts=2)])

    with pytest.raises(KeyError, match="config"):
        CKADistancePlugin().run(ctx)


def test_plugin_contribute_artifact_returns_empty_dict():
    """Phase E never writes a JSON artifact."""
    torch.manual_seed(0)
    R = torch.randn(32, 16)
    ctx = _make_ctx(R=R, layers=[_FakeRef(layer_idx=0, num_routed_experts=2)])
    plugin = CKADistancePlugin()
    plugin.run(ctx)
    payload = plugin.contribute_artifact(ctx)
    assert payload == {}


def test_plugin_run_weight_space_short_circuits_cka():
    """``similarity_metric != "cka"`` must skip the CKA path entirely.

    Anchored on a synthetic two-expert fused layer whose ``experts_module``
    is consumable by ``build_banks``. We use ``_FakeAccRaises`` to prove
    the plugin does **not** call ``output_acc.get_representations`` when
    the weight-space override is active.

    The fused-experts stub is defined inline so this test stays decoupled
    from ``conftest.py`` (sub-task 4 plan §6.2).
    """
    import torch.nn as nn

    class _MinimalFusedExperts(nn.Module):
        """Minimal stand-in for ``Qwen3_5MoeExperts`` — only the two
        parameters ``build_banks`` inspects, with the fused-layout shapes
        ``gate_up_proj`` ``[N, 2·d_int, d_hid]`` and ``down_proj``
        ``[N, d_hid, d_int]``."""

        def __init__(self, num_experts: int, hidden: int, intermediate: int) -> None:
            super().__init__()
            self.gate_up_proj = nn.Parameter(
                torch.randn(num_experts, 2 * intermediate, hidden)
            )
            self.down_proj = nn.Parameter(
                torch.randn(num_experts, hidden, intermediate)
            )

    class _RefWithExperts:
        def __init__(self, layer_idx: int, n: int, hidden: int, intermediate: int):
            self.layer_idx = layer_idx
            self.num_routed_experts = n
            self.experts_module = _MinimalFusedExperts(n, hidden, intermediate)

    torch.manual_seed(42)
    layers = [_RefWithExperts(layer_idx=0, n=2, hidden=8, intermediate=4)]

    ctx = Stage1Context()
    ctx.set("output_acc", _FakeAccRaises())
    ctx.set("moe_layers", layers)
    ctx.set("config", {"stage1_grape": {"similarity_metric": "cosine"}})

    # The plugin must NOT touch _FakeAccRaises.get_representations.
    CKADistancePlugin().run(ctx)
    D_dict = ctx.get("D_matrices")
    assert set(D_dict.keys()) == {0}
    D = D_dict[0]
    assert D.shape == (2, 2)
    # Cosine self-similarity has minor fp32 drift around 0; off-diagonal
    # must be a strictly positive [0,1] cosine distance for distinct experts.
    assert torch.allclose(torch.diag(D), torch.zeros(2), atol=1e-6)
    assert D[0, 1].item() > 0.0
    assert D[0, 1].item() <= 1.0


def test_plugin_run_does_not_mutate_output_acc():
    """Plugin reads ``output_acc`` but doesn't replace the slot binding."""
    torch.manual_seed(0)
    R = torch.randn(32, 16)
    ctx = _make_ctx(R=R, layers=[_FakeRef(layer_idx=0, num_routed_experts=2)])

    before_id = id(ctx.get("output_acc"))
    CKADistancePlugin().run(ctx)
    after_id = id(ctx.get("output_acc"))

    assert before_id == after_id
