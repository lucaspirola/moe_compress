"""Unit tests for ``moe_compress.stage1.plugins.grape_merge``.

Verifies:

1. The plugin's class-level Protocol attributes match the registry
   contract (``PipelinePlugin``).
2. The plugin's ``run`` is byte-equivalent to calling the legacy
   ``_grape_greedy_merge`` helper directly on the same input — i.e.
   the migration is observation-preserving.
3. ``contribute_artifact`` returns exactly the historical five-key
   ``stage1_budgets.json`` schema with the correct inner types.
4. Missing required slots cause ``KeyError`` so the orchestrator
   (sub-task 10) gets a clear contract violation rather than a silent
   misbehaviour.
5. The plugin keeps a *copy* of the Stage 1 sub-config, not an alias.
"""

from __future__ import annotations

import pytest
import torch

from moe_compress.budget.solver import BudgetDecomposition
from moe_compress.pipeline.plugin import PipelinePlugin
from moe_compress.pipeline.context import PipelineContext
from moe_compress.stage1.plugins.grape_merge import (
    GrapeMergePlugin,
    _grape_greedy_merge,
)


# ---------------------------------------------------------------------------
# Shared fixtures (kept private — these tests deliberately do NOT use the
# project-wide ``conftest.py`` fixtures; Phase F only needs synthetic
# tensors so the test file stays fast and decoupled).
# ---------------------------------------------------------------------------


def _make_decomposition(global_budget: int = 6) -> BudgetDecomposition:
    """Construct a real ``BudgetDecomposition`` dataclass.

    The plugin only reads ``global_expert_budget``; the other fields
    are inert. We use the real dataclass (not a stub) so the test guards
    against drift if a future plugin reads more decomposition fields.
    """
    return BudgetDecomposition(
        total_reduction_ratio=0.20,
        expert_prune_ratio=0.25,
        svd_rank_ratio=0.0,
        global_expert_budget=global_budget,
        min_experts_per_layer=2,
    )


def _build_synthetic_inputs() -> dict:
    """Synthetic GRAPE-merge input with an SE blacklist.

    Two layers, n=4 experts each, distinct D-matrix patterns so the
    merge sequence is deterministic. ``global_budget=6`` with
    ``blacklist={0: [1, 2]}`` exercises the blacklist code paths.
    """
    n = 4
    D0 = torch.full((n, n), 0.3, dtype=torch.float32)
    D0.fill_diagonal_(0.0)
    D1 = torch.full((n, n), 0.6, dtype=torch.float32)
    D1.fill_diagonal_(0.0)
    return {
        "D_matrices": {0: D0, 1: D1},
        "blacklist": {0: [1, 2]},
        "per_layer_counts": {0: n, 1: n},
        "config": {"stage1_grape": {"entropy_tolerance": 1.0}},
        "moe_layers": [],  # GRAPE never consumes moe_layers; opaque list is fine
        "decomposition": _make_decomposition(global_budget=6),
    }


def _populate_context(inputs: dict) -> PipelineContext:
    ctx = PipelineContext()
    ctx.set("D_matrices", inputs["D_matrices"])
    ctx.set("blacklist", inputs["blacklist"])
    ctx.set("per_layer_targets", inputs["per_layer_counts"])
    ctx.set("moe_layers", inputs["moe_layers"])
    ctx.set("decomposition", inputs["decomposition"])
    ctx.set("config", inputs["config"])
    return ctx


# ---------------------------------------------------------------------------
# 1. Protocol-attribute contract
# ---------------------------------------------------------------------------


def test_plugin_protocol_attributes():
    """Class-level attributes match the plan exactly."""
    plugin = GrapeMergePlugin()
    assert plugin.name == "grape_merge"
    # `paper` must cite arXiv:2604.06542 (GRAPE, Zhang et al. — NOT Liu;
    # the prior assertion mis-attributed first authorship), the
    # "no official code" finding, and the five GRAPE-specific deviations.
    assert "arXiv:2604.06542" in plugin.paper
    assert "Zhang" in plugin.paper
    assert "Algorithm 1" in plugin.paper
    for deviation_token in (
        "D3",
        "D4",
        "D5",
        "D-grape-restart-merge",
        "D-se-blacklist-merge",
        "D-cka-distance",
    ):
        assert deviation_token in plugin.paper
    assert plugin.config_key == "stage1_grape"
    assert plugin.reads == (
        "D_matrices",
        "blacklist",
        "per_layer_targets",
        "decomposition",
        "config",
    )
    assert plugin.writes == (
        "per_layer_target_experts",
        "per_layer_redundancy",
        "achieved_budget",
        "requested_budget",
        "grape_config",
    )
    assert plugin.provides == ()


def test_plugin_is_runtime_checkable_pipelineplugin():
    """``isinstance`` against the runtime-checkable Protocol must succeed."""
    assert isinstance(GrapeMergePlugin(), PipelinePlugin)


def test_plugin_is_enabled_always_true():
    """GRAPE is mandatory — every config enables it."""
    plugin = GrapeMergePlugin()
    assert plugin.is_enabled({}) is True
    assert plugin.is_enabled({"stage1_grape": {}}) is True


# ---------------------------------------------------------------------------
# 2. Byte-equivalence anchor against the legacy helper
# ---------------------------------------------------------------------------


def test_plugin_run_matches_legacy_grape_greedy_merge():
    """End-to-end ``plugin.run`` writes the same per-layer budgets as the
    legacy helper consumed the same way the old inline block did."""
    inputs = _build_synthetic_inputs()
    ctx = _populate_context(inputs)

    GrapeMergePlugin().run(ctx)

    plugin_budgets_str = ctx.get("per_layer_target_experts")
    # Plugin stores budgets keyed by ``str(layer_idx)``. Convert for
    # comparison against the legacy helper, which keys by int.
    plugin_budgets = {int(k): v for k, v in plugin_budgets_str.items()}

    legacy_budgets = _grape_greedy_merge(
        D_matrices=inputs["D_matrices"],
        global_budget=inputs["decomposition"].global_expert_budget,
        per_layer_counts=inputs["per_layer_counts"],
        blacklist=inputs["blacklist"],
        gamma=1.0,
    )

    assert plugin_budgets == legacy_budgets

    # Sanity: total surviving experts == global_budget (effective_budget+SE).
    assert sum(plugin_budgets.values()) == 6
    assert ctx.get("achieved_budget") == 6
    assert ctx.get("requested_budget") == 6



# ---------------------------------------------------------------------------
# 3. Artifact contract (five keys, correct inner types)
# ---------------------------------------------------------------------------


def test_plugin_contribute_artifact_has_exactly_five_keys():
    """The artifact dict must match the historical schema exactly."""
    inputs = _build_synthetic_inputs()
    ctx = _populate_context(inputs)
    plugin = GrapeMergePlugin()
    plugin.run(ctx)

    payload = plugin.contribute_artifact(ctx)

    assert set(payload.keys()) == {
        "per_layer_target_experts",
        "per_layer_redundancy",
        "achieved_budget",
        "requested_budget",
        "config",
    }

    # per_layer_target_experts: dict[str, int]
    pte = payload["per_layer_target_experts"]
    assert isinstance(pte, dict)
    for k, v in pte.items():
        assert isinstance(k, str)
        assert isinstance(v, int)

    # per_layer_redundancy: dict[str, float]
    plr = payload["per_layer_redundancy"]
    assert isinstance(plr, dict)
    for k, v in plr.items():
        assert isinstance(k, str)
        assert isinstance(v, float)

    # achieved_budget / requested_budget are ints
    assert isinstance(payload["achieved_budget"], int)
    assert isinstance(payload["requested_budget"], int)

    # config is a dict equal to inputs["config"]["stage1_grape"]
    assert isinstance(payload["config"], dict)
    assert payload["config"] == inputs["config"]["stage1_grape"]


def test_plugin_writes_string_keys_in_per_layer_dicts():
    """Golden-byte compatibility: keys must be ``str(layer_idx)``, not ints."""
    inputs = _build_synthetic_inputs()
    ctx = _populate_context(inputs)
    GrapeMergePlugin().run(ctx)
    for k in ctx.get("per_layer_target_experts").keys():
        assert isinstance(k, str)
    for k in ctx.get("per_layer_redundancy").keys():
        assert isinstance(k, str)


def test_plugin_grape_config_is_copy_not_alias():
    """The stored ``grape_config`` slot must be a shallow copy of
    ``config["stage1_grape"]`` (legacy ``dict(s1)`` behaviour) — mutating
    the slot must NOT bleed back into the original config dict."""
    inputs = _build_synthetic_inputs()
    ctx = _populate_context(inputs)
    GrapeMergePlugin().run(ctx)

    stored = ctx.get("grape_config")
    stored["entropy_tolerance"] = 999.0
    assert inputs["config"]["stage1_grape"]["entropy_tolerance"] == 1.0


# ---------------------------------------------------------------------------
# 4. Missing-slot KeyError contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "missing_slot",
    [
        "D_matrices",
        "blacklist",
        "per_layer_targets",
        "decomposition",
        "config",
    ],
)
def test_plugin_run_rejects_missing_slot(missing_slot):
    """``plugin.run`` must raise ``KeyError`` mentioning the missing slot
    name when any of its required reads are absent from the context."""
    inputs = _build_synthetic_inputs()
    # Slot-name mapping: "per_layer_targets" is the context slot, but the
    # inputs dict uses "per_layer_counts" — both are populated below.
    ctx = PipelineContext()
    populators = {
        "D_matrices": ("D_matrices", inputs["D_matrices"]),
        "blacklist": ("blacklist", inputs["blacklist"]),
        "per_layer_targets": ("per_layer_targets", inputs["per_layer_counts"]),
        "moe_layers": ("moe_layers", inputs["moe_layers"]),
        "decomposition": ("decomposition", inputs["decomposition"]),
        "config": ("config", inputs["config"]),
    }
    for slot_name, (key, value) in populators.items():
        if slot_name == missing_slot:
            continue
        ctx.set(key, value)

    with pytest.raises(KeyError) as exc:
        GrapeMergePlugin().run(ctx)
    assert missing_slot in str(exc.value)


# ---------------------------------------------------------------------------
# 5. C8 persistent-mask GATE — byte-identical greedy-merge parity golden.
#
# This is a dedicated parity gate for the C8 speed-up that replaces the
# per-iteration ``tmp = D_l.copy()`` mask rebuild with a persistent per-layer
# ``tmp_l``.  The fixture is hand-tuned so that a SINGLE ``_grape_greedy_merge``
# call exercises all four high-risk branches of the greedy loop:
#
#   (a) blacklisted experts        — every layer has a non-empty blacklist
#                                     (layer 0 has TWO, driving floor->0).
#   (b) structural-block trigger   — layer 0 (2 of 4 experts blacklisted, so
#                                     only experts {0,3} are mergeable; after
#                                     one merge no finite pair remains while the
#                                     count is still above the floor) hits the
#                                     ``if not np.isfinite(...).any()`` branch.
#   (c) entropy freeze + restart   — gamma=0.02 makes E_hat just below E_init,
#                                     so the first entropy-reducing merge per
#                                     layer freezes it; once all non-permanently
#                                     -blocked layers are frozen the loop
#                                     restarts (both the top-of-loop and the
#                                     lag-corrected post-selection restart fire).
#   (d) zero-distance pair         — layer 0 has D[0,3]==D[3,0]==0.0, a genuine
#                                     identical-weight pair that argmin SELECTS.
#
# The expected budgets are blessed from the PRE-EDIT implementation; the C8
# edit must leave them byte-identical.  Do NOT regen on failure — a diff here
# means the persistent-mask refactor changed the merge sequence.
# ---------------------------------------------------------------------------


def _grape_gate_inputs() -> dict:
    """Hand-tuned GRAPE input that hits all four high-risk branches."""
    n = 4
    D0 = torch.full((n, n), 0.3, dtype=torch.float32)
    D0.fill_diagonal_(0.0)
    # Genuine zero-distance (identical) pair among the two non-blacklisted
    # experts {0, 3} of layer 0 — branch (d).
    D0[0, 3] = 0.0
    D0[3, 0] = 0.0
    # Layers 1 & 2: tightly clustered (0.1) so a single merge sharply reduces
    # entropy and trips the freeze gate — branch (c).
    D1 = torch.full((n, n), 0.1, dtype=torch.float32)
    D1.fill_diagonal_(0.0)
    D2 = torch.full((n, n), 0.1, dtype=torch.float32)
    D2.fill_diagonal_(0.0)
    return {
        "D_matrices": {0: D0, 1: D1, 2: D2},
        "per_layer_counts": {0: n, 1: n, 2: n},
        # Branch (a): every layer blacklisted; layer 0 has TWO (floor -> 0)
        # which is what lets it reach the structural-block branch (b).
        "blacklist": {0: [1, 2], 1: [3], 2: [0]},
        "global_budget": 3,
        "gamma": 0.02,
    }


# Golden blessed from the PRE-EDIT _grape_greedy_merge on the fixture above.
_GRAPE_GATE_GOLDEN = {0: 3, 1: 2, 2: 2}


def _trace_grape_branches(inputs: dict):
    """Run ``_grape_greedy_merge`` under a line tracer that records which of
    the four high-risk branches fired, matching by SOURCE-LINE CONTENT so the
    detection survives the C8 line-number shifts.  Returns ``(budgets,
    branches)`` where ``branches`` is a dict of bool flags.
    """
    import sys
    import inspect

    src_file = inspect.getsourcefile(_grape_greedy_merge)
    # Build absolute line-number -> stripped-text map for the module file so
    # the tracer can classify a hit by what the executed line *says* (robust
    # to the C8 line-number shifts).
    with open(src_file, "r") as fh:
        file_text = fh.readlines()
    line_text = {i + 1: ln.strip() for i, ln in enumerate(file_text)}

    flags = {
        "structural_block": False,
        "entropy_freeze": False,
        "restart": False,
        "zero_distance_selected": False,
    }

    # Detect each branch by the *content* of the executed line. The
    # zero-distance check reads the masked distance matrix (named ``tmp`` in
    # the pre-edit code and ``tmp_l`` after C8) at the argmin-selected index.
    def tracer(frame, event, arg):
        if event == "line" and frame.f_code.co_filename == src_file:
            txt = line_text.get(frame.f_lineno, "")
            if txt.startswith("structurally_blocked.add(best_layer)"):
                flags["structural_block"] = True
            elif txt.startswith("frozen.add(best_layer)"):
                flags["entropy_freeze"] = True
            elif txt.startswith("frozen.clear()"):
                flags["restart"] = True
            elif txt.startswith("_, j_star = divmod(flat_idx, n)"):
                masked = frame.f_locals.get("tmp")
                if masked is None:
                    masked = frame.f_locals.get("tmp_l")
                flat_idx = frame.f_locals.get("flat_idx")
                if masked is not None and flat_idx is not None:
                    if float(masked.flat[int(flat_idx)]) == 0.0:
                        flags["zero_distance_selected"] = True
        return tracer

    sys.settrace(tracer)
    try:
        budgets = _grape_greedy_merge(
            D_matrices=inputs["D_matrices"],
            global_budget=inputs["global_budget"],
            per_layer_counts=inputs["per_layer_counts"],
            blacklist=inputs["blacklist"],
            gamma=inputs["gamma"],
        )
    finally:
        sys.settrace(None)
    return budgets, flags


def test_grape_gate_all_four_branches_fire():
    """The gate fixture must exercise all four high-risk branches; if any
    stops firing the parity argument for C8 no longer covers it."""
    _, flags = _trace_grape_branches(_grape_gate_inputs())
    assert flags["structural_block"], "structural-block branch did not fire"
    assert flags["entropy_freeze"], "entropy-freeze branch did not fire"
    assert flags["restart"], "entropy-restart branch did not fire"
    assert flags["zero_distance_selected"], (
        "zero-distance pair was not selected by argmin"
    )


def test_grape_gate_budgets_byte_identical_to_golden():
    """C8 persistent-mask parity gate: budgets must stay byte-identical to the
    pre-edit golden.  A mismatch means the refactor changed the merge order."""
    inputs = _grape_gate_inputs()
    budgets = _grape_greedy_merge(
        D_matrices=inputs["D_matrices"],
        global_budget=inputs["global_budget"],
        per_layer_counts=inputs["per_layer_counts"],
        blacklist=inputs["blacklist"],
        gamma=inputs["gamma"],
    )
    assert budgets == _GRAPE_GATE_GOLDEN, (
        f"GRAPE budgets {budgets} != golden {_GRAPE_GATE_GOLDEN}; "
        "the C8 persistent-mask refactor changed the merge sequence."
    )
    assert sum(budgets.values()) == 7
