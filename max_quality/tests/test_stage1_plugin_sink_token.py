"""Unit tests for ``moe_compress.stage1.plugins.sink_token`` (sub-task 7).

Verifies:

1. The plugin's class-level Protocol attributes match the registry
   contract (``PipelinePlugin``).
2. ``is_enabled`` reads
   ``config["stage1_grape"]["super_expert_detection"]["sink_token_enabled"]``
   (default True).
3. ``setup`` constructs ``SinkTokenRoutingAccumulator`` (or ``None`` when
   disabled) and writes it to ``ctx.sink_acc``.
4. ``run`` is a documented no-op in sub-task 7 — reads ``sink_acc`` for
   the side-effect of raising ``KeyError`` if ``setup`` was skipped, logs
   an entry-line, returns.
5. ``contribute_artifact`` returns the canonical four-key payload
   byte-equivalent to the legacy inline ``sink_payload`` literal
   (pre-sub-task-7).
6. Missing required slots cause ``KeyError`` so the orchestrator
   (sub-task 10) gets a clear contract violation rather than a silent
   misbehaviour.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pytest

from moe_compress.pipeline.candidates import CandidateBag
from moe_compress.pipeline.plugin import PipelinePlugin
from moe_compress.pipeline.context import PipelineContext
from moe_compress.stage1.plugins.sink_token import SinkTokenDetectorPlugin
from moe_compress.utils.sink_token_routing import SinkTokenRoutingAccumulator


# ---------------------------------------------------------------------------
# Test fakes — the plugin reads ``len(moe_layers)`` and
# ``getattr(tokenizer, "bos_token_id", None)``; we don't need a real model.
# ---------------------------------------------------------------------------


@dataclass
class _FakeRef:
    """The plugin only reads ``len(moe_layers)``; contents don't matter."""

    layer_idx: int = 0


@dataclass
class _FakeTokenizer:
    bos_token_id: int | None = None


def _default_config(sink_token_enabled: bool = True) -> dict:
    return {
        "stage1_grape": {
            "super_expert_detection": {
                "sink_token_enabled": sink_token_enabled,
                "sink_token_score_ratio": 10.0,
                "sink_token_freq_threshold": 0.99,
                "sink_token_max_per_layer_cap": 10,
            }
        }
    }


def _populated_ctx(
    moe_layers=None,
    tokenizer=None,
    config=None,
    n_per_layer: int = 128,
) -> PipelineContext:
    ctx = PipelineContext()
    ctx.set(
        "moe_layers",
        moe_layers if moe_layers is not None else [_FakeRef(0), _FakeRef(1)],
    )
    ctx.set(
        "tokenizer",
        tokenizer if tokenizer is not None else _FakeTokenizer(bos_token_id=2),
    )
    ctx.set("config", config if config is not None else _default_config())
    ctx.set("n_per_layer", n_per_layer)
    return ctx


# ---------------------------------------------------------------------------
# Protocol-attribute contract
# ---------------------------------------------------------------------------


def test_plugin_protocol_attributes():
    p = SinkTokenDetectorPlugin()
    assert p.name == "sink_token"
    # `paper` must cite arXiv:2507.23279 (sink-token Figures 6 / 20 / 21
    # are the structural-signature observation) AND the golden official-code
    # commit (ZunhaiSu/Super-Experts-Profilling @
    # 573aead3127ae593ba267758b832944f8fed1485) — the detection criterion
    # itself is project-original (deviation D-sink-token-routing).
    assert "arXiv:2507.23279" in p.paper
    assert "573aead3127ae593ba267758b832944f8fed1485" in p.paper
    assert "D-sink-token-routing" in p.paper
    assert p.config_key == "stage1_grape.super_expert_detection.sink_token_enabled"
    assert p.reads == (
        "moe_layers",
        "tokenizer",
        "config",
        "n_per_layer",
        "sink_acc",
        "candidate_bag",
    )
    assert p.writes == ("sink_acc", "candidate_bag")
    assert p.provides == ("sink_routing",)


def test_plugin_is_runtime_checkable_pipelineplugin():
    assert isinstance(SinkTokenDetectorPlugin(), PipelinePlugin)


# ---------------------------------------------------------------------------
# ``is_enabled`` config branching
# ---------------------------------------------------------------------------


def test_plugin_is_enabled_default_true():
    assert SinkTokenDetectorPlugin().is_enabled({}) is True


def test_plugin_is_enabled_explicit_false():
    cfg = {"stage1_grape": {"super_expert_detection": {"sink_token_enabled": False}}}
    assert SinkTokenDetectorPlugin().is_enabled(cfg) is False


def test_plugin_is_enabled_explicit_true():
    cfg = {"stage1_grape": {"super_expert_detection": {"sink_token_enabled": True}}}
    assert SinkTokenDetectorPlugin().is_enabled(cfg) is True


# ---------------------------------------------------------------------------
# ``setup`` — constructs ``SinkTokenRoutingAccumulator`` (or ``None``)
# ---------------------------------------------------------------------------


def test_plugin_setup_builds_accumulator_when_enabled():
    plugin = SinkTokenDetectorPlugin()
    ctx = _populated_ctx(
        moe_layers=[_FakeRef(0), _FakeRef(1)],
        tokenizer=_FakeTokenizer(bos_token_id=2),
        n_per_layer=128,
    )

    plugin.setup(ctx)

    sink_acc = ctx.get("sink_acc")
    assert isinstance(sink_acc, SinkTokenRoutingAccumulator)
    assert sink_acc.num_layers == 2
    assert sink_acc.num_experts == 128
    assert sink_acc.bos_token_id == 2


def test_plugin_setup_writes_none_when_disabled():
    plugin = SinkTokenDetectorPlugin()
    ctx = _populated_ctx(config=_default_config(sink_token_enabled=False))

    plugin.setup(ctx)

    # Slot is explicitly written — bare ctx.get would raise if it wasn't.
    assert ctx.get("sink_acc") is None


def test_plugin_setup_handles_tokenizer_without_bos_token_id():
    plugin = SinkTokenDetectorPlugin()
    ctx = _populated_ctx(tokenizer=object())  # bare object — no bos_token_id attr

    plugin.setup(ctx)

    sink_acc = ctx.get("sink_acc")
    assert isinstance(sink_acc, SinkTokenRoutingAccumulator)
    assert sink_acc.bos_token_id is None


def test_plugin_setup_handles_tokenizer_with_none_bos_token_id():
    plugin = SinkTokenDetectorPlugin()
    ctx = _populated_ctx(tokenizer=_FakeTokenizer(bos_token_id=None))

    plugin.setup(ctx)

    sink_acc = ctx.get("sink_acc")
    assert isinstance(sink_acc, SinkTokenRoutingAccumulator)
    assert sink_acc.bos_token_id is None


def test_plugin_setup_handles_empty_moe_layers():
    plugin = SinkTokenDetectorPlugin()
    ctx = _populated_ctx(moe_layers=[], n_per_layer=0)

    plugin.setup(ctx)

    sink_acc = ctx.get("sink_acc")
    assert isinstance(sink_acc, SinkTokenRoutingAccumulator)
    assert sink_acc.num_layers == 0
    assert sink_acc.num_experts == 0


# ---------------------------------------------------------------------------
# ``run`` — sub-task-8 candidate-add step (shared CandidateBag, "sink_token" tag)
# ---------------------------------------------------------------------------


def _run_ctx_with_sink_acc(sink_acc, config=None) -> PipelineContext:
    """Build a run-ready ctx: ``sink_acc`` + ``candidate_bag`` + ``config``."""
    ctx = PipelineContext()
    ctx.set("sink_acc", sink_acc)
    ctx.set("candidate_bag", CandidateBag())
    ctx.set("config", config if config is not None else _default_config())
    return ctx


def test_plugin_run_adds_candidates_with_sink_token_tag():
    """``run`` calls ``apply_sink_token_candidate_selection`` and adds each
    returned (l, e) to the shared ``CandidateBag`` with tag ``"sink_token"``.

    The mocked ``sink_acc`` has expert 3 in layer 0 sink-dominated (high
    freq + high score ratio); experts 5 and 7 are normal-dominated.
    """
    from types import SimpleNamespace

    sink_acc = SimpleNamespace(
        mean_router_score_sink={(0, 3): 1.0, (0, 5): 0.05, (0, 7): 0.05},
        mean_router_score_normal={(0, 3): 0.05, (0, 5): 1.0, (0, 7): 1.0},
        freq_on_sink={(0, 3): 1.0, (0, 5): 0.0, (0, 7): 0.0},
    )
    ctx = _run_ctx_with_sink_acc(sink_acc)

    SinkTokenDetectorPlugin().run(ctx)

    bag: CandidateBag = ctx.get("candidate_bag")
    assert bag.by_tag("sink_token") == {0: [3]}


def test_plugin_run_noop_when_disabled():
    """When ``sink_token_enabled=False`` the plugin's ``setup`` writes
    ``sink_acc=None``; the rewritten ``run`` short-circuits on the
    ``sink_acc is None`` check and adds zero candidates."""
    plugin = SinkTokenDetectorPlugin()
    ctx = _populated_ctx(config=_default_config(sink_token_enabled=False))
    plugin.setup(ctx)
    # Sanity: setup wrote None into the slot.
    assert ctx.get("sink_acc") is None
    ctx.set("candidate_bag", CandidateBag())

    # Must not raise — ``sink_acc=None`` is the documented disabled state.
    plugin.run(ctx)

    # Slot remains None and no candidates were added.
    assert ctx.get("sink_acc") is None
    assert len(ctx.get("candidate_bag")) == 0


def test_plugin_run_rejects_missing_sink_acc():
    plugin = SinkTokenDetectorPlugin()
    ctx = PipelineContext()  # never called setup
    ctx.set("config", _default_config())

    with pytest.raises(KeyError, match="sink_acc"):
        plugin.run(ctx)


def test_plugin_run_rejects_missing_config():
    plugin = SinkTokenDetectorPlugin()
    ctx = PipelineContext()
    ctx.set("sink_acc", None)
    ctx.set("candidate_bag", CandidateBag())

    with pytest.raises(KeyError, match="config"):
        plugin.run(ctx)


def test_plugin_run_rejects_missing_candidate_bag():
    """The ``candidate_bag`` read fires only when ``run`` does NOT
    short-circuit — i.e. ``sink_acc`` is a real (non-None) accumulator."""
    from types import SimpleNamespace

    plugin = SinkTokenDetectorPlugin()
    sink_acc = SimpleNamespace(
        mean_router_score_sink={(0, 3): 1.0},
        mean_router_score_normal={(0, 3): 0.05},
        freq_on_sink={(0, 3): 1.0},
    )
    ctx = PipelineContext()
    ctx.set("sink_acc", sink_acc)
    ctx.set("config", _default_config())
    # candidate_bag deliberately omitted.

    with pytest.raises(KeyError, match="candidate_bag"):
        plugin.run(ctx)


# ---------------------------------------------------------------------------
# ``contribute_artifact`` — four-key block, byte-equivalent
# ---------------------------------------------------------------------------


def _ctx_with_artifact_inputs(
    sink_acc=None,
    candidates=None,
) -> PipelineContext:
    """Build a ctx pre-populated with finalized sink_acc + candidates.

    The plugin doesn't run Phase B; we set the finalized per-(l, e) dicts
    on the accumulator directly so contribute_artifact can read them.
    """
    ctx = PipelineContext()
    if sink_acc is None:
        sink_acc = SinkTokenRoutingAccumulator(
            num_layers=1, num_experts=4, bos_token_id=None,
        )
        sink_acc.mean_router_score_sink = {(0, 1): 0.9}
        sink_acc.mean_router_score_normal = {(0, 1): 0.05}
        sink_acc.freq_on_sink = {(0, 1): 1.0}
    ctx.set("sink_acc", sink_acc)
    ctx.set(
        "candidates",
        candidates
        if candidates is not None
        else {(0, 1): ["sink_token"], (0, 2): ["aimer"]},
    )
    return ctx


def test_plugin_contribute_artifact_four_keys():
    plugin = SinkTokenDetectorPlugin()
    ctx = _ctx_with_artifact_inputs()

    payload = plugin.contribute_artifact(ctx)

    assert set(payload.keys()) == {
        "mean_router_score_sink",
        "mean_router_score_normal",
        "freq_on_sink",
        "candidates",
    }
    for key in payload:
        assert isinstance(payload[key], dict), f"payload[{key!r}] is not a dict"


def test_plugin_contribute_artifact_score_keying():
    plugin = SinkTokenDetectorPlugin()
    ctx = _ctx_with_artifact_inputs()

    payload = plugin.contribute_artifact(ctx)

    for dict_key in ("mean_router_score_sink", "mean_router_score_normal", "freq_on_sink"):
        for key, value in payload[dict_key].items():
            assert re.match(r"^L\d+E\d+$", key), (
                f"unexpected key {key!r} in {dict_key!r}"
            )
            assert value is None or isinstance(value, float)


def test_plugin_contribute_artifact_scrubs_nan_inf():
    plugin = SinkTokenDetectorPlugin()
    sink_acc = SinkTokenRoutingAccumulator(
        num_layers=1, num_experts=4, bos_token_id=None,
    )
    sink_acc.mean_router_score_sink = {
        (0, 0): float("nan"),
        (0, 1): float("inf"),
        (0, 2): float("-inf"),
        (0, 3): 0.5,
    }
    sink_acc.mean_router_score_normal = {}
    sink_acc.freq_on_sink = {}
    ctx = _ctx_with_artifact_inputs(sink_acc=sink_acc, candidates={})

    payload = plugin.contribute_artifact(ctx)

    assert payload["mean_router_score_sink"]["L0E0"] is None
    assert payload["mean_router_score_sink"]["L0E1"] is None
    assert payload["mean_router_score_sink"]["L0E2"] is None
    assert payload["mean_router_score_sink"]["L0E3"] == 0.5


def test_plugin_contribute_artifact_candidates_inversion():
    plugin = SinkTokenDetectorPlugin()
    candidates = {
        (0, 1): ["sink_token"],
        (0, 2): ["aimer", "sink_token"],
        (0, 3): ["phase_c"],  # no sink_token tag → excluded
        (1, 0): ["sink_token"],
    }
    ctx = _ctx_with_artifact_inputs(candidates=candidates)

    payload = plugin.contribute_artifact(ctx)

    assert payload["candidates"] == {"0": [1, 2], "1": [0]}


def test_plugin_contribute_artifact_disabled_returns_empty_score_dicts():
    """When ``sink_token_enabled=False`` and the plugin's ``setup`` wrote
    ``sink_acc=None``, the three score dicts collapse to {} (matching the
    legacy inline ``if sink_acc is not None else {}`` behaviour). The
    ``candidates`` key is still computed normally from the ctx slot."""
    plugin = SinkTokenDetectorPlugin()
    ctx = PipelineContext()
    ctx.set("sink_acc", None)
    ctx.set("candidates", {(0, 1): ["sink_token"]})

    payload = plugin.contribute_artifact(ctx)

    assert payload["mean_router_score_sink"] == {}
    assert payload["mean_router_score_normal"] == {}
    assert payload["freq_on_sink"] == {}
    assert payload["candidates"] == {"0": [1]}


def test_plugin_contribute_artifact_byte_equivalent_to_legacy_inline():
    """Byte-anchor — mirror the inline ``sink_payload`` literal exactly.

    Legacy logic (pre-sub-task-7):

        sink_payload = {
            "mean_router_score_sink": (
                {f"L{li}E{e}": safe_float(v)
                 for (li, e), v in sink_acc.mean_router_score_sink.items()}
                if sink_acc is not None else {}
            ),
            "mean_router_score_normal": (
                {f"L{li}E{e}": safe_float(v)
                 for (li, e), v in sink_acc.mean_router_score_normal.items()}
                if sink_acc is not None else {}
            ),
            "freq_on_sink": (
                {f"L{li}E{e}": safe_float(v)
                 for (li, e), v in sink_acc.freq_on_sink.items()}
                if sink_acc is not None else {}
            ),
            "candidates": _candidates_by_provenance("sink_token"),
        }
    """
    from moe_compress.pipeline.safe_json import safe_float

    sink_acc = SinkTokenRoutingAccumulator(
        num_layers=2, num_experts=4, bos_token_id=None,
    )
    sink_acc.mean_router_score_sink = {
        (0, 0): 0.9,
        (0, 1): float("nan"),
        (1, 2): 0.99,
    }
    sink_acc.mean_router_score_normal = {
        (0, 0): 0.01,
        (0, 1): 0.5,
        (1, 2): 0.02,
    }
    sink_acc.freq_on_sink = {
        (0, 0): 1.0,
        (0, 1): 0.5,
        (1, 2): 0.99,
    }
    candidates = {
        (0, 0): ["sink_token", "phase_c"],
        (0, 1): ["sink_token"],
        (1, 2): ["sink_token"],
    }

    # Independent re-build of the legacy literal.
    def _candidates_by_provenance(tag: str) -> dict[str, list[int]]:
        out: dict[int, list[int]] = {}
        for (li, e), tags in candidates.items():
            if tag in tags:
                out.setdefault(int(li), []).append(int(e))
        return {str(li): sorted(es) for li, es in out.items()}

    expected = {
        "mean_router_score_sink": {
            f"L{li}E{e}": safe_float(v)
            for (li, e), v in sink_acc.mean_router_score_sink.items()
        },
        "mean_router_score_normal": {
            f"L{li}E{e}": safe_float(v)
            for (li, e), v in sink_acc.mean_router_score_normal.items()
        },
        "freq_on_sink": {
            f"L{li}E{e}": safe_float(v)
            for (li, e), v in sink_acc.freq_on_sink.items()
        },
        "candidates": _candidates_by_provenance("sink_token"),
    }

    ctx = _ctx_with_artifact_inputs(sink_acc=sink_acc, candidates=candidates)
    payload = SinkTokenDetectorPlugin().contribute_artifact(ctx)

    assert payload == expected


# ---------------------------------------------------------------------------
# Missing-slot errors — KeyError per slot
# ---------------------------------------------------------------------------


def test_plugin_setup_rejects_missing_moe_layers():
    plugin = SinkTokenDetectorPlugin()
    ctx = PipelineContext()
    ctx.set("tokenizer", _FakeTokenizer())
    ctx.set("config", _default_config())
    ctx.set("n_per_layer", 4)

    with pytest.raises(KeyError, match="moe_layers"):
        plugin.setup(ctx)


def test_plugin_setup_rejects_missing_tokenizer():
    plugin = SinkTokenDetectorPlugin()
    ctx = PipelineContext()
    ctx.set("moe_layers", [])
    ctx.set("config", _default_config())
    ctx.set("n_per_layer", 4)

    with pytest.raises(KeyError, match="tokenizer"):
        plugin.setup(ctx)


def test_plugin_setup_rejects_missing_config():
    plugin = SinkTokenDetectorPlugin()
    ctx = PipelineContext()
    ctx.set("moe_layers", [])
    ctx.set("tokenizer", _FakeTokenizer())
    ctx.set("n_per_layer", 4)

    with pytest.raises(KeyError, match="config"):
        plugin.setup(ctx)


def test_plugin_setup_rejects_missing_n_per_layer():
    plugin = SinkTokenDetectorPlugin()
    ctx = PipelineContext()
    ctx.set("moe_layers", [])
    ctx.set("tokenizer", _FakeTokenizer())
    ctx.set("config", _default_config())

    with pytest.raises(KeyError, match="n_per_layer"):
        plugin.setup(ctx)


def test_plugin_contribute_artifact_rejects_missing_sink_acc():
    plugin = SinkTokenDetectorPlugin()
    ctx = PipelineContext()
    ctx.set("candidates", {})

    with pytest.raises(KeyError, match="sink_acc"):
        plugin.contribute_artifact(ctx)


def test_plugin_contribute_artifact_rejects_missing_candidates():
    plugin = SinkTokenDetectorPlugin()
    ctx = PipelineContext()
    ctx.set("sink_acc", None)

    with pytest.raises(KeyError, match="candidates"):
        plugin.contribute_artifact(ctx)
