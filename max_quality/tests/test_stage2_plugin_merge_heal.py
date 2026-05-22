"""Per-layer merge-heal plugin module.

Structural contract: the MergeHealPlugin contract, the boolean
(merge_heal_enabled) is_enabled gate, a monkeypatch-drift guard
(T9-T16 lesson), the LIVE S2-11 hook coverage (the plugin exposes
``pre_merge_snapshot`` / ``post_merge`` and NOT ``write_artifacts``), and an
ON-path equivalence test that drives the plugin's hooks with merge-heal
ENABLED and asserts byte-equality vs. a direct ``_capture_mlp_io`` +
``_heal_layer`` reference. Deep algorithm coverage stays in
test_stage2_merge_heal.py — this file does NOT re-test the heal internals.
"""
from __future__ import annotations

import pathlib

import torch

from moe_compress.pipeline.context import PipelineContext
from moe_compress.pipeline.plugin import PipelinePlugin
from moe_compress.stage2.merging import _resize_router_for_kept_experts
from moe_compress.stage2.plugins.merge_heal import (
    MergeHealPlugin,
    _HealConfig,
    _capture_mlp_io,
    _heal_layer,
    _make_shared_out_fn,
)
from moe_compress.utils.activation_shards import ShardWriter
from moe_compress.utils.model_io import build_banks, iter_moe_layers

_CPU = torch.device("cpu")

_HEAL_NAMES = (
    "_HealConfig",
    "_make_shared_out_fn",
    "_capture_mlp_io",
    "_heal_student_moe_output",
    "_heal_lr_at_step",
    "_heal_layer",
    "_summarize_distill_state",
)


def _make_plugin(*, heal_cfg, artifacts_dir, model, batches):
    """Construct a MergeHealPlugin with the S2-11 constructor knob set."""
    return MergeHealPlugin(
        heal_cfg=heal_cfg,
        heal_device=_CPU,
        xd_batches=None,
        batches=batches,
        model=model,
        artifacts_dir=artifacts_dir,
        device=_CPU,
    )


def _disabled_plugin():
    """A plugin whose heal_cfg is OFF — for the structural contract tests."""
    return _make_plugin(
        heal_cfg=_HealConfig({}),  # merge_heal_enabled defaults False
        artifacts_dir=pathlib.Path("."),
        model=object(),
        batches=[],
    )


# --- plugin contract ------------------------------------------------------
def test_plugin_conforms_to_pipeline_plugin():
    assert isinstance(_disabled_plugin(), PipelinePlugin)


def test_plugin_name():
    assert MergeHealPlugin.name == "merge_heal"


# --- is_enabled boolean gate ---------------------------------------------
def test_is_enabled_true_when_flag_true():
    assert _disabled_plugin().is_enabled(
        {"stage2_reap_ream": {"merge_heal_enabled": True}}
    ) is True


def test_is_enabled_false_when_flag_false():
    assert _disabled_plugin().is_enabled(
        {"stage2_reap_ream": {"merge_heal_enabled": False}}
    ) is False


def test_is_enabled_false_when_key_missing():
    assert _disabled_plugin().is_enabled({"stage2_reap_ream": {}}) is False


def test_is_enabled_false_when_block_missing():
    assert _disabled_plugin().is_enabled({}) is False


# --- LIVE S2-11 hooks: structural ----------------------------------------
def test_exposes_live_phase_hooks():
    """S2-11: the plugin owns the ``pre_merge_snapshot`` + ``post_merge``
    phases. It must NOT declare ``write_artifacts`` — the
    ``_summarize_distill_state`` telemetry stays in LayerMergePlugin.
    """
    plugin = _disabled_plugin()
    assert callable(getattr(plugin, "pre_merge_snapshot", None))
    assert callable(getattr(plugin, "post_merge", None))
    assert not hasattr(plugin, "write_artifacts"), (
        "MergeHealPlugin must not declare write_artifacts — telemetry stays "
        "in LayerMergePlugin.write_artifacts"
    )
    # merge is NOT a hook of this plugin (expert-distill owns the merge phase).
    assert not hasattr(plugin, "merge")


def test_pre_merge_snapshot_disabled_writes_none(tmp_path):
    """With merge-heal OFF the snapshot hook writes nemo/xd writers as None —
    the slots ARE published so downstream reads resolve."""
    ctx = PipelineContext()

    class _FakeLayer:
        layer_idx = 0

    ctx.set("layer_ref", _FakeLayer())
    ctx.set("grouped", {0: [0, 1]})  # would-merge layer, but heal OFF
    _make_plugin(
        heal_cfg=_HealConfig({}), artifacts_dir=tmp_path,
        model=object(), batches=[],
    ).pre_merge_snapshot(ctx)
    assert ctx.get("nemo_writer") is None
    assert ctx.get("xd_writer") is None


def test_post_merge_disabled_overwrites_heal_state_none(tmp_path):
    """With merge-heal OFF the post_merge hook overwrites ``heal_state`` (which
    LayerMergePlugin.post_merge defaults to None) — still None, no crash."""
    ctx = PipelineContext()

    class _FakeLayer:
        layer_idx = 0

    ctx.set("layer_ref", _FakeLayer())
    ctx.set("final_kept_ids", (0, 1))
    ctx.set("nemo_writer", None)
    ctx.set("xd_writer", None)
    ctx.set("heal_state", None)  # LayerMergePlugin.post_merge default
    _make_plugin(
        heal_cfg=_HealConfig({}), artifacts_dir=tmp_path,
        model=object(), batches=[],
    ).post_merge(ctx)
    assert ctx.get("heal_state") is None


# --- LIVE S2-11 hooks: ON-path equivalence (merge-heal ENABLED) ----------
def _tiny_moe_model():
    """A tiny randomly-initialized Qwen3-MoE causal LM (CPU, fp32, inference).

    Mirrors the fixture in test_stage2_merge_heal.py so the ON-path test runs
    on the same shape of model the deep heal tests exercise.
    """
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    cfg = Qwen3MoeConfig(
        vocab_size=128, hidden_size=32, intermediate_size=48,
        moe_intermediate_size=16, num_hidden_layers=3, num_attention_heads=4,
        num_key_value_heads=2, num_experts=8, num_experts_per_tok=2,
        norm_topk_prob=True, decoder_sparse_step=1, max_position_embeddings=64,
        head_dim=8,
    )
    torch.manual_seed(0)
    model = Qwen3MoeForCausalLM(cfg)
    model.train(False)
    return model


def _id_batches(model, n_seq=64, seq_len=8, chunk=4, seed=0):
    """Deterministic synthetic token batches (seeded generator)."""
    gen = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, model.config.vocab_size, (n_seq, seq_len),
                        generator=gen)
    return [ids[i:i + chunk] for i in range(0, n_seq, chunk)]


def _heal_cfg(**overrides) -> _HealConfig:
    base = {
        "merge_heal_enabled": True, "merge_heal_lr": 1.0e-3,
        "merge_heal_max_steps": 300, "merge_heal_eval_interval": 20,
        "merge_heal_patience": 20, "merge_heal_minibatch_size": 16,
        "merge_heal_shard_rows": 32, "merge_heal_token_cap": 64,
    }
    base.update(overrides)
    return _HealConfig(base)


_KEPT = [0, 2, 5, 7]


def _merge_layer(ref, final_kept_ids):
    """Apply a merge to one layer: select kept experts + resize the router."""
    banks = build_banks(ref)
    for bank in banks.values():
        bank.select(final_kept_ids)
    _resize_router_for_kept_experts(ref, final_kept_ids)


def test_merge_heal_on_path_matches_reference(tmp_path):
    """ON-path equivalence: drive MergeHealPlugin's pre_merge_snapshot +
    post_merge hooks with merge-heal ENABLED and assert byte-equality vs. a
    direct ``_capture_mlp_io`` + ``_heal_layer`` reference run on an
    independent identically-seeded model.

    The tiny_config gate (test_stage2_pipeline_run_layer) only exercises the
    heal-OFF path; this covers the ON path. Every RNG in the heal toolchain is
    seeded by ``layer_idx``, so two runs on identically-seeded models with the
    same captured tensors and config produce bit-identical kept-expert weights
    and a byte-equal heal_state.
    """
    cfg = _heal_cfg()

    # --- Reference run: direct _capture_mlp_io + merge + _heal_layer ------
    ref_model = _tiny_moe_model()
    ref_layer = list(iter_moe_layers(ref_model))[1]
    ref_batches = _id_batches(ref_model)
    ref_hidden = ref_layer.router.weight.shape[-1]
    ref_writer = ShardWriter(
        tmp_path / "ref_nemo", layer_idx=ref_layer.layer_idx,
        hidden_dim=ref_hidden, shard_rows=cfg.shard_rows,
    )
    _capture_mlp_io(
        ref_model, ref_layer, ref_batches, device=_CPU,
        pool_size=cfg.token_cap, shard_writer=ref_writer,
    )
    _merge_layer(ref_layer, _KEPT)
    _shared_fn = _make_shared_out_fn(ref_layer)
    ref_writer.compute_shared_companions(_shared_fn)
    ref_manifest = ref_writer.finalize(
        split_ratio=1.0 - cfg.holdout_fraction, seed=ref_layer.layer_idx,
    )
    ref_state = _heal_layer(
        layer_ref=ref_layer, final_kept_ids=_KEPT,
        manifest=ref_manifest, manifest_dir=ref_writer.out_dir,
        heal_cfg=cfg, device=_CPU,
    )
    ref_banks = build_banks(ref_layer)
    ref_weights = {
        pos: {
            name: ref_banks[name].get(pos).clone()
            for name in ("gate_proj", "up_proj", "down_proj")
        }
        for pos in range(len(_KEPT))
    }
    ref_router = ref_layer.router.weight.detach().clone()

    # --- Plugin run: drive pre_merge_snapshot + post_merge through hooks --
    plug_model = _tiny_moe_model()
    plug_layer = list(iter_moe_layers(plug_model))[1]
    plug_batches = _id_batches(plug_model)
    plugin = _make_plugin(
        heal_cfg=cfg, artifacts_dir=tmp_path / "plug_artifacts",
        model=plug_model, batches=plug_batches,
    )
    ctx = PipelineContext()
    ctx.set("layer_ref", plug_layer)
    ctx.set("grouped", {c: [c] for c in _KEPT} | {0: [0, 1, 3, 4, 6]})

    # pre_merge_snapshot: captures the layer's pre-merge mlp I/O shards.
    plugin.pre_merge_snapshot(ctx)
    assert ctx.get("nemo_writer") is not None
    assert ctx.get("xd_writer") is None

    # The orchestrator applies the merge (LayerMergePlugin.post_merge) BEFORE
    # the MergeHealPlugin.post_merge hook runs — replicate that here.
    _merge_layer(plug_layer, _KEPT)
    ctx.set("final_kept_ids", tuple(_KEPT))
    ctx.set("heal_state", None)  # LayerMergePlugin.post_merge default

    plugin.post_merge(ctx)
    plug_state = ctx.get("heal_state")
    assert plug_state is not None

    # heal_state must be byte-equal: same seeded model, same captured tensors,
    # same config -> identical optimiser trajectory.
    for key in ("steps", "accepted", "stop_reason", "holdout_mse",
                "plain_merged_holdout_mse", "heal_gap", "train_mse",
                "train_mse_at_best"):
        assert plug_state[key] == ref_state[key], (
            f"heal_state[{key!r}] diverged: plugin={plug_state[key]} "
            f"ref={ref_state[key]}"
        )

    # Kept-expert weights + the resized router must be bit-identical.
    plug_banks = build_banks(plug_layer)
    for pos in range(len(_KEPT)):
        for name in ("gate_proj", "up_proj", "down_proj"):
            assert torch.equal(
                plug_banks[name].get(pos), ref_weights[pos][name]
            ), f"plugin-path bank weight pos={pos} {name} diverged from reference"
    assert torch.equal(plug_layer.router.weight.detach(), ref_router), (
        "plugin-path healed router diverged from the _heal_layer reference"
    )


# --- monkeypatch-drift guard (T9-T16 lesson) -----------------------------
def test_no_stale_monkeypatch_of_heal_symbols():
    """The 7 heal symbols moved to pipeline.plugins.merge_heal. Any test that
    patches one on the monolith namespace must also patch it on the new module
    (or the live path drifts). Fails loudly otherwise.
    """
    tests_dir = pathlib.Path(__file__).parent
    needles = tuple(
        f'setattr(stage2_reap_ream, "{name}"' for name in _HEAL_NAMES
    )
    offenders = []
    for path in sorted(tests_dir.glob("test_*.py")):
        if path.name == pathlib.Path(__file__).name:
            continue
        text = path.read_text()
        for needle in needles:
            if needle in text and "merge_heal" not in text.replace(needle, ""):
                offenders.append(
                    f"{path.name}: patches a heal symbol on monolith only"
                )
    assert not offenders, (
        "monolith-only monkeypatch of a heal symbol — also patch it on "
        "pipeline.plugins.merge_heal:\n" + "\n".join(offenders)
    )
