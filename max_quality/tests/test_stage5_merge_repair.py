"""CPU-only correctness tests for Direction E — Stage 2.5 merge-repair.

Direction E adds a config-gated `merge_repair` mode to `stage5_router_kd`:
  * the merged centroid experts (kept experts that absorbed others during the
    Stage-2 merge) become trainable, on top of the router;
  * a per-layer MoE-block-output MSE term against the teacher is added to the
    existing vocab-logit KL.

These tests exercise the unit-testable surface without a GPU or a real model:
  * the merge-map → merged-centroid identification;
  * which (and only which) params get `requires_grad=True` when the flag is on,
    and the gradient-mask hook that restricts updates to the centroid rows;
  * the MSE-term construction on synthetic teacher/student layer tensors;
  * that with the flag OFF the freeze set is byte-identical to pre-E behaviour;
  * the fail-loud behaviour when a vocab-logits cache is configured together
    with `merge_repair` on.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from moe_compress import stage5_router_kd as s5m
from moe_compress.router_kd import orchestrator as rk_orchestrator
from moe_compress.utils.model_io import iter_moe_layers

# conftest.py provides the `tiny_model` / `tiny_config` fixtures (a synthetic
# fused-experts MoE that is the structural twin of Qwen3_5Moe).


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_merge_map(tmp_path: Path, mapping: dict) -> Path:
    """Write a Stage-2-style merge_map.json under stage2_pruned/."""
    d = tmp_path / "stage2_pruned"
    d.mkdir(parents=True, exist_ok=True)
    # JSON keys are strings — mirror exactly what save_json_artifact produces.
    payload = {
        str(layer): {str(new_idx): list(members) for new_idx, members in groups.items()}
        for layer, groups in mapping.items()
    }
    (d / "merge_map.json").write_text(json.dumps(payload), encoding="utf-8")
    return d / "merge_map.json"


def _merge_map_with_centroids(model: nn.Module) -> dict:
    """Build a merge map where, in every MoE layer, expert 0 is a merged
    centroid (absorbed experts 0 and 1) and every other expert is untouched."""
    mapping: dict = {}
    for ref in iter_moe_layers(model):
        n = ref.num_routed_experts
        assert n >= 3, "fixture must have >=3 experts for this test"
        groups = {0: [0, 1]}  # expert 0 absorbed 0 and 1 -> merged centroid
        for e in range(1, n):
            groups[e] = [e + 1]  # length-1 -> untouched, NOT a repair target
        mapping[ref.layer_idx] = groups
    return mapping


# ---------------------------------------------------------------------------
# merge-map parsing + merged-centroid identification
# ---------------------------------------------------------------------------


def test_load_merge_map_normalizes_string_keys(tmp_path):
    _write_merge_map(tmp_path, {0: {0: [0, 1], 1: [2]}, 3: {0: [0]}})
    loaded = s5m._load_merge_map(tmp_path, None)
    assert loaded == {0: {0: [0, 1], 1: [2]}, 3: {0: [0]}}
    # Keys must be ints, not the JSON strings.
    assert all(isinstance(k, int) for k in loaded)
    assert all(isinstance(k, int) for k in loaded[0])


def test_load_merge_map_missing_fails_loud(tmp_path):
    with pytest.raises(RuntimeError, match="merge map not found"):
        s5m._load_merge_map(tmp_path, None)


def test_load_merge_map_honors_override_path(tmp_path):
    alt = tmp_path / "custom_merge.json"
    alt.write_text(json.dumps({"0": {"0": [0, 1]}}), encoding="utf-8")
    loaded = s5m._load_merge_map(tmp_path, "custom_merge.json")
    assert loaded == {0: {0: [0, 1]}}


def test_merged_centroid_rows_only_real_merges():
    merge_map = {5: {0: [0, 1, 2], 1: [3], 2: [4, 5], 3: [6]}}
    # Rows 0 and 2 absorbed >1 expert -> merged centroids.
    # Rows 1 and 3 are length-1 -> untouched, must NOT be returned.
    assert s5m._merged_centroid_rows(merge_map, 5) == [0, 2]
    # A layer not in the map (e.g. dense layer) has zero merged centroids.
    assert s5m._merged_centroid_rows(merge_map, 99) == []


def test_select_merge_repair_layers_model_agnostic(tiny_model):
    merge_map = _merge_map_with_centroids(tiny_model)
    selected = s5m._select_merge_repair_layers(tiny_model, merge_map)
    layer_indices = {ref.layer_idx for ref, _ in selected}
    expected = {ref.layer_idx for ref in iter_moe_layers(tiny_model)}
    assert layer_indices == expected
    # Every selected layer reports exactly the merged centroid (row 0).
    for _ref, rows in selected:
        assert rows == [0]


def test_select_merge_repair_layers_rejects_out_of_range_row(tiny_model):
    # A centroid row beyond the post-merge expert count == wrong merge map.
    n = next(iter_moe_layers(tiny_model)).num_routed_experts
    bad_map = {ref.layer_idx: {n + 99: [0, 1]} for ref in iter_moe_layers(tiny_model)}
    with pytest.raises(RuntimeError, match="outside the post-merge expert range"):
        s5m._select_merge_repair_layers(tiny_model, bad_map)


def test_select_merge_repair_layers_skips_layers_without_merges(tiny_model):
    # Map exists but every group is length-1 -> no merged centroids anywhere.
    no_merge = {
        ref.layer_idx: {e: [e] for e in range(ref.num_routed_experts)}
        for ref in iter_moe_layers(tiny_model)
    }
    assert s5m._select_merge_repair_layers(tiny_model, no_merge) == []


# ---------------------------------------------------------------------------
# unfreeze: requires_grad scope + gradient-mask hook
# ---------------------------------------------------------------------------


def test_unfreeze_merged_experts_sets_requires_grad_on_expert_tensors(tiny_model):
    # Start from the production freeze state: only the router weight is trainable.
    s5m._freeze_non_routers(tiny_model, ["mlp.gate.weight"])
    merge_map = _merge_map_with_centroids(tiny_model)
    repair_layers = s5m._select_merge_repair_layers(tiny_model, merge_map)
    handles = s5m._unfreeze_merged_experts(tiny_model, repair_layers)

    # The fused expert tensors of every repair layer must now be trainable.
    for ref, _rows in repair_layers:
        for tname in ("gate_up_proj", "down_proj"):
            p = getattr(ref.experts_module, tname)
            assert p.requires_grad, f"{tname} should be trainable after unfreeze"
    # One hook per unfrozen expert tensor (2 per layer for fused experts).
    assert len(handles) == 2 * len(repair_layers)

    for h in handles.values():
        h.remove()


def test_gradient_mask_zeros_non_centroid_rows(tiny_model):
    """The grad-mask hook must let gradient through ONLY for centroid rows."""
    s5m._freeze_non_routers(tiny_model, ["mlp.gate.weight"])
    merge_map = _merge_map_with_centroids(tiny_model)  # centroid = row 0
    repair_layers = s5m._select_merge_repair_layers(tiny_model, merge_map)
    handles = s5m._unfreeze_merged_experts(tiny_model, repair_layers)

    ref0 = repair_layers[0][0]
    gate_up = getattr(ref0.experts_module, "gate_up_proj")  # [E, 2*int, hid]
    # A loss that touches EVERY expert row uniformly.
    loss = (gate_up ** 2).sum()
    loss.backward()

    grad = gate_up.grad
    assert grad is not None
    # Row 0 (the merged centroid) keeps its gradient.
    assert grad[0].abs().sum().item() > 0.0
    # Every other expert row was zeroed by the mask hook.
    for e in range(1, gate_up.shape[0]):
        assert torch.count_nonzero(grad[e]).item() == 0, (
            f"non-centroid expert row {e} should have zero gradient"
        )

    for h in handles.values():
        h.remove()


def test_unfreeze_does_not_touch_non_repair_params(tiny_model):
    """Embedding / lm_head / shared-expert must stay frozen after unfreeze."""
    s5m._freeze_non_routers(tiny_model, ["mlp.gate.weight"])
    merge_map = _merge_map_with_centroids(tiny_model)
    repair_layers = s5m._select_merge_repair_layers(tiny_model, merge_map)
    handles = s5m._unfreeze_merged_experts(tiny_model, repair_layers)

    trainable = {n for n, p in tiny_model.named_parameters() if p.requires_grad}
    # Router weights stay trainable.
    assert any("mlp.gate.weight" in n for n in trainable)
    # Expert fused tensors are now trainable.
    assert any("experts.gate_up_proj" in n for n in trainable)
    # But nothing outside routers + experts leaked in.
    for n in trainable:
        assert ("mlp.gate.weight" in n) or ("experts." in n), (
            f"unexpected trainable param leaked: {n}"
        )
    assert not any("embed" in n for n in trainable)
    assert not any("lm_head" in n for n in trainable)
    assert not any("shared_expert" in n for n in trainable)

    for h in handles.values():
        h.remove()


# ---------------------------------------------------------------------------
# per-layer MSE term construction
# ---------------------------------------------------------------------------


def test_merge_repair_mse_zero_when_outputs_identical():
    s = {0: torch.randn(2, 4, 8), 1: torch.randn(2, 4, 8)}
    t = {k: v.clone() for k, v in s.items()}
    mse = s5m._merge_repair_mse(s, t, [0, 1])
    assert torch.allclose(mse, torch.zeros(()), atol=1e-12)


def test_merge_repair_mse_matches_manual_mean():
    torch.manual_seed(0)
    s = {0: torch.randn(2, 3, 5), 7: torch.randn(2, 3, 5)}
    t = {0: torch.randn(2, 3, 5), 7: torch.randn(2, 3, 5)}
    mse = s5m._merge_repair_mse(s, t, [0, 7])
    manual = 0.5 * (
        ((s[0] - t[0]) ** 2).mean() + ((s[7] - t[7]) ** 2).mean()
    )
    assert torch.allclose(mse, manual, atol=1e-6)


def test_merge_repair_mse_backpropagates_into_student():
    """The MSE term must carry gradient into the student layer outputs."""
    student_out = torch.randn(2, 3, 5, requires_grad=True)
    teacher_out = torch.randn(2, 3, 5)  # detached fixed target
    mse = s5m._merge_repair_mse({4: student_out}, {4: teacher_out}, [4])
    mse.backward()
    assert student_out.grad is not None
    assert student_out.grad.abs().sum().item() > 0.0


def test_merge_repair_mse_missing_layer_fails_loud():
    s = {0: torch.randn(1, 2, 3)}
    t = {0: torch.randn(1, 2, 3)}
    with pytest.raises(RuntimeError, match="output missing from"):
        s5m._merge_repair_mse(s, t, [0, 1])  # layer 1 never captured


def test_merge_repair_mse_shape_mismatch_fails_loud():
    s = {0: torch.randn(1, 2, 3)}
    t = {0: torch.randn(1, 2, 4)}
    with pytest.raises(RuntimeError, match="shape mismatch"):
        s5m._merge_repair_mse(s, t, [0])


def test_merge_repair_mse_empty_layer_list_is_zero():
    mse = s5m._merge_repair_mse({}, {}, [])
    assert float(mse) == 0.0


# ---------------------------------------------------------------------------
# forward-hook capture of MoE-block outputs
# ---------------------------------------------------------------------------


def test_layer_output_capture_records_block_outputs(tiny_model):
    layer_indices = {ref.layer_idx for ref in iter_moe_layers(tiny_model)}
    cap = s5m._LayerOutputCapture(tiny_model, layer_indices, detach=True)
    try:
        ids = torch.randint(0, 32, (2, 6), dtype=torch.long)
        tiny_model(input_ids=ids)
        # Every requested MoE layer produced a captured tensor.
        assert set(cap.outputs.keys()) == layer_indices
        for li, tensor in cap.outputs.items():
            assert tensor.shape == (2, 6, tiny_model.config.hidden_size)
            # detach=True -> no autograd history retained.
            assert not tensor.requires_grad
    finally:
        cap.remove()


def test_layer_output_capture_keeps_grad_when_not_detached(tiny_model):
    layer_indices = {ref.layer_idx for ref in iter_moe_layers(tiny_model)}
    cap = s5m._LayerOutputCapture(tiny_model, layer_indices, detach=False)
    try:
        ids = torch.randint(0, 32, (1, 4), dtype=torch.long)
        tiny_model(input_ids=ids)
        for tensor in cap.outputs.values():
            assert tensor.grad_fn is not None
    finally:
        cap.remove()


def test_layer_output_capture_remove_unregisters_hooks(tiny_model):
    layer_indices = {ref.layer_idx for ref in iter_moe_layers(tiny_model)}
    cap = s5m._LayerOutputCapture(tiny_model, layer_indices, detach=True)
    cap.remove()
    cap.clear()
    ids = torch.randint(0, 32, (1, 4), dtype=torch.long)
    tiny_model(input_ids=ids)
    # Hooks removed -> nothing captured on a subsequent forward.
    assert cap.outputs == {}


# ---------------------------------------------------------------------------
# flag-OFF: byte-identical freeze set + loss
# ---------------------------------------------------------------------------


def test_flag_off_freeze_set_is_unchanged(tiny_model, tiny_config):
    """With merge_repair absent/off the trainable set == router-only, exactly
    as on pre-Direction-E `main`."""
    patterns = tiny_config["stage5_router_kd"]["trainable_name_patterns"]
    s5m._freeze_non_routers(tiny_model, patterns)
    baseline = {n for n, p in tiny_model.named_parameters() if p.requires_grad}

    # The merge_repair config block is the gate; default-off => no unfreeze.
    mr_cfg = tiny_config["stage5_router_kd"].get("merge_repair") or {}
    assert bool(mr_cfg.get("enabled", False)) is False

    # Re-applying the freeze with the flag off yields the identical set.
    s5m._freeze_non_routers(tiny_model, patterns)
    after = {n for n, p in tiny_model.named_parameters() if p.requires_grad}
    assert after == baseline
    # And it is exactly the router weights — no expert tensors.
    assert all("mlp.gate.weight" in n for n in after)
    assert after, "router params must be trainable"


def test_flag_off_loss_equals_pure_kl():
    """When merge_repair is off the combined loss is exactly the vocab KL.

    Exercises the *production* `_combine_kd_loss` — the same function `run()`
    calls — with `mse_term=None` (the flag-off path). It must return the exact
    `kl_loss` tensor object, so `loss` is byte-identical to pre-E `main`.
    """
    torch.manual_seed(0)
    student_logits = torch.randn(1, 5, 32, requires_grad=True)
    teacher_logits = torch.randn(1, 5, 32)
    kl_loss = s5m._chunked_vocab_kl(student_logits, teacher_logits, 1.0, chunk_size=512)

    # Flag-off path: run() passes mse_term=None. mse_weight is irrelevant and
    # must not change the result (test a non-zero weight to prove it).
    loss = s5m._combine_kd_loss(kl_loss, None, mse_weight=3.0)
    assert loss is kl_loss  # SAME object — no MSE term, no new graph node


# ---------------------------------------------------------------------------
# fail-loud: merge_repair + teacher_logits_cache are incompatible
# ---------------------------------------------------------------------------


def test_merge_repair_with_logits_cache_fails_loud(tiny_model, tiny_config, tmp_path,
                                                   monkeypatch):
    """`run()` must refuse merge_repair when a vocab-logits cache is set: the
    cache has no per-layer hidden states, so the MSE term is impossible."""
    from moe_compress import stage2
    from moe_compress.utils import model_io as mio

    # Build a real (trivial) teacher-logits cache so the cache branch in run()
    # is taken — format_version 1, shapes aligned with the tiny config.
    # The fixture config lacks vocab_size (run()'s cache topology check needs
    # it); set it to the embedding vocab so the cache passes topology checks
    # and the merge_repair guard is the line that fires.
    s5 = tiny_config["stage5_router_kd"]
    seq_len = int(s5["max_sequence_length"])
    n_samples = int(s5["max_calibration_samples"])
    vocab = tiny_model.embed.num_embeddings
    tiny_model.config.vocab_size = vocab
    cache = {
        "format_version": 1,
        "batch_size": int(s5["batch_size"]),
        "sequence_length": seq_len,
        "num_samples": n_samples,
        "logits": torch.zeros(n_samples * seq_len, vocab, dtype=torch.float32),
    }
    cache_path = tmp_path / "teacher_logits.pt"
    torch.save(cache, cache_path)

    # A merge map so the merge_repair config is otherwise valid.
    _write_merge_map(tmp_path, _merge_map_with_centroids(tiny_model))

    cfg = json.loads(json.dumps(tiny_config))  # deep copy of the plain-dict cfg
    cfg["stage5_router_kd"]["teacher_logits_cache"] = str(cache_path)
    cfg["stage5_router_kd"]["merge_repair"] = {"enabled": True, "mse_weight": 1.0}

    # Stub the heavy save so run() does not try to serialize a real checkpoint
    # (it must raise long before reaching that point anyway).
    def _noop_save(model, tokenizer, path, **kwargs):
        Path(path).mkdir(parents=True, exist_ok=True)
        return Path(path)

    # RK-8: the real Router-KD orchestrator binds save_compressed_checkpoint /
    # build_calibration_tensor by direct import — patch them there.
    monkeypatch.setattr(rk_orchestrator, "save_compressed_checkpoint", _noop_save)
    monkeypatch.setattr(rk_orchestrator, "build_calibration_tensor",
                        lambda *a, **k: torch.zeros(n_samples, seq_len, dtype=torch.long))

    class _TinyTok:
        name_or_path = "tiny"
        eos_token_id = 0

        def save_pretrained(self, *_a, **_k):
            return None

    with pytest.raises(RuntimeError, match="incompatible with .*teacher_logits_cache"):
        s5m.run(tiny_model, _TinyTok(), cfg, tmp_path, device=None,
                no_resume=True, stage_key="stage2p5")


# ---------------------------------------------------------------------------
# loss-combination weighting (the run() formula, exercised directly)
# ---------------------------------------------------------------------------


def test_loss_combination_applies_mse_weight():
    """`loss = kl_loss + mse_weight * mse_term` — the weight scales only MSE.

    Exercises the production `_combine_kd_loss` directly (the function `run()`
    uses), so a regression in the combine formula is caught here."""
    kl_loss = torch.tensor(2.0)
    s = {0: torch.full((1, 2, 3), 1.0)}
    t = {0: torch.full((1, 2, 3), 3.0)}  # per-element (1-3)^2 = 4 -> mse = 4.0
    mse_term = s5m._merge_repair_mse(s, t, [0])
    assert torch.allclose(mse_term, torch.tensor(4.0))

    for w in (0.0, 0.5, 1.0, 3.0):
        loss = s5m._combine_kd_loss(kl_loss, mse_term, mse_weight=w)
        assert torch.allclose(loss, torch.tensor(2.0 + w * 4.0))
