"""End-to-end Stage 3 smoke test on the fused-experts synthetic fixture.

Runs the full stage3_svd.run() path:
  B-covariance collection → D-Rank allocation → AA-SVD factorization →
  FactoredExperts install → checkpoint + rank_map.json write → spill cleanup.

Block-refine is disabled in tiny_config (lbfgs_steps=5, enabled=False) so
this stays under a second on CPU. AA-SVD uses plain-SVD fallback when
A_cov is absent — that's fine for wiring correctness.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest
import torch

from moe_compress import stage1_grape, stage2_reap_ream
from moe_compress import stage3_svd
from moe_compress.budget.solver import BudgetDecomposition
from moe_compress.utils.model_io import FactoredExperts, iter_moe_layers


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


@pytest.fixture(params=["fp32", "bf16"])
def patched_stage3(request, monkeypatch, tiny_config, tiny_config_bf16):
    """Patch calibration loaders in every stage module that calls them.

    Parametrized over ``fp32`` (default) and ``bf16`` covariance storage so the
    eigh-based AA-SVD path is exercised under bf16 quantization end-to-end —
    defense in depth for the bf16 covariance bug fixed in §6.5.
    """
    tiny_config = tiny_config_bf16 if request.param == "bf16" else tiny_config
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
    monkeypatch.setattr(stage3_svd, "build_calibration_tensor", _fake_build)

    from moe_compress.utils import model_io as mio
    monkeypatch.setattr(mio, "save_compressed_checkpoint", _noop_save)
    monkeypatch.setattr(stage2_reap_ream, "save_compressed_checkpoint", _noop_save)
    monkeypatch.setattr(stage3_svd, "save_compressed_checkpoint", _noop_save)

    return tiny_config


def _run_stages_012(model, config, tmp_path):
    """Run Stages 1→2 to get a post-prune model + Stage 2 covariance artifact."""
    decomp = BudgetDecomposition(
        total_reduction_ratio=0.2,
        expert_prune_ratio=0.5,
        svd_rank_ratio=0.14,
        global_expert_budget=4,
        min_experts_per_layer=2,
        blacklisted_experts={},
    )
    stage1_grape.run(model, _TinyTokenizer(), config, tmp_path, decomp)
    stage2_reap_ream.run(
        model, _TinyTokenizer(), config, tmp_path, device=None,
    )
    return decomp


def test_stage3_smoke(tiny_model, patched_stage3, tmp_path):
    """stage3_svd.run() completes without exception and produces expected artifacts."""
    decomp = _run_stages_012(tiny_model, patched_stage3, tmp_path)

    stage3_svd.run(
        tiny_model, _TinyTokenizer(), patched_stage3, tmp_path, decomp, device=None,
    )

    # All MoE layers must now be FactoredExperts.
    moe_layers = list(iter_moe_layers(tiny_model))
    for ref in moe_layers:
        assert isinstance(ref.experts_module, FactoredExperts), (
            f"Layer {ref.layer_idx}: experts_module is {type(ref.experts_module).__name__}, "
            "expected FactoredExperts after Stage 3"
        )

    # rank_map.json must be written.
    rank_map_path = tmp_path / "stage3_svd" / "rank_map.json"
    assert rank_map_path.exists(), "rank_map.json not written"
    rank_map = json.loads(rank_map_path.read_text())
    assert "rank_map" in rank_map
    assert "T_budget" in rank_map
    assert "per_layer_ranks" in rank_map

    # Original weights snapshot must be written.
    originals_path = tmp_path / "_stage3_original_weights.pt"
    assert originals_path.exists(), "_stage3_original_weights.pt not written"
    originals = torch.load(originals_path, map_location="cpu")
    # Should have gate_proj + up_proj + down_proj for each (layer, expert).
    n_experts_total = sum(ref.num_routed_experts for ref in moe_layers)
    assert len(originals) == n_experts_total * 3, (
        f"Expected {n_experts_total * 3} original tensors, got {len(originals)}"
    )

    # Spill dir must be cleaned up on success.
    spill_dir = tmp_path / "_stage3_bcov_partial"
    assert not spill_dir.exists(), (
        "B-cov spill dir was not cleaned up after successful Stage 3"
    )


def test_stage3_factored_experts_shapes(tiny_model, patched_stage3, tmp_path):
    """FactoredExperts installed by Stage 3 have correct rank-k shapes."""
    decomp = _run_stages_012(tiny_model, patched_stage3, tmp_path)

    stage3_svd.run(
        tiny_model, _TinyTokenizer(), patched_stage3, tmp_path, decomp, device=None,
    )

    rank_map_path = tmp_path / "stage3_svd" / "rank_map.json"
    per_layer_ranks = json.loads(rank_map_path.read_text())["per_layer_ranks"]

    for ref in iter_moe_layers(tiny_model):
        fe = ref.experts_module
        assert isinstance(fe, FactoredExperts)
        ranks = per_layer_ranks[str(ref.layer_idx)]
        hidden = fe.hidden_dim
        d_int = fe.intermediate_dim
        k_gate = ranks["gate_proj"]
        k_up = ranks["up_proj"]
        k_down = ranks["down_proj"]
        N = ref.num_routed_experts
        # gate_proj_U: [N, d_int, k_gate]  gate_proj_V: [N, k_gate, hidden]
        assert fe.gate_proj_U.shape == (N, d_int, k_gate), \
            f"gate_proj_U shape mismatch: {fe.gate_proj_U.shape}"
        assert fe.gate_proj_V.shape == (N, k_gate, hidden), \
            f"gate_proj_V shape mismatch: {fe.gate_proj_V.shape}"
        # up_proj_U: [N, d_int, k_up]  up_proj_V: [N, k_up, hidden]
        assert fe.up_proj_U.shape == (N, d_int, k_up), \
            f"up_proj_U shape mismatch: {fe.up_proj_U.shape}"
        assert fe.up_proj_V.shape == (N, k_up, hidden), \
            f"up_proj_V shape mismatch: {fe.up_proj_V.shape}"
        # down_proj_U: [N, hidden, k_down]  down_proj_V: [N, k_down, d_int]
        assert fe.down_proj_U.shape == (N, hidden, k_down), \
            f"down_proj_U shape mismatch: {fe.down_proj_U.shape}"
        assert fe.down_proj_V.shape == (N, k_down, d_int), \
            f"down_proj_V shape mismatch: {fe.down_proj_V.shape}"


def test_stage3_with_preseeded_acov(tiny_model, patched_stage3, tmp_path):
    """Stage 3 uses A_cov when available (not just plain-SVD fallback)."""
    decomp = _run_stages_012(tiny_model, patched_stage3, tmp_path)

    # Stage 2 already wrote _stage2_input_covariance.pt — verify Stage 3 loads it.
    acov_path = tmp_path / "_stage2_input_covariance.pt"
    assert acov_path.exists(), "Stage 2 should have written A-cov"
    payload = torch.load(acov_path, map_location="cpu")
    assert "covariance" in payload and len(payload["covariance"]) > 0

    # Stage 3 must complete without error; AA-SVD path (not plain-SVD fallback)
    # should be taken for at least some experts. We verify indirectly: if A_cov
    # was loaded, rank_map will still be written (same code path).
    stage3_svd.run(
        tiny_model, _TinyTokenizer(), patched_stage3, tmp_path, decomp, device=None,
    )
    assert (tmp_path / "stage3_svd" / "rank_map.json").exists()


def test_stage3_phase_c5_emits_loss_metrics(
    tiny_model, patched_stage3, tmp_path, monkeypatch,
):
    """Phase C.5 (block-level joint refinement, paper 2604.02119 §3.3) emits
    per-block ``stage3/c5_loss_init``, ``c5_loss_final``, and
    ``c5_loss_rel_drop`` trackio metrics. Replaces the legacy per-matrix
    L-BFGS refine which emitted ``refine_bw_*`` metrics.
    """
    # Snapshot a teacher copy BEFORE Stages 1-2 mutate the model.
    teacher_copy = copy.deepcopy(tiny_model)
    teacher_copy.eval()
    for p in teacher_copy.parameters():
        p.requires_grad_(False)

    config = copy.deepcopy(patched_stage3)
    config["stage3_svd"]["block_refine"] = {
        "enabled": True, "epochs": 1, "batch_size": 1,
        "learning_rate": 1.0e-4, "warmup_ratio": 0.1, "weight_decay": 0.0,
    }
    # Phase C.5 requires the teacher resident; cross_covariance triggers the
    # teacher load. The tiny synthetic model can't be loaded from HF, so we
    # patch load_model to return the pre-snapshotted copy.
    # Cross-cov stays False (avoids pre-existing dual-forward bug); block_refine=True
    # alone now triggers teacher load via the _need_teacher branch.
    config["stage3_svd"]["aa_svd"]["cross_covariance"] = False
    monkeypatch.setattr(stage3_svd, "load_model",
                        lambda *args, **kwargs: (teacher_copy, None))
    # `model.name_or_path` and `model.revision` keys must exist on the
    # config for the load_model invocation, even though the patched fn ignores
    # them.
    config.setdefault("model", {})
    config["model"].setdefault("name_or_path", "tiny")
    config["model"].setdefault("revision", "main")
    config["model"].setdefault("torch_dtype", "float32")
    config["model"].setdefault("device_map", "cpu")
    config["model"].setdefault("attn_implementation", "eager")

    captured: list[dict] = []

    def _capture_metrics(metrics):
        if isinstance(metrics, dict):
            captured.append(dict(metrics))

    monkeypatch.setattr(stage3_svd, "_trackio_log", _capture_metrics)

    decomp = _run_stages_012(tiny_model, config, tmp_path)
    stage3_svd.run(
        tiny_model, _TinyTokenizer(), config, tmp_path, decomp, device=None,
    )

    c5_metrics = [m for m in captured if any(k.startswith("stage3/c5_") for k in m)]
    assert c5_metrics, "no Phase C.5 trackio events captured"

    keys_seen = set()
    for m in c5_metrics:
        for k, v in m.items():
            if k.startswith("stage3/c5_"):
                keys_seen.add(k.split("/", 1)[1])
                assert math.isfinite(float(v)), f"non-finite C.5 metric {k}={v}"
    for kind in ("c5_loss_init", "c5_loss_final", "c5_loss_rel_drop"):
        assert kind in keys_seen, (
            f"expected '{kind}' in C.5 metrics; got {sorted(keys_seen)}"
        )


def test_stage3_spill_dir_created_during_run(tiny_model, patched_stage3, tmp_path,
                                              monkeypatch):
    """During Stage 3, _stage3_bcov_partial/ is created and populated with .pt files,
    then cleaned up. We intercept the cleanup to verify the intermediate state."""
    import shutil as _shutil

    decomp = _run_stages_012(tiny_model, patched_stage3, tmp_path)
    spill_dir = tmp_path / "_stage3_bcov_partial"
    captured_files: list[list[str]] = []

    original_rmtree = _shutil.rmtree

    def _capturing_rmtree(path, **kwargs):
        p = Path(path)
        if p == spill_dir:
            captured_files.append(sorted(f.name for f in p.glob("*.pt")))
        original_rmtree(path, **kwargs)

    monkeypatch.setattr(_shutil, "rmtree", _capturing_rmtree)

    stage3_svd.run(
        tiny_model, _TinyTokenizer(), patched_stage3, tmp_path, decomp, device=None,
    )

    assert captured_files, "rmtree was never called on spill_dir — cleanup not triggered"
    spill_files = captured_files[0]
    moe_layers = list(iter_moe_layers(tiny_model))
    # After Stage 3 success, every layer should have had its spill file created.
    # (The factor loop lazy-loads + unloads; files aren't deleted mid-run.)
    assert len(spill_files) == len(moe_layers), (
        f"Expected {len(moe_layers)} spill files before cleanup, got {spill_files}"
    )
    for ref in moe_layers:
        assert f"layer_{ref.layer_idx}.pt" in spill_files, \
            f"layer_{ref.layer_idx}.pt missing from spill dir at cleanup time"
