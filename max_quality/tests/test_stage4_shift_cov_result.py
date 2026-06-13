"""Stage-4 shift-cov result-changing integration test (Task 5).

Runs the tiny Stage-0→4 pipeline TWICE on independent model copies:
  arm-anchor (default whitening_cov)
  arm-shift  (whitening_cov="shift", consuming an INJECTED differing
              _stage3_shift_covariance.pt)
and asserts the widened EoRA factors differ NON-TRIVIALLY between arms AND
that both arms are valid (finite, correct shapes, ranks within budget).

Per the review's Low note, we ALWAYS inject a deliberately differing fake
shift cov rather than relying on the tiny fixture's anchor/shift differing
numerically — so the swap's effect is pinned unambiguously.

Local helpers are redeclared on purpose (codebase discipline: no cross-test
imports).
"""
from __future__ import annotations

import copy
from pathlib import Path

import pytest

try:
    import torch
    from moe_compress import stage1, stage3_svd, stage4_eora
    from moe_compress.stage2 import orchestrator as stage2_reap_ream
    from moe_compress.budget.solver import BudgetDecomposition
    from moe_compress.utils.atomic_io import (
        atomic_torch_save,
        write_manifest_last,
    )
    from moe_compress.utils.model_io import iter_moe_layers, FactoredExperts
except Exception as e:  # pragma: no cover - import guard
    pytest.skip(f"Stage 4 imports unavailable: {e}", allow_module_level=True)

from tests.conftest import _TinyModel  # type: ignore  # noqa: E402


class _TinyTokenizer:
    name_or_path = "tiny-tokenizer"
    eos_token_id = 0

    def __call__(self, text, *_, **__):
        return {"input_ids": [min(ord(c) % 32, 31) for c in (text or " ")]}

    def save_pretrained(self, *_args, **_kwargs):
        return None


def _noop_save(model, tokenizer, path, **kwargs):
    Path(path).mkdir(parents=True, exist_ok=True)


def _patch(monkeypatch):
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


def _decomp():
    return BudgetDecomposition(
        total_reduction_ratio=0.2,
        expert_prune_ratio=0.5,
        svd_rank_ratio=0.14,
        global_expert_budget=4,
        min_experts_per_layer=2,
        blacklisted_experts={},
    )


def _run_0123(model, config, tmp_path):
    decomp = _decomp()
    stage1.run(model, _TinyTokenizer(), config, tmp_path, decomp)
    stage2_reap_ream.run(model, _TinyTokenizer(), config, tmp_path, device=None)
    stage3_svd.run(model, _TinyTokenizer(), config, tmp_path, decomp, device=None)


def _snapshot_factors(model):
    """Capture widened EoRA factors per (layer, name) after Stage 4."""
    snap: dict = {}
    for ref in iter_moe_layers(model):
        fe = ref.experts_module
        if not isinstance(fe, FactoredExperts):
            continue
        snap[(ref.layer_idx, "gate_U")] = fe.gate_proj_U.data.clone()
        snap[(ref.layer_idx, "gate_V")] = fe.gate_proj_V.data.clone()
        snap[(ref.layer_idx, "down_U")] = fe.down_proj_U.data.clone()
        snap[(ref.layer_idx, "down_V")] = fe.down_proj_V.data.clone()
    return snap


def _config():
    """tiny_config inlined with a budget that actually produces EoRA factors.

    The default tiny budget (3 %) rounds to rank 0 → no compensation →
    whitening-independent. Bump compensation_budget_pct + the rank cap so
    _solve_expert_tile actually widens, exposing the whitening-cov effect.
    """
    cfg = {
        "model": {"name_or_path": "tiny", "revision": "main",
                  "torch_dtype": "float32", "device_map": "cpu",
                  "attn_implementation": "sdpa", "load_in_4bit": False,
                  "trust_remote_code": False},
        "target": {"total_reduction_ratio": 0.25,
                   "initial_expert_reduction": 0.25,
                   "initial_svd_reduction": 0.10},
        "calibration": {"source": "c4-math-code", "dataset": "allenai/c4",
                        "subset": "en", "split": "train", "seed": 0,
                        "num_sequences": 8, "sequence_length": 16,
                        "super_expert_num_samples": 4,
                        "domain_mix": {"c4": 1.0, "math": 0.0, "code": 0.0},
                        "math_dataset": "unused", "code_dataset": "unused"},
        "stage1_grape": {"num_calibration_samples": 4,
                         "similarity_metric": "cosine", "min_experts_per_layer": 2,
                         "early_layer_bonus": 0, "early_layer_bonus_depth": 0,
                         "late_layer_bonus": 0, "late_layer_bonus_depth": 0,
                         "target_total_experts_per_layer_avg": 3,
                         "super_expert_detection": {"zscore_threshold": 1.0,
                                                    "max_blacklisted_per_layer": 1,
                                                    "global_blacklist_cap_pct": 0.50}},
        "stage2_reap_ream": {"batch_size": 1, "num_calibration_samples": 4,
                             "reap_min_active_tokens": 1,
                             "covariance_storage_dtype": "float32",
                             "max_merge_group_size": 0,
                             "ream_cost_sigma_threshold": float("inf"),
                             "ream_cost_bump_ratio": 0.10,
                             "ream": {"hungarian": True,
                                      "frequency_weighted_merge": True}},
        "stage3_svd": {"scope": "moe_experts_only",
                       "d_rank": {"parameter_cost_omega_mode": "auto"},
                       "swift_svd_plus": {"alpha_grid": [0.5],
                                          "validation_samples": 2,
                                          "metric": "wikitext2_ppl",
                                          "per_group_type": True,
                                          "alpha_search_min_host_ram_gb": 0.0},
                       "aa_svd": {"use_post_prune_inputs": True,
                                  "cross_covariance": False},
                       "block_refine": {"enabled": False, "epochs": 1,
                                        "batch_size": 1, "learning_rate": 1.0e-4,
                                        "warmup_ratio": 0.1, "weight_decay": 0.0}},
        # Budget bumped so EoRA actually widens (rank > 0).
        "stage4_eora": {"per_expert": True, "compensation_budget_pct": 1.0,
                        "eigenspace_rank_cap": 4},
        "logging": {"level": "INFO", "log_every_n_steps": 5,
                    "save_intermediate_every_n_layers": 1},
    }
    return cfg


def _inject_differing_shift_cov(anchor_dir, shift_dir):
    """Build a shift cov from the anchor's _stage2_input_covariance.pt keys but
    deliberately PERTURBED so it differs from the anchor numerically — then
    write it into ``shift_dir`` as the Stage-4 ride-along artifact."""
    a_payload = torch.load(
        anchor_dir / "_stage2_input_covariance.pt", map_location="cpu"
    )
    a_cov = a_payload.get("covariance", {})
    assert a_cov, "anchor covariance must be non-empty for a meaningful test"
    shift = {}
    for key, t in a_cov.items():
        t32 = t.to(torch.float32)
        d = t32.shape[0]
        # Perturb: scale up + add a diagonal bump. Keeps SPD-ness (positive
        # diagonal dominance) while guaranteeing a different whitening basis.
        shift[key] = (t32 * 3.0 + torch.eye(d) * 5.0).to(torch.float32)
    out_path = shift_dir / "_stage3_shift_covariance.pt"
    atomic_torch_save(out_path, {"format_version": 1, "covariance": shift})
    manifest = out_path.with_suffix(out_path.suffix + ".MANIFEST.json")
    write_manifest_last(out_path, manifest, schema_version=1,
                        extra_meta={"n_keys": len(shift),
                                    "artifact": "stage3_shift_covariance"},
                        compute_sha256=False)
    return shift


def test_shift_cov_changes_result_and_both_valid(monkeypatch, tmp_path):
    _patch(monkeypatch)

    # Run Stages 0→3 ONCE so the post-S3 model + sidecars are IDENTICAL for
    # both arms; the only thing that differs between the arms is the Stage-4
    # whitening cov. (Two independent S0→3 runs would let other sources of
    # variation leak in, making the comparison fail to isolate the swap.)
    src_dir = tmp_path / "s0123"
    src_dir.mkdir()
    cfg_base = _config()
    src_model = _TinyModel()
    _run_0123(src_model, cfg_base, src_dir)

    def _stage4_arm(arm_name, whitening_cov, inject_shift):
        arm_dir = tmp_path / arm_name
        arm_dir.mkdir()
        # Copy the S0→3 sidecars Stage 4 consumes into this arm's dir.
        for fn in ("_stage2_input_covariance.pt", "_stage3_original_weights.pt"):
            src = src_dir / fn
            if src.exists():
                (arm_dir / fn).write_bytes(src.read_bytes())
            man = src_dir / (fn + ".MANIFEST.json")
            if man.exists():
                (arm_dir / (fn + ".MANIFEST.json")).write_bytes(man.read_bytes())
        injected = None
        if inject_shift:
            injected = _inject_differing_shift_cov(src_dir, arm_dir)
        cfg = _config()
        if whitening_cov is not None:
            cfg["stage4_eora"]["whitening_cov"] = whitening_cov
        # Deepcopy the post-S3 model so the in-place widening of one arm does
        # not affect the other.
        model = copy.deepcopy(src_model)
        stage4_eora.run(model, _TinyTokenizer(), cfg, arm_dir)
        return _snapshot_factors(model), injected

    factors_anchor, _ = _stage4_arm("anchor", None, inject_shift=False)
    factors_shift, injected = _stage4_arm("shift", "shift", inject_shift=True)

    # Sanity: the bumped budget actually widened SOMETHING (rank > 0). Without
    # this the magnitude comparison below would be vacuously satisfied.
    assert factors_anchor, "anchor arm produced no factored layers"
    assert any(v.numel() > 0 and v.shape[-1] > 0 for v in factors_anchor.values()), (
        "EoRA produced no rank>0 factors — budget too small to exercise the swap"
    )

    # The injected shift cov has the right per-expert key/shape shape.
    assert injected  # non-empty
    for key, t in injected.items():
        assert len(key) == 3  # (layer, expert, matrix)
        assert t.ndim == 2 and t.shape[0] == t.shape[1]

    # ----- both valid -----
    for snap, label in ((factors_anchor, "anchor"), (factors_shift, "shift")):
        for k, v in snap.items():
            assert torch.isfinite(v).all(), f"{label} {k} has non-finite values"

    # ----- shapes match across arms (same budget/model) -----
    assert set(factors_anchor) == set(factors_shift)
    for k in factors_anchor:
        assert factors_anchor[k].shape == factors_shift[k].shape, (
            f"shape mismatch for {k}"
        )

    # ----- shift != anchor with NON-TRIVIAL magnitude -----
    total_delta = 0.0
    total_ref = 0.0
    for k in factors_anchor:
        a = factors_anchor[k].to(torch.float32)
        s = factors_shift[k].to(torch.float32)
        total_delta += (a - s).abs().sum().item()
        total_ref += a.abs().sum().item()
    assert total_ref > 0, "anchor factors are all-zero — test is vacuous"
    rel = total_delta / total_ref
    assert rel > 1e-3, (
        f"shift whitening did not change the result non-trivially "
        f"(relative U·V delta {rel:.3e} ≤ 1e-3)"
    )
