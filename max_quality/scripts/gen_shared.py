#!/usr/bin/env python3
"""Stub the three shared Stage-1 artifacts for a BY-THE-BOOK REAP/REAM ablation
WITHOUT running Stage-1 GRAPE (which OOMs in Phase D and is unnecessary for the
by-the-book uniform-budget / empty-blacklist case).

Why this is correct (and not a hack)
-------------------------------------
For a by-the-book ablation the protected/super-expert set is EMPTY and the
per-layer budget is UNIFORM. GRAPE's whole job — non-uniform per-layer budget
allocation via CKA + SE-blacklist integration — is therefore a no-op here:
its output would be a uniform per-layer survivor count and an empty blacklist.
The downstream runner (``run_reap_ream_35pct``) re-pins K via
``write_uniform_budget`` anyway (run_probe.py:233), so the exact per-layer K we
write is NOT load-bearing; only the SCHEMA / key-presence is. We therefore
synthesize the three artifacts directly from the budget solver (which only
needs model *structure*, not a forward pass) and skip the expensive GRAPE run.

The three artifacts (consumed at ``<probe-root>/_shared/``):

  1. budget_decomposition.json  — exactly ``BudgetDecomposition.as_dict()``
       (run_pipeline.py:225). Loaded back as
       ``BudgetDecomposition(**{k:v for k in __dataclass_fields__})``
       (run_pipeline.py:231-233). Carries a positive ``eora_overhead_pct`` so
       the runner's net-provenance check (run_reap_ream_35pct.py:578-588) does
       NOT warn.

  2. stage1_budgets.json        — the 5-key grape_merge payload
       (grape_merge.py:443-448 / docstring 435-441):
       ``per_layer_target_experts`` (dict[str,int]),
       ``per_layer_redundancy`` (dict[str,float]),
       ``achieved_budget`` (int), ``requested_budget`` (int),
       ``config`` (dict). Stage 2 reads ONLY ``per_layer_target_experts``
       (orchestrator.py:819-822, KeyErrors if absent). The runner re-pins K via
       ``write_uniform_budget`` (run_probe.py:246-256), which requires a
       non-empty ``per_layer_target_experts`` dict to learn the layer set.

  3. stage1_blacklist.json      — the 7-top-level-key schema
       (artifacts.py:14-22 REQUIRED_BLACKLIST_TOP_LEVEL_KEYS):
       ``blacklist, per_expert_max, config, blacklist_provenance,
       dual_signal, aimer, sink_token``. EMPTY by-the-book: every
       per-layer/per-expert dict empty. Stage 2 reads ONLY
       ``blacklist_payload.get("blacklist", {})`` (orchestrator.py:823-824),
       which is tolerant; the other 6 keys are presence-only for the
       Stage-1→Stage-2 contract.

Usage
-----
    python -m scripts.gen_shared \
        --model /root/volume/models/Qwen3.6-35B-A3B \
        --out   /path/to/probe-root/_shared \
        [--target 0.35] [--ep-sp-ratio 2.0] [--min-experts 128] [--eora-pct 0.03]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("gen_shared")

# Expected geometry for Qwen3.6-35B-A3B (defensive asserts; FAIL LOUD on drift).
_EXPECTED_MOE_LAYERS = 40
_EXPECTED_EXPERTS_PER_LAYER = 256


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stub the three shared Stage-1 artifacts for a by-the-book "
                    "REAP/REAM ablation (no GRAPE).",
    )
    p.add_argument("--model", required=True,
                   help="Local HF model dir (e.g. /root/volume/models/Qwen3.6-35B-A3B).")
    p.add_argument("--out", required=True,
                   help="Output _shared/ directory for the three artifacts.")
    p.add_argument("--target", type=float, default=0.35,
                   help="Net total reduction target (default 0.35).")
    p.add_argument("--ep-sp-ratio", type=float, default=2.0,
                   help="ep:sp knob ratio passed to the solver (default 2.0).")
    p.add_argument("--min-experts", type=int, default=128,
                   help="min_experts_per_layer floor (default 128).")
    p.add_argument("--eora-pct", type=float, default=0.03,
                   help="EoRA overhead pct — must be > 0 so the artifact is "
                        "net-aware and the runner's provenance check stays quiet "
                        "(default 0.03).")
    return p.parse_args(argv)


def _load_model_for_solver(model_path: str):
    """Load the model on CPU so the solver can read its structure/param counts.

    The solver (budget/solver.py:solve) only inspects parameter *counts*
    (count_parameters / count_expert_parameters) and the per-layer expert
    geometry (iter_moe_layers) — no forward pass, no GPU needed. We therefore
    load on CPU (device_map=None) to avoid any VRAM pressure; correctness is
    identical because the solver never runs the model.
    """
    from moe_compress.utils.model_io import load_model

    log.info("Loading model on CPU for solver structure inspection: %s", model_path)
    model, _tok = load_model(
        model_path,
        torch_dtype="bfloat16",
        device_map=None,            # CPU; solver only reads param counts + geometry
        attn_implementation="sdpa",
        load_in_4bit=False,
        trust_remote_code=False,
    )
    return model


def _moe_geometry(model) -> dict[int, int]:
    """Return {layer_idx: num_routed_experts} for every MoE layer.

    iter_moe_layers yields MoELayerRef with .layer_idx and .num_routed_experts
    (model_io.py:327-342, MoELayerRef.num_routed_experts property at :226).
    """
    from moe_compress.utils.model_io import iter_moe_layers

    geom = {ref.layer_idx: ref.num_routed_experts for ref in iter_moe_layers(model)}
    if not geom:
        raise RuntimeError(
            "iter_moe_layers found NO MoE layers on the model — cannot stub "
            "Stage-1 artifacts. Is this the right checkpoint?"
        )
    return geom


def _assert_expected_geometry(geom: dict[int, int]) -> None:
    n_layers = len(geom)
    experts = sorted(set(geom.values()))
    if n_layers != _EXPECTED_MOE_LAYERS:
        raise AssertionError(
            f"Expected {_EXPECTED_MOE_LAYERS} MoE layers for Qwen3.6-35B-A3B, "
            f"found {n_layers}: {sorted(geom)}. Refusing to stub artifacts for "
            "an unexpected geometry (would silently corrupt the ablation)."
        )
    if experts != [_EXPECTED_EXPERTS_PER_LAYER]:
        raise AssertionError(
            f"Expected every MoE layer to have {_EXPECTED_EXPERTS_PER_LAYER} "
            f"routed experts, found distinct counts {experts}. Refusing to stub."
        )


def _solve_budget(model, *, target: float, ep_sp_ratio: float,
                  min_experts: int, eora_pct: float):
    """Run the budget solver with an EMPTY blacklist (by-the-book).

    Mirrors run_pipeline.py:200-207 / 217-224 exactly (blacklisted_experts={},
    eora_overhead_pct=eora_pct) so the produced decomposition is identical to
    what a real Stage-1 run would have written — minus GRAPE's per-layer split.
    """
    from moe_compress.budget import solver as budget_solver

    bd = budget_solver.solve(
        model,
        target_total_reduction=target,
        ep_sp_knob_ratio=ep_sp_ratio,
        min_experts_per_layer=min_experts,
        blacklisted_experts={},      # by-the-book: NO super-expert protection
        eora_overhead_pct=eora_pct,  # > 0 -> net-aware (provenance check quiet)
    )
    if bd is None:
        raise RuntimeError("budget_solver.solve returned None — cannot proceed.")
    if not (bd.eora_overhead_pct > 0.0):
        raise RuntimeError(
            f"Solver produced GROSS-only decomposition "
            f"(eora_overhead_pct={bd.eora_overhead_pct}); the runner's "
            "net-provenance check (run_reap_ream_35pct.py:578-588) would WARN. "
            "Pass --eora-pct > 0."
        )
    return bd


def _build_budget_decomposition_json(bd) -> dict:
    """budget_decomposition.json == BudgetDecomposition.as_dict() (solver.py:151)."""
    return bd.as_dict()


def _build_stage1_budgets_json(bd, geom: dict[int, int],
                               min_experts: int) -> tuple[dict, int]:
    """Build the 5-key stage1_budgets.json (grape_merge.py:443-448).

    survivors_per_layer = round((1 - ep) * experts_per_layer), clamped to the
    min_experts floor (so the uniform K never violates the floor the merge path
    enforces). ep = bd.expert_prune_ratio (the solver's prune knob). The exact K
    is NOT load-bearing (the runner re-pins via write_uniform_budget,
    run_probe.py:246-256) but it must be a positive int per existing MoE layer so
    the layer-set is learnable and the schema is valid.
    """
    ep = float(bd.expert_prune_ratio)
    layer_idxs = sorted(geom)
    # Uniform survivor count (by-the-book). All layers share experts_per_layer
    # (asserted upstream), so a single survivors value applies to every layer.
    experts_per_layer = next(iter(geom.values()))
    survivors = int(round((1.0 - ep) * experts_per_layer))
    survivors = max(survivors, int(min_experts))
    survivors = min(survivors, experts_per_layer)
    if survivors <= 0:
        raise RuntimeError(
            f"Derived non-positive survivors ({survivors}) from ep={ep}; "
            "refusing to write a degenerate budget."
        )

    per_layer_target_experts = {str(li): survivors for li in layer_idxs}
    # per_layer_redundancy: grape writes dict[str,float] (docstring 436). Stage 2
    # never reads it (orchestrator.py:819-822 only reads per_layer_target_experts);
    # by-the-book has no measured redundancy, so emit schema-valid 0.0 per layer.
    per_layer_redundancy = {str(li): 0.0 for li in layer_idxs}

    achieved_budget = survivors * len(layer_idxs)
    # requested_budget == the solver's global surviving-expert budget.
    requested_budget = int(bd.global_expert_budget)

    payload = {
        "per_layer_target_experts": per_layer_target_experts,
        "per_layer_redundancy": per_layer_redundancy,
        "achieved_budget": int(achieved_budget),
        "requested_budget": requested_budget,
        # The legacy 5th key is literally "config": dict(s1) (grape_merge.py
        # docstring 439-441). We have no Stage-1 YAML here; a by-the-book marker
        # dict is schema-valid (nothing reads it) and self-documenting.
        "config": {
            "stubbed_by": "scripts/gen_shared.py",
            "by_the_book": True,
            "grape_skipped": True,
            "uniform_survivors_per_layer": survivors,
        },
    }
    return payload, survivors


def _build_stage1_blacklist_json(geom: dict[int, int], *,
                                 target: float, ep_sp_ratio: float,
                                 min_experts: int, eora_pct: float) -> dict:
    """Build the EMPTY 7-key stage1_blacklist.json (artifacts.py:14-22).

    Every protected/super-expert set is empty (by-the-book). All 7 top-level
    keys are present with schema-valid empty values so the Stage-1→Stage-2
    contract (and any loader iterating the keys) does not KeyError. Stage 2
    reads ONLY ``blacklist`` (orchestrator.py:823-824), which is {} here.

    Sub-block shapes mirror the real orchestrator output (orchestrator.py:643-696)
    and the plugin contribute_artifact returns (ma_detection.py:439-447,
    aimer.py:335-340, sink_token.py:351-359) — all empty.
    """
    return {
        # blacklist: {str(layer_idx): [expert_idx, ...]} — EMPTY (no protection).
        "blacklist": {},
        # per_expert_max: {f"L{li}E{e}": float} — empty (no candidates measured).
        "per_expert_max": {},
        # config: the inner 15-key SE-detection config block (orchestrator.py:653-678).
        # Nothing downstream reads it; record the stub provenance instead.
        "config": {
            "stubbed_by": "scripts/gen_shared.py",
            "by_the_book": True,
            "super_expert_detection_skipped": True,
            "target_total_reduction": float(target),
            "ep_sp_knob_ratio": float(ep_sp_ratio),
            "min_experts_per_layer": int(min_experts),
            "eora_overhead_pct": float(eora_pct),
        },
        # blacklist_provenance: {f"L{li}E{e}": [tag, ...]} — empty (no entries).
        "blacklist_provenance": {},
        # dual_signal block (ma_detection.py:439-447) — three per-layer maps, empty.
        "dual_signal": {
            "residual_growth_per_layer": {},
            "moe_output_growth_per_layer": {},
            "moe_output_max_per_layer": {},
        },
        # aimer block (aimer.py:335-340) — three maps, empty.
        "aimer": {
            "scores": {},
            "bottom_pct_per_layer": {},
            "candidates": {},
        },
        # sink_token block (sink_token.py:351-359) — four maps, empty.
        "sink_token": {
            "mean_router_score_sink": {},
            "mean_router_score_normal": {},
            "freq_on_sink": {},
            "candidates": {},
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    from moe_compress.utils.model_io import save_json_artifact

    model = _load_model_for_solver(args.model)
    geom = _moe_geometry(model)
    _assert_expected_geometry(geom)
    log.info("MoE geometry OK: %d layers x %d experts.",
             len(geom), next(iter(geom.values())))

    bd = _solve_budget(
        model, target=args.target, ep_sp_ratio=args.ep_sp_ratio,
        min_experts=args.min_experts, eora_pct=args.eora_pct,
    )
    log.info(
        "Solver: ep=%.4f sp=%.4f K_global=%d (net target=%.4f gross=%.4f "
        "eora_pct=%.3f)",
        bd.expert_prune_ratio, bd.svd_rank_ratio, bd.global_expert_budget,
        bd.target_net_reduction, bd.projected_total_reduction, bd.eora_overhead_pct,
    )

    # Free the model ASAP — the rest is pure JSON assembly.
    del model
    try:
        import gc
        gc.collect()
    except Exception:
        pass

    decomp_json = _build_budget_decomposition_json(bd)
    budgets_json, survivors = _build_stage1_budgets_json(
        bd, geom, min_experts=args.min_experts)
    blacklist_json = _build_stage1_blacklist_json(
        geom, target=args.target, ep_sp_ratio=args.ep_sp_ratio,
        min_experts=args.min_experts, eora_pct=args.eora_pct)

    decomp_path = out_dir / "budget_decomposition.json"
    budgets_path = out_dir / "stage1_budgets.json"
    blacklist_path = out_dir / "stage1_blacklist.json"

    save_json_artifact(decomp_json, decomp_path)
    save_json_artifact(budgets_json, budgets_path)
    save_json_artifact(blacklist_json, blacklist_path)

    # ---- Summary ---------------------------------------------------------
    n_layers = len(geom)
    blacklist_empty = (
        not blacklist_json["blacklist"]
        and not blacklist_json["per_expert_max"]
        and not blacklist_json["blacklist_provenance"]
    )
    print("\n=== gen_shared: wrote 3 shared Stage-1 artifacts ===")
    print(f"  {decomp_path}")
    print(f"  {budgets_path}")
    print(f"  {blacklist_path}")
    print("--- summary ---")
    print(f"  ep (expert_prune_ratio) : {bd.expert_prune_ratio:.4f}")
    print(f"  sp (svd_rank_ratio)     : {bd.svd_rank_ratio:.4f}")
    print(f"  K_global                : {bd.global_expert_budget}")
    print(f"  n_moe_layers            : {n_layers}")
    print(f"  survivors_per_layer     : {survivors} (uniform; "
          f"{survivors}/{next(iter(geom.values()))} kept)")
    print(f"  eora_overhead_pct       : {bd.eora_overhead_pct:.3f} "
          f"(net-aware -> provenance check quiet)")
    print(f"  blacklist EMPTY         : {blacklist_empty} (by-the-book)")
    print(f"  blacklist 7-key schema  : "
          f"{sorted(blacklist_json) == sorted(['blacklist','per_expert_max','config','blacklist_provenance','dual_signal','aimer','sink_token'])}")

    if not blacklist_empty:
        raise RuntimeError("blacklist is not empty — by-the-book invariant violated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
