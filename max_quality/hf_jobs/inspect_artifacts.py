"""Read the bucket artifacts and report on them without touching GPU.

Use between staged HF Jobs submissions — ``--stop-after-stage N`` exits,
this script reads whatever was written to the bucket, prints a sanity
summary, and either blesses moving on or flags the issue.

Run locally:

    python hf_jobs/inspect_artifacts.py stage0
    python hf_jobs/inspect_artifacts.py stage1
    python hf_jobs/inspect_artifacts.py stage2
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from huggingface_hub import HfFileSystem


BUCKET_URI = "buckets/pirola/moe-cache"
ARTIFACTS_DIR_IN_BUCKET = "artifacts"


def _read_json_from_bucket(relative_path: str) -> dict:
    fs = HfFileSystem()
    uri = f"{BUCKET_URI}/{ARTIFACTS_DIR_IN_BUCKET}/{relative_path}"
    with fs.open(uri, "r") as fh:
        return json.load(fh)


def _list_bucket_files(prefix: str) -> list[str]:
    fs = HfFileSystem()
    uri = f"{BUCKET_URI}/{ARTIFACTS_DIR_IN_BUCKET}/{prefix}"
    try:
        return sorted(fs.ls(uri, detail=False))
    except FileNotFoundError:
        return []


def inspect_stage0() -> int:
    data = _read_json_from_bucket("stage1_blacklist.json")
    blacklist = data.get("blacklist", {})
    per_max   = data.get("per_expert_max", {})
    layers = sorted({int(k) for k in blacklist}) or []
    print(f"Stage 0 artifacts:")
    print(f"  layers with blacklisted experts : {len(blacklist)}")
    print(f"  total experts blacklisted       : {sum(len(v) for v in blacklist.values())}")
    print(f"  per_expert_max entries          : {len(per_max)}")
    if not per_max:
        print("  ERROR: per_expert_max is empty — Stage 0 did not run forward.")
        return 1
    # Sample stats
    values = sorted((float(v) for v in per_max.values()), reverse=True)
    print(f"  top-10 max|down_proj|           : {[round(v, 3) for v in values[:10]]}")
    print(f"  median max|down_proj|           : {round(values[len(values)//2], 3)}")
    # Sanity: no layer should blacklist more than ~15% of its experts given
    # cap_per_layer=4 and 256 experts.
    for li, experts in blacklist.items():
        if len(experts) > 10:
            print(f"  WARNING: layer {li} has {len(experts)} blacklisted — cap was 4.")
    return 0


def inspect_stage1() -> int:
    data = _read_json_from_bucket("stage1_budgets.json")
    budgets = {int(k): int(v) for k, v in data["per_layer_target_experts"].items()}
    print(f"Stage 1 artifacts:")
    print(f"  layers                : {len(budgets)}")
    # Schema names this `requested_budget` (was historically `global_budget`).
    print(f"  requested budget      : {data['requested_budget']}")
    print(f"  achieved budget       : {data['achieved_budget']}")
    print(f"  per-layer budget min  : {min(budgets.values())}")
    print(f"  per-layer budget max  : {max(budgets.values())}")
    print(f"  mean                  : {sum(budgets.values()) / len(budgets):.1f}")
    # Invariants
    if min(budgets.values()) < 9:     # config validator enforces ≥9
        print(f"  ERROR: budget fell below min_experts=9 floor.")
        return 1
    if max(budgets.values()) > 256:
        print(f"  ERROR: budget above original 256 experts.")
        return 1
    return 0


def inspect_stage2() -> int:
    # merge_map under stage2_pruned/
    fs = HfFileSystem()
    mm_uri = f"{BUCKET_URI}/{ARTIFACTS_DIR_IN_BUCKET}/stage2_pruned/merge_map.json"
    try:
        with fs.open(mm_uri, "r") as fh:
            mm = json.load(fh)
    except FileNotFoundError:
        print("ERROR: merge_map.json not found — Stage 2 did not write it.")
        return 1

    budgets = _read_json_from_bucket("stage1_budgets.json")["per_layer_target_experts"]
    print(f"Stage 2 artifacts:")
    print(f"  merged layers                        : {len(mm)}")
    print(f"  Σ surviving experts across layers    : "
          f"{sum(len(g) for g in mm.values())}")
    # Every surviving expert (new_idx) should have a non-empty group of original IDs.
    bad = 0
    for li, groups in mm.items():
        expected = int(budgets[li])
        if len(groups) != expected:
            print(f"  ERROR: layer {li}: merge_map has {len(groups)} slots but "
                  f"budget is {expected}")
            bad += 1
        # Check contiguity: new_idx must span 0..expected-1
        slots = sorted(int(k) for k in groups)
        if slots != list(range(expected)):
            print(f"  ERROR: layer {li}: merge_map keys are not 0..{expected-1}: {slots[:10]}…")
            bad += 1
    if bad:
        return 1
    print("  merge_map shape OK")
    return 0


def main(argv) -> int:
    if len(argv) < 1:
        print(__doc__, file=sys.stderr)
        return 2
    which = argv[0]
    return {
        "stage0": inspect_stage0,
        "stage1": inspect_stage1,
        "stage2": inspect_stage2,
    }[which]()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
