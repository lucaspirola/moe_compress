#!/usr/bin/env python3
"""Derive the box ablation config from reap_faithful.yaml.

Sets the local model path (no HF re-download), points calibration at the box
self-traces JSONL (so every stage reads the replay, not the streamed mix), turns
Stage-3 cov AUTO-BATCH ON (the now-fixed speedup), and — when --shard is given —
shards both models across the visible GPUs (device_map=balanced) so the dual-model
cross-cov fits, plus pins a fixed cov WINDOW (the auto window-sizer mis-estimates
the dual-model + per-expert cross-cov footprint and OOMs). Pure yaml round-trip.
"""
import sys, argparse, yaml

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("dst")
ap.add_argument("--base", default="/root/work", help="box base dir")
ap.add_argument("--shard", action="store_true", help="device_map=balanced (multi-GPU sharding)")
ap.add_argument("--cov-window", type=int, default=0, help="fixed multi_gpu.cov_window_size (0=leave auto)")
args = ap.parse_args()

MODEL = f"{args.base}/models/Qwen3.6-35B-A3B"
JSONL = f"{args.base}/run/self_traces_489ee0e1b17b43b0.jsonl"

with open(args.src) as f:
    cfg = yaml.safe_load(f)

m = cfg.setdefault("model", {})
m["name_or_path"] = MODEL
if args.shard:
    m["device_map"] = "balanced"   # shard teacher+student across all visible GPUs

cfg.setdefault("calibration", {})["jsonl_path"] = JSONL
cfg.setdefault("target", {})["net_of_eora"] = True

s3 = cfg.setdefault("stage3_svd", {})
s3["cov_batch_size"] = "auto"
s3["auto_batch"] = {"enabled": True, "headroom_frac": 0.1, "max_cap": 256}

if args.cov_window > 0:
    mg = cfg.setdefault("multi_gpu", {})
    mg["cov_window_size"] = args.cov_window   # pin G (auto over-sizes for dual-model cross-cov -> OOM)

with open(args.dst, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)

print(f"wrote {args.dst}")
print(f"  model.name_or_path = {m['name_or_path']}")
print(f"  model.device_map   = {m.get('device_map', '(default)')}")
print(f"  calibration.jsonl_path = {cfg['calibration']['jsonl_path']}")
print(f"  target.net_of_eora = {cfg['target']['net_of_eora']}")
print(f"  stage3_svd.cov_batch_size = {s3['cov_batch_size']} auto_batch={s3['auto_batch']}")
print(f"  multi_gpu.cov_window_size = {cfg.get('multi_gpu', {}).get('cov_window_size', '(auto)')}")
