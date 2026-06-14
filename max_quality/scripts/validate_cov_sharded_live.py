#!/usr/bin/env python3
"""LIVE validation of the device_map=balanced (multi-GPU SHARDED) Stage-3 cov path.

The s234 ablation runs Stage-3 cov collection with the model(s) sharded across GPUs
via device_map=balanced (the only way the dual-model cross-cov fits). That multi-GPU
cov path was landed but NEVER run on real >=2 GPUs. This validates it BEFORE the
multi-hour ablation relies on it:

  (1) B-cov EQUIVALENCE: collect B-cov on the 1-GPU model and on the 2-GPU sharded
      model; assert allclose (sharding must not change the math — only fp/device
      reduction-order noise). gate/up + factored down.
  (2) CROSS-COV SMOKE: load teacher+student BOTH sharded (device_map=balanced), run
      the real cross-cov dual-forward collection, assert C is finite + non-zero and
      it doesn't crash on cross-device tensors. (No 1-GPU reference possible: two 35B
      models don't co-fit one card — that's the whole reason for sharding.)

Synthetic calib at the real seq_len; cov_window pinned (auto over-sizes the dual-model
cross-cov footprint and OOMs).
"""
import argparse, sys, gc
import torch

from moe_compress.utils.model_io import iter_moe_layers
from moe_compress.utils.calibration import iter_batches
from moe_compress.utils.activation_hooks import InputCovarianceAccumulator
from moe_compress.stage3.plugins import covariance_collection as cc


def load(path, device_map):
    from transformers import AutoModelForCausalLM
    kw = dict(dtype=torch.bfloat16, low_cpu_mem_usage=True)
    if device_map is None:
        m = AutoModelForCausalLM.from_pretrained(path, **kw).to("cuda:0")
    else:
        m = AutoModelForCausalLM.from_pretrained(path, device_map=device_map, **kw)
    m.train(False)
    return m


def collect_B(model, calib, window, device):
    B = InputCovarianceAccumulator(); B.set_storage_dtype(torch.float32)
    cc._collect_covariances(
        model, list(iter_moe_layers(model)), iter_batches(calib, 1), B, device=device,
        teacher_model=None, C_acc=None, cov_window_size=window, calib=calib, cov_auto=False)
    B.finalize_all()
    return B.covariance


def compare(ref, shard):
    k1, k2 = set(ref), set(shard)
    assert k1 == k2, f"key mismatch {len(k1^k2)}"
    eq = cl = fail = 0; worst = 0.0
    for k in sorted(k1, key=str):
        a, b = ref[k].float(), shard[k].float()
        if torch.allclose(a, b, rtol=1e-3, atol=1e-4):
            if torch.equal(a, b): eq += 1
            else: cl += 1
            worst = max(worst, (a - b).abs().max().item())
        else:
            fail += 1
            print(f"  FAIL {k}: maxabs={(a-b).abs().max().item():.3e}")
    print(f"[B-equiv] {len(k1)} keys: bitwise={eq} allclose={cl} FAIL={fail} worst={worst:.3e}")
    return fail == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-seq", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    print(f"[setup] visible GPUs={torch.cuda.device_count()}", flush=True)
    assert torch.cuda.device_count() >= 2, "need >=2 GPUs"
    import json
    with open(f"{args.model}/config.json") as f:
        mc = json.load(f)
    vocab = mc.get("vocab_size") or mc["text_config"]["vocab_size"]
    g = torch.Generator().manual_seed(args.seed)
    calib = torch.randint(0, vocab, (args.n_seq, args.seq_len), generator=g)

    # (1) B-cov 1-GPU reference
    print("[1a] B-cov 1-GPU (device_map=None, cuda:0) ...", flush=True)
    m1 = load(args.model, None)
    cov_ref = collect_B(m1, calib, args.window, torch.device("cuda:0"))
    del m1; gc.collect(); torch.cuda.empty_cache()

    # (1) B-cov 2-GPU sharded
    print("[1b] B-cov 2-GPU sharded (device_map=balanced) ...", flush=True)
    m2 = load(args.model, "balanced")
    print("     shard map:", {k: str(v) for k, v in list(m2.hf_device_map.items())[:6]}, "...", flush=True)
    cov_shard = collect_B(m2, calib, args.window, torch.device("cuda:0"))
    b_ok = compare(cov_ref, cov_shard)
    del m2; gc.collect(); torch.cuda.empty_cache()

    # (2) cross-cov smoke: teacher + student both sharded
    print("[2] cross-cov SMOKE: teacher+student both device_map=balanced ...", flush=True)
    student = load(args.model, "balanced")
    teacher = load(args.model, "balanced")
    teacher.train(False)
    for p in teacher.parameters(): p.requires_grad_(False)
    Bc = InputCovarianceAccumulator(); Bc.set_storage_dtype(torch.float32)
    Cc = InputCovarianceAccumulator(); Cc.set_storage_dtype(torch.float32)
    small = calib[: min(4, args.n_seq)]
    cross_ok = True; reason = ""
    try:
        cc._collect_covariances(
            student, list(iter_moe_layers(student)), iter_batches(small, 1), Bc,
            device=torch.device("cuda:0"),
            teacher_model=teacher, teacher_moe_layers=list(iter_moe_layers(teacher)),
            C_acc=Cc, cov_window_size=args.window, calib=small, cov_auto=False)
        Bc.finalize_all(); Cc.finalize_all()
        nC = len(Cc.covariance)
        finite = all(torch.isfinite(v).all().item() for v in Cc.covariance.values())
        nonzero = any(v.abs().sum().item() > 0 for v in Cc.covariance.values())
        cross_ok = nC > 0 and finite and nonzero
        reason = f"C_keys={nC} finite={finite} nonzero={nonzero}"
    except Exception as e:
        cross_ok = False; reason = f"CRASH: {repr(e)[:200]}"
    print(f"[cross-smoke] {'PASS' if cross_ok else 'FAIL'} — {reason}", flush=True)

    ok = b_ok and cross_ok
    print(f"\n[RESULT] sharded-cov validation = {'PASS' if ok else 'FAIL'} "
          f"(B-equiv={'ok' if b_ok else 'FAIL'}, cross-smoke={'ok' if cross_ok else 'FAIL'})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
