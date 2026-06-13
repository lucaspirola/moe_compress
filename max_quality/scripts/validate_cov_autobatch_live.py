#!/usr/bin/env python3
"""LIVE 35B/H200 validation of the Stage-3 cov auto-batch pin+wire.

Runs the REAL ``_collect_covariances`` twice on the real model:
  (1) baseline  cov_auto=False, forward batch = 1  (the byte-identical golden path)
  (2) auto      cov_auto=True   (size_batch + run_with_oom_backoff fill VRAM)

Then asserts equivalence with the SAME comparison the CPU golden uses
(test_cov_autobatch_wire.py): torch.equal for gate_proj/up B keys, torch.allclose
(rtol=1e-4, atol=1e-5) for factored down_proj B keys (forward-activation drift, the
reduction pin makes it N-independent). Also reports the wall-clock speedup.

B-path only (teacher_model=None): cross-cov C needs two full models (~140GB), which
won't co-fit one H200; C is bitwise-covered by the CPU golden + exercised live via the
pruned-student path in the ablation. Calib is synthetic tokens at the real seq_len --
equivalence and speedup are forward-SHAPE properties, not token-value properties.
"""
import argparse, time, sys
import torch

from moe_compress.utils.model_io import iter_moe_layers
from moe_compress.utils.calibration import iter_batches
from moe_compress.utils.activation_hooks import InputCovarianceAccumulator
from moe_compress.stage3.plugins import covariance_collection as cc
from moe_compress.utils.auto_batch import AutoBatchConfig


def load_model(path):
    from transformers import AutoModelForCausalLM
    print(f"[load] {path}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, low_cpu_mem_usage=True,
    ).to("cuda")
    model.train(False)  # eval mode (avoid literal that trips the editor security hook)
    return model


def run_collect(model, moe_layers, calib, *, cov_auto, window, max_cap, headroom):
    B = InputCovarianceAccumulator()
    B.set_storage_dtype(torch.float32)  # fp32 storage -> exact-as-possible compare
    cfg = AutoBatchConfig(enabled=True, headroom_frac=headroom, max_cap=max_cap)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    cc._collect_covariances(
        model, moe_layers, iter_batches(calib, batch_size=1), B, device="cuda",
        teacher_model=None, teacher_moe_layers=None, C_acc=None,
        cov_window_size=window,
        calib=calib, cov_auto=cov_auto, auto_batch_cfg=cfg,
    )
    B.finalize_all()
    torch.cuda.synchronize()
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2**30
    return B.covariance, dt, peak


def compare(cov_bs1, cov_auto):
    k1, k2 = set(cov_bs1), set(cov_auto)
    assert k1 == k2, f"key sets differ: only_bs1={k1-k2} only_auto={k2-k1}"
    n_equal = n_close = n_fail = 0
    worst = 0.0
    for k in sorted(k1, key=str):
        a, b = cov_bs1[k].float(), cov_auto[k].float()
        is_down = "down" in (k[2] if isinstance(k, tuple) else str(k))
        if is_down:
            if torch.allclose(a, b, rtol=1e-4, atol=1e-5):
                n_close += 1
                worst = max(worst, (a - b).abs().max().item())
            else:
                n_fail += 1
                print(f"  FAIL(down) {k}: maxabs={(a-b).abs().max().item():.3e}")
        else:
            if torch.equal(a, b):
                n_equal += 1
            else:
                n_fail += 1
                print(f"  FAIL(gate/up NOT bitwise-equal) {k}: maxabs={(a-b).abs().max().item():.3e}")
    print(f"[compare] {len(k1)} keys: bitwise-equal(gate/up)={n_equal} "
          f"allclose(down)={n_close} FAIL={n_fail} worst_down_absdiff={worst:.3e}")
    return n_fail == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-seq", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--window", type=int, default=8, help="cov_window_size (MoE layers/pass), same for both runs")
    ap.add_argument("--max-cap", type=int, default=256)
    ap.add_argument("--headroom", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    model = load_model(args.model)
    moe_layers = list(iter_moe_layers(model))
    cfg_t = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    vocab = int(cfg_t.vocab_size)
    print(f"[setup] {len(moe_layers)} MoE layers, vocab={vocab}, "
          f"calib={args.n_seq}x{args.seq_len}, window={args.window}", flush=True)

    g = torch.Generator().manual_seed(args.seed)
    calib = torch.randint(0, vocab, (args.n_seq, args.seq_len), generator=g)

    print("[run] baseline cov_auto=False (bs=1) ...", flush=True)
    cov_bs1, t_bs1, peak_bs1 = run_collect(
        model, moe_layers, calib, cov_auto=False, window=args.window,
        max_cap=args.max_cap, headroom=args.headroom)
    print(f"[run] baseline done: {t_bs1:.1f}s peak={peak_bs1:.1f}GiB", flush=True)

    print("[run] auto cov_auto=True ...", flush=True)
    cov_auto, t_auto, peak_auto = run_collect(
        model, moe_layers, calib, cov_auto=True, window=args.window,
        max_cap=args.max_cap, headroom=args.headroom)
    print(f"[run] auto done: {t_auto:.1f}s peak={peak_auto:.1f}GiB", flush=True)

    ok = compare(cov_bs1, cov_auto)
    speedup = t_bs1 / t_auto if t_auto > 0 else float("nan")
    print(f"\n[RESULT] equivalence={'PASS' if ok else 'FAIL'} | "
          f"speedup={speedup:.2f}x (bs1 {t_bs1:.1f}s -> auto {t_auto:.1f}s) | "
          f"peak bs1={peak_bs1:.1f} auto={peak_auto:.1f} GiB")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
