#!/usr/bin/env python3
"""LIVE 2-GPU data-parallel validation of the Stage-3 cov DP reduce.

Proves the load-bearing claim: with the per-sequence reduction pin, two replicas
that AUTO-SIZE their forward batch INDEPENDENTLY (per-GPU VRAM) produce a key-wise
DP reduce (``_reduce_spilled_cov_dirs``, fp32 sum of finalized Grams) that equals
a single-GPU bs=1 run -- so replicas need NOT agree on a batch (no min-agreement).

Drives the REAL code: each replica is a spawned process pinned to one GPU via
CUDA_VISIBLE_DEVICES, calls the REAL ``_collect_covariances`` with ``spill_dir``
(→ real ``spill_layer_to_disk``), and the parent reduces with the REAL
``_reduce_spilled_cov_dirs``. Reference = one replica, full calib, bs=1.

  REF : 1 replica  · full calib       · cov_auto=False (bs=1)  → out_ref
  DP  : 2 replicas · disjoint shards   · cov_auto=True  (auto) → out_dp

Assert out_dp == out_ref: torch.equal for gate/up B keys, torch.allclose
(rtol=1e-4, atol=1e-5) for factored down_proj B keys. B-path only (teacher=None),
consistent with the 1-GPU validation. Synthetic calib at the real seq_len.
"""
import argparse, json, os, sys
from pathlib import Path
import torch
import torch.multiprocessing as mp


def _worker(rank, visible, model_path, shard_file, spill_dir, cov_auto,
            window, max_cap, headroom):
    os.environ["CUDA_VISIBLE_DEVICES"] = visible  # MUST precede CUDA init
    import torch as t
    from transformers import AutoModelForCausalLM
    from moe_compress.utils.model_io import iter_moe_layers
    from moe_compress.utils.calibration import iter_batches
    from moe_compress.utils.activation_hooks import InputCovarianceAccumulator
    from moe_compress.stage3.plugins import covariance_collection as cc
    from moe_compress.utils.auto_batch import AutoBatchConfig

    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=t.bfloat16, low_cpu_mem_usage=True).to("cuda")
    model.train(False)
    moe_layers = list(iter_moe_layers(model))
    shard = t.load(shard_file)
    t.cuda.reset_peak_memory_stats()
    B = InputCovarianceAccumulator(); B.set_storage_dtype(t.float32)
    cc._collect_covariances(
        model, moe_layers, iter_batches(shard, batch_size=1), B, device="cuda",
        spill_dir=Path(spill_dir), teacher_model=None, teacher_moe_layers=None,
        C_acc=None, cov_window_size=window,
        calib=shard, cov_auto=cov_auto,
        auto_batch_cfg=AutoBatchConfig(enabled=True, headroom_frac=headroom, max_cap=max_cap),
    )
    peak = t.cuda.max_memory_allocated() / 2**30
    print(f"[worker{rank}] CUDA_VISIBLE_DEVICES={visible} cov_auto={cov_auto} "
          f"shard={tuple(shard.shape)} peak={peak:.1f}GiB spill={spill_dir}", flush=True)


def _run_procs(spawn_specs):
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_worker, args=a) for a in spawn_specs]
    for p in procs: p.start()
    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"replica process exited with code {p.exitcode}")


def load_canonical(out_dir):
    d = {}
    for p in sorted(Path(out_dir).glob("layer_*.pt")):
        payload = torch.load(p, map_location="cpu", weights_only=True)
        for k, tns in payload["covariance"].items():
            d[k] = tns
    return d


def compare(ref, dp):
    k1, k2 = set(ref), set(dp)
    assert k1 == k2, f"key sets differ: only_ref={len(k1-k2)} only_dp={len(k2-k1)}"
    n_equal = n_close = n_fail = 0; worst = 0.0
    for k in sorted(k1, key=str):
        a, b = ref[k].float(), dp[k].float()
        is_down = "down" in (k[2] if isinstance(k, tuple) else str(k))
        if is_down:
            if torch.allclose(a, b, rtol=1e-4, atol=1e-5):
                n_close += 1; worst = max(worst, (a - b).abs().max().item())
            else:
                n_fail += 1; print(f"  FAIL(down) {k}: maxabs={(a-b).abs().max().item():.3e}")
        else:
            if torch.equal(a, b): n_equal += 1
            else:
                n_fail += 1; print(f"  FAIL(gate/up not bitwise) {k}: maxabs={(a-b).abs().max().item():.3e}")
    print(f"[compare] {len(k1)} keys: bitwise-equal(gate/up)={n_equal} "
          f"allclose(down)={n_close} FAIL={n_fail} worst_down_absdiff={worst:.3e}")
    return n_fail == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--n-seq", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--max-cap", type=int, default=256)
    ap.add_argument("--headroom", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    ngpu = torch.cuda.device_count()
    print(f"[setup] visible GPUs={ngpu}", flush=True)
    assert ngpu >= 2, f"need >=2 GPUs, have {ngpu}"

    with open(Path(args.model) / "config.json") as f:
        mc = json.load(f)
    vocab = mc.get("vocab_size") or mc.get("text_config", {}).get("vocab_size")
    print(f"[setup] vocab={vocab} calib={args.n_seq}x{args.seq_len} window={args.window}", flush=True)

    work = Path(args.workdir); work.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(args.seed)
    calib = torch.randint(0, vocab, (args.n_seq, args.seq_len), generator=g)
    half = args.n_seq // 2
    torch.save(calib, work / "shard_ref.pt")
    torch.save(calib[:half], work / "shard0.pt")
    torch.save(calib[half:], work / "shard1.pt")

    from moe_compress.stage3.plugins.covariance_collection import _reduce_spilled_cov_dirs

    print("[REF] 1 replica, full calib, bs=1 (cov_auto=False) ...", flush=True)
    _run_procs([(0, "0", args.model, str(work / "shard_ref.pt"),
                 str(work / "r_ref"), False, args.window, args.max_cap, args.headroom)])
    _reduce_spilled_cov_dirs([work / "r_ref"], work / "out_ref", storage_dtype=torch.float32)

    print("[DP] 2 replicas, disjoint shards, cov_auto=True (independent auto-size) ...", flush=True)
    _run_procs([
        (0, "0", args.model, str(work / "shard0.pt"), str(work / "r0"), True, args.window, args.max_cap, args.headroom),
        (1, "1", args.model, str(work / "shard1.pt"), str(work / "r1"), True, args.window, args.max_cap, args.headroom),
    ])
    _reduce_spilled_cov_dirs([work / "r0", work / "r1"], work / "out_dp", storage_dtype=torch.float32)

    ref = load_canonical(work / "out_ref")
    dp = load_canonical(work / "out_dp")
    ok = compare(ref, dp)
    print(f"\n[RESULT] DP-reduce equivalence = {'PASS' if ok else 'FAIL'} "
          f"(2-replica independent-auto reduce vs 1-GPU bs=1)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
