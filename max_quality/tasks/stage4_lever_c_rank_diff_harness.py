#!/usr/bin/env python3
"""Lever-C rank-diff harness (HUMAN-GATED — NOT a golden regen).

The tiny golden fixture produces ``compensated_params=0`` / empty ``rank_map``,
so it cannot exercise real EoRA ranks. This standalone harness drives the
production full-SVD ``take_eff`` (``min(r, U_p.shape[1])`` with
``torch.linalg.svd(delta_prime, full_matrices=False)``) against the Lever-C
Gram-side ``take_eff`` (``min(r, min(d_out, n_keep))``) on representative,
production-like shapes and reports the FLIP COUNT.

A flip is the only surface that could change ``eora_ranks.json``. Expected: 0.
This script does NOT run ``MOE_REGEN_GOLDEN=1`` and does NOT touch any golden.

Run:  python3 max_quality/tasks/stage4_lever_c_rank_diff_harness.py
"""
from __future__ import annotations

import torch


def production_take_eff(delta_prime: torch.Tensor, r: int) -> int:
    U_p, _, _ = torch.linalg.svd(delta_prime, full_matrices=False)
    return min(r, int(U_p.shape[1]))


def gram_take_eff(delta_prime: torch.Tensor, r: int) -> int:
    d_out_, n_keep_ = delta_prime.shape
    return min(r, min(d_out_, n_keep_))


def main() -> int:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}")
    # Representative shapes per the plan envelope: production gate/up have
    # d_out≈2048; n_keep (post-noise-floor eigen-count) up to ~768; r<=128.
    # Both rectangular directions are exercised (n_keep<d_out and n_keep>d_out).
    d_outs = (2048, 768)
    n_keeps = (64, 128, 256, 512, 768, 1024, 2048)
    rs = (32, 64, 96, 128)
    seeds = (1, 2, 3)

    flips = 0
    cases = 0
    flip_rows = []
    for d_out in d_outs:
        for n_keep in n_keeps:
            for r in rs:
                for seed in seeds:
                    g = torch.Generator().manual_seed(
                        hash((d_out, n_keep, r, seed)) & 0x7FFFFFFF
                    )
                    # Mix full-rank and rank-deficient delta_prime to probe
                    # the near-zero / tie boundary the plan flagged.
                    dp = torch.randn(d_out, n_keep, generator=g, dtype=torch.float32)
                    if seed % 2 == 0:
                        # inject a near-degenerate tail (rank-deficient block)
                        k = max(1, min(d_out, n_keep) // 4)
                        dp[:, -k:] = dp[:, :1] * 1e-7
                    dp = dp.to(dev)
                    prod = production_take_eff(dp, r)
                    gram = gram_take_eff(dp, r)
                    cases += 1
                    if prod != gram:
                        flips += 1
                        flip_rows.append((d_out, n_keep, r, seed, prod, gram))

    print(f"cases={cases}  flips={flips}")
    if flip_rows:
        print("FLIP ROWS (d_out, n_keep, r, seed, prod_take_eff, gram_take_eff):")
        for row in flip_rows:
            print("  ", row)
        print("RESULT: NON-ZERO FLIPS — golden would change; HUMAN sign-off required.")
        return 1
    print("RESULT: 0 flips — Gram-side take_eff matches production SVD; golden byte-identical.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
