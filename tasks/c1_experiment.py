"""C1 empirical decision: does the bounded Gram δ_gate reconstruction change
the ACTUAL downstream merge/centroid assignments vs the reference cdist path?

Reference path  : cdist on explicitly-normalized rows (current production code).
Gram path       : reconstruct d = sqrt(2 - 2 cos) from G = full^T @ full (fp64).

Downstream consumers exercised (mirrors ream_cost.py:303-339 + ream_cost_post.py:215):
  1. sim_gate_full  -> sub-select (nc, c) -> cost = 1 - (sim_gate + sim_expert)/2
  2. argpartition top-K candidate filter (ream_cost_post.py:215) -> candidate SET per row
  3. 'pre' path: per-row argmin of cost = the centroid each non-centroid merges to
We compare BOTH the candidate sets and the final argmin assignments between paths.
"""
import torch
import numpy as np
import torch.nn.functional as F

torch.set_printoptions(precision=10)
np.random.seed(0)
torch.manual_seed(0)

# ---------------------------------------------------------------------------
# Reference (production) path — verbatim transcription of
# activation_hooks.py compute_gate_similarity_matrix (lines 501-542).
# ---------------------------------------------------------------------------
def ref_delta_gate(full: torch.Tensor, ids: list[int]) -> torch.Tensor:
    full = full.to(torch.float64)
    col = torch.tensor(ids, dtype=torch.long)
    mat = full.index_select(1, col).t().contiguous()         # [n, T]
    mat = F.normalize(mat, p=2, dim=1)
    mat = torch.where(torch.isnan(mat), torch.zeros_like(mat), mat)
    n = len(ids)
    if mat.abs().max() < 1e-9:
        return torch.zeros(n, n, dtype=torch.float64)
    d = torch.cdist(mat, mat, p=2)
    sim = 1.0 - d / d.max().clamp(min=1e-12)
    sim.fill_diagonal_(1.0)
    sim.clamp_(0.0, 1.0)
    return sim

# ---------------------------------------------------------------------------
# Gram path — the plan's §2.2/§2.3 reconstruction, accumulating G over batches.
# ---------------------------------------------------------------------------
def gram_delta_gate(batches: list[torch.Tensor], ids: list[int]) -> torch.Tensor:
    # online accumulation of G = sum_t v[t] v[t]^T  in fp64
    E = batches[0].shape[1]
    G = torch.zeros(E, E, dtype=torch.float64)
    for b in batches:
        x = b.to(torch.float64)
        G += x.transpose(0, 1) @ x
    col = torch.tensor(ids, dtype=torch.long)
    G_sub = G.index_select(0, col).index_select(1, col).to(torch.float64)
    n = len(ids)
    norms = G_sub.diagonal().clamp_min(0).sqrt()
    nz = norms > 0
    if not bool(nz.any()) or float(norms.max()) < 1e-9:
        return torch.zeros(n, n, dtype=torch.float64)
    unit = nz.to(torch.float64)
    denom = (norms[:, None] * norms[None, :]).clamp_min(1e-300)
    cos = torch.where(nz[:, None] & nz[None, :], G_sub / denom, torch.zeros_like(G_sub))
    d = (unit[:, None] + unit[None, :] - 2.0 * cos).clamp_min(0.0).sqrt()
    sim = 1.0 - d / d.max().clamp_min(1e-12)
    sim.fill_diagonal_(1.0)
    sim.clamp_(0.0, 1.0)
    return sim

# ---------------------------------------------------------------------------
# Build a REALISTIC fixture mimicking real router logits:
#   - many experts (E=128), realistic token count
#   - a base set of distinct directions
#   - SEVERAL near-colinear pairs/clusters (delta_gate's operating point —
#     it hunts for similar experts to merge), at varying closeness.
# Router logits in fp32 (that is how they arrive from the model).
# ---------------------------------------------------------------------------
def build_fixture(E=128, T=8192, n_batches=5):
    base = torch.randn(E, T, dtype=torch.float64)            # per-expert logit profile rows
    # Inject near-colinear structure: make families of experts share a direction
    # with tiny fp32-scale perturbations (the catastrophic-cancellation regime).
    # Closeness levels chosen to straddle the reviewer's reproduced failure band.
    near_pairs = [
        (1, 0, 1e-6),   # reviewer's reported failure point
        (3, 2, 1e-5),
        (5, 4, 1e-4),
        (7, 6, 1e-3),
        (9, 8, 1e-2),
        (11, 10, 1e-7),  # deeper into cancellation
        (13, 12, 1e-9),  # reviewer's ~0.99-error point
    ]
    for dst, src, eps in near_pairs:
        base[dst] = base[src] + eps * torch.randn(T, dtype=torch.float64)
    # also a tight cluster of 4 experts (centroid-competition stress)
    for k in (20, 21, 22):
        base[k] = base[19] + (10.0 ** -(4 + (k - 20))) * torch.randn(T, dtype=torch.float64)
    full = base.t().contiguous()                              # [T, E]
    # cast to fp32 — exactly how logits arrive from the model / get stored
    full32 = full.to(torch.float32)
    # split into uneven batches
    sizes = np.array_split(np.arange(T), n_batches)
    batches = [full32[s[0]:s[-1] + 1] for s in sizes]
    return full32, batches, near_pairs

full32, batches, near_pairs = build_fixture()
E = full32.shape[1]
all_ids = list(range(E))

sim_ref = ref_delta_gate(full32, all_ids)
sim_gram = gram_delta_gate(batches, all_ids)

abs_err = (sim_ref - sim_gram).abs()
print("=== sim matrix accuracy (delta_gate, full %dx%d) ===" % (E, E))
print("max abs err :", float(abs_err.max()))
print("mean abs err:", float(abs_err.mean()))
print("near-colinear pair sim values (ref vs gram, abs err):")
for dst, src, eps in near_pairs:
    r = float(sim_ref[dst, src]); g = float(sim_gram[dst, src])
    print(f"  eps={eps:>7.0e}  pair({dst},{src})  ref={r:.10f}  gram={g:.10f}  err={abs(r-g):.3e}")

# ---------------------------------------------------------------------------
# DOWNSTREAM: build the cost matrix and assignment exactly like ream_cost.py.
# We need sim_expert too. Use a realistic random sim_expert in [0,1] (the
# gate path is what differs; sim_expert is identical between the two paths, so
# it just adds common noise that the ranking must survive). Make the cost
# DOMINATED by sim_gate in the near-colinear regime to stress the decision.
# ---------------------------------------------------------------------------
def downstream(sim_gate_full, sim_expert_full, noncentroid_ids, centroid_ids, topk=48):
    all_n_ids = list(range(sim_gate_full.shape[0]))
    id_to_row = {e: i for i, e in enumerate(all_n_ids)}
    nc_rows = [id_to_row[e] for e in noncentroid_ids]
    c_cols = [id_to_row[e] for e in centroid_ids]
    sim_gate_sub = sim_gate_full[np.ix_(nc_rows, c_cols)].numpy().astype(np.float64)
    sim_exp_sub = sim_expert_full[np.ix_(nc_rows, c_cols)].numpy().astype(np.float64)
    cost = 1.0 - (sim_gate_sub + sim_exp_sub) / 2.0
    np.clip(cost, 0.0, 1.0, out=cost)
    # top-K candidate filter (ream_cost_post.py:215)
    k_cand = min(topk, cost.shape[1])
    topk_idx = np.argpartition(cost, k_cand - 1, axis=1)[:, :k_cand]
    cand_sets = [frozenset(row.tolist()) for row in topk_idx]
    # 'pre' path assignment: each non-centroid merges to argmin-cost centroid
    argmin = cost.argmin(axis=1)
    return cost, cand_sets, argmin

# Use a realistic sim_expert that is NOT correlated with sim_gate, so the
# combined cost ordering genuinely depends on the sim_gate values where they
# matter. Identical for both paths.
sim_expert_full = torch.rand(E, E, dtype=torch.float64)
sim_expert_full = (sim_expert_full + sim_expert_full.t()) / 2  # symmetric, [0,1]

# Realistic centroid/non-centroid split: ~25% centroids. Put the near-colinear
# experts deliberately on BOTH sides so the cancellation lands inside the
# candidate competition.
centroid_ids = sorted(set([0, 2, 4, 6, 8, 10, 12, 19] + list(range(30, 30 + 24))))
noncentroid_ids = [e for e in all_ids if e not in centroid_ids]

for topk in (48, 8, 4, 1):
    cost_r, cand_r, arg_r = downstream(sim_ref, sim_expert_full, noncentroid_ids, centroid_ids, topk)
    cost_g, cand_g, arg_g = downstream(sim_gram, sim_expert_full, noncentroid_ids, centroid_ids, topk)
    cand_flips = sum(1 for a, b in zip(cand_r, cand_g) if a != b)
    arg_flips = int((arg_r != arg_g).sum())
    print(f"\n=== downstream topk={topk} ===")
    print(f"  cost max abs err          : {np.abs(cost_r - cost_g).max():.3e}")
    print(f"  top-K candidate-set flips : {cand_flips} / {len(cand_r)} rows")
    print(f"  pre-path argmin flips     : {arg_flips} / {len(arg_r)} rows")
    if arg_flips:
        rows = np.where(arg_r != arg_g)[0]
        for ci in rows[:10]:
            print(f"    nc={noncentroid_ids[ci]}: ref->c{centroid_ids[arg_r[ci]]} "
                  f"gram->c{centroid_ids[arg_g[ci]]} "
                  f"(cost_r={cost_r[ci,arg_r[ci]]:.6f}/{cost_r[ci,arg_g[ci]]:.6f} "
                  f"cost_g={cost_g[ci,arg_r[ci]]:.6f}/{cost_g[ci,arg_g[ci]]:.6f})")
