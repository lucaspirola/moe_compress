"""Reproduce the reviewer's ~1.9e-4 (eps 1e-6) / ~0.99 (eps 1e-9) errors, then
show the plan's fp64 path is robust in the SAME regime. The reviewer's huge
error comes from doing the WHOLE distance reconstruction in fp32:
  cos in fp32, then d=sqrt(2-2cos) in fp32 -> catastrophic cancellation.
The plan's path keeps G AND the 2-2cos subtraction in fp64.
"""
import torch, torch.nn.functional as F
torch.manual_seed(0)

def ref(full, ids):
    mat = F.normalize(full[:, ids].t().contiguous().to(torch.float64), p=2, dim=1)
    d = torch.cdist(mat, mat, p=2)
    sim = 1 - d/d.max().clamp(min=1e-12); sim.fill_diagonal_(1.0); return sim.clamp(0,1)

def gram_in_dtype(full, ids, dt):
    # FULL reconstruction carried out in dtype `dt` (reviewer's all-fp32 style)
    x = full.to(dt)
    G = x.transpose(0,1) @ x
    Gs = G[ids][:, ids]
    norms = Gs.diagonal().clamp_min(0).sqrt()
    cos = Gs / (norms[:,None]*norms[None,:]).clamp_min(torch.finfo(dt).tiny)
    d = (2 - 2*cos).clamp_min(0).sqrt()      # <-- cancellation happens HERE in dtype dt
    d64 = d.to(torch.float64)
    sim = 1 - d64/d64.max().clamp_min(1e-12); sim.fill_diagonal_(1.0); return sim.clamp(0,1)

T, E = 8192, 16
base = torch.randn(E, T, dtype=torch.float64)
pairs = [(1,0,1e-6),(3,2,1e-9)]
for dst,src,eps in pairs:
    base[dst] = base[src] + eps*torch.randn(T, dtype=torch.float64)
full = base.t().contiguous().to(torch.float32)
ids = list(range(E))
R = ref(full, ids)
print("Reviewer-style ALL-FP32 reconstruction (cos+subtraction in fp32):")
G32 = gram_in_dtype(full, ids, torch.float32)
for dst,src,eps in pairs:
    print(f"  eps={eps:.0e} pair({dst},{src}): ref={float(R[dst,src]):.8f} "
          f"fp32recon={float(G32[dst,src]):.8f} err={float((R[dst,src]-G32[dst,src]).abs()):.3e}")
print("\nPlan FP64 reconstruction (G and 2-2cos in fp64):")
G64 = gram_in_dtype(full, ids, torch.float64)
for dst,src,eps in pairs:
    print(f"  eps={eps:.0e} pair({dst},{src}): ref={float(R[dst,src]):.8f} "
          f"fp64recon={float(G64[dst,src]):.8f} err={float((R[dst,src]-G64[dst,src]).abs()):.3e}")
