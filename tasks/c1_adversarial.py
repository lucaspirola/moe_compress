"""Adversarial worst-case for the fp64 Gram path under the ACTUAL downstream
decision, with sim_gate as the SOLE cost driver (sim_expert constant) so any
sim ranking error directly flips assignments. Includes exact-duplicate experts
and a dense near-colinear cluster competing for the same centroid.
"""
import torch, numpy as np, torch.nn.functional as F
torch.manual_seed(7); np.random.seed(7)

def ref(full, ids):
    mat = F.normalize(full[:, ids].t().contiguous().to(torch.float64), p=2, dim=1)
    mat = torch.where(mat.isnan(), torch.zeros_like(mat), mat)
    d = torch.cdist(mat, mat, p=2)
    sim = 1 - d/d.max().clamp(min=1e-12); sim.fill_diagonal_(1.0); return sim.clamp(0,1)

def gram(batches, ids):
    E = batches[0].shape[1]
    G = torch.zeros(E,E,dtype=torch.float64)
    for b in batches: x=b.to(torch.float64); G += x.transpose(0,1)@x
    Gs = G[ids][:,ids]
    norms = Gs.diagonal().clamp_min(0).sqrt(); nz = norms>0; unit=nz.double()
    denom=(norms[:,None]*norms[None,:]).clamp_min(1e-300)
    cos=torch.where(nz[:,None]&nz[None,:], Gs/denom, torch.zeros_like(Gs))
    d=(unit[:,None]+unit[None,:]-2*cos).clamp_min(0).sqrt()
    sim=1-d/d.max().clamp_min(1e-12); sim.fill_diagonal_(1.0); return sim.clamp(0,1)

E,T=64,16384
base=torch.randn(E,T,dtype=torch.float64)
# dense near-colinear cluster around centroid expert 0, decreasing closeness
cluster=[0]
for k in range(1,12):
    base[k]=base[0]+(10.0**-(np.random.uniform(6,9)))*torch.randn(T,dtype=torch.float64)
    cluster.append(k)
base[12]=base[0].clone()   # EXACT duplicate (cos==1 exactly)
full=base.t().contiguous().to(torch.float32)
batches=[full[i*T//6:(i+1)*T//6] for i in range(6)]
ids=list(range(E))
R=ref(full,ids); Gm=gram(batches,ids)
print("max abs sim err (adversarial):", float((R-Gm).abs().max()))

# downstream with sim_gate as SOLE driver
centroids=[0,30,31,32,33,34,35,36,37,38,39,40]
noncent=[e for e in ids if e not in centroids]
def assign(sim):
    rows=[ids.index(e) for e in noncent]; cols=[ids.index(e) for e in centroids]
    sg=sim[np.ix_(rows,cols)].numpy().astype(np.float64)
    cost=1-(sg+0.5)/2  # sim_expert constant 0.5 -> sim_gate is the only ranker
    np.clip(cost,0,1,out=cost)
    k=min(4,len(centroids))
    cand=[frozenset(r.tolist()) for r in np.argpartition(cost,k-1,axis=1)[:,:k]]
    return cost, cand, cost.argmin(axis=1)
cr,candr,ar=assign(R); cg,candg,ag=assign(Gm)
print("cost max abs err:", np.abs(cr-cg).max())
print("top-K(4) candidate-set flips:", sum(1 for a,b in zip(candr,candg) if a!=b), "/", len(candr))
print("argmin assignment flips:", int((ar!=ag).sum()), "/", len(ar))
# also report the tightest cluster members' assignment
for e in cluster[1:]+[12]:
    ci=noncent.index(e)
    print(f"  nc={e}: ref->c{centroids[ar[ci]]} gram->c{centroids[ag[ci]]} "
          f"(min cost ref={cr[ci].min():.8f} gram={cg[ci].min():.8f})")
