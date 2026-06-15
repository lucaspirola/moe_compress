#!/usr/bin/env python3
"""Hard guardrail: the self-traces JSONL is domain-tagged AND a downsized draw
(e.g. 512) preserves the calibration domain mix.

Why this exists: the replay loader (_draw_self_traces_indices -> _distribute_counts
over the empirical per-domain weights) is DESIGNED to keep the same percentage
split at every draw size. But if the JSONL rows lack a ``domain`` field the loader
SILENTLY degenerates to a single {"unknown": 1.0} bucket and only *warns* — the 512
draw then becomes a flat draw with NO stratification. Given the M2 sidecar history
this must FAIL the run, not warn. Exit 0 + DOMAIN_MIX_OK on success.

Tests the REAL draw path (imports the loader's own functions), so a regression in
the stratification logic also trips this. Run with the venv torch, post-download:

    python box_verify_domain_mix.py [jsonl_path] [n_draw] [seed]
"""
import sys
from pathlib import Path

# Make moe_compress importable when run standalone (box sets PYTHONPATH too).
_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir():
    sys.path.insert(0, str(_SRC))

from moe_compress.utils.calibration import (  # noqa: E402
    _load_self_traces_state,
    _draw_self_traces_indices,
)

JSONL = sys.argv[1] if len(sys.argv) > 1 else "/root/work/run/self_traces_489ee0e1b17b43b0.jsonl"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 512
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 2  # Stage-3 cov uses seed_offset=+2

# Tagging quality threshold: more than this fraction of rows untagged ("unknown")
# means the mix is unreliable even if some rows carry a domain.
MAX_UNKNOWN_FRAC = 0.01

state = _load_self_traces_state(JSONL)
by_domain = state["by_domain"]
weights = state["weights"]  # empirical full-set mix (sums to 1)
total_rows = len(state["rows"])

print(f"[domain-mix] {JSONL}")
print(f"[domain-mix] rows={total_rows} domains={len(by_domain)} "
      f"full_mix={{{', '.join(f'{d}:{w:.2%}' for d, w in sorted(weights.items()))}}}")

# (1) HARD: the JSONL must be domain-tagged (not the degenerate {'unknown'} bucket,
#     and not mostly-untagged — partial tagging skews the mix just as badly).
assert set(by_domain) != {"unknown"}, (
    "self-traces JSONL is NOT domain-tagged — loader degenerated to {'unknown': 1.0}; "
    "the 512 draw would have NO stratification. Regenerate with the domain-tagged "
    "build_self_traces_calib.py."
)
assert len(by_domain) >= 2, f"expected >=2 domains, got {sorted(by_domain)}"
unknown_frac = weights.get("unknown", 0.0)
assert unknown_frac <= MAX_UNKNOWN_FRAC, (
    f"{unknown_frac:.2%} of rows are untagged ('unknown') > {MAX_UNKNOWN_FRAC:.0%} — "
    "the empirical mix is unreliable. Regenerate with full domain tags."
)

# (2) HARD: the N-draw must reproduce the full mix per domain. Largest-remainder
#     guarantees |served_count - weight*N| < 1, so per-domain served fraction is
#     within 1/N of the full weight for ANY correct draw. A larger drift means the
#     stratification broke.
selected = _draw_self_traces_indices(state, n_requested=N, seed=SEED)
assert len(selected) == N, f"draw returned {len(selected)} rows, requested {N} (domain pool shortfall?)"
domain_of = state["domain_of"]
served = {}
for i in selected:
    served[domain_of[i]] = served.get(domain_of[i], 0) + 1

tol = 1.0 / N + 1e-9
worst = 0.0
for d, w in weights.items():
    frac = served.get(d, 0) / N
    drift = abs(frac - w)
    worst = max(worst, drift)
    flag = "  <-- DRIFT" if drift > tol else ""
    print(f"[domain-mix]   {d:>12}: full={w:.2%} served@{N}={frac:.2%} (Δ={drift:.4f}){flag}")
    assert drift <= tol, (
        f"domain {d!r} drifted {drift:.4f} > tol {tol:.4f} in the {N}-draw — "
        "stratification is NOT preserving the calibration mix."
    )

print(f"[domain-mix] worst per-domain drift={worst:.4f} (tol={tol:.4f}) at N={N}, seed={SEED}")
print("DOMAIN_MIX_OK")
