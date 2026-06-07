# Cudagraph-fast calibration capture — design

**Status:** design (2026-06-07). User decision: rearchitect capture to run fast
(cudagraphs on) so the full 8k corpus is cheap. Supersedes the eager-only
`_calib_buf_rows` workaround.

## Root cause (GPU-confirmed)

Capture accumulation runs in a **Python callback** (`_ch.dispatch` → registered
callback). Python cannot execute inside a CUDA-graph replay, so every signal
yields zero under cudagraphs (the production default). Also `_calib_buf_rows =
max_cudagraph_capture_size` is 0 under `enforce_eager` → 0-row side-store. Both
confirmed at runtime; the eager+buf_rows poke produced 624 reap tok-acts/layer.

## Principle

CUDA-graph capture records CUDA **ops**, not Python. So the fix is: each capture
signal accumulates via **pure GPU tensor ops** (`scatter_add_` /
`scatter_reduce_` / matmul-accumulate) into **persistent** per-(layer,expert)
accumulator tensors, issued from the callback so they're recorded into the
captured graph and re-execute every replay. No `.cpu()`, `.item()`, RNG, or
data-dependent Python control flow on the capture path. This works identically
under eager and under cudagraph → durable + fast.

## Mechanism (reap_scores, the probe's signal)

In `TritonExperts.apply`, after the Triton kernel fills `_unweighted_slice`
(`[n_tok, top_k, hidden]`, the per-token unweighted expert outputs — the
persistent `_calib_unweighted_buf` side-store is RETAINED; it's the data source):

```
norms   = _unweighted_slice.reshape(-1, H).float().norm(dim=-1)      # [n_tok*top_k]
contrib = norms * topk_weights.reshape(-1).float()                   # gate-weighted
ids     = topk_ids.reshape(-1)                                       # [n_tok*top_k]
score_accum[rank].scatter_add_(0, ids, contrib)                     # persistent [L,E] fp32
count_accum[rank].scatter_add_(0, ids, ones_like(contrib))         # persistent [L,E] i64
```

- `rank` (layer index) is a Python int at **capture** time → resolves to a fixed
  accumulator slice frozen into the graph; each of the 40 layers records its own
  scatter into its own slice. `topk_ids`/`topk_weights` are live tensors
  recomputed each replay → correct indices/values per replay.
- `topk_weights` must be available at the dispatch point; stash per-layer into a
  persistent tensor (no Python dict).
- Final read of accumulators (one Python read, outside any graph) → reap S_j =
  score_accum / (optionally count) per upstream Eq.9.

`per_expert_max`: `scatter_reduce_(amax)` of `|f|_∞`. `routing_stats`:
`scatter_add_` of freq + gate-weight (same `router` payload).

## Correctness handling

- **Per-layer accumulator selection** frozen at capture (layer_rank constant per
  call-site) — verified design; the capture pass runs all 40 layers once with
  correct rank each, recording 40 distinct scatters.
- **Bucket/variable num_tokens:** scatter over flat `n_tok*top_k` indices handles
  any batch size; each cudagraph bucket records its own scatter. Works for decode
  buckets AND chunked-prefill.
- **buf_rows fix:** size the side-store from `max(max_cudagraph_capture_size,
  max_num_batched_tokens)` (or a floor) so it's never 0 — covers eager + prefill
  chunks. (Belt-and-suspenders even though in-graph accumulation is primary.)
- **No host sync on the capture path** — accumulators stay on GPU; read once at
  dump.

## Scope (FINAL — user decision 2026-06-07: EVERYTHING in-graph)

ALL ten signals get the in-graph GPU-accumulation treatment now:
- **Scatter/reduce signals:** `reap_scores`, `per_expert_max`, `routing_stats`,
  `router_logits_stats`, `imatrix` (dense + MoE), `wanda` — persistent
  per-(layer,expert) accumulators via `scatter_add_`/`scatter_reduce_`.
- **Reservoir/block signals** (`output_reservoir`, `block_outputs`,
  stage2_profile reservoir component): replace RNG reservoir sampling with
  **deterministic fixed-stride** selection into a persistent capped buffer
  (graph-safe, no RNG, no data-dependent shapes; keep every ⌈N/cap⌉-th token).
- **`input_cov` (+ stage2_profile covariance component): in-graph GPU
  accumulation, but FLAG-GATED OFF by default** (its ~172 GB all-resident
  footprint OOMs a single H200). Two run modes when enabled:
  - **(a) all-resident** — full cudagraph-fast, requires ≥172 GB total VRAM
    (e.g. 2×H200, TP/offload split). Default for multi-GPU.
  - **(b) per-layer CPU-offload (1×H200)** — accumulate one layer's per-expert
    Gram on GPU, copy to pinned CPU after each layer, reuse the GPU slice for the
    next layer. Runs the per-layer accumulation as GPU ops; orchestration is
    per-layer (not one whole-model graph) so the resident footprint is one
    layer (~4.3 GB) not all 40. Slower but fits 1×H200.
  The default (flag off) means normal probe/calibration runs never allocate the
  172 GB and never OOM; input_cov is opt-in for when Stage 3/4 actually need it.

## Verify gates (HF GPU, before trusting)

1. With cudagraphs ON (default, NOT enforce_eager), reap `token_counts.sum() > 0`
   and ≈ `n_tokens × top_k` over a known prompt (content, not file size).
2. Capture is non-zero over BOTH a decode-heavy run and a prefill-heavy
   (forward-only) run.
3. Numerical sanity: reap scores rank experts sensibly (not uniform/NaN).

## Pipeline

Patch (this design) → plan→review→impl→review (all-none, separate agents) → HF
CPU wheel rebuild (single-arch sm_90a fast path first; multi-arch for canonical)
→ HF GPU verify (gates above) → DataCrunch full 8k capture (cudagraph-fast) →
6-model probe.

## Out of scope

No change to generation, the v2 mix, or plugin/stage consumers (they read the
same sidecar files). v3 forward-only replay (feat/calib-v3-replay) remains valid
as a *capture driver* but is orthogonal to this fix.
