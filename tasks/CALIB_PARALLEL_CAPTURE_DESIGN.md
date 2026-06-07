# Parallel (data-parallel) calibration capture — design

**Goal:** run the in-graph capture over the 8000-row corpus as **N independent
single-GPU processes** (one per GPU, e.g. 4×H200), each over a **disjoint
shard**, then **merge** the N partial sidecars into one identical to a single
full run. ~N× wall-clock speedup at the same total GPU-hours.

Branch: `feat/calib-parallel-capture` (has the `--replay-from` driver + the
cudagraph in-graph wheel patches).

## Why data-parallel (not tensor-parallel)
Every in-graph capture signal is an **associative reduction**, so partial
results from disjoint data shards combine exactly. Tensor-parallel (tp=N) shards
ONE model → per-rank/sharded accumulators + cross-rank gather + non-deterministic
logits (driver pins tp=1). Data-parallel keeps the proven tp=1 path per process.

## Disjointness guarantee (the hard requirement)
- `shard_split.py` splits `self_traces_<key>.jsonl` into N **contiguous,
  non-overlapping** files: rows `[k*ceil(8000/N):(k+1)*...]` → `shard_k.jsonl`
  in `shard_k/` dirs.
- HARD verify before any launch: (a) `sum(line_count(shard_k)) == total`,
  (b) the multiset union of shard rows == original (assert no row in two shards
  and none dropped — compare a stable per-row key, e.g. `_attempt_idx`/`seed_idx`,
  as disjoint sets covering the full range). Abort if not exactly disjoint+complete.
- Each process is given ONLY its `shard_k.jsonl` (`--replay-from shard_k.jsonl`)
  and writes sidecars under that file's own `sidecars/<stem>/` dir → no shared
  input row, no output write collision. Provable.

## Required driver fix
`_load_teacher_vllm` must force `kernel_config={"moe_backend":"triton"}` so the
model runs `TritonExperts` (the kernel-interior signals — reap, per_expert_max,
output_reservoir — only dispatch there; default auto picks FlashInfer → no
capture). Add a `--moe-backend` flag (default "triton" for calibration).

## Merge utility (`merge_sidecars.py`) — correctness-critical
Input: N per-shard sidecar dirs. Output: one merged sidecar dir at the canonical
8000-jsonl `sidecars/<stem>/` path (so a probe config with
`calibration.jsonl_path = <8000 jsonl>` resolves it). Per-signal reductions
(must be EXACTLY equal to a single full run):

- **reap_scores (probe-critical):** payload has `reap_scores` (= per-shard mean
  `score_sum/count`) + `token_counts`. Merge:
  `combined[l,e] = Σ_k(reap_k[l,e] * count_k[l,e]) / max(Σ_k count_k[l,e], 1)`,
  `counts = Σ_k count_k`. Exactly reproduces the single-run mean. Unit-tested.
- **per_expert_max:** element-wise `max` across shards. token_counts summed.
- **routing_stats:** `freq = Σ freq_k`; `mean_weight = Σ(mean_weight_k*freq_k)/max(Σfreq_k,1)`.
- **imatrix:** dense `.dat` not pipeline-consumed (Stage 6 uses llama.cpp's own) —
  sum the per-channel sq-sums if present; low priority.
- **input_cov:** off by default → usually absent; if present, sum the Gram
  accumulators (additive). 
- **block_outputs:** concatenate the per-shard `hidden_states` `[n,H]` in shard
  order → `[Σn, H]`; `n_prompts_in_subset` summed. (block-refine uses them as a
  sample set; per-row pairing preserved within each row.)
- **output_reservoir:** concatenate samples then re-cap to the configured cap
  (best-effort; deterministic — take first `cap`). Documented as approximate.
- **router_logits_stats:** no sidecar emitted (HIGH-1 fix) → N/A.
- **stage2_profile:** replay-only/live-disabled → N/A (not produced in-graph).

Merge must validate all N shards agree on `(n_layers, n_experts)` and schema
version; abort on mismatch.

## Orchestration (`run_parallel_capture.sh`)
1. `shard_split.py <8000 jsonl> N` → shard_0..N-1 (+ disjoint/complete assert).
2. For k in 0..N-1: `CUDA_VISIBLE_DEVICES=k nohup python build_self_traces_calib_vllm.py
   --replay-from shard_k/shard_k.jsonl --capture-reap-scores [+others] --moe-backend triton ...`
   (each a full tp=1 replay-capture on its shard; cudagraphs ON).
3. Wait for all N (poll per-shard completion marker / sidecar present).
4. `merge_sidecars.py shard_0/sidecars ... → <8000 jsonl>/sidecars/<stem>/`.
5. Verify merged reap_scores: load, `token_counts.sum() == Σ shard counts`,
   non-zero, `[40,256]`.

## Smoke (replay-path, the gate before the full run)
A script that loads the model (cudagraphs ON, moe_backend=triton) and runs the
ACTUAL replay path (`--replay-from` over ~8 rows of a shard, max_tokens=1) +
asserts reap `token_counts > 0` AND a traced-region signal (block_out) fires —
matching the real run, not generic generation. This is the go/no-go on the
fixed wheel before the 4-way run.

## CPU tests (no GPU)
- `shard_split`: split a synthetic 10-row jsonl into 4 → assert disjoint+complete.
- `merge_sidecars`: build synthetic per-shard reap payloads from a known
  single-run partition → assert merged == single-run (exact for reap mean +
  counts; max for pem; weighted-mean for routing).

## Process
New code (driver moe_backend fix, shard_split, merge, orchestration, smoke) →
plan/review → implement → code-review to all-none (merge correctness is the
critical review focus). Then rent 4×H200, smoke, run, merge, probe.
