# Ablation Run — Handoff Document

**Last updated**: 2026-05-09 by the agent that built this. Read this *first* if you're picking up the Stage 2 v2 §8 ablation matrix run mid-flight.

---

## 1. Goal

Run the Stage 2 v2 ablation matrix (A0..A11) defined in [`stage2_assignment_revision.md` §8](./stage2_assignment_revision.md). For each row, produce `stage6_eval.json` with WikiText-2 PPL + lm-eval (ARC, HellaSwag) + HumanEval pass@1 + MATH-500 pass@1, then compare A_i vs A0 to validate Stage 2 v2 design choices.

**Pipeline shape per ablation** (user-mandated, deviates from production):
```
Stage 1 (shared once across all 12)  →  Stage 2  →  Stage 2.5  →  Stage 6
```
Stages 3, 4, 5 are skipped. Stage 2 alone hits the 35% reduction target via `target.expert_svd_ratio: 100.0`. Stage 6 reads `stage5_final/` which is a symlink to `stage2p5_final/` (bridge inserted by `_bridge_stage25_to_stage6` in `run_ablations.py`).

**Acceptance is NOT enforced** by the harness. Per user directive: collect all 12 metrics, surface them, let the user decide which configs to ship as defaults.

---

## 2. Current Operational State

### Active job

| Field | Value |
|---|---|
| **Latest job ID** | `69fe7a28aff1cd33e8f3178f` (SECOND resubmit) |
| **Flavor** | `a100-large` ($2.50/h) |
| **Timeout** | 48h |
| **Bucket** | `pirola/moe-ablations` mounted at `/mnt/cache` |
| **Status as of write** | SCHEDULING (queued, ~1h28m) |
| **Trackio** | https://huggingface.co/spaces/pirola/trackio (project `moe-compress-strategy-a`) |
| **Job page** | `https://huggingface.co/jobs/pirola/<JOB_ID>` |

### Job history (queue stalls have been the entire blocker)

| Job ID | Outcome | Notes |
|---|---|---|
| `69fd1f75317220dbbd1a6321` | RAN on H200 — canceled | Found Phase D CKA was 6.7h CPU-bound; we killed it to GPU-vectorize. |
| `69fd5691317220dbbd1a6477` | A100, canceled @ 8h35m queue | Never picked up. |
| `69fdd28e317220dbbd1a68bd` | A100, canceled @ 11h48m queue | Never picked up. |
| `69fe7a28aff1cd33e8f3178f` | A100, current | If it stalls > ~12h, cancel + resubmit. |

**Pattern**: A100 queue has been pathological for ~24h. User wants to keep on A100 ($2.50/h) rather than escalate to H200 ($5/h). User's idea of "submit 12 jobs in parallel and cancel the rest as soon as one starts" was rejected due to bucket-collision risk + ToS concerns (see `feedback_double_loop_protocol.md` analysis if discussed).

### Cancel + resubmit pattern (proven)

```bash
# Cancel
hf jobs cancel <JOB_ID>

# Resubmit (same flavor, same code from HF dataset)
cd max_quality && DETACH=1 ./hf_jobs/submit_ablations.sh
# Returns "Job started with ID: <NEW_ID>" — track that.
```

The harness is **resume-safe** across cancels: each ablation skips if its `stage6_eval.json` exists in the bucket. Stage 1 outputs in `_shared/` survive too, so resubmits skip pre-flight if it ran.

---

## 3. Code Sync Workflow (CRITICAL)

The HF Jobs entrypoint downloads code from the `pirola/moe-compress` HF dataset **at runtime** (after job allocation). This means:

1. **Local edits** must be committed to GitHub AND synced to the HF dataset before the job starts running.
2. **Sync command**:
   ```bash
   cd /home/lucas/ai/moe_compress
   hf upload --repo-type dataset pirola/moe-compress . \
     --include "max_quality/src/**" \
     --include "max_quality/configs/**" \
     --include "max_quality/hf_jobs/**" \
     --include "max_quality/tests/**"
   ```
3. **Hot-patch property**: while a job is in SCHEDULING, you can push code changes that will be picked up when it transitions to RUNNING (entrypoint downloads after allocation, not at submit time). Useful but not a designed contract.

The entrypoint script itself (`hf_jobs/entrypoint_ablations.py`) is uploaded to a separate `pirola/jobs-artifacts` bucket at submission time and is **frozen**. Changes to the entrypoint require a fresh `submit_ablations.sh` invocation.

---

## 4. Bucket / Trackio State

### `pirola/moe-ablations` bucket layout (mounted at `/mnt/cache`)

```
/mnt/cache/
├── ablations/
│   ├── _shared/                      # Stage 1 outputs, teacher cache, shared across A0..A11
│   │   ├── stage1_blacklist.json
│   │   ├── stage1_budgets.json
│   │   ├── teacher_eval_cache.json   # filled by A0, hit by A1..A11
│   │   └── _calibration_cache/
│   ├── A0/                            # per-ablation; isolated artifacts
│   │   ├── stage1_*.json              # hardlinked from _shared/
│   │   ├── stage2_pruned/
│   │   ├── stage2p5_final/
│   │   ├── stage5_final → stage2p5_final  # symlink bridge to Stage 6
│   │   └── stage6_eval.json
│   ├── A1/ ... A11/                   # same shape
│   └── _summary.json                  # post-loop aggregation
├── code/                              # downloaded from pirola/moe-compress; recreated each run
└── hf_cache/                          # model snapshot (~70GB Qwen3.6-35B-A3B); KEEP across runs
```

### Bucket cleanup operations

```bash
# Wipe per-run artifacts (keep model snapshot)
hf buckets rm hf://buckets/pirola/moe-ablations/ablations -R -y
hf buckets rm hf://buckets/pirola/moe-ablations/code -R -y
hf buckets rm hf://buckets/pirola/moe-ablations/hf_cache/trackio -R -y
hf buckets rm hf://buckets/pirola/moe-ablations/hf_cache/xet -R -y

# Wipe Trackio DB (separate bucket)
hf buckets rm hf://buckets/pirola/trackio-bucket/trackio -R -y
```

The model snapshot in `hf_cache/hub/models--Qwen--Qwen3.6-35B-A3B/` saves ~30 min download per run — keep it across submissions.

---

## 5. Optimizations Completed (Recent Commits)

| Commit | Change | Wall-clock impact |
|---|---|---|
| `aa95ce4` | Pre-existing test fixes + buffer-persistence bug | none (correctness) |
| `9edd56a` | Disable imatrix in ablation harness | -10-30 min/ablation = -2-6h total |
| `68ea836` | GPU-vectorize Phase B reservoir + Phase D CKA + Stage 2 cost matrix; remove AA-SVD Path 2 | Stage 1: 8.5h → ~75 min on H200. Stage 2: -10 min/ablation. AA-SVD: correctness fix. |
| `3e518b0` | calibration set/dict TypeError | bug fix |
| `47ecc7b` | solver kwarg + preflight trackio init | bug fix |
| `6ed9282` | Initial harness | feature |

**Key file pointers**:
- `src/moe_compress/stage1_grape.py:556` — `_cka_distance_matrix` GPU path with auto CPU per-pair fallback when reservoir under-fills
- `src/moe_compress/utils/activation_hooks.py:508` — `ExpertOutputAccumulator` GPU-resident reservoir (Phase B)
- `src/moe_compress/stage2_reap_ream.py:2607` — `_permutation_align_to_centroid` (cdist on GPU now; single CPU sync at Hungarian)
- `src/moe_compress/stage3_svd.py:1482` — `_precompute_eigh` (Path 2 removed; A is `del`-ed before branching)
- `src/moe_compress/run_ablations.py:87` — `_build_ablation_config` (caps Stage 2 calibration + disables imatrix)

### Optimizations deferred (NOT a bottleneck after measurement)

- **Stage 1 Phase E GRAPE solver**: pure CPU numpy, ~180 ms total — irrelevant. Don't bother.
- **Stage 2 EM refinement loop**: gets cdist GPU benefit transitively from cost matrix fix. No additional work needed.

---

## 6. Wall-Clock Projection (after all optimizations)

On A100 (~1.6× slower than H200 for fp32 GEMM):

| Stage | Per-ablation | Notes |
|---|---|---|
| Stage 1 pre-flight | ~120 min (one-time) | 4000 calibration samples. Phase A/B forward dominant. |
| Stage 2 + 2.5 + 6 per ablation | ~5-7h | Stage 2 layer-by-layer dominant; Stage 6 includes batched generate(). |
| **Full A0..A11** | **~70-85h** | Will need ~2 job submissions at 48h timeout (resume-safe). |

If queue keeps stalling: a single 48h H200 job would complete the matrix in ~50h, $250 vs ~$200 on A100 with stalls. Same order.

---

## 7. Monitoring Pattern (what I've been doing)

The user invokes `/loop` periodically. The loop self-paces with `ScheduleWakeup` at ~30-min cadence. Each iteration:

```bash
# Status
hf jobs inspect <JOB_ID> | python3 -c "import json,sys; d=json.load(sys.stdin); print(d[0]['status']['stage'])"

# Logs (when available)
hf jobs logs <JOB_ID> | tail -80

# Watch for these progression markers (in order):
#   "Trackio initialized"             ← entrypoint started
#   "Stage 1 Phase A: MA-formation"   ← phase A started
#   "calibration forward N/256"       ← phase A/B forward progress (every 64 batches)
#   "Stage 1 Phase B: profiling"      ← phase B started
#   "Stage 1 Phase C: blacklisted"    ← phase C done
#   "Stage 1 Phase D: computing CKA"  ← phase D started
#   "CKA matrix: layer K/40"          ← phase D progress per-layer (~1 sec/layer with GPU fix)
#   "Stage 1 Phase E: ..."            ← GRAPE solver
#   "Pre-flight Stage 1 complete"
#   "[A0] starting (deltas={})"       ← first ablation begins
#   "Stage 2 layer ..."               ← Stage 2 progress
#   "stage6_eval.json"                ← A0 done; report numbers
```

**Decision rules**:
- SCHEDULING < 6h → wait, 30-min cadence.
- SCHEDULING > 6-8h → consider cancel + resubmit.
- SCHEDULING > 12h → user's threshold for cancel (recent: 11h48m).
- RUNNING + log progress → 30-min cadence, watch for stage transitions.
- RUNNING + log silence > 1h → grep for "GRAPE: converged" or "Phase X complete"; if true silence on RUNNING with no progress, investigate (rare).
- Failure traceback → diagnose root cause, push fix to GitHub + HF dataset, resubmit (job is in `_shared/` so Stage 1 won't repeat).

---

## 8. Quick-Start for a Fresh Agent

```bash
# 1. Check current job (replace with latest ID from this doc or `hf jobs ps`)
hf jobs inspect 69fe7a28aff1cd33e8f3178f

# 2. If SCHEDULING and < 12h: wait. If > 12h: cancel + resubmit (see §2).

# 3. If RUNNING: tail logs and look for progression markers (§7).

# 4. If a Phase X completes successfully but later one fails:
#    - The harness is resume-safe; just push the fix and resubmit.
#    - Don't manually clean buckets unless you want a true clean state.

# 5. If A0 completes (stage6_eval.json visible at bucket /ablations/A0/stage6_eval.json):
#    - Report the metrics
#    - Continue monitoring A1..A11

# 6. If all 12 done: aggregate _summary.json and surface to user.
```

**Things NOT to do without asking**:
- Don't escalate to H200 (more expensive; user has explicitly chosen A100).
- Don't change the calibration sizes — they were deliberated.
- Don't run more than 1 job at a time — bucket collision risk (the proposed 12-parallel queue-racing was rejected).
- Don't enable imatrix in the ablation harness (we just turned it off; saves 2-6h).
- Don't add Stage 1 Phase E GPU port — measured at ~180 ms total, not worth it.

---

## 9. External GPU Runbook (vast.ai / RunPod / Lambda)

HF Jobs queue stalls drove a parallel deployment path: the same harness, packaged as a Docker image, runs on any commodity GPU host. Use this when the HF Jobs queue is pathological (current state) or when you want sub-5-min allocation latency.

- **Image**: `ghcr.io/lucaspirola/moe-compress:latest` (also `:sha-<short>` for reproducibility).
- **Build**: `.github/workflows/docker-build.yml` runs on push to main and rebuilds whenever `requirements.txt`, `docker/**`, or the workflow itself changes.
- **Runbook**: [`max_quality/docker/README.md`](../docker/README.md) — vast.ai filter command, full `docker run` invocation, env-var reference, local sanity checks.

The image is a **deps-only carrier**: code is `git clone`d at container start by `docker/bootstrap.sh`, and the model snapshot lives on a host-mounted `/cache` volume (mount as `-v /workspace/cache:/cache` on vast.ai). This way you can iterate on code without rebuilding the image, and a 70 GB snapshot survives across rentals.

Both the HF Jobs path and the Docker path consume the same canonical [`max_quality/requirements.txt`](../requirements.txt) — adding or upgrading a dep there propagates to both.

---

## 10. Reference Docs (read these for technical depth)

- [`stage2_assignment_revision.md`](./stage2_assignment_revision.md) — Stage 2 v2 spec including §8 ablation matrix definition (A0..A11 deltas, expected outcomes)
- [`hf_jobs_operations.md`](./hf_jobs_operations.md) — HF Jobs ops reference
- [`huggingface_jobs_and_buckets.md`](./huggingface_jobs_and_buckets.md) — HF Jobs/Buckets API reference
- [`pipeline_status_2026-04-27.md`](./pipeline_status_2026-04-27.md) — earlier pipeline status snapshot
- [`stage_memory_profiles.md`](./stage_memory_profiles.md) — per-stage memory budgets

The full git log (`git log --oneline | head -20`) is the source of truth for *why* each optimization was applied.
