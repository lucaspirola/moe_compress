# Strategy A — Maximum Quality MoE Compression for Qwen3.6-35B-A3B @ 30%

Implementation of Strategy A from
[`pirola/moe-compression-workflow/VALIDATED_STRATEGIES.md`](https://huggingface.co/pirola/moe-compression-workflow/blob/main/VALIDATED_STRATEGIES.md),
targeting **Qwen/Qwen3.6-35B-A3B** with **30% total parameter reduction** (expert
pruning + non-uniform SVD compounding).

Plan file: `~/.claude/plans/using-https-huggingface-co-pirola-moe-co-mutable-galaxy.md`

## Pipeline

| Stage | Module | What it does | A100 cost |
|-------|--------|--------------|-----------|
| 0 | `stage0_super_experts` | `down_proj` max-activation → blacklist | ~2 min |
| 1 | `stage1_grape` | per-layer redundancy → non-uniform expert budgets | ~5 min |
| 2 | `stage2_reap_ream` | REAP scoring + REAM merge (sequential) | ~1.5 h |
| 3 | `stage3_svd` | D-Rank + Swift-SVD+ + AA-SVD + block refine | ~45 min |
| 4 | `stage4_eora` | training-free low-rank compensation | ~5 min |
| 5 | `stage5_router_kd` | router-only KL distillation | ~20 min |
| 6 | `stage6_validate` | WikiText-2 PPL + zero-shot + gen | ~10 min |

## First step for any new model: generate calibration self-traces

This tool is model-agnostic — the test case here is Qwen3.6-35B-A3B but the
pipeline works against any HF MoE teacher. **Before the first run for a new
teacher**, you must generate a calibration trace JSONL by running the teacher
once in reasoning mode over a prompt set. The pipeline fails loudly at Stage 1
if the trace JSONL is missing and prints the exact command for your model.

Why: every stage that consumes calibration (Stage 1 GRAPE profiling, Stage 2
REAP+REAM merge cost, Stage 2 SH per-layer heal, Stage 2.5 router-KD) sees
THE SAME `calibration.source`. Generic chat-templated SFT corpora (Tülu-3,
FineWeb, OpenMath, ...) get you the outer chat-template wrapping (Gemma 3 QAT
rule — necessary for the model to keep speaking chat post-compression), but
they do NOT contain the teacher's actual `<think>...</think>` reasoning
traces. For reasoning-mode models the most quality-critical token positions
are *inside* that block — and without self-traces, the routers (Stage 2.5)
and merged expert weights (Stage 2 SH heal) are never supervised there.

```bash
# One-shot pre-step. ~5-6h on a single H200 with a BF16 teacher; deterministic
# under (teacher_repo, revision, prompts, max_new_tokens) so the JSONL is
# reproducible and the cache invalidates automatically. Cache-key suffix
# is folded into the output filename → multiple teachers coexist on disk.
#
# --num-prompts is the GENERATION budget. The script tags each row with
# `_complete: bool` (true iff </think> + EOS landed in-trace); the downstream
# loader filters `_complete=false`. For Qwen3-thinking with hard math in
# the mix, expect ~70-80% completeness → oversize num-prompts by ~1.3× the
# target complete-row count.
python max_quality/scripts/build_self_traces_calib.py \
    --teacher <YOUR_TEACHER_REPO> \
    --num-prompts 6500 --max-new-tokens 16384 \
    --output artifacts/_shared/self_traces.jsonl

# Then point the run config at the source:
#   calibration:
#     source: self-traces
#     jsonl_path: artifacts/_shared/self_traces_<hash16>.jsonl   # printed by the script
```

Re-run the script only when the teacher revision or the prompt set changes.
See the script's `--help` for full options.

## Quick start (A100 80 GB)

```bash
source /home/lucas/ai/venv/bin/activate
pip install -r requirements.txt
python -m moe_compress.run_pipeline \
    --config configs/qwen36_35b_a3b_30pct.yaml \
    --model Qwen/Qwen3.6-35B-A3B \
    --artifacts-dir ./artifacts \
    --target-ratio 0.30
```

Resume from a specific stage (e.g. after an OOM mid-Stage-3):

```bash
python -m moe_compress.run_pipeline --config ... --resume-from-stage 3
```

## Local smoke test (RTX 5080, 16 GB)

```bash
pytest tests/ -v                         # synthetic MoE unit tests
pytest tests/test_smoke_qwen3_0_5b.py    # end-to-end on a small MoE model
```

## Run on Hugging Face Jobs (recommended)

HF Jobs provisions an A100 on demand, runs the pipeline, then releases the GPU
automatically when the script exits — no idle billing. Persistent state lives
in a private HF bucket mounted at `/mnt/cache`.

**One-time setup (done in commit 12e1fa0):**
- Bucket `pirola/moe-cache` (holds HF snapshot cache + pipeline artifacts)
- Dataset repo `pirola/moe-compress-code` (pipeline source, fetched by the job
  on start)

**Submit a run:**

```bash
./hf_jobs/submit.sh                      # default: a100-large, 30% target
TARGET_RATIO=0.25 ./hf_jobs/submit.sh    # lighter compression
FLAVOR=h200 ./hf_jobs/submit.sh          # faster, costs 2× more
DETACH=1 ./hf_jobs/submit.sh             # return immediately; follow logs
                                         # via `hf jobs logs $JOB_ID -f`
```

Cost at $2.50/h (a100-large) × ~2.75 h ≈ **$7 per run**, plus ~$2/month for
the bucket. The GPU is released on any script exit (success, pipeline error,
or `SIGTERM`), so there is no idle tail.

**Before the first real run**, do a 10-minute CPU dry-run to confirm auth,
mounts, and code delivery:

```bash
hf jobs uv run hf_jobs/dry_run.py \
    --flavor cpu-basic --timeout 10m \
    --volume hf://buckets/pirola/moe-cache:/mnt/cache \
    --secrets HF_TOKEN \
    --env CODE_REPO=pirola/moe-compress-code \
    --env CACHE_MOUNT=/mnt/cache
```

**Results land at** `pirola/qwen3-6-35b-a3b-strategy-a-<pct>pct-<utc-timestamp>`
(private model repo, auto-created). The final `stage5_final/` safetensors are
uploaded as the main model; `stage*.json` per-stage artifacts land under
`artifacts/`. Set `RESULT_REPO` to override the destination.

**Re-runs are cheap**: the bucket retains the model snapshot (~70 GB) and the
calibration token cache, so subsequent jobs skip those downloads.

**Durability**: each heavy stage (2–5) uploads its checkpoint to a per-stage
Hub repo (`<result_repo>-stage{N}`) immediately on completion, so a job that
crashes or is cancelled later still preserves every completed stage. Resume
the next stage with `PRIOR_STAGE_REPO=<that-stage-repo>`. Bucket writes are
**not** durable on `hf jobs cancel` — see
[`docs/huggingface_jobs_and_buckets.md`](docs/huggingface_jobs_and_buckets.md)
for the full operating model.

## Protected components (never touched by pruning/SVD)

- Shared expert at every MoE layer
- Attention weights (DeltaNet and full-attention projections)
- Embeddings, `lm_head`, layer norms
- Router weights (except Stage 5, which updates *only* these)
- Super experts on the Stage 0 blacklist

## Risk register

| Risk | Level | Mitigation |
|------|-------|------------|
| REAM + variable-N′_l interaction | med-high | Per-layer MSE monitor in Stage 2; bump budget 10% on outlier |
| AA-SVD on MoE weights | medium | SVD limited to expert matrices; block-level L-BFGS refine |
| Full multi-stage pipeline untested | high | Per-stage checkpointing; Stage 6 hard gate on quality metrics |
| DeltaNet hybrid attention | unknown | All attention weights frozen across all stages |

## Success criteria (Stage 6 vs uncompressed)

- WikiText-2 PPL: ≤ +3% relative
- ARC-C / HellaSwag: ≤ 1.5 pp absolute drop
- HumanEval / MATH-500: ≤ 3 pp absolute drop
- Actual param reduction: ≥ 30.0%
