# HF Jobs Operations Guide

## A — GPU Flavors and Availability

| Flavor | GPU | VRAM | CPU RAM (cgroup) | vCPU | Notes |
|--------|-----|------|------------------|------|-------|
| `a100-large` | A100 SXM4 80 GB | 80 GB | 142 GB | 12 | **Pipeline standard. Fits BF16 student alone.** |
| `a100x4` | 4× A100 SXM4 | 4×80 GB | ~500 GB | 48 | Multi-GPU or when 142 GB cgroup RAM is too tight |
| `t4-medium` | T4 | 16 GB | 32 GB | 4 | Cheap smoke testing only — never use for production stages |
| `a100-small` | A100 40 GB | 40 GB | ~80 GB | 8 | **Do not use — see known bug below** |

### Known Hardware Bug — Avoid `a100-small`

`a100-small` (40 GB A100) consistently fails with CUDA device initialization errors on the first
kernel launch. The root cause is unknown; Hugging Face support is aware. Jobs appear to schedule
successfully but fail within 30–60 s. **Always use `a100-large` (80 GB) or larger.**

Symptom: `CUDA error: initialization error` / `cudaErrorInitializationError` early in job output.

### Availability

`a100-large` queues can back up 10–20 min during peak hours (14:00–20:00 UTC weekdays). If a job
stays in `SCHEDULING` for >30 min, cancel and resubmit — the second attempt usually gets a slot
within 5 min.

### RAM Reporting Trap

`psutil.virtual_memory().total` on HF Jobs reports the **host** RAM (~1–2 TB), **not** the cgroup
limit (142 GB for `a100-large`). The cgroup OOM-killer fires at 142 GB with SIGKILL (exit code 137)
without any Python-visible exception. Monitor per-process RSS via Trackio `sys/proc_rss_gb`. If RSS
exceeds ~120 GB, the next large allocation will trigger the kill.

### Timeout

The default job timeout is **30 minutes**. HF kills the pod with SIGKILL at timeout — the same as
a cancel. Always pass `--timeout 8h` (or higher) in `submit.sh`. There is no unlimited option.

```bash
hf jobs run <image> <command> --flavor a100-large --timeout 8h
```

---

## B — Crash-Resume Pattern (Reference Implementation)

The canonical implementation is Stage 3 in `src/moe_compress/stage3_svd.py`. Every stage should
follow these rules:

### Rules

1. **One spill file per resumable unit** (layer, optimizer step) in `artifacts_dir/_stageN_partial/`.
   The partial dir is created at the top of `run()` with `mkdir(parents=True, exist_ok=True)`.

2. **Atomic writes only.** Write to `file.pt.tmp`, then `os.replace(tmp, final)`. A SIGKILL
   mid-write leaves at most a `.tmp` file; the resume path never mistakes it for a valid artifact.

3. **Skip check at loop head.**
   ```python
   if (partial_dir / f"layer_{idx}.pt").exists():
       # load and restore, then continue
       continue
   ```
   Check presence BEFORE any expensive computation on that unit.

4. **Validate `format_version`.** Every payload must include `"format_version": 1`. Load code must
   check this and raise `RuntimeError` if it doesn't match. Future format changes bump the version.

5. **Hard-fail on expected-but-missing files.** If a resume file should be present (e.g., a layer
   file listed in a manifest) but isn't, raise `RuntimeError` with the path. Silent skip would
   produce a silently degraded model.

6. **Background spill executor.** For stages with sequential per-layer compute, use
   `ThreadPoolExecutor(max_workers=1)` + `drain_done_futures` (from `utils/futures.py`) so disk I/O
   overlaps the next unit's GPU compute. Always join all futures in a `try/finally`.

7. **Cleanup on success.** `shutil.rmtree(partial_dir, ignore_errors=True)` after the final
   checkpoint is written and `save_compressed_checkpoint` succeeds. This prevents a later re-run
   with a different config from silently reusing stale spills.

8. **Stage 5 exception.** Do NOT clean up `_stage5_partial/` on success. Router KD training is
   stochastic; checkpoints are useful for debugging convergence and post-mortem analysis. The
   directory is small (≤ 2 checkpoints × a few MB each).

### Stage Coverage

| Stage | Partial dir | Unit | Spill content |
|-------|-------------|------|---------------|
| Stage 2 | `_stage2_partial/` | Per MoE layer | `merge_{idx}.json` + `layer_{idx}.pt` (cov) |
| Stage 3 | `_stage3_bcov_partial/` | Per MoE layer | `layer_{idx}.pt` (B-cov) |
| Stage 4 | `_stage4_partial/` | Per MoE layer | `layer_{idx}.pt` (FactoredExperts U/V + ranks) |
| Stage 5 | `_stage5_partial/` | Per N optimizer steps | `step_{n}.pt` (router state + optim state) |

### Stage 5 Specifics

Stage 5 saves checkpoints every `checkpoint_every_n_steps` optimizer steps (configured in YAML).
Only the two most recent checkpoints are kept to bound disk use. On resume:

1. The latest checkpoint is loaded.
2. Router parameters are restored into the student model.
3. Optimizer state is restored with `load_state_dict()`, then `_move_optimizer_state_to_device()`
   moves all optimizer tensors to the correct CUDA device (required because the checkpoint was
   saved with `map_location="cpu"`).
4. Training fast-forwards by skipping all epochs/batches already processed.

### Bucket Durability Warning

Writes to HF bucket mounts (`/mnt/cache`) are **not durable on SIGKILL or cancel**. The FUSE driver
does not flush on pod exit with code 137. Only Hub commits survive. Every stage must upload its
output to a Hub repo before the next stage starts. See `utils/hub_upload.py`.

---

## C — Operational Gotchas

### `hf jobs cancel` Grace Period

`hf jobs cancel` sends SIGTERM. The grace period is ~5 seconds. Any in-flight `torch.save` larger
than ~500 MB will not complete before the pod is killed. This is why atomic `.tmp → os.replace` and
per-layer granularity (not per-run) matter — partial writes leave only a `.tmp` file, which the
resume path ignores.

### Two-Run Trackio Duplication

If you submit two jobs with different `PRIOR_STAGE_REPO` pointing at the same artifacts, Trackio
shows two separate runs. View them independently via `trackio.init(..., name=run_name)` — each run
gets a unique name derived from the result repo.

### Stage 3 Retry Must Delete Partial Dir

If you want to change `svd_rank_ratio` between Stage 3 retries, delete
`_stage3_bcov_partial/` from the bucket first. If the partial dir exists, Stage 3 reuses the old
B-cov silently even if the config changed.

```bash
hf buckets remove <bucket>/_stage3_bcov_partial --recursive --yes
```

### Staged Hub Uploads Between Stages

Each stage must commit its output to a Hub repo before the next stage starts:

```
Stage N completes → hub_upload.py → Hub commit → next stage reads from Hub
```

Never chain stages in a single job relying on bucket persistence. A SIGKILL between stages loses
everything from the completed stage.

### psutil RAM on HF Jobs

```python
# WRONG — reports host RAM (~2 TB), not cgroup limit
import psutil
print(psutil.virtual_memory().total)  # ~2 199 023 255 552

# CORRECT — read cgroup limit directly
with open("/sys/fs/cgroup/memory/memory.limit_in_bytes") as f:
    print(int(f.read()))  # ~152 473 600 000 (142 GiB for a100-large)
```
