# Using Hugging Face Jobs, Buckets, and Hub for the MoE compression pipeline

Operational reference for running multi-stage pipelines on HF Jobs without losing
work. Sources at the bottom.

## TL;DR

| Storage           | Durable on SIGKILL? | When data is committed                | Use for                                    |
|-------------------|---------------------|---------------------------------------|--------------------------------------------|
| Bucket (FUSE)     | **No**              | On `close()` (streaming) or async after close (advanced-writes) | In-pod scratch / intermediate caches      |
| Hub repo (commit) | **Yes**             | When `upload_*` returns HTTP 200      | Stage outputs that the next job will read  |

**Rule of thumb**: never rely on bucket-resident files surviving a job
cancellation or crash. The Hub commit is the only durability boundary.

---

## 1. The three primitives

### Hub repos — `huggingface_hub.HfApi`

Git-based, versioned, atomic at the commit level. Created via `create_repo`,
populated with `upload_folder` (small) or `upload_large_folder` (multi-GB,
resumable). Once the commit returns, the data is durable on Xet/S3 storage.

```python
from huggingface_hub import HfApi
api = HfApi()
api.create_repo("user/my-stage-output", repo_type="model", private=True, exist_ok=True)
api.upload_large_folder(
    folder_path="./artifacts/stage2_pruned",
    repo_id="user/my-stage-output",
    repo_type="model",
)
```

`upload_large_folder` writes progress to `./cache/huggingface/` so an interrupted
run resumes where it left off.

### Buckets — `hf://buckets/<namespace>/<bucket>/...`

S3-like, **non-versioned**, mutable, fast. Ideal for cross-job caches (model
snapshots, calibration tensors) and in-pod intermediate state.

CLI:
```bash
hf buckets list <user>/<bucket> -R                # recursive listing
hf buckets cp ./file hf://buckets/<user>/<bucket>/path
hf buckets sync ./local hf://buckets/<user>/<bucket>/path
```

In Jobs, mount as a volume (read-write):
```bash
hf jobs uv run script.py --volume hf://buckets/<user>/<bucket>:/mnt/cache
```

### Jobs — `hf jobs ...`

UV-script or Docker workloads on HF GPUs. Pay-per-second.

```bash
hf jobs uv run hf_jobs/entrypoint.py \
    --flavor a100-large \
    --volume hf://buckets/<user>/moe-cache:/mnt/cache \
    --secrets HF_TOKEN \
    --env "RESUME_FROM_STAGE=2" \
    --env "STOP_AFTER_STAGE=2"
```

Useful CLI:
```bash
hf jobs ps -a                       # list jobs
hf jobs inspect <id>                # status JSON
hf jobs logs <id> [--follow]        # logs
hf jobs cancel <id>                 # see "Cancellation" below
hf jobs stats                       # CPU/MEM/GPU usage of running jobs
hf jobs hardware                    # available flavors
```

---

## 2. Bucket durability — the trap that bit us

The HF docs state: *"Volume mounts in Jobs and Spaces are the same idea as
hf-mount, managed for you by the platform."* hf-mount in turn documents two
write modes:

- **Streaming (default)**: writes buffer in process memory, upload on
  `close()`. *"A crash before close means data loss."*
- **Advanced-writes**: stage to local disk first, async debounced flush after
  close (default 2 s window, up to 30 s). *"A crash before flush completes
  means data loss."*

Implication: `Path.write_bytes(...)` returning, or `model.save_pretrained(...)`
logging "wrote model.safetensors", does **not** mean the file is on Xet/S3.
It means the file is in a per-pod buffer. SIGKILL discards the buffer.

### Cancellation

`hf jobs cancel <id>` semantics are not documented (no SIGTERM grace-period
guarantee). Assume worst case: the pod is killed promptly and any in-flight
bucket writes are lost. Even artifacts written hours earlier may not be durable
if the FUSE writer hasn't flushed them.

### Timeouts (the same trap, dressed differently)

A job that hits its `--timeout` is killed exactly the same way as `hf jobs cancel`:
status flips to `ERROR` with `"message": "Job timeout"` and the pod terminates
with no flush guarantee. Bucket writes from the killed pod are lost.

**HF Jobs' default timeout is 30 minutes.** Omitting `--timeout` does NOT mean
"unlimited" — it means "use the (small) account default." Any heavy stage will
hit it. Always pass an explicit `--timeout` sized for the work plus headroom.
`hf_jobs/submit.sh` defaults to `TIMEOUT=8h` for one heavy stage; override via
`TIMEOUT=12h ./hf_jobs/submit.sh` for multi-stage or longer runs.

### Implication for multi-stage pipelines

Anything we want a future job to read **must** go to the Hub before the
producing job exits. Use the bucket only for ephemeral or recomputable state.

---

## 3. The right pattern for our compression pipeline

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ Job: Stage 2│ --> │ Job: Stage 3│ --> │ Job: Stage 4│ ...
└─────────────┘     └─────────────┘     └─────────────┘
       │                   ▲ │                  ▲
       │                   │ │                  │
       ▼                   │ ▼                  │
┌──────────────┐ download  ┌──────────────┐ download
│ Hub repo:    │←──────────│ Hub repo:    │
│ stage2 out   │           │ stage3 out   │
└──────────────┘           └──────────────┘
```

One stage per job. Each job ends naturally — `_upload_results` in
`hf_jobs/entrypoint.py` pushes the stage output (`stage{N}_*/` plus its
sidecars) to a fresh Hub repo. The next job sets `PRIOR_STAGE_REPO` to that
repo and `_restore_prior_checkpoint` downloads it into the bucket before the
pipeline starts.

### Submitting a stage

```bash
RESUME_FROM_STAGE=2 STOP_AFTER_STAGE=2 DETACH=1 ./hf_jobs/submit.sh
```

The auto-named result repo follows the pattern
`pirola/qwen3-6-35b-a3b-strategy-a-30pct-stop2-<UTC-timestamp>`. Capture it
from logs:

```bash
hf jobs logs <id> | grep "Result repo will be"
```

### Resuming the next stage

```bash
PRIOR_STAGE_REPO=pirola/qwen3-6-35b-a3b-strategy-a-30pct-stop2-20260425-1430 \
RESUME_FROM_STAGE=3 STOP_AFTER_STAGE=3 DETACH=1 ./hf_jobs/submit.sh
```

The entrypoint (`_restore_prior_checkpoint`) downloads that repo into
`/mnt/cache/artifacts/stage{N-1}_*/` and hoists `_stage*_*.pt` sidecars up to
`/mnt/cache/artifacts/`.

### What NOT to do

- ❌ Combine multiple stages in one job (`STOP_AFTER_STAGE=3` while
  `RESUME_FROM_STAGE=2`). If you cancel mid-Stage-3, the in-memory Stage 2
  checkpoint is lost and the bucket copy may not have flushed.
- ❌ Cancel a job to "restart with a config tweak". Bucket state at cancel
  time is unreliable. Wait for clean exit, fix config, resubmit.
- ❌ Treat bucket-resident files as durable across jobs. Always go through
  Hub for cross-job state.

---

## 4. Authentication, env, and cache

```bash
# Login (once per machine)
hf auth login                       # paste a token from huggingface.co/settings/tokens
hf auth whoami                      # confirm

# Pass HF_TOKEN to jobs via --secrets (preferred) or --env (avoid for tokens)
hf jobs uv run ... --secrets HF_TOKEN
```

In Jobs, point HF cache to the bucket so model snapshots survive across runs:

```python
os.environ["HF_HOME"] = "/mnt/cache/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/mnt/cache/hf_cache/hub"
```

This is what our entrypoint does. The first job downloads the 35 B model;
subsequent jobs hit the bucket cache (saving ~5 minutes per cold start).

---

## 5. Useful ad-hoc commands

```bash
# Inspect a partial Hub upload
python -c "from huggingface_hub import HfApi; \
  print('\n'.join(sorted(HfApi().list_repo_files('pirola/<repo>', repo_type='model'))))"

# Check what's actually on the bucket (vs what the job claimed to write)
hf buckets list pirola/moe-cache/artifacts -R

# Tail a running job's logs
hf jobs logs <id> --follow

# Filter past jobs
hf jobs ps -a --filter status=error
hf jobs ps -a --filter "command=*entrypoint.py"
```

---

## 6. References

- [Storage Buckets](https://huggingface.co/docs/hub/storage-buckets)
- [Access Patterns (volume mounts in Jobs)](https://huggingface.co/docs/hub/en/storage-buckets-access)
- [hf-mount README](https://github.com/huggingface/hf-mount) — durability semantics
- [Manage Jobs](https://huggingface.co/docs/hub/en/jobs-manage)
- [Upload files to the Hub](https://huggingface.co/docs/huggingface_hub/en/guides/upload) — `upload_folder`, `upload_large_folder`, `CommitScheduler`
- [hf CLI](https://huggingface.co/docs/huggingface_hub/guides/cli)
