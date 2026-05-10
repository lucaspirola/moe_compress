# Portable runtime: vast.ai / RunPod / Lambda / any GPU host

Companion to the HF Jobs path (`hf_jobs/entrypoint_ablations.py`). Both consume the same canonical [`max_quality/requirements.txt`](../requirements.txt). Use this when:

- HF Jobs queue is stalled (current symptom: 8–12h waits for a100-large)
- you want sub-5-min allocation latency
- you want cheaper $/h on commodity providers (vast.ai DC A100 ~ $0.80/h vs HF $2.50/h)

Image is built by `.github/workflows/docker-build.yml` and pushed to:

- `ghcr.io/lucaspirola/moe-compress:latest` — mutable, convenience
- `ghcr.io/lucaspirola/moe-compress:sha-<short>` — immutable, reproducibility

---

## What the image carries

- CUDA 12.6.3 + cuDNN (devel — `causal-conv1d` and `flash-linear-attention` compile kernels at install)
- Python 3.11
- torch ≥ 2.5, < 2.11 from the cu124 wheel index
- All deps in `requirements.txt`
- `bootstrap.sh` as `ENTRYPOINT`
- `/cache` declared as `VOLUME`

What it does **not** carry: the model snapshot (~70 GB) or the project code. Both are fetched at container start into the host-mounted `/cache` volume.

---

## vast.ai operator runbook

Prerequisite: install the `vastai` CLI (`pip install vastai`) and set the API key once with `vastai set api-key <KEY>` — it persists at `~/.config/vastai/vast_api_key` and is auto-read by every subsequent invocation. Upload your SSH public key at https://cloud.vast.ai/account/.

### 1. Find an instance

Filter for an A100 80 GB DC node with reasonable bandwidth:

```bash
vastai search offers \
    'gpu_name=A100_SXM4 gpu_ram>=80 datacenter=true reliability>0.99 inet_up>=100 inet_down>=200' \
    -o dph_total
```

Pick the cheapest row meeting the filter. Note the `id` column.

### 2. Launch

```bash
vastai create instance <OFFER_ID> \
    --image ghcr.io/lucaspirola/moe-compress:latest \
    --disk 200 \
    --env '-e HF_TOKEN=hf_xxx -e ONLY=A0 -e PREFLIGHT_ONLY=0 -e UPLOAD_ON_SUCCESS=1' \
    --ssh
```

Notes:
- `--disk 200` — 200 GB ephemeral disk for the model snapshot + ablation artifacts. If you mount a persistent vast.ai volume at `/workspace/cache`, drop this to ~30 GB and reuse the snapshot across rentals.
- The image's `ENTRYPOINT` is `bootstrap.sh`, so the container starts the harness automatically — no SSH needed unless you want to tail logs.
- `--ssh` is still recommended so you can `ssh root@<host> -p <port>` and `docker logs -f <container>` if something looks wrong.

### 3. Tail logs

```bash
ssh root@<host> -p <port> 'docker logs -f $(docker ps -q | head -1)'
```

Watch for:

- `[bootstrap] Code HEAD = <sha>` — repo cloned
- `snapshot_download complete` — model resident on `/cache`
- `Trackio initialized` (per-ablation `trackio.init` call from `run_ablations.py`)
- `>>> RUN COMPLETE` — final line; safe to destroy

### 4. Verify and destroy

After `>>> RUN COMPLETE`:

```bash
# spot-check that artifacts uploaded (only if UPLOAD_ON_SUCCESS=1)
hf api repos/get pirola/moe-ablations  # should list new files under ablations/

# destroy the instance — billing stops the moment it's gone
vastai destroy instance <INSTANCE_ID>
```

If you forgot `UPLOAD_ON_SUCCESS=1`, SSH in and:

```bash
hf upload --repo-type bucket pirola/moe-ablations /workspace/cache/ablations \
    --include "_shared/**" \
    --include "*/stage6_eval.json" \
    --include "_summary.json"
```

then destroy.

---

## Environment reference

`bootstrap.sh` reads these. `HF_TOKEN` is the only required one.

| Var | Default | Purpose |
|---|---|---|
| `HF_TOKEN` | _(required)_ | HF Hub token with read+write on `pirola` namespace |
| `MODEL_REPO` | `Qwen/Qwen3.6-35B-A3B` | Model to compress |
| `NUM_SEQUENCES` | `1000` | Calibration sample count for each ablation |
| `ONLY` | _(empty)_ | Comma-separated subset of ablation IDs (e.g. `A0,A3,A7`); empty = all 12 |
| `PREFLIGHT_ONLY` | `0` | If `1`, exits after Stage 1 pre-flight (no per-ablation work) — use to validate hardware/cache cheaply |
| `CONFIG_PATH` | `configs/qwen36_35b_a3b_30pct.yaml` | Pipeline config, relative to `max_quality/` |
| `CODE_REPO_URL` | `https://github.com/lucaspirola/moe_compress.git` | Where to clone code from |
| `CODE_REF` | `main` | Git ref to checkout |
| `CACHE_MOUNT` | `/cache` | Host-mounted persistent volume mount point |
| `TRACKIO_SPACE_ID` | `pirola/trackio` | Trackio Space for per-run dashboards |
| `HF_ARTIFACTS_BUCKET` | `pirola/moe-ablations` | Bucket to receive `_shared/`, `*/stage6_eval.json`, `_summary.json` |
| `UPLOAD_ON_SUCCESS` | `0` | If `1`, push the artifact subset to `HF_ARTIFACTS_BUCKET` after a clean run |
| `DESTROY_HINT` | `vastai destroy instance $VAST_CONTAINERLABEL` | Final-line hint for the operator |

---

## Local sanity checks

These do not need a GPU — they only verify the image is well-formed:

```bash
# Build (run from repo root)
docker build -f max_quality/docker/Dockerfile \
    -t moe-compress:dev max_quality/

# bootstrap should fail loudly on missing HF_TOKEN
docker run --rm moe-compress:dev
# expected: [bootstrap] HF_TOKEN: HF_TOKEN is required ... → exit 1
```

If you have a local CUDA box, you can also check the kernels load:

```bash
docker run --rm --gpus all moe-compress:dev \
    python -c "import torch, fla, causal_conv1d; print(torch.cuda.get_device_name(0))"
```

---

## Troubleshooting

| Symptom | Likely cause | Action |
|---|---|---|
| First container start sits at "Pulling fs layer" for 5–10 min | Image is ~30 GB; vast.ai host hasn't pulled it before | Wait. Subsequent rentals on the same machine reuse the cached image. |
| `[bootstrap] FATAL: nvidia-smi not on PATH` | Container started without `--gpus all` | Misconfigured host runtime; destroy the instance and pick a different offer. |
| `huggingface-cli login` fails with 401 | `HF_TOKEN` invalid, expired, or read-only | Confirm scope at https://huggingface.co/settings/tokens — write is needed for `UPLOAD_ON_SUCCESS=1` and for Trackio dashboards. |
| `snapshot_download` HTTP 429 | HF Hub anonymous rate limit | The bootstrap is idempotent on `/cache`; relaunch with the same `CACHE_MOUNT` and the partial snapshot resumes. |
| GHCR `docker pull` fails with `denied` | Image visibility regressed to private | Flip back to public via https://github.com/users/lucaspirola/packages/container/moe-compress/settings (web UI only — GitHub's REST API has no PATCH-visibility endpoint for user-owned packages). |
| NCCL hang at model load | Multi-GPU launch on a host with broken NCCL fabric | Multi-GPU is not validated for this image — keep `num_gpus=1` in the offer filter. |

---

## Cost reference (2026-05)

| Provider | GPU | $/h | Boot-to-running |
|---|---|---|---|
| vast.ai DC | A100 80GB | ~$0.80 | < 5 min |
| vast.ai community | A100 80GB | ~$0.50 | < 5 min, lower reliability |
| HF Jobs | a100-large | $2.50 | 8–12h queue (current pathological state) |

A full 12-ablation run is ~36–60 GPU-hours on a single A100, so vast.ai DC is roughly $30–50 vs HF's $90–150 — and starts immediately.

---

## Why the image is set up this way

See [`/home/lucas/.claude/plans/plan-the-implementation-of-giggly-dusk.md`](../../../../../.claude/plans/plan-the-implementation-of-giggly-dusk.md) for the full design rationale (image base choice, registry choice, why code is cloned at runtime instead of baked in, why we don't auto-destroy, etc.). Short version: the image is a deps-only carrier so we can iterate on code without rebuilding, and the model snapshot stays on a persistent volume so it isn't re-downloaded per rental.
