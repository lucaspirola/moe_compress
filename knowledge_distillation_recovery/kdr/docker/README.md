# kdr — vast.ai operator runbook

Self-contained guide to launching a kdr training job on a vast.ai
instance. The base image is shared with `max_quality`
(`ghcr.io/lucaspirola/moe-compress:latest`); kdr ships only `bootstrap.sh`
and this file.

## Hardware tier

ZAYA1-8B is 8.4 B total / 760 M active parameters. **Single-GPU** is the
intended configuration:

| Tier              | Mode      | Wall-clock (50 M tokens) | Wall-clock (200 step smoke) |
| ----------------- | --------- | ------------------------ | --------------------------- |
| 1× H200 (141 GB)  | `bf16`    | ~25 min                  | ~5 min                      |
| 1× H200 (141 GB)  | `da_qad`  | ~30 min                  | ~6 min                      |
| 1× A100-80GB      | `bf16`    | ~40 min                  | ~8 min                      |
| 1× A100-80GB      | `da_qad`  | ~50 min                  | ~10 min                     |

Multi-GPU ZeRO-3 is overkill for ZAYA1's footprint and adds NCCL setup
complexity; pick the highest-VRAM single-GPU offer that fits the budget.

The vast.ai **offer filter** that matches the above:

```
gpu_name=H200 OR gpu_name in ['A100_SXM4', 'A100_PCIE']
gpu_ram >= 80
verified=true
disk_space >= 200
```

200 GB disk covers: image (~30 GB), HF cache (~17 GB teacher + 17 GB
student = 34 GB), partials staging (~50 GB), final artifact (~17 GB),
slack.

## Required env vars (LLR-0032)

Set on the vast.ai instance via `-e KEY=value` when starting the container,
or via the dashboard's "Environment" pane.

| Variable        | Description                                                                   |
| --------------- | ----------------------------------------------------------------------------- |
| `HF_TOKEN`      | HF Hub token with **write** access to the partials + recovered repos.         |
| `STUDENT_REPO`  | HF Hub repo ID of the student model. ZAYA1: `Zyphra/ZAYA1-reasoning-base`.    |
| `CACHE_MOUNT`   | Absolute path on the instance for snapshots + artifacts. e.g. `/workspace`.   |
| `KDR_CONFIG`    | YAML config path. Repo-relative or absolute. e.g. `knowledge_distillation_recovery/kdr/configs/zaya1_8b_da_qad_nvfp4_int4kv.yaml`. |
| `KDR_MODE`      | `"bf16"` or `"da_qad"`. Embedded in `run_id`.                                 |

Optional:

| Variable                  | Default                                        |
| ------------------------- | ---------------------------------------------- |
| `PARTIALS_REPO_PREFIX`    | `pirola/kdr-partials` (suffix `-{run_id}`)     |
| `RECOVERED_REPO_PREFIX`   | `pirola/kdr-recovered` (suffix `-{run_id}`)    |
| `MOE_COMPRESS_GIT_URL`    | `https://github.com/lucaspirola/moe_compress.git` |
| `MOE_COMPRESS_GIT_REF`    | `main`                                         |

`bootstrap.sh` aborts (exit 2) with a clear message if any required var is
unset or empty.

## Running

```bash
# On a freshly-launched vast.ai instance with the moe_compress image:
curl -sSL https://raw.githubusercontent.com/lucaspirola/moe_compress/main/knowledge_distillation_recovery/kdr/docker/bootstrap.sh \
    | bash
```

Or, equivalently, pre-bake the script into the launch command:

```
"command": "bash /workspace/bootstrap.sh"
```

Sequence (per `bootstrap.sh`):

1. **Env-var validation** (LLR-0032). Fail fast before downloading 17 GB.
2. **Repo clone** of `moe_compress` at the configured ref into `${CACHE_MOUNT}/moe_compress`.
3. **Zyphra transformers fork install** (LLR-0035). `force-reinstall` over the
   image's stock transformers; required for `ZayaForCausalLM`.
4. **HF auth** + `snapshot_download` of the student into `${CACHE_MOUNT}/student`.
5. **`run_id` derivation** (LLR-0031). `sha256(canonical_config_dump || \x00 ||
   student_sha || \x00 || mode)[:16]`. Same inputs → same hash → can resume.
6. **Partials query** (LLR-0033). `HfApi().list_repo_files` on
   `${PARTIALS_REPO_PREFIX}-${RUN_ID}`; pick the highest-step partial whose
   `_SAVE_COMPLETE` sentinel is present; `snapshot_download` only that subdir.
7. **Trainer invocation**: `python -m kdr.cli.train --config <yaml> --student
   <cache>/student --mode <mode> --artifacts-dir <cache>/artifacts
   [--resume-from <step_dir>]`.
8. **Final upload** (LLR-0030). `kdr_${MODE}_recovered/` → recovered repo.

## Sanity checks before launching

- `huggingface-cli whoami` returns the expected user.
- The student repo is accessible: `huggingface-cli download ${STUDENT_REPO}
  config.json --local-dir /tmp` succeeds.
- The `KDR_CONFIG` YAML validates locally:
  ```bash
  python -c "
  import yaml
  from kdr.config import Config
  with open('${KDR_CONFIG}') as f:
      Config.model_validate(yaml.safe_load(f))
  print('OK')
  "
  ```

## Resume semantics

The same `(config, student_sha, mode)` triple deterministically yields the
same `run_id`. A re-launched instance against the same job:

- re-derives the run_id (same hash);
- queries the partials repo;
- finds the highest-step partial with `_SAVE_COMPLETE` present;
- snapshot-downloads it locally;
- the trainer's `--resume-from <local_path>` reads the step from the dir
  name and skips the calibration micros that were consumed before the
  crash (see `_LoopState.run`).

Partials missing the sentinel are silently skipped. They represent
mid-write crashes; loading them would corrupt the run.

## Stopping / billing

After the final upload prints its URL, the instance is idle but still
billing. Destroy it via the vast.ai dashboard or:

```bash
curl -sSf "https://console.vast.ai/api/v0/instances/${VAST_INSTANCE_ID}/" \
     -H "Authorization: Bearer ${VAST_API_KEY}" -X DELETE
```

`bootstrap.sh` prints this hint at exit.

## Troubleshooting

| Exit code | Cause                                                  | Action                                    |
| --------- | ------------------------------------------------------ | ----------------------------------------- |
| 2         | Required env var missing OR `KDR_MODE` malformed       | Fix env, restart                          |
| 3         | Zyphra transformers fork install failed                | Check network; fork moved? See LLR-0035   |
| 4         | git clone or snapshot_download failed                  | Check `HF_TOKEN` scope; check disk space  |
| 5         | Trainer crashed mid-run                                | Check stderr; if persistent, file an issue. Re-launching against the same `run_id` resumes from the latest partial |
| 6         | Final artifact uploaded but `from_pretrained` round-trip failed in a fresh Python process | Artifact is on HF Hub; inspect the load error. Common causes: missing tokenizer files in the save (kdr bug — file an issue), Zyphra transformers fork mismatch (re-pin `MOE_COMPRESS_GIT_REF`), or HF Hub propagation lag (rare; relaunch after a few minutes — the run is idempotent on the same `run_id`) |

## Out of scope

- **GGUF / MLX deployment paths.** kdr emits compressed-tensors only;
  GGUF / MLX users re-quantize from the BF16-recovered output and accept the
  recovery loss.
- **Multi-GPU ZeRO-3.** Single-GPU is sufficient for ZAYA1; multi-GPU
  configurations work in principle (the loop dispatches via `Accelerator`)
  but are not validated in Phase 7.
