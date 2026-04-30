# Algorithm Reference

## 11. Durability and Crash-Resume Model

### Inter-Stage Durability

HF Jobs bucket FUSE mounts are **not durable** under SIGKILL or timeout. The durability boundary is per-stage Hub uploads:

```
<base_repo>-stage2   ← Stage 2 output + covariance sidecar
<base_repo>-stage3   ← Stage 3 output + originals sidecar
<base_repo>-stage4   ← Stage 4 output
<base_repo>-stage5   ← Final compressed model
```

Each heavy stage (2–5) uploads its checkpoint to a per-stage Hub repo immediately on completion. The bucket is treated as scratch cache only.

### Within-Stage Crash-Resume

All partial checkpoint files are written via `.tmp` → `os.replace` (atomic on POSIX). A SIGKILL mid-write leaves at most a `.tmp` file, never a truncated final file. Dangling `.tmp` files are cleaned up at stage startup.

**`--no-resume` flag:** When passed to `run_pipeline.py`, disables all within-stage resume behaviour. Each stage runs unconditionally from scratch with no partial-file I/O. Stage 1 and Stage 6 are unaffected (they have no resume files).

| Stage | Resume Mechanism | Granularity | `--no-resume` Effect |
|-------|-----------------|-------------|---------------------|
| 1 | None (stateless, ~5 min) | N/A | None — JSONs are outputs, not resume files |
| 2 | `_stage2_partial/merge_{i}.json` + `layer_{i}.pt` | Per MoE layer | Skip all partial I/O |
| 3 | `_stage3_bcov_partial/`, `_stage3_ccov_partial/` spills; `_stage3_alpha_result.json` | Per covariance phase + α search | Delete existing spills; skip α cache |
| 4 | `_stage4_partial/layer_{i}.pt` | Per MoE layer | Skip all partial I/O |
| 5 | `_stage5_partial/step_{N}.pt` (rolling window of 2) | Per optimizer step (every 100 steps) | Skip all checkpoint I/O |
| 6 | None (stateless by design) | N/A | None — teacher_eval_cache is a speedup cache, not resume |

### Resume Safety Properties

**Stage 2 critical invariant:** Covariance remapping (`_remap_covariance_for_layer`) happens BEFORE the snapshot (`_snapshot_cov_layer`), which happens BEFORE the merge JSON write (`_write_merge_json`). A layer is considered complete only when BOTH `.json` and `.pt` exist. If `.pt` exists without `.json` (orphaned by crash between snapshot and JSON write), the `.pt` is deleted and the layer is reprocessed from scratch. This prevents double-remap corruption.

**Stage 3 covariance reuse:** On re-entry, if all per-layer B-cov spill files exist in `_stage3_bcov_partial/`, Phase A (covariance collection) is skipped entirely — including the teacher model load (~70 GB, ~60s). The α search result is cached in `_stage3_alpha_result.json` and reused on re-entry (~33 min saved).

**Stage 3 originals snapshot:** `_stage3_original_weights.pt` is saved immediately after `_snapshot_originals()` returns, BEFORE the α search and Phase D factoring. Stage 4 can access originals even if Stage 3 crashes during factoring.

**Stage 4 double-widen guard:** When `_stage3_original_weights.pt` is absent but `stage4_eora/eora_ranks.json` exists, Stage 4 detects a double-widen attempt and raises `AssertionError`. This protects against in-process re-runs (notebooks, test harnesses) where `widen_rank()` would silently double-apply EoRA correction.

**Stage 5 deferred teacher load:** The teacher model is loaded lazily on the first live batch (after fast-forward completes), not before the training loop. This eliminates wasted load time on resume.

### Format Version Enforcement

Every partial checkpoint carries a `format_version` field. On resume, the version is checked before any state is restored. A mismatch raises an error with an actionable message ("delete `_stage{N}_partial/` and re-run"). This prevents silent corruption when checkpoint format changes across code versions.
