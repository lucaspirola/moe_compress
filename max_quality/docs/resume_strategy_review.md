# Resume Strategy Review — All 6 Pipeline Stages

**Date:** 2026-04-30
**Hardware target:** HF H200 — 256 GB Host RAM, 141 GB VRAM
**Reviewer:** ML Intern automated audit

---

## Summary of All Findings

### Critical Bugs Found (must fix before next production run)

| # | Stage | Bug | Impact | Fix |
|---|-------|-----|--------|-----|
| 1 | Cross-cutting | `save_json_artifact()` writes directly to final path via `path.open("w")` — **not atomic**. A crash mid-write produces a truncated, invalid JSON file. | Stage 1→2 handoff corruption; any `save_json_artifact` call in any stage | `.tmp` + `os.replace()` pattern. Also, `budget_decomposition.json` uses raw `.write_text()` — route through `save_json_artifact` |
| 2 | 2 | Orphaned `.pt` without matching `.json` in `_stage2_partial/`. If crash occurs after `_snapshot_cov_layer` but before `_write_merge_json`, the `.pt` has post-remap covariance but no `.json` → on resume, layer is re-processed, `_remap_covariance_for_layer` runs again on already-remapped data → **double-remap → silent numerical corruption** | Corrupted covariance fed to Stage 3 → degraded SVD factoring for all downstream layers | In resume scan, if `layer_{i}.pt` exists without `merge_{i}.json`, delete the orphaned `.pt` and log a warning before reprocessing |
| 3 | 4 | `widen_rank()` is not idempotent. In-process re-run (e.g. notebook, or crash between `widen_rank` and `_spill_layer` if Stages 3→4 share a process) would double-apply EoRA correction → **doubled rank, noise-amplified factors** | Silently corrupted FactoredExperts for affected layers | Assert `fe.ranks[name] == stage3_rank` before every `widen_rank()` call for non-resumed layers; snapshot Stage 3 ranks before the loop |

### High-Severity Gaps (significant time/correctness impact)

| # | Stage | Gap | Impact | Fix |
|---|-------|-----|--------|-----|
| 4 | Cross-cutting | No `--no-resume` CLI flag to disable all resume/checkpoint behavior | Cannot guarantee clean-slate runs; no way to disable intermediate file writes for CI/debugging | Add `--no-resume` to `run_pipeline.py`, thread `no_resume: bool` to each stage's `run()` |
| 5 | Cross-cutting | No pre-flight validation of Stage 1 artifacts when `--resume-from-stage >= 2` | Cryptic `KeyError`/`FileNotFoundError` deep inside Stage 2 code instead of clear startup error | `_validate_stage1_artifacts()` before entering Stage 2 |
| 6 | 3 | `_collect_covariances()` always re-runs (~90 min + 70 GB teacher load), even when all spill files already exist from a prior interrupted run | Wasted ~90 min on Stage 3 re-runs after Phase D crash | Check spill completeness before entering Phase A; skip teacher load if all spills exist |
| 7 | 3 | `_stage3_original_weights.pt` saved AFTER Phase D (factoring loop). If Stage 3 crashes during factoring, Stage 4 has no originals snapshot | Stage 4 cannot run after Stage 3 crash; must re-run entire Stage 3 | Move `torch.save(originals, ...)` to immediately after `_snapshot_originals()`, before α search |
| 8 | 3 | α search result (~33 min, 11 candidates) not persisted | α search re-runs on every Stage 3 re-entry | Save best α to `_stage3_alpha_result.json`; reload on re-run |
| 9 | 5 | Teacher model loaded before fast-forward loop. On resume, fast-forward iterates without using teacher → wasted ~60s load time | Unnecessary teacher load on resume | Deferred/lazy teacher loading: load only on first live batch |

### Medium/Low Fixes

| # | Stage | Fix | Priority |
|---|-------|-----|----------|
| 10 | 2, 3, 4, 5 | Clean dangling `.tmp` files at startup (glob `*.tmp`, delete) | Medium |
| 11 | 4 | Restore `effective_ranks` from spill payload on resume (currently may only restore `ranks`) | Medium |
| 12 | 5 | Save `gradient_accumulation` in checkpoint; assert match on resume | Medium |
| 13 | 6 | Normalize `revision` in teacher cache key (`or "main"` for `None`) | Medium |
| 14 | 6 | Background GGUF thread: atomic write (`.tmp` + rename) | Medium |
| 15 | Cross-cutting | Post-Stage-1 read-back validation of all 3 JSON files | Low |

---

## Per-Stage Detailed Plans

### Stage 1 — Super Expert Detection + GRAPE

**Current resume model:** None (stateless, JSON-only output, ~5 min on H200). `--resume-from-stage 2+` skips Stage 1. Re-running is always safe.

**`--no-resume` effect:** None. Stage 1's JSON files are final outputs (not intermediate resume files). They must always be written for Stage 2 to proceed.

**Changes needed:**
1. Make `save_json_artifact()` atomic (`.tmp` + `os.replace`) — affects all stages
2. Route `budget_decomposition.json` through `save_json_artifact` (currently uses raw `.write_text()`)
3. Add `_validate_stage1_artifacts()` in `run_pipeline.py`: (a) called when `--resume-from-stage >= 2`, (b) called as read-back after Stage 1 completes

---

### Stage 2 — REAP + REAM

**Current resume model:** Per-layer `_stage2_partial/` with `merge_{i}.json` + `layer_{i}.pt`. Both atomic via `.tmp` + `os.replace`. Completed layers replayed from partial files on resume. `partial_dir` cleaned on success.

**`--no-resume` effect:** Skip all `_stage2_partial/` I/O — no directory creation, no partial writes, no resume scanning. Final outputs (`_stage2_input_covariance.pt`, `stage2_pruned/`, `merge_map.json`) always written.

**Changes needed:**
1. **CRITICAL:** Delete orphaned `.pt` without matching `.json` in resume scan
2. Clean `.tmp` at startup
3. Add `no_resume: bool = False` parameter to `run()`
4. Thread `no_resume` from `run_pipeline.py`

**Critical invariant preserved:** `_remap_covariance_for_layer()` → `_snapshot_cov_layer()` → `_write_merge_json()` order is correct and must not change.

---

### Stage 3 — Non-Uniform SVD

**Current resume model:** B-cov/C-cov per-layer spills to `_stage3_bcov_partial/`. No Phase D (factoring) resume. No α search persistence. Spill dirs cleaned on success.

**`--no-resume` effect:** Delete existing spill dirs on startup (force fresh covariance collection). Skip α result file. Spill files still written within the stage for memory management — they are always needed operationally.

**Changes needed:**
1. **HIGH:** Covariance spill reuse — check if all spill files exist before calling `_collect_covariances()`. Skip Phase A entirely (including teacher model load) if complete.
2. **HIGH:** Move `_stage3_original_weights.pt` write to before Phase D
3. **HIGH:** α search result persistence to `_stage3_alpha_result.json`
4. Add `no_resume: bool = False` parameter to `run()`
5. Thread `no_resume` from `run_pipeline.py`

---

### Stage 4 — EoRA Residual Compensation

**Current resume model:** Per-layer `_stage4_partial/layer_{i}.pt`. Each spill has format_version, U/V, ranks, effective_ranks. Atomic writes. Partial dir cleaned on success.

**`--no-resume` effect:** Skip all `_stage4_partial/` I/O — no directory, no spills, no resume loading.

**Changes needed:**
1. **CRITICAL:** Double-widen guard — assert `fe.ranks[name] == stage3_ranks[layer_idx][name]` before every `widen_rank()` for non-resumed layers. Snapshot Stage 3 ranks before the loop.
2. Restore `effective_ranks` from spill payload on resume
3. Clean `.tmp` at startup
4. Add `no_resume: bool = False` parameter
5. Thread `no_resume` from `run_pipeline.py`

---

### Stage 5 — Router KD

**Current resume model:** Step-boundary `_stage5_partial/step_{N}.pt`. Rolling 2-checkpoint window. Atomic writes. Fast-forward on resume via `epoch < resume_epoch or (epoch == resume_epoch and i <= resume_batch_i)`.

**`--no-resume` effect:** Skip all `_stage5_partial/` I/O — no directory, no checkpoints, no resume scanning, no fast-forward.

**Changes needed:**
1. **HIGH:** Deferred teacher loading — load only on first live batch (saves ~60s on resume)
2. Save `gradient_accumulation` in checkpoint; assert match on resume
3. Clean `.tmp` at startup
4. Add `no_resume: bool = False` parameter
5. Thread `no_resume` from `run_pipeline.py`

---

### Stage 6 — Validation

**Current resume model:** None by design ("Stage 6 is stateless. Re-running is always safe."). Teacher eval cache is the only persistence (speedup, not resume).

**`--no-resume` effect:** None. Stage 6 already never resumes. The teacher_eval_cache is a performance cache controlled by its own `enabled` config key, independent of `--no-resume`.

**Changes needed:**
1. Normalize `revision` in teacher cache key (`or "main"` for `None`)
2. Background GGUF thread: atomic write pattern (`.tmp` + `os.replace`)

---

## Updated §11 for ALGORITHM_REFERENCE.md

Replace the existing §11 text with:

```markdown
## 11. Durability and Crash-Resume Model

### Inter-Stage Durability

HF Jobs bucket FUSE mounts are **not durable** under SIGKILL or timeout. The durability boundary is per-stage Hub uploads:

\`\`\`
<base_repo>-stage2   ← Stage 2 output + covariance sidecar
<base_repo>-stage3   ← Stage 3 output + originals sidecar
<base_repo>-stage4   ← Stage 4 output
<base_repo>-stage5   ← Final compressed model
\`\`\`

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

**Stage 4 double-widen guard:** Before calling `widen_rank()` for any non-resumed layer, an assertion verifies that the current `fe.ranks[name]` matches the Stage 3 rank (captured in a snapshot before the loop). A rank mismatch indicates double-application of EoRA, which would silently corrupt the factors. The guard is unreachable in the normal pipeline flow (process restart reloads from checkpoint), but protects against in-process re-runs (notebooks, test harnesses).

**Stage 5 deferred teacher load:** The teacher model is loaded lazily on the first live batch (after fast-forward completes), not before the training loop. This eliminates wasted load time on resume.

### Format Version Enforcement

Every partial checkpoint carries a `format_version` field. On resume, the version is checked before any state is restored. A mismatch raises an error with an actionable message ("delete `_stage{N}_partial/` and re-run"). This prevents silent corruption when checkpoint format changes across code versions.
```

---

## Implementation Priority Order

1. **Immediate (before next run):** Fix #1 (`save_json_artifact` atomic), Fix #2 (Stage 2 orphan .pt), Fix #3 (Stage 4 double-widen guard)
2. **Before next run:** Fix #4 (`--no-resume` flag), Fix #5 (pre-flight validation), Fix #7 (originals timing)
3. **High value, moderate effort:** Fix #6 (Stage 3 cov reuse), Fix #8 (α cache), Fix #9 (deferred teacher)
4. **Polish:** Fixes #10-15 (`.tmp` cleanup, effective_ranks, grad_accum check, revision normalization, GGUF atomic)
