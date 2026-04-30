# Resume Strategy Review — All 6 Pipeline Stages

**Date:** 2026-04-30
**Hardware target:** HF H200 — 256 GB Host RAM, 141 GB VRAM
**Reviewer:** ML Intern automated audit

---

## Executive Summary

Each of the 6 pipeline stages was independently audited for correctness, robustness, speed, and disable-resume capability. The audit found:

- **2 critical bugs** (Stage 2 orphan .pt double-remap, Stage 4 double-widen on in-process re-run)
- **3 high-severity gaps** (Stage 3 no Phase D resume, covariance reuse, originals snapshot timing)
- **1 cross-cutting gap** (save_json_artifact not atomic; no `--no-resume` flag)
- **Multiple medium/low fixes** across all stages

All fixes preserve backward compatibility (default behavior unchanged). The `--no-resume` flag is the single pipeline-wide mechanism to disable all resume/checkpoint behavior.

---

## Stage 1 — Super Expert Detection + GRAPE

**Current resume:** None (stateless, JSON-only output). `--resume-from-stage 2+` skips Stage 1.

| # | Severity | Issue | Fix |
|---|----------|-------|-----|
| 1 | **P0** | `save_json_artifact` writes directly to final path (non-atomic) | `.tmp` + `os.replace` pattern |
| 2 | **P0** | `budget_decomposition.json` uses raw `.write_text()` (non-atomic) | Route through `save_json_artifact` |
| 3 | **P0** | No pre-flight validation of Stage 1 artifacts when `--resume-from-stage >= 2` | `_validate_stage1_artifacts()` before Stage 2 entry |
| 4 | **P2** | Read-back validation after Stage 1 completes | Call `_validate_stage1_artifacts()` after Stage 1 |
| 5 | — | `--no-resume` does not affect Stage 1 (correct — its JSONs are outputs, not resume files) | No change needed |

---

## Stage 2 — REAP + REAM

**Current resume:** Per-layer `_stage2_partial/` with `merge_{i}.json` + `layer_{i}.pt`. Atomic writes via `.tmp` + `os.replace`.

| # | Severity | Issue | Fix |
|---|----------|-------|-----|
| 1 | **CRITICAL** | Orphan `.pt` without matching `.json` causes double-remap corruption on resume | Delete orphaned `.pt` files in resume scan |
| 2 | **HIGH** | Dangling `.tmp` files from prior crashes | Glob and delete `*.tmp` at startup |
| 3 | **FEATURE** | No mechanism to disable partial-file writes | `no_resume: bool = False` parameter: skip all `_stage2_partial` I/O |
| 4 | — | `_stage2_input_covariance.pt` written after loop (correct — needs all layers) | No change needed |

---

## Stage 3 — Non-Uniform SVD

**Current resume:** B-cov/C-cov spill per-layer to `_stage3_bcov_partial/`. No Phase D (factoring) resume. No α search persistence.

| # | Severity | Issue | Fix |
|---|----------|-------|-----|
| 1 | **CRITICAL** | Phase D crash leaves model in hybrid state; no per-layer factoring resume | Covariance spill reuse check (skip Phase A if complete) |
| 2 | **HIGH** | `_collect_covariances()` always re-runs, even if all spills exist | Spill completeness sentinel + skip |
| 3 | **HIGH** | `_stage3_original_weights.pt` saved after Phase D; Stage 4 needs it on crash | Move save to before Phase D (after `_snapshot_originals()`) |
| 4 | **HIGH** | α search result (~33 min) not persisted | Save to `_stage3_alpha_result.json`; reload on re-run |
| 5 | **FEATURE** | No `--no-resume` support | `no_resume: bool`: delete existing spills on startup |

---

## Stage 4 — EoRA Residual Compensation

**Current resume:** Per-layer `_stage4_partial/layer_{i}.pt`. Atomic writes.

| # | Severity | Issue | Fix |
|---|----------|-------|-----|
| 1 | **CRITICAL** | `widen_rank()` is not idempotent; in-process re-run double-applies EoRA | Rank assertion before `widen_rank()` (guard against already-widened fe) |
| 2 | **HIGH** | `effective_ranks` may not be fully restored on resume | Explicitly restore from spill payload |
| 3 | **MEDIUM** | Dangling `.tmp` files | Glob and delete at startup |
| 4 | **FEATURE** | No `--no-resume` support | `no_resume: bool`: skip all partial-dir I/O |

---

## Stage 5 — Router KD

**Current resume:** Step-boundary checkpoints in `_stage5_partial/step_{N}.pt`. Rolling 2-checkpoint window. Atomic writes.

| # | Severity | Issue | Fix |
|---|----------|-------|-----|
| 1 | **HIGH** | Teacher loaded before fast-forward (wasted ~60s on resume) | Deferred/lazy teacher loading after fast-forward completes |
| 2 | **MEDIUM** | No `gradient_accumulation` invariant check on resume | Save in checkpoint, assert on load |
| 3 | **MEDIUM** | Dangling `.tmp` files | Glob and delete at startup |
| 4 | **FEATURE** | No `--no-resume` support | `no_resume: bool`: skip checkpoint read/write |

---

## Stage 6 — Validation

**Current resume:** None (stateless by design). Teacher eval cache is the only persistence.

| # | Severity | Issue | Fix |
|---|----------|-------|-----|
| 1 | **MEDIUM** | `revision=None` vs `revision="main"` cache key mismatch | `or "main"` normalization |
| 2 | **MEDIUM** | Background GGUF thread writes non-atomically to final path | `.tmp` + `os.replace` pattern |
| 3 | — | Stage 6 never resumes (correct by design) | No change needed |
| 4 | — | Teacher eval cache is a speedup, not a resume mechanism | No change needed |

---

## Cross-Cutting: `--no-resume` Flag

Added to `run_pipeline.py` argument parser. Threaded to each stage's `run()` as `no_resume=True`. Per-stage behavior:

| Stage | Effect of `--no-resume` |
|-------|------------------------|
| 1 | No effect (Stage 1 has no resume files; its JSONs are outputs) |
| 2 | Skip `_stage2_partial/` creation, reading, and writing |
| 3 | Delete existing `_stage3_bcov_partial/`, `_stage3_ccov_partial/`; skip α result cache |
| 4 | Skip `_stage4_partial/` creation, reading, and writing |
| 5 | Skip `_stage5_partial/` creation, reading, and writing |
| 6 | No effect (Stage 6 already never resumes) |

---

## Implementation Status

All fixes documented in this file. Code changes applied via separate commits.
