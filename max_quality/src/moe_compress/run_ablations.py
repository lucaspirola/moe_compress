"""Stage 2 v2 ablation harness — runs A0..A11 sequentially in one job.

Pipeline shape per ablation: Stage 1 (shared) → Stage 2 → Stage 2.5 → Stage 6.
Stages 3, 4, 5 are skipped — Stage 2 alone hits the 35% total-reduction target
because we set ``target.expert_svd_ratio = 100.0`` so the budget solver
allocates ~all of the savings to expert pruning.

The driver is **idempotent**: ablations whose ``stage6_eval.json`` already
exists are skipped on re-invocation. This makes job-timeout + resubmit a
zero-config recovery path — the harness picks up where it left off.

Per-stage Hub uploads are intentionally disabled for ablation runs (set via
``PIPELINE_HUB_RESULT_REPO_BASE=""``); the bucket-mounted artifact dir is
the durability boundary, not the Hub. Twelve ablations × per-stage uploads
would create 36+ junk repos under ``pirola/``.
"""
from __future__ import annotations

import os

# Switch PyTorch's CUDA caching allocator to expandable segments BEFORE any
# torch import. With the per-layer GPU-resident covariance accumulator
# (256 experts × ~100 MB matrices) plus per-batch gated-output tensors and
# repeated dense [T, max_K, d_hid] chunk allocations, the default
# fixed-block allocator fragments after a handful of profile layers and
# eventually fails to satisfy a large allocation — surfacing as a hard
# CUDA segfault rather than a clean OutOfMemoryError. Reproduced as a
# crash at layer 7 batch ~20 of Stage 2 on Qwen3.6-35B-A3B. The
# expandable-segments allocator grows the same region instead of carving
# fixed blocks, so coalesced free space remains usable. No code change
# needed once this env var is set before torch's CUDA initialization.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Cap the hf-xet (HF Xet storage) tokio worker pool BEFORE any huggingface_hub
# import. Left unbounded it spawns ~79 threads per HF transfer; together with
# OpenMP + pyarrow worker pools that pushed the process past ~185 live threads
# — the thread-count blowup that made the periodic faulthandler watchdog
# (removed below) race frame mutation and segfault. 8 is ample for our I/O.
os.environ.setdefault("HF_XET_NUM_CONCURRENT_RANGE_GETS", "8")

# Disable inductor's GEMM autotuner BEFORE torch imports. torch 2.11.0+cu130's
# `torch._grouped_mm` routes through inductor autotune on dynamic group-size
# signatures; with the Qwen3.6-35B-A3B MoE Stage 2 REAP profile (active-experts
# cardinality changes once merging starts at layer 12), the autotuner deadlocks
# — main thread hangs 2 min, faulthandler dump fingers `torch/nn/functional.py
# in grouped_mm`, then SIGSEGV. Reproduced on H200 SXM5 (driver 580.126.09) AND
# B200 (590.48.01) on the cu130 image; the prior cu128 wheel did not autotune
# this op so it worked. Disabling autotune keeps the same kernel at the same
# cuBLASLt-13 grouped-GEMM speed (no compromise) — only the autotune layer is
# turned off. See pytorch/pytorch issues #158042, #159378, #156202.
os.environ.setdefault("TORCHINDUCTOR_MAX_AUTOTUNE_GEMM", "0")
os.environ.setdefault("TORCHINDUCTOR_MAX_AUTOTUNE", "0")

# Disable persistent FX-graph + AOTAutograd caches BEFORE torch imports.
# Symptom (reproduced 2026-05-14, H200 SXM5, torch 2.11.0+cu130, Stage 2.5
# router KD with `torch.compile(student, mode='default')`): clean training
# for 100-200 optimizer steps, then SIGSEGV in either
# `_aot_autograd/runtime_wrappers.py:2735 impl_fn` (compiled backward) or
# `triton/runtime/autotuner.py:252 → jit.py:744` (autotune re-entry on a
# stale cached kernel). Same failure class as pytorch/pytorch#144609 —
# Inductor's persistent FX-graph cache reuses serialized compiled-backward
# artifacts whose CUDA device handles get invalidated between autograd
# buffer releases, producing non-deterministic segfaults at autograd-
# engine re-invocation. NO speed compromise — torch.compile stays ON, the
# in-process kernel cache stays hot, only the persistent on-disk artifact
# cache is bypassed.
os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "0")
os.environ.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "0")

# Inductor's compile_worker uses `fork` by default — on a process that has
# already initialised CUDA + multithreaded native libs (BLAS, OpenMP), this
# is undefined behaviour and causes random SIGSEGV in the child during
# compilation. Stage 6 (eval-only with torch.compile(dynamic=True)) hit this
# every ~5-10 min during lm_eval's task-rotation (each new task triggers
# fresh compile jobs at runtime, long after CUDA was already up). Switching
# the worker startup to `spawn` and serialising compile threads is the
# supported escape hatch (pytorch/pytorch#148651). Zero speed cost — only
# the compiler's process topology changes; compiled kernel output is
# byte-identical, and runtime execution is unaffected. Also bump the
# Dynamo recompile/cache ceiling so dynamic-shape eval doesn't fall back
# to eager mid-run when the default 8-entry cache fills up.
os.environ.setdefault("TORCHINDUCTOR_WORKER_START", "spawn")
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
os.environ.setdefault("TORCHDYNAMO_CACHE_SIZE_LIMIT", "512")
os.environ.setdefault("TORCHDYNAMO_RECOMPILE_LIMIT", "512")

import argparse
import copy
import faulthandler
import json
import logging
import math
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import yaml

# Install faulthandler's fatal-signal handler so a genuine SIGSEGV/SIGABRT
# still dumps a Python traceback to stderr.
#
# We deliberately do NOT arm faulthandler.dump_traceback_later(repeat=True):
# that periodic watchdog walks EVERY thread's frame chain without the GIL,
# and with the ~185 threads this harness spawns (hf-xet tokio pool, OpenMP,
# pyarrow) it races frame mutation and SIGSEGVs *inside the dumper thread
# itself* — confirmed 2026-05-16 via core-dump backtrace (crash in
# PyCode_Addr2Line with rdi=0x20, a freed frame's garbage f_code pointer).
# It fired whenever a Stage 2/2.5 step ran longer than 120s, which for days
# masqueraded as random multi-host "hardware" instability. Hang forensics
# are covered instead by the per-ablation _last_alive.json breadcrumb.
faulthandler.enable()

from .run_pipeline import main as run_pipeline_main  # noqa: F401 (subprocess entry below)
from .utils.model_io import load_json_artifact, save_json_artifact
from .utils.runtime_monitor import (
    flush as _rt_flush,
    install_signal_handlers as _rt_install_signal_handlers,
    set_path as _rt_set_path,
    update as _rt_update,
)
from .utils.trackio_log import trackio_log as _trackio_log

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ablation matrix — mirrors max_quality/docs/stage2_assignment_revision.md § 8
# ---------------------------------------------------------------------------

# Each row's deltas are applied on top of the base config's `stage2_reap_ream`
# block. All v2 flags absent from a row's deltas inherit their baseline value
# (greedy / pre / none / false / 0) from the YAML.
_A4_BASE: dict[str, Any] = {
    "assignment_solver": "auto",
    "cost_alignment": "post",
    "cost_whitening": "diag",
    "cost_asymmetric": True,
    "em_refinement_rounds": 2,
    "capacity_util_threshold": 0.25,
    "cost_topk_filter": 24,
}
_A7_BASE: dict[str, Any] = {
    "assignment_solver": "auto",
    "cost_alignment": "post",
    "cost_whitening": "full",
    "cost_asymmetric": True,
    "em_refinement_rounds": 3,
    "capacity_util_threshold": 0.25,
    "cost_topk_filter": 48,
}
_A8_BASE: dict[str, Any] = {**_A7_BASE, "expert_distill_steps": 500}

ABLATION_DELTAS: list[tuple[str, dict[str, Any]]] = [
    ("A0",  {}),
    ("A1",  {"assignment_solver": "auto"}),
    ("A1_oldkd", {"assignment_solver": "auto", "stage5_router_kd": {"epochs": 1, "max_calibration_samples": 3000, "kd_temperature_start": 1.0, "kd_temperature_end": 1.0}}),
    ("A2",  {"assignment_solver": "auto", "cost_alignment": "post", "cost_whitening": "diag"}),
    ("A3",  {"assignment_solver": "auto", "cost_alignment": "post", "cost_whitening": "diag", "cost_asymmetric": True}),
    ("A4",  _A4_BASE),
    ("A5",  {**_A4_BASE, "cost_topk_filter": 16}),
    ("A6",  {**_A4_BASE, "em_refinement_rounds": 3}),
    ("A7",  _A7_BASE),
    ("A8",  _A8_BASE),
    ("A9",  {**_A8_BASE, "assignment_solver": "sinkhorn"}),
    ("A10", {**_A8_BASE, "expert_distill_steps": 200}),
    ("A11", {**_A8_BASE, "expert_distill_min_freq_sum": 0.5}),
]


# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------


def _build_ablation_config(
    base: dict, deltas: dict, *, num_sequences: int,
    teacher_cache_path: Path,
    teacher_model_repo: str | None = None,
    stage5_max_calibration_samples: int | None = None,
    stage5_max_sequence_length: int | None = None,
    stage6_mode: str = "full",
) -> dict:
    """Apply per-ablation deltas + the 35%-via-Stage-2-only target to the
    base config. The teacher-cache path forces all 12 ablations to read /
    write the same Stage 6 cache file (filled by A0, hit by A1..A11).

    Optional Stage 5 overrides (applied uniformly across all ablations in the
    invocation) lower H200 VRAM pressure without touching A0's BF16 baseline:
      - teacher_model_repo: swap the KD teacher (e.g. FP8 quant).
      - stage5_max_calibration_samples / stage5_max_sequence_length: shrink
        the Stage 2.5 calibration footprint.
    """
    cfg = copy.deepcopy(base)
    cfg.setdefault("target", {})
    cfg["target"]["total_reduction_ratio"] = 0.35
    cfg["target"]["expert_svd_ratio"] = 100.0  # force ~100% expert pruning, ~0% SVD
    cfg.setdefault("calibration", {})
    cfg["calibration"]["num_sequences"] = num_sequences
    s2 = cfg.setdefault("stage2_reap_ream", {})
    # Stage 2 reads its own per-stage knob (stage2_reap_ream.num_calibration_samples,
    # see stage2_reap_ream.py:111) — overriding only the global calibration.num_sequences
    # would leave Stage 2 at its YAML-specified value (4000 in prod). Cap it here so the
    # --num-sequences flag actually bounds Stage 2 work.
    s2["num_calibration_samples"] = num_sequences
    for k, v in deltas.items():
        if k == "stage5_router_kd" and isinstance(v, dict):
            cfg.setdefault("stage5_router_kd", {}).update(v)
        else:
            s2[k] = v
    s5 = cfg.setdefault("stage5_router_kd", {})
    if teacher_model_repo:
        s5["teacher_model_repo"] = teacher_model_repo
    if stage5_max_calibration_samples is not None:
        s5["max_calibration_samples"] = stage5_max_calibration_samples
    if stage5_max_sequence_length is not None:
        s5["max_sequence_length"] = stage5_max_sequence_length
    cfg.setdefault("stage6_validate", {})
    cfg["stage6_validate"].setdefault("teacher_eval_cache", {})
    cfg["stage6_validate"]["teacher_eval_cache"]["cache_path"] = str(teacher_cache_path)
    # Stage 6 mode: 'thermometer' swaps the expensive full eval for the cheap
    # stage6alt directional signal. The thermometer's teacher-BPT cache is
    # sweep-shared (sits beside the full-eval teacher cache in shared_dir) so
    # all 12 ablations reuse one teacher measurement.
    cfg["stage6_validate"]["mode"] = stage6_mode
    if stage6_mode == "thermometer":
        therm = cfg["stage6_validate"].setdefault("thermometer", {})
        therm["teacher_cache_path"] = str(
            teacher_cache_path.parent / "thermometer_teacher_cache.json"
        )
    # Disable imatrix for ablations: it runs convert_hf_to_gguf.py + llama-imatrix
    # via subprocess (10-30 min CPU work for a 35B model) and produces a GGUF
    # quantization calibration artifact we don't consume in the ablation analysis.
    # Per-ablation savings: 10-30 min × 12 = 2-6h total.
    cfg["stage6_validate"].setdefault("imatrix", {})
    cfg["stage6_validate"]["imatrix"]["enabled"] = False
    return cfg


# ---------------------------------------------------------------------------
# Per-ablation runner
# ---------------------------------------------------------------------------


def _stage6_artifact(stage6_mode: str) -> str:
    """Final Stage 6 artifact filename for the given mode.

    'full' → stage6_eval.json (stage6_validate); 'thermometer' →
    stage6alt_eval.json (stage6alt_thermometer). Used as the completion gate
    and the bucket-upload target.
    """
    return "stage6alt_eval.json" if stage6_mode == "thermometer" else "stage6_eval.json"


def _is_complete(ablation_dir: Path, stage6_mode: str = "full") -> bool:
    """Skip-if-already-run signal. Stage 6's final artifact is the gate."""
    return (ablation_dir / _stage6_artifact(stage6_mode)).exists()


# ---------------------------------------------------------------------------
# Sweep progress: leaderboard + summary (regenerated after EVERY ablation)
# ---------------------------------------------------------------------------

def _fmt(v, prec: int = 4) -> str:
    """Leaderboard cell formatter — '—' for missing, 'inf' for non-finite."""
    if v is None:
        return "—"
    if isinstance(v, float):
        return "inf" if not math.isfinite(v) else f"{v:.{prec}f}"
    return str(v)


def _render_leaderboard(results: dict, failures: list, rows: list,
                        stage6_mode: str) -> str:
    """Render a ranked markdown leaderboard from the per-ablation results.

    Robust to partial sweeps: only rows present in `results` are tabled;
    failed and not-yet-run rows are listed below. Regenerated after every
    ablation by `_persist_progress`, so a mid-sweep process death still
    leaves rows 0..N-1 visible.
    """
    done = {a for a, _ in failures}
    out = [
        f"# Ablation leaderboard — stage6 mode: {stage6_mode}",
        "",
        f"_completed {len(results)}/{len(rows)} · failed {len(failures)} · "
        f"regenerated after every row_",
        "",
    ]
    if stage6_mode == "thermometer":
        # Rank by bpt_gap ascending (lower = less compression damage);
        # missing/non-finite gaps sink to the bottom.
        def _gap(r):
            g = r.get("bpt_gap")
            return g if isinstance(g, (int, float)) and math.isfinite(g) else float("inf")
        ranked = sorted(results.items(), key=lambda kv: _gap(kv[1]))
        out += [
            "| rank | ablation | bpt_gap | top1_agree | student_bpt | "
            "teacher_bpt | ARC-E | HSwag | corpus | knob delta |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for i, (aid, r) in enumerate(ranked, 1):
            corpus = (r.get("corpus") or {}).get("name", "?")
            deltas = r.get("_deltas") or {}
            delta_s = ", ".join(f"{k}={v}" for k, v in deltas.items()) or "(baseline)"
            out.append(
                f"| {i} | {aid} | {_fmt(r.get('bpt_gap'))} | "
                f"{_fmt(r.get('top1_agreement'))} | {_fmt(r.get('student_bpt'))} | "
                f"{_fmt(r.get('teacher_bpt'))} | "
                f"{_fmt(r.get('student_arc_easy_acc_norm'), 3)} | "
                f"{_fmt(r.get('student_hellaswag_acc_norm'), 3)} | "
                f"{corpus} | {delta_s} |"
            )
        out += [
            "",
            "Ranking: lower `bpt_gap` = less compression damage. On the "
            "`nemotron` corpus the gap is RELATIVE (Stage-2.5 adaptation is "
            "common-mode across rows) — rank rows against each other, not "
            "against zero. Tiebreak: higher `top1_agree`, then ARC-E+HSwag.",
        ]
    else:
        out += ["| ablation | wikitext2_ppl | knob delta |", "|---|---|---|"]
        for aid, r in results.items():
            ppl = r.get("student", {}).get("wikitext2_ppl")
            deltas = r.get("_deltas") or {}
            delta_s = ", ".join(f"{k}={v}" for k, v in deltas.items()) or "(baseline)"
            out.append(f"| {aid} | {_fmt(ppl)} | {delta_s} |")
    if failures:
        out += ["", "## Failed"]
        out += [f"- **{aid}**: {err}" for aid, err in failures]
    pending = [aid for aid, _ in rows if aid not in results and aid not in done]
    if pending:
        out += ["", "## Not yet run", ", ".join(pending)]
    return "\n".join(out) + "\n"


def _persist_progress(ablations_root: Path, results: dict, failures: list,
                      rows: list, stage6_mode: str, upload_bucket: str,
                      hf_token: str) -> None:
    """Write `_summary.json` + `_leaderboard.md` and upload them to the bucket.

    Called after EVERY ablation (success or failure) so a mid-sweep process
    death — e.g. a Stage 6 SIGSEGV, which kills the whole process, not just a
    row — still leaves the completed rows' results durable on the bucket.
    Fully guarded: a write/upload failure here must never abort the sweep.
    """
    try:
        summary = {
            "stage6_mode": stage6_mode,
            "results": results,
            "failures": [{"ablation_id": a, "error": e} for a, e in failures],
            "num_completed": len(results),
            "num_failed": len(failures),
            "total_ablations_planned": len(rows),
        }
        summary_path = ablations_root / "_summary.json"
        leaderboard_path = ablations_root / "_leaderboard.md"
        save_json_artifact(summary, summary_path)
        leaderboard_path.write_text(
            _render_leaderboard(results, failures, rows, stage6_mode),
            encoding="utf-8",
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("progress persist (local write) failed: %s", exc)
        return
    if upload_bucket and hf_token:
        # Upload on a daemon thread with a bounded join. A hung HF connection
        # (TCP open, no response) does NOT raise — without the timeout it would
        # stall the main GPU thread between rows for the ~15-30 min of OS TCP
        # keepalive. The local files are already written above, so abandoning a
        # slow upload only delays bucket visibility by one row.
        def _upload_progress() -> None:
            try:
                from huggingface_hub import HfApi
                HfApi(token=hf_token).batch_bucket_files(
                    bucket_id=upload_bucket,
                    add=[(str(summary_path), "_summary.json"),
                         (str(leaderboard_path), "_leaderboard.md")],
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("progress upload failed (files remain on disk): %s",
                            exc)
        t = threading.Thread(target=_upload_progress, daemon=True,
                             name="progress-upload")
        t.start()
        t.join(timeout=30)
        if t.is_alive():
            log.warning("progress upload >30s — abandoning (next row retries)")


def _upload_ablation_bg(
    ablation_id: str,
    ablation_dir: Path,
    shared_dir: Path,
    bucket: str,
    hf_token: str,
    upload_shared: bool,
    stage6_artifact: str = "stage6_eval.json",
) -> None:
    """Upload one ablation's results to HF Hub. Runs in a background thread so
    the GPU can start the next ablation immediately. Writes uploaded.flag on
    success so the orchestrator knows this ablation is durable on Hub."""
    _log = logging.getLogger(__name__)
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=hf_token)

        if upload_shared and shared_dir.exists():
            _log.info("[%s] uploading _shared/ to bucket %s", ablation_id, bucket)
            add_list = []
            for f in shared_dir.rglob("*"):
                if not f.is_file():
                    continue
                rel = f.relative_to(shared_dir)
                rel_str = str(rel)
                if rel_str.endswith(".lock") or rel_str.endswith(".py") or "__pycache__" in rel_str:
                    continue
                add_list.append((str(f), f"_shared/{rel_str}"))
            if add_list:
                api.batch_bucket_files(bucket_id=bucket, add=add_list)

        _log.info("[%s] uploading %s to bucket %s",
                  ablation_id, stage6_artifact, bucket)
        api.batch_bucket_files(
            bucket_id=bucket,
            add=[(str(ablation_dir / stage6_artifact),
                  f"{ablation_id}/{stage6_artifact}")],
        )

        (ablation_dir / "uploaded.flag").touch()
        _log.info("[%s] upload complete → %s/%s/%s",
                  ablation_id, bucket, ablation_id, stage6_artifact)
    except Exception as exc:  # noqa: BLE001
        _log.warning("[%s] background upload failed (artifacts remain on disk): %s",
                     ablation_id, exc)


def _hardlink_or_copy(src: Path, dst: Path) -> None:
    """Hardlink src → dst when on the same filesystem; fall back to copy."""
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)


def _seed_stage1_artifacts(ablation_dir: Path, shared_dir: Path) -> None:
    """Hardlink the three Stage 1 outputs from _shared/ into the ablation
    dir so Stage 2 reads them as if Stage 1 had run locally."""
    for name in ("stage1_blacklist.json", "stage1_budgets.json",
                 "budget_decomposition.json"):
        src = shared_dir / name
        if not src.exists():
            raise RuntimeError(
                f"_seed_stage1_artifacts: shared Stage 1 artifact missing: {src}. "
                "Pre-flight Stage 1 step must run before any ablation."
            )
        _hardlink_or_copy(src, ablation_dir / name)


def _write_ablation_config(cfg: dict, ablation_dir: Path) -> Path:
    """Write the per-ablation config YAML into the ablation dir for forensic
    record + pass-through to run_pipeline.main()."""
    cfg_path = ablation_dir / "ablation_config.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return cfg_path


def _run_one_ablation(
    *, ablation_id: str, deltas: dict[str, Any], base_config: dict,
    shared_dir: Path, ablations_root: Path, model_repo: str,
    num_sequences: int, teacher_cache_path: Path,
    teacher_model_repo: str | None = None,
    stage5_max_calibration_samples: int | None = None,
    stage5_max_sequence_length: int | None = None,
    stage6_mode: str = "full",
) -> dict[str, Any]:
    """Drive one ablation through Stage 2 → 2.5 → Stage 6. Stage 1 artifacts
    are seeded from ``shared_dir``; Stage 6 reads the shared teacher cache."""
    ablation_dir = ablations_root / ablation_id
    ablation_dir.mkdir(parents=True, exist_ok=True)
    _s6_artifact = _stage6_artifact(stage6_mode)

    if _is_complete(ablation_dir, stage6_mode):
        log.info("[%s] already complete — loading prior result", ablation_id)
        return json.loads((ablation_dir / _s6_artifact).read_text())

    log.info("[%s] starting (deltas=%s)", ablation_id, deltas)
    t_start = time.monotonic()

    # Per-ablation breadcrumb: <ablation_dir>/_last_alive.json captures the
    # last-known state (layer/batch/phase + signal on terminal exit) so the
    # next run can diagnose where the prior attempt died without parsing logs.
    _rt_set_path(ablation_dir / "_last_alive.json")
    _rt_update(ablation_id=ablation_id, phase="ablation_start",
               deltas={k: str(v) for k, v in deltas.items()})

    # Per-ablation Trackio run.
    try:
        import trackio
        try:
            run = trackio.init(
                project="moe-compress-strategy-a",
                name=f"ablation-{ablation_id}",
                space_id=os.environ.get("TRACKIO_SPACE_ID", "pirola/trackio"),
                config={"ablation_id": ablation_id, **deltas},
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("[%s] trackio.init failed: %s — continuing without it", ablation_id, exc)
            run = None
    except ImportError:
        run = None

    try:
        cfg = _build_ablation_config(
            base_config, deltas,
            num_sequences=num_sequences,
            teacher_cache_path=teacher_cache_path,
            teacher_model_repo=teacher_model_repo,
            stage5_max_calibration_samples=stage5_max_calibration_samples,
            stage5_max_sequence_length=stage5_max_sequence_length,
            stage6_mode=stage6_mode,
        )
        cfg_path = _write_ablation_config(cfg, ablation_dir)
        _seed_stage1_artifacts(ablation_dir, shared_dir)

        # Stage 2 + Stage 2.5 — one pipeline call (Stage 2.5 is folded into the
        # Stage-2 block of run_pipeline.main, so --stop-after-stage 2 runs both).
        # Run as a FRESH SUBPROCESS, not in-process: doing Stage 2/2.5 and then
        # Stage 6 in one long-lived process accumulates ~130 GB of non-freed
        # resident memory, so the Stage-6 step OOM-kills at ~182 GB (host RAM
        # 178 GB, and the leftover is not swap-reclaimable). A subprocess is
        # reaped on exit, returning all memory to the OS — each stage-group
        # starts clean. Identical work and args; env/cwd inherited.
        rc1 = subprocess.run(
            [sys.executable, "-m", "moe_compress.run_pipeline",
             "--config", str(cfg_path),
             "--model", model_repo,
             "--artifacts-dir", str(ablation_dir),
             "--target-ratio", "0.35",
             "--resume-from-stage", "2",
             "--stop-after-stage", "2"],
            check=False,
        ).returncode
        if rc1 != 0:
            raise RuntimeError(f"[{ablation_id}] Stage 2/2.5 returned exit code {rc1}")

        # Stage 6 loads from stage2p5_final/ via run_pipeline._load_for_stage's
        # fallback path (0871b98). The previous `stage5_final → stage2p5_final`
        # symlink bridge is no longer needed and has been removed; the load-time
        # fallback handles both the full-pipeline path (stage5_final/ present)
        # and the ablation harness path (stage2p5_final/ only) without a
        # filesystem-side workaround.
        # Stage 6 — also a fresh subprocess (see the Stage 2/2.5 note above):
        # this is the step that OOM-killed when it inherited the accumulated
        # process. Clean process => ~125 GB peak, well within 178 GB RAM.
        rc2 = subprocess.run(
            [sys.executable, "-m", "moe_compress.run_pipeline",
             "--config", str(cfg_path),
             "--model", model_repo,
             "--artifacts-dir", str(ablation_dir),
             "--target-ratio", "0.35",
             "--resume-from-stage", "6",
             "--stop-after-stage", "6"],
            check=False,
        ).returncode
        if rc2 != 0:
            raise RuntimeError(f"[{ablation_id}] Stage 6 returned exit code {rc2}")

        if not _is_complete(ablation_dir, stage6_mode):
            raise RuntimeError(
                f"[{ablation_id}] Stage 6 succeeded but {_s6_artifact} missing"
            )

        elapsed = time.monotonic() - t_start
        result = json.loads((ablation_dir / _s6_artifact).read_text())
        result["_ablation_id"] = ablation_id
        result["_deltas"] = deltas
        result["_elapsed_seconds"] = elapsed
        log.info("[%s] complete in %.1f min", ablation_id, elapsed / 60.0)
        # Mark the breadcrumb as cleanly completed so a forensic read of
        # _last_alive.json after a successful run doesn't look like a
        # mid-batch crash. Force-flush past the throttle so the final
        # state lands on disk before this function returns.
        _rt_update(phase="ablation_done", elapsed_seconds=elapsed)
        _rt_flush()
        return result
    finally:
        if run is not None:
            try:
                run.finish()
            except Exception as exc:  # noqa: BLE001
                log.warning("[%s] trackio.run.finish failed: %s", ablation_id, exc)


# ---------------------------------------------------------------------------
# Pre-flight: shared Stage 1 + shared teacher cache
# ---------------------------------------------------------------------------


def _preflight(
    *, base_config: dict, shared_dir: Path, model_repo: str,
    num_sequences: int, teacher_cache_path: Path,
    teacher_model_repo: str | None = None,
    stage5_max_calibration_samples: int | None = None,
    stage5_max_sequence_length: int | None = None,
    stage6_mode: str = "full",
) -> None:
    """Run Stage 1 once on the base config. Idempotent — skips if artifacts
    already present. The teacher cache is filled by A0's Stage 6 (no separate
    teacher-only mode in Stage 6).
    """
    shared_dir.mkdir(parents=True, exist_ok=True)
    needed = ("stage1_blacklist.json", "stage1_budgets.json",
              "budget_decomposition.json")
    if all((shared_dir / n).exists() for n in needed):
        log.info("Pre-flight Stage 1 already complete in %s", shared_dir)
        return

    log.info("Pre-flight: running Stage 1 once into %s", shared_dir)
    cfg = _build_ablation_config(
        base_config, deltas={},
        num_sequences=num_sequences,
        teacher_cache_path=teacher_cache_path,
        teacher_model_repo=teacher_model_repo,
        stage5_max_calibration_samples=stage5_max_calibration_samples,
        stage5_max_sequence_length=stage5_max_sequence_length,
        stage6_mode=stage6_mode,
    )
    cfg_path = _write_ablation_config(cfg, shared_dir)
    rc = run_pipeline_main([
        "--config", str(cfg_path),
        "--model", model_repo,
        "--artifacts-dir", str(shared_dir),
        "--target-ratio", "0.35",
        "--resume-from-stage", "1",
        "--stop-after-stage", "1",
    ])
    if rc != 0:
        raise RuntimeError(f"Pre-flight Stage 1 failed with exit code {rc}")
    for n in needed:
        if not (shared_dir / n).exists():
            raise RuntimeError(
                f"Pre-flight Stage 1 returned 0 but {n} is missing from {shared_dir}"
            )
    log.info("Pre-flight Stage 1 complete")


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stage 2 v2 ablation harness (A0..A11)",
    )
    parser.add_argument("--config", required=True,
                        help="Base YAML config (e.g., qwen36_35b_a3b_30pct.yaml)")
    parser.add_argument("--model", required=True,
                        help="HF model repo for the base model")
    parser.add_argument("--ablations-root", required=True,
                        help="Root directory for per-ablation artifacts (will create A0..A11 subdirs)")
    parser.add_argument("--num-sequences", type=int, default=1000,
                        help="Calibration sequence count for ablations (default 1000)")
    parser.add_argument("--only", default=None,
                        help="Comma-separated subset of ablation IDs to run (e.g., A0,A4,A8). Default: all 12.")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run A0 only with tiny calibration; for local plumbing verification")
    parser.add_argument("--stage6-mode", choices=["full", "thermometer"],
                        default="full",
                        help="Stage 6 eval mode for the sweep. 'full' (default) "
                        "runs the comprehensive suite per row (~$50-120/row). "
                        "'thermometer' runs the cheap stage6alt directional eval "
                        "(~$0.22/row) — use it to sweep A0..A11, then re-run the "
                        "winning row with --stage6-mode full for the deliverable.")
    parser.add_argument("--preflight-only", action="store_true",
                        help="Run Stage 1 pre-flight, write _shared/ artifacts, then exit. "
                        "For split-platform workflows: run pre-flight on a high-VRAM GPU "
                        "(e.g. H200) once, then run the per-ablation loop on cheaper hardware "
                        "that consumes the bucket-stored _shared/ outputs.")
    # Stage 5 VRAM-reduction levers — applied uniformly to every ablation in
    # this invocation. Either or both may be set; both default to null (no
    # override, base config values).
    parser.add_argument("--teacher-model-repo", default=None,
                        help="Lever (a): override stage5_router_kd.teacher_model_repo "
                        "for every ablation. Use Qwen/Qwen3.6-35B-A3B-FP8 to fit "
                        "Stage 2.5 on H200 (143 GB). Default: null (BF16 teacher).")
    parser.add_argument("--stage5-max-calibration-samples", type=int, default=None,
                        help="Lever (b): override stage5_router_kd.max_calibration_samples. "
                        "Smaller calibration set → smaller Stage 2.5 activation peak "
                        "and fewer KD steps. Default: null (use config value).")
    parser.add_argument("--stage5-max-sequence-length", type=int, default=None,
                        help="Lever (b): override stage5_router_kd.max_sequence_length. "
                        "Shorter sequences also shrink the activation peak. "
                        "Default: null (use config value).")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    log.info("========== Stage 2 v2 Ablation Harness ==========")

    base_config = yaml.safe_load(Path(args.config).read_text())
    ablations_root = Path(args.ablations_root)
    ablations_root.mkdir(parents=True, exist_ok=True)
    # Install SIGTERM/SIGINT/SIGHUP handlers + atexit flusher so the breadcrumb
    # records terminal signals. SIGSEGV can't be reliably caught from Python;
    # per-batch _rt_update() calls in the hot paths cover that case.
    _rt_install_signal_handlers()
    shared_dir = ablations_root / "_shared"
    teacher_cache_path = shared_dir / "teacher_eval_cache.json"

    # Disable per-stage Hub upload for ablations — bucket is the durability layer.
    os.environ.pop("PIPELINE_HUB_RESULT_REPO_BASE", None)

    num_sequences = 8 if args.smoke_test else args.num_sequences

    # Filter ablations to run.
    if args.only:
        only_ids = {x.strip() for x in args.only.split(",") if x.strip()}
        rows = [(aid, d) for (aid, d) in ABLATION_DELTAS if aid in only_ids]
    elif args.smoke_test:
        rows = ABLATION_DELTAS[:1]  # A0 only
    else:
        rows = ABLATION_DELTAS

    log.info("Will run %d ablation(s): %s", len(rows), [r[0] for r in rows])

    # Pre-flight: shared Stage 1 (blacklist + budgets, identical across ablations).
    _preflight(
        base_config=base_config,
        shared_dir=shared_dir,
        model_repo=args.model,
        num_sequences=num_sequences,
        teacher_cache_path=teacher_cache_path,
        teacher_model_repo=args.teacher_model_repo,
        stage5_max_calibration_samples=args.stage5_max_calibration_samples,
        stage5_max_sequence_length=args.stage5_max_sequence_length,
        stage6_mode=args.stage6_mode,
    )

    # Stop here if we only wanted the shared Stage 1 artifacts (e.g., the
    # H200-pre-flight + A100-ablations split workflow). _preflight wrote
    # everything needed to shared_dir/{stage1_blacklist,stage1_budgets,
    # budget_decomposition}.json — pickup happens transparently on whatever
    # platform runs the per-ablation loop next.
    if args.preflight_only:
        log.info("--preflight-only: exiting after pre-flight. _shared/ ready at %s", shared_dir)
        return 0

    # Background upload state. When HF_ARTIFACTS_BUCKET is set, each completed
    # ablation is uploaded in a daemon thread while the GPU works on the next
    # one. uploaded.flag is written on success so the orchestrator can tell
    # which ablations are durable on Hub and safe to use when switching GPUs.
    upload_bucket = os.environ.get("HF_ARTIFACTS_BUCKET", "").strip()
    hf_token = os.environ.get("HF_TOKEN", "")
    upload_threads: list[tuple[str, threading.Thread]] = []
    first_upload = True  # _shared/ is uploaded once alongside the first ablation

    # Per-ablation loop.
    results: dict[str, Any] = {}
    failures: list[tuple[str, str]] = []
    for i, (aid, deltas) in enumerate(rows):
        log.info("[%d/%d] === %s ===", i + 1, len(rows), aid)
        try:
            result = _run_one_ablation(
                ablation_id=aid, deltas=deltas, base_config=base_config,
                shared_dir=shared_dir, ablations_root=ablations_root,
                model_repo=args.model, num_sequences=num_sequences,
                teacher_cache_path=teacher_cache_path,
                teacher_model_repo=args.teacher_model_repo,
                stage5_max_calibration_samples=args.stage5_max_calibration_samples,
                stage5_max_sequence_length=args.stage5_max_sequence_length,
                stage6_mode=args.stage6_mode,
            )
            results[aid] = result
            # Sanity: A0 must populate the teacher cache so A1..A11 hit it
            # instead of re-scoring the teacher. The cache file differs by mode.
            _teacher_cache_file = (
                teacher_cache_path.parent / "thermometer_teacher_cache.json"
                if args.stage6_mode == "thermometer" else teacher_cache_path
            )
            if aid == "A0" and not _teacher_cache_file.exists():
                raise RuntimeError(
                    f"A0 completed but {_teacher_cache_file.name} was not written. "
                    "Subsequent ablations would re-run teacher scoring. Halting."
                )
            # Kick off background upload immediately — next ablation starts
            # on the GPU while CPU streams this one to HF Hub.
            if upload_bucket and hf_token:
                t = threading.Thread(
                    target=_upload_ablation_bg,
                    args=(aid, ablations_root / aid, shared_dir,
                          upload_bucket, hf_token, first_upload,
                          _stage6_artifact(args.stage6_mode)),
                    daemon=True,
                    name=f"upload-{aid}",
                )
                t.start()
                upload_threads.append((aid, t))
                first_upload = False
        except Exception as exc:  # noqa: BLE001
            log.exception("[%s] failed", aid)
            failures.append((aid, str(exc)))
            # Continue to the next ablation; failed run is retryable on the
            # next job invocation (the harness is idempotent).

        # Regenerate the leaderboard after EVERY row (success or failure) so a
        # mid-sweep process death leaves completed rows durable on the bucket.
        _persist_progress(ablations_root, results, failures, rows,
                          args.stage6_mode, upload_bucket, hf_token)

    # Wait for all background uploads to finish before writing the summary.
    # Each thread has already started; GPU work is done at this point.
    if upload_threads:
        log.info("Waiting for %d background upload(s) to finish…", len(upload_threads))
        for aid, t in upload_threads:
            t.join(timeout=600)
            if t.is_alive():
                log.warning("[%s] upload thread still alive after 10 min — giving up", aid)

    # Final aggregate. The per-row _persist_progress calls already wrote the
    # summary + leaderboard incrementally; this is the terminal snapshot.
    _persist_progress(ablations_root, results, failures, rows,
                      args.stage6_mode, upload_bucket, hf_token)
    log.info("Wrote summary + leaderboard to %s", ablations_root)

    # Concise dashboard line.
    log.info("=" * 60)
    log.info("Ablation summary (completed=%d, failed=%d):",
             len(results), len(failures))
    for aid, _ in rows:
        if aid in results:
            r = results[aid]
            if args.stage6_mode == "thermometer":
                log.info("  %s : bpt_gap=%s top1_agree=%s", aid,
                         _fmt(r.get("bpt_gap")), _fmt(r.get("top1_agreement")))
            else:
                log.info("  %s : ppl=%s", aid,
                         r.get("student", {}).get("wikitext2_ppl", "?"))
        else:
            log.info("  %s : FAILED or skipped", aid)
    log.info("=" * 60)
    log.info("Leaderboard: %s", ablations_root / "_leaderboard.md")

    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
