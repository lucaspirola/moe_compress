"""HF Jobs entrypoint for Chapter 1 — Structural Recovery at BF16.

UV script (PEP 723). Run via ``hf_jobs/submit.sh``. Two-phase orchestration:

  Phase 1 (optional, BP#9 teacher correction):
    accelerate launch --use_deepspeed --deepspeed_config_file ... \
        -m structural_recovery.teacher_correction \
        --config <yaml> --artifacts-dir <dir>
    → writes ``artifacts/teacher_corrected_bf16/``

  Phase 2 (KD):
    accelerate launch --use_deepspeed --deepspeed_config_file ... \
        -m structural_recovery.run_recovery \
        --config <yaml> --student <repo> --artifacts-dir <dir> \
        [--teacher-source artifacts/teacher_corrected_bf16]
    → writes ``artifacts/chapter1_recovered/``

Each phase is a fresh Python process so DeepSpeed engine state is clean.

Layout (mirrors max_quality/hf_jobs/entrypoint.py):

  /mnt/cache/                    (HF bucket mount, persistent across runs)
    ├── code/                    snapshot of pirola/moe-compress-code
    ├── code_recovery/           snapshot of pirola/structural-recovery-code
    ├── hf_cache/                HF_HOME (model snapshots persist here)
    └── recovery_artifacts/      Chapter 1 outputs

Required env vars (set by submit.sh):
  HF_TOKEN         Read+write on ``pirola`` namespace
  STUDENT_REPO     The result repo from a max_quality run (mandatory)
Optional env vars:
  CACHE_MOUNT      ``/mnt/cache`` (default)
  CODE_REPO        ``pirola/moe-compress-code``
  RECOVERY_REPO    ``pirola/structural-recovery-code``
  CONFIG_PATH      ``configs/qwen36_35b_a3b_chapter1_{smoke,light}.yaml``
  DS_CONFIG_PATH   ``ds_configs/zero3_offload_optim.json``
  RESULT_REPO      Override final upload destination (auto if empty)
  SMOKE            ``1`` → use smoke YAML, single-GPU, no DeepSpeed
  SKIP_TEACHER_CORRECTION  ``1`` → skip Phase 1
"""

# /// script
# requires-python = ">=3.10"
# dependencies = [
#     # Match max_quality's pin (CUDA 12.9 host driver compatibility on HF Jobs).
#     "torch>=2.5.0,<2.11.0",
#     "transformers>=4.57.0",
#     "accelerate>=1.0.0",
#     "datasets>=3.0.0",
#     "safetensors>=0.4.5",
#     "tokenizers>=0.20.0",
#     "sentencepiece>=0.2.0",
#     "huggingface_hub>=0.26.0",
#     "einops>=0.8.0",
#     "numpy>=1.26.0",
#     "scipy>=1.11.0",
#     "bitsandbytes>=0.44.0",
#     "deepspeed>=0.15.0",
#     "nvidia-modelopt>=0.21.0",
#     "pyyaml>=6.0",
# ]
# ///

from __future__ import annotations

import logging
import os
import shutil
import signal
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

LOG = logging.getLogger("hf_jobs.recovery_entrypoint")


CACHE_MOUNT             = Path(os.environ.get("CACHE_MOUNT", "/mnt/cache"))
CODE_REPO               = os.environ.get("CODE_REPO",       "pirola/moe-compress-code")
RECOVERY_REPO           = os.environ.get("RECOVERY_REPO",   "pirola/structural-recovery-code")
STUDENT_REPO            = os.environ.get("STUDENT_REPO",    "")
CONFIG_PATH             = os.environ.get("CONFIG_PATH",     "")
DS_CONFIG_PATH          = os.environ.get("DS_CONFIG_PATH",  "ds_configs/zero3_offload_optim.json")
RESULT_REPO             = os.environ.get("RESULT_REPO",     "")
SMOKE                   = os.environ.get("SMOKE", "0") not in ("0", "false", "False", "")
SKIP_TEACHER            = os.environ.get("SKIP_TEACHER_CORRECTION", "0") not in ("0", "false", "False", "")
TEACHER_CORRECTED_REPO  = os.environ.get("TEACHER_CORRECTED_REPO", "")
ALLOW_SINGLE_GPU        = os.environ.get("ALLOW_SINGLE_GPU", "0") not in ("0", "false", "False", "")


def _main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    LOG.info("========== Chapter 1 — Structural Recovery on HF Jobs ==========")
    LOG.info("STUDENT_REPO=%s  SMOKE=%s  SKIP_TEACHER=%s",
             STUDENT_REPO, SMOKE, SKIP_TEACHER)
    LOG.info("CODE_REPO=%s  RECOVERY_REPO=%s", CODE_REPO, RECOVERY_REPO)
    LOG.info("TEACHER_CORRECTED_REPO=%s", TEACHER_CORRECTED_REPO or "<run Phase 1>")

    _sanity_check()

    code_dir       = CACHE_MOUNT / "code"
    recovery_dir   = CACHE_MOUNT / "code_recovery"
    hf_home        = CACHE_MOUNT / "hf_cache"
    artifacts_dir  = CACHE_MOUNT / "recovery_artifacts"
    for p in (code_dir, recovery_dir, hf_home, artifacts_dir):
        p.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)
    # NOTE: TRANSFORMERS_CACHE is deprecated in transformers>=4.36 and emits
    # a FutureWarning. HF_HOME alone is sufficient — transformers and
    # huggingface_hub both pick up <HF_HOME>/hub for the model snapshot cache.
    LOG.info("HF_HOME=%s  artifacts_dir=%s", hf_home, artifacts_dir)

    # Resolve the result repo name once — _resolve_result_repo() embeds a
    # timestamp, so calling it twice produces different repos for Phase 1
    # failure vs. final upload.
    result_repo = _resolve_result_repo()
    LOG.info("Result repo (resolved once): %s", result_repo)

    # 1. Download both code repos fresh on every run.
    _download_code(CODE_REPO,     code_dir)
    _download_code(RECOVERY_REPO, recovery_dir)

    # 2. Resolve config path (default: smoke vs light).
    config_rel = CONFIG_PATH or (
        "configs/qwen36_35b_a3b_chapter1_smoke.yaml" if SMOKE
        else "configs/qwen36_35b_a3b_chapter1_light.yaml"
    )
    config_arg = str(recovery_dir / config_rel)
    ds_config_arg = str(recovery_dir / DS_CONFIG_PATH)

    # 3. Prime model snapshots (idempotent on cache hit). Read the YAML so we
    #    prime the actual teacher the run will use — light tier uses BF16,
    #    smoke tier uses FP8, and hardcoding either is wrong for the other.
    import yaml
    with open(config_arg) as _f:
        _cfg = yaml.safe_load(_f) or {}
    phase2_teacher = _cfg["teacher"]["name_or_path"]
    _prime_snapshot(phase2_teacher, hf_home)

    # If teacher correction will run, also prime its BF16 source. The source
    # is teacher_correction.bf16_teacher_name_or_path if set; otherwise it's
    # derived by stripping `-FP8` (so a BF16 phase2 teacher passes through
    # unchanged and we skip the redundant prime).
    tcc = _cfg.get("teacher_correction") or {}
    if tcc.get("enabled") and not SKIP_TEACHER and not SMOKE:
        bf16_teacher = (
            tcc.get("bf16_teacher_name_or_path")
            or phase2_teacher.removesuffix("-FP8").removesuffix("-fp8")
        )
        if bf16_teacher != phase2_teacher:
            # If the derived BF16 sibling does not exist on the Hub, fail fast
            # with a clear error rather than letting _prime_snapshot retry-
            # then-fail with an opaque RepositoryNotFoundError after wasting
            # ~2 minutes on retry backoff.
            try:
                from huggingface_hub import HfApi
                HfApi().model_info(bf16_teacher)
            except Exception as err:                                 # noqa: BLE001
                raise RuntimeError(
                    f"Teacher correction is enabled but the derived BF16 "
                    f"teacher repo {bf16_teacher!r} (stripped from "
                    f"{phase2_teacher!r}) is not accessible on the Hub: {err}. "
                    "Either set teacher_correction.bf16_teacher_name_or_path "
                    "in the YAML, or set SKIP_TEACHER_CORRECTION=1."
                ) from None
            _prime_snapshot(bf16_teacher, hf_home)

    _prime_snapshot(STUDENT_REPO, hf_home)

    # 4. PYTHONPATH prepends both code dirs.
    pythonpath = (
        f"{recovery_dir}/src:{code_dir}/src:" + os.environ.get("PYTHONPATH", "")
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = pythonpath

    # 5. Phase 1: Teacher correction (Light tier only — smoke skips).
    teacher_source: str | None = None
    if SMOKE:
        if TEACHER_CORRECTED_REPO:
            LOG.warning(
                "TEACHER_CORRECTED_REPO=%s is set but SMOKE=1 — ignoring. "
                "Smoke uses the FP8 teacher from the YAML; teacher correction "
                "is not part of the smoke path. Unset SMOKE or "
                "TEACHER_CORRECTED_REPO to remove this warning.",
                TEACHER_CORRECTED_REPO,
            )
        LOG.info("SMOKE=1: skipping Phase 1 (teacher correction).")
    elif TEACHER_CORRECTED_REPO:
        LOG.info("TEACHER_CORRECTED_REPO set: downloading Phase 1 output from Hub.")
        _restore_teacher_corrected_checkpoint(TEACHER_CORRECTED_REPO, artifacts_dir)
        teacher_source = str(artifacts_dir / "teacher_corrected_bf16")
    elif SKIP_TEACHER:
        LOG.info("SKIP_TEACHER_CORRECTION=1: skipping Phase 1.")
    else:
        LOG.info("=== Phase 1: Teacher correction ===")
        rc = _run_phase(
            entry="structural_recovery.teacher_correction",
            extra_args=["--artifacts-dir", str(artifacts_dir)],
            config_arg=config_arg,
            ds_config_arg=ds_config_arg,
            env=env,
            use_deepspeed=True,
        )
        if rc != 0:
            LOG.error("Phase 1 (teacher correction) failed (rc=%d).", rc)
            _upload_results(artifacts_dir, result_repo, ok=False)
            return rc
        teacher_source = str(artifacts_dir / "teacher_corrected_bf16")
        if not Path(teacher_source).exists():
            LOG.warning("Phase 1 reported success but %s missing; "
                        "Phase 2 will fall back to teacher.name_or_path "
                        "from the YAML (%s).",
                        teacher_source, phase2_teacher)
            teacher_source = None
        else:
            # Upload Phase 1 output to Hub NOW — synchronously, before Phase 2
            # begins. This is the only durability boundary: if Phase 2 crashes
            # or the job is killed, this upload survives and can be passed as
            # TEACHER_CORRECTED_REPO on the next run to skip Phase 1.
            _upload_phase1_to_hub(artifacts_dir, result_repo)

    # 6. Phase 2: Distillation.
    LOG.info("=== Phase 2: KD distillation ===")
    extra = ["--student", STUDENT_REPO, "--artifacts-dir", str(artifacts_dir)]
    if SMOKE:
        extra.append("--smoke")
    if teacher_source:
        extra.extend(["--teacher-source", teacher_source])
    rc = _run_phase(
        entry="structural_recovery.run_recovery",
        extra_args=extra,
        config_arg=config_arg,
        ds_config_arg=ds_config_arg,
        env=env,
        use_deepspeed=not SMOKE,
    )

    # 7. Upload artifacts (best-effort).
    _upload_results(artifacts_dir, result_repo, ok=(rc == 0))
    return rc


# ---------------------------------------------------------------------------
# Phase runner
# ---------------------------------------------------------------------------


def _run_phase(*, entry: str, extra_args: list[str], config_arg: str,
               ds_config_arg: str, env: dict, use_deepspeed: bool) -> int:
    """Launch one phase as a subprocess. DeepSpeed via accelerate launch.

    Always passes ``--num_processes`` explicitly — accelerate's auto-detect
    is unreliable on HF Jobs (CUDA_VISIBLE_DEVICES isn't always honoured),
    and a silent fallback to single-process under DS3 would corrupt training.
    """
    import torch
    n_gpus = torch.cuda.device_count()

    if use_deepspeed:
        cmd = [
            "accelerate", "launch",
            "--num_processes", str(n_gpus),
            "--num_machines", "1",
            "--use_deepspeed",
            "--deepspeed_config_file", ds_config_arg,
            "--mixed_precision", "bf16",
            "-m", entry,
            "--config", config_arg,
        ] + extra_args
    else:
        # Smoke / single-GPU: skip accelerate, just plain python.
        cmd = [
            sys.executable, "-m", entry,
            "--config", config_arg,
        ] + extra_args

    LOG.info("subprocess (n_gpus=%d, deepspeed=%s): %s",
             n_gpus, use_deepspeed, " ".join(cmd))
    try:
        completed = subprocess.run(cmd, env=env, check=False)
        return completed.returncode
    except BaseException as exc:                                 # noqa: BLE001
        LOG.error("subprocess raised: %s\n%s", exc, traceback.format_exc())
        return 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sanity_check() -> None:
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is not set. Pass --secrets HF_TOKEN.")
    if not CACHE_MOUNT.exists():
        raise RuntimeError(
            f"Expected bucket mount at {CACHE_MOUNT}. Pass "
            "--volume hf://buckets/pirola/moe-cache:/mnt/cache."
        )
    if not STUDENT_REPO:
        raise RuntimeError(
            "STUDENT_REPO is empty. Pass STUDENT_REPO=<the result repo from "
            "your max_quality run> via --env."
        )

    import torch
    avail = torch.cuda.is_available()
    LOG.info("torch=%s cuda=%s avail=%s device_count=%d",
             torch.__version__, getattr(torch.version, "cuda", "?"),
             avail, torch.cuda.device_count())
    if not avail:
        raise RuntimeError(
            "torch.cuda.is_available() is False — refusing to run on CPU."
        )
    if not SMOKE and torch.cuda.device_count() < 2:
        # The Light tier YAML calibrates LR schedule, gradient_accumulation,
        # and total_tokens for 4 GPUs. Running on 1 GPU silently quarters the
        # effective batch size and burns ~$60 of compute on a misconfigured
        # run. Refuse unless the operator explicitly opts in.
        if not ALLOW_SINGLE_GPU:
            raise RuntimeError(
                f"device_count={torch.cuda.device_count()} < 2 with SMOKE=0. "
                "The Light tier expects multi-GPU; the YAML's LR / grad_accum "
                "/ total_tokens are calibrated for 4 GPUs. Either submit with "
                "FLAVOR=a100x4 (or h200x2), or pass ALLOW_SINGLE_GPU=1 to "
                "acknowledge the schedule mismatch."
            )
        LOG.warning(
            "ALLOW_SINGLE_GPU=1 honored; running Light tier on %d GPU(s) — "
            "LR schedule and effective batch size will be off-spec.",
            torch.cuda.device_count(),
        )


def _with_retry(fn, label: str, max_attempts: int = 3) -> None:
    """Call fn() up to max_attempts times with exponential backoff."""
    import time
    delays = [30, 90]
    for attempt in range(1, max_attempts + 1):
        try:
            fn()
            return
        except Exception as exc:  # noqa: BLE001
            if attempt == max_attempts:
                raise
            wait = delays[min(attempt - 1, len(delays) - 1)]
            LOG.warning("%s: attempt %d/%d failed (%s); retrying in %ds",
                        label, attempt, max_attempts, exc, wait)
            time.sleep(wait)


def _download_code(repo_id: str, dest: Path) -> None:
    from huggingface_hub import snapshot_download
    LOG.info("snapshot_download %s -> %s", repo_id, dest)
    if dest.exists():
        for p in dest.iterdir():
            if p.is_dir() and p.name not in ("__pycache__",):
                shutil.rmtree(p, ignore_errors=True)
            else:
                p.unlink(missing_ok=True)

    def _do():
        snapshot_download(
            repo_id, repo_type="dataset", local_dir=dest,
            allow_patterns=[
                "*.py", "*.yaml", "*.yml", "*.json", "*.txt", "*.md",
                # Use ``**`` recursive globs (some hub versions don't recurse on
                # ``configs/*``-style single-star patterns).
                "configs/**", "ds_configs/**", "src/**", "hf_jobs/**", "tests/**",
            ],
        )

    _with_retry(_do, label=f"_download_code({repo_id})")


def _prime_snapshot(repo_id: str, hf_home: Path) -> None:
    """Prime a MODEL snapshot in the HF cache (idempotent)."""
    from huggingface_hub import snapshot_download
    LOG.info("Priming snapshot %s -> %s", repo_id, hf_home / "hub")

    def _do():
        snapshot_download(
            repo_id, repo_type="model",
            cache_dir=hf_home / "hub", allow_patterns=["*"],
        )

    _with_retry(_do, label=f"_prime_snapshot({repo_id})")


def _resolve_result_repo() -> str:
    if RESULT_REPO:
        return RESULT_REPO
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    stem = (STUDENT_REPO.split("/", 1)[-1] if "/" in STUDENT_REPO else STUDENT_REPO).lower()
    tag = "-smoke" if SMOKE else ""
    return f"pirola/{stem}-chapter1{tag}-{ts}"


def _upload_phase1_to_hub(artifacts_dir: Path, result_repo_base: str) -> None:
    """Upload Phase 1 (teacher_corrected_bf16/) to Hub as <base>-phase1.

    Blocks until Hub commit returns 200 — the only durability guarantee across
    job cancellations. Called synchronously after Phase 1 succeeds and BEFORE
    Phase 2 begins so the teacher checkpoint survives a Phase 2 crash.
    """
    from huggingface_hub import HfApi
    repo_id = f"{result_repo_base}-phase1"
    src = artifacts_dir / "teacher_corrected_bf16"
    if not src.exists():
        LOG.warning("_upload_phase1_to_hub: %s missing — skipping.", src)
        return

    api = HfApi()
    try:
        api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
    except Exception as err:  # noqa: BLE001
        LOG.warning("create_repo(%s): %s — continuing.", repo_id, err)

    LOG.info("=== Uploading Phase 1 checkpoint to Hub: %s ===", repo_id)
    api.upload_large_folder(
        folder_path=str(src), repo_id=repo_id, repo_type="model",
    )

    cfg_path = artifacts_dir / "resolved_config.yaml"
    if cfg_path.exists():
        try:
            api.upload_file(
                path_or_fileobj=str(cfg_path),
                path_in_repo="artifacts/resolved_config.yaml",
                repo_id=repo_id, repo_type="model",
            )
        except Exception as err:  # noqa: BLE001
            LOG.warning("Phase 1 config upload failed: %s", err)

    LOG.info("Phase 1 durable on Hub -> https://huggingface.co/%s", repo_id)


def _restore_teacher_corrected_checkpoint(repo_id: str, artifacts_dir: Path) -> None:
    """Download teacher_corrected_bf16/ from Hub. Idempotent on complete cache.

    Guards against partial downloads: only skips the download when the
    ``_SAVE_COMPLETE`` sentinel written by ``_save_bf16`` is present.
    Checking only for the index file is insufficient — a SIGKILL mid-download
    can leave the index but miss shards, causing a cryptic I/O error later.
    """
    from huggingface_hub import snapshot_download
    dest = artifacts_dir / "teacher_corrected_bf16"
    sentinel = dest / "_SAVE_COMPLETE"
    if sentinel.exists():
        LOG.info("teacher_corrected_bf16 complete at %s; skipping download.", dest)
        return
    if dest.exists():
        LOG.warning(
            "%s exists but _SAVE_COMPLETE is absent — prior download was "
            "incomplete. Re-downloading.", dest,
        )
        shutil.rmtree(dest, ignore_errors=True)
    LOG.info("Downloading teacher corrected checkpoint from %s -> %s", repo_id, dest)
    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id, repo_type="model", local_dir=str(dest),
        ignore_patterns=["*.metadata", "job_status.txt", "artifacts/*"],
    )
    LOG.info("Teacher correction checkpoint restored from Hub.")


def _upload_results(artifacts_dir: Path, repo_id: str, *, ok: bool) -> None:
    from huggingface_hub import HfApi
    api = HfApi()
    try:
        api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
    except Exception as err:                                     # noqa: BLE001
        LOG.warning("create_repo(%s): %s — continuing.", repo_id, err)

    final_dir = artifacts_dir / "chapter1_recovered"
    if not final_dir.exists():
        # Fall back to the most recent valid partial. Exclude .tmp dirs (from
        # interrupted atomic renames) and dirs missing _SAVE_COMPLETE.
        partials = sorted(
            [
                p for p in artifacts_dir.glob("chapter1_recovered_partial_step*")
                if p.is_dir()
                and not p.name.endswith(".tmp")
                and (p / "_SAVE_COMPLETE").exists()
            ],
            key=lambda p: int(p.name.split("step")[-1]) if p.name.split("step")[-1].isdigit() else -1,
        )
        if partials:
            LOG.warning("chapter1_recovered missing; uploading %s.", partials[-1])
            final_dir = partials[-1]

    if final_dir.exists():
        try:
            api.upload_large_folder(
                folder_path=str(final_dir), repo_id=repo_id, repo_type="model",
            )
        except Exception as err:  # noqa: BLE001
            LOG.error(
                "upload_large_folder failed: %s — job_status.txt will still be written.",
                err,
            )

    aux = ["resolved_config.yaml", "chapter1_final_metrics.json"]
    for name in aux:
        p = artifacts_dir / name
        if not p.exists():
            continue
        try:
            api.upload_file(
                path_or_fileobj=str(p),
                path_in_repo=f"artifacts/{name}",
                repo_id=repo_id, repo_type="model",
            )
        except Exception as err:                                     # noqa: BLE001
            # Aux files are diagnostic. A transient upload failure must not
            # prevent the terminal-state ``job_status.txt`` write below — the
            # operator's only signal that the job actually finished.
            LOG.warning("aux upload %s failed: %s — continuing.", name, err)

    status_path = artifacts_dir / "_job_status.txt"
    status_path.write_text(
        f"{'SUCCESS' if ok else 'FAILURE'} at "
        f"{datetime.now(timezone.utc).isoformat()}\n"
        f"CODE_REPO={CODE_REPO}\nRECOVERY_REPO={RECOVERY_REPO}\n"
        f"STUDENT_REPO={STUDENT_REPO}\nSMOKE={SMOKE}  "
        f"SKIP_TEACHER={SKIP_TEACHER}\n"
    )
    api.upload_file(
        path_or_fileobj=str(status_path),
        path_in_repo="job_status.txt",
        repo_id=repo_id, repo_type="model",
    )
    LOG.info("Upload complete -> https://huggingface.co/%s", repo_id)


if __name__ == "__main__":
    # SIGTERM handling: HF Jobs sends SIGTERM on cancel/timeout. We exit 143
    # so the orchestrator surfaces the cancellation explicitly rather than as
    # an unrelated exit code. Caveat: this handler is installed on the parent
    # process; in-flight uploads inside ``_upload_results`` (which runs after
    # the accelerate child returns) may be cut mid-shard. Hub commits are
    # atomic per file but not per folder, so a SIGTERM during the final
    # ``upload_large_folder`` may leave a partial repo. The next run's
    # auto-resume re-uploads from the local partial dir.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    sys.exit(_main())
