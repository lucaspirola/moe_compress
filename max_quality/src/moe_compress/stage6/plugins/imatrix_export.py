"""Imatrix export — live owner of the Stage 6 post-eval imatrix / GGUF phase.

Paper / spec source
--------------------
**Imatrix** = importance matrix, the per-channel calibration profile
produced by ``llama-imatrix`` for use by GGUF post-training quantizers
(K/M-quants, IQ-quants). Not a single paper; the ``llama-imatrix``
tool ships with llama.cpp upstream.

Stage 6 implementation note: convert the student to F16 GGUF (the
quantization-input format), run ``llama-imatrix`` against the
WikiText-2-train calibration corpus
(:mod:`stage6.plugins.eval_environment` builds the corpus), and write
the ``eval_text_concat.txt`` debug side-channel.

Gated by ``imatrix.enabled`` (default ON when the ``imatrix`` subdict is
present or omitted; opt-out by setting ``imatrix.enabled: false`` in the
Stage 6 config). The default-True behaviour matches the production YAML
contract (``configs/qwen36_35b_a3b_30pct.yaml`` sets it explicitly).

Reference code
--------------
``ggerganov/llama.cpp`` — ``llama-imatrix`` CLI; standard library, no
project-pinned SHA. Invoked as a subprocess.

This plugin is the LIVE owner of the Stage 6 post-eval imatrix / GGUF
pipeline. The orchestrator walks ``start_gguf_convert`` (cache-MISS
only, before the teacher-side eval loop) and later ``export_imatrix``
(after evals complete) — see ``stage6/orchestrator.py``. Together they
convert the student checkpoint to F16 GGUF, run ``llama-imatrix``
against the WikiText-2-train calibration corpus, and write the
``eval_text_concat.txt`` debug side-channel.

Hook layout
-----------
* ``start_gguf_convert(ctx)`` — kicks off the background F16-GGUF
  conversion thread (Optimization #8) so the CPU-bound conversion
  overlaps with the GPU-bound teacher eval. Publishes ``gguf_thread``
  and ``gguf_result`` to ``ctx``.
* ``export_imatrix(ctx)`` — joins the bg thread, dispatches one of
  three imatrix branches (skip / prebuilt-GGUF / sequential fallback),
  and unconditionally writes ``eval_text_concat.txt``.

The module-level helpers (``_background_gguf_convert``,
``_write_eval_text_concat``, ``_run_llama_imatrix_with_prebuilt_gguf``,
``_generate_imatrix``, ``_find_llama_cpp_dir``) are subprocess-driving
implementation details consumed by the hooks above; they are kept
module-private (PEP 8 leading-underscore) and are not part of the
public surface.

Circular-import contract (mirror of ``stage6/plugins/teacher_provider.py``):
this module imports only from ``..context`` / ``...utils`` / sibling
plugin modules (``eval_environment``) / stdlib.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from ..context import PipelineContext
from ...utils.trackio_log import trackio_log as _trackio_log
from .eval_environment import _IMATRIX_CALIB_FILENAME, _atomic_write_text

log = logging.getLogger(__name__)


# F-C-C-1: Spec §9 -- the eval-text concat (eval prompts seen by the model
# during PPL/zero-shot/generative) is captured to eval_text_concat.txt as a
# debugging side-channel ONLY. The imatrix calibration-corpus filename
# (_IMATRIX_CALIB_FILENAME) lives in stage6/plugins/eval_environment and is
# imported above. Single source of truth for the eval-text-concat filename.
_EVAL_TEXT_CONCAT_FILENAME: str = "eval_text_concat.txt"


# ---------------------------------------------------------------------------
# Background GGUF conversion (Optimization #8)
# ---------------------------------------------------------------------------

def _background_gguf_convert(icfg: dict, artifacts_dir: Path, result: dict) -> None:
    """Convert the student model to F16 GGUF in background (CPU-bound)."""
    if not icfg.get("enabled", True):
        return

    llama_cpp_dir = _find_llama_cpp_dir(icfg.get("llama_cpp_dir"))
    if llama_cpp_dir is None:
        log.warning("GGUF convert (background): llama.cpp not found — skipping.")
        return

    convert_py = llama_cpp_dir / "convert_hf_to_gguf.py"
    if not convert_py.exists():
        log.warning("GGUF convert (background): convert script missing — skipping.")
        return

    model_dir = artifacts_dir / "stage5_final"
    if not model_dir.exists():
        log.warning("GGUF convert (background): stage5_final not found — skipping.")
        return

    # L2/L3: free-space check is parametrized (default 40 GB ≈ Llama-3-8B F16
    # headroom; raise for larger MoE F16 GGUFs via icfg.min_free_gb) and
    # tolerant of artifacts_dir not yet existing (fall back to parent).
    disk_root = artifacts_dir if artifacts_dir.exists() else artifacts_dir.parent
    free_gb = shutil.disk_usage(str(disk_root)).free / 1e9
    min_free_gb = float(icfg.get("min_free_gb", 40))
    if free_gb < min_free_gb:
        log.warning(
            "GGUF convert (background): only %.1f GB free (< min_free_gb=%.1f) — skipping.",
            free_gb, min_free_gb,
        )
        return

    f16_path = artifacts_dir / "model_f16.gguf"
    f16_tmp = artifacts_dir / "model_f16.gguf.tmp"
    log.info("GGUF convert (background): %s → F16 GGUF", model_dir)
    env = {
        **os.environ,
        "LD_LIBRARY_PATH": str(llama_cpp_dir / "build" / "bin") + ":" + os.environ.get("LD_LIBRARY_PATH", ""),
    }
    try:
        t0 = time.monotonic()
        stderr_log = artifacts_dir / "gguf_convert_stderr.log"
        with open(stderr_log, "w") as _stderr_fh:
            subprocess.run(
                [sys.executable, str(convert_py), str(model_dir),
                 "--outtype", "f16", "--outfile", str(f16_tmp)],
                env=env, check=True, timeout=3600, stderr=_stderr_fh,
            )
        os.replace(f16_tmp, f16_path)
        dt = time.monotonic() - t0
        result["f16_path"] = f16_path
        log.info("GGUF convert (background): done in %.1fs (%.1f GB)",
                 dt, f16_path.stat().st_size / 1e9)
    except subprocess.TimeoutExpired as exc:
        log.error("GGUF convert (background): timed out after 3600s (%s)", exc)
        f16_tmp.unlink(missing_ok=True)
        return
    except subprocess.CalledProcessError as exc:
        stderr_snippet = ""
        if stderr_log.exists():
            try:
                stderr_snippet = stderr_log.read_text(errors="replace")[-2000:]
            except Exception:
                pass
        log.warning("GGUF convert (background): failed (%s): %s", exc, stderr_snippet)
        f16_tmp.unlink(missing_ok=True)
        return
    except Exception as exc:
        log.warning("GGUF convert (background): failed (%s)", exc)
        f16_tmp.unlink(missing_ok=True)
        return


def _write_eval_text_concat(texts: list[str], artifacts_dir: Path) -> Path:
    """Write the eval-text concat (debug side-channel) atomically.

    F-C-C-1: this file is the concatenation of every prompt seen by the model
    during PPL/zero-shot/generative evals. It is NOT the imatrix calibration
    corpus — that is `calibration_wiki_train.txt` per Spec §9.
    """
    path = artifacts_dir / _EVAL_TEXT_CONCAT_FILENAME
    joined = "\n\n".join(t.strip() for t in texts if t and t.strip())
    _atomic_write_text(path, joined)
    log.info(
        "eval_text_concat: %d docs (%d chars) → %s (debug only — not used by imatrix)",
        len(texts), len(joined), path,
    )
    return path


def _run_llama_imatrix_with_prebuilt_gguf(
    eval_text_concat: list[str], icfg: dict, artifacts_dir: Path, gguf_result: dict,
) -> None:
    """Run llama-imatrix using the pre-built F16 GGUF from background thread.

    F-C-C-1: imatrix calibration corpus is the WikiText-2 *train* split
    (calibration_wiki_train.txt). The eval_text_concat list captured during
    PPL/zero-shot/generative evals is written as a debugging side-channel
    (eval_text_concat.txt) but is NOT used by llama-imatrix.
    """
    # Always write eval_text_concat.txt as a debug artifact.
    _write_eval_text_concat(eval_text_concat, artifacts_dir)

    if not icfg.get("enabled", True):
        return

    # F-C-C-1: imatrix calibration source is the WikiText-2 *train* split.
    calib_path = artifacts_dir / _IMATRIX_CALIB_FILENAME
    if not calib_path.exists() or calib_path.stat().st_size == 0:
        log.warning(
            "imatrix: calibration corpus %s missing/empty; skipping imatrix generation. "
            "Spec §9 requires this file (WikiText-2 train split). "
            "It is built automatically at the top of run() — check the earlier warning "
            "from _build_imatrix_calibration_corpus.",
            calib_path,
        )
        return

    f16_path = gguf_result.get("f16_path")
    if f16_path is None or not f16_path.exists():
        # M1: safe fall-through. By contract, export_imatrix only invokes this
        # prebuilt path AFTER the bg thread has been joined; if it timed out
        # with the thread still alive, export_imatrix takes the skip-imatrix
        # branch instead (see imatrix_skipped sentinel). So reaching here
        # means the bg thread is DEAD but produced no f16_path (failed /
        # disabled / no llama.cpp). Sequential _generate_imatrix is the only
        # writer of model_f16.gguf in that case — no concurrent-writer race.
        log.warning(
            "imatrix: pre-built GGUF not available (bg thread joined without "
            "producing f16_path) — falling back to sequential _generate_imatrix"
        )
        _generate_imatrix(eval_text_concat, icfg, artifacts_dir)
        return

    llama_cpp_dir = _find_llama_cpp_dir(icfg.get("llama_cpp_dir"))
    if llama_cpp_dir is None:
        log.warning("llama_cpp_dir not found; skipping imatrix generation via prebuilt GGUF")
        return

    imatrix_bin = llama_cpp_dir / "build" / "bin" / "llama-imatrix"
    if not imatrix_bin.exists():
        log.warning("imatrix: llama-imatrix binary not found — skipping.")
        return

    imatrix_out = artifacts_dir / "imatrix.gguf"
    ngl = int(icfg.get("ngl", 99))
    ctx = int(icfg.get("ctx_size", 2048))
    env = {
        **os.environ,
        "LD_LIBRARY_PATH": str(llama_cpp_dir / "build" / "bin") + ":" + os.environ.get("LD_LIBRARY_PATH", ""),
    }
    log.info("imatrix: running llama-imatrix (ngl=%d, ctx=%d) → %s", ngl, ctx, imatrix_out)
    imatrix_stderr_log = artifacts_dir / "llama_imatrix_stderr.log"
    try:
        with open(imatrix_stderr_log, "w") as _stderr_fh:
            subprocess.run(
                [str(imatrix_bin),
                 "-m", str(f16_path), "-f", str(calib_path),
                 "-o", str(imatrix_out), "--output-format", "gguf",
                 "--no-ppl", "-ngl", str(ngl), "-c", str(ctx)],
                env=env, check=True, timeout=7200, stderr=_stderr_fh,
            )
        log.info("imatrix: saved (%.1f MB)", imatrix_out.stat().st_size / 1e6)
    except subprocess.TimeoutExpired as exc:
        log.warning("imatrix subprocess timed out after %ss; skipping imatrix", exc.timeout)
        return
    except subprocess.CalledProcessError as exc:
        stderr_snippet = ""
        if imatrix_stderr_log.exists():
            try:
                stderr_snippet = imatrix_stderr_log.read_text(errors="replace")[-2000:]
            except Exception:
                pass
        log.warning("imatrix: llama-imatrix failed (%s): %s. Calibration text at %s.",
                    exc, stderr_snippet, calib_path)
    except Exception as exc:
        log.warning("imatrix: llama-imatrix failed (%s). Calibration text at %s.", exc, calib_path)


# ---------------------------------------------------------------------------
# imatrix calibration + GGUF conversion (full sequential path)
# ---------------------------------------------------------------------------


def _generate_imatrix(eval_text_concat: list[str], icfg: dict, artifacts_dir: Path) -> None:
    """Sequential GGUF + llama-imatrix fallback path.

    F-C-C-1: imatrix calibration corpus is `calibration_wiki_train.txt` (the
    WikiText-2 train split written at the top of run()). The eval_text_concat
    list captured during evals is written as a debug-only side-channel.
    """
    # Always write the eval-text concat as a debug artifact, regardless of imatrix enable.
    _write_eval_text_concat(eval_text_concat, artifacts_dir)

    if not icfg.get("enabled", True):
        log.info("imatrix: disabled via config.")
        return

    calib_path = artifacts_dir / _IMATRIX_CALIB_FILENAME
    if not calib_path.exists() or calib_path.stat().st_size == 0:
        log.warning(
            "imatrix: calibration corpus %s missing/empty; skipping imatrix generation. "
            "Spec §9 requires this file (WikiText-2 train split).",
            calib_path,
        )
        return

    llama_cpp_dir = _find_llama_cpp_dir(icfg.get("llama_cpp_dir"))
    if llama_cpp_dir is None:
        log.warning("imatrix: llama.cpp not found; skipping imatrix generation.")
        return

    imatrix_bin = llama_cpp_dir / "build" / "bin" / "llama-imatrix"
    convert_py  = llama_cpp_dir / "convert_hf_to_gguf.py"
    if not imatrix_bin.exists() or not convert_py.exists():
        log.warning("imatrix: binaries missing under %s; skipping.", llama_cpp_dir)
        return

    model_dir = artifacts_dir / "stage5_final"
    if not model_dir.exists():
        log.warning("imatrix: stage5_final not found at %s; skipping.", model_dir)
        return

    # L2/L3: parametrized free-space threshold + tolerant of artifacts_dir
    # not yet existing (fall back to parent for the disk_usage probe).
    disk_root = artifacts_dir if artifacts_dir.exists() else artifacts_dir.parent
    free_gb = shutil.disk_usage(str(disk_root)).free / 1e9
    min_free_gb = float(icfg.get("min_free_gb", 40))
    if free_gb < min_free_gb:
        log.warning(
            "imatrix: only %.1f GB free (< min_free_gb=%.1f); skipping GGUF conversion.",
            free_gb, min_free_gb,
        )
        return

    f16_path = artifacts_dir / "model_f16.gguf"
    f16_tmp = artifacts_dir / "model_f16.gguf.tmp"
    env = {
        **os.environ,
        "LD_LIBRARY_PATH": str(llama_cpp_dir / "build" / "bin") + ":" + os.environ.get("LD_LIBRARY_PATH", ""),
    }
    log.info("imatrix: converting %s → F16 GGUF", model_dir)
    stderr_log = artifacts_dir / "gguf_convert_stderr.log"
    try:
        f16_tmp.unlink(missing_ok=True)
        with open(stderr_log, "w") as stderr_fh:
            subprocess.run(
                [sys.executable, str(convert_py), str(model_dir),
                 "--outtype", "f16", "--outfile", str(f16_tmp)],
                env=env, check=True, timeout=3600, stderr=stderr_fh,
            )
        os.replace(f16_tmp, f16_path)
        log.info("imatrix: GGUF ready (%.1f GB)", f16_path.stat().st_size / 1e9)
    except subprocess.TimeoutExpired as exc:
        f16_tmp.unlink(missing_ok=True)
        log.warning("imatrix: GGUF conversion timed out after %ss: %s; skipping.", exc.timeout, exc)
        return
    except subprocess.CalledProcessError as exc:
        f16_tmp.unlink(missing_ok=True)
        try:
            tail = stderr_log.read_text()[-2000:]
        except Exception:
            tail = ""
        log.warning("imatrix: GGUF conversion failed (%s): %s; skipping.", exc, tail)
        return
    except Exception as exc:  # noqa: BLE001
        f16_tmp.unlink(missing_ok=True)
        log.warning("imatrix: GGUF conversion failed (%s); skipping.", exc)
        return

    imatrix_out = artifacts_dir / "imatrix.gguf"
    ngl = int(icfg.get("ngl", 99))
    ctx = int(icfg.get("ctx_size", 2048))
    log.info("imatrix: running llama-imatrix (ngl=%d, ctx=%d) → %s", ngl, ctx, imatrix_out)
    imatrix_stderr_log = artifacts_dir / "llama_imatrix_stderr.log"
    try:
        with open(imatrix_stderr_log, "w") as stderr_fh:
            subprocess.run(
                [str(imatrix_bin),
                 "-m", str(f16_path), "-f", str(calib_path),
                 "-o", str(imatrix_out), "--output-format", "gguf",
                 "--no-ppl", "-ngl", str(ngl), "-c", str(ctx)],
                env=env, check=True, timeout=7200, stderr=stderr_fh,
            )
        log.info("imatrix: saved (%.1f MB)", imatrix_out.stat().st_size / 1e6)
    except subprocess.TimeoutExpired as exc:
        log.error("imatrix: llama-imatrix timed out after 7200s (%s). Calibration text at %s.", exc, calib_path)
        return
    except subprocess.CalledProcessError as exc:
        try:
            tail = imatrix_stderr_log.read_text()[-2000:]
        except Exception:
            tail = ""
        log.warning("imatrix: llama-imatrix failed (%s): %s. Calibration text at %s.", exc, tail, calib_path)
    except Exception as exc:  # noqa: BLE001
        log.warning("imatrix: llama-imatrix failed (%s). Calibration text at %s.", exc, calib_path)


def _find_llama_cpp_dir(override: str | None = None) -> Path | None:
    candidates: list[Path] = []
    if override:
        candidates.append(Path(override))
    env_dir = os.environ.get("LLAMA_CPP_DIR")
    if env_dir:
        candidates.append(Path(env_dir))
    on_path = shutil.which("llama-imatrix")
    if on_path:
        candidates.append(Path(on_path).parent.parent.parent)
    # No further fallback beyond the three candidates above (config, env var, PATH search).

    for p in candidates:
        if (p / "build" / "bin" / "llama-imatrix").exists() and (p / "convert_hf_to_gguf.py").exists():
            return p
    return None


class ImatrixExportPlugin:
    """Stage 6 imatrix-export plugin -- live owner of the post-eval pipeline.

    Owns the Stage 6 post-eval imatrix / GGUF pipeline:

    * Background F16-GGUF conversion that overlaps with teacher eval
      (Optimization #8), kicked off by ``start_gguf_convert``.
    * Late-phase llama-imatrix dispatch (with prebuilt-GGUF +
      sequential fallback branches), dispatched by ``export_imatrix``.
    * Unconditional eval-text-concat debug write (spec §9).

    The orchestrator walks ``start_gguf_convert`` on the cache-MISS path
    before the teacher-side eval loop, and walks ``export_imatrix``
    after evals complete — see ``stage6/orchestrator.py``.
    """

    name = "imatrix_export"
    paper = "Stage 6 imatrix / GGUF export — llama.cpp llama-imatrix; opt-out via imatrix.enabled=false (default ON). See module docstring."
    config_key = "stage6_validate.imatrix.enabled"
    reads: tuple[str, ...] = (
        "config", "artifacts_dir", "eval_text_concat", "cached_teacher_results",
        # Intra-plugin handoff slots: start_gguf_convert WRITES these, then
        # export_imatrix READS them. Declared in both `reads` and `writes` so
        # the framework's data-dependency contract (plugin.py docstring) is
        # accurate even for handoffs that stay within one plugin across phases.
        "gguf_thread", "gguf_result",
    )
    writes: tuple[str, ...] = (
        "gguf_thread", "gguf_result", "imatrix_skipped",
    )
    # gguf_thread/gguf_result are run-scoped handles the early hook publishes
    # for the late hook to consume; imatrix_skipped is the F-CR2-M-1 sentinel
    # surfaced to the artifact / trackio. None is a calibration-pass
    # accumulator, so `provides` is empty -- same convention as the sibling
    # plugins.
    provides: tuple[str, ...] = ()

    def is_enabled(self, config: dict) -> bool:
        """Gate on ``stage6_validate.imatrix.enabled`` (default ON).

        A Stage 6 config that omits the ``imatrix`` subdict (or omits the
        ``enabled`` key inside it) triggers the GGUF + llama-imatrix
        pipeline; opt out by setting ``imatrix.enabled: false``. The
        production YAML (``configs/qwen36_35b_a3b_30pct.yaml``) sets the
        key explicitly to ``true``, so the in-code default only affects
        users who omit the subdict entirely (smoke tests, ad-hoc configs).
        """
        return bool(
            (config.get("stage6_validate", {}) or {})
            .get("imatrix", {})
            .get("enabled", True)
        )

    def contribute_artifact(self, ctx: Any) -> dict:
        return {}

    def start_gguf_convert(self, ctx: PipelineContext) -> None:
        """Phase hook -- early-phase background F16-GGUF conversion kickoff.

        Dispatched by ``stage6/orchestrator.py`` on the cache-MISS path,
        immediately before the teacher-side eval loop. The CPU-bound
        GGUF conversion runs concurrently with the GPU-bound teacher eval
        (Optimization #8); the resulting ``f16_path`` is consumed by the
        late ``export_imatrix`` hook.

        Behaviour::

            if s6.get("imatrix", {}).get("enabled", True):
                gguf_thread = threading.Thread(
                    target=_background_gguf_convert,
                    args=(s6.get("imatrix", {}), artifacts_dir, gguf_result),
                    daemon=True,
                    name="gguf-convert",
                )
                gguf_thread.start()

        The orchestrator already gates this hook by the teacher-cache
        result (only invoked on cache MISS), so the body does NOT
        re-check ``cached_teacher_results``.

        The thread handle + the result dict are published to ctx so
        ``export_imatrix`` can join the thread and read ``f16_path``.
        """
        config = ctx.get("config")
        artifacts_dir = ctx.get("artifacts_dir")
        s6 = config["stage6_validate"]

        # Inner gate (default ON): when imatrix is disabled in config we
        # publish empty handles so ``export_imatrix`` can short-circuit
        # consistently. ``is_enabled`` is the registry-level gate (whole
        # plugin); this inner check covers configs where the orchestrator
        # invokes the hook anyway (e.g. is_enabled is bypassed in tests).
        if not s6.get("imatrix", {}).get("enabled", True):
            ctx.set("gguf_thread", None)
            ctx.set("gguf_result", {})
            return

        gguf_result: dict = {}
        gguf_thread = threading.Thread(
            target=_background_gguf_convert,
            args=(s6.get("imatrix", {}), artifacts_dir, gguf_result),
            daemon=True,
            name="gguf-convert",
        )
        gguf_thread.start()
        log.info("Stage 6: GGUF conversion started in background (CPU-bound)")
        ctx.set("gguf_thread", gguf_thread)
        ctx.set("gguf_result", gguf_result)

    def export_imatrix(self, ctx: PipelineContext) -> None:
        """Phase hook -- late-phase imatrix dispatch.

        Dispatched by ``stage6/orchestrator.py`` after the eval loop has
        completed (GPU is now free from the teacher). The hook:

        1. **Wait for the background GGUF thread** -- ``gguf_thread.join(
           timeout=3700)``. If the thread is still alive at timeout AND
           the bg thread has not already published a valid ``f16_path``,
           set the F-CR2-M-1 ``imatrix_skipped`` sentinel and surface to
           trackio. If ``f16_path`` is published AND exists on disk we
           treat the thread as effectively done (the bg-thread contract
           is: ``f16_path`` is set only after ``os.replace`` of the tmp
           file completes — no race possible).
        2. **Imatrix-skipped path** -- write the eval-text-concat debug
           artifact (unconditional per spec §9) and return.
        3. **Prebuilt-GGUF path** -- when the teacher cache MISSED AND
           the background thread produced an ``f16_path``, invoke
           ``_run_llama_imatrix_with_prebuilt_gguf``.
        4. **Sequential-fallback path** -- otherwise (cache HIT, or
           bg thread failed) invoke ``_generate_imatrix``; its internal
           ``enabled`` guard short-circuits when imatrix is disabled.

        L1: if ``_find_llama_cpp_dir`` returns ``None`` along the fall-
        through path we set ``imatrix_skipped`` for observability parity
        with the bg-thread-timeout sentinel.

        Optional ctx slots:
          * ``eval_text_concat`` (list[str] -- empty list if missing)
          * ``cached_teacher_results`` (dict | None -- None if missing,
            i.e. treat as cache MISS for the gating)
          * ``gguf_thread`` (Thread | None -- None if ``start_gguf_convert``
            short-circuited or didn't run)
          * ``gguf_result`` (dict -- empty dict if missing)
        """
        config = ctx.get("config")
        artifacts_dir = ctx.get("artifacts_dir")
        s6 = config["stage6_validate"]

        eval_text_concat = (
            ctx.get("eval_text_concat") if ctx.has("eval_text_concat") else []
        )
        cached_teacher_results = (
            ctx.get("cached_teacher_results")
            if ctx.has("cached_teacher_results") else None
        )
        gguf_thread = ctx.get("gguf_thread") if ctx.has("gguf_thread") else None
        gguf_result = ctx.get("gguf_result") if ctx.has("gguf_result") else {}

        log.info("Stage 6: starting post-eval imatrix pipeline")
        # Optimization #8: If GGUF conversion was running in background, wait for it.
        # Then run llama-imatrix (which needs the GPU, now freed from teacher).
        imatrix_skipped = False
        if gguf_thread is not None:
            log.info("Stage 6: waiting for background GGUF conversion to complete")
            gguf_thread.join(timeout=3700)
            if gguf_thread.is_alive():
                # M2: bg thread contract — f16_path is set only AFTER os.replace
                # of the tmp file completes, so a populated+on-disk f16_path
                # means the conversion is effectively done even if the Python
                # Thread object hasn't fully wound down. Only skip when there
                # is no usable f16_path.
                published = gguf_result.get("f16_path")
                if published is not None and published.exists():
                    log.warning(
                        "GGUF convert thread still flagged alive after %.0f s, "
                        "but f16_path=%s is populated and exists on disk — "
                        "treating bg thread as effectively done and proceeding "
                        "with prebuilt-GGUF path.",
                        3700, published,
                    )
                else:
                    # F-CR2-M-1: SKIP imatrix entirely when the bg thread is still
                    # alive AND has not published a usable f16_path. The daemon bg
                    # thread continues writing to model_f16.gguf.tmp and would race
                    # with _generate_imatrix's sequential fallback, both of which
                    # call os.replace on the same target path. By skipping, no
                    # concurrent writer exists; the bg thread's eventual replace
                    # just updates the GGUF for the next run.
                    log.error(
                        "GGUF convert thread still alive after %.0f s timeout AND no "
                        "f16_path published; SKIPPING imatrix entirely to avoid "
                        "concurrent-writer race on model_f16.gguf",
                        3700,
                    )
                    imatrix_skipped = True
        f16_path = None if imatrix_skipped else gguf_result.get("f16_path")

        if imatrix_skipped:
            # Sentinel: surface to dashboard via trackio. Do NOT call _generate_imatrix:
            # it would spawn a sequential GGUF write that races the still-live bg thread.
            _trackio_log({"stage6/imatrix_skipped": 1.0})
            ctx.set("imatrix_skipped", True)
            # eval_text_concat.txt is an unconditional debug side-channel per spec
            # §9 -- write it even on the skipped-imatrix path so the dashboard
            # has the captured prompts available for triage.
            try:
                _write_eval_text_concat(eval_text_concat, artifacts_dir)
            except Exception as exc:  # noqa: BLE001
                log.warning("imatrix-skipped path: eval_text_concat write failed (%s)", exc)
            return

        # L1: observability parity — if the llama.cpp binaries are missing we
        # cannot run imatrix at all. Surface that as ``imatrix_skipped`` on the
        # context (same sentinel emitted by the bg-thread-timeout path above).
        # We only set the sentinel; the eval-text-concat write happens inside
        # the helper paths below regardless of imatrix availability.
        icfg = s6.get("imatrix", {})
        if icfg.get("enabled", True) and _find_llama_cpp_dir(icfg.get("llama_cpp_dir")) is None:
            log.warning(
                "imatrix: llama.cpp not found at any candidate location — "
                "setting imatrix_skipped sentinel for observability parity."
            )
            _trackio_log({"stage6/imatrix_skipped": 1.0})
            ctx.set("imatrix_skipped", True)
            # Still proceed below: the helper paths will short-circuit on the
            # same llama.cpp-missing check and will produce the eval_text_concat
            # debug write per spec §9.

        if cached_teacher_results is None and f16_path is not None:
            _run_llama_imatrix_with_prebuilt_gguf(
                eval_text_concat, s6.get("imatrix", {}), artifacts_dir, gguf_result,
            )
        else:
            # This else covers two sub-cases:
            #   (a) Teacher was cached -- no background GGUF conversion was started, so
            #       gguf_result is empty and we fall through here. _generate_imatrix
            #       performs its own GGUF conversion sequentially if imatrix is enabled;
            #       if imatrix is disabled it returns immediately via its `enabled` guard.
            #   (b) Background GGUF conversion was started but failed/produced no output --
            #       cached_teacher_results is None but gguf_result has no f16_path.
            #       _generate_imatrix will retry the full GGUF + imatrix pipeline.
            # In both cases _generate_imatrix's internal `enabled` guard ensures we do
            # nothing unnecessary when imatrix is disabled in config.
            _generate_imatrix(eval_text_concat, s6.get("imatrix", {}), artifacts_dir)


# N3: the leading-underscore helpers below are module-private per PEP 8
# but remain in ``__all__`` for legacy ``stage6_validate`` monolith
# re-import compat (see ``stage6_validate.py`` re-export block) and for
# the unit-test suite that imports them by name from this module. They
# should be dropped from ``__all__`` when the monolith re-export is
# removed; the public surface is ``ImatrixExportPlugin`` alone.
__all__ = [
    "_EVAL_TEXT_CONCAT_FILENAME",
    "_background_gguf_convert",
    "_write_eval_text_concat",
    "_run_llama_imatrix_with_prebuilt_gguf",
    "_generate_imatrix",
    "_find_llama_cpp_dir",
    "ImatrixExportPlugin",
]
