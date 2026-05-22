"""Imatrix export (S6-6 of the Stage 6 plugin-architecture refactor).

Home of the Stage 6 imatrix / GGUF concern, extracted from the legacy
``stage6_validate.py`` monolith. The imatrix-export plugin owns the
post-eval pipeline that converts the student checkpoint to F16 GGUF,
runs ``llama-imatrix`` against the WikiText-2-train calibration corpus,
and writes the ``eval_text_concat.txt`` debug side-channel.

Pattern A vs Pattern B
----------------------
S6-6 covers a MIXED pattern (mirror of S6-5):

* **Pattern A -- relocated verbatim**: ``_EVAL_TEXT_CONCAT_FILENAME``,
  ``_background_gguf_convert``, ``_write_eval_text_concat``,
  ``_run_llama_imatrix_with_prebuilt_gguf``, ``_generate_imatrix`` and
  ``_find_llama_cpp_dir`` below are character-identical copies of the
  monolith bodies. ``stage6_validate.py`` re-imports the 5 FUNCTIONS
  (a ``# noqa: F401`` block) so ``run()`` and external callers/tests
  keep their original import path. The ``_EVAL_TEXT_CONCAT_FILENAME``
  constant is NOT re-imported by the monolith: only the relocated
  ``_write_eval_text_concat`` references it, and the plugin module's
  module-local copy is the single source of truth.
* **Pattern B -- reproduced in TWO inert hooks**: the imatrix pipeline
  in the monolith ``run()`` is split across two phases -- an EARLY
  kickoff (``threading.Thread(target=_background_gguf_convert,
  ...).start()`` immediately before the teacher-side eval loop) and a
  LATE join + llama-imatrix dispatch + eval-text-concat write
  (``gguf_thread.join(...)`` then one of the three imatrix branches
  followed by ``_write_eval_text_concat``). Both blocks are INLINE
  ``run()`` code in the monolith -- there is nothing standalone to
  relocate. The ``start_gguf_convert`` and ``export_imatrix`` hooks
  below REPRODUCE those inline blocks faithfully; the monolith
  ``run()`` is NOT modified for them. This is an intentional,
  temporary logic duplication that resolves at S6-8 when the monolith
  ``run()`` is deleted and these hooks are wired live.

Circular-import contract (mirror of ``stage6/plugins/teacher_provider.py``):
this module imports only from ``..context`` / ``...utils`` / sibling
plugin modules (``eval_environment``) / stdlib -- NEVER from
``stage6_validate``, ``stage6.orchestrator`` or ``orchestrator`` at any
scope (module-top OR function-local). The monolith re-imports *this*
module at load time, so a ``from ..stage6_validate import ...`` here
would deadlock the import; nothing in this module does that.

``ImatrixExportPlugin`` is registered-but-INERT at S6-6 -- no orchestrator
walk or test invokes its ``start_gguf_convert`` or ``export_imatrix``
hooks. S6-8 plugs the hooks into the live Stage 6 plugin sequencer and
deletes the monolith ``run()``.
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
# imported above. Module-LOCAL constant -- the monolith does NOT re-import it
# (only _write_eval_text_concat references it, and that function is also
# relocated here, so the single source of truth lives in this module).
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

    free_gb = shutil.disk_usage(artifacts_dir).free / 1e9
    if free_gb < 40:
        log.warning("GGUF convert (background): only %.1f GB free — skipping.", free_gb)
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
        log.warning("imatrix: pre-built GGUF not available — falling back to full pipeline")
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

    free_gb = shutil.disk_usage(artifacts_dir).free / 1e9
    if free_gb < 40:
        log.warning("imatrix: only %.1f GB free; skipping GGUF conversion.", free_gb)
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
    """Stage 6 imatrix-export plugin (S6-6 -- registered-but-INERT).

    Owns the Stage 6 post-eval imatrix / GGUF pipeline: the background
    F16-GGUF conversion that overlaps with teacher eval (Optimization #8),
    the late-phase llama-imatrix dispatch (with prebuilt-GGUF + sequential
    fallback branches), and the eval-text-concat debug write. The
    standalone helpers (Pattern A) are relocated verbatim above and
    re-imported by the monolith; the ordering glue around them (the early
    thread-start + the late join + dispatch + concat write) is reproduced
    in the ``start_gguf_convert`` and ``export_imatrix`` hooks below
    (Pattern B).

    S6-6 wires this class into the plugin registry as metadata only -- no
    orchestrator walk or test invokes ``start_gguf_convert`` /
    ``export_imatrix``. S6-8 plugs the hooks into the live Stage 6 plugin
    sequencer and deletes the monolith ``run()``.
    """

    name = "imatrix_export"
    paper = (
        "imatrix-guided quantisation (llama.cpp `llama-imatrix` against the "
        "WikiText-2 train split per Spec §9; Stage 6 post-eval export -- "
        "Optimization #8 overlaps GGUF conversion with teacher eval)."
    )
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
        """Gate on ``stage6_validate.imatrix.enabled`` (default True).

        Defaults to ``True`` to faithfully reproduce the monolith ``run()``'s
        per-call-site default at line ``if s6.get("imatrix", {}).get("enabled",
        True):`` -- a Stage 6 config that omits the ``imatrix`` subdict
        triggers the GGUF + llama-imatrix pipeline exactly as the monolith
        does. At S6-8 the registry-level gate must match the monolith's
        behavior for byte-identical-by-construction wiring; "safer defaults"
        belong at the YAML/config layer, not the plugin metadata.
        """
        return bool(
            (config.get("stage6_validate", {}) or {})
            .get("imatrix", {})
            .get("enabled", True)
        )

    def contribute_artifact(self, ctx: Any) -> dict:
        return {}

    def start_gguf_convert(self, ctx: PipelineContext) -> None:
        """Phase hook -- early-phase GGUF thread kickoff (S6-8 wiring surface).

        INERT at S6-6: no orchestrator walk or test invokes this hook. S6-8
        replaces the Stage 6 orchestrator body with the plugin sequencer and
        dispatches this hook in place of the monolith's inline ``run()``
        thread-start. The body below reproduces that inline block faithfully
        -- it is dead code at S6-6 but S6-8 relies on it once the monolith
        ``run()`` is deleted.

        Reproduces the monolith ``run()``'s thread-start (immediately before
        the teacher-side eval loop)::

            if s6.get("imatrix", {}).get("enabled", True):
                gguf_thread = threading.Thread(
                    target=_background_gguf_convert,
                    args=(s6.get("imatrix", {}), artifacts_dir, gguf_result),
                    daemon=True,
                    name="gguf-convert",
                )
                gguf_thread.start()

        Note the monolith only starts the thread on the **non-cache-hit**
        path (inside the ``else:`` of ``if cached_teacher_results is not
        None``). The orchestrator's phase dispatch sequence at S6-8 wires
        this hook AFTER the teacher-provider's cache-hit shortcut has
        returned, so the same gating happens by construction; this hook
        body therefore reproduces ONLY the inner
        ``if s6.get("imatrix", ...): ... start()`` block and does NOT
        re-check ``cached_teacher_results`` itself.

        The thread handle + the result dict are published to ctx so
        ``export_imatrix`` can join the thread and read ``f16_path``.
        """
        config = ctx.get("config")
        artifacts_dir = ctx.get("artifacts_dir")
        s6 = config["stage6_validate"]

        # Reproduce the monolith's default-True semantics here (matches the
        # `if s6.get("imatrix", {}).get("enabled", True):` form in run()).
        # NOTE: intentionally a different default than `is_enabled`
        # (False) -- `is_enabled` decides whether the plugin runs AT ALL on
        # a given config; the inner guard here matches the monolith's
        # per-call-site default for the icfg passed to the background
        # thread.
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
        """Phase hook -- late-phase imatrix dispatch (S6-8 wiring surface).

        INERT at S6-6: no orchestrator walk or test invokes this hook. S6-8
        replaces the Stage 6 orchestrator body with the plugin sequencer and
        dispatches this hook in place of the monolith's inline ``run()``
        post-eval imatrix block. The body below reproduces that inline
        block faithfully -- it is dead code at S6-6 but S6-8 relies on it
        once the monolith ``run()`` is deleted.

        Reproduces the monolith ``run()``'s post-eval block:

        1. **Wait for the background GGUF thread** -- ``gguf_thread.join(
           timeout=3700)``; if the thread is still alive, set the
           F-CR2-M-1 ``imatrix_skipped`` sentinel and surface to trackio.
        2. **Imatrix-skipped path** -- write the eval-text-concat debug
           artifact (unconditional per spec §9) and return.
        3. **Prebuilt-GGUF path** -- when the teacher cache MISSED AND
           the background thread produced an ``f16_path``, invoke
           ``_run_llama_imatrix_with_prebuilt_gguf``.
        4. **Sequential-fallback path** -- otherwise (cache HIT, or
           bg thread failed) invoke ``_generate_imatrix``; its internal
           ``enabled`` guard short-circuits when imatrix is disabled.

        Optional ctx slots (with the monolith's same defaults):
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
        gguf_thread_timed_out = False
        imatrix_skipped = False
        if gguf_thread is not None:
            log.info("Stage 6: waiting for background GGUF conversion to complete")
            gguf_thread.join(timeout=3700)
            if gguf_thread.is_alive():
                # F-CR2-M-1: SKIP imatrix entirely when the bg thread is still alive after
                # the timeout. The daemon bg thread continues writing to model_f16.gguf.tmp
                # and would race with _generate_imatrix's sequential fallback, both of which
                # call os.replace on the same target path. By skipping, no concurrent writer
                # exists; the bg thread's eventual replace just updates the GGUF for the next
                # run. The prebuilt-only GGUF (without imatrix) remains acceptable for
                # downstream serving.
                log.error(
                    "GGUF convert thread still alive after %.0f s timeout; SKIPPING imatrix "
                    "entirely to avoid concurrent-writer race on model_f16.gguf",
                    3700,
                )
                gguf_thread_timed_out = True
                imatrix_skipped = True
        f16_path = None if gguf_thread_timed_out else gguf_result.get("f16_path")
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
        elif cached_teacher_results is None and f16_path is not None:
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


__all__ = [
    "_EVAL_TEXT_CONCAT_FILENAME",
    "_background_gguf_convert",
    "_write_eval_text_concat",
    "_run_llama_imatrix_with_prebuilt_gguf",
    "_generate_imatrix",
    "_find_llama_cpp_dir",
    "ImatrixExportPlugin",
]
