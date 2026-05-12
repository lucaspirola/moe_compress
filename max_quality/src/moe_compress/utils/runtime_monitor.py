"""Lightweight runtime instrumentation for the ablation harness.

Two facilities:

- ``snapshot_telemetry()`` — return a short string with VRAM / host-RAM /
  GPU-util / GPU-temp values. Cheap (~ms) and safe to call every batch.
  Spliced into per-batch DIAG log lines in Stage 2 and Stage 2.5 so
  memory creep, thermal throttling, or GPU idling shows up directly in
  the log stream.

- Module-level breadcrumb (``set_path`` / ``update`` /
  ``install_signal_handlers``) — per-ablation single-file state record
  at ``<ablation_dir>/_last_alive.json``. Updated cheaply on the hot
  path; atomically flushed. Survives any crash (SIGSEGV, OOM kill, hard
  reboot) — the next run inspects the file and knows exactly where the
  prior attempt died, without parsing log files. SIGSEGV cannot be
  reliably caught from Python, so periodic ``update()`` calls in hot
  loops are how the last-known state is captured; the signal handlers
  catch SIGTERM / SIGINT / SIGHUP / clean ``atexit``.

All functions are no-ops if their optional dependencies are missing
(``psutil``, ``pynvml``) or if no breadcrumb path has been set — this
file must never raise from the production hot path.
"""
from __future__ import annotations

import atexit
import json
import os
import signal
import time
from pathlib import Path
from typing import Any

try:
    import psutil  # type: ignore
except ImportError:
    psutil = None  # type: ignore

try:
    import pynvml  # type: ignore
    _PYNVML_INITIALIZED = False
except ImportError:
    pynvml = None  # type: ignore
    _PYNVML_INITIALIZED = False

import torch


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------


def _maybe_init_pynvml() -> bool:
    global _PYNVML_INITIALIZED
    if pynvml is None:
        return False
    if not _PYNVML_INITIALIZED:
        try:
            pynvml.nvmlInit()
            _PYNVML_INITIALIZED = True
        except Exception:
            return False
    return _PYNVML_INITIALIZED


def snapshot_telemetry() -> str:
    """Return a short telemetry string for DIAG log splicing.

    Example: ``vram=89.2/143GB host_free=920GB gpu_util=18% temp=58C``.
    Returns ``"no-telemetry"`` if no probes succeed (e.g. CPU-only env).
    """
    parts: list[str] = []

    if torch.cuda.is_available():
        try:
            free, total = torch.cuda.mem_get_info()
            parts.append(f"vram={(total - free) / 1e9:.1f}/{total / 1e9:.0f}GB")
        except Exception:
            pass

    if psutil is not None:
        try:
            vm = psutil.virtual_memory()
            parts.append(f"host_free={vm.available / 1e9:.0f}GB")
        except Exception:
            pass

    if _maybe_init_pynvml():
        try:
            # Resolve via UUID so CUDA_VISIBLE_DEVICES remapping doesn't surface
            # the wrong physical card. torch's `current_device()` is the CVD-
            # remapped index, while `pynvml.nvmlDeviceGetHandleByIndex` expects
            # the *physical* index — passing one to the other reports a different
            # GPU's metrics. UUID lookup is invariant under remapping.
            h = None
            if torch.cuda.is_available():
                dev_idx = torch.cuda.current_device()
                uuid = getattr(torch.cuda.get_device_properties(dev_idx), "uuid", None)
                if uuid is not None:
                    # NVML expects the "GPU-<uuid>" form; torch's `.uuid` is a
                    # uuid.UUID whose str() is the bare hex (no prefix). Adding
                    # the prefix unless already present makes this robust across
                    # torch versions (newer torch returns a str already prefixed).
                    uuid_str = str(uuid)
                    if not uuid_str.startswith("GPU-"):
                        uuid_str = f"GPU-{uuid_str}"
                    h = pynvml.nvmlDeviceGetHandleByUUID(uuid_str)
            if h is None:
                # Either no CUDA or torch's uuid attr is unavailable (older torch);
                # fall back to physical index 0 — wrong on remapped multi-GPU hosts
                # but better than crashing the telemetry probe.
                h = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
            parts.append(f"gpu_util={util.gpu}% temp={temp}C")
        except Exception:
            pass

    return " ".join(parts) if parts else "no-telemetry"


# ---------------------------------------------------------------------------
# Breadcrumb
# ---------------------------------------------------------------------------

_state: dict[str, Any] = {"phase": "uninit", "pid": os.getpid()}
_path: Path | None = None
_handlers_installed = False
_write_every_n_calls = 10  # File-write throttle for hot-path calls; flush() bypasses.
_call_count = 0
_parent_mkdir_done = False


def set_path(path: Path | str | None) -> None:
    """Set the breadcrumb output path and reset state.

    ``None`` disables breadcrumb writes. Resetting state prevents stale
    keys from one ablation leaking into the next ablation's
    ``_last_alive.json`` (e.g., layer/batch from a prior crash).
    """
    global _path, _state, _call_count, _parent_mkdir_done
    _path = Path(path) if path is not None else None
    _state = {"phase": "init", "pid": os.getpid()}
    _call_count = 0
    _parent_mkdir_done = False
    if _path is not None:
        _state["ts_started"] = time.time()
        flush()


def update(**fields: Any) -> None:
    """Merge fields into the in-memory state; throttle disk writes.

    The state dict is updated unconditionally (cheap — single dict.update).
    Disk write happens only every ``_write_every_n_calls`` calls so the
    hot path's per-batch cost stays microsecond-level. Use ``flush()`` to
    force an immediate write (signal handlers and atexit do this).

    Never raises — breadcrumb errors must never take down the harness.
    """
    global _call_count
    if _path is None:
        return
    _state.update(fields)
    _state["ts_updated"] = time.time()
    _call_count += 1
    if _call_count % _write_every_n_calls == 0:
        _write_to_disk()


def flush() -> None:
    """Force-write the current state to disk. Used by signal/atexit paths."""
    if _path is None:
        return
    _state["ts_updated"] = time.time()
    _write_to_disk()


def _write_to_disk() -> None:
    """Atomic write of _state to _path. Swallows all errors."""
    global _parent_mkdir_done
    try:
        if not _parent_mkdir_done:
            _path.parent.mkdir(parents=True, exist_ok=True)  # type: ignore[union-attr]
            _parent_mkdir_done = True
        tmp = _path.with_suffix(".tmp")  # type: ignore[union-attr]
        tmp.write_text(json.dumps(_state, default=str))
        os.replace(tmp, _path)  # type: ignore[arg-type]
    except Exception:
        pass


def install_signal_handlers() -> None:
    """Register SIGTERM/SIGINT/SIGHUP handlers + atexit flusher.

    Idempotent: safe to call multiple times. SIGSEGV cannot be reliably
    caught from Python — periodic ``update()`` calls in hot loops are
    the fallback for that case.
    """
    global _handlers_installed
    if _handlers_installed:
        return

    def _on_signal(signum, _frame):
        try:
            name = signal.Signals(signum).name
        except Exception:
            name = f"signal({signum})"
        # Bypass the throttle so the signal name is durable before re-raise.
        _state["signal"] = name
        _state["phase"] = f"interrupted_by_{name}"
        flush()
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        try:
            signal.signal(sig, _on_signal)
        except (OSError, ValueError):
            pass  # Some signals can't be set in non-main threads.

    def _atexit_flush() -> None:
        # Don't stomp a terminal phase already recorded by the harness
        # (e.g. ``ablation_done`` from _run_one_ablation's success path).
        # That state is the forensic answer "did the last ablation finish
        # cleanly?" — overwriting it with ``atexit`` would make a clean
        # completion look indistinguishable from a process-exit-mid-run.
        if _state.get("phase") != "ablation_done":
            _state["phase"] = "atexit"
        flush()
    atexit.register(_atexit_flush)
    _handlers_installed = True
