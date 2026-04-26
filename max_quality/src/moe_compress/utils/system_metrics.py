"""Background system-metrics sampler.

Spawns a daemon thread that, every ``interval_sec``, samples GPU + CPU + RAM
state and pushes it to:

1. Python ``logging`` (one INFO line per tick — durable on stdout/HF Jobs logs
   even if Trackio is unavailable).
2. Trackio Run (optional — if a ``trackio_run`` Run object is supplied).

Designed for HF Jobs: the previous Stage 3 OOM (exit 137, no traceback) was
invisible because nothing logged the CPU RAM curve. With this in place a stall
or pre-OOM climb shows up on the dashboard within one tick.

Use as a context manager from ``hf_jobs/entrypoint.py``:

    run = trackio.init(...)
    with SystemMetrics(interval_sec=30, trackio_run=run):
        run_pipeline_main(...)

NB: ``trackio_run`` is the Run **object** returned by ``trackio.init(...)``,
not the trackio module. Trackio's module-level ``trackio.log`` reads a
thread-local current-run pointer — it raises from any thread that didn't
call ``trackio.init`` itself. ``run.log`` is object-bound and works from
any thread.

Graceful degradation: missing ``pynvml`` or ``psutil`` only emits a warning
and disables the corresponding fields; the thread keeps running and never
takes down the pipeline.
"""
from __future__ import annotations

import logging
import threading

log = logging.getLogger("moe_compress.system_metrics")


class SystemMetrics:
    def __init__(
        self,
        interval_sec: float = 30.0,
        gpu_index: int = 0,
        trackio_run=None,
    ) -> None:
        self.interval_sec = float(interval_sec)
        self.gpu_index = int(gpu_index)
        self.run = trackio_run        # Trackio Run object, or None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._pynvml = None
        self._nvml_handle = None
        self._psutil = None
        self._tick = 0

    def start(self) -> None:
        try:
            import pynvml
            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
        except Exception as exc:                     # noqa: BLE001
            log.warning("pynvml unavailable (%s) — GPU metrics disabled", exc)
            self._pynvml = None

        try:
            import psutil
            self._psutil = psutil
            # Priming call. cpu_percent(interval=None) reports CPU usage
            # *since the previous call*, so the first real sample shows the
            # average over the first interval_sec window. This is system-wide
            # CPU, not just our process — fine on dedicated HF Jobs A100
            # hosts where we're the only tenant. Switch to
            # psutil.Process().cpu_percent() if we ever move to shared CPU
            # flavors.
            psutil.cpu_percent(interval=None)
        except Exception as exc:                     # noqa: BLE001
            log.warning("psutil unavailable (%s) — CPU/RAM metrics disabled", exc)
            self._psutil = None

        self._thread = threading.Thread(
            target=self._loop, name="system-metrics", daemon=True,
        )
        self._thread.start()
        log.info("SystemMetrics started (interval=%.1fs, gpu=%d, trackio_run=%s)",
                 self.interval_sec, self.gpu_index,
                 "on" if self.run is not None else "off")

    def stop(self, timeout: float | None = None) -> None:
        # Drop the Trackio Run reference FIRST so any in-flight tick that's
        # already past the ``if self.run is not None`` check still races
        # against a no-op rather than ``run.log`` after ``run.finish()``
        # (which raises). The wrapper in ``_emit`` catches the exception
        # either way, but this avoids per-tick warning spam during shutdown.
        self.run = None
        self._stop.set()
        if self._thread is not None:
            # Allow the thread to complete its current sample (which may
            # take ~1s for pynvml) plus a small safety margin.
            join_timeout = timeout if timeout is not None else self.interval_sec + 5.0
            self._thread.join(timeout=join_timeout)
        if self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:                        # noqa: BLE001
                pass
        log.info("SystemMetrics stopped after %d sample(s)", self._tick)

    def __enter__(self) -> "SystemMetrics":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    # ------------------------------------------------------------------

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_sec):
            try:
                sample = self._sample()
            except Exception as exc:                 # noqa: BLE001
                log.warning("metrics sample failed: %s", exc)
                continue
            self._tick += 1
            self._emit(sample)

    def _sample(self) -> dict[str, float]:
        out: dict[str, float] = {}

        if self._pynvml is not None and self._nvml_handle is not None:
            try:
                mem = self._pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                util = self._pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                out["sys/gpu_util_pct"] = float(util.gpu)
                out["sys/vram_used_gb"] = mem.used / 1e9
                out["sys/vram_total_gb"] = mem.total / 1e9
                out["sys/vram_pct"] = 100.0 * mem.used / max(mem.total, 1)
            except Exception as exc:                 # noqa: BLE001
                log.warning("pynvml sample failed: %s", exc)

        try:
            import torch
            if torch.cuda.is_available():
                out["sys/torch_alloc_gb"] = torch.cuda.memory_allocated() / 1e9
                out["sys/torch_reserved_gb"] = torch.cuda.memory_reserved() / 1e9
        except Exception:                            # noqa: BLE001
            pass

        if self._psutil is not None:
            try:
                vm = self._psutil.virtual_memory()
                out["sys/cpu_pct"] = float(self._psutil.cpu_percent(interval=None))
                # Host-wide memory — fluctuates with page cache slosh from
                # other tenants / unrelated mmap activity. Useful for a
                # macroscopic view but NOT the right signal for "is our
                # accumulator growing".
                out["sys/ram_used_gb"] = vm.used / 1e9
                out["sys/ram_total_gb"] = vm.total / 1e9
                out["sys/ram_pct"] = float(vm.percent)
                # Per-process RSS — the bulk of what THIS pipeline holds
                # (Python heap + PyTorch CPU tensor storage + mmaps this
                # process has touched). Doesn't include other tenants'
                # page cache, so it's a tighter bound on our own footprint.
                # Can still fluctuate down when the kernel evicts cold
                # mmap'd file pages from this process, but the floor is
                # the actual heap.
                proc = self._psutil.Process()
                meminfo = proc.memory_info()
                out["sys/proc_rss_gb"] = meminfo.rss / 1e9
                out["sys/proc_vms_gb"] = meminfo.vms / 1e9
            except Exception as exc:                 # noqa: BLE001
                log.warning("psutil sample failed: %s", exc)

        # Peak RSS since process start — monotonically non-decreasing and
        # therefore the cleanest "did our accumulator actually grow"
        # signal. Linux reports KB; we convert to GB.
        try:
            import resource
            ru = resource.getrusage(resource.RUSAGE_SELF)
            out["sys/maxrss_gb"] = ru.ru_maxrss / (1024 * 1024)
        except Exception:                            # noqa: BLE001
            pass

        return out

    def _emit(self, sample: dict[str, float]) -> None:
        # One terse stdout line covering the most common stall signals.
        # Includes proc_rss / maxrss alongside host-wide ram so a stage
        # author can see the difference between page-cache fluctuation
        # (host ram dips) and actual process growth (proc_rss / maxrss
        # only ever climbs, never dips).
        gpu = sample.get("sys/gpu_util_pct")
        vram = sample.get("sys/vram_used_gb")
        vram_pct = sample.get("sys/vram_pct")
        cpu = sample.get("sys/cpu_pct")
        ram = sample.get("sys/ram_used_gb")
        ram_pct = sample.get("sys/ram_pct")
        proc_rss = sample.get("sys/proc_rss_gb")
        maxrss = sample.get("sys/maxrss_gb")
        log.info(
            "sys: gpu=%s%% vram=%s/%sGB(%s%%) cpu=%s%% "
            "ram=%s/%sGB(%s%%) proc_rss=%sGB maxrss=%sGB",
            _fmt(gpu), _fmt(vram), _fmt(sample.get("sys/vram_total_gb")), _fmt(vram_pct),
            _fmt(cpu), _fmt(ram), _fmt(sample.get("sys/ram_total_gb")), _fmt(ram_pct),
            _fmt(proc_rss), _fmt(maxrss),
        )
        if self.run is not None:
            try:
                # Object-bound .log on the Run — works from any thread
                # (unlike module-level trackio.log which is thread-local).
                self.run.log(sample)
            except Exception as exc:                 # noqa: BLE001
                log.warning("run.log failed: %s", exc)


def _fmt(x):
    if x is None:
        return "-"
    return f"{x:.1f}"
