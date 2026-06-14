"""VRAM-aware per-plugin auto-batch resolver (v1: bounded/invariant plugins).

See docs/superpowers/specs/2026-06-11-per-plugin-vram-aware-auto-batch-sizing-design.md.
Default-OFF: when AutoBatchConfig.enabled is False the resolver is never invoked
and every golden stays byte-identical. NOT for reduction-accumulating plugins
(cov Gram / NLL) — those need the v2 pinned-reduction work first.

Config block
------------
Plugins read an optional ``auto_batch`` mapping (parsed via
``AutoBatchConfig.from_dict``); unknown keys are ignored. Keys:

  - ``enabled`` (bool, default ``False``) — master switch. When false the
    resolver is a no-op and the plugin keeps its fixed batch size, so goldens
    stay byte-identical.
  - ``headroom_frac`` (float, default ``0.1``) — fraction of total VRAM held
    back as headroom when sizing the candidate from usable-free memory.
  - ``max_cap`` (int, default ``4096``) — hard upper clamp on the resolved
    batch size.
  - ``probe_samples`` (int, default ``4``) — number of samples the plugin
    feeds the cost probe.

There is NO ``rtol``/``atol`` — the subset allclose self-test was removed
(spec rev4): sizing is a cost-model prediction (``size_batch``) and the REAL
pass is the fit test (``run_with_oom_backoff``).

Scope (phase-a today): the resolver applies ONLY to plugins declared
``FidelityClass.BATCH_INVARIANT`` (``_V1_ELIGIBLE``). For all other plugins,
and whenever ``enabled`` is false, ``resolve_batch`` is a no-op that returns
the caller's fixed batch size.
"""
from __future__ import annotations
import dataclasses
import enum
import logging
import math

log = logging.getLogger(__name__)


class FidelityClass(enum.Enum):
    METRIC_PINNED = "metric_pinned"
    BATCH_INVARIANT = "batch_invariant"
    REDUCTION_BOUNDED = "reduction_bounded"
    REDUCTION_ACCUMULATING = "reduction_accumulating"


# Classes the v1 resolver may auto-size. Only BATCH_INVARIANT is proven-safe with
# NO numeric check (grouping-independent reduction); REDUCTION_BOUNDED is declared
# but NOT auto-wired in v1 (needs its own bounded-drift argument first, per spec
# §4c) — there is no subset self-test to fall back on, so we don't trust it yet.
_V1_ELIGIBLE = frozenset({FidelityClass.BATCH_INVARIANT})


@dataclasses.dataclass(frozen=True)
class AutoBatchConfig:
    enabled: bool = False
    headroom_frac: float = 0.1
    max_cap: int = 4096
    probe_samples: int = 4

    @classmethod
    def from_dict(cls, d: dict | None) -> "AutoBatchConfig":
        d = d or {}
        fields = {f.name for f in dataclasses.fields(cls)}
        kw = {k: v for k, v in d.items() if k in fields}
        # Safety contract is "OFF unless explicitly on": coerce enabled to a real
        # bool so a stray YAML string ("false") cannot silently enable the resolver
        # and break the byte-identical guarantee.
        if "enabled" in kw:
            kw["enabled"] = (kw["enabled"] is True) or (str(kw["enabled"]).strip().lower() == "true")
        return cls(**kw)


def fit_cost(*, peak1: int, peak2: int) -> tuple[float, float]:
    """Fit cost(b)=fixed+b*per_sample from single-forward peaks at b=1 and b=2."""
    per_sample = float(peak2) - float(peak1)
    if per_sample <= 0:
        raise ValueError(f"non-increasing probe peaks: peak1={peak1} peak2={peak2}")
    # Spec §4 formula is fixed = 2*peak1 - peak2 (unclamped). We clamp to >=0 to
    # guard against a noisy probe yielding a spurious negative fixed; this does
    # not change the b=1,2 slope (per_sample) used for sizing.
    fixed = 2.0 * float(peak1) - float(peak2)
    return max(fixed, 0.0), per_sample


def size_candidate(*, total: int, headroom_frac: float,
                   fixed: float, per_sample: float, fixed_batch: int,
                   max_cap: int) -> int:
    """Largest batch whose predicted ABSOLUTE peak fits VRAM, clamped to [floor, cap].

    The two-point probe peaks are ABSOLUTE ``max_memory_allocated`` bytes, so
    ``fixed`` (= 2*peak1 - peak2, the cost line's y-intercept) already includes
    EVERY byte resident at probe time — the model weights, the framework, and any
    co-resident accumulator. The predicted absolute peak at batch ``b`` is
    ``fixed + b*per_sample``; we size the largest ``b`` that satisfies
    ``fixed + b*per_sample <= total*(1 - headroom_frac)``.

    Do NOT subtract a separate ``allocated_baseline`` here: that byte count (the
    resident model) is ALREADY inside ``fixed``. Subtracting it again
    double-counts the model and, whenever the model dominates VRAM
    (``fixed ≈ baseline ≈ total``, i.e. every real large model), drives the result
    negative → clamps to ``fixed_batch`` → auto-batch silently never engages. That
    was the shipped bug; the CPU probes (allocated()==0) masked it. ``headroom`` +
    ``run_with_oom_backoff`` cover memory that grows AFTER the probe (Gram fill).

    Precondition: fixed_batch <= max_cap. The floor wins over the cap (we never
    return below the caller's proven-safe fixed batch), so a misconfigured
    fixed_batch > max_cap would defeat the cap — reject it loudly instead.
    """
    if fixed_batch > max_cap:
        raise ValueError(f"fixed_batch={fixed_batch} exceeds max_cap={max_cap}")
    headroom = headroom_frac * float(total)
    usable = float(total) - headroom
    raw = math.floor((usable - fixed) / per_sample) if per_sample > 0 else fixed_batch
    return int(max(fixed_batch, min(raw, max_cap)))


class MemProbe:
    """Injectable CUDA-memory accessor (bytes). Override in tests."""
    def total(self) -> int: raise NotImplementedError
    def allocated(self) -> int: raise NotImplementedError
    def reset_peak(self) -> None: raise NotImplementedError
    def peak(self) -> int: raise NotImplementedError


class CudaMemProbe(MemProbe):
    def __init__(self, device=None):
        import torch
        self._torch = torch
        self._device = device
    def total(self) -> int:
        _free, total = self._torch.cuda.mem_get_info(self._device)
        return int(total)
    def allocated(self) -> int:
        return int(self._torch.cuda.memory_allocated(self._device))
    def reset_peak(self) -> None:
        self._torch.cuda.reset_peak_memory_stats(self._device)
    def peak(self) -> int:
        return int(self._torch.cuda.max_memory_allocated(self._device))


def size_batch(cost_probe_fn, fixed_batch: int, *, headroom_frac: float,
               max_cap: int, mem: MemProbe | None = None) -> int:
    """Predict the largest forward batch from a two-point cost probe. Never raises:
    a bad/non-increasing/OOMing probe degrades to fixed_batch.

    ``cost_probe_fn(micro_batch) -> int`` runs ONE forward of ``micro_batch``
    sequences with the plugin's identical forward signature and returns the peak
    allocated bytes. It is called only at 1 and 2.
    """
    import torch
    if mem is None:
        mem = CudaMemProbe()
    baseline = mem.allocated()  # for logging only — NOT used in sizing (it is
    # already inside ``fixed``; subtracting it would double-count, see size_candidate)
    try:
        mem.reset_peak(); peak1 = cost_probe_fn(1)
        mem.reset_peak(); peak2 = cost_probe_fn(2)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        log.warning("auto_batch: OOM during cost probe; using floor %d", fixed_batch)
        return int(fixed_batch)
    try:
        fixed, per = fit_cost(peak1=peak1, peak2=peak2)
    except ValueError as exc:
        log.warning("auto_batch: cost probe unusable (%s); using floor %d", exc, fixed_batch)
        return int(fixed_batch)
    candidate = size_candidate(total=mem.total(),
                               headroom_frac=headroom_frac, fixed=fixed, per_sample=per,
                               fixed_batch=fixed_batch, max_cap=max_cap)
    log.info("auto_batch: predicted batch=%d (floor=%d total=%.3g baseline=%.3g "
             "fixed=%.3g per=%.3g)",
             candidate, fixed_batch, float(mem.total()), float(baseline), fixed, per)
    return int(candidate)


def resolve_batch(cost_probe_fn, fixed_batch: int, fidelity_class: FidelityClass,
                  cfg: AutoBatchConfig, mem: MemProbe | None = None) -> int:
    """Class-gated sizing. Returns fixed_batch (no probe) when disabled or the
    fidelity class is not v1-eligible; else the size_batch prediction."""
    if not cfg.enabled or fidelity_class not in _V1_ELIGIBLE:
        return int(fixed_batch)
    return size_batch(cost_probe_fn, fixed_batch, headroom_frac=cfg.headroom_frac,
                      max_cap=cfg.max_cap, mem=mem)


def run_with_oom_backoff(run_fn, start_batch: int, floor: int):
    """Run run_fn(batch) at start_batch; on CUDA OOM empty_cache + back off
    gently toward floor (x0.75, capped at attempt-1) and rerun, e.g.
    18->13->9 instead of 18->9. The real pass is the fit test. Re-raises if
    the floor OOMs."""
    import torch
    attempt = max(int(start_batch), int(floor))
    while True:
        try:
            return run_fn(attempt)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if attempt <= floor:
                log.error("auto_batch: OOM at floor batch=%d; unrecoverable", floor)
                raise
            new = max(min(int(attempt * 0.75), attempt - 1), floor)
            log.warning("auto_batch: OOM at batch=%d; retrying at %d", attempt, new)
            attempt = new
