"""Instrumented forward for fused Qwen3_5MoeExperts.

Replaces the old per-`nn.Linear` forward-hook strategy. Because the fused
``Qwen3_5MoeExperts.forward`` has no per-expert sub-modules to hook, we
monkey-patch the whole forward with an instrumented replica that emits
user-supplied callbacks at each of the three key points:

    input          : (sel_state)                     — input to gate_up_proj
    intermediate   : (act_fn(gate) * up)             — input to down_proj
    down           : (down_proj output)              — expert output

Each callback signature:

    def cb(layer_idx, expert_idx, tensor, context) -> None

where ``context`` is a dict with ``top_k_weights``, ``top_k_pos``, ``token_idx``
so the callee can compute REAP scores (g_j · ||f_j||) without re-reading the
routing metadata.

Usage:

    from moe_compress.utils.activation_hooks import instrument_experts

    callbacks = {
        "down":         down_max_cb,     # Stage 0
        "input":        cov_cb,          # Stage 2/3 gate_up_proj input cov
        "intermediate": int_cov_cb,      # Stage 2/3 down_proj input cov
    }
    with instrument_experts(layer_ref, callbacks):
        for batch in batches:
            model(input_ids=batch)

The instrumentation is per-layer. Install on each MoE layer you want to
observe; caller handles which layers' data to collect.

This module also keeps the previously-used accumulator dataclasses
(``DownProjMaxAccumulator``, ``ReapAccumulator``, ``InputCovarianceAccumulator``)
because Stages 0/2/3 still use their API — only the hook plumbing below them
changed.
"""
from __future__ import annotations

import contextlib
import logging
import threading
import types
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_io import MoELayerRef, FactoredExperts

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Accumulators (API preserved from pre-refactor; stages import these)
# ---------------------------------------------------------------------------


@dataclass
class DownProjMaxAccumulator:
    """Per-(layer, expert) max(|x|) accumulator, GPU-resident during forward.

    Keeps a 0-dim CUDA tensor per expert and runs ``torch.maximum`` on it
    without syncing. :meth:`finalize` transfers to CPU once at end of
    profiling (not per-expert, per-sample).
    """
    per_expert_max: dict[tuple[int, int], float] = field(default_factory=dict)
    _gpu: dict[tuple[int, int], torch.Tensor] = field(default_factory=dict)

    def update(self, layer_idx: int, expert_idx: int, x: torch.Tensor) -> None:
        cur = x.detach().abs().amax()                       # 0-dim, stays on device
        key = (layer_idx, expert_idx)
        prev = self._gpu.get(key)
        if prev is None:
            self._gpu[key] = cur
        else:
            prev.copy_(torch.maximum(prev, cur))

    def finalize(self) -> None:
        for key, tensor in self._gpu.items():
            val = float(tensor.cpu().item())
            if val > self.per_expert_max.get(key, 0.0):
                self.per_expert_max[key] = val
        self._gpu.clear()


@dataclass
class ReapAccumulator:
    """REAP score accumulator, GPU-resident during layer profiling.

    Instead of ``sums[k] += float(tensor.cpu().item())`` (which stalls the GPU
    on every expert × sample event), we keep a per-expert 0-dim tensor on the
    same device as the forward and only transfer to CPU via
    :meth:`finalize_layer`.
    """
    sums: dict[tuple[int, int], float] = field(default_factory=lambda: defaultdict(float))
    counts: dict[tuple[int, int], int] = field(default_factory=lambda: defaultdict(int))
    freq: dict[tuple[int, int], int] = field(default_factory=lambda: defaultdict(int))
    _gpu_sums: dict[tuple[int, int], torch.Tensor] = field(default_factory=dict)

    def add_gpu(self, key: tuple[int, int], contrib: torch.Tensor, n_tokens: int) -> None:
        cur = self._gpu_sums.get(key)
        if cur is None:
            # ``.clone()`` so the stored accumulator owns its storage. Caller
            # ``record_reap`` computes a fresh 0-dim tensor per call today so
            # the detach-only path would also work, but clone hardens against
            # future callers who reuse buffers.
            self._gpu_sums[key] = contrib.detach().clone()
        else:
            cur.add_(contrib.detach())
        self.counts[key] = self.counts.get(key, 0) + n_tokens
        self.freq[key] = self.freq.get(key, 0) + n_tokens

    def finalize_layer(self, layer_idx: int) -> None:
        keys = [k for k in self._gpu_sums if k[0] == layer_idx]
        for k in keys:
            gpu = self._gpu_sums.pop(k)
            self.sums[k] = self.sums.get(k, 0.0) + float(gpu.cpu().item())

    def finalize_all(self) -> None:
        layer_ids = {k[0] for k in self._gpu_sums}
        for li in layer_ids:
            self.finalize_layer(li)

    def score(self, layer_idx: int, expert_idx: int) -> float:
        k = (layer_idx, expert_idx)
        n = self.counts.get(k, 0)
        if n == 0:
            return 0.0
        return self.sums.get(k, 0.0) / n


@dataclass
class InputCovarianceAccumulator:
    """Per-(layer, expert, matrix_name) streaming covariance accumulator.

    Two-tier storage:
      * ``_gpu``: transient covariances for the *currently-hot layer* kept
        resident on the forward's device. Summed in fp32 on GPU to avoid
        CPU-sync on every expert × sample event.
      * ``covariance``: CPU-resident final results, in ``storage_dtype``
        (default bf16 for disk economy). Populated by :meth:`finalize_layer`
        once per-layer profiling completes.

    Stage 2 drives profiling one layer at a time, so only one layer's worth
    of per-expert GPU covariances is live simultaneously (≤ ~256 experts
    × 2048×2048 fp32 ≈ 4.3 GB — fits well under an 80 GB A100).

    Aliasing: ``gate_proj`` and ``up_proj`` share input inside
    ``Qwen3_5MoeExperts`` so writes with ``matrix_name="up_proj"`` are
    ignored (``gate_proj`` already covers it, and :meth:`get` returns the
    gate_proj entry for up_proj lookups).
    """

    covariance: dict[tuple[int, int, str], torch.Tensor] = field(default_factory=dict)
    token_count: dict[tuple[int, int, str], int] = field(default_factory=lambda: defaultdict(int))
    storage_dtype: torch.dtype = torch.float32
    _alias_gate_up: bool = True
    # GPU-resident transient storage, keyed the same way as ``covariance``.
    _gpu: dict[tuple[int, int, str], torch.Tensor] = field(default_factory=dict)
    _gpu_token_count: dict[tuple[int, int, str], int] = field(default_factory=lambda: defaultdict(int))
    # Thread lock guarding ``covariance`` and ``token_count`` dict mutations.
    # Stage 3 runs ``spill_layer_to_disk`` on a background thread (overlapping
    # I/O with the next layer's forward pass) while the main thread continues
    # to ``finalize_layer`` for layer N+1. The two threads touch different
    # KEYS but the SAME dict objects; CPython dict resize during another
    # thread's pop/insert can corrupt the hash table even under the GIL.
    # RLock so pop loops + helper methods can re-enter safely.
    _lock: "threading.RLock" = field(default_factory=lambda: threading.RLock())  # noqa: F821

    def set_storage_dtype(self, dtype: torch.dtype) -> None:
        self.storage_dtype = dtype

    def update(
        self, layer_idx: int, expert_idx: int, matrix_name: str, x: torch.Tensor
    ) -> None:
        """Accumulate xᵀx on the *same device as x*. No CPU sync here."""
        if self._alias_gate_up and matrix_name == "up_proj":
            return
        flat = x.detach().reshape(-1, x.shape[-1])
        if flat.numel() == 0:
            return
        flat_f32 = flat.to(torch.float32)
        cov = flat_f32.transpose(0, 1) @ flat_f32            # stays on GPU
        key = (layer_idx, expert_idx, matrix_name)
        # _gpu / _gpu_token_count are touched only by the calling thread
        # (forward hooks fire synchronously on the main thread) — no lock.
        cur = self._gpu.get(key)
        if cur is None:
            self._gpu[key] = cov
        else:
            cur.add_(cov)
        self._gpu_token_count[key] = self._gpu_token_count.get(key, 0) + flat.shape[0]

    def finalize_layer(self, layer_idx: int) -> None:
        """Move every covariance for ``layer_idx`` from GPU to CPU in
        ``storage_dtype``. Call once after a layer's profile is done; the
        per-expert GPU tensors are freed afterwards."""
        keys = [k for k in self._gpu if k[0] == layer_idx]
        for k in keys:
            gpu_cov = self._gpu.pop(k)
            cpu_cov = gpu_cov.to(self.storage_dtype).cpu()
            with self._lock:
                prev = self.covariance.get(k)
                if prev is None:
                    self.covariance[k] = cpu_cov
                else:
                    self.covariance[k] = (
                        prev.to(torch.float32) + cpu_cov.to(torch.float32)
                    ).to(self.storage_dtype)
                self.token_count[k] = (
                    self.token_count.get(k, 0) + self._gpu_token_count.pop(k, 0)
                )

    def finalize_all(self) -> None:
        """Move every GPU-resident covariance to CPU. Use when layers are
        instrumented simultaneously (e.g. Stage 0) instead of one-at-a-time."""
        layer_ids = {k[0] for k in self._gpu}
        for li in layer_ids:
            self.finalize_layer(li)

    def spill_layer_to_disk(self, layer_idx: int, dir_path) -> None:
        """Persist all in-memory entries for ``layer_idx`` to a single
        ``layer_{layer_idx}.pt`` file under ``dir_path``, then drop them
        from the in-memory dict. Bounds per-layer accumulators to disk —
        required on hosts where the full per-(layer, expert) cov dict
        would exceed the cgroup memory limit (a100-large = 142 GB).
        Call :meth:`load_layer_from_disk` to bring a layer back when
        the factor loop needs it.

        Writes are **atomic**: tensors are saved to ``layer_{idx}.pt.tmp``
        and only renamed to the final path after ``torch.save`` returns
        successfully. A SIGKILL/OOM mid-write therefore leaves at most a
        ``.tmp`` file (which the resume path doesn't recognize), never a
        truncated final ``.pt`` that would silently load as garbage on
        resume.
        """
        import os
        from pathlib import Path
        # Snapshot keys + payload under the lock so a concurrent
        # finalize_layer (different keys but same dict) can't trigger a
        # dict-resize while we're iterating.
        with self._lock:
            keys = [k for k in self.covariance if k[0] == layer_idx]
            if not keys:
                return
            payload = {
                "format_version": 1,
                "covariance": {k: self.covariance[k] for k in keys},
                "tokens": {k: self.token_count.get(k, 0) for k in keys},
            }
        out = Path(dir_path) / f"layer_{layer_idx}.pt"
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(out.suffix + ".tmp")
        torch.save(payload, tmp)        # disk I/O — releases GIL, no lock
        os.replace(tmp, out)            # atomic on POSIX, including FUSE
        with self._lock:
            for k in keys:
                self.covariance.pop(k, None)

    def load_layer_from_disk(self, layer_idx: int, dir_path) -> bool:
        """Restore a previously spilled layer's entries to the in-memory
        dict. Returns True if loaded, False if the file doesn't exist
        (caller should treat as not-yet-computed)."""
        from pathlib import Path
        p = Path(dir_path) / f"layer_{layer_idx}.pt"
        if not p.exists():
            return False
        try:
            payload = torch.load(p, map_location="cpu")
        except Exception as exc:                     # noqa: BLE001
            # Most likely a torn .pt that snuck past the .tmp+rename
            # guard (e.g. an old run on a previous version of this
            # code). Raise with the path so it's actionable.
            raise RuntimeError(
                f"Failed to torch.load spill file {p}: {exc!r}. "
                "The file is likely corrupt — delete it (or the whole "
                "spill dir) and re-run the affected layer."
            ) from exc
        if not isinstance(payload, dict) or "covariance" not in payload:
            raise RuntimeError(
                f"Spill file {p} has unexpected layout: "
                f"got {type(payload).__name__} "
                f"(keys={list(payload.keys()) if isinstance(payload, dict) else 'n/a'}). "
                "Likely a legacy or corrupt format — delete and re-run."
            )
        fmt = int(payload.get("format_version", 0))
        if fmt != 1:
            raise RuntimeError(
                f"Spill file {p} has format_version={fmt} (expected 1). "
                "Regenerate by deleting the per-layer spill dir and re-running."
            )
        with self._lock:
            self.covariance.update(payload["covariance"])
            for k, n in payload.get("tokens", {}).items():
                self.token_count[k] = n
        return True

    def unload_layer(self, layer_idx: int) -> None:
        """Drop in-memory entries for a layer (used after a lazy-load+use
        cycle during the factor phase). Token counts are left in place
        since they're tiny."""
        with self._lock:
            for k in [k for k in self.covariance if k[0] == layer_idx]:
                self.covariance.pop(k, None)

    def get(self, key: tuple[int, int, str]) -> torch.Tensor | None:
        if key in self.covariance:
            return self.covariance[key]
        if self._alias_gate_up and key[2] == "up_proj":
            alt = (key[0], key[1], "gate_proj")
            return self.covariance.get(alt)
        return None


# ---------------------------------------------------------------------------
# Common REAP recorder (called from an 'intermediate'- or 'down'-point callback)
# ---------------------------------------------------------------------------


def record_reap(
    acc: ReapAccumulator,
    layer_idx: int,
    expert_idx: int,
    gate_vals: torch.Tensor,
    expert_outs: torch.Tensor,
) -> None:
    """``gate_vals`` [T], ``expert_outs`` [T, hidden]. Accumulates on GPU —
    the CPU transfer happens in :meth:`ReapAccumulator.finalize_layer`."""
    if gate_vals.numel() == 0:
        return
    leading = int(expert_outs.shape[0]) if expert_outs.dim() >= 2 else int(expert_outs.numel())
    if gate_vals.numel() != leading:
        raise RuntimeError(
            f"REAP: gate_vals.numel()={gate_vals.numel()} != expert_outs[0]={leading} "
            f"(layer={layer_idx}, expert={expert_idx}). "
            "Instrumented forward is out of sync with the reference dispatch."
        )
    norms = expert_outs.to(torch.float32).norm(dim=-1)
    contrib = (gate_vals.to(torch.float32) * norms).sum()
    acc.add_gpu((layer_idx, expert_idx), contrib, int(gate_vals.numel()))


# ---------------------------------------------------------------------------
# Instrumented forward for one MoE layer
# ---------------------------------------------------------------------------


CallbackFn = Callable[[int, int, torch.Tensor, dict], None]


@contextlib.contextmanager
def instrument_experts(
    layer_ref: MoELayerRef,
    callbacks: dict[str, CallbackFn],
):
    """Install an instrumented forward on ``layer_ref.mlp.experts`` that
    emits callbacks at the three observation points, then restore on exit.

    Works for both ``Qwen3_5MoeExperts`` (fused) and ``FactoredExperts``
    (our rank-k replacement). The two paths share the same dispatch
    structure — only the per-expert matmul sequence differs.

    Accepted callback keys:
      - ``input``         : called with sel_state per (layer, expert, batch)
      - ``intermediate``  : called with act_fn(gate) * up (down_proj input)
      - ``down``          : called with down output
      - ``gate_up_out``   : called with the raw pre-chunk gate_up projection
      - ``gate_up_in``    : alias for ``input`` (clarity)

    Context dict passed to each callback:
      {"top_k_weights": [T], "top_k_pos": [T], "token_idx": [T]}
    """
    experts = layer_ref.experts_module
    is_factored = isinstance(experts, FactoredExperts)
    original_forward = experts.forward
    layer_idx = layer_ref.layer_idx

    def _cb(name, eidx, tensor, ctx):
        fn = callbacks.get(name)
        if fn is None:
            return
        fn(layer_idx, int(eidx), tensor, ctx)

    if is_factored:
        def wrapped(self, hidden_states, top_k_index, top_k_weights):
            final = torch.zeros_like(hidden_states)
            with torch.no_grad():
                mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
                hit = (mask.sum(dim=(-1, -2)) > 0).nonzero()
            for expert_idx in hit:
                e = expert_idx[0]
                if e == self.num_experts:
                    continue
                top_k_pos, token_idx = torch.where(mask[e])
                sel = hidden_states[token_idx]
                ctx = {"top_k_weights": top_k_weights[token_idx, top_k_pos],
                       "top_k_pos": top_k_pos, "token_idx": token_idx}
                _cb("input", e, sel, ctx)
                _cb("gate_up_in", e, sel, ctx)
                gate = F.linear(F.linear(sel, self.gate_proj_V[e]), self.gate_proj_U[e])
                up   = F.linear(F.linear(sel, self.up_proj_V[e]),   self.up_proj_U[e])
                intermediate = self.act_fn(gate) * up
                _cb("intermediate", e, intermediate, ctx)
                down = F.linear(F.linear(intermediate, self.down_proj_V[e]),
                                self.down_proj_U[e])
                _cb("down", e, down, ctx)
                down = down * top_k_weights[token_idx, top_k_pos, None]
                final.index_add_(0, token_idx, down.to(final.dtype))
            return final
    else:
        def wrapped(self, hidden_states, top_k_index, top_k_weights):
            final = torch.zeros_like(hidden_states)
            with torch.no_grad():
                mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
                hit = (mask.sum(dim=(-1, -2)) > 0).nonzero()
            for expert_idx in hit:
                e = expert_idx[0]
                if e == self.num_experts:
                    continue
                top_k_pos, token_idx = torch.where(mask[e])
                sel = hidden_states[token_idx]
                ctx = {"top_k_weights": top_k_weights[token_idx, top_k_pos],
                       "top_k_pos": top_k_pos, "token_idx": token_idx}
                _cb("input", e, sel, ctx)
                _cb("gate_up_in", e, sel, ctx)
                gate_up = F.linear(sel, self.gate_up_proj[e])
                _cb("gate_up_out", e, gate_up, ctx)
                gate, up = gate_up.chunk(2, dim=-1)
                intermediate = self.act_fn(gate) * up
                _cb("intermediate", e, intermediate, ctx)
                down = F.linear(intermediate, self.down_proj[e])
                _cb("down", e, down, ctx)
                down = down * top_k_weights[token_idx, top_k_pos, None]
                final.index_add_(0, token_idx, down.to(final.dtype))
            return final

    experts.forward = types.MethodType(wrapped, experts)
    try:
        yield
    finally:
        experts.forward = original_forward


# ---------------------------------------------------------------------------
# Generic calibration runner (no hooks by itself)
# ---------------------------------------------------------------------------


def run_calibration(
    model: nn.Module,
    batches,
    *,
    device=None,
    extra_forward_kwargs: dict | None = None,
    per_batch_callback: Callable[[int], None] | None = None,
    log_every: int = 64,
) -> None:
    model.eval()
    n_total = len(batches) if hasattr(batches, "__len__") else None
    with torch.no_grad():
        for i, batch in enumerate(batches):
            if device is not None:
                batch = batch.to(device)
            model(input_ids=batch, **(extra_forward_kwargs or {}))
            if per_batch_callback is not None:
                per_batch_callback(i)
            if log_every > 0 and (i + 1) % log_every == 0:
                if n_total is not None:
                    log.info("calibration forward %d/%d", i + 1, n_total)
                else:
                    log.info("calibration forward %d", i + 1)


# ---------------------------------------------------------------------------
# Router-output hook (Stage 5 uses this instead of the fused-experts shim)
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def capture_router_outputs(layer_refs: list[MoELayerRef]):
    """Collect **pre-softmax** router logits for each given layer.

    Why a pre-forward hook: ``Qwen3_5MoeTopKRouter.forward`` overwrites its
    first return value with a softmax'd tensor before returning, so a
    ``register_forward_hook`` on the router gives post-softmax probabilities
    rather than logits. For Stage 5 KD we need the raw scores, so we
    recompute ``F.linear(hidden, router.weight)`` ourselves in a pre-forward
    hook. Cheap (one matmul that also runs inside the router) and always
    correct.
    """
    storage: dict[int, list[torch.Tensor]] = {ref.layer_idx: [] for ref in layer_refs}
    handles: list = []

    def _pre_factory(li, router):
        def _h(_m, inputs):
            x = inputs[0]
            if hasattr(router, "hidden_dim"):
                x = x.reshape(-1, router.hidden_dim)
            logits = F.linear(x, router.weight)
            if getattr(router, "bias", None) is not None:
                logits = logits + router.bias
            storage[li].append(logits.detach())
        return _h

    for ref in layer_refs:
        h = ref.router.register_forward_pre_hook(_pre_factory(ref.layer_idx, ref.router))
        handles.append(h)
    try:
        yield storage
    finally:
        for h in handles:
            h.remove()
