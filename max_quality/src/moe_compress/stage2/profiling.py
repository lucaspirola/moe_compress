"""Stage 2 profiling harness: per-layer early-exit forward + REAP/REAM/cov hooks.

Extracted from ``stage2_reap_ream.py`` in Task 3 of the plugin-architecture
refactor. Public surface is unchanged: ``stage2_reap_ream`` re-imports both
``_profile_layer`` and ``_LayerInputAccumulator`` at module scope so external
call-sites (tests in particular) keep working without modification.
"""
from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

from ..utils.activation_hooks import (
    InputCovarianceAccumulator,
    ReamCostAccumulator,
    ReapAccumulator,
    _EarlyExitException,
    capture_router_outputs,
    early_exit_after_layer,
    instrument_experts,
    record_reap,
)
from ..utils.model_io import MoELayerRef
from ..utils.runtime_monitor import (
    snapshot_telemetry as _rt_snap,
    update as _rt_update,
)

log = logging.getLogger(__name__)


class _LayerInputAccumulator:
    """Reservoir-sample hidden states arriving at a single MoE layer.

    Captured during the profile pass via a forward-pre hook on the decoder
    layer. Used by step 7b expert distillation to provide the calibration
    inputs ``x`` that feed both the merged-centroid student forward and the
    pre-merge group-member target forward.

    Sample size is capped at ``max_samples`` (default 8192 tokens) so the
    host RAM cost is bounded even on long calibration runs. With
    ``hidden_size=2048`` and bf16, a full buffer is ~32 MB.

    Caller contract on first add: when the very first ``add()`` call carries
    ``n > max_samples`` tokens, the buffer is initialised with the deterministic
    prefix ``flat[:max_samples]`` and the remaining ``n - max_samples`` tokens
    are silently discarded (i.e. they are not subject to reservoir replacement).
    This matches the scalar-loop baseline behavior — the original implementation
    extended the buffer one token at a time until reaching ``max_samples``, then
    returned without ever revisiting the prefix. Callers feeding the accumulator
    in a single oversized first batch will get a non-uniform sample biased to
    the prefix; feed in multiple ``add()`` calls (or pre-shuffle) for a truly
    uniform sample.

    A seeded ``torch.Generator`` (default seed = 0) is used for the reservoir
    coin flips so the captured calibration set is bit-reproducible across
    runs; callers can override ``seed`` with the layer index for per-layer
    independence (the Stage 2 driver does this).

    Algorithm: vectorized batch implementation of Vitter (1985) Algorithm R.
    See ``tasks/SC_FAST_PLAN_V3.md`` §4 / Optimization C (lines 277-296) for
    the measured 45x speedup (1.89 ms/batch vs 85.9 ms/batch baseline) and
    Vitter, J.S. (1985). "Random sampling with a reservoir." *ACM Transactions
    on Mathematical Software*, 11(1):37-57 for the underlying algorithm.

    Deviation D1 from textbook scalar Algorithm R: this implementation
    processes one batch of N tokens in a single vectorized step (one rand(N)
    call for coin flips, one randint(0, max_samples, (n_kept,)) call for
    target slots). The RNG stream produced by ``self._generator`` for any
    given seed will therefore differ from the scalar-loop implementation,
    yielding different *sample identities* in the final buffer. The marginal
    probability that any token survives the reservoir after N total tokens
    is unchanged at min(max_samples/N, 1.0), so the distribution is identical
    Algorithm R. When two kept tokens in the same batch select the same
    target slot, CPU ``index_copy_`` resolves in batch order (last write
    wins), matching the scalar loop's per-token sequential semantics
    exactly -- there is no statistical bias introduced by the vectorization.
    """

    def __init__(self, max_samples: int = 8192, *, seed: int = 0) -> None:
        self.max_samples = max_samples
        self.buffer: torch.Tensor | None = None
        self.seen = 0
        self._generator = torch.Generator(device="cpu").manual_seed(int(seed))

    def add(self, hidden: torch.Tensor) -> None:
        # Step 0: flatten and move to CPU — preserves existing contract.
        flat = hidden.reshape(-1, hidden.shape[-1]).detach().to("cpu")  # (n, H) — may be a view; clone/cat below ensures contiguity where needed
        n = flat.shape[0]
        if n == 0:
            return

        # ---------------------------------------------------------------
        # Phase A — First-ever call: deterministic prefix take.
        # Identical to current implementation; does NOT consume generator.
        # ---------------------------------------------------------------
        if self.buffer is None:
            take = min(n, self.max_samples)
            self.buffer = flat[:take].contiguous().clone()
            self.seen = n
            return

        # ---------------------------------------------------------------
        # Phase B — Buffer not yet full: fill remaining capacity first.
        # Also does NOT consume generator (every arriving token below cap
        # is kept unconditionally, probability = 1.0).
        # ---------------------------------------------------------------
        current_size = self.buffer.shape[0]
        if current_size < self.max_samples:
            remaining = self.max_samples - current_size
            fill_count = min(n, remaining)
            self.buffer = torch.cat(
                [self.buffer, flat[:fill_count]], dim=0
            ).contiguous()          # stays contiguous after cat
            self.seen += fill_count
            # If all n tokens fit below the cap, done.
            if fill_count == n:
                return
            # Otherwise trim flat to the unprocessed tail and fall through.
            flat = flat[fill_count:].contiguous()
            n = n - fill_count

        # ---------------------------------------------------------------
        # Phase C — Buffer is full (shape[0] == max_samples).
        # Vectorized Algorithm R for the remaining n tokens.
        # ---------------------------------------------------------------
        # pos[i] = 1-indexed global position of flat[i] in the entire stream
        # seen before this call  → (self.seen + 1) .. (self.seen + n)
        pos = torch.arange(
            self.seen + 1, self.seen + n + 1, dtype=torch.float64
        )                                              # (n,) float64 for precision

        # keep_prob[i] = min(max_samples / pos[i], 1.0)
        # clamp to 1.0 is theoretically inert in Phase C (pos >= max_samples+1 always,
        # so max_samples/pos < 1.0), but kept as defense-in-depth against any future
        # refactor that re-routes a partial-fill batch through Phase C.
        keep_probs = torch.clamp_max(
            self.max_samples / pos, 1.0
        ).to(torch.float32)                            # (n,) float32 for rand comparison

        # One uniform draw per token using the seeded generator.
        coin = torch.rand(n, generator=self._generator)   # (n,)  — CPU generator

        # Boolean mask of tokens that win their coin flip.
        keep_mask = coin < keep_probs                  # (n,) bool

        n_kept = int(keep_mask.sum())
        if n_kept > 0:
            # Indices into flat[] of kept tokens.
            kept_local = keep_mask.nonzero(as_tuple=False).squeeze(1)  # (n_kept,)

            # Uniform random target slot in [0, max_samples) per kept token.
            target_slots = torch.randint(
                0, self.max_samples, (n_kept,),
                generator=self._generator,
            )                                          # (n_kept,) int64

            # Vectorised in-place scatter: buffer[target_slots[k]] = flat[kept_local[k]]
            self.buffer.index_copy_(0, target_slots, flat[kept_local])

        self.seen += n

    def get(self) -> torch.Tensor | None:
        return self.buffer


def _profile_layer(
    model,
    layer_ref: MoELayerRef,
    batches,
    reap_acc: ReapAccumulator,
    cov_acc: InputCovarianceAccumulator,
    ream_acc: ReamCostAccumulator,
    *,
    device=None,
    layer_input_acc: "_LayerInputAccumulator | None" = None,
) -> None:
    """Profile a single MoE layer with early-exit forward.

    REAM sequential merging (paper 2604.04356, §4, Fig 1(b)) requires
    that each layer is profiled on hidden states reflecting all prior
    merges.  All metrics (REAP scores, REAM δ_gate/δ̃_expert, input
    covariance) depend only on hidden states arriving *at* this layer,
    not on downstream layers.  We therefore abort the forward pass
    immediately after this layer completes via :func:`early_exit_after_layer`,
    avoiding O(40−L) unnecessary layer-forwards per batch.

    Total layer-forwards across 40 sequential profiling passes:
    1+2+…+40 = 820 (vs 40×40 = 1600 without early exit).
    """
    layer_idx = layer_ref.layer_idx
    n_experts = layer_ref.num_routed_experts
    was_training = model.training
    model.eval()

    # Resolve `device` from model parameters if the caller left it None,
    # so finalize_batch's compute_device always lands on the model's GPU
    # rather than torch.cuda.current_device() (which may diverge in
    # multi-GPU or thread-context scenarios).
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    # Cumulative token offset: tracks the global start index of each batch.
    # Using cumulative addition (not batch_idx * fixed_size) handles the last
    # partial batch when num_calibration_samples % batch_size != 0.
    _batch_offset = 0  # cumulative token start of current batch
    _next_offset = 0   # cumulative token count after current batch

    # DIAG: per-hook time accumulators (input_cb, intermediate_cb, down_cb, call_count)
    # Reset at start of each batch by the batch loop below.
    import time as _diag_time_mod
    _diag_cb = [0.0, 0.0, 0.0, 0]
    # B-C-C-1: full-softmax cache for the current batch's router logits.
    # Spec §5 line 339 + D-ream-sparse-routing require σ(x)_e (the
    # un-renormalized full softmax over ALL experts), not the top-k
    # renormalized weights returned by Qwen3_5MoeTopKRouter.forward.
    # Populated by an experts-module pre-forward hook that runs AFTER the
    # router pre-forward hook (which captures the raw logits) but BEFORE any
    # expert forward (which fires down_cb). down_cb reads _full_softmax[0]
    # to obtain σ(x)_e at active token positions.
    _full_softmax: list[torch.Tensor | None] = [None]

    def input_cb(li, e, tensor, ctx):
        _t = _diag_time_mod.monotonic()
        cov_acc.update(li, e, "gate_proj", tensor)
        _diag_cb[0] += _diag_time_mod.monotonic() - _t
        _diag_cb[3] += 1

    def intermediate_cb(li, e, tensor, ctx):
        _t = _diag_time_mod.monotonic()
        cov_acc.update(li, e, "down_proj", tensor)
        ream_acc.record_neuron_activations(li, e, tensor)
        _diag_cb[1] += _diag_time_mod.monotonic() - _t

    def down_cb(li, e, tensor, ctx):
        _t = _diag_time_mod.monotonic()
        # _batch_offset is only read here, never assigned; no nonlocal declaration needed.
        record_reap(reap_acc, li, e, ctx["top_k_weights"], tensor)
        # B-C-C-1: pass σ(x)_e (full softmax over all experts) at active token
        # positions for this expert, NOT ctx["top_k_weights"] (renormalized to
        # sum=1 over top-k). The pre-forward hook installed below populates
        # _full_softmax[0] before any expert forward fires.
        token_idx = ctx["token_idx"]
        fs = _full_softmax[0]
        if fs is not None:
            # Index the cached [T, n_experts] full-softmax tensor at the
            # active token positions for this expert. fs lives on the same
            # device as the router logits (typically GPU); index with
            # token_idx on its native device to avoid a CPU↔GPU round-trip.
            # Result shape: [|active|]; ensure it lands on the expert-output
            # device for the downstream (gate * expert_output) multiply.
            sigma_e = fs[token_idx.to(fs.device), e].to(tensor.device)
        else:
            # B-iter5-L-4 (code): hook ordering is spec-required to populate
            # _full_softmax[0] before any expert forward fires; reaching this
            # branch indicates a real ordering bug. Log at ERROR so it is not
            # missed; keep the fallback to top_k_weights so the run completes.
            log.error(
                "down_cb: full-softmax cache empty for layer %d expert %d — "
                "falling back to top_k_weights (renormalized; spec-degraded). "
                "This indicates a hook-ordering bug — the experts pre-forward "
                "hook (_populate_full_softmax) should always run before any "
                "expert forward fires.",
                li, e,
            )
            sigma_e = ctx["top_k_weights"]
        ream_acc.record_gated_output(
            li, e, sigma_e, tensor,
            token_idx, _batch_offset,
        )
        _diag_cb[2] += _diag_time_mod.monotonic() - _t

    # B-C-C-1: pre-forward hook on the experts module that computes the full
    # softmax from the latest captured router logits. Runs after the router
    # pre-forward hook (which appends to router_logits_storage[layer_idx])
    # but before the experts forward (which fires down_cb). Because
    # capture_router_outputs's hook is a *router* pre-forward hook and this
    # one is an *experts* pre-forward hook, ordering is guaranteed by the
    # decoder layer's call sequence (router runs first, dispatches to experts).
    def _populate_full_softmax(_module, _inputs):
        if router_logits_storage[layer_idx]:
            batch_logits = router_logits_storage[layer_idx][-1]
            # F.softmax over the last (expert) dim → [T, n_experts] σ(x)_e values.
            # .float() avoids dtype mismatch when the router runs in bf16.
            # Keep on-device (router-logits device) — down_cb indexes with
            # on-device token_idx, avoiding a CPU↔GPU round-trip per expert.
            _full_softmax[0] = F.softmax(batch_logits.float(), dim=-1)
        else:
            _full_softmax[0] = None

    try:
        with instrument_experts(
            layer_ref,
            {"input": input_cb, "intermediate": intermediate_cb, "down": down_cb},
        ), capture_router_outputs([layer_ref]) as router_logits_storage, \
             early_exit_after_layer(model, layer_idx):
            # Install the experts pre-forward hook AFTER capture_router_outputs
            # so the router hook fires first per batch.
            _experts_handle = layer_ref.experts_module.register_forward_pre_hook(
                _populate_full_softmax
            )
            # Phase 3: optionally capture the layer-input hidden states for
            # per-merge-group expert distillation (spec § 5 step 7b / M8).
            # Hook on the decoder layer module — its first input is the
            # hidden_states tensor that the layer's forward operates on.
            _layer_in_handle = None
            if layer_input_acc is not None:
                def _capture_layer_input(_module, inputs):
                    if inputs and inputs[0] is not None:
                        layer_input_acc.add(inputs[0])
                _layer_in_handle = layer_ref.layer_module.register_forward_pre_hook(
                    _capture_layer_input
                )
            try:
                # DIAG: layer-1 hang investigation — log every batch so we can see
                # if the forward pass is making progress and how long each batch takes.
                import time as _diag_time
                _diag_t0 = _diag_time.monotonic()
                _diag_count = 0
                log.info("DIAG layer %d: entering batch loop (calibration tensor + early-exit forwards) | %s", layer_idx, _rt_snap())
                _rt_update(stage="stage2", layer=int(layer_idx), batch=0, phase="profile_layer_start")
                for batch in batches:
                    _diag_t_batch = _diag_time.monotonic()
                    # Reset per-batch hook timers
                    _diag_cb[0] = 0.0; _diag_cb[1] = 0.0; _diag_cb[2] = 0.0; _diag_cb[3] = 0
                    # `device` is guaranteed non-None after the resolution block
                    # at the top of _profile_layer.
                    batch = batch.to(device)
                    _batch_offset = _next_offset
                    router_logits_storage[layer_idx].clear()
                    _full_softmax[0] = None
                    _diag_t_fwd = _diag_time.monotonic()
                    with torch.no_grad():
                        try:
                            model(input_ids=batch)
                        except _EarlyExitException:
                            pass  # expected — target layer completed
                    _diag_fwd_dt = _diag_time.monotonic() - _diag_t_fwd
                    if router_logits_storage[layer_idx]:
                        batch_logits = router_logits_storage[layer_idx][-1]
                        ream_acc.record_router_logits(layer_idx, batch_logits, _batch_offset)
                    ream_acc.finalize_batch(layer_idx, n_experts, compute_device=device)
                    ream_acc.record_batch_token_count(layer_idx, batch.shape[0] * batch.shape[1])
                    _next_offset += batch.shape[0] * batch.shape[1]
                    _diag_count += 1
                    _rt_update(stage="stage2", layer=int(layer_idx), batch=int(_diag_count),
                               phase="profile_layer_batch")
                    _diag_dt = _diag_time.monotonic() - _diag_t_batch
                    if _diag_count <= 3 or _diag_count % 10 == 0:
                        log.info(
                            "DIAG layer %d batch %d: total=%.2fs fwd=%.2fs hooks: input=%.2fs intermed=%.2fs down=%.2fs (n_cb=%d) | cum=%.1fs | %s",
                            layer_idx, _diag_count, _diag_dt, _diag_fwd_dt,
                            _diag_cb[0], _diag_cb[1], _diag_cb[2], _diag_cb[3],
                            _diag_time.monotonic() - _diag_t0, _rt_snap(),
                        )
                log.info("DIAG layer %d: batch loop complete — %d batches in %.1fs, now post-profile work | %s",
                         layer_idx, _diag_count, _diag_time.monotonic() - _diag_t0, _rt_snap())
            finally:
                _experts_handle.remove()
                if _layer_in_handle is not None:
                    _layer_in_handle.remove()
    finally:
        if was_training:
            model.train()
