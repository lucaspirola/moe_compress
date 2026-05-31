"""Stage 2 profile-pass sidecar WRITER (Plugin #12 REDO — Optimization A).

This module is the writer half of the cache-or-live provider pair. It is
imported by ``build_self_traces_calib_vllm.py`` under the alias
``vllm.calibration_stage2_profile`` (the actual import target after the
vLLM patch is applied lives at ``vllm/calibration_stage2_profile.py``;
the file here is the canonical source — the patch ships a verbatim copy
into the vLLM tree).

Design — single writer, one sidecar, four data streams:
  * Router hook → gate logit profiles (Bug #2 fix: raw per-batch list,
    NOT pre-collapsed).
  * Expert-out-unweighted hook → per-batch ``(token_indices, gated)``
    per expert, fed into a real :class:`ReamCostAccumulator` so the
    Eq. 8 numerator is computed via the SAME finalize_batch code path
    the live profiling uses (Bug #1 fix: per-token jointly-active pair
    cosines, NOT cos(mean_i, mean_j)).
  * Layer-input hook → :class:`_LayerInputAccumulator` per layer rank
    (Crit-3 fix: always captured when --capture-stage2-profile is on).
  * Per-batch token-count → fed into the same ReamCostAccumulator's
    ``record_batch_token_count`` for the Eq. 8 denominator (Bug #3 fix:
    EXACT |X|, NOT Sum_e token_counts_e).

Cov accumulation is fed by the ``expert_in`` (gate_proj) + ``expert_mid``
(down_proj) hooks (:func:`_expert_in_handler` / :func:`_expert_mid_handler`),
which reconstruct the SAME per-(token,slot) row set the live Stage-2
``instrument_experts`` path uses (``torch.where(mask[e])`` -> ``token_idx``)
and feed it through the shared :class:`InputCovarianceAccumulator.update`.
gate cov is ``[hidden_size, hidden_size]``; down cov is
``[moe_intermediate_size, moe_intermediate_size]``. ``setup`` calls
``set_storage_dtype`` IMMEDIATELY so the writer never relies on the default
fp32 dtype at ``activation_hooks.py:961``.

Public API used by the driver:
    * :func:`setup`
    * :func:`dump_stage2_profile`
    * :func:`dump_stage2_profile_checkpoint`
    * :func:`load_stage2_profile_checkpoint`
    * :func:`set_n_prompts_accumulated`
    * :func:`get_n_prompts_accumulated`
    * :func:`record_batch_token_count`

Internal callbacks exposed for unit testing (plan section 8.1):
    * :func:`_on_router_callback`
    * :func:`_on_expert_out_unweighted_callback`
    * :func:`_on_layer_in_callback`
    * :func:`_finalize_batch_for_layer`
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from ..stage2.profiling import _LayerInputAccumulator
from ..utils.activation_hooks import (
    InputCovarianceAccumulator,
    ReamCostAccumulator,
)
from ..utils.cached_calibration_signals import (
    SCHEMA_VERSIONS,
    Stage2ProfilePayloadV4,
    save_stage2_profile_v4,
)

log = logging.getLogger(__name__)

# Reservoir cap for the per-layer input-tokens streaming sample (Plugin #1
# Opt-C / Vitter Algorithm R). Matches the default at
# ``moe_compress.stage2.profiling._LayerInputAccumulator.__init__`` so the
# captured calibration set is identical whether the data arrives from the
# vLLM `layer_in` hook (this writer) or from the live profile pass
# (``moe_compress.stage2.profiling._profile_layer``). Env-overridable so SC
# experiments can raise it (per PLAN_PLUGIN_14:284 -- bounded by the host
# RAM budget). Sampled at import; new processes pick up overrides.
_LAYER_INPUT_MAX_SAMPLES: int = int(
    os.getenv("VLLM_CALIB_LAYER_INPUT_MAX_SAMPLES", "8192")
)


@dataclass
class _WriterState:
    """Single shared accumulator state for the run.

    Held module-level (the writer is a singleton per process; the
    pre-import env-var gate VLLM_CALIB_CAPTURE_STAGE2_PROFILE keeps a
    stale state from a previous process from leaking through). Reset by
    :func:`setup`.
    """
    ream_acc: ReamCostAccumulator = field(default_factory=ReamCostAccumulator)
    cov_acc: InputCovarianceAccumulator = field(default_factory=InputCovarianceAccumulator)
    # Per-layer-idx streaming reservoirs. Populated by the `layer_in`
    # callback via :class:`_LayerInputAccumulator` (Vitter Algorithm R,
    # per-layer seeded by ``layer_idx``). Finalised to a bf16 ``[N, hidden]``
    # tensor list at dump time. Empty dict on runs without the
    # ``VLLM_CALIB_CAPTURE_LAYER_IN=1`` env gate.
    layer_input_reservoir: dict[int, _LayerInputAccumulator] = field(default_factory=dict)
    layer_idx_to_rank: dict[int, int] = field(default_factory=dict)
    rank_to_layer_idx: dict[int, int] = field(default_factory=dict)
    n_layers: int = 0
    n_experts: int = 0
    top_k: int = 0
    model_hash: str = ""
    configured_cov_storage_dtype: str = "float16"
    n_prompts_accumulated: int = 0


_state: _WriterState = _WriterState()


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------
def setup(
    llm: Any | None = None,
    *,
    cov_storage_dtype: str = "float16",
    n_layers: int | None = None,
    n_experts: int | None = None,
    top_k: int | None = None,
    model_hash: str | None = None,
    layer_idx_to_rank: dict[int, int] | None = None,
) -> None:
    """Initialize the writer state.

    Called once from ``build_self_traces_calib_vllm.py`` after the vLLM
    LLM object exists but BEFORE any prompt is sampled (so the
    callbacks register before the first router hook fires).

    The env gate VLLM_CALIB_CAPTURE_STAGE2_PROFILE is checked by the
    upstream callback dispatcher; ``setup`` itself is unconditional --
    callers that did not set the env should not invoke us.

    Args:
        llm: the vLLM LLM object. When ``None``, the function operates
            in test / standalone mode (no model introspection).
        cov_storage_dtype: one of {"float16","bfloat16","float32"}.
            Pinned IMMEDIATELY on the writer's cov_acc so the default
            fp32 at ``activation_hooks.py:961`` is overridden before
            any ``update`` call can land.
        n_layers / n_experts / top_k / model_hash: optional cross-
            validation metadata.
        layer_idx_to_rank: optional explicit layer_idx -> layer_rank
            mapping.
    """
    global _state
    _state = _WriterState()
    if cov_storage_dtype not in ("float16", "bfloat16", "float32"):
        raise ValueError(
            f"setup: cov_storage_dtype={cov_storage_dtype!r} not in "
            f"{{'float16','bfloat16','float32'}}"
        )
    cov_dtype = getattr(torch, cov_storage_dtype)
    _state.cov_acc.set_storage_dtype(cov_dtype)
    _state.configured_cov_storage_dtype = cov_storage_dtype

    if n_layers is not None:
        _state.n_layers = int(n_layers)
    if n_experts is not None:
        _state.n_experts = int(n_experts)
        _state.ream_acc.num_experts = int(n_experts)
    if top_k is not None:
        _state.top_k = int(top_k)
    if model_hash is not None:
        _state.model_hash = str(model_hash)

    if layer_idx_to_rank is not None:
        _state.layer_idx_to_rank = dict(layer_idx_to_rank)
        _state.rank_to_layer_idx = {r: li for li, r in _state.layer_idx_to_rank.items()}
        if _state.n_layers == 0:
            _state.n_layers = len(_state.layer_idx_to_rank)
    elif llm is not None:
        try:
            _populate_layer_map_from_llm(llm)
        except Exception as exc:
            log.warning(
                "stage2-profile-writer: setup couldn't discover layers "
                "from llm (%s)", exc,
            )

    log.info(
        "stage2-profile-writer: setup ok (cov_storage_dtype=%s, "
        "n_layers=%d, n_experts=%d, top_k=%d)",
        cov_storage_dtype, _state.n_layers, _state.n_experts, _state.top_k,
    )


def _populate_layer_map_from_llm(llm: Any) -> None:
    """Best-effort layer-rank discovery on the vLLM LLM object."""
    model = None
    for attr_path in (
        ("llm_engine", "model_executor", "driver_worker", "model_runner", "model"),
        ("llm_engine", "model_executor", "model"),
    ):
        node = llm
        for a in attr_path:
            node = getattr(node, a, None)
            if node is None:
                break
        if node is not None:
            model = node
            break
    if model is None:
        return
    ranks: dict[int, int] = {}
    next_rank = 0
    for name, _mod in getattr(model, "named_modules", lambda: [])():
        if "layers" not in name:
            continue
        if "moe" not in name.lower() and "experts" not in name.lower():
            continue
        for tok in name.split("."):
            if tok.isdigit():
                li = int(tok)
                if li not in ranks:
                    ranks[li] = next_rank
                    next_rank += 1
                break
    if ranks:
        _state.layer_idx_to_rank = ranks
        _state.rank_to_layer_idx = {r: li for li, r in ranks.items()}
        if _state.n_layers == 0:
            _state.n_layers = len(ranks)


def set_n_prompts_accumulated(n: int) -> None:
    _state.n_prompts_accumulated = int(n)


def get_n_prompts_accumulated() -> int:
    return int(_state.n_prompts_accumulated)


def record_batch_token_count(layer_idx: int, n_tokens: int) -> None:
    """Forward to ReamCostAccumulator.record_batch_token_count."""
    _state.ream_acc.record_batch_token_count(int(layer_idx), int(n_tokens))


# ---------------------------------------------------------------------------
# Internal callbacks.
# ---------------------------------------------------------------------------
def _on_router_callback(
    layer_idx: int, logits: torch.Tensor, batch_offset: int,
) -> None:
    """Router hook handler -- folds per-batch logits into the online _gate_gram."""
    _state.ream_acc.record_router_logits(
        int(layer_idx), logits, int(batch_offset),
    )


def _on_expert_out_unweighted_callback(
    layer_idx: int,
    expert_idx: int,
    gated: torch.Tensor,
    token_indices: torch.Tensor,
    batch_offset: int,
    *,
    gate_weights: torch.Tensor | None = None,
) -> None:
    """Expert-output-unweighted hook handler.

    ``gated`` should be the WEIGHTED expert output sigma(x)_e * E_e(x). If
    callers pass the unweighted output separately as ``gate_weights``,
    the multiply is done here.
    """
    if gate_weights is not None:
        _state.ream_acc.record_gated_output(
            int(layer_idx), int(expert_idx),
            gate_weights=gate_weights,
            expert_output=gated,
            token_indices=token_indices,
            batch_offset=int(batch_offset),
        )
    else:
        T = int(token_indices.numel())
        ones = torch.ones((T,), dtype=gated.dtype, device=gated.device)
        _state.ream_acc.record_gated_output(
            int(layer_idx), int(expert_idx),
            gate_weights=ones,
            expert_output=gated,
            token_indices=token_indices,
            batch_offset=int(batch_offset),
        )


def _expert_in_handler(**kw: Any) -> None:
    """``expert_in`` hook -- gate_proj input covariance per expert.

    Mirrors the LIVE Stage-2 ``instrument_experts`` row set exactly: the
    "input"/gate_proj cb receives ``hidden_states[token_idx]`` where
    ``token_idx`` comes from ``torch.where(mask[e])`` (per-(token,slot)).
    The vLLM ``expert_in`` dispatch hands us the full ``hidden_states
    [n_tok, hidden]`` + ``topk_ids [n_tok, top_k]``; we reconstruct the
    per-slot row set for each expert ``e`` and feed it through the shared
    ``InputCovarianceAccumulator.update`` (the SAME accumulator the live
    path uses; ``subᵀ@sub`` + token_count are accumulated internally).

    Per-slot ``(topk_ids == e).nonzero()`` rows match the live contract and
    are robust to a future sampler that repeats an expert within a row;
    under ``torch.topk`` (distinct experts per token) this equals the
    unique-token mask. CPU-resident ``sub`` (option A): the matmul runs on
    CPU so it is safe under vLLM CUDA-graph capture.
    """
    if _state.cov_acc is None:
        return
    try:
        layer_idx = int(kw["layer_idx"])
        hidden = kw["hidden_states"]
        topk_ids = kw["topk_ids"]
    except KeyError:
        return
    hs = hidden.detach().reshape(-1, hidden.shape[-1]).to("cpu")  # [n_tok, hidden]
    ids = topk_ids.detach().to("cpu")
    if ids.dim() == 1:
        ids = ids.unsqueeze(-1)  # -> [n_tok, top_k]
    n_experts = _state.n_experts or (
        int(ids.max().item()) + 1 if ids.numel() else 0
    )
    for e in range(n_experts):
        # Per-(token,slot) rows: (topk_ids == e).nonzero() -> (token, slot);
        # column 0 is the per-slot token index (mirrors torch.where(mask[e])).
        slot_tok = (ids == e).nonzero(as_tuple=False)  # [n_assign, 2]
        if slot_tok.numel() == 0:
            continue
        token_idx = slot_tok[:, 0]
        sub = hs.index_select(0, token_idx)  # [n_assign, hidden] (CPU)
        _state.cov_acc.update(layer_idx, e, "gate_proj", sub)


def _expert_mid_handler(**kw: Any) -> None:
    """``expert_mid`` hook -- down_proj input covariance per expert.

    The live ``intermediate``/down_proj cb uses the SAME per-(token,slot)
    axis as gate (both flow from ``torch.where(mask[e])`` -> ``token_idx``
    in ``instrument_experts``); there is NO gate/down axis difference in the
    live cov path. The vLLM ``expert_mid`` dispatch hands us ``intermediate
    [n_tok, top_k, interm]`` + ``topk_ids [n_tok, top_k]``; we flatten to
    ``[n_tok*top_k, interm]`` (naturally per-slot — the kernel runs one
    down_proj matmul per (token, slot) pair) and select the per-slot rows
    for each expert. CPU matmul (option A) for CUDA-graph safety.
    """
    if _state.cov_acc is None:
        return
    try:
        layer_idx = int(kw["layer_idx"])
        interm = kw["intermediate"]
        topk_ids = kw["topk_ids"]
    except KeyError:
        return
    interm_dim = interm.shape[-1]
    flat = interm.detach().reshape(-1, interm_dim).to("cpu")  # [n_tok*top_k, interm]
    flat_ids = topk_ids.detach().reshape(-1).to("cpu")  # [n_tok*top_k]
    n_experts = _state.n_experts or (
        int(flat_ids.max().item()) + 1 if flat_ids.numel() else 0
    )
    for e in range(n_experts):
        rows = (flat_ids == e).nonzero(as_tuple=False).reshape(-1)
        if rows.numel() == 0:
            continue
        sub = flat.index_select(0, rows)  # [n_e_slots, interm] (CPU)
        _state.cov_acc.update(layer_idx, e, "down_proj", sub)


def _finalize_batch_for_layer(layer_idx: int, n_experts: int) -> None:
    """Drain _batch_gated_indexed and accumulate the Eq. 8 numerator.

    Direct passthrough to :meth:`ReamCostAccumulator.finalize_batch`,
    which mirrors lines 260-448 of ``activation_hooks.py`` exactly
    (Bug #1 fix: per-token jointly-active pair cosines, NOT
    cos(mean_i, mean_j)).
    """
    _state.ream_acc.finalize_batch(int(layer_idx), int(n_experts))


def _record_neuron_act(
    layer_idx: int, expert_idx: int, neuron_act_sum: torch.Tensor,
    n_tokens: int,
) -> None:
    """Test helper -- stash neuron mean state directly."""
    key = (int(layer_idx), int(expert_idx))
    _state.ream_acc._neuron_act_sum[key] = neuron_act_sum.detach().to(torch.float32).cpu()
    _state.ream_acc._neuron_act_count[key] = int(n_tokens)


def _on_layer_in_callback(
    layer_idx: int, hidden: torch.Tensor,
) -> None:
    """``layer_in`` hook handler -- stream per-batch layer input into the reservoir.

    Constructs a per-layer :class:`_LayerInputAccumulator` lazily on first
    sighting (seed = layer_idx for per-layer determinism, matching the live
    profile pass's contract at ``moe_compress.stage2.profiling._profile_layer``).
    Subsequent calls feed the Vitter Algorithm R reservoir via the
    accumulator's vectorised ``add`` method (Plugin #1 Opt-C).
    """
    li = int(layer_idx)
    acc = _state.layer_input_reservoir.get(li)
    if acc is None:
        # Per-layer seed = layer_idx (cross-run determinism + per-layer
        # independence; matches profiling.py:86).
        acc = _LayerInputAccumulator(
            max_samples=_LAYER_INPUT_MAX_SAMPLES, seed=li,
        )
        _state.layer_input_reservoir[li] = acc
    acc.add(hidden)


def _record_layer_input_reservoir(
    layer_idx: int, reservoir: torch.Tensor,
) -> None:
    """Test helper -- stash a pre-finalised reservoir snapshot directly.

    Preserves the original ``dict[int, Tensor]`` test-API contract (tests
    call this to inject a known reservoir without driving the streaming
    callback). Builds a fresh accumulator with the layer-seeded RNG, sets
    ``.buffer`` and ``.seen`` from the snapshot, and stores it on the
    state. Downstream consumers (``dump_stage2_profile``) read the
    ``.buffer`` field uniformly.
    """
    li = int(layer_idx)
    acc = _LayerInputAccumulator(
        max_samples=_LAYER_INPUT_MAX_SAMPLES, seed=li,
    )
    buf = reservoir.detach().to("cpu", dtype=torch.bfloat16).contiguous()
    acc.buffer = buf
    acc.seen = int(buf.size(0))
    _state.layer_input_reservoir[li] = acc


# ---------------------------------------------------------------------------
# Dump.
# ---------------------------------------------------------------------------
def dump_stage2_profile(jsonl_path: Path | str) -> None:
    """Serialize the writer state into a schema-v3 sidecar.

    Per plan section 10 writer-side serialization order:
      1. Finalize all pending GPU covariances for each layer.
      2. Cross-validate cov_acc.storage_dtype against the configured value.
      3. Build Stage2ProfilePayloadV4 with layer_rank-keyed dicts.
      4. Atomic torch.save via save_stage2_profile_v4.
    """
    jsonl_path = Path(jsonl_path)

    # Step 1.
    if _state.rank_to_layer_idx:
        for rank in sorted(_state.rank_to_layer_idx):
            layer_idx = _state.rank_to_layer_idx[rank]
            _state.cov_acc.finalize_layer(layer_idx)
    else:
        _state.cov_acc.finalize_all()

    # Step 2 -- assert configured cov_storage_dtype matches the live value.
    live_dtype_str = str(_state.cov_acc.storage_dtype).split(".")[-1]
    if live_dtype_str != _state.configured_cov_storage_dtype:
        raise AssertionError(
            f"dump_stage2_profile: cov_acc.storage_dtype="
            f"{live_dtype_str!r} != configured "
            f"{_state.configured_cov_storage_dtype!r}. A code path "
            f"mutated storage_dtype after setup."
        )

    # Step 3 -- translate layer_idx-keyed accumulator data to layer_rank-
    # keyed payload dicts.
    if _state.rank_to_layer_idx:
        layer_ranks = sorted(_state.rank_to_layer_idx)
        n_layers = len(layer_ranks)
        l2r = _state.layer_idx_to_rank
    else:
        observed = (
            set(_state.ream_acc._gate_gram)
            | {k[0] for k in _state.cov_acc.covariance}
            | set(_state.layer_input_reservoir)
        )
        layer_ranks = sorted(observed)
        n_layers = len(layer_ranks)
        l2r = {li: i for i, li in enumerate(layer_ranks)}
        for li, r in l2r.items():
            _state.rank_to_layer_idx[r] = li
            _state.layer_idx_to_rank[li] = r

    n_experts = _state.n_experts
    if n_experts == 0:
        candidates: set[int] = set()
        for (_, e, _) in _state.cov_acc.covariance:
            candidates.add(int(e))
        for (_, e) in _state.ream_acc._neuron_act_sum:
            candidates.add(int(e))
        n_experts = (max(candidates) + 1) if candidates else 0

    if n_experts > 0:
        sim_tensor = torch.zeros(
            (n_layers, n_experts, n_experts), dtype=torch.float64,
        )
        for rank in range(n_layers):
            layer_idx = _state.rank_to_layer_idx[rank]
            row = _state.ream_acc._sim_tensor.get(layer_idx)
            if row is not None:
                sim_tensor[rank] = row.detach().to(torch.float64).cpu()
    else:
        sim_tensor = torch.zeros((n_layers, 0, 0), dtype=torch.float64)

    total_tokens = torch.zeros((n_layers,), dtype=torch.int64)
    for rank in range(n_layers):
        layer_idx = _state.rank_to_layer_idx[rank]
        total_tokens[rank] = int(
            _state.ream_acc._total_tokens_by_layer.get(layer_idx, 0)
        )

    # δ_gate router-logit Gram: a bounded [n_layers, E, E] fp64 tensor mirroring
    # the sim_tensor build loop. _gate_gram[layer] is the full [E, E] Gram over
    # all experts; rehydrated into ReamCostAccumulator._gate_gram by the reader.
    if n_experts > 0:
        gate_gram = torch.zeros(
            (n_layers, n_experts, n_experts), dtype=torch.float64,
        )
        for rank in range(n_layers):
            layer_idx = _state.rank_to_layer_idx[rank]
            g = _state.ream_acc._gate_gram.get(layer_idx)
            if g is not None:
                gate_gram[rank] = g.detach().to(torch.float64).cpu()
    else:
        gate_gram = torch.zeros((n_layers, 0, 0), dtype=torch.float64)

    neuron_act_sum: dict = {}
    neuron_act_count: dict = {}
    for (layer_idx, e), v in _state.ream_acc._neuron_act_sum.items():
        rank = l2r.get(layer_idx)
        if rank is None:
            continue
        neuron_act_sum[(rank, int(e))] = v.detach().to(
            "cpu", dtype=torch.float32,
        ).contiguous()
    for (layer_idx, e), c in _state.ream_acc._neuron_act_count.items():
        rank = l2r.get(layer_idx)
        if rank is None:
            continue
        neuron_act_count[(rank, int(e))] = int(c)

    cov_dtype = getattr(torch, _state.configured_cov_storage_dtype)
    cov_payload: dict = {}
    cov_tc: dict = {}
    for (layer_idx, e, m), v in _state.cov_acc.covariance.items():
        rank = l2r.get(layer_idx)
        if rank is None:
            continue
        cov_payload[(rank, int(e), str(m))] = v.detach().to(
            "cpu", dtype=cov_dtype, copy=True,
        ).contiguous()
    for (layer_idx, e, m), n in _state.cov_acc.token_count.items():
        rank = l2r.get(layer_idx)
        if rank is None:
            continue
        cov_tc[(rank, int(e), str(m))] = int(n)

    # Finalise each per-layer accumulator to a bf16 ``[N, hidden]`` tensor.
    # Empty/missing layers fall back to the legacy ``(0, 0)`` placeholder
    # so the schema-v3 reader's empty-shape guard keeps working as a
    # defensive fallback (per plan §5.a).
    layer_input_reservoir: list = []
    for rank in range(n_layers):
        layer_idx = _state.rank_to_layer_idx[rank]
        acc = _state.layer_input_reservoir.get(layer_idx)
        if acc is None or acc.buffer is None or acc.buffer.numel() == 0:
            layer_input_reservoir.append(
                torch.zeros((0, 0), dtype=torch.bfloat16)
            )
        else:
            layer_input_reservoir.append(
                acc.buffer.detach().to("cpu", dtype=torch.bfloat16).contiguous()
            )

    payload = Stage2ProfilePayloadV4(
        format_version=4,
        schema_version=SCHEMA_VERSIONS["stage2_profile"],
        model_hash=_state.model_hash,
        n_layers=n_layers,
        n_experts=int(n_experts),
        top_k=int(_state.top_k),
        cov_storage_dtype=_state.configured_cov_storage_dtype,
        total_tokens_per_layer=total_tokens,
        gate_gram=gate_gram,
        sim_tensor=sim_tensor,
        neuron_act_sum=neuron_act_sum,
        neuron_act_count=neuron_act_count,
        cov_acc=cov_payload,
        cov_token_count=cov_tc,
        layer_input_reservoir=layer_input_reservoir,
    )
    save_stage2_profile_v4(payload, jsonl_path)
    # Per plan §2.e: replace the prior H-1 "no hook yet" warning with an
    # info-log reporting how many of the ``n_layers`` reservoirs got
    # populated. All-empty after CRITICAL-1 landing typically means the
    # operator forgot to set ``VLLM_CALIB_CAPTURE_LAYER_IN=1`` (the new
    # capture env gate) -- escalate to a warning in that case so it
    # surfaces in the driver logs.
    populated = sum(1 for t in layer_input_reservoir if t.numel() > 0)
    if layer_input_reservoir and populated == 0:
        log.warning(
            "stage2-profile-writer: layer_input_reservoir is empty for "
            "all %d layers -- VLLM_CALIB_CAPTURE_LAYER_IN was not set "
            "to '1' (or no layer_in callback fired). SC "
            "cost_alignment='output' will fall back to the live forward "
            "pass for full-hit layers.",
            n_layers,
        )
    else:
        log.info(
            "stage2-profile-writer: layer_input_reservoir populated for "
            "%d/%d layers", populated, n_layers,
        )
    log.info(
        "stage2-profile-writer: wrote %d-layer x %d-expert sidecar "
        "(cov_storage_dtype=%s, n_prompts=%d) next to %s",
        n_layers, n_experts, _state.configured_cov_storage_dtype,
        _state.n_prompts_accumulated, jsonl_path,
    )


# ---------------------------------------------------------------------------
# Checkpoint -- crash-resume serialization.
# ---------------------------------------------------------------------------
_CKPT_SCHEMA = 2


def dump_stage2_profile_checkpoint(path: str | Path) -> None:
    """Atomic write the live writer state to ``path``.

    Uses torch.save (the same pickle-backed primitive every other
    sidecar in this repo uses); the file is consumed only by the
    matching ``load_stage2_profile_checkpoint`` in the same trusted
    process tree.
    """
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    state_payload = {
        "format_version": _CKPT_SCHEMA,
        "schema_version": _CKPT_SCHEMA,
        "n_prompts_accumulated": _state.n_prompts_accumulated,
        "configured_cov_storage_dtype": _state.configured_cov_storage_dtype,
        "n_layers": _state.n_layers,
        "n_experts": _state.n_experts,
        "top_k": _state.top_k,
        "model_hash": _state.model_hash,
        "layer_idx_to_rank": dict(_state.layer_idx_to_rank),
        "ream_acc_sim_tensor": dict(_state.ream_acc._sim_tensor),
        "ream_acc_total_tokens": dict(_state.ream_acc._total_tokens_by_layer),
        "ream_acc_gate_gram": dict(_state.ream_acc._gate_gram),
        "ream_acc_neuron_act_sum": dict(_state.ream_acc._neuron_act_sum),
        "ream_acc_neuron_act_count": dict(_state.ream_acc._neuron_act_count),
        "cov_acc_covariance": dict(_state.cov_acc.covariance),
        "cov_acc_token_count": dict(_state.cov_acc.token_count),
        "cov_acc_storage_dtype": str(_state.cov_acc.storage_dtype).split(".")[-1],
        # Serialise each accumulator as ``(buffer, seen, max_samples,
        # generator_state)``. The seed is recovered from the layer_idx
        # contract (the accumulator was constructed with seed=layer_idx),
        # but seed alone is NOT enough once Phase C has consumed RNG draws
        # -- a fresh ``manual_seed(layer_idx)`` would re-emit the already-
        # consumed prefix of the stream. We therefore serialise the live
        # ``torch.Generator.get_state()`` tensor (uint8[5056] on CPU) and
        # restore it via ``set_state`` on load so the resumed RNG stream
        # is byte-identical to the non-resumed path even after Phase C
        # entry (profiling.py:57-79 determinism contract; reservoir Phase C
        # at profiling.py:128 is where the generator first gets consumed).
        "layer_input_reservoir": {
            int(li): (
                None if acc.buffer is None else acc.buffer.detach().cpu().clone(),
                int(acc.seen),
                int(acc.max_samples),
                acc._generator.get_state().clone(),
            )
            for li, acc in _state.layer_input_reservoir.items()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_payload, tmp)
    os.replace(tmp, path)


def load_stage2_profile_checkpoint(path: str | Path) -> int:
    """Hydrate the writer state from a checkpoint. Returns prompts seen.

    Raises ``ValueError`` on schema mismatch -- the caller deletes the
    stale file and restarts from zero (mirrors imatrix / reap-scores).
    """
    path = Path(path)
    if not path.exists():
        return 0
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    if int(loaded.get("schema_version", -1)) != _CKPT_SCHEMA:
        raise ValueError(
            f"stage2-profile checkpoint at {path} has schema_version="
            f"{loaded.get('schema_version')!r}, expected {_CKPT_SCHEMA}. "
            f"Delete the checkpoint to regenerate."
        )
    _state.n_prompts_accumulated = int(loaded.get("n_prompts_accumulated", 0))
    # L-2 (round 2): cross-validate the LIVE operator-configured cov
    # storage dtype (set by ``setup``) against the value the writer was
    # running when the checkpoint was dumped. Capture the live value
    # BEFORE overwriting `_state.configured_cov_storage_dtype` so the
    # check catches divergence between operator intent and the resumed
    # checkpoint (round-1 fix compared the checkpoint against itself).
    live_configured_cov_storage_dtype = _state.configured_cov_storage_dtype
    checkpoint_configured_cov_storage_dtype = str(
        loaded.get(
            "configured_cov_storage_dtype",
            live_configured_cov_storage_dtype,
        ),
    )
    if (
        checkpoint_configured_cov_storage_dtype
        != live_configured_cov_storage_dtype
    ):
        raise ValueError(
            f"Checkpoint configured_cov_storage_dtype mismatch: live="
            f"{live_configured_cov_storage_dtype!r}, checkpoint="
            f"{checkpoint_configured_cov_storage_dtype!r}. Delete the "
            f"checkpoint to regenerate."
        )
    _state.configured_cov_storage_dtype = (
        checkpoint_configured_cov_storage_dtype
    )
    # Secondary check: the cov_acc's own storage_dtype tag in the
    # checkpoint must agree with the configured value (same fail-loud
    # pattern as the dump-time assert in dump_stage2_profile).
    cov_acc_storage_dtype_from_payload = str(
        loaded.get(
            "cov_acc_storage_dtype",
            _state.configured_cov_storage_dtype,
        ),
    )
    if (
        cov_acc_storage_dtype_from_payload
        != _state.configured_cov_storage_dtype
    ):
        raise ValueError(
            f"Checkpoint cov_acc dtype mismatch: configured="
            f"{_state.configured_cov_storage_dtype!r}, accumulator="
            f"{cov_acc_storage_dtype_from_payload!r}. Delete the "
            f"checkpoint to regenerate."
        )
    cov_dtype = getattr(torch, _state.configured_cov_storage_dtype)
    _state.cov_acc.set_storage_dtype(cov_dtype)
    _state.n_layers = int(loaded.get("n_layers", _state.n_layers))
    # Preserve any prior ``setup``-pinned n_experts when the checkpoint
    # is missing the field — silently overwriting to 0 here would
    # break finalize_batch's sim-row shape (sweep finding round 2).
    _state.n_experts = int(loaded.get("n_experts", _state.n_experts))
    # H-2: ReamCostAccumulator.num_experts is needed by finalize_batch's
    # similarity-matrix sizing; if a checkpoint resumes BEFORE setup()
    # ever pinned it (or after a fresh process where the default is 0),
    # finalize_batch would build a malformed sim row. Mirror n_experts
    # onto the accumulator at load time so resume sees the right shape.
    _state.ream_acc.num_experts = _state.n_experts
    _state.top_k = int(loaded.get("top_k", _state.top_k))
    _state.model_hash = str(loaded.get("model_hash", _state.model_hash))
    _state.layer_idx_to_rank = dict(loaded.get("layer_idx_to_rank", {}))
    _state.rank_to_layer_idx = {
        r: li for li, r in _state.layer_idx_to_rank.items()
    }
    # Uniform full-replace pattern: checkpoint load happens after
    # ``setup`` zeroes the state, so every accumulator dict is the
    # checkpoint snapshot verbatim (sweep finding round 2 -- prior
    # mix of ``= dict(...)`` vs ``.update(...)`` was inconsistent).
    _state.ream_acc._sim_tensor = dict(loaded.get("ream_acc_sim_tensor", {}))
    _state.ream_acc._total_tokens_by_layer = dict(
        loaded.get("ream_acc_total_tokens", {})
    )
    _state.ream_acc._gate_gram = dict(
        loaded.get("ream_acc_gate_gram", {})
    )
    _state.ream_acc._neuron_act_sum = dict(
        loaded.get("ream_acc_neuron_act_sum", {})
    )
    _state.ream_acc._neuron_act_count = dict(
        loaded.get("ream_acc_neuron_act_count", {})
    )
    _state.cov_acc.covariance = dict(loaded.get("cov_acc_covariance", {}))
    _state.cov_acc.token_count = dict(loaded.get("cov_acc_token_count", {}))
    # Reconstruct the per-layer accumulators. The post-MEDIUM-fix payload
    # is a 4-tuple ``(buffer, seen, max_samples, generator_state)`` where
    # ``generator_state`` is the uint8 tensor returned by
    # ``torch.Generator.get_state()``. We restore it via ``set_state`` so
    # the resumed RNG stream is byte-identical to the non-resumed path
    # EVEN AFTER Phase C entry (profiling.py:57-79 determinism contract;
    # the live Phase C ``torch.rand``/``torch.randint`` calls at
    # profiling.py:144,155 share the per-layer generator).
    #
    # Backward compat: a pre-MEDIUM-fix 3-tuple checkpoint
    # ``(buffer, seen, max_samples)`` lacks the generator state. We log a
    # WARN and fall back to seed-re-init (seed = layer_idx) -- byte-
    # identical resume is preserved as long as the pre-checkpoint
    # accumulator never entered Phase C (i.e. the buffer never reached
    # ``max_samples``). Operators who must guarantee byte-identical
    # resume after Phase C entry must re-dump a fresh checkpoint with
    # the new schema.
    #
    # Legacy bare-tensor payload (pre-CRITICAL-1) is also tolerated -- it
    # gets promoted to an accumulator with buffer set + seen =
    # buffer.size(0); generator is seed-re-initialised (same caveat).
    raw_reservoir = loaded.get("layer_input_reservoir", {}) or {}
    new_reservoir: dict[int, _LayerInputAccumulator] = {}
    legacy_3tuple_count = 0
    for li, payload_entry in raw_reservoir.items():
        li_int = int(li)
        if isinstance(payload_entry, tuple):
            if len(payload_entry) == 4:
                buf, seen, max_samples, generator_state = payload_entry
                acc = _LayerInputAccumulator(
                    max_samples=int(max_samples), seed=li_int,
                )
                acc.buffer = (
                    None if buf is None
                    else buf.detach().cpu().to(torch.bfloat16).contiguous()
                )
                acc.seen = int(seen)
                # Restore the RNG state byte-for-byte.
                acc._generator.set_state(generator_state.clone())
            elif len(payload_entry) == 3:
                buf, seen, max_samples = payload_entry
                acc = _LayerInputAccumulator(
                    max_samples=int(max_samples), seed=li_int,
                )
                acc.buffer = (
                    None if buf is None
                    else buf.detach().cpu().to(torch.bfloat16).contiguous()
                )
                acc.seen = int(seen)
                legacy_3tuple_count += 1
            else:
                raise ValueError(
                    f"layer_input_reservoir[{li_int}] tuple has unsupported "
                    f"arity {len(payload_entry)} (expected 3 or 4). Delete "
                    f"the checkpoint to regenerate."
                )
        else:
            # Legacy: payload was a bare tensor snapshot.
            acc = _LayerInputAccumulator(
                max_samples=_LAYER_INPUT_MAX_SAMPLES, seed=li_int,
            )
            if payload_entry is not None and payload_entry.numel() > 0:
                acc.buffer = (
                    payload_entry.detach().cpu()
                    .to(torch.bfloat16).contiguous()
                )
                acc.seen = int(payload_entry.size(0))
        new_reservoir[li_int] = acc
    if legacy_3tuple_count > 0:
        log.warning(
            "stage2_profile_writer.load_checkpoint: %d layer reservoir(s) "
            "loaded from a pre-MEDIUM-fix 3-tuple payload without "
            "generator state; falling back to seed-re-init "
            "(seed=layer_idx). Byte-identical RNG resume is preserved only "
            "if the accumulator never entered Phase C pre-checkpoint "
            "(buffer.size(0) < max_samples). Re-dump a fresh checkpoint "
            "with the new schema to eliminate this warning.",
            legacy_3tuple_count,
        )
    _state.layer_input_reservoir = new_reservoir
    return _state.n_prompts_accumulated


# ---------------------------------------------------------------------------
# Test/debug helpers.
# ---------------------------------------------------------------------------
def _get_state() -> _WriterState:
    """Test-only accessor for the module singleton."""
    return _state


def _reset_state_for_tests() -> None:
    """Test helper: reset the singleton between cases."""
    global _state
    _state = _WriterState()
