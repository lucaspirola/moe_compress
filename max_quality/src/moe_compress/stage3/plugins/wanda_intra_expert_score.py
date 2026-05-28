"""Routing-weighted Wanda intra-expert importance score (MoE-Pruner Linear hook).

Paper
-----
"MoE-Pruner: Pruning Mixture-of-Experts Large Language Model using the
Hints from Its Router" — Xie et al., arXiv:2410.12013 (2024). Extends the
Wanda metric (Sun et al., arXiv:2306.11695, ICLR 2024) to MoE by weighting
each token's activation by the router's gate weight ``g_e`` that assigned
the token to expert ``e``:

    score(W_j) = |W_j| · sqrt( E_t [ (x_t · g_{e,t})² ] )

where the expectation is over tokens ``t`` in the calibration set that were
routed to expert ``e``, ``x_t`` is the per-channel input activation row
entering the expert's linear layer (gate_proj / up_proj input is the
hidden state pre-routing; down_proj input is the post-act intermediate),
``W_j`` is the j-th expert's weight matrix, and ``g_{e,t}`` is the scalar
routing weight the router assigned to ``(token t, expert e)``.

The score is per-(input-channel) — i.e. for ``W ∈ R^{d_out × d_in}`` the
score has the same shape (``|W_{out,in}|`` broadcast against
``sqrt(scalar_row[in])``), letting an external pruner (e.g. unstructured
magnitude prune over the score) decide which weights inside each expert
are least important to keep.

Use
---
Complementary to Stage 2 REAP (which prunes whole experts). This score
enables intra-expert sparsity: drop low-importance individual weights
within an expert that survives REAP. The Stage 3 ablation grid (A0..A11)
consumes the per-(layer, expert, matrix) score tensor map to compare
intra-expert pruning strategies; the score itself is published to
``ctx["stage3.wanda_intra_expert_score"]`` and optionally to a JSON-keyed
.pt sidecar at the F-H-7 namespaced calibration-sidecar path
``<jsonl.parent>/sidecars/<jsonl.stem>/wanda_intra_expert_score.pt``
(resolved via :func:`moe_compress.utils.cached_calibration_signals.sidecar_path`
with signal name ``"wanda_intra_expert_score"``). The previous layout
``artifacts_dir/_stage3_wanda_intra_expert_score.pt`` was out-of-band of
the calibration-sidecar pipeline and collided across multi-run sweeps
that share an ``artifacts_dir`` (audit W-2, 2026-05-29); the namespaced
layout coexists with W-1's ``wanda_scalar_row.pt`` under the same
``<jsonl>/sidecars/<stem>/`` directory under distinct signal names.

Upstream reference (clean-room)
-------------------------------
fusion_bench (tanganke/fusion_bench, MIT-licensed, © 2024 Anke Tang):

  * ``fusion_bench/method/moe_pruner/hooks/mixtral.py:10`` —
    ``MoEPrunerHookFnForMixtralLinear`` — the Linear-side accumulator that
    maintains ``scalar_row`` (running ``E[(x · g)²]``) per input channel
    and yields ``|W| · sqrt(scalar_row)`` on ``compute()``.
  * ``fusion_bench/method/moe_pruner/hooks/mixtral.py:53`` —
    ``MoEPrunerHookFnForMixtralGate`` — the Gate-side hook that recomputes
    softmax-then-topk on the router logits, assigns the per-token routing
    weight ``g_{e,t}`` to each Linear hook before the expert forward, and
    relies on a side-channel (``hook._routing_weights``) to plumb that
    scalar into the Linear accumulator.
  * ``fusion_bench/method/moe_pruner/moe_pruner.py:75`` — orchestrator
    ``MoEPruner.run`` — drives the per-layer dual hook install + forward.

Verified live at upstream HEAD on 2026-05-28 (commit fetched the same day
from a fresh clone of github.com/tanganke/fusion_bench). This module is a
clean-room reimplementation from the math in the docstring above; no
upstream code is copied verbatim. The line numbers cited remain stable so
a reviewer can re-derive the math from the reference.

Pattern H (clean-room) compliance: the algorithm is the math, expressed
through this codebase's own primitives (``instrument_experts`` context
dict's ``top_k_weights`` slot; the running-mean update via
``InputCovarianceAccumulator``-style ``nsamples`` tracking; the
``ExpertMatrixBank`` lookup for ``|W|``). The MoE-Pruner upstream uses a
``_routing_weights`` side channel between two forward hooks; our pipeline
already publishes the routing weight in the callback ``ctx`` dict — the
clean-room version is structurally cleaner and avoids the side channel.

Deviations from upstream
------------------------
**D-zero-extra-forward (RESOLVED via calibration sidecar — W-1)**.
The brief promised "zero extra forward cost" because the routing
weights are already collected during the covariance pass.
Resolution: ``scalar_row`` is captured as a calibration sidecar by
``vllm.calibration_wanda_scalar_row`` (gated on
``--capture-wanda-scalar-row``). On cache HIT, this plugin skips its
per-layer calibration sweep entirely and hydrates the accumulator
state directly from the sidecar via
``_WandaScalarRowAccumulator.from_payload`` — true zero extra
forward.

On cache MISS (e.g. a sidecar was not captured), the plugin falls
back to the per-layer calibration sweep at lines ~498-528 (mirrors
``_collect_covariances`` structure) — the original ~2× Stage 3 cal
cost is retained as the fallback path so production runs that omit
the capture flag still succeed.

See:
* ``tasks/PLAN_W1_WANDA_SCALAR_ROW_CAPTURE.md`` for the plan
* ``stage3/plugins/wanda_scalar_row_cache.py`` for the cache provider
* ``vllm/calibration_wanda_scalar_row.py`` for the writer

**Honest cost** (cache MISS only): ~2× Stage 3 calibration
wall-clock when ``--capture-wanda-scalar-row`` was NOT set during
calibration. Cache HIT: zero extra forward.

**Honest cost (cache HIT)**: the score map produced on the HIT
path is a STRICT SUBSET of the MISS path. The vLLM writer
(``vllm/calibration_wanda_scalar_row.py``) emits only
``"gate_proj"`` scalar_row entries -- the post-activation
intermediate seen by ``down_proj`` is not visible at the
``expert_in`` hook, so no ``"down_proj"`` keys are written. On
HIT, ``_compute_scores`` consequently emits ``{"gate_proj":
Tensor}`` per (layer, expert) and no ``"down_proj"`` entry; on
MISS, the per-layer calibration sweep emits both. The
``D-gate-up-share`` deviation aliases ``up_proj`` onto
``gate_proj`` at compute time on both paths, so the
``up_proj`` consumer is unaffected.

Downstream consumers (the D-Rank allocator, the future
rank-redistribution allocator) MUST handle a score map without
``"down_proj"`` keys gracefully on the HIT path -- either by
falling back to a covariance-only score for ``down_proj`` or by
skipping the wanda-weighting for that matrix slot. Today no
allocator consumes the wanda score map directly (it is
published for future use, see the plugin's "Placed between
collect_covariances and allocate_ranks" comment in the
orchestrator), so the HIT-vs-MISS asymmetry is latent; it
becomes load-bearing the moment an allocator wires it in.

A future patch may compose the fallback path with
``_collect_covariances`` (audit W-2) for the no-sidecar path; this is
independent of W-1 and tracked at
``tasks/todo_wanda_compose_with_collect_covariances.md``.

**D-fused-experts-architecture**. Upstream targets MixtralForCausalLM
(per-expert ``nn.Linear`` triples) and DeepseekV2ForCausalLM. Our base
model is Qwen3.6-35B-A3B which uses a fused ``Qwen3_5MoeExperts`` (a
single ``gate_up_proj`` + ``down_proj`` stacked tensor per layer, NOT a
per-expert Linear). The score is therefore computed against
``ExpertMatrixBank.get(expert_idx)`` — the bank's view of the per-expert
slice of the fused tensor — instead of an ``nn.Linear.weight`` attribute.
The math (``|W| · sqrt(scalar_row)``) is identical to upstream; only the
weight-lookup primitive differs.

**D-gate-up-share** (consistent with covariance_collection D6). The
``gate_proj`` and ``up_proj`` of each expert share the same pre-routing
hidden state input, so ``scalar_row`` is collected once under
``matrix_name="gate_proj"`` and the same scalar_row is reused for
``up_proj`` at compute time (the ``|W|`` factor differs between the two
matrices; ``sqrt(scalar_row)`` is identical). ``down_proj`` has its own
``scalar_row`` (input is the post-act intermediate, distinct from
gate_proj's input).

Pattern B (sidecar format_version)
----------------------------------
The optional JSON-keyed .pt sidecar carries a top-level
``format_version`` field (currently ``1``). Bump on incompatible payload
changes.

Pattern C (config validation at top of run)
-------------------------------------------
:meth:`_validate_config` runs as the FIRST statement of
:meth:`collect_wanda_scores` and rejects unknown keys with a ``ValueError``
listing the typo + the allowed key set, so an operator's typo (e.g.
``enabld`` for ``enabled``) fails loud instead of silently falling
through to default-OFF.

Opt-in gating
-------------
``is_enabled`` gates on ``stage3.wanda_intra_expert.enabled`` (default
``False``) so the plugin is a strict no-op for every row that does not
explicitly request it. When disabled, the registry's
:meth:`PluginRegistry.enabled` filter drops the plugin and the
orchestrator's :func:`walk_phases` call for ``collect_wanda_scores``
becomes a byte-identical no-op.

Circular-import note (mirror of ``stage3/plugins/covariance_collection.py``):
this module imports only from ``...utils.*``, ``...pipeline.*`` and
stdlib — never from ``stage3_svd`` or ``stage3.orchestrator``.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from ...pipeline.context import PipelineContext
from ...utils.activation_hooks import instrument_experts
from ...utils.atomic_io import atomic_torch_save, write_manifest_last
from ...utils.cached_calibration_signals import SCHEMA_VERSIONS
from ...utils.model_io import MATRIX_NAMES, build_banks

log = logging.getLogger(__name__)


# Sidecar payload schema version (Pattern B). Bump on incompatible changes.
# F-N-2 / NIT-1: schema version sourced from the central registry
# so any future v1 -> v2 bump invalidates writer + reader atomically.
_ARTIFACT_FORMAT_VERSION: int = SCHEMA_VERSIONS["wanda_intra_expert_score"]

# Recognised config keys (Pattern C). Any other key under
# ``stage3.wanda_intra_expert`` raises ``ValueError``.
#
# W-2 (2026-05-29): ``sidecar_filename`` is REMOVED — the sidecar path
# is now derived from the calibration JSONL stem via
# :func:`sidecar_path(..., "wanda_intra_expert_score")` so it lives
# under the namespaced ``<jsonl>/sidecars/<stem>/`` layout. Any
# pre-W-2 config carrying ``sidecar_filename`` will now raise via the
# Pattern C unknown-key validator (caller gets the typo list + the
# allowed-key set), failing loud instead of silently writing to the
# wrong place.
_ALLOWED_CFG_KEYS: frozenset[str] = frozenset(
    (
        "enabled",
        "write_sidecar",
        "score_dtype",
        "scalar_row_dtype",
    )
)

# Allowed values for the ``score_dtype`` / ``scalar_row_dtype`` knobs.
_ALLOWED_DTYPES: frozenset[str] = frozenset(
    ("float32", "float16", "bfloat16")
)
_DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _dtype_from_name(name: str) -> torch.dtype:
    return _DTYPE_MAP[name]


# ---------------------------------------------------------------------------
# Running-mean scalar_row accumulator (one entry per (layer, expert, matrix))
# ---------------------------------------------------------------------------


class _WandaScalarRowAccumulator:
    """Per-(layer, expert, matrix) running mean of ``(x · g_e)²`` per input channel.

    Mirrors the math of fusion_bench's ``MoEPrunerHookFnForMixtralLinear``
    (mixtral.py:10) but accumulates the unbiased running mean directly
    instead of staging through a single ``nn.Linear`` per call. The
    update rule is the same Welford-flavored running mean upstream uses:

        scalar_row *= n_old / (n_old + n_batch)
        n_total = n_old + n_batch
        scalar_row += ||x · g||₂² (over rows) / n_total

    where ``n_batch`` is the number of token rows arriving in this batch
    (= ``x.shape[0]``) and ``||·||₂²`` is the per-channel squared L2 norm
    along the row axis (= sum of squares per channel).

    Storage is ``fp32`` by default (the entire scoring formula is
    numerically dominated by the sum-of-squares which accumulates over
    thousands of tokens — fp16 would underflow on a long enough run);
    the per-call accumulation runs in fp32 on the layer's device and the
    final scalar_row is moved to CPU + cast to ``scalar_row_dtype`` for
    sidecar storage.
    """

    def __init__(self, scalar_row_dtype: torch.dtype = torch.float32) -> None:
        # (layer_idx, expert_idx, matrix_name) -> fp32 GPU tensor [d_in]
        self._gpu: dict[tuple[int, int, str], torch.Tensor] = {}
        # Sample count per key (token-row count, accumulated across batches)
        self._nsamples: dict[tuple[int, int, str], int] = {}
        # Finalized CPU tensors, cast to scalar_row_dtype
        self._cpu: dict[tuple[int, int, str], torch.Tensor] = {}
        self._dtype = scalar_row_dtype
        # When True, the accumulator was hydrated from a calibration
        # sidecar (see ``from_payload``); update() is forbidden.
        self._frozen: bool = False

    def update(
        self,
        layer_idx: int,
        expert_idx: int,
        matrix_name: str,
        x_rows: torch.Tensor,       # [T, d_in] — input rows for one expert
        g_weights: torch.Tensor,    # [T]       — per-row scalar routing weight
    ) -> None:
        """Accumulate one batch of token rows for ``(layer, expert, matrix)``.

        ``x_rows`` is the activation matrix entering the expert's linear
        (rows = tokens routed to this expert in this batch, cols = input
        channels). ``g_weights`` is the per-row scalar routing weight
        the router assigned to ``(token, expert)``. The squared L2 norm
        (per channel) of the weighted rows is added into the running
        mean.
        """
        if self._frozen:
            raise RuntimeError(
                "_WandaScalarRowAccumulator.update: accumulator was "
                "hydrated from a calibration sidecar (via "
                "from_payload) and is frozen — calling .update() now "
                "would double-count the calibration data. "
                "Construct a fresh accumulator if you need to "
                "accumulate live token rows."
            )
        if x_rows.numel() == 0:
            return
        if x_rows.dim() != 2:
            raise ValueError(
                f"_WandaScalarRowAccumulator.update: x_rows must be 2D "
                f"[T, d_in], got shape {tuple(x_rows.shape)}"
            )
        if g_weights.shape != (x_rows.shape[0],):
            raise ValueError(
                f"_WandaScalarRowAccumulator.update: g_weights shape "
                f"{tuple(g_weights.shape)} does not match x_rows leading "
                f"dim {x_rows.shape[0]}"
            )
        key = (layer_idx, expert_idx, matrix_name)
        n_batch = int(x_rows.shape[0])
        n_old = self._nsamples.get(key, 0)
        n_total = n_old + n_batch
        # Promote to fp32 on the input device (matches upstream's
        # inp.type(torch.float32) cast before the squared norm).
        x_fp32 = x_rows.detach().to(torch.float32)
        g_fp32 = g_weights.detach().to(torch.float32).reshape(-1, 1)
        # Per-channel sum of squares of (x_c · g)
        # ``(x_fp32 * g_fp32)`` is [T, d_in]; the squared L2 norm per
        # channel is the per-column sum-of-squares = ``(.).pow(2).sum(0)``.
        # Upstream's ``torch.norm(inp * routing_weights, p=2, dim=1)**2``
        # operates on the transposed [d_in, T] view; mathematically
        # identical, expressed without the transpose here.
        sq_per_channel = (x_fp32 * g_fp32).pow(2).sum(dim=0)
        cur = self._gpu.get(key)
        if cur is None:
            # First batch for this key: init with the per-channel mean.
            self._gpu[key] = sq_per_channel / float(n_total)
        else:
            # Running mean update — mirror of upstream's two-step:
            #   scalar_row *= n_old / n_total
            #   scalar_row += sq_per_channel / n_total
            cur.mul_(float(n_old) / float(n_total))
            cur.add_(sq_per_channel / float(n_total))
        self._nsamples[key] = n_total

    def finalize_layer(self, layer_idx: int) -> None:
        """Move all entries for ``layer_idx`` from GPU fp32 to CPU
        ``scalar_row_dtype`` and free the GPU tensors. Call once per
        layer, after the calibration batches for that layer have been
        consumed.
        """
        keys = [k for k in self._gpu if k[0] == layer_idx]
        for k in keys:
            self._cpu[k] = self._gpu[k].detach().to(
                device="cpu", dtype=self._dtype,
            )
            del self._gpu[k]

    def get_scalar_row(
        self, layer_idx: int, expert_idx: int, matrix_name: str,
    ) -> torch.Tensor | None:
        return self._cpu.get((layer_idx, expert_idx, matrix_name))

    @property
    def cpu_entries(self) -> dict[tuple[int, int, str], torch.Tensor]:
        return self._cpu

    @classmethod
    def from_payload(
        cls,
        payload,
        *,
        scalar_row_dtype: torch.dtype,
        expected_d_in_by_matrix: dict[str, int] | None = None,
    ) -> "_WandaScalarRowAccumulator":
        """Hydrate a frozen accumulator from a calibration sidecar.

        The returned accumulator is finalize-ready: its update path is
        NOT meant to be called again (the payload represents finalized
        running means from the calibration phase). Calling ``.update()``
        after ``.from_payload()`` raises ``RuntimeError`` (the
        ``_frozen`` guard set here defends against silent
        double-counting).

        ``payload`` is a
        :class:`moe_compress.utils.cached_calibration_signals.WandaScalarRowPayload`
        (the dict shape + dataclass shape both supported; the dataclass
        is canonical).

        Optional ``expected_d_in_by_matrix`` (e.g.
        ``{"gate_proj": 2048, "down_proj": 768}``) — when provided, each
        cached ``sigma_x_g_squared`` tensor is asserted to match
        ``Tensor[d_in]`` for its matrix's expected ``d_in``. Catches
        geometric drift across runs (polish item #2 from the planner
        review).

        Pattern mirrors ``dict.fromkeys`` / ``torch.Tensor.from_numpy``
        (alternate constructor that internalises the encapsulation —
        replaces the v1 plan's direct ``_cpu`` / ``_nsamples`` pokes
        from outside the class; L-2 plan-reviewer-v1 fold).
        """
        sigma_dict = payload.sigma_x_g_squared
        token_counts = payload.token_counts
        self = cls(scalar_row_dtype=scalar_row_dtype)
        for key, sigma in sigma_dict.items():
            li, e, name = key
            if not isinstance(sigma, torch.Tensor):
                raise ValueError(
                    f"_WandaScalarRowAccumulator.from_payload: "
                    f"sigma_x_g_squared[{key}] is "
                    f"{type(sigma).__name__}, expected torch.Tensor."
                )
            if sigma.dim() != 1:
                raise ValueError(
                    f"_WandaScalarRowAccumulator.from_payload: "
                    f"sigma_x_g_squared[{key}] has dim {sigma.dim()}, "
                    f"expected 1D Tensor[d_in]."
                )
            if expected_d_in_by_matrix is not None:
                exp_d_in = expected_d_in_by_matrix.get(name)
                if exp_d_in is not None and int(sigma.shape[0]) != int(exp_d_in):
                    raise ValueError(
                        f"_WandaScalarRowAccumulator.from_payload: "
                        f"sigma_x_g_squared[{key}] has d_in="
                        f"{int(sigma.shape[0])}, expected {int(exp_d_in)} "
                        f"for matrix '{name}'. Geometric drift across "
                        f"runs — delete the sidecar to regenerate."
                    )
            self._cpu[(int(li), int(e), str(name))] = sigma.to(scalar_row_dtype)
            count = token_counts.get(key, 0)
            self._nsamples[(int(li), int(e), str(name))] = int(count)
        self._frozen = True
        return self


# ---------------------------------------------------------------------------
# Score computation: |W| · sqrt(scalar_row) per (layer, expert, matrix)
# ---------------------------------------------------------------------------


def _compute_scores(
    moe_layers,
    scalar_row_acc: _WandaScalarRowAccumulator,
    *,
    score_dtype: torch.dtype,
) -> dict[int, dict[int, dict[str, torch.Tensor]]]:
    """Materialize the per-weight Wanda score map.

    Returns a nested dict ``{layer_idx: {expert_idx: {matrix_name: Tensor}}}``
    where each tensor has the same shape as the corresponding expert's
    weight matrix ``W ∈ R^{d_out × d_in}``: ``|W| * sqrt(scalar_row)`` with
    ``scalar_row`` broadcast against the input-channel axis. CPU,
    ``score_dtype``.

    For matrices without their own scalar_row (``up_proj`` reuses
    ``gate_proj``'s scalar_row by D-gate-up-share — they share the same
    pre-routing hidden state input), the gate_proj scalar_row is reused.
    """
    out: dict[int, dict[int, dict[str, torch.Tensor]]] = {}
    for ref in moe_layers:
        banks = build_banks(ref)
        per_layer: dict[int, dict[str, torch.Tensor]] = {}
        for e in range(ref.num_routed_experts):
            per_expert: dict[str, torch.Tensor] = {}
            for name in MATRIX_NAMES:
                # gate/up share scalar_row (auto-cov-style alias).
                lookup_name = "gate_proj" if name == "up_proj" else name
                scalar_row = scalar_row_acc.get_scalar_row(
                    ref.layer_idx, e, lookup_name,
                )
                if scalar_row is None:
                    # No tokens routed to this (layer, expert) for this
                    # matrix during calibration — score is undefined; skip.
                    continue
                W = banks[name].get(e).detach().to(
                    device="cpu", dtype=torch.float32,
                )
                # |W| · sqrt(scalar_row) — scalar_row [d_in] broadcasts
                # against |W| [d_out, d_in].
                score = W.abs() * scalar_row.to(torch.float32).clamp_min(0.0).sqrt().reshape(1, -1)
                per_expert[name] = score.to(score_dtype)
            if per_expert:
                per_layer[e] = per_expert
        if per_layer:
            out[ref.layer_idx] = per_layer
    return out


# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------


class WandaIntraExpertScorePlugin:
    """Stage 3 routing-weighted Wanda intra-expert score plugin.

    See module docstring for the math, upstream citation, deviation log
    (D-zero-extra-forward, D-fused-experts-architecture, D-gate-up-share)
    and pattern compliance (B / C / H).

    Default OFF (``stage3.wanda_intra_expert.enabled`` defaults to False)
    so this plugin is a strict no-op for every row that does not
    explicitly opt in. When disabled, the orchestrator's
    ``walk_phases(("collect_wanda_scores",), plugins, run_ctx)`` call is
    a byte-identical no-op (registry filter drops the plugin via
    :meth:`is_enabled`).
    """

    name: str = "wanda_intra_expert_score"
    paper: str = (
        "MoE-Pruner (arXiv:2410.12013) Linear-side accumulator extended from "
        "Wanda (arXiv:2306.11695). Score: |W_j| * sqrt(E[(x · g_e)^2]). "
        "Clean-room reimplementation from "
        "fusion_bench/method/moe_pruner/hooks/mixtral.py:10 "
        "(MoEPrunerHookFnForMixtralLinear) and "
        "fusion_bench/method/moe_pruner/hooks/mixtral.py:53 "
        "(MoEPrunerHookFnForMixtralGate); upstream license MIT, "
        "© 2024 Anke Tang. Verified live at upstream HEAD 2026-05-28. "
        "Deviations: D-zero-extra-forward (RESOLVED via W-1 calibration "
        "sidecar -- on cache HIT the per-layer calibration sweep is "
        "skipped entirely; cache MISS retains the own-calibration-pass "
        "fallback), D-fused-experts-architecture "
        "(weight lookup via ExpertMatrixBank.get instead of nn.Linear.weight "
        "for Qwen3_5MoeExperts), D-gate-up-share (gate/up share scalar_row "
        "per covariance_collection D6). Pattern H (clean-room), B "
        "(sidecar format_version), C (config-validation-at-top)."
    )
    config_key: str = "stage3.wanda_intra_expert.enabled"
    reads: tuple[str, ...] = (
        "model",
        "moe_layers",
        "batches",
        "device",
        "config",
        # W-2 (2026-05-29): the sidecar is now namespaced under the
        # calibration JSONL stem (replaces the prior ``artifacts_dir``
        # dependency). The orchestrator sets ``calibration_jsonl_path``
        # (absolute ``Path``) on ``run_ctx`` at the same site as
        # ``artifacts_dir`` (orchestrator line 210). When absent the
        # plugin still computes the in-memory score map but skips the
        # sidecar write with a WARNING (operators see the actionable
        # cause rather than a silent NO-OP).
        "calibration_jsonl_path",
        # Optional cache slot populated by
        # ``Stage3WandaScalarRowCacheProvider`` (W-1). On HIT, the
        # plugin short-circuits its per-layer calibration sweep — see
        # tasks/PLAN_W1_WANDA_SCALAR_ROW_CAPTURE.md §7.1.
        "stage3.wanda_scalar_row",
    )
    writes: tuple[str, ...] = (
        "stage3.wanda_intra_expert_score",
        "stage3.wanda_intra_expert_metadata",
    )
    provides: tuple[str, ...] = ()

    def is_enabled(self, config: dict) -> bool:
        """Gate on ``config["stage3"]["wanda_intra_expert"]["enabled"]``.

        Default OFF — opt-in only. The lookup is defensive: any missing
        intermediate dict returns False so plugin enablement never
        raises on a partial config.
        """
        try:
            return bool(config["stage3"]["wanda_intra_expert"]["enabled"])
        except (KeyError, TypeError):
            return False

    def contribute_artifact(self, ctx: Any) -> dict:
        """Optional pipeline-level artifact contribution.

        When the score has been written to context (the plugin ran),
        publish a summary descriptor (NOT the full score tensors — those
        are gigabytes; the sidecar .pt is the canonical store). The dict
        carries ``format_version`` (Pattern B) so a future artifact
        consumer can detect schema mismatches.
        """
        if not ctx.has("stage3.wanda_intra_expert_score"):
            return {}
        score_map = ctx.get("stage3.wanda_intra_expert_score")
        n_layers = len(score_map)
        n_keys = sum(
            len(per_expert)
            for per_layer in score_map.values()
            for per_expert in per_layer.values()
        )
        return {
            "format_version": _ARTIFACT_FORMAT_VERSION,
            "wanda_intra_expert_score_summary": {
                "n_layers": n_layers,
                "n_score_tensors": n_keys,
            },
        }

    # ------------------------------------------------------------------
    # Phase hook: collect_wanda_scores
    # ------------------------------------------------------------------

    def collect_wanda_scores(self, ctx: PipelineContext) -> None:
        """Phase hook — collect ``scalar_row`` per (layer, expert, matrix),
        compute the per-weight ``|W| · sqrt(scalar_row)`` score map, and
        publish to ``ctx["stage3.wanda_intra_expert_score"]`` (+ optional
        sidecar).

        Runs an independent calibration pass through the student model
        using ``instrument_experts`` — same primitive
        ``covariance_collection`` uses; the per-callback ``ctx`` dict
        carries the ``top_k_weights`` slot this plugin needs, so no
        router-side hook is required (clean-room divergence from
        upstream's two-hook side-channel design — see module D-zero-
        extra-forward).
        """
        # Pattern C: validate config FIRST, before any calibration work.
        config = ctx.get("config")
        wanda_cfg_raw: dict = (
            config.get("stage3", {}).get("wanda_intra_expert", {})
        )
        cfg = self._validate_config(wanda_cfg_raw)
        if not cfg["enabled"]:
            # is_enabled should have already gated this, but a defensive
            # check keeps the hook idempotent when called directly.
            log.debug(
                "wanda_intra_expert_score: enabled=False, skipping "
                "(is_enabled gate should have already filtered)"
            )
            return

        model = ctx.get("model")
        moe_layers = ctx.get("moe_layers")
        batches = ctx.get("batches")
        device = ctx.get("device") if ctx.has("device") else None
        # W-2: sidecar path derives from the calibration JSONL stem via
        # the F-H-7 ``sidecar_path`` helper. The orchestrator sets this
        # slot at line 210; if absent, the in-memory score map is still
        # published but the sidecar write is skipped with a WARNING.
        calib_jsonl_path: Path | None = (
            Path(ctx.get("calibration_jsonl_path"))
            if ctx.has("calibration_jsonl_path")
            else None
        )

        score_dtype = _dtype_from_name(cfg["score_dtype"])
        scalar_row_dtype = _dtype_from_name(cfg["scalar_row_dtype"])

        log.info(
            "wanda_intra_expert_score: starting (%d MoE layers, "
            "score_dtype=%s, scalar_row_dtype=%s, write_sidecar=%s)",
            len(moe_layers), cfg["score_dtype"], cfg["scalar_row_dtype"],
            cfg["write_sidecar"],
        )

        # --- CACHE HIT: short-circuit per W-1 -----------------------------
        # Mirrors Stage3InputCovCacheProvider's contract — the cache
        # provider has already validated schema + manifest by the time it
        # populates ``ctx["stage3.wanda_scalar_row"]``. On HIT, hydrate
        # the accumulator via ``from_payload`` and skip the per-layer
        # sweep entirely (the writer in
        # ``vllm/calibration_wanda_scalar_row.py`` captured the same
        # running mean during the calibration forward — see D-zero-
        # extra-forward in this module's docstring).
        if ctx.has("stage3.wanda_scalar_row"):
            payload = ctx.get("stage3.wanda_scalar_row")
            log.info(
                "wanda_intra_expert_score: cache HIT (%d entries) — "
                "skipping per-layer calibration sweep",
                len(payload.sigma_x_g_squared),
            )
            # Build the expected d_in map from the model banks so the
            # classmethod can range-check the cached tensor shapes
            # against the live config (polish item #2 — catches
            # geometric drift across runs).
            expected_d_in: dict[str, int] = {}
            if moe_layers:
                _banks = build_banks(moe_layers[0])
                for _name in ("gate_proj", "down_proj"):
                    if _name in _banks:
                        try:
                            _W = _banks[_name].get(0)
                            expected_d_in[_name] = int(_W.shape[1])
                        except (KeyError, AttributeError, IndexError):
                            # If the bank can't materialise expert 0,
                            # skip the shape assertion — the consumer's
                            # own _compute_scores will surface a real
                            # error later if the cache is incompatible.
                            pass
            acc = _WandaScalarRowAccumulator.from_payload(
                payload,
                scalar_row_dtype=scalar_row_dtype,
                expected_d_in_by_matrix=expected_d_in or None,
            )
        else:
            acc = _WandaScalarRowAccumulator(scalar_row_dtype=scalar_row_dtype)

            # --- Per-layer calibration pass -------------------------------
            # Mirror of _collect_covariances structure: one calibration
            # sweep PER MoE layer (sequential), to bound peak hook state.
            # The cache-HIT short-circuit above resolves D-zero-extra-
            # forward; this fallback is the cache-MISS path retained for
            # operators who did not pass ``--capture-wanda-scalar-row``.
            for k, ref in enumerate(moe_layers):
                log.info(
                    "  wanda layer %d/%d (idx=%d) — calibration pass",
                    k + 1, len(moe_layers), ref.layer_idx,
                )

                def _input_cb(li, e, tensor, cb_ctx, _acc=acc):
                    # gate_proj input — pre-routing hidden state (also
                    # serves up_proj by D-gate-up-share).
                    _acc.update(
                        li, e, "gate_proj",
                        tensor, cb_ctx["top_k_weights"],
                    )

                def _intermediate_cb(li, e, tensor, cb_ctx, _acc=acc):
                    # down_proj input — post-act intermediate.
                    _acc.update(
                        li, e, "down_proj",
                        tensor, cb_ctx["top_k_weights"],
                    )

                with instrument_experts(
                    ref,
                    {"input": _input_cb, "intermediate": _intermediate_cb},
                ):
                    for batch in batches:
                        if device is not None:
                            batch = batch.to(device)
                        with torch.no_grad():
                            model(input_ids=batch)

                acc.finalize_layer(ref.layer_idx)

        # --- Compute |W| · sqrt(scalar_row) -------------------------------
        log.info(
            "wanda_intra_expert_score: computing |W| · sqrt(scalar_row) "
            "for %d layers", len(moe_layers),
        )
        score_map = _compute_scores(
            moe_layers, acc, score_dtype=score_dtype,
        )

        metadata = {
            "n_layers": len(score_map),
            "n_score_tensors": sum(
                len(per_expert)
                for per_layer in score_map.values()
                for per_expert in per_layer.values()
            ),
            "score_dtype": cfg["score_dtype"],
            "scalar_row_dtype": cfg["scalar_row_dtype"],
        }

        ctx.set("stage3.wanda_intra_expert_score", score_map)
        ctx.set("stage3.wanda_intra_expert_metadata", metadata)

        # --- Optional sidecar (Pattern B + atomic + manifest-LAST) --------
        # W-2: write to the F-H-7 namespaced layout
        # ``<jsonl.parent>/sidecars/<jsonl.stem>/wanda_intra_expert_score.pt``
        # via the shared :func:`sidecar_path` helper, so two Stage 3 runs
        # that share an ``artifacts_dir`` (e.g. A0..A11 ablation sweep rows
        # with different ``score_dtype``) no longer collide on the sidecar.
        if cfg["write_sidecar"]:
            if calib_jsonl_path is None:
                log.warning(
                    "wanda_intra_expert_score: write_sidecar=True but "
                    "ctx['calibration_jsonl_path'] is not set — skipping "
                    "sidecar write. The orchestrator promotes this slot at "
                    "stage3/orchestrator.py line 210; an unset value usually "
                    "means the plugin was invoked outside the Stage 3 "
                    "orchestrator (e.g. a unit test) or that wiring is "
                    "missing. The in-memory score map at "
                    "ctx['stage3.wanda_intra_expert_score'] is unaffected.",
                )
            else:
                from ...utils.cached_calibration_signals import (
                    sidecar_path as _sidecar_path,
                )
                # Local name ``sidecar_pt`` avoids shadowing the imported
                # helper inside this function body.
                sidecar_pt = _sidecar_path(
                    calib_jsonl_path, "wanda_intra_expert_score",
                )
                manifest_path = sidecar_pt.with_suffix(".pt.MANIFEST.json")
                sidecar_pt.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    "format_version": _ARTIFACT_FORMAT_VERSION,
                    "wanda_intra_expert_score": score_map,
                    "metadata": metadata,
                }
                try:
                    manifest_path.unlink(missing_ok=True)
                except OSError:
                    pass
                atomic_torch_save(sidecar_pt, payload)
                write_manifest_last(
                    sidecar_pt,
                    manifest_path,
                    schema_version=_ARTIFACT_FORMAT_VERSION,
                    extra_meta={
                        "artifact": "wanda_intra_expert_score",
                        **metadata,
                    },
                    compute_sha256=False,
                )
                log.info(
                    "wanda_intra_expert_score: wrote sidecar %s (manifest %s)",
                    sidecar_pt, manifest_path,
                )

    # ------------------------------------------------------------------
    # Config validation (Pattern C)
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_config(wanda_cfg: dict) -> dict:
        """Reject unknown keys + range-check the values. Pattern C.

        Returns a typed dict of the validated config with defaults
        applied. Unknown keys raise ``ValueError`` listing the typo so
        operators see ``enabld`` mis-keys instead of having them silently
        fall through to default-OFF.
        """
        unknown = set(wanda_cfg.keys()) - _ALLOWED_CFG_KEYS
        if unknown:
            raise ValueError(
                f"wanda_intra_expert: unknown config keys "
                f"{sorted(unknown)!r} under stage3.wanda_intra_expert. "
                f"Allowed keys: {sorted(_ALLOWED_CFG_KEYS)!r}."
            )
        cfg = {
            "enabled": bool(wanda_cfg.get("enabled", False)),
            "write_sidecar": bool(wanda_cfg.get("write_sidecar", True)),
            "score_dtype": str(wanda_cfg.get("score_dtype", "float32")),
            "scalar_row_dtype": str(
                wanda_cfg.get("scalar_row_dtype", "float32")
            ),
        }
        if cfg["score_dtype"] not in _ALLOWED_DTYPES:
            raise ValueError(
                f"wanda_intra_expert: score_dtype must be one of "
                f"{sorted(_ALLOWED_DTYPES)!r}, got {cfg['score_dtype']!r}."
            )
        if cfg["scalar_row_dtype"] not in _ALLOWED_DTYPES:
            raise ValueError(
                f"wanda_intra_expert: scalar_row_dtype must be one of "
                f"{sorted(_ALLOWED_DTYPES)!r}, got {cfg['scalar_row_dtype']!r}."
            )
        return cfg


__all__ = [
    "WandaIntraExpertScorePlugin",
    "_WandaScalarRowAccumulator",
    "_compute_scores",
    "_ARTIFACT_FORMAT_VERSION",
]
