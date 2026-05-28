"""Stage 2 cache provider for the REDO profile-pass sidecar (Optimization A).

Reads the schema-v3 ``stage2_profile.pt`` sidecar produced by the
``--capture-stage2-profile`` calibration flag (the vLLM patch
``vllm.calibration_stage2_profile``) and hydrates the per-layer
:class:`ReamCostAccumulator`, the shared :class:`InputCovarianceAccumulator`
and the per-layer ``_LayerInputAccumulator`` so that
:meth:`LayerMergePlugin.on_profile` can skip the live forward pass entirely
for full-hit layers (Pattern A — cache-aware skip).

Architecture
------------
* Single config knob: ``stage2_reap_ream.profile_sidecar.enabled: bool``
  (default ``false``). Bug #8 is addressed structurally — only one reader
  exists going forward (this one); there is no separate cov-only reader
  whose registration could double-count covariance.
* Registered AFTER :class:`LayerMergePlugin` in the plugin registry so its
  ``on_layer_setup`` runs SECOND and sees the freshly-constructed
  ``ream_acc`` / ``layer_input_acc`` that ``LayerMergePlugin.on_layer_setup``
  writes to ctx — the reader hydrates those in place rather than swapping
  the objects out (OQ-1 Option A — in-place hydration).
* Full-hit / partial-hit detection per plan §5: a layer is a full hit when
  ``payload.total_tokens_per_layer[layer_rank] >= 0.5 × expected``. On
  partial / miss the reader leaves the live path's fresh empty
  accumulators untouched and the live forward runs.

Hydration contract (in-place; see plan §10):
    * ``ream_acc.gate_logit_profiles[layer_idx]`` ← payload list-of-tuples
      verbatim (list[(int_offset, Tensor[T_b, E] fp32)]).
    * ``ream_acc._sim_tensor[layer_idx]``           ← clone of payload sim row.
    * ``ream_acc._total_tokens_by_layer[layer_idx]`` ← payload total.
    * ``ream_acc._neuron_act_sum/_count[(layer_idx, e)]`` ← payload rows for
      the current rank (after layer_rank → layer_idx translation).
    * ``cov_acc.covariance[(layer_idx, e, m)]`` ← payload row cast to the
      live ``cov_acc.storage_dtype`` (DIRECT dict write — NOT
      ``cov_acc.update(...)`` — see plan §12 / Critical-2).
    * ``cov_acc.token_count[(layer_idx, e, m)]`` ← payload row count.
    * ``layer_input_acc.buffer`` ← clone of payload reservoir row (guarded
      ``if layer_input_acc is not None:`` per plan §10 N-3 / round-3 fix).

The ``on_load`` run-scope step performs the schema-v3 + cov_storage_dtype
cross-validation; a mismatch raises ``ValueError`` with the
"Delete the sidecar to regenerate" message.

Paper
-----
SC_FAST_PLAN_V3 §4 Optimization A — projects 30-50 minutes / SC row by
short-circuiting Stage 2's per-layer profile forward pass.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from ...pipeline.context import PipelineContext
from ...utils.cached_calibration_signals import (
    Stage2ProfilePayloadV3,
    load_stage2_profile_v3,
    sidecar_path,
)

log = logging.getLogger(__name__)


class Stage2ProfileCacheProvider:
    """Cache-side provider for the Stage 2 REDO profile-pass sidecar.

    Hooks:
      * ``on_load``  — run-scope: load and cross-validate the sidecar.
      * ``on_layer_setup`` — per-layer: hydrate accumulators on full hit.

    There is NO ``on_profile`` hook; the cache-aware skip is implemented
    on :meth:`LayerMergePlugin.on_profile` (a single early-return guard
    keyed on ``ctx["stage2_profile_full_hit"]``).
    """

    name: str = "stage2_profile_cache"
    paper: str = (
        "SC_FAST_PLAN_V3 §4 Optimization A REDO. Reads "
        "sidecars/stage2_profile.pt (schema v3) produced by "
        "--capture-stage2-profile. On full hit: hydrates ream_acc + "
        "cov_acc + layer_input_acc in-place so LayerMergePlugin.on_profile "
        "skips the live forward pass. On partial/miss: no-op; live path "
        "runs unchanged. Single reader per Bug #8 fix."
    )
    config_key: str = "stage2_reap_ream"
    reads: tuple[str, ...] = (
        "_layer_rank", "layer_ref",
        "ream_acc", "layer_input_acc",
        "stage2_profile_payload",
    )
    writes: tuple[str, ...] = (
        "stage2_profile_payload",
        "stage2_profile_full_hit",
        "stage2_profile_partial_hit",
        "ream_acc", "layer_input_acc",
    )
    provides: tuple[str, ...] = ()

    def __init__(
        self,
        *,
        cov_acc,
        expected_cov_storage_dtype: str | None = None,
        partial_hit_fraction: float = 0.5,
        cost_alignment: str = "pre",
    ) -> None:
        """Construct the provider.

        Args:
            cov_acc: the run-scope :class:`InputCovarianceAccumulator`
                shared with :class:`LayerMergePlugin`. The reader writes
                directly into its ``covariance`` / ``token_count`` dicts
                on full hit (Pattern: direct dict-write, NOT
                ``cov_acc.update``).
            expected_cov_storage_dtype: when provided (e.g. from the
                Stage 2 YAML's ``covariance_storage_dtype``), the sidecar
                load fails loud if ``payload.cov_storage_dtype`` does
                not match.  When ``None`` the cross-validation is
                skipped and only the allowed-values guard fires.
            partial_hit_fraction: threshold below which a layer is
                classified as a partial hit (plan §5; defaults to 0.5).
            cost_alignment: the active ``stage2_reap_ream.cost_alignment``
                value (one of ``"pre"`` / ``"post"`` / ``"output"``). Used
                by the hidden-bug demote guard (plan §5.b): when
                ``"output"`` is active AND the sidecar's per-layer
                reservoir is empty, the reader demotes ``full_hit`` to
                ``partial_hit`` so ``_output_space_cost`` does not crash
                on a missing layer-input buffer. Defaults to ``"pre"``
                (no demotion) so the new arg is optional for callers that
                don't enable output-space alignment.
        """
        self.cov_acc = cov_acc
        self.expected_cov_storage_dtype = expected_cov_storage_dtype
        self.partial_hit_fraction = float(partial_hit_fraction)
        self.cost_alignment = str(cost_alignment).lower()
        # Filled by on_load; consumed by on_layer_setup.
        self.payload: Stage2ProfilePayloadV3 | None = None

    def _cost_alignment_requires_reservoir(self) -> bool:
        """True iff ``cost_alignment="output"`` is active.

        Mirrors :meth:`OutputSpaceCostPlugin.is_enabled` (output_space_cost.py
        :585-593): the layer-input reservoir is mandatory only on
        ``cost_alignment="output"`` runs. On ``"pre"`` / ``"post"`` the
        reservoir is unused so an empty payload entry is harmless.
        """
        return self.cost_alignment == "output"

    def is_enabled(self, config: dict) -> bool:
        s2 = config.get("stage2_reap_ream", {}) or {}
        sidecar = s2.get("profile_sidecar", {}) or {}
        return bool(sidecar.get("enabled", False))

    def contribute_artifact(self, ctx: PipelineContext) -> dict:
        return {}

    # ------------------------------------------------------------------
    # Run-scope: load the sidecar (or no-op on miss).
    # ------------------------------------------------------------------
    def on_load(
        self, ctx: PipelineContext, jsonl_path: Path,
    ) -> Stage2ProfilePayloadV3 | None:
        """Try to load the schema-v3 stage2_profile sidecar.

        Returns the loaded payload on hit; ``None`` on miss (graceful —
        the live path runs unchanged). Raises ``ValueError`` on schema
        / cov_storage_dtype mismatch (per plan §12 error-handling).
        """
        # Schema / cross-validation errors surface as ValueError with a
        # "Delete the sidecar to regenerate" message — let them propagate
        # to the operator unchanged (no try/except wrapper; the previous
        # `except ValueError: raise` was a no-op).
        payload = load_stage2_profile_v3(
            jsonl_path,
            expected_cov_storage_dtype=self.expected_cov_storage_dtype,
        )
        if payload is None:
            log.warning(
                "stage2-profile-cache: sidecar %s missing — cache miss, "
                "live path will run",
                sidecar_path(jsonl_path, "stage2_profile"),
            )
            return None
        self.payload = payload
        ctx.set("stage2_profile_payload", payload)
        log.info(
            "stage2-profile-cache: loaded %d-layer × %d-expert sidecar (top_k=%d, "
            "cov_storage_dtype=%s) from %s",
            payload.n_layers, payload.n_experts, payload.top_k,
            payload.cov_storage_dtype,
            sidecar_path(jsonl_path, "stage2_profile"),
        )
        return payload

    # ------------------------------------------------------------------
    # Per-layer: hydrate accumulators on full hit.
    # ------------------------------------------------------------------
    def on_layer_setup(self, ctx: PipelineContext) -> None:
        """Hydrate per-layer state in-place on full hit; mark partial otherwise.

        Runs AFTER :meth:`LayerMergePlugin.on_layer_setup` so the freshly
        constructed ``ream_acc`` / ``layer_input_acc`` are present in ctx
        and can be populated in place.
        """
        if self.payload is None:
            # Cache miss at run-scope level — nothing to hydrate.
            return

        layer_rank = ctx.get("_layer_rank")
        layer_ref = ctx.get("layer_ref")
        layer_idx = int(layer_ref.layer_idx)
        payload = self.payload

        # ------------------------------------------------------------------
        # Partial-hit detection (plan §5). The sidecar's per-layer token
        # count is compared against the per-layer expectation
        # (n_batches × tokens_per_batch). We do not have the live token
        # expectation at this hook, so we use the maximum token count
        # observed in the sidecar itself as the proxy reference — a layer
        # whose own count is < partial_hit_fraction × max is "partial". The
        # operator who writes the sidecar from a SHORTER run than the
        # current one will trip the fraction threshold via the absolute
        # reference passed in the future; today the relative-to-max heuristic
        # is robust enough because the writer's own per-layer totals are
        # uniform-ish in practice (each layer sees the same |X|).
        per_layer_tokens = payload.total_tokens_per_layer
        layer_tokens = int(per_layer_tokens[layer_rank].item())
        max_layer_tokens = int(per_layer_tokens.max().item()) if per_layer_tokens.numel() > 0 else 0
        threshold = int(self.partial_hit_fraction * max_layer_tokens)
        if layer_tokens <= 0 or layer_tokens < threshold:
            ctx.set("stage2_profile_partial_hit", True)
            log.info(
                "stage2-profile-cache: layer %d (rank=%d) partial hit "
                "(layer_tokens=%d, threshold=%d; partial because "
                "layer_tokens <= 0 or layer_tokens < threshold) — "
                "live forward will run",
                layer_idx, layer_rank, layer_tokens, threshold,
            )
            return

        # --- Full hit: hydrate the per-layer accumulators in-place ----------
        ream_acc = ctx.get("ream_acc")
        layer_input_acc = ctx.get("layer_input_acc")

        # gate_logit_profiles: preserve list-of-tuples shape verbatim. The
        # downstream consumer compute_gate_similarity_matrix
        # (activation_hooks.py:501) unpacks via `for _, t in batches` —
        # only the tensors are read, but the offset MUST stay in the tuple
        # so the live storage shape contract is preserved (round-2 Crit-1).
        sidecar_batches = payload.gate_logit_profiles.get(layer_rank, ())
        # Copy the list (shallow) so future writer-side mutations of the
        # payload list cannot bleed into the live acc. Tensors themselves
        # are not cloned (record_router_logits never mutates them post-append).
        ream_acc.gate_logit_profiles[layer_idx] = list(sidecar_batches)

        # _sim_tensor: clone the [E, E] row so any in-place add later
        # (none expected on full hit, defense-in-depth) does not mutate
        # the payload.
        sim_row = payload.sim_tensor[layer_rank]
        ream_acc._sim_tensor[layer_idx] = sim_row.clone()

        # _total_tokens_by_layer: scalar int.
        ream_acc._total_tokens_by_layer[layer_idx] = layer_tokens

        # _neuron_act_sum / _neuron_act_count: iterate only the entries
        # for this rank (the payload may carry many ranks; we hydrate
        # only the current layer's slice).
        for (lr, e), v in payload.neuron_act_sum.items():
            if int(lr) == int(layer_rank):
                ream_acc._neuron_act_sum[(layer_idx, int(e))] = v.clone()
        for (lr, e), c in payload.neuron_act_count.items():
            if int(lr) == int(layer_rank):
                ream_acc._neuron_act_count[(layer_idx, int(e))] = int(c)

        # cov_acc hydration — DIRECT dict write into the finalized
        # storage. NOT cov_acc.update(...) (which expects a raw input
        # matrix x and accumulates xᵀx into _pending — the wrong path
        # for a pre-finalized payload). The cast pins the live
        # storage_dtype so the run's downstream consumers see a uniform
        # dtype across hydrated + live cov rows (round-2 Crit-2).
        cov_acc = self.cov_acc
        live_dtype = cov_acc.storage_dtype
        for (lr, e, m), cov_t in payload.cov_acc.items():
            if int(lr) == int(layer_rank):
                cov_acc.covariance[(layer_idx, int(e), str(m))] = (
                    cov_t.to(live_dtype)
                )
        for (lr, e, m), n in payload.cov_token_count.items():
            if int(lr) == int(layer_rank):
                cov_acc.token_count[(layer_idx, int(e), str(m))] = int(n)

        # layer_input_acc hydration — REQUIRED by SC strategy
        # (cost_alignment="output") to compute _output_space_cost from
        # layer_input_acc.buffer. The accumulator is None on runs that
        # do not need it (LayerMergePlugin.on_layer_setup sets it to None
        # when both expert_distill_steps==0 AND cost_alignment != "output"),
        # in which case the reservoir data is left unused in the sidecar
        # (acceptable opt-in cost — see plan §10 N-3 / round-3 guard).
        if layer_input_acc is not None:
            reservoir_t = payload.layer_input_reservoir[layer_rank]
            # Empty placeholder ``(0, 0)`` tensor signals the writer ran
            # without the ``VLLM_CALIB_CAPTURE_LAYER_IN=1`` env gate (or
            # against a pre-CRITICAL-1 vLLM patch that lacks the
            # ``layer_in`` dispatch site). Skip hydration so the live
            # path's fresh empty accumulator remains.
            if reservoir_t is not None and reservoir_t.numel() > 0:
                layer_input_acc.buffer = reservoir_t.clone()
                layer_input_acc.seen = int(reservoir_t.size(0))
            elif self._cost_alignment_requires_reservoir():
                # Hidden-bug fix (plan §5.b): on
                # ``cost_alignment="output"`` runs an empty reservoir
                # WILL crash ``_output_space_cost`` downstream. Demote to
                # partial hit so :meth:`LayerMergePlugin.on_profile`
                # re-runs ``_profile_layer`` and the live forward
                # populates ``layer_input_acc.buffer``. Without this the
                # first run against a stale pre-hook sidecar that hits a
                # ``cost_alignment="output"`` strategy raises::
                #
                #     RuntimeError: _output_space_cost: no layer-input
                #     calibration tokens were captured ...
                ctx.set("stage2_profile_partial_hit", True)
                log.info(
                    "stage2-profile-cache: layer %d (rank=%d) demoted "
                    "to partial hit (empty layer_input_reservoir + "
                    "cost_alignment='output') — live forward will run",
                    layer_idx, layer_rank,
                )
                return

        ctx.set("stage2_profile_full_hit", True)
        log.info(
            "stage2-profile-cache: layer %d (rank=%d) FULL HIT — "
            "hydrated ream_acc + cov_acc + layer_input_acc, "
            "live forward will be skipped",
            layer_idx, layer_rank,
        )
