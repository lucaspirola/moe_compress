"""Faithful-to-upstream REAP expert PRUNER (pure structural drop, no merge).

This plugin implements a one-shot REAP expert *prune*: it keeps the
top-``(n_experts − n_prune)`` experts by REAP saliency and **drops** the
bottom ``n_prune`` outright (their expert tensors and router rows are
sliced away). No merge, no SVD, no heal — and, critically, **no router
rescale** after the drop. It is the structural analog of upstream
``prune()``; the surviving router logits are renormalized by the model's
own forward (``config.norm_topk_prob``), not baked into the weights.

Mode gating
-----------
The plugin is INERT by default: ``is_enabled`` returns True only when
``stage2_reap_ream.prune_mode == "faithful_prune"`` (default ``"merge"``).
With the default mode every existing config / golden snapshot is
byte-identical (the plugin is never registered into the enabled set, and
the orchestrator keeps the merge path).

Paper
-----
Thangarasa et al., "REAP the Experts: Why Pruning Prevails for One-Shot
MoE Compression" — arXiv:2510.13999 (ICLR 2026).

Official code
-------------
``CerebrasResearch/reap`` @ ``1970473c51ca3caeb98c10392f15b3a08a672974``.
The drop loop lives in ``src/reap/prune.py:82-145``; the selection is
``torch.topk(saliency, n_experts_to_prune, largest=False)``
(``prune.py:101``), and the kept set is the complement
(``prune.py:105``). Upstream performs NO router rescale after the slice
(``prune.py:142``).

Upstream parity (the kept-set formula)
--------------------------------------
``compute_assignment`` computes ``final_kept_ids`` so that, with an empty
protected set, it is byte-identical to upstream's

    experts_to_prune = torch.topk(scores, n_prune, largest=False).indices
    retained         = [i for i in range(n) if i not in experts_to_prune]

``n_prune`` is a single GLOBAL scalar computed ONCE from the first MoE
layer's expert count and reused at every layer (the Qwen3.x stack is
homogeneous so absolute-count and fixed-fraction coincide).

**D-keep-rounded (drop-count derivation).** We derive the drop so the KEPT
count is ``round((1 - prune_fraction) * n_experts0)`` (i.e. ``n_prune =
n_experts0 - round((1 - prune_fraction) * n_experts0)``). Upstream computes
``int(total_experts * compression_ratio)`` (``prune.py:258-261``), a
TRUNCATION. The two agree whenever ``n_experts0 * prune_fraction`` is
integral; they differ only on a non-integral product. We deliberately use
keep-rounded — the exact statement of the "keep the top-(1 -
compression_ratio) experts" intent — so the production case (256 experts,
0.35) keeps ``round(166.4) = 166`` (drops 90), matching the REAM survivor
count K=166, rather than upstream's ``int(89.6) = 89`` (keep 167, an
off-by-one vs REAM). With an empty protected set the SELECTION (which
experts survive given ``n_prune``) is still byte-identical to upstream's
``torch.topk`` formula; only the scalar ``n_prune`` rounding differs.

Deviations (vs upstream default)
--------------------------------
- **D-prune-fraction-source:** we expose an explicit ``prune_fraction``
  config knob instead of deriving the drop count from the Stage-1
  GRAPE-allocated per-layer budget (``ctx.get("target")``). That budget
  is a function of the SVD-aware param-split reduction config, NOT a
  clean expert-drop fraction — so faithful mode BYPASSES it entirely and
  computes ``n_prune`` from ``prune_fraction`` (matches upstream's direct
  ``compression_ratio`` semantics).
- **D-protected-experts (GENUINE DIVERGENCE):** we always hard-exclude
  protected (super / shared) experts from the drop candidates (a Stage-1
  contract). Upstream's super-expert preservation is OFF by default
  (``args.py:514-515,522-523`` both ``default=False``), so the upstream
  *default* run has no protected set and drops the pure bottom-``n_prune``.
  When our protected set is empty, our selection is byte-identical to
  upstream's formula (proven by the protected=∅ test).
- **D-prune-no-imatrix/cov:** faithful mode collects no covariance /
  imatrix (upstream's drop collects none). Stage 3+ are skipped anyway.
- **D-fp-fused-storage:** our Qwen3.x stores experts in the fused
  ``Qwen3_5MoeExperts`` (``gate_up_proj`` / ``down_proj`` stacked
  tensors); we slice them via ``ExpertMatrixBank.select`` — upstream's own
  fused path (``prune.py:137-140``, Llama-4) does the same; the kept rows
  are identical.

Wiring (option (a))
-------------------
``ReapPrunePlugin`` OWNS ``compute_assignment`` / ``merge`` (no-op) /
``post_merge`` (the drop) / ``write_artifacts`` (its own merge-JSON
payload + a resume sentinel ``.pt``).

Score source — vLLM sidecar ONLY (FAIL LOUD)
--------------------------------------------
The pruner sources REAP saliency from the ``--capture-reap-scores`` vLLM
sidecar (the ``scores`` ctx slot hydrated by
``Stage2ReapScoresCacheProvider.on_score`` on a cache hit). Faithful mode
drops ``LayerMergePlugin`` — the only ``on_profile`` hook that populates a
live ``ReapAccumulator`` — so the sidecar is the ONLY score source. If
``scores`` is absent (cache miss / sidecar not built), ``compute_assignment``
raises a descriptive ``RuntimeError`` rather than running its own HF
forward-pass rescore (which would diverge from the vLLM calibration
distribution). The operator must run the vLLM calibration with
``--capture-reap-scores`` first.

In faithful mode the orchestrator DROPS ``LayerMergePlugin`` (its ``write_artifacts`` would ``ctx.get()``
~12 bump-loop slots the faithful path never sets, and ``ctx.get`` raises
``KeyError`` on a missing slot). The orchestrator also routes faithful
layers through ``walk_phases(_STAGE2_LAYER_PHASES, ...)`` (the plain
``compute_assignment`` slot) instead of ``_run_assignment`` (the bump
loop), so no cost matrix / solver / merge / heal machinery runs.

Resume (sentinel ``.pt``)
-------------------------
``resume.py`` treats a layer as completed only when BOTH
``merge_{idx}.json`` AND ``layer_{idx}.pt`` exist. Faithful mode collects
no covariance, so ``_snapshot_cov_layer`` would never write the ``.pt`` —
silently re-running every layer on resume. ``write_artifacts`` therefore
writes an empty-but-valid sentinel ``.pt``
(``{"format_version": 1, "covariance": {}, "tokens": {}}``) so the
resume gate passes and the layer is replayed (``bank.select`` +
``_resize_router_for_kept_experts``) instead of re-profiled. The replay's
``_merge_experts_inplace(record.grouped, …)`` is a no-op for faithful
singletons because it skips groups of ``len(members) <= 1``
(``merging.py:161-163``).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from ...pipeline.context import PipelineContext
from ...utils.model_io import build_banks
from ..merging import _resize_router_for_kept_experts
from ..shared_io import _durable_rename, _write_merge_json

log = logging.getLogger(__name__)

PRUNE_MODE_KEY = "prune_mode"
FAITHFUL_PRUNE = "faithful_prune"
MERGE_MODE = "merge"


def faithful_prune_enabled(config: dict) -> bool:
    """True iff ``stage2_reap_ream.prune_mode == "faithful_prune"``.

    Single source of truth for the faithful-mode gate; the orchestrator,
    ``ReapPrunePlugin.is_enabled``, and the LayerMergePlugin-drop branch
    all consult this so they never disagree.
    """
    s2 = config.get("stage2_reap_ream", {}) or {}
    return s2.get(PRUNE_MODE_KEY, MERGE_MODE) == FAITHFUL_PRUNE


def compute_final_kept_ids(
    scores: np.ndarray,
    *,
    n_experts: int,
    n_prune: int,
    protected: list[int],
) -> tuple[list[int], list[int]]:
    """Top-``(n_experts − n_prune)`` experts by saliency ∪ protected.

    Returns ``(final_kept_ids, pruned_expert_ids)`` — both sorted.

    Mirrors upstream ``torch.topk(scores, n_prune, largest=False)`` (drop
    the bottom ``n_prune``) ≡ keep the top ``faithful_target``. Protected
    experts are NEVER dropped (D-protected-experts); with an empty
    protected set this reduces to the pure upstream formula.

    Tie-break: deterministic keep-LOWEST-index. ``np.argsort`` is stable, so
    equal scores keep ascending expert-index order; the descending pass
    (``-scores``) then keeps the lowest index of a tie and drops the higher.
    This does NOT claim parity with ``torch.topk(largest=False)`` on ties —
    torch's tie order is implementation-defined and differs CPU vs CUDA (e.g.
    ``scores=[0.5, 0.5, 0.9], n_prune=1`` → torch drops index 0, we drop index
    2). Parity is unattainable and irrelevant for continuous REAP saliencies
    (which are never exactly tied); what matters is our own determinism.
    """
    protected_set = set(protected)
    faithful_target = n_experts - n_prune  # kept count (including protected)
    if faithful_target < len(protected_set):
        raise RuntimeError(
            f"faithful prune target {faithful_target} (n_experts={n_experts} "
            f"− n_prune={n_prune}) is smaller than the protected count "
            f"{len(protected_set)}; prune_fraction is too aggressive for this "
            "layer's protected/blacklisted experts"
        )
    # Descending saliency. np.argsort(-scores) is stable → ties resolve to
    # ascending index, so the kept set prefers the lowest-index expert on a tie
    # (deterministic keep-lowest-index; NOT a torch.topk tie-order match — see
    # the docstring).
    order = [int(e) for e in np.argsort(-scores)]
    n_kept_nonprotected = faithful_target - len(protected_set)
    kept_nonprotected = [
        e for e in order if e not in protected_set
    ][:n_kept_nonprotected]
    final_kept_ids = sorted(protected_set | set(kept_nonprotected))
    pruned_expert_ids = sorted(set(range(n_experts)) - set(final_kept_ids))
    return final_kept_ids, pruned_expert_ids


class ReapPrunePlugin:
    """Faithful REAP pruner — INERT unless ``prune_mode == "faithful_prune"``.

    Owns ``compute_assignment`` (top-K selection, BYPASSES
    ``ctx.get("target")``), ``merge`` (no-op default), ``post_merge`` (the
    drop), and ``write_artifacts`` (own payload §3.7 + sentinel ``.pt``).
    See module docstring for the upstream-parity formula, the deviations,
    the wiring, and the resume design.
    """

    name = "reap_prune"
    paper = (
        "REAP one-shot pure expert PRUNE: keep top-(n−n_prune) by Eq.9 "
        "saliency, drop the rest (no router rescale) — arXiv:2510.13999 "
        "(Thangarasa et al.). Official code: CerebrasResearch/reap @ "
        "1970473c51ca3caeb98c10392f15b3a08a672974 (prune.py:82-145). "
        "Deviations: D-prune-fraction-source (explicit prune_fraction, "
        "bypasses Stage-1 budget), D-protected-experts (hard-exclude "
        "protected; upstream default has none), D-prune-no-imatrix/cov, "
        "D-fp-fused-storage. See module docstring."
    )
    config_key = "stage2_reap_ream"
    reads: tuple[str, ...] = (
        "layer_ref", "reap_scores_payload", "scores", "freq", "protected",
        "n_experts", "partial_dir",
    )
    writes: tuple[str, ...] = (
        "final_kept_ids", "grouped", "ream_centroid_ids", "pruned_expert_ids",
        "distill_state", "heal_state", "protected",
    )
    provides: tuple[str, ...] = ()

    def __init__(
        self,
        *,
        prune_fraction: float,
        blacklist: dict[int, list[int]],
        merge_map: dict[int, dict[int, list[int]]],
        partial_dir: Path | None,
    ) -> None:
        # No logic in __init__ — a faithful re-host of the knobs the hooks read.
        self.prune_fraction = float(prune_fraction)
        self.blacklist = blacklist
        # Run-scope mutable scratchpad (mirrors LayerMergePlugin's ownership of
        # ``merge_map``); mutated in write_artifacts, read by the orchestrator
        # after the per-layer loop.
        self.merge_map = merge_map
        self.partial_dir = partial_dir
        # Global drop count, computed ONCE from the first MoE layer's expert
        # count (upstream prune.py:258-261). Cached on first compute_assignment.
        self._n_prune: int | None = None

    def is_enabled(self, config: dict) -> bool:
        """Enabled only in faithful-prune mode; inert (dropped) otherwise."""
        return faithful_prune_enabled(config)

    def contribute_artifact(self, ctx) -> dict:
        return {}

    # ------------------------------------------------------------------
    # compute_assignment — top-K selection (BYPASSES ctx.get("target"))
    # ------------------------------------------------------------------
    def compute_assignment(self, ctx: PipelineContext) -> None:
        """Compute ``final_kept_ids`` = top-(n−n_prune) by REAP score ∪ protected.

        BYPASSES ``ctx.get("target")`` (the Stage-1 GRAPE budget); the drop
        count is ``n_experts0 - round(n_experts0 * (1 - prune_fraction))``
        (keep ``round((1 - prune_fraction) * n_experts0)``) computed once from
        the first MoE layer and reused (homogeneous stack). Publishes the slots
        the standard post-assign schedule + ``write_artifacts`` consume.
        """
        layer_ref = ctx.get("layer_ref")
        n_experts = ctx.get("n_experts")
        # FAIL LOUD if the vLLM REAP-scores sidecar is absent. Faithful mode does
        # NOT run its own HF forward-pass rescoring — the project requires
        # vLLM-sourced REAP scores (the ``--capture-reap-scores`` sidecar) for
        # faithfulness/consistency; an HF rescore would diverge from the
        # calibration distribution.
        #
        # CRITICAL — gate on ``reap_scores_payload``, NOT ``scores``: the
        # ``scores`` slot is an UNRELIABLE proxy for "a sidecar was loaded". On a
        # real cache miss ``Stage2ReapScoresCacheProvider.on_score`` returns
        # without setting ``scores`` (it only sets the payload + scores on a hit,
        # reap_scores_cache.py:64,74-84), and then ``ReapScoringPlugin.on_score``
        # (reap_scoring.py:200-219) runs and publishes an ALL-ZERO ``scores``
        # vector (its live ReapAccumulator is EMPTY because faithful mode drops
        # LayerMergePlugin, the only ``on_profile`` feeder). That would make
        # ``ctx.has("scores")`` True and silently prune on argsort(-zeros) — the
        # exact silent failure this guard exists to prevent.
        # ``reap_scores_payload`` is set ONLY when ``load_reap_scores`` succeeds
        # (reap_scores_cache.py:64), so it is the true "real sidecar present"
        # signal.
        if not ctx.has("reap_scores_payload"):
            raise RuntimeError(
                f"Layer {layer_ref.layer_idx}: faithful_prune mode requires REAP "
                "saliency from the vLLM --capture-reap-scores sidecar, but its "
                "payload was not loaded (cache miss / sidecar absent). Run the "
                "vLLM calibration with --capture-reap-scores FIRST "
                "(build_self_traces_calib_vllm.py), then re-run Stage 2. "
                "Faithful prune does NOT fall back to a fresh HF forward-pass "
                "rescore (it would diverge from the calibration distribution), "
                "and it refuses the all-zero ReapAccumulator scores the live "
                "ReapScoringPlugin would otherwise publish on a sidecar miss."
            )
        scores = ctx.get("scores")

        # Protected experts (super / shared, from stage1_blacklist.json) — never
        # dropped. Publish ``protected`` early (the post-assign schedule + the
        # merge JSON read it). ``overwrite=True`` is intentional: ``protected``
        # is the slot ``_run_assignment`` normally owns (orchestrator.py:271),
        # but ``_run_assignment`` never runs in faithful mode — this plugin is
        # its faithful-mode replacement, so it (re)writes the slot itself.
        protected = sorted(set(self.blacklist.get(layer_ref.layer_idx, [])))
        protected_set = set(protected)
        ctx.set("protected", tuple(protected), overwrite=True)

        # Global n_prune, computed ONCE from layer 0 (upstream prune.py:258-261)
        # and reused per layer (Qwen3.x is homogeneous). Derive the drop count so
        # the KEPT count is ``round((1 - prune_fraction) * n_experts)`` — the
        # upstream "keep the top-(1 - compression_ratio) experts" intent stated
        # exactly. For n_experts=256, prune_fraction=0.35 this keeps
        # round(166.4)=166 (drops 90), vs the old ``int(256*0.35)=89`` which
        # kept 167 (off-by-one vs the REAM 166 survivor count). The keep-rounded
        # form is inert whenever ``n_experts * fraction`` is integral (the cases
        # the existing test suite exercises) and only changes the non-integral
        # production case.
        if self._n_prune is None:
            self._n_prune = n_experts - round(
                n_experts * (1.0 - self.prune_fraction)
            )
        n_prune = self._n_prune

        final_kept_ids, pruned_expert_ids = compute_final_kept_ids(
            scores, n_experts=n_experts, n_prune=n_prune, protected=protected,
        )
        if not final_kept_ids:
            raise RuntimeError(
                f"Layer {layer_ref.layer_idx}: final_kept_ids is empty — "
                f"prune_fraction={self.prune_fraction} dropped every expert"
            )

        # Singleton groups → no merge math. ream_centroid_ids are the kept,
        # non-protected experts (used only for the merge_map / forensic logging).
        grouped = {e: [e] for e in final_kept_ids}
        ream_centroid_ids = [e for e in final_kept_ids if e not in protected_set]

        ctx.set("final_kept_ids", tuple(final_kept_ids))
        ctx.set("grouped", grouped)
        ctx.set("ream_centroid_ids", tuple(ream_centroid_ids))
        ctx.set("pruned_expert_ids", tuple(pruned_expert_ids))
        log.info(
            "  layer %d: faithful prune — keep %d / %d experts "
            "(protected=%d, dropped=%d, n_prune=%d)",
            layer_ref.layer_idx, len(final_kept_ids), n_experts,
            len(protected), len(pruned_expert_ids), n_prune,
        )

    # ------------------------------------------------------------------
    # merge — NO-OP (faithful mode never merges); set the downstream default
    # ------------------------------------------------------------------
    def merge(self, ctx: PipelineContext) -> None:
        """No merge in faithful mode. Set ``distill_state=None`` default.

        Mirrors ``LayerMergePlugin.merge``'s ``distill_state=None`` default so
        ``write_artifacts`` never hits a missing-slot ``KeyError``. No
        ``_merge_experts_inplace``, no covariance touch.
        """
        ctx.set("distill_state", None)

    # ------------------------------------------------------------------
    # post_merge — the structural drop (bank.select + router slice, NO rescale)
    # ------------------------------------------------------------------
    def post_merge(self, ctx: PipelineContext) -> None:
        """Slice expert tensors + router rows to ``final_kept_ids``.

        The fused analog of upstream ``prune.py:137-145``. NO router rescale
        (upstream ``prune.py:142`` only slices) — the model's own forward
        renormalizes the surviving top-k at inference time.
        """
        layer_ref = ctx.get("layer_ref")
        final_kept_ids = list(ctx.get("final_kept_ids"))

        banks = build_banks(layer_ref)
        for bank in banks.values():
            bank.select(final_kept_ids)
        _resize_router_for_kept_experts(layer_ref, final_kept_ids)

        ctx.set("heal_state", None)

    # ------------------------------------------------------------------
    # write_artifacts — own merge-JSON payload (§3.7) + resume sentinel .pt
    # ------------------------------------------------------------------
    def write_artifacts(self, ctx: PipelineContext) -> dict[str, Any]:
        """Write the per-layer merge JSON + sentinel ``.pt`` (faithful payload).

        Emits ONLY the fields ``resume.py`` reads (no bump-loop forensics).
        ``freq`` covers the FULL original expert set (``resume.py`` derives
        ``n_pre_merge = len(freq)`` and asserts it == ``num_routed_experts``).
        ``grouped`` / ``merge_map_layer`` are singletons with NON-EMPTY member
        lists (passing the ``resume.py`` no-empty-member guard). The sentinel
        ``.pt`` makes the ``resume.py`` both-files gate recognize the layer as
        completed (faithful mode collects no covariance).
        """
        partial_dir = ctx.get("partial_dir")
        layer_ref = ctx.get("layer_ref")
        layer_idx = layer_ref.layer_idx
        grouped = ctx.get("grouped")
        freq = ctx.get("freq")
        final_kept_ids = list(ctx.get("final_kept_ids"))
        pruned_expert_ids = list(ctx.get("pruned_expert_ids"))

        # merge_map: new_idx → [orig_eid] singletons (NON-EMPTY member lists).
        # Populated UNCONDITIONALLY (mirrors LayerMergePlugin) — the finalize
        # step writes the aggregate merge_map.json from this run-scope dict even
        # in no-resume mode (partial_dir is None).
        merge_map_layer = {
            new_idx: [eid] for new_idx, eid in enumerate(final_kept_ids)
        }
        self.merge_map[layer_idx] = merge_map_layer

        # Partial-dir checkpoint writes only when resume is enabled.
        if partial_dir is None:
            return {}

        # Sentinel covariance .pt written BEFORE the merge JSON so the
        # .pt-before-.json resume invariant holds (resume.py orphan-cleanup
        # deletes a .pt with no JSON; a JSON with no .pt re-runs the layer).
        self._write_sentinel_cov(partial_dir, layer_idx)

        _write_merge_json(
            partial_dir, layer_idx, final_kept_ids, grouped, freq,
            merge_map_layer,
            mean_cost_per_pair=None,        # no merges → no cost
            assignment_solver_used="none",  # faithful prune uses no solver
            cost_alignment_used="pre",      # neutral default
            em_rounds_completed=0,
            distill_state=None,
            heal_state=None,
            pruned_expert_ids=pruned_expert_ids,  # additive (upstream parity)
            stage2_run_id=(
                ctx.get("stage2_run_id") if ctx.has("stage2_run_id") else None
            ),
        )
        log.info(
            "  layer %d: wrote faithful-prune merge JSON + sentinel .pt "
            "(kept=%d, pruned=%d)",
            layer_idx, len(final_kept_ids), len(pruned_expert_ids),
        )
        return {}

    @staticmethod
    def _write_sentinel_cov(partial_dir: Path, layer_idx: int) -> None:
        """Write an empty-but-valid ``layer_{idx}.pt`` so resume completes.

        Mirrors ``_snapshot_cov_layer``'s schema with empty maps and the same
        atomic tmp + ``_durable_rename`` write. ``_accumulate_payload``
        tolerates empty ``covariance`` / ``tokens`` dicts (it iterates the maps
        → no-op), so replay accumulates nothing.
        """
        import torch  # local import: keep module CPU-import-safe / lazy.

        payload = {"format_version": 1, "covariance": {}, "tokens": {}}
        tmp = partial_dir / f"layer_{layer_idx}.pt.tmp"
        final = partial_dir / f"layer_{layer_idx}.pt"
        torch.save(payload, tmp)
        _durable_rename(tmp, final)
