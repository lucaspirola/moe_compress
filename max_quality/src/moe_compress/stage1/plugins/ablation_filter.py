"""Phase D — Ablation Filter (load-bearing causal-ΔNLL filter).

Paper: ALGORITHM_REFERENCE.md §4 Phase D and §12 D-causal-ablation-validation.
Migrated from the legacy Stage 1 ablation-filter module in sub-task 5 of the
Stage 1 → plugin-architecture refactor.

The plugin's externally observable behaviour is **byte-identical** to the
legacy ``run_ablation_filter`` + ``_write_ablation_filter_artifact`` pair:
same baseline-NLL semantics, same per-candidate ablation forward-pass, same
threshold-filter rule (ΔNLL > threshold), same six-key JSON payload schema.
Verified via the golden snapshot test for ``stage1_ablation_filter.json``
and ``stage1_blacklist.json`` (Phase D feeds the Phase-C-artifact assembly
below).

Ablation semantics
------------------
The ``down`` callback in :func:`instrument_experts` receives the down_proj
output by reference *before* it is multiplied by routing weights and
``index_add_``-ed into the layer's output. Calling ``tensor.zero_()`` on
the callback argument therefore zeroes the tensor that the next two lines
of the wrapped forward consume — see ``activation_hooks.py`` lines 1099-
1103 (factored) and 1132-1135 (fused). The forward runs under
``torch.no_grad()`` so in-place mutation is safe.

The legacy ``run_phase_f`` v5 entry point is kept here verbatim for
back-compat with any out-of-tree caller. New code calls
:class:`AblationFilterPlugin` (or the module-level ``run_ablation_filter``
directly).
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import torch

from ...utils.activation_hooks import instrument_experts
from ...utils.calibration import build_calibration_tensor, iter_batches, spec_from_config
from ...utils.model_io import iter_moe_layers, save_json_artifact
from ...pipeline.context import PipelineContext

log = logging.getLogger(__name__)


class AblationFilterPlugin:
    """Causal-ΔNLL ablation filter (Phase D — load-bearing final-blacklist producer).

    Reads the Phase C candidate set + a held-out calibration slice; ablates
    each candidate by zeroing its down_proj output during a forward pass;
    keeps candidates whose ΔNLL exceeds ``ablation_filter_threshold``.

    The plugin produces two outputs:

    1. **In-memory:** the three context slots ``blacklist``,
       ``candidate_deltas``, ``baseline_nll`` (read by the downstream
       Phase-C-artifact assembly and the Phase E/F delegations).
    2. **JSON file:** the ``stage1_ablation_filter.json`` payload returned
       by :meth:`contribute_artifact` — the orchestrator writes it via
       :func:`utils.model_io.save_json_artifact`. The plugin itself does
       not write to disk.

    When ``stage1_grape.ablation_filter.enabled`` is ``False`` the plugin
    falls back to using the candidate set verbatim as the blacklist (with
    empty ``candidate_deltas`` and ``baseline_nll=0.0``) — matching the
    legacy short-circuit at the top of ``run_ablation_filter``.
    """

    name: str = "ablation_filter"
    paper: str = (
        "Stage 1 ALGORITHM_REFERENCE.md §4 Phase D — D-causal-ablation-validation"
    )
    config_key: str = "stage1_grape.ablation_filter"
    reads: tuple[str, ...] = (
        "candidates",
        "model",
        "tokenizer",
        "config",
        "artifacts_dir",
        "device",
    )
    writes: tuple[str, ...] = (
        "blacklist",
        "candidate_deltas",
        "baseline_nll",
        # Private bookkeeping for ``contribute_artifact`` — not consumed by
        # any downstream plugin or the legacy orchestrator code. Kept in
        # ``writes`` so the Protocol contract is honest about every slot
        # the plugin touches.
        "ablation_filter_threshold",
        "ablation_filter_config",
    )
    # Phase D runs its own dedicated ablation forward pass (one per candidate);
    # it does not consume any shared accumulator from Phase B.
    provides: tuple[str, ...] = ()

    def is_enabled(self, config: dict) -> bool:
        """Read ``config["stage1_grape"]["ablation_filter"]["enabled"]``; default True.

        ``False`` does **not** skip Phase D entirely — the plugin still
        runs and the candidate set is used as the blacklist verbatim.
        ``is_enabled`` reflects the orchestrator-visible flag for
        the orchestrator's gating.
        """
        s1 = config.get("stage1_grape", {})
        af = s1.get("ablation_filter", {})
        return bool(af.get("enabled", True))

    def run(self, ctx: PipelineContext) -> None:
        """Execute Phase D end-to-end.

        Reads ``candidates``, ``model``, ``tokenizer``, ``config``,
        ``artifacts_dir``, ``device`` from ``ctx``; writes ``blacklist``,
        ``candidate_deltas``, ``baseline_nll``, ``ablation_filter_threshold``,
        ``ablation_filter_config`` back.

        Delegates the ablation work to :func:`run_ablation_filter` (the
        moved legacy worker — byte-identical to the original).
        """
        candidates = ctx.get("candidates")
        model = ctx.get("model")
        tokenizer = ctx.get("tokenizer")
        config = ctx.get("config")
        artifacts_dir = ctx.get("artifacts_dir")
        device = ctx.get("device")

        s1 = config["stage1_grape"]
        af = s1.get("ablation_filter", {})
        threshold = float(af.get("blacklist_threshold", 0.001))

        blacklist, candidate_deltas, baseline_nll = run_ablation_filter(
            model, tokenizer, config, artifacts_dir,
            candidates=candidates,
            device=device,
        )

        ctx.set("blacklist", blacklist)
        ctx.set("candidate_deltas", candidate_deltas)
        ctx.set("baseline_nll", baseline_nll)
        ctx.set("ablation_filter_threshold", threshold)
        # Plugin-internal default config dict — only the three keys the
        # plugin itself can derive from its own inputs. The orchestrator
        # may overwrite this slot with a wider mix that includes Phase-C
        # state, in which case the golden-snapshot byte-anchor uses the
        # wider dict.
        ctx.set(
            "ablation_filter_config",
            {
                "holdout_samples": int(af.get("holdout_samples", 100)),
                "ablation_filter_threshold": threshold,
                "ablation_filter_batch_size": int(af.get("batch_size", 32)),
            },
        )

    def contribute_artifact(self, ctx: PipelineContext) -> dict:
        """Return the ``stage1_ablation_filter.json`` payload.

        Identical six-key schema to the legacy
        :func:`_write_ablation_filter_artifact` (the pre-sub-task-5
        ablation-filter module):

        Returns
        -------
        dict
            Exactly six top-level keys:
              - ``baseline_mean_nll`` : float
              - ``ablation_filter_threshold`` : float
              - ``candidate_count`` : int
              - ``blacklist_count`` : int
              - ``candidates`` : dict[str, {"delta_nll", "provenance",
                "passed_filter"}]
              - ``config`` : dict (whatever ``ablation_filter_config``
                holds at call time — the plugin's default subset OR the
                orchestrator's wider overlay; see the overwrite path in
                §4.2.3 of subtask_5_plan.md)
        """
        candidates: dict[tuple[int, int], list[str]] = ctx.get("candidates")
        deltas: dict[tuple[int, int], float] = ctx.get("candidate_deltas")
        baseline_nll: float = ctx.get("baseline_nll")
        threshold: float = ctx.get("ablation_filter_threshold")
        blacklist: dict[int, list[int]] = ctx.get("blacklist")
        config_dict: dict = ctx.get("ablation_filter_config")

        blacklisted_pairs = {(li, e) for li, exps in blacklist.items() for e in exps}
        candidates_payload: dict[str, dict] = {}
        for (li, e), provenance in candidates.items():
            key = f"L{li}E{e}"
            delta = deltas.get((int(li), int(e)))
            candidates_payload[key] = {
                "delta_nll": float(delta) if delta is not None else None,
                "provenance": list(provenance),
                "passed_filter": (int(li), int(e)) in blacklisted_pairs,
            }

        return {
            "baseline_mean_nll": float(baseline_nll),
            "ablation_filter_threshold": float(threshold),
            "candidate_count": len(candidates),
            "blacklist_count": sum(len(v) for v in blacklist.values()),
            "candidates": candidates_payload,
            "config": dict(config_dict),
        }


# ---------------------------------------------------------------------------
# Module-level helpers — moved verbatim from the legacy Stage 1
# ablation-filter module (pre-sub-task-5). This is the single source of truth.
# ---------------------------------------------------------------------------


def rank_top_nonblacklisted(
    per_expert_max: dict[tuple[int, int], float],
    blacklist: dict[int, list[int]],
    L: set[int],
    top_k: int,
) -> dict[int, list[int]]:
    """For each l ∈ L, return top-`top_k` non-blacklisted expert ids by per_expert_max desc."""
    blacklisted = {(li, e) for li, lst in blacklist.items() for e in lst}
    by_layer: dict[int, list[tuple[int, float]]] = {}
    for (li, e), v in per_expert_max.items():
        if li not in L or (li, e) in blacklisted:
            continue
        by_layer.setdefault(li, []).append((e, v))
    out: dict[int, list[int]] = {}
    for li, lst in by_layer.items():
        lst.sort(key=lambda t: -t[1])
        out[li] = [e for e, _ in lst[:top_k]]
    return out


def _measure_corpus_nll(model, batches, device) -> float:
    """Mean per-token NLL across the held-out slice. Cross-entropy on shifted labels."""
    total_nll = 0.0
    total_tokens = 0
    model.eval()
    with torch.no_grad():
        for batch in batches:
            batch = batch.to(device) if device is not None else batch
            out = model(input_ids=batch, labels=batch)
            ntok = (batch.shape[0] * (batch.shape[1] - 1))  # shift-by-1
            total_nll += float(out.loss.item()) * ntok
            total_tokens += ntok
    return total_nll / max(total_tokens, 1)


def _ablate_expert_context(layer_ref, expert_idx: int):
    """Context manager that zeros the named expert's down_proj output for its lifetime.

    Relies on ``instrument_experts``'s ``down`` callback receiving the
    down_proj output by reference *before* it is multiplied by routing
    weights — see module docstring for the activation_hooks reference.
    """
    def _zero_cb(li, e, tensor, _ctx):
        if e == expert_idx:
            tensor.zero_()
    return instrument_experts(layer_ref, {"down": _zero_cb})


def run_phase_f(
    model,
    tokenizer,
    config: dict,
    artifacts_dir: Path,
    blacklist: dict[int, list[int]],
    per_expert_max: dict[tuple[int, int], float],
    L: set[int],
    *,
    device=None,
) -> Path:
    """Deprecated v5 entry point — write the old post-hoc ablation report.

    v6 promotes ablation from report to filter: prefer :func:`run_ablation_filter`
    on the Phase C candidate set. This function is retained for callers that
    still want the legacy ``stage1_post_hoc_ablation.json`` report (blacklist
    ΔNLL + top-K non-blacklisted ΔNLL).
    """
    warnings.warn(
        "run_phase_f is deprecated; v6 uses run_ablation_filter on the Phase C "
        "candidate set as the load-bearing final filter. See ALGORITHM_REFERENCE.md "
        "§4 Phase D and §12 D-causal-ablation-validation.",
        DeprecationWarning,
        stacklevel=2,
    )
    s1 = config["stage1_grape"]
    cal = config["calibration"]
    pf = s1.get("post_hoc_ablation", {})
    if not bool(pf.get("enabled", True)):
        log.info("Stage 1 Phase F: disabled in config; skipping")
        return artifacts_dir / "stage1_post_hoc_ablation.json"

    holdout_samples = int(pf.get("holdout_samples", 100))
    top_k = int(pf.get("topk_nonblacklisted", 5))
    # The ablation forward pass is forward-only with a single per-expert hook;
    # the held-out NLL is invariant to batch size as long as token coverage is
    # complete. bs=8 cuts the per-ablation batch count from 100 to 13. The
    # Phase B accumulators (max_acc, output_acc) remain resident in run()'s
    # scope during Phase F, so the memory budget is the same as Phase B's
    # ~120-130 GB at bs=8 — still ~20-30 GB headroom on the 150.8 GB H200.
    batch_size = int(pf.get("batch_size", 8))

    # Held-out slice: deterministic seed offset distinct from Phase A/B
    spec = spec_from_config(cal, num_sequences_override=holdout_samples, seed_offset=999)
    calib = build_calibration_tensor(
        tokenizer, spec, cache_dir=artifacts_dir / "_calibration_cache_phase_f",
    )
    eval_batches = iter_batches(calib, batch_size=batch_size)

    moe_layers = {ref.layer_idx: ref for ref in iter_moe_layers(model)}

    # Baseline: no ablation
    baseline_nll = _measure_corpus_nll(model, eval_batches, device)
    log.info("Stage 1 Phase F: baseline mean NLL = %.4f", baseline_nll)

    impacts: dict[str, dict] = {"blacklisted": {}, "top_nonblacklisted": {}}

    # Blacklisted ablations
    for li, exps in blacklist.items():
        ref = moe_layers[int(li)]
        for e in exps:
            with _ablate_expert_context(ref, e):
                nll = _measure_corpus_nll(model, eval_batches, device)
            impacts["blacklisted"][f"L{li}E{e}"] = nll - baseline_nll

    # Top-K non-blacklisted candidates per l ∈ L
    candidates = rank_top_nonblacklisted(per_expert_max, blacklist, L, top_k=top_k)
    for li, exps in candidates.items():
        ref = moe_layers[li]
        for e in exps:
            with _ablate_expert_context(ref, e):
                nll = _measure_corpus_nll(model, eval_batches, device)
            impacts["top_nonblacklisted"][f"L{li}E{e}"] = nll - baseline_nll

    out_path = artifacts_dir / "stage1_post_hoc_ablation.json"
    save_json_artifact(
        {
            "baseline_mean_nll": baseline_nll,
            "delta_nll": impacts,
            "config": {
                "holdout_samples": holdout_samples,
                "topk_nonblacklisted": top_k,
                "ma_formation_layers": sorted(L),
            },
        },
        out_path,
    )
    log.info("Stage 1 Phase F: wrote %d ablation results to %s",
             len(impacts["blacklisted"]) + len(impacts["top_nonblacklisted"]),
             out_path)
    return out_path


def _apply_threshold_filter(
    deltas: dict[tuple[int, int], float],
    threshold: float,
) -> dict[int, list[int]]:
    """Keep candidates whose ΔNLL > threshold; group by layer with sorted expert ids."""
    bl: dict[int, list[int]] = {}
    for (li, e), d in deltas.items():
        if d > threshold:
            bl.setdefault(int(li), []).append(int(e))
    for li in bl:
        bl[li] = sorted(bl[li])
    return bl


def run_ablation_filter(
    model,
    tokenizer,
    config: dict,
    artifacts_dir: Path,
    candidates: dict[tuple[int, int], list[str]],
    *,
    device=None,
) -> tuple[dict[int, list[int]], dict[tuple[int, int], float], float]:
    """Phase D — ablate every Phase C candidate; return validated blacklist + per-candidate ΔNLL.

    Parameters
    ----------
    candidates: ``{(layer_idx, expert_idx): [provenance_tags...]}``
        Phase C output. Provenance is ignored by the filter itself but is
        carried through to the artifact for audit.

    Returns
    -------
    blacklist: ``{layer_idx: sorted_expert_ids}``
        The validated final blacklist (ΔNLL > ``ablation_filter_threshold``).
    per_candidate_dnll: ``{(layer_idx, expert_idx): ΔNLL}``
        ΔNLL for every candidate (kept and rejected alike).
    baseline_nll: float
        Held-out NLL with no ablation.

    Reads ``config["stage1_grape"]["ablation_filter"]``:
      enabled (default True), holdout_samples (default 100),
      blacklist_threshold (default 0.001), batch_size (default 32).
    """
    s1 = config["stage1_grape"]
    af = s1.get("ablation_filter", {})
    if not bool(af.get("enabled", True)):
        log.info(
            "Stage 1 Phase D: ablation_filter disabled; falling back to candidate set as blacklist"
        )
        bl: dict[int, list[int]] = {}
        for (li, e), _ in candidates.items():
            bl.setdefault(int(li), []).append(int(e))
        for li in bl:
            bl[li] = sorted(bl[li])
        return bl, {}, 0.0

    holdout_samples = int(af.get("holdout_samples", 100))
    threshold = float(af.get("blacklist_threshold", 0.001))
    batch_size = int(af.get("batch_size", 32))

    spec = spec_from_config(
        config["calibration"], num_sequences_override=holdout_samples, seed_offset=999
    )
    calib = build_calibration_tensor(
        tokenizer, spec, cache_dir=artifacts_dir / "_calibration_cache_phase_d",
    )
    eval_batches = iter_batches(calib, batch_size=batch_size)

    moe_layers = {ref.layer_idx: ref for ref in iter_moe_layers(model)}
    baseline_nll = _measure_corpus_nll(model, eval_batches, device)
    log.info(
        "Stage 1 Phase D: baseline mean NLL = %.4f over %d candidates",
        baseline_nll, len(candidates),
    )

    deltas: dict[tuple[int, int], float] = {}
    for (li, e), _provenance in candidates.items():
        ref = moe_layers.get(int(li))
        if ref is None:
            log.warning(
                "Phase D: layer %d not found in moe_layers; skipping (l=%d, e=%d)", li, li, e
            )
            continue
        with _ablate_expert_context(ref, int(e)):
            nll = _measure_corpus_nll(model, eval_batches, device)
        deltas[(int(li), int(e))] = nll - baseline_nll

    blacklist = _apply_threshold_filter(deltas, threshold)
    log.info(
        "Stage 1 Phase D: ablation-filter kept %d / %d candidates (threshold=%.4f)",
        sum(len(v) for v in blacklist.values()), len(deltas), threshold,
    )
    return blacklist, deltas, baseline_nll


def _write_ablation_filter_artifact(
    artifacts_dir: Path,
    *,
    candidates: dict[tuple[int, int], list[str]],
    deltas: dict[tuple[int, int], float],
    baseline_nll: float,
    threshold: float,
    blacklist: dict[int, list[int]],
    config_dict: dict,
) -> Path:
    """Write ``stage1_ablation_filter.json`` per the v6 schema (see ALGORITHM_REFERENCE.md §4)."""
    blacklisted_pairs = {(li, e) for li, exps in blacklist.items() for e in exps}
    candidates_payload: dict[str, dict] = {}
    for (li, e), provenance in candidates.items():
        key = f"L{li}E{e}"
        delta = deltas.get((int(li), int(e)))
        candidates_payload[key] = {
            "delta_nll": float(delta) if delta is not None else None,
            "provenance": list(provenance),
            "passed_filter": (int(li), int(e)) in blacklisted_pairs,
        }

    payload = {
        "baseline_mean_nll": float(baseline_nll),
        "ablation_filter_threshold": float(threshold),
        "candidate_count": len(candidates),
        "blacklist_count": sum(len(v) for v in blacklist.values()),
        "candidates": candidates_payload,
        "config": dict(config_dict),
    }
    out_path = artifacts_dir / "stage1_ablation_filter.json"
    save_json_artifact(payload, out_path)
    return out_path
