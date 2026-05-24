"""Causal-ΔNLL ablation filter — load-bearing final-blacklist producer.

Paper
-----
**None — there is no paper for this filter.** It is project-original.

The Super-Experts paper arXiv:2507.23279 validates its detected SE set via
*global* ablation experiments — pruning the full SE set together vs.
pruning random expert sets of the same size — and reports Avg. / ARC-c /
WikiPPL / etc. impact on downstream task suites (audit/spec_compliance/
01_papers/2507.23279/source.md L379-L383 (dynamic-SE-pruning motivation)
+ L574-L588 (Table 3 caption + Baseline/Prune SEs/Random rows)).
It does NOT use ablation as a *per-expert* filter to admit/reject
candidates one at a time.

This plugin runs a per-candidate ablation pass: for every ``(l, e)`` in
the Phase-C candidate pool, install a forward hook that zeros expert
``e``'s ``down_proj`` output, measure ΔNLL on a held-out slice, and admit
``(l, e)`` to the final blacklist only when ``ΔNLL > ablation_filter_threshold``.

Official code
-------------
None. The companion paper's official repo
(ZunhaiSu/Super-Experts-Profilling @
``573aead3127ae593ba267758b832944f8fed1485``) implements the three-way AND
detector and Table 3-style global-ablation evaluation, but does not
implement per-expert ablation-filter blacklist construction.

Deviation: D-causal-ablation-validation
---------------------------------------
Project-original (and load-bearing) — this filter produces Stage 1's
final blacklist. Procedure:

  1. Held-out slice: ``holdout_samples`` (=100) calibration sequences
     drawn with a deterministic seed offset distinct from earlier
     calibration passes; cached at ``_calibration_cache_phase_d/``.
  2. Baseline: forward over the slice with no ablation; record mean
     per-token NLL ``baseline_nll``.
  3. For each candidate ``(l, e)``: install a forward hook that zeros
     expert ``e``'s ``down_proj`` output; measure ``ablated_nll``;
     ``ΔNLL = ablated_nll − baseline_nll``. Remove hook; restore.
  4. Filter: ``blacklist = {(l, e) | ΔNLL > ablation_filter_threshold}``
     (default ``0.001`` ≈ 0.1 % PPL impact). Per-candidate ΔNLL retained
     in ``stage1_ablation_filter.json`` for audit.

Cost: ``|candidates|`` × held-out-forward time. At
``ablation_filter_batch_size = 8`` and 100 holdout samples
(~13 batches per candidate) each candidate takes ~15 s →
~15–30 min for a 60–100-candidate pool. The earlier-pipeline
accumulators stay resident through this pass (downstream CKA still
needs them), so resident memory is the earlier-pass footprint plus the
held-out cache.

Why this is load-bearing
------------------------
v4 produced a 158-expert blacklist of which only 5 had measurable
ablation ΔNLL (144 dead-weight, 9 false positives that *hurt* PPL when
protected). Static-threshold detection is fragile across architectures —
each new architecture shifts the right thresholds and the right values
are unknown until the model is run. Ablation evidence is ground truth.

v6 promoted ablation from report-only (the legacy ``run_phase_f``) to
the load-bearing final filter and rewrote Phase C as a candidate-pool
generator (three-way AND ∪ AIMER ∪ sink-token ∪ magnitude-top-K) gated
by this pass.

Ablation semantics
------------------
The ``down`` callback in :func:`instrument_experts` receives the
``down_proj`` output by reference *before* it is multiplied by routing
weights and ``index_add_``-ed into the layer's output. Calling
``tensor.zero_()`` on the callback argument therefore zeroes the tensor
that the next two lines of the wrapped forward consume — see
``activation_hooks.py`` ``wrapped_factored`` (L1300 ``_cb("down", ...)``,
L1301-L1302 routing-weight multiply + ``final.index_add_``) and
``wrapped_fused`` (L1332 ``_cb("down", ...)``, L1333-L1334 same two-line
consumer). Line numbers may drift across edits; the function names are
the stable anchor. The forward runs under ``torch.no_grad()`` so
in-place mutation is safe.

Git archaeology
---------------
- ``1b4e3bd``/``da71126`` (2026-05-10) "feat(stage1): run_ablation_filter
  — promote ablation from report to filter" — the v5→v6 architectural
  switch. ``run_ablation_filter`` accepts the Phase-C candidate dict and
  returns ``(validated_blacklist, per_candidate_dnll, baseline_nll)``;
  ``_apply_threshold_filter`` extracted as a unit-testable threshold
  helper; ``_write_ablation_filter_artifact`` introduced for the new
  ``stage1_ablation_filter.json`` schema. Same commit marks the legacy
  ``run_phase_f`` as deprecated.
- ``51c49bf``/``473241b``: "Phase C now candidate generation; Phase D
  ablation filter" — orchestrator wiring of the v6 architecture.
- ``0e497fd``/``f236d82``: "Merge stage1 v6: ablation-as-filter
  architecture" — landed the architectural switch on the integration
  branch.
- ``7bc65b3``/``10fdf05`` (2026-05-10, after job 6a00caf0) "fix(stage1):
  drop ablation_filter.batch_size 32→8 (v4-proven; bs=32 OOM)" — the
  first H200 run OOM'd on the baseline ``_measure_corpus_nll`` forward
  because the ``ForCausalLMLoss`` bf16→fp32 logits upcast wanted
  ``32 × 2048 × 151936 × 4`` ≈ 38 GB at bs=32. Dropped to bs=8 (logits
  upcast ~9.5 GB); matches the v4-proven Phase F batch size and leaves
  ~40 GB free with the model + earlier-pipeline accumulators resident.

Naming-history note
-------------------
The legacy stage-1 monolith called this "Phase D" (the ablation filter
slot in the A → B → C → D → E → F phase chain). The current plugin
architecture has no phase taxonomy. Log strings and Trackio keys retain
``"Stage 1 Phase D"`` for dashboard back-compat; the deprecated
``run_phase_f`` (v5 entry point) keeps its name verbatim for any
out-of-tree caller. New prose drops the labels.

Plugin contract
---------------
``writes`` covers five slots: the three downstream-readable outputs
(``blacklist``, ``candidate_deltas``, ``baseline_nll``) plus two
internal bookkeeping slots (``ablation_filter_threshold``,
``ablation_filter_config``) consumed only by :meth:`contribute_artifact`.
``provides = ()`` — this filter runs its own dedicated ablation forward
pass (one per candidate) and does NOT consume any shared accumulator
from the earlier calibration pass.

``contribute_artifact`` returns the six-key
``stage1_ablation_filter.json`` payload: ``baseline_mean_nll``,
``ablation_filter_threshold``, ``candidate_count``, ``blacklist_count``,
``candidates`` (per-key ``{delta_nll, provenance, passed_filter}``),
``config`` (the resident ``ablation_filter_config`` dict).

When ``stage1_grape.ablation_filter.enabled`` is ``False`` the worker
falls back to using the candidate set verbatim as the blacklist (empty
``candidate_deltas`` and ``baseline_nll = 0.0``) — matching the legacy
short-circuit at the top of :func:`run_ablation_filter`.

Naming-note: ``blacklist_threshold`` vs ``ablation_filter_threshold``
---------------------------------------------------------------------
The config key is ``stage1_grape.ablation_filter.blacklist_threshold``
(read at :func:`run` / :func:`run_ablation_filter`); the context slot
+ artifact key is ``ablation_filter_threshold`` (written by
:func:`run` / read by :meth:`contribute_artifact`). They refer to the
*same value* via the v6 ablation-as-load-bearing-blacklist-filter
architecture rename — the YAML keeps the original "blacklist" framing
(the per-candidate threshold for admission to the final blacklist) and
the downstream artifact uses the v6 "ablation_filter" framing. Both
names are intentionally kept (YAML compat for in-flight configs + clear
artifact provenance) and are NOT to be unified by rename — both are
referenced elsewhere (configs/*.yaml, stage1_ablation_filter.json
consumers).
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
    """Causal-ΔNLL per-expert ablation filter — load-bearing final-blacklist producer.

    Reads the Phase-C candidate pool + a held-out calibration slice;
    ablates each candidate by zeroing its ``down_proj`` output during a
    forward pass; keeps candidates whose ΔNLL exceeds
    ``ablation_filter_threshold``. See the module docstring for the
    paper / official-code citations (both negative — the filter is
    project-original), the full deviation rationale, and the v4→v6
    ablation-as-filter promotion archaeology.

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
        "Causal-ΔNLL per-expert ablation filter (project-original; no paper). "
        "arXiv:2507.23279 validates SE detection via global ablations only "
        "(source.md L379-L383 dynamic-SE-pruning motivation + L574-L588 "
        "Table 3 caption + Baseline/Prune SEs/Random rows); official code "
        "ZunhaiSu/Super-Experts-Profilling @ "
        "573aead3127ae593ba267758b832944f8fed1485 implements no per-expert "
        "ablation-filter blacklist construction. Deviation: "
        "D-causal-ablation-validation — see module docstring for the v6 "
        "ablation-as-filter rationale, the 158→5 v4-evidence motivation, "
        "and the load-bearing role in final-blacklist construction."
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

        Two-layer gating semantics — these two paths are NOT equivalent:

          1. **Orchestrator gate (primary).** Returning ``False`` here
             typically tells the orchestrator *not to invoke* :meth:`run`
             at all. Whether the orchestrator honors this gate (and what
             it does to fill the downstream ``blacklist`` / ``candidate_deltas``
             / ``baseline_nll`` slots when it skips) is the orchestrator's
             responsibility — see :mod:`moe_compress.stage1.orchestrator`.
          2. **Inner short-circuit (fallback).** If the orchestrator does
             invoke :meth:`run` *anyway* (e.g. tests calling the plugin
             directly, or a future orchestrator policy that always runs
             every plugin), the short-circuit at the top of
             :func:`run_ablation_filter` (re-reads ``af["enabled"]``)
             falls back to using the candidate set verbatim as the
             blacklist with empty ``candidate_deltas`` and
             ``baseline_nll = 0.0``.

        The inner short-circuit is the fallback path only if ``run`` IS
        invoked while ``enabled=False`` — it is NOT what ``is_enabled``
        itself reports. Inverting the gate so this method always returns
        ``True`` and the inner short-circuit becomes the sole disabled-
        config path would change orchestrator behavior and is therefore
        deferred; the current dual-path design is preserved.
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
                "ablation_filter_batch_size": int(af.get("batch_size", 8)),
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
        "run_phase_f is deprecated; v6 uses run_ablation_filter on the candidate "
        "pool as the load-bearing final filter. See deviation "
        "D-causal-ablation-validation in the AblationFilterPlugin module "
        "docstring.",
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
      blacklist_threshold (default 0.001), batch_size (default 8).
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
    batch_size = int(af.get("batch_size", 8))

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
    """Write ``stage1_ablation_filter.json`` per the v6 schema (see :class:`AblationFilterPlugin.contribute_artifact`)."""
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
