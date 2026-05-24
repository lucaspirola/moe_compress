"""Sink-token-routing Super-Expert candidate detector.

Paper
-----
Su et al., "Unveiling Super Experts in Mixture-of-Experts Large Language
Models" — arXiv:2507.23279 (audit/spec_compliance/01_papers/2507.23279/source.md).

§5.1 paragraph at L717–L723 + Figure 6 caption L728–L734 (source.md) and
Appendix F Figures 20–21 (source.md L2479–L2486) document the paper's
empirical observation:

    "This routing behavior of SEs ensures that the attention sink token is
     strongly activated at the SEs. ... The sink token subsequently produces
     activation outliers."   — source.md L721, L723

Figures 6 / 20 / 21 are descriptive histograms of expert-router score
distributions for sink vs non-sink tokens, confirming sink-token-dominated
routing as a structural signature of SEs. **The paper does NOT define a
sink-token-based detection criterion** — its Algorithm 1 uses the three-way
AND on residual-stream magnitudes (see ``three_way_and.py``); sink-token
routing is presented as post-hoc analysis, not as a detector.

Official code
-------------
ZunhaiSu/Super-Experts-Profilling @ commit
``573aead3127ae593ba267758b832944f8fed1485`` (2025-09-25) —
github.com/ZunhaiSu/Super-Experts-Profilling.

Inspection of the repo (commit above) confirms the official code implements
the three-way AND criterion only and contains **no sink-token-routing
detector**. The detector implemented here is therefore project-original; the
paper supplies the *signal* (sink-token-dominated routing correlates with
SEs), and this plugin operationalizes it as a candidate-pool contributor.

Deviation: D-sink-token-routing
-------------------------------
Project-original detection criterion derived from the paper's descriptive
observation. During the calibration pass the orchestrator-wired
:class:`CalibrationEngine` hooks each MoE layer's router output + input-ids
and routes per-batch updates into the
:class:`SinkTokenRoutingAccumulator`; this plugin aggregates the resulting
per-(layer, expert) statistics:

    mean_router_score_on_sink_tokens          (``score_sink``)
    mean_router_score_on_normal_tokens        (``score_normal``)
    activation_frequency_on_sink_tokens       (``freq_on_sink``)

Sink tokens are positions where ``input_id == tokenizer.bos_token_id`` ∪
position 0 of each sequence. Per-layer normalization:
``freq_on_sink[(l, e)] = sink_fires[(l, e)] / total_sink_tokens_seen_by_layer[l]``
— per-layer denominator (all layers see identical calibration data, so the
counts coincide; the per-layer contract eliminates the ambiguity that
previously caused a num_layers-fold double-count in legacy code paths).

Candidate-add criterion (all three must hold for ``(l, e)``):

    score_sink / max(score_normal, ε)  >  sink_token_score_ratio        (=10.0)
    freq_on_sink                       >  sink_token_freq_threshold     (=0.99)
    not exceeding                         sink_token_max_per_layer_cap  (=10)
                                          per layer (sorted by score-ratio desc)

Candidates added to the shared ``CandidateBag`` with provenance tag
``"sink_token"``. The final blacklist is gated by the downstream
ablation-filter pass — sink-token candidacy alone is necessary, not
sufficient, for the SE label.

Why the thresholds and the per-layer cap exist (v6 vs v4)
---------------------------------------------------------
v4 ran this detector as an **auto-extension** with thresholds 5.0 / 0.95 and
no cap. Empirically on Qwen3-30B-A3B-2507 (the project's target model),
v4's ``freq_on_sink`` saturated at 1.0 for many non-SE experts — sink-token
routing turns out to be broadly distributed on this architecture rather
than SE-specific. v4 produced a 158-expert blacklist of which 146 came from
this detector; only 1 of those 146 had measurable ablation-pass ΔNLL.

v6 (this code) demotes the detector from auto-extension to
candidate-pool-only and tightens the thresholds:

    score_ratio        5.0   →  10.0   (tightened)
    freq_threshold     0.95  →  0.99   (tightened)
    max_per_layer_cap  ∞     →  10     (new — bounds the candidate set on
                                        architectures where the score+freq
                                        criterion still over-fires)

The per-layer cap selects the top-N by ``score_ratio`` descending. Final
blacklist remains gated by the ablation filter.

Naming-history note
-------------------
The legacy stage-1 monolith called this "Phase C₃" (third sub-source of a
unified Phase C candidate-collection stage). The current plugin
architecture has no phase taxonomy — plugin enable/disable is the flag
that gates execution. Log-string identifiers and the existing test/Trackio
key conventions retain ``"Phase C₃"`` / ``"phase_c"`` strings for dashboard
back-compat; new prose drops the labels.

Git archaeology
---------------
- ``623278e``/``9a2bd0d``: initial introduction as auto-extension at
  ``sink_token_score_ratio=5.0`` and ``sink_token_freq_threshold=0.95``.
- ``65658a0``/``a354f79``: reviewer-recommended config rev; values unchanged
  at 5.0 / 0.95, default flipped to disabled pending Phase F evidence.
- ``a2e34db``/``aa2ed94``: v6 re-enabled tighter — 10.0 / 0.99 with
  ``sink_token_max_per_layer_cap=10`` introduced, detector demoted from
  auto-extension to candidate-only.

Output context slots / artifact contract
----------------------------------------
Three methods (one more than the ``PipelinePlugin`` Protocol mandates):

1. ``setup`` — runs BEFORE the calibration pass. Constructs the
   ``SinkTokenRoutingAccumulator`` (or assigns ``None`` when
   ``sink_token_enabled=False``) and writes the ``sink_acc`` slot. The
   ``CalibrationEngine``'s per-batch handler reads ``sink_acc`` back and
   calls ``sink_acc.update(layer_idx, ids, scores, routed_pos)`` per batch
   per MoE layer, wired via ``HookKind.ROUTER_LOGITS_PER_BATCH +
   HookKind.INPUT_IDS_PER_BATCH``.
2. ``run`` — calls ``apply_sink_token_candidate_selection`` on
   ``sink_acc``'s post-finalize aggregates and adds each returned
   ``(l, e)`` to the shared ``CandidateBag`` with tag ``"sink_token"``.
   Short-circuits when ``sink_acc is None``.
3. ``contribute_artifact`` — returns the four-key ``sink_token`` block of
   ``stage1_blacklist.json``: ``mean_router_score_sink``,
   ``mean_router_score_normal``, ``freq_on_sink``, ``candidates``.

``writes = ("sink_acc", "candidate_bag")`` because ``setup`` writes
``sink_acc`` and ``run`` mutates ``candidate_bag`` in place via
``add(l, e, "sink_token")``. ``provides = ("sink_routing",)`` is the
declarative metadata that the orchestrator translates into the
calibration-pass ``HookSpec``.

Implementation note: the per-batch reduction is vectorized — per-layer
arrays of shape ``(num_experts,)`` populated via
``torch.nn.functional.one_hot`` on the routed indices, replacing an
earlier per-expert Python loop that was the dominant calibration-pass
walltime cost on H200 (~5 sec/batch). The vectorization also fixed a
num_layers-fold double-count in ``freq_on_sink`` (the 2026-05-10 H200
run artifact had max ``freq_on_sink = 1/40 = 0.025`` instead of 1.0,
which left the 0.95 threshold unreachable).
"""

from __future__ import annotations

import logging

from ...pipeline.candidates import CandidateBag
from ...pipeline.safe_json import safe_float
from ...utils.sink_token_routing import (
    SinkTokenRoutingAccumulator,
    apply_sink_token_candidate_selection,
)
from ...pipeline.context import PipelineContext

log = logging.getLogger(__name__)


class SinkTokenDetectorPlugin:
    """Sink-token-routing Super-Expert candidate detector.

    Operationalizes the paper's empirical observation (arXiv:2507.23279
    Figures 6/20/21) — sink-token-dominated routing correlates with Super
    Experts — into a project-original detection criterion (deviation
    D-sink-token-routing). Flags ``(l, e)`` as a candidate when

        score_sink / max(score_normal, ε) > sink_token_score_ratio
        AND
        freq_on_sink > sink_token_freq_threshold
        AND
        not exceeding sink_token_max_per_layer_cap per layer (sorted by
        score-ratio desc).

    See the module docstring for the paper / official-code citations, the
    full deviation rationale, the v6-vs-v4 threshold tightening, and the
    three-method (``setup`` / ``run`` / ``contribute_artifact``) contract.

    The ``candidate_bag`` slot appears in both ``reads`` and ``writes``:
    ``run`` reads the bag instance and mutates it in place via
    ``.add(l, e, "sink_token")``. Final blacklist requires ablation-filter
    evidence — sink-token candidacy is necessary, not sufficient.
    """

    name: str = "sink_token"
    paper: str = (
        "Sink-token routing as a Super-Expert structural signature — "
        "arXiv:2507.23279 (Su et al., 'Unveiling Super Experts in MoE LLMs') "
        "§5.1 Figure 6 + Appendix F Figures 20-21 (descriptive observation); "
        "official code ZunhaiSu/Super-Experts-Profilling @ "
        "573aead3127ae593ba267758b832944f8fed1485 implements no sink-token "
        "detector. Detection criterion is project-original — see deviation "
        "D-sink-token-routing in the module docstring."
    )
    config_key: str = "stage1_grape.super_expert_detection.sink_token_enabled"
    reads: tuple[str, ...] = (
        "moe_layers",
        "tokenizer",
        "config",
        "n_per_layer",
        "sink_acc",         # the accumulator instance (post-finalize).
        "candidate_bag",    # shared write surface.
    )
    writes: tuple[str, ...] = (
        "sink_acc",
        "candidate_bag",    # mutated in place via .add(l, e, "sink_token").
    )
    # The calibration-pass needs the sink-routing accumulator wired
    # (router-logits + input-ids per batch). The orchestrator translates
    # this declarative string into a CalibrationEngine HookSpec declaring
    # HookKind.ROUTER_LOGITS_PER_BATCH + HookKind.INPUT_IDS_PER_BATCH.
    provides: tuple[str, ...] = ("sink_routing",)

    def is_enabled(self, config: dict) -> bool:
        """Read ``config["stage1_grape"]["super_expert_detection"]["sink_token_enabled"]``;
        default ``True``.

        ``False`` does **not** skip the plugin entirely — :meth:`setup`
        still runs and writes ``sink_acc = None`` (matching the legacy
        inline ``else: sink_acc = None``). ``is_enabled`` reflects the
        orchestrator-visible flag for the orchestrator's gating.
        """
        s1 = config.get("stage1_grape", {})
        se = s1.get("super_expert_detection", {})
        return bool(se.get("sink_token_enabled", True))

    def setup(self, ctx: PipelineContext) -> None:
        """Construct the :class:`SinkTokenRoutingAccumulator` for the calibration pass.

        Reads ``moe_layers``, ``tokenizer``, ``config``, ``n_per_layer``
        from ``ctx``; writes ``sink_acc`` back (a
        :class:`SinkTokenRoutingAccumulator` instance, or ``None`` if
        ``sink_token_enabled=False``).

        Called BEFORE the calibration pass. The
        :class:`CalibrationEngine`'s per-batch handler reads ``sink_acc``
        and invokes ``sink_acc.update(layer_idx, ids, scores, routed_pos)``
        per batch per MoE layer. After the calibration pass the
        orchestrator calls ``sink_acc.finalize()``; the plugin's
        :meth:`contribute_artifact` then reads the finalized per-(l, e)
        dicts.

        Not on the ``PipelinePlugin`` Protocol — see sub-task 7 plan §2.4.
        Sub-task 10 introduces a ``SetupCapablePlugin`` Protocol subtype
        (or an opt-in registry method) when more plugins need a
        pre-calibration setup step.
        """
        moe_layers = ctx.get("moe_layers")
        tokenizer = ctx.get("tokenizer")
        config: dict = ctx.get("config")
        n_per_layer: int = ctx.get("n_per_layer")

        s1 = config["stage1_grape"]
        se_cfg = s1.get("super_expert_detection", {})
        sink_token_enabled = bool(se_cfg.get("sink_token_enabled", True))

        sink_acc: SinkTokenRoutingAccumulator | None
        if sink_token_enabled:
            bos_token_id = getattr(tokenizer, "bos_token_id", None)
            sink_acc = SinkTokenRoutingAccumulator(
                num_layers=len(moe_layers),
                num_experts=n_per_layer,
                bos_token_id=int(bos_token_id) if bos_token_id is not None else None,
            )
        else:
            sink_acc = None

        log.info(
            "Stage 1 Phase C₃ (sink-token) setup: enabled=%s, num_layers=%d, "
            "num_experts=%d, bos_token_id=%s.",
            sink_token_enabled,
            len(moe_layers),
            n_per_layer,
            (None if sink_acc is None else sink_acc.bos_token_id),
        )

        ctx.set("sink_acc", sink_acc)

    def run(self, ctx: PipelineContext) -> None:
        """Execute sink-token candidate generation.

        Reads ``sink_acc`` + ``candidate_bag`` + ``config`` from ``ctx``.
        Calls :func:`apply_sink_token_candidate_selection` on
        ``sink_acc``'s three aggregates (post-finalize), then adds each
        returned (l, e) to the shared ``CandidateBag`` with tag
        ``"sink_token"``.

        Short-circuit semantics:
        - ``sink_acc is None`` (setup ran with ``sink_token_enabled=False``):
          no-op; no candidates added.
        - ``sink_token_enabled=False``: setup wrote ``sink_acc=None``; this
          branch short-circuits on the None check (matches legacy
          ``_collect_candidates:991`` ``if sink_enabled and sink_acc is not
          None:`` guard).

        Reads ``sink_acc`` — raises ``KeyError`` if :meth:`setup` was
        skipped. Reads ``candidate_bag`` only when not short-circuiting.
        """
        sink_acc: SinkTokenRoutingAccumulator | None = ctx.get("sink_acc")
        config: dict = ctx.get("config")

        s1 = config["stage1_grape"]
        se_cfg = s1.get("super_expert_detection", {})
        sink_enabled = bool(se_cfg.get("sink_token_enabled", True))
        sink_score_ratio = float(se_cfg.get("sink_token_score_ratio", 10.0))
        sink_freq_threshold = float(se_cfg.get("sink_token_freq_threshold", 0.99))
        sink_max_per_layer_cap = int(se_cfg.get("sink_token_max_per_layer_cap", 10))

        if not sink_enabled or sink_acc is None:
            log.info(
                "Stage 1 Phase C₃ (sink-token) run: short-circuited "
                "(sink_enabled=%s, sink_acc=%s).",
                sink_enabled,
                "<None>" if sink_acc is None else "<finalized>",
            )
            return

        candidate_bag: CandidateBag = ctx.get("candidate_bag")

        sink_pairs = apply_sink_token_candidate_selection(
            sink_acc.mean_router_score_sink,
            sink_acc.mean_router_score_normal,
            sink_acc.freq_on_sink,
            score_ratio=sink_score_ratio,
            freq_threshold=sink_freq_threshold,
            max_per_layer_cap=sink_max_per_layer_cap,
        )
        for (li, e) in sink_pairs:
            candidate_bag.add(int(li), int(e), "sink_token")

        log.info(
            "Stage 1 Phase C₃ (sink-token) run: added %d candidates "
            "(score_ratio=%.3g, freq_threshold=%.3g, cap=%d).",
            len(sink_pairs), sink_score_ratio, sink_freq_threshold,
            sink_max_per_layer_cap,
        )

    def contribute_artifact(self, ctx: PipelineContext) -> dict:
        """Return the four-key ``sink_token`` block of ``stage1_blacklist.json``.

        Identical schema to the legacy inline construction (pre-sub-task-7):

        Returns
        -------
        dict
            Exactly four top-level keys:
              - ``mean_router_score_sink`` : ``{f"L{li}E{e}": float | None}``
                — every ``(layer_idx, expert_idx)`` from
                ``sink_acc.mean_router_score_sink`` with non-finite floats
                scrubbed to ``None``. Empty dict if ``sink_acc is None``.
              - ``mean_router_score_normal`` : same shape, from
                ``sink_acc.mean_router_score_normal``.
              - ``freq_on_sink`` : same shape, from
                ``sink_acc.freq_on_sink``.
              - ``candidates`` : ``{str(li): sorted([expert_idx, ...])}``
                — derived from the unified ``candidates`` dict on the ctx
                (materialised from the shared ``CandidateBag`` via
                ``to_provenance_dict()`` BEFORE the delegation block
                calls this method) by inverting on the ``"sink_token"``
                tag.
        """
        sink_acc: SinkTokenRoutingAccumulator | None = ctx.get("sink_acc")
        candidates: dict[tuple[int, int], list[str]] = ctx.get("candidates")

        # Three score-dicts — byte-identical to the legacy inline literal
        # (pre-sub-task-7).
        if sink_acc is not None:
            mean_router_score_sink = {
                f"L{li}E{e}": safe_float(v)
                for (li, e), v in sink_acc.mean_router_score_sink.items()
            }
            mean_router_score_normal = {
                f"L{li}E{e}": safe_float(v)
                for (li, e), v in sink_acc.mean_router_score_normal.items()
            }
            freq_on_sink = {
                f"L{li}E{e}": safe_float(v)
                for (li, e), v in sink_acc.freq_on_sink.items()
            }
        else:
            mean_router_score_sink = {}
            mean_router_score_normal = {}
            freq_on_sink = {}

        # Invert ``candidates`` on the "sink_token" tag — byte-identical to
        # the legacy inline ``_candidates_by_provenance("sink_token")``
        # (pre-sub-task-7).
        sink_candidates_by_layer: dict[int, list[int]] = {}
        for (li, e), tags in candidates.items():
            if "sink_token" in tags:
                sink_candidates_by_layer.setdefault(int(li), []).append(int(e))

        return {
            "mean_router_score_sink": mean_router_score_sink,
            "mean_router_score_normal": mean_router_score_normal,
            "freq_on_sink": freq_on_sink,
            "candidates": {
                str(li): sorted(es)
                for li, es in sink_candidates_by_layer.items()
            },
        }
