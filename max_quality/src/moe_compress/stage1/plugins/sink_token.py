"""Phase C₃ — Sink-token routing detector.

Paper: 2507.23279 Figures 20-21 (sink-token-dominated routing as a structural
signature of Super Experts). Migrated from the legacy Stage 1 module in
sub-task 7 of the Stage 1 → plugin-architecture refactor.

The plugin owns three responsibilities:

1. **Accumulator construction** — :meth:`setup` builds the
   :class:`~moe_compress.utils.sink_token_routing.SinkTokenRoutingAccumulator`
   (or assigns ``None`` when ``sink_token_enabled=False``) and writes the
   ``sink_acc`` slot on the :class:`Stage1Context`. Called BEFORE Phase B's
   calibration pass so the :class:`CalibrationEngine`'s per-batch handler
   can read the instance for its ``sink_acc.update(...)`` calls — wired via
   the ``ROUTER_LOGITS_PER_BATCH`` + ``INPUT_IDS_PER_BATCH`` hook kinds.
2. **Candidate-pool contribution** — :meth:`run` (rewritten in sub-task 8
   from a no-op) calls :func:`apply_sink_token_candidate_selection` on
   ``sink_acc``'s post-finalize aggregates and adds each returned (l, e)
   to the shared ``CandidateBag`` with tag ``"sink_token"``.
3. **Artifact contribution** — :meth:`contribute_artifact` returns the
   four-key ``sink_token`` block of ``stage1_blacklist.json``:
   ``mean_router_score_sink``, ``mean_router_score_normal``, ``freq_on_sink``,
   ``candidates``. The first three flatten ``sink_acc``'s post-finalize
   per-(layer, expert) dicts into ``f"L{li}E{e}"`` keyed dicts with NaN/Inf
   scrubbed; the fourth inverts the unified ``candidates`` dict on the
   ``"sink_token"`` tag.

The plugin's ``writes`` field is ``("sink_acc", "candidate_bag")`` —
:meth:`setup` writes ``sink_acc``; :meth:`run` mutates the shared
``candidate_bag`` in place via ``add(l, e, "sink_token")``. The
``candidate_bag`` slot appears in both ``reads`` and ``writes`` for that
reason — see `tasks/refactor_stage1/subtask_8_plan.md` §2.4 for the
rationale on extending ``run`` to own the candidate-add step.

The plugin's ``provides`` field is ``("sink_routing",)`` — declarative
metadata advertising that Phase B needs the sink-routing accumulator. The
string label is conceptual in sub-task 7 (no consumer wires it); sub-task
10's orchestrator translates it to a :class:`HookSpec` declaring
``HookKind.ROUTER_LOGITS_PER_BATCH | HookKind.INPUT_IDS_PER_BATCH``.
"""

from __future__ import annotations

import logging

from .._framework.candidates import CandidateBag
from .._framework.safe_json import safe_float
from ...utils.sink_token_routing import (
    SinkTokenRoutingAccumulator,
    apply_sink_token_candidate_selection,
)
from ..context import Stage1Context

log = logging.getLogger(__name__)


class SinkTokenDetectorPlugin:
    """Sink-token routing detector (Phase C₃).

    Identifies experts whose routing is dominated by sink tokens (BOS or
    leading position 0) — a structural signature of Super Experts per
    2507.23279 Figures 20-21. The detector flags an expert if
    ``mean_router_score_on_sink / max(mean_router_score_on_normal, EPS)
    > score_ratio`` AND ``freq_on_sink > freq_threshold``, with a per-layer
    cap on the candidate count (ranked by score-ratio descending).

    Three-method shape:

    - :meth:`setup` runs BEFORE Phase B. Constructs
      :class:`SinkTokenRoutingAccumulator` (or ``None`` if disabled) and
      writes the ``sink_acc`` slot on the context. The
      :class:`CalibrationEngine`'s per-batch handler reads the slot back
      and calls ``sink_acc.update(...)`` per batch.
    - :meth:`run` (rewritten in sub-task 8) calls
      :func:`apply_sink_token_candidate_selection` on ``sink_acc``'s
      post-finalize aggregates and adds each returned (l, e) to the
      shared ``CandidateBag`` with tag ``"sink_token"``. Short-circuits
      when ``sink_acc is None`` (disabled path).
    - :meth:`contribute_artifact` returns the four-key ``sink_token`` block
      of ``stage1_blacklist.json``.

    Scope note: as of sub-task 8 the plugin owns the sink-token
    candidate-add step end-to-end — :meth:`run` mutates the shared
    ``CandidateBag`` rather than leaving the union to a legacy
    ``_collect_candidates``. The ``candidate_bag`` slot
    appears in both ``reads`` and ``writes``: it is read (the bag
    instance) and mutated in place. See sub-task 8 plan §2.4.
    """

    name: str = "sink_token"
    paper: str = "Sink-token routing as a Super-Expert structural signature (arXiv:2507.23279, Figures 20-21)"
    config_key: str = "stage1_grape.super_expert_detection.sink_token_enabled"
    reads: tuple[str, ...] = (
        "moe_layers",
        "tokenizer",
        "config",
        "n_per_layer",
        "sink_acc",         # sub-task 8: the accumulator instance (post-finalize).
        "candidate_bag",    # sub-task 8: shared write surface.
    )
    writes: tuple[str, ...] = (
        "sink_acc",
        "candidate_bag",    # sub-task 8: mutated in place via .add(l, e, "sink_token").
    )
    # Phase B needs the sink-routing accumulator wired (router-logits +
    # input-ids per batch). String label is declarative metadata in sub-task
    # 7; sub-task 10's orchestrator translates it to a CalibrationEngine
    # HookSpec declaring HookKind.ROUTER_LOGITS_PER_BATCH +
    # HookKind.INPUT_IDS_PER_BATCH.
    provides: tuple[str, ...] = ("sink_routing",)

    def is_enabled(self, config: dict) -> bool:
        """Read ``config["stage1_grape"]["super_expert_detection"]["sink_token_enabled"]``;
        default ``True``.

        ``False`` does **not** skip Phase C₃ entirely — the plugin's
        :meth:`setup` still runs and writes ``sink_acc = None`` (matching
        the legacy inline ``else: sink_acc = None``). ``is_enabled``
        reflects the orchestrator-visible flag for the orchestrator's
        gating.
        """
        s1 = config.get("stage1_grape", {})
        se = s1.get("super_expert_detection", {})
        return bool(se.get("sink_token_enabled", True))

    def setup(self, ctx: Stage1Context) -> None:
        """Construct the :class:`SinkTokenRoutingAccumulator` for Phase B.

        Reads ``moe_layers``, ``tokenizer``, ``config``, ``n_per_layer``
        from ``ctx``; writes ``sink_acc`` back (a
        :class:`SinkTokenRoutingAccumulator` instance, or ``None`` if
        ``sink_token_enabled=False``).

        Called BEFORE Phase B's calibration pass. The
        :class:`CalibrationEngine`'s per-batch handler reads ``sink_acc``
        and invokes ``sink_acc.update(layer_idx, ids, scores, routed_pos)``
        per batch per MoE layer. After Phase B the orchestrator calls
        ``sink_acc.finalize()``; the plugin's :meth:`contribute_artifact`
        then reads the finalized per-(l, e) dicts.

        Not on the ``PipelinePlugin`` Protocol — see sub-task 7 plan §2.4.
        Sub-task 10 introduces a ``SetupCapablePlugin`` Protocol subtype
        (or an opt-in registry method) when more plugins need a pre-Phase-B
        setup step.
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
            "Stage 1 Phase C3 (sink-token) setup: enabled=%s, num_layers=%d, "
            "num_experts=%d, bos_token_id=%s.",
            sink_token_enabled,
            len(moe_layers),
            n_per_layer,
            (None if sink_acc is None else sink_acc.bos_token_id),
        )

        ctx.set("sink_acc", sink_acc)

    def run(self, ctx: Stage1Context) -> None:
        """Execute Phase C₃ candidate generation (sub-task 8).

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

    def contribute_artifact(self, ctx: Stage1Context) -> dict:
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
