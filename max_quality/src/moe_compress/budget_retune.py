"""Damage-aware retune of GRAPE per-layer expert budget against measured Stage-2 merge cost.

Paper
-----
**No paper.** Project-original tool — STRATEGY_NEXT "Direction A". The
baseline Stage 1 GRAPE allocation (see :mod:`stage1.plugins.grape_merge`,
arXiv:2604.06542) uses CKA-redundancy + entropy gate; this retune
re-solves the per-layer budget against the *measured* Stage-2 REAM
merge cost so high-damage layers can keep more experts (and
low-damage layers donate).

Official code
-------------
None — Direction A is project-original. Documented at
``STRATEGY_NEXT.md`` §A in the project root.

Why a re-solve and not a conserved-per-layer-total knapsack
-----------------------------------------------------------
The original Direction A held the total kept-expert count fixed and
shifted single experts between layers that already carried a damage
signal. That proved to be a structural no-op: GRAPE hard-floors every
merged layer at ``N // 2`` and the retune floor was also ``N // 2``,
so no merged layer could donate; meanwhile the layers GRAPE protected
at ``N`` carry no damage signal at all (they had no merges). Direction
A v2 re-solves the **global** budget under the GRAPE entropy gate
biased by per-layer ``mean_cost_per_pair``.

Original module header retained:

Direction A — retune the per-layer expert budget against *measured* Stage-2 damage.

Stage 1 (GRAPE) allocates a non-uniform per-layer expert budget
(``per_layer_target_experts`` in ``stage1_budgets.json``) using CKA-redundancy
plus an entropy gate.  That allocation deliberately does **not** minimise merge
damage: GRAPE is activation-redundancy-aware, not merge-cost-aware.

Stage 2 (``stage2_reap_ream.py``), in contrast, *measures* the actual merge
damage for the budget it was handed.  For every MoE layer it writes a per-layer
record to ``<artifacts>/_stage2_partial/merge_<layer>.json`` whose
``mean_cost_per_pair`` field is the mean REAM assignment cost across the pairs
that were merged in that layer — i.e. the average damage incurred per merge.

This module is a *budget-retune tool*.  Given a completed baseline Stage-2 run,
it reads the measured per-layer damage, then performs a **damage-aware
allocation re-solve** of the per-layer budget.

----------------------------------------------------------------------
Why a re-solve and not a conserved-per-layer-total knapsack
----------------------------------------------------------------------
The original Direction A held the *total* kept-expert count fixed and shifted
single experts between layers that already carried a damage signal.  That
proved to be a structural no-op: GRAPE hard-floors every *merged* layer at
``N // 2`` and the retune floor was also ``N // 2``, so no merged layer could
donate, while the layers GRAPE *protected* at ``N`` carry no damage signal at
all (Stage 2 never merged them) — leaving the knapsack with zero donor freedom.

The redesign (Direction A proposal, option d) replaces the conserved-total
exchange with a *global-budget-conserving* re-allocation:

  * the **global** kept-expert total is conserved (so achieved compression is
    still pinned) — but the *per-layer split* is fully re-solved;
  * the per-layer **floor** is ``max(N_l // floor_divisor, n_protected_l)``
    with a configurable ``floor_divisor`` (default ``2`` — byte-identical to
    GRAPE's ``N // 2`` invariant; ``> 2`` is opt-in and lets donor layers
    exist *below* ``N // 2``);
  * layers GRAPE protected at ``N`` (no measured ``mean_cost_per_pair``) are
    scored with a **redundancy prior** derived from GRAPE's own
    ``per_layer_redundancy`` so they too can be re-allocated — clearly flagged
    as *predicted, not measured* in the output provenance;
  * if the floors make the global budget infeasible the tool **fails loud**
    with :class:`BudgetInfeasibleError` instead of silently shipping an
    under-compressed model.

It writes a NEW ``stage1_budgets.json`` (a different path — the input is never
clobbered) carrying the reallocated ``per_layer_target_experts``.  Stage 2 can
then be re-run pointed at the new file; it already consumes
``per_layer_target_experts`` verbatim (see ``stage2_reap_ream.run``), so no
change to Stage 2's merge logic is required.

----------------------------------------------------------------------
Model-agnostic contract
----------------------------------------------------------------------
NOTHING is hardcoded to any particular model.  Every quantity is derived from
the run's own artifacts:

  * number of MoE layers          -> the set of ``merge_<layer>.json`` files
  * each layer's total experts    -> ``len(freq)`` in that layer's merge JSON
                                     (Stage 2 enforces ``freq`` keys == range(N),
                                     so this is the routed-expert count N_l)
  * each layer's current budget   -> ``per_layer_target_experts`` in the input
                                     ``stage1_budgets.json``
  * per-layer floor               -> ``max(N_l // floor_divisor,
                                     n_protected_l)`` — GRAPE's half-experts
                                     floor (``floor_divisor == 2``) raised so a
                                     layer is never dropped below its count of
                                     protected (blacklisted super-) experts,
                                     read from ``stage1_blacklist.json`` when
                                     present (see :mod:`stage1` package docstring)
  * redundancy prior              -> ``per_layer_redundancy`` in the input
                                     ``stage1_budgets.json`` (GRAPE's R̃^l)

----------------------------------------------------------------------
The marginal-cost damage model
----------------------------------------------------------------------
We have at most ONE scalar of measured damage per layer: ``mean_cost_per_pair``.
We do not have a full damage-vs-budget curve.  The honest, conservative model
used here is:

    predicted_layer_damage(kept) = marginal_cost_l * n_merged_pairs(kept)

where ``n_merged_pairs(kept) = N_l - kept`` is the number of non-centroid
experts folded into a centroid when ``kept`` experts survive.  This treats the
*per-pair* damage as a layer-local constant and the *count* of merges as the
lever.  Under that model the marginal cost of removing one more expert from a
layer is exactly ``marginal_cost_l``, so minimising the total predicted damage
for a fixed global budget is a separable, constant-marginal allocation: start
every layer at its ceiling ``N_l`` and repeatedly remove one expert from the
layer with the smallest ``marginal_cost`` that is still above its floor, until
the global budget is met.  This is the standard exchange argument applied to
the *whole* allocation rather than to conserved-total transfers — and, unlike
the old knapsack, it can move a layer off ``N``.

``marginal_cost_l`` is:

  * the **measured** ``mean_cost_per_pair`` when Stage 2 merged that layer
    (``has_signal`` is true);
  * a **predicted** prior derived from GRAPE's ``per_layer_redundancy`` for
    signal-less layers — more redundant (smaller R̃^l) ⇒ cheaper to merge.
    These layers are flagged ``predicted=True`` so downstream can audit the
    measured-vs-predicted split.

If *no* layer has a usable measured signal the tool refuses to run rather than
inventing a damage model from priors alone.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

# Default filename for the retuned output. Kept distinct from the input
# "stage1_budgets.json" so the input is never clobbered.
DEFAULT_OUTPUT_NAME = "stage1_budgets.retuned.json"

# Stage-2 partial-checkpoint layout — these mirror constants used inside
# stage2_reap_ream.py. They are layout facts of *this pipeline*, not facts
# about any particular model.
_PARTIAL_DIRNAME = "_stage2_partial"
_MERGE_JSON_GLOB = "merge_*.json"

# Stage-1 blacklist artifact. Its ``blacklist`` key maps str(layer) -> list of
# protected (super-)expert ids that Stage 2 never merges. Used to raise the
# per-layer floor. A layout fact of this pipeline, not a model-specific one.
_BLACKLIST_NAME = "stage1_blacklist.json"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class LayerDamage:
    """Per-layer state assembled from the Stage-1 + Stage-2 artifacts."""

    layer_idx: int
    total_experts: int          # N_l — routed experts in the layer (ceiling)
    current_budget: int         # kept-experts assigned by GRAPE (Stage 1)
    floor: int                  # minimum kept-experts: max(N_l//K, n_protected)
    mean_cost_per_pair: float | None  # measured Stage-2 damage; None == no signal
    redundancy: float | None = None   # GRAPE per_layer_redundancy R̃^l prior

    @property
    def has_signal(self) -> bool:
        """True iff Stage 2 measured a usable, strictly-positive per-pair cost."""
        return self.mean_cost_per_pair is not None and self.mean_cost_per_pair > 0.0


@dataclass
class RetuneResult:
    """Output of :func:`retune_budgets`."""

    new_budgets: dict[int, int]                 # layer_idx -> retuned kept-experts
    old_budgets: dict[int, int]                 # layer_idx -> original kept-experts
    total_kept: int                             # conserved GLOBAL budget (sum)
    transfers: int                              # net experts moved vs the input
    predicted_damage_before: float
    predicted_damage_after: float
    floor_divisor: int = 2                      # the N//K floor divisor used
    layers_without_signal: list[int] = field(default_factory=list)
    layers_predicted: list[int] = field(default_factory=list)

    def as_output_payload(self, source_payload: dict) -> dict:
        """Build the JSON payload for the new stage1_budgets.json.

        ``source_payload`` is the parsed input stage1_budgets.json; every key
        other than ``per_layer_target_experts`` / ``achieved_budget`` is copied
        through unchanged so Stage 2 / downstream see a familiar artifact.

        The ``budget_retune`` provenance block records the global-budget
        conservation, the ``floor_divisor`` used, and the measured-vs-predicted
        split (``layers_predicted`` are the signal-less layers whose marginal
        cost came from the redundancy prior, not from a Stage-2 measurement).
        """
        out = dict(source_payload)
        out["per_layer_target_experts"] = {
            str(k): int(v) for k, v in sorted(self.new_budgets.items())
        }
        out["achieved_budget"] = int(self.total_kept)
        out["budget_retune"] = {
            "tool": "moe_compress.budget_retune",
            "transfers_applied": self.transfers,
            "predicted_damage_before": self.predicted_damage_before,
            "predicted_damage_after": self.predicted_damage_after,
            "floor_divisor": int(self.floor_divisor),
            "global_budget_conserved": int(self.total_kept),
            "layers_without_damage_signal": sorted(self.layers_without_signal),
            # layers_cost_predicted == layers_without_damage_signal by
            # construction; their cost is redundancy-prior-derived OR — when
            # the layer carries no redundancy value — a conservative max-cost
            # fallback, so "predicted" here is not always a true prediction.
            "layers_cost_predicted": sorted(self.layers_predicted),
            "n_layers_measured": (
                len(self.new_budgets) - len(self.layers_predicted)
            ),
            "n_layers_predicted": len(self.layers_predicted),
        }
        return out


class NoDamageSignalError(RuntimeError):
    """Raised when the Stage-2 artifacts carry no usable per-layer damage signal."""


class BudgetInfeasibleError(RuntimeError):
    """Raised when the per-layer floors make the global budget unreachable.

    The damage-aware re-solve must place exactly ``global_budget`` kept-experts
    across the layers, each within ``[floor_l, N_l]``.  If the global budget is
    below ``sum(floor_l)`` or above ``sum(N_l)`` no valid allocation exists.
    Rather than silently shipping an under- or over-compressed model the tool
    fails loud — this is the §4 robustness fix from the Direction A proposal.
    """


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------
def _layer_idx_from_merge_path(path: Path) -> int:
    """Extract the integer layer index from a ``merge_<idx>.json`` filename."""
    stem = path.stem  # "merge_12"
    prefix = "merge_"
    if not stem.startswith(prefix):
        raise ValueError(f"unexpected merge-JSON filename: {path.name}")
    return int(stem[len(prefix):])


def load_stage2_damage(artifacts_dir: Path) -> dict[int, tuple[int, float | None]]:
    """Read every Stage-2 per-layer merge record.

    Returns ``{layer_idx: (total_experts, mean_cost_per_pair)}``.

    * ``total_experts`` is ``len(freq)`` — Stage 2 enforces that the ``freq``
      dict has keys exactly ``range(N)``, so its length is the routed-expert
      count for that layer (the per-layer ceiling).
    * ``mean_cost_per_pair`` is the measured damage, or ``None`` when Stage 2
      recorded no merge cost for the layer.

    Raises FileNotFoundError if no Stage-2 partial dir / merge JSONs exist.
    """
    partial_dir = artifacts_dir / _PARTIAL_DIRNAME
    if not partial_dir.is_dir():
        raise FileNotFoundError(
            f"No Stage-2 partial directory at {partial_dir}. "
            "budget_retune needs a completed baseline Stage-2 run; the "
            "per-layer merge JSONs live under <artifacts>/_stage2_partial/. "
            "Note: stage2_reap_ream.run deletes _stage2_partial/ on a fully "
            "successful finish — retune against a run whose partials are still "
            "present, or before the final cleanup."
        )
    merge_files = sorted(partial_dir.glob(_MERGE_JSON_GLOB))
    if not merge_files:
        raise FileNotFoundError(
            f"No merge_*.json files under {partial_dir}. The Stage-2 run "
            "produced no per-layer merge records to retune against."
        )

    out: dict[int, tuple[int, float | None]] = {}
    for mf in merge_files:
        layer_idx = _layer_idx_from_merge_path(mf)
        data = json.loads(mf.read_text(encoding="utf-8"))

        fv = int(data.get("format_version", 0))
        if fv != 2:
            raise ValueError(
                f"{mf} has format_version={fv} (expected 2). budget_retune "
                "only understands the Stage-2 v2 merge-JSON schema."
            )
        if "freq" not in data:
            raise ValueError(
                f"{mf} is missing the 'freq' field — cannot derive the layer's "
                "total expert count. File is corrupt or pre-v2."
            )
        # len(freq) == routed-expert count N_l. Stage 2 guarantees the freq
        # keys are exactly range(N) (it raises otherwise on resume), so the
        # length is an exact, model-agnostic count.
        total_experts = len(data["freq"])
        if total_experts <= 0:
            raise ValueError(f"{mf}: 'freq' is empty — layer has no experts?")

        # mean_cost_per_pair is the verified damage field. It is JSON null
        # when Stage 2 merged nothing (budget >= N_l) or when all pair costs
        # were zero — both mean "no usable damage signal for this layer".
        mcp = data.get("mean_cost_per_pair", None)
        mcp_val: float | None = None if mcp is None else float(mcp)

        out[layer_idx] = (total_experts, mcp_val)

    return out


def load_stage1_budgets(budgets_path: Path) -> dict:
    """Load and lightly validate the input stage1_budgets.json."""
    if not budgets_path.is_file():
        raise FileNotFoundError(f"stage1_budgets.json not found: {budgets_path}")
    payload = json.loads(budgets_path.read_text(encoding="utf-8"))
    if "per_layer_target_experts" not in payload:
        raise ValueError(
            f"{budgets_path} has no 'per_layer_target_experts' key — not a "
            "valid Stage-1 budgets artifact."
        )
    return payload


def load_protected_counts(blacklist_path: Path) -> dict[int, int]:
    """Read per-layer protected-expert counts from stage1_blacklist.json.

    Stage 1 (GRAPE) blacklists *super-experts* — experts whose ablation damage
    is so high they must never be merged. Stage 2 always keeps every protected
    expert, so a layer's true kept-experts floor is ``max(N_l // 2,
    n_protected_l)``, not ``N_l // 2`` alone.

    The artifact's ``blacklist`` key maps ``str(layer_idx) -> [expert_idx,...]``;
    this returns ``{layer_idx: n_protected}``. Layers absent from the blacklist
    have no protected experts (the caller's ``.get(li, 0)`` maps them to 0).

    Fully model-agnostic: the count is whatever Stage 1 recorded for the model
    under compression.
    """
    payload = json.loads(blacklist_path.read_text(encoding="utf-8"))
    if "blacklist" not in payload:
        raise ValueError(
            f"{blacklist_path} has no 'blacklist' key — not a valid Stage-1 "
            "blacklist artifact (stage1_blacklist.json)."
        )
    return {int(k): len(v) for k, v in payload["blacklist"].items()}


def assemble_layers(
    stage1_payload: dict,
    stage2_damage: dict[int, tuple[int, float | None]],
    protected_counts: dict[int, int] | None = None,
    *,
    floor_divisor: int = 2,
) -> list[LayerDamage]:
    """Join Stage-1 budgets with Stage-2 measured damage into LayerDamage rows.

    The layer set must agree between the two artifacts: a mismatch means the
    budgets file and the Stage-2 run describe different models / runs.

    ``protected_counts`` maps ``layer_idx -> n_protected`` (from
    :func:`load_protected_counts`); it raises the per-layer floor so the
    retune can never strand a blacklisted super-expert. When ``None`` the
    floor falls back to ``N_l // floor_divisor`` alone.

    ``floor_divisor`` (default ``2``) sets the per-layer floor as
    ``max(N_l // floor_divisor, n_protected_l)``.  ``2`` reproduces GRAPE's
    ``N // 2`` invariant exactly (byte-identical default path).  A value
    ``> 2`` deliberately lowers the floor below ``N // 2`` so donor layers can
    exist — this is an opt-in, unvalidated quality regime; the caller is
    expected to have logged a loud warning.

    Per-layer ``redundancy`` is taken from the input artifact's optional
    ``per_layer_redundancy`` block (GRAPE's R̃^l, written by Stage 1).  It is
    only consulted as a *prior* for signal-less layers; absence leaves it
    ``None`` and the retune treats those layers conservatively.
    """
    if floor_divisor < 1:
        raise ValueError(
            f"floor_divisor must be >= 1, got {floor_divisor}."
        )
    s1_budgets = {
        int(k): int(v) for k, v in stage1_payload["per_layer_target_experts"].items()
    }
    s1_layers = set(s1_budgets)
    s2_layers = set(stage2_damage)
    if s1_layers != s2_layers:
        only_s1 = sorted(s1_layers - s2_layers)
        only_s2 = sorted(s2_layers - s1_layers)
        raise ValueError(
            "Layer-set mismatch between stage1_budgets.json and the Stage-2 "
            f"merge JSONs. Only in budgets: {only_s1}; only in Stage-2: "
            f"{only_s2}. The two artifacts must come from the same run."
        )

    # Optional GRAPE redundancy prior — used only for signal-less layers.
    raw_red = stage1_payload.get("per_layer_redundancy") or {}
    redundancies: dict[int, float] = {
        int(k): float(v) for k, v in raw_red.items()
    }

    layers: list[LayerDamage] = []
    for li in sorted(s1_layers):
        total_experts, mcp = stage2_damage[li]
        current_budget = s1_budgets[li]
        # Per-layer floor. With floor_divisor == 2 this is GRAPE's
        # half-experts floor (see :mod:`stage1` package docstring); the floor is also
        # raised so a layer is never dropped below its count of protected
        # (blacklisted super-) experts. Without a blacklist artifact
        # n_protected is 0 and this reduces to the N//floor_divisor convention.
        n_protected = (protected_counts or {}).get(li, 0)
        floor = max(total_experts // floor_divisor, n_protected)
        if not (0 < current_budget <= total_experts):
            raise ValueError(
                f"Layer {li}: current budget {current_budget} is outside "
                f"(0, {total_experts}] — stage1_budgets.json is inconsistent "
                "with the Stage-2 expert counts."
            )
        if current_budget < floor:
            # The input already violates the (configured) floor. We surface
            # this rather than silently 'fixing' it, because it means the
            # upstream Stage-1 run used a different floor than the one the
            # retune was configured with.
            raise ValueError(
                f"Layer {li}: current budget {current_budget} is below the "
                f"floor {floor} (= max({total_experts} // {floor_divisor}, "
                f"n_protected={n_protected})). The input budgets file was "
                "produced with a different floor convention; retuning "
                "against it would be unsound."
            )
        layers.append(
            LayerDamage(
                layer_idx=li,
                total_experts=total_experts,
                current_budget=current_budget,
                floor=floor,
                mean_cost_per_pair=mcp,
                redundancy=redundancies.get(li),
            )
        )
    return layers


# ---------------------------------------------------------------------------
# Damage-aware allocation re-solve
# ---------------------------------------------------------------------------
def _predicted_marginal_costs(
    layers: list[LayerDamage],
) -> tuple[dict[int, float], list[int]]:
    """Per-layer marginal merge cost for the re-solve.

    For a layer with a measured Stage-2 signal the marginal cost is the
    measured ``mean_cost_per_pair`` directly.

    For a *signal-less* layer (GRAPE protected it at ``N``, so Stage 2 never
    merged it) there is no measurement.  We fall back to a **prior** derived
    from GRAPE's ``per_layer_redundancy`` R̃^l: a more redundant layer (smaller
    R̃) is cheaper to merge, so the predicted marginal cost rises with R̃.  The
    prior is anchored into the *measured* cost range — ``[min_measured,
    max_measured]`` — so a predicted layer competes on the same scale as the
    measured ones rather than dominating or being dominated by an arbitrary
    constant.  When the measured range is degenerate (single signal layer) the
    prior collapses to that single measured cost.

    A signal-less layer with no redundancy value at all is given the *maximum*
    measured cost — the conservative choice: it is treated as expensive to
    merge, so the re-solve will not drain it without strong evidence.

    Returns ``(marginal_cost_by_layer, predicted_layer_idxs)``.
    """
    measured = [
        ld.mean_cost_per_pair for ld in layers if ld.has_signal
    ]  # all floats, > 0
    lo = min(measured)  # type: ignore[type-var]
    hi = max(measured)  # type: ignore[type-var]
    span = hi - lo

    costs: dict[int, float] = {}
    predicted: list[int] = []
    for ld in layers:
        if ld.has_signal:
            costs[ld.layer_idx] = float(ld.mean_cost_per_pair)  # type: ignore[arg-type]
            continue
        predicted.append(ld.layer_idx)
        if ld.redundancy is None:
            # No prior available — be conservative: treat as the most
            # expensive layer so the re-solve won't drain it unjustified.
            costs[ld.layer_idx] = float(hi)
        else:
            # R̃ ∈ [0, 1] (GRAPE min-max normalises it). Higher R̃ -> diverse
            # experts -> more expensive to merge. Clamp defensively.
            r = min(1.0, max(0.0, float(ld.redundancy)))
            costs[ld.layer_idx] = float(lo + span * r)
    return costs, predicted


def _predicted_total_damage(
    budgets: dict[int, int],
    layers_by_idx: dict[int, LayerDamage],
    marginal_costs: dict[int, float],
) -> float:
    """Sum of predicted damage over all layers under the constant-marginal model.

    predicted_layer_damage = marginal_cost_l * n_merged_pairs
                           = marginal_cost_l * (N_l - kept)

    Every layer contributes — measured layers via their measured per-pair cost,
    signal-less layers via their redundancy-prior cost.
    """
    total = 0.0
    for li, kept in budgets.items():
        ld = layers_by_idx[li]
        n_merged = ld.total_experts - kept
        total += marginal_costs[li] * n_merged
    return total


def retune_budgets(
    layers: list[LayerDamage], *, floor_divisor: int = 2
) -> RetuneResult:
    """Damage-aware re-solve of the per-layer expert budget.

    This replaces the original conserved-per-layer-total knapsack with a
    **global-budget-conserving allocation re-solve** (Direction A proposal,
    option d).

    Algorithm:

      1. The conserved quantity is the **global** kept-expert total
         ``global_budget = sum(current_budget_l)`` — achieved compression is
         still pinned, but the per-layer split is fully re-solved.
      2. Each layer is assigned a constant ``marginal_cost`` — its measured
         ``mean_cost_per_pair`` when Stage 2 merged it, else a redundancy-prior
         estimate (see :func:`_predicted_marginal_costs`).
      3. Feasibility: a valid allocation places ``kept_l ∈ [floor_l, N_l]`` on
         every layer summing to ``global_budget``. If ``global_budget`` is
         outside ``[sum(floor_l), sum(N_l)]`` the floors make the target
         unreachable -> :class:`BudgetInfeasibleError` (fail loud rather than
         ship an under-/over-compressed model).
      4. Greedy minimiser: start every layer at its ceiling ``N_l``; repeatedly
         remove one expert from the layer with the smallest ``marginal_cost``
         that is still above its floor, until ``sum(kept) == global_budget``.
         For the separable constant-marginal model this is the exact minimiser
         of total predicted damage — the standard exchange argument applied to
         the whole allocation (it *can* move a layer off ``N``, unlike the old
         conserved-total knapsack).

    At least one layer must carry a *measured* signal: with no measurement at
    all the redundancy prior has nothing to anchor to, and the tool refuses to
    invent a damage model (:class:`NoDamageSignalError`).

    ``transfers`` is reported as the net number of experts that moved versus
    the input allocation (``sum |new - old| / 2``) — a like-for-like
    "how much did the retune change things" scalar.
    """
    if not layers:
        raise ValueError("retune_budgets: no layers supplied.")

    layers_by_idx = {ld.layer_idx: ld for ld in layers}
    old_budgets: dict[int, int] = {
        ld.layer_idx: ld.current_budget for ld in layers
    }
    global_budget = sum(old_budgets.values())

    signal_layers = [ld for ld in layers if ld.has_signal]
    no_signal = [ld.layer_idx for ld in layers if not ld.has_signal]

    if not signal_layers:
        raise NoDamageSignalError(
            "No layer carries a usable Stage-2 damage signal "
            "(every merge JSON has mean_cost_per_pair == null or <= 0). "
            "There is nothing measured to retune against — aborting rather "
            "than inventing a damage model from priors alone. Check that the "
            "Stage-2 run actually performed merges (budgets below the layer "
            "expert counts) and recorded non-zero pair costs."
        )

    # ---- Feasibility check (the §4 robustness fix) -----------------------
    sum_floor = sum(ld.floor for ld in layers)
    sum_ceil = sum(ld.total_experts for ld in layers)
    if global_budget < sum_floor or global_budget > sum_ceil:
        raise BudgetInfeasibleError(
            f"Global kept-expert budget {global_budget} is outside the "
            f"feasible range [{sum_floor}, {sum_ceil}] implied by the "
            f"per-layer floors (floor_divisor={floor_divisor}) and ceilings. "
            "The floors make the target compression unreachable; the re-solve "
            "cannot place the budget without breaching a floor or ceiling. "
            "Lower the compression ratio, or pass a larger --floor-divisor so "
            "donor layers can drop below N//2 (an opt-in, unvalidated quality "
            "regime)."
        )

    marginal_costs, predicted = _predicted_marginal_costs(layers)
    damage_before = _predicted_total_damage(
        old_budgets, layers_by_idx, marginal_costs
    )

    # ---- Greedy re-solve: start at the ceiling, remove cheapest ----------
    # Start every layer at its ceiling N_l, then remove the cheapest-marginal
    # expert until the global budget is met. Because the marginal cost is a
    # layer-local constant, the cheapest layer should be drained *fully* (down
    # to its floor) before the next-cheapest is touched — that is the exact
    # minimiser for the separable constant-marginal model. Ties on marginal
    # cost are broken by layer index for a deterministic, reproducible result.
    budgets: dict[int, int] = {
        ld.layer_idx: ld.total_experts for ld in layers
    }
    n_to_remove = sum_ceil - global_budget
    order = sorted(
        layers, key=lambda ld: (marginal_costs[ld.layer_idx], ld.layer_idx)
    )
    removed = 0
    for ld in order:
        if removed >= n_to_remove:
            break
        # Drain this (cheapest-remaining) layer as far as its floor allows,
        # capped by how many experts still need removing globally.
        sheddable = min(ld.total_experts - ld.floor, n_to_remove - removed)
        budgets[ld.layer_idx] -= sheddable
        removed += sheddable
    if removed != n_to_remove:
        # Unreachable: feasibility was checked above. Defensive guard only.
        raise RuntimeError(
            "budget_retune re-solve could not place the global budget "
            "despite passing the feasibility check — this indicates a "
            "bug in the re-solve logic."
        )

    damage_after = _predicted_total_damage(
        budgets, layers_by_idx, marginal_costs
    )
    # Net experts moved vs. the input allocation (each move is one out + one
    # in across two layers, hence the /2).
    transfers = sum(
        abs(budgets[li] - old_budgets[li]) for li in budgets
    ) // 2

    # ---- Hard invariants -------------------------------------------------
    assert sum(budgets.values()) == global_budget, (
        f"global kept-experts not conserved: "
        f"{sum(budgets.values())} != {global_budget}"
    )
    for ld in layers:
        b = budgets[ld.layer_idx]
        assert ld.floor <= b <= ld.total_experts, (
            f"layer {ld.layer_idx}: retuned budget {b} breaches "
            f"[floor={ld.floor}, ceiling={ld.total_experts}]"
        )
    assert damage_after <= damage_before + 1e-9, (
        f"predicted damage increased ({damage_before} -> {damage_after}) — "
        "under the constant-marginal damage model the greedy re-solve must be "
        "monotone non-increasing; a non-constant model would invalidate this."
    )

    return RetuneResult(
        new_budgets=budgets,
        old_budgets=old_budgets,
        total_kept=global_budget,
        transfers=transfers,
        predicted_damage_before=damage_before,
        predicted_damage_after=damage_after,
        floor_divisor=floor_divisor,
        layers_without_signal=no_signal,
        layers_predicted=predicted,
    )


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------
def retune_from_artifacts(
    artifacts_dir: str | Path,
    *,
    budgets_path: str | Path | None = None,
    blacklist_path: str | Path | None = None,
    output_path: str | Path | None = None,
    floor_divisor: int = 2,
) -> tuple[RetuneResult, Path]:
    """End-to-end: read artifacts, retune, write a new stage1_budgets.json.

    Args:
        artifacts_dir: directory of a completed baseline Stage-2 run. Must
            contain ``stage1_budgets.json`` and ``_stage2_partial/merge_*.json``.
        budgets_path: override path to the input stage1_budgets.json
            (default: ``<artifacts_dir>/stage1_budgets.json``).
        blacklist_path: override path to ``stage1_blacklist.json``
            (default: ``<artifacts_dir>/stage1_blacklist.json``). Used to raise
            the per-layer floor by each layer's protected-expert count. When the
            file is absent the floor falls back to ``N//floor_divisor`` with a
            warning.
        output_path: where to write the retuned budgets
            (default: ``<artifacts_dir>/stage1_budgets.retuned.json``).
            MUST differ from the input path — the input is never clobbered.
        floor_divisor: per-layer floor is ``max(N//floor_divisor,
            n_protected)``. Default ``2`` reproduces GRAPE's ``N//2`` invariant
            (byte-identical default path). A value ``> 2`` lowers the floor
            below ``N//2`` so donor layers can exist — an opt-in, unvalidated
            quality regime; a loud warning is emitted in that case.

    Returns:
        ``(RetuneResult, written_output_path)``.
    """
    if floor_divisor < 1:
        raise ValueError(f"floor_divisor must be >= 1, got {floor_divisor}.")
    if floor_divisor > 2:
        log.warning(
            "budget_retune: floor_divisor=%d > 2 — the per-layer floor is "
            "BELOW the N//2 spec invariant. Donor layers may drop below half "
            "their experts; this is an unvalidated quality regime. Measure "
            "the quality cost (thermometer bpt_gap) before adopting any sweep "
            "that uses this.",
            floor_divisor,
        )
    elif floor_divisor < 2:
        log.warning(
            "budget_retune: floor_divisor=%d (< 2) makes every layer's floor "
            "equal its ceiling N — the re-solve cannot place any budget below "
            "N and will raise BudgetInfeasibleError for any real compression "
            "target. Almost certainly a misconfiguration.",
            floor_divisor,
        )
    artifacts_dir = Path(artifacts_dir)
    budgets_path = (
        Path(budgets_path) if budgets_path is not None
        else artifacts_dir / "stage1_budgets.json"
    )
    output_path = (
        Path(output_path) if output_path is not None
        else artifacts_dir / DEFAULT_OUTPUT_NAME
    )
    if output_path.resolve() == budgets_path.resolve():
        raise ValueError(
            f"output_path ({output_path}) must differ from the input "
            f"budgets_path ({budgets_path}) — the input must not be clobbered."
        )

    blacklist_path = (
        Path(blacklist_path) if blacklist_path is not None
        else artifacts_dir / _BLACKLIST_NAME
    )
    protected_counts: dict[int, int] | None = None
    if blacklist_path.is_file():
        protected_counts = load_protected_counts(blacklist_path)
    else:
        log.warning(
            "No blacklist artifact at %s — falling back to the N//2 per-layer "
            "floor. For a model that blacklists more than half of a layer's "
            "experts as super-experts this floor is too low; pass an explicit "
            "--blacklist-path to retune such a model soundly.",
            blacklist_path,
        )

    stage1_payload = load_stage1_budgets(budgets_path)
    stage2_damage = load_stage2_damage(artifacts_dir)
    layers = assemble_layers(
        stage1_payload, stage2_damage, protected_counts,
        floor_divisor=floor_divisor,
    )
    result = retune_budgets(layers, floor_divisor=floor_divisor)

    out_payload = result.as_output_payload(stage1_payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp.write_text(json.dumps(out_payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(output_path)

    log.info(
        "budget_retune: %d experts moved; predicted damage %.6g -> %.6g "
        "(global kept-experts %d, conserved); %d layer(s) had no damage "
        "signal (cost predicted from redundancy prior); floor_divisor=%d; "
        "wrote %s",
        result.transfers, result.predicted_damage_before,
        result.predicted_damage_after, result.total_kept,
        len(result.layers_predicted), result.floor_divisor, output_path,
    )
    return result, output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m moe_compress.budget_retune",
        description=(
            "Direction A: retune the per-layer expert budget against the "
            "measured Stage-2 merge damage. Reads a completed baseline "
            "Stage-2 run's artifacts and writes a new stage1_budgets.json "
            "with the kept-experts re-solved toward the layers that merge "
            "most expensively. The GLOBAL kept-expert count is conserved "
            "exactly (achieved compression is pinned); signal-less layers "
            "are scored with GRAPE's redundancy prior."
        ),
    )
    p.add_argument(
        "artifacts_dir",
        help="Directory of a completed baseline Stage-2 run "
             "(contains stage1_budgets.json and _stage2_partial/).",
    )
    p.add_argument(
        "--budgets-path", default=None,
        help="Override the input stage1_budgets.json path "
             "(default: <artifacts_dir>/stage1_budgets.json).",
    )
    p.add_argument(
        "--blacklist-path", default=None,
        help="Override the stage1_blacklist.json path "
             "(default: <artifacts_dir>/stage1_blacklist.json). Raises the "
             "per-layer floor by each layer's protected-expert count; if "
             "absent the floor falls back to N//floor_divisor with a warning.",
    )
    p.add_argument(
        "--floor-divisor", type=int, default=2,
        help="Per-layer floor = max(N // floor-divisor, n_protected). "
             "Default 2 reproduces GRAPE's N//2 invariant exactly. A value "
             ">2 lowers the floor below N//2 so donor layers can exist — an "
             "opt-in, UNVALIDATED quality regime; measure the quality cost "
             "(thermometer bpt_gap) before using it in a sweep.",
    )
    p.add_argument(
        "--output-path", default=None,
        help="Where to write the retuned budgets "
             f"(default: <artifacts_dir>/{DEFAULT_OUTPUT_NAME}). "
             "Must differ from the input path.",
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Print a per-layer before/after budget table.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = _build_parser().parse_args(argv)
    try:
        result, output_path = retune_from_artifacts(
            args.artifacts_dir,
            budgets_path=args.budgets_path,
            blacklist_path=args.blacklist_path,
            output_path=args.output_path,
            floor_divisor=args.floor_divisor,
        )
    except (
        FileNotFoundError, ValueError, NoDamageSignalError,
        BudgetInfeasibleError,
    ) as exc:
        log.error("budget_retune failed: %s", exc)
        return 1

    if args.verbose:
        print(f"\n{'layer':>6} {'old':>5} {'new':>5} {'delta':>6}  cost-source")
        for li in sorted(result.new_budgets):
            old = result.old_budgets[li]
            new = result.new_budgets[li]
            delta = new - old
            src = "predicted" if li in result.layers_predicted else "measured"
            mark = "" if delta == 0 else ("  <-- " + ("+" if delta > 0 else "") + str(delta))
            print(f"{li:>6} {old:>5} {new:>5} {delta:>+6}  {src}{mark}")
        print(
            f"\nglobal kept-experts: {result.total_kept} (conserved)  |  "
            f"experts moved: {result.transfers}  |  "
            f"floor_divisor: {result.floor_divisor}  |  "
            f"layers measured/predicted: "
            f"{len(result.new_budgets) - len(result.layers_predicted)}/"
            f"{len(result.layers_predicted)}  |  "
            f"predicted damage: {result.predicted_damage_before:.6g} -> "
            f"{result.predicted_damage_after:.6g}"
        )
    print(f"wrote retuned budgets -> {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
