"""Direction A — retune the per-layer expert budget against *measured* Stage-2 damage.

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
it reads the measured per-layer damage, then re-solves the per-layer budget
allocation with a knapsack-style greedy transfer:

  * the **total** kept-expert count is held fixed (conserved exactly);
  * kept-experts are shifted away from layers that merge *cheaply* toward
    layers that merge *expensively*;
  * a per-layer **floor** (``total_experts // 2``) and **ceiling**
    (the layer's full expert count) are respected.

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
  * per-layer floor               -> ``max(N_l // 2, n_protected_l)`` — GRAPE's
                                     half-experts floor, raised so a layer is
                                     never dropped below its count of protected
                                     (blacklisted super-) experts, read from
                                     ``stage1_blacklist.json`` when present
                                     (see ALGORITHM_REFERENCE.md §4)

----------------------------------------------------------------------
The marginal-cost damage model
----------------------------------------------------------------------
We only have ONE scalar of measured damage per layer: ``mean_cost_per_pair``.
We do not have a full damage-vs-budget curve.  The honest, conservative model
used here is:

    predicted_layer_damage(kept) = mean_cost_per_pair_l * n_merged_pairs(kept)

where ``n_merged_pairs(kept) = N_l - kept`` is the number of non-centroid
experts folded into a centroid when ``kept`` experts survive.  This treats the
*per-pair* damage as a layer-local constant (the only thing Stage 2 actually
measured) and the *count* of merges as the lever.  Under that model the
marginal cost of removing one more expert from a layer is exactly
``mean_cost_per_pair_l`` and the marginal saving of adding one back is the
same.  The greedy transfer therefore moves a unit of budget whenever the
most-expensive recipient layer's per-pair cost strictly exceeds the
cheapest-marginal donor layer's per-pair cost — a standard exchange argument
that is optimal for this separable, constant-marginal model.

Layers whose ``mean_cost_per_pair`` is ``null`` (Stage 2 writes ``null`` when
nothing was merged, e.g. budget >= N_l, or all-zero pair costs) carry no usable
damage signal: they are pinned at their current budget and excluded from the
transfer on both sides.  If *no* layer has a usable signal the tool refuses to
run rather than inventing one.
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
    floor: int                  # minimum kept-experts: total_experts // 2
    mean_cost_per_pair: float | None  # measured Stage-2 damage; None == no signal

    @property
    def has_signal(self) -> bool:
        """True iff Stage 2 measured a usable, strictly-positive per-pair cost."""
        return self.mean_cost_per_pair is not None and self.mean_cost_per_pair > 0.0


@dataclass
class RetuneResult:
    """Output of :func:`retune_budgets`."""

    new_budgets: dict[int, int]                 # layer_idx -> retuned kept-experts
    old_budgets: dict[int, int]                 # layer_idx -> original kept-experts
    total_kept: int                             # conserved total (sum of either)
    transfers: int                              # number of one-expert moves applied
    predicted_damage_before: float
    predicted_damage_after: float
    layers_without_signal: list[int] = field(default_factory=list)

    def as_output_payload(self, source_payload: dict) -> dict:
        """Build the JSON payload for the new stage1_budgets.json.

        ``source_payload`` is the parsed input stage1_budgets.json; every key
        other than ``per_layer_target_experts`` / ``achieved_budget`` is copied
        through unchanged so Stage 2 / downstream see a familiar artifact.
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
            "layers_without_damage_signal": sorted(self.layers_without_signal),
        }
        return out


class NoDamageSignalError(RuntimeError):
    """Raised when the Stage-2 artifacts carry no usable per-layer damage signal."""


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
) -> list[LayerDamage]:
    """Join Stage-1 budgets with Stage-2 measured damage into LayerDamage rows.

    The layer set must agree between the two artifacts: a mismatch means the
    budgets file and the Stage-2 run describe different models / runs.

    ``protected_counts`` maps ``layer_idx -> n_protected`` (from
    :func:`load_protected_counts`); it raises the per-layer floor so the
    retune can never strand a blacklisted super-expert. When ``None`` the
    floor falls back to ``N_l // 2`` alone.
    """
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

    layers: list[LayerDamage] = []
    for li in sorted(s1_layers):
        total_experts, mcp = stage2_damage[li]
        current_budget = s1_budgets[li]
        # Per-layer floor. GRAPE never drops a layer below half its experts
        # (ALGORITHM_REFERENCE.md §4) AND always keeps every protected
        # (blacklisted super-) expert, so the true floor is the larger of the
        # two. Without a blacklist artifact n_protected is 0 and this reduces
        # to the N//2 convention.
        n_protected = (protected_counts or {}).get(li, 0)
        floor = max(total_experts // 2, n_protected)
        if not (0 < current_budget <= total_experts):
            raise ValueError(
                f"Layer {li}: current budget {current_budget} is outside "
                f"(0, {total_experts}] — stage1_budgets.json is inconsistent "
                "with the Stage-2 expert counts."
            )
        if current_budget < floor:
            # The input already violates the floor. We surface this rather
            # than silently 'fixing' it, because it means the upstream Stage-1
            # run used a different floor than total_experts // 2.
            raise ValueError(
                f"Layer {li}: current budget {current_budget} is below the "
                f"floor {floor} (= max({total_experts} // 2, "
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
            )
        )
    return layers


# ---------------------------------------------------------------------------
# Knapsack reallocation
# ---------------------------------------------------------------------------
def _predicted_total_damage(
    budgets: dict[int, int], layers_by_idx: dict[int, LayerDamage]
) -> float:
    """Sum of predicted damage over all layers under the constant-marginal model.

    predicted_layer_damage = mean_cost_per_pair * n_merged_pairs
                           = mean_cost_per_pair * (N_l - kept)

    Layers without a damage signal contribute 0 (we have nothing to predict).
    """
    total = 0.0
    for li, kept in budgets.items():
        ld = layers_by_idx[li]
        if not ld.has_signal:
            continue
        n_merged = ld.total_experts - kept
        total += ld.mean_cost_per_pair * n_merged  # type: ignore[operator]
    return total


def retune_budgets(layers: list[LayerDamage]) -> RetuneResult:
    """Greedy knapsack reallocation of kept-experts against measured damage.

    Algorithm (cheapest-marginal donor -> most-expensive recipient):

      Repeat:
        * DONOR candidates  = layers with a damage signal that can still give
          up one expert without breaching their floor (kept > floor).
        * RECIPIENT candidates = layers with a damage signal that can still
          take one expert without breaching their ceiling (kept < N_l).
        * Pick the donor with the SMALLEST mean_cost_per_pair and the recipient
          with the LARGEST mean_cost_per_pair.
        * If recipient.cost > donor.cost (strictly), move one expert
          donor -> recipient and repeat. Otherwise stop.

    Each move is damage-neutral-or-better: it removes one merge from the
    recipient (saving ``recipient_cost``) and adds one merge to the donor
    (costing ``donor_cost``); the net change is ``donor_cost - recipient_cost``
    which is strictly negative by the loop condition. The total kept-expert
    count is invariant (one out, one in). Termination is guaranteed: total
    predicted damage strictly decreases each step and is bounded below by 0,
    and the integer lattice of budgets is finite.

    Layers without a usable damage signal are pinned at their current budget.
    """
    if not layers:
        raise ValueError("retune_budgets: no layers supplied.")

    layers_by_idx = {ld.layer_idx: ld for ld in layers}
    budgets: dict[int, int] = {ld.layer_idx: ld.current_budget for ld in layers}
    old_budgets = dict(budgets)
    total_kept = sum(budgets.values())

    signal_layers = [ld for ld in layers if ld.has_signal]
    no_signal = [ld.layer_idx for ld in layers if not ld.has_signal]

    if not signal_layers:
        raise NoDamageSignalError(
            "No layer carries a usable Stage-2 damage signal "
            "(every merge JSON has mean_cost_per_pair == null or <= 0). "
            "There is nothing measured to retune against — aborting rather "
            "than inventing a damage model. Check that the Stage-2 run "
            "actually performed merges (budgets below the layer expert "
            "counts) and recorded non-zero pair costs."
        )

    damage_before = _predicted_total_damage(budgets, layers_by_idx)

    transfers = 0
    # Hard cap on iterations: each transfer strictly reduces an integer-bounded
    # quantity, but cap anyway as a defensive guard against any future bug.
    max_iters = total_kept * len(layers) + len(layers) + 1
    for _ in range(max_iters):
        donors = [
            ld for ld in signal_layers if budgets[ld.layer_idx] > ld.floor
        ]
        recipients = [
            ld for ld in signal_layers if budgets[ld.layer_idx] < ld.total_experts
        ]
        if not donors or not recipients:
            break

        # Cheapest donor: smallest per-pair cost. Most-expensive recipient:
        # largest per-pair cost. mean_cost_per_pair is not None for signal layers.
        donor = min(donors, key=lambda ld: ld.mean_cost_per_pair)  # type: ignore[arg-type,return-value]
        recipient = max(recipients, key=lambda ld: ld.mean_cost_per_pair)  # type: ignore[arg-type,return-value]

        if donor.layer_idx == recipient.layer_idx:
            # Only one signal layer is simultaneously the cheapest donor and
            # most-expensive recipient — no cross-layer transfer possible.
            break

        if recipient.mean_cost_per_pair <= donor.mean_cost_per_pair:  # type: ignore[operator]
            # No strictly-improving transfer remains.
            break

        budgets[donor.layer_idx] -= 1
        budgets[recipient.layer_idx] += 1
        transfers += 1
    else:
        # Loop ran to the cap without converging — should be impossible.
        raise RuntimeError(
            "budget_retune greedy loop hit its iteration cap without "
            "converging — this indicates a bug in the transfer logic."
        )

    damage_after = _predicted_total_damage(budgets, layers_by_idx)

    # ---- Hard invariants -------------------------------------------------
    assert sum(budgets.values()) == total_kept, (
        f"total kept-experts not conserved: {sum(budgets.values())} != {total_kept}"
    )
    for ld in layers:
        b = budgets[ld.layer_idx]
        assert ld.floor <= b <= ld.total_experts, (
            f"layer {ld.layer_idx}: retuned budget {b} breaches "
            f"[floor={ld.floor}, ceiling={ld.total_experts}]"
        )
    for ld in layers:
        if not ld.has_signal:
            assert budgets[ld.layer_idx] == old_budgets[ld.layer_idx], (
                f"layer {ld.layer_idx} has no damage signal but its budget "
                "changed — signal-less layers must be pinned."
            )
    assert damage_after <= damage_before + 1e-9, (
        f"predicted damage increased ({damage_before} -> {damage_after}) — "
        "greedy transfer must be monotone non-increasing."
    )

    return RetuneResult(
        new_budgets=budgets,
        old_budgets=old_budgets,
        total_kept=total_kept,
        transfers=transfers,
        predicted_damage_before=damage_before,
        predicted_damage_after=damage_after,
        layers_without_signal=no_signal,
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
            file is absent the floor falls back to ``N//2`` with a warning.
        output_path: where to write the retuned budgets
            (default: ``<artifacts_dir>/stage1_budgets.retuned.json``).
            MUST differ from the input path — the input is never clobbered.

    Returns:
        ``(RetuneResult, written_output_path)``.
    """
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
    layers = assemble_layers(stage1_payload, stage2_damage, protected_counts)
    result = retune_budgets(layers)

    out_payload = result.as_output_payload(stage1_payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp.write_text(json.dumps(out_payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(output_path)

    log.info(
        "budget_retune: %d transfers; predicted damage %.6g -> %.6g "
        "(total kept-experts %d, conserved); %d layer(s) had no damage signal; "
        "wrote %s",
        result.transfers, result.predicted_damage_before,
        result.predicted_damage_after, result.total_kept,
        len(result.layers_without_signal), output_path,
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
            "with the kept-experts reallocated toward the layers that merge "
            "most expensively. Total kept-expert count is conserved exactly."
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
             "absent the floor falls back to N//2 with a warning.",
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
        )
    except (FileNotFoundError, ValueError, NoDamageSignalError) as exc:
        log.error("budget_retune failed: %s", exc)
        return 1

    if args.verbose:
        print(f"\n{'layer':>6} {'old':>5} {'new':>5} {'delta':>6}  signal")
        for li in sorted(result.new_budgets):
            old = result.old_budgets[li]
            new = result.new_budgets[li]
            delta = new - old
            sig = "no" if li in result.layers_without_signal else "yes"
            mark = "" if delta == 0 else ("  <-- " + ("+" if delta > 0 else "") + str(delta))
            print(f"{li:>6} {old:>5} {new:>5} {delta:>+6}  {sig}{mark}")
        print(
            f"\ntotal kept-experts: {result.total_kept} (conserved)  |  "
            f"transfers: {result.transfers}  |  "
            f"predicted damage: {result.predicted_damage_before:.6g} -> "
            f"{result.predicted_damage_after:.6g}"
        )
    print(f"wrote retuned budgets -> {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
