# HC-SMoE Upstream Bit-by-Bit Alignment Audit

Repo state:
- Upstream: `wazenmai/HC-SMoE` cloned at `/tmp/hcsmoe_align/upstream`
  (depth=1, default branch).
- Ours: `max_quality/src/moe_compress/stage1/plugins/damage_curve_dp.py`
  (554 LoC).
- Branch: `fix/damage-curve-dp-upstream-alignment` off `main`
  (`434eee2`).

Definitions:
- **ALIGN** = our code changes to match upstream exactly.
- **KEEP-DEVIATION-WITH-JUSTIFICATION** = upstream's mechanism cannot be
  matched because our pipeline's API surface / fitness function
  fundamentally differs; documented as a `D-tag` with upstream
  `<file>:<line>` cite + rationale.
- **OPEN-QUESTION** = uncertain whether the deviation is forced or
  optional; flagged for user input.

---

## Headline finding

**HC-SMoE upstream contains NO DP, NO knapsack, NO damage curve, NO
output-MSE additivity machinery.**

Verified by exhaustive grep over the entire `wazenmai/HC-SMoE`
repository:

```
$ grep -rn -i "knapsack\|dynamic.*program\|damage" /tmp/hcsmoe_align/upstream/ --include="*.py" --include="*.md"
(no output)
```

(No knapsack-related symbols. No "damage" symbols. No "dynamic
programming" symbols.)

The **single** upstream mechanism that maps onto our plugin's purpose
(varying per-layer expert counts) is `_assign_num_groups_per_layer` —
a global frequency-threshold rank cut:

- `upstream/hcsmoe/merging/grouping_mixtral.py:142-173`
- `upstream/hcsmoe/merging/grouping_qwen.py:130-176`

These are the **only** lines we can byte-diff our plugin against. Every
other algorithmic surface (DP recurrence, damage-curve cumsum,
marginal-prior derivation, traceback, +inf at-floor convention,
`_PRIOR_EPS` clamp) has **no upstream counterpart**. Those surfaces
remain anchored on R4 (arXiv:2308.10438) which has no public code.

The plugin docstring (lines 21-27, current) already characterises
HC-SMoE's mechanism as:

> HC-SMoE's implementation is crude (global frequency threshold
> determines per-layer counts as a side-effect)

— a paper-text paraphrase, never byte-confirmed against the actual
HC-SMoE source until this audit. This audit's primary job is to
**ground that characterisation** with concrete upstream file:line cites
and decide whether the project's DP-knapsack-on-damage-curve approach
is a legitimate principled departure (it is) or an accidental drift
(it is not).

---

## Item 1 — Per-layer expert count derivation

**Upstream**: `upstream/hcsmoe/merging/grouping_mixtral.py:142-173`,
`upstream/hcsmoe/merging/grouping_qwen.py:130-176`. Reproduced
verbatim (mixtral variant):

```python
def _assign_num_groups_per_layer(
        self,
        num_average_groups: int,
        merging_layers: List[int],
) -> Dict[str, int]:
    num_grouping_layers = len(merging_layers)
    total_num_groups = num_average_groups * num_grouping_layers + self.num_experts * (
            len(self.sparse_layer_indices) - num_grouping_layers
    )
    all_usage_frequency = []
    usage_frequency_dict = deepcopy(self._usage_frequency_state_dict)
    for i, layer_idx in enumerate(self.sparse_layer_indices):
        ffn_name = f"model.layers.{layer_idx}.block_sparse_moe"
        all_usage_frequency.append(usage_frequency_dict[ffn_name])

    all_usage_frequency = torch.cat(all_usage_frequency, dim=0)
    sorted_usage_frequency, sorted_indices = torch.sort(
        all_usage_frequency, descending=True)
    num_groups_per_layer = dict()

    # Note: When threshold is 0.0, the actual number of groups is
    # smaller than total_num_groups.
    if num_average_groups == self.num_experts:
        total_num_groups = total_num_groups - 1
    frequency_threshold = sorted_usage_frequency[total_num_groups]
    print(f"[HC-SMoE] Frequency threshold: {frequency_threshold}")

    for i, layer_idx in enumerate(self.sparse_layer_indices):
        ffn_name = f"model.layers.{layer_idx}.block_sparse_moe"
        num_groups_per_layer[ffn_name] = torch.sum(
            (usage_frequency_dict[ffn_name] >= frequency_threshold).long()
        ).item()

    return num_groups_per_layer
```

Mechanism in plain English:

1. Form a single global vector of expert usage frequencies (token-
   routing histogram), one entry per expert across all layers.
2. Sort descending. Take the `total_num_groups`-th entry as the
   global "frequency threshold".
3. For each layer, count how many of its experts have usage frequency
   `>=` the threshold (mixtral) / `>` the threshold (qwen). That count
   is the per-layer surviving-expert budget.

This is a **global rank cut on per-expert usage frequency**. There is
no DP, no cost-minimisation objective, no damage signal, no per-layer
floor enforcement (a layer can collapse to 0 surviving experts if all
its experts route rarely), no blacklist handling. The mechanism is
crude in exactly the sense the plugin docstring already documents.

The qwen variant adds two refinements: (a) non-merging layers get
their frequencies stamped to `ones_like` so they survive (line
`grouping_qwen.py:146-147`); (b) `frequency_threshold == 1.0` raises
(line `grouping_qwen.py:167-168`).

**Ours**:
`max_quality/src/moe_compress/stage1/plugins/damage_curve_dp.py:255-396`
(the `run()` method). Mechanism in plain English:

1. Build per-layer CKA-off-diagonal cumulative damage curves
   `D_ℓ(k) = Σ_{i=1..k} sort_asc(off_diag(D_matrices[ℓ]))[i]`
   (lines 320-326, helper `_build_damage_curves` lines 415-471).
2. Solve the 1D knapsack DP
   `min Σ_ℓ D_ℓ(k_ℓ)  s.t.  Σ_ℓ k_ℓ = G`
   (lines 347-353, helper `_solve_knapsack_dp` lines 479-554).
3. Publish the marginal damage at the DP optimum as
   `stage1_grape.merge_cost_prior` for GRAPE to consume (lines
   355-396).

Our cost signal is **CKA-distance-based**, not usage-frequency-based.
Our objective is **explicit cost minimisation** under a global
merge-count constraint, not an implicit rank cut. Our output is a
**multiplicative prior into GRAPE**, not a direct per-layer expert
count.

**Verdict**: **KEEP-DEVIATION-WITH-JUSTIFICATION**.

Reasoning:

a) HC-SMoE's mechanism does not solve the same problem we solve.
   HC-SMoE picks per-layer counts as a side-effect of global-frequency
   rank cutting; our plugin picks per-layer counts to minimise an
   additive damage proxy (CKA-based per R4's additivity theorem).
   These are mathematically incomparable mechanisms — the project's
   recipe is a principled refinement of the "vary per-layer counts"
   idea HC-SMoE introduced, not an attempt to clone HC-SMoE's specific
   threshold trick.

b) The project's SC plan (`tasks/SC_STAGE12_COMPREHENSIVE_PLAN.md` §7
   A1) anchors the algorithmic basis on R4's additivity theorem
   (arXiv:2308.10438 Theorem 1) which **has no public code** — it is
   genuinely textbook-DP. The HC-SMoE anchor is an idea-precedent, not
   a code-precedent.

c) Adopting HC-SMoE's mechanism verbatim would require a usage-
   frequency provider (we have `routing_stats_cache.py` but the SC
   plan wires CKA into Stage 1 for cost-of-merge estimation, not
   usage-frequency for survivor selection). Switching cost signals
   would invalidate the plugin's R4-additivity theoretical basis.

The plugin docstring at lines 21-27 ("HC-SMoE's implementation is
crude … the DP-knapsack-on-damage-curve here is a principled
refinement of the same idea") is **correct** and now grounded on
verified upstream code. Action: replace the paper-text-only
paraphrase with a concrete upstream `<file>:<line>` cite + a one-
sentence summary of upstream's actual mechanism.

A new deviation tag `D-no-hcsmoe-knapsack-upstream` will be added to
the docstring and the `paper:` attribute to make this audit
discoverable from the plugin itself.

---

## Item 2 — Cost signal: usage-frequency vs CKA-distance

**Upstream**: usage frequency from token routing. Computed by
`compute_all_usages` (called from `merging-mixtral.py:165` and
`merging-qwen.py:210, 213, 229`). Stored in
`grouping_mixtral.py:_usage_frequency_state_dict` /
`grouping_qwen.py:_usage_frequency_state_dict`. Has no theoretical
"damage" interpretation — it's a routing-traffic proxy.

**Ours**: CKA off-diagonal pair distance from `D_matrices` (produced
by `cka_distance.py`). Plugin reads `D_matrices` at
`damage_curve_dp.py:274` and uses `triu_indices(n, k=1)` extraction
at `:443`.

**Verdict**: **KEEP-DEVIATION-WITH-JUSTIFICATION**.

This is the **D-cka-substitute-for-output-mse** deviation already
documented in the plugin docstring (lines 85-101). The current
docstring frames it as "CKA substituted for output-space MSE because
Stage 2 cost machinery isn't available at Stage 1". That framing is
**correct** but is now also a deviation from HC-SMoE upstream's
usage-frequency cost signal.

The plugin uses CKA for two reasons:

1. R4's per-layer δᵢ is best estimated by an **output-space** cost
   (paper Rec 2). The Stage 2 output-space cost machinery isn't
   available at Stage 1, so CKA — already computed by
   `cka_distance.py` for GRAPE's primary merge primitive — is the
   closest Stage 1-available proxy. Empirically CKA and output-MSE
   share the same "smaller distance ⇒ smaller merge damage" monotone
   ordering at small `k`.

2. HC-SMoE's usage frequency is a **traffic** proxy, not a **damage**
   proxy. A high-traffic expert isn't necessarily expensive to merge
   (it might be merging into a near-clone); a low-traffic expert may
   still be expensive to merge if no near-clone exists. Using usage
   frequency would break R4's additivity theorem's δᵢ interpretation.

Action: extend the docstring's `D-cka-substitute-for-output-mse`
section with a one-sentence note that the alternative — HC-SMoE's
usage-frequency cost — was deliberately rejected because it breaks
R4's δᵢ semantics.

---

## Item 3 — Output: per-layer expert counts vs multiplicative prior into GRAPE

**Upstream**: `num_groups_per_layer: Dict[str, int]` mapped to FFN
module names (`grouping_mixtral.py:159, 167-171`). Consumed directly
at `grouping_mixtral.py:359` (and similar): `num_groups_in_layer =
num_groups_per_layer[ffn_name] if self.dynamic_group else num_groups`.
The per-layer count is the **definitive** survivor budget — clustering
runs at that count for each layer.

**Ours**:
`damage_curve_dp.py:381-384` writes three ctx slots
(`damage_curves`, `dp_optimum`, `merge_cost_prior_computed`) and
mutates `config["stage1_grape"]["merge_cost_prior"]`. GRAPE then
**refines** the DP starting point via its entropy-aware greedy
(documented `D-dp-prior-as-marginal` lines 103-113).

**Verdict**: **KEEP-DEVIATION-WITH-JUSTIFICATION**.

This is the `D-dp-prior-as-marginal` deviation already documented in
the plugin docstring. Re-justification under the new HC-SMoE-grounded
context:

a) HC-SMoE's mechanism has no analog to GRAPE's entropy gate γ;
   HC-SMoE's per-layer count is a hard side-effect of the global
   threshold cut. Our DP is a *biaser* into GRAPE's existing
   entropy-aware greedy. The DP optimum k*_ℓ alone would lose
   GRAPE's γ regulariser.

b) The SC baseline (`SC_FAST_PLAN_V3` SC=0.1293) is a GRAPE-only
   pipeline. Publishing the DP as a multiplicative prior preserves
   GRAPE-only as the disabled-default (byte-identical when
   `stage1_grape.damage_curve_dp.enabled=False`, verified by
   `test_run_disabled_does_nothing`).

c) `D-dp-prior-as-marginal` is the project's intentional "DP is a
   biaser, not a replacement" stance. Adopting upstream HC-SMoE's
   "per-layer count = DP optimum, end of story" would require ripping
   out GRAPE entirely, which is a separate larger redesign decision.

Action: add a sentence to the `D-dp-prior-as-marginal` block noting
that HC-SMoE's direct-per-layer-count output was deliberately not
adopted because it would bypass GRAPE's entropy gate.

---

## Item 4 — Floor / minimum surviving experts

**Upstream**: **No per-layer floor.** Reading
`grouping_mixtral.py:142-173` end-to-end: a layer whose experts all
fall below the global threshold gets `num_groups_per_layer[ffn_name] =
0`. There is no clamp to a minimum; subsequent clustering at
`num_groups=0` would presumably fail catastrophically but upstream
never guards against it explicitly. In the qwen variant
(`grouping_qwen.py:147`) non-merging layers get `usage_frequency_dict
= ones_like` so they always survive — but *merging* layers have no
floor protection.

A `group_limit` knob exists (`grouping_qwen.py:62, 150`) but it's a
*maximum* group size, not a per-layer floor.

**Ours**: explicit floor via `_floor.py:per_layer_floor` (lines
`damage_curve_dp.py:314-318`). The DP's `k_max[li]` is exactly
`N_ℓ − total_floor_ℓ` so the DP can never plan more merges than the
floor allows. Floor is configurable via
`stage1_grape.grape_floor_divisor` (default 2; lines 286-291).

**Verdict**: **KEEP-DEVIATION-WITH-JUSTIFICATION**.

The floor exists because GRAPE downstream **enforces** a floor (per
`grape_merge.py` D5 — see `_floor.py:29-33`); the DP must plan against
the **same** floor GRAPE will enforce, or the DP optimum is infeasible.
Adopting HC-SMoE's no-floor convention would either (a) cause the DP
to over-merge layers GRAPE will then reject (creating a feasibility
mismatch the plugin's `global_merges > sum(k_max)` guard at lines
332-339 already handles), or (b) require eliminating GRAPE's floor
which would shred the SC=0.1293 baseline's robustness.

Action: add a new deviation tag `D-floor-vs-hcsmoe-no-floor` to
explicitly document this departure with an upstream `<file>:<line>`
cite to `grouping_mixtral.py:169-171` (the no-floor count derivation).

---

## Item 5 — Blacklist / immovable-expert handling

**Upstream**: **No blacklist.** Grep confirms: no
"blacklist" / "immovable" / "protected" / "super-expert" symbol exists
in HC-SMoE:

```
$ grep -rn -i "blacklist\|immovable\|protected\|super.expert" /tmp/hcsmoe_align/upstream/ --include="*.py"
(no output)
```

Every expert is mergeable in HC-SMoE.

**Ours**: `damage_curve_dp.py:275` reads `blacklist: dict[int,
list[int]]` from ctx; `_build_damage_curves` excludes any pair
touching a blacklisted expert at lines 444-449. The floor
calculation in `per_layer_floor` accounts for the blacklist
(`_floor.py:85-87`).

**Verdict**: **KEEP-DEVIATION-WITH-JUSTIFICATION**.

Blacklist support exists because the project's Stage 1 pipeline has a
super-expert detector (`ma_detection.py` + `sink_token.py` +
`three_way_and.py` voters, with `ablation_filter.py` aggregator). The
detector flags experts whose ablation triggers catastrophic damage;
those experts must not be touched by the merger. This is a
project-specific protection layer absent from HC-SMoE because
HC-SMoE's task scope doesn't model super-experts.

Action: add a deviation tag `D-blacklist-vs-hcsmoe-no-blacklist`
documenting this with an upstream cite.

---

## Item 6 — Initialization / RNG conventions

**Upstream**: in the per-layer count derivation
(`_assign_num_groups_per_layer`), there is **no RNG use** — the
function is deterministic given the usage frequency dict. RNG enters
HC-SMoE only inside `cluster_experts` (k-means initial-centre
selection at `clustering.py:49 first_center_idx = torch.randint(...)`).

**Ours**: `damage_curve_dp.py:run()` is **also deterministic** given
`D_matrices`, `blacklist`, `per_layer_targets`, and `decomposition`.
No RNG calls. (`np.random` is used only inside `_build_damage_curves`
tests, not the runtime path.)

**Verdict**: **ALIGN** (already aligned by accident — both functions
are deterministic). No action.

---

## Item 7 — Hyperparameters / defaults / ε values

**Upstream**: `group_limit` default 4 (`grouping_mixtral.py:39`,
`grouping_qwen.py:40`); `data_limit` 50000 (mixtral) / 1000000 (qwen);
`start_layer` 0; `dynamic_group` False (NB: the per-layer-count
derivation only fires when `dynamic_group=True`); various `cluster` /
`linkage` strings. **No ε values** in `_assign_num_groups_per_layer`.

**Ours**: `_PRIOR_EPS = 1e-12` (`damage_curve_dp.py:190`); plugin
gated by `stage1_grape.damage_curve_dp.enabled` (default False,
matching the "disabled-byte-identical-default" project convention);
`grape_floor_divisor` default 2 (line 286). No counterpart to
HC-SMoE's `group_limit` / `data_limit` because we don't run clustering
in this plugin.

**Verdict**: **KEEP-DEVIATION-WITH-JUSTIFICATION**.

`_PRIOR_EPS` is the `D-prior-floor-eps` deviation already documented
(lines 116-134). It exists because GRAPE's selection rule is
multiplicative; HC-SMoE's selection rule isn't multiplicative so no
ε analog is needed upstream. Action: cross-reference the new HC-SMoE
upstream cite in the `D-prior-floor-eps` block to make the divergence
trace explicit.

---

## Item 8 — Output schema / config side-channel

**Upstream**: `_assign_num_groups_per_layer` returns `Dict[str, int]`
keyed by `f"model.layers.{layer_idx}.block_sparse_moe"` (mixtral) /
`f"model.layers.{layer_idx}.mlp"` (qwen). The dict is consumed in-
process; nothing is persisted.

**Ours**: writes three ctx slots and mutates
`config["stage1_grape"]["merge_cost_prior"]` (string-keyed
`{str(li): float(prior[li])}` per GRAPE's hook contract at
`grape_merge.py:171-176`). Diagnostics on ctx, no JSON artifact
(`contribute_artifact` returns `{}`).

**Verdict**: **KEEP-DEVIATION-WITH-JUSTIFICATION**.

Output format is driven by the project's pipeline-context plugin
architecture (`PipelineContext` from
`max_quality/src/moe_compress/pipeline/context.py`) which has no
upstream analog. The string-keyed config-side-channel is required by
GRAPE's existing inert hook (`grape_merge.py:171-176`) — changing it
would force a GRAPE-side change for no semantic gain.

No code action; the existing docstring contract section ("Output
context contract", lines 156-167) already documents this exhaustively.

---

## Item 9 — Independent-pairs upper bound

**Upstream**: no analog (no damage curve in upstream).

**Ours**: `D_ℓ(k)` is the cumsum of the k smallest off-diagonal CKA
distances, which treats merged pairs as independent (lines 443-453).
Documented as `D-independent-pairs-assumption` (lines 136-154) — a
strict upper bound on R4's per-layer δᵢ that biases the DP toward
layers with sparser low-distance pair structure.

**Verdict**: **KEEP-DEVIATION-WITH-JUSTIFICATION**.

This deviation is paper-side (against R4, not HC-SMoE) and already
fully documented. No action from this audit. Listed here only for
completeness.

---

## Summary table

| # | Surface | Upstream cite | Our cite | Verdict |
|---|---------|---------------|----------|---------|
| 1 | Per-layer count derivation | `grouping_mixtral.py:142-173`, `grouping_qwen.py:130-176` | `damage_curve_dp.py:255-396` | KEEP-DEVIATION — `D-no-hcsmoe-knapsack-upstream` (new) |
| 2 | Cost signal (freq vs CKA) | `grouping_mixtral.py:152-157` | `damage_curve_dp.py:274, 443-453` | KEEP-DEVIATION — extend `D-cka-substitute-for-output-mse` |
| 3 | Output format (counts vs prior) | `grouping_mixtral.py:169-171` | `damage_curve_dp.py:381-384` | KEEP-DEVIATION — extend `D-dp-prior-as-marginal` |
| 4 | Per-layer floor | (no upstream) `grouping_mixtral.py:142-173` | `damage_curve_dp.py:314-318` + `_floor.py:85-87` | KEEP-DEVIATION — `D-floor-vs-hcsmoe-no-floor` (new) |
| 5 | Blacklist | (no upstream) | `damage_curve_dp.py:275, 444-449` | KEEP-DEVIATION — `D-blacklist-vs-hcsmoe-no-blacklist` (new) |
| 6 | RNG | (deterministic) `grouping_mixtral.py:142-173` | `damage_curve_dp.py:run()` (deterministic) | ALIGN — no action |
| 7 | Hyperparameters / ε | (none in count derivation) | `_PRIOR_EPS = 1e-12` (line 190) | KEEP-DEVIATION — cross-ref `D-prior-floor-eps` |
| 8 | Output schema / side-channel | (in-process Dict) | ctx slots + config mutation | KEEP-DEVIATION — no doc change (already exhaustive) |
| 9 | Independent-pairs UB | (no analog) | `damage_curve_dp.py:443-453` | KEEP-DEVIATION — paper-side; no HC-SMoE-side action |

**Net verdict**: Zero ALIGN-with-code-change items. The plugin's
algorithmic surfaces all legitimately depart from HC-SMoE upstream
because (a) HC-SMoE solves a different sub-problem (rank cut, not
cost minimisation), (b) HC-SMoE has no floor / blacklist analogs that
the project's super-expert protection requires, and (c) the project's
DP is grounded on R4's additivity theorem which HC-SMoE does not
reference.

The audit's deliverable is therefore **documentation-only**: convert
the plugin's existing paper-paraphrase characterisation of HC-SMoE
into upstream-grounded `<file>:<line>` cites + add three new deviation
tags to make the divergences from HC-SMoE upstream discoverable from
the plugin itself.

---

## OPEN-QUESTIONs

**None.** Every surface either aligns with upstream (Item 6) or has a
principled, paper-grounded reason to deviate (Items 1-5, 7-9). No
surfaces are in the "uncertain whether forced or optional" bucket.

---

## Action list (commits in this branch)

1. Replace plugin docstring's paper-only HC-SMoE characterisation
   (lines 21-27) with upstream-grounded cite to
   `grouping_mixtral.py:142-173` + `grouping_qwen.py:130-176`.
2. Add new deviation tag `D-no-hcsmoe-knapsack-upstream` documenting
   the entire algorithmic-shape departure from upstream.
3. Extend `D-cka-substitute-for-output-mse` (Item 2) with the
   usage-frequency-rejection rationale.
4. Extend `D-dp-prior-as-marginal` (Item 3) with the HC-SMoE-direct-
   output rejection rationale.
5. Add new deviation tag `D-floor-vs-hcsmoe-no-floor` (Item 4).
6. Add new deviation tag `D-blacklist-vs-hcsmoe-no-blacklist` (Item 5).
7. Cross-reference HC-SMoE in `D-prior-floor-eps` (Item 7).
8. Update the `paper:` attribute to enumerate the new D-tags so
   `test_plugin_protocol_attributes` covers them.
9. Update tests to assert the new D-tag enumeration in `paper`.
10. Run the full `test_stage1_*` suite to confirm zero behavioural
    regression (the changes are documentation-only — no algorithmic
    change is made).
