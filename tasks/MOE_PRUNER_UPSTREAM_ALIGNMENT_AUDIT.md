# MoE-Pruner ↔ `expert_distill.py` upstream alignment audit

## Scope statement (READ FIRST)

This audit cross-references **our** post-merge expert distillation plugin
(`max_quality/src/moe_compress/stage2/plugins/expert_distill.py`) against a
**third-party** implementation of MoE-Pruner (arXiv:2410.12013) — the
`tanganke/fusion_bench` repo (MIT-licensed, 223 stars, active), specifically:

| file | lines | role |
| --- | --- | --- |
| `fusion_bench/method/moe_pruner/moe_pruner.py` | 304 | top-level `MoEPruner` algorithm (calibration + per-layer pruning loop) |
| `fusion_bench/method/moe_pruner/hooks/hook.py` | 23 | `BaseHookFn` ABC |
| `fusion_bench/method/moe_pruner/hooks/mixtral.py` | 93 | Mixtral linear + gate hooks |
| `fusion_bench/method/moe_pruner/hooks/deepseek_v2.py` | 85 | DeepseekV2 linear + gate hooks |
| `fusion_bench/method/moe_pruner/utils/prune.py` | 318 | `prepare_calibration_input`, `WrappedGPT`-style Wanda pruning |
| `fusion_bench/method/moe_pruner/utils/layerwrapper.py` | 61 | `WrappedGPT` scaler-row accumulator |
| `fusion_bench/method/moe_pruner/utils/data.py` | 155 | `get_loaders` (c4 + wikitext2) |
| `fusion_bench/method/moe_pruner/utils/score.py` | 41 | load-balance scoring (unused by the pruner) |
| **total** | **1093** | |

Upstream clone path: `/tmp/moe_pruner_align/upstream/`.

### Authoritative status

- This is **NOT the paper authors' code**. The MoE-Pruner paper (Xie et al.
  arXiv:2410.12013) cites `github.com/yanyue-xie/moe-pruner` which today
  returns **HTTP 404** (verified 2026-05-28).
- `tanganke/fusion_bench` is a faithful re-implementation by Anke Tang
  (`fusion_bench` author, no formal affiliation with the paper's authors).
- "Alignment" here therefore means **matches Anke Tang's published
  implementation**, NOT "matches the paper authors' code". The paper text
  remains the authoritative spec; upstream is a cross-check on a plausible
  implementation.

### Algorithmic scope of our plugin

Our `expert_distill.py` is **the post-prune expert-wise distillation step**
described in MoE-Pruner Eq. 10 — NOT the pruning step. It lives *downstream*
of any pruning/merging pass; in our pipeline it runs in Stage 2 between
`_merge_experts_inplace` and `bank.select`, training the merged centroid
against the freq-weighted-additive forward of pre-merge group members.

**Critical finding (surface coverage)**: `tanganke/fusion_bench`'s
`moe_pruner/` module implements ONLY the Wanda-style pruning loop — it does
**NOT** implement the paper's Eq. 10 expert-wise distillation. A repo-wide
`grep -r distill fusion_bench/method/moe_pruner/` returns zero matches:

```
$ grep -rln "distill" /tmp/moe_pruner_align/upstream/fusion_bench/method/moe_pruner/
(no output)
```

This means our `expert_distill.py` **has no direct counterpart in upstream**.
The byte-by-byte alignment exercise the agent prompt requests cannot be
performed against `tanganke/fusion_bench` for the distillation step — there
is no upstream code path to align against.

What remains in scope:

1. **Citation hygiene** — replace the dead `yanyue-xie/moe-pruner` URL with
   the live `tanganke/fusion_bench` URL, with a *third-party caveat* so
   future readers don't conflate the two.
2. **Cross-checking shared upstream surfaces** (calibration data plumbing,
   layer walk, hook architecture) — but only where our pipeline overlaps,
   i.e. concepts that flow into the distillation target.
3. **Documenting the absence**: confirm in writing that upstream's pruning
   metric, hook capture of `|x|·|r|`, sparsity ratios, and N:M structured
   pruning are **out of scope** for our plugin (they're pre-merge concerns,
   not post-merge distillation).

The remainder of this audit walks the surface list from the agent prompt
and tags each with **IN-SCOPE** (overlap with our plugin) or
**OUT-OF-SCOPE** (upstream's domain, not ours).

---

## Surface-by-surface audit

### 1. Calibration sample collection — OUT-OF-SCOPE

**Upstream** (`moe_pruner.py:101-146`): `_prepare_calibration_data` builds a
c4 dataloader via `get_loaders` (`utils/data.py:138-155`), then catches
inputs to `model.model.layers[0]` via a `Catcher` module
(`utils/prune.py:117-140`), pinning `nsamples=128` activations of shape
`(128, model.seqlen, hidden)` cached to
`outputs/cache/<model>/calibration_data.pkl` (`moe_pruner.py:113-119`).

**Ours**: layer inputs are captured by a separate Stage 1/2 infrastructure
(`_LayerInputAccumulator`, `profile_layer`) that is NOT inside this plugin.
The plugin reads `layer_input_acc.get()` (`expert_distill.py:461-462`); the
sampling mechanism is reservoir-style (`_LayerInputAccumulator`), not
contiguous like upstream.

Verdict: **OUT-OF-SCOPE**. Upstream's calibration path is the pruner's
upstream concern; we are downstream of merging and consume whatever the
profile step published. Not a deviation, a layering boundary.

### 2. Hook architecture — OUT-OF-SCOPE

**Upstream** (`hooks/mixtral.py:10-50`, `hooks/deepseek_v2.py:11-47`): a
forward hook on each expert's linear sublayer that accumulates a per-input-
column scaler `‖inp * routing_weights‖² / nsamples` (the Wanda formula with
the router gate baked in). A separate `Gate` hook intercepts the router
forward, computes `softmax → topk → one_hot`, slices per-expert routing
weights, and stashes them as `_routing_weights` on the linear hook for the
linear hook to read on the next forward.

**Ours**: NO forward hooks on linear sublayers. The distillation target is
built from a **CPU snapshot** of pre-merge expert weights
(`_snapshot_pre_merge_layer_experts`, `expert_distill.py:154-180`) and a
freq-weighted sum re-evaluated via `_swiglu_forward` on captured
`layer_inputs`. The router is NOT consulted inside the plugin — `freq`
arrives via `ctx.get("freq")`, computed elsewhere.

Verdict: **OUT-OF-SCOPE**. Hook capture is the pruner's importance-score
infrastructure; we operate on pre-merge weights + post-profile layer
inputs. Not a deviation.

### 3. Pruning metric `|W| · sqrt(‖x·r‖²)` — OUT-OF-SCOPE

**Upstream** (`hooks/mixtral.py:26-29`, `hooks/deepseek_v2.py:44-47`):
```python
def compute(self):
    return torch.abs(self.linear.weight) * torch.sqrt(
        self.scalar_row.reshape(1, -1)
    )
```
where `scalar_row = ‖inp * routing_weights‖² / nsamples` accumulated over
calibration samples. This is the paper's importance metric — note the
**sqrt** wrapper on the squared-norm accumulator, so the effective per-
column factor is the L2 norm `‖inp · r‖₂` (matching the paper's
"|W| · |x| · |r|" claim at the per-output-neuron level when `|x·r|` is
broadcast across rows).

**Ours**: no pruning metric. We do not score-and-prune; we are the
distillation step *after* merging.

Verdict: **OUT-OF-SCOPE**. This metric defines which weights get zeroed in
upstream's pruning step; our centroid weights are determined by the prior
REAM merge (cosine-clustered, freq-weighted average), not by upstream's
metric. Not a deviation — a different algorithmic stage.

### 4. Per-output-neuron scope — OUT-OF-SCOPE

**Upstream** (`utils/prune_utils.py` via `unstructured_magnitude_prune_`):
the mask is applied row-wise (`W_metric.shape[1]` = input dim per row, sort
along `dim=-1`), confirming the per-output-neuron scope claim in the paper.

**Ours**: not applicable (no pruning).

Verdict: **OUT-OF-SCOPE**.

### 5. Sparsity / N:M structured — OUT-OF-SCOPE

**Upstream** (`moe_pruner.py:254-274`): supports `PruningType.UNSTRUCTURED`
(scalar `sparsity_ratio`) and `PruningType.SEMISTRUCTURED` (`n`, `m`
giving N:M sparsity). Paper-reported 50% sparsity = UNSTRUCTURED 0.5.

**Ours**: no sparsity ratio. Centroid count is determined by `target_experts`
upstream of our plugin (the REAM merge).

Verdict: **OUT-OF-SCOPE**.

### 6. One-shot vs iterative pruning — OUT-OF-SCOPE

**Upstream** (`moe_pruner.py:160-285`): one pass over `model.model.layers`,
each layer pruned in-place once. **One-shot, layer-sequential.**

**Ours**: our distillation runs per merged group (`expert_distill.py:473`)
with up to `expert_distill_steps` AdamW iterations on the centroid. So
**iterative within the distillation step**, but one-shot at the
group/layer level (no second pass over the layer).

Verdict: **OUT-OF-SCOPE** for the pruning aspect; we are post-prune /
post-merge. The iterativity here is *intra-step* gradient descent on Eq.
10's loss, which is exactly what the paper asks for.

### 7. Expert-wise KD post-prune (Eq. 10) — IN-SCOPE / DEVIATION

**Upstream**: **does not implement it.** Verified via repo-wide grep — zero
references to "distill" in `fusion_bench/method/moe_pruner/`. The paper's
99% recovery at 50% sparsity result is presumably reproducible via a
separate fine-tune harness, but `tanganke/fusion_bench` ships only the
pruner.

**Paper** (Eq. 10):
```
L_KD = L_CE + λ · Σ_{j,i} MSE(E_i^{j,teacher}, E_i^{j,student})
```
where `j` indexes layers and `i` indexes experts. Teacher of expert `i` is
the **pretrained unpruned expert `i`** (one-to-one same-index pairing
across the unpruned/pruned banks).

**Ours** (`expert_distill.py:183-332`):
- Loss is **MSE only**, no CE term. (Deviation `D-expert-distill-mse`.)
- Pairing is **per-merge-group additive** rather than per-expert
  same-index: the centroid expert is trained to reproduce the
  freq-weighted *sum* of its pre-merge group members' forward outputs.
  This makes algorithmic sense for our setting — after REAM merge there
  is no "same-index teacher expert"; the centroid was assembled from a
  group of original experts and has no 1:1 counterpart in the unpruned
  bank. The deviation is documented in lines 32-73 (D-expert-distill-mse).
- v1 implementation drops the per-token routing weight `g_e^orig(x)` and
  the TopK gate (D-expert-distill-mse-v1, lines 75-103). Two coupled
  simplifications that the docstring frankly flags will be lifted in v2.

Verdict: **DEVIATION (intentional, documented)**. Our plugin diverges from
the paper's Eq. 10 in pairing (per-merge-group additive vs per-expert
same-index) and in dropping the CE term + per-token routing weight in v1.
Upstream `tanganke/fusion_bench` ships no comparable implementation, so no
"upstream alignment" claim can be made on this surface.

### 8. Forward-pass plumbing (layer walk + cache swap) — OUT-OF-SCOPE

**Upstream** (`moe_pruner.py:160-285`): the classic "inps/outs ping-pong"
where `outs = layer(inps)` then `inps, outs = outs, inps` (lines 277-285)
to walk activations forward without re-running the entire model. Inputs to
the first layer are caught via `Catcher`; subsequent layers consume the
previous layer's outputs.

**Ours**: layer walk happens in `stage2/orchestrator.py` (the Stage 2
driver), not in our plugin. Our plugin receives one layer's worth of
context at a time via `ctx`. No ping-pong inside the plugin.

Verdict: **OUT-OF-SCOPE**.

### 9. Hyperparameters — IN-SCOPE for naming, OUT-OF-SCOPE for values

**Upstream** (`moe_pruner.py:54-73`):
- `nsamples` — calibration sample count (paper uses 128).
- `seed` — RNG seed for c4 sampling.
- `device` — accelerator.
- `prune_type`, `sparsity_ratio`, `n`, `m`, `max_seqlen` — pruning knobs.

**Ours** (`expert_distill.py:375-399`):
- `expert_distill_steps` — AdamW step count.
- `expert_distill_lr`, `expert_distill_betas` — optimizer.
- `expert_distill_token_cap` — token subsample cap (8192 default).
- `expert_distill_skip_singletons` — bool, skip groups of size 1.
- `expert_distill_plateau_steps`, `expert_distill_plateau_eps` —
  early-break gate.

No naming overlap with upstream (upstream has no distill knobs). Values
must be set per the paper's Eq. 10 fine-tune appendix (CE coefficient λ,
distillation epochs, optimizer); the paper's Algorithm 1 / Appendix B
should be consulted for v2 knob values when CE+TopK are added.

Verdict: **OUT-OF-SCOPE** for byte-alignment; **OPEN-QUESTION** for paper
fidelity (see § OPEN-QUESTIONS below).

### 10. Other differences — IN-SCOPE: citation hygiene

The dead URL is the only **IN-SCOPE** divergence between our plugin and
upstream's published-implementation reality.

**Current state** (verified):
- `expert_distill.py:25` — `MoE-Pruner has official code at github.com/yanyue-xie/moe-pruner` — **404 dead link**.
- `expert_distill.py:356` — same dead URL in the `paper` field of `ExpertDistillPlugin`.

**Fix**: replace with `tanganke/fusion_bench` *along with a third-party
caveat* so future readers don't read it as "the paper authors' code".

Verdict: **ALIGN** (citation update only — no algorithmic change).

---

## Summary table

| # | Surface | Status | Action |
| - | - | - | - |
| 1 | Calibration sample collection | OUT-OF-SCOPE | None — upstream of plugin |
| 2 | Hook architecture | OUT-OF-SCOPE | None — pruner-only infra |
| 3 | Pruning metric `\|W\|·\|x·r\|` | OUT-OF-SCOPE | None — pruning not distill |
| 4 | Per-output-neuron scope | OUT-OF-SCOPE | None |
| 5 | Sparsity / N:M | OUT-OF-SCOPE | None |
| 6 | One-shot vs iterative | OUT-OF-SCOPE | None |
| 7 | Expert-wise KD (Eq. 10) | DEVIATION (documented) | None — D-expert-distill-mse + D-expert-distill-mse-v1 already disclose; upstream has no counterpart |
| 8 | Forward-pass plumbing | OUT-OF-SCOPE | None |
| 9 | Hyperparameters | OUT-OF-SCOPE / OPEN-Q | None for now |
| 10 | Citation URLs | **ALIGN** | Update `expert_distill.py:25` + `:356` to cite `tanganke/fusion_bench` with third-party caveat |

---

## OPEN-QUESTIONS

1. **Paper Eq. 10 includes a CE term** (`L_CE + λ · Σ MSE`). Our plugin
   uses **MSE only** — no LM cross-entropy. The CE term in the paper is
   computed on the full model output; including it would require a forward
   pass through the entire pruned model on labeled tokens, which is a
   different infrastructure (it's effectively a fine-tune harness, not a
   per-merge-group local distill). **Question**: do we want a separate
   fine-tune phase that adds the CE term post-stage-2, or is the MSE-only
   per-group distill a deliberately-scoped local refinement?
   Current plugin docstring frames the CE drop as part of
   D-expert-distill-mse but does not give a recommendation. The paper's
   99% recovery result depends on the *combined* L_KD; MSE-only here
   should not be reported as "matching MoE-Pruner".

2. **TopK gate + per-token routing weight (v1 simplifications)** are
   slated for v2 per D-expert-distill-mse-v1 (lines 75-103). Status?
   Module docstring says "Phase 3 v2 will lift both simplifications" —
   not in scope for this alignment branch, but tracked as an open spec
   gap.

3. **Upstream `tanganke/fusion_bench` has no distill code** — should we
   look elsewhere for an Eq. 10 reference implementation? Candidates: the
   paper authors' personal repos (couldn't find one), any third-party
   replication (none located via `grep moe-pruner` on HF/GitHub).
   **Recommendation**: treat the paper text as authoritative and document
   in the citation that no public implementation of Eq. 10 has been
   located.

---

## Decision log for this branch

- **No algorithmic changes** to `_distill_merged_group` or
  `_snapshot_pre_merge_layer_experts` — all surfaces 1-9 either don't
  overlap with upstream or are documented intentional deviations.
- **Citation fix only**: update line 25 and line 356 of
  `expert_distill.py` to cite `tanganke/fusion_bench` (a third-party
  re-implementation) instead of the dead `yanyue-xie/moe-pruner` link,
  with a sentence flagging the third-party status.
- **Tests**: no test changes — the citation update touches docstrings only.

---

## Verification commands

```bash
# Upstream layout
find /tmp/moe_pruner_align/upstream/fusion_bench/method/moe_pruner -name "*.py" | xargs wc -l

# Confirm upstream has zero distill code
grep -rln "distill" /tmp/moe_pruner_align/upstream/fusion_bench/method/moe_pruner/
# (no output)

# Confirm dead/live URLs
curl -sI https://github.com/yanyue-xie/moe-pruner | head -1  # 404
curl -sI https://github.com/tanganke/fusion_bench | head -1  # 200

# Confirm current citation state
grep -n "yanyue-xie\|moe-pruner\|tanganke" \
  max_quality/src/moe_compress/stage2/plugins/expert_distill.py
```
