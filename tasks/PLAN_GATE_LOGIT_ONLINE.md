# PLAN — Online δ_gate accumulation (eliminate the unbounded `gate_logit_profiles` blowup)

Status: PLAN (not implemented). A plan-reviewer reviews this before any code lands.

**Line numbers below are from this plan branch's worktree.** The implementer MUST re-grep for the named symbols rather than trust exact line offsets — the cited offsets drift by ~±15 lines across checkouts; logic is identical. Verified symbol locations (re-grepped at the shared checkout `d06a13f`, 2026-05-31):
- `activation_hooks.py`: `gate_logit_profiles` field declaration L118; docstring bullet L86; `record_router_logits` body append L173; `compute_gate_similarity_matrix` L450 (snapshot L492, cdist L532, dist2sim L536); `clear_layer` pop L612.
- `stage2_profile_writer.py`: `observed` set L392; glp payload build L431-436; `gate_logit_profiles=` kwarg L498; `_CKPT_SCHEMA = 1` L539; checkpoint dump key L564; checkpoint load restore L685-686; `schema_version` build L491.
- These are the authoritative anchors; the §3 prose offsets that disagree defer to this list.

Base for `git apply --check`: **the brief names `ad7125a431e176d4161099480a66f0169609a690`, but that commit is NOT present in this checkout** (`git cat-file -t` → "could not get object info"; `git fetch origin <sha>` → "not our ref"). See §8 — the build sequence flags the base-SHA discrepancy for the reviewer to resolve before merge (likely an upstream-vLLM-tree SHA the patch targets, not a repo SHA).

---

## 0. CRITICAL CORRECTION TO THE TASK BRIEF (raise, don't substitute)

The brief states the raw `gate_logit_profiles` logits "exist only to compute the REAM δ_gate cost (per-token jointly-active expert-pair cosines, folded into `_sim_tensor`)" and instructs folding them into `_sim_tensor [L,E,E]`.

**This premise is factually wrong, verified by reading the code.** Two *distinct* REAM cost terms exist, fed by two *distinct* hooks into two *distinct* accumulators:

| Term | Eq. | Source hook | Accumulator | Statistic |
|---|---|---|---|---|
| **δ̃_expert** | Eq. 8 | `record_gated_output` (gated expert outputs σ·E(x)) | `_sim_tensor [E,E]` fp64 | Σ_t per-token cosine of **gated outputs** over jointly-active token pairs |
| **δ_gate** | Eq. 5 | `record_router_logits` (raw pre-softmax router logits) | `gate_logit_profiles` (the UNBOUNDED list) | per-expert **logit-profile vector** over the token axis → L2-normalize → pairwise Euclidean distance → observed-max dist2sim |

The raw logits do **NOT** feed `_sim_tensor`. They feed `compute_gate_similarity_matrix` (`activation_hooks.py:450-542`), the **sole** code-level consumer (confirmed by `grep`: `ream_cost.py` `sim_gate_full = ream_acc.compute_gate_similarity_matrix(li, all_n_ids)` is the only call; `stage2_profile_cache.py` only re-hydrates the list; the `expert_distill.py` mention is a docstring describing a *hypothetical v2*, not a consumer).

**`_sim_tensor` is already online-accumulated and bounded — it is not the offender and must NOT be touched.** Folding δ_gate into it is mathematically impossible: δ_gate is a different statistic (logit-vector cosine, not gated-output cosine) over a different population (all tokens, not jointly-active pairs).

**Corrected fix direction (same spirit, correct target):** introduce a SECOND bounded `[E,E]` online accumulator — a **router-logit Gram matrix** `G` (the per-expert sum-of-squares is just `diag(G)`) — that captures the complete sufficient statistics δ_gate needs, accumulate it per batch, discard the raw logits each batch, and reconstruct the δ_gate matrix at finalize to within a documented accuracy budget that preserves all merge assignments (§2.4). `_sim_tensor` (Eq. 8) is left entirely alone.

The reconstruction is **numerically faithful, NOT bit-exact** — see §2.4 for the accuracy budget and the empirical merge-invariance proof (C1). The real bar is that δ_gate's *result* — the downstream merge/centroid assignments produced by `ream_cost.py` — is preserved, **not** bit-equality of the sim matrix. δ_gate accumulates `G` and performs the `2−2cos` reconstruction in **fp64**, which keeps the residual sim error ~5 orders of magnitude below the cost gaps that decide assignments. **Carrying the same reconstruction in fp32 is NOT safe**: it triggers catastrophic cancellation in `√(2−2cos)` for near-colinear experts (the reviewer reproduced ~3e-4 sim error at profile-noise 1e-6 with an all-fp32 path). fp64 throughout the Gram + reconstruction is therefore load-bearing, not a nicety — §2.4.

---

## 1. The current δ_gate math (file:line, verbatim semantics)

`ReamCostAccumulator.compute_gate_similarity_matrix` (`max_quality/src/moe_compress/utils/activation_hooks.py:450-542`):

```
492   batches = list(self.gate_logit_profiles.get(layer_idx, ()))      # list[(offset, [T_b, E] fp32)]
501   full = torch.cat([t for _, t in batches], dim=0)                 # [T_total, E]
506   col_idx = torch.tensor(expert_ids, dtype=torch.long)
508   mat = full.index_select(1, col_idx).t().contiguous()            # [n, T_total]  (one row per expert)
516   mat = F.normalize(mat, p=2, dim=1)                               # L2-normalize EACH EXPERT'S ROW over token axis
520   mat = where(isnan(mat), 0, mat)                                 # zero-norm experts → all-zero row
528   if mat.abs().max() < 1e-9: return zeros(n, n)                    # all-zero early exit
532   d = torch.cdist(mat, mat, p=2)                                   # [n, n] pairwise Euclidean dist of NORMALIZED rows
536   sim = 1.0 - d / d.max().clamp(min=1e-12)                         # observed-max dist2sim
537   sim.fill_diagonal_(1.0)
540   sim.clamp_(0.0, 1.0)
```

Reference parity (`ream/ream.py:37-42` per the docstring at `:453-459`): dist2sim uses the **observed** `d.max()` over the full `n×n` matrix, not a fixed constant.

**Exact statistic.** Let `v_e ∈ R^{T_total}` be expert `e`'s logit-profile vector (column `e` of `full`, i.e. the pre-softmax routing logit at every token, INCLUDING tokens where `e` was not top-k — `full` is the dense `[T,E]` router output, every entry populated). Then:

- normalized row `û_e = v_e / ‖v_e‖₂` (zero vector → 0 by the `isnan` guard),
- `d_{ij} = ‖û_i − û_j‖₂`,
- `sim_{ij} = 1 − d_{ij} / max_{p,q} d_{pq}`.

This is **NOT** `cos(mean_i, mean_j)` and **NOT** a per-token jointly-active pair cosine. It is the cosine geometry of the full per-expert logit-profile vectors:

```
d_{ij}² = ‖û_i‖² + ‖û_j‖² − 2 û_i·û_j = 2 − 2·cos(v_i, v_j)        (for non-zero v_i, v_j; ‖û‖=1)
```

so `d_{ij}` is a deterministic function of the **cosine similarity** `c_{ij} = cos(v_i, v_j)` alone (zero-vector edge cases in §2.3).

---

## 2. Equivalence: online Gram form ≈ current batched form (numerically faithful, fp64; merge-invariant — §2.4)

### 2.1 Sufficient statistics
`cos(v_i, v_j) = (v_i · v_j) / (‖v_i‖₂ · ‖v_j‖₂)`. Every term decomposes into **token-additive sums**:

- inner product `v_i · v_j = Σ_t v_i[t]·v_j[t]`  → entry `(i,j)` of the **Gram matrix** `G = Σ_t v[t] v[t]ᵀ` (shape `[E,E]`),
- squared norm `‖v_i‖₂² = Σ_t v_i[t]²` = `G[i,i]` (the diagonal of `G`).

`G` is additive over tokens, hence over batches:
```
G = Σ_{batch b} ( Σ_{t∈b} v_b[t] v_b[t]ᵀ ) = Σ_b G_b ,    G_b = full_bᵀ @ full_b   ([E,E])
```
A bounded `[E,E]` fp64 matrix is a **complete** sufficient statistic for every `c_{ij}` the current code computes — `G` retains every degree of freedom the cosine geometry depends on. **No information δ_gate needs is lost by discarding the raw logits after folding each batch into `G`.** (Information-completeness is exact; the *numerical reconstruction* of `d` from `G` is finite-precision — see §2.4.)

### 2.2 Reconstructing the identical δ_gate matrix from `G`
At finalize, sub-select rows/cols of `G` for the requested `expert_ids` (→ `G_sub [n,n]`):
```
norms = sqrt(diag(G_sub))                              # [n]; ‖v_e‖₂ per requested expert
cos   = G_sub / (norms[:,None] * norms[None,:])        # [n,n]; = û_i·û_j   (guarded for zero norms — §2.3)
d2    = (2 - 2*cos).clamp_min(0)                        # clamp_min guards fp negatives near 0
d     = sqrt(d2)                                        # ≈ cdist of the normalized rows (fp64; see §2.4 for the error bound)
sim   = 1 - d / d.max().clamp_min(1e-12)
sim.fill_diagonal_(1.0); sim.clamp_(0,1)
```
`d` here is the **same closed-form** as `torch.cdist(mat, mat, p=2)` (both equal `√(2−2·cos)` for unit rows), but the two are NOT bit-identical: cdist computes `‖û_i−û_j‖` directly on normalized rows, whereas the Gram path recovers it from `√(2−2·cos)`, which loses precision via catastrophic cancellation when `cos≈1` (near-colinear experts — δ_gate's operating point). **Two divergence sources, not one:** (a) fp summation order in forming `Σ_t v_i[t]v_j[t]` (one `cat`+`normalize`+`cdist` vs. running `+= full_bᵀ@full_b`); (b) the `2−2cos` cancellation. Both are bounded to ~1e-8 sim error by carrying `G` AND the `2−2cos` subtraction in **fp64** (§2.4 quantifies; an all-fp32 reconstruction blows (b) up to ~3e-4). `d.max()`, dist2sim, diagonal-set and final clamp are then identical operations on this `d`.

### 2.3 Edge cases — proven to match the current guards
The faithful replication uses the explicit-norm distance formula so the zero-vector convention matches `where(isnan,0)`:
```
unit  = (norms > 0).double()                                      # [n]; 1 if expert has signal, else 0
nz    = norms > 0
cos   = where(nz[:,None] & nz[None,:], G_sub/(norms_i*norms_j), 0)
d2    = (unit[:,None] + unit[None,:] - 2*cos).clamp_min(0)
d     = sqrt(d2)
```
Cases:
1. **Both non-zero:** `1 + 1 − 2cos = 2 − 2cos` ✓ (matches §2.2).
2. **One zero (i):** `0 + 1 − 0 = 1` → `d=1` ✓ — matches "all-zero û_i row vs unit û_j row" `‖0 − û_j‖ = 1`.
3. **Both zero:** `0 + 0 − 0 = 0` → `d=0` ✓ — matches `‖0 − 0‖ = 0`.
This reproduces the current `F.normalize` + `where(isnan,0)` semantics exactly.
4. **All-zero early exit** (`mat.abs().max() < 1e-9` in the current code; re-grep — the normalize/early-exit block lives in `compute_gate_similarity_matrix`): the normalized matrix is ~zero only when EVERY expert profile is zero-normed (a normalized non-zero row has unit-magnitude entries ≥ 1/√T ≫ 1e-9). The faithful online predicate is `max(norms) < 1e-9` → return `zeros(n,n)`. Confirmed equivalent on the §2.4 / §7 fixture.
5. **`d.max()==0`** (all experts identical direction): `clamp_min(1e-12)` then `1 − 0 = 1` everywhere, then `clamp_(0,1)` — identical in both forms.

**Equivalence verdict: numerically faithful + MERGE-INVARIANT (not bit-exact).** The reconstructed `d` is the same closed form as `cdist(normalize(full),...)` but diverges by ~1e-8 (sim) in fp64 due to (a) Gram summation order and (b) `2−2cos` cancellation at `cos≈1`. §2.4 shows empirically that this error does NOT change any downstream merge/centroid assignment — the load-bearing property. No non-additive step breaks the online reduction: `normalize` and `cdist` are deferred to finalize and applied to the *reconstructed* `d`, which depends on `G` only through additive sums.

### 2.4 Accuracy budget + merge-invariance (C1 — empirical decision)

The reviewer flagged that `d=√(2−2cos)` is numerically dangerous for near-colinear experts (cancellation when `cos≈1`), which IS δ_gate's operating regime. **Resolution: keep the bounded `[E,E]` fp64 Gram. The error is real but stays ~5 orders of magnitude below the cost gaps that decide merges; assignments do not flip.** The decision was made empirically against the ACTUAL downstream consumer (`ream_cost.py:303` → `_ream_cost_matrix` → `argpartition` top-K filter + cost-argmin assignment), not by inspection.

**Root cause of the reviewer's large error (reproduced):** the ~3e-4 (at noise 1e-6) / ~0.99 (at 1e-9) errors arise ONLY when the *entire* reconstruction (`cos` and the `2−2cos` subtraction) is carried in **fp32**. Reproduced exactly:

```
Reviewer-style ALL-FP32 reconstruction (cos+subtraction in fp32):
  eps=1e-06 pair(1,0): ref=0.99999931 fp32recon=0.99965906 err=3.402e-04
  eps=1e-09 pair(3,2): ref=0.99999999 fp32recon=1.00000000 err=5.452e-09
Plan FP64 reconstruction (G and 2-2cos in fp64):
  eps=1e-06 pair(1,0): ref=0.99999931 fp64recon=0.99999931 err=4.626e-10
  eps=1e-09 pair(3,2): ref=0.99999999 fp64recon=1.00000000 err=5.452e-09
```

fp64 collapses the catastrophic case from 3.4e-4 → 4.6e-10. **This is why §3.1 mandates fp64 for BOTH the Gram accumulation AND the `2−2cos` reconstruction — it is load-bearing, not decorative.** (fp32 *Gram accumulation* alone — with fp64 reconstruction — is ~80× worse than fp64 Gram but still ~7e-7; the dominant risk is fp32 in the subtraction, so both must be fp64.)

**Merge-invariance experiment (the real bar).** Realistic fixture: `E=128`, `T=8192`, 5 uneven batches, fp32 logits (as they arrive from the model), with seven injected near-colinear pairs spanning `eps ∈ {1e-2 … 1e-9}` (incl. the reviewer's 1e-6 and 1e-9 points) plus a tight 4-expert cluster. Both paths (reference cdist-on-normalized vs fp64 Gram) feed the actual `ream_cost.py` cost build (`cost = 1 − (sim_gate + sim_expert)/2`), the `np.argpartition` top-K candidate filter, and the `pre`-path cost-argmin assignment, over a realistic 25%-centroid split:

```
=== sim matrix accuracy (delta_gate, full 128x128) ===
max abs err : 3.4455374109398917e-09
near-colinear pairs: eps=1e-06 err=3.391e-10 ... eps=1e-09 err=0.000e+00
=== downstream topk=48/8/4/1 (all identical) ===
  cost max abs err          : 1.723e-09
  top-K candidate-set flips : 0 / 96 rows
  pre-path argmin flips     : 0 / 96 rows
```

**Adversarial stress** (`sim_expert` held constant so `sim_gate` is the SOLE cost ranker; an exact-duplicate expert `cos==1`, plus a dense 11-member near-colinear cluster all competing for one centroid):

```
max abs sim err (adversarial): 1.82e-08
cost max abs err: 9.10e-09
top-K(4) candidate-set flips: 0 / 52
argmin assignment flips: 0 / 52
  nc=12 (exact duplicate): ref->c0 gram->c0  (min cost ref=0.25000000 gram=0.25000001)
  [all 11 cluster members: ref->c0 gram->c0]
```

**Verdict: merges do NOT flip.** Worst-case fp64 sim error ~1.8e-8 ≪ the inter-candidate cost gaps; top-K candidate sets and final argmin assignments are byte-identical to the reference on both the realistic and adversarial fixtures. The bounded `[E,E]` Gram is therefore **sufficient** — no redesign (mean-centering / capped-exact fallback) is needed. Reproducible scripts committed alongside this plan: `tasks/c1_experiment.py` (realistic merge-invariance), `tasks/c1_reviewer.py` (reproduces the reviewer's fp32 ~3e-4 error + fp64 fix), `tasks/c1_adversarial.py` (exact-duplicate / dense-cluster stress); run under `torch 2.11.0+cu130`.

**Accuracy budget (documented contract):**
- Per-entry sim error vs reference: ≤ ~2e-8 (fp64 Gram + fp64 reconstruction), even for exact-duplicate / near-colinear experts.
- Merge/centroid assignment: **invariant** (0 flips) — this is the acceptance bar, not bit-equality.
- fp64 is mandatory for both accumulation and reconstruction; fp32 anywhere in the `2−2cos` path is rejected (would breach the budget).

---

## 3. Exact edits — BOTH surfaces kept logic-byte-equivalent

New bounded state on `ReamCostAccumulator` (replaces `gate_logit_profiles`):
```python
# Per-layer router-logit Gram accumulator G[i,j] = Σ_t v_i[t]·v_j[t], [E,E] fp64 CPU.
# diag(G) = Σ_t v_e[t]² = ‖v_e‖₂². Bounded: E²·8 = 512 KB/layer at E=256
# (vs. the ~570-920 GB unbounded raw-logit list it replaces).
_gate_gram: dict[int, torch.Tensor] = field(default_factory=dict)
```

### 3.1 `max_quality/src/moe_compress/utils/activation_hooks.py` (canonical)

**(a) `record_router_logits` (≈151-173)** — fold-and-discard. New body:
```python
def record_router_logits(self, layer_idx, logits, batch_offset):
    # batch_offset kept in signature for API compat; no longer used —
    # G accumulation is order-independent, needs no global token index.
    x = logits.detach().to(torch.float64)          # [T_b, E]; fp64 MANDATORY (§2.4 — fp32 here breaches the accuracy budget)
    gram_b = (x.transpose(0, 1) @ x).cpu()          # [E, E] = Σ_{t∈b} v[t]v[t]ᵀ
    with self._lock:
        g = self._gate_gram.get(layer_idx)
        if g is None:
            self._gate_gram[layer_idx] = gram_b
        else:
            g.add_(gram_b)
    # raw logits go out of scope → freed each batch (the whole point)
```
Compute `gram_b` on the logits' device (GPU) then one `[E,E]` CPU transfer/batch (512 KB) — also a wire-size reduction vs. the prior full `[T_b,E]` CPU transfer.

**(b) `compute_gate_similarity_matrix` (450-542)** — read `_gate_gram` instead of concatenating `gate_logit_profiles`. Replace the snapshot+cat+normalize+cdist block (≈491-532) with the §2.2/§2.3 reconstruction:
```python
with self._lock:
    g = self._gate_gram.get(layer_idx)
if g is None:
    return torch.zeros(n, n, dtype=torch.float32)
col = torch.tensor(expert_ids, dtype=torch.long)
try:
    G_sub = g.index_select(0, col).index_select(1, col).to(torch.float64)   # [n,n]
except IndexError as exc:
    raise IndexError(... f"layer {layer_idx} gram with {g.shape[0]} rows ...") from exc
norms = G_sub.diagonal().clamp_min(0).sqrt()                                # [n]
nz = norms > 0
if not bool(nz.any()) or float(norms.max()) < 1e-9:                        # all-zero early exit (§2.3.4)
    return torch.zeros(n, n, dtype=torch.float32)
unit  = nz.to(torch.float64)
denom = (norms[:, None] * norms[None, :]).clamp_min(1e-300)
cos   = torch.where(nz[:, None] & nz[None, :], G_sub / denom, torch.zeros_like(G_sub))
d     = (unit[:, None] + unit[None, :] - 2.0 * cos).clamp_min(0.0).sqrt()   # fp64; 2−2cos cancellation bounded by fp64 (§2.4)
sim   = 1.0 - d / d.max().clamp_min(1e-12)
sim.fill_diagonal_(1.0)
sim.clamp_(0.0, 1.0)
return sim.to(torch.float32)
```
Keep the `n==0 → zeros(0,0)` guard (≈482-484) and the IndexError-wrapping contract (now against `g.shape[0]`). **All intermediates (`G_sub`, `cos`, `d`) stay fp64 until the final `.to(torch.float32)` — see §2.4; an fp32 `2−2cos` subtraction would breach the accuracy budget.** The output downcast to fp32 matches the current return dtype and is post-`d.max()`-normalization, so it does not re-introduce cancellation.

**(c) `clear_layer` (≈609-613)** — replace `self.gate_logit_profiles.pop(layer_idx, None)` with `self._gate_gram.pop(layer_idx, None)`.

**(d) Remove** the `gate_logit_profiles` field (≈118-120) and rewrite the class docstring bullet (≈85-99) to describe `_gate_gram`.

### 3.2 `max_quality/src/moe_compress/calibration/stage2_profile_writer.py` (canonical writer)
- **`dump_stage2_profile`**: build a `gate_gram` payload tensor `[n_layers, E, E]` fp64 by mirroring the `sim_tensor` loop (≈415-422, reading `_state.ream_acc._gate_gram.get(layer_idx)`); REMOVE the `gate_logit_profiles` dict build (≈431-436) and the `gate_logit_profiles=` kwarg (≈498). Update the `observed` set (≈392) `set(_state.ream_acc.gate_logit_profiles)` → `set(_state.ream_acc._gate_gram)`.
- **`dump_stage2_profile_checkpoint`**: replace `"ream_acc_gate_logit_profiles": dict(...)` (≈564) with `"ream_acc_gate_gram": dict(_state.ream_acc._gate_gram)`.
- **`load_stage2_profile_checkpoint`**: replace the glp restore (≈685-686) with `_state.ream_acc._gate_gram = dict(loaded.get("ream_acc_gate_gram", {}))`. Bump `_CKPT_SCHEMA` 1 → 2 (≈539): the checkpoint carries a renamed key, so an old checkpoint must fail-loud and regenerate — the existing schema-mismatch `ValueError` path (≈606-609) already does this.

### 3.3 `max_quality/src/moe_compress/utils/cached_calibration_signals.py` (schema)
- `Stage2ProfilePayloadV3` → **`Stage2ProfilePayloadV4`** (schema bump). Remove the `gate_logit_profiles: dict` field; add `gate_gram: torch.Tensor  # [n_layers, E, E] fp64`. Update docstring (drop the "Bug #2 raw per-batch list" bullet, document the Gram).
- `SCHEMA_VERSIONS["stage2_profile"]` **3 → 4** (≈123) with a comment noting the v3→v4 break (raw logits replaced by Gram).
- `save_stage2_profile_v3` → `save_stage2_profile_v4`: remove the glp CPU-move loop and serialize `gate_gram` (one contiguous fp64 tensor, like `sim_tensor`).
- `load_stage2_profile_v3` → `load_stage2_profile_v4`: drop glp handling, surface `gate_gram`. The `_check_schema("stage2_profile", ...)` reads `SCHEMA_VERSIONS` (≈1122) so it picks up 4 automatically; confirm the dataclass isinstance check targets V4.
- Update `__all__` and the importer rename (`stage2_profile_writer.py` import; the vLLM patch import block).

### 3.4 `max_quality/src/moe_compress/stage2/plugins/stage2_profile_cache.py` (hydration consumer)
- Drop the `gate_logit_profiles` hydration (the `sidecar_batches`/`ream_acc.gate_logit_profiles[layer_idx] = list(...)` block); instead `ream_acc._gate_gram[layer_idx] = payload.gate_gram[layer_rank].clone()` (mirror the `_sim_tensor` hydration). Update the module docstring bullet.

### 3.4b `max_quality/src/moe_compress/calibration/__init__.py` (N1 — docstring only)
The package docstring names `Stage2ProfilePayloadV3` and `save_stage2_profile_v3` as the shared symbols re-exported from `cached_calibration_signals` (~L12-13). After §3.3 renames these to `V4`/`save_stage2_profile_v4`, update the docstring text accordingly. Docstring-only; no code/`__all__` change in this file (the actual re-exports, if any, are resolved by §3.3's `__all__` edit).

### 3.5 vLLM patch twin — `max_quality/patches/vllm_calibration_stage2_profile.patch`
The patch is a **single-hunk new-file add** (`--- /dev/null` … `@@ -0,0 +1,N @@`). Its body is a **verbatim copy** of `stage2_profile_writer.py` plus a thin vLLM callback-dispatch shim (`_router_handler`, `register_callback("router", _router_handler)`).

**(H2) STEP 0 — materialize the working copy first.** `vllm/calibration_stage2_profile.py` does NOT exist as a tracked working file in this repo (confirmed: `ls vllm/calibration_stage2_profile.py` → No such file; the directory `vllm/` does not exist). It lives ONLY as the `+`-body inside the patch. The §3.5 edits + hunk-regen presuppose a working file to edit, so the implementer MUST first extract it from the patch body:
```bash
mkdir -p vllm
# strip the leading '+' from each body line (skip the diff/header lines) to reconstruct the file:
git apply --include='vllm/calibration_stage2_profile.py' max_quality/patches/vllm_calibration_stage2_profile.patch
#   (applies the single new-file hunk into ./vllm/calibration_stage2_profile.py)
# OR, if applying in-place is undesirable, sed the '^+' bodylines out of the @@ hunk.
ls -l vllm/calibration_stage2_profile.py   # MUST exist before proceeding
```
This working copy is a build-time scratch artifact (not committed — `vllm/` is the patch target tree, not repo source); it exists only so §3.5's "edit the canonical working copy then `git diff --no-index`" regen step has a file to operate on. Then apply the SAME logic edits as §3.2 to that working copy:
- `_router_handler` calls `record_router_logits` — call site unchanged (the accumulator now folds internally).
- Remove the patch's `gate_logit_profiles` payload build, the `gate_logit_profiles=` kwarg, the checkpoint dump key, and the checkpoint load; add the `gate_gram` equivalents identical to §3.2.
- Update the import block for the `Stage2ProfilePayloadV4` / `save_stage2_profile_v4` rename.
- **Recompute the hunk header:** the file is created whole, so the only header is `@@ -0,0 +1,N @@` with `N` = the new `+`-line count of the body. **Regenerate, do not hand-count:** edit the canonical `vllm/calibration_stage2_profile.py` working copy, then `git diff --no-index /dev/null vllm/calibration_stage2_profile.py` (or `git apply -R` + re-add) to emit the corrected header.
- `MANIFEST.md` (H3) — the patch is regenerated, so MORE than the prose changes. Update ALL of:
  - The "Schema bump" prose line (currently reads `bumped from 1 to 3` — note: **fix the existing typo too, it should be `1 to 4`**, not `3 to 4`; `SCHEMA_VERSIONS["stage2_profile"]` was already at 3 in this branch's writer, but the MANIFEST prose still says "1 to 3", so the corrected end state is "bumped from 1 to 4").
  - **`Patch 2 MD5`** (currently `fefbcec8b4f230317bdb16be808eecc8`, the `| Patch 2 MD5 |` table row): recompute via `md5sum max_quality/patches/vllm_calibration_stage2_profile.patch` AFTER regen and paste the new hash.
  - **`Patch 2 line count`** (currently `812`, the `| Patch 2 line count |` table row): recompute via `wc -l` after regen.
  - The **`## Verifying locally`** self-check block: the `# expect: fefbcec8…` comment under the `md5sum …stage2_profile.patch` line, and the `# expect: 812` comment under its `wc -l` line — both must match the regenerated values.
  - Patch 1 (`vllm_calibration_hooks.patch`) is NOT touched by this plan; its MD5/line-count/expect values stay as-is.

**Twin invariant:** after edits the patch `+`-body MUST remain a verbatim superset of `stage2_profile_writer.py` (writer logic + vLLM shim only). The reviewer verifies by diffing the patch `+`-body against the canonical writer (modulo the shim functions). Preserves the standing "canonical ↔ patch byte-equivalent in logic" contract.

---

## 4. Resumability
- `_sim_tensor` (Eq. 8): **already** checkpointed (`stage2_profile_writer.py` dump/restore) and sidecar-persisted — untouched by this plan, survives resume as before.
- `_gate_gram` (NEW, δ_gate): the running accumulator that replaces the raw list. MUST be persisted for resume — both the checkpoint (§3.2 `"ream_acc_gate_gram"`) and the sidecar (§3.3 `gate_gram` field) carry it.
- **No separate count needed.** `G` encodes both the off-diagonal numerator and the per-expert sum-of-squares (diagonal); δ_gate's dist2sim is scale-invariant in the cosine, so no token count enters the δ_gate path (unlike Eq. 8, whose `_total_tokens_by_layer` denominator is already persisted). Confirmed: `compute_gate_similarity_matrix` (450-542) never references `_total_tokens_by_layer`. Persist only `_gate_gram`.

### 4.1 (M1) Stored Gram is the FULL `[n_layers, E_total, E_total]` — subset is a compute-time view
The persisted/accumulated `_gate_gram[layer]` is the **full** `[E_total, E_total]` Gram over ALL experts in the layer (every column of the dense router-logit output, populated for every token incl. non-top-k). It is **NOT** a candidate/centroid subset and is **NOT** pre-restricted to non-protected (non-SE) experts. The subsetting happens entirely at compute time, AFTER rehydration, inside `compute_gate_similarity_matrix`:
- `ream_cost.py:303` passes `all_n_ids` (the COMPLETE non-protected expert list for the layer) as `expert_ids`;
- the function does `G_sub = G.index_select(0, col).index_select(1, col)` to view only those rows/cols;
- `d.max()` is then taken over that `[n,n]` non-protected sub-population — matching the current code's spec invariant (the docstring at `compute_gate_similarity_matrix`: "`expert_ids` MUST contain ALL non-protected experts … `D.max()` must be over the full non-protected population N, not just a nc+c subset").
Storing the full `[E_total,E_total]` Gram (not a pre-subset) is what makes this possible: the SE/protected-expert exclusion and the `d.max()`-over-non-protected-population are applied at compute time from the rehydrated full matrix, never baked into storage. (Memory note: `E_total²·8` bytes — e.g. 512 KB/layer at `E_total=256` — already accounts for the full population; §6.)

---

## 5. Test plan
1. **NEW `test_gate_gram_online_equals_batched`** (new `max_quality/tests/test_gate_gram_equivalence.py`): synthetic `[T,E]` logit fixture split into ≥3 uneven batches; assert the online path (`record_router_logits` per batch → `compute_gate_similarity_matrix`) matches a **reference** doing the OLD math directly (`cat` → `F.normalize` → `cdist` → dist2sim) on the full tensor.
   - **Tolerance (HONEST, per §2.4 budget): `torch.allclose(online, ref, atol=2e-7, rtol=0)`.** Do NOT use 1e-9 — that is bit-equality framing and is FALSE for near-colinear experts (the `2−2cos` reconstruction loses precision exactly where δ_gate operates; observed fp64 error ~2e-8, with margin to 2e-7). The acceptance bar is numerical faithfulness within budget, not bit-equality.
   - **REQUIRED fixture coverage:** the fixture MUST include at least one **near-colinear expert pair** (e.g. `v_j = v_i + eps`, `eps≈1e-6` on fp32 logits) — this is the catastrophic-cancellation case §2.4 identified; a fixture of only well-separated experts would vacuously pass and hide a regression to fp32. Also parametrize the three §2.3 edge cases: one zero-norm expert, all-zero matrix, all-identical experts (`d.max()==0`).
   - **NEW `test_gate_gram_merge_invariant`** (same file): the §2.4 merge-invariance guard, condensed — build the near-colinear fixture, compute `sim` both ways, feed both through the `ream_cost.py` cost build + `np.argpartition` top-K filter + cost-argmin, and assert the candidate sets AND argmin assignments are identical. This regression-guards the actual acceptance bar (merge-invariance), not just the sim tolerance. Also add a `test_gate_gram_rejects_fp32` (or an xfail-documenting comment) showing an all-fp32 reconstruction breaches `atol=2e-7` — locks in WHY fp64 is mandatory.
2. **Update `test_stage2_profile_sidecar_writer_math.py`**: replace `test_gate_logit_profiles_preserved_verbatim` and its glp assertions with a `gate_gram`-shape/round-trip assertion. The Bug #1 sim_tensor tests (`test_writer_sim_tensor_matches_reference`, `test_writer_sim_tensor_distinguishes_from_buggy_path`) are UNCHANGED (Eq. 8 untouched) and MUST still pass — they regression-guard that `_sim_tensor` was not disturbed.
3. **Update `test_stage2_profile_sidecar_roundtrip.py`**: drop `gate_logit_profiles` build/assert; add `gate_gram` round-trip (`torch.equal(loaded.gate_gram, original.gate_gram)`, dtype fp64).
4. **Update `test_stage2_profile_sidecar_partial_hit.py`** and **`test_cached_calibration_signals.py`**: swap glp payload construction/asserts for `gate_gram`.
5. **`test_stage2_assignment_v2.py`** — **NO CHANGE NEEDED (L1).** Re-grepped (2026-05-31): this test defines its OWN stub `compute_gate_similarity_matrix(self, layer_idx, expert_ids)` on a fake accumulator (one `def` at ~L675); it never touches `gate_logit_profiles`, `_gate_gram`, `record_router_logits`, or any `Stage2ProfilePayloadV*`. It is decoupled from this change and must continue to pass as-is — keep it in the §8 regression set as the integration guard that the public δ_gate API contract is unchanged.
6. Any test pinning `schema_version==3` → 4; any `Stage2ProfilePayloadV3` reference → V4.
7. **(H1) Two ADDITIONAL payload-constructing tests that hard-break on the V3→V4 rename** — both pass `Stage2ProfilePayloadV3(..., gate_logit_profiles=...)` and will fail to import/construct after §3.3:
   - `test_stage2_profile_layer_in_hook.py`: `_build_payload_with_empty_reservoir` (~L425-448) imports `Stage2ProfilePayloadV3` (~L39) and constructs it with `gate_logit_profiles=gate_logit_profiles` (~L429, L448), consumed by ≥3 tests (~L467, L496, L522). Rename the import → `Stage2ProfilePayloadV4`, drop the glp dict build, pass `gate_gram=` (a small `[n_layers,E,E]` fp64 tensor; an empty/zeros tensor is fine for these reservoir-focused tests).
   - `test_stage2_profile_sidecar_cov_storage_dtype.py`: `_build_payload(cov_storage_dtype)` (~L31-43) imports `Stage2ProfilePayloadV3` (~L20) and constructs it with `gate_logit_profiles={0: []}` (~L43). Rename → V4, swap the glp kwarg for `gate_gram=`.
   These were omitted from the original §5 list; with §3.3 renaming the dataclass they are NOT optional — they break the suite at collection time.

---

## 6. Memory accounting (why this fixes the blowup)
- Before: `gate_logit_profiles` = append-only `[T_b, 256] fp32` per batch, concatenated → ~24M tokens × 256 × 4 B held across 40 layers in the vLLM single-pass path ≈ **570-920 GB host RAM** + serialized to sidecar/checkpoint.
- After: `_gate_gram` = one `[256,256] fp64` per layer = **512 KB/layer**, **~20 MB across 40 layers** — bounded, constant in token count. Sidecar/checkpoint shrink by the same factor.

---

## 7. Equivalence fixture detail (for the reviewer to run)
The load-bearing proof is the §2.4 merge-invariance experiment (assignments do not flip); the §5.1 numerical-equivalence test is its unit-level guard. Minimal reference (pseudocode, fp64):
```python
def ref_delta_gate(full, ids):       # full: [T,E] fp64
    mat = F.normalize(full[:, ids].t(), p=2, dim=1)      # [n,T]
    mat = torch.where(mat.isnan(), torch.zeros_like(mat), mat)
    if mat.abs().max() < 1e-9: return torch.zeros(len(ids), len(ids))
    d = torch.cdist(mat, mat)
    sim = 1 - d / d.max().clamp_min(1e-12); sim.fill_diagonal_(1.); return sim.clamp(0,1)
```
Assert `ref_delta_gate(full, ids) ≈ online_from_gram(full split into batches, ids)`.

---

## 8. Build sequence + verification
1. Branch from current HEAD (this plan branch's base). **RAISE for reviewer: the brief's base SHA `ad7125a` is absent from this repo** — confirm the intended base before the implementer runs `git apply --check`. If `ad7125a` is the upstream-vLLM SHA the patch targets, `git apply --check max_quality/patches/vllm_calibration_stage2_profile.patch` must run in a *vLLM* clone at that SHA, not this repo.
2. Implement §3.1-§3.4b (canonical), run the touched-module suite (incl. the H1 payload tests and the new gram tests):
   `pytest max_quality/tests/test_stage2_profile_sidecar_writer_math.py max_quality/tests/test_stage2_profile_sidecar_roundtrip.py max_quality/tests/test_stage2_profile_sidecar_partial_hit.py max_quality/tests/test_cached_calibration_signals.py max_quality/tests/test_stage2_profile_layer_in_hook.py max_quality/tests/test_stage2_profile_sidecar_cov_storage_dtype.py max_quality/tests/test_stage2_assignment_v2.py max_quality/tests/test_gate_gram_equivalence.py -q`
3. **(H2)** Materialize the patch working copy FIRST (§3.5 STEP 0): `git apply --include='vllm/calibration_stage2_profile.py' max_quality/patches/vllm_calibration_stage2_profile.patch` → confirm `vllm/calibration_stage2_profile.py` exists.
4. Apply the §3.5 logic edits to that working copy, then regenerate the patch: `git diff --no-index /dev/null vllm/calibration_stage2_profile.py > max_quality/patches/vllm_calibration_stage2_profile.patch` (auto-correct hunk header), then `git apply --check` against the vLLM tree at its pinned SHA.
5. Diff the patch `+`-body against `stage2_profile_writer.py` to confirm the twin invariant (§3.5).
6. **(H3)** Update `MANIFEST.md`: recompute `Patch 2 MD5` (`md5sum`) and `Patch 2 line count` (`wc -l`) on the regenerated patch, update the matching `# expect:` self-check comments, and fix the schema-bump prose to "bumped from 1 to 4".

---

## 9. Files touched (summary)
- `max_quality/src/moe_compress/utils/activation_hooks.py` — `_gate_gram` field, `record_router_logits`, `compute_gate_similarity_matrix`, `clear_layer`, docstring.
- `max_quality/src/moe_compress/calibration/stage2_profile_writer.py` — dump/checkpoint/load glp→gram; `_CKPT_SCHEMA` 1→2.
- `max_quality/src/moe_compress/utils/cached_calibration_signals.py` — `Stage2ProfilePayloadV4`, `SCHEMA_VERSIONS` 3→4, `save_/load_stage2_profile_v4`, `__all__`.
- `max_quality/src/moe_compress/stage2/plugins/stage2_profile_cache.py` — hydrate `_gate_gram`.
- `max_quality/src/moe_compress/calibration/__init__.py` — (N1) docstring V3→V4 only.
- `max_quality/patches/vllm_calibration_stage2_profile.patch` + `max_quality/patches/MANIFEST.md` — twin edits + hunk-header regen + (H3) MD5/line-count/expect regen + schema note.
- Tests: `test_stage2_profile_sidecar_writer_math.py`, `_roundtrip.py`, `_partial_hit.py`, `test_cached_calibration_signals.py`, + (H1) `test_stage2_profile_layer_in_hook.py`, `test_stage2_profile_sidecar_cov_storage_dtype.py`, + new `test_gate_gram_equivalence.py` (equivalence + merge-invariance + fp32-rejection). **`test_stage2_assignment_v2.py` — NO change (L1), kept in the regression set only.**

NOT touched: `_sim_tensor` / Eq. 8 / `record_gated_output` / `finalize_batch` / `compute_delta_expert` — the brief's stated target was wrong; those are already bounded and must stay byte-identical.
