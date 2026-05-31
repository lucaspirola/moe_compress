# PLAN — Online δ_gate accumulation (eliminate the unbounded `gate_logit_profiles` blowup)

Status: PLAN (not implemented). A plan-reviewer reviews this before any code lands.

**Line numbers below are from this plan branch's worktree (HEAD `463cb1b`).** The implementer should re-grep for the named symbols rather than trust exact line offsets (the shared checkout `d06a13f` is offset by ~±15 lines in the writer; logic is identical).

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

**Corrected fix direction (same spirit, correct target):** introduce a SECOND bounded `[E,E]` online accumulator — a **router-logit Gram matrix** `G` (the per-expert sum-of-squares is just `diag(G)`) — that captures *exactly* the sufficient statistics δ_gate needs, accumulate it per batch, discard the raw logits each batch, and reconstruct the identical δ_gate matrix at finalize. `_sim_tensor` (Eq. 8) is left entirely alone.

The equivalence below is **EXACT** (modulo fp summation order — the same ~5e-4 fp32 drift the codebase already tolerates everywhere, see `finalize_batch` docstring "up to ~5e-4 relative drift"). δ_gate accumulates in **fp64**, so its drift is far below even that.

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

## 2. Equivalence proof: online Gram form == current batched form (EXACT, fp64)

### 2.1 Sufficient statistics
`cos(v_i, v_j) = (v_i · v_j) / (‖v_i‖₂ · ‖v_j‖₂)`. Every term decomposes into **token-additive sums**:

- inner product `v_i · v_j = Σ_t v_i[t]·v_j[t]`  → entry `(i,j)` of the **Gram matrix** `G = Σ_t v[t] v[t]ᵀ` (shape `[E,E]`),
- squared norm `‖v_i‖₂² = Σ_t v_i[t]²` = `G[i,i]` (the diagonal of `G`).

`G` is additive over tokens, hence over batches:
```
G = Σ_{batch b} ( Σ_{t∈b} v_b[t] v_b[t]ᵀ ) = Σ_b G_b ,    G_b = full_bᵀ @ full_b   ([E,E])
```
A bounded `[E,E]` fp64 matrix is the EXACT sufficient statistic for every `c_{ij}` the current code computes. **No information δ_gate needs is lost by discarding the raw logits after folding each batch into `G`.**

### 2.2 Reconstructing the identical δ_gate matrix from `G`
At finalize, sub-select rows/cols of `G` for the requested `expert_ids` (→ `G_sub [n,n]`):
```
norms = sqrt(diag(G_sub))                              # [n]; ‖v_e‖₂ per requested expert
cos   = G_sub / (norms[:,None] * norms[None,:])        # [n,n]; = û_i·û_j   (guarded for zero norms — §2.3)
d2    = (2 - 2*cos).clamp_min(0)                        # clamp_min guards fp negatives near 0
d     = sqrt(d2)                                        # == cdist of the normalized rows, EXACTLY
sim   = 1 - d / d.max().clamp_min(1e-12)
sim.fill_diagonal_(1.0); sim.clamp_(0,1)
```
`d` here equals `torch.cdist(mat, mat, p=2)` from the current code **element-for-element** (both equal `√(2−2·cos)` for unit rows). `d.max()`, the dist2sim, the diagonal-set and the final clamp are then byte-identical operations on byte-identical `d`. **The only numerical difference is fp summation order** in forming `Σ_t v_i[t]v_j[t]` (one big `cat`+`normalize`+`cdist` vs. a running `+= full_bᵀ@full_b`). Accumulating `G` in **fp64** drives this ≪ the 5e-4 fp32 drift the codebase already accepts.

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
4. **All-zero early exit** (`activation_hooks.py:528`, `mat.abs().max() < 1e-9`): the normalized matrix is ~zero only when EVERY expert profile is zero-normed (a normalized non-zero row has unit-magnitude entries ≥ 1/√T ≫ 1e-9). The faithful online predicate is `max(norms) < 1e-9` → return `zeros(n,n)`. The reviewer confirms the threshold form on the §7 fixture.
5. **`d.max()==0`** (all experts identical direction): `clamp_min(1e-12)` then `1 − 0 = 1` everywhere, then `clamp_(0,1)` — identical in both forms.

**Equivalence verdict: EXACT** — the reconstructed `d` is element-wise equal to `cdist(normalize(full),...)`; only the fp accumulation order of `G` differs, mitigated by fp64. No non-additive step breaks the online reduction: `normalize` and `cdist` are deferred to finalize and applied to the *reconstructed* `d`, which depends on `G` only through additive sums.

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
    x = logits.detach().to(torch.float64)          # [T_b, E]; fp64 for exact Gram accumulation
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
d     = (unit[:, None] + unit[None, :] - 2.0 * cos).clamp_min(0.0).sqrt()
sim   = 1.0 - d / d.max().clamp_min(1e-12)
sim.fill_diagonal_(1.0)
sim.clamp_(0.0, 1.0)
return sim.to(torch.float32)
```
Keep the `n==0 → zeros(0,0)` guard (≈482-484) and the IndexError-wrapping contract (now against `g.shape[0]`).

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

### 3.5 vLLM patch twin — `max_quality/patches/vllm_calibration_stage2_profile.patch`
The patch is a **single-hunk new-file add** (`--- /dev/null` … `@@ -0,0 +1,N @@`). Its body is a **verbatim copy** of `stage2_profile_writer.py` plus a thin vLLM callback-dispatch shim (`_router_handler`, `register_callback("router", _router_handler)`). Apply the SAME logic edits as §3.2 to the patch body:
- `_router_handler` calls `record_router_logits` — call site unchanged (the accumulator now folds internally).
- Remove the patch's `gate_logit_profiles` payload build, the `gate_logit_profiles=` kwarg, the checkpoint dump key, and the checkpoint load; add the `gate_gram` equivalents identical to §3.2.
- Update the import block for the `Stage2ProfilePayloadV4` / `save_stage2_profile_v4` rename.
- **Recompute the hunk header:** the file is created whole, so the only header is `@@ -0,0 +1,N @@` with `N` = the new `+`-line count of the body. **Regenerate, do not hand-count:** edit the canonical `vllm/calibration_stage2_profile.py` working copy, then `git diff --no-index /dev/null vllm/calibration_stage2_profile.py` (or `git apply -R` + re-add) to emit the corrected header.
- `MANIFEST.md` "Schema bump" line (currently `SCHEMA_VERSIONS["stage2_profile"] bumped … to 3`) → 4.

**Twin invariant:** after edits the patch `+`-body MUST remain a verbatim superset of `stage2_profile_writer.py` (writer logic + vLLM shim only). The reviewer verifies by diffing the patch `+`-body against the canonical writer (modulo the shim functions). Preserves the standing "canonical ↔ patch byte-equivalent in logic" contract.

---

## 4. Resumability
- `_sim_tensor` (Eq. 8): **already** checkpointed (`stage2_profile_writer.py` dump/restore) and sidecar-persisted — untouched by this plan, survives resume as before.
- `_gate_gram` (NEW, δ_gate): the running accumulator that replaces the raw list. MUST be persisted for resume — both the checkpoint (§3.2 `"ream_acc_gate_gram"`) and the sidecar (§3.3 `gate_gram` field) carry it.
- **No separate count needed.** `G` encodes both the off-diagonal numerator and the per-expert sum-of-squares (diagonal); δ_gate's dist2sim is scale-invariant in the cosine, so no token count enters the δ_gate path (unlike Eq. 8, whose `_total_tokens_by_layer` denominator is already persisted). Confirmed: `compute_gate_similarity_matrix` (450-542) never references `_total_tokens_by_layer`. Persist only `_gate_gram`.

---

## 5. Test plan
1. **NEW `test_gate_gram_online_equals_batched`** (new `max_quality/tests/test_gate_gram_equivalence.py`): synthetic `[T,E]` logit fixture split into ≥3 uneven batches; assert the online path (`record_router_logits` per batch → `compute_gate_similarity_matrix`) equals a **reference** doing the OLD math directly (`cat` → `F.normalize` → `cdist` → dist2sim) on the full tensor. Tolerance: `torch.allclose(online, ref, atol=1e-9, rtol=1e-9)` — fp64 Gram makes this near-exact; the only divergence source is fp accumulation order, ≪ 1e-9 in fp64. Parametrize the three §2.3 edge cases: one zero-norm expert, all-zero matrix, all-identical experts (`d.max()==0`).
2. **Update `test_stage2_profile_sidecar_writer_math.py`**: replace `test_gate_logit_profiles_preserved_verbatim` and its glp assertions with a `gate_gram`-shape/round-trip assertion. The Bug #1 sim_tensor tests (`test_writer_sim_tensor_matches_reference`, `test_writer_sim_tensor_distinguishes_from_buggy_path`) are UNCHANGED (Eq. 8 untouched) and MUST still pass — they regression-guard that `_sim_tensor` was not disturbed.
3. **Update `test_stage2_profile_sidecar_roundtrip.py`**: drop `gate_logit_profiles` build/assert; add `gate_gram` round-trip (`torch.equal(loaded.gate_gram, original.gate_gram)`, dtype fp64).
4. **Update `test_stage2_profile_sidecar_partial_hit.py`** and **`test_cached_calibration_signals.py`**: swap glp payload construction/asserts for `gate_gram`.
5. **`test_stage2_assignment_v2.py`** (calls `compute_gate_similarity_matrix`): should pass unchanged IF the reconstruction is faithful — the integration guard that the public δ_gate API result is preserved. If its fixture populates `gate_logit_profiles` directly, redirect it to `record_router_logits` / `_gate_gram`.
6. Any test pinning `schema_version==3` → 4; any `Stage2ProfilePayloadV3` reference → V4.

---

## 6. Memory accounting (why this fixes the blowup)
- Before: `gate_logit_profiles` = append-only `[T_b, 256] fp32` per batch, concatenated → ~24M tokens × 256 × 4 B held across 40 layers in the vLLM single-pass path ≈ **570-920 GB host RAM** + serialized to sidecar/checkpoint.
- After: `_gate_gram` = one `[256,256] fp64` per layer = **512 KB/layer**, **~20 MB across 40 layers** — bounded, constant in token count. Sidecar/checkpoint shrink by the same factor.

---

## 7. Equivalence fixture detail (for the reviewer to run)
The §5.1 test is the load-bearing proof. Minimal reference (pseudocode, fp64):
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
2. Implement §3.1-§3.4 (canonical), run the touched-module suite:
   `pytest max_quality/tests/test_stage2_profile_sidecar_writer_math.py max_quality/tests/test_stage2_profile_sidecar_roundtrip.py max_quality/tests/test_stage2_profile_sidecar_partial_hit.py max_quality/tests/test_cached_calibration_signals.py max_quality/tests/test_stage2_assignment_v2.py max_quality/tests/test_gate_gram_equivalence.py -q`
3. Regenerate the patch twin (§3.5): edit the canonical `vllm/calibration_stage2_profile.py` working copy, then `git diff --no-index /dev/null vllm/calibration_stage2_profile.py > max_quality/patches/vllm_calibration_stage2_profile.patch` (auto-correct hunk header), then `git apply --check` against the vLLM tree at its pinned SHA.
4. Diff the patch `+`-body against `stage2_profile_writer.py` to confirm the twin invariant (§3.5).
5. Update `MANIFEST.md` schema-bump note (3 → 4).

---

## 9. Files touched (summary)
- `max_quality/src/moe_compress/utils/activation_hooks.py` — `_gate_gram` field, `record_router_logits`, `compute_gate_similarity_matrix`, `clear_layer`, docstring.
- `max_quality/src/moe_compress/calibration/stage2_profile_writer.py` — dump/checkpoint/load glp→gram; `_CKPT_SCHEMA` 1→2.
- `max_quality/src/moe_compress/utils/cached_calibration_signals.py` — `Stage2ProfilePayloadV4`, `SCHEMA_VERSIONS` 3→4, `save_/load_stage2_profile_v4`, `__all__`.
- `max_quality/src/moe_compress/stage2/plugins/stage2_profile_cache.py` — hydrate `_gate_gram`.
- `max_quality/patches/vllm_calibration_stage2_profile.patch` + `max_quality/patches/MANIFEST.md` — twin edits + hunk-header regen + schema note.
- Tests: `test_stage2_profile_sidecar_writer_math.py`, `_roundtrip.py`, `_partial_hit.py`, `test_cached_calibration_signals.py`, `test_stage2_assignment_v2.py`, + new `test_gate_gram_equivalence.py`.

NOT touched: `_sim_tensor` / Eq. 8 / `record_gated_output` / `finalize_batch` / `compute_delta_expert` — the brief's stated target was wrong; those are already bounded and must stay byte-identical.
