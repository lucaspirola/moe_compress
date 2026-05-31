# PLAN — Stage-2 profile-sidecar covariance capture (fix the empty `cov_acc` payload)

Status: PLAN (not implemented). A plan-reviewer reviews this before any code lands.

**Intent verdict: BUG-TO-FIX (not docstring-only).** Covariance IS in scope for Optimization A — the
reader `Stage2ProfileCacheProvider.on_layer_setup` direct-writes `payload.cov_acc` into the live
`InputCovarianceAccumulator` on full hit, and `LayerMergePlugin.on_profile` skips the live cov forward
(`_profile_layer` + `cov_acc.finalize_layer`) on full hit. So an empty cov payload makes a full-hit layer
emit ZERO covariance into `_stage2_input_covariance.pt`, silently corrupting Stage 3/4. The writer must feed
`cov_acc`. The false docstring claim is corrected as part of the fix (it is a symptom, not the root cause).

**Line numbers are from the shared checkout HEAD `463c9f9` (canonical writer) / the patch twin.** The
implementer must re-grep named symbols; offsets drift between the canonical writer and the vLLM patch body
(re-verified against `origin/main` `463cb1b` during plan revision — symbol offsets had drifted ~600-800 lines
in the big hooks patch; named-symbol re-grep is mandatory before any edit).

---

## 0. Verified evidence (read with my own tools, not summarized)

| Claim | File:anchor | Verified |
|---|---|---|
| `cov_acc` created + dtype-pinned + finalized + serialized + checkpointed | `stage2_profile_writer.py:91,154,367-372,459-470,567-569,694-695` | YES |
| NO callback feeds `cov_acc` — only router / expert_out_unweighted / layer_in registered | patch `vllm_calibration_stage2_profile.patch:234-236`; canonical has no `cov_acc.update` anywhere | YES (grep: zero `cov_acc.update` calls in writer) |
| Reader direct-writes `payload.cov_acc` → live `cov_acc.covariance` on full hit | `stage2_profile_cache.py:287-294` | YES |
| `LayerMergePlugin.on_profile` skips BOTH `_profile_layer` AND `cov_acc.finalize_layer(layer_idx)` on full hit | `layer_merge.py:493` (`on_profile`), `:514` (early `return`), `:523` (skipped `_profile_layer` call), `:533` (skipped `cov_acc.finalize_layer`) | YES |
| Downstream `_snapshot_cov_layer` reads `cov_acc.covariance[(layer_idx,e,name)]` for the layer | `layer_merge.py:677` → `shared_io.py:70` (def), `:82` (`covariance` clone) | YES |
| Live Stage 2 cov captures **gate_proj AND down_proj** | `profiling.py:230` (`cov_acc.update(li,e,"gate_proj",tensor)`), `:236` (`"down_proj"`) | YES |
| **LIVE cov row axis is `torch.where(mask[e])` per-(token,slot) for BOTH gate AND down** — there is NO gate/down axis difference in the live path | `activation_hooks.py:1314`/`:1351` (`top_k_pos, token_idx = torch.where(mask[e])`; `mask[e]` is `[top_k, T]`); `sel = hidden_states[token_idx]` feeds BOTH the `"input"`→gate_proj cb (`:1318`/`:1355`) and the `"intermediate"`→down_proj cb (`:1328`/`:1361`) via the SAME `token_idx` | YES |
| imatrix `_on_expert_in` (gate) uses **any-top-k unique-token mask**, `_on_expert_mid` (down) uses **(token,slot) per-slot axis** — these DIFFER, but ONLY inside imatrix's Σx² statistic, NOT the live cov path | hooks patch `_on_expert_in :6510`, `_on_expert_mid :6553` (docstring `:6565` "this hook uses the (token, slot) accounting axis rather than the per-expert any-top-k mask used by `_on_expert_in`") | YES |
| input_cov `_on_expert_in` (gate cov) uses `mask_2d.any(dim=-1)` unique-token rows; comment claims "matches the Stage 2 update semantics" | hooks patch `_on_expert_in :7379`, mask `:7461` (`mask_2d.any(dim=-1).nonzero…`) | YES — equivalence holds ONLY under `torch.topk` (distinct experts per token, no within-row repeat); see §3.1 |
| Legacy `_stage2_input_covariance.pt` dumps the whole `cov_acc.covariance` (gate+down) | `shared_io.py:202,239`; gated by `MOE_SKIP_STAGE2_COV_SAVE=1` at `orchestrator.py:1555-1559` | YES |
| `input_cov` sidecar is **gate_proj ONLY**, keyed by absolute `layer_idx` | hooks patch `_on_expert_in :7379`, key write `:7467` (`key = (layer_idx, int(e), "gate_proj")`) | YES |
| vLLM `expert_in` / `expert_mid` dispatches fire `(layer_idx=layer.moe_layer_id, hidden_states/intermediate, topk_ids)`, gated by `_ch._CAPTURE_EXPERT` / `_ch._CAPTURE_EXPERT_MID`; injected into vLLM `fused_moe.py` via the patch's regex rewrite (`_current_layer_idx = layer.moe_layer_id` regex at hooks patch `:1848`/`:1927`); `_current_layer_idx` is set when `_CAPTURE_EXPERT_UNWEIGHTED or _CAPTURE_EXPERT_MID` (already satisfied — opt-a sets `_EXPERT_UNWEIGHTED`) | hooks patch `:1848`, `:1927` (re-grep `moe_layer_id`; the literal dispatch sites are regex-injected, not a static `_ch.dispatch(...)` line — offsets here are from `463c9f9` and drift) | YES |
| opt-a env block sets ROUTER + EXPERT_UNWEIGHTED only — NOT `VLLM_CALIB_CAPTURE_EXPERT` / `_EXPERT_MID` | `PLAN_PLUGIN_12_opt_a_redo.md §6.2` | YES (this is why a naive `expert_in` registration would never fire) |

**Root cause:** `setup()` builds `cov_acc` and the dump/checkpoint plumbing was written end-to-end, but the
data-feed hook was never wired. The schema, reader, and `on_profile` skip all already assume a populated cov
payload. This is a wiring gap, not a design descope.

**Schema-version note (N2):** the BASE fix changes NO checkpoint schema — `_CKPT_SCHEMA` is currently `1`
(`stage2_profile_writer.py:539`). But per the §6 sequencing (gate-logit-online lands FIRST), `_CKPT_SCHEMA`
will already be `2` by the time this fix lands (gate-logit-online bumps it 1→2 for the gram sequencing). This
fix neither reads nor bumps it; it inherits whatever value gate-logit-online left. Do NOT assert `_CKPT_SCHEMA == 1`
in any new test — assert against the live constant.

---

## 1. The fix at a glance

Register a cov-feeding hook in the stage2_profile writer that mirrors the live `_profile_layer` cov exactly:
**gate_proj** from `hidden_states` (via the `expert_in` dispatch) and **down_proj** from the post-SwiGLU
intermediate (via the `expert_mid` dispatch), both fed through the existing `InputCovarianceAccumulator.update`
(the same accumulator the live path uses), with `finalize_layer` already called at dump time. Then the four
existing serialize/checkpoint/reader/skip paths Just Work because they were already built for a non-empty cov.

Two surfaces change in lockstep (the canonical-↔-patch byte-equivalent-logic contract):
- `max_quality/src/moe_compress/calibration/stage2_profile_writer.py` (canonical)
- `max_quality/patches/vllm_calibration_stage2_profile.patch` (vLLM twin)

Plus driver env wiring in `build_self_traces_calib_vllm.py` (turn on the two dispatch gates).

No schema change is required for the BASE fix (the `cov_acc`/`cov_token_count` fields already exist in
`Stage2ProfilePayloadV3`). See §6 for the gate-logit-online composition (that plan DOES bump the schema).

---

## 2. Cov topology decision (holistic — three artifacts)

Three covariance artifacts exist in the repo today:

| Artifact | Producer | Consumer | Key axis | Scope |
|---|---|---|---|---|
| `input_cov` sidecar | `vllm/calibration_input_cov.py` (`--capture-input-covariance`) | Stage 3 `covariance_collection.py`, Stage 4 `eora_inputs.py` (via `input_cov_cache.py` providers) | absolute `layer_idx` | **gate_proj only** |
| `_stage2_input_covariance.pt` | Stage 2 `_save_covariance` (live `cov_acc`) | Stage 3/4 (same loaders) | absolute `layer_idx` | gate+down |
| stage2_profile `cov_acc` (THIS) | stage2_profile writer (after fix) → reader hydrates live `cov_acc` | Stage 2 `_snapshot_cov_layer` / `_remap_covariance_for_layer`; then flows into `_stage2_input_covariance.pt` | `layer_rank` on disk → `layer_idx` after reader translation | **must match live: gate+down** |

**Recommended end-state (keep all three separate — do NOT collapse):**

1. **stage2_profile cov MUST capture gate_proj AND down_proj.** A full-hit layer skips the live cov forward,
   and the eventual `_stage2_input_covariance.pt` is built from the hydrated `cov_acc` (gate+down). Capturing
   gate-only would silently drop every layer's down_proj cov on full hit — Stage 3/4 `down_proj` SVD inputs
   would vanish. (`input_cov` gets away with gate-only because Stage 3 has a documented down_proj fallback
   (Corollary 3.3 / B-only, deviation D6) — that fallback does NOT apply when the live Stage 2 path would
   have produced a real down_proj cov. Matching the live path is the byte-equivalence contract.)

2. **`input_cov` does NOT become redundant; keep it separate.** It is gate-only, `layer_idx`-keyed, and feeds
   Stage 3/4 on runs that did NOT run `--capture-stage2-profile`. Re-pointing Stage 3/4 at the stage2_profile
   sidecar would require: (a) `layer_rank → layer_idx` re-keying in the Stage 3/4 loaders, (b) confirming the
   down_proj entries are actually wanted there (input_cov deliberately omits them), and (c) coupling the
   Stage 3/4 cache to a Stage-2-internal artifact. Out of scope and not worth the coupling.

3. **`_stage2_input_covariance.pt` stays.** On a full-hit Stage 2 run it is now reconstructed from the hydrated
   stage2_profile cov instead of a live forward — exactly the Optimization-A win. Operators who disable Stage
   3/4 still set `MOE_SKIP_STAGE2_COV_SAVE=1` to drop it (`orchestrator.py:1555`).

**Avoiding double-storage of the same ~91GB cov:** the stage2_profile cov and `_stage2_input_covariance.pt`
hold the SAME data on a full-hit run, but at different lifecycle points (sidecar = calibration output;
`.pt` = Stage 2 output). They are not co-resident as duplicate sidecars; the `.pt` is derived. The genuine
redundancy is **stage2_profile cov vs `input_cov` sidecar** when an operator runs BOTH `--capture-input-covariance`
AND `--capture-stage2-profile` on the same calibration pass — they'd write gate_proj twice (once gate-only at
`layer_idx`, once gate+down at `layer_rank`). **Recommendation:** document that operators capturing
`--capture-stage2-profile` for a Stage-3/4-enabled run do NOT also need `--capture-input-covariance` (the
stage2_profile path is a superset for Stage 2; Stage 3/4 continue to read the separate input_cov sidecar only
on runs that lack stage2_profile). Do NOT auto-disable one from the other (keeps the features decoupled per
Bug #8's structural-independence principle). This is a doc note, not code.

**Cov dimensions (M2 — corrected against the actual model config).** The cov matrices are `[d_in, d_in]`
where `d_in` is the INPUT dim of each projection:
- **gate_proj** input = `hidden_states` → `d_in = hidden_size` → cov `[hidden_size, hidden_size]`.
- **down_proj** input = post-SwiGLU intermediate → `d_in = moe_intermediate_size` → cov
  `[moe_intermediate_size, moe_intermediate_size]`.

Verified from `Qwen/Qwen3.6-35B-A3B` `config.json` (`config.json#text_config`): **`hidden_size = 2048`,
`moe_intermediate_size = 512`, `num_experts = 256`, `num_experts_per_tok = 8`, `num_hidden_layers = 40`.**
So:
- gate cov = `[2048, 2048]` → fp16 ≈ **8.4 MB**, fp32 ≈ 16.8 MB per (layer, expert).
- down cov = `[512, 512]` → fp16 ≈ **0.52 MB**, fp32 ≈ 1.05 MB per (layer, expert).

Note this INVERTS the old plan's claim: gate now DOMINATES (gate `[2048,2048]` ≫ down `[512,512]`), the
opposite of the `down_proj [d_hid,d_hid]=[5120,5120]≈25.6 GB` figure the old draft copied.

**Upstream docstring inaccuracy (flag, do NOT fix here):** `activation_hooks.py:942-948` is wrong for this
model — it states `d_hid≈5120, intermediate≈2048` and labels gate_proj `[2048,2048]≈4 GB` / down_proj
`[5120,5120]≈25.6 GB`. The real config is `hidden=2048` / `moe_intermediate=512`, and gate/down are also
swapped relative to the docstring's own dim labels (gate input = hidden, down input = intermediate). Leave the
docstring as-is for this fix (separate cleanup); just do not propagate its numbers.

**Disk delta of the fix (recomputed).** At `cov_storage_dtype=float16` the per-(layer,expert) gate+down cov is
≈ 8.4 + 0.52 = **8.9 MB**. Per layer × 256 experts ≈ **2.28 GB**; across 40 layers ≈ **~91 GB** total on disk
(fp16). This is the dominant component of the sidecar (vs. `sim_tensor` ~210 MB, `layer_input_reservoir`
~3 GB). **Peak GPU `_pending` per layer** is fp32 and single-layer: gate ≈ 4.29 GB + down ≈ 0.27 GB ≈
**~4.56 GB** (NOT the docstring's ~30 GB — that used the wrong/swapped dims). **The implementer MUST note the
~91 GB on-disk fp16 delta in `--capture-stage2-profile` help text and MANIFEST** so operators size disk
accordingly. It is the price of skipping the live cov forward. (Recompute if the target model's config differs —
the formula is `Σ_layers Σ_experts (hidden² + moe_intermediate²) × bytes(cov_storage_dtype)`.)

---

## 3. The cov-capture hooks (mirror `_profile_layer` exactly)

### 3.1 gate_proj — `expert_in` hook

The live `_profile_layer.input_cb` calls `cov_acc.update(li, e, "gate_proj", tensor)` where `tensor` is the
per-expert gate_proj input. **The authoritative reference is the LIVE Stage-2 path `instrument_experts`**
(`activation_hooks.py:1314`/`:1351`), NOT the imatrix code. The live path computes, per expert `e`:
```python
top_k_pos, token_idx = torch.where(mask[e])   # mask[e] is [top_k, T]; one (slot,token) pair per nonzero
sel = hidden_states[token_idx]                # rows indexed by token_idx -> "input"/gate_proj cb (:1318/:1355)
```
i.e. the gate cov rows are **per-(token,slot)** — `torch.where(mask[e])` yields one `(top_k_pos, token_idx)`
pair for every routing assignment of `e`, and `sel` is `hidden_states` indexed by that `token_idx`. The vLLM
`expert_in` dispatch hands the callback the FULL `hidden_states [n_tok, hidden]` + `topk_ids [n_tok, top_k]`;
the handler must reconstruct the SAME per-slot row set.

**Equivalence invariant (load-bearing).** Under `torch.topk` routing each token's top-k experts are DISTINCT
(no within-row repeat), so for a given `e` the per-slot row set (`torch.where(mask[e])` → `token_idx`) and the
unique-token row set (`(topk_ids==e).any(dim=-1)`) are EQUAL — every token routes to `e` via at most one slot.
The `input_cov` `_on_expert_in` (`:7379`, mask `:7461` `mask_2d.any(dim=-1)`) and the imatrix `_on_expert_in`
(`:6510`) both rely on exactly this invariant. Therefore `.any(dim=-1)` is acceptable for gate cov **ONLY under
the documented `torch.topk`-distinct-experts invariant.** To be robust if routing ever changes (e.g. a future
sampler that can repeat an expert within a row), **prefer matching the live per-slot form directly** —
reconstruct `token_idx` from `torch.where((topk_ids == e).T)` (mirroring `mask[e]` being `[top_k, T]`) rather
than de-duping with `.any(dim=-1)`. The two are byte-identical today; the per-slot form is the live contract.

Reuse the row-gathering machinery of `vllm/calibration_input_cov.py::_on_expert_in` (the CPU-residency +
`index_select` skeleton, patch `:7379-7480`), but (a) route the result into the writer's
`_state.cov_acc.update(layer_idx, e, "gate_proj", sub)` instead of `input_cov`'s private `_COV_ACCUM` dict, and
(b) use the per-slot `token_idx` row set to match the live `instrument_experts` semantics exactly.

New handler in the writer:
```python
def _expert_in_handler(**kw):
    if not _CAPTURE or _state.cov_acc is None:
        return
    try:
        layer_idx = int(kw["layer_idx"]); hidden = kw["hidden_states"]; topk_ids = kw["topk_ids"]
    except KeyError:
        return
    hs = hidden.detach().reshape(-1, hidden.shape[-1])    # [n_tok, hidden]
    ids = topk_ids.detach()
    if ids.dim() == 1:
        ids = ids.unsqueeze(-1)                           # -> [n_tok, top_k]
    n_experts = _state.n_experts or (int(ids.max().item()) + 1 if ids.numel() else 0)
    for e in range(n_experts):
        # Per-(token,slot) rows, matching live instrument_experts torch.where(mask[e]):
        # mask[e] is [top_k, T] -> nonzero gives one (slot, token) pair per assignment.
        # `token_idx` is the per-slot token index; under torch.topk this equals the
        # unique-token set, but we use the per-slot form to honor the live contract.
        slot_tok = (ids == e).nonzero(as_tuple=False)     # [n_assign, 2] -> (token, slot)
        if slot_tok.numel() == 0:
            continue
        token_idx = slot_tok[:, 0]                        # per-slot token rows
        sub = hs.index_select(0, token_idx)               # [n_assign, hidden]
        _state.cov_acc.update(layer_idx, e, "gate_proj", sub)   # accumulates subᵀ@sub into _pending
```
Register `_ch.register_callback("expert_in", _expert_in_handler)` (see §M1/§3 registration-location note:
the `register_callback` lives in the PATCH TWIN's `setup` block under the `_CALLBACKS_REGISTERED` guard,
NOT the canonical writer's `setup` — the canonical `setup` does no registration).

**Fidelity note (RAISE for reviewer):** the live `input_cb` row set is `hidden_states[token_idx]` where
`token_idx` comes from `torch.where(mask[e])` (`activation_hooks.py:1314`/`:1351`) — i.e. one row per
routing assignment (per-slot). The handler above reproduces that exact per-slot set. Under `torch.topk` this is
identical to `(topk_ids==e).any(dim=-1)` (distinct experts per token), so the simpler unique-mask would also be
byte-correct TODAY — but the per-slot form is the live contract and is robust to a future sampler that repeats
an expert within a row. Gate equivalence with the §5.1 byte-equivalence test against `_profile_layer`.

### 3.2 down_proj — `expert_mid` hook

The live `intermediate_cb` calls `cov_acc.update(li, e, "down_proj", tensor)` where `tensor` is the per-expert
post-SwiGLU intermediate (the down_proj input). **Critically, the live path uses the SAME per-(token,slot)
axis as gate** — in `instrument_experts` (`activation_hooks.py:1328`/`:1361`) the `"intermediate"`→down_proj
callback receives `intermediate` computed from `sel = hidden_states[token_idx]` for the SAME `token_idx` from
`torch.where(mask[e])`. There is **NO gate/down axis difference in the live cov path** (unlike imatrix, where
`_on_expert_in` uses a unique-token mask and `_on_expert_mid :6553` uses a per-slot mask — that asymmetry is
internal to imatrix's Σx² statistic and is NOT the reference here). The vLLM `expert_mid` dispatch hands the
callback `intermediate [n_tok, top_k, interm]` + `topk_ids [n_tok, top_k]`; reshape to `[n_tok*top_k, interm]`
and select the per-slot rows for `e` (this is naturally per-slot because the kernel runs one down_proj matmul
per (token, slot) pair, and the live path's `token_idx` covers exactly those assignments). So:
```python
def _expert_mid_handler(**kw):
    if not _CAPTURE or _state.cov_acc is None:
        return
    try:
        layer_idx = int(kw["layer_idx"]); interm = kw["intermediate"]; topk_ids = kw["topk_ids"]
    except KeyError:
        return
    interm_dim = interm.shape[-1]
    flat = interm.detach().reshape(-1, interm_dim)        # [n_tok*top_k, interm]
    flat_ids = topk_ids.detach().reshape(-1)              # [n_tok*top_k]
    n_experts = _state.n_experts or (int(flat_ids.max().item()) + 1 if flat_ids.numel() else 0)
    for e in range(n_experts):
        rows = (flat_ids == e).nonzero(as_tuple=False).reshape(-1)
        if rows.numel() == 0:
            continue
        sub = flat.index_select(0, rows)                  # [n_e_slots, interm]
        _state.cov_acc.update(layer_idx, e, "down_proj", sub)
```
Register `_ch.register_callback("expert_mid", _expert_mid_handler)` in the PATCH TWIN's `setup` block (same
`_CALLBACKS_REGISTERED` guard as gate; see §M1/§3 registration-location note). The canonical writer's `setup`
does no `register_callback`.

**Fidelity note (RAISE for reviewer):** the live down axis is the SAME per-(token,slot) axis as gate — both
flow from `torch.where(mask[e])` → `token_idx` in `instrument_experts` (`:1328`/`:1361`). The per-slot reshape
above matches it directly. **Do NOT model gate vs down on the imatrix asymmetry** — that asymmetry
(`_on_expert_in` unique-mask vs `_on_expert_mid` per-slot) is internal to imatrix's Σx² and is NOT the live cov
contract. For the live cov path, gate and down use the IDENTICAL per-slot axis; the §5.x fidelity guard should
assert exactly this (gate and down both per-slot, no de-dup divergence). Pin both against `_profile_layer`'s
`input_cb`/`intermediate_cb` row sets with the byte-equivalence test (§5.1/§5.2).

### 3.3 dtype / storage / aliasing — already correct via `InputCovarianceAccumulator`

Because the writer reuses the SAME `InputCovarianceAccumulator` the live path uses:
- **up_proj aliasing** is handled (`_alias_gate_up=True`, `update` ignores `"up_proj"` — `activation_hooks.py:989`).
- **fp32 accumulate, `storage_dtype` cast at finalize** — already pinned by `setup`'s `set_storage_dtype`
  (`stage2_profile_writer.py:154`); the dump-time assert (`:375-382`) and the reader cross-validation already
  guard it.
- **`finalize_layer` at dump** already drains `_pending → covariance` (`stage2_profile_writer.py:367-372`).
- **token_count** is auto-accumulated by `update` (`_gpu_token_count`) and serialized via `cov_token_count`.

**CUDA-graph safety (RAISE for reviewer):** the live `update` does the `xᵀx` matmul ON-DEVICE (GPU). The
`expert_in`/`expert_mid` dispatches fire from inside a CUDA-graph-captured MoE forward region. The `input_cov`
writer deliberately does the matmul ON CPU for exactly this reason (hooks patch `_on_expert_in :7379`,
CPU-residency rationale docstring `:7388-7405`, `hs = hidden_states.detach().to("cpu", …)` at `:7419`:
device-side alloc/cross-stream sync under capture risks a crash). The writer's `cov_acc.update` keeps it
on-device. **Two options — reviewer picks:**
- **(A)** Pre-`.to("cpu")` the `sub` slice in the handler before `update` (mirrors input_cov; CPU matmul;
  safe under capture; the on-disk values are identical, only the compute device differs — `update` casts to
  fp32 either way). RECOMMENDED for capture-path safety; matches the proven input_cov pattern.
- **(B)** Leave on-device and rely on the same multi-stream `torch.cuda.synchronize()` discipline the live
  path uses before `finalize_layer`. Lower overhead but unproven under vLLM graph capture.
  The implementer should default to **(A)** (proven safe; the per-expert row counts are small per the input_cov
  docstring) unless a pre-flight benchmark shows it is a bottleneck.

---

## 4. Driver wiring (`build_self_traces_calib_vllm.py`) — turn on the dispatch gates

The dispatches in §3 only fire when the env gates are set BEFORE the first vllm import:
- `expert_in` → gated by `_ch._CAPTURE_EXPERT` (`VLLM_CALIB_CAPTURE_EXPERT=1`).
- `expert_mid` → gated by `_ch._CAPTURE_EXPERT_MID` (`VLLM_CALIB_CAPTURE_EXPERT_MID=1`); ALSO requires
  `_current_layer_idx` to be set, which happens when `_ch._CAPTURE_EXPERT_UNWEIGHTED or _ch._CAPTURE_EXPERT_MID`
  (hooks patch `:10271` — re-grep, offsets drift) — already satisfied because opt-a sets
  `VLLM_CALIB_CAPTURE_EXPERT_UNWEIGHTED=1`.

**Edit the `if args.capture_stage2_profile:` env block** (`build_self_traces_calib_vllm.py:1199-1209` —
re-verify, offsets drift). Today it sets ONLY `VLLM_CALIB_CAPTURE_STAGE2_PROFILE=1` + `_ROUTER=1` +
`_EXPERT_UNWEIGHTED=1` + `VLLM_USE_FLASHINFER_MOE_FP16=0` — it does NOT set `_EXPERT` / `_EXPERT_MID`, which is
exactly why the cov hooks would never fire. ALSO set:
```python
os.environ["VLLM_CALIB_CAPTURE_EXPERT"] = "1"       # fires expert_in (gate_proj cov)
os.environ["VLLM_CALIB_CAPTURE_EXPERT_MID"] = "1"   # fires expert_mid (down_proj cov)
```
(and add them to the adjacent `log.info` summary). Verify the exact env-var names against
`vllm/calibration_hooks.py` (`grep _CAPTURE_EXPERT` in the patch: `_CAPTURE_EXPERT`, `_CAPTURE_EXPERT_MID`,
`_CAPTURE_EXPERT_UNWEIGHTED` are distinct module flags). The argparse `--capture-stage2-profile` help text
(`build_self_traces_calib_vllm.py:860-880` — the `help=` string starting ~`:862`) MUST be updated to list the
two new env vars and the **~91 GB fp16 cov disk delta** (§2).

**RAISE for reviewer:** confirm enabling `VLLM_CALIB_CAPTURE_EXPERT` does NOT also activate the imatrix
`expert_in` handler (`calibration_imatrix.py::_on_expert_in`) in a way that double-runs or conflicts on the
same calibration pass. The `register_callback` registry chains multiple callbacks per hook
(`test_chained_callbacks_coexist_on_expert_in`, patch `:955`), so both the imatrix handler AND the
stage2_profile handler would fire if imatrix capture is also on. They write to disjoint accumulators
(imatrix `_moe_accumulators` vs. writer `_state.cov_acc`), so coexistence is safe — but the implementer must
confirm `--capture-stage2-profile` does not *unintentionally* turn on imatrix capture, and vice-versa.

---

## 5. Test plan (no vLLM; drive the writer callbacks directly)

Extend `max_quality/tests/test_stage2_profile_sidecar_writer_math.py` (the existing §8.1 harness).

1. **`test_writer_cov_gate_proj_matches_reference`** — drive `_s2p._expert_in_handler(layer_idx, hidden_states,
   topk_ids)` over synthetic batches; build a reference by feeding the IDENTICAL per-expert row sets into a
   fresh live `InputCovarianceAccumulator.update(li, e, "gate_proj", sub)` + `finalize_layer`. After
   `dump_stage2_profile` + `load_stage2_profile_v3`, assert `payload.cov_acc[(rank, e, "gate_proj")]` is
   `torch.allclose` (fp16 storage tolerance) to the reference `cov.covariance[(layer_idx, e, "gate_proj")]`,
   and `cov_token_count` matches exactly.
2. **`test_writer_cov_down_proj_matches_reference`** — same for `_expert_mid_handler` + `"down_proj"`.
3. **`test_writer_cov_axis_matches_instrument_experts`** (the load-bearing fidelity guard) — **H2: the old
   "token routing to the same expert via two top-k slots" fixture is UNREALIZABLE under `torch.topk` (each
   token's top-k experts are distinct), so it is dropped.** Re-spec: build a REALIZABLE `topk_ids`
   (`[n_tok, top_k]` with distinct experts per row, e.g. random `torch.topk` over synthetic router logits),
   run it through the actual live `instrument_experts` wrapper (or directly compute its
   `torch.where(mask[e])` → `token_idx` row set per expert), and assert that BOTH the writer's
   `_expert_in_handler` per-expert row multiset (→ gate_proj cov + token_count) AND `_expert_mid_handler`
   per-expert row multiset (→ down_proj cov + token_count) EQUAL the live `instrument_experts`
   `input_cb`/`intermediate_cb` row sets for that same `topk_ids`. The assertion is on the row multiset and the
   resulting `torch.allclose` cov + exact `token_count`, for BOTH `gate_proj` and `down_proj`. This catches a
   wrong masking axis (§3.1/§3.2). Additionally keep a cheap **same-axis guard**: assert gate and down use the
   IDENTICAL per-slot row set for the same `topk_ids` (the live path has no gate/down axis difference) — this is
   the only surviving role of the dropped fixture: ensuring the two handlers do not diverge.
4. **`test_writer_cov_up_proj_aliased`** — drive an `update` with `"up_proj"`; assert it is ignored (no
   `(*, *, "up_proj")` key appears) — confirms aliasing is inherited from the shared accumulator.
5. **End-to-end reader roundtrip** — feed the loaded payload into `Stage2ProfileCacheProvider.on_layer_setup`
   with a fresh run-scope `cov_acc`; assert `cov_acc.covariance[(layer_idx, e, m)]` for both matrix names are
   hydrated and `LayerMergePlugin.on_profile`'s full-hit early-return leaves them intact (i.e. a full hit no
   longer yields an empty `_stage2_input_covariance.pt`).
6. **Docstring correction guard** — update `stage2_profile_writer.py:24-28` to state cov is captured via the
   `expert_in` (gate_proj) + `expert_mid` (down_proj) hooks (remove the false "reuses … directly so … byte-
   identical" claim and replace with the accurate hook description). No test asserts docstrings, but the plan
   reviewer checks it.

(Payload-class / loader names above target `Stage2ProfilePayloadV3` / `load_stage2_profile_v3` on the current
main, but per §6 sequencing this fix lands AFTER gate-logit-online, which renames V3→V4 — use whichever
payload/loader version is live when the fix lands.)

Resumability is already covered by the existing checkpoint tests — the cov_acc was ALWAYS in the checkpoint
(`stage2_profile_writer.py:567-569,694-695`); feeding it changes the *content* of `cov_acc.covariance` /
`token_count`, not the checkpoint schema. **This fix bumps NO checkpoint schema.** Note (N2): `_CKPT_SCHEMA`
is `1` on current main (`:539`) but will already be `2` after gate-logit-online lands first — this fix neither
reads nor bumps it. Do NOT hardcode `_CKPT_SCHEMA == 1` in any test; assert against the live constant. Confirm
an existing `test_stage2_profile_*_checkpoint` round-trips a populated cov.

---

## 6. Composition with `plan/gate-logit-online` (CRITICAL — same files)

`origin/plan/gate-logit-online` (`tasks/PLAN_GATE_LOGIT_ONLINE.md`, head `70963a0`) edits a LARGELY
OVERLAPPING file set. Overlap matrix:

| File | gate-logit-online edits | THIS plan edits | Conflict? |
|---|---|---|---|
| `activation_hooks.py` | `ReamCostAccumulator`: `_gate_gram` field, `record_router_logits`, `compute_gate_similarity_matrix`, `clear_layer` | NONE (cov hooks reuse `InputCovarianceAccumulator.update` unchanged) | **No overlap** — different class (`ReamCostAccumulator` vs `InputCovarianceAccumulator`). Clean. |
| `stage2_profile_writer.py` | `dump` glp→gram build; checkpoint `ream_acc_gate_logit_profiles`→`ream_acc_gate_gram`; `_CKPT_SCHEMA` 1→2 | ADD `_expert_in_handler` + `_expert_mid_handler` + their `register_callback` in `setup`; correct docstring | **Same file, disjoint regions.** gate-logit touches the gate-logit dump/ckpt loop + `setup` does NOT change there; THIS touches `setup`'s callback registration + adds two handlers. Sequence-able; minor proximity in `setup`. |
| `cached_calibration_signals.py` | `Stage2ProfilePayloadV3`→**V4**, `SCHEMA_VERSIONS` 3→4, `save/load_v3`→`v4`, drop `gate_logit_profiles`/add `gate_gram` | NONE for base fix (cov fields already exist) | **No edit from THIS plan.** But the rename V3→V4 means THIS plan's tests/reader references to `Stage2ProfilePayloadV3` / `save_stage2_profile_v3` must follow whichever lands first. |
| `stage2_profile_cache.py` | drop glp hydration, add `_gate_gram` hydration | NONE (cov hydration `:287-294` already correct) | **No overlap** in the cov block; both touch the reader file but disjoint hunks. |
| `vllm_calibration_stage2_profile.patch` | regen twin (glp→gram, V4 rename, hunk-header recompute) | regen twin (add the two cov handlers + registrations) | **HARD OVERLAP — both regenerate the whole single-hunk patch.** Must be sequenced on ONE branch; the second to land regenerates from the canonical writer that already contains the first's edits. |
| `MANIFEST.md` | schema-bump note 3→4 | cov-capture + disk-delta note | Same file, different lines. Trivial. |
| `build_self_traces_calib_vllm.py` | (none) | env-block: add `VLLM_CALIB_CAPTURE_EXPERT` + `_EXPERT_MID`; help text | **gate-logit does not touch the driver.** No conflict. |

**Sequencing recommendation: land BOTH on a single shared branch, gate-logit-online FIRST, then this fix.**
Rationale:
- gate-logit-online does the schema bump (V3→V4) and the patch-twin regen. If this cov fix lands first, the
  gate-logit regen would have to absorb the two new cov handlers into its V4 regen anyway.
- Landing gate-logit first means: (1) the schema is already V4; (2) THIS plan adds the cov handlers to a
  writer that already has the gram edits; (3) the single patch-twin regen at the end of THIS plan picks up
  BOTH sets of edits (gram + cov) in one `git diff --no-index /dev/null vllm/calibration_stage2_profile.py`.
- THIS plan's payload references then target `Stage2ProfilePayloadV4` / `save_stage2_profile_v4` (no `cov_acc`
  field change — gate-logit-online keeps `cov_acc`/`cov_token_count` in V4; it only removes
  `gate_logit_profiles`). Verify: V4 still carries `cov_acc: dict` + `cov_token_count: dict` (gate-logit-online
  §3.3 only drops `gate_logit_profiles`, adds `gate_gram` — the cov fields are untouched). **Confirmed
  compatible.**
- The patch-twin byte-equivalence diff (gate-logit-online §3.5 / §8.4) MUST be re-run after BOTH land so the
  `+`-body remains a verbatim superset of the canonical writer (now with gram + cov handlers + vLLM shim).

**If they land on separate branches instead:** the patch-twin regen WILL conflict (both rewrite the whole
single-hunk file). Flag to the implementer: never merge both patch regens blindly — regenerate once from the
final canonical writer.

---

## 7. Constraints honored

- **Source-patch only, no monkey-patching.** The cov handlers are registered via the existing
  `_ch.register_callback` registry (the same mechanism router/expert_out/layer_in use). No `setattr` on
  production code. The reused `InputCovarianceAccumulator.update` is the live API.
- **Canonical ↔ vLLM-twin byte-equivalent in logic.** The two handlers + their registrations are added to
  BOTH `stage2_profile_writer.py` and the patch body identically (the patch body already carries the
  router/expert_out/layer_in handlers as verbatim copies + a vLLM shim — see patch `:292-360`). Regenerate the
  patch hunk header (`git diff --no-index /dev/null vllm/calibration_stage2_profile.py`), do not hand-count.
- **No cost-total commentary, no unilateral actions.** This is a plan only.

---

## 8. Files touched (summary)

- `max_quality/src/moe_compress/calibration/stage2_profile_writer.py` (canonical) — add `_expert_in_handler` +
  `_expert_mid_handler` (testable handler cores live here); correct the docstring at `:24-28`. **Do NOT add
  `register_callback` here** — the canonical `setup` (`:114`) performs no registration; that is the patch
  twin's job (see next bullet). The two handler functions are mirrored byte-for-byte across both surfaces.
- `max_quality/patches/vllm_calibration_stage2_profile.patch` (twin) — same two handler bodies, PLUS the two
  new `_ch.register_callback("expert_in", _expert_in_handler)` / `register_callback("expert_mid",
  _expert_mid_handler)` lines added INSIDE the existing `_CALLBACKS_REGISTERED` guard in the twin's `setup`
  block (currently `:234-236`, alongside `router`/`expert_out_unweighted`/`layer_in`). Then **regenerate the
  whole single-hunk patch** (`git diff --no-index /dev/null vllm/calibration_stage2_profile.py`) — do NOT
  hand-edit hunk headers.
- `max_quality/scripts/build_self_traces_calib_vllm.py` — env block (`:1199-1209`): add
  `VLLM_CALIB_CAPTURE_EXPERT=1` + `VLLM_CALIB_CAPTURE_EXPERT_MID=1` (+ log line); update the
  `--capture-stage2-profile` help text (`:860-880`) with the two env vars + the **~91 GB fp16 cov disk delta**.
- `max_quality/patches/MANIFEST.md` — (a) note cov capture + ~91 GB disk delta in the prose; (b) **regenerate
  Patch 2's line count + MD5**: recompute `wc -l` + `md5sum` of the regenerated
  `vllm_calibration_stage2_profile.patch` and update the table rows `:23` (Patch 2 line count, currently 812)
  and `:24` (Patch 2 MD5, currently `fefbcec8b4f230317bdb16be808eecc8`), AND the `## Verifying locally`
  `# expect:` self-checks at `:39` (MD5) and `:41` (line count). (Both values WILL change — the patch body
  grows by the two cov handlers + 2 registration lines. If gate-logit-online lands first, regenerate ONCE from
  the final twin that already carries the gram edits, so the MANIFEST reflects the combined patch.)
- `max_quality/tests/test_stage2_profile_sidecar_writer_math.py` — §5 tests (gate, down, axis-fidelity,
  aliasing, reader roundtrip).
- (No `cached_calibration_signals.py` / `stage2_profile_cache.py` edits for the BASE fix — those fields/paths already exist.)

NOT touched: `ReamCostAccumulator`, `_sim_tensor`, the schema dataclass (base fix), the reader's cov block
(`stage2_profile_cache.py:287-294` is already correct). If sequenced after gate-logit-online, follow the V4 rename.

---

## 9. Reviewer checklist

1. §3.1/§3.2 masking axis matches the LIVE `instrument_experts` per-(token,slot) row set for BOTH gate AND
   down (NOT modeled on the imatrix gate/down asymmetry); the §5 item-3 fidelity test is load-bearing.
2. §3.3 CUDA-graph safety — pick option A (CPU matmul) vs B (on-device + sync); default A.
3. §2 topology — confirm down_proj is genuinely consumed by Stage 3/4 on a full-hit run (so gate-only would corrupt).
4. §2 cov dims — gate `[hidden_size,hidden_size]`, down `[moe_intermediate_size,moe_intermediate_size]`;
   ~91 GB fp16 disk delta recomputed from the real config (hidden=2048, moe_intermediate=512, 256 experts, 40 layers).
5. §4 env-var names verified against `vllm/calibration_hooks.py` (`_CAPTURE_EXPERT` / `_CAPTURE_EXPERT_MID`);
   driver env block `:1199-1209` does NOT set them today.
6. §4 imatrix coexistence — enabling `VLLM_CALIB_CAPTURE_EXPERT` does not unintentionally double-capture.
7. §8 registration goes in the PATCH TWIN's `_CALLBACKS_REGISTERED` guard, NOT the canonical `setup`.
8. §8 MANIFEST Patch 2 line-count + MD5 regenerated (`:23`/`:24`/`:39`/`:41`).
9. §6 sequencing — confirm V4 keeps `cov_acc`/`cov_token_count`; regenerate the patch twin ONCE from the final canonical writer.
10. No checkpoint-schema bump from this fix (cov_acc was always in the checkpoint); `_CKPT_SCHEMA` inherits gate-logit-online's value (will be 2).
