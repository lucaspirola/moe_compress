# PLAN — Stage-2 profile-sidecar covariance capture (fix the empty `cov_acc` payload)

Status: PLAN (not implemented). A plan-reviewer reviews this before any code lands.

**Intent verdict: BUG-TO-FIX (not docstring-only).** Covariance IS in scope for Optimization A — the
reader `Stage2ProfileCacheProvider.on_layer_setup` direct-writes `payload.cov_acc` into the live
`InputCovarianceAccumulator` on full hit, and `LayerMergePlugin.on_profile` skips the live cov forward
(`_profile_layer` + `cov_acc.finalize_layer`) on full hit. So an empty cov payload makes a full-hit layer
emit ZERO covariance into `_stage2_input_covariance.pt`, silently corrupting Stage 3/4. The writer must feed
`cov_acc`. The false docstring claim is corrected as part of the fix (it is a symptom, not the root cause).

**Line numbers are from the shared checkout HEAD `463c9f9` (canonical writer) / the patch twin.** The
implementer must re-grep named symbols; offsets drift between the canonical writer and the vLLM patch body.

---

## 0. Verified evidence (read with my own tools, not summarized)

| Claim | File:anchor | Verified |
|---|---|---|
| `cov_acc` created + dtype-pinned + finalized + serialized + checkpointed | `stage2_profile_writer.py:91,154,367-372,459-470,567-569,694-695` | YES |
| NO callback feeds `cov_acc` — only router / expert_out_unweighted / layer_in registered | patch `vllm_calibration_stage2_profile.patch:234-236`; canonical has no `cov_acc.update` anywhere | YES (grep: zero `cov_acc.update` calls in writer) |
| Reader direct-writes `payload.cov_acc` → live `cov_acc.covariance` on full hit | `stage2_profile_cache.py:287-294` | YES |
| `LayerMergePlugin.on_profile` skips BOTH `_profile_layer` AND `cov_acc.finalize_layer(layer_idx)` on full hit | `layer_merge.py:513-514` (early return), `:523-533` (skipped body incl. `finalize_layer`) | YES |
| Downstream `_snapshot_cov_layer` reads `cov_acc.covariance[(layer_idx,e,name)]` for the layer | `layer_merge.py:677` → `shared_io.py:64-82` | YES |
| Live Stage 2 cov captures **gate_proj AND down_proj** | `profiling.py:230` (`cov_acc.update(li,e,"gate_proj",tensor)`), `:236` (`"down_proj"`) | YES |
| Legacy `_stage2_input_covariance.pt` dumps the whole `cov_acc.covariance` (gate+down) | `shared_io.py:202,239`; gated by `MOE_SKIP_STAGE2_COV_SAVE=1` at `orchestrator.py:1555-1559` | YES |
| `input_cov` sidecar is **gate_proj ONLY**, keyed by absolute `layer_idx` | patch `:6964-6971` (gate-only), `:7237` (`(layer_idx,e,"gate_proj")`) | YES |
| vLLM `expert_in` dispatch fires `(layer_idx=layer.moe_layer_id, hidden_states, topk_ids)` gated by `_ch._CAPTURE_EXPERT` | patch `vllm_calibration_hooks.patch:9943-9952` | YES |
| vLLM `expert_mid` dispatch fires `(layer_idx=_current_layer_idx, intermediate=[n_tok,top_k,interm], topk_ids)` gated by `_ch._CAPTURE_EXPERT_MID`; `_current_layer_idx` set when `_CAPTURE_EXPERT_UNWEIGHTED or _CAPTURE_EXPERT_MID` | patch `:9579-9588`, `:9929-9930` | YES |
| opt-a env block sets ROUTER + EXPERT_UNWEIGHTED only — NOT `VLLM_CALIB_CAPTURE_EXPERT` / `_EXPERT_MID` | `PLAN_PLUGIN_12_opt_a_redo.md §6.2` | YES (this is why a naive `expert_in` registration would never fire) |

**Root cause:** `setup()` builds `cov_acc` and the dump/checkpoint plumbing was written end-to-end, but the
data-feed hook was never wired. The schema, reader, and `on_profile` skip all already assume a populated cov
payload. This is a wiring gap, not a design descope.

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

**Avoiding double-storage of the same ~80GB cov:** the stage2_profile cov and `_stage2_input_covariance.pt`
hold the SAME data on a full-hit run, but at different lifecycle points (sidecar = calibration output;
`.pt` = Stage 2 output). They are not co-resident as duplicate sidecars; the `.pt` is derived. The genuine
redundancy is **stage2_profile cov vs `input_cov` sidecar** when an operator runs BOTH `--capture-input-covariance`
AND `--capture-stage2-profile` on the same calibration pass — they'd write gate_proj twice (once gate-only at
`layer_idx`, once gate+down at `layer_rank`). **Recommendation:** document that operators capturing
`--capture-stage2-profile` for a Stage-3/4-enabled run do NOT also need `--capture-input-covariance` (the
stage2_profile path is a superset for Stage 2; Stage 3/4 continue to read the separate input_cov sidecar only
on runs that lack stage2_profile). Do NOT auto-disable one from the other (keeps the features decoupled per
Bug #8's structural-independence principle). This is a doc note, not code.

**Disk delta of the fix:** capturing cov ADDS the gate+down covariance to the stage2_profile sidecar — the same
~50-80GB documented for `_stage2_input_covariance.pt` / the `input_cov` sidecar (`calibration_input_cov.py`
docstring: "~50-70 GB" gate-only; gate+down is larger, dominated by the `down_proj [d_hid,d_hid]` blocks —
the accumulator docstring at `activation_hooks.py:945-948` estimates ~4 GB gate + ~25.6 GB down = ~30 GB
*peak GPU `_pending`* per layer; on-disk in fp16 across all layers is the ~50-80GB range). At the default
`cov_storage_dtype=float16` this is the dominant component of the sidecar (vs. `sim_tensor` ~210 MB,
`layer_input_reservoir` ~3 GB). **The implementer MUST note this in `--capture-stage2-profile` help text and
MANIFEST** so operators size disk accordingly. It is the price of skipping the live cov forward.

---

## 3. The cov-capture hooks (mirror `_profile_layer` exactly)

### 3.1 gate_proj — `expert_in` hook

The live `_profile_layer.input_cb` calls `cov_acc.update(li, e, "gate_proj", tensor)` where `tensor` is the
per-expert gate_proj input (the hidden state rows the kernel dispatches to expert `e`). The vLLM `expert_in`
dispatch hands the callback the FULL `hidden_states [n_tok, hidden]` + `topk_ids [n_tok, top_k]`; the callback
masks per-expert and calls `update`. This is exactly what `vllm/calibration_input_cov.py::_on_expert_in`
already does (patch `:7149-7249`) — **reuse that masking logic verbatim**, but route it into the writer's
`_state.cov_acc.update(layer_idx, e, "gate_proj", sub)` instead of `input_cov`'s private `_COV_ACCUM` dict.

New handler in the writer:
```python
def _expert_in_handler(**kw):
    if not _CAPTURE or _state.cov_acc is None:
        return
    try:
        layer_idx = int(kw["layer_idx"]); hidden = kw["hidden_states"]; topk_ids = kw["topk_ids"]
    except KeyError:
        return
    hs = hidden.detach().reshape(-1, hidden.shape[-1])
    ids = topk_ids.detach()
    if ids.dim() == 1:
        ids = ids.unsqueeze(-1)
    n_experts = _state.n_experts or (int(ids.max().item()) + 1 if ids.numel() else 0)
    for e in range(n_experts):
        rows = (ids == e).any(dim=-1).nonzero(as_tuple=False).reshape(-1)
        if rows.numel() == 0:
            continue
        sub = hs.index_select(0, rows)               # [n_e, hidden]
        _state.cov_acc.update(layer_idx, e, "gate_proj", sub)   # accumulates subᵀ@sub into _pending
```
Register with `_ch.register_callback("expert_in", _expert_in_handler)` in `setup()` alongside the existing
router/expert_out/layer_in registrations.

**Fidelity note (RAISE for reviewer):** the live `input_cb` fires once per (layer, expert) with the kernel's
*actual* per-expert dispatched rows; the vLLM hook reconstructs the per-expert rows from `topk_ids`. For
**unique tokens per expert** these are the same set (a token routed to `e` contributes its row once). The
`input_cov` writer documents this exact equivalence (patch `:7227-7230`: "unique keeps each token at most once
… matches the Stage 2 update semantics"). The implementer MUST confirm against `_profile_layer`'s actual
per-expert row set (`profiling.py` early-exit forward + the expert dispatch) that the masking produces the
identical row multiset — i.e. that the live path also de-dups a token that lands in expert `e` via two top-k
slots. If the live path counts such a token TWICE (per-slot), the mask must use per-(token,slot) rows, not
`.any(dim=-1)`. This is the single highest-risk byte-equivalence point; gate it with a unit test (§5.1).

### 3.2 down_proj — `expert_mid` hook

The live `intermediate_cb` calls `cov_acc.update(li, e, "down_proj", tensor)` where `tensor` is the per-expert
post-SwiGLU intermediate (the down_proj input). The vLLM `expert_mid` dispatch hands the callback
`intermediate [n_tok, top_k, interm]` + `topk_ids`. Per the existing imatrix `_on_expert_mid` (patch
`:9526+`), the down statistic uses the **(token, slot) accounting axis** — each `(t,k)` pair maps to exactly
one expert, because the kernel runs one down_proj matmul per (token, slot) pair. So:
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
Register `_ch.register_callback("expert_mid", _expert_mid_handler)` in `setup()`.

**Fidelity note (RAISE for reviewer):** confirm `_profile_layer`'s down_proj `intermediate_cb` axis. If the
live path feeds the per-expert intermediate with **one row per dispatched (token,slot)** (which is the natural
kernel granularity), the per-slot mask above matches. If it instead de-dups to unique tokens, switch to the
unique-token mask. The gate (§3.1) and down (§3.2) axes may legitimately DIFFER (the imatrix code uses
any-top-k mask for gate, (token,slot) for down) — do not assume they are the same. Pin both against
`_profile_layer` with the byte-equivalence test (§5.1/§5.2).

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
writer deliberately does the matmul ON CPU for exactly this reason (patch `:7158-7177`: device-side alloc under
capture risks a crash). The writer's `cov_acc.update` keeps it on-device. **Two options — reviewer picks:**
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
  `_current_layer_idx` to be set, which happens when `_CAPTURE_EXPERT_UNWEIGHTED or _CAPTURE_EXPERT_MID`
  (patch `:9929`) — already satisfied because opt-a sets `VLLM_CALIB_CAPTURE_EXPERT_UNWEIGHTED=1`.

**Edit `PLAN_PLUGIN_12_opt_a_redo.md §6.2`'s env block** (the pre-import block at ~line 978 in the driver) to
ALSO set:
```python
os.environ["VLLM_CALIB_CAPTURE_EXPERT"] = "1"       # fires expert_in (gate_proj cov)
os.environ["VLLM_CALIB_CAPTURE_EXPERT_MID"] = "1"   # fires expert_mid (down_proj cov)
```
Verify the exact env-var names against `vllm/calibration_hooks.py` (`grep _CAPTURE_EXPERT` in the patch:
`_CAPTURE_EXPERT`, `_CAPTURE_EXPERT_MID`, `_CAPTURE_EXPERT_UNWEIGHTED` are distinct module flags). The
argparse `--capture-stage2-profile` help (driver §6.1) MUST be updated to list the two new env vars and the
~50-80GB cov disk delta (§2).

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
3. **`test_writer_cov_axis_matches_profile_layer`** (the load-bearing fidelity guard) — construct a fixture
   where a token routes to the same expert via two top-k slots; assert the writer's per-expert row multiset
   (and resulting cov + token_count) matches what `_profile_layer`'s `input_cb`/`intermediate_cb` produce for
   that fixture. This is the test that catches a wrong masking axis (§3.1/§3.2 fidelity notes). It MUST FAIL
   if the gate hook uses (token,slot) where the live path de-dups, or vice-versa.
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

Resumability is already covered by the existing checkpoint tests — the cov_acc was ALWAYS in the checkpoint
(`stage2_profile_writer.py:567-569,694-695`); feeding it changes the *content* of `cov_acc.covariance` /
`token_count`, not the checkpoint schema. **No checkpoint-schema bump needed for the base fix** (the
`_CKPT_SCHEMA` and `Stage2ProfilePayloadV3` fields are unchanged). Confirm an existing
`test_stage2_profile_*_checkpoint` round-trips a populated cov.

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

- `max_quality/src/moe_compress/calibration/stage2_profile_writer.py` — add `_expert_in_handler`,
  `_expert_mid_handler`; register both in `setup`; correct the §24-28 docstring. (Mirror exactly in the patch twin.)
- `max_quality/patches/vllm_calibration_stage2_profile.patch` — same two handlers + registrations + hunk-header regen.
- `max_quality/scripts/build_self_traces_calib_vllm.py` — env block: add `VLLM_CALIB_CAPTURE_EXPERT=1` +
  `VLLM_CALIB_CAPTURE_EXPERT_MID=1`; update `--capture-stage2-profile` help text (env vars + ~50-80GB cov disk delta).
- `max_quality/patches/MANIFEST.md` — note cov capture + disk delta.
- `max_quality/tests/test_stage2_profile_sidecar_writer_math.py` — §5 tests (gate, down, axis-fidelity, aliasing, reader roundtrip).
- (No `cached_calibration_signals.py` / `stage2_profile_cache.py` edits for the BASE fix — those fields/paths already exist.)

NOT touched: `ReamCostAccumulator`, `_sim_tensor`, the schema dataclass (base fix), the reader's cov block
(`stage2_profile_cache.py:287-294` is already correct). If sequenced after gate-logit-online, follow the V4 rename.

---

## 9. Reviewer checklist

1. §3.1/§3.2 masking axis matches `_profile_layer`'s actual per-expert row multiset (the §5.3 fidelity test is load-bearing).
2. §3.3 CUDA-graph safety — pick option A (CPU matmul) vs B (on-device + sync); default A.
3. §2 topology — confirm down_proj is genuinely consumed by Stage 3/4 on a full-hit run (so gate-only would corrupt).
4. §4 env-var names verified against `vllm/calibration_hooks.py` (`_CAPTURE_EXPERT` / `_CAPTURE_EXPERT_MID`).
5. §4 imatrix coexistence — enabling `VLLM_CALIB_CAPTURE_EXPERT` does not unintentionally double-capture.
6. §6 sequencing — confirm V4 keeps `cov_acc`/`cov_token_count`; regenerate the patch twin ONCE from the final canonical writer.
7. No checkpoint-schema bump needed for the base fix (cov_acc was always in the checkpoint).
