# PLAN — A7 (capture-only hook) + A1 (windowed single-pass covariance collection)

Author: planner/architect pass. Branch `plan/stage3-a7-a1` off `main` (`e395ad0`).
Scope: design only — NO production code in this doc. Every code claim below is
`file:line`-verified against `main`. Papers: AA-SVD arXiv:2604.02119, Swift-SVD
arXiv:2604.01609 (both YYMM ≤ 2606, valid).

---

## 0. TL;DR — the crux verdict (read this first)

**A7+A1 is NOT byte-identical to the current golden artifacts. It IS byte-identical
to a NEW all-native golden, and that new golden is *more* faithful to inference,
not less. Recommendation: option (b) — regenerate the Stage-3 golden against the
capture-only (A7) native path, gated by a one-time native-vs-loop covariance
equivalence + rank/PPL non-regression check.**

Why no scheme is byte-identical to *today's* artifacts:

- The current pipeline captures covariance through `instrument_experts`, which
  **replaces** the experts' `forward` with a per-expert Python loop using
  `F.linear` (`activation_hooks.py:1453-1521`). That per-expert-loop fp reduction
  order is baked into today's golden.
- The model's **real** (inference) expert forward is neither that loop nor a
  per-token matmul — for `FactoredExperts` it is a **padded batched `bmm`**
  (`model_io.py:982-994`), a *third* distinct fp reduction order.
- So the current golden's numerics correspond to an instrumentation artifact (the
  Python loop) that never runs at inference. Any capture-only design that lets the
  native forward run will differ from it at the fp-rounding level — and is closer
  to what the deployed model actually computes.

The honest framing (matches the fidelity audit `analysis/stage3-speedup-fidelity`):

- (a) *Truly byte-identical to current*: **impossible** for any windowed scheme
  (proven in §2). Reject.
- (b) *Byte-identical to a NEW all-native golden, + more correct*: **recommended**.
  Requires golden regen + the equivalence gate in §7. ✅
- (c) *A quality trade*: NOT what this is. The covariance *definitions* (AA-SVD
  Thm 3.2 `C = X_pre^T X_post`, `S = X_post^T X_post`) are unchanged; only the
  fp reduction order of the captured activations moves, within the same fp
  tolerance the pipeline already accepts across DP shards and devices
  (`test_multigpu_stage3.py:176-179, 209`).

---

## 1. Verified analysis of the current N-pass design

### 1.1 The N×-redundant structure (the 80–90% cost)

`_collect_covariances` (`covariance_collection.py:279-546`) runs **one full
dual-forward per MoE layer**:

- Outer loop `for k, ref in enumerate(moe_layers)` (`:449`) — N iterations
  (N≈40 for the 35B target).
- Per iteration it instruments **only** layer `k` on the student (`:475-477`) and,
  for cross-cov, only the matching teacher layer (`:484-486`).
- Inner loop `for batch_idx, batch in enumerate(batches)` (`:489`) forwards the
  **full-depth** teacher (`:494-496`) then the full-depth student (`:497-499`),
  both under `no_grad`.
- The module docstring at `:312-317` ("We hook ALL layers on BOTH models
  simultaneously") is **STALE** — it describes the original intent; the live code
  does one layer per pass. The `:438-446` NOTE is the accurate description and
  states the N-pass design is a **memory** tradeoff (legacy label "D9"), explicitly
  "Wall-clock cost is ~N× the simultaneous design". This is exactly the redundancy
  A1 removes.

Net: both 35B models are forwarded full-depth N times. A1 wants 1 pass (or
`ceil(N/G)` passes for a window of G layers) → up to N× / G× speedup on the
dominant cost.

### 1.2 The accumulators are order-stable per key (A1's correctness floor)

The Gram accumulators are pure additive sums keyed by `(layer, expert, matrix)`:

- `InputCovarianceAccumulator.update` (`activation_hooks.py:983-1021`): computes
  `cov = flatᵀ @ flat` on-device and `cur.add_(cov)` per key (`:1012, :1018-1020`).
- `update_cross` (`:1023-1052`): `cur.add_(cross_f32...)` per key (`:1049-1051`)
  for the precomputed `X_preᵀ X_post`.
- `finalize_layer` (`:1054-1084`): pops one layer's pending keys, casts to
  `storage_dtype`, single GPU→CPU transfer, merges into `covariance` dict.

Each token contributes to exactly one `(layer, expert)` key per pass, so the
additive order **within a key** is independent of how many layers are hooked
simultaneously. Hooking G layers in one pass produces the same per-key partials as
hooking them one-at-a-time, in the same intra-key order. **The accumulator layer
is not where A1 breaks byte-identity** — confirmed by the fidelity audit and by
the existing DP reduce being a key-wise sum (`_reduce_spilled_cov_dirs`,
`covariance_collection.py:193-256`).

### 1.3 WHERE byte-identity actually breaks — `instrument_experts` replaces forward

`instrument_experts` (`activation_hooks.py:1406-1529`) is a contextmanager that
**swaps** `experts.forward` for a wrapper (`:1525` `experts.forward =
types.MethodType(forward_fn, experts)`; restored `:1529`). The two wrappers:

- `wrapped_fused` (`:1491-1521`): per-expert Python `for` loop; `sel =
  hidden_states[token_idx]`, `gate_up = F.linear(sel, self.gate_up_proj[e])`,
  `intermediate = act_fn(gate)*up`, `down = F.linear(intermediate, ...)`,
  `final.index_add_(0, token_idx, ...)`.
- `wrapped_factored` (`:1453-1488`): same loop but rank-k
  `F.linear(F.linear(sel, V[e]), U[e])`.

The **native** forwards are different fp reduction orders:

- `FactoredExperts.forward` (`model_io.py:906-995`) — **padded batched `bmm`**:
  gathers active-expert tokens into `[n_active, max_tokens, d_hid]` (`:961-968`),
  6 `bmm`s (`:982-986`), un-pads + single `index_add_` (`:989-994`). This is a
  *different* fp reduction than the per-expert `F.linear` loop.
- The fused (unfactored) native path is a fused/grouped GEMM (analogous claim;
  the wrapper's `wrapped_fused` reproduces its math with a Python loop).

**Consequence for A1 (verified mechanism):**

- *Current N-pass*: when collecting layer L, layers `0..L-1` run the **native**
  forward; only L runs the Python loop. So `X_pre^{(L)}`/`X_post^{(L)}` sit
  downstream of native upstream MoE forwards, and L's own captured activations are
  taken inside the Python-loop forward.
- *A1 with `instrument_experts` on all/G layers*: layers `0..L-1` now run the
  **Python loop** too → the residual stream into L shifts at fp level →
  `X_pre/X_post` (and `C`, `S`) at L move vs today. **Not byte-identical.** (This
  is the fidelity audit's "single biggest fidelity risk".)
- *A7 capture-only (no forward replacement)*: upstream layers stay native, so the
  residual stream is the **real** inference stream. But L's *own* captured
  activation is now taken from the native (bmm/GEMM) forward instead of the Python
  loop → still differs from today at L (different reduction order at the captured
  layer itself). **Also not byte-identical to today — but byte-identical to a new
  all-native baseline, and that baseline equals the deployed model's actual
  numerics.**

So the choice is not "A7 byte-identical vs A1 not"; it is "which *new* baseline".
A7's all-native baseline is the inference-faithful one. See §2.

---

## 2. THE CENTRAL RESOLUTION: byte-identical-vs-new-golden

### 2.1 Is (i) — A7+A1 == NEW all-native golden — real? YES.

Claim: with A7 capture-only hooks on a window of G layers, the captured
`(X_pre, X_post)` per layer, and hence `C` and `S`, are **byte-identical to what a
single full-depth native forward of the (unmodified) student+teacher would
produce**, regardless of G or window boundaries, because:

1. Upstream layers run the native forward in *all* windows → identical residual
   stream into every captured layer in every window (the stream does not depend on
   which downstream layers are hooked; a capture-only hook is side-effect-free).
2. The captured quantities are deterministic functions of that native stream:
   - `X_pre`/`X_post` for the **gate_proj** key = the experts' input hidden state
     gathered by `token_idx` (`input_cb` at `covariance_collection.py:362-419`;
     `_teacher_input_cb` at `:336-360`). These are **pure input gathers** — they
     do not depend on the expert matmul at all, only on routing (`token_idx`,
     which is a function of the router output on the native stream). So the
     gate-side `C`/`S` are invariant to *the captured layer's* forward internals;
     they move only with the upstream stream — which A7 keeps native. ✅
   - `X_post` for the **down_proj** key = `intermediate = act_fn(gate)*up`
     (`intermediate_cb` at `:421-428`, fed by the `"intermediate"` callback). This
     **does** depend on the captured layer's gate/up matmul. Under A7 it is read
     from the native bmm/GEMM result → different fp order than the current Python
     loop → this is the one key whose value moves *even without upstream drift*.
3. Across windows, each layer is captured in exactly one window, and the
   accumulator is keyed by `(layer,expert,matrix)` with no cross-window
   interaction (§1.2). So `ceil(N/G)` windowed passes == 1 conceptual native pass,
   per key, bit-for-bit (given a fixed device/dtype).

Therefore A7+A1 is **byte-reproducible against a regenerated all-native golden**
and that golden is a deterministic function of the real model — the cleaner
reference.

### 2.2 Is (ii) — reproduce the current Python-loop numerics across a windowed pass — real? NO.

To keep today's numerics, every captured layer L would need (a) its **upstream**
layers on the native forward AND (b) **itself** on the Python loop. In a single
windowed pass that hooks `{L, L+1, ..., L+G-1}`, layer L+1's upstream now includes
L, which must be on the Python loop for L's own capture — but then L+1 sees a
Python-loop upstream, contradicting (a) for L+1. The two requirements are mutually
exclusive for any window with ≥2 hooked layers. Confirmed infeasible. (You *could*
emulate it with per-layer "run upstream native, swap only L to loop" — but that is
exactly the current N-pass, no speedup.) Reject (ii).

### 2.3 Recommendation + why (b) is acceptable / more correct

**Recommend (b): regenerate the Stage-3 golden against the A7 all-native path.**

Justification:
- **Definitional fidelity is preserved.** AA-SVD Thm 3.2 (PDF lines 575-577, per
  the fidelity audit) says the solution "operates only on the covariance matrices
  … its cost is independent of the number of calibration tokens" — the covariance
  *value* is all that matters; collection order/instrumentation is irrelevant *if
  the captured activations are the same numbers*. A7 captures the same defined
  quantities (`C = X_preᵀX_post`, `S = X_postᵀX_post`, gate-only cross per D6),
  just read off the native forward. No paper quantity changes.
- **More inference-faithful.** The native bmm/GEMM forward is what the deployed
  model runs; the Python-loop forward is a measurement artifact. Calibrating the
  SVD against the activations the model *actually* produces is strictly more
  correct than calibrating against a reduction order that exists only during
  instrumentation.
- **The fp delta is the same class the pipeline already tolerates.** DP shards
  (`test_multigpu_stage3.py:302-330`, rtol=1e-6) and cross-device passes
  (`:176-179, 209`, rtol=1e-5) are already accepted as "fp-tolerance equal, not
  bit-exact". Native-vs-loop is the same kind of reordering.

**Required validation before blessing the new golden (the gate, see §7):**
1. **Native-vs-loop covariance equivalence** on `tiny_model`: collect `C`/`S` via
   the current `instrument_experts` path and via the new A7 capture-only path in a
   **single-layer** configuration (so upstream drift is excluded and only the
   captured-layer forward-order differs); assert `allclose(rtol=1e-4, atol=1e-5)`
   per key. This isolates the down_proj key delta and bounds it.
2. **1-pass-vs-N-pass equivalence** (A1): A7 single windowed pass (all layers) vs
   A7 per-layer passes must be **byte-identical** (atol=0 on CPU) — proving the
   windowing itself adds zero error on top of the native baseline.
3. **Rank/PPL non-regression** on one real arm: regenerate `rank_map` under A7,
   diff against the current golden's ranks; require no rank flips beyond a
   pre-agreed tolerance, and WikiText-2 PPL within noise. (The fidelity audit notes
   the rank-flip risk is *plausible but unmeasured* — this gate measures it.)

If gate (3) shows rank flips that move PPL, escalate to the user before blessing —
do NOT silently ship a quality change (per RAISE-don't-substitute).

---

## 3. A7 design — capture-only expert input/output hook

### 3.1 Quantities to capture (must equal today's, modulo native-vs-loop order)

`instrument_experts` today emits five callbacks (`activation_hooks.py:1418-1427`);
the cov collector uses exactly two:

- `"input"` → `input_cb` (`covariance_collection.py:362`): `sel = hidden_states[
  token_idx]` per expert → student `S` (gate_proj key) + cross `C`.
- `"intermediate"` → `intermediate_cb` (`:421`): `act_fn(gate)*up` per expert →
  student `S` (down_proj key, the b4f882c down-cov capture).
- teacher side uses only `"input"` → `_teacher_input_cb` (`:336`).

A7 must reproduce, per `(layer, expert)`:
- `X = hidden_states[token_idx]` (the gate_proj expert input), and
- `intermediate = act_fn(gate_e)*up_e` for the down_proj input,

with the **same `token_idx` / `top_k_pos` / `top_k_weights` context** the loop
builds at `:1470-1471, :1507-1508`.

### 3.2 The hook mechanism (no forward replacement)

A7 registers, per hooked experts module, on the **native** forward:

1. **`register_forward_pre_hook(..., with_kwargs=True)`** — receives `(module,
   args, kwargs)` where `args == (hidden_states, top_k_index, top_k_weights)`
   (native sig `model_io.py:906-911`; fused sig identical, mirrored by the wrappers
   `:1454, :1491`). From these inputs A7 **re-derives the exact per-expert routing**
   the loop uses, with the identical ops:
   `mask = F.one_hot(top_k_index, num_experts).permute(2,1,0)` then
   `top_k_pos, token_idx = where(mask[e])` (identical to `:1461-1468` /
   `model_io.py:941-959`). `X = hidden_states[token_idx]` is then **bit-identical**
   to the loop's `sel` — it is a pure gather, independent of forward internals.
   → fires the `input`/`gate_up_in` equivalents (student `S`-gate + cross `C`).
   The teacher experts module gets the same pre-hook (input-only) for `X_pre`.

2. **down_proj input** (`intermediate`): the pre-hook alone cannot produce
   `act_fn(gate)*up` without doing the matmul. Two A7 sub-options:
   - **A7-pre-recompute (RECOMMENDED):** inside the pre-hook, after deriving
     per-expert `sel`, recompute `intermediate` for the **down_proj key only** with
     the native expert weights in the native shape — i.e. for FactoredExperts use
     the same `bmm` form as `model_io.py:982-985` on the gathered tokens (NOT the
     per-expert `F.linear` loop), so the captured `intermediate` matches the native
     forward's reduction order. This duplicates the gate/up matmul (≈⅔ of the
     expert flops) but keeps the real forward untouched and the captured value
     native-consistent. Cost is acceptable because A1 removes the N× factor that
     dwarfs it.
   - **A7-forward-hook (alternative):** a `register_forward_hook` cannot see the
     internal `inter` tensor (it is local to forward, not the output). To capture
     `inter` natively without recompute would require the model's forward to expose
     it (a `capture` flag on `FactoredExperts.forward` that stashes `inter`/`down`
     pre-routing-weight). That is a *small, surgical* native-code change and is the
     cleanest long-term option, but it touches production forward — flag for user
     decision. Default to A7-pre-recompute to avoid editing the inference forward.

   Either way the captured `intermediate` is byte-identical to the native bmm
   result (same op, same gather), satisfying §2.1's down_proj requirement.

   **Zero-pad cancellation (why mirroring the padded bmm shape is safe).** The
   native `inter` (`model_io.py:985`) is **padded** `[n_active, max_tokens, d_int]`
   with zero pad rows (`silu(0)*0 = 0`), whereas the loop's `intermediate`
   (`activation_hooks.py:1481`) is **un-padded** per expert. But `B_acc.update`
   reshapes to `(-1, d_int)` (`activation_hooks.py:1003`) and forms `flatᵀ@flat`
   (`:1012`) — the zero pad rows contribute **zero** outer-products, so the padded
   (native bmm) recompute and the un-padded loop yield an **IDENTICAL** down_proj
   Gram. This is *why* the A7 recompute may safely use the native bmm shape; the
   §7.1 byte-match unit test should therefore feed the padded `inter` (zero rows
   included) and still assert exact equality.

3. **No output capture is needed for cov** (the loop's `down`/`gate_up_out`
   callbacks are unused by the collector). So A7 is genuinely "input + intermediate
   capture only"; the experts' real output (the residual contribution) flows
   through the **native** forward unmodified. This is the whole point: upstream
   stays native.

### 3.3 Reentrancy / lifecycle

- A7 hooks are plain `RemovableHandle`s; register on entry, `.remove()` on exit
  (mirror the contextmanager pattern of `instrument_experts:1526-1529`). No
  forward swap → the `_instrument_experts_patched` reentrancy guard
  (`:1438-1443`) is irrelevant to A7; A7 and `instrument_experts` must NOT be
  active on the same module simultaneously (assert/guard).
- A7 must be a **sibling** capture path selectable by config, with
  `instrument_experts` retained as the default/fallback (clean degrade, §4.4).

### 3.4 Keying invariant (preserved)

Per-`(layer, expert, matrix)` keying is preserved exactly: A7 derives `layer_idx`
from the `MoELayerRef` (`model_io.py:215-247`), `expert_idx` from the `where(mask)`
enumeration, and `matrix_name ∈ {gate_proj, down_proj}` from which capture point
fired — identical to today (`update`/`update_cross` calls at
`covariance_collection.py:371, 417, 422`).

---

## 4. A1 design — windowed single-pass collection

### 4.1 Window the layer loop

Replace the per-layer outer loop (`covariance_collection.py:449`) with a
**windowed** loop: partition `moe_layers` into contiguous windows of size G; for
each window, register A7 capture hooks on **all G layers** (student + teacher),
forward each batch **once** (teacher then student, `no_grad`), then
`finalize_layer` + spill **each** layer in the window and remove its hooks.
Passes = `ceil(N/G)` instead of N → **G× speedup**, all native (so §2.1 holds).

`G = N` is the full single-pass; `G = 1` degrades to **today's structure but on the
native forward** (still a new-golden, NOT the current golden — see §2). The window
exists purely for VRAM, exactly as the `:438-446` NOTE describes.

### 4.2 `_teacher_hidden` lifetime — the real restructure

Today `_teacher_hidden` is a single-layer `{token_idx → row}` dict cleared every
batch (`:469, :492`) and every layer. Under A1 it must hold **G layers'** teacher
rows for the duration of one batch's student forward:

- Structure stays `dict[layer_idx → dict[token_idx → row]]` (it already is keyed by
  layer, `:334, :354-360`), so multi-layer storage needs no schema change — just
  do **not** restrict it to one layer.
- Clear per **batch** (not per layer): clear at the top of each batch
  (keep `:492`), populate all G teacher layers during the teacher forward, consume
  all G during the student forward, then clear. This bounds teacher-hidden RAM to
  **G × (one layer's matched rows)** per batch — the dominant new RAM term for
  cross-cov and the input to VRAM auto-sizing (§4.3).
- A4 (vectorize the dict→dense `index_select`, from the speedup analysis) composes
  here and **reduces** this RAM/CPU cost; recommend landing A4 alongside A1 since
  A1 multiplies the per-batch teacher-row volume by G. (A4 is paper-faithful per
  the fidelity audit; out of scope to design fully here but note the synergy.)

### 4.3 VRAM auto-sizing of G

G is bounded by the activations that must be **simultaneously resident** for one
windowed batch. Auto-size with a **new** `_resolve_cov_window(config, devices)`
that follows the config-resolution *shape* of `_resolve_cov_replicas`
(`orchestrator.py:79-100`) but **adds a VRAM probe** — `_resolve_cov_replicas` is
config-ONLY (no probe). The probe uses `torch.cuda.mem_get_info()` (already used at
`utils/runtime_monitor.py:78`), a NEW mechanism, not a mirror of the existing
config-only resolver:

- Probe free VRAM via `torch.cuda.mem_get_info()` per device.
- Estimate per-layer hook residency: for the captured batch, the dominant terms
  are (a) the gathered expert inputs for `S`/`C` accumulation (transient, freed
  after each `update`), and (b) the per-layer `_pending` Gram on-device until
  `finalize_layer` (≈ `d_hid² · 4 bytes` for gate + `d_int²·4` for down, per
  layer — the on-device fp32 cov, `update:1012`). The cov tensors are the
  persistent term across a window because `finalize_layer` runs at window end.
  → `G ≈ floor((free_VRAM − headroom) / per_layer_resident_bytes)`, clamped to
  `[1, N]`.
- **Mitigation to raise G:** call `finalize_layer` + spill **eagerly** as each
  layer's contribution completes within the window is NOT possible (all layers'
  hooks fire every batch), BUT you can `finalize_layer(L)` for the whole window
  only after the last batch. To bound the persistent cov term, spill is per-layer
  at window end (reuse the existing background spill executor `:430-436, :505-518`).
  The teacher-hidden RAM (host, not VRAM) from §4.2 is the other knob.
- Config: add `multi_gpu.cov_window_size` (int, default `auto`) — under the
  **`multi_gpu`** block that `_resolve_cov_replicas` already reads
  (`orchestrator.py:93` `config.get("multi_gpu")`, `cov_replicas` at `:94`), NOT a
  separate `multigpu` block (which does not exist → would silently default G=1).
  `auto` → VRAM-probe; explicit int → clamp to `[1, N]`. Default behavior with the
  key absent = `G=1` native (clean degrade, NOT current golden — documented).

### 4.4 Compose with multi-GPU (DP + sharding, already landed)

- **DP replicas** (`run_dp_covariance_collection:673-748`, `_cov_replica_worker:
  580-670`): each replica calls `_collect_covariances` over a disjoint calib shard
  and spills to its own subdir; parent key-wise reduces
  (`_reduce_spilled_cov_dirs:193-256`). A1 lives **inside** `_collect_covariances`,
  so each replica independently gets the G× windowing → **DP (G_dp replicas) ×
  A1 (G× fewer passes) multiply**. No change to the reduce or shard logic; the
  disjoint-shard token-space argument (`_shard_calib:558-577`) is unaffected
  (windowing changes *which layers* are hooked per pass, not *which tokens* a
  replica owns). ✅
- **Model sharding** (`device_map="auto"`): A7's per-row `.to(tgt_device)`
  coercion (`covariance_collection.py:396-399` — NOTE: the implementer should trust
  this cite, NOT the stale `test_multigpu_stage3.py:171` comment which still points
  at `covariance_collection.py:310-313`; pre-existing repo line drift, the real
  coercion is `:396-399`) and the accumulator's `.to(device=cur.device)`
  (`activation_hooks.py:1051`) already handle teacher/
  student on different GPUs; A1 inherits this unchanged. Larger G raises per-card
  activation residency → feeds the §4.3 VRAM probe per device. A6 (bigger batch)
  competes with G for the same VRAM; the auto-sizer must budget both.
- VRAM auto-sizing of G must run **per replica** (each replica sees its own GPU
  subset) — compute G inside `_cov_replica_worker` after pinning
  `CUDA_VISIBLE_DEVICES` (`:580-636`), not in the parent.

---

## 5. Per-file change list (implementation, for the builder — NOT done here)

| File | Change |
|------|--------|
| `utils/activation_hooks.py` | **New** `capture_experts(layer_ref, callbacks, *, capture_intermediate: bool)` contextmanager: registers `forward_pre_hook(with_kwargs=True)` that re-derives `(token_idx, top_k_pos)` via the same `one_hot/where`, fires `input`/`gate_up_in` with `sel`, and (if `capture_intermediate`) recomputes `intermediate` in the **native bmm/GEMM shape** to fire `intermediate`; `.remove()` on exit. Assert the module is not already `instrument_experts`-patched (`:1438`). Does NOT touch `instrument_experts` (kept as fallback). |
| `stage3/plugins/covariance_collection.py` | (1) `_collect_covariances`: replace per-layer outer loop (`:449`) with a windowed loop over `_iter_windows(moe_layers, G)`; register A7 `capture_experts` on all layers in the window (student + teacher) instead of one `instrument_experts`; forward once per batch per window; `finalize_layer`+spill each window layer at window end. (2) `_teacher_hidden`: clear per-batch only; populate/consume all G window layers. (3) Add `cov_window_size`/`G` param threaded from config; add `_resolve_cov_window(config, devices)` VRAM auto-sizer — config-resolution shape of `_resolve_cov_replicas` PLUS a new `torch.cuda.mem_get_info()` probe (not a pure mirror; the existing resolver is config-only). (4) Thread G into `_cov_replica_worker` (compute after device-pin) and `_collect_covariances` signature. (5) Fix the STALE docstring `:312-317`. Keep `instrument_experts` path selectable for the fallback/golden-regen-A/B. |
| `stage3/orchestrator.py` | Read `multi_gpu.cov_window_size` (same block as `cov_replicas`, `:93-94`); pass G into both the in-process `_collect_covariances` dispatch (via the `CovarianceCollectionPlugin.collect_covariances` ctx slot, `covariance_collection.py:852-886`) and `run_dp_covariance_collection`. Add a `cov_window_size` ctx slot. |
| `config` (recipe yaml/defaults) | Add `multi_gpu.cov_window_size: auto` (under the existing `multi_gpu` block, alongside `cov_replicas`); document `G=1 ⇒ native per-layer (new golden, not current)`. |
| `tests/golden/stage3/rank_map.*.json` | **Regenerate** under A7 (MOE_REGEN_GOLDEN=1) — the blessed new all-native baseline. Human-gated bless after the §7 rank/PPL gate. |
| `tests/test_multigpu_stage3.py` | Add A1 windowed-vs-per-layer equivalence (atol=0 CPU) + A7-vs-instrument single-layer equivalence (rtol=1e-4). |
| `tasks/` deviation log | Record D-A7: "cov captured from native forward (was Python-loop); golden rebaselined; more inference-faithful." |

---

## 6. Build sequence

1. **A7 capture-only hook** (`capture_experts`) + a unit test asserting, on
   `tiny_model` **single layer**, that A7-captured `C`/`S` match
   `instrument_experts`-captured `C`/`S` to `rtol=1e-4, atol=1e-5` (the
   native-vs-loop bound). Land A7 *behind a config flag*, default off.
2. **A1 windowing** inside `_collect_covariances` (G param, windowed loop,
   per-batch `_teacher_hidden`), with `G=1` reproducing A7-per-layer. Unit test:
   windowed (G=all) vs per-layer (G=1) **byte-identical** (atol=0 CPU).
3. **VRAM auto-sizer** `_resolve_cov_window` + per-replica wiring in
   `_cov_replica_worker`; orchestrator + config threading.
4. **Golden regen + gate** (§7): regenerate `rank_map.*.json` under A7; run the
   rank-diff + PPL non-regression on one real arm; human bless.
5. Flip A7+A1 to default once the gate passes; keep `instrument_experts` as a
   documented fallback (`cov_window_size`/`cov_capture_mode` config).

Each step is independently testable on CPU/`tiny_model`; only step 4's PPL leg
needs a real arm (one GPU).

---

## 7. Test plan

### 7.1 New unit tests (CPU, `tiny_model`, no multi-GPU box)

- **`test_a7_capture_matches_instrument_single_layer`**: for one MoE layer, run the
  current `instrument_experts` collect and the A7 `capture_experts` collect over
  the same seeded batches; assert per-key `C`/`S` `allclose(rtol=1e-4, atol=1e-5)`.
  Isolates the native-vs-loop delta (no upstream drift, single layer). Asserts the
  delta is bounded, justifying the new golden.
- **`test_a1_windowed_equals_perlayer`**: A7 windowed (G=N, single pass) vs A7
  per-layer (G=1); assert **byte-identical** (`atol=0`) on CPU per key. Proves
  windowing adds zero error on top of the native baseline. Runs in-process — **no
  real multi-GPU box needed** (the existing `test_cov_dp_equivalence:358` and
  `test_cov_sharding_equivalence:168` patterns are the templates).
- **`test_a1_window_sizes_consistent`**: G ∈ {1, 2, N} all produce byte-identical
  per-key cov on CPU — window boundary independence.
- **`test_teacher_hidden_window_lifetime`**: assert `_teacher_hidden` holds exactly
  the current window's layers during a batch and is cleared between batches (no
  cross-window leakage) — guards the §4.2 restructure.

### 7.2 Golden regeneration + equivalence gate (the §2.3 gate)

- Regenerate `rank_map.fp32.json` / `rank_map.bf16.json` /
  `rank_map.alpha.{fp32,bf16}.json` (`test_stage3_golden_snapshot.py:155-294`)
  under A7 via `MOE_REGEN_GOLDEN=1`. The snapshot test then pins the **new**
  all-native bytes.
- **Rank-diff report**: compare new vs old `per_layer_ranks` (`rank_map.json`); the
  bless is human-gated. Require the diff to be empty *or* explained-and-approved.
- **PPL non-regression (one real arm, 1 GPU)**: WikiText-2 PPL of the A7-calibrated
  Stage-3 model vs the current-golden-calibrated model within run-to-run noise.
  This is the only leg needing a GPU; everything else is CPU-CI.

### 7.3 Multi-GPU composition (existing harness)

- `test_multigpu_stage3.py` DP-equivalence (`:358`) and reduce
  (`:259, :302`) must still pass with A1 windowing inside each replica (the reduce
  is unchanged; only the per-replica pass count drops).

### 7.4 Regression guard

- Full `pytest max_quality/tests/test_stage3_*.py test_multigpu_stage3.py
  test_smoke_stage3.py` green before merge.

---

## 8. Risks + the quality-validation gate

| Risk | Severity | Mitigation |
|------|----------|------------|
| **Rank flips from native-vs-loop cov delta** (the down_proj key, §2.1, moves even single-layer) | HIGH (output-affecting if it flips ranks) | §7.2 rank-diff + PPL gate; human bless; escalate (RAISE-don't-substitute) if PPL moves. The audit calls this "plausible but unmeasured" — the gate measures it. |
| **A7 down_proj recompute fp-mismatch** if recompute uses `F.linear` loop instead of native `bmm` shape | MED | A7-pre-recompute MUST mirror `model_io.py:982-985` bmm shape, not the per-expert `F.linear` loop, or the down_proj key won't match the native forward. Unit-test asserts byte-match of recomputed `inter` vs a native-forward-exposed `inter` on tiny. |
| **VRAM under-/over-size of G** → OOM or left-on-table speedup | MED | Conservative headroom in `_resolve_cov_window`; G clamped `[1,N]`; OOM → auto-halve-G retry; log chosen G. Measure on first real sharded run (mirror the A6 MEASURED note `orchestrator.py:197-206`). |
| **`_teacher_hidden` RAM blow-up at large G** (cross-cov) | MED | Per-batch clear (§4.2) bounds it to `G × matched-rows`; land A4 (`index_select`) to shrink per-layer cost; G auto-size accounts for host RAM too. |
| **A7/instrument_experts double-active on a module** | LOW | Assert/guard; A7 and `instrument_experts` are mutually exclusive per module. |
| **Stale-docstring / config drift** (`:312-317` already wrong) | LOW | Fix the docstring in the same change; document `G=1 ≠ current golden`. |
| **Someone reads "byte-identical" as "== current artifacts"** | LOW (process) | This doc + the deviation-log D-A7 entry state plainly: byte-identical to the NEW all-native golden, NOT the current one. |

**Gate to ship (all must hold):** (1) A7-vs-loop single-layer cov within rtol=1e-4;
(2) A1 windowed==per-layer atol=0; (3) regenerated golden blessed after rank-diff;
(4) WikiText-2 PPL non-regression on one real arm. If (3)/(4) reveal a quality
move, STOP and escalate — A7+A1 is a *perf+fidelity* change, not a quality trade,
and must not silently become one.

---

## 9. One-paragraph honest summary (for the deviation log)

A7+A1 collects Stage-3 covariance from the model's **real native forward**
(`FactoredExperts` padded `bmm` / fused GEMM) instead of the per-expert Python loop
that `instrument_experts` injects today. It is therefore **not byte-identical to the
current golden artifacts** (which bake in the Python-loop fp reduction order), but it
**is byte-identical to a regenerated all-native golden**, which is *more* faithful to
inference because the deployed model never runs the Python loop. The covariance
*definitions* (AA-SVD Thm 3.2 `C = X_preᵀX_post`, `S = X_postᵀX_post`, gate-only per
D6) are unchanged; only the fp reduction order of the captured activations moves,
within the fp tolerance the pipeline already accepts across DP shards and devices.
A1 windows G layers per pass (`ceil(N/G)` passes vs N, auto-sized from VRAM, composing
multiplicatively with the landed DP/sharding levers) and is proven byte-identical
across window sizes. Ship is gated on a one-time native-vs-loop equivalence +
rank/PPL non-regression check; a quality move there escalates, it does not ship.
