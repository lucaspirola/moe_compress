# PLAN B0 — Calibration capture-hook fix (empty sidecars on Qwen3.6-35B-A3B)

Status: PLANNING ONLY (no production code in this branch). CPU/read-only diagnosis.
Branch: `feat/b0-hook-fix`.

---

## 0. TL;DR

The calib-v2 vLLM run produced 8000 good self-traces but **every** MoE-internal
capture sidecar was empty ("no MoE layers seen" / "wrote 0 entries").

**Root cause (high confidence on the mechanism, partial confidence on which of two
variants dominated):** `Qwen/Qwen3.6-35B-A3B` does **not** map to the
`qwen3_moe` vLLM module that one of our patch hunks instruments. Its declared
architecture is `Qwen3_5MoeForConditionalGeneration` (model_type `qwen3_5_moe`),
which in vLLM is served by the `qwen3_5` module and **reuses
`Qwen3NextSparseMoeBlock` imported from `qwen3_next`** — a *different* MoE block
class than the patched `Qwen3MoeSparseMoeBlock`. Therefore:

* The **model-file** hunk (`vllm/model_executor/models/qwen3_moe.py`, dispatches
  `layer_in` + `block_out`) is **dead code** for this model → explains empty
  `block_outputs` and the empty `layer_input_reservoir` in the stage2 profile.

* The **kernel-level** hooks (`router`, `expert_in`, `expert_out_weighted`,
  `expert_out_unweighted`, `expert_mid`, `linear_in`, `lm_head_in`) are
  dispatched from **model-agnostic** sites (`MoERunner.apply`,
  `TritonExperts.apply`, `LinearBase` subclasses, `LogitsProcessor._get_logits`).
  These *should* fire for Qwen3.6's FusedMoE — yet they too produced zero
  entries. That points to a **second, dominant failure**: the writers'
  `setup()` discovered **zero MoE layers** (`_N_LAYERS == 0`), so no
  accumulators were allocated / no callbacks meaningfully accumulated. The most
  likely reasons (cannot be fully separated without the live model on a GPU):
  - **(B0-a)** `setup()`'s `_resolve_model()` attribute-walk did not find the
    live `nn.Module` under the **vLLM V1 engine** worker layout used by the run,
    so the `named_modules()` discovery loop never ran → `_N_LAYERS=0`; **and/or**
  - **(B0-b)** the model was loaded via **`trust_remote_code=True`** (the driver
    forces it; model_type `qwen3_5_moe`). If the d20260531 wheel's base commit
    (`ad7125a`, vLLM v0.21.0) did **not** ship an in-tree `qwen3_5` module, the
    model loaded the HF repo's custom modeling code, whose MoE block may not be
    vLLM's `FusedMoE`/`MoERunner`/`TritonExperts` at all → no `moe_layer_id`,
    kernel hooks never reached. The run log line "Using TRITON Unquantized MoE
    backend" argues *against* a fully-custom MoE path but does **not** rule out
    B0-a.

**Honesty:** which of B0-a vs B0-b (or both) dominated **cannot be proven from
source alone** — it needs the live module tree on a GPU. The fix is therefore
designed as (1) a best-effort hook-binding hardening that is correct under
*either* variant, plus (2) a **fail-fast assertion** that aborts the run the
moment the first chunk yields empty captures. The fail-fast is independently
valuable and always correct, and is the true validator (Phase-B $1 smoke).

---

## 1. Evidence (file:line, verified against actual source)

### 1.1 Our patch instruments `qwen3_moe.py`, not Qwen3.6's block

* Model-file hunk targets `Qwen3MoeSparseMoeBlock` / `Qwen3MoeModel`:
  `max_quality/patches/vllm_calibration_hooks.patch:10392` (diff header for
  `vllm/model_executor/models/qwen3_moe.py`), with the `layer_in` dispatch at
  patch line **10416–10421** and `block_out` at **10428–10448**, plus the L2
  max-layer hunk in `Qwen3MoeModel.forward` at **10452–10476**.

* Build script confirms the intent: it references "the Qwen3MoeSparseMoeBlock
  dispatch sites": `max_quality/scripts/hf_jobs_build_patched_vllm.sh:62`.

### 1.2 Qwen3.6-35B-A3B's actual architecture

* HF `config.json` for `Qwen/Qwen3.6-35B-A3B`:
  `architectures = ["Qwen3_5MoeForConditionalGeneration"]`, `model_type =
  "qwen3_5_moe"` (fetched from
  `https://huggingface.co/Qwen/Qwen3.6-35B-A3B/raw/main/config.json`).

* vLLM registry (`vllm/model_executor/models/registry.py`, v0.21.0) maps
  `"Qwen3_5MoeForConditionalGeneration": ("qwen3_5",
  "Qwen3_5MoeForConditionalGeneration")` — i.e. module `qwen3_5`, **not**
  `qwen3_moe`. (`Qwen3MoeForCausalLM` is a *separate* key → `qwen3_moe`.)

* `vllm/model_executor/models/qwen3_5.py` (v0.21.0) builds its MoE block as
  `Qwen3NextSparseMoeBlock` **imported from `qwen3_next`**, used via
  `self.mlp = Qwen3NextSparseMoeBlock(...)` and
  `self.moe_layers.append(layer.mlp.experts)`. So Qwen3.6's MoE block class is
  `Qwen3NextSparseMoeBlock`, never the patched `Qwen3MoeSparseMoeBlock`.

* `Qwen3NextSparseMoeBlock.forward` calls `self.experts(...)` where
  `self.experts` is a `FusedMoE` constructed with a `prefix=` arg and **no**
  explicit `moe_layer_id` (verified in `qwen3_next.py`, v0.21.0). The hybrid
  `GatedDeltaNetAttention` ("linear_attention") layers confirm the GDN log line.

### 1.3 Kernel hooks ARE model-agnostic (so they *should* have fired)

* `router` / `expert_in` / `expert_out_weighted` dispatched from
  `MoERunner.apply`: patch **10260–10310**; `_current_layer_idx` set from
  `layer.moe_layer_id`: patch **10271–10272**.
* `expert_out_unweighted` + `expert_mid` from `TritonExperts.apply`: patch
  **10016–10031** and **9915–9931**.
* `moe_layer_id` is a `FusedMoE` auto-incrementing class counter
  (`FusedMoE._next_moe_layer_id`), assigned in `FusedMoE.__init__`
  (`vllm/model_executor/layers/fused_moe/layer.py`, v0.21.0) — present on
  *any* model's FusedMoE, including Qwen3.6's.
* `linear_in` from `LinearBase` subclasses (patch `linear.py` hunk header at
  **10315**), `lm_head_in` from `LogitsProcessor._get_logits` (patch
  `logits_processor.py` header at **10366**) — fully model-agnostic; imatrix's
  primary surfaces.

### 1.4 The decisive symptom: writers saw ZERO MoE layers

* `calibration_reap_scores.setup()` discovers MoE layers by walking
  `model.named_modules()` for modules with **both** `moe_layer_id: int` and
  `num_experts: int > 0`: patch **8600–8615**. If the model isn't resolved or
  the experts module lacks those attrs, `_N_LAYERS = 0`.
* `dump_reap_scores` then logs "no MoE layers seen; skipping dump" when
  `_N_LAYERS == 0`: patch **8705–8709** — the exact run-log message.
* `_resolve_model()` tries only four hard-coded attribute paths (patch
  **8557–8563**), all shaped for older `driver_worker`/`model_runner` layouts;
  none is guaranteed for the V1 engine worker layout used by this run.

**Conclusion from §1.3 + §1.4:** because even the model-agnostic linear/router
hooks came back empty, the dominant failure is at **discovery/wiring**
(`_N_LAYERS=0`), not only the dead `qwen3_moe.py` hunk. The dead model-file hunk
*additionally* zeroes `block_outputs` + `layer_input_reservoir`. Distinguishing
B0-a (bad `_resolve_model` walk) from B0-b (trust_remote_code custom MoE)
requires the live module tree → GPU.

---

## 2. The fix

### 2.1 Hook-binding fix (model-file dispatches: `layer_in` + `block_out`)

The `layer_in` and `block_out` dispatches must be added to the MoE block class
Qwen3.6 actually uses. Two non-exclusive options:

* **Primary (preferred): instrument `Qwen3NextSparseMoeBlock.forward` in
  `vllm/model_executor/models/qwen3_next.py`.** Add the same two dispatch
  blocks the patch currently adds to `Qwen3MoeSparseMoeBlock`:
  - `layer_in` after the `sequence_parallel_chunk` step, before the
    `is_internal_router` branch (the post-chunk `hidden_states` is exactly what
    `self.experts` receives) — mirror patch lines **10408–10421**.
  - `block_out` in both the internal-router and external-gate branches after
    `final_hidden_states = self.experts(...)` — mirror patch lines
    **10428–10448**.
  - `layer_idx` source: `Qwen3NextSparseMoeBlock` has **no** `moe_layer_id`
    field of its own; use `self.experts.moe_layer_id` (the FusedMoE counter),
    matching how `Qwen3MoeSparseMoeBlock` already does it.
  Because `qwen3_5` imports this exact class from `qwen3_next`, one edit covers
  both `Qwen3NextForCausalLM` and `Qwen3_5MoeForConditionalGeneration`.

* **Belt-and-suspenders: also keep the `qwen3_moe.py` hunk** (harmless for other
  models; it remains the correct site for `Qwen3MoeForCausalLM`).

* **L2 max-layer hunk:** the `Qwen3MoeModel.forward` early-exit (patch
  **10452–10476**) also does not apply to Qwen3.6's `Qwen3NextModel.forward`.
  Out of scope for B0 (not a capture, and the run did not use it), but flag it:
  any future `VLLM_CALIB_MAX_LAYER` use on Qwen3.6 needs the same port.

File:line targets to ADD (in the rebuilt wheel's source tree):
`vllm/model_executor/models/qwen3_next.py` → `Qwen3NextSparseMoeBlock.forward`,
immediately after the `if self.is_sequence_parallel: hidden_states =
sequence_parallel_chunk(hidden_states)` line and after each
`final_hidden_states = self.experts(...)`.

### 2.2 Discovery / wiring fix (covers the dominant `_N_LAYERS=0` failure)

This is the higher-priority half because it gates the model-agnostic hooks too.

* **Harden `_resolve_model()`** in every writer that has it (`reap_scores`,
  `imatrix`, `input_cov`, `wanda_scalar_row`, `per_expert_max`,
  `routing_stats`, `router_logits_stats`, `output_reservoir`,
  `block_outputs`, and the `stage2_profile._populate_layer_map_from_llm`).
  Add the **V1-engine** attribute paths actually used by the run, e.g.:
  - `("llm_engine","engine_core","engine_core","model_executor","driver_worker","model_runner","model")`
  - `("llm_engine","engine_core","model_executor","driver_worker","worker","model_runner","model")`
  - a **fallback BFS**: if the four explicit paths miss, walk all attributes
    breadth-first for the first object whose `named_modules()` yields a module
    with `moe_layer_id`. (Bounded depth, stop on first hit.)
  The exact V1 path **must be confirmed on the live GPU object** (print
  `type(llm.llm_engine)` and dir-walk) — see §4. Until then the BFS fallback is
  the robust catch-all.

* **Discovery predicate hardening:** match MoE experts by `moe_layer_id`
  presence as today (patch **8601**), but ALSO treat a module as MoE if it is a
  `FusedMoE` instance even when `num_experts` is exposed under a different attr
  (e.g. `global_num_experts` / `logical_num_experts`). Qwen3.6's FusedMoE may
  surface expert count under a different name; the current strict `num_experts:
  int > 0` filter (patch **8602–8605**) could silently skip it. Read the live
  attr names on GPU before finalizing (§4).

* **Lazy fallback already half-present:** `reap_scores.setup` warns and intends
  "lazily allocate at first router dispatch" when the model can't be resolved
  (patch **8590–8594**), but `_on_expert_out_unweighted` then **raises** on an
  unknown `layer_idx` (patch **8670–8680**) — so lazy allocation never actually
  works. Decide one policy: either (a) make lazy allocation real (grow
  accumulators on first sight of a layer_idx, keyed by raw `moe_layer_id`), or
  (b) keep the hard-fail but guarantee discovery via the BFS fallback above.
  **Recommended: (b)** — discovery is cheaper to make reliable than to retrofit
  dynamic-growth into every writer, and the fail-fast (§2.3) turns any residual
  miss into an immediate, cheap abort.

### 2.3 Fail-fast assertion in the driver (ALWAYS-CORRECT, independent value)

Add a post-first-chunk assertion that aborts loudly if any **enabled** capture
has zero accumulated entries. This is the single most valuable deliverable: it
caps the cost of any future hook/model mismatch at one chunk (~seconds–minutes)
instead of a full 5h run.

**Where:** `max_quality/scripts/build_self_traces_calib_vllm.py`, inside the
chunk loop (`for chunk_start in range(...)` at **line 1989**), guarded to run
**exactly once** after the **first chunk that actually called `llm.generate()`**
(i.e. `gen_chunk` non-empty; the generate call is at **line 2033**, post-output
processing ends ~**line 2045**). Place the check after output processing and
before the per-writer checkpoint blocks (which begin ~**line 2109**).

**What it asserts** (only for captures whose `--capture-*` flag is set):

| Capture flag                | Non-empty predicate (probe the live writer module state)                          |
|-----------------------------|------------------------------------------------------------------------------------|
| `--capture-reap-scores`     | `_reap._N_LAYERS > 0` AND `any(t.sum() > 0 for t in _reap._REAP_TOKEN_COUNTS.values())` |
| `--capture-input-covariance`| `len(_icov`-accumulator covariance dict`) > 0` (≥1 (layer,expert,matrix) entry)    |
| `--capture-imatrix`         | imatrix accumulator has ≥1 recorded linear/expert channel-sum entry                |
| `--capture-stage2-profile`  | `_s2p._state.ream_acc._gate_gram` non-empty OR cov_acc.covariance non-empty        |
| `--capture-wanda-scalar-row`| ≥1 (layer,expert) row accumulated                                                  |
| `--capture-per-expert-max`  | ≥1 expert max entry                                                                |
| `--capture-routing-stats`   | ≥1 layer routing-count entry                                                       |
| `--capture-router-logits-stats` | ≥1 layer stats entry                                                           |
| `--capture-output-reservoir`| ≥1 reservoir sample                                                                |
| `--capture-block-outputs`   | ≥1 block_out dispatch counted                                                      |
| `--capture-layer-input-reservoir` (with stage2) | ≥1 non-empty per-layer reservoir buffer                        |

Each writer must expose a tiny **public introspection** helper (e.g.
`captured_entry_count() -> int`) so the driver does not reach into private
`_state` (cleaner + testable). The driver collects `{name: count}` for every
enabled capture; if **any enabled** capture reports `0`, it:
1. logs an ERROR naming every empty capture, the resolved model class, and a
   hint ("hooks did not bind to this model's MoE path — see PLAN_B0_HOOK_FIX);
2. raises `SystemExit(2)` **before** writing any checkpoint or continuing,
   so the run aborts at chunk 0 with the partial JSONL intact.

A `--allow-empty-captures` escape hatch (default off) lets an operator bypass
the abort for debugging.

**Subtlety (TEACHER_FORCED-only first chunk):** if the first chunk is all-TF
(`gen_chunk` empty, no `generate()`), captures legitimately have zero entries.
The assertion must trigger only **after the first chunk with a non-empty
`gen_chunk`**. Track a `first_gen_chunk_checked: bool`.

### 2.4 Wheel rebuild — REQUIRED

§2.1 and §2.2 change vLLM source (model-file dispatch + writer `setup()` /
introspection helpers), so the wheel **must be rebuilt**.

* The fix lands in BOTH patches:
  - `vllm_calibration_hooks.patch`: add the `qwen3_next.py` hunk; harden every
    writer's `_resolve_model()` + discovery predicate; add
    `captured_entry_count()` to each writer.
  - `vllm_calibration_stage2_profile.patch`: harden
    `_populate_layer_map_from_llm` + add `captured_entry_count()`.
* Rebuild via `max_quality/scripts/hf_jobs_build_patched_vllm.sh`. Required
  bumps in that script:
  - new `wc -l` / `md5sum` expected values for both patches (lines **84**, **91**);
  - the branch/tag the `curl` pulls from (lines **80**, **87**) → point at the
    branch carrying the B0 fix (e.g. `feat/b0-hook-fix` or a new
    `calib-v2-b0` tag);
  - README patch-hash lines **228–229**.
* After upload, bump `VLLM_WHEEL_FILE` in the run config to the new
  `d<DATE>` wheel (the d20260531 wheel is stale for this fix).
* **Base-commit check (resolves B0-b):** before rebuild, confirm whether
  v0.21.0 @ `ad7125a` actually ships `vllm/model_executor/models/qwen3_5.py`
  and `qwen3_next.py`. If it does NOT, `git apply` of the new `qwen3_next.py`
  hunk will fail — meaning Qwen3.6 was loaded via `trust_remote_code` custom
  code at runtime, and the model-file hook approach is moot for that wheel
  (we'd need a newer vLLM base OR to drop trust_remote_code so the in-tree
  `qwen3_5` path is used). This single `git apply --check` is the cheapest test
  that disambiguates B0-a vs B0-b. Add it to the build script's Phase 5.

---

## 3. Re-capture strategy (after the fixed wheel is built)

Per the failure memo, a full 5h generation re-run is NOT required: the captures
observe the MoE forward over the saved 8000 prompts. A **forward-only
teacher-forced replay** of the existing `self_traces_*.jsonl` prompts+completions
(no sampling) harvests every sidecar in ~30–60 min GPU. The driver already
supports TEACHER_FORCED entries (§ `_synth_teacher_forced_rows`, driver
**line 521 / 2020**) — but note TF rows currently **skip** `generate()`. The
replay path must run the prompts through a forward that triggers the MoE hooks
(a `generate(max_new_tokens=1)` or a dedicated forward-replay mode), NOT the
synth-only TF path. Specify the replay mode as a follow-up; out of scope for the
B0 plan beyond noting it.

---

## 4. Validation path (Phase B, on GPU — the real validator)

Cheap GPU smoke (~$1, ~50 prompts) that MUST be run before any full re-capture:

1. Launch the fixed wheel on 1×H200 spot with the model and **all** the
   `--capture-*` flags the probe needs (reap_scores, input_cov, imatrix,
   wanda, stage2_profile, per_expert_max, routing_stats, router_logits_stats,
   output_reservoir, block_outputs), `--chunk-size 50`, ~50–100 prompts.
2. **First, dump the live module tree** (one-off): print
   `type(llm.llm_engine).__module__`, the resolved model class, and the first
   few `named_modules()` entries that have `moe_layer_id` + the attr name the
   FusedMoE exposes its expert count under. This **empirically confirms B0-a vs
   B0-b** and pins the exact `_resolve_model` path + discovery attr. Feed back
   into §2.2 if the BFS fallback wasn't sufficient.
3. The §2.3 fail-fast assertion is the pass/fail gate: the smoke **passes** iff
   the run does NOT abort at chunk 0, i.e. every enabled capture has nonzero
   entries after the first generate chunk.
4. Spot-check magnitudes: `reap_scores` has `_N_LAYERS ==` model's MoE-layer
   count (Qwen3.6: confirm vs config `num_hidden_layers` minus dense/linear
   layers), `input_cov` has ~`n_layers * n_experts * 2` (gate+down) entries,
   imatrix has ≥1 entry per linear surface.

Only after the smoke passes do we run the forward-only re-capture (§3).

---

## 5. Test plan

### 5.1 CPU-testable now (no GPU, no live model)

* **Fail-fast assertion logic (highest value).** Unit-test the driver's
  post-first-chunk check with **mocked writer modules**: inject fake
  `captured_entry_count()` returning `0` for one enabled capture and `>0` for
  others; assert the driver raises `SystemExit(2)` and the error message names
  the empty capture. Second case: all `>0` → no raise. Third case: first chunk
  all-TF (`gen_chunk` empty) → assertion deferred, no raise. Pure-Python; no
  torch/CUDA. (NO monkeypatching production code — pass the writer modules in
  via a small injectable registry / helper, per repo policy.)
* **Discovery predicate (unit-testable).** Build a tiny fake `nn.Module` tree
  with one module carrying `moe_layer_id` + expert-count attr nested under a
  V1-style attribute chain; assert the hardened `_resolve_model()` + discovery
  loop finds it and the BFS fallback finds it when the explicit paths miss.
  Run on CPU with `torch.nn.Module` stubs (no model weights).
* **`captured_entry_count()` helpers.** Per-writer unit test: empty accumulator
  → 0; after a synthetic dispatch → >0. Reuse the existing per-writer smoke
  tests in the patch's `tests/test_calibration_*_smoke.py` set as the harness.
* **Patch base-commit check.** `git apply --check` of the new `qwen3_next.py`
  hunk against a v0.21.0 checkout — confirms the dispatch site exists in the
  wheel's base (the B0-a/B0-b disambiguator, §2.4). CPU-only.

### 5.2 Requires GPU (Phase B)

* End-to-end hook firing on the live Qwen3.6 module tree (the §4 smoke). This is
  the only place the real "did the hooks bind?" question is answered. The CPU
  tests prove the *logic*; the GPU smoke proves the *binding*.

---

## 6. Confidence statement (honesty)

* **High confidence:** Qwen3.6-35B-A3B is `Qwen3_5MoeForConditionalGeneration` /
  `qwen3_5` and uses `Qwen3NextSparseMoeBlock`, NOT the patched
  `Qwen3MoeSparseMoeBlock` → the model-file `layer_in`/`block_out` hunk is dead
  code for it (explains empty `block_outputs` + `layer_input_reservoir`).
  [config.json + registry.py + qwen3_5.py all verified against source.]
* **High confidence:** the run-log "no MoE layers seen" means `_N_LAYERS == 0`,
  i.e. writer discovery found no FusedMoE with `moe_layer_id`+expert-count —
  this is what zeroed even the model-agnostic kernel hooks.
* **Medium / NOT proven from source:** *why* discovery found zero layers —
  B0-a (`_resolve_model` walk wrong for the V1 engine layout) vs B0-b
  (trust_remote_code custom MoE without `moe_layer_id`). **This cannot be
  decided without the live model on a GPU.** The plan handles both: hardened
  discovery + BFS fallback (B0-a), the base-commit `git apply --check` and the
  GPU module-tree dump (B0-b), and the fail-fast assertion as the universal,
  always-correct safety net.
