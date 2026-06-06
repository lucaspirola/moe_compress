# PLAN B0 — Calibration capture-hook fix (empty sidecars on Qwen3.6-35B-A3B)

Status: PLANNING ONLY (no production code in this branch). CPU/read-only diagnosis.
Branch: `feat/b0-hook-fix`.

All file:line evidence in this revision is re-verified against the pinned vLLM
clone `/tmp/vllm_b0 @ ad7125a` (v0.21.0), our two patches under
`max_quality/patches/`, and the driver/launch scripts under
`max_quality/scripts/`.

---

## 0. TL;DR

The calib-v2 vLLM run produced 8000 good self-traces but **every** MoE-internal
capture sidecar was empty ("no MoE layers seen" / "wrote 0 entries").

**Root cause is now CLEAR — two source-confirmed, independent defects (no GPU
needed to decide):**

* **C1 (DOMINANT): EngineCore runs in a subprocess by default**, so callbacks
  registered in the driver process never fire and `_resolve_model()` finds no
  model. This zeroes **every** hook — including the model-agnostic
  linear/router/lm_head ones — and forces `_N_LAYERS=0`. Fixed by running the
  engine **in-process** via `VLLM_ENABLE_V1_MULTIPROCESSING=0`. **No wheel
  rebuild needed for C1.**

* **C2: the discovery predicate is wrong** — it gates on
  `getattr(module, "num_experts")`, but `FusedMoE` **never** sets
  `self.num_experts`. The predicate is therefore always False for **any** model,
  so even with C1 fixed, `_N_LAYERS` would still be 0. Fixed by switching the
  predicate to `global_num_experts`. **Needs wheel rebuild.**

A **secondary** model-file gap (M3) additionally zeroes `block_outputs` +
`layer_input_reservoir`: the `layer_in`/`block_out` dispatch was added to
`Qwen3MoeSparseMoeBlock`, but Qwen3.6 uses `Qwen3NextSparseMoeBlock`. Porting
the hook (M3) restores those two captures. **Needs wheel rebuild.**

The prior plan's "needs GPU to decide B0-a vs B0-b" hedge, the `trust_remote_code`
hypothesis, and the inert BFS fallback are **dropped** — all disproved from
source (see §1.6).

---

## 1. Evidence (file:line, re-verified against /tmp/vllm_b0 @ ad7125a)

### 1.1 C1 — EngineCore is a subprocess by default

* **Default is multiprocessing ON.** `vllm/envs.py:129`
  (`VLLM_ENABLE_V1_MULTIPROCESSING: bool = True` type-decl) and
  `vllm/envs.py:1112-1114` (env-parse lambda:
  `bool(int(os.getenv("VLLM_ENABLE_V1_MULTIPROCESSING", "1")))`). Verified.
* **`model_executor` is set ONLY in-process.** `vllm/v1/engine/llm_engine.py:122-124`:
  ```python
  if not multiprocess_mode:
      # for v0 compatibility
      self.model_executor = self.engine_core.engine_core.model_executor
  ```
  and `:148` passes `multiprocess_mode=envs.VLLM_ENABLE_V1_MULTIPROCESSING` into
  `__init__`. Verified.
* **The driver never overrides it.** `max_quality/scripts/build_self_traces_calib_vllm.py`
  constructs the engine via `_load_teacher_vllm` → `LLM(**kwargs)`
  (`:272`); it passes no `enable_multiprocessing`-style override and sets no
  `VLLM_ENABLE_V1_MULTIPROCESSING` env (grep: the only env writes are the
  `VLLM_CALIB_CAPTURE_*` block at `:1166+` and the cache/jobs block at
  `:192-200`). Verified — no override.

**Mechanism:** with MP=1, the worker (where `MoERunner.apply`, `TritonExperts.apply`,
`LinearBase`, `LogitsProcessor`, and the model forward run, hence where
`dispatch()` executes) lives in a **separate subprocess** from the driver where
the writers' `register_callback()` ran. The worker's `_CALLBACKS` dict is empty
→ every dispatch is a no-op. Simultaneously, the driver's `_resolve_model()`
walk has no `model_executor` to find (it is `None` unless `not multiprocess_mode`)
→ `_N_LAYERS=0` → "no MoE layers seen". This explains why **even the
model-agnostic** linear/router/lm_head hooks produced zero entries.

### 1.2 C2 — discovery predicate keys off a non-existent attribute

* **Writers gate discovery on `num_experts`.** In
  `max_quality/patches/vllm_calibration_hooks.patch`, every writer's `setup()`
  discovery loop does:
  ```
  +            num_experts = getattr(module, "num_experts", None)
  +                    and isinstance(num_experts, int) and num_experts > 0):
  ```
  Verified at patch lines **5451/5453, 7352/7354, 7817/7819, 8259/8261,
  8602/8604, 8987/8989, 9420/9422, 10686/10688**, plus the standalone
  `n_exp = getattr(module, "num_experts", None)` at **6701**. (These match the
  review's cited 5453 / 6701 / 7354 / 7819 / 8261 / 8604 / 8989 / 9422 / 10688.)
* **`FusedMoE` never sets `self.num_experts`.** In
  `/tmp/vllm_b0/vllm/model_executor/layers/fused_moe/layer.py`, `FusedMoE.__init__`
  sets only:
  - `self.moe_layer_id` (`:298`, auto-increment class counter)
  - `self.global_num_experts = num_experts + num_redundant_experts` (`:338`)
  - `self.logical_num_experts = num_experts` (`:339`)
  - `self.local_num_experts = local_num_experts` (`:414`)
  - `self.moe_layer_id` block at `:297-299`.

  `grep -n "self\.num_experts" layer.py` → **no match** (verified). So the
  predicate `getattr(module, "num_experts", None)` is **always None** → the
  `isinstance(..., int) and ... > 0` guard is always False → the module is never
  counted → `_N_LAYERS=0` for **any** MoE model, not just Qwen3.6.

### 1.3 M3 — model-file hook on the wrong block class

* Patch model-file hunk targets `Qwen3MoeSparseMoeBlock` /
  `Qwen3MoeModel` (`vllm/model_executor/models/qwen3_moe.py`), diff header at
  patch **10392**, `layer_in` dispatch **10416-10421**, `block_out`
  **10428-10448**, L2 max-layer hunk **10452-10476**.
* Qwen3.6-35B-A3B is `Qwen3_5MoeForConditionalGeneration` →
  `/tmp/vllm_b0/vllm/model_executor/models/registry.py:551-554`:
  ```python
  "Qwen3_5MoeForConditionalGeneration": (
      "qwen3_5",
      "Qwen3_5MoeForConditionalGeneration",
  ),
  ```
  → module `qwen3_5`, **not** `qwen3_moe`. Verified.
* `qwen3_5` reuses `Qwen3NextSparseMoeBlock` from `qwen3_next`. Its
  `forward` is `/tmp/vllm_b0/vllm/model_executor/models/qwen3_next.py:175-201`:
  - post-`sequence_parallel_chunk` hidden_states at `:181`
    (`hidden_states = sequence_parallel_chunk(hidden_states)`),
  - `final_hidden_states = self.experts(...)` in **both** the
    `is_internal_router` branch (`:185-187`) and the external-gate branch
    (`:191-193`).
  - The block has **no** `moe_layer_id` field of its own; the FusedMoE counter
    lives on `self.experts.moe_layer_id` (layer.py:298). Verified.

So the patched `Qwen3MoeSparseMoeBlock` dispatch is dead code for Qwen3.6 →
empty `block_outputs` + empty `layer_input_reservoir`.

### 1.4 stage2_profile uses NAME-string discovery → needs ONLY C1

* `max_quality/patches/vllm_calibration_stage2_profile.patch:255-291`
  (`_populate_layer_map_from_llm`) discovers layers by **module-name string**:
  it walks `model.named_modules()`, requires `"layers"` in the name and
  (`"moe"` or `"experts"`) in `name.lower()`, then parses the digit token as the
  layer index. It does **NOT** use the `num_experts` predicate (the only
  `num_experts` references in that patch are `_state.ream_acc.num_experts`
  assignments at `:213` and `:806`, not discovery). Verified.
* Therefore stage2_profile is broken **only** by C1 (no resolved model under
  MP=1 → the walk's two `attr_path`s at `:259-261` both return None → early
  `return`). Once C1 makes `model_executor` resolvable in-process, stage2's
  name-string walk succeeds **without** C2. Scope: stage2 needs **C1 only**.

### 1.5 In-process delegation chain works once MP=0 (UniProcExecutor)

With MP=0 and tp=1 (single-GPU calibration driver), the executor is
`UniProcExecutor`. The `_resolve_model()` walk reaches the live model via:
* `vllm/v1/executor/uniproc_executor.py:48` — `self.driver_worker =
  WorkerWrapperBase(rpc_rank=0)` (the worker wrapper).
* `vllm/v1/worker/worker_base.py:319-320` — `def __getattr__(self, attr):
  return getattr(self.worker, attr)` (delegates any attr to the real worker).
* `vllm/v1/worker/gpu_worker.py:167` — the worker exposes the model as
  `self.model_runner.model` (used at `:167` for buffer save). Verified.

So under MP=0, `llm_engine.model_executor.driver_worker` (→ `WorkerWrapperBase`
→ `__getattr__` → real GPU worker) → `.model_runner.model` resolves the model in
the **same process** as setup + dispatch + dump. The existing `_resolve_model`
attribute-path approach works; **no BFS fallback is needed** (deleted, §1.6).

### 1.6 Disproved hypotheses (dropped from the plan)

* **B0-b (`trust_remote_code` custom MoE):** disproved — registry.py:551-554
  maps the architecture to the **in-tree** `qwen3_5` module; the in-tree class
  is used, so the model's MoE is vLLM's `FusedMoE`. No custom modeling code.
* **B0-a "which V1 attr path?" uncertainty:** resolved — §1.5 pins the exact
  in-process chain. Under MP=0 the existing path resolves.
* **BFS fallback:** deleted. It could never have helped under MP=1 (the model
  object does not exist in the driver process regardless of how hard you walk),
  and is unnecessary under MP=0 (§1.5).

---

## 2. The fix

### 2.1 C1 — run EngineCore in-process (DOMINANT; NO wheel rebuild)

Set `VLLM_ENABLE_V1_MULTIPROCESSING=0` so `multiprocess_mode` is False, which
(a) makes `llm_engine.model_executor` resolvable (llm_engine.py:122-124) and
(b) runs the model forward + all dispatch sites in the **same process** as the
writers' registered callbacks.

**Where (both, belt-and-suspenders):**

1. **Driver — `max_quality/scripts/build_self_traces_calib_vllm.py`.** Set
   `os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"` **before any vLLM
   import**. The import is `from vllm import LLM` at **:247**, inside
   `_load_teacher_vllm` (`:216`). The safest site is **module-level**, in the
   top-of-file env block alongside the existing `MAX_JOBS`/`NVCC_THREADS`/
   `VLLM_CACHE_ROOT` writes (currently inside the early setup, `:192-200`) — but
   it must be a plain top-level statement guaranteed to run before `import vllm`
   anywhere. Use a hard set (`os.environ[...] = "0"`), **not** `setdefault`, so a
   stale shell export cannot re-enable MP. Add a one-line `log.info` recording
   that calibration forces in-process mode.
2. **Launch — `max_quality/scripts/run_calib_vllm.sh`.** Add
   `export VLLM_ENABLE_V1_MULTIPROCESSING=0` to the export block at
   **:40-50** (next to `MAX_JOBS`/`NVCC_THREADS`/`VLLM_CACHE_ROOT`). Redundant
   with (1) but documents the requirement at the launch surface and covers any
   alternate entry point.

**Notes:**
* In-process is correct for single-GPU (tp=1) calibration; the driver is
  single-GPU per the calib-v2 memo. It also **improves determinism** (no
  cross-process scheduling nondeterminism in capture).
* C1 fixes **all** model-agnostic hooks (router/linear/lm_head/expert) AND the
  `_resolve_model`/`_N_LAYERS` discovery path for `reap_scores`, `imatrix`,
  `input_cov`, `wanda_scalar_row`, `per_expert_max`, `routing_stats`,
  `router_logits_stats`, `output_reservoir`, AND stage2_profile (§1.4).
* **C1 needs no wheel rebuild** — it is an environment change only.

### 2.2 C2 — fix the discovery predicate (needs wheel rebuild)

Change the discovery predicate from `num_experts` to `global_num_experts` at
**every** writer `setup()` site in `vllm_calibration_hooks.patch`. The attribute
`global_num_experts` is set unconditionally in `FusedMoE.__init__`
(layer.py:338) for any MoE model, so it is the correct, model-agnostic key.

Sites to change (verified present in our patch; each is a
`getattr(module, "num_experts", None)` read on the line above plus the
`isinstance(...)>0` guard, except 6701 which is a standalone read):

| Patch line(s)            | Writer (by setup block)        |
|--------------------------|--------------------------------|
| 5451 / 5453              | (writer A)                     |
| 6701 (`n_exp`)           | (standalone expert-count read) |
| 7352 / 7354              | (writer B)                     |
| 7817 / 7819              | (writer C)                     |
| 8259 / 8261              | (writer D)                     |
| 8602 / 8604              | reap_scores                    |
| 8987 / 8989              | (writer E)                     |
| 9420 / 9422              | (writer F)                     |
| 10686 / 10688            | (writer G)                     |

For each: replace the `getattr(module, "num_experts", None)` read with
`getattr(module, "global_num_experts", None)` (keep the existing
`isinstance(x, int) and x > 0` guard and the cross-layer
`n_experts_seen` consistency check unchanged — they remain correct once the
value is non-None). Keep the local variable name or rename to `n_experts`
consistently; do not change downstream uses other than the attribute string.

> Note: the **dispatch** path already uses the correct attr —
> `vllm_calibration_hooks.patch:10279` reads `layer.logical_num_experts` to slice
> router logits. C2 is strictly the **discovery/setup** predicate. Optionally use
> `logical_num_experts` (the true routed count) instead of `global_num_experts`
> for the discovery count to match the dispatch convention; either resolves the
> always-False bug. Recommend `global_num_experts` for "is this a FusedMoE?"
> presence and note that count semantics (with/without redundant experts) do not
> affect discovery (we only need `> 0`).

### 2.3 M3 — port the block-level hook to Qwen3NextSparseMoeBlock (needs rebuild)

Add a new hunk to `vllm_calibration_hooks.patch` instrumenting
`Qwen3NextSparseMoeBlock.forward` in
`vllm/model_executor/models/qwen3_next.py` (the class `qwen3_5` reuses, hence
covers Qwen3.6 and `Qwen3NextForCausalLM` with one edit). Mirror the existing
`Qwen3MoeSparseMoeBlock` dispatch:

* `layer_in` dispatch: insert immediately **after**
  `qwen3_next.py:181` (the `hidden_states = sequence_parallel_chunk(hidden_states)`
  line, inside the `if self.is_sequence_parallel:` block — or, to also cover the
  non-SP path, after the `hidden_states = hidden_states.view(-1, hidden_dim)` at
  `:179` and before the `is_internal_router` branch). Dispatch the post-shape
  `hidden_states` (exactly what `self.experts` receives). Mirror patch
  **10408-10421**.
* `block_out` dispatch: insert after **each**
  `final_hidden_states = self.experts(...)` — both the `is_internal_router`
  branch (`:185-187`) and the external-gate branch (`:191-193`). Mirror patch
  **10428-10448**.
* `layer_idx` source: `self.experts.moe_layer_id` (the FusedMoE counter,
  layer.py:298) — `Qwen3NextSparseMoeBlock` has no `moe_layer_id` of its own,
  same convention as `Qwen3MoeSparseMoeBlock`.

**Keep** the existing `qwen3_moe.py` hunk (still correct for
`Qwen3MoeForCausalLM`). M3 restores `block_outputs` + `layer_input_reservoir`
for Qwen3.6 only; the kernel/linear captures are restored by C1+C2 independent
of M3.

**L2 max-layer hunk (out of scope, flagged):** the `Qwen3MoeModel.forward`
early-exit (patch 10452-10476) does not apply to Qwen3.6's `Qwen3NextModel`. Not
a capture and unused by the run; any future `VLLM_CALIB_MAX_LAYER` use on
Qwen3.6 needs the same port.

**Cheap CI:** add `git apply --check` of the new `qwen3_next.py` hunk against a
v0.21.0 (`ad7125a`) checkout to the build script (Phase 5) — confirms the
dispatch site exists in the wheel base. (Not a B0-a/B0-b disambiguator anymore —
that question is already resolved §1.6 — just a build-hygiene gate.)

### 2.4 Fail-fast assertion in the driver (M1 — keep; always-correct)

Add a post-first-chunk assertion that aborts loudly if any **enabled** capture
has zero accumulated entries, capping the cost of any future hook/model mismatch
at one chunk.

**Where:** `max_quality/scripts/build_self_traces_calib_vllm.py`, inside the
chunk loop (`for chunk_start in range(...)` at **:1989**). Place it **after**
the session-elapsed log block (`:2092-2097`) and **before** the
`block_outputs` `close_subset` gate (`:2105+`, `_bo.close_subset()` at `:2111`).
The `generate()` call is at **:2033**, gated on `if gen_chunk:` (`:2027`).

**Run-once / TF-edge guard:** track `first_gen_chunk_checked: bool`. Run the
assertion only after the **first chunk with a non-empty `gen_chunk`** (i.e. one
that actually called `llm.generate()` at :2033). An all-TF first chunk
(`gen_chunk` empty) legitimately has zero captures → defer.

**What it asserts (only for captures whose `--capture-*` flag is set):** each
writer exposes a **public** `captured_entry_count() -> int` (NO private `_state`
reach-in, NO monkeypatch — per repo policy). The driver builds `{name: count}`
for every enabled capture; if **any enabled** capture reports `0`:
1. log ERROR naming every empty capture, the resolved model class, and the hint
   "hooks did not bind to this model's MoE path — see PLAN_B0_HOOK_FIX";
2. raise `SystemExit(2)` **before** any checkpoint/close_subset, so the run
   aborts at chunk 0 with partial JSONL intact.

`--allow-empty-captures` (default off) bypasses the abort for debugging.

The `captured_entry_count()` helpers are added in the **patch** (one per writer
+ stage2), so they ride the wheel rebuild; the driver-side check is pure-Python
(no rebuild for the driver edit itself).

### 2.5 Wheel rebuild — REQUIRED for C2 + M3 + entry-count helpers

C1 (env) needs **no** rebuild. C2 (§2.2), M3 (§2.3), and the per-writer
`captured_entry_count()` helpers (§2.4) change patch source → rebuild via
`max_quality/scripts/hf_jobs_build_patched_vllm.sh`:

* Both patches change:
  - `vllm_calibration_hooks.patch`: C2 predicate at the 9 sites (§2.2) + the new
    `qwen3_next.py` M3 hunk + `captured_entry_count()` per writer.
  - `vllm_calibration_stage2_profile.patch`: `captured_entry_count()` for
    stage2 (its discovery is already name-string based; **no C2 change**).
* Required bumps in `hf_jobs_build_patched_vllm.sh`:
  - new `wc -l` + `md5sum` expected values — `vllm_calibration_hooks.patch` at
    **:84** (currently `11272 lines, MD5 9aaf47abd4c44bf2b2a62edd7e28014f`),
    `vllm_calibration_stage2_profile.patch` at **:91** (currently
    `900 lines, MD5 176e1bc4ee08d32d0b2a12dc73b4fec4`);
  - the `curl` source branch/tag at **:80** and **:87** (currently
    `calib-v2-stage2-profile-complete`) → point at the branch carrying the B0
    fix (e.g. `feat/b0-hook-fix` or a new `calib-v2-b0` tag);
  - README patch-hash lines **:228-229** (the `— NNNN lines, MD5 ...` strings);
  - add the `git apply --check` of the `qwen3_next.py` hunk to Phase 5 (§2.3).
* After upload, bump `VLLM_WHEEL_FILE` in the run config to the new `d<DATE>`
  wheel (the d20260531 wheel is stale for C2/M3).

---

## 3. Re-capture strategy (after the fixed wheel + C1 env)

A full 5h generation re-run is NOT required: captures observe the MoE forward
over the saved 8000 prompts. A **forward-only teacher-forced replay** of the
existing `self_traces_*.jsonl` prompts+completions harvests every sidecar in
~30-60 min GPU. The driver supports TEACHER_FORCED rows
(`_synth_teacher_forced_rows`, driver ~`:521` / `:2020`) but those rows **skip**
`generate()`; the replay must run a forward that triggers the MoE hooks
(`generate(max_new_tokens=1)` or a dedicated forward-replay mode), NOT the
synth-only TF path. **The re-capture run must also export
`VLLM_ENABLE_V1_MULTIPROCESSING=0`** (C1) — without it the replay captures empty
too. Specify the replay mode as a follow-up; out of scope beyond noting it.

---

## 4. Validation path (Phase B, on GPU)

Cheap GPU smoke (~$1, ~50-100 prompts) before any full re-capture, **with
`VLLM_ENABLE_V1_MULTIPROCESSING=0`**:

1. Launch the fixed wheel on 1×H200 spot, the model, and **all** the
   `--capture-*` flags the probe needs, `--chunk-size 50`.
2. The §2.4 fail-fast assertion is the pass/fail gate: the smoke **passes** iff
   the run does NOT abort at chunk 0 — i.e. every enabled capture has nonzero
   entries after the first generate chunk.
3. Spot-check magnitudes: `reap_scores._N_LAYERS ==` Qwen3.6's MoE-layer count
   (confirm vs config: total layers minus dense/linear-attention layers);
   `input_cov` ~`n_layers * n_experts * 2` (gate+down) entries; imatrix ≥1 per
   linear surface; `block_outputs`/`layer_input_reservoir` non-empty (validates
   M3).

(The prior plan's "first dump the live module tree to decide B0-a vs B0-b" step
is removed — §1.6 settled it from source. An optional one-line
`log.info(type(llm.llm_engine).__module__, resolved model class)` at startup is
fine as a sanity breadcrumb, not a decision gate.)

---

## 5. Test plan

### 5.1 CPU-testable now (no GPU, no live model)

* **Fail-fast assertion logic (highest value).** Unit-test the driver
  post-first-chunk check with **injected** fake writer modules exposing
  `captured_entry_count()` (one returns 0, others >0) → assert `SystemExit(2)`
  and the error names the empty capture. Cases: all >0 → no raise; first chunk
  all-TF (`gen_chunk` empty) → deferred, no raise. Pure-Python; no torch/CUDA.
  NO monkeypatching production code — pass writers via a small injectable
  registry/helper.
* **C2 predicate (unit-testable).** Build a tiny fake `nn.Module` carrying
  `moe_layer_id` + `global_num_experts` (and crucially **no** `num_experts`) →
  assert the hardened discovery loop counts it as 1 MoE layer; assert the old
  `num_experts` predicate would have missed it. CPU `torch.nn.Module` stubs.
* **`captured_entry_count()` helpers.** Per-writer: empty accumulator → 0; after
  a synthetic dispatch → >0. Reuse `tests/test_calibration_*_smoke.py`.
* **M3 patch apply check.** `git apply --check` of the new `qwen3_next.py` hunk
  against a v0.21.0 (`ad7125a`) checkout. CPU-only.

### 5.2 Requires GPU (Phase B)

* End-to-end hook firing on the live Qwen3.6 module tree under MP=0 (the §4
  smoke). The CPU tests prove the *logic*; the GPU smoke proves the *binding*
  (C1 in-process wiring + C2 discovery + M3 block hook all firing together).

---

## 6. Confidence statement

* **CONFIRMED (source) — C1:** `VLLM_ENABLE_V1_MULTIPROCESSING` defaults True
  (envs.py:129, :1112-1114); `model_executor` set only `if not multiprocess_mode`
  (llm_engine.py:122-124, :148); driver never overrides. Engine runs in a
  subprocess → callbacks/`_resolve_model` in the driver process are inert →
  every hook empty, `_N_LAYERS=0`. Fix = env `=0`, no rebuild.
* **CONFIRMED (source) — C2:** `FusedMoE` never sets `self.num_experts`
  (layer.py — only `global_num_experts`:338 / `logical_num_experts`:339 /
  `local_num_experts`:414 / `moe_layer_id`:298); the patch's
  `getattr(module,"num_experts")` predicate (9 sites) is always False → always
  `_N_LAYERS=0`. Fix = use `global_num_experts`, needs rebuild.
* **CONFIRMED (source) — M3:** Qwen3.6 → `qwen3_5` (registry.py:551-554) →
  `Qwen3NextSparseMoeBlock` (qwen3_next.py:175-201), not the patched
  `Qwen3MoeSparseMoeBlock`. Port the block hook (uses `self.experts.moe_layer_id`),
  needs rebuild. Restores `block_outputs` + `layer_input_reservoir` only.
* **DISPROVED (source):** B0-b `trust_remote_code` (in-tree `qwen3_5` is used);
  B0-a "unknown V1 attr path" (UniProcExecutor delegation pinned §1.5,
  uniproc_executor.py:48 → worker_base.py:319-320 → gpu_worker.py:167); the BFS
  fallback (deleted — useless across the process boundary, unneeded in-process).
* **Scope:** stage2_profile needs **C1 only** (name-string discovery,
  stage2 patch :255-291); all kernel/linear writers need **C1 + C2**;
  `block_outputs`/`layer_input_reservoir` additionally need **M3**.
