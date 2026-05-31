# PLAN — Stage-6 eval optimizations + bs-invariance correctness fix

Base: `origin/main` @ `0ae66896942cd8ecbca1cf62c81aaec60f889590` (verified via `git rev-parse origin/main`).
Branch: `plan/stage6-opt`. **Plan only — no production edits in this branch.**
All file:line citations below were read from real blobs (`git show origin/main:<path>`), not from the working tree, and re-verified against the measurement-pass numbers (drift corrected inline where found).

> NOTE ON PATHS: the spec referenced `max_quality/src/moe_compress/configs/...`,
> `tests/golden/stage6/...`, and `qwen36_35b_a3b_30pct.yaml:~570`. The actual
> on-disk locations on `origin/main` are:
> - config: `max_quality/configs/qwen36_35b_a3b_30pct.yaml` (NOT under `src/`); `gen_batch_size: 8` is at **line 570** (exact match).
> - golden: `max_quality/tests/golden/stage6/stage6_eval.json` (NOT under `src/moe_compress/tests/`).
> - the plugin/tool sources live under `max_quality/src/moe_compress/`:
>   - `tools/eval_harness.py`, `stage6/plugins/{humaneval,math500,wikitext_ppl,teacher_provider}.py`, `stage6alt/plugins/bpt_metric.py`.
> - tests live under `max_quality/tests/`.
> All `file:line` cites below are relative to those real paths.

---

## 0. Ground truth established by the blob read (do not re-derive)

### 0.1 The golden is a DISABLED-EVAL skeleton — all three items are golden-safe *by construction*
`max_quality/tests/golden/stage6/stage6_eval.json` (read in full) contains only
integers / booleans / empty dicts / strings:
```json
{ "delta": {}, "measured_reduction": {...ints/null...}, "overall_pass": false,
  "student": {}, "teacher": {},
  "thresholds": { "skipped_checks": {
     "humaneval_pass_at_1_drop_ok": "generative eval disabled in config",
     "wikitext2_ppl_increase_ok":   "wikitext2 eval disabled in config", ... } } }
```
The snapshot test (`max_quality/tests/test_stage6_golden_snapshot.py`, docstring
"Why a plain byte compare is safe here") states the snapshot is captured with
**all eval families disabled** + a pre-baked teacher-cache hit, so the artifact
"carries only integers, booleans, empty dicts and strings — there are no computed
float metrics". Consequence: the golden never executes generate(), the PPL
forward, or the BPT argmax. None of items 1-3 can move a single byte of it.
We will still RUN the byte-compare as a regression gate, but the safety is
structural, not numeric.

### 0.2 `_humaneval` runs for BOTH student and teacher (Item-2 2x payoff is real)
`stage6/plugins/humaneval.py:149` defines `_humaneval`;
`stage6/plugins/teacher_provider.py:600-603` calls it on the teacher
(`teacher_results["humaneval_pass_at_1"] = _humaneval(teacher, ...)`), and the
student call sites are `humaneval.py:403` (the `eval_task` hook) and
`humaneval.py:493` (the inert-plugin scoring path). Any scoring-path change
applies to both sides automatically (single function).

### 0.3 The PPL/forward path does NOT slice logits in the plugin (Item-3 nuance)
`stage6/plugins/wikitext_ppl.py:194` is `out = model(input_ids=batch, labels=batch)`
and it reads ONLY `out.loss` (`:199`, `loss_val = float(out.loss.item())`).
The plugin never touches `out.logits`. The `(B,L,V)` materialization + the fp32
loss upcast happen INSIDE HF `...ForCausalLM.forward` when `labels=` is passed
(HF `ForCausalLMLoss`: `logits.float()` over the whole tensor -> a single fused
`cross_entropy(reduction="mean")`). `bpt_metric.py:130` has the same
`model(input_ids=batch, labels=batch)` pattern PLUS an explicit
`out.logits[:, :-1, :].argmax(dim=-1).to("cpu")` at `:150-152` (gated by
`collect_argmax`). This distinction is why Item 3 keeps ONLY the bit-identical
argmax-chunking change (see Item 3 — M1 decision).

### 0.4 Host start-method = fork (Item-2 primary risk)
`python3 -c 'import multiprocessing as mp; print(mp.get_start_method())'` -> `fork`
on the RTX 5080 host (Python 3.12.3). Fork-after-CUDA-init is the headline risk
for the ProcessPool (Item-2 Risks) — we force `spawn`.

### 0.5 `lsa_pool` is a ThreadPool justified ONLY by scipy GIL-release
`utils/lsa_pool.py` docstring: ThreadPool is correct there because scipy >=1.12
LSA "releases the GIL". HumanEval scoring runs user code in-process under the
GIL -> a ThreadPool would be a no-op. Confirms the spec's "use ProcessPool, NOT
lsa_pool/ThreadPool" mandate.

### 0.6 Two leaf-utility / circular-import contracts constrain Item-2's wiring
These are documented invariants we MUST honor (they shape the constant-placement
and worker-module decisions below):
- **`tools/eval_harness.py:1-21` leaf-utility contract**: this module imports
  only stdlib + `torch` (`import torch` at `eval_harness.py:28`) and MUST NEVER
  import a stage/pipeline module. It is Pattern-A: every symbol is a
  character-identical copy of the monolith and re-imported back by the monolith.
  Consequence: `_STAGE6_ATTN_IMPLEMENTATION` is *duplicated* as a module-local
  copy in `humaneval.py:141` and `teacher_provider.py:82` (each with a
  "keep in sync" comment) — it is NOT cross-imported.
- **`teacher_provider.py:37-43` circular-import contract**: it imports only from
  `..context` / `...utils` / sibling plugin modules
  (`eval_environment`, `wikitext_ppl`, `zero_shot_lm_eval`, `humaneval`,
  `math500`) / stdlib / torch — and NEVER from `stage6_validate`,
  `stage6.orchestrator`, or `...tools.eval_harness`, at module-top OR
  function-local scope. (`humaneval.py:115-119` DOES import private names from
  `...tools.eval_harness`, so that module may freely use eval_harness symbols.)

---

## ITEM 1 — Honest bs-invariance docstrings + pinned generative geometry (golden-safe; HIGHEST PRIORITY)

### Defect
Three docstrings claim batched generate() is "numerically identical to bs=1".
Measurement: 11/16 completions differ at bs=8 (bf16 + left-pad reduction-order
drift). The FORWARD / PPL / loglikelihood path IS batch-invariant (verified).
We make the claim honest and pin the generative geometry. NO change to the
bs=8 numeric behavior — metrics stay exactly what bs=8 produces.

### Sites (verified file:line)
1. `tools/eval_harness.py:52-53` — `_generate_batched` docstring (def at `:47`):
   "Greedy decoding produces deterministic outputs regardless of batching." (`:52`)
   "Numerically identical to serial generation." (`:53`). These are the false
   claims.
2. `stage6/plugins/humaneval.py:34-35` — module docstring:
   "The gate's batched-vs-bs=1 numerical-identity claim (#3, #4 in
   VALIDATED_STRATEGIES) holds under greedy decoding only." (verified exact.)
3. `stage6/plugins/wikitext_ppl.py:26-29` — module docstring:
   "configurable batch_size; **numerically identical to batch_size=1** (no
   per-batch noise). This is the project's VALIDATED_STRATEGIES §Stage 6
   Optimization #1." ("numerically identical to batch_size=1" is on `:27`; the
   VALIDATED_STRATEGIES reference spans `:26-29`.) This claim is TRUE for the
   forward/PPL path -> keep it, scope it explicitly.
4. `stage6/plugins/math500.py:391` — the THIRD `gen_batch_size` parse site
   (`gen_batch_size = int(s6.get("gen_batch_size", 8))`, `<= 0` raise at `:392-395`).
   Functionally identical to the humaneval / teacher_provider sites; `math500_accuracy`
   is a batch-geometry-dependent pinned generative metric, so it MUST get the same
   advisory WARN (see M3). `math500.py` ALREADY imports from `...tools.eval_harness`
   (`:85`), so it can import `PINNED_GEN_BATCH_SIZE` DIRECTLY (no local mirror).
5. Config narrative `max_quality/configs/qwen36_35b_a3b_30pct.yaml:568-569` —
   "Greedy decode (do_sample=False) is deterministic regardless of batch." (`:568`)
   "Left-padding + attention_mask ensures no cross-contamination." (`:569`). Same
   overclaim in prose; correct it too. `gen_batch_size: 8` is at `:570`.

### Before/after sketch
- `eval_harness.py:_generate_batched` docstring — replace the two false sentences
  (`:52-53`):
  "Greedy decoding (do_sample=False) is deterministic run-to-run for a FIXED
  batch geometry. It is NOT bit-identical across batch sizes: bf16 matmul
  reduction order + left-pad placement shift near-tied argmax (~11/16
  completions differ at bs=8 vs bs=1). Callers that report generative metrics
  MUST pin batch_size for run-to-run reproducibility. The forward/PPL/
  loglikelihood path IS batch-invariant; only generate() is geometry-dependent."
- `humaneval.py:34-35` — scope the VALIDATED_STRATEGIES reference: #3/#4 cover the
  batched-generate PLUMBING (left-pad, EOS-truncation, eager pin); the metrics
  humaneval_pass_at_1 / math500_accuracy are batch-geometry-dependent, pinned via
  gen_batch_size.
- `wikitext_ppl.py:26-29` — KEEP "numerically identical to batch_size=1" but
  qualify: "(forward/PPL path only — this metric reads out.loss, a batch-invariant
  reduction; the generate()-based generative metrics are NOT batch-invariant —
  see eval_harness._generate_batched)."

### Pin gen_batch_size so generative metrics are reproducible (M3 — WARN, not raise)
Measurement found bs=8 IS run-to-run deterministic; pinning locks the geometry.
**DECISION (M3): plain advisory WARN at all THREE parse sites — NOT a hard raise, and
NOT the over-engineered conditional/`HUMANEVAL_LIMIT`-guarded raise.** The value is
parsed + validated in three places (`humaneval.py:429`, `teacher_provider.py:548`,
`math500.py:391`) and a silent change at ANY of them would silently re-bless
generative metrics (`humaneval_pass_at_1` AND `math500_accuracy`); a loud WARN
delivers reproducibility-of-reporting without breaking `HUMANEVAL_LIMIT`/smoke runs
(operators legitimately run other geometries).

At the three existing parse sites (each already does
`gen_batch_size = int(s6.get("gen_batch_size", 8))` then a `<= 0` raise):
```python
gen_batch_size = int(s6.get("gen_batch_size", 8))
if gen_batch_size <= 0:
    raise ValueError(...)                       # existing — unchanged
# Item-1: generative metrics (humaneval_pass_at_1, math500_accuracy) are
# batch-geometry-dependent (bf16 + left-pad reduction drift). Pin the geometry
# so reported numbers are reproducible run-to-run. Advisory only — smoke runs
# (HUMANEVAL_LIMIT) legitimately use other geometries, so do NOT raise here.
if gen_batch_size != PINNED_GEN_BATCH_SIZE:
    log.warning("gen_batch_size=%d differs from the pinned generative geometry "
                "%d; humaneval/math500 numbers are NOT comparable to pinned runs.",
                gen_batch_size, PINNED_GEN_BATCH_SIZE)
```

#### Constant placement (H2 — respect both contracts; do NOT cross-import into teacher_provider)
`PINNED_GEN_BATCH_SIZE` follows the EXACT pattern already used for
`_STAGE6_ATTN_IMPLEMENTATION` (see 0.6):
- **Canonical definition** next to `_STAGE6_ATTN_IMPLEMENTATION` in
  `tools/eval_harness.py` (`:38` region): `PINNED_GEN_BATCH_SIZE: int = 8`, added
  to `__all__`.
- **`humaneval.py`** already imports private names from `...tools.eval_harness`
  (`:115-119`) — so add `PINNED_GEN_BATCH_SIZE` to that existing import block and
  use it at the `:429` parse site. This is contract-safe.
- **`math500.py`** already imports from `...tools.eval_harness` (`:85`) — so add
  `PINNED_GEN_BATCH_SIZE` to that existing import block and use it at the `:391`
  parse site, emitting the same advisory WARN after the parse/validation (`:392`).
  This is contract-safe — DIRECT import, NO local mirror needed (unlike
  teacher_provider).
- **`teacher_provider.py`** has a HARD circular-import contract (0.6 /
  `teacher_provider.py:37-43`) that forbids importing from `...tools.eval_harness`.
  Therefore define a **LOCAL MIRROR** there, next to its existing module-local
  `_STAGE6_ATTN_IMPLEMENTATION` copy (`teacher_provider.py:82`), with a "keep in
  sync with eval_harness.PINNED_GEN_BATCH_SIZE" comment — mirroring exactly how
  `_STAGE6_ATTN_IMPLEMENTATION` is duplicated rather than imported. Use the mirror
  at the `:548` parse site. **Do NOT cross-import the constant into
  teacher_provider.**
- Config comment at `qwen36_35b_a3b_30pct.yaml:570` updated to "PINNED generative
  geometry — do not change without re-blessing generative metrics."

### Why
Honest invariance scoping + a loud geometry pin = reproducible generative metrics
without altering the numbers we already produce, while respecting the two
import contracts (0.6).

### Test that proves it
- New unit `test_generate_batched_docstring_scopes_invariance` — asserts
  `_generate_batched.__doc__` no longer contains "Numerically identical" and DOES
  contain "NOT bit-identical" (cheap doc-contract pin, mirrors existing constant-pin tests).
- New unit `test_gen_batch_size_pin_warns_on_mismatch` — call the parse path (or a
  tiny extracted helper) with gen_batch_size=4 and assert a WARNING via caplog (no
  raise). Cover all three sites (humaneval, teacher_provider, math500) — math500
  uses the directly-imported `PINNED_GEN_BATCH_SIZE`, the others as wired in H2.
- New unit `test_pinned_gen_batch_size_mirror_in_sync` — assert
  `eval_harness.PINNED_GEN_BATCH_SIZE == teacher_provider`'s local mirror value
  (mirror-drift guard, same spirit as any `_STAGE6_ATTN_IMPLEMENTATION` pin).
- Regression: `pytest max_quality/tests/test_stage6_golden_snapshot.py` stays green
  (byte-identical — 0.1 guarantees it).

### Risk / rollback
Risk near-zero (docstrings + one advisory warn + one constant + one mirror + one
config comment). Rollback: revert the doc edits + delete the constant/mirror/warn.
No metric, no golden, no generation behavior touched.

---

## ITEM 2 — HumanEval scoring -> ProcessPool (Tier-1: timeout-robustness + daemon-thread leak fix)

### Current code (verified)
- Serial scoring loop: `stage6/plugins/humaneval.py:243-261` —
  `for i, (raw_stub, completion, test, ep) in enumerate(zip(...)): if _check_humaneval(...): passes += 1`.
- Post-loop leak warning: `humaneval.py:255-259`
  (`if leaked_counter[0]: log.warning("HumanEval: %d exec threads leaked ...")`).
- `_check_humaneval` body: `humaneval.py:264-321` (def at `:264`). Builds `src`,
  runs the model code in a DAEMON thread
  (`_t = threading.Thread(target=_exec_target, daemon=True)` at `:306`, `_t.start()`
  `:307`, `_t.join(timeout=exec_timeout_secs)` `:308`), and on `_t.is_alive()`
  (`:309`) returns False while incrementing a leaked-thread counter
  (`_leaked_counter[0] += 1` at `:312`). Final `return True` at `:320`. The leaked
  daemon thread keeps running until interpreter exit — the documented LEAK. All
  ranges verified; the leak is real.
- `_check_humaneval` is ALREADY a top-level module function (picklable by reference)
  — no relocation needed for the wrapper; but see the worker-module decision (H1).
- `_extract_code_from_chat_response` lives at `tools/eval_harness.py:251`
  (top-level), but its **defining module imports torch at `eval_harness.py:28`**
  and the regex constants it uses (`_THINK_BLOCK_RE`, `_PY_FENCE_RE`,
  `_TRAILING_PROSE_RE`) are module-level in eval_harness. See H1.

### H1 — the spawn worker MUST be torch-free (corrects the old plan's false claim)
Under `spawn`, each child process re-imports the **defining module of the worker
function** to unpickle it. If the worker imports from (or lives in)
`tools/eval_harness.py`, every spawned child re-imports eval_harness ->
**imports torch in EVERY child** + re-triggers eval_harness module-level side
effects. The old plan's "worker imports only stdlib + `_extract_code_from_chat_response`
-> small spawn cost" claim was FALSE: `_extract_code_from_chat_response`'s module
pulls torch.

**FIX:** create a NEW torch-free leaf module — e.g.
`stage6/plugins/_humaneval_worker.py` (stdlib + `re` only; NO torch, NO stage
imports) — that contains BOTH:
  1. `_score_humaneval_one(prompt, completion, test_src, entry_point) -> bool`
     (the picklable worker), and
  2. a COPY of the pure regex/string extraction logic
     (`_THINK_BLOCK_RE` / `_PY_FENCE_RE` / `_TRAILING_PROSE_RE` + the
     `_extract_code_from_chat_response` body — all pure `re`/string, no torch).
The worker imports its extraction helper from THIS leaf module, NOT from
`eval_harness`. Spawn children then pay only stdlib + `re` import cost.
- Keep the eval_harness copy of `_extract_code_from_chat_response` as-is (still
  used on the generation side); the worker module holds an independent, torch-free
  copy. Add a "keep in sync with tools/eval_harness._extract_code_from_chat_response"
  comment on both (same duplication discipline as `_STAGE6_ATTN_IMPLEMENTATION`).
- A unit test asserts the worker module is import-clean: importing
  `stage6/plugins/_humaneval_worker.py` does NOT pull `torch` into `sys.modules`
  (run in a fresh interpreter / assert `"torch" not in sys.modules` after a
  subprocess import).

### Target design
Replace the serial loop + daemon-thread timeout with a
`concurrent.futures.ProcessPoolExecutor`:
- (a) The picklable worker `_score_humaneval_one(prompt, completion, test_src,
  entry_point) -> bool` lives in the torch-free leaf module (H1). It does
  extraction (local copy) + runs the model code and returns True/False — NO
  threading inside the worker (the process IS the isolation + kill boundary now).
  Greedy decode deterministic -> result is a pure function of inputs.
  - KEEP `_check_humaneval` in `humaneval.py` with its EXACT current signature +
    bool contract as a thin wrapper delegating to `_score_humaneval_one`, because
    the plugin tests pin it positionally:
    `test_stage6_plugin_humaneval.py:183`
    (`_check_humaneval(prompt, completion, test_src, "add") is True`) and `:202`
    (wrong solution -> is False). DO NOT change `_check_humaneval`'s signature.
    (The `_leaked_counter` mutable-box arg stays on the wrapper for the existing
    in-process tests; it is NOT passed across the process boundary.)
- (b) Submit all problems, then WAIT on ALL futures against a SINGLE SHARED
  deadline — `deadline = time.monotonic() + exec_timeout_secs` applied across the
  batch via `concurrent.futures.wait(futures, timeout=remaining)` in a loop, NOT
  per-future `fut.result(timeout=exec_timeout_secs)` (which re-serializes the
  timeout: N x 10s worst case). Problems whose future is unfinished at the
  deadline score False (matches current timeout->False).
- (c) HARD-TERMINATE stuck workers (N3 — prefer the portable form): on deadline,
  `executor.shutdown(wait=False, cancel_futures=True)` and **re-create the pool**
  for any subsequent batch, rather than reaching into the private
  `executor._processes`. The public shutdown + pool re-creation is portable across
  CPython minors. *If* a stricter immediate-kill is required on the pinned 3.12
  host, document the private-API coupling explicitly before touching
  `executor._processes`. This FIXES the leak either way: a killed/terminated
  process reclaims the runaway work (vs the daemon thread that runs to interpreter
  exit).
- Pool: `ProcessPoolExecutor(max_workers=min(os.cpu_count(), N), mp_context=multiprocessing.get_context("spawn"))`.
  FORCE spawn even though the host default is fork (0.4) — fork-after-CUDA-init can
  deadlock the worker; spawn gives a clean interpreter. Because the worker module is
  torch-free (H1), spawn import cost is small + one-time.

### Before/after (loop only; humaneval.py:243-261)
Before: serial `for ... if _check_humaneval(...): passes += 1`.
After: submit `_score_humaneval_one` futures, shared-deadline wait,
`passes = sum(1 for f in futures if f.done() and f.result() is True)`
(order-independent sum), terminate unfinished workers (shutdown+recreate),
tally `terminated = #unfinished` for the post-loop warning (now "terminated" not
"leaked"). Update the `:255-259` warning text accordingly.

### Why
- Fixes the daemon-thread leak (real correctness/resource bug).
- Timeout-robustness: shared deadline caps total scoring wall-time at ~one
  exec_timeout_secs, not Nx. Measured 4x ONLY in the timeout-heavy/degraded-student
  regime (negligible when everything passes fast) — frame the payoff as LEAK FIX +
  TIMEOUT ROBUSTNESS, not a blanket 4x. 2x reach because it runs for student AND
  teacher (0.2: `teacher_provider.py:600`).

### Tier-1 invariants to assert
- Per-problem pass/fail identical to the daemon-thread path (same src construction,
  same run, same exception->False, same timeout->False).
- Final passes/total is an order-independent sum.
- Greedy decode deterministic -> `_score_humaneval_one` is a pure function.

### Pins to preserve
- `test_stage6_plugin_humaneval.py:183` + `:202` (the `_check_humaneval` 4-arg
  positional bool contract) — keep `_check_humaneval` as the public wrapper.
- `test_stage6_plugin_humaneval.py:77` (`stage6_validate._check_humaneval is
  he._check_humaneval` is-identity assertion) — keeping `_check_humaneval` as a
  wrapper IN `humaneval.py` (already required for the `:183/:202` bool-contract pins
  above) also satisfies this; do NOT relocate the symbol out of `humaneval.py`.
- `stage6_eval.json` golden (0.1 — disabled, untouched).

### In-code-decision reversals to update (N2 — REQUIRED, both blocks)
This change brings subprocess isolation INTO scope, which directly contradicts two
documented in-code decisions that MUST be rewritten as part of this item:
- The N3 comment in `_exec_target` (`humaneval.py:292-300`): currently
  "Subprocess isolation is the only real fix and is intentionally out of scope
  (D-humaneval-greedy rationale)." Rewrite to reflect that subprocess isolation is
  now implemented via the ProcessPool worker (and the signal.alarm caveat is moot
  once the model code runs in a child process, not a daemon thread).
- The module docstring in-process rationale (`humaneval.py:23-26` and
  `:37-41`): "(b) Exec-based scoring runs **in-process**, NOT in a subprocess
  sandbox ..." and "In-process exec is documented because subprocess isolation
  would slow eval substantially with no signal-quality benefit ...". Rewrite both
  blocks to document the ProcessPool design.
- The "Known limitations" cross-reference (`humaneval.py:38-40` -> the
  `stage6_validate.py` "Known limitations" note about daemon-thread leakage / no
  syscall interruption): update that note too so it reflects subprocess workers.

### Nitpicks
- **N1**: after the rewrite, `import threading` (`humaneval.py:111`) is unused —
  remove it if no other reference remains. Add the new imports
  (`concurrent.futures`, `multiprocessing`, `time`; `os` is already imported).
- Security posture UNCHANGED conceptually (still runs model code) but now in a
  child process (MORE isolated than the in-process daemon thread). H1 security
  WARNING at `humaneval.py:220-231` stays; update wording from "daemon thread" to
  "subprocess workers".

### Tests that prove it
- Existing `test_check_humaneval_correct_solution` (:183) + `_wrong_solution` (:202)
  MUST stay green unchanged (wrapper preserves contract).
- New `test_score_humaneval_one_picklable` — `pickle.dumps(_score_humaneval_one)`
  succeeds (guards against accidental closure capture).
- New `test_humaneval_worker_module_is_torch_free` — import the worker module in a
  fresh subprocess and assert `"torch" not in sys.modules` (H1 guard).
- New `test_humaneval_pool_timeout_scores_false` — a worker that sleeps past a tiny
  exec_timeout_secs scores False AND leaves no live child after shutdown. Proves the
  leak fix.
- New `test_humaneval_pool_order_independent` — shuffle completion order, assert
  identical passes total.

### Risks / rollback
- fork vs spawn (PRIMARY): host default is fork (0.4); forking after the model is on
  CUDA can hang the child. MITIGATION: force mp_context="spawn" + torch-free worker
  module (H1). The torch-free-import test is the regression guard.
- Pickle: args are str/int (picklable); worker is top-level in a leaf module
  (picklable). The pickle unit test guards regressions.
- Worker-kill correctness on Linux: Process.terminate() sends SIGTERM; a worker stuck
  in a C extension may ignore it briefly. With shutdown(wait=False, cancel_futures=True)
  + pool re-creation the OS reaps on exit; strictly better than the current
  never-dies daemon thread. Escalate to SIGKILL after a grace period if stricter kill
  is needed.
- Rollback: revert to the serial loop + daemon-thread _check_humaneval and delete the
  worker leaf module. Self-contained to humaneval.py + the new worker file.

---

## ITEM 3 — BPT argmax seq/batch chunking (Tier-1: memory only, BIT-IDENTICAL)

### M1 decision — DROP the PPL loss-path rewrite (old Item 3a); keep ONLY the argmax chunking (old Item 3b)
The old plan proposed dropping `labels=` in `wikitext_ppl.py` and hand-rolling a
chunked fp32 cross-entropy sum. **That CANNOT be bit-identical** and is therefore
DROPPED for this Tier-1 round:
- HF `ForCausalLMLoss` does `logits.float()` on the WHOLE `(B,L,V)` then a single
  fused `cross_entropy(reduction="mean")`. A chunked `reduction="sum"` +
  Python-float accumulate differs in reduction order AND dtype path -> NOT
  bit-identical -> would violate the user's Tier-1 / no-re-bless choice.
- The `wikitext_ppl.py` `labels=`/`out.loss` path (`:194`, `:199`, nll
  reconstruction `:210`) is therefore **left UNTOUCHED**.
- NOTE: if a loss-path OOM is later *demonstrated at scale*, the PPL chunking
  returns as a SEPARATE tolerance-bounded **Tier-2** decision (explicit
  metric-stability sign-off, not bit-identity) — it is NOT part of this Tier-1
  round.

What REMAINS in Item 3 is the argmax chunking only, which IS truly bit-identical.

### Current code (verified)
- `stage6alt/plugins/bpt_metric.py:130` — `out = model(input_ids=batch, labels=batch)`
  (forward at `:130`); reads `out.loss` (`:135`). The argmax side at `:150-152` —
  `argmax_chunks.append(out.logits[:, :-1, :].argmax(dim=-1).to("cpu"))`
  (gated by `collect_argmax`; `.append(` on `:150`, the `out.logits[...].argmax`
  expression on `:151`). Argmax already moves to CPU.
- The `(B,L,V)` logits tensor that backs `out.logits[:, :-1, :]` is the memory
  peak on the argmax path. Prod geometry: B=8 (bpt batch), L=2048
  (`sequence_length: 2048`, config `:534`/`:593`), V=151936 -> `(B,L,V)` bf16
  ~= 4.98 GB; the materialized logits are the documented OOM risk on smaller
  cards. (The HF-internal fp32 CE upcast is on the loss path, which we are NOT
  touching per M1.)

### Target design — argmax over vocab, chunk over B/L only (bit-identical)
`argmax` over the vocab axis is EXACTLY decomposable: compute argmax per
seq-chunk (chunk the B or L axis), `.to("cpu")`, concat. `argmax(full)[k] ==
argmax(chunk_k)` exactly — argmax has no floating accumulation, and ties broken by
first-index are preserved **as long as we keep the vocab axis whole and chunk only
B/L** (never split the vocab axis). Result tensor is identical.
```python
# bpt_metric.py:150-152, collect_argmax branch:
# was: argmax_chunks.append(out.logits[:, :-1, :].argmax(dim=-1).to("cpu"))
# now: chunk over B (or L), argmax each slice over the (whole) vocab axis,
#      move each to CPU, append; downstream torch.cat(dim=0) is unchanged.
shift = out.logits[:, :-1, :]                 # (B, L-1, V)
for b0 in range(0, shift.shape[0], argmax_chunk_b):
    sl = shift[b0:b0 + argmax_chunk_b]         # (b, L-1, V) — vocab axis WHOLE
    argmax_chunks.append(sl.argmax(dim=-1).to("cpu"))
```
Shape contract `(num_seqs, seq_len-1)` and dtype `long`
(`test_stage6alt_plugin_bpt.py:356-357`) preserved; `torch.cat(argmax_chunks,
dim=0)` at `bpt_metric.py:177` unchanged.

### Why
Bounds the argmax-path logits peak to `(chunk_b, L-1, V)` instead of the full
`(B, L-1, V)`. Memory-only; metric bit-identical (no float accumulation in argmax).

### Tests that prove it
- ARGMAX bit-identity (load-bearing): new `test_bpt_argmax_chunked_equals_unchunked`
  — `torch.equal(chunked_argmax, full_argmax)` on tiny_model.
- Existing pins stay green: `test_bpt_from_nll_collect_argmax_shape`
  (`test_stage6alt_plugin_bpt.py:342-357`, shape `(2,3)`, dtype long, bs=1),
  `test_bpt_from_nll_finite_path` (:327),
  `test_bpt_from_nll_requires_eager_attention` (:309).
- Golden byte-compare (0.1).

### Risks / rollback
- argmax tie-breaking: safe as long as we never split the vocab axis (only B/L).
  Plan: chunk B/L ONLY.
- Eager-attn pin (`bpt_metric.py:97-106`) UNTOUCHED.
- Blackwell grouped_mm shim UNTOUCHED (spec mandate; verified not in this file).
- The `wikitext_ppl.py` loss path is UNTOUCHED (M1).
- Rollback: revert the argmax branch to the single-shot
  `out.logits[:, :-1, :].argmax(...)`; self-contained to bpt_metric.py.

---

## ORDERING

1. Item 1 first (highest priority, lowest risk, pure docs + advisory + constant).
   Lands the honesty fix + geometry pin so any subsequent perf work reports against
   a pinned geometry.
2. Item 3 second (argmax memory chunking; bit-identical; unblocks running BPT on
   smaller cards). The argmax-equality test gates it.
3. Item 2 last (ProcessPool; most moving parts — spawn / pickle / torch-free worker
   / kill). Benefits from Item-3 having lowered memory pressure so end-to-end smoke
   can exercise scoring.

Each item is independent (different files, no shared edits) -> could land in parallel,
but the above order minimizes integration risk.

---

## TESTING PLAN ON HOST RTX 5080 (16 GB, Python 3.12.3, mp=fork)

1. Unit suite (fast, no GPU pressure):
   `pytest max_quality/tests/test_stage6_plugin_humaneval.py \
          max_quality/tests/test_stage6_plugin_wikitext.py \
          max_quality/tests/test_stage6alt_plugin_bpt.py \
          max_quality/tests/test_stage6_plugin_math500.py -v`
   plus the new docstring / mirror-sync / pickle / torch-free-worker / timeout /
   argmax-equality tests.
2. Golden gate (must stay byte-identical):
   `pytest max_quality/tests/test_stage6_golden_snapshot.py \
          max_quality/tests/test_stage6alt_golden_snapshot.py -v`
3. Item-2 leak proof: the new timeout test asserts no live child remains after
   shutdown; additionally a manual `ps --ppid <pid>` sanity check shows no orphaned
   children after a deliberately-hung snippet.
4. Item-3 memory proof on the 16 GB card: run `_bpt_from_nll` on tiny_model with a
   small `argmax_chunk_b`; for a real-geometry smoke use B=8, L=2048, real vocab and
   watch nvidia-smi peak stay under the no-chunking peak.
5. fork->spawn + torch-free verification (Item 2): assert
   `multiprocessing.get_context("spawn")` is what the pool uses; confirm the worker
   leaf module does NOT import torch (fresh-interpreter import + `"torch" not in
   sys.modules`).

---

## CONSTRAINTS RE-AFFIRMED
- tests/golden/stage6/stage6_eval.json stays byte-identical (structural — 0.1).
- No generation batch-behavior change (Item 1 documents/pins; does not re-batch).
- Eager attention pin (F-S-M-1) untouched in all three items.
- Blackwell grouped_mm shim untouched (not present in the edited files; verified).
- No metric re-bless. Item 3 is bit-identical by construction (argmax over whole
  vocab, chunk B/L only); the non-bit-identical PPL loss rewrite is DROPPED (M1).
- Leaf-utility (`eval_harness.py:1-21`) + circular-import (`teacher_provider.py:37-43`)
  contracts respected: `PINNED_GEN_BATCH_SIZE` canonical in eval_harness, imported by
  humaneval, MIRRORED (not imported) in teacher_provider; the spawn worker lives in a
  new torch-free leaf module (H1).
- No monkeypatch in new tests (project rule) — use caplog, direct calls, real tiny_model
  fixture, subprocess imports.

> L2 NOTE: `math500.py`'s module DOCSTRING contains NO false "numerically
> identical / bs=1" claim (its `:48` mention of "identical copies of the monolith
> definitions" is about Pattern-A symbol copies, not generate() invariance).
> Mirroring the Item-1 docstring scoping note into math500.py would be ADDITIVE, not
> a fix — dropped from this round (optional only). This is SEPARATE from the
> `gen_batch_size` WARN pin at `math500.py:391`, which IS in scope (Item 1 Site 4 /
> M3) because `math500_accuracy` is a batch-geometry-dependent generative metric.
