# PLAN — Stage-6 eval optimizations + bs-invariance correctness fix

Base: `origin/main` @ `0ae66896942cd8ecbca1cf62c81aaec60f889590` (verified via `git rev-parse origin/main`).
Branch: `plan/stage6-opt`. **Plan only — no production edits in this branch.**
All file:line citations below were read from real blobs (`git show origin/main:<path>`), not from the working tree, and re-verified against the measurement-pass numbers (drift noted inline where found).

> NOTE ON PATHS: the spec referenced `max_quality/src/moe_compress/configs/...`,
> `tests/golden/stage6/...`, and `qwen36_35b_a3b_30pct.yaml:~570`. The actual
> on-disk locations on `origin/main` are:
> - config: `max_quality/configs/qwen36_35b_a3b_30pct.yaml` (NOT under `src/`); `gen_batch_size: 8` is at **line 570** (exact match).
> - golden: `max_quality/tests/golden/stage6/stage6_eval.json` (NOT under `src/moe_compress/tests/`).
> - tests live under `max_quality/tests/`.

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
`stage6/plugins/humaneval.py:70` exports `_humaneval`; `teacher_provider.py:600-603`
calls it on the teacher (`teacher_results["humaneval_pass_at_1"] = _humaneval(teacher, ...)`),
and the student call site is the `eval_task` hook at `humaneval.py:493-498`. Any
scoring-path change applies to both sides automatically (single function).

### 0.3 The PPL/forward path does NOT slice logits in the plugin (Item-3 nuance)
`stage6/plugins/wikitext_ppl.py:194` is `out = model(input_ids=batch, labels=batch)`
and it reads ONLY `out.loss` (`:196`, `loss_val = float(out.loss.item())`).
The plugin never touches `out.logits`. The `(B,L,V)` materialization + the fp32
loss upcast happen INSIDE HF `...ForCausalLM.forward` when `labels=` is passed
(`logits.float()` -> `cross_entropy`). So for wikitext_ppl the only lever is to
STOP passing `labels=` and compute the loss ourselves in vocab-chunks from
`out.logits` (then we control the upcast + chunking). `bpt_metric.py:130` has the
same `model(input_ids=batch, labels=batch)` pattern PLUS an explicit
`out.logits[:, :-1, :].argmax(dim=-1).to("cpu")` at `:150-152` (gated by
`collect_argmax`). This distinction drives two different chunking sketches below.

### 0.4 Host start-method = fork (Item-2 primary risk)
`python3 -c 'import multiprocessing as mp; print(mp.get_start_method())'` -> `fork`
on the RTX 5080 host (Python 3.12.3). Fork-after-CUDA-init is the headline risk
for the ProcessPool (Item-2 Risks).

### 0.5 `lsa_pool` is a ThreadPool justified ONLY by scipy GIL-release
`utils/lsa_pool.py` docstring: ThreadPool is correct there because scipy >=1.12
LSA "releases the GIL". HumanEval scoring runs `exec()` of PURE-Python user
code -> holds the GIL -> a ThreadPool would be a no-op. Confirms the spec's
"use ProcessPool, NOT lsa_pool/ThreadPool" mandate.

---

## ITEM 1 — Honest bs-invariance docstrings + pinned generative geometry (golden-safe; HIGHEST PRIORITY)

### Defect
Three docstrings claim batched generate() is "numerically identical to bs=1".
Measurement: 11/16 completions differ at bs=8 (bf16 + left-pad reduction-order
drift). The FORWARD / PPL / loglikelihood path IS batch-invariant (verified).
We make the claim honest and pin the generative geometry. NO change to the
bs=8 numeric behavior — metrics stay exactly what bs=8 produces.

### Sites (verified file:line)
1. `tools/eval_harness.py:52-53` — `_generate_batched` docstring:
   "Greedy decoding produces deterministic outputs regardless of batching.
   Numerically identical to serial generation." This is the false claim.
   (Spec said ~53 — exact.)
2. `stage6/plugins/humaneval.py:34-35` — module docstring:
   "The gate's batched-vs-bs=1 numerical-identity claim (#3, #4 in
   VALIDATED_STRATEGIES) holds under greedy decoding only." (Spec ~34-35 — exact.)
3. `stage6/plugins/wikitext_ppl.py:27-30` — module docstring:
   "configurable batch_size; numerically identical to batch_size=1 ...
   VALIDATED_STRATEGIES Stage 6 Optimization #1." (Spec ~28-29 — within 1 line.)
   This claim is TRUE for the forward/PPL path -> keep it, scope it explicitly.
4. Config narrative `max_quality/configs/qwen36_35b_a3b_30pct.yaml:567-569` —
   "Greedy decode (do_sample=False) is deterministic regardless of batch.
   Left-padding + attention_mask ensures no cross-contamination." Same overclaim
   in prose; correct it too.

### Before/after sketch
- `eval_harness.py:_generate_batched` docstring — replace the two false sentences:
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
- `wikitext_ppl.py:27-30` — KEEP "numerically identical to batch_size=1" but
  qualify: "(forward/PPL path only — this metric reads out.loss, a batch-invariant
  reduction; the generate()-based generative metrics are NOT batch-invariant —
  see eval_harness._generate_batched)."
- Mirror the one-liner scoping into `math500.py` module docstring (shares
  _generate_batched; verify exact line at implement time, math500.py:48-51 region).

### Pin gen_batch_size so generative metrics are reproducible
Measurement found bs=8 IS run-to-run deterministic; pinning locks the geometry.
DECISION: ASSERT-in-code (advisory WARN), not document-only — the value is parsed
+ validated in two places (`humaneval.py:429-432`, `teacher_provider.py:548-551`)
and a silent change there would silently re-bless generative metrics.

At the two existing parse sites that read `s6.get("gen_batch_size", 8)`:
```python
gen_batch_size = int(s6.get("gen_batch_size", 8))
if gen_batch_size <= 0:
    raise ValueError(...)                       # existing
# Item-1: generative metrics (humaneval_pass_at_1, math500_accuracy) are
# batch-geometry-dependent (bf16 + left-pad reduction drift). Pin the geometry
# so reported numbers are reproducible run-to-run.
if gen_batch_size != PINNED_GEN_BATCH_SIZE:
    log.warning("gen_batch_size=%d differs from the pinned generative geometry "
                "%d; humaneval/math500 numbers are NOT comparable to pinned runs.",
                gen_batch_size, PINNED_GEN_BATCH_SIZE)
```
WARN (not raise) because operators legitimately use HUMANEVAL_LIMIT/smoke runs at
other geometries; a hard raise would break smoke. `PINNED_GEN_BATCH_SIZE = 8`
lives next to the existing `_STAGE6_ATTN_IMPLEMENTATION` constant in
`eval_harness.py` (single source, imported by humaneval + teacher_provider).
Config comment at `qwen36_35b_a3b_30pct.yaml:570` updated to "PINNED generative
geometry — do not change without re-blessing generative metrics."

> Raise-vs-warn is the one open design choice. Default to WARN per the smoke-run
> caveat; flip to a guarded raise (allow override only when HUMANEVAL_LIMIT>0) if
> the user prefers hard enforcement.

### Why
Honest invariance scoping + a loud geometry pin = reproducible generative metrics
without altering the numbers we already produce.

### Test that proves it
- New unit `test_generate_batched_docstring_scopes_invariance` — asserts
  `_generate_batched.__doc__` no longer contains "Numerically identical" and DOES
  contain "NOT bit-identical" (cheap doc-contract pin, mirrors existing constant-pin tests).
- New unit `test_gen_batch_size_pin_warns_on_mismatch` — call the parse path (or a
  tiny extracted helper) with gen_batch_size=4 and assert a WARNING via caplog.
- Regression: `pytest max_quality/tests/test_stage6_golden_snapshot.py` stays green
  (byte-identical — 0.1 guarantees it).

### Risk / rollback
Risk near-zero (docstrings + one advisory warn + one config comment). Rollback:
revert the doc edits + delete the constant/assert. No metric, no golden, no
generation behavior touched.

---

## ITEM 2 — HumanEval scoring -> ProcessPool (Tier-1: timeout-robustness + daemon-thread leak fix)

### Current code (verified)
- Serial scoring loop: `stage6/plugins/humaneval.py:243-261` —
  `for i, (raw_stub, completion, test, ep) in enumerate(zip(...)): if _check_humaneval(...): passes += 1`.
- `_check_humaneval` body: `humaneval.py:264-320`. Builds `src`, runs
  `exec(src, ns, ns)` inside a DAEMON thread (`_t = threading.Thread(target=_exec_target,
  daemon=True)`, `:314-316`), `join(timeout=exec_timeout_secs)`, and on `_t.is_alive()`
  returns False while incrementing a leaked-thread counter (`:317-323`). The leaked
  daemon thread keeps running its exec() until interpreter exit — the documented
  LEAK (spec cited ~:255-259; on origin/main the leak tally is the
  `_leaked_counter[0] += 1` block at `humaneval.py:318-323` and the post-loop
  warning at `:255-259`). Both ranges verified; the leak is real.
- `_check_humaneval` is ALREADY a top-level module function (picklable by reference)
  — no relocation needed.
- Worker-needed helper `_extract_code_from_chat_response` lives at
  `tools/eval_harness.py:251` (top-level, importable, picklable by reference).

### Target design
Replace the serial loop + daemon-thread timeout with a
`concurrent.futures.ProcessPoolExecutor`:
- (a) Keep a top-level, PICKLABLE worker with IDENTICAL bool pass/fail semantics.
  Current signature:
  `_check_humaneval(prompt, completion, test_src, entry_point, *, exec_timeout_secs=10, _leaked_counter=None, _problem_index=0) -> bool`.
  The `_leaked_counter` mutable-box arg CANNOT cross a process boundary -> split into:
  - a pure picklable worker `_score_humaneval_one(prompt, completion, test_src, entry_point) -> bool`
    that does `_extract_code_from_chat_response` + `exec(src, ns, ns)` and returns
    True/False (NO threading inside the worker — the process IS the isolation +
    kill boundary now). Greedy decode deterministic -> result is a pure function of inputs.
  - KEEP `_check_humaneval` with its EXACT current signature + bool contract as a
    thin wrapper delegating to `_score_humaneval_one`, because the plugin tests pin
    it positionally: `test_stage6_plugin_humaneval.py:183`
    (`_check_humaneval(prompt, completion, test_src, "add") is True`) and `:202`
    (wrong solution -> is False). DO NOT change `_check_humaneval`'s signature.
- (b) Submit all problems, then WAIT on ALL futures against a SINGLE SHARED
  deadline — `deadline = time.monotonic() + exec_timeout_secs` applied across the
  batch via `concurrent.futures.wait(futures, timeout=remaining)` in a loop, NOT
  per-future `fut.result(timeout=exec_timeout_secs)` (which re-serializes the
  timeout: N x 10s worst case). Problems whose future is unfinished at the
  deadline score False (matches current timeout->False).
- (c) HARD-TERMINATE stuck workers: on deadline,
  `executor.shutdown(wait=False, cancel_futures=True)` then
  `for p in executor._processes.values(): p.terminate()` (or recreate the pool;
  or pebble.ProcessPool if a dep is acceptable — default to stdlib + explicit
  terminate()). This FIXES the leak: a killed process reclaims the runaway exec()
  (vs the daemon thread that runs to interpreter exit).
- Pool: `ProcessPoolExecutor(max_workers=min(os.cpu_count(), N), mp_context=multiprocessing.get_context("spawn"))`.
  FORCE spawn even though the host default is fork (0.4) — fork-after-CUDA-init can
  deadlock the worker; spawn gives a clean interpreter. The worker imports only
  stdlib + `_extract_code_from_chat_response`, so spawn import cost is small + one-time.

### Before/after (loop only; humaneval.py:243-261)
Before: serial `for ... if _check_humaneval(...): passes += 1`.
After: submit `_score_humaneval_one` futures, shared-deadline wait,
`passes = sum(1 for f in futures if f.done() and f.result() is True)`
(order-independent sum), terminate unfinished workers, tally `leaked = #unfinished`
for the existing post-loop warning (now "terminated" not "leaked").

### Why
- Fixes the daemon-thread leak (real correctness/resource bug).
- Timeout-robustness: shared deadline caps total scoring wall-time at ~one
  exec_timeout_secs, not Nx. Measured 4x ONLY in the timeout-heavy/degraded-student
  regime (negligible when everything passes fast) — frame the payoff as LEAK FIX +
  TIMEOUT ROBUSTNESS, not a blanket 4x. 2x reach because it runs for student AND teacher (0.2).

### Tier-1 invariants to assert
- Per-problem pass/fail identical to the daemon-thread path (same src construction,
  same exec, same exception->False, same timeout->False).
- Final passes/total is an order-independent sum.
- Greedy decode deterministic -> `_score_humaneval_one` is a pure function.

### Pins to preserve
- `test_stage6_plugin_humaneval.py:183` + `:202` (the `_check_humaneval` 4-arg
  positional bool contract) — keep `_check_humaneval` as the public wrapper.
- `stage6_eval.json` golden (0.1 — disabled, untouched).

### Tests that prove it
- Existing `test_check_humaneval_correct_solution` (:183) + `_wrong_solution` (:202)
  MUST stay green unchanged (wrapper preserves contract).
- New `test_score_humaneval_one_picklable` — `pickle.dumps(_score_humaneval_one)`
  succeeds (guards against accidental closure capture).
- New `test_humaneval_pool_timeout_scores_false` — a worker that sleeps past a tiny
  exec_timeout_secs scores False AND leaves no live child (`executor._processes`
  empty after shutdown). Proves the leak fix.
- New `test_humaneval_pool_order_independent` — shuffle completion order, assert
  identical passes total.

### Risks / rollback
- fork vs spawn (PRIMARY): host default is fork (0.4); forking after the model is on
  CUDA can hang the child. MITIGATION: force mp_context="spawn". Spawn re-imports the
  worker module -> worker must be import-clean and not pull CUDA/torch at import (it
  imports only `_extract_code_from_chat_response` + stdlib). VERIFY no torch import is
  triggered transitively by that import.
- Pickle: args are str/int (picklable); worker is top-level (picklable). The pickle
  unit test guards regressions.
- Worker-kill correctness on Linux: Process.terminate() sends SIGTERM; a worker stuck
  in a C extension may ignore it briefly. ACCEPTED: shutdown(wait=False) + no join — the
  OS reaps on exit; strictly better than the current never-dies daemon thread. Escalate
  to SIGKILL after a grace period if stricter kill is needed.
- Security posture UNCHANGED: still exec() of model code, now in a child process (MORE
  isolated than the in-process daemon thread). H1 security WARNING at humaneval.py:222-227
  stays; update wording from "daemon threads" to "subprocess workers".
- Rollback: revert to the serial loop + daemon-thread _check_humaneval. Self-contained
  to humaneval.py.

---

## ITEM 3 — BPT/PPL logits vocab/seq chunking (Tier-1: memory only, bit-identical)

### Current code (verified)
- `stage6alt/plugins/bpt_metric.py:130` — `out = model(input_ids=batch, labels=batch)`;
  `:150-152` — `argmax_chunks.append(out.logits[:, :-1, :].argmax(dim=-1).to("cpu"))`
  (gated by collect_argmax). Argmax already moves to CPU.
- `stage6/plugins/wikitext_ppl.py:194` — `out = model(input_ids=batch, labels=batch)`;
  reads only `out.loss` (`:196`). Logits + fp32 upcast are INTERNAL to HF forward (0.3).
- Prod geometry: B=8 (ppl_batch_size: 8), L=2048 (sequence_length: 2048, config
  `:534`/`:593`), V=151936 -> (B,L,V) bf16 ~= 4.98 GB; HF's internal logits.float() for
  cross_entropy ~= 9.96 GB transient. On the 16 GB RTX 5080 this OOMs outright; on
  <94 GB prod cards it is the documented OOM risk. Spec's ~5 GB / ~10 GB confirmed.

### Target design (TWO distinct sketches because the two call sites differ — 0.3)

3a. wikitext_ppl.py (loss-only): stop passing labels=; request logits and compute NLL
ourselves in vocab chunks so we never hold a full fp32 (B,L,V).
```python
out = model(input_ids=batch)                 # no labels -> HF does NOT upcast/CE
logits = out.logits                          # (B, L, V) bf16
shift_logits = logits[:, :-1, :]             # (B, L-1, V)
shift_labels = batch[:, 1:]                  # (B, L-1)
# chunk over the flattened (B*(L-1)) token axis; for each chunk compute
# F.cross_entropy(chunk.float(), labels_chunk, reduction="sum") and accumulate.
```
Bounds the fp32 upcast to (chunk, V) instead of (B*(L-1), V). NLL sum is an exact
reduction. CAVEAT: HF's labels= path returns reduction="mean" (out.loss), and the
plugin reconstructs the sum via `loss_val * (numel - B)` (wikitext_ppl.py:212, the L-3
comment). Switching to our own reduction="sum" over fp32 logits changes the UPCAST
GRANULARITY but the math is the same fp32 cross-entropy. THIS IS THE ONE PLACE
BIT-IDENTITY MUST BE PROVEN, NOT ASSUMED — fp32 summation equals all-at-once to the
last ULP only if we sum in the same order; plan a tolerance-AND-exact test (below). If
strict bit-identity cannot be guaranteed, this becomes a Tier-1 "metric-stable within
fp32 ULP" change; we re-bless nothing because the golden is disabled (0.1) and the
wikitext test only asserts isfinite + >0 (see Tests).

3b. bpt_metric.py (loss + argmax): the loss side is identical to 3a. The argmax side
(out.logits[:, :-1, :].argmax(dim=-1)) is the memory peak that is EASY to make
bit-identical: argmax over the vocab axis is EXACTLY decomposable — compute argmax per
seq-chunk (chunk the L or B axis), concat on CPU. argmax(full)[k] == argmax(chunk_k)
exactly (argmax has no floating accumulation; ties broken by first-index, preserved if
we keep the vocab axis whole and chunk only B/L). So chunk over B (or L), argmax each,
.to("cpu"), concat -> identical tensor. Shape contract (num_seqs, seq_len-1) and dtype
long (test :356-357) preserved.

### Before/after
- wikitext_ppl.py:194-212: `model(...labels=batch)` + `loss_val*(numel-B)` ->
  `model(input_ids=batch)` + chunked fp32 CE-sum accumulator. Keep the None-loss /
  non-finite / skip-batch guards (:197-211) reframed around the computed sum.
- bpt_metric.py:130 + 150-152: same loss change; argmax becomes a B/L-chunked loop
  appending CPU argmax slices.

### Why
Cuts peak from ~15 GB transient to ~(chunk*V*4 bytes) — fits <94 GB cards and the 16 GB
host (small chunk + small B in the test). Memory-only; metric intended bit-identical.

### Tests that prove it
- BIT-IDENTITY (the load-bearing test): new
  `test_wikitext_ppl_chunked_equals_unchunked` — run the OLD labels=-path PPL and the
  NEW chunked PPL on tiny_model over the SAME chunks; assert
  `ppl_new == pytest.approx(ppl_old, rel=0, abs=0)` if strict, else document the fp32
  ULP tolerance actually achieved and pin THAT. Same for BPT.
- ARGMAX bit-identity: new `test_bpt_argmax_chunked_equals_unchunked` —
  `torch.equal(chunked_argmax, full_argmax)`.
- Existing pins stay green: `test_bpt_from_nll_collect_argmax_shape`
  (test_stage6alt_plugin_bpt.py:342-357, shape (2,3), dtype long, bs=1),
  `test_bpt_from_nll_finite_path` (:327), `test_wikitext2_ppl_returns_finite_float`
  (test_stage6_plugin_wikitext.py:236-260), `test_wikitext2_ppl_eager_attn_assert` (:223),
  `test_bpt_from_nll_requires_eager_attention` (:309).
- Golden byte-compare (0.1).

### Risks / rollback
- Bit-identity of the loss path is NOT free (3a caveat): HF's internal CE vs our chunked
  CE can differ at fp32 ULP if reduction order differs. MITIGATION: accumulate in fp64
  (nll_sum is already a Python float = fp64) and sum chunk sums in a fixed order; PROVE
  with the equality test before merge. If exact bit-identity fails, STOP and re-scope to
  "memory fix, metric stable to <1e-6 rel" with explicit user sign-off (do not silently
  re-bless).
- argmax tie-breaking: safe as long as we never split the vocab axis (only B/L). Plan:
  chunk B/L ONLY.
- Eager-attn pin (bpt_metric.py:100-106, wikitext_ppl analogue) UNTOUCHED.
- Blackwell grouped_mm shim UNTOUCHED (spec mandate; verified not in these files).
- Rollback: revert to labels=batch + out.loss; self-contained per file.

---

## ORDERING

1. Item 1 first (highest priority, lowest risk, pure docs+advisory). Lands the honesty
   fix + geometry pin so any subsequent perf work reports against a pinned geometry.
2. Item 3 second (memory; unblocks running the real eval on the 16 GB host / <94 GB
   cards at all). Bit-identity test gates it.
3. Item 2 last (ProcessPool; most moving parts — spawn/pickle/kill). Benefits from
   Item-3 having made the forward path runnable on the host so end-to-end smoke can
   exercise scoring.

Each item is independent (different files, no shared edits) -> could land in parallel,
but the above order minimizes integration risk.

---

## TESTING PLAN ON HOST RTX 5080 (16 GB, Python 3.12.3, mp=fork)

1. Unit suite (fast, no GPU pressure):
   `pytest max_quality/tests/test_stage6_plugin_humaneval.py \
          max_quality/tests/test_stage6_plugin_wikitext.py \
          max_quality/tests/test_stage6alt_plugin_bpt.py \
          max_quality/tests/test_stage6_plugin_math500.py -v`
   plus the new bit-identity / pickle / timeout tests.
2. Golden gate (must stay byte-identical):
   `pytest max_quality/tests/test_stage6_golden_snapshot.py \
          max_quality/tests/test_stage6alt_golden_snapshot.py -v`
3. Item-2 leak proof: the new timeout test asserts executor._processes is empty
   post-shutdown; additionally a manual `ps --ppid <pid>` sanity check shows no orphaned
   children after a deliberately-hung snippet.
4. Item-3 memory proof on the 16 GB card: run _wikitext2_ppl / _bpt_from_nll on
   tiny_model with a small chunk; for a real-geometry smoke use B=1, L=2048, real vocab
   and watch nvidia-smi peak stay well under 16 GB with chunking vs OOM without (the
   OOM-without is itself the demonstration on this card).
5. fork->spawn verification (Item 2): assert multiprocessing.get_context("spawn") is what
   the pool uses; confirm a worker does NOT import torch (verify the transitive import set
   at implement time in a fresh interpreter).

---

## CONSTRAINTS RE-AFFIRMED
- tests/golden/stage6/stage6_eval.json stays byte-identical (structural — 0.1).
- No generation batch-behavior change (Item 1 documents/pins; does not re-batch).
- Eager attention pin (F-S-M-1) untouched in all three items.
- Blackwell grouped_mm shim untouched (not present in the edited files; verified).
- No metric re-bless; bit-identity proven for Item 3 before merge, else escalate.
- No monkeypatch in new tests (project rule) — use caplog, direct calls, real tiny_model fixture.
