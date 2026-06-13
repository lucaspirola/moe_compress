# Auto-Batch v2 — ablation_filter NLL Pin + Wire Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use `- [ ]` checkboxes.

**Goal:** Pin `ablation_filter`'s (Stage-1 GRAPE Phase D) corpus-NLL reduction to a fixed per-sequence grouping so it's independent of the forward batch, then auto-size that batch (`size_batch`+`run_with_oom_backoff`) to fill VRAM. Gated behind `stage1_grape.ablation_filter.batch_size: "auto"` + `auto_batch.enabled`. **Default (fixed `batch_size`, fused HF CE) is byte-identical** — the stage1_ablation_filter golden is untouched. The pin doubles as a memory win (manual per-seq CE can chunk the fp32-logits upcast that forced bs=32→8).

**Architecture / fidelity:**
- `_measure_corpus_nll(model, batches, device)` (`ablation_filter.py:383-395`) is a **token-mean** assembled from per-batch fused `out.loss` (HF `ForCausalLMLoss`): `total_nll += loss.item()*ntok; return total_nll/total_tokens`. Reduction-accumulating (drift grows with N), same class as cov.
- **The pin (auto path only):** a new `_measure_corpus_nll_pinned` computes logits (`labels=None`) and a **per-sequence CE** in fixed sequence order — `F.cross_entropy(logits[i,:-1].float(), batch[i,1:], reduction='sum')` accumulated `total_nll += per_seq_sum`, `total_tokens += seq-1`. This matches HF's shift/ignore semantics (all tokens count, fp32 CE) but with the reduction grouping pinned per-sequence → **independent of forward batch**. Used ONLY on the auto path; the DEFAULT keeps the fused `model(labels=batch)` path → golden byte-identical.
- **Why the discrete `ΔNLL > threshold` (default 0.001) is safe:** `baseline_nll` and `ablated_nll` are BOTH measured by the same pinned fn, so once pinned, **ΔNLL is batch-invariant** — the threshold decision is deterministic across batch sizes. Residual = ordinary forward non-determinism (exists at bs=1 too); the ~0.001 margin vs ~1e-6 forward noise is ~1000×. No-worse-than-bs=1; no decision-margin guard needed.
- **Wiring is SIMPLE (vs cov):** `_measure_corpus_nll` returns a FRESH float (no persistent accumulator) → `run_with_oom_backoff(lambda bs: _measure_corpus_nll_pinned(model, iter_batches(holdout, bs), device), start=sized, floor=fixed)` needs NO discard/reset. Size once (cost probe = a 1-seq and 2-seq forward via size_batch), reuse for baseline + every candidate.
- **Honest fidelity:** call `size_batch`/`run_with_oom_backoff` DIRECTLY (cov-class reduction-accumulating, allclose) — NOT `resolve_batch`'s `_V1_ELIGIBLE` gate.

**Tech Stack:** PyTorch, pytest. Code root `max_quality/`. CPU-only for design+impl+golden; live speedup deferred (GRAPE Phase D on a real model).

**Spec:** §5/§10 step 2 (ablation_filter, after its NLL pin). Builds on v1 (`utils/auto_batch.py`) + the cov-wire pattern (main `280d4e4`). NOTE: `block_refine` was classified **METRIC-PINNED** (minibatch-SGD → batch changes trained weights) → NOT auto-batchable, intentionally NOT built (like `gen`).

---

## File Structure
- **Modify** `src/moe_compress/stage1/plugins/ablation_filter.py` —
  - Add `_measure_corpus_nll_pinned(model, batches, device)` (per-sequence CE, fixed order).
  - Add `_ablation_is_auto(af, s1) -> bool` = `af.get("batch_size")=="auto" and AutoBatchConfig.from_dict(s1.get("auto_batch")).enabled` (mirror `_cov_is_auto`). NOTE (Review N1): `af` = `s1["ablation_filter"]` (holds `batch_size`); the `auto_batch` block lives under `stage1_grape` (`s1`), NOT under `af` — confirm the config tree and pass both. `headroom_frac`/`max_cap` come from `AutoBatchConfig.from_dict(s1.get("auto_batch"))`.
  - In `run_ablation_filter`: when auto → size the holdout batch once via `size_batch` (cost probe = a 1-seq and 2-seq `_measure_corpus_nll_pinned` forward), then measure baseline + every candidate's NLL through `run_with_oom_backoff(... _measure_corpus_nll_pinned ...)`. Non-auto → the existing fused `_measure_corpus_nll` at the fixed int (unchanged).
  - `batch_size` parse: keep returning the int floor when not "auto"; `_ablation_is_auto` reads the "auto" sentinel separately (don't break the existing `int(af.get("batch_size", 8))`).
  - `from ...utils.auto_batch import size_batch, run_with_oom_backoff, AutoBatchConfig, CudaMemProbe` (NOT resolve_batch/FidelityClass).
- **Create** `tests/test_ablation_autobatch.py` — forward-free/tiny-model CPU: pinned NLL grouping-independence; default-off byte-identity (fused path, no probe); auto invokes size_batch+backoff; ΔNLL batch-invariance under the pin.
- **Goldens** `tests/golden/stage1/stage1_ablation_filter.json` — NOT TOUCHED (default fixed bs → fused path → byte-identical).

---

## Conventions
- Logger `log = logging.getLogger(__name__)`. No GPU in unit tests. Commit per task; trailer `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

---

## Task 1: `_measure_corpus_nll_pinned` (per-sequence CE) — grouping-independence

**Files:** Modify `ablation_filter.py`; Test `tests/test_ablation_autobatch.py`.

`_measure_corpus_nll_pinned(model, batches, device)`: MUST put the model in inference mode + wrap the whole loop in `torch.no_grad()` (mirror the fused `_measure_corpus_nll:387-388` — WITHOUT `no_grad` the autograd graph over the `[bs,seq,151936]` fp32 logits defeats the memory win and risks OOM, Review N3). Per batch, `logits = model(input_ids=batch).logits`; for each sequence i IN ORDER: `ce = F.cross_entropy(logits[i, :-1].float(), batch[i, 1:].to(device), reduction='sum'); total_nll += float(ce.item()); total_tokens += batch.shape[1]-1`; return `total_nll/max(total_tokens,1)`. The per-seq accumulation is the fixed grouping → batch-independent.

- [ ] **Step 1: Failing test** — tiny LM (monkeypatched/synthetic `model(input_ids=...).logits`); assert `_measure_corpus_nll_pinned` over a batch of B sequences == the same B sequences fed one-per-batch (per-seq grouping invariant: `torch.equal`/exact float on CPU). Plus: it matches the fused `_measure_corpus_nll` within `allclose(atol=1e-5)` (faithful token-mean — use a GENEROUS atol; real seq×vocab sums carry more fp order-noise than a tiny synthetic, Review N2).
- [ ] **Step 2: Run** → FAIL (`_measure_corpus_nll_pinned` undefined). **Step 3: Implement.** **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat(ablation-pin): per-sequence pinned corpus NLL`.

---

## Task 2: Wire size_batch + run_with_oom_backoff (gated, default byte-identical)

**Files:** Modify `ablation_filter.py` (`run_ablation_filter` + `_ablation_is_auto`); extend the test.

- [ ] **Step 1:** Add `_ablation_is_auto(af, s1)`; import the auto_batch helpers. `cfg = AutoBatchConfig.from_dict(s1.get("auto_batch"))`. `batch_size` parse: `auto = _ablation_is_auto(af, s1)`; `fixed_bs = int(af.get("batch_size", 8)) if not auto else 8` (floor when auto).
- [ ] **Step 2:** In `run_ablation_filter`, when `auto`: before the baseline measure, `size_batch(cost_probe_fn, fixed_batch=fixed_bs, headroom_frac=cfg.headroom_frac, max_cap=cfg.max_cap, mem=CudaMemProbe(device))` (use `cfg.max_cap`, cov-consistent — NOT a separate module const, Review L2) where `cost_probe_fn(mb)` runs ONE `_measure_corpus_nll_pinned(model, [holdout[:mb]], device)` (reset_peak/max_memory_allocated) → `nll_bs`. (Note: `size_batch` probes at mb=1 and mb=2 internally — two single-batch forwards — Review L1.) Then a helper `measure(model_)` = `run_with_oom_backoff(lambda b: _measure_corpus_nll_pinned(model_, iter_batches(holdout, b), device), start_batch=nll_bs, floor=fixed_bs)`; use it for `baseline_nll` AND each candidate's `ablated_nll`. Non-auto: the existing `_measure_corpus_nll(model, eval_batches, device)` path, verbatim.
- [ ] **Step 3: Tests:** (a) default-off → fused `_measure_corpus_nll` path, `size_batch`/`run_with_oom_backoff` NOT called (spies); (b) auto → size_batch called once + each NLL via run_with_oom_backoff; (c) ΔNLL batch-invariance: baseline-ablated computed at two different forced batch sizes under the pin → identical ΔNLL (within forward-drift tol); (d) no resolve_batch/FidelityClass import.
- [ ] **Step 4: Implement.** **Step 5: Run** new tests → PASS.
- [ ] **Step 6: GOLDEN GUARDRAIL** — `cd max_quality && python3 -m pytest tests/test_stage1_plugin_ablation_filter.py tests/test_stage1_golden_snapshot.py tests/test_ablation_autobatch.py -q` MUST pass UNCHANGED, no `MOE_REGEN_GOLDEN`. Default config has no `batch_size:"auto"` → fused path → byte-identical `stage1_ablation_filter.json`. If it changes → STOP, report.
- [ ] **Step 7: Commit** `feat(ablation-wire): auto-size + OOM-backoff the holdout NLL forward (gated, default byte-identical)`.

---

## Task 3: Docs
- [ ] Document `ablation_filter.batch_size: "auto"` (+ auto_batch.enabled): pinned per-seq NLL → ΔNLL batch-invariant; auto-sizes the holdout forward with OOM-backoff (also relieves the fp32-logits OOM that forced bs=8). Default fixed int = byte-identical. Note block_refine is METRIC-PINNED (not auto-batchable). Commit `docs(ablation): batch_size auto + block_refine ineligible note`.

---

## Out of scope
- `block_refine` — METRIC-PINNED (minibatch-SGD), intentionally NOT wired. Add a one-line code comment marking it ineligible (Task 3).
- Re-blessing any golden. Live GRAPE-Phase-D speedup (deferred).

## After this plan
Standard plan/review → impl/review loops, all-none. This completes the v2 auto-batch reduction-accumulating plugins (cov done; ablation_filter here; block_refine correctly excluded).
