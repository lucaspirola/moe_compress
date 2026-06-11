# Stage 3 Speed-Up Analysis — quality-neutral opportunities

Scope: `max_quality/src/moe_compress/stage3/*` (+ `utils/activation_hooks.py`).
Goal: cut Stage-3 wall-time **without changing the fp output** (SVD factors,
ranks, EoRA factors within fp tolerance). Multi-GPU model-sharding + DP
cov-reduce + task-parallel EoRA already landed (`e395ad0`); this hunts for
ORTHOGONAL wins that compound with it.

Convention: **VERIFIED** = read in code with file:line; **INFERRED** = derived
from the structure but not directly measured. No timing was measured in this
pass (read-only); all % figures are structural estimates, flagged as such.

---

## 1. Wall-time breakdown (live run: cross_covariance=true, 40 layers, bs≈4)

Reported live profile: cov collection ≈ **27 min/layer × 40 ≈ 16 h/arm**, plus
11-candidate alpha grid + 25-epoch block_refine. Ranked dominant costs:

| # | Sub-step | Code | Est. share of Stage-3 wall | Why it costs that |
|---|----------|------|---------------------------|-------------------|
| 1 | **Cov collection dual-forward** | `covariance_collection.py:438-499` | **~80-90%** (~16 h) | Runs **one full calibration pass PER MoE layer**, sequential (`for k, ref in enumerate(moe_layers)`, line 449), and inside each pass forwards BOTH the 35B teacher and the 35B student over the *entire* depth (lines 494-499). Only the current layer's experts are instrumented (`instrument_experts`, lines 475-486). VERIFIED. |
| 2 | **Alpha-grid PPL validation** | `swift_svd_alpha.py:556-682` | **~5-10%** (~30 min) | 11 candidates × (factor full model → WikiText-2 PPL eval → restore). `_factor_model_at_ranks` re-runs `eigh(B)` per (layer,expert) **inside** the candidate loop (line 446). VERIFIED. Only engaged when `validation_samples>0` (prod = 512). |
| 3 | **block_refine (Phase C.5)** | `block_refine.py:344-577` | **~3-8%** | 25 epochs × N MoE blocks × AdamW. Teacher targets ALREADY cached per-batch (line 469-534), so per-epoch teacher recompute is gone. Student forward + backward is irreducible compute. VERIFIED cache. |
| 4 | **SVD factorization (main loop)** | `aa_svd_factor.py:480+`, `_aa_svd` | **~1-2%** | One eigh + one SVD per (layer,expert,matrix); gate/up already share one eigh via `_precompute_eigh`+`_aa_svd_precomputed`. VERIFIED reuse already in place. |
| 5 | **Model loads** | `orchestrator.py:350-439` | **<1%** | Teacher load ~60s, skipped on resume when block_refine off (line 304-323). VERIFIED. |

The picture is overwhelmingly dominated by **#1** — the N×-redundant
full-depth dual-forward. Everything else is rounding error against 16 h.

---

## 2. Speed-up ideas

### (A) QUALITY-NEUTRAL — pure perf, output unchanged within fp tolerance

| ID | Idea | File:line | Why output is UNCHANGED | Est. speedup | Difficulty |
|----|------|-----------|--------------------------|--------------|------------|
| **A1** | **Single-pass cov collection** — instrument ALL MoE layers at once (not one layer per full pass) so the dual-forward runs **once** total instead of N times. | `covariance_collection.py:438-499` (the per-layer `for` loop + the inner `for batch` loop) | The accumulator `B_acc.update`/`C_acc.update_cross` is keyed by `(layer, expert, matrix)` (`activation_hooks.py:1014, 1042`) and is a pure linear sum of per-token outer products — **independent of how many layers are hooked simultaneously**. Hooking all layers in one ExitStack produces byte-identical per-key Gram sums; only the iteration order over (layer) changes, and summation of fp32 partials within a key is order-stable here because each token contributes to exactly one (layer,expert) key per pass. The current code ALREADY documents this as an intentional *memory* tradeoff ("Wall-clock cost is ~N× the simultaneous design", line 438-446), not a correctness one. Peak RAM is the blocker, mitigated by the existing per-layer spill (each layer still `finalize_layer`+spills as its forward completes within the single pass). | **~N× ≈ up to ~40× on cov collection** (the dominant 80-90%); realistically bounded by per-layer-spill cadence and the teacher-hidden RAM for cross-cov. Even a partial batching (hook G layers per pass) gives ~G×. | **High** — must restructure `_teacher_hidden` lifetime (currently cleared per layer, line 469/492) to hold all hooked layers' teacher rows for one pass; cross-cov RAM grows ∝ layers-hooked. Mitigate by hooking a window of layers and spilling as you go. THE BIG ONE. |
| **A2** | **Cache `eigh(B)`/rhs across the 11 alpha candidates** in the PPL validation search. | `swift_svd_alpha.py:436-449` (inside the `for ref` loop, itself inside the `for alpha` loop at line 636) | `_precompute_eigh(B,A,C)` depends ONLY on the covariances B,A,C — NOT on `k`/alpha (`aa_svd_factor.py:187-262`; `A` is `del`-eted, line 230). Across all 11 candidates B/A/C are identical (same spill files). Only `k` changes, and `k` is consumed solely by `_aa_svd_precomputed` (line 462-464). Precomputing the `_EighDecomp` per (layer,expert) ONCE and reusing it for all 11 candidates yields bit-identical factors. The spectral-proxy path ALREADY does exactly this caching (`grouped_svs_cache`, line 907/942/1130); the validation path simply never got it. | **~10-11× on the alpha search** (eigh is the bulk of `_factor_model_at_ranks`; SVD of `M=W@rhs` is cheaper and still per-candidate but unavoidable since `k` differs). Net Stage-3: ~5-9% × (10/11) ≈ **a few %**. | **Medium** — hoist a `{(layer,expert): _EighDecomp}` cache outside the candidate loop in `_swift_svd_plus_alpha_search_validation`; per-layer the B/C spill is already lazy-loaded. RAM: one layer's decomps at a time if loop stays layer-major. |
| **A3** | **Cross-arm teacher-cov reuse** — the two REAP/REAM arms at the SAME net budget share the SAME teacher model; the teacher-side `X_pre` (and thus the *B-only* student-independent quantities) differ per arm only via student routing, BUT the **A_cov (Stage-2 input cov)** and the teacher forward activations are arm-independent. | `orchestrator.py:386-419` (DP path reloads teacher per replica), run-runner level | A_cov is loaded from a shared Stage-2 sidecar already (`orchestrator.py:146-180`) — VERIFIED shared. The cross-cov `C=X_pre^T X_post` is NOT arm-independent (X_post depends on the arm's pruned student routing — `covariance_collection.py:362-419`), so C cannot be shared. **Only the teacher *load* and A_cov are shareable**, which is already done. Genuine reuse is limited. | Small (teacher load is <1%). Listed for completeness — mostly already realized. | Low, but low payoff. |
| **A4** | **Vectorize the per-token cross-cov Python loop** (already flagged MEDIUM-2 in-code). | `covariance_collection.py:359-360, 397-403` (`_teacher_input_cb` builds a `{tidx: row}` dict; `input_cb` stacks row-by-row) | Building `X_pre` by indexing the teacher tensor with the student's `token_idx` (a single `index_select`) instead of a Python dict-build + per-row `torch.stack` produces the SAME `X_pre.T @ X_post` matmul input — pure reshaping, identical fp result. The matmul `cross = X_pre.T @ X_post` (line 411) is unchanged. | Removes ~256 small tensor builds/batch on the CPU side; speeds the CROSS-COV bookkeeping, which overlaps GPU but can be the CPU bottleneck at small batch. Est. **5-15% on cov-collection CPU overhead**, more when bs is small (current bs=4..16). | **Medium** — needs the teacher hook to store one `[n_tok, d]` tensor per layer keyed densely by position, then `index_select(0, token_idx)`. Care with token-position alignment across the dict→tensor change. Compounds with A1. |
| **A5** | **fp16 (not bf16) cov storage is already chosen; keep `eigh` in fp32 not fp64.** Confirm no stray fp64. | `aa_svd_factor.py:207-209` (`eigh` runs in **fp32**), `covariance_collection.py:97-111` (fp16 store) | The eigendecomposition is already fp32 on-device (line 207), NOT fp64 — matching the "factors fp32-GPU" rule from prior Blackwell notes. No fp64 svdvals on the hot path was found in stage3 factor code. So there is **no fp64-on-GPU penalty to remove here** (unlike the rank-deciding spectra elsewhere). This is a NO-OP confirmation, not a change — flag so nobody "optimizes" it by dropping to bf16 (which the docstring says worsens rank outcomes). | 0% (already optimal); listed to PREVENT a regressive change. | n/a |
| **A6** | **Larger cov batch_size once model-sharded.** Cov forward is `no_grad` (config note `batch_size:16 — B-cov forward runs under no_grad → cheap`). With model sharding freeing per-card VRAM, raise bs. | `orchestrator.py:190-206` comment explicitly says sharding "frees activation headroom for a larger batch_size … MEASURED on first sharded run" | The Gram accumulator sums per-token outer products; the per-token contribution is independent of batch grouping → **batch size does not change the covariance** (sum is associative over tokens, fp32 partials). Identical output. Larger bs = fewer kernel launches + better GEMM utilization. | **~1.5-3×** on cov collection depending on the achievable bs (activation-bound; measure). Compounds with A1 (single pass) multiplicatively. | **Low** — config bump + a MEASURED VRAM check (the code already anticipates this). |
| **A7** | **Skip the slow Python expert-loop forward for cov collection** — `instrument_experts` replaces the fused/grouped expert GEMM with a per-expert Python `for` loop (`activation_hooks.py:1463-1520`). For NON-instrumented layers the native fast path runs; but the instrumented layer pays the Python-loop tax on BOTH teacher and student. | `activation_hooks.py:1453-1521` | The hook only needs the per-expert INPUT (`sel`) for the covariance; the *output* (down/index_add) is recomputed but discarded by cov collection. If the cov pass needs only inputs, a lighter hook that captures inputs via a pre-hook on the native fused forward (no Python expert loop) yields the SAME captured `X` → SAME Gram. Under A1 (all layers hooked) this tax would otherwise hit EVERY layer, so removing it is important. | Hard to size without measuring the Python-loop overhead vs GEMM; INFERRED meaningful at 256 experts/layer. Could be **1.2-2×** on the instrumented forward. | **Medium-High** — requires a capture-only hook path that doesn't replace forward with the Python loop; must preserve exact `token_idx`/routing semantics used by cross-cov. Risk of subtle divergence → validate Gram byte-equality. |

### (B) QUALITY-TRADING — DO NOT treat as free; need validation

| ID | Idea | Risk |
|----|------|------|
| **B1** | Use the **spectral-proxy** alpha path (`validation_samples=0`) instead of the 11-candidate end-to-end PPL grid. | Changes which alpha is selected → different ranks → **different output**. The code explicitly removed the silent proxy fallback as "non-paper-compliant" (`swift_svd_alpha.py:333`, `orchestrator.py:625-681`). This is a STATISTICAL trade, not a perf optimization. Saves ~all of #2 but must be quality-validated. |
| **B2** | Fewer alpha candidates (coarsen `alpha_grid` to e.g. 6). | Coarser grid can miss the PPL-optimal alpha → different ranks. Quality trade. |
| **B3** | Fewer calibration sequences / fewer block_refine epochs (25→fewer) / drop `cross_covariance`. | All change the statistical estimate or the objective → different factors. Explicitly out of scope per the quality-neutral constraint. |
| **B4** | bf16 (instead of fp16) cov storage to halve disk/transfer. | Docstring (`covariance_collection.py:106-109`) reports fp16 gives "cleaner Stage-3 rank-deficiency outcomes than bf16" — i.e. rank decisions can flip. Quality trade. |
| **B5** | Drop block_refine entirely or reduce to dense-only. | Removes the cross-block cascade correction (paper §3.3). Quality trade. |

---

## 3. Recommended ordered quality-neutral wins (biggest bang first)

1. **A1 — single-pass (or windowed) cov collection.** Kills the N× redundancy
   that IS the 16 h. Even a windowed variant (hook G≈4-8 layers/pass) gives
   G×. Highest payoff, highest effort. Restructure `_teacher_hidden` lifetime +
   per-layer spill-as-you-go. *Compounds multiplicatively with A6 and the
   already-landed DP/model-sharding levers (DP splits the SINGLE pass across
   replicas; A1 removes the N factor; together they multiply).* 

2. **A6 — bump cov `batch_size` on the sharded run.** Trivial config change,
   the code already anticipates it; ~1.5-3×, multiplies with A1.

3. **A4 — vectorize the per-token cross-cov loop.** Removes the documented
   CPU hotspot; matters more as A1 increases hooked-layer count. Medium effort.

4. **A2 — cache `eigh`/rhs across alpha candidates in the PPL validation
   search.** ~10× on the alpha search (a few % of Stage-3). Self-contained,
   medium effort, mirrors the existing `grouped_svs_cache` pattern.

5. **A7 — capture-only cov hook (skip the Python expert-loop forward).**
   Higher risk (must prove Gram byte-equality), defer until A1 lands since A1
   makes the per-layer-forward tax hit every layer.

A3 and A5 are confirmations/already-realized — no action, but A5 is a
**guardrail**: do not "optimize" the fp32 eigh down to bf16.

### Interaction with the just-landed multi-GPU work
- **DP cov-reduce** (lever 2) splits ONE pass across G replicas. A1 removes the
  N-pass factor *within* each replica → the two multiply (G × N speedup
  combined). Fully orthogonal. (`covariance_collection.py:673-748`.)
- **Model sharding** frees per-card VRAM → enables **A6** (bigger bs). Orthogonal.
- **Task-parallel EoRA** (Stage 4) does not touch Stage-3 cov/alpha/refine — no overlap.

---

## 4. Explicit "DO NOT do without validation" list

- Do **not** set `validation_samples=0` / use spectral proxy to skip the PPL
  grid (B1) — changes selected alpha → different model.
- Do **not** coarsen `alpha_grid` (B2).
- Do **not** cut calibration sequences, block_refine epochs, or disable
  `cross_covariance` (B3) — these are the algorithm, not overhead.
- Do **not** switch cov storage fp16→bf16 (B4) — can flip rank decisions.
- Do **not** drop/reduce block_refine (B5).
- Do **not** "simplify" the fp32 `eigh(B)` to bf16/fp16 (A5 guardrail) — fp32 is
  the deliberate precision-safe choice; lower precision risks rank flips.

---

### Verified vs inferred summary
- VERIFIED in code: the N-pass-per-layer structure (A1), the per-candidate eigh
  recompute in the PPL path + the existing spectral-proxy cache (A2), teacher-
  target caching already present in block_refine (#3), fp32 eigh (A5), shared
  A_cov load (A3), `no_grad` cov forward + sharding-frees-headroom comment (A6),
  the Python expert-loop forward in `instrument_experts` (A7), the per-token
  cross-cov Python loop + its in-code MEDIUM-2 flag (A4).
- INFERRED (not measured this pass): all % speedup figures (structural
  estimates), and the magnitude of the Python-loop tax in A7. Recommend a
  one-layer timing probe before committing A1/A7 effort.
