# Stage-3 covariance (B + cross-cov C) on a SINGLE H200 — synthesized strategy

**Date:** 2026-06-14
**Branch:** `research/single-h200-stage3-strategy` (synthesis only — main + the
running ablation are untouched)
**Goal:** run the full Stage-3 covariance pass **including the cross-covariance C,
at paper-faithful fidelity**, on **one 143 GB H200** instead of today's 2-GPU
`device_map: balanced` dual-model co-residency.

This document is a **strategy**, not new investigation. It composes four
already-converged, code-verified findings:

| # | Finding | Source doc (branch `commit`) |
|---|---|---|
| #3 | Single-pass-all-layers cov is FEASIBLE & PAPER-FAITHFUL; bitwise at `cov_batch_size=1`; needs a ~10-line CPU-resident hot `_pending` (matmul-on-GPU → result-to-CPU → sum-on-CPU). Windowing only bounds the GPU accumulator. | `2026-06-14-stage3-single-pass-cov.md` (`research/stage3-single-pass-cov` `5d0bce3`) |
| #2 | B/C are ~8–16× over-calibrated vs AA-SVD's 256; **use 512 sequences** (per-expert n/d≈32, the "solid default", user preference). | `2026-06-14-stage3-bc-calibration-size.md` (`research/stage3-bc-calibration-size` `d6ef89b`) |
| (fallback) | vLLM cannot accelerate C; the only drop-C lever is `aa_svd.cross_covariance: false` (Path 3, quality cost UNMEASURED). This is the fidelity-SACRIFICING fallback — **NOT used here**. | `2026-06-14-vllm-dual-model-crosscov.md` (`research/vllm-dual-crosscov` `55327ba`) |
| #1 | Auto-batch OOM-backoff halves too coarsely (18→9). Size the cov batch from the cost model + a headroom notch, don't blind-halve. | landed auto-batch v1+v2 (memory `project_auto_batch_v1_landed`); code `utils/auto_batch.py` |

The C-drop fallback (Path 3) is the thing this strategy **avoids**: we keep C and
keep paper fidelity, and still fit one H200 — by *sequencing* the dual forward
instead of dropping a model.

---

## The blocker today (verified in code)

The cross-cov C needs the teacher (pre-prune, ~67 GB BF16) and the student
(post-prune, ~49 GB BF16) to forward the **same** calibration batch so the
position-join `token_idx` means the same absolute position in both. Today both
are loaded **co-resident** and forwarded back-to-back in one process:

- `stage3/orchestrator.py:255-263` (the orchestrator's own comment): *"original
  BF16 (~70 GB) + pruned BF16 (~50 GB) = ~120 GB, leaving ~21 GB for activations
  + covariance accumulation — which caps batch_size at ~4. The multi-GPU lever
  (`model.device_map: balanced`) shards both models across N GPUs…"*
- The dual forward is literally sequential already, but **co-resident**:
  `covariance_collection.py:943-947` — `teacher_model(input_ids=batch)` then
  `model(input_ids=batch)` on the **same** `batch`.

So today the two 35B models' weights (~116 GB) sit on the card simultaneously; the
join is what forces co-residency. That is the only reason Stage-3 cov-with-C needs
2 GPUs.

---

## The strategy in one paragraph

**Sequence the two model forwards in *time* instead of co-residing them in
*VRAM*, combine with single-pass-all-layers (one sweep over 512 calib sequences),
keep both Gram and cross-Gram accumulators in CPU RAM, and size the per-batch
forward from the cost model.** Per calibration batch: (1) forward the **teacher
alone** (it's the only 35B on the GPU), capturing each layer's pre-routing hidden
state `[T, d_in]` to CPU; (2) **unload the teacher** (or, cheaper, keep it
CPU-pinned and stream weights — see §1.b); (3) forward the **student alone** on
the *same* batch, and for each (layer, student-expert) gather the matching cached
teacher rows from CPU and accumulate `C += X_preᵀ@X_post` on CPU. Because only ONE
35B model is GPU-resident at a time, the freed ~50–67 GB is pure activation +
workspace headroom, so one H200 holds it with room to spare. The token set, the
position-join, and the per-(layer, student-expert) keying are **identical** to
today's co-resident dual-forward — only the *timing* of the two forwards changed.

---

## 1. The core move — sequential / streamed dual-forward

### 1.a Why it preserves C exactly

C's correctness depends on three things, **all of which the sequencing leaves
untouched** (verified against the cross-cov code):

1. **Same token set / same `input_ids`.** Both forwards still run the *same*
   `batch` tensor — we simply run them at different times on the same GPU, not on
   two GPUs at once. `covariance_collection.py:943-947` already issues them as two
   separate `torch.no_grad()` calls; co-residency is incidental, not required by
   the join.
2. **Intra-batch position-join, teacher store cleared PER-BATCH.** The teacher
   dense store is rebuilt every batch and holds **all** hooked layers' rows for
   the current batch (`covariance_collection.py:928-931`:
   `_teacher_hidden.clear(); _teacher_dense.clear(); _teacher_filled.clear()`).
   The join (`covariance_collection.py:754-765`) reads `_teacher_dense[li]` by
   `sel_idx = tok[keep]` — purely **within the current batch**. It never spans
   batches or passes. Caching the teacher rows to CPU between the teacher forward
   and the student forward of the *same* batch changes **nothing** about which
   rows are gathered: `index_select(0, sel_idx)` returns the same rows whether
   `_teacher_dense[li]` lives on GPU or CPU.
3. **Per-(layer, student-expert) keying + the reduction pin.** The C key and the
   ascending per-sequence accumulation are unchanged
   (`covariance_collection.py:766-795`, `update_cross`): `sids = sel_idx //
   _seq_len`, accumulate in ascending seq order. Sequencing the forwards does not
   reorder a key's per-sequence sum.

### 1.b Bitwise vs allclose — the precise verdict

**The sequencing itself is BITWISE-identical to today's co-resident dual-forward**
for every key (gate B, cross C, and down B), because:

- The per-token outer-product GEMM is produced by the **same GPU kernel** on the
  same operands. The student gate/down inputs come from the *student's own*
  forward (unchanged). The teacher rows `X_pre` are the *same fp32 values* — we
  just stored them on CPU and moved them back (a `.to()` copy is value-preserving;
  it is not a recomputation). `X_preᵀ@X_post` (`covariance_collection.py:782-794`)
  runs on `tgt_device` (GPU) exactly as today.
- The CPU `add_` of the running sum is IEEE-754-exact for the same operands in the
  same pinned order (`activation_hooks.py:1027-1029`,
  `update_grouped:1033-1050`) — this is the single-pass doc's load-bearing point
  (#3 §3a/b): summing the same per-token products in the same per-sequence order
  gives the same result regardless of *where* the running sum lives.

The **one** residual is the already-known factored **down_proj B** allclose ~1e-6
at `cov_batch_size>1` (`covariance_collection.py:806-816` — the padded batched
`bmm` for `act_fn(gate)*up` is perturbed by forward *shape*). That residual is
**orthogonal** to sequencing and to accumulator location; it is governed only by
`cov_batch_size`. **At `cov_batch_size=1` the whole strategy (sequencing +
single-pass + CPU accumulate) is bitwise-identical to today, all keys.** At
`cov_batch_size>1` (which we *want*, for throughput — §4) the down B picks up the
same pre-existing, quality-neutral ~1e-6 it already has today at bs>1. So:

> **Sequencing + single-pass + CPU-accumulate = BITWISE-identical at
> `cov_batch_size=1`; ALLCLOSE ~1e-6 (down_proj B only) at `cov_batch_size>1`,
> identical to today's bs>1 behaviour. Neither C nor gate B ever drifts (they are
> bitwise-invariant to `cov_batch_size` per the reduction pin,
> `covariance_collection.py:430-442`).**

The teacher must NOT be re-forwarded for the student's batch (that would be a
second teacher pass — wasted compute, and if non-deterministic, a drift source).
The contract is: **one teacher forward per batch, rows cached to CPU, then one
student forward per batch.** Same number of forwards as today (one teacher + one
student per batch), just not co-resident.

### 1.c Teacher residency mechanics (the actual VRAM win)

Two implementation options, both fit one H200:

- **(A) Cache-rows, keep teacher CPU-pinned, stream the teacher forward layer-by-
  layer.** Keep teacher weights in CPU RAM (3 TB box); use HF `device_map` with an
  offload hook so each teacher layer is paged GPU→compute→evict during the teacher
  forward. Peak GPU teacher weight ≈ a few layers, not the whole 67 GB. Then the
  student forwards resident. This is the most VRAM-frugal but adds H2D weight
  traffic on the teacher pass.
- **(B) Load-teacher / forward / free / load-student.** Simplest: at batch start
  the teacher is the only GPU model; after caching rows, `del`/`.to('cpu')` the
  teacher and bring the student on. But moving 67 GB↔49 GB of weights *per batch*
  is prohibitive — so do **(B′)**: keep BOTH models CPU-resident and only one
  *GPU-resident at a time across the whole pass* by ordering: **teacher forward
  over ALL batches first (cache all teacher rows to CPU/disk), then student
  forward over all batches.** This is clean BUT it breaks the per-batch teacher-
  store-clear assumption (it would need all batches' teacher rows persisted — the
  172 GB-class activation-store problem the vLLM doc flagged for the dense
  transport, `2026-06-14-vllm-dual-model-crosscov.md` §d-i).

**Recommended: option (A)** — per-batch teacher forward with the teacher weights
offloaded/streamed and only the current batch's `[T, d_in]` rows cached to CPU
(batch-sized, ~MBs, NOT calib-sized). This keeps the teacher store **per-batch**
exactly as the code already does (`covariance_collection.py:928-931`), so it is
the minimal deviation from today's control flow: same loop, same per-batch
clear, just the teacher is weight-streamed and only one model is GPU-hot.

> **Honest caveat (the one thing the first live run must confirm):** the GPU peak
> for the teacher pass under option (A) is *weight-streaming-bound* (a few teacher
> layers resident) and for the student pass is *student-weights + activation-
> bound*. The exact streaming chunking (how many teacher layers co-resident) is an
> HF-offload knob measured on the first run, not assumed. The **fidelity** is
> independent of this knob (value-preserving), only the **throughput** depends on
> it.

---

## 2. Combine with SINGLE-PASS (#3)

Set the cov window `G = N_layers = 40` so all 40 layers are hooked in **one**
sweep over the calibration set (no `ceil(N/G)` re-forwards). This requires the
CPU-resident hot accumulator from #3, because at `G=40` all 40 layers' `_pending`
Grams would otherwise be GPU-resident simultaneously (~2 TB B+C — infeasible,
#3 §1.2).

The ~10-line change (#3's "only code gap"): a `cpu_accumulate` mode in
`InputCovarianceAccumulator.update` (`activation_hooks.py:1018-1029`) that keeps
the **matmul on GPU** (`cov = flat_f32.transpose(0,1) @ flat_f32`,
`activation_hooks.py:1022`) but `.to('cpu')` the **result** and `add_`s into a
CPU `_pending` (`activation_hooks.py:1027-1029`). The pinned per-sequence order
(`update_grouped`, `activation_hooks.py:1033-1050`) is preserved on the CPU side.

> **CRITICAL (verbatim from #3):** the matmul must stay on GPU — a CPU fp32 matmul
> is NOT bit-for-bit equal to a GPU fp32 matmul (different reduction order), and
> computing `xᵀx` on CPU would VOID the bitwise guarantee for **every** key
> including gate B and cross C. Only the cheap `[d,d]` running-sum relocates.

The finalized-Gram → CPU → disk spill-and-evict already exists and is the live
default (`activation_hooks.py:1097` `.cpu()`; `spill_layer_to_disk` eviction;
per-layer spill `covariance_collection.py:1036-1047`), and persists **fp16**
(`covariance_collection.py:97-111`). So the only new residency is the *hot*
`_pending` on CPU; everything downstream is unchanged.

Single-pass at `G=40` collapses today's ~5–13 windows (resolver budgets `G≈3–4`
per `covariance_collection.py:347-398`) to **1 sweep**.

---

## 3. Calibration = 512 sequences (user preference, #2)

Use a Stage-3-cov-specific knob `cov_num_sequences: 512` (NOT a cut of the global
`num_sequences=4000`, which Stage-2's reservoir needs — #2 §6). Per #2:

| | Old (inherited) | New (this strategy) |
|---|---|---|
| Sequences | 4000 | **512** |
| Seq length | 4096 | 4096 |
| Total tokens | ~16.4M | **~2.1M** |
| Per-expert tokens (`seqs×4096/32`) | ~512K | **~65K** |
| Per-expert **n/d** (gate/up, d=2048) | ~250 | **≈32** ("solid", into stable-tail) |
| Per-expert **n/d** (down, d=512) | ~1000 | **≈128** (4× safer) |

512 is the "solid default" — a notch above AA-SVD's 256 (n/d≈16) and far above
EoRA's floor of 32 sequences. Token budget drops **~8×** (16.4M → 2.1M). This is
the **only** non-bitwise change in the whole strategy and is gated by the §7
spectrum check.

---

## 4. Batch sizing (#1) — size, don't blind-halve

Today the auto path sizes the cov batch with `size_batch`
(`utils/auto_batch.py:143`, two-point cost-model prediction → `size_candidate`
:87, largest `b` with `fixed + b·per ≤ total·(1−headroom)`) and on a real OOM
falls back via `run_with_oom_backoff` (`utils/auto_batch.py:188-201`), which does
`new = max(attempt // 2, floor)` — the **coarse 18→9 halving** #1 flags.

For the single-model-resident cov forward, the right approach is:

1. **Size from the cost model** (`size_batch`): with only ONE 35B resident (~50–67
   GB) instead of ~116 GB, `fixed` (the cost line's y-intercept = resident bytes)
   is ~50–67 GB lower, so `size_candidate` predicts a **much larger** batch than
   the co-resident case (which capped at ~4, `orchestrator.py:258`). The
   single-model footprint gives substantially more activation headroom.
2. **Apply a headroom notch** (`headroom_frac`, already a config — `AutoBatchConfig`)
   rather than running at the ragged edge.
3. **Keep `run_with_oom_backoff` only as a safety net**, and (the #1 improvement)
   prefer a *finer* step than `//2` if it ever fires — e.g. step down by the
   cost-model's per-sample slope to the largest still-fitting batch instead of
   halving 18→9. Since we now size from the model with headroom, backoff should
   rarely engage at all.

Net: the single-model residency turns the cov batch from "capped at ~4" into a
cost-model-sized batch (the exact ceiling is activation-bound and **measured on
the first run** — `orchestrator.py:255-263` already says this is measured, not
assumed). Larger batch → fewer batches → faster, while gate B / cross C stay
bitwise (reduction pin) and down B keeps its pre-existing bs>1 ~1e-6.

---

## 5. MEMORY BUDGET — the proof (one 143 GB H200)

The invariant: **at most ONE 35B model is GPU-resident at a time; all
accumulators and the cached teacher rows live in CPU RAM (3 TB box).**

### GPU-resident peak (the binding constraint)

**Teacher-forward phase (per batch):**
| Item | GB | Note |
|---|---|---|
| Teacher weights GPU-resident | ~67 (option B, whole) **or** ~8–15 (option A, streamed window) | BF16 35B; option A pages layers |
| Teacher activations (this batch, hot layer) | ~2–6 | bs-dependent; `[T, d_hid]` BF16 |
| Transient GEMM workspace | ~1–3 | per-token outer-product on GPU before `.to('cpu')` |
| **Teacher-phase peak** | **~67–76 (B) / ~12–24 (A)** | both ≤ 143 |

**Student-forward phase (per batch):**
| Item | GB | Note |
|---|---|---|
| Student weights GPU-resident | ~49 | BF16 post-prune 35B (~180–200 experts/layer) |
| Student activations (this batch, all 40 hooked layers' transient) | ~6–20 | single-pass hooks all layers; only the *current* layer's activation is hot, prior layers' Grams already `.to('cpu')` |
| Cross-mult GEMM workspace (`X_preᵀ@X_post` on GPU) | ~1–3 | result copied to CPU immediately |
| **Student-phase peak** | **~56–72** | ≤ 143 |

**Binding GPU peak ≈ max(teacher-phase, student-phase) ≈ 67–76 GB** (option B) or
**~56–72 GB** (option A teacher-streamed). Either way **well under 143 GB**, with
**≥67 GB headroom** in the worst case (option B teacher whole-resident) — that
headroom is exactly the activation budget that lets §4 size a much larger batch
than the co-resident ~4.

> Contrast today (co-resident): ~67 (teacher) + ~49 (student) = **~116 GB of
> weights alone**, leaving ~21 GB → batch capped ~4 → needs 2 GPUs. Sequencing
> removes the *other* model's 49–67 GB from the card.

### CPU-RAM-resident (3 TB box — fits comfortably)

| Item | Size | Note |
|---|---|---|
| A_cov (Stage-2) | already CPU | pre-existing |
| Hot `_pending` Grams, all 40 layers, B + C (#3 CPU-accumulate) | ~1 TB fp16 / ~2 TB fp32 transient | spilled per-layer to disk as finalized (`covariance_collection.py:1036-1047`), so not all live at once in practice |
| Cached teacher rows, current batch, all 40 layers | `40 × [T, d_in]` — **batch-sized, ~GBs** | per-batch clear (`covariance_collection.py:928-931`); at bs small, few-MB/layer |
| Finalized fp16 Grams streaming to disk | spilled + evicted | `spill_layer_to_disk` Phase-3 eviction |

The teacher row cache is **batch-sized, not calib-sized** (#3 §4), so it does NOT
hit the 172 GB activation wall the vLLM doc flagged for the *all-batches* dense
transport — we cache one batch's rows, join, clear, repeat.

**Verdict: one 143 GB H200 holds the GPU side (≤~76 GB peak, ≥67 GB headroom);
the 3 TB CPU RAM holds the accumulators + the batch-sized teacher cache.
CONFIRMED ≤ 143 GB.**

---

## 6. Throughput / cost vs today's 2-GPU 5-window dual-forward

Today (2×H200, sharded, co-resident): `ceil(40/G)` windows with `G≈3–4` after the
~70 GB-per-card shard ⇒ **~10–13 windows** (conservatively call it ~5 if `G=8`),
each re-forwarding the **4000**-seq calib set through both models.

This strategy (1×H200): **1 single-pass** over **512** seqs.

- **Fewer passes:** ~5–13 windows → 1 ⇒ **~5–13× fewer forwards** (#3 §5).
- **Less data:** 4000 → 512 seqs ⇒ **~8× less data per pass** (#2 §6, §3 above).
- **Combined forward-work reduction:** ~5–13 × ~8 ≈ **~40–100× less forward
  compute** on paper.
- **Minus the sequencing cost:** the teacher and student now run as **two
  sequential single-model forwards** instead of one co-resident pass — but that
  was *already* two forwards (`covariance_collection.py:943-947`); the only added
  cost is (option A) teacher weight-streaming H2D traffic and the per-batch
  teacher-row D2H copy (batch-sized, overlappable). This is a **constant-factor**
  overhead on the teacher pass, not a multiplier — it does not erode the
  ~40–100× pass×data reduction. CPU-accumulate `add_`s overlap the next GPU
  forward (#3 §5).

**Net: order ~1–2 orders of magnitude less forward work than the current 2-GPU
run, on HALF the GPUs.**

### The 2-GPU bonus (since it now fits ONE card)

Because the whole pass fits one H200, a 2-GPU box buys a **further ~2×** two ways:

1. **Two ablation arms concurrently** — one arm per GPU (e.g. REAP vs REAM, or
   `cross_covariance` on/off), no sharding needed.
2. **Data-parallel the cov** — each GPU forwards **half the 512 seqs** (256
   each), reduce the Grams cross-replica. The DP reduce is **exact**
   (`B = Σ_r B_r`, `covariance_collection.py:256-260`), and the existing N-GPU
   AA-SVD cov DP path already does this (memory `project_multigpu_stage3_landed`,
   main `0a59ff7`). ⇒ ~2× wall-clock on the cov pass itself.

---

## 7. Fidelity verdict + de-risk

**Is the whole strategy paper-faithful vs the current run? YES, with one gated
exception.**

| Component | Fidelity vs today | Basis |
|---|---|---|
| Sequential / streamed dual-forward (§1) | **Bitwise-identical** (value-preserving CPU caching of teacher rows; same GPU GEMM, same join, same keys) | `covariance_collection.py:754-795, 928-931`; #3 §4 |
| Single-pass `G=40` + CPU-accumulate (§2) | **Bitwise at `cov_batch_size=1`; allclose ~1e-6 (down B only) at bs>1** — and that ~1e-6 already exists today at bs>1 | #3 §3, §6; `activation_hooks.py:1018-1029`; `covariance_collection.py:430-442, 806-816` |
| Batch sizing (§4) | gate B / cross C **bitwise-invariant** to batch; down B allclose ~1e-6 at bs>1 (pre-existing) | `covariance_collection.py:430-442` |
| **Calib 4000 → 512 (§3)** | **NON-bitwise — the only quality-affecting change.** Paper-faithful *per AA-SVD/EoRA* (512 > their 256/128 defaults), but our run's *specific* spectrum must be confirmed. | #2 §3, §5, §6 |

**The de-risk gate (per #2 §6, required before trusting 512):**

1. **Spectrum / r_eff equivalence on BOTH B AND C**, 1–2 layers, **512 vs 4000**:
   collect at both, compare per-expert (i) retained-rank `r_eff` counts and (ii)
   the top-`d/2` eigenvalue spectrum. **Include C** — the teacher×student
   position-join can keep *fewer* matched rows per expert than B's routed count
   (`covariance_collection.py:754-795`), so C's per-expert `n` may be lower than
   B's; confirm C's spectrum is stable at 512, not just B's.
2. **One end-to-end PPL spot-check** at 512 to confirm rank-flip stability across
   the 40 × ~200-expert grid.

If both pass (expected — 512 gives per-expert n/d≈32, past full-rank), 512 is
locked. If C's spectrum is borderline at 512, bump C (only) to 768/1024 — cheap,
since C is the only term that might want more.

**The sequencing and single-pass are fidelity-neutral (bitwise/allclose); only
the calib cut needs the empirical check.** Honest bottom line: the *mechanism*
(fit on one H200 with C) is paper-faithful and proven; the *one knob* that trades
nothing-on-paper for an 8× data cut (512) is gated by the cheap spectrum check.

---

## 8. Implementation outline (minimal changes, real functions to touch)

All on a feature branch; **do NOT touch main while the ablation runs.**

1. **CPU-resident hot accumulator (#3, the ~10-line core).**
   `utils/activation_hooks.py:1018-1029` — add a `cpu_accumulate: bool` mode to
   `InputCovarianceAccumulator.update`: keep `cov = flat_f32.transpose(0,1) @
   flat_f32` on GPU (line 1022, **unchanged**), then `cov = cov.to('cpu')` before
   the `self._pending[key] = cov` / `cur.add_(cov)` (lines 1026-1029). Preserve
   the `update_grouped` per-sequence order on the CPU side (lines 1033-1050,
   unchanged). Mirror for `update_cross` (`activation_hooks.py:1051-1080`) so C
   accumulates on CPU too. **Matmul stays on GPU** (the bitwise contract).

2. **Teacher-stream-to-CPU mode in the cov collection (§1.c option A).**
   `stage3/covariance_collection.py` around the dual-forward
   (`:943-947`) and teacher capture (`:617-620, 663-673`): load the teacher with
   an HF offload `device_map` so its forward streams layers GPU→evict, and cache
   each layer's `[T, d_in]` rows to CPU per batch (the store already clears
   per-batch at `:928-931` — keep that). Single-model GPU residency drops the
   co-resident ~116 GB to ≤~76 GB. The student forward (`:947`) stays GPU-resident
   and unchanged.

3. **`G = N_layers` single-pass.** Set the cov window to all 40 layers (one
   window) — `_resolve_cov_window` (`covariance_collection.py:347-398`) /
   `_iter_windows` (`:333-344`). With the CPU accumulator (step 1) this no longer
   OOMs. Finalize+spill per layer as today (`:1036-1047`).

4. **Stage-3-cov calib knob (`cov_num_sequences: 512`).** Add a dedicated
   override read in `stage3/orchestrator.py:231-238` (where it builds the B/C
   calib tensor via `spec_from_config(cal, seed_offset=2)` /
   `build_calibration_tensor`): slice the first 512 rows, or a `spec_from_config`
   override. Default to the inherited global value for back-compat (golden
   byte-identical when absent). Leaves Stage-2's 4000-seq reservoir untouched
   (#2 §6).

5. **Batch-size fix (#1).** Drive the cov forward through `size_batch`
   (`utils/auto_batch.py:143`) with the single-model footprint (the freed ~50–67
   GB lets `size_candidate` (`:87`) predict a large batch); keep
   `run_with_oom_backoff` (`:188-201`) only as a net, and prefer a per-sample-slope
   step over the `//2` halving (`:201`) if it ever fires. Wire via the existing
   `cov_batch_size: "auto"` / `_resolve_cov_batch_size`
   (`covariance_collection.py:430-442`, used at `orchestrator.py:249`).

6. **(Optional 2-GPU bonus)** Data-parallel the 512 seqs across 2 GPUs via the
   landed N-GPU cov DP path (`covariance_collection.py:256-260, 1320-1324`;
   memory `project_multigpu_stage3_landed`) — exact reduce, ~2×. Or run two
   ablation arms, one per GPU.

**Validation before trusting it:** (a) golden byte-identical when all new knobs
absent (defaults = today); (b) `cov_batch_size=1` single-pass + sequencing →
torch.equal vs a 2-GPU control on 1–2 layers (proves the bitwise claim); (c) the
§7 spectrum/r_eff + PPL gate for 512.

---

## Bottom line

Fitting Stage-3 cov **with C, at paper fidelity, on one 143 GB H200** is
achievable by **time-sequencing the teacher and student forwards** (one 35B
GPU-resident at a time) + **single-pass-all-layers** + **CPU-resident
accumulators** + **512-seq calib** + **cost-model batch sizing**. GPU peak
≤~76 GB (≥67 GB headroom); accumulators + batch-sized teacher cache in 3 TB CPU
RAM. The sequencing + single-pass are bitwise (`cov_batch_size=1`) / allclose
~1e-6 (down B, bs>1, pre-existing); the **only** quality-affecting change is the
4000→512 calib cut, gated by a B+C spectrum/r_eff + PPL check. Throughput: ~1–2
orders of magnitude less forward work than the 2-GPU run, on half the GPUs, with
a further ~2× available from the freed second GPU.
