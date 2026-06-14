# Can patched vLLM capture Stage-3 cross-covariance C faster than HF dual-forward?

**Date:** 2026-06-14
**Branch:** `research/vllm-dual-crosscov` (analysis only — no runner/ablation code touched)
**Question:** Could the patched vLLM fork EVER load BOTH the teacher (pre-prune) and the
pruned/healed student co-resident and capture the Stage-3 cross-covariance **C** faster than
the current HF-eager sharded dual-forward?

## VERDICT: **INFEASIBLE** (as a faster-than-HF C capture).

The patched vLLM's fast in-graph SYRK captures a **per-expert SELF-Gram keyed by the model's
OWN routing**. The Stage-3 cross-cov C is a **cross-model join**: the teacher's hidden state
at *the token positions the STUDENT routed to each student-expert*. Reproducing that join in
vLLM requires (a) two engines driven in deterministic lockstep on identical token sequences,
(b) extracting per-token pre-routing hidden states out of two paged/continuous-batched graphs
and re-joining them by absolute token position, and (c) loading a non-native compressed
student checkpoint into vLLM. Each is a hard blocker; together they erase any throughput win.
The vLLM SYRK patch, as built, is structurally the wrong shape for C and cannot be extended to
a cross-SYRK without rebuilding the join in Python outside the graph — which is exactly the
slow part we'd be trying to avoid.

**Highest-value finding (d-ii):** C is NOT droppable and the marginals (A_cov, B) do NOT
suffice for Path 1 — but C also is NOT needed on the fast path. See "Simpler alternatives"
below: the realistic speedup is to drop the teacher's *dense* per-token activation transport,
not to move C into vLLM.

---

## a. What EXACTLY is C (precise definition, with file:line)

**C = E[X_pre^T · X_post] per `(layer, student_expert, "gate_proj")`**, where:

- `X_post` = the **student's** gate_proj input rows for the tokens the **student** routes to
  student-expert `e` (`covariance_collection.py:531-535`, `:675-697`).
- `X_pre` = the **teacher's** pre-routing hidden state **at those same absolute token
  positions** (`covariance_collection.py:553-569`, `:754-765`).

The contraction is literally `X_pre.T @ X_post` accumulated per expert
(`covariance_collection.py:739`, `:782-784`, `:792-794`), drained by
`InputCovarianceAccumulator.update_cross` (`activation_hooks.py:1051-1080`).

**The teacher↔student expert correspondence is NOT 1:1 and is deliberately avoided.** The
student is pruned/merged (teacher 256 experts → student ~180-200,
`covariance_collection.py:543-546`). C is attributed **per student-expert** using the
**student's** routing to select which teacher token-rows to gather
(`covariance_collection.py:560-569`):

> "for each (layer, expert, token_idx) in the window, look up the corresponding X_pre and
> accumulate C += X_pre^T @ X_post" — where `token_idx` is the **student's**
> `torch.where(mask[e])` dispatch (`activation_hooks.py:1668`, `:1707-1712`).

So C is **not** on a shared pre-routing hidden state in the index-free sense — the teacher's
*input* to the MoE block is shared across experts (pre-routing,
`covariance_collection.py:643-645`), but **which teacher rows enter expert `e`'s C is decided
by the student's routing**. The join key is the **absolute token position within the shared
`input_ids` batch** (`_teacher_dense[layer]` is indexed `[T, d_in]` by `token_idx`,
`covariance_collection.py:663-673`; the student gathers it via
`index_select(0, sel_idx)` with `sel_idx = tok[keep]`, `:759-765`).

**This is the load-bearing fact for feasibility:** C's correctness depends on the teacher and
student forwards seeing the **identical `input_ids` tensor in the same process**, so that
`token_idx` means the same absolute position in both
(`covariance_collection.py:943-947` — `teacher_model(input_ids=batch)` then
`model(input_ids=batch)`, same `batch`). The HF eager path guarantees this trivially: one
dense `[rows, seq]` tensor, one process, deterministic row order, both models hooked.

## b. Can two vLLM engines run in lockstep for an aligned cross-SYRK?

**No, not without defeating the purpose.**

- vLLM is built around a **single** model engine + scheduler + paged KV cache + continuous
  batching. Two models = two `LLM`/`LLMEngine` instances = two independent graphs with
  independent schedulers. There is no shared `input_ids` row-order contract between them.
- The fast in-graph SYRK reads `topk_ids` from **this** model's router
  (`moe_runner.py:678`, `:689-691` — `_flat = topk_ids.reshape(-1)`, counting-sort, then
  `gram_grouped_accum`). It has **no notion of "the other model's routing"** and no notion of
  absolute token position across engines. It is a pure self-Gram per the local router.
- Even forcing `--enforce-eager` + offline `LLM.generate(prompt_token_ids=...)` on both
  engines with identical prompts, vLLM does not expose a stable per-token absolute-position
  index that survives prefill chunking, paged-attention block layout, and continuous-batch
  packing. The whole point of C is the per-position join; vLLM's value (PagedAttention +
  continuous batching, `build_self_traces_calib_vllm.py:23-33`) is precisely the layer that
  scrambles deterministic dense token order. You'd have to disable batching/paging to recover
  order — at which point vLLM is no faster than (and arguably slower than) HF eager for a
  forward-only pass.
- The two models also have **different layer/expert counts** (pruned student), so even a
  notional "same step" doesn't align expert-wise; the join must be by token position, which
  vLLM does not surface.

## c. Could the SYRK patch be extended to a CROSS-SYRK (teacher.T @ student)?

**Not within the graph, and not cheaply.** The existing op
`torch.ops.calib_gram.gram_grouped_accum(cov, count, x_sorted, offsets)`
(`moe_runner.py:693-697`, allocated `calibration_input_cov.py:194-207`) consumes **one**
model's `x_sorted` grouped by **one** model's `offsets` (one counting-sort over one router's
`topk_ids`). A cross-SYRK needs `X_pre` (teacher rows) and `X_post` (student rows) for the
**same** absolute positions, grouped by the **student's** expert assignment. The two streams
live in **two separate engine processes/graphs**; there is no in-graph point where both are
resident. Handing the teacher's per-token pre-routing hidden state from engine A to engine B,
re-indexed by absolute position, is exactly the Python-side join the HF path already does — so
the "cross-SYRK" degenerates into: vLLM teacher forward → dump per-token hidden states → feed
to a second pass. That is alternative (d-i), analyzed below, and it does not need a graph patch
at all.

## d. Simpler alternatives that still use vLLM for speed

### (d-i) Teacher pre-routing activations via fast vLLM, then a student pass cross-multiplies

**Does C decompose this way? YES — C is linear and position-keyed, so it decomposes**, but the
join is the cost, not the teacher forward.

C only needs, per layer, the teacher's **pre-routing MoE input** `h_teacher[pos] ∈ R^{d_in}`
for every token position `pos` (it's identical across the layer's experts —
`covariance_collection.py:643-645`). That is a single `[T, d_in]` tensor per layer per token,
**independent of the teacher's routing**. So one COULD:

1. Run the teacher once through vLLM, capture `block_in`/pre-MoE hidden states per layer per
   token. vLLM already has a `block_out` hook family (`calibration_hooks.py:61-62`) and the
   in-graph capture machinery; a `layer_in`/pre-MoE-input capture is the same shape. Stream
   `[T, d_in]` per layer to disk/CPU.
2. Run the student (HF or vLLM) and, per student-expert dispatch, `index_select` the stored
   teacher rows by `token_idx` and accumulate `X_pre.T @ X_post` — i.e. the **existing**
   `input_cb` join (`covariance_collection.py:754-794`) reading teacher rows from disk instead
   of from a co-resident teacher.

**Cost/benefit of (d-i):**
- **Pro:** removes the teacher from co-residency (frees ~70 GB,
  `orchestrator.py:258-260`), and the teacher forward gets vLLM throughput.
- **Con (fatal to the "faster" claim):** the persisted teacher activation is
  `[T_total, d_in]` fp32 **per layer** — for the 35B at full calib this is the same
  multi-GB-per-layer dense activation tensor that the input_cov OFFLOAD path already found to
  be a **172 GB wall** (see MEMORY: `input_cov_offload_172gb_wall`). You trade VRAM co-residency
  for an enormous activation-streaming bill, plus you STILL run a full student forward with the
  Python per-expert join (the actual slow part). The teacher forward is not the bottleneck; the
  **per-(layer,expert) Python gather + GEMM on the student side** is, and (d-i) keeps it
  verbatim.
- **Determinism risk:** the student pass must produce `token_idx` against the **same** absolute
  positions the teacher capture used. With HF student that's fine (dense order). With a vLLM
  student it reintroduces the position-alignment problem of (b).

Net: (d-i) is a *memory* refactor, not a *speed* win, and it's strictly dominated by simply
**not loading the teacher densely** — which the current code can already approximate (it can
run S-only Path 3, or capture the teacher block input once).

### (d-ii) Is C separable / low-rank — do the marginals (A_cov, B) suffice?

**NO — and this is the most important finding.** Path 1 (Theorem 3.2) is
`M = W · C · B^{-1} · L_B` (`aa_svd_factor.py:364-368`, `:254-256`). The codebase **already
tried** substituting a marginal for C: the retired "Path 2" put the pre-prune
auto-covariance A into the C slot and it **broke**, producing `U·V ≈ W·A·B^{-1}·L_B` instead
of approximating `W` (`aa_svd_factor.py:374-378`; `covariance_collection.py:59-67`). So the
**joint** C is mathematically load-bearing for Path 1; the marginals A and B provably do not
substitute.

**BUT** — C is droppable in the sense that the pipeline has a first-class fallback: **Path 3 /
Corollary 3.3 `M = W · L_B`** (`aa_svd_factor.py:370-372`), used whenever
`aa_svd.cross_covariance: false` (`orchestrator.py:267`). The measured quality gap is
**~0.2 PPL** (`aa_svd_factor.py:67` — "Quality gap is ~0.2 PPL"). down_proj **already** runs
Path 3 unconditionally (no down cross-cov exists — `covariance_collection.py:80-82`,
`:824-829`). So the honest framing is:

- You cannot make C *cheaper* by approximating it from marginals (Path 2 is dead).
- You CAN skip C entirely (Path 3) for a ~0.2 PPL cost — that is the only "drop C" lever, and
  it requires **no vLLM work at all** (flip one config key).

If the goal is "stop paying for the slow dual-forward," the lever is `cross_covariance: false`
(accept ~0.2 PPL), not a vLLM cross-SYRK.

## e. Can vLLM even load the compressed student?

**No (natively).** A_cov (Stage-2 self-cov) was captured on the **original** model, which vLLM
loads natively. The Stage-3 student is `stage2p5_final` (or `stage2_pruned`) —
**stacked `FactoredExperts`** (`orchestrator.py:452-456`), a project-specific compressed
checkpoint format (merged/pruned expert banks + low-rank factors). vLLM has no loader for this
layout; it expects a standard HF MoE weight layout (`calibration_input_cov.py:143-145` assumes
`[E, hidden, intermediate]` unquantized). Running the student through vLLM would require
writing a vLLM model definition for `FactoredExperts` — a large, separate effort with no other
payoff. This alone makes a **two-vLLM-engine** cross-cov a major build even before the
alignment problems of (b).

## f. Net assessment — is any vLLM path actually faster?

**No.** Accounting honestly:

| Lever | VRAM | Speed vs HF dual-forward | Effort | Quality |
|---|---|---|---|---|
| Current HF sharded dual-forward | ~120 GB co-resident (`orchestrator.py:258-260`) | baseline | shipped | exact C |
| Two vLLM engines + cross-SYRK (the asked question) | 2 engines | **slower** (must disable paging/batching for position alignment, §b) + can't load student (§e) | **very large** (vLLM FactoredExperts loader + cross-engine join + new op) | exact C |
| (d-i) vLLM teacher capture → disk → student join | frees teacher (~70 GB) | **not faster** (172 GB activation wall; student Python join unchanged) | medium | exact C |
| (d-ii) `cross_covariance: false` → Path 3 | frees teacher entirely | **much faster** (single forward, no teacher) | **zero** (config flip) | −~0.2 PPL |

The vLLM SYRK is fast precisely because it's **in-graph, single-model, routed-by-self**, with
**no host sync and no cross-model join** (`calibration_input_cov.py:23-25`,
`moe_runner.py:662-697`). Every property that makes it fast is a property C cannot have: C is
cross-model, position-joined, and (on the student side) needs the student's per-expert
dispatch in Python. Forcing C into vLLM removes exactly the properties that gave the speedup,
so the win is **illusory**.

**Recommendation:** If the dual-forward cost is the pain, the realistic options are, in order:
1. **Accept Path 3** (`aa_svd.cross_covariance: false`) for ~0.2 PPL and delete the teacher
   forward entirely — zero engineering, biggest speedup. gate/up join down to Path 3 like
   down_proj already is.
2. If C must stay, keep the HF dual-forward but **drop dense teacher activation transport**:
   the teacher only needs its pre-MoE block input per token (one `[T, d_in]` per layer,
   routing-independent), so the teacher could run **truncated/early-exit** per window
   (vLLM already has `VLLM_CALIB_MAX_LAYER`, `calibration_hooks.py:340-389`) — but this is a
   micro-opt on the existing HF path, not a vLLM-engine swap.
3. Do **not** build the two-engine vLLM cross-SYRK. It is INFEASIBLE-as-faster and
   FEASIBLE-only-WITH-MAJOR-EFFORT-and-no-speedup.

---

### Load-bearing facts (file:line)

1. **C is a cross-model, position-joined, per-student-expert quantity** — teacher hidden state
   at the positions the *student* routes to expert `e`:
   `covariance_collection.py:531-569`, `:754-794`; `activation_hooks.py:1051-1080`.
2. **Alignment requires identical `input_ids` in one process** (teacher then student on the
   same `batch`): `covariance_collection.py:943-947`; join key is absolute `token_idx` into
   `_teacher_dense[layer]` (`:663-673`, `:759-765`).
3. **The vLLM SYRK is a single-model self-Gram routed by the model's OWN `topk_ids`** (counting
   sort over local routing, in-graph, no host sync): `moe_runner.py:662-697`
   (`_flat = topk_ids.reshape(-1)`, `gram_grouped_accum`); allocator
   `calibration_input_cov.py:186-207`.
4. **Marginals do NOT substitute for C** — retired Path 2 (A→C slot) produced
   `W·A·B^{-1}·L_B`, not `W`: `aa_svd_factor.py:374-378`; `covariance_collection.py:59-67`.
   The only "drop C" lever is Path 3 (`M = W·L_B`, ~0.2 PPL): `aa_svd_factor.py:67`, `:370-372`;
   gated by `orchestrator.py:267`.
5. **The student is a non-native `FactoredExperts` checkpoint** vLLM cannot load:
   `orchestrator.py:452-456`; vLLM assumes `[E, hidden, intermediate]` unquantized
   (`calibration_input_cov.py:143-145`).
