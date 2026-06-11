# Auto-Batch v2 — Stage-3 Covariance Gram Reduction-Pin Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use `- [ ]` checkboxes.

**Goal:** Pin the Stage-3 covariance Gram accumulation to a **fixed per-sequence grouping** so the accumulated covariance is **independent of the cov forward batch size** (`cov_batch_size`). This is the v2 precondition that lets `cov_batch_size` rise (to fill VRAM / kill the ~27 min/layer cov wall) without N-scaling reduction drift. **This plan does the PIN ONLY** — it does NOT wire the auto-batch resolver into cov (that is a separate follow-on, after the pin is proven).

**Architecture / fidelity (read carefully):**
- The cov Gram `cov = flatᵀ@flat; cur.add_(cov)` (`activation_hooks.py:1021-1029`) accumulates **per `update` call**, keyed by `(layer, expert, matrix)`. At `cov_batch_size=1` each `update` receives exactly **one source sequence's** expert-tokens, and the per-key running sum accumulates in **sequence order**. Raising the batch today merges B sequences' tokens into one `update` → different matmul/accumulation grouping → fp-reassociation → the Gram changes (this is why `cov_batch_size` defaults to 1 "for the byte-identical golden").
- **The pin:** at any forward batch B, split each expert's captured rows by source sequence (`seq_id = token_idx // seq_len`) and call `update` **once per sequence, in ascending `seq_id` order**, with an **order-preserving boolean select** (so each per-sequence matmul has the identical operands+row-order it would have at bs=1). Because `add_` is **per-key (per-expert)** and different experts are independent keys, the *only* order that matters is the per-expert one, which this reproduces as sequence-ascending = the bs=1 grouping. → **reduction-order drift = 0.**
- **What is and isn't byte-identical:** the **default path (bs=1 / pin is a no-op)** stays **byte-identical** — the cov golden does NOT move. A future **enabled big-batch** run is **allclose** (not bitwise) to bs=1, because the forward *activations* are batch-shape-dependent on GPU (~1e-6, the unavoidable v1 reality) — but the pin removes the *N-scaling reduction* component, leaving only that bounded, **N-independent** forward drift (quality-neutral). The pin's correctness (grouping-independence) is provable **byte-identically and forward-free** by feeding identical synthetic activations grouped two ways.

**Tech Stack:** PyTorch, pytest. Code root `max_quality/`; all commands from there. CPU-only (no GPU needed for this plan).

**Spec:** `docs/superpowers/specs/2026-06-11-per-plugin-vram-aware-auto-batch-sizing-design.md` §5 v2 (rev5). Builds on v1 (`utils/auto_batch.py`, merged `5e9ffb5`).

---

## File Structure

- **Modify** `src/moe_compress/utils/activation_hooks.py` — add ONLY the accumulator helper `InputCovarianceAccumulator.update_grouped(layer, expert, matrix, x, seq_ids)` (per-sequence-ascending split). **Do NOT thread `seq_len` into the hook ctx** — the experts pre-hook receives `hidden_states` already flattened to `[T, d]` and `top_k_index` as `[T, top_k]`; `seq_len` is not visible there. (Review H1.)
- **Modify** `src/moe_compress/stage3/plugins/covariance_collection.py` — the cov callbacks are closures in the batch loop; bind `_seq_len = batch.shape[1]` as a **free variable** there exactly like the existing `_teacher_T = batch.shape[0]*batch.shape[1]` (`:730`, read by the closures `:497-499`). In `input_cb`/`intermediate_cb`/down callbacks, when the captured rows span >1 distinct sequence, route the B-accumulation through the per-sequence split; the cross-cov **C** split is done **on the pre-matmul operands inside `input_cb`** (NOT via `update_cross` — it receives an already-formed product). bs=1 / single-sequence → the existing single `update`/`update_cross` calls, byte-for-byte unchanged.
- **Create** `tests/test_cov_reduction_pin.py` — forward-free CPU unit tests: grouping-independence, default-path no-op, the **padded factored-down_proj** case, and the **cross-cov operand split**.
- **Goldens** `tests/golden/stage3*` — NOT TOUCHED. The pin is a no-op at `cov_batch_size=1` (the golden's setting) → byte-identical, no re-bless. The byte-identical `rank_map.json` snapshot is the **binding guard for the C path** (the separate `test_stage3_cross_cov.py` is only allclose 1e-5 — weaker; do not rely on it).

---

## Conventions
- Logger `log = logging.getLogger(__name__)`.
- No GPU in tests. Synthetic activations (CPU fp32) — the pin is a pure-reduction property, no forward needed.
- Commit per task; trailer `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

---

## Task 1: Pinned-grouping accumulator API (forward-free, the core property)

**Files:** Modify `src/moe_compress/utils/activation_hooks.py`; Test `tests/test_cov_reduction_pin.py`.

The pin's correctness is a property of the accumulator: **feeding rows grouped per-sequence (ascending) yields the same running Gram, byte-for-byte, as feeding them as a single block would at bs=1 — and is independent of how a big batch merged them.** Add `update_grouped(layer, expert, matrix, x, seq_ids)`: split `x`'s rows by `seq_ids` (a 1-D int tensor, one per row), iterate `sorted(unique(seq_ids))`, and for each `s` call the existing `update(layer, expert, matrix, x[seq_ids == s])` (boolean select preserves row order). When `seq_ids` is None or has a single unique value, it's exactly one `update` (no behavior change).

- [ ] **Step 1: Failing test** (`tests/test_cov_reduction_pin.py`)

```python
import torch, pytest
from moe_compress.utils.activation_hooks import InputCovarianceAccumulator

def _gram(acc, key):  # drain the pending GPU/CPU sum for a key
    return acc._pending[key].clone()

def test_grouped_equals_sequential_bytewise():
    torch.manual_seed(0)
    d = 8
    # 3 sequences of differing token counts routed to expert e
    rows = [torch.randn(n, d, dtype=torch.float32) for n in (5, 3, 4)]
    seq_ids = torch.cat([torch.full((n,), s) for s, n in enumerate((5, 3, 4))])
    x = torch.cat(rows, 0)
    key = (0, 0, "gate_proj")

    # reference: feed each sequence as its own update, in ascending seq order
    ref = InputCovarianceAccumulator(); ref.set_storage_dtype(torch.float32)
    for r in rows:
        ref.update(0, 0, "gate_proj", r)

    # pinned grouped: one call with a merged block + seq_ids
    got = InputCovarianceAccumulator(); got.set_storage_dtype(torch.float32)
    got.update_grouped(0, 0, "gate_proj", x, seq_ids)

    assert torch.equal(_gram(got, key), _gram(ref, key))   # BYTE-identical

def test_grouped_is_order_invariant_in_input_block_but_pins_seq_grouping():
    # shuffling the row order WITHIN the merged block (keeping seq_ids aligned)
    # must NOT change the result (each per-seq matmul gets the same row set in
    # the same per-seq relative order via boolean select).
    torch.manual_seed(1); d = 6
    x = torch.randn(12, d, dtype=torch.float32)
    seq_ids = torch.tensor([0,0,1,2,2,2,0,1,1,2,0,1])
    a = InputCovarianceAccumulator(); a.set_storage_dtype(torch.float32)
    a.update_grouped(0,0,"gate_proj", x, seq_ids)
    # a stable-by-seq reference: gather rows per ascending seq in original order
    ref = InputCovarianceAccumulator(); ref.set_storage_dtype(torch.float32)
    for s in sorted(set(seq_ids.tolist())):
        ref.update(0,0,"gate_proj", x[seq_ids == s])
    assert torch.equal(a._pending[(0,0,"gate_proj")], ref._pending[(0,0,"gate_proj")])

def test_update_grouped_single_sequence_equals_plain_update():
    torch.manual_seed(2); d = 4
    x = torch.randn(7, d, dtype=torch.float32)
    seq_ids = torch.zeros(7, dtype=torch.long)
    g = InputCovarianceAccumulator(); g.set_storage_dtype(torch.float32)
    g.update_grouped(0,0,"down_proj", x, seq_ids)
    p = InputCovarianceAccumulator(); p.set_storage_dtype(torch.float32)
    p.update(0,0,"down_proj", x)
    assert torch.equal(g._pending[(0,0,"down_proj")], p._pending[(0,0,"down_proj")])
```

- [ ] **Step 2: Run** `python3 -m pytest tests/test_cov_reduction_pin.py -q` → FAIL (`update_grouped` undefined).
- [ ] **Step 3: Implement** `update_grouped` in `InputCovarianceAccumulator` (read the class first; reuse `update`):

```python
def update_grouped(self, layer_idx, expert_idx, matrix_name, x, seq_ids):
    """Pinned per-sequence Gram accumulation: split x's rows by seq_ids and
    accumulate one update per source sequence in ASCENDING seq order. This makes
    the running Gram independent of the forward batch's sequence-merging, while
    each per-sequence matmul gets the identical row set (boolean select preserves
    order) the bs=1 path would. seq_ids None / single distinct value => one plain
    update (byte-identical to today's bs=1 path). update_grouped OWNS the
    single-vs-split decision — callers do not pre-guard on token count."""
    if seq_ids is None:
        self.update(layer_idx, expert_idx, matrix_name, x); return
    uniq = torch.unique(seq_ids, sorted=True)         # ascending order is load-bearing
    if uniq.numel() <= 1:
        self.update(layer_idx, expert_idx, matrix_name, x); return
    for s in uniq.tolist():
        self.update(layer_idx, expert_idx, matrix_name, x[seq_ids == s])
```

- [ ] **Step 4: Run** → PASS. **Step 5: Commit** `feat(cov-pin): InputCovarianceAccumulator.update_grouped (per-sequence pinned Gram)`.

CRITICAL: pass `sorted=True` to `torch.unique` EXPLICITLY (not a comment-only assumption) — the ascending per-key accumulation order is load-bearing and must not regress on a torch upgrade. `update_grouped` is the **single source of truth** for split-vs-single; callers (Task 2) must NOT add a `tensor.shape[0]`-based pre-guard (a single sequence can legitimately have up to `seq_len·top_k` rows — Review Med-1).

---

## Task 2: Route the cov callbacks through the per-sequence split (closure `seq_len`)

**Files:** Modify `covariance_collection.py` (callback routing only — NO activation_hooks ctx change); extend `tests/test_cov_reduction_pin.py`.

- [ ] **Step 1:** Read the cov callbacks + their enclosing batch loop in `covariance_collection.py` (`input_cb`/`intermediate_cb` ~`:559-634`, loop ~`:717-741`). Confirm the callbacks are closures that already read the free var `_teacher_T` (`:730`, `:497-499`). Bind a parallel free var **`_seq_len = batch.shape[1]`** in the same loop scope so the callbacks read it per batch. (`batch` is `[B, seq_len]`; `iter_batches` slices rows only → uniform `seq_len`.) `_seq_len` is the ONLY new wiring; do NOT touch `activation_hooks.py`'s hooks (Review H1 — `seq_len` is not visible at the flattened-input hook).

- [ ] **Step 2: B-path** (gate_proj/up via `input_cb`; down_proj via the down callback). Replace the bare `B_acc.update(li, e, name, tensor)`. `tok = ctx["token_idx"]` is the flat `[T]` index. **Factored down_proj is PADDED** (`inter_padded[i]` is `[max_tokens, d_int]` while `len(tok) ≤ max_tokens`) → split on the **unpadded prefix** `tensor[: tok.shape[0]]` (pad rows are zero → contribute nothing to the Gram → dropping them is byte-safe; VERIFY zero in the impl):
```python
tok = ctx.get("token_idx")
if _seq_len and tok is not None:
    rows = tensor[: tok.shape[0]]                 # unpadded prefix (down_proj is padded)
    B_acc.update_grouped(li, e, name, rows, tok // _seq_len)   # owns single-vs-split
else:
    B_acc.update(li, e, name, tensor)             # no seq_len context: UNCHANGED
```
At bs=1 `tok // _seq_len` is all-equal → `update_grouped` falls through to one `update` → byte-identical. Do NOT add a `tensor.shape[0]`-based pre-guard (Review Med-1: one sequence can have up to `seq_len·top_k` rows).

- [ ] **Step 3: C-path (cross-cov) — split the OPERANDS, on the KEPT rows.** The cross term is formed in `input_cb` (~`:605-624`) as `cross = X_pre.T @ X_post` then `C_acc.update_cross(li, e, name, cross, n_tokens)`. `update_cross` receives the already-formed matrix and CANNOT be split (Review H3). **CRITICAL (Review H3-followup):** `X_pre`/`X_post` are **`keep`-filtered** — `sel_idx = tok[keep]` (`:610/:613`), `X_post = det_post[keep]` (`:616`) — so they have `keep.sum()` rows, NOT `len(tok)`. The seq-id source MUST therefore be **`sel_idx // _seq_len`** (the kept-row identities, 1:1 with the `X_pre`/`X_post` rows), NOT `tok // _seq_len`. Carry the real token count (`n_tokens = int(keep.sum())`) in the `else` branch and `int(m.sum())` in the split branch. Read the real var names at `:605-624` and adapt:
```python
sids = sel_idx // _seq_len                                   # kept-row seq ids (aligns with X_pre/X_post)
if _seq_len and torch.unique(sids).numel() > 1:
    for s in torch.unique(sids, sorted=True).tolist():
        m = sids == s
        C_acc.update_cross(li, e, name, X_pre[m].T @ X_post[m], int(m.sum()))
else:
    C_acc.update_cross(li, e, name, X_pre.T @ X_post, n_tokens)   # bs=1: UNCHANGED (n_tokens=int(keep.sum()))
```
Invariant: split the **kept** pre-matmul rows by sequence, ascending seq; byte-identical at bs=1 (one sequence → one call).

- [ ] **Step 4: Failing tests** (extend `tests/test_cov_reduction_pin.py`, forward-free):
  - **Factored padded down_proj:** `tensor` `[max_tokens, d]`, `tok` length `< max_tokens`, trailing pad rows zero → assert routed split == `update_grouped(tensor[:len(tok)], seq_ids)`, and that the zero pad rows make drop-vs-keep byte-equal.
  - **C operand split (incl. a KEPT-row drop):** synthetic `X_pre`/`X_post` over ≥2 sequences built from a `tok` + a `keep` mask that **drops at least one row** (so `sel_idx = tok[keep]` has fewer rows than `tok`); derive `sids = sel_idx // _seq_len`; assert the per-sequence-operand cross accumulation == feeding each seq's `X_pre[m].T@X_post[m]` to `update_cross` ascending, AND that deriving sids from the full `tok` (the buggy way) would mis-length — this pins the kept-row invariant forward-free.
  - **No-`_seq_len` / single-seq:** routes through plain `update`/`update_cross`, byte-identical.

- [ ] **Step 5: Implement** Steps 1–3. **Step 6: Run** the new tests → PASS.

- [ ] **Step 7: GOLDEN GUARDRAIL** — `python3 -m pytest tests/test_stage3_golden_snapshot.py tests/test_smoke_stage3.py -q` MUST pass UNCHANGED (no `MOE_REGEN_GOLDEN`). Pin is a no-op at `cov_batch_size=1` → cov Gram byte-identical; the byte-identical `rank_map.json` snapshot is also the binding guard for the C path (the allclose-1e-5 `test_stage3_cross_cov.py` is weaker — don't rely on it). If it changes → STOP, report. Also run `tests/test_stage2_cov_manifest.py tests/test_input_cov_offload_streaming.py tests/test_multigpu_stage3.py` (cross-path / shared-hook / multi-GPU — no regression).

- [ ] **Step 8: Commit** `feat(cov-pin): route cov B+C capture through per-sequence pinned grouping (default-off no-op)`.

## Task 3: (docs) Note the pin + the cov_batch_size implication

**Files:** Modify `covariance_collection.py` `_resolve_cov_batch_size` docstring (`:351-401`).

- [ ] Update the docstring: with the per-sequence pin, raising `cov_batch_size` no longer changes the *reduction grouping* (only the unavoidable forward-activation drift remains), so a bigger cov batch is now quality-neutral (allclose, N-independent) rather than golden-breaking. Note the auto-batch wiring is a SEPARATE follow-on. No test (docs). Run the unit + golden suite to confirm green. Commit `docs(cov-pin): cov_batch_size is now reduction-grouping-invariant`.

---

## Out of scope (do NOT implement here)
- Wiring `resolve_batch`/`run_with_oom_backoff` into the cov capture, co-resolving with `_resolve_cov_window` G, classifying cov eligible — the NEXT v2 plan.
- The DP `min(candidate)` agreement; the live GPU speedup validation.
- ablation_filter / block_refine pins.
- Re-blessing ANY golden.

## After this plan
Standard **plan/review loop** (all 5 categories → all-none) BEFORE execution, then **implementation/review loop** (same rules) during execution. GPU validation of the live speedup is deferred to a real cov run (H200), like the existing multi-GPU Stage-3 validation.
