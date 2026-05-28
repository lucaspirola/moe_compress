# PLAN — MoE-Pruner Option (a): bubble teacher LM-head vocab logits into `expert_distill`

**Headline feasibility**: **BLOCKED — needs user decision before any code lands.**

**Branch family (canonical)**: `fix/expert-distill-paper-ce-via-pathb`
(the prior direct-implementer branch — see commit `d75549a` —
already raised the same architectural blocker and halted with a doc.
This plan supersedes that ad-hoc halt with a structured proposal.)

**Status**: PLAN ONLY. No source files touched. Pending user decision
on the Option (a) vs Option (2) vs Option (3) split surfaced in §10.

---

## 0. TL;DR

User wants Option (a): replace the per-layer **feature-level KL**
adaptation in `expert_distill._feature_kl_ce` (Lift 1, commit
`d041169`) with a true **vocab-level CE** that matches MoE-Pruner
paper Eq. 10's `L_CE` verbatim, by piping the Path-B teacher
`_stage5_teacher_logits.pt` cache into Stage 2.

After reading the code carefully, **Option (a) has an unbudgeted
student-side cost** that the original framing missed:

- Path-B gives us the teacher vocab logits exactly (already
  computed, ~30 GB sidecar, manifested, validated). That half is
  cheap to plumb.
- But `L_CE` is a categorical CE over the vocabulary axis — it
  needs BOTH operands in vocab-coordinate space. The student in
  `_distill_merged_group` is a **single merged-expert SwiGLU
  triplet** that returns `(T, hidden)`. Projecting that hidden
  output to `(T, vocab)` requires forwarding through every
  remaining downstream layer + final RMSNorm + LM head on every
  gradient step.
- That forward inflates per-step cost by 3–4 orders of magnitude
  (rough cost analysis in §2.B), which converts this plugin from
  "local per-merge-group refinement" into "end-to-end fine-tune
  harness". The whole architectural reason this plugin exists is
  to avoid that.

The decision the user actually needs to make is **NOT "which sidecar
schema do we add"** — it is **"where does L_CE live"**. There are
four real options, enumerated in §10. **This plan documents Option
(a) as the user requested AND surfaces the architectural alternative
(separate fine-tune stage) that the audit's OPEN-QUESTION § 1 already
flagged.** Per CLAUDE.md "RAISE, don't substitute" and
`feedback_raise_dont_substitute.md`, I will not silently pivot to a
hacky variant; this plan stops at the user-decision boundary.

If the user reads §2 and confirms "yes, take the cost hit, I want the
full downstream forward inside `_distill_merged_group`" — the
implementer steps in §3–§9 are the recipe. If the user reads §2 and
chooses Option (2) (separate fine-tune stage) or Option (3)
(LayerNorm-aware KL) instead — a new plan should be written; this
one is moot.

---

## 1. Strategy summary (what Option (a) would mean concretely)

Reuse Path-B teacher cache `_stage5_teacher_logits.pt` (already
shipped — see `max_quality/hf_jobs/precompute_teacher_logits.py` and
`max_quality/src/moe_compress/router_kd/plugins/teacher.py:170-469`)
as the **teacher operand** of paper Eq. 10's `L_CE`. To make the
**student operand** comparable, two new pieces of plumbing:

1. **Token-position metadata sidecar** at Stage 2 profile time
   (Pattern O, atomic-write + manifest-last). The Stage 2
   `_LayerInputAccumulator` already reservoir-samples 8192
   per-layer hidden inputs (see `stage2/profiling.py:34-167`); we
   add a parallel `int64[T]` of flat token-position indices
   `(seq_idx · L + pos)` per surviving token so each reservoir
   sample can be matched back to its row in
   `_stage5_teacher_logits.pt[token_idx, :]`. Without this index,
   the teacher logits are useless to Stage 2: we have hidden
   inputs but no anchor to vocab logits.

2. **Student-to-vocab projection inside `_distill_merged_group`**.
   For each gradient step, the student SwiGLU output `(T, hidden)`
   must be (i) reassembled into the layer-N MoE block sum,
   (ii) added to the residual stream entering layer N,
   (iii) forwarded through layers N..end (~half the decoder on
   Qwen3.6-35B-A3B at the worst-case mid-layer position),
   (iv) RMSNorm, (v) LM head → `(T, vocab)`. Only then is a true
   vocab-level CE between Path-B teacher logits and this projected
   student logits paper-faithful.

The new ctx slots + sidecar contract carry the Pattern O / Pattern B
discipline; the cost concern in §2 has nothing to do with the
sidecar contract — it is a per-step compute cost.

---

## 2. Architectural risk

### 2.A — The student-to-vocab projection is the BLOCKER

The student in `_distill_merged_group` (file:
`stage2/plugins/expert_distill.py:378-685`) is built as:

```
p_gate = nn.Parameter(init_gate)       # (intermediate, hidden)
p_up   = nn.Parameter(init_up)         # (intermediate, hidden)
p_down = nn.Parameter(init_down)       # (hidden, intermediate)
optim  = AdamW([p_gate, p_up, p_down], ...)
out    = _swiglu_forward(p_gate, p_up, p_down, x_all)   # (T, hidden)
```

That `(T, hidden)` tensor lives in **layer-N hidden-state geometry**,
NOT vocab geometry. It is one **single** expert's contribution to
the MoE-block sum at layer N. To get from there to vocab logits, the
math requires:

1. Run the layer-N router to obtain the per-token top-k gating
   weights, including the centroid's weight (call it `g_c(x)`).
   But the merged centroid only contributes when it is in
   `top_k(σ_orig(x))` — for tokens where the centroid is NOT
   routed, its contribution to the MoE block is zero, and the
   downstream forward depends only on the OTHER top-k experts'
   forwards — which are NOT in scope for this gradient step
   (they are not the optimized parameter set).
2. Compute the other top-k - 1 experts' SwiGLU outputs (with their
   CURRENT post-merge weights — they may themselves be merged
   centroids from earlier layers' or earlier groups' merges).
3. Sum the per-expert contributions weighted by the (post-resize,
   post-stage-2.5) router.
4. Add to the residual stream entering layer N.
5. Forward through ALL remaining decoder layers (~`94 - N`
   layers, where N is the current layer index). Each remaining
   layer is itself an MoE layer with its own router + experts.
6. Apply final RMSNorm.
7. Project through `lm_head: (hidden) → (vocab)`.

ONLY THEN do we have `student_vocab_logits: (T, vocab)` and can
compare against `teacher_vocab_logits: (T, vocab)` from Path-B.

### 2.B — Cost projection (per-step + Stage-2-total)

Production config: `configs/qwen36_35b_a3b_30pct.yaml`.

Per-step cost of one downstream forward at layer N:
- `(94 - N + 1)` decoder layers × `attn + MoE` per layer
  × 8192 tokens × hidden=2048 × intermediate=11008
- At the mid-stack worst case (N ≈ 47), ~47 layers of full
  decoder forward per step.
- Empirically: layer-forward on 8192 tokens at this hidden/intermediate
  is **~50-100 ms** on a single H100 in BF16 — call it 75 ms/layer.
  47 layers × 75 ms = **~3.5 seconds per gradient step**.

Current step cost (the local distill the plugin is designed for):
- 3 GEMMs `(T, hidden) → (T, intermediate) → (T, intermediate) → (T, hidden)`
  + SwiGLU non-linearity + MSE + KL → **~5-10 ms** per step (no
  empirical pin from this worktree, but the order-of-magnitude is
  set by 8192-token SwiGLU on a single H100, which is sub-10 ms).

Slowdown factor: **3.5 s / 7 ms ≈ 500×** at the per-step level.
This is a 3-orders-of-magnitude regression, not a 30% one.

Stage-2-total cost projection (94 layers × ~32 groups/layer ×
`expert_distill_steps=500`):
- Current (local distill): ~1.5 M steps × 7 ms ≈ ~3 hours of
  Stage-2 distillation wall, on top of the profile pass.
- Option (a) literal: ~1.5 M steps × 3.5 s ≈ ~60 days on a
  single H100. (Multi-GPU parallelization helps but doesn't bring
  it under a day.)

Even with `expert_distill_steps` cut to 50 and groups halved, the
projection is **multi-day Stage 2** — completely outside the
project's wall-clock budget.

### 2.C — VRAM topology change

Today `_distill_merged_group` needs:
- `p_gate, p_up, p_down` (one expert × 3 matrices) — fp32 ≈ ~300 MB
- `x_all` (8192 × hidden) — ~32 MB bf16
- gradient + AdamW state on the centroid only — ~1.2 GB fp32
- `_LayerInputAccumulator` per-layer reservoirs (only the current
  layer's is hot)

Option (a) literal needs:
- All the above
- The **full pruned student model** resident on the same device
  (94 MoE layers × ~64 experts × 3 matrices) — ~30 GB BF16
- `output_router_logits=False` is fine but each remaining layer
  needs to be forward-able, so the model can't be sharded across
  CPU — it has to be on the GPU
- The Path-B teacher cache (~30 GB on CPU, mmap'd) — already
  budgeted under D-teacher-cache
- A per-step gradient that **flows back through 47 layers** — that
  is an end-to-end backward pass through half the model on a
  trainable subset of one expert, which means activation
  checkpointing on the frozen layers (or 8x VRAM blowup from
  storing intermediate activations).

Stage 2 currently runs on **a single H100/RTX 6000 Pro** (see
`project_vast_rtx6000_released.md` memory). Option (a) literal would
push Stage 2 onto the same multi-GPU regime as Stage 5 router-KD
(80+ GB VRAM tight even for student-only), and the activation
checkpoint plumbing does not exist in Stage 2 today.

### 2.D — Verdict on architectural risk

**Architectural risk = BLOCKER (catastrophic-cost regime).**

Option (a) is **not "a larger architectural change but tractable"**
as the OPEN-QUESTION block in `expert_distill.py:118-168` originally
described it. It is "rewrite Stage 2 into a fine-tune harness". The
OPEN-QUESTION block underspecifies the student-side cost; this plan's
§2.A–§2.C is the corrected analysis.

Per CLAUDE.md Section 0 (this prompt's instruction), surfacing as
Open Question before designing around it. **Do not implement Option
(a) literal without explicit user confirmation that the multi-day
Stage 2 regression is acceptable.**

---

## 3. Files to create / modify (IF user confirms Option (a) literal)

### 3.A — New file: token-position metadata writer

`max_quality/src/moe_compress/utils/token_position_index.py` (NEW)
- Helpers to build `int64[T]` flat token-position indices per
  reservoir-surviving token, parallel to the reservoir buffer.
- Uses `utils/atomic_io.atomic_torch_save` + `write_manifest_last`
  per Pattern O.

### 3.B — Modified: reservoir to track position metadata

`max_quality/src/moe_compress/stage2/profiling.py`
- Lines 34–166 (`_LayerInputAccumulator`):
  - Add `seed_field` parameter — global `(seq_idx · L + pos)`
    int64 for every token entering `add()`.
  - Parallel reservoir buffer `pos_buffer: torch.Tensor | None`
    that follows the SAME survive/replace decisions as `buffer`
    (Phase A prefix-take, Phase B fill, Phase C
    `index_copy_(0, target_slots, ...)`).
  - New `get_positions(self) -> torch.Tensor | None` accessor
    parallel to `get(self)`.
- Lines 169+ (`_profile_layer`):
  - Plumb `(seq_idx, pos)` into the forward-pre hook that feeds
    `layer_input_acc.add(hidden)`. The hook currently only sees
    the hidden tensor; the per-batch driver knows
    `_batch_offset` (line 211) which can be combined with
    `torch.arange(B*L)` to produce the flat per-token index.

### 3.C — Modified: profile-pass writer dumps the position sidecar

`max_quality/src/moe_compress/stage2/calibration/stage2_profile_writer.py`
- `dump_stage2_profile` (around line 301-467): bump
  `SCHEMA_VERSIONS["stage2_profile"]` to schema 4, add
  `layer_input_positions: dict[layer_idx, int64[T]]` to the
  payload.
- Resume validation in `load_stage2_profile_checkpoint`
  (lines 511-614): include `layer_input_positions` cross-check.
- The shape contract: `positions.shape == (buffer.shape[0],)`
  AND every value `< num_calibration_samples *
  max_sequence_length`.

### 3.D — New file or modified: teacher-cache loader in Stage 2

If we follow Stage 5's pattern, **either** lift a new plugin
`max_quality/src/moe_compress/stage2/plugins/teacher_logits_cache.py`
that mirrors `router_kd/plugins/teacher.py:TeacherCachePlugin`'s
load + validate + slice contract, **or** add a Stage-2-local helper
that reuses the same `read_and_validate_manifest` + payload schema.

Recommendation: **new Stage 2 plugin file**, name
`TeacherLogitsCacheStage2Plugin`. Reasons:
- The Path-B cache file format is already shared (schema 1,
  format_version 1, manifest-last). Stage 5's reader at
  `router_kd/plugins/teacher.py:281-393` is the canonical loader;
  re-use its validation block verbatim (manifest check, vocab
  guard, num_samples guard, sequence_length guard).
- Stage 2 needs a DIFFERENT slice arithmetic from Stage 5 —
  Stage 5 walks batches sequentially in calibration order;
  Stage 2 needs random-access by `(seq_idx, pos)` flat index. A
  separate plugin with its own `provide_teacher_vocab_logits_by_position`
  hook is cleaner than overloading Stage 5's hook.
- Pattern O contract is **identical** to the existing Stage 5
  reader: that's the explicit goal of sidecar-sharing — Stage 2
  consumes the SAME `.pt` + `.MANIFEST.json` Stage 5 produces.

### 3.E — Modified: `expert_distill.py` — vocab CE term

`max_quality/src/moe_compress/stage2/plugins/expert_distill.py`
- Lines 90–168 (OPEN-QUESTION block): rewrite to point to this
  plan + Option (a) realization or Option (2) reroute, depending
  on user decision.
- Lines 320–346 (`_feature_kl_ce`): keep as legacy back-compat,
  gated by the new `expert_distill_use_paper_ce_term=False`
  fallback path.
- Lines 378–685 (`_distill_merged_group`): add new path
  `_distill_merged_group_with_paper_ce`:
  - Loads the **full student model** from ctx (`ctx.get("model")`
    — already published at `stage2/orchestrator.py:1155`).
  - For each gradient step:
    1. Compute `student_hidden = _swiglu_forward(p_gate, p_up,
       p_down, x_all)` — `(T, hidden)`.
    2. Pull `seq_idx, pos` from `layer_input_positions[layer_idx]`
       (length-T int64).
    3. Pull `teacher_vocab_logits = teacher_cache[positions]` —
       `(T, vocab)` bf16 → fp32.
    4. Project `student_hidden → student_vocab_logits` via
       `_project_through_downstream_stack(model, layer_idx,
       student_hidden, positions, residual_stream)`. This is
       the new helper — see §3.F.
    5. `loss_ce = F.cross_entropy(student_vocab_logits,
       softmax(teacher_vocab_logits, temperature=1.0))`
       or, more faithfully, `F.kl_div(F.log_softmax(student),
       F.softmax(teacher), reduction='batchmean')` matching
       Router-KD's vocab KL (which IS paper Eq. 10's L_CE up
       to the additive entropy constant — `teacher_entropy`).
    6. `loss = loss_ce + ce_lambda * mse_loss`.
- Lines 749–751 (knobs):
  - Rename `expert_distill_use_ce_term` → keep as legacy alias
    flag with deprecation log; new flag
    `expert_distill_use_paper_ce_term` (default `True`
    post-implementation, OFF iff cache is missing) selects
    between the new vocab-CE path and the feature-KL fallback.
  - New knob `expert_distill_teacher_logits_cache` (default
    `None`, inherited from `stage5_router_kd.teacher_logits_cache`
    if a single source of truth is desired).
  - New knob `expert_distill_projection_grad_checkpoint` (default
    `True` — without this the activation memory of the downstream
    forward blows up on the per-step backward).

### 3.F — New file: downstream-projection helper

`max_quality/src/moe_compress/stage2/downstream_projection.py` (NEW)
- `_project_through_downstream_stack(model, layer_idx,
  student_hidden, positions, residual_cache) -> torch.Tensor`
- Builds the layer-N MoE block sum (centroid + other top-k
  experts, with the OTHER experts frozen).
- Adds to the residual stream entering layer N — this requires
  capturing `residual_in_layer_N` per token at profile time
  (NEW state — not currently in `_LayerInputAccumulator`; this
  is a SECOND new sidecar entry, parallel to
  `layer_input_positions` and `buffer`).
- Forwards through layers `N+1..end` using
  `torch.utils.checkpoint.checkpoint_sequential` on the frozen
  decoder layers (gradient ONLY needs to flow through the
  centroid's `p_gate, p_up, p_down`).
- Returns `(T, vocab)` student logits.

### 3.G — Modified: orchestrator wires the cache + downstream model

`max_quality/src/moe_compress/stage2/orchestrator.py`
- After line 1155 (`run_ctx.set("model", model)`):
  - Register `TeacherLogitsCacheStage2Plugin` if
    `expert_distill_use_paper_ce_term=True` and cache path
    is configured.
  - Validate cache vs Stage 2 calibration shape: cache covers
    `stage5_router_kd.max_calibration_samples` × `max_sequence_length`,
    but Stage 2 may run on `stage2.num_sequences` (= 4000 today
    per `qwen36_35b_a3b_30pct.yaml`; cf. the secondary blocker
    in `tasks/todo_expert_distill_paper_ce_blocker.md` lines
    160-187). If `stage2.num_sequences > stage5.num_samples`,
    HARD-RAISE — refuse to enable paper-CE rather than silently
    truncating.

### 3.H — Modified: config defaults + docs

`max_quality/configs/qwen36_35b_a3b_30pct.yaml` — add example
opt-in stanza in `stage2_reap_ream:`:
```yaml
expert_distill_use_paper_ce_term: false  # set true to enable Option (a)
expert_distill_teacher_logits_cache: null  # path inherited from stage5 if null
expert_distill_projection_grad_checkpoint: true
```

`max_quality/configs/*` — any other configs that set
`expert_distill_use_ce_term` get the deprecation alias treated
identically to today's behavior.

---

## 4. New ctx slots

| Slot name | Producer | Consumer | Type | Purpose |
|-----------|----------|----------|------|---------|
| `teacher_logits_cache_stage2` | `TeacherLogitsCacheStage2Plugin.load_teacher_cache` (one-time setup, mirrors Stage 5's `load_teacher_cache`) | `ExpertDistillPlugin.merge` reads it inside `_distill_merged_group_with_paper_ce` | `dict[str, Any] \| None` — the mmap'd `.pt` payload (same shape as Stage 5's `teacher_logits_cache` slot — see `router_kd/plugins/teacher.py:404`) | Vocab logits operand of paper Eq. 10's `L_CE` |
| `layer_input_positions` | `_LayerInputAccumulator` (Stage 2 profile pass) → dumped by `stage2_profile_writer.dump_stage2_profile` → reloaded into ctx at run-time | `ExpertDistillPlugin.merge` reads `layer_input_positions[layer_idx]` (length-T int64) | `dict[int, torch.Tensor]` (layer_idx → int64[T]) | Per-reservoir-sample anchor into the Path-B cache's flat row index |
| `layer_input_residuals` | `_LayerInputAccumulator` (Stage 2 profile pass) | `ExpertDistillPlugin.merge` reads `layer_input_residuals[layer_idx]` | `dict[int, torch.Tensor]` (layer_idx → bf16[T, hidden]) | Residual stream entering layer N — needed to reassemble downstream input from the trained centroid's contribution |

Writer site for slots 2–3: a NEW forward-pre hook on the residual
stream alongside the existing `_LayerInputAccumulator.add()` call
inside `_profile_layer` (`stage2/profiling.py:169+`).

Reader site for all three: inside `_distill_merged_group_with_paper_ce`
(replacement for `_distill_merged_group` when paper-CE is enabled).

---

## 5. New sidecar contract (Pattern O — extends the existing
`_stage5_teacher_logits.pt` contract OR adds a NEW per-stage
sidecar)

### 5.A — Reuse `_stage5_teacher_logits.pt` (RECOMMENDED if Option (a) lands)

The cache writer (`hf_jobs/precompute_teacher_logits.py`) already
emits Pattern O:
- `_stage5_teacher_logits.pt` — `format_version=1`, atomic write,
  fsync, parent fsync.
- `_stage5_teacher_logits.pt.MANIFEST.json` — written LAST via
  `write_manifest_last` after the `.pt` is durable.

**Stage 2 reuses the EXACT same file**, no new sidecar produced by
Stage 2 (only consumed). Cross-stage sharing is the design goal.

Validation contract on Stage 2's side (identical to Stage 5's
`TeacherCachePlugin.load_teacher_cache:281-393`):
1. `manifest.json` must exist and validate via
   `read_and_validate_manifest(... expected_schema_version=1)`.
2. `format_version == 1` in the payload itself (defense in depth).
3. `cache_payload["logits"].shape[-1] == model.config.vocab_size`
   (defends against tokenizer mismatch).
4. `cache_payload["logits"].shape[0] == num_samples * sequence_length`
   (token-count contract).
5. NEW Stage-2-specific guard:
   `cache.num_samples >= stage2.num_sequences`. If Stage 2 was
   calibrated on more sequences than Stage 5, RAISE — there are
   tokens with no teacher.

### 5.B — NEW per-layer position sidecar (`_stage2_layer_positions.pt`)

This is a SECOND new sidecar Stage 2 writes during the profile
pass. It is NOT cross-stage shared — only Stage 2 consumes it.

Schema:
```python
{
    "format_version": 1,
    "schema": "stage2_layer_input_positions",
    "layer_input_positions": {
        layer_idx: torch.int64[T]   # T <= max_samples=8192
        for layer_idx in moe_layer_indices
    },
    "layer_input_residuals": {
        layer_idx: torch.bfloat16[T, hidden]
        for layer_idx in moe_layer_indices
    },
    "max_samples": 8192,
    "num_calibration_samples": int,
    "max_sequence_length": int,
    "model_hash": str,    # cross-validate vs the model the profile
                          # pass ran against
}
```

Write site: extension of `dump_stage2_profile` in
`stage2_profile_writer.py`, OR a separate `.pt` produced alongside
`stage2_profile.pt` — the second option is preferred so a Stage 2
run without Option (a) doesn't pay the extra-bytes cost.

Pattern O discipline:
- `atomic_torch_save(path, payload)`
- `write_manifest_last(path, manifest_path, schema_version=1,
  extra_meta={...})` — `schema_version=1`, `extra_meta` carries
  `num_layers`, `max_samples`, `hidden`, `total_position_count`
  for forensics.
- Reader: `read_and_validate_manifest(path, manifest_path,
  expected_schema_version=1)` BEFORE the torch.load + mmap.

---

## 6. Config knobs (Pattern C — consumed verbatim, no implicit coupling)

| Knob | Default | Type | Semantics |
|------|---------|------|-----------|
| `expert_distill_use_paper_ce_term` | `True` post-Option(a)-lands (gated to `False` if no cache configured) | `bool` | Master switch for Option (a). When True, the new `_distill_merged_group_with_paper_ce` path runs. When False, the legacy `_distill_merged_group` runs with the existing v1/v2 target and feature-KL CE fallback (unchanged). |
| `expert_distill_use_ce_term` | DEPRECATED — alias for back-compat | `bool` | Logs a deprecation warning on read. If `expert_distill_use_paper_ce_term=False` and this is `True`, the legacy feature-KL CE path activates (today's Lift 1 behavior). Removed in v3. |
| `expert_distill_ce_lambda` | `1.0` | `float` | Paper Eq. 10 λ. Tuned per-config. NEW magnitude regime under Option (a) — expect to retune from scratch (vocab CE has DIFFERENT loss magnitude than feature KL). |
| `expert_distill_teacher_logits_cache` | `None` (inherits from `stage5_router_kd.teacher_logits_cache` if set there) | `str \| None` | Path to `_stage5_teacher_logits.pt`. Same artifact Stage 5 consumes — single producer (`hf_jobs/precompute_teacher_logits.py`), two consumers. |
| `expert_distill_projection_grad_checkpoint` | `True` | `bool` | Activation checkpointing on the downstream forward (layers `N+1..end`). Required to fit the per-step backward into VRAM; the only reason this is configurable is to allow benchmark experiments to disable it on H200/B200-class hardware where activation memory is bigger. |
| `expert_distill_position_sidecar` | `None` | `str \| None` | Path to the NEW Stage 2 layer-position sidecar (§5.B). If `None`, Stage 2 builds it during the profile pass and writes it to `artifacts_dir / "_stage2_layer_positions.pt"`. |

Validation site for all six knobs: `ExpertDistillPlugin.__init__`
(line 752+) — raise `ValueError` on bad combinations:
- `use_paper_ce_term=True` + `teacher_logits_cache is None` and
  Stage 5 also has no cache → RAISE.
- `use_paper_ce_term=True` + `use_ce_term=True` → log conflict,
  paper-CE wins, feature-KL is ignored.
- `use_paper_ce_term=True` + cache vocab_size != model vocab_size
  → RAISE (mirrors Stage 5's L380-393 guard).

---

## 7. Test plan

Each test pins a specific contract from this plan. ALL tests must
exist before any source touched (per superpowers
test-driven-development); reviewer must verify the tests fail
before the implementation lands.

### Test 1: `test_layer_input_accumulator_position_parallel.py`

**Contract**: `_LayerInputAccumulator.buffer` and
`_LayerInputAccumulator.pos_buffer` move in lockstep through all
three phases (prefix-take, fill, Vitter reservoir replacement).

Steps:
1. Build accumulator, feed batches with known `(seq_idx, pos)`
   labels.
2. After Phase A (deterministic prefix), assert
   `pos_buffer[:max_samples]` equals the input positions in
   the first batch's first `max_samples` slots.
3. After Phase B (fill), assert positions continue to match.
4. After Phase C (Vitter replacement), assert
   `pos_buffer[k] == input_position[kept_local[k]]` for every
   surviving sample — i.e., for every slot in `target_slots`,
   the position came from the same source token that the hidden
   buffer at that slot came from.
5. Reproducibility: with the same seed and same input sequence,
   `pos_buffer` is bit-identical across runs.

### Test 2: `test_position_sidecar_pattern_o.py`

**Contract**: The new `_stage2_layer_positions.pt` sidecar
follows Pattern O (atomic + manifest-last, schema-versioned,
manifest validates against payload).

Steps:
1. Write a fake sidecar via the production writer path,
   manifest path is the LAST file to appear on disk.
2. Simulate a torn write (delete the manifest, leave the .pt) —
   reader RAISES `ManifestMismatchError`.
3. Simulate a torn write (truncate the .pt by 1 byte after
   manifest is written) — reader RAISES with a size mismatch.
4. Schema-version bump (write with schema=2, read with
   schema=1) — reader RAISES.

### Test 3: `test_teacher_cache_stage2_consumption.py`

**Contract**: The Stage 2 cache reader consumes the SAME
`_stage5_teacher_logits.pt` Stage 5 already validates, with the
SAME `format_version=1` + manifest contract.

Steps:
1. Build a synthetic Path-B cache (small vocab/seq/N for test
   speed) using the same code path as
   `hf_jobs/precompute_teacher_logits.py` (atomic write +
   manifest-last).
2. Stage 5's `TeacherCachePlugin.load_teacher_cache` consumes
   it — green today.
3. Stage 2's `TeacherLogitsCacheStage2Plugin.load_teacher_cache`
   consumes the same file — also green.
4. Cross-validation: `(seq_idx, pos)` lookup in Stage 2 matches
   `(epoch * num_batches + batch_index) * tokens_per_batch +
   token_within_batch` lookup in Stage 5 for the same tokens.
5. Sequence-count mismatch (cache has 3000 samples, Stage 2
   profile has 4000) → Stage 2 RAISES at orchestrator wire-up,
   does NOT silently truncate.

### Test 4: `test_distill_merged_group_paper_ce_loss_signal.py`

**Contract**: When the cache + position sidecar match, the new
paper-CE path produces a NON-zero gradient on the centroid's
trainable parameters, AND the loss decreases over steps on a
canned example.

Steps:
1. Tiny fake model (2 layers, 4 experts, hidden=64, vocab=128)
   + fake teacher logits computed by the SAME model on a known
   token batch.
2. Initialize a merged-centroid distill against the fake
   teacher logits.
3. Assert `loss[0] > loss[10]` (loss decreased over 10 AdamW
   steps).
4. Assert `p_gate.grad.abs().sum() > 0` after first
   `backward()` (gradient flow through the downstream forward
   is functional).
5. Mass-conservation: when `centroid` is set to its frozen
   pre-merge weights AND `top_k > 1` includes the centroid AND
   teacher logits = student logits at step 0, `loss[0] ≈ 0`
   (sanity check that the projection geometry is right).

### Test 5: `test_paper_ce_cost_smoke.py` (cost regression smoke)

**Contract**: A SINGLE per-step paper-CE forward+backward at
production scale (94 layers, 8192 tokens) completes within a
configurable wall-clock budget. This is the **gating** test for
"is Option (a) shippable" — if it fails, the user has direct
empirical confirmation of the §2.B cost projection.

Steps:
1. Skip-unless-GPU mark.
2. Load a real Qwen3.6-35B-A3B-shaped model (or a config-shrunk
   version per CI budget — see existing `tests/conftest.py`
   patterns).
3. Time ONE step of `_distill_merged_group_with_paper_ce` on
   layer N=47 (mid-stack), 8192 tokens.
4. Assert wall-clock < `paper_ce_smoke_max_seconds` (default
   `10.0`, configurable via env var `PAPER_CE_SMOKE_MAX_S` for
   investigation).
5. Print wall-clock + projected Stage-2-total. Test is a SOFT
   fail (xfail) if it exceeds budget — this is observation, not
   gating CI, but documents the cost in test output.

---

## 8. Migration path

**Recommendation: keep `expert_distill_use_paper_ce_term=False`
by default in v2 even after the code lands**, until §7 Test 5
demonstrates the per-step cost is acceptable (which §2.B
projects it will NOT be at production scale).

Migration ladder:
1. **Pre-merge to main**: ship the plumbing (cache reader,
   position sidecar, downstream projection helper) with the
   flag DEFAULT OFF. Existing Lift 1 feature-KL path remains
   the default — no regression to current runs.
2. **First production trial**: opt-in via config knob for
   one `qwen36_35b_a3b_30pct` run. Measure actual Stage 2 wall.
3. **Flip default to True**: only after the trial shows the
   wall is tolerable AND the resulting model quality on Stage 6
   evals is materially better than Lift 1 feature-KL.
4. **Deprecate `expert_distill_use_ce_term`**: only after
   step 3.

Alternative (Option (2) from §10): leave `expert_distill_*`
alone on main, build a NEW Stage 2.6 / Stage 4.5 fine-tune
plugin that puts L_CE at the model level. This plan does NOT
cover that work — a separate plan would be written.

---

## 9. Blockers / OPEN-QUESTIONS

### B1. PRIMARY: student-to-vocab projection cost
**Status**: BLOCKER until user explicitly accepts the cost
regression OR pivots to Option (2)/(3). Documented in §2.
Same blocker as `tasks/todo_expert_distill_paper_ce_blocker.md`
on the prior direct-implementer branch.

### B2. SECONDARY: Path-B coverage window vs Stage 2 calibration
window
**Status**: Solvable but needs a decision. Path-B currently
covers `stage5_router_kd.max_calibration_samples=3000`. Stage 2
calibration runs on `stage2.num_sequences=4000` (cf.
`configs/qwen36_35b_a3b_30pct.yaml`). Three resolutions:
- (i) Restrict Stage 2 to `min(N_stage5, N_stage2)` — coupling
  knob, breaks current behavior for non-Option-(a) runs unless
  config-gated.
- (ii) Extend Path-B precompute to cover all 4000 Stage 2
  sequences — +33% cache size (~40 GB) + +33% precompute time
  (~60 min instead of 45).
- (iii) Run Path-B with `max_calibration_samples = max(N_stage5,
  N_stage2)` permanently — paper-CE coverage AND Stage 5
  unaffected (Stage 5 reads only first N).
  Recommended: (iii).

### B3. SECONDARY: Path-B cache freshness across Stage 2 runs with
different calibration sources
**Status**: Solvable. The Path-B cache is keyed by tokenizer +
`spec_from_config(cal)`. If a Stage 2 calibration switches
sources (e.g., the recent `qwen3-pretrain-mix` change in commit
`9c5abe2`), Path-B must be regenerated. The manifest's
`extra_meta` includes the model repo but NOT the calibration
spec hash today. Recommend adding `calibration_spec_hash` to
`extra_meta` and validating on read.

### B4. SECONDARY: residual-stream sidecar size
**Status**: Solvable, document the cost. `layer_input_residuals`
adds `94 layers × 8192 tokens × 2048 hidden × 2 bytes = ~3 GB`
per Stage 2 run — fits in the existing artifacts budget but is
the biggest single sidecar Stage 2 has produced.

### B5. OPEN: does the user prefer "Option (a) literal" or "Option
(2) separate fine-tune stage"?
**Status**: USER DECISION REQUIRED. This is the meta-question
this plan exists to surface. Without an answer, no code lands.

### B6. OPEN: if Option (a) literal, what is the acceptable
per-step wall regression?
**Status**: USER DECISION REQUIRED. The §7 Test 5 budget is
TBD-by-user. If the user says "10 seconds is fine, I'll run
Stage 2 for a week", the implementation proceeds; if the user
says "100 ms max", it does NOT.

---

## 10. Options for the user (verbatim from the prior direct-implementer halt)

These are the same five options the prior halt enumerated.
Plan §1-§9 above covers Option (a) literal (option 5 here);
the others are summarized for the user's decision-making.

| # | Option | Effort | Cost regression | Paper-fidelity | Recommended? |
|---|--------|--------|-----------------|----------------|--------------|
| (1) | **Accept Lift 1 feature-KL as final.** Keep `expert_distill_use_ce_term=True` on main (commit `d041169`). Document feature-KL as the project's paper-Eq.10 realization with the disclosed `D-expert-distill-ce-term` deviation. | Zero | None | Per-layer adaptation, shift-invariant — already disclosed | **Cheapest.** Matches the audit's "deliberately-scoped local refinement" framing. |
| (2) | **Move L_CE to a separate fine-tune phase** (new Stage 2.6 / Stage 4.5 / Stage 5-extension) that uses the full pruned model + Path-B cache to compute true L_CE at the model level. Per-merge-group distill stays as local MSE refinement. | High (new orchestrator stage) | None at Stage 2; new wall at Stage 2.6 (but ~comparable to Stage 5 which already pays this cost) | **Paper-faithful** | **Audit's preferred path.** The right architectural home. |
| (3) | **Try Option (b) from the OPEN-QUESTION block** — closer to (1) in cost, partial improvement. LayerNorm-aware KL or feature-wise MSE on softmaxes. Closes the constant-shift gap without claiming paper-fidelity. | Low | None | Closer to paper than feature-KL but not faithful | Cheapest semantic upgrade. |
| (4) | **Drop the CE term entirely** (`expert_distill_use_ce_term=False` default). Ship v2 paper-faithful target + MSE-only loss. The audit verdict on surface #7 is `DEVIATION (documented)`. | Zero (revert Lift 1 default) | None | Most honest about the per-layer scope limit | Reasonable if (1) is unsatisfying. |
| (5) | **Force-implement Option (a) literal** with the full downstream forward inside `_distill_merged_group`. This plan §1-§9. | Very high (new plugin, new sidecar, new helper module, ~6-12 weeks engineering) | **3 orders of magnitude per step** | **Paper-faithful** | **NOT RECOMMENDED.** §2.B cost analysis. |

---

## 11. What this plan will NOT silently do

Per `feedback_raise_dont_substitute.md` and CLAUDE.md "RAISE,
don't substitute":

- Quietly pick Option (5) and call it Option (a). (§2 surfaces
  the regression; user must explicitly accept.)
- Quietly pick Option (3) and call it Option (a). (Plan
  enumerates Option (3) as a separate path requiring a separate
  plan.)
- Add the position sidecar in preparation for Option (a)
  without resolving B5 first. (§3.B is gated on B5.)
- Pretend the existing feature-KL is "paper Eq. 10 verbatim"
  by relabeling. (Lift 1's docstring already calls it a
  per-layer adaptation; this plan preserves that honesty.)

---

## 12. Plan path

`tasks/PLAN_MOE_PRUNER_CE_TERM.md`

**Feasibility headline**: **BLOCKED** — Option (a) literal is
architecturally workable but produces a ~500× per-step cost
regression that the user must explicitly accept before any
implementation. The plan's §3-§9 is the recipe if accepted;
§10 lists the four alternatives that the user should choose
between before any code lands.

**Critical blocker**: B1 (student-to-vocab projection) and B5
(user-decision required between Option (a) literal vs Option
(2) separate-fine-tune-stage).
