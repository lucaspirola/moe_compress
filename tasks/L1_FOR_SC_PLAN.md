# L1 for SC — End-to-End Implementation Plan

**Branch**: `feat/calibration-v2` (HEAD `6ff3636`)
**Status**: Spec draft. NOT IMPLEMENTED.
**Context**: L1 was scope-cut in the calibration-v2 campaign (commit `9647362` per
`tasks/calib_v2_writers_todo.md`). The user has since asked for a plan that revives L1
**only as a Stage 2 cost backend for SC**, not as a full REAP+REAM driver refactor.
The scope here is intentionally narrower than the original L1 — see "What L1 means
here" below.

---

## 0. What L1 means here (scope discipline)

The campaign's "L1" was REAP+REAM as **N vLLM passes with `update_weights` between
rounds** — replace the Stage 2 driver loop. That scope tripped 3 architectural
blockers, all valid; this plan does NOT revive that scope.

**This plan's L1** = use the existing patched vLLM session (the one we already
spin up to write calibration sidecars) as a forward-evaluation engine for Stage
2's **`_output_space_cost`** inner loop. The Stage 2 driver stays HF-side. The
"cost backend" becomes vLLM-backed for SC.

This is materially smaller in scope: no full driver refactor, no model porting,
no cudagraph stress at Stage-2 granularity. The 3 blockers from `9647362` still
need to be defeated, but the surface area is one function (the per-pair forward
in `_output_space_cost`), not the whole driver.

**The hard finding from Phase 1 below**: under the current code, the per-pair
cost is **96% Hungarian neuron-permutation solve** and **0.4 ms / 1.6%
SwiGLU forward** (`MOE_COMPRESS_REPORT.md:250-254`). vLLM cannot speed up
Hungarian. So an L1 with the current cost shape would only attack the 1.6%.
The plan therefore pivots to two distinct uses for L1, both larger-than-1.6%:

1. **Phase 2.5 — Global-cost variant (Direction-C upgrade)**: replace the
   single-expert local SwiGLU with a multi-layer rollout cost. This is
   IMPOSSIBLE today on HF (full-model forward × tens of thousands of pairs
   = days of compute). It becomes tractable under L1 because vLLM's batched
   forward amortizes across pairs in a single `LLM.generate(prompts=...)`
   call (kv-cache reuse, captured CUDA graph).
2. **Phase 2.6 — Optimizer/heal ablation**: drive A/B variants of the Stage
   2.5 router-KD or per-layer heal optimizer in a single persistent vLLM
   session. The dominant cost there is the forward, not Hungarian — L1's
   forward speed-up directly converts to ablation throughput.

Stage 2's current `_output_space_cost` is **NOT** a good first consumer of an
L1 backend on its own. It is the unit-test surface for the primitives, but
the production payoff is in 2.5 / 2.6, not in shaving 1.6% off SC.

---

## 1. Phase 1 verification report (read-only, completed)

The Phase-1 read-only verification is in the conversation history this plan
was produced from. Key facts the plan depends on:

- SC = baseline + Direction C only (`{cost_alignment: "output",
  capacity_util_threshold: 0}`). Source:
  `max_quality/src/moe_compress/run_ablations.py:195`.
- Direction C iterates per (child m, candidate centroid c) and forwards a
  **single-expert** SwiGLU on calibration tokens (`_swiglu_forward` at
  `max_quality/src/moe_compress/stage2/plugins/output_space_cost.py:157`,
  called from `_output_space_cost` at `:276`). It does NOT forward the
  whole HF model. The "HF model forward" framing of the L1 motivation
  is **wrong** for the current code shape.
- ~3h/row cost (`MOE_COMPRESS_REPORT.md:250`) is dominated by the
  per-pair Hungarian neuron-permutation solve in
  `_tentative_merged_weights` (~25 ms/pair, ~96%); SwiGLU is ~0.4 ms.
  The Hungarian doesn't GPU-vectorize.
- vLLM 0.21 V1 exposes `LLM.collective_rpc(method, timeout, args, kwargs)`
  AND `LLM.apply_model(func: Callable[[nn.Module], _R]) -> list[_R]`
  (`/tmp/vllm-patched/vllm/entrypoints/llm.py:646` and `:678`). `apply_model`
  is the cleaner public path.
- vLLM ships `replace_parameter(layer, name, new_data, prefer_copy=True)`
  (`/tmp/vllm-patched/vllm/model_executor/utils.py:47`) whose docstring
  states: *"This preserves the parameter's storage address (`data_ptr`),
  which is required for captured CUDA graphs to remain valid across weight
  updates (e.g. in RL training loops)."* Upstream-blessed for L1.
- vLLM `FusedMoE` weight layout: `w13_weight[E, 2I, H]` (gate||up),
  `w2_weight[E, H, I]` — confirmed at
  `/tmp/vllm-patched/vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py:97`.
- HF transformers (5.5.3) `Qwen3MoeExperts` layout: `gate_up_proj[E, 2I, H]`
  + `down_proj[E, H, I]` — confirmed at
  `/home/lucas/.local/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:223-224`.
- **Stage 2 already accesses HF's fused layout**, NOT a `ModuleList[nn.Linear]`
  per expert: `build_banks` at
  `max_quality/src/moe_compress/utils/model_io.py:515-550` slices
  `gate_up_proj` and uses `down_proj` directly. The L1 validation harness's
  Test 3 description ("16 separate `nn.Linear` experts per layer") is
  **outdated**.

Implication: the vLLM↔HF weight translation is essentially the identity
permutation (slice the same fused tensor). The "weight layout adapter" risk
is much smaller than the campaign halt-trigger memo implied.

---

## 2. Implementation plan

### Phase 2.1 — The three primitives in isolation

New module: `max_quality/src/moe_compress/utils/vllm_runtime.py`. ~120 LoC.

**Three primitives:**

1. `vllm_model(llm: LLM) -> nn.Module` — single-worker reach into the live
   model.
   ```python
   def vllm_model(llm):
       """Return the in-process nn.Module on the driver worker.

       Uses LLM.apply_model (public, vllm-blessed for single-worker)
       which routes through collective_rpc internally. Multi-worker
       returns the local-rank model — Stage 2 callers must run with
       tensor_parallel_size=1 for the primitives' contract to hold.
       """
       models = llm.apply_model(lambda m: m)
       assert len(models) == 1, "vllm_runtime requires tensor_parallel_size=1"
       return models[0]
   ```

   Why not `collective_rpc(lambda w: w.model_runner.model)`: closures may not
   serialize across worker processes under V1's multi-worker setup. `apply_model`
   exists exactly to paper over the V0/V1 attribute-chain drift. The
   `l1_validation_harness.py` reaches via
   `llm.llm_engine.model_executor.driver_worker.model_runner.model` — that
   path is V0-specific and breaks on V1; do not use it in new code.

2. `update_weights_inplace(llm, name_to_tensor: dict[str, torch.Tensor]) -> None`
   ```python
   def update_weights_inplace(llm, name_to_tensor):
       """Push named tensors into the live vLLM model with prefer_copy
       semantics. CUDA-graph-safe per vllm.model_executor.utils.replace_parameter.

       Caller owns dtype/device coercion — primitive does NOT cast.
       Raises ValueError on shape/dtype/device mismatch (NOT silent re-register).
       """
       def _push(model, mapping):
           from vllm.model_executor.utils import replace_parameter
           for qualname, new in mapping.items():
               # Walk dotted path to the *parent* module + leaf attr.
               parts = qualname.split(".")
               parent = model
               for p in parts[:-1]:
                   parent = getattr(parent, p)
               leaf = parts[-1]
               cur = getattr(parent, leaf)
               if cur.shape != new.shape:
                   raise ValueError(f"{qualname}: shape {tuple(cur.shape)} vs {tuple(new.shape)}")
               if cur.dtype != new.dtype:
                   raise ValueError(f"{qualname}: dtype {cur.dtype} vs {new.dtype}")
               if cur.device != new.device:
                   raise ValueError(f"{qualname}: device {cur.device} vs {new.device}")
               replace_parameter(parent, leaf, new, prefer_copy=True)
       llm.apply_model(lambda m: _push(m, name_to_tensor))
   ```

   Hard failure modes the primitive surfaces (do NOT swallow):
   * `prefer_copy=True` falls through to `setattr` re-register if shape/dtype/
     device disagree — that path breaks CUDA graphs silently. Guard upstream.
   * `convert_to_unquantized_kernel_format` may have shuffled `w13_weight` /
     `w2_weight` into a backend-specific layout at load time (AITER `is_shuffled`,
     ROCm padding). Under the campaign operating point
     (`VLLM_USE_FLASHINFER_MOE_FP16=0`, NVIDIA, no AITER), the unquantized
     Triton backend uses the canonical layout; the primitive asserts
     `not getattr(w, "is_shuffled", False)` and `w.stride()[-1] == 1` to fail
     loud on any future deployment surprise.

3. `hf_to_vllm_experts(hf_layer, layer_idx: int) -> dict[str, torch.Tensor]`
   ```python
   def hf_to_vllm_experts(hf_layer, layer_idx):
       """Translate HF Qwen3MoeExperts -> vLLM w13_weight/w2_weight stacked form.

       Both HF (transformers >= 5.x) and vLLM use the same [E, 2I, H] / [E, H, I]
       fused layout. The translation is in practice the identity; the wrapper
       exists so a future model adapter can intercept it (and so we never
       touch raw .data attributes outside this one place).

       Gate||up ordering: vLLM's w13_weight is concat([gate, up], dim=0)[E, 2I, H].
       HF's gate_up_proj is also [E, 2I, H]. We do NOT auto-detect order; we
       pin it to gate||up and add a smoke test (Phase 2.1 isolation test 3) that
       breaks loudly if a future HF revision flips the order.
       """
       experts = hf_layer.experts
       w13 = experts.gate_up_proj.detach()
       w2 = experts.down_proj.detach()
       prefix = f"model.layers.{layer_idx}.mlp.experts"
       return {f"{prefix}.w13_weight": w13, f"{prefix}.w2_weight": w2}
   ```

**Isolation tests** (each ~30 min on a tiny model, ~$1.50 / each at A10G spot $0.50/hr — H100 not required):

* `test_vllm_model_singleton` — load `PrimeIntellect/qwen3-moe-tiny` (the
  same model the campaign's `l1_validation_harness.py` uses; `670M`, fits
  in a 24 GB A10G alongside HF), assert `vllm_model(llm)` returns a single
  `nn.Module` instance, assert `model.model.layers[1].mlp.experts.w13_weight`
  is reachable. ~5 min.
* `test_update_weights_inplace_cuda_graph_survives` — clone of harness Test 1
  (zero one expert via `update_weights_inplace`, generate same prompt cold/warm/
  post-update; assert warm-class latency post-update AND text changes). Distinct
  from the harness in that it goes through the production primitive, not raw
  `.copy_()`. ~10 min.
* `test_hf_to_vllm_translation_byte_identity` — harness Test 3a (zero pre-
  translation diff at fp32, < `1e-4` after bf16 round-trip; the gate||up order
  is hard-pinned, not auto-detected). ~15 min.

**Halt-triggers (specific, not "L1 is impossible"):**

* Primitive 1 returns >1 model → tensor_parallel_size!=1 was active; the
  Stage 2 integration contract requires TP=1. Surface "Stage 2 must run
  vLLM with TP=1 in this design".
* Primitive 2 hits the shape/dtype/device mismatch path → there's a hidden
  shuffle in the load path under our operating point. Surface specific
  shape/dtype line; do not proceed to Phase 2.2.
* Primitive 3 sees `gate_up_proj` shape != `[E, 2I, H]` → HF revision drift;
  surface and pin `transformers==5.5.x` upstream.

GPU budget: 1 × A10G spot × 1 h ≈ $0.50. (Tiny model, not the 35B teacher.)

### Phase 2.2 — Integrate primitives into Stage 2 (READ-ONLY proof — NOT THE SC WIN)

This phase exists to prove correctness, not to win speed. The current
SC bottleneck is Hungarian (96%), not the SwiGLU forward (1.6%). Replacing
the SwiGLU with a vLLM forward will be SLOWER on the 35B teacher because
of per-pair launch overhead. The deliverable is the byte-identity guarantee,
not throughput.

**Touch points:**

* `_output_space_cost` at
  `max_quality/src/moe_compress/stage2/plugins/output_space_cost.py:276`.
  The current per-pair forward at lines 421-438 calls `_swiglu_forward(W_m...)`
  and `_swiglu_forward(merged...)` on raw weight tensors.
* Replacement strategy (gated on a new YAML knob `stage2_reap_ream.cost_engine:
  "hf" | "vllm"`, default `"hf"` — byte-identical to today when off):
  1. Build the merged weights HF-side (no change to `_tentative_merged_weights`).
  2. Push the (centroid_id, child_id) merged tensors into vLLM via
     `update_weights_inplace` — targeting `w13_weight[centroid_id]` and
     `w2_weight[centroid_id]` SLICES (vLLM's expert dim 0 is `[E, ...]`).
  3. Forward a sentinel prompt batch through vLLM whose hidden-state at layer
     L equals `x_all` (the cached `layer_inputs` reservoir). This is the hard
     part — see "Data-flow constraints" below.
  4. Capture the per-expert routed output via the existing patched
     `expert_out_unweighted` hook (already wired by the V1+V2 writers; the
     patch exposes this with `VLLM_CALIB_CAPTURE_EXPERT_UNWEIGHTED=1`,
     MANIFEST line 199 onward). Read the layer-L expert-m output, compute
     the same routing-weighted MSE.
  5. Push the originals back via `update_weights_inplace` to revert.

**Data-flow constraint that breaks the naive design:**

`_output_space_cost` operates on a 1024-token reservoir of **hidden states at
layer L** (`layer_inputs`). vLLM's `LLM.generate` takes **token IDs**, not
hidden states. There is no public API to seed a forward at a mid-network
hidden state. Two real options:

* **Option A (rejected — too invasive)**: patch vLLM to expose a
  `forward_from_hidden_states(layer_idx, x)` entrypoint. New patch surface,
  new MD5, regenerate wheel. ~2-3 days of patch design + a new wheel
  rebuild on HF Jobs.

* **Option B (recommended)**: don't seed at layer L. Instead, accept that
  the cost is now a function of the **token-ID** calibration prompts, not the
  pre-cached layer-L hidden states. The cost-relevant quantity is "expert m's
  routed output on tokens that route to m at layer L" — which the vLLM
  forward already captures via `expert_out_unweighted`. The reservoir
  becomes a sub-sample of TOKEN IDS rather than HIDDEN STATES. This changes
  the cost matrix's input distribution slightly (post-attention hidden
  vs cached); the byte-identity gate (Phase 2.3) will measure whether the
  drift is tolerable.

Memory budget on a 141 GB H200:
* Teacher BF16 weights: ~70 GB.
* vLLM KV cache @ 1024 tokens × 36 layers × 2 × 64 heads × 128 d_head × bf16:
  ~3 GB at the calibration batch.
* HF model in Stage 2 driver: ~70 GB (BF16 35B).
* **Total at peak: ~143 GB — does NOT fit.** Two H200s required, OR FP8
  teacher in vLLM (already supported by `TEACHER_MODEL_REPO=Qwen/Qwen3.6-35B-A3B-FP8`
  per `max_quality/docker/run_strategy_sweep.sh:35`). With FP8 teacher (~35 GB)
  + BF16 HF student (~70 GB) + vLLM KV (~3 GB) = ~108 GB. Fits.

The FP8 teacher route adds a numerical-fidelity question: the per-pair cost
under FP8 vLLM vs BF16 HF is not byte-identical even before any other source
of drift. The Phase 2.3 gate must tolerate FP8 noise (~0.02 BPT gap shift,
based on the run_strategy_sweep operating point).

**Cross-cutting design choice**: do NOT drop the HF model after Stage 2 builds
the merged weights. The merged-weights HF tensor `_tentative_merged_weights`
is recomputed PER PAIR; if we drop HF, we lose the permutation cache + ream_acc
state. Keep HF resident; the FP8 vLLM teacher is the lever that makes it fit.

GPU budget: 1 × H200 × 5 h SC run × $3.39/hr = $17. (Single SC run, not the full sweep.)

### Phase 2.3 — Validation gate

Run SC end-to-end with `stage2_reap_ream.cost_engine: "vllm"` and compare
to the historical SC `bpt_gap = 0.1293` (`MOE_COMPRESS_REPORT.md:203`).

* Tolerance: `|new_bpt_gap - 0.1293| < 0.02` (covers fp8 teacher noise + bf16
  hidden-state distribution drift).
* On failure, the gate produces the per-layer cost-matrix diff
  (`cost_vllm[m, c] - cost_hf[m, c]` heatmap) and the per-pair assignment
  diff. We do NOT proceed to Phase 2.4 without the gate green.

GPU budget: 1 × H200 × 5 h × $3.39 = $17. If the gate fails, +1 retry budgeted
= $34 worst case.

**Decision criterion if the gate fails by a small margin** (e.g. 0.02 < gap <
0.05): keep the HF backend as default, document the vLLM backend as
"for global-cost extension only" (Phase 2.5 use case). Do NOT silently raise
the tolerance — surface to user.

### Phase 2.4 — SCD diagnostic instrumentation

Once Phase 2.3 passes, the vLLM backend becomes the substrate for measuring
why SCD regresses (`bpt_gap 0.1868 vs SC 0.1293` —
`MOE_COMPRESS_REPORT.md:204`). Hypothesis: 2-opt swaps lower the LOCAL
single-layer cost but cascade to higher downstream layer cost.

**Instrumentation**: around `_two_opt_refine` at
`max_quality/src/moe_compress/stage2/plugins/two_opt_refine.py:73`, before
accepting a strictly-lowering swap, evaluate the **multi-layer rollout cost**
(layer L → final output residual under tentative merge) for both pre-swap and
post-swap assignments, log both. Do NOT change the acceptance rule yet — this
is diagnostic, not prescriptive.

* Multi-layer rollout: HF-side build merged weights as today, push to vLLM,
  run vLLM `generate(prompts, max_new_tokens=0)` to get logits, measure
  vs teacher logits. ~10s per swap candidate at 128-prompt subset (the
  block-outputs subset from Item 7 of the campaign).
* SCD has ~2-3 swaps per layer that get accepted on the static cost; ×
  ~21 merge-amenable layers × 2 (pre+post) × 10s = ~30 min of instrumented
  cost per SCD run. Tractable.

Deliverable: a CSV `(layer, swap_id, local_cost_delta, global_cost_delta,
accepted_by_local, would_be_accepted_by_global)`. If `global_cost_delta`
disagrees with `local_cost_delta` on most accepted swaps, the proxy gap is
confirmed.

GPU budget: 1 × H200 × 6 h × $3.39 = $20.

### Phase 2.5 — Global-cost extension (Direction-C upgrade)

Gated on Phase 2.4 confirming the local-vs-global gap. The Phase 2.4
instrumentation already implements the global-cost evaluator; Phase 2.5
just promotes it from diagnostic to objective.

**Variants to ablate (each ~5 h H200, ~$17 each):**

* **SC-global**: replace the local single-expert cost in `_output_space_cost`
  with the multi-layer rollout cost (full residual through layers L+1..N-1
  after tentative merge at L). The token cost: per-pair rollout = forward of
  128 prompts × (N-L) layers. With L₀ ≈ 4 (first MoE layer) and N=36
  (Qwen3.6-35B-A3B layer count), that's ~32 layers/forward × ~2400 pairs/layer
  × 21 layers ≈ 1.6M layer-forwards per run. Single H100 vLLM can do
  ~5k layer-forwards/sec batched at this size → ~320s. With Python overhead
  it'll be slower; budget 1-2 h per SC-global Stage 2.
* **SC-global-SCD**: same plus 2-opt refinement on the global cost.

Halt-trigger: if SC-global Stage 2 wall-clock exceeds 8 h, fall back to a
**sampled** global cost (eval rollout on a 10% random sub-sample of pairs,
use the local cost otherwise). Surface the sampling rate to user.

GPU budget: ~$50-100 (2-3 variants × 5-10 h).

### Phase 2.6 — Optimizer ablation for Stage 2.5 healing on SC

This is the **highest-ROI L1 consumer** because:
1. The dominant cost is the optimizer forward, NOT Hungarian — L1's forward
   speedup converts directly.
2. The `tasks/todo.md` note (on the OTHER repo's `feat/heal-lr-schedule`
   branch — `/home/lucas/moe_compress/tasks/todo.md:1-8`) frames the
   question as: does the heal optimizer's LR/warmup unlock more SC
   recovery? Answer requires running heal × several optimizer variants.

**Setup**: at the end of an SC Stage 2 run, hold the merged-state pruned
model in memory, spin up vLLM with merged weights, then drive variants of:
* CPU-offloaded Adam (numerically faithful; expensive memory but slow).
* 8-bit Adam (cheap, slight numerical drift).
* Partial unfreeze (heal only N most-damaged layers).
* Constant LR vs cosine schedule (per the heal-lr-schedule branch).

Each variant trains for 16k steps with the same calibration data. Compare
`bpt_gap` deltas vs SC's 0.1293 baseline.

Why vLLM here: the heal training loop calls `_LayerOutputCapture` for
teacher-side targets. Today that's an HF teacher hook (`router_kd/plugins/
merge_repair.py::_LayerOutputCapture`). Under L1, the teacher's per-layer
block outputs come from the patched vLLM hook (already wired via Item 7:
`calib-v2-block-outputs-writer`). The student-side optimizer still runs HF,
but each step's teacher forward is replaced by a vLLM batched call —
~5-8× faster on a 35B teacher.

GPU budget: ~$100-150 (3-4 variants × 8-12 h heal each).

---

## 3. Cross-cutting concerns

### Patch identity / wheel discipline (per MANIFEST.md)

This plan does NOT introduce any new vLLM source patches in Phase 2.1-2.3.
All three primitives use public-API surface (`apply_model`, `replace_parameter`).
The `expert_out_unweighted` hook for Phase 2.2 step 4 is already in the
shipped wheel (tag `calib-v2-max-layer-early-exit`, MD5
`a8da5e321ac7fb30f1648fba3476bea6`).

Phase 2.5 might need a `forward_from_hidden_states` entrypoint IF Phase 2.2
Option B's token-ID distribution drift exceeds the validation gate. That
would be the only patch bump in this plan. The patch contents would be
~80-120 LoC (an entrypoint analogous to the existing `Qwen3MoeModel.forward`
with `start_layer` override + `inputs_embeds` accepted as a tensor argument).
If invoked, MANIFEST.md gets a new section, new tag (`calib-v2-l1-rollout`
or similar), MD5 update, new wheel build on HF Jobs.

### No PR language

Per the user's standing directive (`feedback_no_pr_language` in memory),
all commits land directly on `feat/calibration-v2` with FF-only merges from
sub-branches. No PRs. Each phase's work goes on a sub-branch
(`feat/l1-primitives`, `feat/l1-cost-backend`, `feat/l1-global-cost`,
`feat/l1-heal-ablation`); each sub-branch FFs into `feat/calibration-v2`
when its gate passes.

### GPU budget (cumulative)

| Phase | GPU | Time | Cost |
|---|---|---|---|
| 2.1 | A10G spot | 1 h | $0.50 |
| 2.2 (dev) | H200 spot | 6 h | $20 |
| 2.3 (validation gate) | H200 spot | 5 h ± retry | $17-34 |
| 2.4 (instrumentation) | H200 spot | 6 h | $20 |
| 2.5 (global-cost) | H200 spot | 15-25 h | $50-85 |
| 2.6 (heal ablation) | H200 spot | 30-50 h | $100-170 |
| **Total** | | | **~$210-330** |

Dev time (planning + implementing + reviewing, NOT including GPU wait):
* Phase 2.1: 4-6 h (primitives are mechanical).
* Phase 2.2: 12-16 h (data-flow design is the hard part).
* Phase 2.3-2.4: 6-8 h.
* Phase 2.5: 16-24 h (if the patch is needed, +24 h).
* Phase 2.6: 12-16 h.
* Total: ~60-90 h of dev.

### Halt triggers (specific, surface to user, do not auto-resolve)

* Phase 2.1 isolation tests fail on `prefer_copy` → write up exact mismatch
  (shape / dtype / device / shuffle flag); STOP.
* Phase 2.2 vLLM peak memory exceeds H200 with FP8 teacher → surface
  "Phase 2 requires 2 × H200 OR FP8 student", do NOT silently shrink the
  calibration token cap.
* Phase 2.3 `bpt_gap` drift > 0.05 → STOP, surface per-layer cost-matrix
  diff, do not raise tolerance silently.
* Phase 2.5 SC-global Stage 2 wall-clock > 8 h → fall back to sampled
  global cost, surface the sample rate.
* Any phase: if the patched wheel needs ANY change, bump tag + MD5 +
  MANIFEST per the existing campaign discipline. Do NOT silently re-apply.

---

## 4. Risks the plan enumerates

### R1 — `collective_rpc(lambda)` closure serialization across workers

vLLM ships a Ray-based / mp-based worker fan-out for `tensor_parallel_size>1`;
closure-pickling may fail at the boundary. **Mitigation**: pin the design
to `tensor_parallel_size=1` (single worker, no fan-out) and assert it at
primitive entry. The 35B-A3B teacher fits on a single H200 in FP8.
Multi-GPU scale-up is out of scope.

### R2 — `replace_parameter(prefer_copy=True)` and torch.compile re-trace

vLLM uses `@support_torch_compile` on `Qwen3MoeModel.forward`. The L2
max_layer patch (MANIFEST line 64) reads `_CALIB_MAX_LAYER` ONCE at
forward entry and *deliberately allows recompile on value change*. Weight
mutations via `prefer_copy=True` preserve `data_ptr` — which is the
sentinel torch.compile uses for tensor identity in its CUDA-graph
specialization. Therefore weight in-place updates should NOT trigger
recompile. **Verification**: Phase 2.1 test 2 measures wall-clock of
post-update forward against the warm baseline. If post-update >
1.5 × warm, recompile is happening; STOP and investigate.

### R3 — MoE backend kernels hold internal weight references

`UnquantizedMoEMethod._setup_kernel` (`unquantized_fused_moe_method.py:150`)
captures `layer.w13_weight` / `layer.w2_weight` into a `moe_kernel` at
load time. The `is_weight_update` path uses `prefer_copy=True` → preserves
storage → kernels see the new data. **But**: if any backend (AITER,
ROCm-padded) has shuffled the storage, our pushed unshuffled tensor will
be silently misinterpreted. **Mitigation**: Phase 2.1 primitive 2 asserts
`not getattr(w, "is_shuffled", False)` and `w.stride()[-1] == 1` (canonical
contiguous). Under `VLLM_USE_FLASHINFER_MOE_FP16=0` + NVIDIA + Triton
backend (the operating point), this holds.

### R4 — HF↔vLLM weight layout drift across model revisions

This plan pins `transformers==5.5.x` (per the patched-wheel build
requirements in `max_quality/docs/calibration_v2_data_capture_plan.md:206`).
A future HF revision could flip `gate||up` to `up||gate`. **Mitigation**:
Phase 2.1 test 3 asserts the gate-first ordering byte-for-byte; CI catches
the drift.

### R5 — Stage 2 holds HF model in CPU memory between layers

Stage 2's per-layer driver (`stage2/orchestrator.py`) currently moves
expert tensors GPU↔CPU between layers to fit in VRAM. The vLLM session
keeps the full teacher GPU-resident throughout. **Total VRAM peak**
(see Phase 2.2 memory accounting): ~108 GB with FP8 vLLM teacher + BF16
HF student. Fits H200, does NOT fit H100 (80 GB). **Mitigation**: pin
H200 (or 2 × H100). Do NOT design for H100 in this plan.

### R6 — The 96% Hungarian bottleneck swamps any forward speedup

This is the killer risk. **vLLM cannot speed up the Hungarian solve.**
If Phase 2.1-2.3 lands cleanly and Phase 2.4 diagnostics show the
global-cost variant is the actual win, the per-pair forward cost grows
(multi-layer rollout vs single-expert SwiGLU) and the Hungarian share
drops below 50%. Until then, L1 for the current Direction-C cost shape
is **architectural cleanup, not a speedup**. The plan must communicate
this clearly: the SC win from L1 comes from Phase 2.5 (global cost) or
2.6 (heal ablation), not 2.2.

### R7 — Sub-cap calibration tokens at layer L (Phase 2.2 Option B drift)

vLLM forwards from token-IDs, not from cached layer-L hidden states.
The drift between "cost computed on cached hidden states" and "cost
computed on freshly-forwarded hidden states" is bounded by attention
non-determinism in the layers 0..L-1, which is itself small but non-zero
under FP8. The Phase 2.3 gate's 0.02 tolerance subsumes this; the
patched-vLLM `forward_from_hidden_states` extension (Phase 2.5 IF needed)
would close it.

---

## 5. Decision points (go/no-go per phase)

| Phase | Go condition | No-go fallback |
|---|---|---|
| 2.1 | All 3 isolation tests PASS | Surface specific failure; stop the chain. The campaign halt-trigger memo from `9647362` was right. |
| 2.2 | Implementation green + smoke test on tiny model | Drop Option B, design Option A patch (+1 wheel bump + week of dev). |
| 2.3 | `bpt_gap` within 0.02 of 0.1293 | Surface drift origin (FP8 vs hidden-state drift); negotiate larger tolerance with user OR drop vLLM backend as default. |
| 2.4 | Per-swap local/global divergence visible | If SCD-like assignments are local-faithful AND global-faithful, the SCD regression has a different cause; surface and rethink. |
| 2.5 | SC-global `bpt_gap < 0.1293` | If equal or worse, the local-vs-global hypothesis is wrong; document and stop. |
| 2.6 | Any heal variant moves SC `bpt_gap` by >0.02 | If none do, the "optimizer was binding" hypothesis is wrong; healing IS dead for SC and we ship SC as-is (per the user's todo.md `if this still ≈ S0, healing is dead and we ship SC` reading). |

---

## 6. What this plan does NOT do

* Does NOT revive the original L1 (full REAP+REAM as N vLLM passes).
* Does NOT touch Stage 1 GRAPE (unchanged across Stage-2 strategy rows).
* Does NOT introduce new vLLM patches in Phases 2.1-2.4 (only Phase 2.5 IF
  the hidden-state drift exceeds tolerance).
* Does NOT remove the HF model from the Stage 2 driver. The merged-weight
  math (permutation cache, REAM accumulators) stays HF-side.
* Does NOT change SC's algorithmic semantics — Phase 2.3 is byte-near-identity;
  Phase 2.5 IS a semantic change (global cost) and is gated on a fresh ablation.
