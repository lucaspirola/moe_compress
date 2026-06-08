# REAP faithful_prune — GPU-acceleration research (READ-ONLY, measured)

Repo: /home/lucas/ai/moe_compress @ main (HEAD f984738; prompt said 3d715d6 — the
prune path is unchanged, plugin/primitives identical).
GPU: RTX 5080 16GB, shared. At measurement time 0% util, 0–7.3GB used → absolute
numbers reported (uncontended window). Method noted per-row.

## THE KEY QUESTION: where do the tensors live during Stage 2 faithful_prune?

Traced: `run_pipeline.py:156` sets `device = cuda if available else cpu` and loads
the model via `load_model(..., device_map=config["model"]["device_map"])`
(run_pipeline.py:565). Config `qwen36_35b_a3b_30pct.yaml:14` → `device_map: auto`.

`load_model` (utils/model_io.py:81-99) passes `device_map="auto"` straight to HF
accelerate (only the `load_in_4bit` branch rewrites it to `{"":0}`).

**Therefore placement is decided by accelerate at load time:**
- **H200 production target (141GB, model ~70GB):** everything fits → all experts
  on GPU. The faithful drop (`bank.select` / `_resize_router` `index_select`) runs
  **GPU-resident → already optimal.** No win available; PROVEN below.
- **GPU too small / contended (16GB local box, or a smaller card):** accelerate
  `auto` spills layers to CPU. `index_select` on those layers runs on CPU →
  catastrophically slow on bf16 (no native CPU bf16 SIMD). This is the only
  regime where a GPU win exists.

The `select`/`_resize_router` index_select runs on `stacked.device` /
`router.weight.device` (model_io.py:477, merging.py:423) — i.e. wherever the
layer currently lives. The functions are device-agnostic; the device is inherited.

## What faithful_prune actually does per layer (post_merge, reap_prune.py:352-367)
`build_banks(ref)` → for each of {gate_up_proj, down_proj}: `bank.select(kept)`
(index_select+clone, model_io.py:478) → `_resize_router_for_kept_experts`
(index_select+clone on router.weight, merging.py:425). NO covariance, NO calibration
forward, NO solver/merge/heal (all dropped in faithful mode). So this gather IS the
dominant per-layer compute.

## Real dims (Qwen3.6-35B-A3B, text_config — from HF cache config.json)
num_experts=256, hidden=2048, moe_intermediate=512, top_k=8, 40 layers, bf16.
(Prompt's "~128 experts, hidden~4096" was off; used the REAL dims.)
Fused per layer: gate_up_proj [256,1024,2048]=1.07GB, down_proj [256,2048,512]=0.54GB.
prune_fraction 0.30 → keep 180 / 256.

## MEASUREMENTS

### M1 — synthetic index_select+clone (gate_up + down + router), per layer
(/tmp/reap_select_timing.py, 20 iters, warmup 3, absolute, uncontended)
| path | per-layer | whole-model (40L) |
|---|---|---|
| CPU-resident | 2736 ms | 109.4 s |
| **GPU-resident (H200 path)** | **5.56 ms** | **0.22 s** |
| GPU + H2D+D2H transfer | 498 ms | 19.9 s |

speedup CPU→GPU-resident **492x**; CPU→GPU-with-transfer **5.5x**.
bit-exact: `torch.equal(GPU gather, CPU gather)` = **True**.

### M2 — the REAL production functions (ExpertMatrixBank.select + _resize_router)
(/tmp/reap_real_funcs_timing.py — imports the actual functions, builds a
Qwen3_5MoeExperts-shaped module, runs the exact post_merge body; module
reconstructed per-iter since select() is one-shot, construct-time subtracted)
| path | per-layer drop | whole-model (40L) |
|---|---|---|
| CPU-resident | 2156 ms | 86.3 s |
| **GPU-resident** | **5.57 ms** | **0.22 s** |

speedup **387x**; bit-exact (real-func GPU result == CPU result) = **True**.
Confirms M1 with the actual code path.

### M3 — cross-layer batching potential (hunt item #2)
(/tmp/reap_batch_layers.py)
40-layer GPU drop = 462 ms total (11.6 ms/layer incl. clone); gather-only 232 ms.
Each layer has a DIFFERENT kept-id set (REAP saliency is per-layer) → no shared
gather index → **no true cross-layer batching exists.** The per-layer loop already
issues async gathers on one stream. Whole-model GPU cost is sub-second → immaterial.

### M4 — `.clone()` redundancy (micro-finding, NOT a tier-1 win)
`index_select` already returns a fresh, contiguous, independent tensor
(data_ptr ≠ source, is_contiguous True, survives source mutation — measured).
The production `.clone()` after it (model_io.py:478, merging.py:425/428) is a
redundant second full-tensor allocation = ~50% of the GPU op cost (M3). Dropping
it is correctness-neutral, but saves only ~230 ms across the whole 40-layer model
→ not worth touching production for; logged for completeness.

## RANKED TIER-1 (numeric-preserving) FINDINGS

### T1 — Ensure the model is GPU-resident before the faithful drop (CONDITIONAL)
- (a) current cost if model on CPU: **2156 ms/layer (86 s whole model)** — measured M2.
- (b) GPU cost: **5.6 ms/layer (0.22 s)** — measured M2.
- (c) net: **387–492x** if the layers were on CPU; **5.5x** even charging a full
  H2D+D2H round trip (honest, model-on-CPU case, M1).
- (d) bit-exact: index_select is a pure gather; `torch.equal(GPU,CPU)` = True (M1,M2).
- (e) risk: **on the H200 target this is a NO-OP** — `device_map:auto` already puts
  everything on GPU, so the drop is already the 5.6ms path. The win exists ONLY when
  accelerate spills to CPU (undersized/contended GPU). Golden touched: none (the
  drop output is byte-identical regardless of device — proven). Acting on this would
  mean forcing the layer to GPU for the gather when it landed on CPU; only worth it
  on the small-GPU regime, and even then the honest gain is the 5.5x transfer-charged
  number, not 387x, unless the model stays resident.

### VERDICT for the production (H200) path: ALREADY OPTIMAL — proven
On the documented H200 target the experts are GPU-resident, the gather is 5.6ms/layer
(0.22s for the whole model), there is no cross-layer batching to exploit (M3), and the
only inefficiency (redundant .clone, M4) is sub-second across 40 layers. **No Tier-1
GPU win exists for the in-spec H200 run.** The 387x figure is real but applies only to
a CPU-resident regime that the production config does not hit.

## Honest caveats
- All numbers from an uncontended RTX 5080 window; absolute, single-GPU, bf16.
- M2 reconstructs the module each iter (select is one-shot); construct time measured
  and subtracted — residual is the pure select+resize cost.
- CPU bf16 is pathologically slow (likely upcast per-element); the 2156ms is real for
  this box but a server CPU with more cores/AVX512-BF16 would be faster — still
  orders slower than GPU-resident.
- Did NOT load the real 35B model (16GB box); the accelerate `auto` CPU-spill
  behavior is asserted from the code path + documented accelerate semantics + the
  existing memory note "on H200 everything fits on 1 GPU", not from a live 35B load.
