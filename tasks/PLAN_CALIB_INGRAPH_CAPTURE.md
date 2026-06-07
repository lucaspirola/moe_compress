# PLAN — In-graph cudagraph-safe calibration capture (CORRECTED, all signals)

Implements tasks/CALIB_CUDAGRAPH_FAST_CAPTURE_DESIGN.md. Branch feat/calib-cudagraph-capture.
Revised to fix plan-review round-1 findings (C1 custom-op routing, C2/C3, H1-3, M1-4).

---

# CALIB CUDAGRAPH FAST CAPTURE — CORRECTED IMPLEMENTATION PLAN

## 0. Boundary Classification (C1 resolution)

**Custom-op-interior (inline GPU accumulation is graph-safe):** All code called from `_moe_forward` → `_forward_impl` → `_apply_quant_method` → `_quant_method.apply` → `TritonExperts.apply`. This includes:

- `router` / `topk_weights` / `topk_ids` (in `_apply_quant_method`, moe_runner.py:526–544)
- `expert_in` (in `_apply_quant_method`, before `.apply()`)
- `expert_out_weighted` (in `_apply_quant_method`, after `.apply()`)
- `expert_out_unweighted` (inside `TritonExperts.apply`, triton_moe.py)
- `expert_mid` (inside `TritonExperts.apply`)

All of these are inside the `vllm.moe_forward` custom op (moe_runner.py:167–173), which is opaque to Dynamo.

**Traced-region (must use `direct_register_custom_op` wrappers):** Dispatched from code that `@support_torch_compile` traces:

- `linear_in` / `lm_head_in`: dispatched from `ColumnParallelLinear.forward` (linear.py:582) and `LogitsProcessor._get_logits` (logits_processor.py:96). Both are called from `Qwen3MoeModel`/`Qwen3NextModel` which is decorated `@support_torch_compile` (qwen3_moe.py:439, qwen3_next.py:461).
- `block_out`: dispatched from `Qwen3NextSparseMoeBlock.forward` (qwen3_next.py:175) / `Qwen3MoeSparseMoeBlock.forward` (qwen3_moe.py:226). These are methods on sub-modules of `@support_torch_compile` models.
- `layer_in`: dispatched from decoder layer `forward` (qwen3_next.py:401 / qwen3_moe.py:416), also inside the compiled region.

**Writers affected by traced-region signals:**
- `imatrix` (dense path: `linear_in` and `lm_head_in`)
- `block_outputs` (`block_out`)
- `stage2_profile` (`layer_in`, `expert_out_unweighted`, etc.)
- `input_cov` (dense path: `linear_in`)

**Writers that are pure custom-op-interior:**
- `reap_scores`, `per_expert_max`, `routing_stats`, `router_logits_stats`, `wanda_scalar_row`, `output_reservoir` — all consume only `router`, `expert_in`, `expert_out_unweighted`, `expert_mid`, `expert_out_weighted`.

---

## 1. Shared Infrastructure Changes

### 1.1 `vllm/calibration_hooks.py`

Replace `_callbacks` dict + `dispatch()` Python calls with persistent GPU accumulator infrastructure:

```
# Module-level GPU accumulator registries (keyed by layer_idx int)
_REAP_SCORE_ACCUM_GPU:     dict[int, Tensor]   # [E] fp32
_REAP_TOKEN_COUNTS_GPU:    dict[int, Tensor]   # [E] int64
_PEM_ACCUM_GPU:            dict[int, Tensor]   # [E, K] fp32
_ROUTING_FREQ_GPU:         dict[int, Tensor]   # [E] int64
_ROUTING_WSUM_GPU:         dict[int, Tensor]   # [E] int64
_ROUTER_LOGITS_SUM_GPU:    dict[int, Tensor]   # [E] fp32
_ROUTER_LOGITS_SQ_GPU:     dict[int, Tensor]   # [E] fp32
_ROUTER_COUNT_GPU:         dict[int, Tensor]   # [E] int64
_IMATRIX_MoE_GPU:          dict[int, Tensor]   # [E, H] fp32 (expert_in sq-sum)
_IMATRIX_DENSE_GPU:        dict[str, Tensor]   # keyed by layer name, fp32
_WANDA_GPU:                dict[int, Tensor]   # [E, K] fp32
_RESERVOIR_GPU:            dict[int, Tensor]   # [E, cap, H] fp32
_RESERVOIR_CTR_GPU:        dict[int, Tensor]   # [E] int64 (monotonic per-expert)
_BLOCK_OUT_GPU:            dict[int, Tensor]   # [buf_rows, H] bf16 (monotonic)
_BLOCK_OUT_PTR_GPU:        dict[int, Tensor]   # [] int64 scalar write pointer
_TOPK_WEIGHTS_STASH_GPU:   dict[int, Tensor]   # [buf_rows, top_k] fp32
_TOPK_IDS_STASH_GPU:       dict[int, Tensor]   # [buf_rows, top_k] int64
```

Keep `_current_layer_idx: int = -1` and the flag globals (`_CAPTURE_ROUTER`, etc.) as before.

Remove `_callbacks` dict, `dispatch()`, and `register_callback()` entirely. Instead, export the GPU tensor dicts directly so each writer module can hold references allocated during `setup()`.

Add `_CUSTOM_OP_LIB` module-level `torch.library.Library("vllm_calib", "FRAGMENT")` for traced-region op registration.

### 1.2 `_calib_buf_rows` fix (H2)

In `TritonExperts.__init__` (triton_moe.py), replace:

```python
self._calib_buf_rows = vllm_config.compilation_config.max_cudagraph_capture_size
```

With:

```python
from vllm import envs
_ccs = vllm_config.compilation_config.max_cudagraph_capture_size or 0
_mbt = getattr(vllm_config.scheduler_config, "max_num_batched_tokens", 0)
self._calib_buf_rows = max(_ccs, _mbt, 512)
```

Also add at the OOB skip site (inside `TritonExperts.apply`):

```python
if num_tokens > self._calib_buf_rows:
    _warn_once(f"calib: buf_rows={self._calib_buf_rows} < num_tokens={num_tokens}; "
               "prefill skipped for expert_out_unweighted/mid")
    # skip the dispatch block
```

Use a module-level `_WARNED_OOB: set[int]` keyed by layer id to make this warn-once.

### 1.3 `_current_layer_idx` guard broadening (M4)

In `moe_runner.py`, the guard currently reads:

```python
if _CAPTURE_EXPERT_UNWEIGHTED or _CAPTURE_EXPERT_MID:
    _ch._current_layer_idx = layer.moe_layer_id
```

Expand to:

```python
if any([_ch._CAPTURE_EXPERT_UNWEIGHTED, _ch._CAPTURE_EXPERT_MID,
        _ch._CAPTURE_IMATRIX, _ch._CAPTURE_OUTPUT_RESERVOIR,
        _ch._CAPTURE_PER_EXPERT_MAX, _ch._CAPTURE_WANDA]):
    _ch._current_layer_idx = layer.moe_layer_id
```

This ensures `_current_layer_idx` is set before any TritonExperts-interior writer can read it.

### 1.4 New env vars (M3)

Add to `vllm/envs.py` (in the existing calib block):

```python
VLLM_CALIB_INPUT_COV_MODE: str = "off"       # "off" | "resident" | "offload"
VLLM_CALIB_BUF_ROWS_FLOOR: int = 512          # minimum _calib_buf_rows
VLLM_CALIB_STAGE2_PROFILE_MODE: str = "replay" # "replay" | "live" (see H3)
```

The existing vars remain. `VLLM_CALIB_CAPTURE_INPUT_COV` is repurposed: non-empty value = mode string ("resident"/"offload"); empty = off.

### 1.5 Custom-op registration module

Create `vllm/calibration_custom_ops.py`. This module registers all traced-region accumulation ops using `direct_register_custom_op` from `vllm.utils.torch_utils`. It must be imported at module level from each traced-region signal's caller (or from `calibration_hooks.py` which is imported by all callers).

Registration pattern (identical to moe_runner.py:167–173):

```python
from vllm.utils.torch_utils import direct_register_custom_op, vllm_lib

def _calib_dense_imatrix_accum(x: torch.Tensor, layer_name: str) -> None:
    # x: [T, H] — input to a dense linear layer
    # mutates global _IMATRIX_DENSE_GPU[layer_name]
    if layer_name not in _ch._IMATRIX_DENSE_GPU:
        return
    acc = _ch._IMATRIX_DENSE_GPU[layer_name]
    x_sq = (x.float() ** 2).sum(dim=0)   # [H]
    acc.add_(x_sq)

def _calib_dense_imatrix_accum_fake(x: torch.Tensor, layer_name: str) -> None:
    return

direct_register_custom_op(
    op_name="calib_dense_imatrix_accum",
    op_func=_calib_dense_imatrix_accum,
    mutates_args=["x"],   # x not mutated but Dynamo must not DCE the call
    fake_impl=_calib_dense_imatrix_accum_fake,
)
```

`mutates_args=["x"]` is a convention trick to prevent Dynamo from eliminating the call as dead. Actually `mutates_args` should be the name of the first persistent accumulator argument. Since accumulators are module-level dicts (not function args), use `mutates_args=[]` but tag with `torch.Tag.needs_fixed_stride_order` so Dynamo cannot reorder or eliminate it. The custom op is opaque — its body runs in the dispatch path, which is not traced.

Similarly register:

```python
_calib_block_out_accum(hidden: Tensor, layer_idx: int) -> None
_calib_layer_in_accum(hidden: Tensor, layer_idx: int) -> None
```

All three ops: `mutates_args=[]`, `tags=(torch.Tag.needs_fixed_stride_order,)`.

**Insertion points for traced-region callers:**

- `_calib_dense_imatrix_accum`: call from `ColumnParallelLinear.forward` (linear.py:392) after `output = self.quant_method.apply(...)` — pass `x` (the input tensor before the linear) and `self.layer_name`. Also call from `LogitsProcessor._get_logits` (logits_processor.py:96) after `lm_head.quant_method.apply(...)` — pass `hidden_states` and `"lm_head"`.
- `_calib_block_out_accum`: call from `Qwen3NextSparseMoeBlock.forward` (qwen3_next.py:202) and `Qwen3MoeSparseMoeBlock.forward` (qwen3_moe.py:258), after `return final_hidden_states.view(orig_shape)` — insert before the return, passing `final_hidden_states` and `self.moe_layer_id`.
- `_calib_layer_in_accum`: call from decoder layer `forward` in both models (qwen3_next.py:401 / qwen3_moe.py:416), at the top of the layer forward before the attention call, passing `hidden_states` and `self.layer_idx`.

**Gate pattern for all three custom ops:** Guard the call with `if _ch._CAPTURE_IMATRIX:` / `if _ch._CAPTURE_BLOCK:` / `if _ch._CAPTURE_LAYER_IN:`. These are Python bool checks read at trace time — when False, Dynamo folds the call away entirely (the guard is a constant). When True, the custom op call is recorded into the graph and re-executes on every replay.

---

## 2. Per-Writer Implementation

### 2.1 `calibration_reap_scores.py` — custom-op-interior

**GPU accumulators** (allocated in `setup()`, before CUDA graph capture):

```python
_REAP_SCORE_ACCUM_GPU: dict[int, Tensor]   # layer_idx -> [n_experts] fp32, zero-init
_REAP_TOKEN_COUNTS_GPU: dict[int, Tensor]  # layer_idx -> [n_experts] int64, zero-init
_TOPK_WEIGHTS_STASH_GPU: dict[int, Tensor] # layer_idx -> [buf_rows, top_k] fp32
```

**In-graph accumulation** (replacing `_on_router` + `_on_expert_out_unweighted` CPU callbacks):

In `moe_runner.py _apply_quant_method`, after `topk_weights, topk_ids = self.router.select_experts(...)` (line 530), add inline GPU ops:

```python
if _ch._CAPTURE_ROUTER:
    layer_idx = _ch._current_layer_idx  # Python int, constant at capture time
    stash = _ch._TOPK_WEIGHTS_STASH_GPU[layer_idx]
    stash[:topk_weights.shape[0]].copy_(topk_weights)  # graph-recorded copy_
```

In `TritonExperts.apply`, after `invoke_fused_moe_triton_kernel` returns, the `_unweighted_slice` is `self._calib_unweighted_buf[:num_tokens, :, :]`. Add:

```python
if _ch._CAPTURE_EXPERT and _ch._CAPTURE_ROUTER:
    layer_idx = _ch._current_layer_idx
    scores = _ch._REAP_SCORE_ACCUM_GPU[layer_idx]    # [E]
    counts = _ch._REAP_TOKEN_COUNTS_GPU[layer_idx]    # [E]
    # topk_ids shape: [num_tokens, top_k]; unweighted_slice: [num_tokens, top_k, K]
    # norm per (token, expert): [num_tokens, top_k]
    norms = _unweighted_slice.float().norm(dim=-1)     # [T, K_top]
    stash = _ch._TOPK_WEIGHTS_STASH_GPU[layer_idx][:num_tokens]  # [T, K_top]
    weighted_norms = norms * stash                      # [T, K_top] graph-safe multiply
    flat_ids  = topk_ids.reshape(-1).long()             # [T*K_top]
    flat_vals = weighted_norms.reshape(-1)              # [T*K_top]
    scores.scatter_add_(0, flat_ids, flat_vals)
    ones_buf = _ch._ONES_BUF_GPU[layer_idx][:flat_ids.shape[0]]  # [T*K_top] ones, prealloc
    counts.scatter_add_(0, flat_ids, ones_buf)
```

`_ONES_BUF_GPU[layer_idx]` is a `[buf_rows * top_k]` int64 tensor of ones allocated once in `setup()`. Slice `[:flat_ids.shape[0]]` — `flat_ids.shape[0]` is a constant at graph capture time (captured batch size × top_k).

**`captured_entry_count()`** (H1):

```python
def captured_entry_count() -> bool:
    if not _REAP_TOKEN_COUNTS_GPU:
        return False
    return int(next(iter(_REAP_TOKEN_COUNTS_GPU.values())).sum().item()) > 0
```

One `.item()` outside graph, at dump time only.

**`dump()`**: copy GPU tensors to CPU at dump time (outside graph):

```python
for layer_idx, scores_gpu in _REAP_SCORE_ACCUM_GPU.items():
    counts_cpu = _REAP_TOKEN_COUNTS_GPU[layer_idx].cpu().float()
    scores_cpu = scores_gpu.cpu()
    safe_counts = counts_cpu.clamp(min=1.0)
    normalized = scores_cpu / safe_counts
    # serialize to sidecar as before
```

Remove all `_ROUTER_WEIGHTS_STASH` CPU stash, `_on_router`, `_on_expert_out_unweighted` callbacks, `register_callback` calls.

### 2.2 `calibration_per_expert_max.py` — custom-op-interior

**GPU accumulators:**

```python
_PEM_ACCUM_GPU: dict[int, Tensor]  # [n_experts, K] fp32, -inf init
```

**In-graph accumulation** (in `TritonExperts.apply`):

```python
if _ch._CAPTURE_PER_EXPERT_MAX:
    layer_idx = _ch._current_layer_idx
    acc = _ch._PEM_ACCUM_GPU[layer_idx]     # [E, K]
    # _unweighted_slice: [T, top_k, K] — abs-max per (expert, channel)
    abs_vals = _unweighted_slice.float().abs()  # [T, top_k, K]
    flat_ids  = topk_ids.reshape(-1).long()     # [T*top_k]
    flat_vals = abs_vals.reshape(-1, abs_vals.shape[-1])  # [T*top_k, K]
    acc.scatter_reduce_(0, flat_ids.unsqueeze(1).expand_as(flat_vals),
                        flat_vals, reduce="amax", include_self=True)
```

**`captured_entry_count()`**: `int((_PEM_ACCUM_GPU[0] > -1e37).any().item()) > 0` — checks if any cell moved off -inf.

**`dump()`**: `acc.cpu()` per layer at dump time.

### 2.3 `calibration_routing_stats.py` — custom-op-interior

**GPU accumulators:**

```python
_ROUTING_FREQ_GPU:  dict[int, Tensor]  # [E] int64, zero-init
_ROUTING_WSUM_GPU:  dict[int, Tensor]  # [E] fp32, zero-init
```

**In-graph accumulation** (in `moe_runner.py _apply_quant_method`, same location as reap topk stash):

```python
if _ch._CAPTURE_ROUTING_STATS:
    layer_idx = _ch._current_layer_idx
    freq  = _ch._ROUTING_FREQ_GPU[layer_idx]   # [E]
    wsum  = _ch._ROUTING_WSUM_GPU[layer_idx]   # [E]
    flat_ids = topk_ids.reshape(-1).long()
    flat_w   = topk_weights.reshape(-1)
    ones_i64 = _ch._ONES_I64_BUF_GPU[layer_idx][:flat_ids.shape[0]]
    freq.scatter_add_(0, flat_ids, ones_i64)
    wsum.scatter_add_(0, flat_ids, flat_w)
```

**`dump()`**: both tensors `.cpu()` at dump time.

### 2.4 `calibration_router_logits_stats.py` — custom-op-interior

**GPU accumulators:**

```python
_RLOGITS_SUM_GPU:   dict[int, Tensor]  # [E] fp32, zero-init
_RLOGITS_SQ_GPU:    dict[int, Tensor]  # [E] fp32, zero-init
_RLOGITS_COUNT_GPU: dict[int, Tensor]  # [E] int64, zero-init
```

**In-graph accumulation**: Computed from `router_logits` in `moe_runner.py _apply_quant_method` before `select_experts` call (router_logits is available as a tensor argument):

```python
if _ch._CAPTURE_ROUTER_LOGITS_STATS:
    layer_idx = _ch._current_layer_idx
    # router_logits: [T, E]
    # BOS-path narrowing: when T==1 this is a single-token decode step
    # (documented LOW item: BOS at position 0 has sink logit behavior;
    # no special-case needed — uniform accumulation is correct for stats)
    lsum  = _ch._RLOGITS_SUM_GPU[layer_idx]    # [E]
    lsq   = _ch._RLOGITS_SQ_GPU[layer_idx]     # [E]
    lcnt  = _ch._RLOGITS_COUNT_GPU[layer_idx]  # [E]
    rl    = router_logits.float()               # [T, E]
    lsum.add_(rl.sum(dim=0))                    # graph-safe: reduce over T
    lsq.add_((rl ** 2).sum(dim=0))
    t_count = _ch._SCALAR_ONES_GPU[layer_idx]  # [] int64 = 1, prealloc
    lcnt.add_(t_count * rl.shape[0])            # constant * runtime int
```

Note: `rl.shape[0]` is a constant at capture time (the batch size slot). The multiply `t_count * rl.shape[0]` is graph-safe.

**`captured_entry_count()`**: `_RLOGITS_COUNT_GPU[0].sum().item() > 0`.

### 2.5 `calibration_imatrix.py` — split: MoE path (interior) + dense path (traced-region)

**MoE path (custom-op-interior):** `expert_in` signal comes from `moe_runner.py _apply_quant_method` before `_quant_method.apply(...)` — already inside the custom op. Replace the Python loop with:

```python
if _ch._CAPTURE_IMATRIX:
    layer_idx = _ch._current_layer_idx
    acc = _ch._IMATRIX_MoE_GPU[layer_idx]   # [E, H]
    # hidden_states: [T, H], topk_ids: [T, top_k]
    x_sq = hidden_states.float() ** 2        # [T, H]
    flat_ids = topk_ids[:, 0].long()         # [T] — use first expert for routing (top-1 surrogate)
    # For correct per-expert accumulation: scatter x_sq for each (token, expert_slot)
    # Expand for all top_k slots:
    for k in range(topk_ids.shape[1]):       # loop over top_k (small constant, unrolled by Dynamo)
        ids_k = topk_ids[:, k].long()        # [T]
        acc.scatter_add_(0, ids_k.unsqueeze(1).expand(-1, acc.shape[1]), x_sq)
```

`topk_ids.shape[1]` is top_k = small constant (2 or 4) — Dynamo unrolls this loop at compile time, so it becomes a sequence of static `scatter_add_` calls in the graph.

**Dense path (traced-region):** The `_calib_dense_imatrix_accum` custom op (Section 1.5) handles `linear_in` and `lm_head_in`. In `calibration_imatrix.py`:

```python
_IMATRIX_DENSE_GPU: dict[str, Tensor]  # layer_name -> [H] fp32
```

The custom op body reads `_ch._IMATRIX_DENSE_GPU[layer_name]` directly. `setup()` pre-populates the dict with zero-tensors for all layer names that will be captured.

**`captured_entry_count()`**: 
```python
# MoE path
moe_ok = any(v.sum().item() > 0 for v in _ch._IMATRIX_MoE_GPU.values())
# dense path  
dense_ok = any(v.sum().item() > 0 for v in _ch._IMATRIX_DENSE_GPU.values())
return moe_ok or dense_ok
```

### 2.6 `calibration_wanda_scalar_row.py` — custom-op-interior

**GPU accumulators:**

```python
_WANDA_GPU: dict[int, Tensor]  # [E, H] fp32, zero-init
```

**In-graph accumulation** (in `moe_runner.py _apply_quant_method`):

```python
if _ch._CAPTURE_WANDA:
    layer_idx = _ch._current_layer_idx
    acc = _ch._WANDA_GPU[layer_idx]     # [E, H]
    # x: hidden_states [T, H]; g: gate output [T, top_k] (topk_weights)
    # wanda metric: (x_sq * g_sq) per channel, accumulated per expert
    x_sq = (hidden_states.float() ** 2)     # [T, H]
    for k in range(topk_ids.shape[1]):
        ids_k = topk_ids[:, k].long()       # [T]
        w_k   = topk_weights[:, k].float().unsqueeze(1) ** 2  # [T, 1]
        contrib = x_sq * w_k                # [T, H]
        acc.scatter_add_(0, ids_k.unsqueeze(1).expand(-1, acc.shape[1]), contrib)
```

### 2.7 `calibration_output_reservoir.py` — custom-op-interior (C3 fix)

**Design: Deterministic fixed-stride per-expert reservoir (no RNG).**

Replace Vitter-R entirely. The algorithm: for expert `e`, token slot `s` (0-indexed within this dispatch, routing to expert `e`), write to reservoir slot `(ctr_e + s) % cap` where `ctr_e` is the per-expert monotonic token counter. After writing, advance `ctr_e += N_e` where `N_e` = number of tokens routed to expert `e` this dispatch. This gives a deterministic sliding window with stride 1 per token, which is uniform coverage with no intra-dispatch collision (each token writes to a distinct slot `mod cap` because `s` is the within-dispatch local index).

**GPU accumulators:**

```python
_RESERVOIR_GPU:     dict[int, Tensor]  # [E, cap, H] fp32
_RESERVOIR_CTR_GPU: dict[int, Tensor]  # [E] int64 monotonic counter, zero-init
```

`cap` = `VLLM_CALIB_OUTPUT_RESERVOIR_CAP` (default 256).

**In-graph accumulation** (in `TritonExperts.apply`):

```python
if _ch._CAPTURE_OUTPUT_RESERVOIR:
    layer_idx = _ch._current_layer_idx
    res   = _ch._RESERVOIR_GPU[layer_idx]    # [E, cap, H]
    ctrs  = _ch._RESERVOIR_CTR_GPU[layer_idx]  # [E]
    cap   = res.shape[1]
    H     = res.shape[2]
    # _unweighted_slice: [T, top_k, H] (expert outputs for each slot)
    # topk_ids: [T, top_k]
    # For each expert slot k:
    for k in range(topk_ids.shape[1]):  # unrolled by Dynamo
        ids_k = topk_ids[:, k].long()    # [T]
        vals_k = _unweighted_slice[:, k, :].float()  # [T, H]
        # Assign write slots: within-dispatch local rank for each token per expert
        # Build a [T] tensor of within-expert local indices via scatter
        # local_rank[t] = (number of tokens before t that route to same expert as t in slot k)
        # This requires a rank-within-group — not graph-safe with Python loops.
        # CHOSEN APPROACH: compute slot = (ctr_e + within_expert_rank) % cap
        # Compute within_expert_rank via cumsum on indicator per expert:
        E = ctrs.shape[0]
        indicator = torch.zeros(T, E, dtype=torch.int64, device=ids_k.device)
        indicator.scatter_(1, ids_k.unsqueeze(1), 1)   # [T, E] one-hot
        # cumsum along T gives within-dispatch rank (0-based prefix):
        local_rank = indicator.cumsum(dim=0).gather(1, ids_k.unsqueeze(1)).squeeze(1) - 1  # [T]
        base_ctr = ctrs.gather(0, ids_k)  # [T] — each token's expert's current ctr
        write_slot = (base_ctr + local_rank) % cap    # [T]
        # Write: res[ids_k[t], write_slot[t], :] = vals_k[t, :]
        # Flatten to scatter_: linear index = ids_k * cap * H + write_slot * H + ch
        lin_idx = (ids_k * cap + write_slot).unsqueeze(1).expand(-1, H)  # [T, H]
        # res reshaped to [E*cap, H]:
        res_flat = res.view(-1, H)
        res_flat.scatter_(0, lin_idx, vals_k)
        # Advance counters: add N_e to ctr for each expert
        delta = indicator.sum(dim=0)  # [E] number of tokens routed to each expert
        ctrs.add_(delta)
```

This is fully graph-safe: `cumsum`, `gather`, `scatter_`, `add_` are all CUDA ops. `topk_ids.shape[1]` (top_k) is constant at capture time so the `for k` loop is unrolled by Dynamo into a static sequence.

**`captured_entry_count()`**: `_RESERVOIR_CTR_GPU[0].sum().item() > 0`.

**`dump()`**: `_RESERVOIR_GPU[layer_idx][:, :min(ctr, cap), :]` for each layer, `.cpu()`.

**Old CPU fields to remove**: `_RESERVOIR`, per-cell `torch.Generator`, phase-1/2 fill/sample logic.

### 2.8 `calibration_block_outputs.py` — traced-region (C2 fix)

**Design: Pre-sized monotonic write-pointer buffer.**

```python
_BLOCK_OUT_GPU:     dict[int, Tensor]  # [subset * max_seq_len, H] bf16
_BLOCK_OUT_PTR_GPU: dict[int, Tensor]  # [] int64 scalar (write pointer per layer)
```

`subset = VLLM_CALIB_BLOCK_OUTPUTS_SUBSET_SIZE` (default 128). `max_seq_len` = model config max sequence length. Allocated in `setup()` per MoE layer index.

**In-graph accumulation** (via `_calib_block_out_accum` custom op, Section 1.5):

```python
def _calib_block_out_accum(hidden: torch.Tensor, layer_idx: int) -> None:
    if not _ch._CAPTURE_BLOCK:
        return
    if layer_idx not in _ch._BLOCK_OUT_GPU:
        return
    buf = _ch._BLOCK_OUT_GPU[layer_idx]   # [cap, H]
    ptr = _ch._BLOCK_OUT_PTR_GPU[layer_idx]  # [] int64
    T = hidden.shape[0]
    cap = buf.shape[0]
    # Only write if space remains (monotonic, no wrap)
    # This check is a runtime branch — graph-safe because ptr is a GPU tensor
    # and the condition is computed on GPU:
    space = (cap - ptr).clamp(min=0)
    write_T = space.clamp(max=T).item()  # .item() here is ONE read at graph boundary
    # ALTERNATIVE: use torch.where to avoid .item():
    write_T_gpu = torch.minimum(space, torch.tensor(T, device=space.device))
    buf_slice = buf[ptr:ptr+write_T_gpu]  # dynamic slice — NOT graph-safe with tensor index
```

`ptr:ptr+write_T_gpu` with a tensor index is NOT graph-safe (dynamic shape). Use `torch.ops.aten.slice_scatter` or a static upper bound.

**Revised approach:** Since buf size is `subset * max_seq_len` and we only write once per prompt (at inference time), the write pointer advances by exactly `T` tokens per dispatch. Cap is never exceeded during a correctly sized calibration run. If it is exceeded, the OOB tokens are silently dropped. This can be implemented with a static-shape conditional:

```python
def _calib_block_out_accum(hidden: torch.Tensor, layer_idx: int) -> None:
    buf = _ch._BLOCK_OUT_GPU[layer_idx]     # [cap, H] bf16
    ptr_tensor = _ch._BLOCK_OUT_PTR_GPU[layer_idx]  # [] int64
    T = hidden.shape[0]    # constant at capture time
    cap = buf.shape[0]     # constant at capture time
    ptr = int(ptr_tensor.item())  # read pointer value — ONE .item() per forward pass
    end = min(ptr + T, cap)
    write_n = end - ptr
    if write_n > 0:
        buf[ptr:end].copy_(hidden[:write_n].to(buf.dtype))
    ptr_tensor.fill_(end)
```

The `.item()` call makes this NOT fully in-graph. However, `_calib_block_out_accum` is a `direct_register_custom_op` — its body runs eagerly (not traced by Dynamo). The CUDA graph does NOT capture inside custom op bodies; it records the custom op call as an opaque dispatch. Therefore `.item()` inside the custom op body is fine: it runs outside the graph's recorded CUDA stream.

Wait — this is the critical subtlety. A `direct_register_custom_op` op IS recorded into the CUDA graph only if its implementation issues CUDA kernels. Python-level `.item()` inside a custom op body would cause a `cudaStreamSynchronize` during capture, which is illegal.

**Correct approach:** Use a pure GPU conditional with `torch.clamp` and `scatter_`:

```python
def _calib_block_out_accum(hidden: torch.Tensor, layer_idx: int) -> None:
    buf = _ch._BLOCK_OUT_GPU[layer_idx]     # [cap, H] bf16
    ptr_tensor = _ch._BLOCK_OUT_PTR_GPU[layer_idx]  # [] int64
    T = hidden.shape[0]    # constant at capture time (captured batch size)
    # Write range: [ptr, ptr+T) clamped to [0, cap)
    # Compute write indices for each of T rows: ptr + arange(T), clamped to cap
    arange = _ch._ARANGE_BUF_GPU[layer_idx][:T]  # prealloc [buf_rows] arange, slice to T
    write_idx = (ptr_tensor + arange).clamp(max=buf.shape[0] - 1)  # [T]
    # Mask out tokens that fall outside cap (write_idx == cap-1 may collide — use valid_mask)
    valid_mask = (ptr_tensor + arange) < buf.shape[0]  # [T] bool
    # Only scatter valid tokens:
    valid_hidden = hidden * valid_mask.unsqueeze(1).to(hidden.dtype)  # zero out invalid
    buf.scatter_(0, write_idx.unsqueeze(1).expand(-1, buf.shape[1]),
                 valid_hidden.to(buf.dtype))
    # Advance pointer by T (clamped to cap):
    new_ptr = (ptr_tensor + T).clamp(max=buf.shape[0])
    ptr_tensor.copy_(new_ptr)
```

All ops are pure CUDA: `clamp`, `scatter_`, `copy_`. Graph-safe. The overflow case writes `hidden * 0` (zero) into the last slot — a minor corruption of the last slot when over-full, acceptable since `setup()` is sized to never overflow under normal use.

**`captured_entry_count()`**: `_BLOCK_OUT_PTR_GPU[0].item() > 0` at dump time.

**`dump()`**:

```python
for layer_idx, buf in _BLOCK_OUT_GPU.items():
    valid = int(_BLOCK_OUT_PTR_GPU[layer_idx].item())
    data = buf[:valid].cpu()
    # serialize using existing BlockHiddenPayload schema
    payload = BlockHiddenPayload(hidden=data, n_tokens=valid)
    save_sidecar(layer_idx, payload)
```

`BlockHiddenPayload` schema unchanged: `hidden: [n_tokens, H]`, `n_tokens: int`.

### 2.9 `calibration_input_cov.py` — flag-gated dual-mode (C1 + M2)

Mode controlled by `VLLM_CALIB_INPUT_COV_MODE` (M3 env var):

**Mode "off"**: no-op, no allocation.

**Mode "resident"**: GPU accumulators hold the full `[E, H, H]` covariance matrix.

**Mode "offload"**: GPU holds only a `[buf_rows, H]` input buffer per expert per layer; after each forward, a CPU thread does `bmm` on CPU.

**Resident-mode peak memory analysis (M2):**

For Qwen3-30B (n_experts=128, H=2048, top_k=8, n_moe_layers=48):
- Accumulator: 128 × 2048 × 2048 × 4 bytes × 48 layers = 128 × 16M × 48 = 98,304 MB ≈ 96 GB per TP rank
- With TP=4 sharding H → H/4=512 per rank: 128 × 512 × 512 × 4 × 48 = 128 × 1M × 48 ≈ 6 GB per rank — feasible
- Per-forward `[E, N, H/4]` temp (E=128, N=buf_rows=512, H/4=512): 128 × 512 × 512 × 2 = 67 MB (bf16)
- `bmm [E, H/4, H/4]`: materially same as accumulator slice, no second allocation (using `baddbmm_` in-place into accumulator)
- **Confirmed:** `bmm` with `out=accumulator` does NOT materialize a second `[E, H, H]` — PyTorch `baddbmm_` mutates in place.
- **Full-H unsharded** (if TP=1): 96 GB just for covs on top of 60 GB model weights — not viable on a single 80 GB H100 without off-loading. TP=8 required, or use offload mode.

**GPU accumulators (resident mode):**

```python
_INPUT_COV_GPU: dict[(int,int), Tensor]  # (layer_idx, expert_idx) -> [H_local, H_local] fp32
```

**In-graph accumulation** (in `moe_runner.py`, inside custom op boundary):

```python
if _ch._CAPTURE_INPUT_COV and _ch._INPUT_COV_MODE == "resident":
    layer_idx = _ch._current_layer_idx
    x = hidden_states.float()       # [T, H]
    for k in range(topk_ids.shape[1]):
        ids_k = topk_ids[:, k]      # [T]
        for e in range(n_experts):  # NOT graph-safe: Python loop over E
```

This Python loop over experts is not acceptable for large E. Use batched gather + `baddbmm_`:

```python
if _ch._CAPTURE_INPUT_COV and _ch._INPUT_COV_MODE == "resident":
    layer_idx = _ch._current_layer_idx
    cov_stack = _ch._INPUT_COV_STACK_GPU[layer_idx]  # [E, H, H] fp32
    x = hidden_states.float()   # [T, H]
    # For each top_k slot:
    for k in range(topk_ids.shape[1]):  # Dynamo unrolls (k is small constant)
        ids_k = topk_ids[:, k].long()   # [T]
        # Gather x per expert: x_gathered[e] = rows of x where ids_k==e
        # Use scatter to build [E, buf_rows, H] batched input tensor:
        x_batched = _ch._INPUT_COV_TEMP_GPU[layer_idx]  # [E, buf_rows, H] fp32, zero each fwd
        x_batched.zero_()
        # local_rank per expert (same construction as reservoir):
        E = cov_stack.shape[0]
        indicator = torch.zeros(x.shape[0], E, dtype=torch.float32, device=x.device)
        indicator.scatter_(1, ids_k.unsqueeze(1), 1.0)
        local_rank = indicator.cumsum(dim=0).gather(1, ids_k.unsqueeze(1)).squeeze(1).long() - 1
        dest_e  = ids_k               # [T] which expert
        dest_r  = local_rank          # [T] which row within expert
        lin_idx = (dest_e * x_batched.shape[1] + dest_r)  # [T]
        x_batched.view(-1, x.shape[1]).scatter_(0, lin_idx.unsqueeze(1).expand(-1, x.shape[1]), x)
        # baddbmm_ in place: cov_stack += x_batched.T @ x_batched per expert
        cov_stack.baddbmm_(x_batched.transpose(1, 2), x_batched)  # [E, H, H] += [E, H, buf] @ [E, buf, H]
```

**Offload mode**: instead of `baddbmm_` GPU, after filling `x_batched`, enqueue a CPU task: `pool.submit(cpu_cov_update, x_batched.cpu(), layer_idx)`. This offload happens inside the custom op body (not in graph), so the async submit is fine. The GPU graph only records the `x_batched.zero_()` + scatter fill.

**H3 / stage2_profile:** See Section 3 below.

### 2.10 `calibration_stage2_profile.py` — routed to forward-only replay (H3)

**Locating the post-step hook (H3):** `execute_model` in `vllm/v1/worker/gpu/model_runner.py` returns at line 1183 with `return None` (last PP rank returns None; `sample_tokens` is a separate call at line 1185). There is no `post_model_hook` or step-level callback surface in the current vLLM v1 worker path. The model runner stores `execute_model_state` (line 1169) but that is only for bridging to `sample_tokens`, not for user hooks.

**Decision:** Stage2_profile and input_cov-offload mode's aggregation step are routed via the `feat/calib-v3-replay` forward-only replay driver. The live vLLM capture path records only the raw tensor data (GPU accumulators). Post-run, the replay driver loads the checkpoint and re-executes `ReamCostAccumulator` + `InputCovarianceAccumulator` from `moe_compress.stage2.profiling` against the captured outputs.

`calibration_stage2_profile.py` in live mode: only registers setup()/dump() stubs; the 5 callbacks (`router`, `expert_out_unweighted`, `layer_in`, `expert_in`, `expert_mid`) are NOT registered during live vLLM capture (they remain CPU-callback-based and are unused under cudagraph).

The replay driver will call them after loading the dumped GPU tensor checkpoints. This is explicitly marked in the `VLLM_CALIB_STAGE2_PROFILE_MODE` env var ("replay" = default).

---

## 3. `moe_runner.py` Patch Insertion Points

All inline GPU accumulation for custom-op-interior writers is inserted in `_apply_quant_method` (moe_runner.py:496) and in `TritonExperts.apply` (triton_moe.py).

**`_apply_quant_method` additions** (after line 530, after `topk_weights, topk_ids = ...`):

```python
# === CALIB: in-graph GPU accumulation (graph-safe: inside vllm.moe_forward custom op) ===
if any([_ch._CAPTURE_ROUTER, _ch._CAPTURE_ROUTING_STATS,
        _ch._CAPTURE_ROUTER_LOGITS_STATS, _ch._CAPTURE_WANDA,
        _ch._CAPTURE_INPUT_COV, _ch._CAPTURE_OUTPUT_RESERVOIR,
        _ch._CAPTURE_IMATRIX, _ch._CAPTURE_PER_EXPERT_MAX]):
    _layer_idx = _ch._current_layer_idx  # int constant at capture time
    if _ch._CAPTURE_ROUTER:
        _ch._TOPK_WEIGHTS_STASH_GPU[_layer_idx][:topk_weights.shape[0]].copy_(topk_weights)
    if _ch._CAPTURE_ROUTING_STATS:
        # routing freq + wsum (Section 2.3)
        ...
    if _ch._CAPTURE_ROUTER_LOGITS_STATS:
        # router_logits mean/var (Section 2.4)
        ...
    if _ch._CAPTURE_WANDA:
        # wanda metric (Section 2.6)
        ...
    if _ch._CAPTURE_IMATRIX:
        # MoE imatrix (Section 2.5 MoE path)
        ...
    if _ch._CAPTURE_INPUT_COV:
        # input_cov resident/offload (Section 2.9)
        ...
# === END CALIB ===
```

The outer `if any([...])` guard is folded to a constant by Dynamo (all `_CAPTURE_*` flags are module-level booleans read at trace time). When all flags are False, the entire block is DCE'd.

**`TritonExperts.apply` additions** (after `invoke_fused_moe_triton_kernel`, before return):

```python
if self._calib_buf_rows > 0 and num_tokens <= self._calib_buf_rows:
    _unweighted_slice = self._calib_unweighted_buf[:num_tokens]  # [T, top_k, K]
    if _ch._CAPTURE_EXPERT and _ch._CAPTURE_ROUTER:
        # reap_scores (Section 2.1)
        ...
    if _ch._CAPTURE_PER_EXPERT_MAX:
        # per_expert_max (Section 2.2)
        ...
    if _ch._CAPTURE_OUTPUT_RESERVOIR:
        # output_reservoir (Section 2.7)
        ...
else:
    _warn_once(self.moe_layer_id, num_tokens, self._calib_buf_rows)
```

---

## 4. `vllm/envs.py` Hunk (M3)

Add to the existing calib env var block (after `VLLM_CALIB_BLOCK_OUTPUTS_SUBSET_SIZE`):

```python
VLLM_CALIB_INPUT_COV_MODE: str = os.getenv("VLLM_CALIB_INPUT_COV_MODE", "off")
# "off" | "resident" | "offload"
VLLM_CALIB_BUF_ROWS_FLOOR: int = int(os.getenv("VLLM_CALIB_BUF_ROWS_FLOOR", "512"))
VLLM_CALIB_STAGE2_PROFILE_MODE: str = os.getenv("VLLM_CALIB_STAGE2_PROFILE_MODE", "replay")
# "replay" = use forward-only replay driver post-capture
# "live"   = register callbacks during capture (only works eager/non-compiled)
```

---

## 5. Implementation File Map

**Files to create:**
- `/vllm/calibration_custom_ops.py` — `direct_register_custom_op` registrations for `calib_dense_imatrix_accum`, `calib_block_out_accum`, `calib_layer_in_accum`; exposes `torch.ops.vllm.calib_*` for use by traced-region callers

**Files to modify:**
- `/vllm/calibration_hooks.py` — remove `_callbacks`/`dispatch`/`register_callback`; add GPU accumulator dicts; add `_ARANGE_BUF_GPU`, `_ONES_BUF_GPU`, `_ONES_I64_BUF_GPU`, `_SCALAR_ONES_GPU`; add `_current_layer_idx` guard broadening
- `/vllm/calibration_reap_scores.py` — full rewrite to in-graph GPU (Section 2.1)
- `/vllm/calibration_per_expert_max.py` — full rewrite to in-graph GPU (Section 2.2)
- `/vllm/calibration_routing_stats.py` — full rewrite (Section 2.3)
- `/vllm/calibration_router_logits_stats.py` — full rewrite (Section 2.4)
- `/vllm/calibration_imatrix.py` — MoE path rewrite + dense path via custom op (Section 2.5)
- `/vllm/calibration_wanda_scalar_row.py` — full rewrite (Section 2.6)
- `/vllm/calibration_output_reservoir.py` — full rewrite, deterministic fixed-stride (Section 2.7)
- `/vllm/calibration_block_outputs.py` — monotonic write-pointer + custom op receiver (Section 2.8)
- `/vllm/calibration_input_cov.py` — flag-gated dual-mode (Section 2.9)
- `/vllm/calibration_stage2_profile.py` — stub setup/dump; disable live callbacks; add replay-mode note (Section 2.10)
- `/vllm/model_executor/layers/fused_moe/runner/moe_runner.py` — inline GPU accum in `_apply_quant_method`; broadened `_current_layer_idx` guard (M4); import `calibration_custom_ops`
- `/vllm/model_executor/layers/fused_moe/experts/triton_moe.py` — `_calib_buf_rows` fix (H2); OOB warn-once; inline GPU accum in `apply()`
- `/vllm/model_executor/layers/linear.py` — insert `torch.ops.vllm.calib_dense_imatrix_accum(input_, self._calib_layer_name)` in `ColumnParallelLinear.forward` (line 582), guarded by `if _ch._CAPTURE_IMATRIX`
- `/vllm/model_executor/layers/logits_processor.py` — insert `torch.ops.vllm.calib_dense_imatrix_accum(hidden_states, "lm_head")` in `_get_logits` (line 96), guarded by `if _ch._CAPTURE_IMATRIX`
- `/vllm/model_executor/models/qwen3_moe.py` — insert `torch.ops.vllm.calib_block_out_accum(final_hidden_states, self.moe_layer_id)` in `Qwen3MoeSparseMoeBlock.forward` (line 258); insert `torch.ops.vllm.calib_layer_in_accum(hidden_states, self.layer_idx)` in decoder layer forward (line 416), guarded by respective flags
- `/vllm/model_executor/models/qwen3_next.py` — same as qwen3_moe.py equivalents (SparseMoeBlock.forward line 202; decoder layer forward line 401)
- `/vllm/envs.py` — add three new env vars (M3, Section 4)

---

## 6. Build Sequence

**Phase 1 — Infrastructure (no behavior change):**
- [ ] Create `calibration_custom_ops.py` with the three `direct_register_custom_op` registrations and empty (pass) bodies
- [ ] Add GPU accumulator dict declarations to `calibration_hooks.py`; remove `_callbacks`/`dispatch`/`register_callback`; add prealloc buffer dicts
- [ ] Add new env vars to `envs.py`
- [ ] Fix `_calib_buf_rows` in `triton_moe.py` (H2)
- [ ] Broaden `_current_layer_idx` guard in `moe_runner.py` (M4)

**Phase 2 — Custom-op-interior writers (reap, pem, routing_stats, router_logits_stats, wanda):**
- [ ] Rewrite each writer's `setup()` to allocate GPU tensors (pre-capture)
- [ ] Replace CPU callbacks in each with in-graph GPU ops in `moe_runner.py` and `triton_moe.py`
- [ ] Fix `captured_entry_count()` for each (H1)
- [ ] Update `dump()` for each (GPU→CPU copy at dump time)
- [ ] Verify: smoke test with `enforce_eager=False`, `VLLM_COMPILE=1`, token counts non-zero post-decode

**Phase 3 — Reservoir rewrite (C3):**
- [ ] Rewrite `calibration_output_reservoir.py` to deterministic fixed-stride per-expert (Section 2.7)
- [ ] Verify: distribution check — per-expert write counts should be uniform across cap slots

**Phase 4 — Traced-region signals (C1):**
- [ ] Implement `_calib_dense_imatrix_accum`, `_calib_block_out_accum`, `_calib_layer_in_accum` custom op bodies
- [ ] Insert calls in `linear.py`, `logits_processor.py`, `qwen3_moe.py`, `qwen3_next.py`
- [ ] Rewrite `calibration_imatrix.py` MoE path (in-graph) + dense path (via custom op)
- [ ] Rewrite `calibration_block_outputs.py` with monotonic write pointer (C2)
- [ ] Verify: `block_out` accumulator non-empty after compile=ON forward

**Phase 5 — input_cov + stage2_profile (H3):**
- [ ] Implement `calibration_input_cov.py` resident mode (full GPU `baddbmm_` path)
- [ ] Implement offload mode (enqueue CPU task from custom op body)
- [ ] Stub out `calibration_stage2_profile.py` for replay-mode
- [ ] Verify: resident mode accumulator shape + memory footprint at TP=4

**Phase 6 — Verify gates:**
- [ ] Gate 1: `VLLM_CALIB_CAPTURE_EXPERT=1 VLLM_COMPILE=1` — decode-only batch, verify `_REAP_TOKEN_COUNTS_GPU[0].sum().item() > 0`
- [ ] Gate 2: `VLLM_CALIB_CAPTURE_BLOCK=1 VLLM_COMPILE=1` — verify `_BLOCK_OUT_PTR_GPU[0].item() > 0` after prefill (traced-region signal test)
- [ ] Gate 3: `VLLM_CALIB_CAPTURE_IMATRIX=1 VLLM_COMPILE=1` — verify `_IMATRIX_DENSE_GPU["model.layers.0.self_attn.q_proj"].sum().item() > 0` (dense path, traced-region)
- [ ] Gate 4: full 8000-trace calib run → all sidecars non-empty → downstream REAP finalize completes without shape errors
- [ ] Gate 5: resume from in-flight v1 CPU-checkpoint errors on schema version mismatch (checkpoint version bump hard-errors cleanly)

---

## 7. Checkpoint Schema Version Bump

The existing CPU-checkpoint schema for all writers changes (GPU tensors replace CPU lists/dicts). Any in-flight v1 checkpoint is incompatible. Add to each writer's `load_checkpoint()`:

```python
if data.get("schema_version", 0) < 2:
    raise ValueError(
        "Calibration checkpoint schema v1 is incompatible with v2 (GPU accumulator format). "
        "Delete the checkpoint directory and restart capture."
    )
```

This hard-errors on resume (LOW/NIT item) rather than silently corrupting state.

---

## 8. LOW/NIT Items

- **Router logits BOS-path:** BOS token (position 0, single-token prefill) produces a sink logit distribution. Document in `calibration_router_logits_stats.py` module docstring that BOS tokens are accumulated without filtering — callers should be aware that position-0 statistics may be biased toward the sink expert.
- **Redundant `.long()`**: The original `_on_router` callbacks had `.long()` on already-int64 tensors. The new in-graph path uses `topk_ids` directly from `select_experts` which returns `int64`; drop all `.long()` casts.
- **Prealloc `ones`/`arange` buffers**: `_ch._ONES_BUF_GPU[layer_idx]` (`[buf_rows * top_k]` int64 filled with 1), `_ch._ARANGE_BUF_GPU[layer_idx]` (`[buf_rows]` int64 arange) are allocated once in `setup()` and sliced per dispatch. This avoids `torch.ones`/`torch.arange` allocation inside the graph (allocation in-graph creates dynamic memory which disrupts graph replay).
