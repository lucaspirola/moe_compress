# PLAN — In-graph cudagraph-safe calibration capture (ROUND 3, all signals)

Implements tasks/CALIB_CUDAGRAPH_FAST_CAPTURE_DESIGN.md. Branch feat/calib-cudagraph-capture.
Fixes round-2: C1-RESIDUAL (named-arg mutates_args custom-op retention), H-NEW-1 (input_cov clamp), M-NEW-1/2/3.

---

# CALIB CUDAGRAPH FAST CAPTURE — CORRECTED FULL IMPLEMENTATION PLAN (Round 3)

## 0. Ground Truth from Patch + Source (M-NEW-3 / Version Skew)

All insertion-point citations below are relative to the **patched file symbols**, not upstream line numbers, because the patch modifies `MoERunner._apply_quant_method` / `.forward` significantly. The relevant placement facts confirmed from the patch:

- **`_current_layer_idx` assignment** lives in `MoERunner.forward` (the non-monolithic path), guarded by `if _ch._CAPTURE_EXPERT_UNWEIGHTED or _ch._CAPTURE_EXPERT_MID` (patch line 10375). It is NOT in `_apply_quant_method`.
- **`router` dispatch** fires in `MoERunner.forward`, in the non-monolithic branch, after `topk_weights, topk_ids = self.router.select_experts(...)`. Patch line 10377–10390.
- **`expert_in` dispatch** fires immediately after the router dispatch in the same non-monolithic branch. Patch line 10391–10397.
- **`expert_out_weighted` dispatch** fires after `fused_out = self._quant_method.apply(...)` returns, still in `MoERunner.forward` non-monolithic branch. Patch line 10407–10414.
- **`expert_out_unweighted` and `expert_mid` dispatches** fire inside `TritonExperts.apply`, reading `_ch._current_layer_idx` which was set by the runner earlier in the same call chain.
- **`layer_in` and `block_out` dispatches** fire in `Qwen3MoeSparseMoeBlock.forward` and `Qwen3NextSparseMoeBlock.forward`, using `self.experts.moe_layer_id` (NOT `self.moe_layer_id` — the SparseMoeBlock has no such attribute of its own).
- **`linear_in` dispatch** fires in `ReplicatedLinear.forward`, `ColumnParallelLinear.forward`, and `RowParallelLinear.forward`, all using `self.prefix`.
- **`lm_head_in` dispatch** fires in `LogitsProcessor._get_logits`, passing `hidden_states` (before the matmul).

**Implication for the plan:** All GPU accumulation for `router`, `expert_in`, `expert_out_weighted` must be inserted in `MoERunner.forward` (non-monolithic branch), not `_apply_quant_method`. This is still inside the `vllm.moe_forward` custom op body (confirmed: `_forward_impl` calls `_apply_quant_method` which calls `forward` through the runner chain — the entire stack is opaque to Dynamo). The `layer_in`, `block_out`, and `linear_in`/`lm_head_in` dispatch sites remain in traced-region code.

---

## 1. Boundary Classification (C1 — final)

**Custom-op-interior (inline GPU accumulation is graph-safe):**
All code reachable from `MoERunner.forward` in the non-monolithic path: `router`, `expert_in`, `expert_out_weighted` dispatch sites in `MoERunner.forward`; `expert_out_unweighted` and `expert_mid` sites in `TritonExperts.apply`. All are inside `vllm.moe_forward` (confirmed: `direct_register_custom_op`, moe_runner.py:167–173).

**Traced-region — requires `direct_register_custom_op` wrapper (C1-RESIDUAL fix):**
- `linear_in` → `ColumnParallelLinear.forward`, `ReplicatedLinear.forward`, `RowParallelLinear.forward` — called from `@support_torch_compile` model forward
- `lm_head_in` → `LogitsProcessor._get_logits` — called from compiled model
- `block_out` → `Qwen3MoeSparseMoeBlock.forward`, `Qwen3NextSparseMoeBlock.forward` — inside `@support_torch_compile` model
- `layer_in` → same SparseMoeBlock forward methods — same compiled region

**Affected writers by boundary:**
- Custom-op-interior only: `reap_scores`, `per_expert_max`, `routing_stats`, `router_logits_stats`, `wanda_scalar_row`, `output_reservoir`
- Traced-region only: `block_outputs` (block_out signal)
- Mixed (MoE path interior, dense path traced): `imatrix` (MoE via `expert_in`/`expert_mid`, dense via `linear_in`/`lm_head_in`)
- Traced-region: `stage2_profile` (layer_in, plus TritonExperts-interior signals — routed to replay, see Section 2.10)
- Traced-region + interior mixed: `input_cov` (dense `linear_in` traced; MoE `expert_in` interior)

---

## 2. C1-RESIDUAL Fix: Correct `direct_register_custom_op` Pattern

**The bug in the prior plan:** `mutates_args=[]` with a None-returning op that only touches module-level globals is DCE'd by Inductor. Dynamo sees no tensor aliasing, no output, no mutation of named args — the op is dead.

**The fix:** Pass the actual GPU accumulator tensor as a named function argument and declare it in `mutates_args`. Dynamo then sees a tensor mutation and cannot eliminate the call. This matches the proven vLLM pattern: `lora_shrink` uses `mutates_args=["output_tensor"]`, `unified_attention_with_output` uses `mutates_args=["output", "output_block_scale"]`, `fused_moe_lora` uses `mutates_args=["output"]`.

**Complication:** The accumulator is keyed by (layer_idx, layer_name, etc.) — it is a different tensor for every call site. But `layer_idx` is a compile-time constant (Python int frozen at trace time), so the specific accumulator tensor resolved by `_ch._IMATRIX_DENSE_GPU[layer_idx]` at trace time is also a fixed tensor address. The caller fetches the accumulator by key and passes it as an argument.

**Pattern for all three traced-region ops:**

```python
# In vllm/calibration_custom_ops.py

def _calib_imatrix_dense_accum(accum: torch.Tensor, x: torch.Tensor) -> None:
    """accum: [H] fp32, in-place add of x.float()^2.sum(dim=0). mutates accum."""
    accum.add_(x.float().pow(2).sum(dim=0))

def _calib_imatrix_dense_accum_fake(accum: torch.Tensor, x: torch.Tensor) -> None:
    return  # no output; accum mutation declared via mutates_args

direct_register_custom_op(
    op_name="calib_imatrix_dense_accum",
    op_func=_calib_imatrix_dense_accum,
    mutates_args=["accum"],
    fake_impl=_calib_imatrix_dense_accum_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


def _calib_block_out_accum(accum: torch.Tensor, ptr: torch.Tensor,
                            arange: torch.Tensor, hidden: torch.Tensor) -> None:
    """accum: [cap, H] bf16 (write buffer), ptr: [] int64 (write pointer).
    Writes hidden[:T] into accum at monotonic positions, advances ptr by T."""
    T = hidden.shape[0]
    cap = accum.shape[0]
    write_idx = (ptr + arange[:T]).clamp(max=cap - 1)  # [T]
    valid = (ptr + arange[:T]) < cap                    # [T] bool
    src = hidden * valid.unsqueeze(1).to(hidden.dtype)   # zero out OOB
    accum.scatter_(0, write_idx.unsqueeze(1).expand(-1, accum.shape[1]),
                   src.to(accum.dtype))
    ptr.copy_((ptr + T).clamp(max=cap))

def _calib_block_out_accum_fake(accum, ptr, arange, hidden):
    return

direct_register_custom_op(
    op_name="calib_block_out_accum",
    op_func=_calib_block_out_accum,
    mutates_args=["accum", "ptr"],
    fake_impl=_calib_block_out_accum_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


def _calib_layer_in_accum(accum: torch.Tensor, ptr: torch.Tensor,
                           arange: torch.Tensor, hidden: torch.Tensor) -> None:
    """Same write-pointer pattern as block_out but for layer_in."""
    T = hidden.shape[0]
    cap = accum.shape[0]
    write_idx = (ptr + arange[:T]).clamp(max=cap - 1)
    valid = (ptr + arange[:T]) < cap
    src = hidden * valid.unsqueeze(1).to(hidden.dtype)
    accum.scatter_(0, write_idx.unsqueeze(1).expand(-1, accum.shape[1]),
                   src.to(accum.dtype))
    ptr.copy_((ptr + T).clamp(max=cap))

def _calib_layer_in_accum_fake(accum, ptr, arange, hidden):
    return

direct_register_custom_op(
    op_name="calib_layer_in_accum",
    op_func=_calib_layer_in_accum,
    mutates_args=["accum", "ptr"],
    fake_impl=_calib_layer_in_accum_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)
```

**Caller pattern in traced-region code (e.g., `ColumnParallelLinear.forward`):**

```python
# At trace time: _ch._CAPTURE_IMATRIX is a Python bool constant.
# When True, Dynamo sees the constant-True branch and records the custom op call.
# When False, Dynamo folds the branch and the call is DCE'd (correctly -- no accumulator).
if _ch._CAPTURE_IMATRIX and self.prefix:
    _accum = _ch._IMATRIX_DENSE_GPU.get(self.prefix)  # tensor, resolved at trace time
    if _accum is not None:
        torch.ops.vllm.calib_imatrix_dense_accum(_accum, input_)
```

`_ch._IMATRIX_DENSE_GPU.get(self.prefix)` is evaluated at trace time: `self.prefix` is a compile-time attribute (string constant per layer instance), and the dict is pre-populated by `setup()` before any forward. The resolved tensor `_accum` is a specific GPU tensor with a stable address — Dynamo records it as a constant in the graph. The `mutates_args=["accum"]` declaration tells Dynamo this call mutates that specific tensor, preventing DCE.

**Verify gate (added per C1-RESIDUAL):** After compiling with `TORCH_COMPILE_DEBUG=1` or using `depyf` to dump the compiled graph, check that `torch.ops.vllm.calib_block_out_accum` (or `calib_imatrix_dense_accum`) appears as a node in the graph IR for the traced-region callers. The verify gate for `block_out` is: `VLLM_CALIB_CAPTURE_BLOCK=1 VLLM_COMPILE=1`, dump graph, grep for `calib_block_out_accum`. Token count non-zero is necessary but not sufficient — must confirm the op node survives in the compiled graph.

---

## 3. Shared Infrastructure Changes

### 3.1 `vllm/calibration_hooks.py`

Remove `_callbacks`, `dispatch()`, `register_callback()`. Add module-level GPU accumulator dicts:

```python
# Per-layer GPU accumulators (allocated by each writer's setup(), pre-capture)
_REAP_SCORE_ACCUM_GPU:     dict[int, Tensor] = {}   # [E] fp32
_REAP_TOKEN_COUNTS_GPU:    dict[int, Tensor] = {}   # [E] int64
_PEM_ACCUM_GPU:            dict[int, Tensor] = {}   # [E, K] fp32  (-inf init)
_ROUTING_FREQ_GPU:         dict[int, Tensor] = {}   # [E] int64
_ROUTING_WSUM_GPU:         dict[int, Tensor] = {}   # [E] fp32
_RLOGITS_SUM_GPU:          dict[int, Tensor] = {}   # [E] fp32
_RLOGITS_SQ_GPU:           dict[int, Tensor] = {}   # [E] fp32
_RLOGITS_COUNT_GPU:        dict[int, Tensor] = {}   # [E] int64
_IMATRIX_MoE_GPU:          dict[int, Tensor] = {}   # [E, H] fp32
_IMATRIX_DENSE_GPU:        dict[str, Tensor] = {}   # prefix -> [H] fp32
_WANDA_GPU:                dict[int, Tensor] = {}   # [E, H] fp32
_RESERVOIR_GPU:            dict[int, Tensor] = {}   # [E, cap, H] fp32
_RESERVOIR_CTR_GPU:        dict[int, Tensor] = {}   # [E] int64
_BLOCK_OUT_GPU:            dict[int, Tensor] = {}   # [cap, H] bf16
_BLOCK_OUT_PTR_GPU:        dict[int, Tensor] = {}   # [] int64
_LAYER_IN_GPU:             dict[int, Tensor] = {}   # [cap, H] bf16
_LAYER_IN_PTR_GPU:         dict[int, Tensor] = {}   # [] int64
_INPUT_COV_GPU:            dict[int, Tensor] = {}   # [E, H, H] fp32 (resident mode)
_TOPK_WEIGHTS_STASH_GPU:   dict[int, Tensor] = {}   # [buf_rows, top_k] fp32
_TOPK_IDS_STASH_GPU:       dict[int, Tensor] = {}   # [buf_rows, top_k] int64
# Shared prealloc scratch buffers (allocated by setup(), pre-capture):
_ARANGE_BUF_GPU:           dict[int, Tensor] = {}   # [buf_rows] int64 arange 0..buf_rows-1
_ONES_I64_BUF_GPU:         dict[int, Tensor] = {}   # [buf_rows*top_k] int64 ones
```

Keep `_current_layer_idx: int = -1`, all `_CAPTURE_*` flag globals, `_CALIB_MAX_LAYER`.

Note on in-graph allocation: vLLM's CUDA graph implementation uses a managed memory pool, so intermediate tensor allocations inside custom op bodies (e.g., `torch.zeros`, `cumsum` output) do NOT break graph replay — they are replayed against the same pool. Pre-allocating `_ARANGE_BUF_GPU` is an optional perf optimization (avoids repeated small allocations), not a correctness requirement. The plan pre-allocates them anyway for predictability.

### 3.2 `_calib_buf_rows` fix (H2)

In `TritonExperts.__init__`:

```python
from vllm import envs as _envs
_ccs = (vllm_config.compilation_config.max_cudagraph_capture_size or 0)
_mbt = getattr(getattr(vllm_config, "scheduler_config", None),
               "max_num_batched_tokens", 0) or 0
_floor = getattr(_envs, "VLLM_CALIB_BUF_ROWS_FLOOR", 512)
self._calib_buf_rows = max(_ccs, _mbt, _floor)
```

At the OOB skip in `TritonExperts.apply`, before the calibration accumulation block:

```python
if num_tokens > self._calib_buf_rows:
    if self._moe_layer_id not in _WARNED_OOB:
        _WARNED_OOB.add(self._moe_layer_id)
        logger.warning(
            "calib: buf_rows=%d < num_tokens=%d for layer %d; "
            "prefill tokens SKIPPED for expert_out_unweighted/mid capture. "
            "Increase VLLM_CALIB_BUF_ROWS_FLOOR or max_num_batched_tokens.",
            self._calib_buf_rows, num_tokens, self._moe_layer_id,
        )
    # skip the entire calibration accumulation block
```

`_WARNED_OOB: set[int] = set()` is a module-level set in `triton_moe.py`.

### 3.3 `_current_layer_idx` guard broadening (M4)

In `MoERunner.forward` (non-monolithic path), replace:

```python
if _ch._CAPTURE_EXPERT_UNWEIGHTED or _ch._CAPTURE_EXPERT_MID:
    _ch._current_layer_idx = layer.moe_layer_id
```

With:

```python
if (_ch._CAPTURE_EXPERT_UNWEIGHTED or _ch._CAPTURE_EXPERT_MID or
        _ch._CAPTURE_IMATRIX or _ch._CAPTURE_OUTPUT_RESERVOIR or
        _ch._CAPTURE_PER_EXPERT_MAX or _ch._CAPTURE_WANDA_SCALAR_ROW or
        _ch._CAPTURE_REAP_SCORES or _ch._CAPTURE_INPUT_COV):
    _ch._current_layer_idx = layer.moe_layer_id
```

This ensures every TritonExperts-interior accumulation sees a valid layer index regardless of which subset of flags is enabled.

### 3.4 New env vars (M3) — `vllm/envs.py`

Add to the existing calib block:

```python
"VLLM_CALIB_INPUT_COV_MODE": lambda: os.getenv("VLLM_CALIB_INPUT_COV_MODE", "off"),
# "off" | "resident" | "offload"
"VLLM_CALIB_BUF_ROWS_FLOOR": lambda: int(os.getenv("VLLM_CALIB_BUF_ROWS_FLOOR", "512")),
"VLLM_CALIB_STAGE2_PROFILE_MODE": lambda: os.getenv("VLLM_CALIB_STAGE2_PROFILE_MODE", "replay"),
# "replay" = post-capture via feat/calib-v3-replay driver; "live" = not supported under compile
```

---

## 4. Per-Writer Implementation

### 4.1 `calibration_reap_scores.py` — custom-op-interior

**GPU accumulators** (allocated by `setup()` before first forward):

```python
_REAP_SCORE_ACCUM_GPU:   dict[int, Tensor]   # [E] fp32, zero-init
_REAP_TOKEN_COUNTS_GPU:  dict[int, Tensor]   # [E] int64, zero-init
_TOPK_WEIGHTS_STASH_GPU: dict[int, Tensor]   # [buf_rows, top_k] fp32
_ONES_I64_BUF_GPU:       dict[int, Tensor]   # [buf_rows*top_k] int64 ones
```

**In-graph accumulation — `MoERunner.forward` non-monolithic branch:**

After `topk_weights, topk_ids = self.router.select_experts(...)`:

```python
if _ch._CAPTURE_REAP_SCORES:
    _li = _ch._current_layer_idx          # int constant at capture time
    _ch._TOPK_WEIGHTS_STASH_GPU[_li][:topk_weights.shape[0]].copy_(topk_weights)
```

`topk_weights.shape[0]` is the captured batch size — constant at graph capture.

**In-graph accumulation — `TritonExperts.apply`** (after OOB check, with valid `_unweighted_slice`):

```python
if _ch._CAPTURE_REAP_SCORES:
    _li = _ch._current_layer_idx
    scores  = _ch._REAP_SCORE_ACCUM_GPU[_li]     # [E]
    counts  = _ch._REAP_TOKEN_COUNTS_GPU[_li]    # [E]
    stash   = _ch._TOPK_WEIGHTS_STASH_GPU[_li][:num_tokens]  # [T, top_k]
    # _unweighted_slice: [T, top_k, K] bf16
    norms   = _unweighted_slice.float().norm(dim=-1)          # [T, top_k]
    weighted = norms * stash                                  # [T, top_k]
    flat_ids  = topk_ids.reshape(-1)                          # [T*top_k]
    flat_vals = weighted.reshape(-1)                          # [T*top_k]
    scores.scatter_add_(0, flat_ids, flat_vals)
    ones = _ch._ONES_I64_BUF_GPU[_li][:flat_ids.shape[0]]    # [T*top_k]
    counts.scatter_add_(0, flat_ids, ones)
```

**`captured_entry_count()`:**

```python
def captured_entry_count() -> bool:
    return any(v.sum().item() > 0 for v in _ch._REAP_TOKEN_COUNTS_GPU.values())
```

**`dump()`** (called outside graph, at end of capture run):

```python
for layer_idx, scores_gpu in _ch._REAP_SCORE_ACCUM_GPU.items():
    counts_cpu = _ch._REAP_TOKEN_COUNTS_GPU[layer_idx].cpu().float().clamp(min=1.0)
    scores_cpu = scores_gpu.cpu()
    normalized = scores_cpu / counts_cpu
    # serialize to sidecar payload (unchanged schema)
```

Remove all CPU stash, `_on_router`, `_on_expert_out_unweighted`, `register_callback` calls.

### 4.2 `calibration_per_expert_max.py` — custom-op-interior

**GPU accumulators:**

```python
_PEM_ACCUM_GPU: dict[int, Tensor]  # [E, K] fp32, -inf init
```

**In-graph accumulation — `TritonExperts.apply`:**

```python
if _ch._CAPTURE_PER_EXPERT_MAX:
    _li = _ch._current_layer_idx
    acc = _ch._PEM_ACCUM_GPU[_li]     # [E, K]
    abs_vals = _unweighted_slice.float().abs()        # [T, top_k, K]
    flat_ids  = topk_ids.reshape(-1)                  # [T*top_k]
    flat_vals = abs_vals.reshape(-1, acc.shape[1])    # [T*top_k, K]
    idx_exp   = flat_ids.unsqueeze(1).expand_as(flat_vals)
    acc.scatter_reduce_(0, idx_exp, flat_vals, reduce="amax", include_self=True)
```

**`captured_entry_count()`:**

```python
return any((v > -1e37).any().item() for v in _ch._PEM_ACCUM_GPU.values())
```

**`dump()`**: `acc.cpu()` per layer.

### 4.3 `calibration_routing_stats.py` — custom-op-interior

**GPU accumulators:**

```python
_ROUTING_FREQ_GPU: dict[int, Tensor]  # [E] int64, zero-init
_ROUTING_WSUM_GPU: dict[int, Tensor]  # [E] fp32, zero-init
```

**In-graph accumulation — `MoERunner.forward`** (same location as reap topk stash, after `topk_weights, topk_ids`):

```python
if _ch._CAPTURE_ROUTING_STATS:
    _li = _ch._current_layer_idx
    freq = _ch._ROUTING_FREQ_GPU[_li]
    wsum = _ch._ROUTING_WSUM_GPU[_li]
    flat_ids = topk_ids.reshape(-1)
    flat_w   = topk_weights.reshape(-1)
    ones = _ch._ONES_I64_BUF_GPU[_li][:flat_ids.shape[0]]
    freq.scatter_add_(0, flat_ids, ones)
    wsum.scatter_add_(0, flat_ids, flat_w)
```

**`captured_entry_count()`:**

```python
return any(v.sum().item() > 0 for v in _ch._ROUTING_FREQ_GPU.values())
```

### 4.4 `calibration_router_logits_stats.py` — custom-op-interior

**GPU accumulators:**

```python
_RLOGITS_SUM_GPU:   dict[int, Tensor]  # [E] fp32, zero-init
_RLOGITS_SQ_GPU:    dict[int, Tensor]  # [E] fp32, zero-init
_RLOGITS_COUNT_GPU: dict[int, Tensor]  # [E] int64, zero-init
```

**In-graph accumulation — `MoERunner.forward`** (before `select_experts`, where `router_logits` is available; note: already sliced to remove shared-expert column if `_fse_fuse_gate`):

```python
if _ch._CAPTURE_ROUTER_LOGITS_STATS:
    _li = _ch._current_layer_idx
    rl   = router_logits.float()                    # [T, E] (already sliced by runner)
    lsum = _ch._RLOGITS_SUM_GPU[_li]
    lsq  = _ch._RLOGITS_SQ_GPU[_li]
    lcnt = _ch._RLOGITS_COUNT_GPU[_li]
    lsum.add_(rl.sum(dim=0))                        # [E]
    lsq.add_((rl * rl).sum(dim=0))                 # [E]
    lcnt.add_(torch.tensor(rl.shape[0], dtype=torch.int64, device=rl.device))
```

Note: `torch.tensor(rl.shape[0], ...)` is a constant at capture time (batch size is fixed per captured graph) — it becomes a scalar literal in the graph. No dynamic allocation.

**BOS-path note (LOW):** BOS token (single-token prefill at position 0) produces a sink-heavy logit distribution. This is accumulated without filtering. Callers of `dump()` statistics should be aware position-0 tokens skew the mean/variance of the top-expert column.

**`captured_entry_count()`:**

```python
return any(v.sum().item() > 0 for v in _ch._RLOGITS_COUNT_GPU.values())
```

### 4.5 `calibration_imatrix.py` — split: MoE interior + dense traced-region

**MoE path (custom-op-interior, using `expert_in` and `expert_mid` signals in `MoERunner.forward` / `TritonExperts.apply`):**

GPU accumulator:

```python
_IMATRIX_MoE_GPU: dict[int, Tensor]  # [E, H] fp32, zero-init
```

In-graph in `MoERunner.forward` (after `topk_ids` available, before `_quant_method.apply`):

```python
if _ch._CAPTURE_IMATRIX:
    _li = _ch._current_layer_idx
    acc = _ch._IMATRIX_MoE_GPU[_li]    # [E, H]
    x_sq = hidden_states.float() ** 2   # [T, H]
    for k in range(topk_ids.shape[1]):  # top_k is small constant, Dynamo unrolls
        ids_k = topk_ids[:, k]          # [T]
        acc.scatter_add_(0, ids_k.unsqueeze(1).expand(-1, acc.shape[1]), x_sq)
```

`topk_ids.shape[1]` is top_k = compile-time constant (2 or 4). Dynamo unrolls the `for k` loop into a static sequence of `scatter_add_` nodes.

Remove the earlier `topk_ids[:,0]` top-1-surrogate line that appeared in the prior plan draft — the top_k loop is the correct path.

**Dense path (traced-region, via `calib_imatrix_dense_accum` custom op):**

GPU accumulator:

```python
_IMATRIX_DENSE_GPU: dict[str, Tensor]  # prefix -> [H] fp32, zero-init
```

`setup()` pre-populates this dict for all layer prefixes that will be captured (by iterating model named modules).

Callers (all three linear forward methods + `LogitsProcessor._get_logits`):

```python
# In ColumnParallelLinear.forward (and Replicated, RowParallel analogously):
if _ch._CAPTURE_IMATRIX and self.prefix:
    _accum = _ch._IMATRIX_DENSE_GPU.get(self.prefix)
    if _accum is not None:
        torch.ops.vllm.calib_imatrix_dense_accum(_accum, input_)

# In LogitsProcessor._get_logits:
if _ch._CAPTURE_IMATRIX:
    _accum = _ch._IMATRIX_DENSE_GPU.get("lm_head")
    if _accum is not None:
        torch.ops.vllm.calib_imatrix_dense_accum(_accum, hidden_states)
```

`_accum` is `None` when the dict has not been pre-populated for this prefix (e.g., when max_layer gate excludes it). The `if _accum is not None` guard folds to a compile-time constant per layer instance because `self.prefix` is a fixed string and the dict is populated at `setup()` before tracing.

**`captured_entry_count()`:**

```python
moe_ok   = any(v.sum().item() > 0 for v in _ch._IMATRIX_MoE_GPU.values())
dense_ok = any(v.sum().item() > 0 for v in _ch._IMATRIX_DENSE_GPU.values())
return moe_ok or dense_ok
```

### 4.6 `calibration_wanda_scalar_row.py` — custom-op-interior

**GPU accumulator:**

```python
_WANDA_GPU: dict[int, Tensor]  # [E, H] fp32, zero-init
```

**In-graph — `MoERunner.forward`** (after `topk_ids` / `topk_weights`):

```python
if _ch._CAPTURE_WANDA_SCALAR_ROW:
    _li = _ch._current_layer_idx
    acc = _ch._WANDA_GPU[_li]     # [E, H]
    x_sq = hidden_states.float() ** 2    # [T, H]
    for k in range(topk_ids.shape[1]):
        ids_k = topk_ids[:, k]           # [T]
        w_k   = topk_weights[:, k].float().unsqueeze(1) ** 2  # [T, 1]
        contrib = x_sq * w_k             # [T, H]
        acc.scatter_add_(0, ids_k.unsqueeze(1).expand(-1, acc.shape[1]), contrib)
```

**`captured_entry_count()`:**

```python
return any(v.sum().item() > 0 for v in _ch._WANDA_GPU.values())
```

### 4.7 `calibration_output_reservoir.py` — custom-op-interior (C3 fix)

**Design: Deterministic fixed-stride per-expert reservoir, no RNG.**

Algorithm: for expert `e`, the write slot for the `s`-th token routed to `e` within this dispatch is `(ctr_e + s) % cap`. After the dispatch, `ctr_e += N_e` (count of tokens routed to `e` this dispatch). Each token gets a distinct slot within the dispatch because `s` is the unique within-expert local index. Adjacent dispatches tile without overlap because `ctr_e` advances by `N_e` (not 1).

**GPU accumulators:**

```python
_RESERVOIR_GPU:     dict[int, Tensor]  # [E, cap, H] fp32
_RESERVOIR_CTR_GPU: dict[int, Tensor]  # [E] int64, zero-init
```

`cap = VLLM_CALIB_OUTPUT_RESERVOIR_CAP` (default 256).

**In-graph accumulation — `TritonExperts.apply`:**

```python
if _ch._CAPTURE_OUTPUT_RESERVOIR:
    _li  = _ch._current_layer_idx
    res  = _ch._RESERVOIR_GPU[_li]      # [E, cap, H]
    ctrs = _ch._RESERVOIR_CTR_GPU[_li]  # [E]
    cap  = res.shape[1]
    H    = res.shape[2]
    T    = _unweighted_slice.shape[0]
    E    = ctrs.shape[0]

    for k in range(topk_ids.shape[1]):  # Dynamo unrolls
        ids_k  = topk_ids[:, k].long()          # [T]
        vals_k = _unweighted_slice[:, k, :].float()  # [T, H]

        # One-hot [T, E] and cumsum to get within-dispatch local rank per expert
        indicator = torch.zeros(T, E, dtype=torch.int64, device=ids_k.device)
        indicator.scatter_(1, ids_k.unsqueeze(1), 1)  # [T, E]
        # cumsum along T gives prefix count; subtract 1 for 0-based local rank
        local_rank = (indicator.cumsum(dim=0)
                       .gather(1, ids_k.unsqueeze(1))
                       .squeeze(1) - 1)            # [T] int64, range [0, N_e)

        base_ctr   = ctrs.gather(0, ids_k)         # [T]: each token's expert ctr
        write_slot = (base_ctr + local_rank) % cap # [T]

        # Flatten to [E*cap, H] for scatter_
        lin_idx = (ids_k * cap + write_slot).unsqueeze(1).expand(-1, H)  # [T, H]
        res.view(-1, H).scatter_(0, lin_idx, vals_k)

        # Advance ctrs: add N_e for each expert
        delta = indicator.sum(dim=0)               # [E]
        ctrs.add_(delta)
```

All ops (`zeros`, `scatter_`, `cumsum`, `gather`, `add_`) are CUDA-graph-safe. The intermediary `indicator` and `local_rank` tensors are allocated from the CUDA graph memory pool during replay — no stability issue.

**`captured_entry_count()`:**

```python
return any(v.sum().item() > 0 for v in _ch._RESERVOIR_CTR_GPU.values())
```

**`dump()`:**

```python
for layer_idx, res in _ch._RESERVOIR_GPU.items():
    ctr = _ch._RESERVOIR_CTR_GPU[layer_idx].cpu()
    # Each expert may have less than cap tokens if total < cap
    # Dump as-is: shape [E, cap, H], valid tokens are [0:min(ctr_e, cap)]
    # The consumer already handles variable per-expert fill.
    data = res.cpu()
    save_reservoir_sidecar(layer_idx, data, ctr)
```

Remove all `_RESERVOIR` CPU dicts, `torch.Generator`, Vitter-R phase-1/2 logic.

### 4.8 `calibration_block_outputs.py` — traced-region (C2 fix + M-NEW-1 + M-NEW-2)

**GPU accumulators:**

```python
_BLOCK_OUT_GPU:     dict[int, Tensor]  # layer_idx -> [cap, H] bf16
_BLOCK_OUT_PTR_GPU: dict[int, Tensor]  # layer_idx -> [] int64
_BLOCK_ARANGE_GPU:  dict[int, Tensor]  # layer_idx -> [cap] int64 arange
```

`cap = VLLM_CALIB_BLOCK_OUTPUTS_SUBSET_SIZE * max_seq_len`, allocated by `setup()`.

**Attribute fix (M-NEW-1):** Both `Qwen3MoeSparseMoeBlock` and `Qwen3NextSparseMoeBlock` use `self.experts.moe_layer_id` — confirmed directly from the patch (lines 10523, 10536, 10610, 10623). The `layer_idx` passed to the custom op is `self.experts.moe_layer_id`.

**Callers in traced-region code (both qwen3_moe.py and qwen3_next.py):**

Replace the existing `_ch.dispatch("block_out", ...)` calls with:

```python
if _ch._CAPTURE_BLOCK:
    _layer_id = self.experts.moe_layer_id  # compile-time int for this instance
    _accum = _ch._BLOCK_OUT_GPU.get(_layer_id)
    _ptr   = _ch._BLOCK_OUT_PTR_GPU.get(_layer_id)
    _arng  = _ch._BLOCK_ARANGE_GPU.get(_layer_id)
    if _accum is not None:
        torch.ops.vllm.calib_block_out_accum(_accum, _ptr, _arng,
                                             final_hidden_states)
```

This appears in all four dispatch sites (internal-router + external-gate branch in both qwen3_moe and qwen3_next).

**`dump()` — `BlockHiddenPayload` fields fix (M-NEW-2):**

The actual `BlockHiddenPayload` schema has fields `(schema_version, layer_idx, n_prompts_in_subset, hidden_states)`.

```python
for layer_idx, buf in _ch._BLOCK_OUT_GPU.items():
    valid_tokens = int(_ch._BLOCK_OUT_PTR_GPU[layer_idx].item())
    if valid_tokens == 0:
        continue
    hidden_data = buf[:valid_tokens].cpu()   # [valid_tokens, H] bf16
    payload = BlockHiddenPayload(
        schema_version=CHECKPOINT_SCHEMA_VERSION,
        layer_idx=layer_idx,
        n_prompts_in_subset=_n_prompts_captured,  # tracked separately by driver
        hidden_states=hidden_data,
    )
    save_block_hidden(layer_idx, payload)
```

**`captured_entry_count()`:**

```python
return any(v.item() > 0 for v in _ch._BLOCK_OUT_PTR_GPU.values())
```

### 4.9 `calibration_input_cov.py` — flag-gated dual-mode + H-NEW-1 fix

**Mode "off":** `setup()` is a no-op. No allocation.

**Mode "resident":** GPU accumulators hold full `[E, H_local, H_local]` covariance per MoE layer.

GPU accumulators:

```python
_INPUT_COV_GPU:      dict[int, Tensor]  # [E, H_local, H_local] fp32
_INPUT_COV_TEMP_GPU: dict[int, Tensor]  # [E, buf_rows, H_local] fp32 (reused per fwd)
```

`H_local = H / tp_size`. `buf_rows = _calib_buf_rows`.

**H-NEW-1 OOB fix:** `local_rank` for a hot expert can exceed `buf_rows` when many tokens in one batch route to the same expert. Fix by clamping/masking before scatter:

**In-graph accumulation — `MoERunner.forward`** (inside custom-op boundary):

```python
if _ch._CAPTURE_INPUT_COV and _ch._INPUT_COV_MODE == "resident":
    _li = _ch._current_layer_idx
    cov = _ch._INPUT_COV_GPU[_li]       # [E, H_local, H_local]
    tmp = _ch._INPUT_COV_TEMP_GPU[_li]  # [E, buf_rows, H_local]
    tmp.zero_()
    x = hidden_states.float()           # [T, H_local]
    E = cov.shape[0]
    buf = tmp.shape[1]

    for k in range(topk_ids.shape[1]):
        ids_k = topk_ids[:, k].long()   # [T]
        indicator = torch.zeros(T, E, dtype=torch.int64, device=ids_k.device)
        indicator.scatter_(1, ids_k.unsqueeze(1), 1)
        local_rank = (indicator.cumsum(dim=0)
                       .gather(1, ids_k.unsqueeze(1))
                       .squeeze(1) - 1)     # [T] range [0, N_e)

        # H-NEW-1: clamp overflow tokens — if local_rank >= buf_rows, drop
        valid_mask = local_rank < buf        # [T] bool
        safe_rank  = local_rank.clamp(max=buf - 1)  # [T], clamped

        dest_e  = ids_k                              # [T]
        dest_r  = safe_rank                          # [T]
        lin_idx = (dest_e * buf + dest_r)            # [T]

        # Zero out overflow tokens (they write to clamped-to-last slot but masked)
        x_masked = x * valid_mask.unsqueeze(1).to(x.dtype)  # [T, H_local]
        tmp.view(-1, x.shape[1]).scatter_(
            0,
            lin_idx.unsqueeze(1).expand(-1, x.shape[1]),
            x_masked,
        )

    # baddbmm_ in-place: cov += tmp.T @ tmp per expert
    # tmp: [E, buf_rows, H]; cov: [E, H, H]
    # baddbmm_ expects [E, H, H] += [E, H, buf] @ [E, buf, H]
    cov.baddbmm_(tmp.transpose(1, 2), tmp)
    # Does NOT materialize a second [E, H, H] tensor — baddbmm_ is in-place. (M2 confirmed)
```

The clamped "last-slot" writes for overflow tokens are zeroed by `x_masked` (the `valid_mask` zeros the input for those tokens), so overflow tokens contribute zero to the accumulation — correct drop semantics.

**Offload mode:** Replace the `baddbmm_` call with an async CPU submit:

```python
x_cpu_future = tmp.cpu()  # D2H copy, async in CUDA stream
_INPUT_COV_OFFLOAD_POOL.submit(_cpu_cov_update, x_cpu_future, _li)
```

`_cpu_cov_update` takes the numpy view and does `cov_cpu += x.T @ x` on CPU. The GPU graph records only the `tmp.zero_()` + scatter fills.

**Peak memory (M2 revisited):**
- TP=4, H_local=512, E=128, 48 layers: `128 × 512 × 512 × 4 × 48 ≈ 6.4 GB` accumulator — feasible.
- TP=1, H=2048, E=128, 48 layers: `128 × 2048 × 2048 × 4 × 48 ≈ 102 GB` — requires offload mode or TP≥4.
- Per-forward temp `[E, buf_rows, H_local]` at TP=4: `128 × 512 × 512 × 4 = 134 MB` — acceptable.
- `baddbmm_` does not allocate a second `[E, H, H]` tensor when called as in-place. Confirmed: PyTorch `baddbmm_` writes into `self` (the accumulator) without temp output allocation.

**`captured_entry_count()`:**

```python
if _ch._INPUT_COV_MODE == "resident":
    return any(v.abs().sum().item() > 0 for v in _ch._INPUT_COV_GPU.values())
return False
```

### 4.10 `calibration_stage2_profile.py` — replay-only (H3)

**H3 resolution:** The `execute_model` method in `vllm/v1/worker/gpu/model_runner.py` returns `None` at line 1183 (last PP rank) and has no post-model-call hook surface. The `sample_tokens` method at line 1185 is a separate scheduler-driven call, not a model-step callback. There is no viable `post_step_hook` surface in the current v1 worker path.

**Decision:** `calibration_stage2_profile.py` operates in replay-only mode under `VLLM_CALIB_STAGE2_PROFILE_MODE=replay` (the default). During live vLLM capture, the module's `setup()` registers no callbacks. Post-capture, the `feat/calib-v3-replay` forward-only replay driver loads the dumped GPU tensor checkpoints and calls `ReamCostAccumulator` and `InputCovarianceAccumulator` from `moe_compress.stage2.profiling` against the replayed outputs.

For `VLLM_CALIB_STAGE2_PROFILE_MODE=live` (non-default), the module registers its five CPU callbacks as before — this only works without `VLLM_COMPILE` (eager mode). Document this clearly in the module docstring.

---

## 5. Implementation File Map

**Files to create:**
- `/vllm/calibration_custom_ops.py` — `direct_register_custom_op` registrations for `calib_imatrix_dense_accum` (`mutates_args=["accum"]`), `calib_block_out_accum` (`mutates_args=["accum", "ptr"]`), `calib_layer_in_accum` (`mutates_args=["accum", "ptr"]`). Module imported by `calibration_hooks.py` at module level to register ops before any forward.

**Files to modify:**

| File | Changes |
|------|---------|
| `vllm/calibration_hooks.py` | Remove `_callbacks`/`dispatch`/`register_callback`; add all GPU accumulator dict declarations; add `_ARANGE_BUF_GPU`, `_ONES_I64_BUF_GPU`; import `calibration_custom_ops` |
| `vllm/calibration_reap_scores.py` | Full rewrite to GPU accumulators + in-graph scatter (Section 4.1) |
| `vllm/calibration_per_expert_max.py` | Full rewrite (Section 4.2) |
| `vllm/calibration_routing_stats.py` | Full rewrite (Section 4.3) |
| `vllm/calibration_router_logits_stats.py` | Full rewrite (Section 4.4) |
| `vllm/calibration_imatrix.py` | MoE path in-graph + dense path via custom op; remove top-1-surrogate line (Section 4.5) |
| `vllm/calibration_wanda_scalar_row.py` | Full rewrite (Section 4.6) |
| `vllm/calibration_output_reservoir.py` | Full rewrite, deterministic fixed-stride (Section 4.7) |
| `vllm/calibration_block_outputs.py` | GPU monotonic-pointer write + custom op receiver; fix payload fields (Section 4.8) |
| `vllm/calibration_input_cov.py` | Dual-mode resident/offload + H-NEW-1 clamp fix (Section 4.9) |
| `vllm/calibration_stage2_profile.py` | Stub live path; document replay mode (Section 4.10) |
| `vllm/model_executor/layers/fused_moe/runner/moe_runner.py` | Inline GPU accum in `MoERunner.forward` non-monolithic branch; broadened `_current_layer_idx` guard (M4) |
| `vllm/model_executor/layers/fused_moe/experts/triton_moe.py` | `_calib_buf_rows` fix (H2); OOB warn-once; inline GPU accum in `TritonExperts.apply` |
| `vllm/model_executor/layers/linear.py` | Replace `_ch.dispatch("linear_in", ...)` with `torch.ops.vllm.calib_imatrix_dense_accum(accum, input_)` in all three linear forward methods; `accum` fetched from `_ch._IMATRIX_DENSE_GPU.get(self.prefix)` |
| `vllm/model_executor/layers/logits_processor.py` | Replace `_ch.dispatch("lm_head_in", ...)` with `torch.ops.vllm.calib_imatrix_dense_accum(accum, hidden_states)` |
| `vllm/model_executor/models/qwen3_moe.py` | Replace `_ch.dispatch("block_out", ...)` with `torch.ops.vllm.calib_block_out_accum(accum, ptr, arng, final_hidden_states)` (both branches); replace `_ch.dispatch("layer_in", ...)` with `torch.ops.vllm.calib_layer_in_accum(accum, ptr, arng, hidden_states)` |
| `vllm/model_executor/models/qwen3_next.py` | Same as qwen3_moe.py equivalents |
| `vllm/envs.py` | Add three new env vars (Section 3.4) |

---

## 6. Build Sequence

**Phase 1 — Infrastructure (no behavior change):**
- [ ] Create `calibration_custom_ops.py` with stub bodies (pass) and correct `mutates_args` declarations; verify `torch.ops.vllm.calib_imatrix_dense_accum` callable in Python
- [ ] Update `calibration_hooks.py`: GPU dict declarations, remove callback infrastructure, import custom_ops
- [ ] Add new env vars to `envs.py`
- [ ] Fix `_calib_buf_rows` in `triton_moe.py` (H2)
- [ ] Broaden `_current_layer_idx` guard in `moe_runner.py` (M4)

**Phase 2 — Custom-op-interior writers:**
- [ ] Rewrite `reap_scores`, `per_expert_max`, `routing_stats`, `router_logits_stats`, `wanda_scalar_row` with GPU accumulators and in-graph ops
- [ ] Add GPU accumulation code to `MoERunner.forward` and `TritonExperts.apply`
- [ ] Fix `captured_entry_count()` for all five writers (use `.values()` not `[0]`)
- [ ] Verify gate: `VLLM_CALIB_CAPTURE_EXPERT=1 VLLM_COMPILE=1` — single decode batch — `_REAP_TOKEN_COUNTS_GPU` sum > 0

**Phase 3 — Reservoir rewrite (C3):**
- [ ] Rewrite `calibration_output_reservoir.py` (Section 4.7)
- [ ] Verify: uniform slot distribution check across 1000 dispatches

**Phase 4 — Traced-region signals (C1-RESIDUAL):**
- [ ] Implement full bodies of `calib_imatrix_dense_accum`, `calib_block_out_accum`, `calib_layer_in_accum`
- [ ] Replace `_ch.dispatch(...)` calls in `linear.py`, `logits_processor.py`, `qwen3_moe.py`, `qwen3_next.py` with `torch.ops.vllm.*` calls; pass named accumulator tensor
- [ ] Rewrite `calibration_imatrix.py` (MoE + dense paths); remove top-1-surrogate line
- [ ] Rewrite `calibration_block_outputs.py` (Section 4.8); fix `BlockHiddenPayload` fields (M-NEW-2)
- [ ] Verify gate: `VLLM_CALIB_CAPTURE_BLOCK=1 VLLM_COMPILE=1` — `_BLOCK_OUT_PTR_GPU` > 0 AND `calib_block_out_accum` node present in `TORCH_COMPILE_DEBUG` graph dump (C1-RESIDUAL confirm)
- [ ] Verify gate: `VLLM_CALIB_CAPTURE_IMATRIX=1 VLLM_COMPILE=1` — dense accumulator for `model.layers.0.self_attn.q_proj` > 0

**Phase 5 — input_cov + stage2_profile:**
- [ ] Implement `input_cov` resident mode with H-NEW-1 clamp (Section 4.9)
- [ ] Implement offload mode
- [ ] Stub `stage2_profile` live path; document replay-only default (Section 4.10)
- [ ] Verify: TP=4 resident-mode accumulator shape `[128, 512, 512]` per layer; `baddbmm_` peak stays under 8 GB extra VRAM

**Phase 6 — Integration verify:**
- [ ] Full 8000-trace calib run with all writers active → all sidecars non-empty → REAP finalize completes without shape errors
- [ ] Resume from v1 CPU checkpoint → schema version mismatch hard-error (verify clean message)
- [ ] `TORCH_COMPILE_DEBUG` graph dump for a compiled forward: confirm `calib_block_out_accum` and `calib_imatrix_dense_accum` nodes present, not DCE'd

---

## 7. Schema Version Bump (LOW)

Every writer's `load_checkpoint()` adds:

```python
if data.get("schema_version", 0) < 2:
    raise RuntimeError(
        "Calibration checkpoint schema v1 is incompatible with v2 "
        "(GPU accumulator format). Delete the checkpoint directory "
        "and restart capture from scratch."
    )
```

In-flight v1 CPU-checkpoint resumes will hard-error on the first `load_checkpoint` call — this is the desired behavior (silent corruption would be worse).
