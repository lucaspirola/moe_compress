# PLAN_PLUGIN_12_opt_a_redo — Stage 2 Profile-Pass Sidecar Cache (Optimization A REDO)

## 1. Goal + Spec Citation

SC_FAST_PLAN_V3 §4 Optimization A ("profile-pass sidecar cache") is the top-1 wall-clock lever for the SC strategy row, projected to save 30-50 minutes per SC row by pre-computing the REAM δ_gate and δ̃_expert accumulators in the vLLM calibration pass and loading them into Stage 2 instead of re-running the per-layer profile forward.

The prior implementation (deleted from main) had eight bugs that this plan explicitly addresses. Every bug-fix decision is spelled out below. The plan is written as a direct implementer spec; a reviewer agent reads this before any code is written.

## 2. Architecture Overview

Five cooperating components, each with a single responsibility:

```
  build_self_traces_calib_vllm.py
        │  --capture-stage2-profile (new flag)
        │  env: VLLM_CALIB_CAPTURE_STAGE2_PROFILE=1
        │       VLLM_CALIB_CAPTURE_ROUTER=1          (gate logits)
        │       VLLM_CALIB_CAPTURE_EXPERT_UNWEIGHTED=1 (gated outputs)
        │       VLLM_USE_FLASHINFER_MOE_FP16=0
        ▼
  vllm/calibration_stage2_profile.py          [NEW vLLM patch — writer]
        │  hooks: router_callback (gate logits)
        │         expert_out_unweighted_callback (gated outputs)
        │  accumulates: gate_logit_profiles (list[(offset, [T_b,E])] per batch)
        │               _batch_gated_indexed per expert per batch
        │               total_tokens_per_layer (int64 per layer)
        │  finalize_batch() mirrors ReamCostAccumulator.finalize_batch()
        │               sim_tensor [n_layers, E, E] fp64
        │  also: neuron_act_sum, neuron_act_count, cov_acc (gate_proj+down_proj)
        │        layer_input_reservoir (always captured; see §5/§6 + Crit-3 fix)
        │  at run end: dump_stage2_profile(jsonl_path) →
        │              sidecars/stage2_profile.pt (schema v3)
        │  checkpointing: dump_stage2_profile_checkpoint / load
        ▼
  utils/cached_calibration_signals.py         [MODIFY — schema + IO]
        │  Stage2ProfilePayloadV3 dataclass
        │  SCHEMA_VERSIONS["stage2_profile"] → 3
        │  save_stage2_profile_v3() / load_stage2_profile_v3()
        ▼
  stage2/plugins/stage2_profile_cache.py      [NEW in-repo — reader]
        │  Stage2ProfileCacheProvider
        │  on_load(): load sidecar → Stage2ProfilePayloadV3
        │  on_layer_setup(): hydrate ream_acc + cov_acc + layer_input_acc
        │                    on FULL HIT ONLY
        │  on_profile(): (empty; cache-aware skip handled in LayerMergePlugin)
        ▼
  stage2/plugins/layer_merge.py               [MODIFY — Pattern A skip]
        │  on_profile(): if ctx.has("stage2_profile_full_hit"): skip
```

**Orchestrator wiring** (pattern: explicit loop, NOT dispatch_first):
`stage2/orchestrator.py` lines 1200-1203 (the `Stage2RoutingStatsCacheProvider` explicit-loop block) show the exact precedent. The new plugin's `on_load` is called via the same explicit-loop pattern because `dispatch_first("on_load", ...)` at line 1188 stops at the first non-None result — if `Stage2ReapScoresCacheProvider` hits first, the stage2-profile provider is never called. The explicit-loop pattern avoids this.

**Reader-pairing scope** (bug #8 fix, see §7 below): the prior Plugin #12 introduced a second reader (`Stage2InputCovCacheProvider`) that read the cov sidecar produced by `--capture-input-covariance`. That second reader was deleted along with the rest of the prior Plugin #12 implementation. This redo registers only ONE new reader — `Stage2ProfileCacheProvider` — behind the single config knob `stage2_reap_ream.profile_sidecar.enabled`. There is no longer any code path that registers an input-cov reader alone, so the historic "double-counting via paired readers" failure mode is structurally impossible. See §7 for the full statement.

## 3. Sidecar Schema v3

The existing `Stage2ProfilePayload` (schema v1) at `cached_calibration_signals.py:153-161` is fully replaced. `SCHEMA_VERSIONS["stage2_profile"]` bumps from 1 to 3 (skipping 2 to signal clean break from the deleted first attempt).

**Pattern K applicability note**: this v1→v3 bump is NOT forward-compatible per Pattern K's letter (which describes adding optional fields without breaking old readers). The existing `_check_schema` enforces strict equality and emits "Delete the sidecar to regenerate" on mismatch. This is intentional because (a) no production v1 sidecar exists in the wild — the prior writer was never shipped on a real calibration run — and (b) the v3 schema is substantially different from v1 (different field set, corrected math, additional fields). Pattern K still applies going forward: v3→v4 bumps SHOULD preserve old readers when they add optional fields only.

**Storage convention (READ FIRST)**: ALL dict keys in this payload use `layer_rank` — the 0-based ordinal index into the MoE layer list. The reader translates `layer_rank → layer_idx` (absolute model layer index) during hydration. Rationale: `layer_rank` is portable across model checkpoints, regardless of where MoE layers start in the model (e.g., dense prefix layers shift `layer_idx` but not `layer_rank`). This convention applies to: `gate_logit_profiles`, `neuron_act_sum`, `neuron_act_count`, `cov_acc`, `cov_token_count`. The tensor-fields like `sim_tensor[n_layers, E, E]`, `total_tokens_per_layer[n_layers]`, and `layer_input_reservoir[n_layers]` already use `layer_rank` by virtue of array indexing (entry `[l]` is the `l`-th MoE layer).

```python
@dataclass
class Stage2ProfilePayloadV3:
    # --- identity / cross-validation ---
    format_version: int                     # = 3  (constant; distinguishes from old v1)
    schema_version: int                     # = 3  (checked by load_stage2_profile_v3)
    model_hash: str                         # SHA-256 of model name + config for sanity
    n_layers: int                           # number of MoE layers
    n_experts: int                          # routed experts per layer
    top_k: int                              # top-k routing (for cross-validation)
    # Cross-validates against the run's `s2.covariance_storage_dtype` setting.
    # Allowed: "float16", "bfloat16", "float32". Reader fails loud
    # ("Delete the sidecar to regenerate") if the value disagrees with
    # orchestrator.py:702 `s2.get("covariance_storage_dtype", "float16")`.
    #
    # Writer-side provenance: this value is set by the driver flag
    # `--stage2-profile-cov-storage-dtype` (default "float16") and passed
    # into `_s2p.setup(llm, cov_storage_dtype=...)`. The writer constructs
    # its `InputCovarianceAccumulator` with `.set_storage_dtype(<dtype>)`
    # IMMEDIATELY (do NOT rely on the default torch.float32 at
    # activation_hooks.py:961). At `dump_stage2_profile` time, the writer
    # MUST cross-validate `str(cov_acc.storage_dtype).split(".")[-1] ==
    # configured_cov_storage_dtype` and raise loud if they disagree — this
    # catches the case where a future change adds a code path that mutates
    # storage_dtype after setup() runs. The driver flag value MUST be
    # propagated to the Stage 2 YAML's `covariance_storage_dtype` for the
    # reader to accept the sidecar.
    cov_storage_dtype: str                  # one of {"float16","bfloat16","float32"}

    # --- BUG #3 FIX: exact per-layer token count (NOT sum of expert token_counts) ---
    # int64 tensor, shape [n_layers]. Entry l = Σ_{batches b} T_b for layer_rank l.
    # Written by record_batch_token_count() in the writer. This is |X| for
    # the Eq. 8 denominator — independent of routing activity.
    total_tokens_per_layer: torch.Tensor    # [n_layers] int64

    # --- BUG #2 FIX: raw gate logit profiles (NOT pre-collapsed) ---
    # dict[layer_rank → list[tuple[int, Tensor[T_b, E] fp32]]]
    # Each entry mirrors EXACTLY the live storage type at activation_hooks.py:118:
    #   gate_logit_profiles[layer_idx]: list[(batch_offset, logits_cpu)]
    # The tuple's first element is the cumulative global-token offset
    # (== batch_idx * batch_size * seq_len under the writer's running offset);
    # the live consumer compute_gate_similarity_matrix (activation_hooks.py:501)
    # unpacks it via `for _, t in batches` — i.e., the offset is preserved for
    # downstream debugging but the cosine math only consumes the tensors.
    # On-disk: each list entry is `(int, CPU fp32 tensor)`.
    gate_logit_profiles: dict               # dict[int → list[tuple[int, Tensor[T_b, E] fp32]]]

    # --- BUG #1 FIX: correct REAM Eq. 8 numerator ---
    # [n_layers, E, E] fp64. Entry [l, i, j] = Σ_{t ∈ jointly-active(i,j)}
    # cos(g_i[t], g_j[t]) accumulated via finalize_batch() (mirrors
    # ReamCostAccumulator.finalize_batch() lines 260-448 exactly).
    # Symmetric, zero diagonal. This is the raw sum; division by
    # total_tokens_per_layer[l] happens in the reader (matches live path).
    sim_tensor: torch.Tensor                # [n_layers, E, E] fp64

    # --- per-neuron activation means for C_act (neuron alignment) ---
    # dict[(layer_rank, expert_idx) → Tensor[d_intermediate] fp32]
    # Reader translates layer_rank → layer_idx on hydration.
    neuron_act_sum: dict                    # {(int, int): Tensor[d_int] fp32}
    neuron_act_count: dict                  # {(int, int): int}

    # --- input covariance for Stage 3/4 (gate_proj + down_proj, NOT up_proj) ---
    # up_proj is aliased to gate_proj in the consumer (InputCovarianceAccumulator)
    # because gate+up share the same input tensor; no separate cov needed.
    # dict[(layer_rank, expert_idx, matrix_name) → Tensor[d, d] in cov_storage_dtype]
    # matrix_name in {"gate_proj", "down_proj"}.
    # FINALIZED covariance (post-finalize_layer). See §10/OQ-3 (resolved).
    cov_acc: dict                           # {(int, int, str): Tensor[d, d] in cov_storage_dtype}
    cov_token_count: dict                   # {(int, int, str): int}

    # --- layer-input reservoir (ALWAYS captured when sidecar is written) ---
    # list[Tensor[N, hidden] bf16], one entry per MoE layer rank (length == n_layers).
    # Consumed by SC strategy's _output_space_cost via layer_input_acc.buffer.
    # Per Crit-3 resolution: --capture-stage2-profile IMPLIES reservoir capture;
    # the field is unconditionally populated. No sub-flag.
    # Storage cost at default config (max_samples=8192, hidden=4096, bf16,
    # n_layers=40) is ~3 GB for Qwen3.6-35B-A3B:
    #     40 × 8192 × 4096 × 2 bytes = 2.68 GB
    # Scales linearly with the per-layer token cap. For SC experiments
    # that raise the cap to ~45 K samples, storage grows to ~15 GB.
    # Both are acceptable as the operator has opted in.
    layer_input_reservoir: list             # list[Tensor[N, hidden] bf16]  (len == n_layers)

    # NOTE: a_down is DROPPED (Bug #6 fix). It was persisted but never
    # consumed. The down_proj input covariance is captured under cov_acc
    # with matrix_name="down_proj" — that IS consumed by Stage 3/4.
```

**Serialization contract**: all tensors moved to CPU before `torch.save`. `gate_logit_profiles` stored as `dict[int, list[tuple[int, torch.Tensor]]]` preserving the live storage format byte-for-byte. `layer_input_reservoir` stored as a list of CPU bf16 tensors (length always == n_layers).

## 4. Writer Math Correction (Bug #1 Fix)

The prior writer computed `cos(mean(g_i), mean(g_j))` (cosine of per-expert mean gated outputs) which is mathematically distinct from the live accumulator's semantics.

The correct algorithm, mirroring `ReamCostAccumulator.finalize_batch` (lines 260-448 in `activation_hooks.py`):

**Per batch `b` at layer `l`:**

1. Collect per-expert payloads from `_batch_gated_indexed[(l, e)]` = `(token_indices[T_e], gated[T_e, d_hid])`. The gated output is `σ(x)_e · E_e(x)` where σ is the FULL UNMASKED softmax weight (not top-k renormalized).

2. Concatenate all experts' (token_index, gated_vector) pairs into flat arrays:
   - `all_eids: [N_total]` long — expert id for each row
   - `all_indices: [N_total]` long — global token index for each row
   - `all_gated: [N_total, d_hid]` float32 — gated output

3. Stable-sort by token index. Use `torch.unique_consecutive` to find runs of the same token. Filter to tokens with ≥ 2 active experts (only jointly-active tokens contribute to the sum).

4. For each jointly-active token `t`, for each pair `(e_i, e_j)` with `e_i ≠ e_j`:
   - Compute `cos(g_i[t], g_j[t])` = L2-norm the two gated vectors, take dot product
   - Zero-norm vectors contribute 0 (not NaN) to the sum
   - Add to `sim_sum[l, e_i, e_j]` and `sim_sum[l, e_j, e_i]` (symmetric)

5. After the loop: `sim_sum.fill_diagonal_(0.0)`. Add to the persistent `_sim_tensor[l]` (fp64).

The implementation uses the same chunked bmm approach (steps 6-8 of `finalize_batch`) to bound peak GPU memory. The writer calls this after each batch's expert callbacks have fired.

The key difference from the deleted implementation: the writer uses per-token jointly-active pair cosines, NOT per-expert mean vectors. The test (Section 8) will fail on the prior implementation because `cos(mean_i, mean_j) ≠ mean(cos_pairs)` when the jointly-active token distribution is non-uniform.

**Total token count**: after each batch `b` at layer `l`, call `record_batch_token_count(l, T_b)` where `T_b = batch.numel() // seq_len` (the actual token count from the batch tensor, NOT `Σ_e token_counts_e`). This mirrors `activation_hooks.py:211-221`.

## 5. Partial-Hit Policy (Bug #4 Fix)

**Decision: skip ream_acc hydration on partial hit. Justification:**

A "partial hit" occurs when the sidecar was written from fewer batches than the current calibration run (e.g., sidecar from 1000 prompts, current run has 2000). On partial hit:

- `gate_logit_profiles[l]` covers only the partial token set: the δ_gate similarity matrix would be computed over fewer tokens than the live run expects, silently biasing results.
- `sim_tensor[l]` is the partial sum: adding new batches to it via `finalize_batch()` would give `Σ_partial + Σ_new_batches` — which is NOT the same as the full sum over all prompts.

The "either full live or full skip" invariant is cleaner and safer:
- On **full hit**: reader hydrates `ream_acc.gate_logit_profiles[l]`, `ream_acc._sim_tensor[l]`, `ream_acc._total_tokens_by_layer[l]`, `ream_acc._neuron_act_sum`, `ream_acc._neuron_act_count`, plus `cov_acc.covariance` + `cov_acc.token_count` for the current layer's `(layer_idx, *, *)` keys, plus `layer_input_acc.buffer` for the current layer **when `layer_input_acc is not None`** (it is None on runs where neither `expert_distill_steps>0` nor `cost_alignment="output"`; see §10 hydration code N-3 guard). `LayerMergePlugin.on_profile` skips the forward pass for this layer (Pattern A).
- On **partial hit** or **miss**: reader sets nothing. `LayerMergePlugin.on_profile` runs the full live forward pass as usual.

"Partial hit" detection: the reader compares `payload.total_tokens_per_layer[layer_rank]` against zero and against the expected token count from the live calibration spec (`n_batches × batch_token_count`). If the sidecar token count for any layer is less than `0.5 × expected`, it is classified as partial. The layer-specific partial flag is stored in `ctx` as `stage2_profile_partial_hit`.

**ctx slot names:**
- `stage2_profile_full_hit: bool` — True when this layer is a full hit
- `stage2_profile_partial_hit: bool` — True when partial (triggers live path)

`LayerMergePlugin.on_profile` checks `ctx.get("stage2_profile_full_hit", False)` and early-returns if True.

## 6. Driver Wiring (Bug #5 Fix)

The wiring pattern follows `--capture-input-covariance` exactly. Five insertion points in `build_self_traces_calib_vllm.py`.

**Capture policy (per Crit-3 resolution)**: `--capture-stage2-profile` is a single binary flag that turns on capture of ALL the payload fields, including `layer_input_reservoir`. No sub-flags. When operators opt in, the reservoir cost (~3 GB at default `max_samples=8192`, up to ~15 GB if the per-layer token cap is raised for SC experiments) is part of the deal — the SC strategy depends on it (`cost_alignment="output"` reads `layer_input_acc.buffer`) and gating it independently would create a latent failure mode where SC silently picks up empty reservoirs.

### 6.1 Argparse block (insert after line 708, before `--capture-per-expert-max`)

```python
p.add_argument("--capture-stage2-profile", action="store_true", default=False,
               help="Capture Stage 2 REAM profile (gate logits + gated outputs "
                    "+ covariance + layer-input reservoir) and write a sidecar at "
                    "<jsonl>/sidecars/stage2_profile.pt (schema v3). "
                    "Requires the vLLM patch vllm.calibration_stage2_profile. "
                    "Auto-enables VLLM_CALIB_CAPTURE_STAGE2_PROFILE=1 + "
                    "VLLM_CALIB_CAPTURE_ROUTER=1 + "
                    "VLLM_CALIB_CAPTURE_EXPERT_UNWEIGHTED=1 + "
                    "VLLM_USE_FLASHINFER_MOE_FP16=0 BEFORE any vllm import. "
                    "Implies layer-input reservoir capture (~3 GB default for "
                    "Qwen3.6-35B-A3B; up to ~15 GB with elevated token_cap; "
                    "required for SC cost_alignment='output'). "
                    "Failures during dump are logged but do NOT re-raise.")
p.add_argument("--stage2-profile-checkpoint-every-chunks", type=int, default=1,
               help="When --capture-stage2-profile is set, dump a checkpoint "
                    "(.stage2_profile.ckpt) every N chunks. Default 1. Set 0 "
                    "to disable. On --resume, checkpoint is hydrated automatically.")
p.add_argument("--stage2-profile-cov-storage-dtype", type=str, default="float16",
               choices=["float16", "bfloat16", "float32"],
               help="When --capture-stage2-profile is set, configure the "
                    "InputCovarianceAccumulator.storage_dtype used by the "
                    "writer. MUST MATCH the Stage 2 config's "
                    "`covariance_storage_dtype` (default 'float16' per "
                    "orchestrator.py:702). On mismatch the reader fails "
                    "loud at load time with 'Delete the sidecar to "
                    "regenerate'. Operators may legitimately choose fp32 "
                    "for higher-precision experiments; the value chosen "
                    "here MUST be propagated to the Stage 2 YAML.")
```

No sub-flag for reservoir capture: it is implied by `--capture-stage2-profile`.

### 6.2 Pre-import env-var block (insert after line 978, after routing-stats block)

```python
if args.capture_stage2_profile:
    os.environ["VLLM_CALIB_CAPTURE_STAGE2_PROFILE"] = "1"
    os.environ["VLLM_CALIB_CAPTURE_ROUTER"] = "1"
    os.environ["VLLM_CALIB_CAPTURE_EXPERT_UNWEIGHTED"] = "1"
    os.environ["VLLM_USE_FLASHINFER_MOE_FP16"] = "0"
    log.info("--capture-stage2-profile: enabled env vars (must precede vllm import)")
```

### 6.3 Setup block (insert after line 1281, after input-cov setup block)

```python
if args.capture_stage2_profile:
    import vllm.calibration_stage2_profile as _s2p  # type: ignore
    _s2p.setup(llm, cov_storage_dtype=args.stage2_profile_cov_storage_dtype)
    log.info("stage2-profile: setup complete -- router + expert_out_unweighted "
             "callbacks registered; cov_storage_dtype=%s",
             args.stage2_profile_cov_storage_dtype)
    s2p_ckpt_path = out_path.with_suffix(".stage2_profile.ckpt")
    if args.resume and s2p_ckpt_path.exists():
        try:
            loaded_prompts = _s2p.load_stage2_profile_checkpoint(str(s2p_ckpt_path))
            if loaded_prompts != already_done:
                log.warning("stage2-profile: checkpoint has %d prompts but JSONL has %d rows",
                            loaded_prompts, already_done)
            else:
                log.info("stage2-profile: hydrated %d-prompt checkpoint from %s",
                         loaded_prompts, s2p_ckpt_path)
        except ValueError as exc:
            log.error("stage2-profile: checkpoint schema mismatch (%s); deleting", exc)
            s2p_ckpt_path.unlink()
```

### 6.4 Periodic checkpoint block (insert after line 1693, after input-cov checkpoint block)

```python
if (args.capture_stage2_profile
        and args.stage2_profile_checkpoint_every_chunks > 0):
    chunk_idx = chunk_start // args.chunk_size
    every = args.stage2_profile_checkpoint_every_chunks
    if (chunk_idx + 1) % every == 0:
        try:
            import vllm.calibration_stage2_profile as _s2p  # type: ignore
            _s2p.set_n_prompts_accumulated(already_done + n_new)
            _s2p.dump_stage2_profile_checkpoint(str(s2p_ckpt_path))
            log.info("stage2-profile: checkpointed %d prompts -> %s",
                     already_done + n_new, s2p_ckpt_path)
        except Exception as exc:
            log.error("stage2-profile checkpoint failed: %s", exc, exc_info=True)
```

### 6.5 Final dump block (insert after line 1869, after input-cov dump block)

```python
if args.capture_stage2_profile:
    try:
        import vllm.calibration_stage2_profile as _s2p  # type: ignore
        _s2p.set_n_prompts_accumulated(already_done + n_new)
        _s2p.dump_stage2_profile(out_path)
        log.info("stage2-profile: dumped sidecar from %d prompts (next to %s)",
                 _s2p.get_n_prompts_accumulated(), out_path)
        if s2p_ckpt_path is not None and s2p_ckpt_path.exists():
            s2p_ckpt_path.unlink()
    except Exception as exc:
        log.error("stage2-profile dump failed: %s", exc, exc_info=True)
```

**Why these five insertions and not more**: mirrors exactly the `--capture-input-covariance` pattern which has these same five locations (args ~685, env ~929, setup ~1252, periodic ~1674, dump ~1857). No other wiring points are needed.

## 7. Reader Pairing Enforcement (Bug #8 Fix)

**Single config knob**: `stage2_reap_ream.profile_sidecar.enabled: bool` (default `false`).

**Restated scope of Bug #8**: the prior Plugin #12 shipped TWO readers — `Stage2ProfileCacheProvider` (this plan's reader) and `Stage2InputCovCacheProvider` (a separate reader that hydrated the cov sidecar produced by the independent `--capture-input-covariance` feature). The historic failure mode was a configuration where an operator registered `Stage2InputCovCacheProvider` alone, which caused the cov sidecar to be hydrated but `LayerMergePlugin.on_profile` to STILL run the full live forward — double-counting input covariance into `cov_acc`.

**This redo addresses Bug #8 structurally, not by hard-failing**: the `Stage2InputCovCacheProvider` was deleted along with the rest of the prior Plugin #12 and is NOT being re-introduced by this plan. Only ONE reader exists going forward — `Stage2ProfileCacheProvider` — registered behind `stage2_reap_ream.profile_sidecar.enabled`. The reader hydrates `cov_acc` directly from `payload.cov_acc` (which is captured by the same writer that produces `payload.sim_tensor`, etc. — one writer, one sidecar, one reader). There is no config path that registers an input-cov reader alone, so the double-counting failure mode is structurally impossible.

The independent `--capture-input-covariance` feature continues to exist as a writer producing its own cov sidecar consumed elsewhere; this plan does NOT register a Stage-2 reader for it. The two features are decoupled.

**Cache-miss behavior**: when `profile_sidecar.enabled=True` but the sidecar file does not exist at expected path, the provider's `on_load` returns `None` and the live path runs — this is a cache miss, NOT an error. The operator gets a log warning.

**Orchestrator registration** (insert AFTER `layer_merge` at line ~1137 in `orchestrator.py`):

The new provider MUST be registered AFTER `LayerMergePlugin` so its `on_layer_setup` runs SECOND (per OQ-1 Option A — see §10). `LayerMergePlugin.on_layer_setup` constructs the fresh `ream_acc` / `layer_input_acc` and writes them to ctx; the reader then hydrates those in-place. Registering BEFORE `LayerMergePlugin` would mean the fresh empty accumulators overwrite the hydrated ones.

NOTE: `Stage2RoutingStatsCacheProvider` (line 1085) is BEFORE `LayerMergePlugin` (line 1137). Do NOT use that position — registration must be AFTER `layer_merge`. A reasonable concrete position is immediately after the `layer_merge,` entry at line 1137 (i.e., between the merge spine and `ExpertDistillPlugin`).

```python
# Profile-sidecar cache reader (Optimization A). Single-knob registration:
# the same flag (`profile_sidecar.enabled`) governs gate-logit hydration,
# cov_acc hydration, and layer_input_acc hydration — all from ONE sidecar
# written by --capture-stage2-profile.
# On full hit: hydrates ream_acc + cov_acc + layer_input_acc so
# LayerMergePlugin.on_profile skips the forward pass.
# On partial hit or miss: no-op; live path runs unchanged.
# Bug #8: only this single reader is ever registered for the profile sidecar.
# The prior Stage2InputCovCacheProvider was deleted and is not re-introduced.
*(
    [Stage2ProfileCacheProvider(cov_acc=cov_acc)]
    if s2.get("profile_sidecar", {}).get("enabled", False)
    else []
),
```

The explicit-loop `on_load` call (mirroring the `Stage2RoutingStatsCacheProvider` precedent at lines 1200-1203) must be added after the `dispatch_first` block (i.e., AFTER line 1203, the existing routing-stats explicit-loop terminator):

```python
# Stage2ProfileCacheProvider.on_load must run explicitly (not via dispatch_first)
# because dispatch_first("on_load", ...) at line 1188 stops at the first non-None
# result. If Stage2ReapScoresCacheProvider hits, this provider's on_load is
# never reached via the chain. Explicit-loop pattern per routing-stats precedent.
for _plug in plugins:
    if isinstance(_plug, Stage2ProfileCacheProvider):
        _plug.on_load(run_ctx, _calib_jsonl_path)
        break
```

## 8. Test Plan

### 8.1 Writer math correctness test (must fail on prior bug-#1 implementation)

**File**: `max_quality/tests/test_stage2_profile_sidecar_writer_math.py`

**Setup**: Construct a synthetic mini-MoE with `n_layers=2`, `n_experts=4`, `top_k=2`, `d_hid=16`, `d_intermediate=32`. Generate `n_batches=3` batches of `T=8` tokens each.

**Step 1** — drive `vllm.calibration_stage2_profile`'s callbacks directly (no vLLM required): call `_s2p._on_router_callback(layer_idx, logits, batch_offset)` and `_s2p._on_expert_out_unweighted_callback(layer_idx, expert_idx, gated, token_indices, batch_offset)` with known synthetic inputs.

**Step 2** — produce reference via live `ReamCostAccumulator`: for each batch, call `acc.record_router_logits(...)`, `acc.record_gated_output(...)`, `acc.finalize_batch(...)`, `acc.record_batch_token_count(...)` with the identical synthetic inputs.

**Step 3** — call `_s2p.dump_stage2_profile(tmp_path)`, load via `load_stage2_profile_v3(tmp_path)`.

**Step 4** — construct a fresh `ReamCostAccumulator`, hydrate it from the loaded payload (same hydration logic the reader plugin uses). Assert byte-identity:
- `payload.sim_tensor[layer_rank]` == `reference_acc._sim_tensor[layer_idx]` (fp64, tolerance 1e-10 for float32 accumulation drift)
- `payload.total_tokens_per_layer[layer_rank]` == `reference_acc._total_tokens_by_layer[layer_idx]`
- `payload.gate_logit_profiles[layer_rank]` is a list of `(int, tensor)` tuples whose contents match `reference_acc.gate_logit_profiles[layer_idx]` element-for-element (same offsets, same tensors)
- After hydration: `reader_acc.compute_gate_similarity_matrix(layer_idx, all_expert_ids)` tensor-equal to `reference_acc.compute_gate_similarity_matrix(layer_idx, all_expert_ids)`

**Step 5** — assert the test FAILS if the writer is replaced with the buggy prior implementation (mean-of-gated-vectors cosine): use a fixture where `mean(cos(pair_tokens)) ≠ cos(mean(v_i), mean(v_j))`, which holds whenever tokens' gated vectors are not all parallel. Include a comment explicitly stating which fixture triggers the bug.

**Step 6** — end-to-end roundtrip: feed the loaded payload into `Stage2ProfileCacheProvider` via `on_layer_setup(ctx)`, verify `ctx.get("ream_acc")` has `_sim_tensor[layer_idx]` equal to reference.

### 8.2 Partial-hit skip test

**File**: `max_quality/tests/test_stage2_profile_sidecar_partial_hit.py`

Construct a sidecar with `total_tokens_per_layer = [100, 100]` (small). Drive `Stage2ProfileCacheProvider.on_layer_setup` with an expected token count of 500. Assert `ctx.get("stage2_profile_full_hit", False) == False` and `ctx.get("stage2_profile_partial_hit", False) == True`. Assert `ream_acc` is NOT hydrated (it is a fresh `ReamCostAccumulator()` from `LayerMergePlugin.on_layer_setup`).

### 8.3 Bug #3 regression test (total_tokens off by top_k)

In the writer-math test (8.1), verify `payload.total_tokens_per_layer[layer_rank]` == `T_batch * n_batches` (exact token count), NOT `T_batch * n_batches * top_k`. This fails on the prior implementation where `_total_tokens_by_layer` was summed from per-expert token_counts.

### 8.4 Sidecar round-trip byte identity (no vLLM)

Construct `Stage2ProfilePayloadV3` in pure Python with synthetic tensors. Call `save_stage2_profile_v3`, then `load_stage2_profile_v3`. Assert all tensor fields are byte-identical (`torch.equal`). Assert `schema_version == 3`. Specifically assert `gate_logit_profiles` roundtrip preserves the `(int, tensor)` tuple structure.

### 8.5 cov_storage_dtype cross-validation test

Construct a sidecar with `cov_storage_dtype="float16"`. Mock the run's `s2.covariance_storage_dtype` as `"bfloat16"`. Assert `load_stage2_profile_v3`-then-validate raises `ValueError` with the "Delete the sidecar to regenerate" message.

## 9. Files to Touch (Exhaustive)

### New files

- `/home/lucas/ai/moe_compress/max_quality/src/moe_compress/stage2/plugins/stage2_profile_cache.py`
  Stage2ProfileCacheProvider class. Hooks: `on_load`, `on_layer_setup`. Bug #4 partial-hit logic here.

- `vllm/calibration_stage2_profile.py` (vLLM patch — lives in the vLLM wheel, NOT in this repo)
  Writer module. Public API: `setup(llm)`, `dump_stage2_profile(jsonl_path)`, `dump_stage2_profile_checkpoint(path)`, `load_stage2_profile_checkpoint(path)`, `set_n_prompts_accumulated(n)`, `get_n_prompts_accumulated()`. Internal: `_on_router_callback`, `_on_expert_out_unweighted_callback`, `_finalize_batch_for_layer`. Bug #1 fix lives here. The writer also captures the layer-input reservoir unconditionally (one entry per MoE layer rank, len == n_layers).

- `max_quality/tests/test_stage2_profile_sidecar_writer_math.py` (Section 8.1)
- `max_quality/tests/test_stage2_profile_sidecar_partial_hit.py` (Section 8.2 + 8.3)
- `max_quality/tests/test_stage2_profile_sidecar_roundtrip.py` (Section 8.4 + 8.5)

### Modified files

- `/home/lucas/ai/moe_compress/max_quality/src/moe_compress/utils/cached_calibration_signals.py`
  - Replace `Stage2ProfilePayload` (v1) with `Stage2ProfilePayloadV3` (Section 3 above). No alias retention — the prior v1 dataclass had no production writer, so no callers exist. (Per Low-8: drop the v1 alias entirely.)
  - `SCHEMA_VERSIONS["stage2_profile"]` → 3
  - Add `save_stage2_profile_v3(payload, jsonl_path)` and `load_stage2_profile_v3(jsonl_path)`
  - `load_stage2_profile_v3` performs the `cov_storage_dtype` cross-validation described in §3 and §8.5 against the active run's configured dtype

- `/home/lucas/ai/moe_compress/max_quality/src/moe_compress/stage2/orchestrator.py`
  - Import `Stage2ProfileCacheProvider` from `stage2.plugins.stage2_profile_cache`
  - Add to `PluginRegistry([...])` AFTER `layer_merge` at line ~1137 (i.e., between the merge spine and `ExpertDistillPlugin`), behind the `profile_sidecar.enabled` gate (Section 7). This satisfies the OQ-1 Option A invariant: `Stage2ProfileCacheProvider.on_layer_setup` must run SECOND so the in-place hydration sees the fresh `ream_acc` / `layer_input_acc` that `LayerMergePlugin.on_layer_setup` constructs.
  - Add explicit `on_load` loop AFTER line 1203 (mirroring the `Stage2RoutingStatsCacheProvider` precedent at lines 1200-1203). See §7.

- `/home/lucas/ai/moe_compress/max_quality/src/moe_compress/stage2/plugins/layer_merge.py`
  - `on_profile()`: add early-return guard at top: `if ctx.get("stage2_profile_full_hit", False): return` (Pattern A, cache-aware skip)
  - `on_layer_teardown()`: add `ctx.set("stage2_profile_full_hit", None, overwrite=True)` and `ctx.set("stage2_profile_partial_hit", None, overwrite=True)` to nullify per-layer cache slots

- `/home/lucas/ai/moe_compress/max_quality/scripts/build_self_traces_calib_vllm.py`
  - Five insertion points per Section 6 (argparse, env-var, setup, periodic checkpoint, final dump)

- `max_quality/patches/MANIFEST.md` (if it exists — add entry for new patch + schema v3 bump)

## 10. Data Flow

```
vLLM calibration run:
  for each prompt batch b:
    router fires → _on_router_callback(layer_idx, logits[T_b, E], offset)
                    → gate_logit_profiles[layer_idx].append((offset, logits_cpu))
                       (mirrors activation_hooks.py:172-173 exactly)
    experts fire → _on_expert_out_unweighted_callback(layer_idx, e, gated[T_e,d], indices, offset)
                    → _batch_gated_indexed[(layer_idx, e)] = (indices, gated)
    layer-input capture (always on per §6 policy):
                    → layer_input_reservoir[layer_rank] += sampled rows
    after all experts fire → _finalize_batch_for_layer(layer_idx, n_experts)
                               [mirrors ReamCostAccumulator.finalize_batch() exactly]
                               → sim_tensor[layer_rank] += sim_sum_f64
                    → record_batch_token_count(layer_rank, T_b)
                               → total_tokens_per_layer[layer_rank] += T_b
  at run end:
    → for each layer_rank l: cov_acc.finalize_layer(l_to_layer_idx[l])
       (drains pending GPU covariances to CPU storage_dtype tensors)
    → construct Stage2ProfilePayloadV3:
         cov_acc           = dict(cov_acc.covariance)         # finalized
         cov_token_count   = dict(cov_acc.token_count)
         cov_storage_dtype = str(cov_acc.storage_dtype).split(".")[-1]
         (other fields per §3)
    → atomic torch.save to sidecars/stage2_profile.pt

Stage 2 run (with profile_sidecar.enabled=true):
  orchestrator.run():
    → Stage2ProfileCacheProvider.on_load(run_ctx, jsonl_path)
       → load_stage2_profile_v3(jsonl_path) → Stage2ProfilePayloadV3
       → cross-validate payload.cov_storage_dtype against
         s2.get("covariance_storage_dtype", "float16"); fail loud on mismatch
       → ctx.set("stage2_profile_payload", payload)
    for each layer l (layer_rank == ctx.get("_layer_rank"),
                       layer_idx  == ctx.get("layer_ref").layer_idx):
      → LayerMergePlugin.on_layer_setup(ctx)
         → constructs fresh ream_acc, layer_input_acc; writes to ctx
      → Stage2ProfileCacheProvider.on_layer_setup(ctx)  [runs AFTER LayerMergePlugin]
         → check payload.total_tokens_per_layer[layer_rank] vs expected
         → on full hit:
             ream_acc        = ctx.get("ream_acc")          # in-place hydration
             layer_input_acc = ctx.get("layer_input_acc")   # in-place hydration
             # gate logit profiles preserved as list[(offset, tensor)]
             ream_acc.gate_logit_profiles[layer_idx] = list(
                 payload.gate_logit_profiles[layer_rank]
             )
             ream_acc._sim_tensor[layer_idx]            = payload.sim_tensor[layer_rank].clone()
             ream_acc._total_tokens_by_layer[layer_idx] = int(payload.total_tokens_per_layer[layer_rank].item())
             for (lr, e), v in payload.neuron_act_sum.items():
                 if lr == layer_rank:
                     ream_acc._neuron_act_sum[(layer_idx, e)] = v.clone()
             for (lr, e), c in payload.neuron_act_count.items():
                 if lr == layer_rank:
                     ream_acc._neuron_act_count[(layer_idx, e)] = int(c)
             # cov_acc hydration — direct dict-write into finalized storage
             # (NOT cov_acc.update(...), which expects the raw input matrix `x`).
             # See OQ-3 (resolved): writer pre-finalized; reader writes finalized
             # entries directly into the same dicts.
             for (lr, e, m), cov_t in payload.cov_acc.items():
                 if lr == layer_rank:
                     cov_acc.covariance[(layer_idx, e, m)] = cov_t.to(
                         cov_acc.storage_dtype
                     )
             for (lr, e, m), n in payload.cov_token_count.items():
                 if lr == layer_rank:
                     cov_acc.token_count[(layer_idx, e, m)] = int(n)
             # layer-input reservoir hydration (Crit-3 fix; required for SC).
             # GUARD: layer_input_acc is None when this run does NOT need it
             # (LayerMergePlugin.on_layer_setup at layer_merge.py:457-464
             # sets it to None when both expert_distill_steps==0 AND
             # cost_alignment != "output"). On those configs the reservoir
             # data persists in the sidecar but is unused by this run —
             # acceptable, since the operator opted into capture via
             # --capture-stage2-profile.
             if layer_input_acc is not None:
                 layer_input_acc.buffer = payload.layer_input_reservoir[layer_rank].clone()
                 layer_input_acc.seen = int(payload.layer_input_reservoir[layer_rank].size(0))
             ctx.set("stage2_profile_full_hit", True)
         → on partial hit or miss:
             ctx.set("stage2_profile_partial_hit", True)
             # ream_acc, cov_acc, layer_input_acc left as the fresh empty
             # accumulators that LayerMergePlugin.on_layer_setup created
      → LayerMergePlugin.on_profile(ctx)
         → if ctx.get("stage2_profile_full_hit", False): return  (Pattern A skip)
         → otherwise: full live forward as today
      → downstream plugins (write_artifacts, etc.) consume the populated
        cov_acc + ream_acc fields normally; see §12 coverage table
```

**OQ-1 RESOLVED (was open in v1 of this plan)**:

`LayerMergePlugin.on_layer_setup` (line 425-468 in `layer_merge.py`) constructs fresh `ReamCostAccumulator()` and `_LayerInputAccumulator()` and writes them to ctx. If `Stage2ProfileCacheProvider.on_layer_setup` runs BEFORE LayerMergePlugin, those fresh objects overwrite the hydrated ones.

**Resolution: Option A (in-place hydration)**. Register `Stage2ProfileCacheProvider` AFTER `LayerMergePlugin` in the `PluginRegistry`. `walk_phases` processes plugins in registration order, so `Stage2ProfileCacheProvider.on_layer_setup` runs second. The reader calls `ctx.get("ream_acc")` / `ctx.get("layer_input_acc")` and hydrates them IN-PLACE (populating internal dicts / setting `.buffer`) — not replacing the objects. This avoids the `overwrite=True` side-effect of implicitly marking the slot as "already set" for downstream plugins.

`cov_acc` is a run-scope shared object constructed once at `orchestrator.run()` line 694; it is NOT re-created per-layer, so ordering against `LayerMergePlugin.on_layer_setup` is irrelevant for it — the reader writes into the same shared instance.

## 11. Build Sequence (Phased Checklist)

### Phase 1 — Schema + IO (no tests yet)

- [ ] Replace `Stage2ProfilePayload` (v1) with `Stage2ProfilePayloadV3` in `cached_calibration_signals.py` (no v1 alias kept)
- [ ] Bump `SCHEMA_VERSIONS["stage2_profile"]` to 3
- [ ] Add `save_stage2_profile_v3` / `load_stage2_profile_v3` (with cov_storage_dtype cross-validation)

### Phase 2 — Reader plugin (no vLLM dependency)

- [ ] Implement `stage2/plugins/stage2_profile_cache.py` — `Stage2ProfileCacheProvider`
- [ ] Hooks: `on_load`, `on_layer_setup` (with partial-hit detection per Section 5, hydration per §10)
- [ ] Register in `orchestrator.py` AFTER `LayerMergePlugin` in plugin list, explicit on_load loop
- [ ] Modify `layer_merge.py` `on_profile` early-return guard
- [ ] Modify `layer_merge.py` `on_layer_teardown` to null the full_hit/partial_hit ctx slots

### Phase 3 — Tests (no vLLM)

- [ ] `test_stage2_profile_sidecar_roundtrip.py` — schema v3 save/load byte identity + cov_storage_dtype cross-validation (8.4 + 8.5)
- [ ] `test_stage2_profile_sidecar_partial_hit.py` — partial-hit skip logic (8.2 + 8.3)
- [ ] `test_stage2_profile_sidecar_writer_math.py` — bug #1 reference test (synthetic callbacks) (8.1)
  - Drive internal writer callbacks directly (not via vLLM)
  - Compare sim_tensor + gate_logit_profiles to reference `ReamCostAccumulator` output
  - Assert test fails on buggy mean-of-means implementation
- [ ] Run full test suite to confirm zero regressions

### Phase 4 — Driver wiring (build_self_traces_calib_vllm.py)

- [ ] Add argparse flags (Section 6.1)
- [ ] Add env-var block (Section 6.2)
- [ ] Add setup block (Section 6.3)
- [ ] Add periodic checkpoint block (Section 6.4)
- [ ] Add final dump block (Section 6.5)

### Phase 5 — vLLM writer patch (separate from Phase 1-4; depends on GPU host)

- [ ] Implement `vllm/calibration_stage2_profile.py`
- [ ] Wire to `vllm/calibration_hooks.py` callback dispatch (VLLM_CALIB_CAPTURE_STAGE2_PROFILE env gate)
- [ ] `setup(llm, cov_storage_dtype: str)`: parse the dtype string, construct the writer's `InputCovarianceAccumulator`, immediately call `set_storage_dtype(torch.<dtype>)`. Store the configured string on the module for cross-validation at dump time.
- [ ] Integrate with `finalize_batch` logic (mirror `ReamCostAccumulator.finalize_batch` exactly)
- [ ] Pre-finalize cov_acc at dump time: `for l in range(n_layers): cov_acc.finalize_layer(layer_idx_of_rank[l])` BEFORE constructing the payload (so the persisted dict is the finalized storage, per OQ-3 resolution)
- [ ] Writer-side dump-time assert: `assert str(cov_acc.storage_dtype).split(".")[-1] == configured_cov_storage_dtype` — raises loud on mismatch
- [ ] Smoke test on CPU-only dummy model
- [ ] Update `max_quality/patches/MANIFEST.md`

### Phase 6 — Integration (requires GPU)

- [ ] Run `build_self_traces_calib_vllm.py --capture-stage2-profile` on small calibration set
- [ ] Verify sidecar schema version, shape, total_tokens values, cov_storage_dtype string
- [ ] Run Stage 2 with `profile_sidecar.enabled=true` against the sidecar
- [ ] Confirm full-hit path triggers (log "stage2_profile_full_hit=True" for every layer)
- [ ] Confirm δ_gate and δ̃_expert values match a baseline Stage 2 run on the same calibration set
- [ ] Confirm SC strategy (cost_alignment="output") consumes layer_input_acc.buffer correctly
- [ ] Measure wall-clock savings on SC strategy row

## 12. Critical Details

### Error handling

- `load_stage2_profile_v3`: returns `None` on file-not-found (cache miss, graceful). Raises `ValueError` on schema_version mismatch with "Delete the sidecar to regenerate" message per `_check_schema` contract. Raises `ValueError` on `cov_storage_dtype` mismatch with the same "Delete the sidecar to regenerate" message.
- Writer dump failures: wrapped in `try/except Exception`, logged at ERROR level, never re-raised — the JSONL is the primary deliverable.
- Checkpoint schema mismatch on `--resume`: delete stale checkpoint, restart from zero (mirrors imatrix / reap-scores pattern).

### State management — reader-side

- The `Stage2ProfileCacheProvider` stores `payload: Stage2ProfilePayloadV3 | None` as a run-scope attribute set by `on_load`. Accessed in `on_layer_setup` via `self.payload`.
- `gate_logit_profiles` stored in the sidecar as `dict[int, list[tuple[int, Tensor]]]` where the dict key is `layer_rank` (0-based). The reader translates: `layer_rank = ctx.get("_layer_rank")` (already set by orchestrator at line 1227), `layer_idx = ctx.get("layer_ref").layer_idx`. The list of `(offset, tensor)` tuples is preserved verbatim — handed off to `ream_acc.gate_logit_profiles[layer_idx]` so that `compute_gate_similarity_matrix` (which unpacks via `for _, t in batches`) consumes it exactly as it would for live capture.
- `cov_acc` is a run-scope shared object (constructed once in `orchestrator.run()` at line 694). On full hit, the reader writes **directly into the finalized storage dicts**:
  ```python
  cov_acc.covariance[(layer_idx, expert_idx, matrix_name)] = payload_tensor.to(cov_acc.storage_dtype)
  cov_acc.token_count[(layer_idx, expert_idx, matrix_name)] = int(n_tokens)
  ```
  NOT via `cov_acc.update(...)`. The live `update(layer_idx, expert_idx, matrix_name, x: Tensor)` API (activation_hooks.py:971) accepts a RAW input matrix `x` and computes `x.T @ x` into `_pending`; passing a finalized covariance there would (a) cause a shape/type mismatch, (b) put data on the wrong (pending vs finalized) accumulator path. Direct dict-write is the correct primitive — this is the same path `InputCovarianceAccumulator.finalize_layer` (line 1063-1072) uses internally.
- The `cov_acc.finalize_layer(layer_ref.layer_idx)` call in `LayerMergePlugin.on_profile` line 496 MUST ALSO be skipped on full hit. The single early-return guard `if ctx.get("stage2_profile_full_hit", False): return` at the top of `on_profile` covers this — the forward pass AND `finalize_layer` are in the same method body.

### State management — writer-side serialization order

The writer MUST follow this exact order at dump time:

1. Drain all pending covariances to finalized storage:
   ```python
   for l in range(n_layers):
       cov_acc.finalize_layer(layer_idx_of_rank[l])
   ```
2. Snapshot the finalized dicts into the payload:
   ```python
   payload.cov_acc          = dict(cov_acc.covariance)
   payload.cov_token_count  = dict(cov_acc.token_count)
   payload.cov_storage_dtype = str(cov_acc.storage_dtype).split(".")[-1]  # "float16" etc.
   ```
3. Translate the dict keys' `layer_idx → layer_rank` as part of step 2 (the writer must store layer_rank-keyed dicts; this mirrors the storage convention in §3).
4. Build the rest of the payload (sim_tensor, gate_logit_profiles, etc.) and `torch.save`.

### State management — write_artifacts coverage

On full hit, `LayerMergePlugin.on_profile` is skipped → no `cov_acc.finalize_layer` call → `cov_acc` must be pre-populated by the reader. Confirm each downstream consumer in `write_artifacts` reads only the fields the reader hydrates:

Function definitions live in `stage2/shared_io.py`; the call sites live inside `LayerMergePlugin.write_artifacts` in `stage2/plugins/layer_merge.py`. Both anchors are given below — the definition for understanding the function, the call site for the dispatch context. (There is no `write_artifacts.py` file.)

| Consumer | Definition | Call site | Reads | Hydrated by reader? |
|---|---|---|---|---|
| `_snapshot_cov_layer` | `stage2/shared_io.py:64` | `layer_merge.py:621` | `cov_acc.covariance[(layer_idx, e, name)]` for current layer's experts | YES — reader writes these on full hit |
| `_snapshot_neuron_means_layer` | `stage2/shared_io.py:85` | `layer_merge.py:625` | `ream_acc._neuron_act_sum[(layer_idx, e)]`, `ream_acc._neuron_act_count[(layer_idx, e)]` | YES — reader writes these (after layer_rank→layer_idx translation) |
| `_remap_covariance_for_layer` | `stage2/shared_io.py:202` | `layer_merge.py:618` | Mutates `cov_acc.covariance` keys for current layer in place | YES — works on hydrated entries; remap is key-rename, content-agnostic |
| `compute_gate_similarity_matrix` | `utils/activation_hooks.py:475+` | (called from REAM cost path) | `ream_acc.gate_logit_profiles[layer_idx]` (list of tuples) | YES — reader writes the verbatim list-of-tuples |
| SC `_output_space_cost` (cost_alignment="output") | (see `OutputSpaceCostPlugin`) | (called from cost dispatch) | `layer_input_acc.buffer` | YES (when `layer_input_acc is not None`) — reader writes `payload.layer_input_reservoir[layer_rank]` into `.buffer`; on configs where `layer_input_acc is None` (no expert_distill, no output cost) the reservoir is left in the sidecar unused, which is intentional |

All consumers' read sets are covered by the reader's hydration set.

### Performance

- `gate_logit_profiles` tensors are CPU fp32, stored as a list of `(int, tensor)` tuples. The sidecar serializes them as-is in `torch.save`. On load they are immediately assigned into `ream_acc.gate_logit_profiles[l]` — no copy, no concatenation at load time (the reader just assigns the list reference; concatenation happens lazily inside `compute_gate_similarity_matrix` when the cost matrix is actually needed).
- `sim_tensor` is `[n_layers, E, E] fp64` — for Qwen3.6-35B-A3B with `n_layers=40, E=256` this is `40 × 256 × 256 × 8 bytes = 210 MB`. Acceptable for sidecar storage.
- `layer_input_reservoir` is ~3 GB for Qwen3.6-35B-A3B at default config (`max_samples=8192, hidden=4096, bf16, n_layers=40` → `2.68 GB`); scales linearly with the per-layer token cap, up to ~15 GB when the cap is raised to ~45 K samples for some SC experiments. Operator-opted-in via `--capture-stage2-profile`.
- The periodic checkpoint path uses `dump_stage2_profile_checkpoint` which serializes the current accumulated state. On large runs this may be slow (>1 GB checkpoint). The default checkpoint-every-chunks=1 can be increased by the operator.

### Security

No user-provided code is executed. `torch.load(..., weights_only=False)` is used (existing pattern in all other signal loaders) to support non-tensor fields (dict keys, ints). The `weights_only=True` restriction would break on non-tensor payload fields; this is an accepted deviation from the strict weights-only contract in this codebase.

## 13. Workflow Contract

This plan will be reviewed by a separate plan-reviewer agent BEFORE the implementer is spawned. The reviewer should specifically check:

1. The OQ-1 resolution (Option A) — confirm in-place hydration is safe when `ReamCostAccumulator` internal dicts are populated directly.
2. The `layer_rank → layer_idx` translation in every hydration path — confirm no off-by-one.
3. The `cov_acc.finalize_layer` skip on full hit — confirm this is correct (finalize_layer is idempotent? No — it moves data from in-progress to finalized storage; skipping it on a pre-populated layer is correct because the data is already finalized in the sidecar).
4. The plugin registration order: `Stage2ProfileCacheProvider` AFTER `LayerMergePlugin`. Confirm walk_phases correctly processes on_layer_setup in plugin-minor order within each phase.
5. The `on_profile` early-return guard in `LayerMergePlugin`: confirm this skips BOTH the forward pass AND `cov_acc.finalize_layer(layer_ref.layer_idx)` (they are both in the same method body at lines 473-496).
6. The cov hydration uses **direct dict-write into `cov_acc.covariance`**, not `cov_acc.update(...)`. The live `update` API takes a raw `x: Tensor` and accumulates `xᵀx` into `_pending`, which is the wrong path for a pre-finalized payload.

### Resolved decisions (previously open)

- **OQ-1** (Section 10 / §11 Phase 2): Resolved as Option A — in-place hydration. Register `Stage2ProfileCacheProvider` after `LayerMergePlugin`; reader calls `ctx.get(...)` on the fresh accumulators and populates them in place.
- **OQ-2** (reservoir capture sub-flag): Resolved against an extra sub-flag. `--capture-stage2-profile` implies reservoir capture unconditionally. Storage cost ~3 GB at default config (scaling up to ~15 GB if the per-layer token cap is raised) is part of the operator's opt-in. The schema's `layer_input_reservoir` is always-present (not Optional). Rationale: SC strategy (`cost_alignment="output"`) requires the reservoir to compute `_output_space_cost`; a separate sub-flag would create a latent failure mode where SC reads an empty buffer.
- **OQ-3** (cov sidecar stores finalized vs raw batch sums): Resolved as **finalized covariance** (post-`finalize_layer`). The writer calls `cov_acc.finalize_layer(l)` for each MoE layer rank BEFORE serializing, then writes `cov_acc.covariance + cov_acc.token_count` directly into the payload. The reader writes back into the same finalized dicts in-place (NOT via `cov_acc.update`, which would re-treat the entry as a raw input matrix and corrupt the accumulator). See §12 "State management — writer-side serialization order" and "State management — reader-side" for the exact code patterns.

### Remaining open questions

None. All previously-open items are resolved above.

### Out of scope (confirmed per task brief)

- vLLM wheel rebuild and HF upload
- H200 SC-row wall-clock validation
- Plugin #13 `profile_sidecar_cache.py` (being dropped)

---
