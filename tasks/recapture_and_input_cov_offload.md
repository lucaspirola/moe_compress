# Recapture all 10 signals over 8000 + build input_cov offload

## Goal
1. Recapture the 9 cheap/medium calibration signals over the full 8000-row corpus
   via forward-only replay, write canonical sidecars, push to HF.
2. Build the input_covariance per-layer CPU-offload (windowed-resident) so the
   172 GB un-shardable Gram can be captured over 8000 without OOM, then write its
   sidecar and push to HF.

## Key findings (from reading the patch + driver, 2026-06-08)
- **No vLLM wheel rebuild needed.** input_cov accumulation in
  `moe_runner.py` is guarded per-layer by `_li in _ch._INPUT_COV_GPU`, so a
  window of allocated layers is captured and absent layers are silently
  skipped. Windowing is pure driver-side orchestration.
- `set_calibration_max_layer(hi)` (already in the wheel, `qwen3_moe.py`
  `Qwen3MoeModel.forward`) early-exits the forward after layer `hi` (inclusive).
- **Two bugs made input_cov sidecars empty before:**
  (a) driver never sets `VLLM_CALIB_INPUT_COV_MODE` -> defaults `"off"` ->
      accumulation guard `_INPUT_COV_MODE=="resident"` is always False;
  (b) `_icov.setup()` allocates ALL 40 layers resident -> 172 GB -> OOM.
  The offload path sets mode=resident AND allocates only a window.
- Architecture: 40 MoE layers, 256 experts, H=2048. Per-layer Gram =
  256*2048*2048*4 = 4.29 GB. Shared temp = 256*(buf_rows+1)*2048*4
  (buf_rows = max(cudagraph, max_num_batched_tokens, 512)). Temp scales with
  max_num_batched_tokens -> keep mbt modest (~1024-2048) on the input_cov pass
  so temp stays ~2-4 GB.

## Deliverable 1: recapture 9 signals (NO code change)
Run `_run_replay` (`--replay-from <8000.jsonl>`) with all capture flags EXCEPT
`--capture-input-covariance`:
  imatrix, reap_scores, wanda_scalar_row, stage2_profile (carries
  layer_input_reservoir), per_expert_max, routing_stats, router_logits_stats,
  output_reservoir, block_outputs.
All share the Triton non-monolithic path (forced by reap_scores). One
forward-only pass over 8000. Verify each sidecar by CONTENT (non-empty,
correct keys/shapes), then push to HF.

## Deliverable 2: input_cov windowed offload (NEW code)
New driver path `--input-cov-offload --input-cov-window-size W` (in replay mode,
only `--capture-input-covariance` set). Algorithm:
- env: VLLM_CALIB_CAPTURE_INPUT_COV=1, VLLM_CALIB_CAPTURE_EXPERT=1,
  VLLM_CALIB_INPUT_COV_MODE=resident (set before vllm import).
- Do NOT call `_icov.setup()` (it allocates all 40 layers).
- Pre-tokenize all rows once; reuse across window passes.
- Allocate ONE shared temp `[E, buf_rows+1, H]` fp32 on GPU.
- For each window [lo..hi] of MoE layer ids (size W):
  - allocate `_ch._INPUT_COV_GPU[li]=[E,H,H]`, `_INPUT_COV_COUNT_GPU[li]=[E]`,
    `_INPUT_COV_TEMP_GPU[li]=shared_temp` for li in window.
  - `set_calibration_max_layer(hi)`; forward over all rows (chunked).
  - snapshot each window layer's Gram + counts to a CPU dict; del GPU entries;
    `torch.cuda.empty_cache()`.
  - per-window disk checkpoint (resume at window granularity).
- After windows: `set_calibration_max_layer(None)`; assemble CovariancePayload;
  `save_covariance(...)` to the canonical sidecar path (same as `dump_input_cov`).
- Verify sidecar by CONTENT (40*256 entries, [2048,2048] each, counts>0).

## Cost
- D1: ~1x full forward over 8000 (forward-only prefill).
- D2: windowed -> ~ (sum of window_hi)/40 x full forward. W~11 -> 4 windows ->
  ~2.6x full forward over 8000. Single H200 (tp=1), volume 0a2fda41.
- Lever (RAISE, default = obey "over the 8000"): Gram is a 2nd-moment statistic
  and converges well before 8000; a 2-3k subset would cut D2 ~3-4x with
  negligible SVD/EoRA loss. Proceeding at 8000 unless told otherwise.

## Order
1. [code] implement `_run_input_cov_offload` + args (free, no GPU).
2. [review] review/fix loop (separate agents, all 5 categories).
3. [run] spin 1xH200, attach volume, setup_gpu_env.sh, D1 pass, verify, push HF.
4. [run] D2 windowed offload, verify, push HF.
5. teardown GPU (keep volume), verify GONE.

## Review section
(filled after implementation)
