# Implementation brief: capture down_proj input covariance via the in-graph CUDA grouped-SYRK kernel

**Hand this to an implementer agent. It is self-contained. Repo: `/home/lucas/ai/moe_compress`, production tree `max_quality/`. Read the cited files yourself before changing anything.**

## Goal
Capture the **down_proj input covariance** (the per-(layer, expert) Gram of the post-SwiGLU intermediate activation — the input to `down_proj`) during vLLM calibration, using the **accelerated in-graph CUDA grouped-SYRK kernel** (NOT an eager Python/CPU hook), and merge it into the existing `covariance.pt` sidecar so Stage 3 (SVD) and Stage 4 (EoRA) can activation-weight `down_proj`. Today only the gate/up input covariance is captured; `down_proj` falls back to plain unweighted SVD across **3** input-covariance (`A_cov`) consumers (Stage-4 EoRA, Stage-3 D-Rank whitening, Stage-3 Swift-SVD ε*), silently degrading the most important matrix. **[CORRECTED 2026-06-09: it is 3, NOT 4 — AA-SVD's rank-k factorization does NOT use `A_cov`; `_precompute_eigh` does `del A` (aa_svd_factor.py:230) and factorizes with the freshly-collected B (post-prune) + C (cross-cov) covariances. The contract is pinned by `test_aa_svd_ignores_A_factor`. So AA-SVD's down path was never A-degraded.]**

## Verified facts (do not re-litigate; verify by reading if you touch the area)
- **down_proj cov is the ONLY missing calibration input for Stage 3+4.** Cross-cov C (Theorem 3.2) and B-cov (post-prune) are computed LIVE in-stage; block_hidden is captured + correct; all other sidecars (stage2_profile/output_reservoir/per_expert_max/reap_scores/routing_stats) are NOT consumed by Stage 3/4. (Audit agent a810ba9a.)
- **The CUDA kernel already supports down — no `.cu` change.** `csrc/calibration/gram_grouped_accum.cu` (in `max_quality/patches/vllm_calibration_hooks.patch`) is dimension-agnostic ("gate/up d_in=hidden_size; down d_in=intermediate_size"), registered as torch op `torch.ops.calib_gram.gram_grouped_accum`.
- **The op is currently called from EXACTLY ONE site:** `vllm_calibration_hooks.patch:10383` inside `MoERunner._apply_quant_method`, on the MoE block input (gate/up, d_in=hidden_size). There is NO second call for the intermediate.
- **The only down/intermediate code is dead + eager:** `_expert_mid_handler` (`vllm_calibration_stage2_profile.patch:431`) is pure eager CPU (`.to("cpu")`, Python per-expert loop) AND unwired (the `expert_mid` dispatch was removed from `TritonExperts.apply` at hooks.patch:9846-9858 under the in-graph rearchitecture; stage2_profile registers no callbacks in replay mode). DO NOT revive the eager path.
- **Shard/assembler already thread `matrix_name` end-to-end** (`max_quality/src/moe_compress/utils/input_cov_offload.py:65,98,150`): `write_layer_shard(..., matrix_name=...)` and `assemble_covariance` already produce `(layer, e, "down_proj")` keys. Default is `"gate_proj"`. NO change needed to the on-disk contract.
- **Consumer side already down-ready (NO change):** the 3 `A_cov` consumers — `stage4/plugins/eora_compensation.py:537-544` keys `(layer,e,"down_proj")` (no gate fallback — by design); `stage3/plugins/d_rank_allocate.py:362-394` (whitening) + `stage3/plugins/swift_svd_alpha.py:757-778` (activation-weighted SVD) consume it. Once captured + merged, they weight down automatically. **NOTE:** `aa_svd_factor.py` `_cov_lookup` (408-415) CAN return the down key, but the factorization `del A`s it (uses B/C, not A) — AA-SVD is NOT an A_cov consumer.

## Model / dims (verified from Qwen/Qwen3.6-35B-A3B config.json → text_config)
- `hidden_size = 2048` (gate/up cov d_in), **`moe_intermediate_size = 512`** (down cov d_in), `num_experts = 256`, `num_experts_per_tok = 8`, `num_hidden_layers = 40`. Multimodal: `AutoModelForImageTextToText` → `Qwen3_5MoeForConditionalGeneration`. **Triton MoE backend** (`grouped_mm`, the calibration forces `VLLM_USE_FLASHINFER_MOE_FP16=0`).
- **down Gram is `[512, 512]` per expert** — 16× SMALLER than gate's `[2048,2048]`. Total all-resident = 40×256×512²×4B ≈ **10.7 GB fp32 / ~5.4 GB bf16**. It **fits resident** — NO windowed offload needed for down (unlike the 172GB gate Gram). This simplifies the capture: a single resident `[E,512,512]` accumulator per layer (or all 40 layers at once, 10.7GB) is fine on an H200.

## The post-SwiGLU intermediate (what to feed the down SYRK)
The down_proj input is `silu(gate_proj(x)) * up_proj(x)`, shape `[num_tokens_routed_to_expert, 512]`, computed INSIDE `TritonExperts.apply()` right before the pre-down quantize — exactly where the deleted `expert_mid` dispatch sat (`hooks.patch:9851`). This tensor is only visible inside `TritonExperts.apply` on the **Triton** backend (FlashInfer's monolithic path has no such hook). **Before implementing, confirm the target run's MoE backend exposes the intermediate at a graph-safe point** (the gate/up SYRK lives in `MoERunner._apply_quant_method`; the intermediate is a different module/scope).

## Required changes (code)
1. **Discovery** (`build_self_traces_calib_vllm.py` `_run_input_cov_offload` @3783 + the `vllm.calibration_input_cov._discover_moe_layers`): also read `moe_intermediate_size` as a second `d_in` for the down group. Today it reads only `hidden_size`.
2. **Second resident accumulator:** allocate a parallel down Gram `[E, 512, 512]` (+ its counting-sort scratch sized for the intermediate). It's small — keep it resident (no windowing). Decide: separate `_ch._INPUT_COV_DOWN_GPU[li]` or a `(group→tensor)` keyed structure. Mirror the gate/up accumulator's lifecycle.
3. **Second in-graph SYRK call** inside `TritonExperts.apply` (hooks.patch, near the old :9851 dispatch site): run the same counting-sort prologue + `gram_grouped_accum` op, but on the post-SwiGLU intermediate, **inside the compiled graph / cudagraph** (graph-safe — no Python callback, no `.to("cpu")`). This is the core change and the hard part.
4. **Offload/capture driver:** in `_run_input_cov_offload`, snapshot/stream the down Gram and call `write_layer_shard(..., matrix_name="down_proj")` for it. Relax the `--input-cov-offload must be SOLE flag` guard (build_self_traces_calib_vllm.py:2143) ONLY as needed to let down ride alongside gate/up in the same run, OR run down as its own capture pass — your call, but document it. Store **bf16** shards (the fp16-overflow fix landed in input_cov_offload.py — raw Gram sums overflow fp16; bf16 is required).
5. **Wheel rebuild:** the new in-graph call site is in the patched fused-MoE path → rebuild the vLLM wheel (`scripts/hf_jobs_build_patched_vllm.sh`, publishes to `pirola/vllm-patched-calib`; bump `VLLM_WHEEL_FILE` in the harness configs). NO `.cu` change.

## Correctness gates (MUST satisfy)
- **Merged sidecar with BOTH key sets.** The regenerated `covariance.pt` must contain `(layer,e,"gate_proj")` AND `(layer,e,"down_proj")` for all 40×256, `schema_version=2`. `load_covariance` (`cached_calibration_signals.py:1452-1463`) does NO key-completeness check — a gate-only or down-only sidecar loads silently and re-degrades. If capturing down in a separate pass, write a merge step that combines the existing gate `covariance.pt` (on HF, `pirola/calib-v2-self-traces/sidecars/self_traces_489ee0e1b17b43b0/covariance.pt`) with the new down keys into one payload.
- **bf16 storage** (not fp16): down Gram is a raw un-normalized sum over ~10^5-10^6 tokens/expert → overflows fp16's 65504. Store bf16 (full fp32 exponent range).
- **CUDA not eager:** the down Gram MUST be accumulated by the `gram_grouped_accum` op inside the graph. If you find yourself writing a Python per-token/per-expert matmul or `.to("cpu")`, STOP — that's the eager path the user explicitly rejected.

## Process (the project's mandatory workflow — see memory feedback_paper_fidelity_review_loop + feedback_review_fix_loop_protocol)
1. **Planner → plan-review loop** (independent reviewer ↔ planner until all-none across CRITICAL/HIGH/MEDIUM/LOW/NITPICK) BEFORE writing code.
2. **Implementer writes CODE ONLY** (no tests, no test runs during implementation).
3. **Paper/spec-fidelity review loop** (does the in-graph SYRK match the gate/up path's contract; deviations documented in docstrings) until clean.
4. **Code-quality review loop** (5 categories) until all-none.
5. **THEN tests + gates.** 6. **Commit FF-only** (no PR language; branch → commit → `merge --ff-only` → push; Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>).
7. Rebuild wheel → provision a Triton-backend H200 → re-capture down cov (forward-only replay of `self_traces_489ee0e1b17b43b0.jsonl` is fine, mirroring the gate/up capture) → merge into covariance.pt → **content-verify** → upload to HF → release the GPU (never leave it idle).

## Acceptance criteria (content-verify, not file-size — feedback_verify_content_not_filesize)
- `covariance.pt` loads (PYTHONPATH includes max_quality/src) as a `CovariancePayload` with **20480 entries** = 10240 `gate_proj` + 10240 `down_proj` keys.
- Sampled `down_proj` Grams: dtype **bfloat16**, shape **`[512,512]`**, **finite (0 inf/0 nan)**, diagonal > 0, symmetric-PSD-ish, token_counts > 0.
- Existing `gate_proj` keys unchanged (byte-identical to the current HF covariance).
- A Stage-3/Stage-4 dry-run (or unit) confirms `_cov_lookup`/EoRA now find the down key (no "falling back to plain SVD" warning for down_proj).
- The capture ran on the **CUDA in-graph path** (verify: torch.compile/cudagraph active, `gram_grouped_accum` invoked for the down group; no eager `expert_mid` callback, no `enforce_eager`).

## Key file references
- `max_quality/patches/vllm_calibration_hooks.patch`: kernel `csrc/calibration/gram_grouped_accum.cu`, op reg ~:187-194, sole SYRK call :10383, deleted expert_mid dispatch :9846-9858 (new call site location ~:9851).
- `max_quality/patches/vllm_calibration_stage2_profile.patch`: eager `_expert_mid_handler` :431 (DEAD — reference only, do not revive).
- `max_quality/scripts/build_self_traces_calib_vllm.py`: `_run_input_cov_offload` :3783, sole-flag guard :2143.
- `max_quality/src/moe_compress/utils/input_cov_offload.py`: `write_layer_shard`/`assemble_covariance` (matrix_name-aware) :65,98,150.
- `max_quality/src/moe_compress/utils/cached_calibration_signals.py`: `SCHEMA_VERSIONS["covariance"]=2` :130, `load_covariance` (no completeness check) :1452-1463.
- A_cov consumers (NO change, for validation): `stage4/plugins/eora_compensation.py:537-544`, `stage3/plugins/d_rank_allocate.py:362-394`, `stage3/plugins/swift_svd_alpha.py:757-778`. (NOT `aa_svd_factor` — it `del A`s and factorizes with B/C.)
- Wheel build: `scripts/hf_jobs_build_patched_vllm.sh` → `pirola/vllm-patched-calib`.

## Out of scope / do NOT
- Do NOT change the `.cu` kernel (already dimension-agnostic). Do NOT change the consumer stages (already down-ready). Do NOT revive the eager `_expert_mid_handler`. Do NOT "add" up_proj keys (up aliases gate by design). Do NOT capture any other signal — down_proj cov is the only gap.
